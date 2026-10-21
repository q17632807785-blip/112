"""10_baselines.py — 在开发集真实数据上 fit，在 sim_train 上评测对照方法。

硬性规则：
  - 永不加载 sim_test.npz 内容（只可核对哈希）。
  - B4 当前为无空间项 provisional，结果写入 baselines/ 并标注，不作为正式 H1 比较。
  - 不修改已发布预注册文档，不推送 GitHub。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sim_core as sc  # noqa: E402
from guard_sim_test import load_sim_train, verify_sim_test_hash_only  # noqa: E402
from methods.base import GeometryInfo, Series, ols_slopes_matrix, cluster_means  # noqa: E402
from methods.simple import B0, B1, B2, B5  # noqa: E402
from methods.eb import EB, EBGK  # noqa: E402
from methods.lmm import B3, B4  # noqa: E402
from methods.poplr import PoPLR  # noqa: E402


def P(*a):
    return os.path.join(ROOT, *a)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cfg():
    return yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))


def make_geo(cfg) -> GeometryInfo:
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    return GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)


def load_real_series(cfg, subjects: set[str]) -> list[Series]:
    eyes = pd.read_parquet(P("data", "eyes.parquet"))
    visits = pd.read_parquet(P("data", "visits.parquet"))
    vl = pd.read_parquet(P("data", "vf_long.parquet"))
    out = []
    E = eyes[eyes.subject.isin(subjects)].sort_values("eye_id")
    for _, e in E.iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        if len(v) < 2:
            continue
        l = vl[vl.eye_id == e.eye_id]
        y = np.full((len(v), 59), np.nan)
        cens = np.zeros((len(v), 59), bool)
        for i, row in enumerate(v.itertuples()):
            sub = l[l.visit == row.visit].sort_values("loc59")
            y[i] = sub.y.to_numpy()
            cens[i] = sub.censored.to_numpy()
        out.append(Series(
            eye_id=str(e.eye_id), subject=str(e.subject),
            t=v.t.to_numpy(float), y=y, censored=cens,
            age=v.age.to_numpy(float), iop=v.iop.to_numpy(float),
        ))
    return out


def vectorized_b1_slopes(t, y, cluster):
    """t (N,k), y (N,k,59) → point/cluster/global slopes."""
    N, k, P = y.shape
    tc = t - t.mean(1, keepdims=True)
    sxx = (tc ** 2).sum(1)  # (N,)
    yc = y - y.mean(1, keepdims=True)
    # b_point (N,59)
    b_pt = (tc[:, :, None] * yc).sum(1) / sxx[:, None]
    # cluster means then OLS
    cm = np.zeros((N, k, 10))
    for c in range(1, 11):
        m = cluster == c
        cm[:, :, c - 1] = y[:, :, m].mean(2)
    cmc = cm - cm.mean(1, keepdims=True)
    b_cl = (tc[:, :, None] * cmc).sum(1) / sxx[:, None]
    ms = y.mean(2)
    msc = ms - ms.mean(1, keepdims=True)
    b_gl = (tc * msc).sum(1) / sxx
    return b_pt, b_cl, b_gl


def vectorized_b5_slopes(t, y, W, cluster):
    # smooth: y_s[n,k,p] = sum_q y[n,k,q] * W[p,q]
    ys = np.einsum("nkq,pq->nkp", y, W)
    return vectorized_b1_slopes(t, ys, cluster)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_vectorized(name, b_pt, b_cl, b_gl, truth_pt, truth_cl, truth_gl, scenarios):
    """按情景等权、簇等权汇总。"""
    scs = sorted(set(scenarios.tolist()))
    per = {}
    cl_rmses = []
    gl_rmses = []
    pt_rmses = []
    for sc in scs:
        m = scenarios == sc
        per[sc] = {
            "cluster_rmse": rmse(b_cl[m], truth_cl[m]),
            "global_rmse": rmse(b_gl[m], truth_gl[m]),
            "point_rmse": rmse(b_pt[m], truth_pt[m]),
            "n": int(m.sum()),
        }
        cl_rmses.append(per[sc]["cluster_rmse"])
        gl_rmses.append(per[sc]["global_rmse"])
        pt_rmses.append(per[sc]["point_rmse"])
    return {
        "method": name,
        "k": int(truth_cl.shape[1] and b_cl.shape[0] >= 0) or 4,  # placeholder overwritten
        "cluster_rmse_eq_scenario": float(np.mean(cl_rmses)),
        "global_rmse_eq_scenario": float(np.mean(gl_rmses)),
        "point_rmse_eq_scenario": float(np.mean(pt_rmses)),
        "per_scenario": per,
    }


def eval_method_loop(method, t, y, cens, scenarios, truth_cl, truth_gl, truth_pt, cluster, max_n=None):
    """慢路径：逐条 estimate。"""
    N = y.shape[0]
    if max_n is not None:
        N = min(N, max_n)
    b_cl = np.full((N, 10), np.nan)
    b_gl = np.full(N, np.nan)
    b_pt = np.full((N, 59), np.nan)
    n_fail = 0
    for i in range(N):
        s = Series(
            eye_id=f"sim_{i}", subject=f"sim_{i}",
            t=t[i], y=y[i], censored=cens[i].astype(bool),
        )
        try:
            est = method.estimate(s)
            if est.failed:
                n_fail += 1
            b_cl[i] = est.slope_cluster
            b_gl[i] = est.slope_global
            b_pt[i] = est.slope_point
        except Exception:
            n_fail += 1
            # B1 fallback
            _, bp, _ = ols_slopes_matrix(t[i], y[i])
            cm = cluster_means(y[i], cluster)
            _, bc, _ = ols_slopes_matrix(t[i], cm)
            b_pt[i] = bp
            b_cl[i] = bc
            b_gl[i] = float(np.nanmean(bp))
        if (i + 1) % 2000 == 0:
            print(f"  {method.name}: {i+1}/{N}", flush=True)
    summary = evaluate_vectorized(
        method.name, b_pt, b_cl, b_gl,
        truth_pt[:N], truth_cl[:N], truth_gl[:N], scenarios[:N],
    )
    summary["n_fail"] = n_fail
    summary["n_eval"] = N
    return summary, b_cl, b_gl


def se_calibration(b_hat, se_hat, b_true):
    """c = quantile(|z|, 0.95) / 1.96。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (b_hat - b_true) / se_hat
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return float("nan")
    return float(np.quantile(np.abs(z), 0.95) / 1.96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--methods", default="B1,B2,B5,EB,EB_GK,B3,B4,PoPLR")
    ap.add_argument("--max-n", type=int, default=None, help="限制 sim_train 条数（调试）")
    ap.add_argument("--skip-slow", action="store_true", help="跳过 PoPLR（置换慢）")
    args = ap.parse_args()

    cfg = load_cfg()
    os.makedirs(P("results", "baselines"), exist_ok=True)
    os.makedirs(P("logs"), exist_ok=True)
    log_path = P("logs", "10_baselines.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(log_path, "w").write("")
    log(f"=== 10_baselines start {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===")
    h = verify_sim_test_hash_only()
    log(f"sim_test seal hash OK: {h}")

    geo = make_geo(cfg)
    split = json.load(open(P("data", "split.json")))
    dev_subj = set(split["dev_subjects"])
    train_series = load_real_series(cfg, dev_subj)
    log(f"dev real series: {len(train_series)} eyes")

    sim = load_sim_train()
    k = args.k
    t = sim[f"k{k}_t"]
    y = sim[f"k{k}_y"]
    cens = sim[f"k{k}_censored"]
    scenarios = sim["meta_scenario"].astype(str)
    truth_cl = sim[f"k{k}_truth_cluster_obs"]
    truth_gl = sim[f"k{k}_truth_global_obs"]
    truth_pt = sim[f"k{k}_truth_point_obs"]
    cluster = sim["cluster_of_point"]
    assert np.array_equal(cluster, geo.cluster)
    N = y.shape[0]
    if args.max_n:
        N = min(N, args.max_n)
        t, y, cens = t[:N], y[:N], cens[:N]
        scenarios = scenarios[:N]
        truth_cl, truth_gl, truth_pt = truth_cl[:N], truth_gl[:N], truth_pt[:N]
    log(f"sim_train eval N={N} k={k}")

    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.skip_slow:
        want = [m for m in want if m != "PoPLR"]

    # instantiate
    methods = {}
    if "B0" in want:
        methods["B0"] = B0(geo)
    if "B1" in want:
        methods["B1"] = B1(geo, min_visits=cfg["methods"]["B1_min_visits"])
    if "B2" in want:
        methods["B2"] = B2(geo)
    if "B5" in want:
        methods["B5"] = B5(geo, **{k2: cfg["methods"]["B5"][k2] for k2 in ("self_weight", "neighbor_weight_total")})
    if "EB" in want:
        methods["EB"] = EB(geo, n_strata=cfg["methods"]["EB"]["sensitivity_strata"])
    if "EB_GK" in want:
        methods["EB_GK"] = EBGK(geo)
    if "B3" in want:
        methods["B3"] = B3(geo)
    if "B4" in want:
        methods["B4"] = B4(geo)
    if "PoPLR" in want:
        methods["PoPLR"] = PoPLR(geo, n_perm=cfg["methods"]["PoPLR"]["n_perm"], seed=cfg["seed"])

    for name, m in methods.items():
        if getattr(m, "needs_fit", False):
            t0 = time.time()
            log(f"fit {name} on {len(train_series)} real dev eyes...")
            m.fit(train_series)
            log(f"  fitted in {time.time()-t0:.1f}s")

    results = []
    cal = {}
    # --- vectorized B1 ---
    if "B1" in methods:
        t0 = time.time()
        b_pt, b_cl, b_gl = vectorized_b1_slopes(t, y, cluster)
        s = evaluate_vectorized("B1", b_pt, b_cl, b_gl, truth_pt, truth_cl, truth_gl, scenarios)
        s["k"] = k
        s["seconds"] = time.time() - t0
        s["n_eval"] = N
        results.append(s)
        log(f"B1 cluster RMSE (eq-scen)={s['cluster_rmse_eq_scenario']:.4f}  ({s['seconds']:.1f}s)")
        # SE calibration for B1 cluster
        # recompute SE via residual — approximate with vectorized residual SE
        tc = t - t.mean(1, keepdims=True)
        sxx = (tc ** 2).sum(1)
        cm = np.stack([y[:, :, cluster == c].mean(2) for c in range(1, 11)], axis=2)
        pred = cm.mean(1, keepdims=True) + b_cl[:, None, :] * tc[:, :, None]
        # actually intercept = mean_y - b*mean_t
        a_cl = cm.mean(1) - b_cl * t.mean(1, keepdims=True)
        pred = a_cl[:, None, :] + b_cl[:, None, :] * t[:, :, None]
        resid = cm - pred
        dof = max(k - 2, 1)
        sigma = np.sqrt((resid ** 2).sum(1) / dof)
        se_cl = sigma / np.sqrt(sxx)[:, None]
        cal["B1_cluster"] = se_calibration(b_cl, se_cl, truth_cl)
        cal["B1_global"] = se_calibration(b_gl, se_cl.mean(1), truth_gl)
        log(f"  SE cal c_cluster={cal['B1_cluster']:.3f}")

    if "B2" in methods:
        t0 = time.time()
        b_pt, b_cl, b_gl = vectorized_b1_slopes(t, y, cluster)  # same denoised
        # B2 only redefines global from MS OLS — identical to global from mean of points when using raw y
        s = evaluate_vectorized("B2", b_pt, b_cl, b_gl, truth_pt, truth_cl, truth_gl, scenarios)
        s["k"] = k
        s["seconds"] = time.time() - t0
        s["n_eval"] = N
        results.append(s)
        log(f"B2 cluster RMSE (eq-scen)={s['cluster_rmse_eq_scenario']:.4f}")

    if "B5" in methods:
        t0 = time.time()
        b_pt, b_cl, b_gl = vectorized_b5_slopes(t, y, methods["B5"].W, cluster)
        s = evaluate_vectorized("B5", b_pt, b_cl, b_gl, truth_pt, truth_cl, truth_gl, scenarios)
        s["k"] = k
        s["seconds"] = time.time() - t0
        s["n_eval"] = N
        results.append(s)
        log(f"B5 cluster RMSE (eq-scen)={s['cluster_rmse_eq_scenario']:.4f}  ({s['seconds']:.1f}s)")

    # loop methods
    for name in ["EB", "EB_GK", "B3", "B4", "PoPLR"]:
        if name not in methods:
            continue
        t0 = time.time()
        log(f"eval {name} loop...")
        s, b_cl, b_gl = eval_method_loop(
            methods[name], t, y, cens, scenarios, truth_cl, truth_gl, truth_pt, cluster, max_n=N,
        )
        s["k"] = k
        s["seconds"] = time.time() - t0
        if name == "B4":
            s["provisional"] = True
            s["spatial_impl"] = "none_provisional"
            s["note"] = "B4 无空间项；spaMM/CAR 待第 3 周测速后以修订锁定"
        results.append(s)
        log(f"{name} cluster RMSE (eq-scen)={s['cluster_rmse_eq_scenario']:.4f}  fail={s.get('n_fail',0)}  ({s['seconds']:.1f}s)")

    # week-5 style contrast B4 vs B1 (provisional)
    by = {r["method"]: r for r in results}
    if "B1" in by and "B4" in by:
        d = by["B4"]["cluster_rmse_eq_scenario"] - by["B1"]["cluster_rmse_eq_scenario"]
        log(f"PROVISIONAL B4-B1 cluster RMSE diff={d:.4f} (负值表示 B4 更好；B4 尚无空间项)")

    out = {
        "k": k,
        "n_eval": N,
        "seed": cfg["seed"],
        "sim_test_hash_verified": h,
        "sim_test_opened": False,
        "se_calibration": cal,
        "results": results,
        "code_sha256": {},
    }
    code_files = [
        "src/guard_sim_test.py",
        "src/10_baselines.py",
        "src/methods/__init__.py",
        "src/methods/base.py",
        "src/methods/simple.py",
        "src/methods/eb.py",
        "src/methods/lmm.py",
        "src/methods/poplr.py",
    ]
    for f in code_files:
        p = P(f)
        if os.path.exists(p):
            out["code_sha256"][f] = sha256_file(p)

    out_path = P("results", "baselines", f"sim_train_k{k}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote {out_path}")
    log("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
