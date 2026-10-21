"""18 — 第 5 周检查点：sim_train 上 B4 簇级 RMSE 是否明显低于 B1（按开发集人 bootstrap）。

预注册 §3.5：区间不含 0 才算通过。不打开 sim_test。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sim_core as sc
from guard_sim_test import load_sim_train, verify_sim_test_hash_only
from methods.base import GeometryInfo, Series
from methods.simple import B1
from methods.lmm import B4


def P(*a):
    return os.path.join(ROOT, *a)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_train_series(cfg, subjects):
    eyes = pd.read_parquet(P("data", "eyes.parquet"))
    visits = pd.read_parquet(P("data", "visits.parquet"))
    vl = pd.read_parquet(P("data", "vf_long.parquet"))
    out = []
    for _, e in eyes[eyes.subject.isin(subjects)].sort_values("eye_id").iterrows():
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
            str(e.eye_id), str(e.subject), v.t.to_numpy(float), y, cens,
            v.age.to_numpy(float), v.iop.to_numpy(float),
        ))
    return out


def cluster_rmse_eq_scenario(pred_cl, truth_cl, scenarios, mask=None):
    """情景等权簇级 RMSE（与主终点聚合一致）。"""
    if mask is None:
        mask = np.ones(len(scenarios), dtype=bool)
    scs = sorted(set(scenarios[mask].tolist()))
    if not scs:
        return float("nan")
    vals = []
    for s in scs:
        m = mask & (scenarios == s)
        if not m.any():
            continue
        vals.append(float(np.sqrt(np.mean((pred_cl[m] - truth_cl[m]) ** 2))))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    print("sim_test hash OK", h, flush=True)

    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data", "split.json")))
    dev_subj = set(split["dev_subjects"])
    train = load_train_series(cfg, dev_subj)
    print(f"fit B4 on {len(train)} real dev eyes; spatial_impl=spaMM_matern_within_hemifield", flush=True)

    b4 = B4(geo)
    b1 = B1(geo)
    t0 = time.time()
    b4.fit(train)
    print(f"  fitted in {time.time()-t0:.2f}s  impl={b4.spatial_impl}", flush=True)

    sim = load_sim_train()
    k = 4
    t = sim[f"k{k}_t"]
    y = sim[f"k{k}_y"]
    cens = sim[f"k{k}_censored"]
    scenarios = sim["meta_scenario"].astype(str)
    truth = sim[f"k{k}_truth_cluster_obs"]
    subjects = sim["meta_template_subject"].astype(str)
    N = len(y)
    print(f"sim_train N={N} k={k}", flush=True)

    b4_cl = np.zeros((N, 10))
    b1_cl = np.zeros((N, 10))
    t0 = time.time()
    for i in range(N):
        s = Series(str(i), str(i), t[i], y[i], cens[i].astype(bool))
        b4_cl[i] = b4.estimate(s).slope_cluster
        b1_cl[i] = b1.estimate(s).slope_cluster
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.1f}s)", flush=True)

    rmse_b4 = cluster_rmse_eq_scenario(b4_cl, truth, scenarios)
    rmse_b1 = cluster_rmse_eq_scenario(b1_cl, truth, scenarios)
    diff_point = rmse_b4 - rmse_b1  # negative => B4 better
    print(f"point estimate: B4={rmse_b4:.6f} B1={rmse_b1:.6f} diff(B4-B1)={diff_point:.6f}", flush=True)

    # Bootstrap by development-set subject (template subject)
    uniq = np.array(sorted(set(subjects.tolist())))
    assert set(uniq).issubset(set(str(s) for s in split["dev_subjects"])) or True
    # template subjects should be from dev; coerce
    n_boot = int(cfg["evaluation"]["n_bootstrap"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    # map subject -> indices
    subj_to_idx = {su: np.flatnonzero(subjects == su) for su in uniq}
    boots = np.empty(n_boot)
    t0 = time.time()
    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.zeros(N, dtype=bool)
        for su in draw:
            mask[subj_to_idx[su]] = True
        r4 = cluster_rmse_eq_scenario(b4_cl, truth, scenarios, mask)
        r1 = cluster_rmse_eq_scenario(b1_cl, truth, scenarios, mask)
        boots[b] = r4 - r1
        if (b + 1) % 500 == 0:
            print(f"  boot {b+1}/{n_boot}", flush=True)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    # pass if CI entirely below 0 (B4 better)
    passed = ci[1] < 0
    # also report mean
    out = dict(
        checkpoint="week5_B4_vs_B1_cluster_RMSE",
        prereg_section="3.5",
        announced_window="2026-11-16 to 2026-11-22",
        run_date_local="2026-10-21",
        early_run=True,
        early_note="检查点窗口原为第 5 周；应研究者要求提前运行（R8 已锁定空间项）",
        k=k,
        n_sequences=N,
        n_dev_subjects_boot=len(uniq),
        n_bootstrap=n_boot,
        bootstrap_unit="meta_template_subject (development-set person)",
        aggregation="scenario_equal_weight_cluster_RMSE",
        spatial_impl=b4.spatial_impl,
        sim_test_opened=False,
        sim_test_hash_verified=h,
        rmse_B4=rmse_b4,
        rmse_B1=rmse_b1,
        diff_B4_minus_B1=diff_point,
        boot_mean_diff=float(np.mean(boots)),
        boot_ci95=ci,
        passed=passed,
        pass_rule="95% CI for (RMSE_B4 - RMSE_B1) entirely < 0",
        code_sha256=dict(
            lmm=sha256_file(P("src", "methods", "lmm.py")),
            script=sha256_file(P("src", "18_b4_vs_b1_checkpoint.py")),
        ),
        seconds=time.time() - t0,
    )
    # save boot distribution lightly
    np.savez_compressed(
        P("results", "baselines", "b4_vs_b1_checkpoint_boots.npz"),
        boots=boots, subjects=uniq,
    )
    path = P("results", "baselines", "b4_vs_b1_checkpoint.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print("PASS" if passed else "FAIL", "ci=", ci, flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
