"""12b — 经典方法在 sim_train 上的 SE 校准系数（第 2.6 节）。"""
from __future__ import annotations

import json, os, sys, time
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc
from guard_sim_test import load_sim_train, verify_sim_test_hash_only
from methods.base import GeometryInfo, Series, ols_slopes_matrix, cluster_means, ols_1d
from methods.simple import B1, B5
from methods.eb import EB, EBGK
from methods.lmm import B3, B4

def P(*a): return os.path.join(ROOT, *a)

def se_cal_c(hat, se, truth):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (hat - truth) / se
    z = z[np.isfinite(z)]
    return float(np.quantile(np.abs(z), 0.95) / 1.96) if len(z) >= 10 else float("nan")

def coverage(hat, se, truth, c=1.0):
    se2 = se * c
    ok = (np.abs(hat - truth) <= 1.96 * se2) & np.isfinite(se2) & (se2 > 0)
    valid = np.isfinite(hat) & np.isfinite(truth) & np.isfinite(se2) & (se2 > 0)
    return float(ok[valid].mean()) if valid.sum() else float("nan")

def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data", "split.json")))
    # load real train
    eyes = pd.read_parquet(P("data","eyes.parquet"))
    visits = pd.read_parquet(P("data","visits.parquet"))
    vl = pd.read_parquet(P("data","vf_long.parquet"))
    train_series = []
    for _, e in eyes[eyes.subject.isin(set(split["dev_subjects"]))].sort_values("eye_id").iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        if len(v) < 2: continue
        l = vl[vl.eye_id == e.eye_id]
        y = np.full((len(v), 59), np.nan); cens = np.zeros((len(v), 59), bool)
        for i, row in enumerate(v.itertuples()):
            sub = l[l.visit == row.visit].sort_values("loc59")
            y[i] = sub.y; cens[i] = sub.censored
        train_series.append(Series(str(e.eye_id), str(e.subject), v.t.to_numpy(float), y, cens,
                                   v.age.to_numpy(float), v.iop.to_numpy(float)))
    sim = load_sim_train(); k = 4
    t, y, cens = sim[f"k{k}_t"], sim[f"k{k}_y"], sim[f"k{k}_censored"]
    scenarios = sim["meta_scenario"].astype(str)
    truth_cl, truth_gl, truth_pt = sim[f"k{k}_truth_cluster_obs"], sim[f"k{k}_truth_global_obs"], sim[f"k{k}_truth_point_obs"]
    cluster = sim["cluster_of_point"]; N = len(y)
    methods = {
        "B1": B1(geo),
        "B5": B5(geo, **cfg["methods"]["B5"]),
        "EB": EB(geo),
        "EB_GK": EBGK(geo),
        "B3": B3(geo),
        "B4": B4(geo),
    }
    for m in methods.values():
        if getattr(m, "needs_fit", False):
            m.fit(train_series)
    out = {"sim_test_hash_verified": h, "sim_test_opened": False, "k": k, "n": N, "methods": {}}
    # B1/B5 vectorized-ish via loop for SE from estimate
    for name, m in methods.items():
        print("cal", name, flush=True)
        t0 = time.time()
        b_cl = np.zeros((N, 10)); se_cl = np.zeros((N, 10))
        b_gl = np.zeros(N); se_gl = np.zeros(N)
        b_pt = np.zeros((N, 59)); se_pt = np.zeros((N, 59))
        for i in range(N):
            est = m.estimate(Series(str(i), str(i), t[i], y[i], cens[i].astype(bool)))
            b_cl[i], se_cl[i] = est.slope_cluster, est.se_cluster
            b_gl[i], se_gl[i] = est.slope_global, est.se_global
            b_pt[i], se_pt[i] = est.slope_point, est.se_point
            if (i + 1) % 3000 == 0: print(f"  {i+1}/{N}", flush=True)
        cal = dict(
            cluster=se_cal_c(b_cl, se_cl, truth_cl),
            global_=se_cal_c(b_gl, se_gl, truth_gl),
            point=se_cal_c(b_pt, se_pt, truth_pt),
        )
        out["methods"][name] = dict(
            se_calibration=cal,
            coverage_raw=dict(cluster=coverage(b_cl, se_cl, truth_cl), global_=coverage(b_gl, se_gl, truth_gl)),
            coverage_calibrated=dict(
                cluster=coverage(b_cl, se_cl, truth_cl, cal["cluster"]),
                global_=coverage(b_gl, se_gl, truth_gl, cal["global_"]),
            ),
            provisional=(name == "B4"),
            seconds=time.time() - t0,
        )
        print(f"  {name} c_cl={cal['cluster']:.3f} cov_raw={out['methods'][name]['coverage_raw']['cluster']:.3f} "
              f"cov_cal={out['methods'][name]['coverage_calibrated']['cluster']:.3f}", flush=True)
    path = P("results", "baselines", "se_calibration_classical_sim_train.json")
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
    print("wrote", path)

if __name__ == "__main__":
    main()
