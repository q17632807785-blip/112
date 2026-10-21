"""16 — 在 sim_train 的 S0/S1 上为各方法估 95% 特异度检出阈值（第 3.6 节步骤 2）。

判定统计量：各簇斜率 z（校准后 SE）的最小值；PoPLR 用 S。
不打开 sim_test。阈值仅供日后解封后使用。
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from guard_sim_test import load_sim_train, verify_sim_test_hash_only
from methods.base import GeometryInfo, Series
from methods.simple import B1, B5
from methods.eb import EB, EBGK
from methods.lmm import B3, B4
import sim_core as sc

def P(*a): return os.path.join(ROOT, *a)

def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    secal = json.load(open(P("results","baselines","se_calibration_classical_sim_train.json")))
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data","split.json")))
    # train series
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
    N = len(y)
    methods = {"B1": B1(geo), "B5": B5(geo, **cfg["methods"]["B5"]), "EB": EB(geo),
               "EB_GK": EBGK(geo), "B3": B3(geo), "B4": B4(geo)}
    for m in methods.values():
        if getattr(m, "needs_fit", False): m.fit(train_series)

    out = {"k": k, "sim_test_opened": False, "sim_test_hash_verified": h, "methods": {}}
    neg = np.isin(scenarios, ["S0", "S1"])
    for name, m in methods.items():
        print("stat", name, flush=True)
        c = secal["methods"][name]["se_calibration"]["cluster"]
        zmin = np.zeros(N)
        for i in range(N):
            est = m.estimate(Series(str(i), str(i), t[i], y[i], cens[i].astype(bool)))
            with np.errstate(divide="ignore", invalid="ignore"):
                z = est.slope_cluster / (est.se_cluster * c)
            z = z[np.isfinite(z)]
            zmin[i] = float(np.min(z)) if len(z) else 0.0
            if (i + 1) % 3000 == 0: print(f"  {i+1}/{N}", flush=True)
        # more negative z → more evidence of progression; threshold = 5th percentile of neg (so 95% of neg are above thr)
        thr = float(np.quantile(zmin[neg], 0.05))
        sens = {s: float(np.mean(zmin[scenarios == s] <= thr)) for s in sorted(set(scenarios.tolist()))}
        out["methods"][name] = dict(
            stat="min_cluster_z_calibrated", se_cal_c_cluster=c, threshold_q05_S0S1=thr,
            specificity_S0S1=float(np.mean(zmin[neg] > thr)), sens_at_thr=sens,
            provisional=(name == "B4"),
        )
        print(f"  thr={thr:.3f} sens_S2={sens['S2']:.3f} sens_S5={sens['S5']:.3f}", flush=True)

    # PoPLR from saved scores
    poplr = json.load(open(P("results","baselines","poplr_sim_train_full.json")))
    scores = np.load(P("results","baselines","poplr_sim_train_full_scores.npz"), allow_pickle=True)
    S = scores["S"]; scn = scores["scenarios"].astype(str)
    out["methods"]["PoPLR"] = dict(
        stat="S", threshold_q95_S0S1=poplr["threshold_S0S1_q95"],
        specificity_S0S1=poplr["specificity_S0S1"], sens_at_thr=poplr["sens_at_thr"],
    )

    # G1-MST: use mean-of-seeds min z if preds exist
    g1 = json.load(open(P("results","baselines","g1mst_sim_train_k4_full.json")))
    c_g1 = g1["se_calibration_mean_of_seeds"]["cluster"]
    zmins = []
    for seed in cfg["model_seeds"]:
        p = np.load(P("results","baselines",f"g1mst_seed{seed}_k4_preds.npz"))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = p["b_cl"] / (p["se_cl"] * c_g1)
        zmin = np.nanmin(z, axis=1)
        zmins.append(zmin)
    # average zmin across seeds (descriptive); threshold on that
    zmin_avg = np.mean(np.stack(zmins), axis=0)
    thr = float(np.quantile(zmin_avg[neg], 0.05))
    sens = {s: float(np.mean(zmin_avg[scenarios == s] <= thr)) for s in sorted(set(scenarios.tolist()))}
    out["methods"]["G1-MST"] = dict(
        stat="min_cluster_z_calibrated_mean_over_seeds", se_cal_c_cluster=c_g1,
        threshold_q05_S0S1=thr, specificity_S0S1=float(np.mean(zmin_avg[neg] > thr)),
        sens_at_thr=sens,
        note="H3 正式算法是每种子各自定阈值再平均灵敏度；此处为训练侧描述性汇总",
    )
    print("G1-MST thr", thr, "sens_S2", sens["S2"], flush=True)

    path = P("results","baselines","detection_thresholds_sim_train.json")
    json.dump(out, open(path,"w"), indent=2, ensure_ascii=False)
    print("wrote", path)

if __name__ == "__main__":
    main()
