"""15b — H4 for G1-MST using all_folds checkpoints; average extrapolation across 5 seeds."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, yaml, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc
from guard_sim_test import verify_sim_test_hash_only
from methods.base import GeometryInfo, Series, cluster_means, ols_slopes_matrix
from methods.g1mst import G1MST, G1MSTConfig
from methods.simple import B1

def P(*a): return os.path.join(ROOT, *a)

def load_series_for_subjects(subjects):
    eyes = pd.read_parquet(P("data","eyes.parquet"))
    visits = pd.read_parquet(P("data","visits.parquet"))
    vl = pd.read_parquet(P("data","vf_long.parquet"))
    out = []
    for _, e in eyes[eyes.subject.isin(subjects)].sort_values("eye_id").iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        if len(v) < 2: continue
        l = vl[vl.eye_id == e.eye_id]
        y = np.full((len(v), 59), np.nan); cens = np.zeros((len(v), 59), bool)
        for i, row in enumerate(v.itertuples()):
            sub = l[l.visit == row.visit].sort_values("loc59")
            y[i] = sub.y; cens[i] = sub.censored
        out.append(Series(str(e.eye_id), str(e.subject), v.t.to_numpy(float), y, cens,
                          v.age.to_numpy(float), v.iop.to_numpy(float)))
    return out

def extrapolate_mae(denoised_list, series, cluster, history=4):
    """denoised_list: list of (history,59) arrays from each seed; average preds."""
    cm_seeds = [cluster_means(d, cluster) for d in denoised_list]
    # average cluster means trajectory across seeds then OLS
    cm = np.mean(np.stack(cm_seeds), axis=0)
    t = series.t[:history]
    a, b, _ = ols_slopes_matrix(t, cm)
    errs = []
    for j in range(history, len(series.t)):
        pred = a + b * series.t[j]
        obs = cluster_means(series.y[j:j+1], cluster)[0]
        errs.append(np.abs(pred - obs))
    return float(np.mean(errs))

def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data","split.json")))
    seeds = list(cfg["model_seeds"])
    ckpt_dir = P("results","g1mst_allfolds")
    missing = []
    for fold in range(5):
        for seed in seeds:
            if not os.path.exists(os.path.join(ckpt_dir, f"seed{seed}_fold{fold}.pt")):
                missing.append((fold, seed))
    if missing:
        print("MISSING checkpoints", len(missing), missing[:5], "..."); 
        json.dump({"error":"missing", "n_missing":len(missing)}, open(P("results","baselines","h4_real_g1mst.json"),"w"))
        return 1

    eye_mae = {}; eye_subject = {}; eye_mae_b1 = {}
    b1 = B1(geo)
    t0 = time.time()
    for fold, val_subj in enumerate(split["all_folds"]):
        val_subj = set(val_subj)
        all_subj = set().union(*[set(f) for f in split["all_folds"]])
        val_series = [s for s in load_series_for_subjects(val_subj) if len(s.t) >= 5]
        models = []
        for seed in seeds:
            gcfg = G1MSTConfig(
                d_model=cfg["g1mst"]["d_model"], n_layers=cfg["g1mst"]["n_layers"],
                ff_width=cfg["g1mst"]["ff_width"], dropout=cfg["g1mst"]["dropout"],
                fourier_freqs=cfg["g1mst"]["fourier_freqs"],
                inference_K=cfg["g1mst"]["inference_K"], mask_frac=cfg["g1mst"]["mask_frac"],
                seed=seed,
            )
            m = G1MST(geo, gcfg)
            m.load(os.path.join(ckpt_dir, f"seed{seed}_fold{fold}.pt"))
            models.append(m)
        print(f"fold {fold} val_ge5={len(val_series)}", flush=True)
        for s in val_series:
            hist = Series(s.eye_id, s.subject, s.t[:4], s.y[:4], s.censored[:4],
                          None if s.age is None else s.age[:4], None if s.iop is None else s.iop[:4])
            dens = [m.denoise(hist) for m in models]
            eye_mae[s.eye_id] = extrapolate_mae(dens, s, geo.cluster, 4)
            eye_subject[s.eye_id] = s.subject
            # B1 for reference on same eyes
            est = b1.estimate(hist)
            cm = cluster_means(est.denoised, geo.cluster)
            a, b, _ = ols_slopes_matrix(hist.t, cm)
            errs = []
            for j in range(4, len(s.t)):
                pred = a + b * s.t[j]
                obs = cluster_means(s.y[j:j+1], geo.cluster)[0]
                errs.append(np.abs(pred - obs))
            eye_mae_b1[s.eye_id] = float(np.mean(errs))

    maes = np.array(list(eye_mae.values()))
    # bootstrap subjects
    rng = np.random.default_rng(cfg["seed"])
    subj_u = sorted(set(eye_subject.values()))
    subj_to_eyes = {su: [e for e,s in eye_subject.items() if s==su] for su in subj_u}
    boots = []
    for _ in range(cfg["evaluation"]["n_bootstrap"]):
        draw = rng.choice(subj_u, size=len(subj_u), replace=True)
        vals = [eye_mae[e] for su in draw for e in subj_to_eyes[su]]
        boots.append(float(np.mean(vals)))
    boots = np.array(boots)
    diff = [eye_mae[e] - eye_mae_b1[e] for e in eye_mae]
    out = dict(
        sim_test_opened=False, sim_test_hash_verified=h,
        n_eyes=len(maes), mae=float(maes.mean()),
        mae_ci95=[float(np.quantile(boots,0.025)), float(np.quantile(boots,0.975))],
        mae_b1=float(np.mean(list(eye_mae_b1.values()))),
        mean_diff_vs_b1=float(np.mean(diff)),
        seconds=time.time()-t0,
    )
    json.dump(out, open(P("results","baselines","h4_real_g1mst.json"),"w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
