"""12_eval_g1mst_sim_train.py — G1-MST / MST-S 在 sim_train 全量评测 + SE 校准。"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc
from guard_sim_test import load_sim_train, verify_sim_test_hash_only
from methods.base import GeometryInfo
from methods.g1mst import G1MST, G1MSTConfig

def P(*a): return os.path.join(ROOT, *a)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def eq_scenario_rmse(pred, truth, scenarios):
    scs = sorted(set(scenarios.tolist())); per = {}
    vals = []
    for s in scs:
        m = scenarios == s
        r = float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)))
        per[s] = r; vals.append(r)
    return float(np.mean(vals)), per

def se_cal_c(hat, se, truth):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (hat - truth) / se
    z = z[np.isfinite(z)]
    return float(np.quantile(np.abs(z), 0.95) / 1.96) if len(z) >= 10 else float("nan")

def coverage(hat, se, truth, c=1.0):
    se2 = se * c
    valid = np.isfinite(hat) & np.isfinite(truth) & np.isfinite(se2) & (se2 > 0)
    if not valid.any(): return float("nan")
    return float((np.abs(hat - truth)[valid] <= 1.96 * se2[valid]).mean())

def ols_batch(t, Y):
    """t (N,k), Y (N,k,p) → a,b,se (N,p)"""
    tc = t - t.mean(1, keepdims=True)
    sxx = (tc ** 2).sum(1).clip(min=1e-12)
    yc = Y - Y.mean(1, keepdims=True)
    b = (tc[:, :, None] * yc).sum(1) / sxx[:, None]
    a = Y.mean(1) - b * t.mean(1, keepdims=True)
    pred = a[:, None, :] + b[:, None, :] * t[:, :, None]
    resid = Y - pred
    dof = max(t.shape[1] - 2, 1)
    sigma = np.sqrt((resid ** 2).sum(1) / dof)
    se = sigma / np.sqrt(sxx)[:, None]
    return a, b, se

def slopes_from_den(t, den, cluster):
    _, b_pt, se_pt = ols_batch(t, den)
    cm = np.stack([den[:, :, cluster == c].mean(2) for c in range(1, 11)], axis=2)
    _, b_cl, se_cl = ols_batch(t, cm)
    ms = den.mean(2)
    _, b_gl, se_gl = ols_batch(t, ms[:, :, None])
    return b_pt, se_pt, b_cl, se_cl, b_gl[:, 0], se_gl[:, 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["g1mst", "msts"], default="g1mst")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--seeds", default=None, help="comma seeds to run (default all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only(); print("sim_test hash OK", h, flush=True)
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    sim = load_sim_train(); k = args.k
    t, y, cens = sim[f"k{k}_t"], sim[f"k{k}_y"], sim[f"k{k}_censored"]
    age = sim[f"k{k}_age"]
    scenarios = sim["meta_scenario"].astype(str)
    truth_cl = sim[f"k{k}_truth_cluster_obs"]
    truth_gl = sim[f"k{k}_truth_global_obs"]
    truth_pt = sim[f"k{k}_truth_point_obs"]
    cluster = sim["cluster_of_point"]; N = len(y)
    print(f"sim_train N={N} k={k} variant={args.variant}", flush=True)

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(cfg["model_seeds"])
    no_temporal = args.variant == "msts"
    ckpt_dir = args.ckpt_dir or P("results", "msts" if no_temporal else "g1mst")
    out_dir = P("results", "baselines"); os.makedirs(out_dir, exist_ok=True)

    per_seed = []
    for seed in seeds:
        path = os.path.join(ckpt_dir, f"seed{seed}_fold0.pt")
        if not os.path.exists(path):
            print("MISSING", path); continue
        gcfg = G1MSTConfig(
            d_model=cfg["g1mst"]["d_model"], n_layers=cfg["g1mst"]["n_layers"],
            ff_width=cfg["g1mst"]["ff_width"], dropout=cfg["g1mst"]["dropout"],
            fourier_freqs=cfg["g1mst"]["fourier_freqs"],
            inference_K=cfg["g1mst"]["inference_K"], mask_frac=cfg["g1mst"]["mask_frac"],
            seed=seed, no_temporal=no_temporal,
        )
        model = G1MST(geo, gcfg); model.load(path)
        print(f"eval seed={seed} K={gcfg.inference_K} batch={args.batch_size}", flush=True)
        t0 = time.time()
        den_all = np.empty_like(y)
        for start in range(0, N, args.batch_size):
            end = min(N, start + args.batch_size)
            den_all[start:end] = model.denoise_array(y[start:end], cens[start:end], t[start:end], age[start:end])
            if end % 2048 < args.batch_size or end == N:
                print(f"    denoise {end}/{N} ({time.time()-t0:.1f}s)", flush=True)
        b_pt, se_pt, b_cl, se_cl, b_gl, se_gl = slopes_from_den(t, den_all, cluster)
        cl_r, cl_per = eq_scenario_rmse(b_cl, truth_cl, scenarios)
        gl_r, gl_per = eq_scenario_rmse(b_gl, truth_gl, scenarios)
        pt_r, _ = eq_scenario_rmse(b_pt, truth_pt, scenarios)
        cal = dict(cluster=se_cal_c(b_cl, se_cl, truth_cl),
                   global_=se_cal_c(b_gl, se_gl, truth_gl),
                   point=se_cal_c(b_pt, se_pt, truth_pt))
        row = dict(
            seed=seed, seconds=time.time()-t0, ckpt=path, ckpt_sha256=sha256_file(path),
            cluster_rmse_eq=cl_r, global_rmse_eq=gl_r, point_rmse_eq=pt_r,
            per_scenario_cluster=cl_per, per_scenario_global=gl_per,
            se_calibration=cal,
            coverage_raw=dict(cluster=coverage(b_cl, se_cl, truth_cl), global_=coverage(b_gl, se_gl, truth_gl)),
            coverage_calibrated=dict(
                cluster=coverage(b_cl, se_cl, truth_cl, cal["cluster"]),
                global_=coverage(b_gl, se_gl, truth_gl, cal["global_"]),
            ),
            inference_K=gcfg.inference_K,
        )
        per_seed.append(row)
        np.savez_compressed(os.path.join(out_dir, f"{args.variant}_seed{seed}_k{k}_preds.npz"),
                            b_cl=b_cl, se_cl=se_cl, b_gl=b_gl, se_gl=se_gl,
                            scenarios=scenarios, meta_template_subject=sim["meta_template_subject"])
        print(f"  seed {seed}: cl={cl_r:.4f} gl={gl_r:.4f} c_cl={cal['cluster']:.3f} ({row['seconds']:.1f}s)", flush=True)

    summary = dict(
        variant=args.variant, k=k, n=N, inference_K=cfg["g1mst"]["inference_K"],
        sim_test_hash_verified=h, sim_test_opened=False,
        cluster_rmse_mean_of_seeds=float(np.mean([r["cluster_rmse_eq"] for r in per_seed])) if per_seed else None,
        global_rmse_mean_of_seeds=float(np.mean([r["global_rmse_eq"] for r in per_seed])) if per_seed else None,
        point_rmse_mean_of_seeds=float(np.mean([r["point_rmse_eq"] for r in per_seed])) if per_seed else None,
        se_calibration_mean_of_seeds=dict(
            cluster=float(np.nanmean([r["se_calibration"]["cluster"] for r in per_seed])),
            global_=float(np.nanmean([r["se_calibration"]["global_"] for r in per_seed])),
            point=float(np.nanmean([r["se_calibration"]["point"] for r in per_seed])),
        ) if per_seed else None,
        per_seed=per_seed,
    )
    out_path = os.path.join(out_dir, f"{args.variant}_sim_train_k{k}_full.json")
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", out_path, flush=True)
    print(json.dumps({kk: summary[kk] for kk in summary if kk != "per_seed"}, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
