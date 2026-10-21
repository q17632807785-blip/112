"""14_poplr_full.py — PoPLR 全量 sim_train（加速版）。"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, yaml
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from guard_sim_test import load_sim_train, verify_sim_test_hash_only

def P(*a): return os.path.join(ROOT, *a)

def ols_b_se(t, Y):
    # t (k,), Y (k,59) or (n_perm,k,59)
    if Y.ndim == 2:
        tc = t - t.mean(); sxx = (tc @ tc)
        yc = Y - Y.mean(0)
        b = (tc[:, None] * yc).sum(0) / sxx
        resid = Y - (Y.mean(0) - b * t.mean() + b * t[:, None])
        # fix intercept
        a = Y.mean(0) - b * t.mean()
        resid = Y - (a + b * t[:, None])
        dof = max(len(t) - 2, 1)
        se = np.sqrt((resid ** 2).sum(0) / dof) / np.sqrt(sxx)
        return b, se
    # Y (R,k,p)
    tc = t - t.mean(); sxx = float(tc @ tc)
    yc = Y - Y.mean(1, keepdims=True)
    b = (tc[None, :, None] * yc).sum(1) / sxx
    a = Y.mean(1) - b * t.mean()
    resid = Y - (a[:, None, :] + b[:, None, :] * t[None, :, None])
    dof = max(len(t) - 2, 1)
    se = np.sqrt((resid ** 2).sum(1) / dof) / np.sqrt(sxx)
    return b, se

def S_from_b_se(b, se, df):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, b / se, 0.0)
    p = stats.t.cdf(z, df)
    if b.ndim == 1:
        neg = b < 0
        if not neg.any(): return 0.0
        return float(-np.sum(np.log(np.clip(p[neg], 1e-300, 1.0))))
    # (R,59)
    R = b.shape[0]
    out = np.zeros(R)
    for r in range(R):
        neg = b[r] < 0
        if neg.any():
            out[r] = -np.sum(np.log(np.clip(p[r, neg], 1e-300, 1.0)))
    return out

def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    n_perm = int(cfg["methods"]["PoPLR"]["n_perm"]); seed = int(cfg["seed"])
    sim = load_sim_train(); k = 4
    t, y = sim[f"k{k}_t"], sim[f"k{k}_y"]
    scenarios = sim["meta_scenario"].astype(str); N = len(y)
    print(f"PoPLR full N={N} n_perm={n_perm}", flush=True)
    S = np.zeros(N); p_perm = np.zeros(N)
    t0 = time.time(); rng = np.random.default_rng(seed)
    df = max(k - 2, 1)
    for i in range(N):
        ti, yi = t[i], y[i]
        b, se = ols_b_se(ti, yi)
        Si = S_from_b_se(b, se, df); S[i] = Si
        # all perms at once
        perms = np.array([rng.permutation(k) for _ in range(n_perm)])
        Yp = yi[perms]  # (R,k,59)
        bp, sep = ols_b_se(ti, Yp)
        null = S_from_b_se(bp, sep, df)
        p_perm[i] = float(np.mean(null >= Si))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} elapsed={time.time()-t0:.1f}s", flush=True)
    neg = np.isin(scenarios, ["S0", "S1"])
    thr = float(np.quantile(S[neg], 0.95))
    sens = {s: float(np.mean(S[scenarios == s] >= thr)) for s in sorted(set(scenarios.tolist()))}
    out = dict(n=N, k=k, n_perm=n_perm, seed=seed, threshold_S0S1_q95=thr,
               specificity_S0S1=float(np.mean(S[neg] < thr)), sens_at_thr=sens,
               mean_S_by_scenario={s: float(S[scenarios == s].mean()) for s in sorted(set(scenarios.tolist()))},
               seconds=time.time()-t0, sim_test_hash_verified=h, sim_test_opened=False)
    np.savez_compressed(P("results","baselines","poplr_sim_train_full_scores.npz"), S=S, p_perm=p_perm, scenarios=scenarios)
    json.dump(out, open(P("results","baselines","poplr_sim_train_full.json"),"w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
