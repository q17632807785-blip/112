"""15_h4_real.py — H4 真实数据锚点（第 3.7 节），不碰 sim_test。

全体 all_folds：每折用其余折训练；≥5 次检查的眼；前 4 次估计簇趋势外推。
G1-MST：若存在对应 all_fold 检查点则用，否则跳过并注明（当前检查点是 dev_folds）。
B4 标 provisional。
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc
from guard_sim_test import verify_sim_test_hash_only
from methods.base import GeometryInfo, Series, cluster_means, ols_slopes_matrix
from methods.simple import B1, B2, B5
from methods.eb import EB, EBGK
from methods.lmm import B3, B4

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

def eye_mae(method, series, cluster, history_visits=4):
    """返回该眼 MAE（第 5 次及以后 × 10 簇平均），或 None。"""
    if len(series.t) < history_visits + 1:
        return None
    hist = Series(series.eye_id, series.subject, series.t[:history_visits], series.y[:history_visits],
                  series.censored[:history_visits],
                  None if series.age is None else series.age[:history_visits],
                  None if series.iop is None else series.iop[:history_visits])
    est = method.estimate(hist)
    # intercept from cluster means OLS on history
    cm = cluster_means(est.denoised if est.denoised is not None else hist.y, cluster)
    a, b, _ = ols_slopes_matrix(hist.t, cm)
    # For EB/B4 use their cluster slopes
    if method.name in ("EB", "EB_GK", "B4") or getattr(method, "name", "") in ("EB", "EB_GK", "B4"):
        b = est.slope_cluster
        # re-estimate intercept from last hist cluster means: a = y_last - b * t_last (simple)
        a = cm[-1] - b * hist.t[-1]
    errs = []
    for j in range(history_visits, len(series.t)):
        pred = a + b * series.t[j]
        obs = cluster_means(series.y[j:j+1], cluster)[0]
        errs.append(np.abs(pred - obs))
    return float(np.mean(errs))  # mean over future visits × clusters

def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    h = verify_sim_test_hash_only()
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data","split.json")))
    all_folds = split["all_folds"]
    eyes = pd.read_parquet(P("data","eyes.parquet"))
    # map eye -> n_visits
    target_eyes = eyes[eyes.n_visits >= 5]
    print(f"H4 candidate eyes (>=5 visits): {len(target_eyes)}", flush=True)

    method_ctors = {
        "B1": lambda: B1(geo),
        "B2": lambda: B2(geo),
        "B5": lambda: B5(geo, **cfg["methods"]["B5"]),
        "EB": lambda: EB(geo),
        "EB_GK": lambda: EBGK(geo),
        "B3": lambda: B3(geo),
        "B4": lambda: B4(geo),
    }
    # per-eye MAE collected across folds
    eye_mae_by_method = {m: {} for m in method_ctors}
    eye_subject = {}
    t0 = time.time()
    for fold_i, val_subj in enumerate(all_folds):
        val_subj = set(val_subj)
        train_subj = set(split["dev_subjects"] + split["holdout_subjects"]) - val_subj
        # actually all subjects = union of all_folds
        all_subj = set()
        for f in all_folds: all_subj |= set(f)
        train_subj = all_subj - val_subj
        train_series = load_series_for_subjects(train_subj)
        val_series = [s for s in load_series_for_subjects(val_subj) if len(s.t) >= 5]
        print(f"fold {fold_i}: train_eyes={len(train_series)} val_ge5={len(val_series)}", flush=True)
        fitted = {}
        for name, ctor in method_ctors.items():
            m = ctor()
            if getattr(m, "needs_fit", False):
                m.fit(train_series)
            fitted[name] = m
        for s in val_series:
            eye_subject[s.eye_id] = s.subject
            for name, m in fitted.items():
                mae = eye_mae(m, s, geo.cluster, history_visits=cfg["evaluation"]["h4_history_visits"])
                if mae is not None:
                    eye_mae_by_method[name][s.eye_id] = mae

    summary = {"sim_test_hash_verified": h, "sim_test_opened": False, "methods": {}, "n_eyes": {}}
    for name, d in eye_mae_by_method.items():
        maes = np.array(list(d.values()))
        subjects = [eye_subject[e] for e in d]
        # bootstrap by subject
        rng = np.random.default_rng(cfg["seed"])
        subj_u = sorted(set(subjects))
        subj_to_eyes = {su: [e for e, su2 in eye_subject.items() if su2 == su and e in d] for su in subj_u}
        boots = []
        for _ in range(cfg["evaluation"]["n_bootstrap"]):
            draw = rng.choice(subj_u, size=len(subj_u), replace=True)
            vals = [d[e] for su in draw for e in subj_to_eyes[su]]
            boots.append(float(np.mean(vals)) if vals else float("nan"))
        boots = np.array(boots)
        summary["methods"][name] = dict(
            mae=float(maes.mean()) if len(maes) else None,
            n_eyes=len(maes),
            mae_ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            provisional=(name == "B4"),
        )
        summary["n_eyes"][name] = len(maes)
        print(f"{name}: MAE={summary['methods'][name]['mae']:.4f} CI={summary['methods'][name]['mae_ci95']}", flush=True)

    # pairwise vs B1
    if "B1" in eye_mae_by_method and eye_mae_by_method["B1"]:
        common = set(eye_mae_by_method["B1"])
        for name in method_ctors:
            if name == "B1": continue
            common &= set(eye_mae_by_method[name])
        diffs = {}
        for name in method_ctors:
            if name == "B1": continue
            d = [eye_mae_by_method[name][e] - eye_mae_by_method["B1"][e] for e in sorted(common)]
            diffs[name] = dict(mean_diff=float(np.mean(d)), n=len(d))
        summary["diff_vs_B1"] = diffs

    summary["seconds"] = time.time() - t0
    summary["note"] = "G1-MST H4 deferred until all_folds checkpoints exist; B4 provisional (no spatial)"
    path = P("results", "baselines", "h4_real_classical.json")
    json.dump(summary, open(path, "w"), indent=2, ensure_ascii=False)
    print("wrote", path)

if __name__ == "__main__":
    main()
