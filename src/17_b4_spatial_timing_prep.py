"""17 — B4 spaMM Matérn 计时预备（第 2.4 / 第 3 周）。

默认 --dry-run：用 5 眼估数量级，不写正式修订。
第 3 周正式：--n-eyes 20 --write-draft 更新 drafts/baselines_r8/。
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def P(*a): return os.path.join(ROOT, *a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eyes", type=int, default=5)
    ap.add_argument("--write-draft", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    split = json.load(open(P("data","split.json")))
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    loc59 = loc[~loc.is_blind_spot.astype(bool)].sort_values("loc59")
    eyes = pd.read_parquet(P("data","eyes.parquet"))
    visits = pd.read_parquet(P("data","visits.parquet"))
    vl = pd.read_parquet(P("data","vf_long.parquet"))
    dev = eyes[eyes.subject.isin(set(split["dev_subjects"]))].sort_values("eye_id")
    rng = np.random.default_rng(cfg["seed"])
    pick = dev.iloc[rng.choice(len(dev), size=min(args.n_eyes, len(dev)), replace=False)]

    # Build long table for spaMM: one row per visit×point
    rows = []
    for _, e in pick.iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        l = vl[vl.eye_id == e.eye_id]
        for _, vv in v.iterrows():
            sub = l[l.visit == vv.visit].sort_values("loc59")
            for _, pt in sub.iterrows():
                xy = loc59[loc59.loc59 == int(pt.loc59)].iloc[0]
                rows.append(dict(
                    y=float(pt.y), t=float(vv.t), eye=str(e.eye_id),
                    x=float(xy.x_deg), yy=float(xy.y_deg),
                    hemi=str(xy.hemifield), loc=int(pt.loc59),
                    cluster=0 if pd.isna(xy.cluster) else int(xy.cluster),
                ))
    df = pd.DataFrame(rows)
    csv_path = P("results","baselines",f"b4_timing_sample_{args.n_eyes}eyes.csv")
    os.makedirs(P("results","baselines"), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} rows={len(df)} eyes={args.n_eyes}", flush=True)

    # Call R spaMM with a simple spatial model (may be slow)
    rscript = f'''
Sys.setenv(R_LIBS_USER="{P("..","Rlib")}")
library(spaMM)
d <- read.csv("{csv_path}")
# Matérn within hemifield approx: fit per hemifield then report time
t0 <- Sys.time()
fit <- tryCatch({{
  fitme(y ~ t + Matern(1|x+yy), data=d, method="REML")
}}, error=function(e) e)
t1 <- Sys.time()
cat("ELAPSED_SEC", as.numeric(difftime(t1,t0,units="secs")), "\\n")
if (inherits(fit, "error")) {{
  cat("ERROR", conditionMessage(fit), "\\n")
}} else {{
  cat("OK\\n")
}}
'''
    rpath = P("results","baselines","b4_timing_run.R")
    open(rpath,"w").write(rscript)
    t0 = time.time()
    import subprocess
    env = os.environ.copy(); env["R_LIBS_USER"] = P("..","Rlib")
    proc = subprocess.run(["Rscript", rpath], capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - t0
    print("R stdout:", proc.stdout[-500:] if proc.stdout else "")
    print("R stderr tail:", (proc.stderr or "")[-300:])
    # parse
    sec = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("ELAPSED_SEC"):
            sec = float(line.split()[1])
    n_dev = int(eyes.subject.isin(set(split["dev_subjects"])).sum())
    # linear extrapolate by eyes
    extrap_h = (sec / args.n_eyes * n_dev / 3600.0) if sec else None
    result = dict(
        n_eyes_timed=args.n_eyes, n_dev_eyes=n_dev, elapsed_sec=sec, wall_sec=elapsed,
        extrapolate_hours_linear_in_eyes=extrap_h,
        over_24h=(extrap_h > 24) if extrap_h is not None else None,
        preferred="car_neighbor_graph_le6deg" if (extrap_h is not None and extrap_h > 24) else "spaMM_matern_within_hemifield",
        note="预备/试跑；第 3 周正式 20 眼结果才写入 R8",
        csv=csv_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    json.dump(result, open(P("results","baselines","b4_timing_dryrun.json"),"w"), indent=2, ensure_ascii=False)
    if args.write_draft and sec is not None:
        draft = P("drafts","baselines_r8",f"timing_dryrun_{args.n_eyes}eyes.md")
        open(draft,"w").write(
            f"# B4 计时试跑（非正式）\n\n- 眼数: {args.n_eyes}\n- 耗时: {sec:.1f}s\n"
            f"- 外推 {n_dev} 眼: {extrap_h:.2f} h\n- 倾向: {result['preferred']}\n"
            f"- 正式窗口: 2026-11-02 至 11-08，n=20\n"
        )
        print("draft", draft)

if __name__ == "__main__":
    main()
