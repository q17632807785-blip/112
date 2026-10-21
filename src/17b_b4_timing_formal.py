"""17b — B4 空间项正式计时（预注册第 2.4 节）：开发集随机 20 眼 spaMM Matérn。"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RLIB = os.path.join(os.path.dirname(ROOT), "Rlib")


def P(*a):
    return os.path.join(ROOT, *a)


def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    seed = 20261021
    n_eyes = 20
    split = json.load(open(P("data", "split.json")))
    loc = pd.read_csv(P(cfg["locations"]["file"]))
    loc59 = loc[~loc.is_blind_spot.astype(bool)].sort_values("loc59")
    eyes = pd.read_parquet(P("data", "eyes.parquet"))
    visits = pd.read_parquet(P("data", "visits.parquet"))
    vl = pd.read_parquet(P("data", "vf_long.parquet"))
    dev = eyes[eyes.subject.isin(set(split["dev_subjects"]))].sort_values("eye_id").reset_index(drop=True)
    n_dev = len(dev)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_dev, size=min(n_eyes, n_dev), replace=False)
    pick = dev.iloc[np.sort(idx)]

    rows = []
    for _, e in pick.iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        l = vl[vl.eye_id == e.eye_id]
        for _, vv in v.iterrows():
            sub = l[l.visit == vv.visit].sort_values("loc59")
            for _, pt in sub.iterrows():
                xy = loc59[loc59.loc59 == int(pt.loc59)].iloc[0]
                cl = 0 if pd.isna(xy.cluster) else int(xy.cluster)
                hemi = str(xy.hemifield) if str(xy.hemifield) in ("sup", "inf") else (
                    "sup" if float(xy.y_deg) >= 0 else "inf"
                )
                rows.append(dict(
                    y=float(pt.y), t=float(vv.t),
                    eye=str(e.eye_id), visit=int(vv.visit),
                    x=float(xy.x_deg), yy=float(xy.y_deg),
                    hemi=hemi, loc=int(pt.loc59),
                    cluster=max(cl, 1),
                    eye_cl=f"{e.eye_id}_c{max(cl, 1)}",
                    eye_vis=f"{e.eye_id}_v{int(vv.visit)}",
                ))
    df = pd.DataFrame(rows)
    os.makedirs(P("results", "baselines"), exist_ok=True)
    csv_path = P("results", "baselines", "b4_timing_formal_20eyes.csv")
    df.to_csv(csv_path, index=False)
    eye_ids = sorted(pick.eye_id.astype(str).tolist())
    print(f"sample eyes={len(eye_ids)} rows={len(df)} seed={seed} n_dev={n_dev}", flush=True)

    rscript = f'''
Sys.setenv(R_LIBS_USER="{RLIB}")
library(spaMM)
d <- read.csv("{csv_path}", stringsAsFactors=FALSE)
d$eye <- factor(d$eye)
d$eye_cl <- factor(d$eye_cl)
d$eye_vis <- factor(d$eye_vis)
d$hemi <- factor(d$hemi)
elapsed <- 0
ok <- TRUE
msg <- ""
for (h in c("sup", "inf")) {{
  dh <- d[d$hemi == h, , drop=FALSE]
  t0 <- Sys.time()
  fit <- tryCatch(
    fitme(y ~ 1 + t + (1 + t | eye) + (1 + t | eye_cl) + (1 | eye_vis) + Matern(1 | x + yy),
          data = dh, method = "REML", verbose = c(TRACE = FALSE)),
    error = function(e) e
  )
  t1 <- Sys.time()
  elapsed <- elapsed + as.numeric(difftime(t1, t0, units = "secs"))
  if (inherits(fit, "error")) {{
    ok <- FALSE
    msg <- paste(msg, h, conditionMessage(fit), sep = " | ")
  }}
}}
cat("ELAPSED_SEC", elapsed, "\\n")
cat("OK_FLAG", if (ok) "TRUE" else "FALSE", "\\n")
cat("MSG", msg, "\\n")
'''
    rpath = P("results", "baselines", "b4_timing_formal.R")
    open(rpath, "w").write(rscript)
    env = os.environ.copy()
    env["R_LIBS_USER"] = RLIB
    wall0 = time.time()
    proc = subprocess.run(["Rscript", rpath], capture_output=True, text=True, env=env, timeout=7200)
    wall = time.time() - wall0
    print("R stdout:\n", proc.stdout[-2000:] if proc.stdout else "(empty)", flush=True)
    print("R stderr tail:\n", (proc.stderr or "")[-1000:], flush=True)

    sec = None
    ok = False
    for line in (proc.stdout or "").splitlines():
        if line.startswith("ELAPSED_SEC"):
            sec = float(line.split()[1])
        if line.startswith("OK_FLAG"):
            ok = line.split()[1] == "TRUE"

    extrap_h = (sec / n_eyes * n_dev / 3600.0) if sec is not None else None
    if not ok:
        preferred = "car_neighbor_graph_le6deg"
        note_extra = "spaMM fit failed on sample; prefer CAR per nonconvergence fallback"
    elif extrap_h is not None and extrap_h > 24:
        preferred = "car_neighbor_graph_le6deg"
        note_extra = "extrapolation > 24h → CAR"
    else:
        preferred = "spaMM_matern_within_hemifield"
        note_extra = "extrapolation ≤ 24h → spaMM Matérn"

    result = dict(
        formal=True,
        date_local="2026-10-21",
        announced_window="2026-11-02 to 2026-11-08",
        early_disclosure=True,
        seed=seed,
        n_eyes_timed=n_eyes,
        n_dev_eyes=n_dev,
        eye_ids=eye_ids,
        elapsed_sec_spamm=sec,
        wall_sec=wall,
        spatmm_ok=ok,
        extrapolate_hours_linear_in_eyes=extrap_h,
        threshold_hours=24,
        over_24h=(extrap_h > 24) if extrap_h is not None else None,
        preferred=preferred,
        model_formula=(
            "per hemifield: y ~ 1 + t + (1+t|eye) + (1+t|eye_cl) + (1|eye_vis) + Matern(1|x+yy), REML"
        ),
        csv=csv_path,
        note="formal timing completed early at investigator request; " + note_extra,
    )
    out = P("results", "baselines", "b4_timing_week3_formal.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
