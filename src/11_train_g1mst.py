"""11_train_g1mst.py — 在真实开发集上训练 G1-MST（主模型不用 sim_train）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sim_core as sc
from guard_sim_test import verify_sim_test_hash_only
from methods.base import GeometryInfo, Series
from methods.g1mst import G1MST, G1MSTConfig
from importlib.util import spec_from_loader  # noqa: F401


def P(*a):
    return os.path.join(ROOT, *a)


def load_real_series(subjects):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--fold", type=int, default=0, help="开发集内部折：用该折作验证，其余训练")
    ap.add_argument("--no-temporal", action="store_true", help="MST-S：去掉时间注意力")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split-folds", choices=["dev_folds","all_folds"], default="dev_folds")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))
    verify_sim_test_hash_only()
    os.makedirs(P("results", "baselines"), exist_ok=True)
    default_dir = ("results/msts" if args.no_temporal else "results/g1mst")
    if args.split_folds == "all_folds":
        default_dir = default_dir + "_allfolds"
    out_dir = args.out_dir or default_dir
    os.makedirs(P(out_dir), exist_ok=True)

    loc = pd.read_csv(P(cfg["locations"]["file"]))
    g = sc.make_geometry(loc)
    geo = GeometryInfo(cluster=g.cluster, neighbor=g.neighbor, center=g.center, xy=g.xy)
    split = json.load(open(P("data", "split.json")))
    folds = split[args.split_folds]
    fold_subjects = set(folds[args.fold])
    if args.split_folds == "all_folds":
        all_subj = set().union(*[set(f) for f in folds])
        train_subj = all_subj - fold_subjects
    else:
        train_subj = set(split["dev_subjects"]) - fold_subjects
    train_series = load_real_series(train_subj)
    val_series = load_real_series(fold_subjects)
    print(f"train eyes={len(train_series)} val eyes={len(val_series)} seed={args.seed}")

    gcfg = G1MSTConfig(
        d_model=cfg["g1mst"]["d_model"],
        n_layers=cfg["g1mst"]["n_layers"],
        ff_width=cfg["g1mst"]["ff_width"],
        dropout=cfg["g1mst"]["dropout"],
        fourier_freqs=cfg["g1mst"]["fourier_freqs"],
        lr=cfg["g1mst"]["lr"],
        weight_decay=cfg["g1mst"]["weight_decay"],
        batch_size=cfg["g1mst"]["batch_size"],
        max_epochs=args.max_epochs or cfg["g1mst"]["max_epochs"],
        early_stop_patience=cfg["g1mst"]["early_stop_patience"],
        inference_K=cfg["g1mst"]["inference_K"],
        mask_frac=cfg["g1mst"]["mask_frac"],
        seed=args.seed,
        no_temporal=args.no_temporal,
    )
    model = G1MST(geo, gcfg)
    print("n_params", model.count_params())
    t0 = time.time()
    hist = model.fit(train_series, val_series=val_series)
    elapsed = time.time() - t0
    path = P(out_dir, f"seed{args.seed}_fold{args.fold}.pt")
    model.save(path)
    meta = dict(
        seed=args.seed, fold=args.fold, n_params=hist["n_params"],
        best_val=hist["best_val"], epochs=len(hist["history"]),
        seconds=elapsed, path=path,
        train_eyes=len(train_series), val_eyes=len(val_series),
        no_temporal=args.no_temporal, variant=("MST-S" if args.no_temporal else "G1-MST"),
    )
    with open(P(out_dir, f"seed{args.seed}_fold{args.fold}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
