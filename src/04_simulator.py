"""04_simulator.py — 预注册第 3.4 节 + 修订 R5：生成 sim_train / sim_test（以及 SA8、SA10 变体）。

用法（在 directionA/ 下）：
  python src/04_simulator.py --pointsets-only
      只用公开的 data/g1_locations.csv 计算 S3/S4 候选点集，写 results/sim_pointsets.json；不读任何受试者数据。
  python src/04_simulator.py --split sim_train --confirm-r5-published [--variant SA8|SA10]
  python src/04_simulator.py --split sim_test  --confirm-r5-published [--variant SA8|SA10]

生成前的硬性检查（任何一项不满足即退出，不写任何文件）：
  1. config.yaml 的 revisions 含 R5；
  2. preregistration.md 修订 R5 中列出的模拟器源码 SHA-256 与磁盘上的文件逐个一致；
  3. sim_test：data/sim_calibration.json 存在且 final = true（第 3.5 节校准已结束），且输出文件尚不存在（不覆盖）。
本脚本从不读取已生成的 sim_test 文件（第 3.4 节封存）；生成后只计算并打印其 SHA-256，写入 logs/。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc  # noqa: E402

SIM_SOURCES = ["src/sim_core.py", "src/04_simulator.py", "src/05_calibrate_sim.py", "src/test_simulator_toy.py"]


def P(*a):
    return os.path.join(ROOT, *a)


def load_cfg():
    return yaml.safe_load(open(P("config.yaml"), encoding="utf-8"))


def pointsets(cfg, geo):
    s3c, ang = sc.s3_candidates(geo)
    s4c, elig = sc.s4_candidates(geo, cfg["simulator"]["scenarios"]["S4"]["max_ecc_deg"],
                                 tuple(cfg["simulator"]["scenarios"]["S4"]["n_points"]))
    for ncl in cfg["simulator"]["scenarios"]["S3"]["n_clusters_spanned"]:
        assert any(c["n_clusters"] == ncl for c in s3c), f"S3 没有跨 {ncl} 个簇的候选弧"
    return s3c, ang, s4c, elig


def guard(cfg, split):
    revs = [r.split()[0] for r in cfg.get("revisions", [])]
    if "R5" not in revs:
        sys.exit("拒绝生成：config.yaml 的 revisions 中没有 R5（S3/S4 规则与模拟器代码哈希须先作为修订发布）。")
    text = open(P("preregistration.md"), encoding="utf-8").read()
    for f in SIM_SOURCES:
        m = re.search(r"`" + re.escape(f) + r"`\s*\|\s*`([0-9a-f]{64})`", text)
        if not m:
            sys.exit(f"拒绝生成：preregistration.md 中找不到 {f} 的 SHA-256。")
        if sc.sha256_file(P(f)) != m.group(1):
            sys.exit(f"拒绝生成：{f} 的 SHA-256 与预注册 R5 不一致。")
    if split == "sim_test":
        cal = P("data", "sim_calibration.json")
        if not os.path.exists(cal) or not json.load(open(cal, encoding="utf-8")).get("final"):
            sys.exit("拒绝生成 sim_test：第 3.5 节校准尚未结束（data/sim_calibration.json 不存在或 final != true）。")


def load_templates(cfg, split, nm):
    """模板眼：sim_train = 开发集全部眼；sim_test = 留出集全部眼（第 3.2 节）。按 eye_id 字符串升序编号。"""
    spl = json.load(open(P("data", "split.json")))
    subj = set(spl["dev_subjects"] if split == "sim_train" else spl["holdout_subjects"])
    eyes = pd.read_parquet(P("data", "eyes.parquet")); visits = pd.read_parquet(P("data", "visits.parquet"))
    vl = pd.read_parquet(P("data", "vf_long.parquet"))
    dev_eyes = eyes[eyes.subject.isin(set(spl["dev_subjects"]))]
    # 基线 MS 三分位：切点取开发集眼的三分位数（同 03_noise_model.py），留出集眼套用开发集切点（R5 规则 N-2）
    q1, q2 = dev_eyes.baseline_ms.quantile(1 / 3), dev_eyes.baseline_ms.quantile(2 / 3)
    ter = lambda ms: 0 if ms <= q1 else (1 if ms <= q2 else 2)
    E = eyes[eyes.subject.isin(subj)].sort_values("eye_id").reset_index(drop=True)
    out = []
    for i, e in E.iterrows():
        v = visits[visits.eye_id == e.eye_id].sort_values("visit")
        l = vl[vl.eye_id == e.eye_id]
        Y = l.pivot_table(index="visit", columns="loc59", values="y").sort_index().to_numpy()
        C = l.pivot_table(index="visit", columns="loc59", values="censored").sort_index().to_numpy().astype(bool)
        tt = int(ter(e.baseline_ms))
        base, _, _ = sc.tobit_template(v.t.to_numpy(), Y, C, nm, tt, tuple(cfg["simulator"]["template_baseline_clip_db"]))
        out.append(sc.Template(eye_id=e.eye_id, subject=str(e.subject), index=int(i), tertile=tt, baseline=base,
                               t_real=v.t.to_numpy(float), iop_real=v.iop.to_numpy(float), age0=float(e.age_baseline)))
    # 补抽用的间隔池：开发集每只眼全部相邻检查间隔（第 3.4 节；sim_test 也用开发集的池）
    dv = visits[visits.subject.isin(set(spl["dev_subjects"]))].sort_values(["eye_id", "visit"])
    pool = np.concatenate([np.diff(g.t.to_numpy(float)) for _, g in dv.groupby("eye_id")])
    return out, np.sort(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["sim_train", "sim_test"])
    ap.add_argument("--variant", choices=["main", "SA8", "SA10"], default="main")
    ap.add_argument("--confirm-r5-published", action="store_true")
    ap.add_argument("--pointsets-only", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(); geo = sc.make_geometry(pd.read_csv(P("data", "g1_locations.csv")))
    s3c, ang, s4c, elig = pointsets(cfg, geo)
    if a.pointsets_only:
        js = dict(rule_constants={k: getattr(sc, k) for k in dir(sc) if k.startswith(("JANS_", "S3_", "NEIGHBOR_"))},
                  phi0_by_loc59=ang, s3_candidates=s3c,
                  s4_eligible_loc59=elig, s4_candidates={str(n): v for n, v in s4c.items()},
                  counts=dict(s3={str(n): sum(c["n_clusters"] == n for c in s3c) for n in (2, 3)},
                              s4={str(n): len(v) for n, v in s4c.items()}))
        os.makedirs(P("results"), exist_ok=True)
        with open(P("results", "sim_pointsets.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(js, f, ensure_ascii=False, indent=1); f.write("\n")
        print(json.dumps(js["counts"], ensure_ascii=False)); return
    if not a.split or not a.confirm_r5_published:
        sys.exit("需要 --split 和 --confirm-r5-published。")
    guard(cfg, a.split)
    S = cfg["simulator"]
    out = P("data", f"{a.split}{'' if a.variant == 'main' else '_' + a.variant}.npz")
    if a.split == "sim_test" and os.path.exists(out):
        sys.exit(f"拒绝覆盖已存在的 {out}（封存文件）。")
    nmj = json.load(open(P("data", "noise_model.json"), encoding="utf-8"))
    nm0 = sc.NoiseModel.from_json(nmj)                     # 模板用未缩放的噪声模型（R5 规则 T-2）
    cal = P("data", "sim_calibration.json")
    kap = json.load(open(cal, encoding="utf-8")) if os.path.exists(cal) else {}
    nm = sc.NoiseModel.from_json(nmj, kap.get("kappa_offset", 1.0), kap.get("kappa_point", 1.0))
    templates, pool = load_templates(cfg, a.split, nm0)
    seed = S["seed_sim_train"] if a.split == "sim_train" else S["seed_sim_test"]
    n_rep = S["n_rep_sim_train"] if a.split == "sim_train" else S["n_rep_sim_test"]
    arr = sc.assemble(templates, sc.SCENARIOS, n_rep, seed, geo, nm, S, s3c, s4c, pool,
                      independent_noise=(a.variant == "SA10"), no_offset=(a.variant == "SA8"))
    sc.write_npz_deterministic(out, arr)
    h = sc.sha256_file(out)
    os.makedirs(P("logs"), exist_ok=True)
    with open(P("logs", f"{os.path.basename(out)}.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  data/{os.path.basename(out)}\nseed={seed} n_rep={n_rep} n_templates={len(templates)} "
                f"kappa_offset={nm.kappa_offset} kappa_point={nm.kappa_point} numpy={np.__version__}\n")
    print(h, out)


if __name__ == "__main__":
    main()
