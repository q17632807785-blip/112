"""5.3 节：按人划分 -> data/split.json（含平衡日志）。

开发集 70% / 留出集 30%，按“人的基线 MS 三分位 × 是否有 ≥5 次检查的眼”分层随机，种子来自 config。
开发集内再按人分 5 折；另做全体 144 人的 5 折（只用于真实数据终点）。
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import ROOT, load_config, get_logger, write_json

cfg = load_config()
sc = cfg["split"]
seed = cfg["seed"]
log = get_logger("02_split")
log.info("seed=%d dev_frac=%.2f", seed, sc["dev_frac"])
rng = np.random.default_rng(seed)

eyes = pd.read_parquet(ROOT / "data" / "eyes.parquet")
subj = eyes.groupby("subject").agg(ms_baseline=("ms_baseline", "mean"), max_visits=("n_visits", "max"),
                                   n_eyes=("eye_id", "size")).reset_index()
subj["ms_tertile"] = pd.qcut(subj.ms_baseline, 3, labels=[0, 1, 2]).astype(int)
subj["ge5"] = (subj.max_visits >= 5).astype(int)
subj["stratum"] = subj.ms_tertile.astype(str) + "_" + subj.ge5.astype(str)
subj = subj.sort_values("subject", key=lambda s: s.astype(int)).reset_index(drop=True)

dev, hold = [], []
for st, g in subj.groupby("stratum", sort=True):
    ids = g.subject.to_numpy().copy()
    rng.shuffle(ids)
    k = int(round(sc["dev_frac"] * len(ids)))
    dev += list(ids[:k]); hold += list(ids[k:])
    log.info("层 %s：%d 人 -> 开发 %d / 留出 %d", st, len(ids), k, len(ids) - k)
assert not set(dev) & set(hold) and len(dev) + len(hold) == 144


def stratified_folds(ids, n_folds, rng):
    sub = subj.set_index("subject").loc[ids]
    folds = [[] for _ in range(n_folds)]
    offset = 0
    for st, g in sub.groupby("stratum", sort=True):
        s = g.index.to_numpy().copy(); rng.shuffle(s)
        for i, x in enumerate(s):
            folds[(i + offset) % n_folds].append(str(x))
        offset += len(s)
    return [sorted(f, key=int) for f in folds]


dev_folds = stratified_folds(dev, sc["dev_n_folds"], rng)
all_folds = stratified_folds(list(subj.subject), sc["all_n_folds"], rng)
assert sorted(sum(dev_folds, []), key=int) == sorted(dev, key=int)
assert sorted(sum(all_folds, []), key=int) == sorted(subj.subject, key=int)

# 平衡
eyes["set"] = np.where(eyes.subject.isin(dev), "dev", "holdout")
bal = {}
lines = ["# 人群划分平衡日志", "", f"种子 {seed}；分层：人的基线 MS（双眼平均）三分位 × 是否有 ≥5 次检查的眼。", "",
         "| 指标 | 开发集 | 留出集 | 检验 |", "|---|---|---|---|"]
d, h = eyes[eyes.set == "dev"], eyes[eyes.set == "holdout"]
rows = [("人数", f"{len(dev)}", f"{len(hold)}", ""),
        ("眼数", f"{len(d)}", f"{len(h)}", ""),
        ("检查次数", f"{d.n_visits.sum()}", f"{h.n_visits.sum()}", "")]
ks_ms = stats.ks_2samp(d.ms_baseline, h.ms_baseline)
mw_nv = stats.mannwhitneyu(d.n_visits, h.n_visits)
rows += [("基线 MS 中位（IQR）", f"{d.ms_baseline.median():.1f}（{d.ms_baseline.quantile(.25):.1f}–{d.ms_baseline.quantile(.75):.1f}）",
          f"{h.ms_baseline.median():.1f}（{h.ms_baseline.quantile(.25):.1f}–{h.ms_baseline.quantile(.75):.1f}）", f"KS p={ks_ms.pvalue:.2f}"),
         ("基线 MS 均值±SD", f"{d.ms_baseline.mean():.1f}±{d.ms_baseline.std():.1f}", f"{h.ms_baseline.mean():.1f}±{h.ms_baseline.std():.1f}", ""),
         ("每眼检查次数 中位（均值）", f"{d.n_visits.median():.0f}（{d.n_visits.mean():.2f}）", f"{h.n_visits.median():.0f}（{h.n_visits.mean():.2f}）", f"MWU p={mw_nv.pvalue:.2f}")]
for k in (4, 5, 6):
    rows.append((f"≥{k} 次检查的眼", f"{(d.n_visits>=k).sum()}（{100*(d.n_visits>=k).mean():.0f}%）",
                 f"{(h.n_visits>=k).sum()}（{100*(h.n_visits>=k).mean():.0f}%）", ""))
rows.append(("PLR2 进展眼", f"{int(d.PLR2.sum())}（{100*d.PLR2.mean():.0f}%）", f"{int(h.PLR2.sum())}（{100*h.PLR2.mean():.0f}%）", ""))
rows.append(("闭角型（ACG）眼", f"{(d.category=='ACG').sum()}", f"{(h.category=='ACG').sum()}", ""))
rows.append(("随访年数 中位", f"{d.followup_years.median():.2f}", f"{h.followup_years.median():.2f}", ""))
for r in rows:
    lines.append("| " + " | ".join(r) + " |")
lines += ["", "开发集 5 折人数：" + " / ".join(str(len(f)) for f in dev_folds),
          "全体 5 折人数：" + " / ".join(str(len(f)) for f in all_folds),
          "", "验收：两组的人无交集 ✅；基线 MS 与检查次数分布接近（见上表检验 p 值）。"]
(ROOT / "logs" / "02_split_balance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
for r in rows:
    log.info("%s | dev %s | holdout %s | %s", *r)

out = dict(seed=seed, created="2026-10-20", strata="baseline_ms_tertile x has_eye_ge5_visits",
           dev_subjects=sorted(map(str, dev), key=int), holdout_subjects=sorted(map(str, hold), key=int),
           dev_eyes=sorted(d.eye_id), holdout_eyes=sorted(h.eye_id),
           dev_folds=dev_folds, all_folds=all_folds,
           balance=dict(ks_ms_baseline_p=ks_ms.pvalue, mwu_n_visits_p=mw_nv.pvalue))
write_json(out, ROOT / "data" / "split.json")
log.info("写出 data/split.json：开发 %d 人 %d 眼，留出 %d 人 %d 眼", len(dev), len(d), len(hold), len(h))
