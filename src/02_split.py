"""02_split.py — 预注册第 3.2 节 / 第 6 部分第 1 条：按人划分开发集/留出集，开发集内 5 折，全体 144 人 5 折。
分层 = 人的基线 MS 三分位（该人各眼第 1 次检查 MS 的平均）× 是否有 ≥5 次检查的眼。种子来自 config.yaml。
各层内：按人号数值排序 → 用种子打乱 → 前 n_hold 个进留出集；n_hold 用最大余数法分配，使总数 = round(0.30×144) = 43。"""
import os, json
import numpy as np, pandas as pd, yaml
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
S = cfg["split"]; SEED = cfg["seed"]
eyes = pd.read_parquet(os.path.join(ROOT, "data", "eyes.parquet"))
subj = eyes.groupby("subject").agg(baseline_ms=("baseline_ms", "mean"), max_visits=("n_visits", "max")).reset_index()
subj["sid_num"] = subj.subject.astype(int); subj = subj.sort_values("sid_num").reset_index(drop=True)
subj["ms_tertile"] = pd.qcut(subj.baseline_ms, 3, labels=False)
subj["has_ge5"] = (subj.max_visits >= 5).astype(int)
subj["stratum"] = subj.ms_tertile.astype(str) + "_" + subj.has_ge5.astype(str)
rng = np.random.default_rng(SEED)
N = len(subj); n_hold_total = int(round(S["holdout_frac"] * N))
strata = sorted(subj.stratum.unique())
sizes = subj.groupby("stratum").size().loc[strata]
quota = sizes * S["holdout_frac"]; base = np.floor(quota).astype(int)
rem = n_hold_total - base.sum(); order = (quota - base).sort_values(ascending=False, kind="stable").index[:rem]
base.loc[order] += 1
hold = []
perm_by_stratum = {}
for st in strata:
    ids = subj.loc[subj.stratum == st, "subject"].tolist()
    p = [ids[i] for i in rng.permutation(len(ids))]; perm_by_stratum[st] = p
    hold += p[: base.loc[st]]
hold = set(hold); dev = [s for s in subj.subject if s not in hold]; hold_l = [s for s in subj.subject if s in hold]
assert len(dev) + len(hold_l) == N and not set(dev) & hold

def folds(ids, k):
    """各层内打乱后轮流分到 k 折（按层依次接续，保证各折大小差 ≤1、层内均衡）。"""
    sub = subj[subj.subject.isin(ids)]; out = [[] for _ in range(k)]; j = 0
    for st in strata:
        s_ids = sub.loc[sub.stratum == st, "subject"].tolist()
        for i in rng.permutation(len(s_ids)):
            out[j % k].append(s_ids[i]); j += 1
    return [sorted(f, key=int) for f in out]

dev_folds = folds(dev, S["dev_n_folds"]); all_folds = folds(subj.subject.tolist(), S["all_n_folds"])
split = {"seed": SEED, "strata": S["strata"],
         "dev_subjects": sorted(dev, key=int), "holdout_subjects": sorted(hold_l, key=int),
         "dev_folds": dev_folds, "all_folds": all_folds}
with open(os.path.join(ROOT, "data", "split.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(split, f, indent=1, ensure_ascii=False); f.write("\n")

e = eyes.copy(); e["set"] = np.where(e.subject.isin(hold), "holdout", "dev")
ks_eye = stats.ks_2samp(e.loc[e.set == "dev", "baseline_ms"], e.loc[e.set == "holdout", "baseline_ms"])
ks_sub = stats.ks_2samp(subj.loc[~subj.subject.isin(hold), "baseline_ms"], subj.loc[subj.subject.isin(hold), "baseline_ms"])
mw = stats.mannwhitneyu(e.loc[e.set == "dev", "n_visits"], e.loc[e.set == "holdout", "n_visits"], alternative="two-sided")
summary = dict(n_dev_subjects=len(dev), n_holdout_subjects=len(hold_l),
               n_dev_eyes=int((e.set == "dev").sum()), n_holdout_eyes=int((e.set == "holdout").sum()),
               ks_baseline_ms_eye_p=float(ks_eye.pvalue), ks_baseline_ms_subject_p=float(ks_sub.pvalue),
               mw_visits_per_eye_p=float(mw.pvalue),
               dev_eyes_ge5=int(((e.set == "dev") & (e.n_visits >= 5)).sum()), holdout_eyes_ge5=int(((e.set == "holdout") & (e.n_visits >= 5)).sum()),
               strata_sizes=sizes.to_dict(), holdout_per_stratum=base.to_dict(),
               dev_fold_sizes=[len(f) for f in dev_folds], all_fold_sizes=[len(f) for f in all_folds])
json.dump(summary, open(os.path.join(ROOT, "results", "split_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
assert sorted(sum(dev_folds, []), key=int) == split["dev_subjects"]
assert sorted(sum(all_folds, []), key=int) == sorted(subj.subject, key=int)
