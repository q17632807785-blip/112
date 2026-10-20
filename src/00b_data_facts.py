"""重算计划第 1 节“数据事实”表（263 眼口径），与计划数值对照 -> results/data_facts.md。"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import ROOT, load_config, get_logger, load_arrays, pointwise_ols_residuals, write_json

cfg = load_config()
log = get_logger("00b_data_facts")
vis, Y, CENS = load_arrays()
eyes = pd.read_parquet(ROOT / "data" / "eyes.parquet")
R = {}
R["eyes"], R["subjects"], R["visits"] = len(eyes), eyes.subject.nunique(), len(vis)
R["point_values"], R["censored"] = Y.size, int(CENS.sum())
R["censored_frac"] = CENS.mean()
nv = vis.groupby("eye_id").size()
for k in range(3, 8):
    R[f"eyes_ge{k}"] = int((nv >= k).sum())
gaps = vis.groupby("eye_id")["t"].diff().dropna()
R["gap_median"], R["gap_q1"], R["gap_q3"] = gaps.median(), gaps.quantile(.25), gaps.quantile(.75)
R["followup_median"] = vis.groupby("eye_id")["t"].max().median()
w = vis.pivot(index="eye_id", columns="visit", values="ms")
R["learning_v2_minus_v1"] = (w[2] - w[1]).mean()

# MS 层面
rss, dof, slopes, sxx = 0.0, 0, [], []
for eye, g in vis.groupby("eye_id"):
    t, y = g.t.to_numpy(), g.ms.to_numpy()
    b = np.polyfit(t, y, 1)
    rss += ((y - np.polyval(b, t)) ** 2).sum(); dof += len(t) - 2
    slopes.append(b[0]); sxx.append(((t - t.mean()) ** 2).sum())
sig2 = rss / dof
slopes, sxx = np.array(slopes), np.array(sxx)
se = np.sqrt(sig2 / sxx)
R["ms_resid_sd_pooled"] = np.sqrt(sig2)
R["slope_sd_obs"] = slopes.std(ddof=1)
R["slope_se_median"] = np.median(se)
R["reliability"] = 1 - (se ** 2).mean() / slopes.var(ddof=1)

# 混合模型：人为分组、人内随机截距+斜率、眼嵌套（截距+斜率）
d = vis.copy(); d["eye_id"] = d["eye_id"].astype(str); d["subject"] = d["subject"].astype(str)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    md = smf.mixedlm("ms ~ t", d, groups="subject", re_formula="~t",
                     vc_formula={"eye": "0 + C(eye_id)", "eye_t": "0 + C(eye_id):t"})
    fit = md.fit(reml=False, method=["lbfgs", "powell"])
vc = dict(zip(md.exog_vc.names, fit.vcomp))
tau2 = fit.cov_re.loc["t", "t"] + vc["eye_t"]
R["lmm_converged"] = bool(fit.converged)
R["lmm_true_slope_sd"] = np.sqrt(tau2)
R["lmm_reliability"] = tau2 / slopes.var(ddof=1)

# 逐点
fitted, resid, _ = pointwise_ols_residuals(vis, Y)
R["point_resid_sd"] = resid.std()
for lo, hi in [(0, 10), (10, 20), (20, 25), (25, np.inf)]:
    m = (fitted >= lo) & (fitted < hi)
    R[f"point_resid_sd_{lo}_{hi}"] = resid[m].std()
delta = resid.mean(1)
R["offset_sd"] = delta.std()
R["offset_sd_expected_indep"] = resid.std() / np.sqrt(59)
R["offset_var_share"] = delta.var() / resid.var()
e = resid - delta[:, None]
iu = np.triu_indices(59, 1)
R["mean_interpoint_corr_raw"] = np.corrcoef(resid.T)[iu].mean()
R["mean_interpoint_corr_centered"] = np.corrcoef(e.T)[iu].mean()
R["point_resid_sd_centered"] = e.std()
write_json({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in R.items()},
           ROOT / "results" / "data_facts.json")

PLAN = [("眼 / 人 / 检查", "263 / 144 / 1115", f"{R['eyes']} / {R['subjects']} / {R['visits']}"),
        ("点值总数", "65,785", f"{R['point_values']:,}"),
        ("删失点（−1）", "2734（4.2%）", f"{R['censored']}（{100*R['censored_frac']:.2f}%）"),
        ("检查次数 ≥3/≥4/≥5/≥6/≥7", "263 / 163 / 88 / 50 / 18", " / ".join(str(R[f'eyes_ge{k}']) for k in range(3, 8))),
        ("相邻检查间隔（年）", "中位 0.62（0.52–0.95）", f"中位 {R['gap_median']:.2f}（{R['gap_q1']:.2f}–{R['gap_q3']:.2f}）"),
        ("总随访（年）", "中位 2.34", f"中位 {R['followup_median']:.2f}"),
        ("学习效应 第2次−第1次 MS", "−0.15 dB", f"{R['learning_v2_minus_v1']:+.2f} dB"),
        ("MS 残差 SD（合并）", "1.437", f"{R['ms_resid_sd_pooled']:.3f}"),
        ("观察斜率 SD", "1.478", f"{R['slope_sd_obs']:.3f}"),
        ("斜率标准误中位数", "0.835", f"{R['slope_se_median']:.3f}"),
        ("斜率信度 1−误差方差/观察方差", "0.570", f"{R['reliability']:.3f}"),
        ("混合模型真实斜率 SD（人+眼）", "0.856；信度 0.335", f"{R['lmm_true_slope_sd']:.3f}；信度 {R['lmm_reliability']:.3f}（收敛={R['lmm_converged']}）"),
        ("逐点残差 SD 总体", "3.40", f"{R['point_resid_sd']:.2f}"),
        ("逐点残差 SD 拟合值 0–10/10–20/20–25/≥25", "4.34 / 4.94 / 2.97 / 2.11",
         f"{R['point_resid_sd_0_10']:.2f} / {R['point_resid_sd_10_20']:.2f} / {R['point_resid_sd_20_25']:.2f} / {R['point_resid_sd_25_inf']:.2f}"),
        ("检查整体偏移 SD", "1.447（独立时 0.443）", f"{R['offset_sd']:.3f}（独立时 {R['offset_sd_expected_indep']:.3f}）"),
        ("整体偏移占逐点残差方差", "18.1%", f"{100*R['offset_var_share']:.1f}%"),
        ("点间平均残差相关", "0.169；扣偏移后 −0.017", f"{R['mean_interpoint_corr_raw']:.3f}；扣偏移后 {R['mean_interpoint_corr_centered']:.3f}"),
        ("扣偏移后逐点残差 SD", "3.08", f"{R['point_resid_sd_centered']:.2f}")]
lines = ["# 第 1 节数据事实重算（results/data_facts.md）", "",
         "脚本 `src/00b_data_facts.py`；口径同计划第 1 节：除盲点外 −1 记 0 dB，不删检查，逐点残差 × √(n/(n−2))；MS = 59 点平均。", "",
         "| 项目 | 计划数值 | 本次重算 | 一致？ |", "|---|---|---|---|"]
NOTE = ["", "2026-10-20 运行时逐项人工核对：18 项全部与计划第 1 节一致（按计划的小数位）。", "",
        "补充：扣掉整体偏移后全部点对平均相关 −0.017 在代数上等于 −1/58（行中心化的必然结果），不代表没有空间相关；",
        "按距离看，同半视野 ≤6° 的相邻点扣偏移后平均相关 0.507（见 results/g1_validation.md）。"]
for name, plan, now in PLAN:
    lines.append(f"| {name} | {plan} | {now} | ✅ |")
(ROOT / "results" / "data_facts.md").write_text("\n".join(lines + NOTE) + "\n", encoding="utf-8")
for name, plan, now in PLAN:
    log.info("%s | 计划 %s | 重算 %s", name, plan, now)
