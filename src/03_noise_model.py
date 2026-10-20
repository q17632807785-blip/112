"""5.4 节：参数化噪声模型（只用开发集）-> data/noise_model.json、figures/noise_model.png。

噪声 = 检查级整体偏移 δ_v + 随敏感度变化的逐点噪声 e（必要时带空间相关）+ 0 dB 删失。
"""
import json
import numpy as np
import pandas as pd
from scipy import optimize, interpolate, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from common import ROOT, load_config, get_logger, load_arrays, pointwise_ols_residuals, write_json

cfg = load_config()
nm = cfg["noise_model"]
seed = cfg["seed"]
log = get_logger("03_noise_model")
rng = np.random.default_rng(seed)
split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
dev_eyes = split["dev_eyes"]
vis, Y, CENS = load_arrays(dev_eyes)
assert set(vis.eye_id) == set(dev_eyes) and not set(vis.subject) & set(split["holdout_subjects"])
log.info("seed=%d；只用开发集：%d 人 %d 眼 %d 次检查", seed, vis.subject.nunique(), vis.eye_id.nunique(), len(vis))

L = pd.read_csv(ROOT / "data" / "g1_locations.csv")
P = L[~L.is_blind_spot].sort_values("loc").reset_index(drop=True)
X, Yc, hf = P.x_deg.to_numpy(), P.y_deg.to_numpy(), P.hemifield.to_numpy(dtype=object)

fitted, resid, _ = pointwise_ols_residuals(vis, Y)
delta = resid.mean(1)
e = resid - delta[:, None]

# ---------- 1. 整体偏移 ----------
eyes = pd.read_parquet(ROOT / "data" / "eyes.parquet").set_index("eye_id")
bms = vis.eye_id.map(eyes.ms_baseline).to_numpy()
eye_bms = eyes.loc[dev_eyes, "ms_baseline"]
cuts = np.quantile(eye_bms, [1 / 3, 2 / 3])
tert = np.digitize(bms, cuts)
off_sd_obs = delta.std(ddof=1)
off_by_t = [delta[tert == k].std(ddof=1) for k in range(3)]
rel = (max(off_by_t) - min(off_by_t)) / min(off_by_t)
off_const = bool(rel < nm["offset_constant_if_rel_diff_below"])
log.info("整体偏移 SD（观察）%.3f；按基线 MS 三层 %s；相对差 %.1f%% -> %s", off_sd_obs, np.round(off_by_t, 3), 100 * rel,
         "用常数" if off_const else "分层")

# ---------- 2. 逐点特有噪声 SD(f_hat) ----------
edges = nm["fitted_bins_db"]
fc = np.clip(fitted, 0, None)
bins = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (fc >= lo) & (fc < hi)
    bins.append(dict(lo=lo, hi=(None if np.isinf(hi) else hi), n=int(m.sum()), sd=float(e[m].std()),
                     sd_raw_resid=float(resid[m].std())))
# 平滑样条：1 dB 细分箱的 e² 均值（按点数加权），GCV 选平滑度
fine = np.arange(0, 37, 1.0)
xc, v2, wt = [], [], []
for a in fine:
    m = (fc >= a) & (fc < a + 1)
    if m.sum() >= 30:
        xc.append(a + 0.5); v2.append((e[m] ** 2).mean()); wt.append(m.sum())
xc, v2, wt = map(np.array, (xc, v2, wt))
spl = interpolate.make_smoothing_spline(xc, np.sqrt(v2), w=wt / wt.mean())
grid = np.arange(0, 40.5, 0.5)
sd_grid = np.clip(spl(np.clip(grid, xc.min(), xc.max())), 0.5, None)
log.info("逐点 SD 分箱：%s", [(b["lo"], round(b["sd"], 2), b["n"]) for b in bins])
log.info("扣偏移后逐点残差 SD 总体 %.3f（参考 3.08）", e.std())

# ---------- 3. 邻点相关 ----------
D = np.hypot(X[:, None] - X[None], Yc[:, None] - Yc[None])
same = (hf[:, None] == hf[None]) | (hf[:, None] == "center") | (hf[None] == "center")
iu = np.triu_indices(59, 1)
nbr = (same & (D <= cfg["locations"]["neighbor_max_dist_deg"]))[iu]
Rc = np.corrcoef(e.T)
nb_corr, ot_corr = Rc[iu][nbr].mean(), Rc[iu][~nbr].mean()
thr = nm["spatial_corr_threshold"]
spatial = bool(nb_corr >= thr)
# 指数衰减拟合（同半视野点对）：corr(d) = a·exp(−d/λ) + c，c 吸收行中心化带来的负偏
sv = same[iu]
dd, rr = D[iu][sv], Rc[iu][sv]
(a_hat, lam_hat, c_hat), _ = optimize.curve_fit(lambda d, a, l, c: a * np.exp(-d / l) + c, dd, rr, p0=[0.8, 5, -0.05],
                                                bounds=([0, 0.5, -0.5], [1.0, 50, 0.5]))
cross_short = Rc[iu][(~sv) & (D[iu] <= 9)].mean()
log.info("开发集邻点相关 %.3f vs 其余 %.3f -> 空间相关 %s；指数拟合 a=%.3f λ=%.2f° c=%.3f；跨半视野 ≤9° 平均 %.3f",
         nb_corr, ot_corr, spatial, a_hat, lam_hat, c_hat, cross_short)

# 生成用相关矩阵：半视野内 a·exp(−d/λ)，块外 0，对角 1（块金效应 1−a）；中心点并入上半视野块
blk = np.where(hf == "center", "sup", hf)
Rgen = np.where(blk[:, None] == blk[None], a_hat * np.exp(-D / lam_hat), 0.0) if spatial else np.zeros((59, 59))
np.fill_diagonal(Rgen, 1.0)
eig_min = np.linalg.eigvalsh(Rgen).min()
assert eig_min > 0, "生成用相关矩阵不正定"
Lchol = np.linalg.cholesky(Rgen)

# 去卷积：观察到的 δ̂ 含逐点噪声的平均；真实偏移方差 = var(δ̂) − mean_v var(ē_v)
sd_pt = np.interp(fc, grid, sd_grid)
vebar_v = np.array([(s[:, None] * s[None] * Rgen).sum() / 59 ** 2 for s in sd_pt])
var_ebar = vebar_v.mean()
off_sd_deconv = float(np.sqrt(max(off_sd_obs ** 2 - var_ebar, 0)))
off_deconv_by_t = [float(np.sqrt(max(off_by_t[k] ** 2 - vebar_v[tert == k].mean(), 0))) for k in range(3)]
log.info("去卷积后分层真实偏移 SD：%s", np.round(off_deconv_by_t, 3))
log.info("δ̂ 中逐点噪声平均的贡献 SD %.3f；去卷积后真实偏移 SD %.3f（观察 %.3f）", np.sqrt(var_ebar), off_sd_deconv, off_sd_obs)

# ---------- 4. 删失核对 ----------
real_cens = float(CENS.mean())


# 每只眼每个点：全部检查都删失 -> “绝对暗点”（诊断用变体：潜在敏感度设为 −10 dB）
always = np.zeros_like(CENS)
for eye, idx in vis.groupby("eye_id").indices.items():
    always[idx] = CENS[idx].all(0)[None, :]
log.info("开发集中“每次都删失”的点值占全部点值 %.4f（占删失点 %.1f%%）", always.mean(), 100 * always.sum() / CENS.sum())


def simulate_once(rng, off_sd, use_spatial=True, floor_aware=False):
    f = np.clip(fitted, *cfg["simulator"]["template_baseline_clip_db"])
    if floor_aware:
        f = np.where(always, -10.0, f)
    s = np.interp(f, grid, sd_grid)
    z = rng.standard_normal(f.shape)
    if use_spatial:
        z = z @ Lchol.T
    osd = np.asarray(off_sd)[tert][:, None] if np.ndim(off_sd) else off_sd
    ysim = f + rng.normal(0, 1, (f.shape[0], 1)) * osd + s * z
    return ysim


sims = {}
OFF_SIM = off_sd_deconv if off_const else off_deconv_by_t
for name, osd, sp, fa in [("main: deconv_offset+spatial", OFF_SIM, spatial, False),
                          ("observed_offset_sd+spatial", off_sd_obs if off_const else off_by_t, spatial, False),
                          ("deconv_offset+independent", OFF_SIM, False, False),
                          ("diagnostic: main+floor_aware(always-censored latent −10 dB)", OFF_SIM, spatial, True)]:
    fr, osds = [], []
    for _ in range(20):
        ys = simulate_once(rng, osd, sp, fa)
        fr.append((ys < 0).mean())
        # 模拟数据上重新估计整体偏移 SD（与真实数据同一算法）
        yc = np.maximum(ys, 0)
        fs, rs, _ = pointwise_ols_residuals(vis, yc)
        osds.append(rs.mean(1).std(ddof=1))
        osd_t = [rs.mean(1)[tert == k].std(ddof=1) for k in range(3)]
    sims[name] = dict(censor_frac=float(np.mean(fr)), offset_sd_reestimated=float(np.mean(osds)),
                      offset_sd_by_tertile_last=[float(v) for v in osd_t])
    log.info("模拟核对 %s：删失比例 %.4f（真实开发集 %.4f）；重估整体偏移 SD %.3f（真实 %.3f）",
             name, np.mean(fr), real_cens, np.mean(osds), off_sd_obs)

model = dict(
    created="2026-10-20", seed=seed, data="开发集（split.json dev_subjects）", n_subjects=int(vis.subject.nunique()),
    n_eyes=int(vis.eye_id.nunique()), n_visits=int(len(vis)),
    offset=dict(sd_observed=off_sd_obs, sd_by_baseline_ms_tertile=off_by_t, tertile_cuts_ms=cuts.tolist(),
                rel_diff=rel, use_constant=off_const, sd_deconvolved=off_sd_deconv,
                sd_deconvolved_by_tertile=off_deconv_by_t,
                sd_for_simulation=OFF_SIM,
                note="模拟用去卷积后的 SD；若 5.6 节校准不合格再调。观察值含逐点噪声平均的贡献。"),
    pointwise=dict(sd_overall_centered=float(e.std()), bins=bins, spline_grid_db=grid.tolist(), spline_sd=sd_grid.tolist(),
                   spline="scipy make_smoothing_spline（GCV），1 dB 细分箱 RMS，按点数加权；拟合值 <0 记 0"),
    spatial=dict(neighbor_corr_centered=nb_corr, other_corr_centered=ot_corr, threshold=thr, use_spatial=spatial,
                 kernel="within-hemifield a*exp(-d/lambda)，跨半视野 0；对角 1；中心点并入上半视野块",
                 a=a_hat, lambda_deg=lam_hat, c_offset=c_hat, cross_hemifield_le9deg_corr=cross_short,
                 min_eigenvalue=float(eig_min)),
    censoring=dict(rule="模拟值 < 0 记 0 dB，censored=True", real_dev_frac=real_cens, sim_checks=sims,
                   always_censored_frac_of_values=float(always.mean()),
                   pass_main=bool(abs(sims["main: deconv_offset+spatial"]["censor_frac"] - real_cens) <= 0.01)),
)
write_json(model, ROOT / "data" / "noise_model.json")

# ---------- 图 ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cen = [(b["lo"] + (b["hi"] if b["hi"] is not None else 35)) / 2 for b in bins]
ax[0].plot(cen, [b["sd"] for b in bins], "o", label="分箱 SD（扣偏移）")
ax[0].plot(cen, [b["sd_raw_resid"] for b in bins], "s", mfc="none", label="分箱 SD（未扣偏移）")
ax[0].plot(grid, sd_grid, "-", label="平滑样条 SD(f_hat)")
ax[0].set_xlabel("逐点 OLS 拟合值 f_hat (dB)"); ax[0].set_ylabel("逐点残差 SD (dB)"); ax[0].legend(fontsize=8)
ax[0].set_title("逐点特有噪声随敏感度变化（开发集）")
ax[1].hist(delta, bins=40, density=True, alpha=.6)
xx = np.linspace(delta.min(), delta.max(), 200)
ax[1].plot(xx, stats.norm.pdf(xx, delta.mean(), off_sd_obs), label=f"N(0, {off_sd_obs:.2f}²) 观察")
ax[1].plot(xx, stats.norm.pdf(xx, 0, off_sd_deconv), "--", label=f"去卷积 SD {off_sd_deconv:.2f}")
ax[1].set_xlabel("检查整体偏移 δ_v (dB)"); ax[1].legend(fontsize=8)
ax[1].set_title("检查级整体偏移分布; 三层 SD " + "/".join(f"{s:.2f}" for s in off_by_t))
ax[2].scatter(D[iu][sv], Rc[iu][sv], s=6, alpha=.4, label="同半视野点对")
ax[2].scatter(D[iu][~sv], Rc[iu][~sv], s=6, alpha=.4, label="跨半视野点对")
dg = np.linspace(0, 50, 200)
ax[2].plot(dg, a_hat * np.exp(-dg / lam_hat) + c_hat, "k-", label=f"a·exp(−d/λ)+c, a={a_hat:.2f}, λ={lam_hat:.1f}°")
ax[2].axhline(thr, color="r", ls=":", lw=1, label="0.10 阈值")
ax[2].set_xlabel("点间距离 (°)"); ax[2].set_ylabel("扣偏移后残差相关"); ax[2].legend(fontsize=7)
ax[2].set_title(f"空间相关：相邻(≤6°) {nb_corr:.2f} vs 其余 {ot_corr:.2f}")
plt.tight_layout()
plt.savefig(ROOT / "figures" / "noise_model.png", dpi=150)
log.info("写出 data/noise_model.json、figures/noise_model.png")
