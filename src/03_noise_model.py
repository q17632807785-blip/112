"""03_noise_model.py — 预注册第 3.3 节 / 第 6 部分第 4 条：噪声模型（只用开发集）。
1) 检查级整体偏移 δ_v（59 点残差平均），按眼基线 MS 三分位分层；去卷积：扣掉逐点噪声（含空间相关）对 δ_v 的平均贡献。
2) 逐点特有噪声 e = r − δ_v：按拟合值分箱 SD + 平滑样条 SD(f̂)。
3) 空间相关：同半视野内 ρ(d) = a·exp(−d/λ)，从 e 的点对相关拟合；跨水平中线不相关；生成时中心点并入上半视野块。
4) 删失：模拟值 < 0 记 0。
自检：用噪声模型（R1 之前的模板口径：OLS 拟合值截到 [0,35]）在开发集真实时间表上生成一次数据，按同样算法重估偏移 SD、删失比例。
只涉及噪声模型本身，不运行任何参赛方法，不生成 sim_train / sim_test。"""
import os, json
import numpy as np, pandas as pd, yaml
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8")); NM = cfg["noise_model"]
D = lambda f: os.path.join(ROOT, "data", f)
split = json.load(open(D("split.json"))); dev = set(split["dev_subjects"])
vl = pd.read_parquet(D("vf_long.parquet")); eyes = pd.read_parquet(D("eyes.parquet"))
loc = pd.read_csv(D("g1_locations.csv")); pts = loc[~loc.is_blind_spot].sort_values("loc59")
XY = pts[["x_deg", "y_deg"]].to_numpy()
vl = vl[vl.subject.isin(dev)]; eyes = eyes[eyes.subject.isin(dev)].reset_index(drop=True)

Y = vl.pivot_table(index=["eye_id", "visit"], columns="loc59", values="y").sort_index()
Cn = vl.pivot_table(index=["eye_id", "visit"], columns="loc59", values="censored").loc[Y.index].to_numpy().astype(bool)
T = vl.groupby(["eye_id", "visit"])["t"].first().loc[Y.index].to_numpy()
Yv = Y.to_numpy(); eye_of = Y.index.get_level_values(0).to_numpy()
F = np.zeros_like(Yv); R = np.zeros_like(Yv); groups = pd.Series(range(len(Y))).groupby(eye_of).indices
for eid, idx in groups.items():
    t = T[idx]; n = len(t); X = np.c_[np.ones(n), t]; b = np.linalg.lstsq(X, Yv[idx], rcond=None)[0]
    F[idx] = X @ b; R[idx] = (Yv[idx] - F[idx]) * np.sqrt(n / (n - 2))
delta = R.mean(1); E = R - delta[:, None]

# ---- 1. 偏移 SD 按基线 MS 三分位（眼级，开发集眼） ----
eyes["tertile"] = pd.qcut(eyes.baseline_ms, 3, labels=False)
cuts = [float(eyes.baseline_ms.quantile(q)) for q in (1 / 3, 2 / 3)]
ter = eyes.set_index("eye_id").loc[eye_of, "tertile"].to_numpy()
rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
off_sd = [rms(delta[ter == k]) for k in range(3)]
off_sd_all = rms(delta)
rel_diff = (max(off_sd) - min(off_sd)) / min(off_sd)
stratify = rel_diff >= NM["offset_constant_if_rel_diff_below"]

# ---- 2. 逐点 SD：分箱 + 平滑样条 ----
fb = np.array(NM["fitted_bins_db"], float); fh = np.clip(F.ravel(), 0, None); ee = E.ravel()
bin_idx = np.digitize(fh, fb[1:-1])
bins = [dict(lo=float(fb[i]), hi=float(fb[i + 1]), n=int((bin_idx == i).sum()), sd=rms(ee[bin_idx == i]),
             mean_fitted=float(fh[bin_idx == i].mean())) for i in range(len(fb) - 1)]
fine_edges = np.arange(0, 37, 1.0); fi = np.digitize(fh, fine_edges[1:-1])
fx, fy, fw = [], [], []
for i in range(len(fine_edges) - 1):
    m = fi == i
    if m.sum() >= 30: fx.append(fh[m].mean()); fy.append(rms(ee[m])); fw.append(np.sqrt(m.sum()))
fx, fy, fw = map(np.array, (fx, fy, fw))
spl = UnivariateSpline(fx, fy, w=fw / fw.mean(), k=3, s=len(fx))
grid = np.arange(0, 40.01, 0.5); gsd = spl(np.clip(grid, fx.min(), fx.max()))
sd_point = lambda f: np.interp(np.clip(f, 0, None), grid, gsd)

# ---- 3. 空间相关 ----
C = np.corrcoef(E.T); dist = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1)); iu = np.triu_indices(59, 1)
same = (XY[:, 1][:, None] * XY[:, 1][None]) > 0                      # 严格同半视野（拟合时不含中心点）
pair = same[iu]; dd = dist[iu][pair]; cc = C[iu][pair]
expf = lambda d, a, lam: a * np.exp(-d / lam)
(a_hat, lam_hat), _ = curve_fit(expf, dd, cc, p0=[0.8, 5], bounds=([0, 0.1], [1, 100]))
nb = (dist <= 6) & (XY[:, 1][:, None] * XY[:, 1][None] >= 0)
dev_nb_corr = float(C[iu][nb[iu]].mean()); dev_ot_corr = float(C[iu][~nb[iu]].mean())
use_spatial = dev_nb_corr >= NM["spatial_corr_threshold"]
block = np.where(XY[:, 1] >= 0, 0, 1)                                   # 生成时中心点并入上半视野块
Rho = np.where(block[:, None] == block[None], a_hat * np.exp(-dist / lam_hat), 0.0); np.fill_diagonal(Rho, 1.0)
if not use_spatial: Rho = np.eye(59)

# ---- 1b. 去卷积：δ_v 中逐点噪声的平均贡献 Var(mean e*) = s'Ρs / 59² ----
S = sd_point(F)
contrib = np.einsum("vi,ij,vj->v", S, Rho, S) / 59 ** 2
off_sd_deconv = [float(np.sqrt(max(0.0, off_sd[k] ** 2 - contrib[ter == k].mean()))) for k in range(3)]
off_sd_deconv_all = float(np.sqrt(max(0, off_sd_all ** 2 - contrib.mean())))

real_censor = float(Cn.mean())

# ---- 自检模拟（只针对噪声模型） ----
rng = np.random.default_rng(cfg["seed"])
L = np.linalg.cholesky(Rho + 1e-10 * np.eye(59))
tmpl = np.clip(F, 0, 35)                                    # R1 之前的模板口径：OLS 拟合值，下限 0 dB
sd_off_v = np.array(off_sd_deconv if stratify else [off_sd_deconv_all] * 3)[ter]
sim = tmpl + sd_off_v[:, None] * rng.standard_normal((len(F), 1)) + (rng.standard_normal((len(F), 59)) @ L.T) * sd_point(tmpl)
sim_c = sim < 0; sim = np.where(sim_c, 0, sim)
Rs = np.zeros_like(sim)
for eid, idx in groups.items():
    t = T[idx]; n = len(t); X = np.c_[np.ones(n), t]; b = np.linalg.lstsq(X, sim[idx], rcond=None)[0]
    Rs[idx] = (sim[idx] - X @ b) * np.sqrt(n / (n - 2))
ds = Rs.mean(1); Es = Rs - ds[:, None]
selfcheck = dict(offset_sd_sim=rms(ds), offset_sd_real=off_sd_all, offset_sd_sim_by_tertile=[rms(ds[ter == k]) for k in range(3)],
                 censor_frac_sim=float(sim_c.mean()), censor_frac_real=real_censor,
                 point_sd_sim=rms(Es), point_sd_real=rms(E),
                 neighbor_corr_sim=float(np.corrcoef(Es.T)[iu][nb[iu]].mean()))

model = {
    "description": "GRAPE 方向 A 噪声模型（只用开发集；预注册第 3.3 节）。R1 之前的模板口径（OLS 下限 0 dB）。",
    "seed": cfg["seed"], "n_dev_subjects": len(dev), "n_dev_eyes": int(len(eyes)), "n_dev_visits": int(len(Y)),
    "dof_correction": NM["dof_correction"],
    "offset": {"strata": NM["offset_strata"], "baseline_ms_tertile_cuts_db": [round(c, 4) for c in cuts],
               "sd_observed_by_tertile": [round(x, 4) for x in off_sd], "sd_observed_pooled": round(off_sd_all, 4),
               "rel_diff": round(rel_diff, 4), "stratified": bool(stratify),
               "sd_deconvolved_by_tertile": [round(x, 4) for x in off_sd_deconv], "sd_deconvolved_pooled": round(off_sd_deconv_all, 4),
               "deconvolution": "sd_deconv^2 = sd_obs^2 - mean_v( s_v' Rho s_v / 59^2 )，s_v = SD_point(f_hat)"},
    "point_sd": {"bins": [dict((k, round(v, 4) if isinstance(v, float) else v) for k, v in b.items()) for b in bins],
                 "spline": {"type": "scipy UnivariateSpline k=3 on 1-dB bins (n>=30), weights sqrt(n), s=n_bins",
                            "eval": "np.interp(clip(f,0,None), grid_db, sd_db)",
                            "grid_db": grid.round(2).tolist(), "sd_db": [round(float(v), 4) for v in gsd]},
                 "sd_overall": round(rms(E), 4)},
    "spatial": {"model": "rho(d) = a*exp(-d/lambda) within hemifield; 0 across horizontal midline; center joins superior block",
                "a": round(float(a_hat), 4), "lambda_deg": round(float(lam_hat), 4), "fit": "least squares on strictly same-hemifield pair correlations of e (dev)",
                "dev_neighbor_corr": round(dev_nb_corr, 4), "dev_other_corr": round(dev_ot_corr, 4), "used": bool(use_spatial)},
    "censoring": {"rule": "sim < 0 -> 0, censored=True", "dev_censor_frac": round(real_censor, 4)},
    "selfcheck": {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in selfcheck.items()},
}
with open(D("noise_model.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(model, f, indent=1, ensure_ascii=False); f.write("\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(fx, fy, "o", ms=3, label="1-dB bins"); ax[0].plot(grid, gsd, "-", label="spline")
ax[0].plot([b["mean_fitted"] for b in bins], [b["sd"] for b in bins], "s", label="spec bins"); ax[0].set_xlabel("fitted (dB)"); ax[0].set_ylabel("SD e (dB)"); ax[0].legend()
for k in range(3): ax[1].hist(delta[ter == k], bins=40, histtype="step", label=f"tertile {k}")
ax[1].set_xlabel("visit offset δ (dB)"); ax[1].legend()
ax[2].plot(dd, cc, ".", ms=2, alpha=.4); xs = np.linspace(0, 50, 200); ax[2].plot(xs, expf(xs, a_hat, lam_hat), "r-")
ax[2].set_xlabel("distance (deg), same hemifield"); ax[2].set_ylabel("corr(e)")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "noise_model.png"), dpi=100)
print(json.dumps({k: model[k] for k in ("offset", "spatial", "censoring", "selfcheck")}, indent=1, ensure_ascii=False))
print("point_sd bins:", [(b["lo"], b["n"], round(b["sd"], 2)) for b in bins], "overall", round(rms(E), 3))
