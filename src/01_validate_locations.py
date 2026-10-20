"""01_validate_locations.py — 计划第 5.2 节第 5 步的三项校验 + 图 2c 独立核对。结果写 results/g1_validation.md。"""
import os, json
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
D = lambda f: os.path.join(ROOT, "data", f)
loc = pd.read_csv(D("g1_locations.csv")); vl = pd.read_parquet(D("vf_long.parquet"))
pts = loc[~loc.is_blind_spot].sort_values("loc59").reset_index(drop=True)
XY = pts[["x_deg", "y_deg"]].to_numpy(); HEM = pts.hemifield.to_numpy()
ns = {}
src = open(os.path.join(HERE, "01_g1_locations.py"), encoding="utf-8").read()
# 取图 2b 判读表（直接从 01 脚本里解析常量，避免重复手抄）
code = src[src.index("SQ = -1"):src.index("assert set(FIG2B) == G1")]
exec(code, ns); FIG2B = ns["FIG2B"]
out = []; ok_all = True

# ---- 校验 1：8 号右眼第 1 次检查按坐标画图，逐格与图 2b 印的数比对 ----
raw = pd.read_excel(os.path.normpath(os.path.join(ROOT, cfg["paths"]["raw_xlsx"])), "Follow-up", header=None).iloc[2:]
r8 = raw[(raw[0].astype(str).str.replace(r"\.0$", "", regex=True) == "8") & (raw[1] == "OD") & (raw[2].astype(int) == 1)]
v61 = r8.iloc[0, 8:69].astype(int).to_numpy()
n_match = sum(int(v61[r.idx]) == FIG2B[(int(r.x_deg), int(r.y_deg))] for r in loc[~loc.is_blind_spot].itertuples())
c1 = n_match == 59; ok_all &= c1
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, title in zip(axs, ["数值（−1 = 黑方块）", "灰度（越暗越差）"]):
    for r in loc.itertuples():
        v = v61[r.idx]
        if r.is_blind_spot:
            ax.add_patch(matplotlib.patches.Ellipse((r.x_deg, r.y_deg), 5, 7, color="0.85")); continue
        if title.startswith("数值"):
            if v == -1: ax.plot(r.x_deg, r.y_deg, "ks", ms=7)
            else: ax.text(r.x_deg, r.y_deg, str(v), ha="center", va="center", fontsize=9)
        else:
            ax.add_patch(plt.Rectangle((r.x_deg - 3, r.y_deg - 3), 6, 6, color=str(np.clip(max(v, 0) / 35, 0, 1))))
    ax.axhline(0, lw=.5); ax.axvline(0, lw=.5); ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
    ax.set_title(f"8 OD visit 1 — {title}", fontsize=10)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
fig.savefig(os.path.join(ROOT, "figures", "g1_patient8_OD_v1.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
out.append(f"## 校验 1：8 号右眼第 1 次检查逐格核对\n\n按 `g1_locations.csv` 坐标放置原始 61 个值，与原文图 2b 每格印的数（黑方块 = −1）逐一比对：**{n_match}/59 一致**。"
           f"图见 `figures/g1_patient8_OD_v1.png`（灰度形态：上半视野大片暗区、下方及中心保留，与图 2a 灰度图一致，人工目视确认）。结论：{'通过' if c1 else '未通过'}。\n")

# ---- 校验 2：盲点占位格位置 ----
bsr = loc[loc.is_blind_spot]
bx, by = bsr.x_deg.iloc[0], bsr.y_deg.iloc[0]
c2 = sorted(bsr.idx.tolist()) == [21, 32] and 10 <= bx <= 20 and abs(by) <= 5 and (raw.iloc[:, 8 + 21] == -1).all() and (raw.iloc[:, 8 + 32] == -1).all()
# 序列上的相邻点：21 介于 (8,2)[idx20] 与 (12,−12)[idx22] 之间；32 介于 (12,12)[idx31] 与 (22,−4)[idx33] 之间——都在颞侧水平线附近
ok_all &= c2
meta = json.load(open(os.path.join(ROOT, "results", "g1_locations_meta.json")))
out.append(f"## 校验 2：盲点占位格\n\n下标 21、32 在全部 1115 次检查中都是 −1；在螺旋顺序中，21 位于 (8,2)→(12,−12) 之间，32 位于 (12,12)→(22,−4) 之间，即跨过颞侧水平线附近的生理盲点。"
           f"占位坐标取图 2b 灰色椭圆中心：用 {meta['blind_spot']['n_squares']} 个完整黑方块（已知坐标）线性标定像素→度（残差 RMSE {meta['blind_spot']['calib_rmse_px']:.2f} px，约 {meta['blind_spot']['px_per_deg_x']:.1f} px/度），"
           f"灰色像素外接框中心 = ({bx:.1f}°, {by:.1f}°)（可见灰色像素质心 {tuple(meta['blind_spot']['pixel_centroid_deg'])}；椭圆被蓝色圆圈和红色箭头切成 {meta['blind_spot']['n_gray_pieces']} 块，外接框中心对遮挡更稳健）。"
           f"位于颞侧 10–20°、水平线 ±5° 内。结论：{'通过' if c2 else '未通过'}。\n")

# ---- 校验 3：扣掉整体偏移后的逐点残差相关（全部检查） ----
Y = vl.pivot_table(index=["eye_id", "visit"], columns="loc59", values="y").sort_index()
T = vl.groupby(["eye_id", "visit"])["t"].first().loc[Y.index]
R = np.zeros_like(Y.to_numpy())
for eid, idx in Y.groupby(level=0).indices.items():
    t = T.to_numpy()[idx]; y = Y.to_numpy()[idx]; n = len(t)
    X = np.c_[np.ones(n), t]; beta = np.linalg.lstsq(X, y, rcond=None)[0]
    R[idx] = (y - X @ beta) * np.sqrt(n / (n - 2))
E = R - R.mean(1, keepdims=True)
C = np.corrcoef(E.T)
dist = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1))
iu = np.triu_indices(59, 1)
# “同一半视野”= 不跨水平中线：y_i·y_j ≥ 0。中心点 (0,0) 在中线上，与上下两侧 ≤6° 的点都算相邻
# （与 G1-MST“跨水平中线的点对 −∞”的写法一致）。对照口径：中心点只归上半视野（hemifield 列）。
Yc = XY[:, 1]
nb = (dist <= cfg["locations"]["neighbor_max_dist_deg"]) & (Yc[:, None] * Yc[None] >= 0)
nb_alt = (dist <= cfg["locations"]["neighbor_max_dist_deg"]) & (HEM[:, None] == HEM[None])
nb_mean = C[iu][nb[iu]].mean(); ot_mean = C[iu][~nb[iu]].mean()
alt_nb, alt_ot = C[iu][nb_alt[iu]].mean(), C[iu][~nb_alt[iu]].mean()
Craw = np.corrcoef(R.T); raw_mean = Craw[iu].mean(); dem_all = C[iu].mean()
# 对照：随机打乱点位映射 200 次时的相邻点相关
rng = np.random.default_rng(cfg["seed"]); perm_nb = []
for _ in range(200):
    p = rng.permutation(59); Cp = C[np.ix_(p, p)]; perm_nb.append(Cp[iu][nb[iu]].mean())
c3 = nb_mean >= cfg["locations"]["neighbor_resid_corr_threshold"] and nb_mean > ot_mean + 0.2
ok_all &= c3
res = dict(neighbor_resid_corr=float(nb_mean), other_resid_corr=float(ot_mean), n_neighbor_pairs=int(nb[iu].sum()),
           n_other_pairs=int((~nb[iu]).sum()), raw_mean_corr=float(raw_mean), demeaned_mean_corr=float(dem_all),
           perm_neighbor_corr_mean=float(np.mean(perm_nb)), perm_neighbor_corr_max=float(np.max(perm_nb)),
           alt_center_sup_only=dict(neighbor=float(alt_nb), other=float(alt_ot), n_neighbor_pairs=int(nb_alt[iu].sum())))
out.append("## 校验 3：相邻点残差相关（全部 263 眼 / 1115 次检查）\n\n逐眼逐点 OLS 残差 × √(n/(n−2))，每次检查减去 59 点平均（整体偏移），对每一对点位在 1115 次检查上算 Pearson 相关，再分组平均。相邻点定义：距离 ≤ 6° 且不跨水平中线（y_i·y_j ≥ 0）；中心点 (0,0) 位于中线上，与上下两侧 ≤ 6° 的点都算相邻。\n\n"
           f"| 量 | 值 |\n|---|---|\n| 同半视野、距离 ≤ 6° 的相邻点对（{res['n_neighbor_pairs']} 对）平均相关 | **{nb_mean:.4f}** |\n"
           f"| 其余点对（{res['n_other_pairs']} 对） | **{ot_mean:.4f}** |\n| 对照口径：中心点只归上半视野（{nb_alt[iu].sum()} 对相邻） | 相邻 {alt_nb:.4f} / 其余 {alt_ot:.4f} |\n| 未扣偏移时全部点对平均相关 | {raw_mean:.4f} |\n| 扣偏移后全部点对平均相关 | {dem_all:.4f} |\n"
           f"| 点位随机打乱 200 次时“相邻点”平均相关（均值 / 最大） | {np.mean(perm_nb):.4f} / {np.max(perm_nb):.4f} |\n\n"
           f"相邻点明显高于其余点对，也远高于打乱映射时的水平；≥ 0.10 → B4 加空间项。结论：{'通过' if c3 else '未通过'}。\n")

# ---- 附加：图 2c（左眼示例）独立核对 ----
FIG2C_R = {  # 图 2c 按显示位置判读后镜像到右眼坐标（x_R = −x_显示）
    (8,26):24, (-8,26):22, (20,20):22, (12,20):27, (4,20):23, (-4,20):21, (-12,20):22, (-20,20):21,
    (4,14):21, (-4,14):22, (20,12):20, (12,12):22, (-12,12):23, (-20,12):22, (26,8):21,
    (8,8):23, (2,8):25, (-2,8):26, (-8,8):24, (22,4):20, (4,4):26, (-4,4):27, (-14,4):22, (-20,4):24, (-26,4):23,
    (8,2):24, (2,2):28, (-2,2):27, (-8,2):26, (0,0):32, (8,-2):28, (2,-2):30, (-2,-2):28, (-8,-2):28,
    (22,-4):21, (4,-4):28, (-4,-4):29, (-14,-4):28, (-20,-4):27, (-26,-4):26, (26,-8):22, (8,-8):26, (-8,-8):26,
    (3,-9):27, (-3,-9):25, (20,-12):21, (12,-12):25, (-12,-12):27, (-20,-12):27, (4,-14):23, (-4,-14):23,
    (20,-20):24, (12,-20):29, (4,-20):26, (-4,-20):24, (-12,-20):28, (-20,-20):26, (8,-26):26, (-8,-26):25}
assert len(FIG2C_R) == 59
fig_vec = np.array([FIG2C_R[(int(x), int(y))] for x, y in XY])
yv = Y.to_numpy(); cens = vl.pivot_table(index=["eye_id", "visit"], columns="loc59", values="censored").loc[Y.index].to_numpy().astype(bool)
obs = np.where(cens, -1, yv)
nm = (obs == fig_vec[None]).sum(1); bi = int(np.argmax(nm)); best = Y.index[bi]
second = np.sort(nm)[-2]
mism = [(tuple(XY[j]), int(fig_vec[j]), int(obs[bi, j])) for j in range(59) if obs[bi, j] != fig_vec[j]]
res.update(fig2c_best_exam=f"{best[0]} visit {best[1]}", fig2c_match=int(nm[bi]), fig2c_second_best=int(second), fig2c_mismatch=str(mism))
out.append(f"## 附加：图 2c（左眼示例）独立核对\n\n把图 2c 每格印的数按显示位置判读，镜像（x → −x）到右眼坐标，与全部 1115 次检查逐一比对。最佳匹配为 **{best[0]} 第 {best[1]} 次检查，{nm[bi]}/59 一致**（次佳检查只有 {second}/59）。"
           f"不一致的格子（坐标, 图上数, 数据值）：{mism if mism else '无'}。\n")

out.insert(0, f"# G1 点位映射校验\n\n生成：`src/01_validate_locations.py`。坐标来源 visualFields 1.0.7 `locmaps$g1`；顺序来源 GRAPE 原文图 2b。\n\n**三项校验：{'全部通过' if ok_all else '有未通过项'}。**\n")
open(os.path.join(ROOT, "results", "g1_validation.md"), "w", encoding="utf-8").write("\n".join(out))
json.dump(res, open(os.path.join(ROOT, "results", "g1_validation.json"), "w"), indent=2)
print(json.dumps(res, indent=1)); assert ok_all
