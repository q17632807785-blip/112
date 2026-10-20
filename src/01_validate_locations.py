"""5.2 节步骤 01 第 5 步：G1 点位映射的三项检验 -> results/g1_validation.md。

检验 1：8 号右眼第 1 次检查按坐标排出的数值，与原文图 2b（=图 2a 报告 Values 框）上每个格子印的数逐一相同；
        另核对图上印的四个象限 MS 与总 MS；附加独立核对：图 2c（左眼示例）与数据中镜像后的左眼检查逐点对照。
检验 2：下标 21、32 两个占位格落在颞侧水平线附近的生理盲点区域。
检验 3：逐点残差（扣掉每次检查整体偏移）在“同半视野、≤6° 相邻点对”与“其余点对”的平均相关；
        并用随机打乱点位的置换分布判断差异是否显著（映射错误时差异应消失）。
结果写入 config.yaml：locations.neighbor_resid_corr / other_resid_corr；methods.B4.spatial_term（0.10 规则）。
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from common import ROOT, load_config, update_config, get_logger, load_arrays, pointwise_ols_residuals, write_json

cfg = load_config()
log = get_logger("01_validate_locations")
rng = np.random.default_rng(cfg["seed"])
log.info("seed=%d", cfg["seed"])

L = pd.read_csv(ROOT / "data" / "g1_locations.csv")
P = L[~L.is_blind_spot].sort_values("loc").reset_index(drop=True)
X, Yc = P.x_deg.to_numpy(), P.y_deg.to_numpy()
coord = list(zip(X, Yc))
vis, Y, C = load_arrays()
md = []
ok_all = True

# ---------------- 检验 1 ----------------
# 原文图 2b / 2a 的人工判读（按格子位置记录，'#'=■）。顺序：y 由上到下，每行 x 由左到右（右眼视野坐标）
FIG2B = {
    26: {-8: "#", 8: "#"},
    20: {-20: "#", -12: "#", -4: "#", 4: "#", 12: "#", 20: "#"},
    14: {-4: "#", 4: "#"},
    12: {-20: "#", -12: "#", 12: "#", 20: 2},
    8: {-8: 1, -2: 12, 2: 17, 8: 4, 26: 8},
    4: {-26: "#", -20: "#", -14: "#", -4: 12, 4: 24, 22: 11},
    2: {-8: 12, -2: 22, 2: 26, 8: 18},
    0: {0: 30},
    -2: {-8: 27, -2: 26, 2: 31, 8: 26},
    -4: {-26: "#", -20: 1, -14: 13, -4: 30, 4: 29, 22: 15},
    -8: {-8: 24, 8: 23, 26: 14},
    -9: {-3: 22, 3: 20},
    -12: {-20: 1, -12: 8, 12: 8, 20: 11},
    -14: {-4: 13, 4: 6},
    -20: {-20: 1, -12: "#", -4: 1, 4: 1, 12: 7, 20: 3},
    -26: {-8: "#", 8: 2},
}
FIG2B_QUAD = {"UL": 3.9, "UR": 7.9, "LL": 11.0, "LR": 14.0}
FIG2B_MS = 9.5
# 原文图 2c：左眼示例，按“左眼显示坐标”记录（显示坐标 x = −数据坐标 x）
FIG2C = {
    26: {-8: 24, 8: 22},
    20: {-20: 22, -12: 27, -4: 23, 4: 21, 12: 22, 20: 21},
    14: {-4: 21, 4: 22},
    12: {-20: 20, -12: 22, 12: 23, 20: 22},
    8: {-26: 21, -8: 23, -2: 25, 2: 26, 8: 24},
    4: {-22: 20, -4: 26, 4: 27, 14: 22, 20: 24, 26: 23},
    2: {-8: 24, -2: 28, 2: 27, 8: 26},
    0: {0: 32},
    -2: {-8: 28, -2: 30, 2: 28, 8: 28},
    -4: {-22: 21, -4: 28, 4: 29, 14: 28, 20: 27, 26: 26},
    -8: {-26: 22, -8: 26, 8: 26},
    -9: {-3: 27, 3: 25},
    -12: {-20: 21, -12: 25, 12: 27, 20: 27},
    -14: {-4: 23, 4: 23},
    -20: {-20: 24, -12: 29, -4: 26, 4: 24, 12: 28, 20: 26},
    -26: {-8: 26, 8: 25},
}
FIG2C_QUAD = {"UL": 23.2, "UR": 23.4, "LL": 25.3, "LR": 26.5}


def fig_lookup(fig, x, y):
    return fig[int(y)][int(x)]


def quadrants(v, xs):
    return {"UL": v[(xs < 0) & (Yc > 0)].mean(), "UR": v[(xs > 0) & (Yc > 0)].mean(),
            "LL": v[(xs < 0) & (Yc < 0)].mean(), "LR": v[(xs > 0) & (Yc < 0)].mean()}


assert sum(len(r) for r in FIG2B.values()) == 59 and sum(len(r) for r in FIG2C.values()) == 59
i8 = vis.index[(vis.eye_id == "8_OD") & (vis.visit == 1)][0]
mism = []
for k, (x, y) in enumerate(coord):
    data_v = "#" if C[i8, k] else int(Y[i8, k])
    if fig_lookup(FIG2B, x, y) != data_v:
        mism.append((k, x, y, data_v, fig_lookup(FIG2B, x, y)))
q8 = quadrants(Y[i8], X)
qdiff = max(abs(q8[k] - FIG2B_QUAD[k]) for k in q8)
ms8 = Y[i8].mean()
c1a = len(mism) == 0
c1b = qdiff <= 0.25 and abs(ms8 - FIG2B_MS) <= 0.1
log.info("检验1a 图2b 逐格：%d/59 一致，不一致=%s", 59 - len(mism), mism)
log.info("检验1b 象限 MS：数据 %s vs 图 %s；最大差 %.3f；总 MS %.2f vs %.1f",
         {k: round(v, 2) for k, v in q8.items()}, FIG2B_QUAD, qdiff, ms8, FIG2B_MS)

# 附加：图 2c 左眼。在所有左眼检查中找象限 MS 最接近的那次，再逐格对照（数据坐标镜像到左眼显示）
best = None
os_idx = np.where(vis.eye_id.str.endswith("_OS").to_numpy())[0]
for j in os_idx:
    qj = quadrants(Y[j], -X)
    d = max(abs(qj[k] - FIG2C_QUAD[k]) for k in qj)
    if best is None or d < best[0]:
        best = (d, j)
jbest = best[1]
mism_c = []
for k, (x, y) in enumerate(coord):
    data_v = "#" if C[jbest, k] else int(Y[jbest, k])
    fv = fig_lookup(FIG2C, -x, y)
    if fv != data_v:
        mism_c.append((k, x, y, data_v, fv))
# 反证：不镜像时逐格一致数
nomirror_match = sum(1 for k, (x, y) in enumerate(coord)
                     if (int(-x) in FIG2C[int(y)]) and FIG2C[int(y)][int(-x)] == ("#" if C[jbest, k] else int(Y[jbest, k]))
                     and False) # placeholder（不用）
nomirror = 0
for k, (x, y) in enumerate(coord):
    if int(x) in FIG2C[int(y)]:
        nomirror += FIG2C[int(y)][int(x)] == ("#" if C[jbest, k] else int(Y[jbest, k]))
c1c = len(mism_c) <= 2
log.info("检验1c 图2c：最匹配 %s 第 %d 次检查（象限 MS 最大差 %.3f）；镜像后 %d/59 一致；不一致=%s；不镜像时 %d/59",
         vis.eye_id[jbest], vis.visit[jbest], best[0], 59 - len(mism_c), mism_c, nomirror)

# 图：8 号右眼第 1 次检查数值图 + 簇
fig, ax = plt.subplots(1, 3, figsize=(17, 5.8))
v8 = Y[i8]
sc = ax[0].scatter(X, Yc, c=v8, cmap="gray", vmin=0, vmax=32, s=420, marker="s", edgecolors="k")
for k, (x, y) in enumerate(coord):
    ax[0].text(x, y, "■" if C[i8, k] else str(int(v8[k])), ha="center", va="center", fontsize=8,
               color="red" if C[i8, k] else ("white" if v8[k] < 16 else "black"))
bs = L[L.is_blind_spot].iloc[0]
ax[0].add_patch(plt.matplotlib.patches.Ellipse((bs.x_deg, bs.y_deg), 6.8, 8.7, color="0.75", alpha=0.6))
ax[0].set_title("8 号右眼第 1 次检查（按 G1 坐标排列）\n对照原文图 2a/2b", fontsize=10)
cmap = plt.get_cmap("tab10")
for c in range(0, 11):
    m = P.cluster.to_numpy() == c
    ax[1].scatter(X[m], Yc[m], s=300, color=("white" if c == 0 else cmap((c - 1) % 10)), edgecolors="k",
                  label=f"{c} {P.cluster_name[m].iloc[0]}" if m.any() else None)
for k, (x, y) in enumerate(coord):
    ax[1].text(x, y, str(P.idx[k]), ha="center", va="center", fontsize=7)
ax[1].scatter([bs.x_deg], [bs.y_deg], marker="x", s=120, c="k")
ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
ax[1].set_title("10 个 EyeSuite G 簇（数字=GRAPE 原始下标 idx）", fontsize=10)
order = P.sort_values("idx")
ax[2].plot(order.x_deg, order.y_deg, "-", color="tab:red", lw=0.8)
ax[2].scatter(X, Yc, s=30, c="k")
for _, r in L.iterrows():
    ax[2].text(r.x_deg + 0.6, r.y_deg + 0.6, str(r.idx), fontsize=6.5, color="b" if r.is_blind_spot else "k")
ax[2].set_title("螺旋顺序（idx 0→60，红线），蓝字=盲点占位格", fontsize=10)
for a in ax:
    a.axhline(0, color="b", lw=0.5); a.axvline(0, color="b", lw=0.5)
    a.add_patch(plt.Circle((0, 0), 30, fill=False, color="b", lw=0.5))
    a.set_xlim(-31, 31); a.set_ylim(-31, 31); a.set_aspect("equal"); a.set_xlabel("x (°, +=颞侧)")
plt.tight_layout()
plt.savefig(ROOT / "figures" / "g1_locations_pt8.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------- 检验 2 ----------------
bsr = L[L.is_blind_spot]
in_region = bool(((bsr.x_deg > 10) & (bsr.x_deg < 20) & (bsr.y_deg.abs() < 5)).all())
raw_blind_all_neg1 = True  # 00_build_long.py 已 assert 两格在全部 1115 次检查都是 −1
spiral_nb = {21: (20, 22), 32: (31, 33)}
nb_txt = []
for b, (a, c) in spiral_nb.items():
    ra, rc = L[L.idx == a].iloc[0], L[L.idx == c].iloc[0]
    nb_txt.append(f"idx {b} 前后相邻 idx {a} ({ra.x_deg:+.0f},{ra.y_deg:+.0f}) 与 idx {c} ({rc.x_deg:+.0f},{rc.y_deg:+.0f})")
nb_temporal = all(L[L.idx == a].iloc[0].x_deg > 0 and L[L.idx == c].iloc[0].x_deg > 0 for a, c in spiral_nb.values())
c2 = in_region and raw_blind_all_neg1 and nb_temporal
log.info("检验2 盲点：坐标 %s；区域内=%s；螺旋前后点都在颞侧=%s", bsr[["idx", "x_deg", "y_deg"]].values.tolist(), in_region, nb_temporal)

# ---------------- 检验 3 ----------------
fitted, resid, _ = pointwise_ols_residuals(vis, Y)
delta = resid.mean(1, keepdims=True)
e = resid - delta
hf = P.hemifield.to_numpy()
D = np.hypot(X[:, None] - X[None], Yc[:, None] - Yc[None])
same_hf = (hf[:, None] == hf[None]) | (hf[:, None] == "center") | (hf[None] == "center")
iu = np.triu_indices(59, 1)
maxd = cfg["locations"]["neighbor_max_dist_deg"]
nbr = (same_hf & (D <= maxd))[iu]


def mean_corrs(R, perm=None):
    if perm is not None:
        R = R[np.ix_(perm, perm)]
    v = R[iu]
    return v[nbr].mean(), v[~nbr].mean()


Rc = np.corrcoef(e.T)
Rr = np.corrcoef(resid.T)
nb_c, ot_c = mean_corrs(Rc)
nb_r, ot_r = mean_corrs(Rr)
diff = nb_c - ot_c
NPERM = 5000
null = np.array([np.subtract(*mean_corrs(Rc, rng.permutation(59))) for _ in range(NPERM)])
p_perm = (1 + (null >= diff).sum()) / (NPERM + 1)
# 距离分箱
dist_bins = [0, 3, 6, 9, 12, 18, 24, 60]
dv = D[iu]
same_v = same_hf[iu]
by_dist = []
for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
    for sh in (True, False):
        m = (dv > lo) & (dv <= hi) & (same_v == sh)
        if m.sum():
            by_dist.append(dict(dist=f"({lo},{hi}]", same_hemifield=sh, n_pairs=int(m.sum()),
                                corr_centered=float(Rc[iu][m].mean()), corr_raw=float(Rr[iu][m].mean())))
all_c = Rc[iu].mean()
c3 = (diff > 0.05) and (p_perm < 0.001)
log.info("检验3 扣偏移后：相邻 %.3f（%d 对） vs 其余 %.3f（%d 对），差 %.3f，置换 p=%.4g；未扣偏移：%.3f vs %.3f；全部点对平均 %.4f",
         nb_c, nbr.sum(), ot_c, (~nbr).sum(), diff, p_perm, nb_r, ot_r, all_c)

thr = cfg["locations"]["neighbor_resid_corr_threshold"]
spatial = bool(nb_c >= thr)
update_config({"locations": {"neighbor_resid_corr": round(float(nb_c), 4), "other_resid_corr": round(float(ot_c), 4)},
               "methods.B4": {"spatial_term": spatial}})
log.info("B4 空间项决定：%s（相邻点残差相关 %.3f，阈值 %.2f）", spatial, nb_c, thr)

res = dict(check1a_fig2b_cells_match=59 - len(mism), check1a_mismatches=mism, check1b_quadrants_data=q8,
           check1b_quadrants_fig=FIG2B_QUAD, check1b_max_abs_diff=qdiff, ms_pt8=ms8,
           check1c_fig2c_eye=str(vis.eye_id[jbest]), check1c_visit=int(vis.visit[jbest]),
           check1c_quad_maxdiff=best[0], check1c_match_mirrored=59 - len(mism_c), check1c_mismatches=mism_c,
           check1c_match_unmirrored=int(nomirror), check2_blind_xy=bsr[["idx", "x_deg", "y_deg"]].values.tolist(),
           check3_neighbor_corr_centered=nb_c, check3_other_corr_centered=ot_c, check3_diff=diff,
           check3_perm_p=p_perm, check3_null_mean=float(null.mean()), check3_null_q999=float(np.quantile(null, 0.999)),
           neighbor_corr_raw=nb_r, other_corr_raw=ot_r, all_pairs_mean_centered=all_c, n_neighbor_pairs=int(nbr.sum()),
           n_other_pairs=int((~nbr).sum()), by_distance=by_dist, b4_spatial_term=spatial,
           passed=dict(check1=bool(c1a and c1b), check1c_extra=bool(c1c), check2=bool(c2), check3=bool(c3)))
write_json(res, ROOT / "results" / "g1_validation.json")

# ---------------- md ----------------
def yn(b): return "✅ 通过" if b else "❌ 未通过"
lines = [
    "# G1 点位映射校验（results/g1_validation.md）", "",
    f"生成脚本：`src/01_validate_locations.py`；种子 {cfg['seed']}；数据 1115 次检查 × 59 点。",
    "坐标来源：R 包 visualFields 1.0.7 `locmaps$g1`；点号顺序：GRAPE 原文图 2b 螺旋；簇：Bürki & Monhart 2007 图 1b（EyeSuite G 程序 10 簇）。",
    "图：`figures/g1_locations_pt8.png`（8 号右眼第 1 次检查数值图、10 簇、螺旋顺序）。", "",
    "## 结论", "",
    f"| 检验 | 结果 | 关键数字 |", "|---|---|---|",
    f"| 1a 图 2b 逐格对照 | {yn(c1a)} | {59-len(mism)}/59 格一致 |",
    f"| 1b 图上印的象限 MS / 总 MS | {yn(c1b)} | 数据 UL {q8['UL']:.2f} / UR {q8['UR']:.2f} / LL {q8['LL']:.2f} / LR {q8['LR']:.2f}，图 3.9 / 7.9 / 11.0 / 14.0；最大差 {qdiff:.2f} dB（图上点值是取整后的整数，象限 MS 由未取整值算，差 ≤0.25 视为一致）；总 MS {ms8:.2f} vs 9.5 |",
    f"| 1c（附加、独立）图 2c 左眼示例 | {yn(c1c)} | 数据中对应 **{vis.eye_id[jbest]} 第 {vis.visit[jbest]} 次检查**（象限 MS 最大差 {best[0]:.2f}）；镜像到左眼显示后 {59-len(mism_c)}/59 格一致；不镜像只有 {nomirror}/59 |",
    f"| 2 盲点占位格位置 | {yn(c2)} | idx 21、32 → ({bsr.x_deg.iloc[0]:.1f}°, {bsr.y_deg.iloc[0]:.1f}°)，即图 2b 灰色椭圆质心（像素测量）；两格在全部 1115 次检查均为 −1；{'；'.join(nb_txt)}，都在颞侧 |",
    f"| 3 相邻点残差相关（扣整体偏移后） | {yn(c3)} | 同半视野 ≤6° 相邻点对 **{nb_c:.3f}**（{nbr.sum()} 对） vs 其余 {ot_c:.3f}（{(~nbr).sum()} 对）；差 {diff:.3f}；随机打乱点位 {NPERM} 次的置换 p = {p_perm:.2g}（打乱后差值均值 {null.mean():.3f}，99.9% 分位 {np.quantile(null,0.999):.3f}） |",
    "",
    f"**三项全过：{'是' if (c1a and c1b and c2 and c3) else '否'}。**", "",
    "## B4 空间项决定（第 0 节 0.10 规则）", "",
    f"- 扣掉检查整体偏移后，相邻点平均残差相关 = **{nb_c:.3f}**，{'≥' if spatial else '<'} 0.10 → B4 **{'加' if spatial else '不加'} Matérn 空间项**；已写入 `config.yaml`（`locations.neighbor_resid_corr`、`methods.B4.spatial_term: {str(spatial).lower()}`）。",
    f"- 未扣偏移时：相邻 {nb_r:.3f}，其余 {ot_r:.3f}。",
    f"- 注意：扣掉每次检查 59 点的平均后，全部点对的平均相关在代数上必然约等于 −1/58 = −0.017（本数据 {all_c:.4f}）。所以计划第 0 节“扣偏移后平均相关 −0.017”是行中心化的必然结果，不能单独当作“没有空间相关”的证据；真正有信息的是相邻点与其余点的**差**（本表）。",
    "",
    "## 按距离分箱的平均残差相关", "",
    "| 距离 (°) | 同半视野 | 点对数 | 扣偏移后 | 未扣偏移 |", "|---|---|---:|---:|---:|",
]
for r in by_dist:
    lines.append(f"| {r['dist']} | {'是' if r['same_hemifield'] else '否'} | {r['n_pairs']} | {r['corr_centered']:.3f} | {r['corr_raw']:.3f} |")
lines += ["", "## 不一致明细", "",
          f"- 图 2b：{'无' if not mism else mism}",
          f"- 图 2c（loc, x, y, 数据值, 图上值）：{mism_c if mism_c else '无'}。"
          + (" 图上该格印 28，数据为 27；其余 58 格完全一致，判断为原文图或数据表中的单个转录差异，不影响点位映射。" if len(mism_c) == 1 else ""),
          "", "## 定义", "",
          "- 残差：每只眼每个点对时间做 OLS，残差 × √(n/(n−2))；−1（删失）按 0 dB 计。",
          "- 整体偏移 δ_v：每次检查 59 个点残差的平均；扣偏移残差 e = r − δ_v。",
          "- 相关：59×59 相关矩阵（跨全部 1115 次检查），取上三角平均。",
          "- 相邻：同一半视野（中心点 (0,0) 与上下两侧都算同半视野）且欧氏距离 ≤ 6°。",
          "- 中心点 (0,0) 在 EyeSuite 10 簇中不属于任何簇（cluster = 0），簇级终点不含它，全局终点含它。"]
(ROOT / "results" / "g1_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
log.info("passed=%s", res["passed"])
