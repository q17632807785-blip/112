"""01_g1_locations.py — 计划第 5.2 节 / 预注册第 6 部分第 2、5 条：G1 点位映射与 10 簇表。
坐标：R 包 visualFields 1.0.7 locmaps$g1（由 R/00_export_g1_coords.R 导出）。
顺序：GRAPE 原文图 2b（右眼，从中心向外螺旋），按图上印的数值逐点人工判读后写在下面，
      并用 8 号右眼第 1 次检查的原始值逐值核对（assert）。
盲点占位格：图 2b 灰色椭圆的像素质心，用图上黑色方块（已知坐标的删失点）做线性标定后换算成度。
输出 data/g1_locations.csv：idx, x_deg, y_deg, hemifield, cluster（另加 loc59, cluster_name, is_blind_spot）。"""
import os, json
import numpy as np, pandas as pd, yaml
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
DATA = os.path.join(ROOT, "data"); BLIND = cfg["data"]["blind_spot_idx"]

vf = pd.read_csv(os.path.join(DATA, "visualFields_g1_coord.csv"))
G1 = set(zip(vf.x.astype(int), vf.y.astype(int)))
assert len(G1) == 59

BS = "BS"
# GRAPE 下标 0–60 → 右眼视野坐标（按图 2b 红色箭头从中心向外数）
SPIRAL = [
    (0, 0), (2, -2), (-2, -2), (-2, 2), (2, 2),                                  # 0–4 中心小圈
    (4, -4), (-4, -4), (-4, 4), (4, 4),                                         # 5–8
    (8, -2), (8, -8), (3, -9), (-3, -9), (-8, -8), (-8, -2), (-8, 2), (-8, 8),  # 9–16
    (-2, 8), (2, 8), (8, 8), (8, 2), BS,                                        # 17–21（21 = 盲点占位）
    (12, -12), (4, -14), (-4, -14), (-12, -12), (-14, -4), (-14, 4),            # 22–27
    (-12, 12), (-4, 14), (4, 14), (12, 12), BS,                                 # 28–32（32 = 盲点占位）
    (22, -4), (20, -12), (20, -20), (12, -20), (4, -20), (-4, -20), (-12, -20), # 33–39
    (-20, -20), (-20, -12), (-20, -4), (-20, 4), (-20, 12), (-20, 20),          # 40–45
    (-12, 20), (-4, 20), (4, 20), (12, 20), (20, 20), (20, 12), (22, 4),        # 46–52
    (26, -8), (8, -26), (-8, -26), (-26, -4), (-26, 4), (-8, 26), (8, 26), (26, 8),  # 53–60
]
assert len(SPIRAL) == 61 and [i for i, c in enumerate(SPIRAL) if c == BS] == BLIND
assert set(c for c in SPIRAL if c != BS) == G1, "spiral coords != visualFields G1 set"

# 图 2b 每个格子上印的数（按坐标判读；黑方块 = −1）
SQ = -1
FIG2B = {
    (0, 0): 30, (2, -2): 31, (-2, -2): 26, (-2, 2): 22, (2, 2): 26, (4, -4): 29, (-4, -4): 30,
    (-4, 4): 12, (4, 4): 24, (8, -2): 26, (8, -8): 23, (3, -9): 20, (-3, -9): 22, (-8, -8): 24,
    (-8, -2): 27, (-8, 2): 12, (-8, 8): 1, (-2, 8): 12, (2, 8): 17, (8, 8): 4, (8, 2): 18,
    (12, -12): 8, (4, -14): 6, (-4, -14): 13, (-12, -12): 8, (-14, -4): 13, (-14, 4): SQ,
    (-12, 12): SQ, (-4, 14): SQ, (4, 14): SQ, (12, 12): SQ,
    (22, -4): 15, (20, -12): 11, (20, -20): 3, (12, -20): 7, (4, -20): 1, (-4, -20): 1,
    (-12, -20): SQ, (-20, -20): 1, (-20, -12): 1, (-20, -4): 1, (-20, 4): SQ, (-20, 12): SQ,
    (-20, 20): SQ, (-12, 20): SQ, (-4, 20): SQ, (4, 20): SQ, (12, 20): SQ, (20, 20): SQ,
    (20, 12): 2, (22, 4): 11, (26, -8): 14, (8, -26): 2, (-8, -26): SQ, (-26, -4): SQ,
    (-26, 4): SQ, (-8, 26): SQ, (8, 26): SQ, (26, 8): 8,
}
assert set(FIG2B) == G1

# 10 簇（预注册第 6 部分第 5 条，逐字照抄）
CLUSTERS = {
    1: ("S_central", [(-2,2), (2,2), (4,4), (8,2)]),
    2: ("S_paracentral_nasal", [(-4,4), (-8,2), (-2,8), (2,8), (-14,4), (-20,4), (-26,4)]),
    3: ("S_arcuate", [(-8,8), (8,8), (-12,12), (-4,14), (4,14), (-20,12)]),
    4: ("S_peripheral", [(12,12), (-20,20), (-12,20), (-4,20), (4,20), (12,20), (-8,26), (8,26)]),
    5: ("S_temporal", [(20,20), (20,12), (22,4), (26,8)]),
    6: ("I_central", [(2,-2), (-2,-2), (4,-4), (8,-2)]),
    7: ("I_paracentral_nasal", [(-4,-4), (8,-8), (3,-9), (-3,-9), (-8,-8), (-8,-2), (-14,-4), (-20,-4), (-26,-4)]),
    8: ("I_arcuate", [(4,-14), (-4,-14), (-12,-12), (-20,-12)]),
    9: ("I_peripheral", [(12,-12), (12,-20), (4,-20), (-4,-20), (-12,-20), (-20,-20), (8,-26), (-8,-26)]),
    10: ("I_temporal", [(22,-4), (20,-12), (20,-20), (26,-8)]),
}
EXPECTED_N = {1: 4, 2: 7, 3: 6, 4: 8, 5: 4, 6: 4, 7: 9, 8: 4, 9: 8, 10: 4}
cl_of = {}
for k, (nm, pts) in CLUSTERS.items():
    assert len(pts) == EXPECTED_N[k], (k, len(pts))
    for p in pts:
        assert p in G1, p; assert p not in cl_of, p
        assert (p[1] > 0) == nm.startswith("S_"), p   # 簇不跨水平中线
        cl_of[p] = k
assert set(G1) - set(cl_of) == {(0, 0)}, "only center point should be unclustered"

# ---- 盲点：图 2b 灰色椭圆质心 ----
def blind_spot_from_figure(png):
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    f = 1900 / 1024
    x0, y0, x1, y1 = int(640 * f), int(40 * f), int(1010 * f), int(390 * f)   # 图 2b 面板
    P = im[y0:y1, x0:x1]; R, G, B = P[..., 0], P[..., 1], P[..., 2]
    black = (R < 70) & (G < 70) & (B < 70)
    lab, n = ndimage.label(black)
    sq = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        fill = (lab[sl] == i).sum() / (h * w)
        if 13 <= h <= 18 and 13 <= w <= 18 and fill > 0.8:   # 完整、未被箭头遮挡的方块
            cy, cx = ndimage.center_of_mass(lab == i); sq.append((cx, cy))
    sq = np.array(sq)
    known = np.array([c for c, v in FIG2B.items() if v == SQ], float)
    assert 12 <= len(sq) <= len(known), (len(sq), len(known))
    # 粗网格搜索比例尺和原点，再用匹配点做线性最小二乘
    best = None
    for s in np.arange(8, 30, 0.05):
        ox = sq[:, 0].mean() - s * known[:, 0].mean(); oy = sq[:, 1].mean() + s * known[:, 1].mean()
        pred = np.c_[ox + s * known[:, 0], oy - s * known[:, 1]]
        d = np.sqrt(((sq[:, None] - pred[None]) ** 2).sum(-1))
        cost = d.min(1).sum()
        if best is None or cost < best[0]: best = (cost, d.argmin(1))
    match = known[best[1]]
    assert len(set(map(tuple, match))) == len(sq), "ambiguous square matching"
    Ax = np.c_[np.ones(len(sq)), match[:, 0]]; cx_ = np.linalg.lstsq(Ax, sq[:, 0], rcond=None)[0]
    Ay = np.c_[np.ones(len(sq)), match[:, 1]]; cy_ = np.linalg.lstsq(Ay, sq[:, 1], rcond=None)[0]
    resid = np.r_[sq[:, 0] - Ax @ cx_, sq[:, 1] - Ay @ cy_]
    gray = (np.abs(R - G) < 10) & (np.abs(G - B) < 12) & (R > 185) & (R < 235)
    glab, gn = ndimage.label(gray)
    # 椭圆被蓝色圆圈和红色箭头切成几块：以最大块为种子，并入其 ±60 px 窗口内所有 ≥20 px 的灰色块
    sizes = ndimage.sum(gray, glab, range(1, gn + 1)); gi = int(np.argmax(sizes)) + 1
    objs = ndimage.find_objects(glab); s0 = objs[gi - 1]
    wy0, wy1, wx0, wx1 = s0[0].start - 60, s0[0].stop + 60, s0[1].start - 60, s0[1].stop + 60
    keep = [i + 1 for i, o in enumerate(objs) if sizes[i] >= 20 and o[0].start >= wy0 and o[0].stop <= wy1
            and o[1].start >= wx0 and o[1].stop <= wx1]
    m = np.isin(glab, keep); ys, xs = np.where(m)
    to_deg = lambda px, py: ((px - cx_[0]) / cx_[1], (py - cy_[0]) / cy_[1])
    bbx, bby = to_deg((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)   # 外接框中心（对遮挡稳健）
    cmx, cmy = to_deg(xs.mean(), ys.mean())                                       # 可见灰色像素质心
    return dict(x_deg=round(float(bbx), 1), y_deg=round(float(bby), 1),
                pixel_centroid_deg=[round(float(cmx), 2), round(float(cmy), 2)],
                bbox_center_deg=[round(float(bbx), 2), round(float(bby), 2)],
                bbox_size_deg=[round(float((xs.max() - xs.min()) / cx_[1]), 2), round(float((ys.max() - ys.min()) / -cy_[1]), 2)],
                n_squares=int(len(sq)), px_per_deg_x=float(cx_[1]), px_per_deg_y=float(-cy_[1]),
                calib_rmse_px=float(np.sqrt((resid ** 2).mean())), gray_px=int(m.sum()), n_gray_pieces=len(keep))

bs = blind_spot_from_figure(os.path.join(ROOT, "figures", "GRAPE_fig2_source.png"))
print("blind spot (fig 2b gray ellipse centroid):", bs)

# ---- 8 号右眼第 1 次检查逐值核对 ----
vl = pd.read_parquet(os.path.join(DATA, "vf_long.parquet"))
REF8 = [30,31,26,22,26,29,30,12,24,26,23,20,22,24,27,12,1,12,17,4,18,-1,8,6,13,8,13,
        -1,-1,-1,-1,-1,-1,15,11,3,7,1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,2,11,14,2,
        -1,-1,-1,-1,-1,8]
n_ok = 0
for i, c in enumerate(SPIRAL):
    if c == BS: continue
    assert FIG2B[c] == REF8[i], (i, c, FIG2B[c], REF8[i]); n_ok += 1
assert n_ok == 59
print(f"fig 2b value-by-value check: {n_ok}/59")

rows = []
k59 = 0
for i, c in enumerate(SPIRAL):
    if c == BS:
        rows.append(dict(idx=i, loc59=-1, x_deg=bs["x_deg"], y_deg=bs["y_deg"], hemifield="blind_spot",
                         cluster="", cluster_name="", is_blind_spot=True))
    else:
        hemi = "sup" if c[1] > 0 or c == (0, 0) else "inf"   # 中心点并入上半视野块（预注册 3.3）
        k = cl_of.get(c)
        rows.append(dict(idx=i, loc59=k59, x_deg=float(c[0]), y_deg=float(c[1]), hemifield=hemi,
                         cluster="" if k is None else k, cluster_name="" if k is None else CLUSTERS[k][0],
                         is_blind_spot=False)); k59 += 1
loc = pd.DataFrame(rows)
loc.to_csv(os.path.join(DATA, "g1_locations.csv"), index=False, lineterminator="\n")
json.dump(dict(blind_spot=bs, fig2b_check=f"{n_ok}/59", coord_source="visualFields 1.0.7 locmaps$g1"),
          open(os.path.join(ROOT, "results", "g1_locations_meta.json"), "w"), indent=2)
print(loc.groupby("cluster_name").size().to_dict())
