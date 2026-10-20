"""5.2 节步骤 01：G1 点位映射表 data/g1_locations.csv。

来源与做法（人工判读 + 脚本固化）：
1. 点号顺序：GRAPE 原文 (Huang et al. 2023, Sci Data 10:520) 图 2b，PDF 第 3 页内嵌位图
   (1900×1484 px, 300 ppi) 裁剪放大 200–300%，从中心点（下标 0）沿红色箭头逐个往外数，
   每个下标用 8 号右眼第 1 次检查印在图上的数值确认。判读记录见下表 SPIRAL。
2. 坐标：R 包 visualFields 1.0.7 的 locmaps$g1（Octopus G1 59 点，右眼视野坐标，+x 为颞侧，
   盲点在 +x）。判读出的每个格子与该表的点一一对应（59 个坐标各用一次，脚本 assert）。
3. 两个盲点占位格（下标 21、32）：坐标取原文图 2b 灰色生理盲点椭圆的质心，
   由像素测量得到 (15.0°, −1.4°)（外圈 30° 圆 = 310 px 半径定比例）。
4. 簇：Octopus EyeSuite G 程序的 10 个簇，取自 Bürki & Monhart 2007
   (“An Update to Octopus Perimetry”, European Ophthalmic Review, 图 1b)，按图逐点判读。
   中心点 (0,0) 在该图中不属于任何簇（cluster = 0）。
   簇编号（本项目自定，EyeSuite 原编号未公开）：
     上半视野 1 中心旁 / 2 旁中心-鼻侧水平 / 3 弓形 / 4 上方周边 / 5 颞侧楔形
     下半视野 6 中心旁 / 7 旁中心-鼻侧水平 / 8 弓形 / 9 下方周边 / 10 颞侧楔形
"""
import numpy as np
import pandas as pd
from common import ROOT, get_logger

log = get_logger("01_g1_locations")

# GRAPE 原始 61 下标 -> G1 坐标 (x°, y°)；'BS' 为盲点占位格
SPIRAL = {
    0: (0, 0), 1: (2, -2), 2: (-2, -2), 3: (-2, 2), 4: (2, 2),
    5: (4, -4), 6: (-4, -4), 7: (-4, 4), 8: (4, 4),
    9: (8, -2), 10: (8, -8), 11: (3, -9), 12: (-3, -9), 13: (-8, -8), 14: (-8, -2), 15: (-8, 2),
    16: (-8, 8), 17: (-2, 8), 18: (2, 8), 19: (8, 8), 20: (8, 2), 21: "BS",
    22: (12, -12), 23: (4, -14), 24: (-4, -14), 25: (-12, -12), 26: (-14, -4), 27: (-14, 4),
    28: (-12, 12), 29: (-4, 14), 30: (4, 14), 31: (12, 12), 32: "BS",
    33: (22, -4), 34: (20, -12), 35: (20, -20), 36: (12, -20), 37: (4, -20), 38: (-4, -20),
    39: (-12, -20), 40: (-20, -20), 41: (-20, -12), 42: (-20, -4), 43: (-20, 4), 44: (-20, 12),
    45: (-20, 20), 46: (-12, 20), 47: (-4, 20), 48: (4, 20), 49: (12, 20), 50: (20, 20),
    51: (20, 12), 52: (22, 4), 53: (26, -8), 54: (8, -26), 55: (-8, -26), 56: (-26, -4),
    57: (-26, 4), 58: (-8, 26), 59: (8, 26), 60: (26, 8),
}
BLIND_SPOT_XY = (15.0, -1.4)

# visualFields 1.0.7 locmaps$g1$coord（59 点）
VF_G1 = [(-8, 26), (8, 26), (-20, 20), (-12, 20), (-4, 20), (4, 20), (12, 20), (20, 20), (-4, 14), (4, 14),
         (-20, 12), (-12, 12), (12, 12), (20, 12), (-8, 8), (-2, 8), (2, 8), (8, 8), (26, 8), (-26, 4), (-20, 4),
         (-14, 4), (-4, 4), (4, 4), (22, 4), (-8, 2), (-2, 2), (2, 2), (8, 2), (0, 0), (-8, -2), (-2, -2), (2, -2),
         (8, -2), (-26, -4), (-20, -4), (-14, -4), (-4, -4), (4, -4), (22, -4), (-8, -8), (8, -8), (26, -8),
         (-3, -9), (3, -9), (-20, -12), (-12, -12), (12, -12), (20, -12), (-4, -14), (4, -14), (-20, -20),
         (-12, -20), (-4, -20), (4, -20), (12, -20), (20, -20), (-8, -26), (8, -26)]

CLUSTERS = {
    1: [(4, 4), (-2, 2), (2, 2), (8, 2)],
    2: [(-2, 8), (2, 8), (-26, 4), (-20, 4), (-14, 4), (-4, 4), (-8, 2)],
    3: [(-20, 12), (-12, 12), (-4, 14), (4, 14), (-8, 8), (8, 8)],
    4: [(-8, 26), (8, 26), (-20, 20), (-12, 20), (-4, 20), (4, 20), (12, 20), (12, 12)],
    5: [(20, 20), (20, 12), (26, 8), (22, 4)],
    6: [(-2, -2), (2, -2), (8, -2), (4, -4)],
    7: [(-8, -2), (-26, -4), (-20, -4), (-14, -4), (-4, -4), (-8, -8), (-3, -9), (3, -9), (8, -8)],
    8: [(-20, -12), (-12, -12), (-4, -14), (4, -14)],
    9: [(12, -12), (-20, -20), (-12, -20), (-4, -20), (4, -20), (12, -20), (-8, -26), (8, -26)],
    10: [(22, -4), (26, -8), (20, -12), (20, -20)],
}
CLUSTER_NAME = {1: "S_central", 2: "S_paracentral_nasal", 3: "S_arcuate", 4: "S_peripheral", 5: "S_temporal",
                6: "I_central", 7: "I_paracentral_nasal", 8: "I_arcuate", 9: "I_peripheral", 10: "I_temporal"}

pts = [v for v in SPIRAL.values() if v != "BS"]
assert len(SPIRAL) == 61 and len(pts) == 59
assert sorted(pts) == sorted(VF_G1), "螺旋映射没有恰好覆盖 visualFields G1 的 59 个坐标"
assert [k for k, v in SPIRAL.items() if v == "BS"] == [21, 32]
cl_pts = [p for v in CLUSTERS.values() for p in v]
assert len(cl_pts) == len(set(cl_pts)) == 58 and set(cl_pts) == set(VF_G1) - {(0, 0)}
pt2cl = {p: c for c, v in CLUSTERS.items() for p in v}
for c, v in CLUSTERS.items():  # 簇不跨水平中线
    assert all((p[1] > 0) == (c <= 5) for p in v), c

rows = []
loc = 0
for idx, v in SPIRAL.items():
    if v == "BS":
        rows.append(dict(idx=idx, loc=-1, is_blind_spot=True, x_deg=BLIND_SPOT_XY[0], y_deg=BLIND_SPOT_XY[1],
                         hemifield="blind", cluster=-1, cluster_name="blind_spot_placeholder"))
        continue
    x, y = v
    hf = "sup" if y > 0 else ("inf" if y < 0 else "center")
    c = pt2cl.get(v, 0)
    rows.append(dict(idx=idx, loc=loc, is_blind_spot=False, x_deg=x, y_deg=y, hemifield=hf, cluster=c,
                     cluster_name=CLUSTER_NAME.get(c, "none_center")))
    loc += 1
df = pd.DataFrame(rows)
df["ecc_deg"] = np.hypot(df.x_deg, df.y_deg).round(3)
df.to_csv(ROOT / "data" / "g1_locations.csv", index=False)
log.info("写出 data/g1_locations.csv：%d 行（59 测试点 + 2 盲点占位格）", len(df))
log.info("各簇点数：%s", df[df.cluster > 0].groupby("cluster").size().to_dict())
