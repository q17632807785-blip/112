"""5.2 节：从 GRAPE 原始表构建长表。

输出 data/vf_long.parquet、data/visits.parquet、data/eyes.parquet。
验收（assert）：1115 次检查；65,785 个点值；删失 2734；无 NaN；
8 号右眼第 1 次检查 61 个原始值与计划 1.2 节一致。
"""
import numpy as np
import pandas as pd
from common import ROOT, load_config, get_logger, write_json

cfg = load_config()
log = get_logger("00_build_long")
xlsx = (ROOT / cfg["paths"]["raw_xlsx"]).resolve()
log.info("raw xlsx: %s", xlsx)

PT8_OD_V1 = [30, 31, 26, 22, 26, 29, 30, 12, 24, 26, 23, 20, 22, 24, 27, 12, 1, 12, 17, 4, 18, -1, 8, 6, 13, 8, 13,
             -1, -1, -1, -1, -1, -1, 15, 11, 3, 7, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 11, 14, 2,
             -1, -1, -1, -1, -1, 8]
assert len(PT8_OD_V1) == 61

# ---------- Follow-up 表 ----------
fu = pd.read_excel(xlsx, sheet_name="Follow-up", header=None)
assert [int(v) for v in fu.iloc[1, 8:69].tolist()] == list(range(61)), "VF 列下标不是 0..60"
fu = fu.iloc[2:].reset_index(drop=True)           # 从第 3 行读起
fu = fu[fu[0].notna()].reset_index(drop=True)
fu["subject"] = fu[0].astype(str).str.replace(r"\.0$", "", regex=True)
fu["laterality"] = fu[1].astype(str).str.strip()
fu["visit"] = fu[2].astype(int)
fu["t"] = fu[3].astype(float)
fu["iop"] = pd.to_numeric(fu[4], errors="coerce")
fu["cfp"] = fu[5].astype(str)
fu["eye_id"] = fu["subject"] + "_" + fu["laterality"]
raw = fu.loc[:, 8:68].astype(float).to_numpy()   # (n_visits, 61)
assert raw.shape[1] == 61

# 硬校验：8 号右眼第 1 次检查
m = (fu["subject"] == "8") & (fu["laterality"] == "OD") & (fu["visit"] == 1)
assert m.sum() == 1
assert raw[m.to_numpy()][0].tolist() == [float(v) for v in PT8_OD_V1], "8 号右眼第 1 次检查与 1.2 节不一致"
log.info("8 号右眼第 1 次检查 61 值与计划 1.2 节完全一致")

blind = cfg["data"]["blind_spot_idx"]
assert np.all(raw[:, blind] == -1), "盲点占位格并非全为 -1"
keep = [i for i in range(61) if i not in blind]
vals = raw[:, keep]                                # (n, 59)
censored = vals == cfg["data"]["censor_code"]
y = np.where(censored, cfg["data"]["censor_value_db"], vals)
assert np.all(vals[~censored] >= 0), "出现 −1 以外的负值"

# ---------- Baseline 表 ----------
bl = pd.read_excel(xlsx, sheet_name="Baseline", header=None).iloc[2:]
bl = bl[bl[0].notna()].reset_index(drop=True)
eyes = pd.DataFrame({
    "subject": bl[0].astype(str).str.replace(r"\.0$", "", regex=True),
    "laterality": bl[1].astype(str).str.strip(),
    "age_baseline": bl[2].astype(float),
    "gender": bl[3].astype(str),
    "iop_baseline": pd.to_numeric(bl[4], errors="coerce"),
    "cct": pd.to_numeric(bl[5], errors="coerce"),
    "total_visits_table": bl[6].astype(int),
    "PLR2": pd.to_numeric(bl[7], errors="coerce"),
    "PLR3": pd.to_numeric(bl[8], errors="coerce"),
    "MD_label": pd.to_numeric(bl[9], errors="coerce"),
    "category": bl[10].astype(str),
    "rnfl_mean": pd.to_numeric(bl[11], errors="coerce"),
    "rnfl_S": pd.to_numeric(bl[12], errors="coerce"),
    "rnfl_N": pd.to_numeric(bl[13], errors="coerce"),
    "rnfl_I": pd.to_numeric(bl[14], errors="coerce"),
    "rnfl_T": pd.to_numeric(bl[15], errors="coerce"),
})
eyes["eye_id"] = eyes["subject"] + "_" + eyes["laterality"]
assert eyes["eye_id"].is_unique

# ---------- visits ----------
visits = fu[["eye_id", "subject", "visit", "t", "iop"]].copy()
visits = visits.merge(eyes[["eye_id", "age_baseline"]], on="eye_id", how="left", validate="m:1")
visits["age"] = visits["age_baseline"] + visits["t"]
visits = visits.drop(columns="age_baseline")
visits["ms"] = y.mean(axis=1)
visits["n_censored"] = censored.sum(axis=1)
visits["has_cfp"] = fu["cfp"].str.strip() != "/"
visits = visits.sort_values(["eye_id", "visit"]).reset_index(drop=True)
order = fu.sort_values(["eye_id", "visit"]).index.to_numpy()
y, censored = y[order], censored[order]

nv = visits.groupby("eye_id").size().rename("n_visits")
eyes = eyes.merge(nv, on="eye_id", how="left", validate="1:1")
first = visits[visits.visit == 1].set_index("eye_id")
eyes["ms_baseline"] = eyes["eye_id"].map(first["ms"])
eyes["followup_years"] = eyes["eye_id"].map(visits.groupby("eye_id")["t"].max())
mismatch = (eyes["n_visits"] != eyes["total_visits_table"]).sum()
log.info("Baseline 表 Total Visits 与 Follow-up 行数不一致的眼数：%d", mismatch)

# ---------- long ----------
nvis = len(visits)
long = pd.DataFrame({
    "eye_id": np.repeat(visits["eye_id"].to_numpy(), 59),
    "subject": np.repeat(visits["subject"].to_numpy(), 59),
    "visit": np.repeat(visits["visit"].to_numpy(), 59),
    "t": np.repeat(visits["t"].to_numpy(), 59),
    "loc": np.tile(np.arange(59), nvis),
    "idx61": np.tile(np.array(keep), nvis),
    "y": y.ravel(),
    "censored": censored.ravel(),
})

# ---------- 验收 ----------
exp = cfg["data"]["expected"]
assert nvis == exp["n_visits"] == 1115, nvis
assert len(long) == exp["n_point_values"] == 65785, len(long)
assert int(long.censored.sum()) == exp["n_censored"] == 2734, long.censored.sum()
assert len(eyes) == exp["n_eyes"] == 263 and eyes.subject.nunique() == exp["n_subjects"] == 144
assert not long[["eye_id", "subject", "visit", "t", "loc", "y", "censored"]].isna().any().any()
assert not visits[["eye_id", "subject", "visit", "t", "ms"]].isna().any().any()
assert not visits.duplicated(["eye_id", "visit"]).any()
assert (visits.groupby("eye_id")["t"].apply(lambda s: s.is_monotonic_increasing)).all()
log.info("visits iop NaN=%d age NaN=%d", visits.iop.isna().sum(), visits.age.isna().sum())

out = ROOT / "data"
long.to_parquet(out / "vf_long.parquet", index=False)
visits.to_parquet(out / "visits.parquet", index=False)
eyes.to_parquet(out / "eyes.parquet", index=False)
summary = dict(n_eyes=len(eyes), n_subjects=int(eyes.subject.nunique()), n_visits=nvis, n_point_values=len(long),
               n_censored=int(long.censored.sum()), censored_frac=float(long.censored.mean()),
               total_visits_mismatch=int(mismatch), keep_idx61=keep)
write_json(summary, out / "build_summary.json")
log.info("全部验收通过：%s", {k: v for k, v in summary.items() if k != "keep_idx61"})
