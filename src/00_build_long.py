"""00_build_long.py — 预注册第 3.1 节 / 计划第 5.2 节：GRAPE 长表构建。
输出 data/vf_long.parquet, data/visits.parquet, data/eyes.parquet。"""
import os, json, hashlib
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
XLSX = os.path.normpath(os.path.join(ROOT, cfg["paths"]["raw_xlsx"]))
DATA = os.path.join(ROOT, cfg["paths"]["data"])
BLIND = cfg["data"]["blind_spot_idx"]; CENS = cfg["data"]["censor_code"]
EXP = cfg["data"]["expected"]

fu = pd.read_excel(XLSX, "Follow-up", header=None).iloc[2:].reset_index(drop=True)
bl = pd.read_excel(XLSX, "Baseline", header=None).iloc[2:].reset_index(drop=True)

def sid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s

fu = fu[fu[0].notna()].reset_index(drop=True)
subj = fu[0].map(sid); eye = fu[1].astype(str).str.strip()
vis = fu[2].astype(int); t = fu[3].astype(float); iop = pd.to_numeric(fu[4], errors="coerce")
raw = fu.iloc[:, 8:69].astype(float).to_numpy()
assert raw.shape[1] == cfg["data"]["n_vf_values_raw"] == 61
assert not np.isnan(raw).any(), "NaN in raw VF"

# 8 号右眼第 1 次检查（计划 1.2 节 / 原文图 2a,2b）
REF8 = [30,31,26,22,26,29,30,12,24,26,23,20,22,24,27,12,1,12,17,4,18,-1,8,6,13,8,13,
        -1,-1,-1,-1,-1,-1,15,11,3,7,1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,2,11,14,2,
        -1,-1,-1,-1,-1,8]
m8 = (subj == "8") & (eye == "OD") & (vis == 1)
assert m8.sum() == 1 and raw[m8.values][0].astype(int).tolist() == REF8, "8 OD v1 mismatch"
assert (raw[:, BLIND] == CENS).all(), "blind-spot placeholders not all -1"

keep = [i for i in range(61) if i not in BLIND]
v59 = raw[:, keep]
cens = v59 == CENS
assert (v59[~cens] >= 0).all()
y = np.where(cens, cfg["data"]["censor_value_db"], v59)

bl_subj = bl[0].map(sid); bl_eye = bl[1].astype(str).str.strip()
base = pd.DataFrame({
    "subject": bl_subj, "eye": bl_eye,
    "age_baseline": bl[2].astype(float), "sex": bl[3].astype(str).str.strip(),
    "iop_baseline": pd.to_numeric(bl[4], errors="coerce"), "cct": pd.to_numeric(bl[5], errors="coerce"),
    "total_visits_official": bl[6].astype(int),
    "label_PLR2": bl[7].astype(int), "label_PLR3": bl[8].astype(int), "label_MD": bl[9].astype(int),
    "diagnosis": bl[10].astype(str).str.strip(),
    "rnfl_mean": pd.to_numeric(bl[11].where(lambda s: s != '/'), errors='coerce'), "rnfl_S": pd.to_numeric(bl[12].where(lambda s: s != '/'), errors='coerce'), "rnfl_N": pd.to_numeric(bl[13].where(lambda s: s != '/'), errors='coerce'),
    "rnfl_I": pd.to_numeric(bl[14].where(lambda s: s != '/'), errors='coerce'), "rnfl_T": pd.to_numeric(bl[15].where(lambda s: s != '/'), errors='coerce')})
base["eye_id"] = base["subject"] + "_" + base["eye"]
assert base["eye_id"].is_unique

visits = pd.DataFrame({"eye_id": subj + "_" + eye, "subject": subj, "eye": eye, "visit": vis, "t": t,
                       "iop": iop, "fundus_image": fu[5].astype(str), "ms": y.mean(1),
                       "n_censored": cens.sum(1)})
visits = visits.merge(base[["eye_id", "age_baseline"]], on="eye_id", how="left", validate="m:1")
assert visits["age_baseline"].notna().all(), "follow-up eye missing in Baseline"
visits["age"] = visits["age_baseline"] + visits["t"]
visits = visits.drop(columns="age_baseline")
order = visits.sort_values(["subject", "eye", "visit"]).index
assert not visits.duplicated(["eye_id", "visit"]).any()
# 每眼内 t 单调、第 1 次 t=0
g = visits.sort_values(["eye_id", "visit"]).groupby("eye_id")
assert (g["t"].first() == 0).all() and g["t"].apply(lambda s: s.is_monotonic_increasing).all()

vf_long = pd.DataFrame({
    "eye_id": np.repeat(visits["eye_id"].values, 59), "subject": np.repeat(visits["subject"].values, 59),
    "visit": np.repeat(visits["visit"].values, 59), "t": np.repeat(visits["t"].values, 59),
    "loc": np.tile(np.array(keep, dtype=int), len(visits)),          # loc = GRAPE 原始下标 0–60（不含 21,32）
    "loc59": np.tile(np.arange(59), len(visits)),
    "y": y.ravel(), "censored": cens.ravel()})

nv = visits.groupby("eye_id").size().rename("n_visits")
ms1 = visits[visits.visit == 1].set_index("eye_id")["ms"].rename("baseline_ms")
eyes = base.set_index("eye_id").join(nv, how="inner").join(ms1).reset_index()
assert (eyes["n_visits"] == eyes["total_visits_official"]).all(), "visit count != Baseline Total Visits"

# 验收
assert len(eyes) == EXP["n_eyes"] == len(base), len(eyes)
assert eyes["subject"].nunique() == EXP["n_subjects"]
assert len(visits) == EXP["n_visits"]
assert len(vf_long) == EXP["n_point_values"]
assert int(vf_long["censored"].sum()) == EXP["n_censored"], int(vf_long["censored"].sum())
for df in (vf_long, visits.drop(columns=["iop"]), eyes.drop(columns=["iop_baseline", "cct"] + [c for c in eyes.columns if c.startswith("rnfl_")])):
    assert not df.isna().any().any()

os.makedirs(DATA, exist_ok=True)
vf_long.to_parquet(os.path.join(DATA, "vf_long.parquet"), index=False)
visits.to_parquet(os.path.join(DATA, "visits.parquet"), index=False)
eyes.to_parquet(os.path.join(DATA, "eyes.parquet"), index=False)
print(f"OK eyes={len(eyes)} subjects={eyes.subject.nunique()} visits={len(visits)} "
      f"values={len(vf_long)} censored={int(vf_long.censored.sum())} "
      f"({vf_long.censored.mean():.4f}); IOP NaN={int(visits.iop.isna().sum())}; RNFL 缺失(\"/\") 眼数={int(eyes.rnfl_mean.isna().sum())}")
print("visits per eye:", eyes.n_visits.value_counts().sort_index().to_dict())
print("diagnosis:", eyes.diagnosis.value_counts().to_dict())
