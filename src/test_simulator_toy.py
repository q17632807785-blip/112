"""test_simulator_toy.py — 模拟器的玩具单元测试（R5）。

只用公开的 G1 点位几何（data/g1_locations.csv）和**固定的玩具噪声模型、玩具模板**（不是 GRAPE 派生的模板，
也不读 noise_model.json、split.json 或任何受试者数据），检查形状、确定性、删失、情景真值。
输出只写到临时目录，测试结束即删除；不生成 sim_train / sim_test。
用法：python src/test_simulator_toy.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
S = cfg["simulator"]
geo = sc.make_geometry(pd.read_csv(os.path.join(ROOT, "data", "g1_locations.csv")))
s3c, _ = sc.s3_candidates(geo)
s4c, elig = sc.s4_candidates(geo, S["scenarios"]["S4"]["max_ecc_deg"], tuple(S["scenarios"]["S4"]["n_points"]))

TOY_NM = sc.NoiseModel(sd_grid=np.array([0.0, 40.0]), sd_vals=np.array([2.0, 2.0]),
                       offset_sd_by_tertile=np.array([1.0, 1.0, 1.0]), a=0.5, lam=5.0, spatial=True)
base_hi = np.full(59, 25.0); base_lo = base_hi.copy(); base_lo[:10] = -5.0
TOY_T = [sc.Template("toyA", "t1", 0, 0, base_hi, np.array([0.0, 1.0, 2.0]), np.array([15.0, 16.0, 17.0]), 60.0),
         sc.Template("toyB", "t2", 1, 2, base_lo, np.arange(7) * 0.5, np.full(7, 14.0), 70.0)]
POOL = np.array([0.5, 1.0])
SEED = 12345
fails = []


def check(name, cond):
    print(("[OK]  " if cond else "[FAIL]") + " " + name)
    if not cond:
        fails.append(name)


# ---- 1. 点集规则 ----
check("S3 候选非空，跨 2 簇和 3 簇都有", {c["n_clusters"] for c in s3c} == {2, 3})
csize = np.bincount(geo.cluster)
check("S3 每个候选：单一半视野、≥3 点、每个触及簇只占部分点、不含中心点、至少一点偏心度 > 10°",
      all(len(set(geo.hemi[c["points"]])) == 1 and len(c["points"]) >= 3 and geo.center not in c["points"]
          and all(np.sum(geo.cluster[c["points"]] == k) < csize[k] for k in c["clusters"])
          and np.hypot(*geo.xy[c["points"]].T).max() > 10 for c in s3c))
check("S4 每个候选：偏心度 ≤ 10°、非中心点同一半视野、在相邻图上连通",
      all(np.hypot(*geo.xy[s].T).max() <= 10 and len({geo.hemi[i] for i in s if i != geo.center}) == 1
          for n in s4c for s in s4c[n]))
check("相邻约定：中心点与上下两侧 ≤6° 的点都相邻（第 6 部分第 3 条，58 对）", int(np.triu(geo.neighbor).sum()) == 58)
check("噪声块：中心点并入上半视野块", geo.noise_block[geo.center] == 0)

# ---- 2. Tobit ----
t = np.array([0.0, 1.0, 2.0, 3.0]); y = np.array([20.0, 19.0, 18.5, 17.0])
a, b, st = sc.tobit_point(t, y, np.zeros(4, bool), 2.0)
ao, bo = np.polyfit(t, y, 1)[::-1]
check("Tobit 无删失 = OLS", st == "ols_no_censoring" and abs(a - ao) < 1e-12 and abs(b - bo) < 1e-12)
check("Tobit 全部删失 → −10", sc.tobit_point(t, np.zeros(4), np.ones(4, bool), 2.0)[0] == -10.0)
a2, _, st2 = sc.tobit_point(t, np.array([3.0, 1.0, 0.0, 0.0]), np.array([0, 0, 1, 1], bool), 2.0)
check("Tobit 部分删失：MLE 收敛、截距有限", st2 == "mle" and np.isfinite(a2))
Yt = np.c_[np.tile(y[:, None], (1, 58)), np.zeros(4)]; Ct = np.zeros((4, 59), bool); Ct[:, -1] = True
bt, _, _ = sc.tobit_template(t, Yt, Ct, TOY_NM, 0)
check("模板截距截到 [−10, 35]", bt.min() >= -10 and bt.max() <= 35 and bt[-1] == -10.0)

# ---- 3. 生成与确定性 ----
args = (TOY_T, sc.SCENARIOS, 3, SEED, geo, TOY_NM, S, s3c, s4c, POOL)
arr1 = sc.assemble(*args); arr2 = sc.assemble(*args)
with tempfile.TemporaryDirectory() as td:
    p1, p2 = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
    sc.write_npz_deterministic(p1, arr1); sc.write_npz_deterministic(p2, arr2)
    h1, h2 = sc.sha256_file(p1), sc.sha256_file(p2)
    z = np.load(p1)
    check("写出文件可读回且与内存一致", all(np.array_equal(z[k], arr1[k], equal_nan=arr1[k].dtype.kind == "f") for k in arr1))
check(f"两次生成字节一致（SHA-256 {h1[:12]}…）", h1 == h2)
N = len(TOY_T) * 8 * 3
check("序列数 = 模板 × 8 情景 × 重复", len(arr1["meta_scenario"]) == N)
check("形状：k{3,4,5,6}_y = (N, k, 59)", all(arr1[f"k{k}_y"].shape == (N, k, 59) for k in S["k_versions"]))
check("形状：truth_cluster (N,10)、truth_global (N,)", all(arr1[f"k{k}_truth_cluster_obs"].shape == (N, 10)
                                                         and arr1[f"k{k}_truth_global_lat"].shape == (N,) for k in S["k_versions"]))
# 顺序无关：单独生成某一条，与整体生成中的同一条相同
L = np.linalg.cholesky(TOY_NM.corr(geo) + 1e-10 * np.eye(59))
_, one = sc.simulate_series(TOY_T[1], "S3", 2, SEED, geo, TOY_NM, L, S, s3c, s4c, POOL)
i = int(np.flatnonzero((arr1["meta_template_eye"] == "toyB") & (arr1["meta_scenario"] == "S3") & (arr1["meta_rep"] == 2))[0])
check("单条重算 = 整体生成中的同一条（与生成顺序无关）", np.array_equal(one[4]["y"], arr1["k4_y"][i]))
check("不同种子 → 不同数据", not np.array_equal(sc.assemble(TOY_T, ["S0"], 1, SEED + 1, geo, TOY_NM, S, s3c, s4c, POOL)["k4_y"],
                                           arr1["k4_y"][arr1["meta_scenario"] == "S0"][[0, 3]]))

# ---- 4. 嵌套、删失、真值 ----
check("嵌套：k=4 的时间与 k=6 的前 4 个相同", np.array_equal(arr1["k4_t"], arr1["k6_t"][:, :4]))
check("模板 ≥k 次时用真实前 k 个时间点", np.allclose(arr1["k6_t"][arr1["meta_template_eye"] == "toyB"][0], np.arange(6) * 0.5))
tA = arr1["k6_t"][arr1["meta_template_eye"] == "toyA"]
check("不足 k 次时从间隔池补抽", np.allclose(tA[:, :3], [0, 1, 2]) and np.isin(np.round(np.diff(tA[:, 2:]), 9), POOL).all())
check("补抽检查 iop = NaN", np.isnan(arr1["k6_iop"][arr1["meta_template_eye"] == "toyA"][:, 3:]).all())
for k in S["k_versions"]:
    yk, ck = arr1[f"k{k}_y"], arr1[f"k{k}_censored"]
    if not ((yk[ck] == 0).all() and (yk[~ck] > 0).all()):
        check(f"删失 k={k}", False)
check("删失：censored ⇔ 观测 = 0，且有删失发生", arr1["k4_censored"].any())
sc_ = arr1["meta_scenario"]; aff = arr1["meta_affected"]; tp = arr1["k4_truth_point_lat"]
check("S0 潜在逐点真值 = −0.1", np.allclose(tp[sc_ == "S0"], -0.1))
check("S5 潜在逐点真值 ∈ {−0.3, −0.6} 且整眼一致", all(np.allclose(r, r[0]) and round(r[0], 9) in (-0.3, -0.6) for r in tp[sc_ == "S5"]))
ok2 = True
for i in np.flatnonzero(sc_ == "S2"):
    ok2 &= np.allclose(tp[i][~aff[i]], -0.1) and np.allclose(tp[i][aff[i]], arr1["meta_slope_draw"][i])
    cl = set(geo.cluster[aff[i]]); ok2 &= len(cl) in (1, 2) and set(np.flatnonzero(np.isin(geo.cluster, list(cl)))) == set(np.flatnonzero(aff[i]))
check("S2 受影响 = 整个 1–2 个簇、斜率为抽到的档；其余 −0.1", ok2)
check("S3/S4/S7 未受影响点 = −0.1（R5 规则 U-1）", all(np.allclose(tp[i][~aff[i]], -0.1) for i in np.flatnonzero(np.isin(sc_, ["S3", "S4", "S7"]))))
check("S3/S7 点集 ∈ S3 候选", all(sorted(np.flatnonzero(aff[i]).tolist()) in [c["points"] for c in s3c] for i in np.flatnonzero(np.isin(sc_, ["S3", "S7"]))))
check("S4 点集 ∈ S4 候选", all(sorted(np.flatnonzero(aff[i]).tolist()) in [s for n in s4c for s in s4c[n]] for i in np.flatnonzero(sc_ == "S4")))
check("S7：k=4 时受影响点潜在 OLS 斜率 = −1", all(np.allclose(tp[i][aff[i]], -1.0) for i in np.flatnonzero(sc_ == "S7")))
check("S7：k=6 时更陡（加速持续，a 固定）", all((arr1["k6_truth_point_lat"][i][aff[i]] < -1.0).all() for i in np.flatnonzero(sc_ == "S7")))
s1 = np.flatnonzero(sc_ == "S1")
check("S1 潜在真值 ≤ 0、阴性", all((arr1["k4_truth_point_lat"][i] <= 1e-12).all() for i in s1) and not arr1["meta_is_progressing"][s1].any())
f6 = arr1["k6_f"]
ok6 = True
for i in np.flatnonzero(sc_ == "S1"):
    j = 2 + int(np.floor(arr1["meta_s1_u"][i] * 5)); d = f6[i][:, 0] - f6[i][0, 0]
    ok6 &= np.allclose(d[: j - 1], 0) and np.allclose(d[j - 1:], -arr1["meta_s1_drop"][i]) and 1 <= arr1["meta_s1_drop"][i] <= 2
check("S1：第 j 次检查起一次性下降 [1,2] dB，j ∈ {2..k}", ok6)
check("观察尺度真值 ≥ 潜在尺度真值的方向（地板只会让斜率变平）",
      np.all(arr1["k4_truth_point_obs"][arr1["meta_template_eye"] == "toyA"] >= arr1["k4_truth_point_lat"][arr1["meta_template_eye"] == "toyA"] - 1e-9))
check("簇级真值 = 簇内逐点真值平均", np.allclose(arr1["k4_truth_cluster_obs"][:, 0], arr1["k4_truth_point_obs"][:, geo.cluster == 1].mean(1)))
# ---- 5. 变体 ----
a8 = sc.assemble(TOY_T, ["S0"], 2, SEED, geo, TOY_NM, S, s3c, s4c, POOL, no_offset=True)
a10 = sc.assemble(TOY_T, ["S0"], 2, SEED, geo, TOY_NM, S, s3c, s4c, POOL, independent_noise=True)
m = arr1["meta_scenario"] == "S0"
check("SA8/SA10 与主版本同时间表（配对）", np.array_equal(a8["k6_t"], arr1["k6_t"][m][[0, 1, 3, 4]]) and np.array_equal(a10["k6_t"], a8["k6_t"]))

# ---- 6. 端到端（玩具项目目录）：04 → 05 → 04 sim_test 的守卫与流程 ----
# 在临时目录里搭一个假项目：复制本目录的模拟器源码与公开点位文件，
# 受试者数据、划分、噪声模型全部是这里随手造的玩具数据（与 GRAPE 无关），结束后整个目录删除。
import json, shutil, subprocess  # noqa: E401


def toy_project(td):
    os.makedirs(os.path.join(td, "src")); os.makedirs(os.path.join(td, "data"))
    for f in ["sim_core.py", "04_simulator.py", "05_calibrate_sim.py", "test_simulator_toy.py"]:
        shutil.copy(os.path.join(HERE, f), os.path.join(td, "src", f))
    shutil.copy(os.path.join(ROOT, "data", "g1_locations.csv"), os.path.join(td, "data", "g1_locations.csv"))
    rng = np.random.default_rng(7)
    eyes, visits, long = [], [], []
    for s in range(1, 9):
        for e in ("OD", "OS"):
            eid = f"{s}_{e}"; n = 3 + (s + len(e)) % 5; tt = np.r_[0, np.cumsum(rng.uniform(0.4, 1.2, n - 1))]
            base = rng.uniform(5, 30, 59)
            Y = np.round(base[None] - 0.3 * tt[:, None] + rng.normal(0, 2, (n, 59)))
            C = Y < 0; Y = np.where(C, 0, Y)
            eyes.append(dict(eye_id=eid, subject=str(s), age_baseline=50.0 + s, baseline_ms=float(Y[0].mean()), n_visits=n))
            for v in range(n):
                visits.append(dict(eye_id=eid, subject=str(s), visit=v + 1, t=float(tt[v]), iop=15.0))
                for p in range(59):
                    long.append(dict(eye_id=eid, subject=str(s), visit=v + 1, t=float(tt[v]), loc59=p, y=float(Y[v, p]), censored=bool(C[v, p])))
    pd.DataFrame(eyes).to_parquet(os.path.join(td, "data", "eyes.parquet"))
    pd.DataFrame(visits).to_parquet(os.path.join(td, "data", "visits.parquet"))
    pd.DataFrame(long).to_parquet(os.path.join(td, "data", "vf_long.parquet"))
    json.dump(dict(dev_subjects=[str(s) for s in range(1, 6)], holdout_subjects=[str(s) for s in range(6, 9)]),
              open(os.path.join(td, "data", "split.json"), "w"))
    json.dump({"offset": {"stratified": True, "sd_deconvolved_by_tertile": [1.0, 1.0, 1.0], "sd_deconvolved_pooled": 1.0},
               "point_sd": {"spline": {"grid_db": [0.0, 40.0], "sd_db": [2.0, 2.0]}},
               "spatial": {"a": 0.5, "lambda_deg": 5.0, "used": True}}, open(os.path.join(td, "data", "noise_model.json"), "w"))
    c = dict(cfg); c["revisions"] = list(cfg["revisions"]) + ["R5 toy"]
    yaml.safe_dump(c, open(os.path.join(td, "config.yaml"), "w", encoding="utf-8"), allow_unicode=True)
    rows = "".join(f"| `src/{f}` | `{sc.sha256_file(os.path.join(td, 'src', f))}` |\n"
                   for f in ["sim_core.py", "04_simulator.py", "05_calibrate_sim.py", "test_simulator_toy.py"])
    open(os.path.join(td, "preregistration.md"), "w", encoding="utf-8").write("toy\n" + rows)


def run(td, *a):
    return subprocess.run([sys.executable, os.path.join(td, "src", a[0]), *a[1:]], cwd=td, capture_output=True, text=True)


if True:
    with tempfile.TemporaryDirectory() as td:
        toy_project(td)
        r = run(td, "04_simulator.py", "--split", "sim_test", "--confirm-r5-published")
        check("sim_test：校准未结束时拒绝生成", r.returncode != 0 and "校准尚未结束" in r.stderr and not os.path.exists(os.path.join(td, "data", "sim_test.npz")))
        r1 = run(td, "04_simulator.py", "--split", "sim_train", "--confirm-r5-published")
        h_a = r1.stdout.split()[0] if r1.returncode == 0 else None
        r1b = run(td, "04_simulator.py", "--split", "sim_train", "--confirm-r5-published")
        check("端到端 sim_train 生成成功且重复运行字节一致", r1.returncode == 0 and h_a == r1b.stdout.split()[0])
        if r1.returncode:
            print(r1.stderr[-2000:])
        zt = np.load(os.path.join(td, "data", "sim_train.npz"))
        check("端到端 sim_train：模板 = 开发集 10 眼 × 8 × 10", len(zt["meta_scenario"]) == 10 * 8 * cfg["simulator"]["n_rep_sim_train"])
        final = False
        for _ in range(5):
            rc = run(td, "05_calibrate_sim.py")
            if rc.returncode:
                print(rc.stderr[-2000:]); break
            cal = json.load(open(os.path.join(td, "data", "sim_calibration.json"), encoding="utf-8"))
            if cal["final"]:
                final = True; break
            run(td, "04_simulator.py", "--split", "sim_train", "--confirm-r5-published")
        check("校准循环：最多 3 轮调参后 final", final and cal["round"] <= cfg["sim_calibration_tolerance"]["max_tuning_rounds"])
        r2 = run(td, "04_simulator.py", "--split", "sim_test", "--confirm-r5-published")
        check("校准结束后 sim_test 生成成功，并写出 SHA-256 日志", r2.returncode == 0 and os.path.exists(os.path.join(td, "logs", "sim_test.npz.sha256")))
        r3 = run(td, "04_simulator.py", "--split", "sim_test", "--confirm-r5-published")
        check("sim_test 已存在时拒绝覆盖", r3.returncode != 0 and "拒绝覆盖" in r3.stderr)
        open(os.path.join(td, "src", "sim_core.py"), "a").write("\n# tamper\n")
        r4 = run(td, "04_simulator.py", "--split", "sim_train", "--confirm-r5-published")
        check("源码哈希与预注册不符时拒绝生成", r4.returncode != 0 and "SHA-256 与预注册 R5 不一致" in r4.stderr)

print(f"\n{len(fails)} 项失败")
sys.exit(1 if fails else 0)
