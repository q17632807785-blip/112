"""05_calibrate_sim.py — 预注册第 3.5 节：模拟器校准（只用开发集真实数据和 sim_train；从不读 sim_test）。

五个统计量（模拟 vs 开发集真实）：MS 残差 SD、检查整体偏移 SD、逐点残差 SD（按拟合值分 7 层，每层）、
观察 MS 斜率的 SD、删失比例。算法与 03_noise_model.py 相同（逐眼/逐序列逐点 OLS，残差乘 sqrt(n/(n−2))，
δ_v = 59 点残差平均，e = r − δ_v，拟合值截到 ≥0 后分层）。
模拟侧：sim_train 的 k = 4 版本，S0、S5、S2 三个情景按开发集观察 MS 斜率三档的眼数占比加权（R5 规则 C-1：
五个统计量都用这一混合，第 3.5 节只对斜率 SD 写明了混合）。
调参（R5 规则 C-2）：只调两个乘性系数 kappa_offset、kappa_point；每轮
  kappa_offset ← kappa_offset × (真实偏移 SD / 模拟偏移 SD)，kappa_point ← kappa_point × (真实逐点 SD / 模拟逐点 SD)（全体合并）；
情景、模板、空间相关参数不调。最多 3 轮；五项全部合格或已调满 3 轮即 final = true。
用法：python src/05_calibrate_sim.py   （每次运行 = 评估当前 sim_train 一轮；不合格且未满 3 轮时写出下一轮系数，
然后重新运行 04_simulator.py --split sim_train 再运行本脚本）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(ROOT, *a)
rms = lambda a, w=None: float(np.sqrt(np.average(np.square(a), weights=w)))


def resid_stats(groups, bins, w_series=None):
    """groups: list of (t (n,), Y (n,59), C (n,59))；w_series: 每条序列的权重。返回统计量字典。"""
    w_series = np.ones(len(groups)) if w_series is None else np.asarray(w_series, float)
    ms_r, ms_w, d_all, d_w, e_all, f_all, e_w, slopes, cens, cens_w = [], [], [], [], [], [], [], [], [], []
    for (t, Y, C), w in zip(groups, w_series):
        n = len(t); X = np.c_[np.ones(n), t]; k = np.sqrt(n / (n - 2))
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; F = X @ b; R = (Y - F) * k
        ms = Y.mean(1); bm = np.linalg.lstsq(X, ms, rcond=None)[0]
        ms_r.append((ms - X @ bm) * k); ms_w.append(np.full(n, w)); slopes.append(bm[1])
        dv = R.mean(1); d_all.append(dv); d_w.append(np.full(n, w))
        e_all.append((R - dv[:, None]).ravel()); f_all.append(np.clip(F, 0, None).ravel()); e_w.append(np.full(R.size, w))
        cens.append(C.ravel()); cens_w.append(np.full(C.size, w))
    cat = np.concatenate
    e, f, ew = cat(e_all), cat(f_all), cat(e_w)
    bi = np.digitize(f, np.asarray(bins, float)[1:-1])
    sl = np.asarray(slopes); sw = w_series
    mu = np.average(sl, weights=sw)
    return dict(ms_resid_sd=rms(cat(ms_r), cat(ms_w)), offset_sd=rms(cat(d_all), cat(d_w)),
                point_resid_sd=rms(e, ew),
                point_resid_sd_by_bin=[rms(e[bi == i], ew[bi == i]) if (bi == i).any() else None for i in range(len(bins) - 1)],
                ms_slope_sd=float(np.sqrt(np.average((sl - mu) ** 2, weights=sw))),
                censor_frac=float(np.average(cat(cens).astype(float), weights=cat(cens_w))))


def main():
    cfg = yaml.safe_load(open(P("config.yaml"), encoding="utf-8")); tol = cfg["sim_calibration_tolerance"]
    bins = cfg["noise_model"]["fitted_bins_db"]
    spl = json.load(open(P("data", "split.json"))); dev = set(spl["dev_subjects"])
    vl = pd.read_parquet(P("data", "vf_long.parquet")); vl = vl[vl.subject.isin(dev)]
    real = []
    for eid, g in vl.groupby("eye_id"):
        Y = g.pivot_table(index="visit", columns="loc59", values="y").sort_index()
        C = g.pivot_table(index="visit", columns="loc59", values="censored").sort_index()
        t = g.groupby("visit").t.first().loc[Y.index].to_numpy(float)
        real.append((t, Y.to_numpy(float), C.to_numpy().astype(bool)))
    R = resid_stats(real, bins)
    # 混合权重：开发集眼按全序列 OLS 的 MS 斜率分三档（第 3.5 节）
    lo, hi = tol["ms_slope_mix_bins_db_per_yr"]
    sl = np.array([np.polyfit(t, Y.mean(1), 1)[0] for t, Y, _ in real])
    pi = {"S0": float(np.mean(sl >= hi)), "S5": float(np.mean((sl >= lo) & (sl < hi))), "S2": float(np.mean(sl < lo))}
    path = P("data", "sim_train.npz")
    assert os.path.basename(path) == "sim_train.npz"          # 只读 sim_train
    z = np.load(path, allow_pickle=False)
    scn = z["meta_scenario"]; Y4, C4, T4 = z["k4_y"], z["k4_censored"], z["k4_t"]
    groups, w = [], []
    for s, p in pi.items():
        idx = np.flatnonzero(scn == s)
        for i in idx:
            groups.append((T4[i], Y4[i], C4[i])); w.append(p / len(idx))
    S = resid_stats(groups, bins, w)
    rel = lambda a, b: abs(a - b) / b
    checks = dict(
        ms_resid_sd=rel(S["ms_resid_sd"], R["ms_resid_sd"]) <= tol["ms_resid_sd_rel"],
        offset_sd=rel(S["offset_sd"], R["offset_sd"]) <= tol["offset_sd_rel"],
        point_resid_sd_by_bin=all(s is not None and rel(s, r) <= tol["point_resid_sd_rel_per_bin"]
                                  for s, r in zip(S["point_resid_sd_by_bin"], R["point_resid_sd_by_bin"]) if r is not None),
        ms_slope_sd=rel(S["ms_slope_sd"], R["ms_slope_sd"]) <= tol["ms_slope_sd_rel"],
        censor_frac=abs(S["censor_frac"] - cfg["noise_model"]["target_censor_frac"]) <= tol["censor_frac_abs"])
    cal_path = P("data", "sim_calibration.json")
    prev = json.load(open(cal_path, encoding="utf-8")) if os.path.exists(cal_path) else dict(round=0, kappa_offset=1.0, kappa_point=1.0, history=[])
    if prev.get("final"):
        sys.exit("校准已 final，不再调整。")
    rnd = prev["round"]; ok = all(checks.values())
    hist = prev["history"] + [dict(round=rnd, kappa_offset=prev["kappa_offset"], kappa_point=prev["kappa_point"],
                                   real=R, sim=S, mix_weights=pi, checks=checks, all_pass=ok)]
    if ok or rnd >= tol["max_tuning_rounds"]:
        new = dict(round=rnd, kappa_offset=prev["kappa_offset"], kappa_point=prev["kappa_point"], final=True, history=hist)
    else:
        new = dict(round=rnd + 1, kappa_offset=prev["kappa_offset"] * R["offset_sd"] / S["offset_sd"],
                   kappa_point=prev["kappa_point"] * R["point_resid_sd"] / S["point_resid_sd"], final=False, history=hist)
    with open(cal_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(new, f, ensure_ascii=False, indent=1); f.write("\n")
    print(json.dumps(dict(round=rnd, checks=checks, final=new["final"], next_kappa=(new["kappa_offset"], new["kappa_point"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
