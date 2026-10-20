"""sim_core.py — 预注册第 3.3–3.5 节与修订 R5 的模拟器核心（纯函数，不读写任何数据文件）。

本模块只做计算：点位几何、S3/S4 点集生成规则、Tobit 模板基线、时间表、情景真实轨迹、噪声、删失、真值。
读 GRAPE 派生文件、写模拟数据集的入口在 04_simulator.py；校准统计量在 05_calibrate_sim.py。
所有随机数都来自 numpy.random.SeedSequence([seed, eye_index, scenario_index, rep]) 派生的独立流，
因此结果只取决于配置种子、模板眼排序和参数，与生成顺序无关。
"""
from __future__ import annotations

import hashlib
import io
import itertools
import zipfile
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
K_MAX = 6                      # 嵌套时间表的最大检查次数（config.simulator.k_versions 的最大值）
N_PTS = 59

# ---------------------------------------------------------------------------
# R5 规则常数（预注册修订 R5 逐条列出；改动任何一个都必须另立修订）
# ---------------------------------------------------------------------------
# Jansonius et al. 2009 (Vision Res 49:2157) 神经纤维束模型，均值参数
JANS_DISC_X, JANS_DISC_Y = 15.0, 2.0      # 视盘中心，视网膜坐标（右眼，中心凹为原点，+x 指向视盘）
JANS_R0 = 4.0
JANS_BETA_SUP, JANS_BETA_INF = -1.9, 0.7
JANS_PHI0_MIN = 60.0                      # 模型只定义 |phi0| ∈ [60°, 180°]
JANS_SOLVE_GRID = 0.001                   # phi0 网格步长（度）
JANS_SOLVE_TOL = 0.05                     # |残差| < 0.05° 视为有解
S3_WINDOW_DEG = 20.0                      # S3 弧：同半视野内入盘角窗口宽度
S3_MIN_POINTS = 3
S3_MIN_MAX_ECC_DEG = 10.0                 # 弧至少有一个点偏心度 > 10°（与 S4 的中心 10° 区分）
NEIGHBOR_MAX_DIST_DEG = 6.0               # 第 6 部分第 3 条
TOBIT_A_BOUNDS = (-100.0, 100.0)          # Tobit 优化框（之后再按 config 截到 [-10, 35]）
TOBIT_B_BOUNDS = (-30.0, 30.0)            # dB/年；防止部分删失时 MLE 发散


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    xy: np.ndarray            # (59, 2) 右眼视野坐标，+x 颞侧
    cluster: np.ndarray       # (59,) 1..10；中心点为 0
    hemi: np.ndarray          # (59,) +1 上半视野，-1 下半视野，0 中心点
    center: int               # 中心点下标
    neighbor: np.ndarray      # (59, 59) bool，第 6 部分第 3 条约定（中心点与两侧 ≤6° 点都相邻）
    noise_block: np.ndarray   # (59,) 生成噪声的块：0 上半视野（含中心点），1 下半视野（第 3.3 节）
    dist: np.ndarray = field(repr=False, default=None)

    @property
    def n_clusters(self) -> int:
        return int(self.cluster.max())


def make_geometry(loc_df) -> Geometry:
    """loc_df = data/g1_locations.csv（pandas DataFrame）。"""
    p = loc_df[~loc_df.is_blind_spot.astype(bool)].sort_values("loc59")
    assert list(p.loc59) == list(range(N_PTS))
    xy = p[["x_deg", "y_deg"]].to_numpy(float)
    cl = p["cluster"].fillna(0).astype(int).to_numpy()
    center = int(np.flatnonzero((xy[:, 0] == 0) & (xy[:, 1] == 0))[0])
    assert cl[center] == 0 and (cl[np.arange(N_PTS) != center] > 0).all()
    hemi = np.sign(xy[:, 1]).astype(int)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    nb = (d <= NEIGHBOR_MAX_DIST_DEG) & (xy[:, 1][:, None] * xy[:, 1][None] >= 0)
    np.fill_diagonal(nb, False)
    block = np.where(xy[:, 1] >= 0, 0, 1)
    return Geometry(xy=xy, cluster=cl, hemi=hemi, center=center, neighbor=nb, noise_block=block, dist=d)


def jansonius_phi0(x_vf: float, y_vf: float):
    """G1 点（右眼视野坐标）→ Jansonius 2009 模型的入盘角 phi0（度）。无解返回 None。
    视野→视网膜：x_ret = x_vf（颞侧视野 = 鼻侧视网膜 = 视盘一侧），y_ret = −y_vf。
    附录 A：x' = x − 15；y' = y − 2(x/15)² (x > 0)，否则 y' = y；r, φ 为 (x', y') 的极坐标。
    方程 (1)(3)–(6)：φ = φ0 + b(φ0)(r − r0)^c(φ0)，在 |φ0| ∈ [60, 180] 的网格上找残差最小的 φ0。"""
    xr, yr = float(x_vf), -float(y_vf)
    xp = xr - JANS_DISC_X
    yp = yr - JANS_DISC_Y * (xr / JANS_DISC_X) ** 2 if xr > 0 else yr
    r = float(np.hypot(xp, yp)); phi = float(np.degrees(np.arctan2(yp, xp)))
    g = np.arange(JANS_PHI0_MIN, 180.0 + 1e-9, JANS_SOLVE_GRID)
    if yp > 0:   # 上方视网膜（下半视野）
        b = np.exp(JANS_BETA_SUP + 3.9 * np.tanh(-(g - 121.0) / 14.0)); c = 1.9 + 1.4 * np.tanh((g - 121.0) / 14.0)
    else:        # 下方视网膜（上半视野）
        g = -g
        b = -np.exp(JANS_BETA_INF + 1.5 * np.tanh(-(-g - 90.0) / 25.0)); c = 1.0 + 0.5 * np.tanh((-g - 90.0) / 25.0)
    h = g + b * np.maximum(r - JANS_R0, 0.0) ** c - phi
    i = int(np.argmin(np.abs(h)))
    return float(g[i]) if abs(h[i]) < JANS_SOLVE_TOL else None


def s3_candidates(geo: Geometry):
    """S3 弧候选点集（R5 规则 S3-1…S3-5）。返回 list[dict]，按 (半视野, 点集) 排序，去重。"""
    csize = np.bincount(geo.cluster, minlength=geo.n_clusters + 1)
    ang = {}
    for i in range(N_PTS):
        if i == geo.center:
            continue
        f = jansonius_phi0(*geo.xy[i])
        if f is not None:
            ang[i] = abs(f)
    ecc = np.hypot(geo.xy[:, 0], geo.xy[:, 1])
    out = {}
    for h, hname in ((1, "sup"), (-1, "inf")):
        pts = [i for i in ang if geo.hemi[i] == h]
        for a in pts:
            run = sorted(i for i in pts if ang[a] <= ang[i] < ang[a] + S3_WINDOW_DEG)
            cl, cnt = np.unique(geo.cluster[run], return_counts=True)
            if (len(run) >= S3_MIN_POINTS and len(cl) in (2, 3) and all(cnt < csize[cl])
                    and ecc[run].max() > S3_MIN_MAX_ECC_DEG):
                out[(hname, tuple(run))] = dict(hemifield=hname, points=list(run), n_clusters=int(len(cl)),
                                                clusters=[int(c) for c in cl])
    return [out[k] for k in sorted(out)], {int(k): round(v, 3) for k, v in sorted(ang.items())}


def s4_candidates(geo: Geometry, max_ecc: float, sizes=(3, 4, 5)):
    """S4 旁中心暗点候选点集（R5 规则 S4-1…S4-3）：偏心度 ≤ max_ecc 的点中，
    在相邻图（第 6 部分第 3 条约定）上连通、且非中心点都在同一半视野的 n 点集合。"""
    ecc = np.hypot(geo.xy[:, 0], geo.xy[:, 1])
    elig = [i for i in range(N_PTS) if ecc[i] <= max_ecc + 1e-9]
    res = {n: [] for n in sizes}
    for n in sizes:
        for comb in itertools.combinations(elig, n):
            hs = {int(geo.hemi[i]) for i in comb if i != geo.center}
            if len(hs) != 1:
                continue
            seen = {comb[0]}; stack = [comb[0]]; cs = set(comb)
            while stack:
                u = stack.pop()
                for v in cs:
                    if v not in seen and geo.neighbor[u, v]:
                        seen.add(v); stack.append(v)
            if len(seen) == n:
                res[n].append(list(comb))
    return res, elig


# ---------------------------------------------------------------------------
# 噪声模型（第 3.3 节；data/noise_model.json）
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    sd_grid: np.ndarray
    sd_vals: np.ndarray
    offset_sd_by_tertile: np.ndarray   # 去卷积后的偏移 SD（模拟用），按基线 MS 三层
    a: float
    lam: float
    spatial: bool
    kappa_offset: float = 1.0          # 第 3.5 节校准系数（默认 1，即未调）
    kappa_point: float = 1.0

    @classmethod
    def from_json(cls, nm: dict, kappa_offset=1.0, kappa_point=1.0):
        off = nm["offset"]
        sd_t = off["sd_deconvolved_by_tertile"] if off["stratified"] else [off["sd_deconvolved_pooled"]] * 3
        return cls(sd_grid=np.asarray(nm["point_sd"]["spline"]["grid_db"], float),
                   sd_vals=np.asarray(nm["point_sd"]["spline"]["sd_db"], float),
                   offset_sd_by_tertile=np.asarray(sd_t, float),
                   a=float(nm["spatial"]["a"]), lam=float(nm["spatial"]["lambda_deg"]),
                   spatial=bool(nm["spatial"]["used"]), kappa_offset=kappa_offset, kappa_point=kappa_point)

    def sd_point(self, f):
        """noise_model.json 的 eval 规则：np.interp(clip(f, 0, None), grid_db, sd_db)。"""
        return np.interp(np.clip(f, 0, None), self.sd_grid, self.sd_vals)

    def corr(self, geo: Geometry, independent=False):
        if independent or not self.spatial:
            return np.eye(N_PTS)
        same = geo.noise_block[:, None] == geo.noise_block[None]
        R = np.where(same, self.a * np.exp(-geo.dist / self.lam), 0.0)
        np.fill_diagonal(R, 1.0)
        return R


# ---------------------------------------------------------------------------
# 模板（修订 R1：逐点 Tobit 截距）
# ---------------------------------------------------------------------------
def _ols(t, Y):
    X = np.c_[np.ones(len(t)), t]
    return np.linalg.lstsq(X, Y, rcond=None)[0]           # (2, m)


def tobit_point(t, y, c, sigma):
    """单点 Tobit（0 dB 左删失，σ 固定）。y 中删失处为 0。返回 (a, b, status)。"""
    if not c.any():
        a, b = _ols(t, y[:, None])[:, 0]
        return float(a), float(b), "ols_no_censoring"
    if c.all():
        return -10.0, 0.0, "all_censored_floor"
    a0, b0 = _ols(t, y[:, None])[:, 0]
    u = ~c

    def nll(p):
        mu = p[0] + p[1] * t
        zu = (y[u] - mu[u]) / sigma
        zc = (0.0 - mu[c]) / sigma
        lc = log_ndtr(zc)
        val = 0.5 * np.sum(zu ** 2) - np.sum(lc)
        dmu = np.zeros_like(t)
        dmu[u] = -(y[u] - mu[u]) / sigma ** 2
        dmu[c] = np.exp(-0.5 * zc ** 2 - 0.5 * np.log(2 * np.pi) - lc) / sigma
        return val, np.array([dmu.sum(), (dmu * t).sum()])

    x0 = np.clip([a0, b0], [TOBIT_A_BOUNDS[0], TOBIT_B_BOUNDS[0]], [TOBIT_A_BOUNDS[1], TOBIT_B_BOUNDS[1]])
    r = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=[TOBIT_A_BOUNDS, TOBIT_B_BOUNDS],
                 options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
    return float(r.x[0]), float(r.x[1]), "mle" if r.success else "mle_not_converged"


def tobit_template(t, Y, C, nm: NoiseModel, tertile: int, clip=(-10.0, 35.0)):
    """一只模板眼的潜在基线（59,）。σ_p = sqrt(mean_v SD_point(f̂_OLS,v)² + SD_offset(tertile)²)（R5 规则 T-1），
    对该眼该点所有检查取同一个 σ，因此无删失时 MLE 严格等于 OLS（与 R1 文字一致）。
    SD_offset 用去卷积后的、未经第 3.5 节校准系数缩放的值（R5 规则 T-2）。"""
    t = np.asarray(t, float); Y = np.asarray(Y, float); C = np.asarray(C, bool)
    F = np.c_[np.ones(len(t)), t] @ _ols(t, Y)
    s_off = float(nm.offset_sd_by_tertile[tertile])
    sig = np.sqrt(np.mean(nm.sd_point(F) ** 2, axis=0) + s_off ** 2)
    base = np.empty(N_PTS); status = []
    for p in range(N_PTS):
        a, _, st = tobit_point(t, Y[:, p], C[:, p], sig[p])
        base[p] = a; status.append(st)
    return np.clip(base, *clip), status, sig


@dataclass
class Template:
    eye_id: str
    subject: str
    index: int                 # 在该划分模板眼排序（eye_id 字符串升序）中的位置
    tertile: int
    baseline: np.ndarray       # (59,) Tobit 潜在基线，t = 0
    t_real: np.ndarray         # (n,) 真实检查时间（年，第 1 次 = 0）
    iop_real: np.ndarray       # (n,)
    age0: float


# ---------------------------------------------------------------------------
# 时间表、情景、真值
# ---------------------------------------------------------------------------
def nested_schedule(tmpl: Template, interval_pool: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """K_MAX 次检查的嵌套时间表：模板前 min(n, K_MAX) 个真实时间点，不足部分从开发集相邻间隔池有放回补抽。
    k 次版本 = 前 k 个时间点（R5 规则 K-1）。返回 (t, iop)，补抽的检查 iop = NaN。"""
    n = min(len(tmpl.t_real), K_MAX)
    t = list(tmpl.t_real[:n]); iop = list(tmpl.iop_real[:n])
    extra = rng.choice(interval_pool, size=K_MAX - n, replace=True) if n < K_MAX else []
    for d in extra:
        t.append(t[-1] + float(d)); iop.append(np.nan)
    return np.asarray(t, float), np.asarray(iop, float)


def ols_slope(t, F):
    """F: (k, m) → (m,) 对 t 的 OLS 斜率。"""
    tc = t - t.mean()
    return (tc[:, None] * (F - F.mean(0))).sum(0) / (tc ** 2).sum()


@dataclass
class ScenarioDraw:
    """一条序列的情景随机参数（与 k 无关；k 相关的部分在 trajectory() 里由这些参数确定）。"""
    scenario: str
    affected: np.ndarray        # (59,) bool
    slope: float = np.nan       # 受影响点的斜率档（S2/S3/S4/S5）
    drop: float = np.nan        # S1 下降幅度
    u_step: float = np.nan      # S1：j_k = 2 + floor(u·(k−1))
    s7_a: float = np.nan        # S7 二次项系数
    pointset_id: int = -1       # S3/S7/S4 候选点集编号；S2/S6 为簇编号位掩码


def draw_scenario(sc: str, geo: Geometry, cfg_sc: dict, s3c: list, s4c: dict, t6: np.ndarray, rng) -> ScenarioDraw:
    aff = np.zeros(N_PTS, bool)
    d = ScenarioDraw(scenario=sc, affected=aff)
    p = cfg_sc[sc]
    if sc == "S0":
        pass
    elif sc == "S1":
        lo, hi = p["step_drop_db_uniform"]
        d.drop = float(rng.uniform(lo, hi)); d.u_step = float(rng.uniform()); aff[:] = True
    elif sc in ("S2", "S6"):
        n = int(rng.choice(p["n_clusters"] if sc == "S2" else cfg_sc["S2"]["n_clusters"]))
        cls = rng.choice(np.arange(1, geo.n_clusters + 1), size=n, replace=False)
        aff[np.isin(geo.cluster, cls)] = True
        d.pointset_id = int(sum(1 << int(c) for c in cls))
        if sc == "S2":
            d.slope = float(rng.choice(p["slopes"]))
    elif sc in ("S3", "S7"):
        ncl = int(rng.choice(cfg_sc["S3"]["n_clusters_spanned"]))
        idx = [i for i, c in enumerate(s3c) if c["n_clusters"] == ncl]
        j = int(idx[int(rng.integers(len(idx)))])
        aff[s3c[j]["points"]] = True; d.pointset_id = j
        if sc == "S3":
            d.slope = float(rng.choice(p["slopes"]))
        else:
            t4 = t6[:4]
            d.s7_a = float(-p["mean_slope_at_k4"] / ols_slope(t4, (t4 ** 2)[:, None])[0])
    elif sc == "S4":
        n = int(rng.choice(p["n_points"]))
        j = int(rng.integers(len(s4c[n])))
        aff[s4c[n][j]] = True; d.pointset_id = n * 1000 + j
        d.slope = float(rng.choice(p["slopes"]))
    elif sc == "S5":
        aff[:] = True; d.slope = float(rng.choice(p["slopes"]))
    else:
        raise ValueError(sc)
    return d


def trajectory(d: ScenarioDraw, baseline: np.ndarray, t: np.ndarray, cfg_sc: dict) -> np.ndarray:
    """潜在真实轨迹 f(t)，(k, 59)。t 为该 k 次版本的时间点（t[0] = 0）。"""
    k = len(t); age = cfg_sc["S0"]["slope_all"]
    T = np.repeat(t[:, None], N_PTS, 1)
    g = age * T                                         # 未受影响的点：同 S0（R5 规则 U-1；S1 除外）
    sc = d.scenario; a = d.affected
    if sc == "S1":
        j = 2 + int(np.floor(d.u_step * (k - 1)))      # j ∈ {2..k} 等概率；第 j 次检查起下降
        g = np.zeros_like(T); g[j - 1:, :] = -d.drop
    elif sc in ("S2", "S3", "S4"):
        g[:, a] = d.slope * T[:, a]
    elif sc == "S5":
        g = d.slope * T
    elif sc == "S6":
        p = cfg_sc["S6"]; tm = 0.5 * (t[0] + t[-1])
        pw = np.where(T <= tm, p["slope_early"] * T, p["slope_early"] * tm + p["slope_late"] * (T - tm))
        g[:, a] = pw[:, a]
    elif sc == "S7":
        g[:, a] = -d.s7_a * T[:, a] ** 2
    return baseline[None, :] + g


def truths(f: np.ndarray, t: np.ndarray, geo: Geometry) -> dict:
    """第 1.3 节：观察尺度 = max(f, 0) 的 OLS；潜在尺度 = f 的 OLS。簇级/全局 = 逐点真值平均。"""
    out = {}
    for name, F in (("obs", np.maximum(f, 0.0)), ("lat", f)):
        bp = ols_slope(t, F)
        out[f"truth_point_{name}"] = bp
        out[f"truth_cluster_{name}"] = np.array([bp[geo.cluster == c].mean() for c in range(1, geo.n_clusters + 1)])
        out[f"truth_global_{name}"] = float(bp.mean())
    return out


def simulate_series(tmpl: Template, sc: str, rep: int, seed: int, geo: Geometry, nm: NoiseModel,
                    L: np.ndarray, cfg_sim: dict, s3c, s4c, interval_pool, no_offset=False):
    """生成一条序列的全部 k 版本（嵌套：同一时间表、同一组标准正态抽样，k 版本取前 k 次）。"""
    ss = np.random.SeedSequence([int(seed), int(tmpl.index), SCENARIOS.index(sc), int(rep)])
    r_sched, r_scen, r_noise = (np.random.default_rng(s) for s in ss.spawn(3))
    t6, iop6 = nested_schedule(tmpl, interval_pool, r_sched)
    cfg_sc = cfg_sim["scenarios"]
    d = draw_scenario(sc, geo, cfg_sc, s3c, s4c, t6, r_scen)
    z_off = r_noise.standard_normal(K_MAX)                 # 检查级整体偏移
    z_pt = r_noise.standard_normal((K_MAX, N_PTS)) @ L.T   # 半视野块内指数相关的逐点标准化噪声
    sd_off = 0.0 if no_offset else nm.kappa_offset * nm.offset_sd_by_tertile[tmpl.tertile]
    res = {}
    for k in cfg_sim["k_versions"]:
        t = t6[:k]
        f = trajectory(d, tmpl.baseline, t, cfg_sc)
        ystar = f + sd_off * z_off[:k, None] + nm.kappa_point * nm.sd_point(f) * z_pt[:k]
        cens = ystar < 0.0
        y = np.where(cens, 0.0, ystar)
        res[k] = dict(t=t, y=y, censored=cens, f=f, iop=iop6[:k], age=tmpl.age0 + t, **truths(f, t, geo))
    return d, res


# ---------------------------------------------------------------------------
# 确定性 npz 写出（固定 zip 时间戳与成员顺序，同样输入 → 同样字节 → 同样 SHA-256）
# ---------------------------------------------------------------------------
def write_npz_deterministic(path: str, arrays: dict):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue(), compresslevel=6)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(templates, scenarios, n_rep, seed, geo, nm, cfg_sim, s3c, s4c, interval_pool,
             independent_noise=False, no_offset=False):
    """按 模板眼 × 情景 × 重复 生成整个数据集，返回可写入 npz 的数组字典。"""
    L = np.linalg.cholesky(nm.corr(geo, independent=independent_noise) + 1e-10 * np.eye(N_PTS))
    ks = cfg_sim["k_versions"]
    meta = dict(scenario=[], template_eye=[], template_subject=[], rep=[], is_progressing=[],
                affected=[], slope_draw=[], s1_drop=[], s1_u=[], s7_a=[], pointset_id=[])
    per_k = {k: {} for k in ks}
    for tmpl in templates:
        for sc in scenarios:
            for rep in range(n_rep):
                d, res = simulate_series(tmpl, sc, rep, seed, geo, nm, L, cfg_sim, s3c, s4c, interval_pool, no_offset)
                meta["scenario"].append(sc); meta["template_eye"].append(tmpl.eye_id)
                meta["template_subject"].append(tmpl.subject); meta["rep"].append(rep)
                meta["is_progressing"].append(bool(cfg_sim["scenarios"][sc]["positive"]))
                meta["affected"].append(d.affected); meta["slope_draw"].append(d.slope)
                meta["s1_drop"].append(d.drop); meta["s1_u"].append(d.u_step); meta["s7_a"].append(d.s7_a)
                meta["pointset_id"].append(d.pointset_id)
                for k in ks:
                    for name, v in res[k].items():
                        per_k[k].setdefault(name, []).append(v)
    out = {"meta_" + k: np.asarray(v) for k, v in meta.items()}
    out["meta_scenario"] = out["meta_scenario"].astype("U2")
    out["meta_template_eye"] = out["meta_template_eye"].astype("U16")
    out["meta_template_subject"] = out["meta_template_subject"].astype("U16")
    for k in ks:
        for name, v in per_k[k].items():
            a = np.asarray(v)
            if name == "y" or name == "f":
                a = a.astype(np.float64)
            out[f"k{k}_{name}"] = a
    out["cluster_of_point"] = geo.cluster.astype(np.int64)
    return out
