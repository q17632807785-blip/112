"""EB 与 EB-GK（预注册第 2.2 / 2.4 节）。"""
from __future__ import annotations

import numpy as np

from .base import Estimate, GeometryInfo, Method, Series, fill_estimate_from_denoised, ols_slopes_matrix
from .simple import B1


class EB(Method):
    """向该眼同簇均值收缩；τ² 按敏感度三层从训练眼估计。"""
    name = "EB"
    needs_fit = True

    def __init__(self, geo: GeometryInfo, n_strata: int = 3):
        self.geo = geo
        self.n_strata = n_strata
        self.tau2 = np.full(n_strata, np.nan)  # 按层
        self.strata_edges = None
        self._b1 = B1(geo)

    def _point_baseline(self, series: Series) -> np.ndarray:
        # 用第一次检查的观测作敏感度分层
        return series.y[0].copy()

    def _stratum(self, baseline: np.ndarray) -> np.ndarray:
        # baseline (59,) → strata index 0..n_strata-1
        edges = self.strata_edges
        s = np.digitize(baseline, edges)  # 1..n_strata
        return np.clip(s - 1, 0, self.n_strata - 1)

    def fit(self, train_series: list[Series]) -> None:
        # 收集训练眼逐点 OLS 斜率与 SE，以及基线敏感度
        bs, ses, bases = [], [], []
        for s in train_series:
            if len(s.t) < 3:
                continue
            _, b, se = ols_slopes_matrix(s.t, s.y)
            bs.append(b)
            ses.append(se)
            bases.append(self._point_baseline(s))
        if not bs:
            self.tau2[:] = 0.0
            self.strata_edges = np.array([-np.inf, np.inf])
            return
        B = np.stack(bs)       # (E, 59)
        S = np.stack(ses)
        Base = np.stack(bases)
        # 用全部训练点的基线三分位作切点（全局切点，再按层估 τ²）
        flat = Base.ravel()
        qs = np.quantile(flat, np.linspace(0, 1, self.n_strata + 1)[1:-1])
        self.strata_edges = qs
        strata = np.digitize(Base, qs)  # 1..n_strata-? → clip
        strata = np.clip(strata, 0, self.n_strata - 1) if len(qs) == 0 else np.clip(strata, 0, self.n_strata - 1)
        # digitize with n-1 edges → 0..n
        if qs.size:
            strata = np.digitize(Base, qs)  # 0..n_strata
            strata = np.clip(strata, 0, self.n_strata - 1)
        else:
            strata = np.zeros_like(Base, dtype=int)
        for k in range(self.n_strata):
            m = strata == k
            if m.sum() < 5:
                self.tau2[k] = 0.0
                continue
            b = B[m]
            s2 = (S[m] ** 2)
            self.tau2[k] = max(0.0, float(np.var(b, ddof=1) - np.mean(s2)))

    def estimate(self, series: Series) -> Estimate:
        if len(series.t) < 3:
            return self._b1.estimate(series)
        _, b, se = ols_slopes_matrix(series.t, series.y)
        base = self._point_baseline(series)
        if self.strata_edges is None or (isinstance(self.strata_edges, np.ndarray) and self.strata_edges.size == 0):
            strata = np.zeros(59, dtype=int)
        else:
            strata = np.digitize(base, self.strata_edges)
            strata = np.clip(strata, 0, self.n_strata - 1)
        tau2 = self.tau2[strata]
        s2 = se ** 2
        denom = tau2 + np.nan_to_num(s2, nan=np.inf)
        w = np.divide(tau2, denom, out=np.zeros_like(tau2, dtype=float), where=denom > 0)
        # 同簇平均斜率（该眼）
        b_eb = b.copy()
        for c in range(1, 11):
            m = self.geo.cluster == c
            if not m.any():
                continue
            b_bar = float(np.nanmean(b[m]))
            b_eb[m] = w[m] * b[m] + (1 - w[m]) * b_bar
        # 中心点：无簇，保留自身（或与全体均值收缩——按预注册只写同簇，中心点无簇则不收缩）
        # 去噪值：用收缩斜率重建 a + b_eb * t（a 仍用逐点 OLS 截距）
        a, _, _ = ols_slopes_matrix(series.t, series.y)
        den = a[None, :] + b_eb[None, :] * series.t[:, None]
        est = fill_estimate_from_denoised(series.t, den, self.geo.cluster)
        # 直接用收缩后的逐点斜率
        est.slope_point = b_eb
        est.se_point = se * np.sqrt(np.maximum(w, 0))  # 粗略：收缩后 SE
        # 簇级：逐点收缩斜率在簇内平均（第 2.3 节：EB 直接给出）
        sc = np.array([float(np.nanmean(b_eb[self.geo.cluster == c])) for c in range(1, 11)])
        sec = np.array([float(np.nanmean(est.se_point[self.geo.cluster == c])) for c in range(1, 11)])
        est.slope_cluster = sc
        est.se_cluster = sec
        est.slope_global = float(np.nanmean(b_eb))
        est.se_global = float(np.nanmean(est.se_point))
        est.extras["w_mean"] = float(np.nanmean(w))
        return est


class EBGK(Method):
    """Gwak–Kang 式：向训练人群该点总体均值收缩；只作描述性对照。"""
    name = "EB_GK"
    needs_fit = True

    def __init__(self, geo: GeometryInfo):
        self.geo = geo
        self.mu = np.zeros(59)
        self.tau2 = np.zeros(59)
        self._b1 = B1(geo)

    def fit(self, train_series: list[Series]) -> None:
        bs, ses = [], []
        for s in train_series:
            if len(s.t) < 3:
                continue
            _, b, se = ols_slopes_matrix(s.t, s.y)
            bs.append(b)
            ses.append(se)
        if not bs:
            return
        B = np.stack(bs)
        S = np.stack(ses)
        self.mu = B.mean(0)
        for p in range(59):
            self.tau2[p] = max(0.0, float(np.var(B[:, p], ddof=1) - np.mean(S[:, p] ** 2)))

    def estimate(self, series: Series) -> Estimate:
        if len(series.t) < 3:
            return self._b1.estimate(series)
        a, b, se = ols_slopes_matrix(series.t, series.y)
        s2 = np.nan_to_num(se ** 2, nan=np.inf, posinf=np.inf)
        denom = self.tau2 + s2
        w = np.divide(self.tau2, denom, out=np.zeros_like(self.tau2), where=denom > 0)
        b_gk = self.mu + w * (b - self.mu)
        den = a[None, :] + b_gk[None, :] * series.t[:, None]
        est = fill_estimate_from_denoised(series.t, den, self.geo.cluster)
        est.slope_point = b_gk
        est.se_point = se * np.sqrt(np.maximum(w, 0))
        est.slope_cluster = np.array([float(np.nanmean(b_gk[self.geo.cluster == c])) for c in range(1, 11)])
        est.se_cluster = np.array([float(np.nanmean(est.se_point[self.geo.cluster == c])) for c in range(1, 11)])
        est.slope_global = float(np.nanmean(b_gk))
        est.se_global = float(np.nanmean(est.se_point))
        return est
