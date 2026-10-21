"""B0 / B1 / B2 / B5（预注册第 2.4 节）。"""
from __future__ import annotations

import numpy as np

from .base import Estimate, GeometryInfo, Method, Series, fill_estimate_from_denoised, ols_1d, ols_slopes_matrix


class B0(Method):
    """末次值不变（仅用于下一次预测描述终点）。"""
    name = "B0"

    def __init__(self, geo: GeometryInfo):
        self.geo = geo

    def estimate(self, series: Series) -> Estimate:
        den = np.repeat(series.y[-1][None, :], len(series.t), axis=0)
        est = fill_estimate_from_denoised(series.t, den, self.geo.cluster)
        # 斜率强制为 0
        est.slope_point[:] = 0.0
        est.se_point[:] = np.nan
        est.slope_cluster[:] = 0.0
        est.se_cluster[:] = np.nan
        est.slope_global = 0.0
        est.se_global = float("nan")
        return est


class B1(Method):
    """逐点线性回归（PLR）；≥3 次检查。"""
    name = "B1"

    def __init__(self, geo: GeometryInfo, min_visits: int = 3):
        self.geo = geo
        self.min_visits = min_visits

    def estimate(self, series: Series) -> Estimate:
        if len(series.t) < self.min_visits:
            est = fill_estimate_from_denoised(series.t, series.y, self.geo.cluster)
            est.failed = True
            est.fallback = "too_few_visits"
            return est
        return fill_estimate_from_denoised(series.t, series.y, self.geo.cluster)


class B2(Method):
    """MS 线性回归（全局）。逐点/簇级仍由去噪值（=原值）的簇均值 OLS 得到。"""
    name = "B2"

    def __init__(self, geo: GeometryInfo):
        self.geo = geo

    def estimate(self, series: Series) -> Estimate:
        est = fill_estimate_from_denoised(series.t, series.y, self.geo.cluster)
        _, b, se, _ = ols_1d(series.t, series.y.mean(1))
        est.slope_global = float(b)
        est.se_global = float(se)
        return est


class B5(Method):
    """空间平滑后再 PLR：自身 0.5 + 同半视野 ≤6° 邻点均分 0.5。"""
    name = "B5"

    def __init__(self, geo: GeometryInfo, self_weight: float = 0.5, neighbor_weight_total: float = 0.5):
        self.geo = geo
        self.self_w = self_weight
        self.nb_w = neighbor_weight_total
        nb = geo.neighbor.astype(bool)
        # 预计算权重矩阵 W (59,59)：den = y @ W.T 等价于 row-wise 加权
        W = np.zeros((59, 59), float)
        for i in range(59):
            nbrs = np.flatnonzero(nb[i])
            W[i, i] = self.self_w
            if len(nbrs) == 0:
                W[i, i] = 1.0
            else:
                W[i, nbrs] = self.nb_w / len(nbrs)
        self.W = W

    def smooth(self, y: np.ndarray) -> np.ndarray:
        # y: (n, 59) → (n, 59)
        return y @ self.W.T

    def estimate(self, series: Series) -> Estimate:
        den = self.smooth(series.y)
        return fill_estimate_from_denoised(series.t, den, self.geo.cluster)
