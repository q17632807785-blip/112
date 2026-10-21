"""统一接口：Series / Estimate（预注册第 2.3 节）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

N_PTS = 59
N_CLUSTERS = 10


@dataclass
class Series:
    eye_id: str
    subject: str
    t: np.ndarray                 # (n,)
    y: np.ndarray                 # (n, 59)；删失记 0
    censored: np.ndarray          # (n, 59) bool
    age: Optional[np.ndarray] = None
    iop: Optional[np.ndarray] = None


@dataclass
class Estimate:
    denoised: np.ndarray          # (n, 59)
    slope_point: np.ndarray       # (59,)
    se_point: np.ndarray          # (59,)
    slope_cluster: np.ndarray     # (10,)
    se_cluster: np.ndarray        # (10,)
    slope_global: float
    se_global: float
    failed: bool = False
    fallback: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class GeometryInfo:
    cluster: np.ndarray           # (59,) 1..10；中心 0
    neighbor: np.ndarray          # (59, 59) bool
    center: int
    xy: np.ndarray


def ols_1d(t: np.ndarray, y: np.ndarray):
    """单序列 OLS 斜率与 SE（残差 SE）。n < 2 返回 nan。"""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 2 or np.allclose(t, t[0]):
        return np.nan, np.nan, np.nan, np.nan
    tc = t - t.mean()
    sxx = float(tc @ tc)
    b = float(tc @ (y - y.mean()) / sxx)
    a = float(y.mean() - b * t.mean())
    resid = y - (a + b * t)
    dof = max(n - 2, 1)
    sigma = float(np.sqrt((resid @ resid) / dof))
    se = sigma / np.sqrt(sxx)
    return a, b, se, sigma


def ols_slopes_matrix(t: np.ndarray, Y: np.ndarray):
    """Y: (n, p) → slope (p,), se (p,), intercept (p,)。"""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    n, p = Y.shape
    if n < 2 or np.allclose(t, t[0]):
        nan = np.full(p, np.nan)
        return nan.copy(), nan.copy(), nan.copy()
    tc = t - t.mean()
    sxx = float(tc @ tc)
    yc = Y - Y.mean(0, keepdims=True)
    b = (tc[:, None] * yc).sum(0) / sxx
    a = Y.mean(0) - b * t.mean()
    resid = Y - (a + b * t[:, None])
    dof = max(n - 2, 1)
    sigma = np.sqrt((resid ** 2).sum(0) / dof)
    se = sigma / np.sqrt(sxx)
    return a, b, se


def cluster_means(Y: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """Y (n, 59) → (n, 10) 簇均值（不含中心点）。"""
    out = np.empty((Y.shape[0], N_CLUSTERS), float)
    for c in range(1, N_CLUSTERS + 1):
        m = cluster == c
        out[:, c - 1] = Y[:, m].mean(1)
    return out


def cluster_slopes_from_denoised(t: np.ndarray, denoised: np.ndarray, cluster: np.ndarray):
    cm = cluster_means(denoised, cluster)
    a, b, se = ols_slopes_matrix(t, cm)
    return b, se


def global_slope_from_denoised(t: np.ndarray, denoised: np.ndarray):
    ms = denoised.mean(1)
    _, b, se, _ = ols_1d(t, ms)
    return float(b), float(se)


def fill_estimate_from_denoised(t, denoised, cluster) -> Estimate:
    _, sp, sep = ols_slopes_matrix(t, denoised)
    sc, sec = cluster_slopes_from_denoised(t, denoised, cluster)
    sg, seg = global_slope_from_denoised(t, denoised)
    return Estimate(
        denoised=np.asarray(denoised, float),
        slope_point=sp, se_point=sep,
        slope_cluster=sc, se_cluster=sec,
        slope_global=sg, se_global=seg,
    )


class Method:
    name: str = "base"
    needs_fit: bool = False

    def fit(self, train_series: list[Series]) -> None:
        return None

    def estimate(self, series: Series) -> Estimate:
        raise NotImplementedError
