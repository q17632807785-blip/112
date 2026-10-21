"""PoPLR（预注册第 2.4 节 D1）。"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Estimate, GeometryInfo, Method, Series, fill_estimate_from_denoised, ols_slopes_matrix
from .simple import B1


class PoPLR(Method):
    """逐点 OLS 单侧 p；S = -sum ln p（只算负斜率）；置换 1000 次。"""
    name = "PoPLR"

    def __init__(self, geo: GeometryInfo, n_perm: int = 1000, seed: int = 20261019):
        self.geo = geo
        self.n_perm = n_perm
        self.seed = seed
        self._b1 = B1(geo)

    def _S_and_p(self, t, y):
        _, b, se = ols_slopes_matrix(t, y)
        # 单侧：H1 slope < 0；t = b/se
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, b / se, 0.0)
        # df = n-2
        df = max(len(t) - 2, 1)
        p = stats.t.cdf(z, df)  # P(T <= z) for negative alternative
        neg = b < 0
        if not np.any(neg):
            S = 0.0
        else:
            p_neg = np.clip(p[neg], 1e-300, 1.0)
            S = float(-np.sum(np.log(p_neg)))
        return S, b, se, p

    def estimate(self, series: Series) -> Estimate:
        if len(series.t) < 3:
            return self._b1.estimate(series)
        t = series.t
        y = series.y
        S, b, se, p = self._S_and_p(t, y)
        rng = np.random.default_rng(self.seed + hash(series.eye_id) % 10_000_000)
        null = np.empty(self.n_perm)
        n = len(t)
        for i in range(self.n_perm):
            # 打乱访视顺序（时间点不变，调换数值）
            perm = rng.permutation(n)
            Sp, _, _, _ = self._S_and_p(t, y[perm])
            null[i] = Sp
        p_perm = float(np.mean(null >= S))
        est = fill_estimate_from_denoised(t, y, self.geo.cluster)
        est.slope_point = b
        est.se_point = se
        est.extras["S"] = S
        est.extras["p_perm"] = p_perm
        est.extras["point_p"] = p
        return est
