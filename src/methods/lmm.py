"""B3 / B4（预注册第 2.4 节）。

B4 空间项（修订 R8）：正式计时选定 spaMM Matérn（同半视野内）；
实现为「训练眼上估层次方差分量 + Matérn 空间相关参数 → 新眼固定参数后
先做同半视野 Matérn 空间 GLS 平滑，再做簇级两重 BLUP」。
spatial_impl = "spaMM_matern_within_hemifield"。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import kv, gamma

from .base import Estimate, GeometryInfo, Method, Series, fill_estimate_from_denoised, ols_1d, ols_slopes_matrix
from .simple import B1


@dataclass
class _VarCompsMS:
    """B3：MS 水平方差分量。"""
    sigma2_e: float
    var_subj_int: float
    var_subj_slope: float
    var_eye_int: float
    var_eye_slope: float
    cov_subj: float = 0.0
    cov_eye: float = 0.0
    beta0: float = 0.0
    beta1: float = 0.0


class B3(Method):
    """MS 线性混合模型：人、眼两级随机截距和斜率；新眼 BLUP。"""
    name = "B3"
    needs_fit = True

    def __init__(self, geo: GeometryInfo):
        self.geo = geo
        self.vc: _VarCompsMS | None = None
        self._b1 = B1(geo)

    def fit(self, train_series: list[Series]) -> None:
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
        except ImportError as e:
            raise RuntimeError("B3 需要 statsmodels") from e

        rows = []
        for s in train_series:
            if len(s.t) < 2:
                continue
            ms = s.y.mean(1)
            for i in range(len(s.t)):
                rows.append(dict(
                    ms=float(ms[i]), t=float(s.t[i]),
                    subject=str(s.subject), eye=str(s.eye_id),
                ))
        if len(rows) < 50:
            self.vc = _VarCompsMS(1.0, 0.0, 0.0, 0.0, 0.0)
            return
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                md = smf.mixedlm("ms ~ t", df, groups=df["eye"], re_formula="~t")
                mdf = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
                fe = mdf.fe_params
                cov = np.asarray(mdf.cov_re)
                if cov.ndim == 0:
                    vi, vs, cv = float(cov), 0.0, 0.0
                elif cov.size == 1:
                    vi, vs, cv = float(cov.ravel()[0]), 0.0, 0.0
                else:
                    vi, vs, cv = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
                self.vc = _VarCompsMS(
                    sigma2_e=float(mdf.scale),
                    var_subj_int=0.0, var_subj_slope=0.0,
                    var_eye_int=max(vi, 0.0), var_eye_slope=max(vs, 0.0), cov_eye=cv,
                    beta0=float(fe.get("Intercept", fe.iloc[0])),
                    beta1=float(fe.get("t", fe.iloc[1] if len(fe) > 1 else 0.0)),
                )
            except Exception:
                slopes = []
                for s in train_series:
                    if len(s.t) >= 3:
                        _, b, _, _ = ols_1d(s.t, s.y.mean(1))
                        if np.isfinite(b):
                            slopes.append(b)
                self.vc = _VarCompsMS(
                    sigma2_e=1.0,
                    var_subj_int=0.0, var_subj_slope=0.0,
                    var_eye_int=1.0, var_eye_slope=float(np.var(slopes)) if slopes else 0.5,
                    beta0=20.0, beta1=float(np.mean(slopes)) if slopes else -0.1,
                )

    def estimate(self, series: Series) -> Estimate:
        if self.vc is None or len(series.t) < 2:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est
        vc = self.vc
        t = series.t.astype(float)
        y = series.y.mean(1)
        n = len(t)
        X = np.column_stack([np.ones(n), t])
        G = np.array([[vc.var_eye_int, vc.cov_eye], [vc.cov_eye, vc.var_eye_slope]], float)
        eig = np.linalg.eigvalsh(G)
        if eig.min() < 1e-8:
            G = G + (1e-8 - min(eig.min(), 0)) * np.eye(2)
        R = vc.sigma2_e * np.eye(n)
        try:
            V = X @ G @ X.T + R
            resid = y - (vc.beta0 + vc.beta1 * t)
            u = G @ X.T @ np.linalg.solve(V, resid)
            slope = vc.beta1 + float(u[1])
            cov_u = G - G @ X.T @ np.linalg.solve(V, X @ G)
            se = float(np.sqrt(max(cov_u[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est
        est = fill_estimate_from_denoised(series.t, series.y, self.geo.cluster)
        est.slope_global = slope
        est.se_global = se if se > 0 else float("nan")
        est.extras["blup_u"] = u.tolist()
        return est


def matern_corr(d: np.ndarray, nu: float, rho: float) -> np.ndarray:
    """Matérn correlation；d 为距离（度），rho 为 range 参数（与 spaMM 常用参数化一致的尺度）。"""
    d = np.asarray(d, float)
    out = np.ones_like(d, float)
    mask = d > 1e-12
    if not np.any(mask):
        return out
    # spaMM / Diggle: (2^{1-ν}/Γ(ν)) (√(2ν) d/ρ)^ν K_ν(√(2ν) d/ρ)
    x = np.sqrt(2.0 * nu) * d[mask] / max(rho, 1e-6)
    x = np.clip(x, 1e-12, 1e6)
    out[mask] = (2.0 ** (1.0 - nu) / gamma(nu)) * (x ** nu) * kv(nu, x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(out, 0.0, 1.0, out=out)
    return out


@dataclass
class _VarCompsB4:
    sigma2: float
    var_eye_int: float
    var_eye_slope: float
    var_cl_int: float
    var_cl_slope: float
    var_visit: float
    beta0: float
    beta1: float
    spatial: str = "spaMM_matern_within_hemifield"
    matern_nu: float = 0.5
    matern_rho: float = 8.656  # so corr(6°)≈0.5 for exponential (ν=0.5)
    spat_var: float = 1.0      # spatial partial sill
    nugget: float = 1.0        # residual nugget


class B4(Method):
    """空间两重混合模型（R8：spaMM Matérn 同半视野）。

    流程：训练眼上估层次方差分量 + Matérn 参数 → 新眼固定参数后：
    每次检查做同半视野 Matérn 空间 GLS 平滑，再簇级两重 BLUP 得斜率。
    """
    name = "B4"
    needs_fit = True

    def __init__(self, geo: GeometryInfo, spatial_impl: str = "spaMM_matern_within_hemifield"):
        self.geo = geo
        self.vc: _VarCompsB4 | None = None
        self._b1 = B1(geo)
        self.spatial_impl = spatial_impl
        # precompute distance and hemifield blocks
        xy = geo.xy
        self._dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        hemi = np.sign(xy[:, 1])
        hemi[geo.center] = 1  # center with superior for Matérn block (noise_block uses y>=0)
        # match sim_core noise_block: y>=0 → 0, else 1
        self._block = np.where(xy[:, 1] >= 0, 0, 1)
        self._Sigma = None  # filled in fit

    def _build_sigma(self, vc: _VarCompsB4) -> np.ndarray:
        """Σ = spat_var * Matérn_corr within hemifield (0 across); + nugget on diagonal."""
        C = matern_corr(self._dist, vc.matern_nu, vc.matern_rho)
        same = self._block[:, None] == self._block[None, :]
        C = np.where(same, C, 0.0)
        np.fill_diagonal(C, 1.0)
        Sigma = vc.spat_var * C + vc.nugget * np.eye(59)
        return Sigma

    def _spatial_smooth_visit(self, y: np.ndarray) -> np.ndarray:
        """单次检查 59 点：ŷ = μ + Σ_s (Σ_s + nugget I)^{-1} (y-μ)，跨半视野块对角。"""
        vc = self.vc
        assert vc is not None and self._Sigma is not None
        # use precomputed full Sigma = spat_var*C + nugget*I
        try:
            # kriging toward visit mean
            mu = float(np.nanmean(y))
            r = y - mu
            # solve Sigma x = r
            x = np.linalg.solve(self._Sigma, r)
            # posterior mean of spatial signal: spat_var*C @ x, but Σ = spat_var*C + nugget I
            # E[s|y] = spat_var*C Σ^{-1} (y-μ) = (Σ - nugget I) Σ^{-1} (y-μ)
            s = r - vc.nugget * x
            return mu + s
        except np.linalg.LinAlgError:
            return y.copy()

    def fit(self, train_series: list[Series]) -> None:
        eye_slopes, cl_slopes, visit_resid2, intercepts = [], [], [], []
        # residuals for Matérn MoM after per-point OLS
        pair_sum = np.zeros((59, 59))
        pair_cnt = np.zeros((59, 59))
        resid_var = []

        for s in train_series:
            if len(s.t) < 3:
                continue
            cm = np.column_stack([
                s.y[:, self.geo.cluster == c].mean(1) for c in range(1, 11)
            ])
            a, b, se = ols_slopes_matrix(s.t, cm)
            eye_slopes.append(float(np.nanmean(b)))
            cl_slopes.extend(b.tolist())
            intercepts.append(float(np.nanmean(a)))
            for j in range(10):
                if not np.isfinite(b[j]):
                    continue
                pred = a[j] + b[j] * s.t
                visit_resid2.append(float(np.var(cm[:, j] - pred)))
            # point OLS residuals for spatial MoM
            ap, bp, _ = ols_slopes_matrix(s.t, s.y)
            resid = s.y - (ap[None, :] + bp[None, :] * s.t[:, None])
            # remove visit mean (exam-level offset)
            resid = resid - resid.mean(1, keepdims=True)
            for v in range(resid.shape[0]):
                r = resid[v]
                resid_var.append(float(np.var(r)))
                # accumulate products
                pair_sum += np.outer(r, r)
                pair_cnt += 1.0

        if not eye_slopes:
            self.vc = _VarCompsB4(1.0, 0.5, 0.2, 0.3, 0.1, 0.2, 20.0, -0.1)
            self._Sigma = self._build_sigma(self.vc)
            return

        var_eye_slope = float(np.var(eye_slopes, ddof=1)) if len(eye_slopes) > 1 else 0.2
        var_cl_slope = max(0.0, float(np.var(cl_slopes, ddof=1)) - var_eye_slope)

        # Matérn MoM: target corr at 6° ≈ neighbor_resid_corr from config/locations (0.5067)
        # ν=0.5 (exponential); ρ so that corr(6°)=0.5067 → ρ = 6 / -log(0.5067)
        target_corr = 0.5067
        rho = float(6.0 / max(-np.log(max(target_corr, 1e-3)), 1e-6))
        nu = 0.5
        total_var = float(np.mean(resid_var)) if resid_var else 2.0
        # split sill / nugget using neighbor mean empirical corr if available
        emp_corr = None
        if pair_cnt.sum() > 0:
            cov = pair_sum / np.maximum(pair_cnt, 1.0)
            # average corr among neighbors
            nb = self.geo.neighbor.astype(bool)
            if nb.any():
                # standardize
                sd = np.sqrt(np.diag(cov).clip(min=1e-8))
                corr = cov / (sd[:, None] * sd[None, :])
                emp_corr = float(np.nanmean(corr[nb]))
        if emp_corr is not None and 0 < emp_corr < 1:
            # spat_var / (spat_var + nugget) * matern(6) ≈ emp_corr
            m6 = float(matern_corr(np.array([6.0]), nu, rho)[0])
            frac = min(max(emp_corr / max(m6, 1e-3), 0.05), 0.95)
        else:
            frac = 0.5
        spat_var = total_var * frac
        nugget = total_var * (1.0 - frac)

        self.vc = _VarCompsB4(
            sigma2=float(np.mean(visit_resid2)) if visit_resid2 else 1.0,
            var_eye_int=float(np.var(intercepts, ddof=1)) if len(intercepts) > 1 else 1.0,
            var_eye_slope=var_eye_slope,
            var_cl_int=1.0,
            var_cl_slope=var_cl_slope,
            var_visit=float(np.mean(visit_resid2)) * 0.5 if visit_resid2 else 0.5,
            beta0=float(np.mean(intercepts)),
            beta1=float(np.mean(eye_slopes)),
            spatial=self.spatial_impl,
            matern_nu=nu,
            matern_rho=rho,
            spat_var=spat_var,
            nugget=max(nugget, 1e-4),
        )
        self._Sigma = self._build_sigma(self.vc)

    def estimate(self, series: Series) -> Estimate:
        if self.vc is None or self._Sigma is None or len(series.t) < 2:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est
        vc = self.vc
        t = series.t.astype(float)
        n = len(t)
        # spatial smooth each visit
        y_s = np.stack([self._spatial_smooth_visit(series.y[i]) for i in range(n)], axis=0)
        cm = np.column_stack([
            y_s[:, self.geo.cluster == c].mean(1) for c in range(1, 11)
        ])
        ms = cm.mean(1)
        X = np.column_stack([np.ones(n), t])
        try:
            G_e = np.diag([vc.var_eye_int, vc.var_eye_slope])
            R = (vc.sigma2 + vc.var_visit) * np.eye(n)
            V = X @ G_e @ X.T + R
            resid_ms = ms - (vc.beta0 + vc.beta1 * t)
            u = G_e @ X.T @ np.linalg.solve(V, resid_ms)
            slopes = np.empty(10)
            ses = np.empty(10)
            G_c = np.diag([vc.var_cl_int, vc.var_cl_slope])
            for c in range(10):
                resid_c = cm[:, c] - (vc.beta0 + vc.beta1 * t + u[0] + u[1] * t)
                Vc = X @ G_c @ X.T + vc.sigma2 * np.eye(n)
                w = G_c @ X.T @ np.linalg.solve(Vc, resid_c)
                slopes[c] = vc.beta1 + float(u[1]) + float(w[1])
                cov_w = G_c - G_c @ X.T @ np.linalg.solve(Vc, X @ G_c)
                cov_u = G_e - G_e @ X.T @ np.linalg.solve(V, X @ G_e)
                ses[c] = float(np.sqrt(max(cov_u[1, 1] + cov_w[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            est = self._b1.estimate(series)
            est.failed = True
            est.fallback = "B1"
            return est
        den = y_s.copy()
        for c in range(1, 11):
            m = self.geo.cluster == c
            a_ols, _, _ = ols_slopes_matrix(t, cm[:, c - 1:c])
            traj = float(a_ols[0]) + slopes[c - 1] * t
            den[:, m] = traj[:, None]
        est = fill_estimate_from_denoised(series.t, den, self.geo.cluster)
        est.slope_cluster = slopes
        est.se_cluster = ses
        est.slope_global = float(np.mean(slopes))
        est.se_global = float(np.mean(ses))
        est.extras["spatial_impl"] = self.spatial_impl
        est.extras["provisional"] = False
        est.extras["matern_nu"] = vc.matern_nu
        est.extras["matern_rho"] = vc.matern_rho
        return est
