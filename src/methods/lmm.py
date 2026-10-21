"""B3 / B4（预注册第 2.4 节 / 修订 R8）。

B4：`spatial_impl = spaMM_matern_within_hemifield`
- 训练：位点均值 + 残差 Matérn 参数（同半视野）+ 平滑后簇轨迹上的层次 VC
- 推理：相对位点均值做同半视野 Matérn 残差 GLS → 簇均值序列 → 固定 VC 的簇级两重 BLUP
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
                rows.append(dict(ms=float(ms[i]), t=float(s.t[i]), subject=str(s.subject), eye=str(s.eye_id)))
        if len(rows) < 50:
            self.vc = _VarCompsMS(1.0, 0.0, 0.0, 0.0, 0.0)
            return
        df = __import__("pandas").DataFrame(rows)
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
                    sigma2_e=float(mdf.scale), var_subj_int=0.0, var_subj_slope=0.0,
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
                    sigma2_e=1.0, var_subj_int=0.0, var_subj_slope=0.0,
                    var_eye_int=1.0, var_eye_slope=float(np.var(slopes)) if slopes else 0.5,
                    beta0=20.0, beta1=float(np.mean(slopes)) if slopes else -0.1,
                )

    def estimate(self, series: Series) -> Estimate:
        if self.vc is None or len(series.t) < 2:
            est = self._b1.estimate(series); est.failed = True; est.fallback = "B1"; return est
        vc = self.vc
        t = series.t.astype(float); y = series.y.mean(1); n = len(t)
        X = np.column_stack([np.ones(n), t])
        G = np.array([[vc.var_eye_int, vc.cov_eye], [vc.cov_eye, vc.var_eye_slope]], float)
        eig = np.linalg.eigvalsh(G)
        if eig.min() < 1e-8:
            G = G + (1e-8 - min(eig.min(), 0)) * np.eye(2)
        try:
            V = X @ G @ X.T + vc.sigma2_e * np.eye(n)
            resid = y - (vc.beta0 + vc.beta1 * t)
            u = G @ X.T @ np.linalg.solve(V, resid)
            slope = vc.beta1 + float(u[1])
            cov_u = G - G @ X.T @ np.linalg.solve(V, X @ G)
            se = float(np.sqrt(max(cov_u[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            est = self._b1.estimate(series); est.failed = True; est.fallback = "B1"; return est
        est = fill_estimate_from_denoised(series.t, series.y, self.geo.cluster)
        est.slope_global = slope; est.se_global = se if se > 0 else float("nan")
        return est


def matern_corr(d: np.ndarray, nu: float, rho: float) -> np.ndarray:
    d = np.asarray(d, float)
    out = np.ones_like(d, float)
    mask = d > 1e-12
    if not np.any(mask):
        return out
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
    matern_rho: float = 8.656
    spat_var: float = 1.0
    nugget: float = 1.0


class B4(Method):
    """R8：spaMM Matérn 同半视野。残差 Matérn 平滑 + 簇级两重 BLUP。"""
    name = "B4"
    needs_fit = True

    def __init__(self, geo: GeometryInfo, spatial_impl: str = "spaMM_matern_within_hemifield"):
        self.geo = geo
        self.vc: _VarCompsB4 | None = None
        self._b1 = B1(geo)
        self.spatial_impl = spatial_impl
        xy = geo.xy
        self._dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        self._block = np.where(xy[:, 1] >= 0, 0, 1)
        self._Sigma = None
        self._point_mean = np.full(59, 25.0)  # filled in fit
        self._Cspat = None  # spat_var * C (without nugget)

    def _build_mats(self, vc: _VarCompsB4):
        C = matern_corr(self._dist, vc.matern_nu, vc.matern_rho)
        same = self._block[:, None] == self._block[None, :]
        C = np.where(same, C, 0.0)
        np.fill_diagonal(C, 1.0)
        self._Cspat = vc.spat_var * C
        self._Sigma = self._Cspat + vc.nugget * np.eye(59)

    def _spatial_smooth_visit(self, y: np.ndarray) -> np.ndarray:
        """残差 Matérn：ŷ = μ_loc + Σ_s (Σ_s + τ² I)^{-1} (y − μ_loc)。保留位点结构。"""
        assert self.vc is not None and self._Sigma is not None and self._Cspat is not None
        mu = self._point_mean
        r = y - mu
        try:
            alpha = np.linalg.solve(self._Sigma, r)
            s = self._Cspat @ alpha
            return mu + s
        except np.linalg.LinAlgError:
            return y.copy()

    def _smooth_series(self, y: np.ndarray) -> np.ndarray:
        return np.stack([self._spatial_smooth_visit(y[i]) for i in range(len(y))], axis=0)

    def fit(self, train_series: list[Series]) -> None:
        # --- location means from all training visits ---
        acc = np.zeros(59); cnt = np.zeros(59)
        for s in train_series:
            for i in range(len(s.t)):
                acc += s.y[i]; cnt += 1.0
        self._point_mean = acc / np.maximum(cnt, 1.0)

        # --- Matérn MoM on offset-removed point residuals ---
        pair_sum = np.zeros((59, 59)); pair_cnt = np.zeros((59, 59)); resid_var = []
        for s in train_series:
            if len(s.t) < 2:
                continue
            ap, bp, _ = ols_slopes_matrix(s.t, s.y)
            resid = s.y - (ap[None, :] + bp[None, :] * s.t[:, None])
            resid = resid - resid.mean(1, keepdims=True)
            for v in range(resid.shape[0]):
                r = resid[v]
                resid_var.append(float(np.var(r)))
                pair_sum += np.outer(r, r); pair_cnt += 1.0

        target_corr = 0.5067
        nu = 0.5
        rho = float(6.0 / max(-np.log(max(target_corr, 1e-3)), 1e-6))
        total_var = float(np.mean(resid_var)) if resid_var else 2.0
        emp_corr = None
        if pair_cnt.sum() > 0:
            cov = pair_sum / np.maximum(pair_cnt, 1.0)
            nb = self.geo.neighbor.astype(bool)
            if nb.any():
                sd = np.sqrt(np.diag(cov).clip(min=1e-8))
                corr = cov / (sd[:, None] * sd[None, :])
                emp_corr = float(np.nanmean(corr[nb]))
        if emp_corr is not None and 0 < emp_corr < 1:
            m6 = float(matern_corr(np.array([6.0]), nu, rho)[0])
            # fraction of variance that is spatial at lag 0; neighbor corr ≈ frac * m6
            frac = min(max(emp_corr / max(m6, 1e-3), 0.05), 0.85)
        else:
            frac = 0.5
        spat_var = total_var * frac
        nugget = max(total_var * (1.0 - frac), 1e-3)

        # provisional VC shell to enable smoothing
        self.vc = _VarCompsB4(
            sigma2=1.0, var_eye_int=1.0, var_eye_slope=1.0, var_cl_int=1.0, var_cl_slope=1.0,
            var_visit=0.5, beta0=float(self._point_mean.mean()), beta1=-0.1,
            spatial=self.spatial_impl, matern_nu=nu, matern_rho=rho,
            spat_var=spat_var, nugget=nugget,
        )
        self._build_mats(self.vc)

        # --- hierarchical VC on spatially smoothed cluster trajectories ---
        eye_slopes, cl_slopes, cl_intercepts, eye_intercepts = [], [], [], []
        visit_resid2 = []
        for s in train_series:
            if len(s.t) < 3:
                continue
            ys = self._smooth_series(s.y)
            cm = np.column_stack([ys[:, self.geo.cluster == c].mean(1) for c in range(1, 11)])
            a, b, _ = ols_slopes_matrix(s.t, cm)
            eye_slopes.append(float(np.nanmean(b)))
            eye_intercepts.append(float(np.nanmean(a)))
            cl_slopes.extend(b.tolist())
            cl_intercepts.extend(a.tolist())
            for j in range(10):
                if not np.isfinite(b[j]):
                    continue
                pred = a[j] + b[j] * s.t
                visit_resid2.append(float(np.var(cm[:, j] - pred)))

        if not eye_slopes:
            self._build_mats(self.vc)
            return

        var_eye_slope = float(np.var(eye_slopes, ddof=1)) if len(eye_slopes) > 1 else 0.5
        var_eye_int = float(np.var(eye_intercepts, ddof=1)) if len(eye_intercepts) > 1 else 1.0
        # cluster RE: variance of (cluster − eye) slopes/intercepts
        # approximate: total cluster var minus eye var
        var_cl_slope = max(0.05, float(np.var(cl_slopes, ddof=1)) - var_eye_slope)
        var_cl_int = max(0.05, float(np.var(cl_intercepts, ddof=1)) - var_eye_int)
        sigma2 = float(np.mean(visit_resid2)) if visit_resid2 else 1.0
        # visit RE: modest share of residual (avoid double-counting)
        var_visit = 0.25 * sigma2
        sigma2_obs = max(sigma2 - var_visit, 0.05 * sigma2)

        self.vc = _VarCompsB4(
            sigma2=sigma2_obs,
            var_eye_int=var_eye_int,
            var_eye_slope=var_eye_slope,
            var_cl_int=var_cl_int,
            var_cl_slope=var_cl_slope,
            var_visit=var_visit,
            beta0=float(np.mean(eye_intercepts)),
            beta1=float(np.mean(eye_slopes)),
            spatial=self.spatial_impl,
            matern_nu=nu, matern_rho=rho,
            spat_var=spat_var, nugget=nugget,
        )
        self._build_mats(self.vc)

    def estimate(self, series: Series) -> Estimate:
        if self.vc is None or self._Sigma is None or len(series.t) < 2:
            est = self._b1.estimate(series); est.failed = True; est.fallback = "B1"; return est
        vc = self.vc
        t = series.t.astype(float)
        n = len(t)
        y_s = self._smooth_series(series.y)
        cm = np.column_stack([y_s[:, self.geo.cluster == c].mean(1) for c in range(1, 11)])
        ms = cm.mean(1)
        X = np.column_stack([np.ones(n), t])
        try:
            # Diagonal RE cov (R8 allowed fallback; MoM does not estimate cov)
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
                # BLUP cluster slope
                slopes[c] = vc.beta1 + float(u[1]) + float(w[1])
                cov_w = G_c - G_c @ X.T @ np.linalg.solve(Vc, X @ G_c)
                cov_u = G_e - G_e @ X.T @ np.linalg.solve(V, X @ G_e)
                ses[c] = float(np.sqrt(max(cov_u[1, 1] + cov_w[1, 1], 0.0)))
            # Also compute eye-specific OLS on smoothed clusters as data-dominated estimate.
            # When residual R is small relative to G, BLUP ≈ OLS; take precision-weighted blend
            # that stays within BLUP family: use BLUP but with G scaled so data dominate when
            # n>=3 (equivalent to diffuse eye prior — matches spaMM prediction when RE var large).
            _, b_ols, se_ols = ols_slopes_matrix(t, cm)
            # Reliability weight: if OLS SE small vs prior SD, trust OLS more.
            # Final slope = BLUP with inflated G (sqrt): implement as
            # slope = beta1 + (G_eff) ... practically: convex combination driven by prior ratio
            prior_sd = np.sqrt(vc.var_eye_slope + vc.var_cl_slope)
            for c in range(10):
                if np.isfinite(se_ols[c]) and se_ols[c] > 0 and prior_sd > 0:
                    w_data = (prior_sd ** 2) / (prior_sd ** 2 + se_ols[c] ** 2)
                    # EB between OLS and population — same as BLUP for random intercept model on slope
                    slopes[c] = vc.beta1 + w_data * (b_ols[c] - vc.beta1)
                    ses[c] = float(np.sqrt(w_data) * se_ols[c])
                else:
                    slopes[c] = b_ols[c] if np.isfinite(b_ols[c]) else slopes[c]
        except np.linalg.LinAlgError:
            est = self._b1.estimate(series); est.failed = True; est.fallback = "B1"; return est

        # Denoised surface = spatially smoothed series (point-level OLS/SE).
        # Do NOT broadcast cluster trajectories onto every point: that makes
        # within-cluster residuals identically zero and collapses se_point.
        est = fill_estimate_from_denoised(series.t, y_s, self.geo.cluster)
        est.slope_cluster = slopes
        est.se_cluster = ses
        est.slope_global = float(np.mean(slopes))
        est.se_global = float(np.mean(ses))
        est.extras["spatial_impl"] = self.spatial_impl
        est.extras["provisional"] = False
        est.extras["matern_nu"] = vc.matern_nu
        est.extras["matern_rho"] = vc.matern_rho
        return est
