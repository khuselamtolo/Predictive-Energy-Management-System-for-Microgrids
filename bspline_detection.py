"""
B-Spline outlier detection for load-profile data.

Implements the detection half of Chen, Li, Lau, Cao & Wang, "Automated
Load Curve Data Cleansing in Power Systems," IEEE Trans. Smart Grid 1(2),
pp. 213-221, 2010 - and, because the same model produces it for free, the
replacement value as well.

The idea, versus the Hampel filter in outlier_detection.py: instead of
comparing each reading to a local median, fit a smooth curve through the
whole window, put a point-wise prediction interval around it, and flag
whatever falls outside. The fitted value is then a ready-made replacement
for the flagged point. Hampel asks "is this far from its neighbours?";
this asks "is this far from the daily shape the rest of the window
implies?" - the two disagree more often than you would expect.

Paper equations, and where they are below:

    y_i     = f(t_i) + eps_i,  eps ~ N(0, sigma^2)                   (1)
    f(t)    = sum_k c_k phi_k(t)                                     (3)
    Phi[i,j]= phi_j(t_i)                                             (5)
    PENSSE  = (y - Phi c)'(y - Phi c) + lambda * c'Rc                (8)-(11)
    c_hat   = (Phi'Phi + lambda R)^-1 Phi'y                          (12)
    y_hat   = S y,  S = Phi (Phi'Phi + lambda R)^-1 Phi'             (14)-(15)
    df      = trace(S)                                               (16)
    s^2{pred}_i = MSE + Var[y_hat]_ii                                (22)
    MSE     = SSE / (n - df)                                         (23)
    Var[y_hat]  = S S' * MSE                                         (24)
    CI_i    = y_hat_i +- z * s{pred}_i                               (25)

Three implementation notes that matter:

1. THE HAT MATRIX IS NEVER FORMED. S is n-by-n; a month of 5-minute data
   is 8 064 points (520 MB) and eight months is ~70 000 (39 GB). Only
   trace(S) and diag(S S') are actually needed, and both reduce to
   K-by-K work plus a pass over the rows of Phi:

       A = Phi'Phi + lambda R,   M = A^-1
       trace(S)      = trace(M Phi'Phi)
       diag(S S')_i  = phi_i' (M Phi'Phi M') phi_i

   Since phi_i has only DEG+1 nonzeros (B-splines have compact support),
   the second one is a per-row 4-by-4 quadratic form. A month-long window
   fits in about a second instead of half a minute.

2. R (eq 10, the integral of the squared second derivative) uses the
   standard P-spline discretisation R ~ D2'D2 (Eilers & Marx) on an
   equally spaced knot grid. It behaves identically as a roughness dial
   and keeps A banded.

3. LAMBDA IS NOT PORTABLE. It depends on the units, the sampling rate and
   the number of basis functions, so a value tuned on one window is not
   meaningful on another dataset. Quote the effective degrees of freedom
   (`df` on the result) instead - that is scale-free.

Missing values are handled the way the paper describes: rows with NaN are
dropped from the fit, but the fitted curve is still evaluated at every
timestamp, so f_hat also supplies an estimate where the data is absent.

CAVEAT - this is least squares, so it is NOT robust the way the Hampel
filter is: the fit is pulled toward the very points it is meant to find.
The roughness penalty is the only thing holding it back. Set lambda too
low and gross outliers are simply absorbed into the curve.

CAVEAT - eq (1) assumes constant error variance, which does not hold for
this data: on the Westridge aggregate the residual spread runs from about
16 kW at low load to 34 kW at peak, so a constant-width interval is too
tight in the evening and too loose overnight, and roughly 2.5x more flags
land in the 17:00-22:00 window than its share of the day. Passing
log_scale=True fits log(load) instead and removes that bias - at the cost
of needing a much smaller lambda, and of requiring strictly positive data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import BSpline
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - surfaced to the user by the GUI
    HAS_SCIPY = False

DEG = 3                    # cubic B-splines
DEFAULT_LAMBDA = 1e4       # smoothing parameter; see note 3 above
DEFAULT_LOG_LAMBDA = 1e1   # a sensible starting point on the log scale
MIN_BASIS = DEG + 2
MAX_BASIS = 2200           # see choose_n_basis
Z_95 = 1.96                # eq (25) with alpha = 0.05


@dataclass
class BSplineResult:
    """Everything the caller needs to draw and explain one fit."""
    fitted: pd.Series
    lower: pd.Series
    upper: pd.Series
    flagged: pd.Series      # boolean, True where observed lies outside the interval
    df: float               # effective degrees of freedom, trace(S)
    n_basis: int
    n_fitted: int           # non-NaN points actually used in the fit
    mse: float
    lam: float
    log_scale: bool

    _span_days: float = 1.0
    basis_capped: bool = False   # True when K < n, i.e. coarser than the data

    @property
    def n_flagged(self) -> int:
        return int(self.flagged.sum())

    @property
    def df_per_day(self) -> float:
        """Effective df per day - the scale-free way to describe the fit."""
        return self.df / max(self._span_days, 1e-9)


def choose_n_basis(n_points: int, max_basis: int = MAX_BASIS) -> int:
    """
    How many basis functions to use.

    K = n (one per observation) is the paper's rule for local-pattern
    modelling, and it is also what makes lambda behave: with K = n on
    5-minute data, lambda = 1e4 holds a steady ~10 effective degrees of
    freedom per day whether the window is one day, three days or a week,
    so the dial means the same thing as you page around. Let K fall
    behind n and the same lambda quietly produces a smoother curve.

    The cap exists because the K-by-K solve is O(K^3): a week (2 016
    points at 5-minute resolution) takes about two seconds, and it grows
    fast from there. Above the cap the basis is coarser than the data,
    the fit can no longer trace a daily profile at the lambda you asked
    for, and you are effectively in the paper's "global pattern" regime -
    which is a legitimate mode for spotting whole corrupted regions, but
    a different one. `BSplineResult.basis_capped` says whether you are in
    it, so the caller can tell the user.
    """
    return int(max(MIN_BASIS, min(n_points - DEG, max_basis)))


def _design_matrix(x: np.ndarray, n_basis: int):
    """Sparse Phi (eq 5) on equally spaced clamped knots spanning x."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    interior = np.linspace(lo, hi, n_basis - DEG + 1)[1:-1]
    knots = np.concatenate([np.full(DEG + 1, lo), interior, np.full(DEG + 1, hi)])
    return BSpline.design_matrix(x, knots, DEG, extrapolate=True).tocsr()


def _difference_penalty(n_basis: int, order: int = 2) -> np.ndarray:
    """D2'D2 - discrete stand-in for R (eq 10); keeps A banded."""
    D = np.diff(np.eye(n_basis), n=order, axis=0)
    return D.T @ D


def _fit_coefficients(x: np.ndarray, y: np.ndarray, valid: np.ndarray,
                      n_basis: int, lam: float):
    """
    Solve the penalised least-squares fit (eq 12) on the valid rows.

    Shared by bspline_flags() (which goes on to build the prediction
    interval) and fit_curve() (which only wants the curve), so the two
    can never drift apart numerically.

    Returns (Phi_all, Phi_valid, G, M, coeffs) where Phi_all spans every
    timestamp - including the invalid ones - so the fitted curve can be
    evaluated where there is no observation. That is what makes this
    usable for imputation.
    """
    Phi_all = _design_matrix(x, n_basis)
    Phi = Phi_all[valid]
    G = (Phi.T @ Phi).toarray()
    A = G + lam * _difference_penalty(n_basis)
    M = np.linalg.solve(A, np.eye(n_basis))
    coeffs = M @ (Phi.T @ y[valid])
    return Phi_all, Phi, G, M, coeffs


def fit_curve(
    series: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    n_basis: Optional[int] = None,
    max_basis: int = MAX_BASIS,
) -> pd.Series:
    """
    Fit a penalised B-spline and return the smooth curve, evaluated at
    every timestamp in `series` - including the ones that are NaN.

    NaNs take no part in the fit but still receive a fitted value, which
    is precisely the imputation step of Chen et al.: drop the corrupted
    reading, fit the underlying curve to what remains, and read the
    replacement off the curve.
    """
    if not HAS_SCIPY:
        raise RuntimeError("The B-spline method needs SciPy. Install it with:  pip install scipy")
    if lam <= 0:
        raise ValueError("lambda must be a positive number")

    y = series.to_numpy(dtype=float)
    valid = np.isfinite(y)
    n_valid = int(valid.sum())
    if n_basis is None:
        n_basis = choose_n_basis(n_valid, max_basis)
    if n_valid <= n_basis or n_valid < MIN_BASIS + 2:
        raise ValueError(
            f"Not enough valid readings to fit a B-spline ({n_valid} for {n_basis} basis functions)."
        )

    x = (series.index - series.index[0]).total_seconds().to_numpy() / 86400.0
    Phi_all, _, _, _, coeffs = _fit_coefficients(x, y, valid, n_basis, lam)
    return pd.Series(Phi_all @ coeffs, index=series.index, name=series.name)


def _row_blocks(Phi):
    """
    Repack a CSR design matrix into dense (n, w) value/column arrays,
    where w = max nonzeros per row (DEG+1 for B-splines). Lets the
    diag(S S') quadratic form be evaluated per row without ever
    materialising an n-by-K product.
    """
    counts = np.diff(Phi.indptr)
    width = int(counts.max())
    n = Phi.shape[0]
    cols = np.zeros((n, width), dtype=np.intp)
    vals = np.zeros((n, width), dtype=float)
    slot = np.arange(width)[None, :] < counts[:, None]
    cols[slot] = Phi.indices
    vals[slot] = Phi.data
    return cols, vals


def bspline_flags(
    series: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    n_basis: Optional[int] = None,
    z: float = Z_95,
    log_scale: bool = False,
    max_basis: int = MAX_BASIS,
) -> BSplineResult:
    """
    Fit a penalised B-spline to `series` and flag readings outside the
    point-wise prediction interval.

    `series` must have a DatetimeIndex. NaNs are allowed: they are left
    out of the fit and never flagged, but the fitted curve and interval
    are still returned for their timestamps.
    """
    if not HAS_SCIPY:
        raise RuntimeError(
            "The B-spline method needs SciPy. Install it with:  pip install scipy"
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must be indexed by timestamp")
    if lam <= 0:
        raise ValueError("lambda must be a positive number")

    obs = series.to_numpy(dtype=float)
    valid = np.isfinite(obs)
    n_valid = int(valid.sum())

    span_days = (series.index[-1] - series.index[0]).total_seconds() / 86400.0
    if n_basis is None:
        n_basis = choose_n_basis(n_valid, max_basis)
    basis_capped = n_basis < n_valid - DEG

    if n_valid <= n_basis or n_valid < MIN_BASIS + 2:
        raise ValueError(
            f"Not enough readings in this window to fit a B-spline "
            f"({n_valid} usable point(s) for {n_basis} basis functions). "
            "Widen the date range."
        )

    y = obs.copy()
    if log_scale:
        if np.nanmin(obs) <= 0:
            raise ValueError(
                "Log scale needs strictly positive readings, and this window "
                "contains zero or negative values. Use the linear scale, or "
                "pick a series/range without them."
            )
        y = np.log(y)

    # x in days since the window start: makes the knot density mean the
    # same thing regardless of the data's sampling rate.
    x = (series.index - series.index[0]).total_seconds().to_numpy() / 86400.0

    Phi_all, Phi, G, M, coeffs = _fit_coefficients(x, y, valid, n_basis, lam)
    yv = y[valid]
    fit_all = Phi_all @ coeffs                               # f_hat at every timestamp

    df = float(np.trace(M @ G))                              # trace(S), eq (16)
    resid = yv - (Phi @ coeffs)
    sse = float(resid @ resid)
    denom = max(n_valid - df, 1.0)
    mse = sse / denom                                        # eq (23)

    # diag(S S') without forming S: phi_i' B phi_i with B = M G M'
    B = M @ G @ M.T
    cols, vals = _row_blocks(Phi_all)
    Bsub = B[cols[:, :, None], cols[:, None, :]]             # (n, w, w)
    diag_ss = np.einsum("ia,iab,ib->i", vals, Bsub, vals)

    s_pred = np.sqrt(mse + mse * diag_ss)                    # eq (22)
    lower_t, upper_t = fit_all - z * s_pred, fit_all + z * s_pred   # eq (25)

    flagged = np.zeros(len(obs), dtype=bool)
    flagged[valid] = (y[valid] < lower_t[valid]) | (y[valid] > upper_t[valid])

    if log_scale:
        fit_all, lower_t, upper_t = np.exp(fit_all), np.exp(lower_t), np.exp(upper_t)

    idx = series.index
    return BSplineResult(
        fitted=pd.Series(fit_all, index=idx),
        lower=pd.Series(lower_t, index=idx),
        upper=pd.Series(upper_t, index=idx),
        flagged=pd.Series(flagged, index=idx),
        df=df,
        n_basis=n_basis,
        n_fitted=n_valid,
        mse=mse,
        lam=lam,
        log_scale=log_scale,
        _span_days=max(span_days, 1e-9),
        basis_capped=basis_capped,
    )


if __name__ == "__main__":
    import sys
    from data_loader import load_load_profile

    path = sys.argv[1] if len(sys.argv) > 1 else "aggregate_total_load.csv"
    data = load_load_profile(path)
    s = data.df[data.households[0]]
    print(f"{data.source_path.name}: {data.households[0]}, {len(s):,} readings\n")
    print(f"{'lambda':>10} {'basis':>7} {'eff. df':>9} {'flagged':>9} {'%':>7}")
    for lam in (1e2, 1e3, 1e4, 1e5, 1e6):
        r = bspline_flags(s, lam=lam)
        print(f"{lam:10.0e} {r.n_basis:7d} {r.df:9.1f} {r.n_flagged:9d} "
              f"{100 * r.flagged.mean():6.2f}%")
