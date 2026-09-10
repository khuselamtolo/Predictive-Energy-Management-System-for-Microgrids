"""
The Data Properties engine: statistics for whatever window is on screen.

Four metrics are required - distribution type, mean, median, max, min -
and the rest come free once you have the array in hand.

"Distribution type" is the one that is easy to answer badly, so the
method is explicit rather than a library call:

  * Fit six candidate families by maximum likelihood and rank them by
    AIC, reporting the winner, the margin over the runner-up, and a KS
    distance. A win by DeltaAIC of 2 is not a real answer, and the panel
    says "indistinguishable" instead of picking one.

  * Test for a point mass at zero FIRST. Over a full day GHI is exactly
    zero for about half its samples, and no continuous distribution
    describes that - every fit would return confident nonsense on the
    single most-inspected channel in a solar file. Zero-inflated columns
    are reported as such and fitted on the non-zero part only.

  * Report effect sizes, never p-values. Shapiro-Wilk is capped at
    n = 5000 and at n = 72,576 every normality test rejects, because
    real data is never exactly normal and the test merely detects that.
    Skewness, excess kurtosis and KS distance say HOW FAR from normal,
    which is the decision-useful quantity.

Fitting six families over 70k points costs a few hundred milliseconds,
so results are cached by (layer id, channel, window) and the moments are
computed separately from the fit - the required four metrics appear
instantly whether or not the fit has finished.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    HAS_SCIPY = True
except ImportError:  # the panel degrades to moments only
    HAS_SCIPY = False

# Families worth trying on power and irradiance data. Normal for a
# well-behaved aggregate; lognormal/gamma/Weibull for the right-skewed
# single-household case; exponential and uniform as sanity anchors.
CANDIDATES = ("norm", "lognorm", "gamma", "weibull_min", "expon", "uniform")

# Share of readings piled at the lower bound above which a continuous fit
# is meaningless and must be reported as a point mass instead.
ZERO_INFLATION_THRESHOLD = 0.20

# The point mass must NOT be tested with exact equality. Raw GHI really is
# 0.000 all night, but the moment it goes through a zero-phase Butterworth
# those readings ring a few tenths either side of zero: measured on the
# filtered test signal, exact zeros fall from 25.3% to 0.02% while 49.8%
# still sit within 1% of the peak. An equality test would therefore see no
# point mass at all and happily report "uniform" for a signal that is half
# darkness - the precise failure this guard exists to prevent. So look for
# a concentration near the series minimum instead, scaled to its range.
FLOOR_RELATIVE = 0.005   # within 0.5% of the observed range counts as "at the floor"
FLOOR_ABSOLUTE = 1e-9
INDISTINGUISHABLE_AIC = 2.0      # ΔAIC below this is not a real preference
POOR_FIT_KS = 0.10               # above this, the winning family is still a bad description
FIT_SAMPLE_CAP = 20_000          # MLE on more than this buys nothing; subsample deterministically


@dataclass
class FitResult:
    family: str
    aic: float
    ks: float
    params: tuple


@dataclass
class WindowStats:
    """Everything the inspector shows for one channel over one window."""

    channel: str
    unit: str
    start: pd.Timestamp
    end: pd.Timestamp

    n: int = 0
    n_missing: int = 0
    pct_missing: float = 0.0
    longest_gap: int = 0

    mean: float = np.nan
    median: float = np.nan
    minimum: float = np.nan
    maximum: float = np.nan
    std: float = np.nan
    iqr: float = np.nan
    mad: float = np.nan
    cv: float = np.nan
    skew: float = np.nan
    excess_kurtosis: float = np.nan
    percentiles: Dict[str, float] = field(default_factory=dict)

    energy_kwh: Optional[float] = None
    zero_fraction: float = 0.0     # share piled at the lower bound
    zero_inflated: bool = False
    floor_level: float = 0.0       # the value that bound sits at
    floor_is_zero: bool = True

    fits: List[FitResult] = field(default_factory=list)
    distribution: str = "not computed"
    verdict: str = ""
    fit_pending: bool = False

    # -- rendering --------------------------------------------------------
    def required_lines(self) -> List[Tuple[str, str]]:
        """The four the brief asks for, plus the shape verdict."""
        u = f" {self.unit}" if self.unit else ""
        return [
            ("Distribution", self.distribution),
            ("Mean", f"{self.mean:,.3f}{u}"),
            ("Median", f"{self.median:,.3f}{u}"),
            ("Minimum", f"{self.minimum:,.3f}{u}"),
            ("Maximum", f"{self.maximum:,.3f}{u}"),
        ]

    def detail_groups(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        u = f" {self.unit}" if self.unit else ""
        coverage = [
            ("Samples", f"{self.n:,}"),
            ("Missing", f"{self.n_missing:,}  ({self.pct_missing:.2f}%)"),
            ("Longest gap", f"{self.longest_gap:,} samples"),
            ("Window", f"{self.start:%Y-%m-%d %H:%M} → {self.end:%Y-%m-%d %H:%M}"),
        ]
        spread = [
            ("Std deviation", f"{self.std:,.3f}{u}"),
            ("IQR", f"{self.iqr:,.3f}{u}"),
            ("MAD", f"{self.mad:,.3f}{u}"),
            ("Range", f"{self.maximum - self.minimum:,.3f}{u}"),
            ("Coeff. of variation", f"{self.cv:.3f}" if np.isfinite(self.cv) else "—"),
        ]
        shape = [
            ("Skewness", f"{self.skew:+.3f}"),
            ("Excess kurtosis", f"{self.excess_kurtosis:+.3f}"),
        ] + [(f"p{k}", f"{v:,.3f}{u}") for k, v in self.percentiles.items()]
        groups = [("Coverage", coverage), ("Spread", spread), ("Shape", shape)]
        if self.energy_kwh is not None:
            groups.append(("Energy", [("Total over window", f"{self.energy_kwh:,.1f} kWh")]))
        if self.fits:
            groups.append(("Distribution fits (ranked by AIC)", [
                (f.family, f"AIC {f.aic:,.0f}   KS D={f.ks:.4f}") for f in self.fits[:4]
            ]))
        return groups

    def as_text(self) -> str:
        out = [f"DATA PROPERTIES — {self.channel}",
               "=" * 52, ""]
        for k, v in self.required_lines():
            out.append(f"  {k:<22} {v}")
        if self.verdict:
            out += ["", "  " + self.verdict]
        for title, rows in self.detail_groups():
            out += ["", title.upper()]
            out += [f"  {k:<22} {v}" for k, v in rows]
        return "\n".join(out)


def _longest_nan_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _verdict_from_moments(skew: float, exkurt: float, values: np.ndarray,
                          is_power: bool = False) -> str:
    """Plain-language shape reading, which is what most users actually want."""
    bits = []
    if abs(skew) < 0.5 and abs(exkurt) < 1:
        bits.append("approximately normal")
    if skew > 1:
        bits.append("right-skewed — a long tail of high readings")
    elif skew < -1:
        bits.append("left-skewed — a long tail of low readings")
    if exkurt > 3:
        bits.append("heavy-tailed — extreme values far more common than a normal would predict")
    if _looks_bimodal(values):
        bits.append("bimodal — two distinct operating levels"
                    + (", typical of a morning and an evening peak" if is_power else ""))
    if not bits:
        bits.append("moderately skewed, no strong tail")
    return "Shape: " + "; ".join(bits) + "."


def _looks_bimodal(values: np.ndarray, bins: int = 40) -> bool:
    """
    Count prominent peaks in a smoothed histogram.

    Bimodality matters specifically for load data - a daily household
    profile usually IS bimodal, and saying "not normal" without saying
    "because it has two peaks" is a wasted diagnosis.
    """
    if len(values) < 200:
        return False
    counts, _ = np.histogram(values, bins=bins)
    if counts.max() == 0:
        return False
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    floor = 0.15 * smooth.max()
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > floor]
    if len(peaks) < 2:
        return False
    # Require a real trough between the two tallest peaks, otherwise a
    # noisy shoulder counts as a mode.
    peaks.sort(key=lambda i: -smooth[i])
    a, b = sorted(peaks[:2])
    trough = smooth[a:b + 1].min()
    return trough < 0.6 * min(smooth[a], smooth[b])


def fit_distributions(values: np.ndarray) -> List[FitResult]:
    """MLE-fit the candidate families and rank them by AIC."""
    if not HAS_SCIPY or len(values) < 30:
        return []

    x = values
    if len(x) > FIT_SAMPLE_CAP:
        # Deterministic stride, so the same window always gives the same
        # answer - a statistic that changes when you re-open the panel is
        # worse than no statistic.
        x = x[:: int(np.ceil(len(x) / FIT_SAMPLE_CAP))]

    results: List[FitResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for family in CANDIDATES:
            dist = getattr(sps, family, None)
            if dist is None:
                continue
            # Families with support on (0, inf) cannot host non-positive data.
            if family in ("lognorm", "gamma", "weibull_min", "expon") and x.min() <= 0:
                continue
            try:
                params = dist.fit(x)
                ll = float(np.sum(dist.logpdf(x, *params)))
                if not np.isfinite(ll):
                    continue
                aic = 2 * len(params) - 2 * ll
                ks = float(sps.kstest(x, family, args=params).statistic)
                results.append(FitResult(family, aic, ks, params))
            except Exception:  # noqa: BLE001 - a family that will not fit is not an error
                continue
    results.sort(key=lambda r: r.aic)
    return results


def floor_mass(values: np.ndarray) -> Tuple[float, float]:
    """
    Fraction of readings piled at the series' lower bound, and where that
    bound is. Tolerance is scaled to the observed range so it survives a
    filter having smeared an exact zero into a near-zero.
    """
    lo, hi = float(values.min()), float(values.max())
    level = lo + max(FLOOR_ABSOLUTE, FLOOR_RELATIVE * (hi - lo))
    return float(np.mean(values <= level)), level


def describe_fit(fits: List[FitResult], zero_inflated: bool, zero_fraction: float,
                 floor_is_zero: bool = True, floor_level: float = 0.0) -> str:
    if zero_inflated:
        where = "at or near zero" if floor_is_zero else f"at the lower bound ({floor_level:,.3g})"
        prefix = f"zero-inflated ({zero_fraction:.1%} {where})"
        if not fits:
            return prefix
        return f"{prefix}; remainder best fits {fits[0].family}"
    if not fits:
        return "not computed" if not HAS_SCIPY else "too few samples to fit"
    best = fits[0]

    # AIC only ranks the candidates against each other - it will happily
    # crown a winner when every one of them is wrong. KS distance is the
    # absolute check, so a bad best fit is reported as a bad fit rather
    # than dressed up as an answer. This fires on raw household data,
    # where a single -999 sentinel drags the minimum below zero, rules
    # out every positive-support family, and leaves "norm" winning with
    # KS D = 0.50.
    if best.ks > POOR_FIT_KS:
        return (f"no standard family fits well (best: {best.family}, KS D={best.ks:.2f}) — "
                "see the shape note below")

    if len(fits) > 1:
        gap = fits[1].aic - best.aic
        if gap < INDISTINGUISHABLE_AIC:
            return (f"{best.family} or {fits[1].family} — indistinguishable on this window "
                    f"(ΔAIC {gap:.1f})")
        return (f"{best.family}  (best of {len(fits)} by AIC; ΔAIC {gap:,.0f} over "
                f"{fits[1].family}; KS D={best.ks:.3f})")
    return f"{best.family}  (KS D={best.ks:.3f})"


def compute(series: pd.Series, channel: str, unit: str = "",
            interval: Optional[pd.Timedelta] = None,
            is_power: bool = False, with_fit: bool = True) -> WindowStats:
    """Full statistics for one series slice. Safe on empty/all-NaN input."""
    start = series.index.min() if len(series) else pd.NaT
    end = series.index.max() if len(series) else pd.NaT
    st = WindowStats(channel=channel, unit=unit, start=start, end=end)

    raw = pd.to_numeric(series, errors="coerce")
    st.n = int(len(raw))
    nan_mask = raw.isna().to_numpy()
    st.n_missing = int(nan_mask.sum())
    st.pct_missing = 100.0 * st.n_missing / st.n if st.n else 0.0
    st.longest_gap = _longest_nan_run(nan_mask)

    values = raw.dropna().to_numpy(dtype=float)
    if values.size == 0:
        st.distribution = "no readings in this window"
        return st

    st.mean = float(np.mean(values))
    st.median = float(np.median(values))
    st.minimum = float(np.min(values))
    st.maximum = float(np.max(values))
    st.std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    q25, q75 = np.percentile(values, [25, 75])
    st.iqr = float(q75 - q25)
    st.mad = float(np.median(np.abs(values - st.median)))
    st.cv = float(st.std / st.mean) if st.mean else np.nan
    st.skew = float(pd.Series(values).skew()) if values.size > 2 else np.nan
    st.excess_kurtosis = float(pd.Series(values).kurtosis()) if values.size > 3 else np.nan
    st.percentiles = {
        k: float(np.percentile(values, p))
        for k, p in (("1", 1), ("5", 5), ("25", 25), ("75", 75), ("95", 95), ("99", 99))
    }

    if is_power and interval is not None:
        st.energy_kwh = float(values.sum() * interval.total_seconds() / 3600.0)

    st.zero_fraction, st.floor_level = floor_mass(values)
    st.zero_inflated = st.zero_fraction >= ZERO_INFLATION_THRESHOLD
    st.floor_is_zero = abs(st.floor_level) < max(1.0, 0.01 * abs(st.maximum))

    fit_values = values[values > st.floor_level] if st.zero_inflated else values
    if fit_values.size < 30:
        fit_values = values
    if with_fit:
        st.fits = fit_distributions(fit_values)
        st.distribution = describe_fit(st.fits, st.zero_inflated, st.zero_fraction,
                                       st.floor_is_zero, st.floor_level)
    else:
        st.fit_pending = True
        st.distribution = "fitting…"

    st.verdict = _verdict_from_moments(st.skew, st.excess_kurtosis, fit_values, is_power)
    if st.zero_inflated:
        where = "at or near zero" if st.floor_is_zero else f"at the lower bound {st.floor_level:,.3g}"
        st.verdict += (
            f" The {st.zero_fraction:.1%} of readings {where} are a point mass, not part of any "
            "continuous distribution — night-time on an irradiance channel, or an idle meter. "
            "The fit above describes only the readings above that floor."
        )
    if st.n >= 5000:
        st.verdict += (
            f" With n = {st.n:,} every formal normality test rejects, so the numbers above are "
            "reported as effect sizes rather than p-values."
        )
    return st


class StatsCache:
    """
    Small LRU keyed on (layer id, channel, window).

    Layers are immutable, so a layer id can never go stale - which is the
    second reason the layer stack is built the way it is.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._store: "OrderedDict[tuple, WindowStats]" = OrderedDict()

    def get(self, key: tuple) -> Optional[WindowStats]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: tuple, value: WindowStats) -> WindowStats:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return value

    def clear(self) -> None:
        self._store.clear()
