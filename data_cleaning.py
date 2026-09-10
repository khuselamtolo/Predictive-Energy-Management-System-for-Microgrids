"""
Outlier removal and B-spline imputation, with an aggregate cleaning report.

This is the "fix them" step that the rest of the pipeline has been
building toward: detection flags a reading, the reading is *popped*
(set to NaN, so it takes no part in anything downstream), and the hole is
filled from a penalised B-spline fitted to the readings around it. That
is the imputation half of Chen, Li, Lau, Cao & Wang (2010) - corrupted
data is replaced by the value the underlying smooth curve implies.

Why fit locally rather than once over the whole series
------------------------------------------------------
A global fit is both wrong and impossible here. Impossible because the
hat matrix is n-by-n and eight months is ~70 000 readings; wrong because
capping the basis to make it tractable leaves a curve far too coarse to
trace a daily load profile, so every imputed value would sit on a smooth
seasonal trend instead of on the right point of the right day.

So each flagged region gets its own fit over a window of surrounding
context - by default an hour either side, widening with the size of the
hole. That is fast, and the curve only has to describe the shape of the
ramp it sits on rather than eight months of seasonality.

Context and smoothing were both chosen by measurement, not taste: see
the constants below for the hold-out grids behind them. The first
version used a much wider window and far heavier smoothing, and the
hold-out test caught it imputing *worse* than plain linear interpolation
(MAE 29.5 kW against 21.0). The current defaults beat linear at every
gap length and tie it at single points.

Two details that matter and are easy to get wrong:

  * Every flagged reading inside the context window is popped before the
    fit, not just the region being repaired. Leaving a neighbouring
    outlier in would drag the very curve being used to replace its
    neighbour.
  * A run longer than `max_gap` is deliberately NOT imputed. Filling a
    long outage with a smooth guess manufactures data that looks real;
    it stays NaN and the report says so. Same safeguard, and the same
    default, as interpolate_and_aggregate.py.

How good is the imputation?
---------------------------
`holdout_validation()` answers that on the user's own data rather than in
principle: hide readings that were never flagged, impute them, and
compare against the values that were actually there - alongside linear
interpolation as a baseline. The result goes into the report, so the
error attached to the filled-in values is measured, not assumed.

Usage:
    from data_cleaning import clean_series
    result = clean_series(series, method="zscore", window=48, n_sigmas=3.0)
    print(result.report())
    result.cleaned.to_csv("aggregate_cleaned.csv")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bspline_detection import fit_curve
from outlier_detection import rolling_zscore

DETECTORS = {"zscore": rolling_zscore}

#: Minimum readings of context either side of a gap for its local fit.
#: Context actually used is max(DEFAULT_CONTEXT, 4 x gap length), so a
#: longer hole gets proportionally more to lean on.
#:
#: 12 samples (1 hour) rather than something wider because it was
#: measured: on the 8-month aggregate, a hold-out test over a grid of
#: context and lambda found MAE rising steadily with context - 17.4 kW
#: at +-12, 19.6 at +-24, 18.6 at +-48, 19.7 at +-72. Five-minute load is
#: strongly autocorrelated, so the nearest readings carry nearly all the
#: information and a wider window just dilutes them.
DEFAULT_CONTEXT = 12

#: Smoothing for the local imputation fits. NOT the same as the lambda
#: used for B-spline *detection* (1e4): that fits a week to find points
#: that stand out, this fits ~an hour to estimate one missing value, and
#: the right amount of smoothing differs by orders of magnitude.
#:
#: Effective lambda is DEFAULT_IMPUTE_LAMBDA x gap length, because the
#: best value rises with the size of the hole - measured MAE by gap
#: length on the aggregate:
#:
#:   gap  lambda=0.1  lambda=1  lambda=10  lambda=100   linear
#:     3       21.71     21.45      21.57       21.67    22.26
#:     6       27.31     25.12      23.17       23.14    24.79
#:    12       31.74     26.48      24.41       24.34    25.79
#:    24       53.89     33.10      27.30       26.64    29.39
#:
#: Short gaps want a curve that tracks local detail; long ones want a
#: stable one, since there is less nearby evidence to justify detail.
DEFAULT_IMPUTE_LAMBDA = 1.0

#: Longest run of consecutive flagged readings that will be imputed.
#: 24 samples = 2 hours at 5-minute data, matching the safeguard in
#: interpolate_and_aggregate.py. Longer runs stay NaN.
DEFAULT_MAX_GAP = 24

#: Basis density for the local fits: one function per 30 minutes of
#: context at 5-minute sampling, which comfortably resolves a ramp
#: without chasing 5-minute noise.
LOCAL_BASIS_PER_SAMPLE = 0.5

# Gap-filling defaults, which are NOT the outlier-cleaning defaults above.
# DEFAULT_IMPUTE_LAMBDA = 1.0 was tuned on unfiltered 5-minute aggregate
# load. Filtered 1-minute irradiance is a different animal: the low-pass
# has already removed the high-frequency content, so the spline only has
# to follow a smooth curve and any extra smoothing penalty hurts.
# Measured on 30 days of 1-minute GHI, filtered at a 30-minute cutoff,
# B-spline MAE against linear interpolation on the same hidden gaps:
#
#     gap length     lambda 0.1        lambda 1.0        linear
#     1 sample       0.011  W/m^2      0.093  W/m^2      0.081 W/m^2
#     5 samples      0.136             0.619             0.493
#     15 samples     1.744             3.810             3.409
#     24 samples     5.477             8.666             8.022
#
# At 0.1 the B-spline wins at every gap length; at 1.0 it loses at every
# one. The default below is therefore 0.1, and it is exposed in the GUI
# because a different sampling rate or cutoff will move the optimum.
DEFAULT_GAPFILL_LAMBDA = 0.1
DEFAULT_GAPFILL_CONTEXT = 60
# Irradiance below this is darkness as far as the fill is concerned.
DEFAULT_DARK_THRESHOLD = 5.0


@dataclass
class CleaningResult:
    original: pd.Series
    cleaned: pd.Series
    flagged: pd.Series          # bool: detected as an outlier
    imputed: pd.Series          # bool: actually replaced by a fitted value
    left_missing: pd.Series     # bool: popped but deliberately not filled
    regions: pd.DataFrame       # one row per contiguous flagged run
    method: str
    window: int
    n_sigmas: float
    lam: float
    context: int
    max_gap: int
    source: str = ""
    holdout: Optional[Dict[str, float]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return int(self.flagged.sum())

    @property
    def n_imputed(self) -> int:
        return int(self.imputed.sum())

    @property
    def n_left_missing(self) -> int:
        return int(self.left_missing.sum())

    def _interval_hours(self) -> float:
        step = pd.Series(self.original.index).diff().median()
        return float(step.total_seconds()) / 3600.0

    def energy_kwh(self) -> Tuple[float, float]:
        """Total energy before and after, in kWh (readings are kW)."""
        h = self._interval_hours()
        return (float(self.original.sum(skipna=True) * h),
                float(self.cleaned.sum(skipna=True) * h))

    def corrections(self) -> pd.DataFrame:
        """Every replaced reading: what it was, what it became."""
        idx = self.imputed[self.imputed].index
        if len(idx) == 0:
            return pd.DataFrame(columns=["timestamp", "original", "imputed", "change"])
        out = pd.DataFrame({
            "timestamp": idx,
            "original": self.original.loc[idx].to_numpy(),
            "imputed": self.cleaned.loc[idx].to_numpy(),
        })
        out["change"] = out["imputed"] - out["original"]
        return out.reindex(out["change"].abs().sort_values(ascending=False).index)

    def report(self) -> str:
        o, c = self.original, self.cleaned
        n = len(o)
        e_before, e_after = self.energy_kwh()
        interval_min = int(round(self._interval_hours() * 60))
        window_hours = self.window * self._interval_hours()

        lines = [
            "Data Cleaning Report",
            "=" * 62,
            f"Source:            {self.source or '(in-memory series)'}",
            f"Series:            {o.name}",
            f"Range:             {o.index[0]}  to  {o.index[-1]}",
            f"Readings:          {n:,} at {interval_min}-minute resolution",
            "",
            "DETECTION",
            "-" * 62,
            f"Method:            {self.method}",
            f"Window:            {self.window} samples ({window_hours:.2f} h, centred)",
            f"Threshold:         {self.n_sigmas:g} sigma",
            f"Flagged:           {self.n_flagged:,} readings "
            f"({100 * self.n_flagged / n:.3f}% of the series)",
            f"Flagged regions:   {len(self.regions):,}"
            + (f"  (longest run {int(self.regions['length'].max())} readings)"
               if len(self.regions) else ""),
            "",
            "IMPUTATION",
            "-" * 62,
            f"Method:            penalised B-spline, lambda={self.lam:g} x gap length, "
            f"context max({self.context}, 4 x gap) samples either side",
            f"Max gap filled:    {self.max_gap} samples "
            f"({self.max_gap * self._interval_hours():.1f} h)",
            f"Imputed:           {self.n_imputed:,} readings",
            f"Left missing:      {self.n_left_missing:,} readings"
            + (" (runs longer than the max gap - a smooth guess over a long "
               "outage would be invented data)" if self.n_left_missing else ""),
        ]

        if self.holdout:
            h = self.holdout
            lines += [
                "",
                "IMPUTATION ACCURACY (hold-out test on this series)",
                "-" * 62,
                f"{h['n']} readings that were NOT flagged were hidden, imputed, and",
                "compared against their true values:",
                f"  B-spline            MAE {h['bspline_mae']:8.2f} kW    "
                f"RMSE {h['bspline_rmse']:8.2f} kW",
                f"  linear interpolation MAE {h['linear_mae']:8.2f} kW    "
                f"RMSE {h['linear_rmse']:8.2f} kW",
                f"  series std for scale     {h['series_std']:8.2f} kW",
            ]

        stats = pd.DataFrame({
            "before": [o.mean(), o.std(), o.min(), o.max(), o.skew(), o.kurtosis()],
            "after": [c.mean(), c.std(), c.min(), c.max(), c.skew(), c.kurtosis()],
        }, index=["mean", "std", "min", "max", "skew", "excess kurtosis"])
        stats["change"] = stats["after"] - stats["before"]

        lines += ["", "EFFECT ON THE SERIES", "-" * 62,
                  f"{'':<18}{'before':>12}{'after':>12}{'change':>12}"]
        for label, row in stats.iterrows():
            lines.append(f"{label:<18}{row['before']:>12.3f}{row['after']:>12.3f}{row['change']:>12.3f}")

        pct = 100 * (e_after - e_before) / e_before if e_before else 0.0
        lines += [
            "",
            f"Total energy:      {e_before:,.1f} kWh  ->  {e_after:,.1f} kWh "
            f"({pct:+.4f}%)",
        ]

        corr = self.corrections()
        if not corr.empty:
            lines += ["", "LARGEST CORRECTIONS", "-" * 62,
                      f"{'timestamp':<22}{'original':>12}{'imputed':>12}{'change':>12}"]
            for _, r in corr.head(15).iterrows():
                lines.append(
                    f"{str(r['timestamp']):<22}{r['original']:>12.2f}"
                    f"{r['imputed']:>12.2f}{r['change']:>+12.2f}"
                )
            if len(corr) > 15:
                lines.append(f"... and {len(corr) - 15} more")

        lines += [
            "",
            "NOTES",
            "-" * 62,
            "- Flagged readings are review candidates, not confirmed faults. A",
            "  detector cannot tell a metering error from a real demand spike;",
            "  it only measures how unusual a reading is against its neighbours.",
            "- Imputed values are model output, not measurements. Anything",
            "  computed from this series afterwards inherits that.",
            "- The original file is never modified.",
        ]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs of True as (start, end) inclusive index positions."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return [(int(r[0]), int(r[-1])) for r in splits]


def _impute_region(values: np.ndarray, index: pd.DatetimeIndex, flagged: np.ndarray,
                   start: int, end: int, context: int, lam: float) -> Optional[np.ndarray]:
    """
    Fit a B-spline to the context around one flagged region and read the
    replacement values off it. Returns None if the local fit is not
    possible (too little surviving context), leaving the caller to record
    the region as unfilled rather than guessing.

    Both the context and the smoothing scale with the length of the hole
    - see the constants at the top of the module for the measurements
    behind that.
    """
    gap_len = end - start + 1
    context = max(context, 4 * gap_len)
    lam = lam * max(1, gap_len)

    lo = max(0, start - context)
    hi = min(len(values), end + 1 + context)

    local = pd.Series(values[lo:hi], index=index[lo:hi])
    # Pop every flagged reading in the window, not just this region -
    # a neighbouring outlier left in would distort the curve.
    local = local.mask(pd.Series(flagged[lo:hi], index=index[lo:hi]))

    n_valid = int(local.notna().sum())
    n_basis = int(max(5, min(n_valid - 4, round(len(local) * LOCAL_BASIS_PER_SAMPLE))))
    if n_valid < n_basis + 2 or n_valid < 12:
        return None

    try:
        curve = fit_curve(local, lam=lam, n_basis=n_basis)
    except (ValueError, RuntimeError):
        return None
    return curve.to_numpy()[start - lo:end + 1 - lo]


def clean_series(
    series: pd.Series,
    method: str = "zscore",
    window: int = 48,
    n_sigmas: float = 3.0,
    lam: float = DEFAULT_IMPUTE_LAMBDA,
    context: int = DEFAULT_CONTEXT,
    max_gap: int = DEFAULT_MAX_GAP,
    source: str = "",
    validate: bool = True,
    seed: int = 0,
) -> CleaningResult:
    """
    Detect outliers over the whole series, pop them, and fill the holes
    from a locally fitted B-spline. The input series is never modified.
    """
    if method not in DETECTORS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(DETECTORS)}")
    if len(series) < 50:
        raise ValueError("Need at least 50 readings to clean a series.")

    name = series.name or "series"
    mask, _, _ = DETECTORS[method](series.to_frame(name), window=window, n_sigmas=n_sigmas)
    flagged = mask[name].to_numpy()

    values = series.to_numpy(dtype=float)
    cleaned = values.copy()
    cleaned[flagged] = np.nan                      # the "pop"

    imputed = np.zeros(len(values), dtype=bool)
    left_missing = np.zeros(len(values), dtype=bool)
    records = []
    notes: List[str] = []

    for start, end in _regions(flagged):
        length = end - start + 1
        if length > max_gap:
            left_missing[start:end + 1] = True
            records.append({"start": series.index[start], "end": series.index[end],
                            "length": length, "imputed": False,
                            "reason": f"run longer than max gap ({max_gap})"})
            continue

        fitted = _impute_region(values, series.index, flagged, start, end, context, lam)
        if fitted is None:
            left_missing[start:end + 1] = True
            records.append({"start": series.index[start], "end": series.index[end],
                            "length": length, "imputed": False,
                            "reason": "not enough clean context for a local fit"})
            continue

        cleaned[start:end + 1] = fitted
        imputed[start:end + 1] = True
        records.append({"start": series.index[start], "end": series.index[end],
                        "length": length, "imputed": True, "reason": ""})

    regions = pd.DataFrame(
        records, columns=["start", "end", "length", "imputed", "reason"]
    )

    if int(left_missing.sum()):
        notes.append(
            f"{int(left_missing.sum())} reading(s) were popped but left empty; see the "
            "regions table for why. Decide explicitly what the forecasting step should "
            "do with them rather than letting them through as silent NaNs."
        )

    holdout = None
    if validate:
        try:
            holdout = holdout_validation(series, flagged, lam=lam, context=context, seed=seed)
        except Exception:  # noqa: BLE001 - validation is a nicety, never fatal
            holdout = None

    return CleaningResult(
        original=series,
        cleaned=pd.Series(cleaned, index=series.index, name=name),
        flagged=pd.Series(flagged, index=series.index),
        imputed=pd.Series(imputed, index=series.index),
        left_missing=pd.Series(left_missing, index=series.index),
        regions=regions, method=method, window=window, n_sigmas=n_sigmas,
        lam=lam, context=context, max_gap=max_gap, source=source,
        holdout=holdout, notes=notes,
    )


@dataclass
class GapFillResult:
    """Outcome of filling holes that were already in the series."""

    original: pd.Series
    filled: pd.Series
    was_missing: pd.Series      # bool: NaN on the way in
    imputed: pd.Series          # bool: actually given a fitted value
    left_missing: pd.Series     # bool: deliberately left as a hole
    regions: pd.DataFrame       # one row per contiguous run of NaN
    lam: float
    context: int
    max_gap: int
    valid_range: Optional[Tuple[float, float]] = None
    unit: str = ""
    source: str = ""
    holdout: Optional[Dict[str, float]] = None
    n_clamped: int = 0
    n_night: int = 0

    @property
    def n_missing(self) -> int:
        return int(self.was_missing.sum())

    @property
    def n_imputed(self) -> int:
        return int(self.imputed.sum())

    @property
    def n_left_missing(self) -> int:
        return int(self.left_missing.sum())

    def _interval_minutes(self) -> float:
        step = pd.Series(self.original.index).diff().median()
        return float(step.total_seconds()) / 60.0

    def corrections(self) -> pd.DataFrame:
        idx = self.imputed[self.imputed].index
        if len(idx) == 0:
            return pd.DataFrame(columns=["timestamp", "filled"])
        return pd.DataFrame({"timestamp": idx, "filled": self.filled.loc[idx].to_numpy()})

    def report(self) -> str:
        o, f = self.original, self.filled
        interval = self._interval_minutes()
        unit = f" {self.unit}" if self.unit else ""
        gap_minutes = self.max_gap * interval

        lines = [
            "GAP FILLING  (penalised B-spline)",
            "=" * 62,
            f"Source:            {self.source or '(in-memory series)'}",
            f"Series:            {o.name}",
            f"Readings:          {len(o):,}   ({interval:g}-minute sampling)",
            "",
            "HOLES FOUND",
            f"  Missing on entry:      {self.n_missing:,}"
            + (f"  ({100 * self.n_missing / len(o):.3f}% of the record)" if len(o) else ""),
            f"  Contiguous runs:       {len(self.regions):,}",
            f"  Filled:                {self.n_imputed:,}",
            f"  Left as holes:         {self.n_left_missing:,}",
        ]

        if len(self.regions):
            lengths = self.regions["length"]
            lines += [
                "",
                "RUN LENGTHS",
                f"  shortest {int(lengths.min())} sample(s)   median {int(lengths.median())}   "
                f"longest {int(lengths.max())} ({int(lengths.max()) * interval:g} minutes)",
                f"  Max gap allowed:       {self.max_gap} samples "
                f"({gap_minutes // 60:.0f}h {gap_minutes % 60:02.0f}m)",
            ]
            too_long = self.regions[~self.regions["imputed"]]
            if len(too_long):
                lines += [
                    "",
                    "LEFT AS HOLES ON PURPOSE",
                    "  A run longer than the max gap has no nearby evidence to fit against.",
                    "  Filling one would invent a stretch of data with nothing behind it, which",
                    "  is worse in a training set than a visible hole.",
                    "",
                ]
                for _, r in too_long.head(10).iterrows():
                    lines.append(f"    {r['start']} → {r['end']}   "
                                 f"{int(r['length']):,} samples   {r['reason']}")
                if len(too_long) > 10:
                    lines.append(f"    … and {len(too_long) - 10} more.")

        if self.n_imputed:
            vals = f[self.imputed]
            lines += [
                "",
                "WHAT WAS PUT IN",
                f"  min    {vals.min():,.3f}{unit}",
                f"  median {vals.median():,.3f}{unit}",
                f"  max    {vals.max():,.3f}{unit}",
            ]
            if self.valid_range is not None:
                lo, hi = self.valid_range
                lines.append(f"  Clamped to the channel's physical range [{lo:g}, {hi:g}]"
                             + (f" — {self.n_clamped:,} fitted value(s) needed it."
                                if self.n_clamped else " — no fitted value needed it."))
            if self.n_night:
                lines.append(
                    f"  {self.n_night:,} hole(s) sat entirely inside a dark period and were set to "
                    "zero rather than splined: irradiance at night is exactly zero, and a curve "
                    "fitted through darkness only adds wiggle.")

        h = self.holdout
        if h:
            better = "better" if h["bspline_mae"] < h["linear_mae"] else "WORSE"
            lines += [
                "",
                "ACCURACY  (measured, not assumed)",
                f"  {h['n']} known-good readings were hidden and re-imputed as if they were holes.",
                f"    B-spline MAE   {h['bspline_mae']:,.3f}{unit}   RMSE {h['bspline_rmse']:,.3f}",
                f"    Linear   MAE   {h['linear_mae']:,.3f}{unit}   RMSE {h['linear_rmse']:,.3f}",
                f"    Series std     {h['series_std']:,.3f}{unit}",
                f"  The B-spline is {better} than linear interpolation on this series.",
            ]
            if h["bspline_mae"] >= h["linear_mae"]:
                lines.append("  Worth knowing before you rely on it — on a series this smooth, "
                             "linear interpolation may be the better choice.")

        lines += [
            "",
            "NOTES",
            "  Only readings that were ALREADY missing are touched; nothing is detected or",
            "  removed here. On solar data the holes are what the filter's valid range screened",
            "  out as physically impossible.",
            "  Each hole is filled from a B-spline fitted to the surrounding context only, so a",
            "  fill reflects local conditions rather than a whole-record average.",
            "  The file on disk is never modified.",
        ]
        return "\n".join(lines)


def _dark_run(values: np.ndarray, index: pd.DatetimeIndex, start: int, end: int,
              context: int, threshold: float) -> bool:
    """
    True when a hole sits entirely inside a dark period.

    Irradiance at night is exactly zero, so a spline fitted through
    darkness contributes nothing but wiggle - and can dip below zero.
    Detected from the surrounding readings rather than a solar-position
    calculation, so it needs no extra dependency and no site coordinates.
    """
    lo = max(0, start - context)
    hi = min(len(values), end + 1 + context)
    neighbours = values[lo:start].tolist() + values[end + 1:hi].tolist()
    finite = [v for v in neighbours if np.isfinite(v)]
    if len(finite) < 4:
        return False
    return bool(np.nanmax(finite) <= threshold)


def impute_missing(
    series: pd.Series,
    lam: float = DEFAULT_GAPFILL_LAMBDA,
    context: int = DEFAULT_GAPFILL_CONTEXT,
    max_gap: int = DEFAULT_MAX_GAP,
    valid_range: Optional[Tuple[float, float]] = None,
    dark_threshold: Optional[float] = None,
    unit: str = "",
    source: str = "",
    validate: bool = True,
    seed: int = 0,
) -> GapFillResult:
    """
    Fill holes that are ALREADY in the series, with no detection step.

    clean_series() detects outliers, pops them, and refills. This does
    only the refilling half, on holes that are already there - which is
    what irradiance needs, because a Z-score cannot see an irradiance
    sensor fault at all (it catches none of them), so the filter's
    physical valid range does the removing and screens the impossible
    readings to NaN. Those NaN are the holes this fills.

    `valid_range` clamps the fitted values: a B-spline through a steep
    sunrise can overshoot below zero, which is not a possible irradiance
    reading. `dark_threshold` sets zero into holes that sit entirely
    inside a dark period, instead of splining through the night.

    The input series is never modified.
    """
    if len(series) < 50:
        raise ValueError("Need at least 50 readings to fill gaps in a series.")

    values = series.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    filled = values.copy()

    imputed = np.zeros(len(values), dtype=bool)
    left = np.zeros(len(values), dtype=bool)
    records = []
    n_clamped = 0
    n_night = 0

    for start, end in _regions(missing):
        length = end - start + 1
        if length > max_gap:
            left[start:end + 1] = True
            records.append({"start": series.index[start], "end": series.index[end],
                            "length": length, "imputed": False,
                            "reason": f"run longer than max gap ({max_gap} samples)"})
            continue

        if dark_threshold is not None and _dark_run(values, series.index, start, end,
                                                    context, dark_threshold):
            filled[start:end + 1] = 0.0
            imputed[start:end + 1] = True
            n_night += 1
            records.append({"start": series.index[start], "end": series.index[end],
                            "length": length, "imputed": True, "reason": "night — set to zero"})
            continue

        fitted = _impute_region(values, series.index, missing, start, end, context, lam)
        if fitted is None:
            left[start:end + 1] = True
            records.append({"start": series.index[start], "end": series.index[end],
                            "length": length, "imputed": False,
                            "reason": "not enough surrounding data for a local fit"})
            continue

        if valid_range is not None:
            lo, hi = valid_range
            outside = int(np.sum((fitted < lo) | (fitted > hi)))
            n_clamped += outside
            fitted = np.clip(fitted, lo, hi)

        filled[start:end + 1] = fitted
        imputed[start:end + 1] = True
        records.append({"start": series.index[start], "end": series.index[end],
                        "length": length, "imputed": True, "reason": ""})

    regions = pd.DataFrame(records, columns=["start", "end", "length", "imputed", "reason"])

    holdout = None
    if validate:
        try:
            holdout = holdout_validation(series, flagged=missing, lam=lam,
                                         context=context, seed=seed)
        except (ValueError, RuntimeError):
            holdout = None

    name = series.name or "series"
    return GapFillResult(
        original=series,
        filled=pd.Series(filled, index=series.index, name=name),
        was_missing=pd.Series(missing, index=series.index, name=name),
        imputed=pd.Series(imputed, index=series.index, name=name),
        left_missing=pd.Series(left, index=series.index, name=name),
        regions=regions, lam=lam, context=context, max_gap=max_gap,
        valid_range=valid_range, unit=unit, source=source,
        holdout=holdout, n_clamped=n_clamped, n_night=n_night,
    )


def holdout_validation(
    series: pd.Series,
    flagged: Optional[np.ndarray] = None,
    n_samples: int = 200,
    lam: float = DEFAULT_IMPUTE_LAMBDA,
    context: int = DEFAULT_CONTEXT,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Measure how accurate the imputation actually is on this series.

    Hides readings that were never flagged, imputes them exactly the way
    a real gap would be imputed, and compares against the values that
    were really there. Linear interpolation is scored on the same hidden
    points as a baseline, so the B-spline's number means something.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    if flagged is None:
        flagged = np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(
        np.isfinite(values) & ~flagged
        & (np.arange(n) > context) & (np.arange(n) < n - context - 1)
    )
    if len(candidates) < 20:
        raise ValueError("Not enough clean interior readings to validate.")
    picks = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)

    bspline_err, linear_err = [], []
    for pos in picks:
        hidden = flagged.copy()
        hidden[pos] = True
        fitted = _impute_region(values, series.index, hidden, pos, pos, context, lam)
        if fitted is None:
            continue
        bspline_err.append(fitted[0] - values[pos])

        lo, hi = max(0, pos - context), min(n, pos + context + 1)
        local = pd.Series(values[lo:hi]).mask(pd.Series(hidden[lo:hi]))
        linear_err.append(float(local.interpolate(limit_direction="both").iloc[pos - lo]) - values[pos])

    if not bspline_err:
        raise ValueError("No hold-out point could be imputed.")

    b = np.asarray(bspline_err)
    l = np.asarray(linear_err)
    return {
        "n": len(b),
        "bspline_mae": float(np.mean(np.abs(b))),
        "bspline_rmse": float(np.sqrt(np.mean(b ** 2))),
        "linear_mae": float(np.mean(np.abs(l))),
        "linear_rmse": float(np.sqrt(np.mean(l ** 2))),
        "series_std": float(np.nanstd(values)),
    }


if __name__ == "__main__":
    import sys
    from data_loader import load_load_profile

    path = sys.argv[1] if len(sys.argv) > 1 else "aggregate_total_load.csv"
    method = sys.argv[2] if len(sys.argv) > 2 else "zscore"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    sigmas = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0

    data = load_load_profile(path)
    s = data.df[data.households[0]]
    result = clean_series(s, method=method, window=window, n_sigmas=sigmas,
                          source=str(data.source_path))
    print(result.report())
