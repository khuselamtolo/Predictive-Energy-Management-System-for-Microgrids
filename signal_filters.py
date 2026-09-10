"""
Zero-phase low-pass filtering for irradiance and other fast-sampled series.

Built for the Stellenbosch GHI record (1-minute, Feb-Sep 2023), where
minute-to-minute swings of several hundred W/m2 make the distribution
strongly non-Gaussian. Those swings are broken cloud passing over the
pyranometer: real physics, not measurement error. Removing them is still
defensible when the goal is PV *plant* output rather than point
irradiance, because a spatially distributed array does its own smoothing
- an array does not see a cloud edge everywhere at once. That is the
justification for low-passing here, and it should be stated as such
rather than described as noise removal.

Two things this module deliberately keeps separate:

PHYSICAL SCREENING comes first, and is not optional
---------------------------------------------------
A linear filter cannot remove an impossible reading; it spreads it over
the neighbouring window. On this dataset a sensor fault held GHI near
2 975 W/m2 for parts of 8-9 May 2023 - including at 03:00, when the true
value is zero - across 713 readings. Feeding that to a Butterworth would
smear a two-day fault into the days around it.

Nor will the outlier detector catch it: a sustained level shift moves the
rolling mean with it, so the interior never deviates. Measured: the
rolling Z-score flags 0 of those 713 readings at every window and
threshold tried. The only thing that catches a fault of that shape is
knowing what the instrument physically cannot report, which is why
`valid_range` exists. Screened readings become NaN and stay NaN - a
two-day outage should not be invented.

ZERO PHASE, because a lag would misalign the data
-------------------------------------------------
`scipy.signal.filtfilt` runs the filter forwards and then backwards, so
the result has no phase shift. A causal filter would delay irradiance by
half its window, and PV output computed from a lagged GHI would sit
several minutes away from the load it is meant to be matched against.
The cost is that the filter is non-causal - it uses future samples - so
it is a data-conditioning step for a historical record, not something
that can run in real time.

Usage:
    from signal_filters import lowpass
    out = lowpass(ghi, cutoff_minutes=30, valid_range=(0, 1400))
    print(out.report())
    out.filtered.to_csv("ghi_filtered.csv")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - surfaced to the user by the GUI
    HAS_SCIPY = False

#: Cutoff, in minutes. 30 rather than something shorter because the
#: variance was measured rather than guessed. Welch spectrum of daytime
#: GHI, detrended by a 2-hour rolling mean, gives the share of
#: fluctuation variance at periods SHORTER than each timescale:
#:
#:     < 2 min    0.0%      < 15 min   23.7%
#:     < 5 min    7.3%      < 30 min   38.2%
#:     < 10 min  16.8%      < 60 min   56.8%
#:
#: A 5-minute cutoff can therefore only ever touch about 7% of the
#: fluctuation - which is why it looks like it is barely doing anything.
#: Cloud shadows are not high-frequency: a shadow takes minutes to tens
#: of minutes to cross, so the variability sits at 10-60 minute periods
#: and the cutoff has to reach into that band to remove it.
#:
#: Measured effect on the full series (order 3):
#:
#:   cutoff   d/dt std   d/dt kurtosis   peak kept   negative ringing
#:      raw      35.60           100.5      100.0%          -
#:    5 min      19.77            58.7       99.6%          0
#:   15 min       8.03            36.6       98.8%          1
#:   30 min       4.51            26.1       97.5%          1
#:   60 min       2.59            12.5       95.5%      1 976
#:
#: 30 minutes is the last setting that still behaves: an eightfold cut in
#: minute-to-minute variation, 97.5% of the peak kept, and effectively no
#: ringing. Past that the filter starts overshooting at the sunrise and
#: sunset edges - 1 976 negative readings at 60 minutes - because it is
#: being asked to smooth across the day/night discontinuity. Daily energy
#: is preserved to better than 0.001% at every setting.
DEFAULT_CUTOFF_MINUTES = 30.0

#: Filter order barely matters here: at every cutoff tried, order 5 moved
#: the statistics by ~1-3% against order 3. The cutoff is the lever, not
#: the roll-off steepness, so this stays at a conservative 3.
DEFAULT_ORDER = 3

#: Sensible ceiling for surface global horizontal irradiance. Clear-sky
#: GHI peaks near 1 000-1 100 W/m2; edge-of-cloud enhancement can push
#: briefly to ~1 400. Anything above that is the instrument, not the sky.
GHI_VALID_RANGE = (0.0, 1400.0)

#: Generous physical bounds for ambient air temperature at this site
#: (observed range on the Stellenbosch record: 2.36 to 36.57 C).
TEMP_VALID_RANGE = (-20.0, 60.0)

#: Temperature is far smoother than irradiance to begin with - 0.092 C
#: minute-to-minute standard deviation against GHI's 35.6 W/m2 - so it
#: needs less smoothing. Its increments are still heavy-tailed
#: (kurtosis 20.6, with single-minute jumps up to 3.23 C that ambient air
#: does not actually do), which is sensor noise worth removing. 15
#: minutes roughly halves the minute-to-minute variation and is also
#: defensible physically: PV cell temperature lags ambient by several
#: minutes of thermal inertia, so sub-15-minute ambient detail does not
#: reach the cell anyway.
TEMP_CUTOFF_MINUTES = 15.0


@dataclass
class FilterResult:
    original: pd.Series
    filtered: pd.Series
    screened: pd.Series          # bool: rejected by valid_range before filtering
    cutoff_minutes: float
    order: int
    valid_range: Optional[Tuple[float, float]]
    sample_minutes: float
    n_reindexed: int = 0         # rows added to make the grid uniform
    notes: List[str] = field(default_factory=list)

    @property
    def n_screened(self) -> int:
        return int(self.screened.sum())

    def report(self) -> str:
        o, f = self.original, self.filtered
        both = o.notna() & f.notna()
        resid = (o[both] - f[both])
        lines = [
            "Low-Pass Filter Report",
            "=" * 62,
            f"Series:            {o.name}",
            f"Range:             {o.index[0]}  to  {o.index[-1]}",
            f"Readings:          {len(o):,} at {self.sample_minutes:g}-minute sampling",
            "",
            "PHYSICAL SCREENING",
            "-" * 62,
        ]
        if self.valid_range is None:
            lines.append("No valid range set - nothing screened. If this series has hard")
            lines.append("physical limits, set them: a filter cannot remove an impossible")
            lines.append("reading, it only spreads it over its neighbours.")
        else:
            lo, hi = self.valid_range
            lines += [
                f"Valid range:       {lo:g} to {hi:g}",
                f"Screened:          {self.n_screened:,} readings "
                f"({100 * self.n_screened / len(o):.3f}%) set to NaN before filtering",
            ]
            if self.n_screened:
                bad = o[self.screened]
                lines.append(f"  worst value:     {bad.abs().max():,.1f}")
                days = sorted({str(d) for d in bad.index.normalize().date})
                lines.append(f"  affected dates:  {', '.join(days[:6])}"
                             + (f" (+{len(days) - 6} more)" if len(days) > 6 else ""))

        lines += [
            "",
            "FILTER",
            "-" * 62,
            f"Type:              Butterworth order {self.order}, zero-phase (filtfilt)",
            f"Cutoff:            {self.cutoff_minutes:g} minutes "
            f"({1000 / (self.cutoff_minutes * 60):.3f} mHz)",
            "                   Cloud variability sits at 10-60 minute periods; a cutoff",
            "                   shorter than that reaches very little of it. See the",
            "                   module header for the measured variance-by-timescale.",
            "Phase:             zero - forwards and backwards, so nothing is time-shifted",
            f"Still NaN after:   {int(f.isna().sum()):,} readings",
        ]

        if len(resid):
            lines += [
                "",
                "WHAT WAS REMOVED",
                "-" * 62,
                f"Residual (original - filtered) std: {resid.std():,.2f}",
                f"Largest single removal:             {resid.abs().max():,.2f}",
            ]
            # Diff the time-aligned series and THEN select, rather than
            # compacting first: dropping the screened readings and
            # diffing what is left manufactures a huge jump across every
            # gap. That inflated the reported kurtosis from 26 to 1 118.
            consecutive = both & both.shift(1, fill_value=False)
            d_before = o.diff()[consecutive]
            d_after = f.diff()[consecutive]
            lines += [
                "",
                "SAMPLE-TO-SAMPLE CHANGE",
                "-" * 62,
                "This is where the non-Gaussian behaviour actually lives. The level",
                "distribution is dominated by the daily cycle; the heavy tails are in",
                "the increments, and they are what the filter targets.",
                f"{'':<12}{'before':>12}{'after':>12}",
                f"{'std':<12}{d_before.std():>12.3f}{d_after.std():>12.3f}",
                f"{'skew':<12}{d_before.skew():>12.3f}{d_after.skew():>12.3f}",
                f"{'kurtosis':<12}{d_before.kurtosis():>12.3f}{d_after.kurtosis():>12.3f}",
                "",
                "LEVEL DISTRIBUTION",
                "-" * 62,
                f"{'':<12}{'before':>12}{'after':>12}",
                f"{'mean':<12}{o[both].mean():>12.3f}{f[both].mean():>12.3f}",
                f"{'std':<12}{o[both].std():>12.3f}{f[both].std():>12.3f}",
                f"{'skew':<12}{o[both].skew():>12.3f}{f[both].skew():>12.3f}",
                f"{'kurtosis':<12}{o[both].kurtosis():>12.3f}{f[both].kurtosis():>12.3f}",
            ]

        # Does the filter push local peaks and troughs by the same amount?
        # For a one-sided signal like GHI - where cloud takes irradiance
        # DOWN far more often and far further than it pushes it up - the
        # answer matters, and it is the thing a symmetric filter is most
        # often accused of getting wrong.
        if len(resid) > 1000:
            local = o[both].rolling(61, center=True, min_periods=20).median()
            active = local.notna() & (local.abs() > 1e-9)
            diff = (f[both] - o[both])[active]
            above = (o[both] > local)[active]
            if int(above.sum()) > 50 and int((~above).sum()) > 50:
                lines += [
                    "",
                    "CONDITIONAL EFFECT (is the smoothing even-handed?)",
                    "-" * 62,
                    f"Local peaks pulled down by:   {diff[above].mean():>10.3f}",
                    f"Local troughs filled up by:   {diff[~above].mean():>10.3f}",
                    f"Net change in the mean:       {diff.mean():>10.4f}",
                    "",
                    "A unity-gain filter cannot bias the mean, and does not - but it",
                    "does move value between peaks and troughs, and the two columns",
                    "above are rarely equal. On irradiance that is expected rather",
                    "than wrong: cloud is one-sided (it shades far more than it",
                    "enhances), so there is more trough than peak to redistribute.",
                    "It is also what a real PV array does, since a cloud edge reaches",
                    "part of the array before the rest. Treat it as the plant-",
                    "smoothing approximation working, not as an artefact - but quote",
                    "these numbers rather than hiding them, and use a shorter cutoff",
                    "if point-irradiance fidelity matters more than plant realism.",
                ]

        lines += ["", "NOTES", "-" * 62]
        lines += [f"- {n}" for n in self.notes]
        lines += [
            "- The fast variation removed here is largely real (cloud transients),",
            "  not measurement error. Filtering is justified when the target is PV",
            "  plant output, which is spatially smoothed anyway - not as a claim",
            "  that the raw readings were wrong.",
            "- filtfilt is non-causal: it uses future samples. Fine for conditioning",
            "  a historical record, unusable in a real-time pipeline.",
            "- The original file is never modified.",
        ]
        return "\n".join(lines)


def _uniform_grid(series: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Put the series on a strictly uniform time grid.

    A Butterworth filter is defined against a fixed sample rate, so an
    irregular index has to be regularised first. Missing timestamps are
    inserted as NaN rather than silently ignored - the Stellenbosch
    record has one such hole, an hour long.
    """
    deltas = pd.Series(series.index).diff().dropna()
    step = deltas.median()
    sample_minutes = float(step.total_seconds()) / 60.0
    if (deltas == step).all():
        return series, sample_minutes, 0

    grid = pd.date_range(series.index[0], series.index[-1], freq=step)
    reindexed = series.reindex(grid)
    return reindexed, sample_minutes, int(len(grid) - len(series))


def lowpass(
    series: pd.Series,
    cutoff_minutes: float = DEFAULT_CUTOFF_MINUTES,
    order: int = DEFAULT_ORDER,
    valid_range: Optional[Tuple[float, float]] = None,
    clamp_output: bool = True,
) -> FilterResult:
    """
    Zero-phase Butterworth low-pass filter.

    Fluctuations faster than `cutoff_minutes` are attenuated; slower
    structure passes. `valid_range` screens physically impossible
    readings to NaN *before* filtering (they stay NaN afterwards).

    Screened points and pre-existing NaNs are interpolated across purely
    so the filter has a continuous signal to run on, then restored to
    NaN. Nothing is imputed here - use the cleaning step for that, which
    reports what it filled and refuses to invent long stretches.

    With `clamp_output`, the filtered result is clipped back into
    `valid_range`. Butterworth ringing around a sharp edge - sunrise, or
    the boundary of a screened block - can otherwise push GHI slightly
    negative, which is a filter artefact rather than data.
    """
    if not HAS_SCIPY:
        raise RuntimeError("The low-pass filter needs SciPy. Install it with:  pip install scipy")
    if cutoff_minutes <= 0:
        raise ValueError("Cutoff must be a positive number of minutes.")
    if order < 1 or order > 10:
        raise ValueError("Filter order must be between 1 and 10.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must be indexed by timestamp")

    notes: List[str] = []
    work, sample_minutes, n_reindexed = _uniform_grid(series)
    if n_reindexed:
        notes.append(
            f"The index was not uniform; {n_reindexed:,} missing timestamp(s) were inserted "
            f"as NaN to give the filter a fixed sample rate."
        )

    if cutoff_minutes <= 2 * sample_minutes:
        raise ValueError(
            f"A {cutoff_minutes:g}-minute cutoff is at or below the Nyquist limit for "
            f"{sample_minutes:g}-minute sampling ({2 * sample_minutes:g} minutes). "
            "Choose a longer cutoff."
        )

    values = work.to_numpy(dtype=float)
    screened = np.zeros(len(values), dtype=bool)
    if valid_range is not None:
        lo, hi = valid_range
        screened = np.isfinite(values) & ((values < lo) | (values > hi))
        values = np.where(screened, np.nan, values)

    missing = ~np.isfinite(values)
    if missing.all():
        raise ValueError("Nothing left to filter after screening.")

    # Interpolate only so filtfilt has a continuous signal; restored below.
    filled = pd.Series(values, index=work.index).interpolate(
        limit_direction="both").to_numpy()

    fs = 1.0 / (sample_minutes * 60.0)          # Hz
    cutoff_hz = 1.0 / (cutoff_minutes * 60.0)
    wn = cutoff_hz / (fs / 2.0)
    b, a = butter(order, wn, btype="low")

    padlen = 3 * max(len(a), len(b))
    if len(filled) <= padlen:
        raise ValueError(f"Series too short for an order-{order} filter ({len(filled)} readings).")
    out = filtfilt(b, a, filled)

    if clamp_output and valid_range is not None:
        out = np.clip(out, valid_range[0], valid_range[1])

    out[missing] = np.nan                        # never invent the screened readings

    filtered = pd.Series(out, index=work.index, name=work.name)
    if int(screened.sum()):
        notes.append(
            f"{int(screened.sum()):,} screened reading(s) are NaN in the output. They are a "
            "sensor fault, not weather - decide explicitly what the downstream model does "
            "with them rather than letting them through."
        )

    return FilterResult(
        original=work,
        filtered=filtered,
        screened=pd.Series(screened, index=work.index),
        cutoff_minutes=cutoff_minutes,
        order=order,
        valid_range=valid_range,
        sample_minutes=sample_minutes,
        n_reindexed=n_reindexed,
        notes=notes,
    )


if __name__ == "__main__":
    import sys
    from data_loader import load_load_profile

    path = sys.argv[1] if len(sys.argv) > 1 else "Stellies_2023.xlsx"
    cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CUTOFF_MINUTES
    data = load_load_profile(path)
    s = data.df[data.households[0]]
    rng = GHI_VALID_RANGE if "ghi" in str(s.name).lower() else None
    print(lowpass(s, cutoff_minutes=cutoff, valid_range=rng).report())
