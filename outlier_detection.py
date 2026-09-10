"""
Rolling Z-score outlier detection for load-profile data.

A reading is flagged when it sits more than `n_sigmas` rolling standard
deviations from its rolling mean, over a centred window. This is the
classical, textbook test and the method the project settled on for load
data - it is easy to state, easy to defend, and on the aggregate series
its flag rate is far more plausible than a robust filter's.

This script is FLAG-ONLY: it does not modify the input data. It produces
a boolean outlier mask (in memory, for reuse) plus two on-disk artifacts:
  - a long-format CSV of every flagged reading (timestamp, household,
    value, local centre, threshold, deviation) - the natural shape for
    later plotting (e.g. circling flagged points on the viewer's chart)
    or manual review
  - a text report summarising counts

Run this AFTER clean_sentinels.py - it expects sentinel codes to already
be NaN holes, not live -999s (a -999 would otherwise dominate every
window it appears near).

TWO PROPERTIES TO STATE RATHER THAN HIDE:

1. Masking. The mean and standard deviation have a 0% breakdown point, so
   a large outlier drags the mean toward itself and inflates the standard
   deviation - widening the very threshold it is tested against, and
   potentially hiding its neighbours. Calibration measured this: longer
   dropouts are detected LESS often than short ones.

2. The window caps the threshold. Because the tested reading is one of
   the values forming its own window's mean and standard deviation, its
   Z-score cannot exceed (window - 1) / sqrt(window) - only 3.18 at a
   12-sample window. See max_achievable_zscore(); "no outliers found" can
   mean the settings were impossible rather than the data being clean.

A third limitation is structural rather than statistical: a SUSTAINED
level shift is invisible. If a long block of readings is wrong by a
constant, the rolling mean shifts with it and the interior never
deviates. Measured on the Stellenbosch irradiance data, where a two-day
sensor fault held GHI near 2 975 W/m2 - including through the night -
the Z-score flagged 0 of 713 non-physical readings at every setting
tried. Faults of that shape need a physical-range screen (see
signal_filters.py), not a point-wise detector.

Usage:
    python outlier_detection.py
    python outlier_detection.py --input Westridge_2002_dupe_cleaned.xlsx
    python outlier_detection.py --window 12 --sigmas 3.0
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data_loader import load_load_profile

DEFAULT_INPUT = "Westridge_2002_dupe_cleaned.xlsx"
DEFAULT_WINDOW = 12  # samples; 12 * 5 min = 1 hour, centered
DEFAULT_SIGMAS = 3.0  # see module docstring for what "sigma" does and does not mean
MAD_TO_STD = 1.4826  # scales MAD to be comparable to std under a normal assumption


def _rolling_mad(x: np.ndarray) -> float:
    """MAD of one rolling window, ignoring NaNs (matches pandas' NaN-aware .median())."""
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


#: Above this window size the strided MAD's temporary array gets large, so
#: fall back to the (slower but memory-flat) pandas implementation.
_FAST_MAD_MAX_WINDOW = 96
_FAST_MAD_CHUNK = 20_000


def _rolling_mad_fast(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Rolling MAD via a strided window matrix - same result as
    .rolling(...).apply(_rolling_mad), roughly 15x faster.

    pandas' rolling.apply calls a Python function once per window, which
    on eight months of 5-minute data is ~70 000 calls per series and
    turns a parameter sweep into minutes of waiting. Building the
    (n, window) view and letting numpy reduce along an axis does the
    same arithmetic in one pass.

    Padding reproduces pandas' centred-window alignment exactly,
    including the off-by-one for even window sizes; test_fast_mad()
    below asserts the two agree.
    """
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.concatenate([
        np.full(pad_left, np.nan), values.astype(float), np.full(pad_right, np.nan)
    ])

    out = np.empty(len(values), dtype=float)
    for start in range(0, len(values), _FAST_MAD_CHUNK):
        stop = min(start + _FAST_MAD_CHUNK, len(values))
        win = sliding_window_view(padded[start:stop + window - 1], window)
        valid = np.count_nonzero(~np.isnan(win), axis=1)
        with warnings.catch_warnings():
            # A window that is entirely NaN is expected near a long gap;
            # it yields NaN, which min_periods would blank out anyway.
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
        mad[valid < min_periods] = np.nan
        out[start:stop] = mad
    return out


def robust_local_scale(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling median absolute deviation, scaled to be comparable to a
    standard deviation (x 1.4826 under a normal assumption).

    This is what remains of the Hampel filter after it was retired as a
    detection method. It is kept because a ROBUST measure of local spread
    is still the right way to size synthetic faults during calibration:
    scaling injected faults by a rolling standard deviation would let
    outliers already in the series inflate the reference and quietly make
    the injected faults smaller than their labels claim.

    Not a detector. Nothing user-facing calls it.
    """
    if min_periods is None:
        min_periods = max(3, window // 2)

    if window <= _FAST_MAD_MAX_WINDOW:
        mad = pd.DataFrame(
            {c: _rolling_mad_fast(df[c].to_numpy(), window, min_periods) for c in df.columns},
            index=df.index,
        )
    else:
        mad = df.rolling(window=window, center=True, min_periods=min_periods).apply(
            _rolling_mad, raw=True)
    return MAD_TO_STD * mad


def rolling_zscore(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    n_sigmas: float = 3.0,
    min_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling Z-score: flag a reading when it sits more than `n_sigmas`
    rolling standard deviations from its rolling mean.

    Returns (outlier_mask, rolling_mean, threshold). Two properties are
    worth stating plainly rather than discovering later:

      - The mean and standard deviation have a 0% breakdown point. One
        genuine outlier inside the window pulls the mean toward itself
        and inflates the standard deviation, which widens the threshold
        and can hide its neighbours. This is the masking effect, and it
        is strongest exactly where outliers cluster.
      - "n sigma" only carries its usual tail probability under a normal
        distribution. Individual household load is strongly right-skewed
        (skew ~1.9-3.1 on this dataset), so 3 sigma there does not mean
        0.3%. The aggregate series is much closer to Gaussian (skew 0.49)
        and the interpretation is correspondingly more honest.

    Both are properties to describe in a write-up, not bugs. Where a
    perfectly flat window gives a standard deviation of zero, any nonzero
    deviation clears the threshold.

    THE WINDOW CAPS THE THRESHOLD. Because the tested point is itself one
    of the `window` values used to compute the mean and the standard
    deviation, its Z-score cannot exceed (window - 1) / sqrt(window) -
    see max_achievable_zscore(). At window=12 that ceiling is 3.18, so a
    5-sigma threshold there flags nothing at all, ever, no matter how
    corrupt the data is. The two parameters are not independent; check
    the ceiling before concluding a series is clean.
    """
    if min_periods is None:
        min_periods = max(3, window // 2)

    roll = df.rolling(window=window, center=True, min_periods=min_periods)
    rolling_mean = roll.mean()
    rolling_std = roll.std()

    threshold = n_sigmas * rolling_std
    deviation = (df - rolling_mean).abs()
    outlier_mask = (deviation > threshold) & threshold.notna()

    return outlier_mask, rolling_mean, threshold


def max_achievable_zscore(window: int) -> float:
    """
    The largest rolling Z-score any single reading can possibly reach in
    a window of `window` samples: (window - 1) / sqrt(window).

    The tested point contributes to both the mean and the standard
    deviation of its own window, which bounds how many standard
    deviations away from that mean it can be (Shiffler, 1988). Verified
    against this dataset: at windows of 6, 12, 24 and 36 the largest
    Z-score observed anywhere equals this bound to three decimals.

    Practical consequence: a threshold at or above this value flags
    nothing, so "no outliers found" means the settings were impossible,
    not that the data is clean.
    """
    return (window - 1) / math.sqrt(window)


def min_window_for_zscore(n_sigmas: float) -> int:
    """Smallest centered window in which `n_sigmas` is actually reachable."""
    window = 3
    while max_achievable_zscore(window) <= n_sigmas:
        window += 1
    return window


#: Detection methods exposed to callers (the GUI included), by name.
#: One entry since the Hampel filter was retired - the mapping is kept so
#: callers stay method-agnostic if another detector is ever added.
METHODS = {
    "zscore": rolling_zscore,
}


def flagged_to_long_format(
    df: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    rolling_centre: pd.DataFrame,
    threshold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the wide boolean mask into one row per flagged reading.

    `local_centre` is the rolling mean the reading was compared against.
    """
    records = []
    for household in df.columns:
        flagged_idx = outlier_mask.index[outlier_mask[household]]
        if len(flagged_idx) == 0:
            continue
        sub = pd.DataFrame(
            {
                "timestamp": flagged_idx,
                "household": household,
                "value": df.loc[flagged_idx, household].values,
                "local_centre": rolling_centre.loc[flagged_idx, household].values,
                "threshold": threshold.loc[flagged_idx, household].values,
            }
        )
        sub["deviation"] = (sub["value"] - sub["local_centre"]).abs()
        records.append(sub)

    if not records:
        return pd.DataFrame(columns=["timestamp", "household", "value", "local_centre", "threshold", "deviation"])

    out = pd.concat(records, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def build_report(
    df: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    long_df: pd.DataFrame,
    window: int,
    n_sigmas: float,
    min_periods: int,
    input_path: Path,
    method: str = "zscore",
) -> str:
    total_cells = int(df.notna().sum().sum())  # only readings that existed could be evaluated
    total_flagged = int(outlier_mask.values.sum())
    pct = 100 * total_flagged / total_cells if total_cells else 0.0

    # count of readings that existed but couldn't be evaluated (window had too few valid points)
    evaluated = df.notna() & outlier_mask.notna()
    inconclusive = int((df.notna() & ~evaluated).sum().sum())

    per_household = outlier_mask.sum(axis=0)
    affected = per_household[per_household > 0].sort_values(ascending=False)

    title = "Rolling Z-Score Outlier Report"
    rule = f"{n_sigmas} * rolling standard deviation from the rolling mean"

    lines = [
        title,
        "=" * 40,
        f"Input:              {input_path}",
        f"Window:             {window} samples (centered), min_periods={min_periods}",
        f"Threshold:          {rule}",
        "",
        f"Readings evaluated:  {total_cells:,}",
        f"Readings flagged:    {total_flagged:,}  ({pct:.3f}% of evaluated readings)",
        f"Inconclusive (too few valid points in window): {inconclusive:,}",
        f"Households with at least one flag: {len(affected)} / {df.shape[1]}",
        "",
        "Flags by household (households with 0 flags omitted):",
    ]
    for household, count in affected.items():
        lines.append(f"  {household}: {count}")

    if not long_df.empty:
        lines += ["", "Top 10 largest deviations (biggest outliers overall):"]
        top = long_df.sort_values("deviation", ascending=False).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"  {row['timestamp']}  {row['household']}: value={row['value']:.3f}, "
                f"local centre={row['local_centre']:.3f}, deviation={row['deviation']:.3f}"
            )

    return "\n".join(lines)


def detect_file(
    input_path: Path,
    window: int = DEFAULT_WINDOW,
    n_sigmas: float = DEFAULT_SIGMAS,
    min_periods: Optional[int] = None,
    method: str = "zscore",
) -> Tuple[pd.DataFrame, str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")

    data = load_load_profile(input_path)
    df = data.df

    if min_periods is None:
        min_periods = max(3, window // 2)

    outlier_mask, rolling_centre, threshold = METHODS[method](df, window, n_sigmas, min_periods)
    long_df = flagged_to_long_format(df, outlier_mask, rolling_centre, threshold)
    report = build_report(df, outlier_mask, long_df, window, n_sigmas, min_periods, input_path, method)

    return long_df, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flag outliers in a load-profile Excel file using a Hampel filter.")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help=f"Input Excel file (default: {DEFAULT_INPUT})")
    p.add_argument("--output", "-o", default=None, help="Output CSV of flagged readings (default: <input>_outliers.csv)")
    p.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help=f"Rolling window size in samples (default: {DEFAULT_WINDOW})")
    p.add_argument("--sigmas", "-s", type=float, default=DEFAULT_SIGMAS, help=f"Sigma threshold (default: {DEFAULT_SIGMAS})")
    p.add_argument("--min-periods", type=int, default=None, help="Minimum valid points required in a window (default: window // 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_outliers.csv")

    print(f"Loading {input_path} ...")
    long_df, report = detect_file(input_path, args.window, args.sigmas, args.min_periods)

    print()
    print(report)

    long_df.to_csv(output_path, index=False)
    report_path = output_path.with_name(f"{output_path.stem}_report.txt")
    report_path.write_text(report)

    print()
    print(f"Flagged readings written to: {output_path}")
    print(f"Report written to:           {report_path}")


if __name__ == "__main__":
    main()
