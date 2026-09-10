"""
Per-household interpolation + cross-household aggregation.

Rationale (see progress notes for the full data behind this): individual
household active-power readings are strongly right-skewed and heavy
tailed, which is why point-wise outlier detection (Hampel/Z-score) on a
single household flags a large share of perfectly normal appliance
spikes. Summing across all 66 households diversifies that away - the
aggregate (total feeder/transformer demand) comes out far closer to a
symmetric, near-Gaussian distribution, which is also the actual
forecasting target. So the pipeline order that makes sense is:

    1. interpolate the small NaN holes per household (this script)
    2. sum across households into one total-demand series (this script)
    3. do outlier detection + imputation on THAT series (next script)

Interpolating before summing matters, not just as tidiness: pandas'
default sum() skips NaN, so a timestamp where most households are
missing (e.g. the shared logging outages found during sentinel
cleaning) would otherwise silently collapse to a tiny fake total
instead of a sensible one. This script sums with skipna=False instead,
so any reading that's still missing after interpolation (see max-gap
below) correctly makes the aggregate NaN at that timestamp too, rather
than quietly undercounting it.

Gaps are capped by --max-gap (in samples, default 24 = 2 hours at
5-minute resolution): a run of missing values longer than this is left
as NaN rather than being smoothed over with a long, likely-wrong
straight line. On this dataset the longest gap is 45 minutes, so the
default won't trigger - it's a safeguard for the full Feb-Sep dataset,
where a longer real outage might show up.

Outputs:
  - <input>_interpolated.xlsx: per-household data, same headerless
    layout as always, gaps filled (a drop-in replacement anywhere the
    cleaned file was used, including the Stage 1 GUI).
  - aggregate_total_load.csv: two columns (timestamp, total_active_power_kw),
    the sum across all households at every timestamp - this is the
    series the next stage (aggregate outlier detection + imputation)
    and eventually the forecasting model should work from.
  - a text report: interpolation counts, any gaps too long to fill,
    and the aggregate's distribution shape before/after.

Usage:
    python interpolate_and_aggregate.py
    python interpolate_and_aggregate.py --input Westridge_2002_dupe_cleaned.xlsx --max-gap 24
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from data_loader import load_load_profile

DEFAULT_INPUT = "Westridge_2002_dupe_cleaned.xlsx"
DEFAULT_MAX_GAP = 24  # samples = 2 hours at 5-minute resolution


def interpolate_households(
    df: pd.DataFrame,
    method: str = "linear",
    max_gap: Optional[int] = DEFAULT_MAX_GAP,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Linearly interpolate NaN holes per household (each column is its own
    time series, so this is independent per household - exactly like the
    Hampel filter's per-column rolling windows).

    Gaps longer than `max_gap` samples are deliberately left as NaN -
    pandas' `limit` caps how many consecutive NaNs get filled walking
    forward from the last valid value, so a longer run is only partially
    filled from each side and whatever's left in the middle stays NaN.

    Returns (interpolated_df, still_missing) where `still_missing` is a
    same-shape boolean DataFrame marking anything that remains NaN
    (either an edge that had no valid neighbour on one side, or the
    untouched middle of an over-long gap).
    """
    interpolated = df.interpolate(
        method=method,
        limit=max_gap,
        limit_direction="both",
    )
    still_missing = interpolated.isna()
    return interpolated, still_missing


def aggregate_total(df: pd.DataFrame) -> pd.Series:
    """
    Sum across households at every timestamp. Deliberately skipna=False:
    if any household is still missing at a timestamp, the total is
    unknown too - it should NOT be silently computed from the rest.
    """
    return df.sum(axis=1, skipna=False)


def build_report(
    raw_df: pd.DataFrame,
    interpolated_df: pd.DataFrame,
    still_missing: pd.DataFrame,
    agg: pd.Series,
    naive_agg: pd.Series,
    max_gap: int,
    input_path: Path,
) -> str:
    n_nan_before = int(raw_df.isna().sum().sum())
    n_nan_after = int(still_missing.sum().sum())
    n_filled = n_nan_before - n_nan_after

    per_household_unfilled = still_missing.sum(axis=0)
    unfilled_households = per_household_unfilled[per_household_unfilled > 0].sort_values(ascending=False)

    lines = [
        "Interpolation + Aggregation Report",
        "=" * 40,
        f"Input:                {input_path}",
        f"Max gap filled:       {max_gap} samples ({max_gap * 5} minutes)",
        "",
        f"NaN before interpolation: {n_nan_before:,}",
        f"NaN filled:               {n_filled:,}",
        f"NaN still missing (gap exceeded max-gap or at series edge): {n_nan_after:,}",
    ]
    if not unfilled_households.empty:
        lines.append("Households with unfilled gaps:")
        for h, c in unfilled_households.items():
            lines.append(f"  {h}: {c}")

    lines += [
        "",
        "Aggregate (total demand across all households) distribution shape:",
        f"  mean={agg.mean():.2f} kW  std={agg.std():.2f} kW  "
        f"skew={agg.skew():.3f}  excess kurtosis={agg.kurtosis():.3f}",
        "  (individual households on this dataset run ~1.9-3.1 skew, ~4-10.6 kurtosis -",
        "   aggregating clearly pulls the distribution back toward symmetric/near-Gaussian.)",
        "",
        "Why interpolate before summing - comparison at the timestamps with the most",
        "simultaneously-missing households (naive sum skips missing households and",
        "silently undercounts; interpolate-first fills them in first):",
    ]

    n_missing_per_row = raw_df.isna().sum(axis=1)
    worst = n_missing_per_row[n_missing_per_row > 0].sort_values(ascending=False).head(5)
    for ts, n_missing in worst.items():
        lines.append(
            f"  {ts}: {int(n_missing)} households missing -> "
            f"naive sum={naive_agg.loc[ts]:.1f} kW, interpolate-first sum={agg.loc[ts]:.1f} kW"
        )

    return "\n".join(lines)


def process_file(
    input_path: Path,
    method: str = "linear",
    max_gap: int = DEFAULT_MAX_GAP,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_load_profile(input_path)
    raw_df = data.df

    interpolated_df, still_missing = interpolate_households(raw_df, method, max_gap)
    agg = aggregate_total(interpolated_df)
    naive_agg = raw_df.sum(axis=1, skipna=True)

    report = build_report(raw_df, interpolated_df, still_missing, agg, naive_agg, max_gap, input_path)
    return interpolated_df, agg, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpolate per-household NaN holes, then sum into a total demand series.")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help=f"Input Excel file (default: {DEFAULT_INPUT})")
    p.add_argument("--household-output", default=None, help="Output Excel file for interpolated per-household data (default: <input>_interpolated.xlsx)")
    p.add_argument("--aggregate-output", default="aggregate_total_load.csv", help="Output CSV for the aggregate series (default: aggregate_total_load.csv)")
    p.add_argument("--method", default="linear", help="pandas interpolation method (default: linear)")
    p.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP, help=f"Max consecutive samples to interpolate over (default: {DEFAULT_MAX_GAP})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    household_output = (
        Path(args.household_output) if args.household_output
        else input_path.with_name(f"{input_path.stem}_interpolated{input_path.suffix}")
    )
    aggregate_output = Path(args.aggregate_output)

    print(f"Loading {input_path} ...")
    interpolated_df, agg, report = process_file(input_path, args.method, args.max_gap)

    print()
    print(report)

    out_df = interpolated_df.reset_index()
    out_df.to_excel(household_output, header=False, index=False, engine="openpyxl")

    agg_df = agg.rename("total_active_power_kw").reset_index()
    agg_df.columns = ["timestamp", "total_active_power_kw"]
    agg_df.to_csv(aggregate_output, index=False)

    report_path = aggregate_output.with_name(f"{aggregate_output.stem}_report.txt")
    report_path.write_text(report)

    print()
    print(f"Interpolated per-household file written to: {household_output}")
    print(f"Aggregate total-load series written to:      {aggregate_output}")
    print(f"Report written to:                           {report_path}")


if __name__ == "__main__":
    main()
