"""
Sentinel value cleaning for load-profile data.

The Westridge export uses -999 as an error/"no reading" code instead of
leaving the cell blank. Left as -999, a single bad reading drags the
y-axis of any plot down to -999 and swamps everything else, and it
would corrupt any statistic (mean, std, min) computed over the column.

This script replaces known sentinel codes with NaN, turning them into
genuine "holes" in the data - the correct representation for missing
values in a pandas/numpy pipeline (matplotlib breaks the line across a
NaN instead of plotting a nonsense value; pandas' isna()/fillna()/
interpolate() and every downstream ML library treat NaN as missing
natively). This script does NOT fill those holes back in - that's a
separate, deliberate imputation step later in the pipeline. For now we
just want the holes to exist and be visible.

Run this once on a raw import before visualising it, assessing data
quality, or doing anything else with it. It never overwrites the
input file - it writes a new "*_cleaned.xlsx" file plus a plain-text
report of exactly what was changed, so the raw data always stays
intact and every cleaning run leaves an audit trail.

Usage:
    python clean_sentinels.py
    python clean_sentinels.py --input raw.xlsx --output cleaned.xlsx
    python clean_sentinels.py --sentinel -999 --sentinel -9999   # multiple codes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from data_loader import load_load_profile

DEFAULT_SENTINELS: Tuple[float, ...] = (-999.0,)
DEFAULT_INPUT = "Westridge_2002_dupe.xlsx"


def replace_sentinels(
    df: pd.DataFrame,
    sentinels: Sequence[float] = DEFAULT_SENTINELS,
    tolerance: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace sentinel error codes with NaN.

    Returns (cleaned_df, hits): `hits` is a same-shape boolean DataFrame,
    True wherever a sentinel was found and replaced - kept separate from
    the cleaned data so it can be reported on and, later, reused as the
    "these were injected, not originally missing" mask for imputation.
    """
    hits = pd.DataFrame(False, index=df.index, columns=df.columns)
    for code in sentinels:
        hits |= (df - code).abs() < tolerance
    cleaned = df.mask(hits)
    return cleaned, hits


def build_report(
    df: pd.DataFrame,
    cleaned: pd.DataFrame,
    hits: pd.DataFrame,
    sentinels: Sequence[float],
    input_path: Path,
    output_path: Path,
) -> str:
    total_cells = df.shape[0] * df.shape[1]
    total_hits = int(hits.values.sum())
    pct = 100 * total_hits / total_cells if total_cells else 0.0

    per_household = hits.sum(axis=0)
    affected = per_household[per_household > 0].sort_values(ascending=False)

    per_timestamp = hits.sum(axis=1)
    worst_timestamps = per_timestamp[per_timestamp > 0].sort_values(ascending=False).head(10)

    nan_before = int(df.isna().sum().sum())
    nan_after = int(cleaned.isna().sum().sum())

    lines = [
        "Sentinel Cleaning Report",
        "=" * 40,
        f"Input:            {input_path}",
        f"Output:           {output_path}",
        f"Sentinel value(s): {list(sentinels)}",
        "",
        f"Total readings:            {total_cells:,}",
        f"Sentinel values replaced:  {total_hits:,}  ({pct:.3f}% of all readings)",
        f"Households affected:       {len(affected)} / {df.shape[1]}",
        f"NaN count before cleaning: {nan_before:,}",
        f"NaN count after cleaning:  {nan_after:,}",
        "",
        "Sentinel count by household (households with 0 hits omitted):",
    ]
    for household, count in affected.items():
        lines.append(f"  {household}: {count}")

    if not worst_timestamps.empty:
        lines += [
            "",
            "Timestamps with the most simultaneous sentinel hits",
            "(a cluster here suggests a shared logging/communication outage,",
            " not 66 independent sensor faults):",
        ]
        for ts, count in worst_timestamps.items():
            lines.append(f"  {ts}: {count} households")

    return "\n".join(lines)


def clean_file(
    input_path: Path,
    output_path: Path,
    sentinels: Sequence[float] = DEFAULT_SENTINELS,
    tolerance: float = 1e-6,
) -> str:
    """Load, clean, and write out one load-profile file. Returns the report text."""
    data = load_load_profile(input_path)
    df = data.df

    cleaned, hits = replace_sentinels(df, sentinels, tolerance)
    report = build_report(df, cleaned, hits, sentinels, input_path, output_path)

    # Write back out in the same headerless layout (timestamp + household
    # columns) the loader expects, so the cleaned file is a drop-in
    # replacement anywhere the raw file was used - including the GUI.
    out_df = cleaned.reset_index()
    out_df.to_excel(output_path, header=False, index=False, engine="openpyxl")

    report_path = output_path.with_name(f"{output_path.stem}_report.txt")
    report_path.write_text(report)

    return report, report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace sentinel error codes (e.g. -999) with NaN in a load-profile Excel file."
    )
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help=f"Input Excel file (default: {DEFAULT_INPUT})")
    p.add_argument("--output", "-o", default=None, help="Output Excel file (default: <input>_cleaned.xlsx)")
    p.add_argument(
        "--sentinel", "-s", type=float, action="append", default=None,
        help="Sentinel value to treat as missing (repeatable). Default: -999",
    )
    p.add_argument("--tolerance", type=float, default=1e-6, help="Floating point match tolerance (default: 1e-6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    sentinels = tuple(args.sentinel) if args.sentinel else DEFAULT_SENTINELS
    output_path = (
        Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")
    )

    print(f"Loading {input_path} ...")
    report, report_path = clean_file(input_path, output_path, sentinels, args.tolerance)

    print()
    print(report)
    print()
    print(f"Cleaned file written to: {output_path}")
    print(f"Report written to:       {report_path}")


if __name__ == "__main__":
    main()
