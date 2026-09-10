"""
Data loading utilities for the Load Profile Visualisation GUI.

Reads a load-profile file - either the multi-household export (first
column = timestamp, every other column = one household's active-power
reading, NO header row) or a single aggregated series with a header
row (e.g. timestamp, total_active_power_kw) - into a tidy pandas
DataFrame indexed by timestamp. Both .xlsx/.xls and .csv are accepted;
which layout a given file uses is auto-detected (see load_load_profile).

The DataFrame's columns are treated generically as "series" throughout
- for a multi-household file that means one column per household
(named Household_01 ... Household_NN); for an aggregate file it's
whatever the file's own header says (e.g. total_active_power_kw), and
there's just one of them. The rest of this module and the GUI don't
need to care which case they're in.

This module has no GUI dependencies so it can be reused directly by
later pipeline stages (data quality checks, anomaly detection,
imputation) without pulling in Tkinter/matplotlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


class LoadProfileData:
    """Wraps the loaded load-profile DataFrame and exposes convenience accessors."""

    def __init__(self, df: pd.DataFrame, source_path: Path):
        self.df = df
        self.source_path = source_path

    @property
    def households(self) -> List[str]:
        """Names of the loaded series (households, or a single aggregate column)."""
        return list(self.df.columns)

    @property
    def start(self) -> pd.Timestamp:
        return self.df.index.min()

    @property
    def end(self) -> pd.Timestamp:
        return self.df.index.max()

    @property
    def n_rows(self) -> int:
        return len(self.df)

    def series_for(
        self,
        household: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Return one series (a household, or the aggregate column), optionally sliced by time."""
        if household not in self.df.columns:
            raise KeyError(f"Unknown series: {household!r}")
        s = self.df[household]
        if start is not None or end is not None:
            s = s.loc[start:end]
        return s


SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls")


def _read_raw(path: Path, header: Optional[int]) -> pd.DataFrame:
    """Read the file with pandas, picking the engine/reader by extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header, engine="openpyxl" if suffix == ".xlsx" else None)
    raise ValueError(
        f"Unsupported file type: {suffix!r}. Expected one of {SUPPORTED_SUFFIXES} "
        f"(a multi-household export or a single aggregated series)."
    )


def _looks_like_header(first_cell) -> bool:
    """True if the file's very first cell doesn't parse as a timestamp - i.e. it's a header label."""
    try:
        parsed = pd.to_datetime(first_cell)
    except (ValueError, TypeError):
        return True
    return pd.isna(parsed)


def load_load_profile(path: Union[str, Path]) -> LoadProfileData:
    """
    Load a load-profile file - either format, auto-detected:

      - Multi-household export (.xlsx/.xls or .csv): first column is a
        timestamp, every other column is one household's active power
        (kW), and there is NO header row (row 0 is real data). Household
        columns are named Household_01 ... Household_NN.
      - Single aggregated series (.xlsx/.xls or .csv), e.g. the output
        of interpolate_and_aggregate.py: first column is a timestamp,
        WITH a header row naming it and the value column(s) (e.g.
        "timestamp", "total_active_power_kw"). Column names are kept
        as-is from the file.

    Detection works by checking whether the file's first cell parses as
    a timestamp: if it does, there's no header (multi-household style);
    if it doesn't, that row is treated as column headers.

    Timestamps are floored to the minute to absorb small sub-second
    jitter seen in some exports (e.g. 00:00:00.020 vs 00:00:00.038)
    without losing any real information at 5-minute resolution.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    peek = _read_raw(path, header=None)
    if peek.shape[1] < 2:
        raise ValueError(
            f"Expected a timestamp column plus at least one data column, got {peek.shape[1]} column(s)."
        )
    has_header = _looks_like_header(peek.iat[0, 0])

    raw = _read_raw(path, header=0 if has_header else None)

    timestamps = pd.to_datetime(raw.iloc[:, 0]).dt.floor("min")

    if has_header:
        value_cols = [str(c) for c in raw.columns[1:]]
    else:
        n_series = raw.shape[1] - 1
        value_cols = [f"Household_{i + 1:02d}" for i in range(n_series)]

    df = raw.iloc[:, 1:].copy()
    df.columns = value_cols
    df.index = pd.DatetimeIndex(timestamps, name="Timestamp")
    df = df.sort_index()

    return LoadProfileData(df, path)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "Westridge_2002_dupe.xlsx"
    data = load_load_profile(target)
    print(f"Loaded {data.source_path.name}")
    print(f"  Rows:        {data.n_rows}")
    print(f"  Households:  {len(data.households)} ({data.households[0]} .. {data.households[-1]})")
    print(f"  Time range:  {data.start} to {data.end}")
