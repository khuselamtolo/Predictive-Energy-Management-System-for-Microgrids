"""
The sequential cleaning pipeline, as composable steps.

clean_sentinels.py and interpolate_and_aggregate.py were written as
command-line tools: they read a path and write a path. The GUI needs
frame-in/frame-out. The good news is that the functions underneath are
already pure - replace_sentinels(), interpolate_households() and
aggregate_total() all take a DataFrame and return one - so this module
is a thin wrapper rather than a reimplementation, and the two scripts
keep working unchanged.

Each step is a PipelineStep: run(dataset) -> StepResult. That is what
makes sentinel replacement and interpolation genuinely decoupled. The
GUI currently runs them as one two-element chain behind one button:

    CHAIN = [SentinelStep(), InterpolateStep()]

Splitting them into two buttons later is a change to that list. Neither
implementation is touched, and neither knows the other exists.

Every step also emits a change ledger - one row per reading it altered,
with the value before, the value after, and why. At the measured
sentinel density (0.162% of cells) the ledger for the 8-month panel is
about 35 KB against a 38.9 MB frame, so it is affordable to keep the
full provenance rather than just the count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from clean_sentinels import DEFAULT_SENTINELS, replace_sentinels
from dataset import ChannelRole, Channel, Dataset, DatasetKind
from interpolate_and_aggregate import (DEFAULT_MAX_GAP, aggregate_total,
                                       interpolate_households)
from layers import empty_ledger, ledger_from_mask

AGGREGATE_COLUMN = "total_active_power_kw"


@dataclass
class StepResult:
    """What one step produced, and everything needed to report on it."""

    dataset: Dataset
    report: str
    ledger: pd.DataFrame = field(default_factory=empty_ledger)
    headline: str = ""
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def n_changed(self) -> int:
        return len(self.ledger)


def _rule(title: str) -> List[str]:
    return [title, "=" * max(40, len(title))]


# ---------------------------------------------------------------- scanning
def scan_sentinels(df: pd.DataFrame,
                   sentinels: Sequence[float] = DEFAULT_SENTINELS,
                   tolerance: float = 1e-6) -> Dict[str, object]:
    """
    Count sentinel codes without changing anything.

    Runs automatically in the background on import so the sentinel card
    can say what is there before the user commits to anything. Measured
    at 0.017 s on the 8-month panel, so it is cheap enough to be
    unconditional.
    """
    hits = pd.DataFrame(False, index=df.index, columns=df.columns)
    for code in sentinels:
        hits |= (df - code).abs() < tolerance
    per_channel = hits.sum(axis=0)
    total_cells = int(df.shape[0] * df.shape[1])
    return {
        "n_hits": int(hits.to_numpy().sum()),
        "n_channels": int((per_channel > 0).sum()),
        "total_cells": total_cells,
        "fraction": (float(hits.to_numpy().sum()) / total_cells) if total_cells else 0.0,
        "n_nan": int(df.isna().to_numpy().sum()),
        "per_channel": per_channel,
    }


def sentinel_candidates(df: pd.DataFrame, min_count: int = 20) -> List[Tuple[float, int]]:
    """
    Values that look like error codes but are not in the configured list.

    A sentinel is a single value repeated implausibly often at an extreme
    of the range - -999, -9999, 9999. Real analogue readings essentially
    never repeat to the last decimal hundreds of times. Reporting these
    means a file using a code we were not told about does not silently
    sail through as if it were clean.

    Exact zero is deliberately excluded: on load data it is a real
    reading, not an error code.
    """
    values = pd.Series(df.to_numpy().ravel()).dropna()
    if values.empty:
        return []
    counts = values.value_counts()
    counts = counts[counts >= min_count]
    lo, hi = values.quantile(0.001), values.quantile(0.999)
    out = []
    for value, count in counts.items():
        if value == 0:
            continue
        if value < lo or value > hi:
            out.append((float(value), int(count)))
    return sorted(out, key=lambda t: -t[1])[:5]


# ------------------------------------------------------------------- steps
class SentinelStep:
    """Replace sentinel error codes with NaN. Does not fill anything."""

    name = "Sentinels"

    def __init__(self, sentinels: Sequence[float] = DEFAULT_SENTINELS):
        self.sentinels = tuple(sentinels)

    def run(self, ds: Dataset) -> StepResult:
        before = ds.df
        cleaned, hits = replace_sentinels(before, self.sentinels)
        ledger = ledger_from_mask(before, cleaned, hits,
                                  f"sentinel code {', '.join(f'{s:g}' for s in self.sentinels)}")

        total_cells = int(before.shape[0] * before.shape[1])
        n_hits = int(hits.to_numpy().sum())
        per_channel = hits.sum(axis=0)
        affected = per_channel[per_channel > 0].sort_values(ascending=False)
        per_ts = hits.sum(axis=1)
        worst = per_ts[per_ts > 0].sort_values(ascending=False).head(10)

        lines = _rule("SENTINEL REPLACEMENT")
        lines += [
            f"Source:                    {ds.source_path.name}",
            f"Sentinel code(s) searched: {', '.join(f'{s:g}' for s in self.sentinels)}",
            "",
            f"Total readings:            {total_cells:,}",
            f"Sentinel values replaced:  {n_hits:,}"
            + (f"  ({100 * n_hits / total_cells:.3f}% of all readings)" if total_cells else ""),
            f"Channels affected:         {len(affected)} / {before.shape[1]}",
            f"Missing before:            {int(before.isna().to_numpy().sum()):,}",
            f"Missing after:             {int(cleaned.isna().to_numpy().sum()):,}",
        ]

        if n_hits == 0:
            lines += ["", "No sentinel codes found. The data may already be clean, or it may",
                      "use a code that is not in the list above - see CANDIDATE CODES below."]
        else:
            lines += ["", "Replaced per channel (channels with none are omitted):"]
            lines += [f"  {name}: {int(count):,}" for name, count in affected.items()]

        if not worst.empty:
            lines += [
                "",
                "Timestamps with the most simultaneous hits. A cluster here points to one",
                "shared logging or communication outage rather than many independent faults:",
            ]
            lines += [f"  {ts}: {int(count)} channels" for ts, count in worst.items()]

        others = sentinel_candidates(before)
        others = [(v, c) for v, c in others
                  if not any(abs(v - s) < 1e-6 for s in self.sentinels)]
        if others:
            lines += [
                "",
                "CANDIDATE CODES NOT SEARCHED FOR",
                "Values repeated far more often than a real analogue reading should be, at",
                "an extreme of the range. If one of these is an error code in your export,",
                "add it to the sentinel list and re-run - it is NOT being handled yet:",
            ]
            lines += [f"  {v:g}  appears {c:,} times" for v, c in others]

        lines += [
            "",
            "NOTES",
            "  Sentinel codes are changed to NaN, leaving holes in th dataset.",
        ]

        headline = (f"{n_hits:,} sentinel reading(s) replaced with NaN across "
                    f"{len(affected)} channel(s)")
        return StepResult(
            dataset=ds.with_frame(cleaned, "Sentinels"),
            report="\n".join(lines),
            ledger=ledger,
            headline=headline if n_hits else "No sentinel codes found",
            stats={"n_hits": n_hits, "n_channels": len(affected)},
        )


class InterpolateStep:
    """Fill NaN holes per channel, leaving over-long gaps deliberately empty."""

    name = "Interpolated"

    def __init__(self, method: str = "linear", max_gap: int = DEFAULT_MAX_GAP):
        self.method = method
        self.max_gap = max_gap

    def run(self, ds: Dataset) -> StepResult:
        before = ds.df
        filled, still_missing = interpolate_households(before, self.method, self.max_gap)

        was_missing = before.isna()
        now_filled = was_missing & ~still_missing
        ledger = ledger_from_mask(before, filled, now_filled,
                                  f"{self.method} interpolation (max gap {self.max_gap})")

        n_before = int(was_missing.to_numpy().sum())
        n_after = int(still_missing.to_numpy().sum())
        n_filled = n_before - n_after
        gap_minutes = int(round(ds.interval.total_seconds() / 60)) * self.max_gap

        unfilled = still_missing.sum(axis=0)
        unfilled = unfilled[unfilled > 0].sort_values(ascending=False)

        lines = _rule("INTERPOLATION")
        lines += [
            f"Method:            {self.method}",
            f"Max gap filled:    {self.max_gap} samples "
            f"({gap_minutes // 60}h {gap_minutes % 60:02d}m at this data's {ds.interval_label()} step)",
            "",
            f"Missing before:    {n_before:,}",
            f"Filled:            {n_filled:,}",
            f"Still missing:     {n_after:,}",
        ]
        if not unfilled.empty:
            lines += ["", "Channels with readings still empty (gap longer than the limit, or at a series edge):"]
            lines += [f"  {name}: {int(count):,}" for name, count in unfilled.items()]

        if n_after:
            lines += [
                "",
                "THESE NEED A DECISION",
                f"  {n_after:,} reading(s) are still NaN and were left that way on purpose. A gap",
                f"  longer than {self.max_gap} samples has no nearby evidence to interpolate from, and",
                "  fabricating one would put invented values into a forecasting training set,",
                "  which is worse than a visible hole. Raise the max gap only if you are",
                "  satisfied the fill would be defensible.",
            ]

        lines += [
            "",
            "NOTES",
            "  Each channel is interpolated independently.",
        ]

        headline = f"{n_filled:,} reading(s) interpolated" + (
            f", {n_after:,} left empty" if n_after else "")
        return StepResult(
            dataset=ds.with_frame(filled, "Interpolated"),
            report="\n".join(lines),
            ledger=ledger,
            headline=headline,
            stats={"n_filled": n_filled, "n_left": n_after},
        )


class AggregateStep:
    """
    Sum across channels into one total-demand series.

    The output Dataset is kind AGGREGATE_SERIES - structurally identical
    to a pre-aggregated .csv imported directly - so from here on there is
    exactly one code path regardless of how the user got here.
    """

    name = "Aggregated"

    def __init__(self, column: str = AGGREGATE_COLUMN):
        self.column = column

    def run(self, ds: Dataset) -> StepResult:
        before = ds.df
        agg = aggregate_total(before)
        naive = before.sum(axis=1, skipna=True)

        out = pd.DataFrame({self.column: agg})
        out.index.name = before.index.name or "Timestamp"

        channels = {self.column: Channel(
            name=self.column, role=ChannelRole.POWER, unit="kW",
            valid_range=None, default_cutoff_min=None, confidence=1.0,
            note="produced by aggregating the household panel",
        )}

        n_unknown = int(agg.isna().sum())
        interval_h = ds.interval.total_seconds() / 3600.0
        energy = float(agg.dropna().sum() * interval_h)

        lines = _rule("AGGREGATION")
        lines += [
            f"Channels summed:     {before.shape[1]}",
            f"Output column:       {self.column}",
            f"Readings:            {len(agg):,}",
            f"Unknown totals:      {n_unknown:,}",
            "",
            "TOTAL DEMAND",
            f"  mean    {agg.mean():.2f} kW",
            f"  median  {agg.median():.2f} kW",
            f"  min     {agg.min():.2f} kW",
            f"  max     {agg.max():.2f} kW",
            f"  std     {agg.std():.2f} kW",
            f"  skew    {agg.skew():.3f}",
            f"  excess kurtosis {agg.kurtosis():.3f}",
            f"  total energy over the record: {energy:,.0f} kWh",
        ]

        per_h_skew = before.skew().dropna()
        if not per_h_skew.empty:
            lines += [
                "",
                "AGGREGATION PULLS THE DISTRIBUTION TOWARD SYMMETRY",
                f"  individual channels: skew {per_h_skew.min():.2f} to {per_h_skew.max():.2f}",
                f"  aggregate:           skew {agg.skew():.2f}",
            ]

        if n_unknown:
            lines += [
                "",
                "WHY SOME TOTALS ARE UNKNOWN",
                f"  {n_unknown:,} timestamp(s) still had at least one channel missing after",
                "  interpolation, so the total there is unknown and is left as NaN.",
            ]
            missing_rows = before.isna().sum(axis=1)
            worst = missing_rows[missing_rows > 0].sort_values(ascending=False).head(5)
            if not worst.empty:
                lines += ["", "  Worst timestamps - naive sum vs. true answer:"]
                for ts, n_missing in worst.items():
                    lines.append(
                        f"    {ts}: {int(n_missing)} channel(s) missing → "
                        f"naive sum {naive.loc[ts]:.1f} kW, reported total "
                        + ("unknown" if pd.isna(agg.loc[ts]) else f"{agg.loc[ts]:.1f} kW")
                    )

        lines += [
            "",
            "NOTES",
            "  The result is a single aggregate series (csv).",
        ]

        headline = (f"{before.shape[1]} channels summed into one series; "
                    f"peak {agg.max():.0f} kW, mean {agg.mean():.0f} kW")
        return StepResult(
            dataset=ds.with_frame(out, "Aggregated", channels=channels,
                                  kind=DatasetKind.AGGREGATE_SERIES),
            report="\n".join(lines),
            ledger=empty_ledger(),  # a reshape, not an edit to existing readings
            headline=headline,
            stats={"n_unknown": n_unknown, "peak": float(agg.max())},
        )


# ------------------------------------------------------------------ chains
def run_chain(steps: Sequence[object], ds: Dataset) -> StepResult:
    """
    Run steps in order, threading the dataset through and concatenating
    the reports into one document with a section per step.

    This is what lets one button run two independent programs without
    either of them knowing the other exists.
    """
    reports, ledgers, headlines, stats = [], [], [], {}
    current = ds
    final_name = ""
    for step in steps:
        result = step.run(current)
        current = result.dataset
        reports.append(result.report)
        if len(result.ledger):
            ledgers.append(result.ledger)
        if result.headline:
            headlines.append(result.headline)
        stats.update({f"{step.name}.{k}": v for k, v in result.stats.items()})
        final_name = step.name

    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else empty_ledger()
    header = (f"Report generated for: {ds.source_path.name}\n"
              f"Steps run: {' → '.join(s.name for s in steps)}\n\n\n")
    return StepResult(
        dataset=current,
        report=header + "\n\n\n".join(reports),
        ledger=ledger,
        headline="; ".join(headlines),
        stats=stats,
    )


CLEAN_CHAIN = (SentinelStep, InterpolateStep)
