"""
Semantic classification of an imported file.

data_loader.py answers a STRUCTURAL question - does this file have a
header row, and how many value columns are there? That is enough to
parse a file, but not enough to know what to do with it. The Stellies
solar export and a two-household aggregate export are structurally
identical: a header row and two numeric columns. Only one of them
should be offered a low-pass filter clamped to 0-1400 W/m^2.

This module adds the second, semantic stage. It labels every column
with a ChannelRole (what the numbers physically are) and the file with
a DatasetKind (where it sits in the pipeline), and those two labels are
what the GUI gates its buttons on - never the file extension.

Classification uses three signals, cheapest first:

  1. Column names, against a lexicon of the terms these exports
     actually use.
  2. Structural shape - a headerless multi-column file is a household
     panel by construction.
  3. A physical fingerprint of the values themselves.

Signal 3 is also a VALIDATOR: it can overrule signals 1 and 2, because
a column's values are harder to get wrong than its label. The one test
that carries most of the weight is the night-time zero fraction.
Irradiance is the only signal in this domain that is identically zero
for roughly half of every 24-hour period, so it stays recognisable even
when the column is called "Column3". When signal 3 contradicts the
column's own name, the Channel keeps a `note` saying so - the GUI shows
it rather than silently overruling what the user's file said.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data_loader import LoadProfileData, load_load_profile
from signal_filters import (DEFAULT_CUTOFF_MINUTES, GHI_VALID_RANGE,
                            TEMP_CUTOFF_MINUTES, TEMP_VALID_RANGE)


class ChannelRole(Enum):
    """What one column physically is, which decides what may be done to it."""

    POWER = "power"              # kW, >= 0, never clamped, Z-score screened
    IRRADIANCE = "irradiance"    # W/m^2, clamped, low-passed
    TEMPERATURE = "temperature"  # deg C, clamped, low-passed
    UNKNOWN = "unknown"


class DatasetKind(Enum):
    """Where the whole file sits in the pipeline."""

    HOUSEHOLD_PANEL = "household_panel"    # N power columns; needs sentinels + aggregation
    AGGREGATE_SERIES = "aggregate_series"  # 1 power column; aggregation already done
    ENVIRONMENTAL = "environmental"        # GHI and/or temperature; filtered, never aggregated


# Signal 1. Deliberately loose: these match against a lower-cased column
# name, and a miss just falls through to the next signal.
ROLE_PATTERNS = {
    ChannelRole.IRRADIANCE: re.compile(
        r"(ghi|dni|dhi|irradian|insolat|solar[_ ]?rad|w[_ ]?m2|w/m)", re.I),
    ChannelRole.TEMPERATURE: re.compile(
        r"(temp|t_?amb|ambient|deg[_ ]?c|°c)", re.I),
    ChannelRole.POWER: re.compile(
        r"(kwh?|k_?w|power|load|demand|active|total|household)", re.I),
}

NAME_CONFIDENCE = 0.9
SHAPE_CONFIDENCE = 0.6
PHYSICAL_CONFIDENCE = 0.8

# What counts as "dark". An absolute threshold is wrong: a pyranometer
# has a noise floor and a thermal offset, so night readings scatter a few
# W/m^2 either side of zero rather than sitting exactly at it. Scale it
# to the column's own peak instead, with a small absolute floor.
ZERO_TOLERANCE = 5.0        # W/m^2 absolute floor
ZERO_RELATIVE = 0.02        # ...or 2% of the column's peak, whichever is larger

# The real discriminator is not HOW MANY zeros there are but WHERE they
# fall. Irradiance is dark for a contiguous block of hours every day and
# never dark in the middle of the day. A vacant household can be at zero
# for most of a month, but its zeros do not organise themselves by hour
# of day like that. Requiring both halves of the pattern is what keeps a
# quiet meter from being mistaken for a pyranometer.
# This guard is not theoretical. Household_43 in the Westridge export is
# within 5 kW of zero for 97.2% of its samples, and four more households
# are above 90% - a plain "mostly zeros" test would have called five
# households pyranometers.
DARK_HOURS_REQUIRED = 6     # hours of the day that are >=90% dark
LIT_HOURS_REQUIRED = 6      # hours of the day that are <=10% dark

# Peak has to be in the band a working pyranometer actually reports.
# Below ~100 W/m^2 nothing useful is being measured; above ~1600 the
# reading is not irradiance at all. A daylight-only household load can
# clear the day/night shape test, so this band is what stops it.
IRRADIANCE_PEAK_BAND = (100.0, 1600.0)

UNITS = {
    ChannelRole.POWER: "kW",
    ChannelRole.IRRADIANCE: "W/m²",
    ChannelRole.TEMPERATURE: "°C",
    ChannelRole.UNKNOWN: "",
}

# Physical limits and the cutoff appropriate to each signal. These are
# the same constants the filter panel already used - centralised here so
# a channel carries its own defaults instead of the GUI guessing.
ROLE_DEFAULTS = {
    ChannelRole.IRRADIANCE: (GHI_VALID_RANGE, DEFAULT_CUTOFF_MINUTES),
    ChannelRole.TEMPERATURE: (TEMP_VALID_RANGE, TEMP_CUTOFF_MINUTES),
    ChannelRole.POWER: (None, None),
    ChannelRole.UNKNOWN: (None, None),
}


@dataclass(frozen=True)
class Channel:
    """One column, plus everything the GUI needs to decide what to offer for it."""

    name: str
    role: ChannelRole
    unit: str
    valid_range: Optional[tuple]      # None => never clamp
    default_cutoff_min: Optional[float]  # None => no low-pass offered
    confidence: float
    note: str = ""                    # set when the fingerprint overruled the name

    @property
    def is_environmental(self) -> bool:
        return self.role in (ChannelRole.IRRADIANCE, ChannelRole.TEMPERATURE)

    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


def _fingerprint(values: pd.Series) -> Dict[str, float]:
    """Cheap physical summary of one column, used by signal 3."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"n": 0, "dark_fraction": 0.0, "has_negative": False,
                "lo": np.nan, "hi": np.nan, "dark_hours": 0, "lit_hours": 0}

    hi = float(x.max())
    dark_level = max(ZERO_TOLERANCE, ZERO_RELATIVE * hi)
    dark = x <= dark_level

    # Where the dark readings fall in the day. This is the part that
    # separates a pyranometer from a meter that happens to read zero a
    # lot: a day/night cycle shows up as some hours almost always dark
    # and others almost never, with nothing in between doing the work.
    dark_hours = lit_hours = 0
    if isinstance(x.index, pd.DatetimeIndex) and len(x) >= 288:
        by_hour = dark.groupby(x.index.hour).mean()
        dark_hours = int((by_hour >= 0.9).sum())
        lit_hours = int((by_hour <= 0.1).sum())

    return {
        "n": float(len(x)),
        "dark_fraction": float(dark.mean()),
        "has_negative": bool((x < -ZERO_TOLERANCE).any()),
        "lo": float(x.min()),
        "hi": hi,
        "dark_hours": dark_hours,
        "lit_hours": lit_hours,
    }


def _role_from_name(name: str) -> Optional[ChannelRole]:
    """Signal 1. Irradiance and temperature are checked before power:
    'solar_power_kw' should read as power, but 'ghi_w_m2' must not be
    dragged to POWER by the stray 'w'."""
    for role in (ChannelRole.IRRADIANCE, ChannelRole.TEMPERATURE, ChannelRole.POWER):
        if ROLE_PATTERNS[role].search(str(name)):
            # "solar power" is power, not irradiance - the more specific
            # word wins when both match.
            if role is ChannelRole.IRRADIANCE and ROLE_PATTERNS[ChannelRole.POWER].search(str(name)):
                if not re.search(r"(ghi|dni|dhi|irradian|w[_ ]?m2|w/m)", str(name), re.I):
                    return ChannelRole.POWER
            return role
    return None


def _role_from_physics(fp: Dict[str, float]) -> Optional[ChannelRole]:
    """
    Signal 3. Only fires when the values are distinctive enough to be
    worth trusting over a column label; otherwise returns None and lets
    the earlier signals stand.
    """
    if fp["n"] < 288:  # less than a day; not enough to see a diurnal cycle
        return None

    # The discriminator: a clean day/night cycle. Both halves are
    # required - hours that are reliably dark AND hours that are
    # reliably lit - so a meter that simply reads zero a lot cannot
    # satisfy it.
    if (fp["dark_hours"] >= DARK_HOURS_REQUIRED
            and fp["lit_hours"] >= LIT_HOURS_REQUIRED
            and not fp["has_negative"]
            and IRRADIANCE_PEAK_BAND[0] <= fp["hi"] <= IRRADIANCE_PEAK_BAND[1]):
        return ChannelRole.IRRADIANCE

    # Sub-zero readings rule out both irradiance and household power.
    # Combined with a small dynamic range, that is ambient temperature.
    if fp["has_negative"] and fp["lo"] >= TEMP_VALID_RANGE[0] and fp["hi"] <= TEMP_VALID_RANGE[1]:
        return ChannelRole.TEMPERATURE

    return None


def classify_channel(name: str, values: pd.Series, structural: Optional[ChannelRole] = None) -> Channel:
    """Run the three signals over one column and return its Channel."""
    fp = _fingerprint(values)
    by_name = _role_from_name(name)
    by_physics = _role_from_physics(fp)

    note = ""
    if by_physics is not None and by_name is not None and by_physics is not by_name:
        # The values disagree with the label. Trust the values, but say
        # so - never silently overrule what the user's own file said.
        role, confidence = by_physics, PHYSICAL_CONFIDENCE
        note = (
            f"named as {by_name.value} but behaves like {by_physics.value} "
            f"({fp['dark_fraction']:.0%} of readings dark, range "
            f"{fp['lo']:.4g}–{fp['hi']:.4g}). Treated as {by_physics.value}."
        )
    elif by_name is not None:
        role, confidence = by_name, NAME_CONFIDENCE
    elif by_physics is not None:
        role, confidence = by_physics, PHYSICAL_CONFIDENCE
    elif structural is not None:
        role, confidence = structural, SHAPE_CONFIDENCE
    else:
        role, confidence = ChannelRole.UNKNOWN, 0.0

    valid_range, cutoff = ROLE_DEFAULTS[role]
    return Channel(
        name=str(name), role=role, unit=UNITS[role],
        valid_range=valid_range, default_cutoff_min=cutoff,
        confidence=confidence, note=note,
    )


def classify_frame(df: pd.DataFrame, had_header: bool) -> Dict[str, Channel]:
    """Classify every column. Signal 2 (structure) is folded in here."""
    # A headerless multi-column export is a household panel by
    # construction - data_loader only auto-names Household_NN when there
    # was no header row to name them.
    structural = ChannelRole.POWER if (not had_header and df.shape[1] >= 2) else None
    if had_header and df.shape[1] == 1:
        structural = ChannelRole.POWER
    return {str(c): classify_channel(c, df[c], structural) for c in df.columns}


def infer_kind(channels: Dict[str, Channel]) -> DatasetKind:
    """Environmental beats everything; after that it is a count of power columns."""
    roles = [c.role for c in channels.values()]
    if any(r in (ChannelRole.IRRADIANCE, ChannelRole.TEMPERATURE) for r in roles):
        return DatasetKind.ENVIRONMENTAL
    n_power = sum(1 for r in roles if r is ChannelRole.POWER) or len(roles)
    return DatasetKind.HOUSEHOLD_PANEL if n_power > 1 else DatasetKind.AGGREGATE_SERIES


def median_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median sampling step. Every window parameter in the app is in
    samples, so 48 means four hours at 5-minute data and 48 minutes at
    1-minute data - the GUI needs this to say which."""
    if len(index) < 2:
        return pd.Timedelta(minutes=5)
    step = pd.Series(index).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return pd.Timedelta(minutes=5)
    return step


class Dataset:
    """
    The standardised structure every module downstream consumes.

    Deliberately exposes the same surface as LoadProfileData (df,
    households, start, end, n_rows, source_path, series_for) so existing
    modules keep working unchanged, and adds the semantic layer on top.

    Treat `df` as read-only. Steps produce a NEW Dataset via with_frame()
    rather than mutating this one - that is what makes the layer stack's
    non-destructive guarantee structural rather than a convention someone
    has to remember.
    """

    def __init__(self, df: pd.DataFrame, channels: Dict[str, Channel],
                 kind: DatasetKind, source_path: Path, layer_name: str = "Raw",
                 interval: Optional[pd.Timedelta] = None, notes: Optional[List[str]] = None):
        self.df = df
        self.channels = channels
        self.kind = kind
        self.source_path = Path(source_path)
        self.layer_name = layer_name
        self.interval = interval if interval is not None else median_interval(df.index)
        self.notes = list(notes or [])

    # -- LoadProfileData-compatible surface ------------------------------
    @property
    def households(self) -> List[str]:
        return [str(c) for c in self.df.columns]

    @property
    def start(self) -> pd.Timestamp:
        return self.df.index.min()

    @property
    def end(self) -> pd.Timestamp:
        return self.df.index.max()

    @property
    def n_rows(self) -> int:
        return len(self.df)

    def series_for(self, household: str, start=None, end=None) -> pd.Series:
        if household not in self.df.columns:
            raise KeyError(f"Unknown series: {household!r}")
        s = self.df[household]
        if start is not None or end is not None:
            s = s.loc[start:end]
        return s

    # -- semantic surface -------------------------------------------------
    @property
    def is_regular(self) -> bool:
        if len(self.df) < 3:
            return True
        steps = pd.Series(self.df.index).diff().dropna()
        return bool((steps == self.interval).all())

    def channel(self, name: str) -> Channel:
        return self.channels.get(str(name), Channel(
            str(name), ChannelRole.UNKNOWN, "", None, None, 0.0))

    @property
    def has_environmental(self) -> bool:
        return any(c.is_environmental for c in self.channels.values())

    def with_frame(self, df: pd.DataFrame, layer_name: str,
                   channels: Optional[Dict[str, Channel]] = None,
                   kind: Optional[DatasetKind] = None,
                   notes: Optional[List[str]] = None) -> "Dataset":
        """A new Dataset over a new frame. Never mutates this one."""
        return Dataset(
            df=df,
            channels=channels if channels is not None else
                     {k: v for k, v in self.channels.items() if k in df.columns},
            kind=kind if kind is not None else self.kind,
            source_path=self.source_path,
            layer_name=layer_name,
            interval=self.interval,
            notes=notes if notes is not None else self.notes,
        )

    def interval_label(self) -> str:
        minutes = self.interval.total_seconds() / 60
        if minutes >= 60 and minutes % 60 == 0:
            return f"{int(minutes // 60)} h"
        return f"{minutes:g} min"

    def kind_label(self) -> str:
        return {
            DatasetKind.HOUSEHOLD_PANEL: "Household panel",
            DatasetKind.AGGREGATE_SERIES: "Aggregate series",
            DatasetKind.ENVIRONMENTAL: "Environmental",
        }[self.kind]

    def samples_to_duration(self, samples: int) -> str:
        """'48 samples (4h 00m)' - the same phrasing the outlier panel used."""
        minutes = int(round(self.interval.total_seconds() / 60)) * samples
        return f"{samples} samples ({minutes // 60}h {minutes % 60:02d}m)"

    def summary(self) -> str:
        return (f"{self.source_path.name}   •   {self.kind_label()}   •   "
                f"{len(self.households)} channel(s)   •   {self.n_rows:,} readings   •   "
                f"{self.interval_label()}   •   {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}")


def from_load_profile(data: LoadProfileData) -> Dataset:
    """Wrap an already-parsed LoadProfileData, running the semantic stage."""
    df = data.df
    # data_loader auto-names columns Household_NN only when the file had
    # no header row, so the naming tells us which structural case it was.
    had_header = not all(str(c).startswith("Household_") for c in df.columns)
    channels = classify_frame(df, had_header)
    notes = [f"{c.name}: {c.note}" for c in channels.values() if c.note]
    return Dataset(df=df, channels=channels, kind=infer_kind(channels),
                   source_path=data.source_path, layer_name="Raw", notes=notes)


def load_dataset(path: Union[str, Path]) -> Dataset:
    """Parse (structural stage) then classify (semantic stage)."""
    return from_load_profile(load_load_profile(path))


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "Westridge_2002_dupe.xlsx"
    ds = load_dataset(target)
    print(ds.summary())
    print(f"  kind:     {ds.kind.value}")
    print(f"  regular:  {ds.is_regular}")
    for ch in list(ds.channels.values())[:8]:
        print(f"  {ch.name:<28} {ch.role.value:<12} conf {ch.confidence:.1f} {ch.note}")
    if len(ds.channels) > 8:
        print(f"  ... and {len(ds.channels) - 8} more")
