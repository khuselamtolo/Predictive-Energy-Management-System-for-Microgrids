"""
Parameter calibration and sensitivity analysis for the outlier detectors.

The problem this solves: "which window and sigma should I use?" has no
answer without knowing what a correct answer looks like, and this data
carries no labelled faults - the -999 sentinels were cleaned upstream and
the aggregate has been interpolated, so whatever is left is unlabelled.

The approach here is synthetic fault injection. Plant faults of known
type, size and position into a copy of the series, run the detector
across a grid of (window, sigma), and score each cell against the known
positions with precision / recall / F1. The winning cell is a calibrated
recommendation; the whole grid is the sensitivity analysis.

    precision = flagged points that were really injected faults
    recall    = injected faults that were actually flagged
    F1        = harmonic mean of the two

Two honest caveats, both worth stating in a write-up rather than hiding:

1. THE RESULT IS ONLY OPTIMAL FOR THE FAULTS YOU INJECT. Calibrating on
   spikes tunes a spike detector. Injection design is a modelling
   assumption, not a neutral measurement - which is why `magnitudes` and
   `kinds` are explicit arguments and why the report breaks recall down
   by fault size rather than quoting one number.

2. PRECISION IS A LOWER BOUND. The original series may contain real
   anomalies of its own. Those get flagged, count as false positives
   against the injected ground truth, and drag precision down even
   though the detector may be right about them. `baseline_flags` in the
   results records how many points the same settings flag in the CLEAN
   series, so this effect is visible instead of silently absorbed.

Fault types (chosen for this dataset - see the project notes):

  spike / dip  a single reading displaced by k local sigma, k drawn from
               `magnitudes`. Using local sigma rather than an absolute kW
               offset keeps difficulty comparable across the seasonal
               trend, and lets recall be reported as a function of k.
  dropout      a run of 1-3 readings collapsed to near zero, the
               aggregate-level equivalent of the shared metering failures
               that produced the -999 clusters in the raw data.

Usage:
    from calibration import calibrate
    result = calibrate(series, method="zscore")
    print(result.best)        # window, sigma, f1, precision, recall
    print(result.grid)        # the full sensitivity surface
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from outlier_detection import (max_achievable_zscore, robust_local_scale,
                               rolling_zscore)

#: Grid searched by default: 30 minutes to 6 hours at 5-minute sampling.
DEFAULT_WINDOWS: Tuple[int, ...] = (6, 12, 24, 36, 48, 72)
DEFAULT_SIGMAS: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

#: Fault sizes in local standard deviations. Spans "subtle enough to be
#: arguable" to "unmistakable", so recall can be reported against size.
DEFAULT_MAGNITUDES: Tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)

DEFAULT_N_FAULTS = 120
DEFAULT_REPEATS = 5
DEFAULT_KINDS: Tuple[str, ...] = ("spike", "dip", "dropout")

DETECTORS = {"zscore": rolling_zscore}


@dataclass
class CalibrationResult:
    grid: pd.DataFrame            # one row per (window, sigma) cell
    best: pd.Series               # the winning row
    recall_by_magnitude: pd.DataFrame
    method: str
    n_faults: int
    repeats: int
    kinds: Tuple[str, ...]
    series_name: str = ""
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        b = self.best
        lines = [
            f"Calibration - {self.method}"
            + (f" on {self.series_name}" if self.series_name else ""),
            "=" * 52,
            f"Injected {self.n_faults} faults ({', '.join(self.kinds)}) x {self.repeats} repeats",
            "",
            f"Best: window = {int(b['window'])} samples, threshold = {b['sigma']:g} sigma",
            f"      F1 = {b['f1']:.3f}   precision = {b['precision']:.3f}   recall = {b['recall']:.3f}",
            f"      flags {b['baseline_flags']:.0f} points in the clean series at these settings",
            "",
            "Recall by fault size (how big does a fault have to be to get caught?):",
        ]
        for _, r in self.recall_by_magnitude.iterrows():
            lines.append(f"  {r['magnitude']:>5} : {100 * r['recall']:5.1f}%")
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


#: Window whose local spread defines "one sigma" for injected fault sizes.
REFERENCE_WINDOW = 48


def _local_scale(values: np.ndarray, window: int = REFERENCE_WINDOW) -> np.ndarray:
    """
    Robust local spread used to size injected faults.

    Deliberately a ROBUST measure (rolling MAD, scaled to std-equivalent)
    rather than a rolling standard deviation. The detector itself uses a
    standard deviation, but sizing injected faults that way would let
    outliers already present in the series inflate the reference, making
    a fault labelled "12 sigma" quietly smaller than its label claims.

    An earlier version used a hand-rolled MAD variant whose median came
    out at 25 kW against 43 kW for the shared definition on the same
    series, which made the recall-by-size table misleading. One
    definition, used everywhere.
    """
    frame = pd.DataFrame({"v": values})
    scale = robust_local_scale(frame, window=window)["v"].to_numpy()
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > 0)])
    return np.where(np.isfinite(scale) & (scale > 0), scale, fallback)


def inject_faults(
    series: pd.Series,
    rng: np.random.Generator,
    n_faults: int = DEFAULT_N_FAULTS,
    kinds: Sequence[str] = DEFAULT_KINDS,
    magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
    dropout_max_len: int = 3,
    edge_margin: int = 100,
    min_gap: int = 20,
    scale: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    Plant faults of known type, size and position into a copy of `series`.

    Returns (corrupted, truth, detail): the damaged series, a boolean
    array marking every corrupted reading, and a table describing each
    fault (position, kind, magnitude) so recall can be broken down by
    size afterwards.

    Faults are kept `min_gap` samples apart and `edge_margin` away from
    the ends, so they don't sit inside each other's rolling windows and
    aren't distorted by the series edges.

    `scale` may be supplied to skip recomputing the local spread - it
    only depends on the untouched series, so a caller running several
    injections should compute it once.
    """
    values = series.to_numpy(dtype=float).copy()
    n = len(values)
    if scale is None:
        scale = _local_scale(values)
    truth = np.zeros(n, dtype=bool)

    usable = np.arange(edge_margin, n - edge_margin - dropout_max_len)
    rng.shuffle(usable)

    records, taken = [], []
    for pos in usable:
        if len(records) >= n_faults:
            break
        if any(abs(pos - t) < min_gap for t in taken[-40:]):
            continue

        kind = kinds[rng.integers(len(kinds))]
        magnitude = float(magnitudes[rng.integers(len(magnitudes))])

        if kind == "dropout":
            length = int(rng.integers(1, dropout_max_len + 1))
            sl = slice(pos, pos + length)
            values[sl] = values[sl] * rng.uniform(0.0, 0.05, size=length)
            truth[sl] = True
            magnitude = float("nan")   # a dropout's size isn't set by us
        else:
            sign = 1.0 if kind == "spike" else -1.0
            values[pos] = max(values[pos] + sign * magnitude * scale[pos], 0.0)
            truth[pos] = True

        records.append({"position": int(pos), "kind": kind, "magnitude": magnitude})
        taken.append(pos)

    detail = pd.DataFrame.from_records(records)
    return pd.Series(values, index=series.index, name=series.name), truth, detail


def _score(flagged: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
    tp = int(np.count_nonzero(flagged & truth))
    fp = int(np.count_nonzero(flagged & ~truth))
    fn = int(np.count_nonzero(~flagged & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _centre_and_scale(detector, frame: pd.DataFrame, window: int, name: str):
    """
    Run a detector once at 1 sigma to recover its rolling centre and
    scale, so every sigma in the grid can then be scored by simple
    thresholding instead of recomputing the rolling statistics. That
    turns an O(windows x sigmas) sweep into an O(windows) one.
    """
    _, centre, threshold = detector(frame, window=window, n_sigmas=1.0)
    deviation = (frame[name] - centre[name]).abs().to_numpy()
    return deviation, threshold[name].to_numpy()


def calibrate(
    series: pd.Series,
    method: str = "zscore",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    n_faults: int = DEFAULT_N_FAULTS,
    repeats: int = DEFAULT_REPEATS,
    kinds: Sequence[str] = DEFAULT_KINDS,
    magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
    seed: int = 0,
    progress=None,
) -> CalibrationResult:
    """
    Search the (window, sigma) grid for the settings that best recover
    injected faults, averaging over `repeats` independent injections so
    the answer isn't an artefact of one random draw.

    `progress`, if given, is called as progress(done, total) so a GUI can
    show how far along the sweep is.
    """
    if method not in DETECTORS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(DETECTORS)}")
    if series.isna().any():
        series = series.dropna()
    if len(series) < 500:
        raise ValueError("Need at least 500 readings to calibrate; widen the range.")

    detector = DETECTORS[method]
    name = series.name or "series"
    notes = []

    # Depends only on the untouched series, so compute it once rather
    # than inside every injection (that alone was most of the runtime).
    injection_scale = _local_scale(series.to_numpy(dtype=float))

    # Baseline: what these settings flag in the UNDAMAGED series. Lets the
    # reader see how much of the "false positive" mass is pre-existing.
    clean_frame = series.to_frame(name)

    rows: Dict[Tuple[int, float], list] = {}
    total = len(windows) * max(repeats, 1)
    done = 0

    baseline: Dict[Tuple[int, float], int] = {}
    for window in windows:
        dev_clean, scale_clean = _centre_and_scale(detector, clean_frame, window, name)
        for sigma in sigmas:
            ok = np.isfinite(scale_clean)
            baseline[(window, sigma)] = int(np.count_nonzero(ok & (dev_clean > sigma * scale_clean)))

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        corrupted, truth, detail = inject_faults(
            series, rng, n_faults=n_faults, kinds=kinds, magnitudes=magnitudes,
            scale=injection_scale,
        )
        frame = corrupted.to_frame(name)

        for window in windows:
            deviation, scale = _centre_and_scale(detector, frame, window, name)
            finite = np.isfinite(scale)
            for sigma in sigmas:
                flagged = finite & (deviation > sigma * scale)
                p, r, f1 = _score(flagged, truth)
                rows.setdefault((window, sigma), []).append((p, r, f1))
            done += 1
            if progress is not None:
                progress(done, total)

    grid_records = []
    for (window, sigma), scores in rows.items():
        arr = np.array(scores)
        reachable = (method != "zscore") or (sigma < max_achievable_zscore(window))
        grid_records.append({
            "window": window,
            "sigma": sigma,
            "precision": arr[:, 0].mean(),
            "recall": arr[:, 1].mean(),
            "f1": arr[:, 2].mean(),
            "f1_std": arr[:, 2].std(),
            "baseline_flags": baseline[(window, sigma)],
            "reachable": reachable,
        })
    grid = pd.DataFrame(grid_records).sort_values(["window", "sigma"]).reset_index(drop=True)

    searchable = grid[grid["reachable"]]
    if searchable.empty:
        raise ValueError("Every cell in the grid is unreachable for this method; widen the windows.")
    best = searchable.loc[searchable["f1"].idxmax()].copy()

    if method == "zscore":
        blocked = int((~grid["reachable"]).sum())
        if blocked:
            notes.append(
                f"{blocked} of {len(grid)} grid cells are mathematically unreachable for a "
                f"Z-score (sigma >= (w-1)/sqrt(w)) and were excluded from the search."
            )

    # Second pass: break recall down by fault size AT THE WINNING CELL.
    # Injection is seeded, so replaying it costs one detector call per
    # repeat and avoids holding every mask in memory.
    best_window, best_sigma = int(best["window"]), float(best["sigma"])
    magnitude_hits: Dict[str, list] = {}
    dropout_runs: Dict[int, list] = {}
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        corrupted, truth, detail = inject_faults(
            series, rng, n_faults=n_faults, kinds=kinds, magnitudes=magnitudes,
            scale=injection_scale,
        )
        deviation, scale = _centre_and_scale(
            detector, corrupted.to_frame(name), best_window, name
        )
        flagged = np.isfinite(scale) & (deviation > best_sigma * scale)
        for _, row in detail.iterrows():
            pos = int(row["position"])
            if row["kind"] == "dropout":
                magnitude_hits.setdefault("dropout", []).append(bool(flagged[pos]))
                length = 1
                while pos + length < len(truth) and truth[pos + length]:
                    length += 1
                dropout_runs.setdefault(length, []).append(bool(flagged[pos]))
            else:
                magnitude_hits.setdefault(f"{row['magnitude']:g}σ", []).append(bool(flagged[pos]))

    def _sort_key(label: str):
        return (1, 0.0) if label == "dropout" else (0, float(label.rstrip("σ")))

    recall_by_magnitude = pd.DataFrame([
        {"magnitude": label, "recall": float(np.mean(hits)), "n": len(hits)}
        for label, hits in sorted(magnitude_hits.items(), key=lambda kv: _sort_key(kv[0]))
    ])

    if dropout_runs and max(dropout_runs) > 1:
        by_len = {L: float(np.mean(v)) for L, v in sorted(dropout_runs.items())}
        if by_len[min(by_len)] - by_len[max(by_len)] > 0.05:
            notes.append(
                "Detection of the first "
                + ", ".join(f"{100 * r:.0f}% at run length {L}" for L, r in by_len.items())
                + ". Consecutive bad readings sit inside each other's window and inflate "
                "the local spread."
            )

    return CalibrationResult(
        grid=grid, best=best, recall_by_magnitude=recall_by_magnitude,
        method=method, n_faults=n_faults, repeats=repeats, kinds=tuple(kinds),
        series_name=str(name), notes=notes,
    )


if __name__ == "__main__":
    import sys
    from data_loader import load_load_profile

    path = sys.argv[1] if len(sys.argv) > 1 else "aggregate_total_load.csv"
    method = sys.argv[2] if len(sys.argv) > 2 else "zscore"
    data = load_load_profile(path)
    s = data.df[data.households[0]]

    result = calibrate(s, method=method)
    print(result.summary())
    print("\nFull grid (F1):")
    pivot = result.grid.pivot(index="window", columns="sigma", values="f1")
    print(pivot.round(3).to_string())
