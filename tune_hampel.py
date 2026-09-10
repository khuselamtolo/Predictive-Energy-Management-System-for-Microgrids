"""
Hampel parameter tuning + sensitivity analysis
==============================================
Systematically choose the (window, sigma) pair for the aggregate-load
Hampel filter, then measure how much that choice actually matters.

There are no ground-truth error labels for this data, so the filter is
scored against faults injected into clean stretches of the real series.
That means the result is only as good as the fault model below: it tells
you which parameters recover *the kinds of errors simulated here*, not
which parameters are correct in some absolute sense. Read the detection
-limit curve and the stability surface, not just the single best cell.

Usage:
    python tune_hampel.py --input aggregate_total_load.csv
    python tune_hampel.py --input agg.csv --column total_active_power_kw \
        --seeds 40 --peak-hours 17 21

Outputs (written next to the input unless --outdir is given):
    hampel_tuning_report.txt   full numbers, written for the write-up
    hampel_tuning.csv          the whole grid, one row per (window, sigma)
    hampel_tuning.png          heatmaps + sensitivity curves
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from outlier_detection import hampel_filter as _project_hampel
    HAVE_PROJECT_FILTER = True
except Exception:  # noqa: BLE001 - script must stand alone if run elsewhere
    HAVE_PROJECT_FILTER = False

MAD_SCALE = 1.4826

DEFAULT_WINDOWS = (6, 12, 24, 48, 96, 144, 288)
DEFAULT_SIGMAS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)


# --------------------------------------------------------------- filter
def hampel_mask(series: pd.Series, window: int, n_sigmas: float) -> np.ndarray:
    """
    Flag points that deviate from a rolling median by more than
    n_sigmas robust standard deviations. Uses the project's own
    implementation when importable so tuning cannot drift from what the
    GUI and CLI actually run.
    """
    if HAVE_PROJECT_FILTER:
        frame = series.to_frame("v")
        mask, _, _ = _project_hampel(frame, window=window, n_sigmas=n_sigmas)
        return mask["v"].to_numpy(dtype=bool)
    med = series.rolling(window=window, center=True, min_periods=1).median()
    dev = (series - med).abs()
    mad = dev.rolling(window=window, center=True, min_periods=1).median()
    return (dev > n_sigmas * MAD_SCALE * mad).to_numpy(dtype=bool) & series.notna().to_numpy()


# ---------------------------------------------------------- fault model
@dataclass
class Fault:
    kind: str
    start: int
    length: int
    magnitude: float  # in robust SDs of the local series


FAULT_KINDS = ("spike", "dropout", "level_shift", "flatline")


def robust_sd(values: np.ndarray) -> float:
    med = np.median(values)
    return float(MAD_SCALE * np.median(np.abs(values - med))) or float(np.std(values))


def inject(
    clean: pd.Series,
    rng: np.random.Generator,
    n_faults: int,
    magnitudes: Sequence[float],
    kinds: Sequence[str] = FAULT_KINDS,
    min_separation: int = 60,
) -> Tuple[pd.Series, np.ndarray, List[Fault]]:
    """
    Copy `clean` and corrupt it in a handful of places.

    Faults are kept apart from each other so that one fault never sits
    inside another's rolling window - otherwise a miss gets blamed on
    the parameters when it was really caused by the injection.
    """
    values = clean.to_numpy(dtype=float).copy()
    truth = np.zeros(len(values), dtype=bool)
    sd = robust_sd(values)
    faults: List[Fault] = []

    taken: List[Tuple[int, int]] = []
    attempts = 0
    while len(faults) < n_faults and attempts < n_faults * 200:
        attempts += 1
        kind = str(rng.choice(kinds))
        length = 1 if kind == "spike" else int(rng.integers(3, 10))
        start = int(rng.integers(min_separation, len(values) - length - min_separation))
        if any(start < e + min_separation and s - min_separation < start + length for s, e in taken):
            continue
        mag = float(rng.choice(magnitudes))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        sl = slice(start, start + length)

        if kind == "spike":
            values[sl] += sign * mag * sd
        elif kind == "level_shift":
            values[sl] += sign * mag * sd
        elif kind == "dropout":
            values[sl] = values[sl] * 0.05  # near-total collapse, e.g. a partial outage
            mag = float("nan")  # amplitude is not a free parameter for this kind
        elif kind == "flatline":
            values[sl] = values[start]  # meter stuck at its last reading
            mag = float("nan")

        truth[sl] = True
        taken.append((start, start + length))
        faults.append(Fault(kind, start, length, mag))

    return pd.Series(values, index=clean.index), truth, faults


# ------------------------------------------------------------- scoring
def score(flagged: np.ndarray, truth: np.ndarray, faults: List[Fault]) -> Dict[str, float]:
    """
    Precision is sample-level: of everything flagged, how much was
    actually corrupt. Recall is event-level: of the faults injected, how
    many were noticed at all. Catching one sample of a 6-sample flatline
    is a detection - you still learn the meter misbehaved there.
    """
    tp = int(np.sum(flagged & truth))
    fp = int(np.sum(flagged & ~truth))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    detected = sum(1 for f in faults if flagged[f.start:f.start + f.length].any())
    recall = detected / len(faults) if faults else float("nan")

    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = f_half = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        b2 = 0.25  # F0.5 - weights precision, i.e. not flagging real peaks
        f_half = (1 + b2) * precision * recall / (b2 * precision + recall)
    return {
        "tp": tp, "fp": fp, "precision": precision, "recall_event": recall,
        "f1": f1, "f_half": f_half, "n_flagged": int(flagged.sum()),
    }


def recall_by_magnitude(
    flagged: np.ndarray, faults: List[Fault], magnitudes: Sequence[float]
) -> Dict[float, Tuple[int, int]]:
    """Only fault kinds whose severity actually scales with magnitude."""
    out = {m: [0, 0] for m in magnitudes}
    for f in faults:
        if np.isnan(f.magnitude) or f.magnitude not in out:
            continue
        out[f.magnitude][1] += 1
        if flagged[f.start:f.start + f.length].any():
            out[f.magnitude][0] += 1
    return {m: (v[0], v[1]) for m, v in out.items()}


# ------------------------------------------------------------ analysis
def longest_clean_run(series: pd.Series) -> pd.Series:
    """Longest stretch with no missing values - the injection substrate."""
    ok = series.notna().to_numpy()
    best_len = best_start = cur_start = 0
    cur = 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, cur_start
        else:
            cur = 0
    return series.iloc[best_start:best_start + best_len]


def run_grid(
    clean: pd.Series, windows: Sequence[int], sigmas: Sequence[float],
    seeds: int, n_faults: int, magnitudes: Sequence[float], rng_seed: int,
) -> pd.DataFrame:
    rows = []
    trials = []
    for s in range(seeds):
        rng = np.random.default_rng(rng_seed + s)
        trials.append(inject(clean, rng, n_faults, magnitudes))

    for w in windows:
        for sig in sigmas:
            per_trial = []
            mags: Dict[float, List[int]] = {m: [0, 0] for m in magnitudes}
            for corrupted, truth, faults in trials:
                flagged = hampel_mask(corrupted, w, sig)
                per_trial.append(score(flagged, truth, faults))
                for m, (hit, tot) in recall_by_magnitude(flagged, faults, magnitudes).items():
                    mags[m][0] += hit
                    mags[m][1] += tot
            agg = {k: float(np.nanmean([t[k] for t in per_trial])) for k in per_trial[0]}
            agg["f_half_sd"] = float(np.nanstd([t["f_half"] for t in per_trial]))
            agg.update({"window": w, "sigma": sig})
            for m, (hit, tot) in mags.items():
                agg[f"recall_at_{m:g}sd"] = hit / tot if tot else float("nan")
            rows.append(agg)
    return pd.DataFrame(rows)


def stability_on_real(
    series: pd.Series, windows: Sequence[int], sigmas: Sequence[float]
) -> pd.DataFrame:
    """Flag counts on the untouched series - no injection, no scoring."""
    rows = []
    n = int(series.notna().sum())
    for w in windows:
        for sig in sigmas:
            flagged = hampel_mask(series, w, sig)
            rows.append({"window": w, "sigma": sig,
                         "n_flagged": int(flagged.sum()),
                         "pct_flagged": 100.0 * flagged.sum() / n})
    return pd.DataFrame(rows)


def peak_hour_fp(
    clean: pd.Series, best_w: int, best_sig: float, peak_hours: Tuple[int, int],
    seeds: int, n_faults: int, magnitudes: Sequence[float], rng_seed: int,
) -> Dict[str, float]:
    """
    Where do the false positives land? If they concentrate in the
    evening ramp, the filter is eating real peaks - the exact failure
    this project is trying to avoid.
    """
    lo, hi = peak_hours
    in_peak = ((clean.index.hour >= lo) & (clean.index.hour < hi)).astype(bool)
    peak_share = float(in_peak.mean())
    fp_in_peak = fp_total = 0
    for s in range(seeds):
        rng = np.random.default_rng(rng_seed + s)
        corrupted, truth, _ = inject(clean, rng, n_faults, magnitudes)
        flagged = hampel_mask(corrupted, best_w, best_sig)
        fp = flagged & ~truth
        fp_total += int(fp.sum())
        fp_in_peak += int((fp & in_peak).sum())
    return {
        "peak_share_of_day": peak_share,
        "fp_total": fp_total,
        "fp_in_peak": fp_in_peak,
        "fp_peak_share": fp_in_peak / fp_total if fp_total else float("nan"),
        "enrichment": (fp_in_peak / fp_total / peak_share) if fp_total and peak_share else float("nan"),
    }


def local_sensitivity(grid: pd.DataFrame, best_w: int, best_sig: float,
                      windows: Sequence[int], sigmas: Sequence[float]) -> List[str]:
    """One-at-a-time moves away from the optimum."""
    lines = []
    best = grid[(grid.window == best_w) & (grid.sigma == best_sig)].iloc[0]
    wi, si = list(windows).index(best_w), list(sigmas).index(best_sig)

    for label, idx, axis in (("window", wi, "window"), ("sigma", si, "sigma")):
        seq = windows if axis == "window" else sigmas
        for step in (-1, 1):
            j = idx + step
            if not (0 <= j < len(seq)):
                continue
            if axis == "window":
                row = grid[(grid.window == seq[j]) & (grid.sigma == best_sig)].iloc[0]
            else:
                row = grid[(grid.window == best_w) & (grid.sigma == seq[j])].iloc[0]
            d = row.f_half - best.f_half
            lines.append(
                f"  {label} {best_w if axis=='window' else best_sig:g} -> {seq[j]:g}: "
                f"F0.5 {best.f_half:.3f} -> {row.f_half:.3f} ({d:+.3f}), "
                f"precision {best.precision:.3f} -> {row.precision:.3f}, "
                f"recall {best.recall_event:.3f} -> {row.recall_event:.3f}"
            )
    return lines


# ---------------------------------------------------------------- plots
def make_plots(grid: pd.DataFrame, stab: pd.DataFrame, windows, sigmas, magnitudes,
               best_w, best_sig, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def heat(ax, pivot, title, fmt="{:.2f}", cmap="viridis"):
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{i:g}" for i in pivot.index])
        ax.set_xlabel("sigma"); ax.set_ylabel("window (samples)")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                ax.text(x, y, fmt.format(pivot.values[y, x]), ha="center", va="center",
                        fontsize=7, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046)

    heat(axes[0, 0], grid.pivot(index="window", columns="sigma", values="f_half"),
         "F0.5 on injected faults (precision-weighted)")
    heat(axes[0, 1], grid.pivot(index="window", columns="sigma", values="precision"),
         "Precision on injected faults")
    heat(axes[1, 0], stab.pivot(index="window", columns="sigma", values="pct_flagged"),
         "% flagged on the untouched series", fmt="{:.1f}", cmap="magma")

    ax = axes[1, 1]
    for w in windows:
        sub = grid[grid.window == w]
        vals = [sub[sub.sigma == best_sig][f"recall_at_{m:g}sd"].iloc[0] for m in magnitudes]
        ax.plot(magnitudes, vals, marker="o", label=f"window={w}")
    ax.axhline(0.5, color="#999", linestyle=":", linewidth=1)
    ax.set_xlabel("fault magnitude (robust SDs)")
    ax.set_ylabel("event recall")
    ax.set_title(f"Detection limit at sigma={best_sig:g}", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(f"Hampel tuning - best cell window={best_w}, sigma={best_sig:g}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")


# ----------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--column", default=None, help="series column (default: first numeric)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--seeds", type=int, default=25, help="independent injection trials")
    p.add_argument("--faults", type=int, default=30, help="faults injected per trial")
    p.add_argument("--magnitudes", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    p.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    p.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    p.add_argument("--peak-hours", type=int, nargs=2, default=[17, 21])
    p.add_argument("--seed", type=int, default=2002)
    args = p.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    col = args.column or df.select_dtypes("number").columns[0]
    series = df[col].astype(float)

    clean = longest_clean_run(series)
    if len(clean) < 2000:
        raise SystemExit(
            f"Longest gap-free run is only {len(clean)} samples - too short to inject into reliably."
        )

    grid = run_grid(clean, args.windows, args.sigmas, args.seeds, args.faults,
                    args.magnitudes, args.seed)
    stab = stability_on_real(series, args.windows, args.sigmas)

    best = grid.sort_values("f_half", ascending=False).iloc[0]
    best_w, best_sig = int(best.window), float(best.sigma)

    # cells statistically indistinguishable from the best
    tol = float(best.f_half_sd) / np.sqrt(args.seeds)
    near = grid[grid.f_half >= best.f_half - tol].sort_values("f_half", ascending=False)

    fp = peak_hour_fp(clean, best_w, best_sig, tuple(args.peak_hours),
                      args.seeds, args.faults, args.magnitudes, args.seed)

    grid.to_csv(outdir / "hampel_tuning.csv", index=False)
    make_plots(grid, stab, args.windows, args.sigmas, args.magnitudes,
               best_w, best_sig, outdir / "hampel_tuning.png")

    L: List[str] = []
    L.append("HAMPEL PARAMETER TUNING + SENSITIVITY ANALYSIS")
    L.append("=" * 62)
    L.append(f"input                : {src}")
    L.append(f"series               : {col}")
    L.append(f"readings             : {len(series):,}  ({int(series.isna().sum())} missing)")
    L.append(f"gap-free run used    : {len(clean):,} samples "
             f"({clean.index[0]:%Y-%m-%d %H:%M} to {clean.index[-1]:%Y-%m-%d %H:%M})")
    L.append(f"moments              : mean {series.mean():.3f}, sd {series.std():.3f}, "
             f"skew {series.skew():.3f}, excess kurtosis {series.kurt():.3f}")
    L.append(f"filter implementation: {'project outlier_detection.hampel_filter' if HAVE_PROJECT_FILTER else 'local fallback'}")
    L.append(f"trials               : {args.seeds} seeds x {args.faults} faults "
             f"({FAULT_KINDS}) at {args.magnitudes} robust SDs")
    L.append("")
    L.append("SCORING")
    L.append("-" * 62)
    L.append("Precision is sample-level, recall is event-level. F0.5 is the headline")
    L.append("because a false positive here means a real demand peak gets treated as")
    L.append("an error, which is more damaging to the forecast than missing a fault.")
    L.append("")
    L.append("BEST CELL")
    L.append("-" * 62)
    L.append(f"window = {best_w} samples, sigma = {best_sig:g}")
    L.append(f"  F0.5 {best.f_half:.3f} (sd across seeds {best.f_half_sd:.3f})")
    L.append(f"  precision {best.precision:.3f}   event recall {best.recall_event:.3f}   F1 {best.f1:.3f}")
    L.append(f"  flags {best.n_flagged:.0f} samples per trial on the corrupted series")
    L.append("")
    L.append(f"Top cells. * marks those within one standard error ({tol:.4f}) of the")
    L.append("best, i.e. not statistically distinguishable from it at this many trials.")
    for _, r in grid.sort_values("f_half", ascending=False).head(10).iterrows():
        star = "*" if r.f_half >= best.f_half - tol else " "
        L.append(f" {star} window {int(r.window):>3}  sigma {r.sigma:>4.1f}   F0.5 {r.f_half:.3f}   "
                 f"P {r.precision:.3f}  R {r.recall_event:.3f}   "
                 f"flags on real data: {int(stab[(stab.window==r.window)&(stab.sigma==r.sigma)].n_flagged.iloc[0])}")
    L.append(f"  ({len(near)} of {len(grid)} cells within one standard error)")
    L.append("")
    L.append("LOCAL SENSITIVITY (one parameter at a time, from the best cell)")
    L.append("-" * 62)
    L += local_sensitivity(grid, best_w, best_sig, args.windows, args.sigmas)
    L.append("")
    L.append("DETECTION LIMIT (event recall by fault size, at the best cell)")
    L.append("-" * 62)
    L.append("Spikes and level shifts only - dropouts and flatlines have no amplitude")
    L.append("parameter, so including them would flatten this curve.")
    row = grid[(grid.window == best_w) & (grid.sigma == best_sig)].iloc[0]
    for m in args.magnitudes:
        L.append(f"  {m:>4.1f} robust SD : recall {row[f'recall_at_{m:g}sd']:.3f}")
    L.append("")
    L.append("FALSE POSITIVES vs THE EVENING PEAK")
    L.append("-" * 62)
    L.append(f"  peak window {args.peak_hours[0]:02d}:00-{args.peak_hours[1]:02d}:00 "
             f"= {fp['peak_share_of_day']*100:.1f}% of the day")
    L.append(f"  false positives landing in it: {fp['fp_in_peak']}/{fp['fp_total']} "
             f"= {fp['fp_peak_share']*100:.1f}%")
    L.append(f"  enrichment vs chance: {fp['enrichment']:.2f}x")
    L.append("  (>1 means the filter preferentially misreads real peaks as errors)")
    L.append("")
    L.append("STABILITY ON THE UNTOUCHED SERIES (no injection)")
    L.append("-" * 62)
    L.append("How many points each cell flags in the real data. A parameter choice")
    L.append("that swings this wildly is one the data cannot justify.")
    L.append("")
    piv = stab.pivot(index="window", columns="sigma", values="n_flagged")
    header = "  window |" + "".join(f"{c:>8.1f}" for c in piv.columns)
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for w, r in piv.iterrows():
        L.append(f"  {w:>6} |" + "".join(f"{v:>8.0f}" for v in r.values))
    L.append("")
    at_best = stab[(stab.window == best_w) & (stab.sigma == best_sig)].iloc[0]
    L.append(f"At the chosen cell: {at_best.n_flagged:.0f} points flagged "
             f"({at_best.pct_flagged:.2f}% of the series).")
    L.append("")
    L.append("HOW TO READ THIS")
    L.append("-" * 62)
    L.append("The best cell is the best against SIMULATED faults. If the injected")
    L.append("faults do not resemble the errors actually present, the ranking does")
    L.append("not transfer. Weight the near-optimal set and the detection limit more")
    L.append("heavily than the single winning cell, and treat the flag counts on the")
    L.append("untouched series as review candidates rather than confirmed errors.")

    report = "\n".join(L)
    (outdir / "hampel_tuning_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
