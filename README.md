# Load Data Preprocessing & Visualisation Pipeline

A one-stop desktop tool for energy time-series: import a household load
panel, a pre-aggregated demand series, or a solar GHI/temperature
record; visualise it and its statistics; assess data quality; detect
anomalies; and clean the data ready for a downstream ML forecasting
task.

**Nothing is ever overwritten.** Each step appends an immutable layer
(Raw → Sentinels → Interpolated → Aggregated → Gaps Filled → Normalised),
**Revert** moves a
pointer rather than restoring a backup, and the file on disk is never
written to.

## Setup

Requires Python 3.9+ with Tkinter (bundled with the standard python.org
installers on Windows/macOS; on Linux you may need to install it
separately, e.g. `sudo apt install python3-tk` on Debian/Ubuntu).

```bash
pip install -r requirements.txt
python app.py
```

## Layout

- **Toolbar** — Import, the file/kind badge, **Layers…**, and
  **Data Properties** (available in every state, for every file type).
- **Pipeline sidebar** — the six steps, in order, each with a status
  glyph: `✓` done, `●` available, `○` blocked, `—` not applicable,
  `⟳` running. **A step is never hidden.** One that does not apply to
  the loaded file stays on screen, greyed, and says *why* — a card that
  disappears reads as a bug. Blue buttons are the legal next actions;
  a disabled step is visibly grey, not blue.
- **Canvas** — the plot, with the series/date/quick-range bar above it
  and a layer breadcrumb showing which layer is being drawn.
- **Inspector** — the Data Properties panel (right, collapsible).
- **Report modals** — a step's report halts the whole UI until you
  choose **Save report…** or **Okay**. Saving does *not* dismiss.
  Dismissing is what commits the result and refreshes the canvas, so
  the data on screen and the numbers in the report can never disagree.

## Using the app

1. **Import File…** — select a `.xlsx`, `.xls`, or `.csv`. The loader
   runs two stages. The *structural* stage decides how to parse (header
   row or not, which reader). The *semantic* stage then labels every
   column with a role and the file with a kind, and that is what the
   sidebar gates on — never the file extension:
   - **Household panel** (e.g. `Westridge_2002_dupe.xlsx`): headerless,
     first column a timestamp, the rest one household's active power
     each. Columns show up as `Household_01`, `Household_02`, …
   - **Aggregate series** (e.g. `aggregate_total_load.csv`): a header
     row plus one value column. Steps 2 and 3 are marked not-applicable.
   - **Environmental** (e.g. `Stellies_2023.xlsx`): GHI and/or ambient
     temperature. The low-pass filter (step 5) and Fill Gaps (step 4)
     apply; sentinels, aggregation and Z-score detection do not.

   Classification uses column names first, then structure, then a
   *physical fingerprint* of the values — which can overrule the other
   two, because values are harder to get wrong than labels. The
   discriminator is a day/night shape test: irradiance is dark for a
   contiguous block of hours every day and never dark at midday. A
   plain "mostly zeros" test would not do: `Household_43` in the
   Westridge export is within 5 kW of zero for 97.2% of its samples.
   When the fingerprint contradicts a column's own name, the app says
   so in the status bar rather than silently overruling the file.

   A sentinel scan runs automatically in the background on import, so
   step 2 tells you what is there before you commit to anything.

2. **Clean Sentinels & Interpolate** — replaces the sentinel
   codes with NaN, then interpolates the holes, leaving gaps longer
   than **Max gap** deliberately empty. These are two independent
   programs run as a chain behind one button; the report has a section
   for each. The report also lists **candidate codes** — values
   repeated far more often than a real analogue reading should be, at
   an extreme of the range — so a file using a code you did not
   configure does not sail through looking clean.

3. **Aggregate** — stays disabled until the step 2 report is
   acknowledged, then sums the panel into one total-demand series.
   Timestamps where any household is still missing are left unknown
   rather than summed from the rest, which would silently undercount.
   The output is the *same kind* as a directly-imported CSV, so
   everything downstream behaves identically either way.

4. **Data Properties** (toolbar) — statistics for the plotted window
   and channel: distribution type, mean, median, min, max, plus
   coverage, spread, shape and energy. Distribution type is answered by
   fitting six families by MLE and ranking them by AIC, reporting the
   winner, the margin, and a KS distance. Three guards keep it honest:
   a winner within ΔAIC 2 is reported as *indistinguishable*; a winner
   with KS D > 0.10 is reported as *no standard family fits well*
   rather than dressed up as an answer; and a point mass at the lower
   bound is detected **before** fitting — GHI is dark for half of every
   day, and no continuous distribution describes that. That last test
   is deliberately not an exact-zero test: after a zero-phase filter,
   exact zeros in the sample record fall from 25.3% to 0.02% while
   49.8% still sit within 1% of peak, so equality would miss the point
   mass entirely and confidently report "uniform".
5. **Series** — pick which series to plot from the dropdown (only one
   option for an aggregate file).
6. **Start / End** — set the date range by typing/using the calendar
   pickers directly, or with the quick-range controls:
   - **Full Range** — the entire loaded dataset
   - **1 Week / 1 Day / 1 Month** — sized relative to whatever start
     date is currently showing (not always the very first day of the
     file), so you can use them to resize the window you're already
     looking at. **1 Month** is a real calendar month (28-31 days,
     whatever the anchor month actually has), not a fixed 30-day block.
   - **◀ Prev / Next ▶** — page backward/forward through the data by
     the length of the current window (e.g. viewing a week and
     clicking Next jumps to the following week). While the window is a
     **1 Month** span, Prev/Next page by a real calendar month instead
     of a fixed day count, so paging through a multi-month file lands
     on true month boundaries (Jan 31 → Feb 28/29 → Mar 31, ...)
     instead of drifting a day or two each page. Switching to another
     preset, or editing the dates and clicking Plot directly, drops
     back to plain day-length paging until 1 Month is pressed again.
   Quick-range buttons redraw the chart immediately. Typing a custom
   date still needs a click on **Plot**.
7. **View** (dropdown next to Plot) — switches what the chart area
   shows for the selected series/range:
   - **Time Series** (default) — the active-power line over time, with
     zoom/pan via the toolbar and hover tooltips showing exact
     timestamp + kW value.
   - **Distribution** — a histogram of the readings in the selected
     window, plus a small stats card (n, mean, median, std, min/max,
     skew, kurtosis, and a missing-count if any). Useful for spotting
     how skewed/spiky a series is before deciding on outlier
     thresholds. Outlier circling doesn't apply to a histogram, so
     Detect Outliers/Clear are disabled while this view is showing and
     re-enable automatically when you switch back.
   Changing the dropdown re-renders immediately with whatever
   series/range is currently selected; Plot, the quick-range presets,
   and Prev/Next all respect whichever view is currently chosen.
8. **Plot** — (re)draws the selected series/range in the current view.
   The Time Series view supports:
   - Zoom/pan via the toolbar under the chart
   - Hover tooltips showing exact timestamp + kW value
9. **Detect Outliers** (sidebar step 4; enabled once a Time Series is plotted) —
   circles flagged readings in red using a **rolling Z-score**: a reading
   more than σ rolling standard deviations from its rolling mean. Both
   controls are editable:
   - **Window (samples)** — the centred rolling window. 12 samples is
     one hour at 5-minute data; the status bar echoes the equivalent
     duration so you can see what you picked.
   - **Threshold (σ)** — how far from the local centre counts as an
     outlier.

   Detection always runs over the *entire* series and then draws only
   the flags inside the visible range, so results don't shift depending
   on where you happen to be scrolled. **Clear** removes the overlay,
   and re-detecting replaces it rather than stacking another on top.
   Nothing in your data is modified: flagged points are review
   candidates, not confirmed errors.

10. **Clean Data** (enabled as soon as a file is loaded) — pops the
   detected outliers and fills the holes.

   Each flagged reading is set to NaN so it takes no part in anything
   downstream, then a penalised B-spline is fitted to the readings
   *around* the hole and the replacement is read off that curve — the
   imputation half of Chen et al. (2010). Fits are local (an hour of
   context either side, widening with the size of the hole) rather than
   one global fit, which would be both too coarse to trace a daily
   profile and computationally impossible over eight months.

   Runs on the whole series, not just the visible window — you're
   cleaning a dataset, not a view. The loaded data is replaced in memory
   so every later plot shows the cleaned series; **Revert** puts the
   original back, and the file on disk is never touched.

   A run of flagged readings longer than the max gap (24 samples = 2
   hours) is deliberately **not** filled — a smooth guess across a long
   outage manufactures data that looks real. Those stay NaN and the
   report says so.

   The **cleaning report** covers detection settings and counts, what
   was imputed versus left empty and why, before/after statistics, the
   change in total energy, and the largest individual corrections. It
   also runs a **hold-out test on your own data**: readings that were
   never flagged are hidden, imputed, and compared against their true
   values, with linear interpolation scored on the same points as a
   baseline — so the error attached to the filled-in values is measured
   rather than assumed. **Save cleaned data + report...** writes the
   cleaned series and the report side by side.

11. **Calibrate...** (enabled as soon as a file is loaded) — picks the
   window and threshold for you, and opens a sensitivity report.

   Since the data has no labelled faults, calibration works by
   **synthetic fault injection**: it plants faults of known type, size
   and position into a copy of the series (isolated spikes/dips sized
   in local σ, plus dropouts collapsing 1–3 readings to near zero),
   sweeps the window × σ grid, and scores every cell by F1 against the
   known positions. The winning cell is written straight into the two
   parameter boxes; the report window shows the whole surface, so you
   can see how sharp or flat the optimum is rather than trusting one
   number.

   The report also breaks recall down **by fault size** — how large a
   fault has to be before the detector reliably catches it — which is
   the more useful number for a write-up than a single F1. Two things
   it deliberately surfaces rather than hides: the result is only
   optimal for the fault types injected, and precision is a lower bound
   because real anomalies already in the series count as false
   positives (the `baseline_flags` column shows how many).

   Runs on a worker thread with progress in the status bar; a sweep over
   eight months of 5-minute data takes about a second.

   **The window caps the threshold.** A reading helps set the mean and
   standard deviation of its own window, so its Z-score can never
   exceed `(w−1)/√w` — only 3.18 at a 12-sample window. Ask for 5σ
   there and *nothing* can ever be flagged, which would look like clean
   data. The app checks this and tells you the minimum window that
   would work instead of silently returning zero.

   **A sustained level shift is invisible to it.** If a block of readings
   is wrong by a constant, the rolling mean moves with it and the
   interior never deviates. Measured on the Stellenbosch irradiance data,
   where a two-day sensor fault held GHI near 2 975 W/m² including
   through the night: 0 of 713 non-physical readings were flagged, at
   every window and threshold tried. Worse, masking means a *longer*
   corrupted run is caught *less* often — 3 samples at ×20 gets all
   three flagged, 10 samples at ×20 gets none. For faults of that shape
   use the low-pass filter's valid range, not the detector.
12. **Low-Pass Filter** (step 5 in the sidebar) — a zero-phase Butterworth for
   fast-sampled sensor data such as irradiance.

   - **Cutoff (minutes)** — fluctuations faster than this are
     attenuated. Default **30 minutes**, and that number is measured
     rather than guessed: on this GHI only **7%** of the daytime
     fluctuation variance sits at periods below 5 minutes, against 24%
     below 15 and 38% below 30. Cloud shadows take minutes to *tens* of
     minutes to cross, so a short cutoff barely reaches them — a
     5-minute setting looks like it is doing almost nothing because it
     structurally cannot do much. Past ~30 minutes the filter starts
     overshooting at the sunrise/sunset edges (1,976 negative readings
     at 60 minutes), which is what caps the useful range.
   - **Order** — filter steepness, 1–10. Default 3; raising it barely
     moves the result (order 5 changes the statistics by 1–3%). The
     cutoff is the lever, not the roll-off.
   - **Valid range** — hard physical limits. Readings outside are
     screened to NaN *before* filtering and stay NaN.
   - **Presets** — **GHI** fills in 0–1400 W/m² with a 30-minute cutoff;
     **Temp** fills in −20 to 60 °C with a 15-minute cutoff. Temperature
     is far smoother to begin with (0.09 °C minute-to-minute against
     GHI's 35.6 W/m²) and PV cell temperature lags ambient by several
     minutes of thermal inertia anyway, so sub-15-minute detail never
     reaches the cell.

   Applied forwards and backwards (`filtfilt`), so nothing is
   time-shifted — a lagged irradiance series would misalign PV output
   against load. The cost is that it is non-causal: fine for
   conditioning a historical record, unusable in real time.

   **Screen before you smooth.** A linear filter cannot remove an
   impossible reading; it spreads it over the filter window. Set the
   valid range for any series with hard physical limits.

   Note what filtering does and doesn't claim: on irradiance the fast
   variation is mostly *real* cloud movement, not measurement error.
   Removing it is defensible because a spatially distributed PV array
   smooths irradiance the same way — it is a plant-output
   approximation, not noise removal. Say that rather than calling the
   raw readings wrong.

   **The filter is even-handed; the signal is not.** Cloud shades far
   more than it enhances — on this record 65% of daytime readings sit
   below 90% of clear sky, only 3% above 105%, and deviations from clear
   sky have skew −1.26. A symmetric filter therefore has an asymmetric
   *effect*: at a 30-minute cutoff it pulls local peaks down ~28 W/m²
   and fills local troughs up ~35 W/m², while leaving the mean and the
   total energy untouched (both change by <0.05%). That is not a bug —
   it is what a distributed array physically does, since a cloud edge
   reaches part of the array before the rest. The report quantifies it
   on every run under **CONDITIONAL EFFECT** so it can be quoted rather
   than hidden. If point-irradiance fidelity matters more than plant
   realism, shorten the cutoff: the redistribution scales with it
   (±5 W/m² at 5 minutes, ±45 at 60).

   Like Clean Data, this replaces the series in memory; **Revert**
   restores the original and the file on disk is never touched.

13. The status bar at the bottom reports how many readings were
   plotted (or summarised, in Distribution view), and flags any
   missing values in the selected window.

## Included sample data

`Westridge_2002_dupe.xlsx` is included so you can try the app
immediately: 66 households, 5-minute active-power readings from
2002-02-01 00:00 to 2002-02-28 23:55 (8,064 rows). Run it through
`clean_sentinels.py` and then `interpolate_and_aggregate.py` to get a
cleaned per-household file and the aggregate series, both of which
open directly in this app too.

## Project structure

- `data_loader.py` — pure-pandas *structural* loading
  (`load_load_profile`, `LoadProfileData`): file type and header
  layout. No GUI dependencies, so every script reuses it directly.
- `dataset.py` — the *semantic* stage on top of it: `ChannelRole`,
  `DatasetKind`, `Channel`, `Dataset` and the three-signal classifier.
  `python dataset.py <file>` prints what it made of a file.
- `layers.py` — the immutable `LayerStack` and the change ledger.
  Whole frames are kept deliberately: measured on the 8-month panel
  (72,576 × 66) a layer is 38.9 MB and 0.131 s, so four layers cost
  ~156 MB — far too cheap to justify lazy evaluation. What is *not*
  a whole frame is the record of what changed: sentinel replacement
  touches 0.162% of cells, so a long-format ledger (35 KB against a
  38.9 MB frame) carries the before/after/reason for every altered
  reading.
- `pipeline.py` — `SentinelStep`, `InterpolateStep`, `AggregateStep`
  and `run_chain`, wrapping the pure functions in the two CLI scripts.
  Splitting step 2 into two buttons is a change to one list; neither
  implementation is touched.
- `statistics.py` — the Data Properties engine: window statistics, the
  AIC ranking, the point-mass and poor-fit guards.
- `widgets.py` — `StepCard` and `ReportModal`.
- `data_cleaning.py` also provides **`impute_missing()`** — the refill
  half of cleaning, with no detection step, for holes that are already
  in the series. This is what irradiance needs. Runnable on its own.
- `normalisation.py` — min-max scaling: `fit`, `transform`,
  `inverse_transform`, `save_params`/`load_params`. Runnable on its own:
  `python normalisation.py aggregate_total_load.csv 0.7`.
- `theme.py` — the light-appearance lock (see **Appearance**).
- `outlier_detection.py` — `rolling_zscore`, plus
  `max_achievable_zscore` / `min_window_for_zscore` and
  `robust_local_scale` (a rolling MAD kept only for sizing synthetic
  faults during calibration). Reused directly by the CLI script and the
  GUI's Detect Outliers button, so the two always agree.
- `signal_filters.py` — the zero-phase Butterworth low-pass and the
  physical-range screening behind section 4. Runnable on its own:
  `python signal_filters.py Stellies_2023.xlsx 5`.
- `data_cleaning.py` — popping outliers, local B-spline imputation, the
  hold-out accuracy test and the cleaning report behind the **Clean
  Data** button. Runnable on its own:
  `python data_cleaning.py aggregate_total_load.csv zscore 48 3.0`.
- `calibration.py` — synthetic fault injection, the (window, σ) grid
  sweep and the sensitivity surface behind the **Calibrate...** button.
  Runnable on its own: `python calibration.py aggregate_total_load.csv`
  prints the recommendation, the recall-by-fault-size table and the full
  F1 grid.
- `bspline_detection.py` — the Chen et al. (2010) penalised B-spline
  cleansing method: fits a smooth curve, flags readings outside its
  prediction interval, and supplies the fitted value as a replacement.
  Evaluated (see the project notes) but **not** wired into the GUI;
  usable directly from a script.
- `app.py` — the Tkinter GUI. Widget enablement is a **pure function of
  state**, computed only in `_refresh_enablement()`; no handler sets a
  pipeline control's state directly. (The previous build set widget
  state at 48 sites across 14 methods, which is where gating bugs live.)
  Workers only read layers and return results through a queue; only the
  main thread appends to the stack or touches a widget, and because
  layers are immutable there is nothing to race on.
- `clean_sentinels.py`, `interpolate_and_aggregate.py` — the remaining
  preprocessing pipeline stages (see the project notes for what each
  does and why).

## Known limitations

- **Importing a large `.xlsx` is the only slow thing in this app.**
  Parsing the 8-month panel takes ~55 s, against 0.13 s to interpolate
  all 66 columns and 0.007 s to aggregate — roughly 400× the whole
  pipeline. It runs on a worker thread so the window stays alive, but
  there is no import cache yet; adding one would turn a re-open into
  ~0.16 s.
- Outlier detection is **not** hard-gated behind aggregation. It works
  on any single power channel, because it always has and removing that
  would be a regression dressed up as guidance — the step 4 card points
  you at aggregating first instead.
- Solar and load sit on different sampling grids (1 min vs 5 min).
  Nothing here resamples them onto a common index yet, which is the
  next thing needed before PV output can be joined to load.
- Date range is day-granularity (a selected day always spans its full
  00:00–23:55). Sub-day custom windows aren't exposed in the UI yet.
- Only one series can be plotted at a time; multi-series comparison
  (e.g. overlaying two households) isn't built yet.
- Cleaning handles isolated spikes/dips and short dropouts. It cannot
  see a sustained **level shift** — see the note under Detect Outliers.
  The low-pass filter's valid range catches the physically impossible
  case; anything else needs a run-aware or changepoint method, which
  isn't built.
- The Hampel filter was removed. The project settled on the rolling
  Z-score for load data; the rolling MAD survives only inside
  calibration, where a robust scale is needed to size injected faults.
- The Z-score's flag rate is held down by masking: one large outlier
  inflates the standard deviation of its own window. Worth stating
  explicitly when reporting results, and quantified in the project notes.
- The Distribution view's stats card is numbers-only (no boxplot yet),
  and only summarises whichever single series/range is currently
  selected — no side-by-side comparison across households.

## Tests

Headless GUI tests run under Xvfb (`xvfb-run -a python3 <suite>.py`).
Eight suites, 373 assertions, all passing:

| Suite | Assertions | Covers |
|---|---|---|
| `smoke_test_pipeline` | 81 | ingestion of all three file shapes, the gating sequence, both report modals, layers, the inspector |
| `smoke_test_filter` | 67 | Butterworth screening/filtering and its report |
| `smoke_test_cleaning` | 47 | pop-and-refill, imputation accuracy, masking, the long-gap safeguard |
| `smoke_test_zscore` | 32 | rolling Z-score, the reachability ceiling, presets |
| `smoke_test_calibration` | 26 | fault injection, the F1 sweep, the worker/queue path |
| `smoke_test_theme` | 21 | the light-appearance lock, contrast, the dark-mode code path |
| `smoke_test_normalisation` | 51 | train-split fitting, leakage, reversibility, saved parameters, the card |
| `smoke_test_gapfill` | 48 | B-spline gap filling, the lambda tuning, night/clamp rules, the card |

## Filling gaps (step 4, "Fill Gaps")

**Clean Data does not work on irradiance, and cannot.** It detects with
a rolling Z-score first, and a Z-score cannot see an irradiance sensor
fault — on the real record it flags *none* of the 713 impossible
readings, because they are not statistical outliers of a rolling window,
they are physically impossible values. What removes them is the low-pass
filter's **valid range**, which screens them to NaN. That leaves holes.

**Fill Gaps is the other half**: it refits holes that are already in the
series from a local penalised B-spline, with no detection step at all.
It works on any channel — solar or load — and only touches readings that
were already missing.

Three rules keep the fill physically honest:

- **Runs longer than Max gap are left alone.** On a 30-day 1-minute GHI
  record with sensor faults, 622 of 636 holes are a single sample and 13
  are two samples — trivial to fill. The remaining one is a 2,880-sample
  (two-day) outage, which is left as a hole. Inventing two days of
  irradiance with nothing nearby to fit against is worse in a training
  set than a visible gap.
- **Fitted values are clamped to the channel's physical range.** A
  B-spline through a steep sunrise can overshoot below zero, which is not
  a possible irradiance reading.
- **Holes entirely inside a dark period are set to exactly zero**, not
  splined. Irradiance at night *is* zero; a curve fitted through darkness
  only adds wiggle and can dip negative. Darkness is detected from the
  surrounding readings, so this needs no site coordinates and no extra
  dependency.

### The smoothing parameter is not the one used for load

`DEFAULT_IMPUTE_LAMBDA = 1.0` was tuned on unfiltered 5-minute aggregate
load. Filtered 1-minute irradiance is a different problem: the low-pass
has already removed the high-frequency content, so the spline only has to
follow a smooth curve and any extra smoothing penalty hurts. Measured on
30 days of 1-minute GHI filtered at a 30-minute cutoff, mean absolute
error against the hidden truth:

| Gap length | λ = 0.1 | λ = 1.0 | Linear interp. |
|---|---|---|---|
| 1 sample | **0.011** | 0.093 | 0.081 |
| 5 samples | **0.136** | 0.619 | 0.493 |
| 15 samples | **1.744** | 3.810 | 3.409 |
| 24 samples | **5.477** | 8.666 | 8.022 |

All in W/m², against a series standard deviation of 293. At λ = 0.1 the
B-spline wins at every gap length; at λ = 1.0 it *loses to linear
interpolation* at every one. `DEFAULT_GAPFILL_LAMBDA` is therefore 0.1,
and **Gap λ** is exposed in the card because a different sampling rate or
cutoff will move the optimum. Every run also reports a measured hold-out
comparison against linear, so if the tuning is wrong for your data the
report says so rather than letting you assume.

## Normalisation (step 6)

Min-max scaling, `x_scaled = (x - min) / (max - min) * (hi - lo) + lo`.
The formula is trivial; three things about it are not.

**The bounds are fitted on the training split only.** If min and max are
computed over the whole record, the largest value the model will ever be
tested on has already influenced the scaling of the data it trained on —
the model never sees a value above 1.0 in training, never learns the
series can exceed what it has seen, and validation error comes out
optimistically biased. The **Train split %** box sets how much of the
record fits the scaler (70% by default); those parameters are then
applied to everything. The train/test boundary is drawn on the chart as
a dashed red line, so the split is visible rather than taken on trust.

**Held-out values can land outside the range, and are not clipped.** A
later peak above the training maximum scales above 1.0. That is not a
bug — it is the honest signal that the fit window did not contain the
full dynamic range. The report counts these excursions per channel and
gives the worst one. On the 8-month aggregate at a 70% split, 4 of
20,909 held-out readings fall outside [0, 1], the worst by 0.0026.

**The saved parameters are the deliverable, not just the scaled data.**
A forecast produced on normalised data comes back in normalised units
and is meaningless until mapped back to kW. **Save report…** writes
three files: the report, the scaled data as CSV, and
`*_scaler.json` — the exact per-channel min/max, the fit window, and
both the forward and inverse formulas. Use that file (never a refit) to
scale future data and to invert model output:

```python
from normalisation import MinMaxScaler, transform, inverse_transform
scaler = MinMaxScaler.load_params("normalisation_report_scaler.json")
X = transform(new_data, scaler)          # same scaling the model learnt
kw = inverse_transform(model_output, scaler)   # back to real units
```

Other behaviour worth knowing:

- **Normalise a cleaned layer.** Min-max is driven entirely by the two
  most extreme readings, so one surviving spike sets the maximum and
  squashes every real reading into a narrow band. The card warns when
  the active layer is still `Raw`, and asks for confirmation.
- A **constant channel** (max == min) is mapped to the bottom of the
  range and flagged in the report rather than dividing by zero — a flat
  channel is almost always a dead sensor.
- Missing values stay missing, through both the transform and the
  inverse.
- Normalised channels keep their role but lose their unit and physical
  limits, so the inspector will not report kWh for unitless data and the
  filter cannot clamp 0–1 values to 0–1400 W/m².
- Reversibility is **verified, not assumed**: every run round-trips the
  data and reports the largest error (2.3e-16 on the aggregate series).

## Appearance

The app is **locked to a light appearance** (`theme.py`). On macOS, Tk's
native `aqua` theme follows the system into Dark Mode, but the report
panes, the statistics inspector and the chart are all drawn light — the
result is not a dark app, it is a broken one. The worst case is a
`tk.Text` given a light background and no explicit foreground: it
inherits the system's *white* text colour, so reports render
white-on-white and read as blank.

Two things make the lock work:

- When the OS is in dark mode the app switches off `aqua` (which ignores
  colour options entirely) in favour of `clam`, whose colours it
  controls. In light mode `aqua` is left alone, so the app still looks
  native. The status bar says when this happened rather than doing it
  silently.
- Every classic Tk widget gets an explicit foreground **and** background
  through the option database, so nothing inherits a system colour.

To follow the OS instead, run with `LOAD_PROFILE_THEME=system` — but
note there is no dark palette for the charts or report panes, so parts
of the window will be light regardless. A proper dark theme is not
built.

`smoke_test_theme.py` walks the whole widget tree and fails on any
classic widget that sets one colour without the other, on any dark
ground, or on any foreground/background pair whose luminance difference
is under 0.30.
