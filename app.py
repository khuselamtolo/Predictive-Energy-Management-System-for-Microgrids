"""
Load Data Preprocessing & Visualisation Pipeline
================================================
A one-stop desktop tool for energy time-series: import a household load
panel, a pre-aggregated demand series, or a solar GHI/temperature
record; look at it; clean it; and get it ready for a downstream ML
forecasting task.

Layout
------
A toolbar across the top, a numbered pipeline down the left, the plot in
the middle, and a statistics inspector on the right. The five sidebar
cards are the pipeline, in order, and they are never hidden - a step
that does not apply to the loaded file stays on screen, greyed, and says
why. Exactly one accent-coloured button exists at any moment: the next
legal action.

State
-----
The app's state is (stage, overlay, busy, modal), and widget enablement
is a PURE FUNCTION of it, computed in one place - _refresh_enablement().
No handler ever calls configure(state=...) on a pipeline control. That
one rule is what makes "Aggregate stays disabled until the interpolation
report is acknowledged" a structural property rather than something six
different handlers have to remember.

Data
----
Nothing is ever overwritten. Every step appends an immutable layer to a
LayerStack (Raw → Sentinels → Interpolated → Aggregated), Revert moves a
pointer rather than restoring a backup, and the file on disk is never
written to. See layers.py for why whole frames are affordable here.

Run with:
    python app.py

Requires: pandas, openpyxl, matplotlib, scipy, mplcursors, tkcalendar
(tkinter ships with Python, but on Linux you may need your distro's
python3-tk package - see README.md).
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

import statistics as stats_engine
from calibration import calibrate
from clean_sentinels import DEFAULT_SENTINELS
from data_cleaning import (DEFAULT_DARK_THRESHOLD, DEFAULT_GAPFILL_LAMBDA,
                           DEFAULT_MAX_GAP as DEFAULT_GAP_LIMIT, clean_series,
                           impute_missing)
from data_loader import LoadProfileData, load_load_profile
from dataset import ChannelRole, Dataset, DatasetKind, from_load_profile, load_dataset
from interpolate_and_aggregate import DEFAULT_MAX_GAP
from layers import LayerStack, empty_ledger, ledger_from_mask
from normalisation import (DEFAULT_FIT_FRACTION, DEFAULT_RANGE, normalise)
from outlier_detection import DEFAULT_WINDOW as OUTLIER_WINDOW
from outlier_detection import (max_achievable_zscore, min_window_for_zscore,
                               rolling_zscore)
from pipeline import (AggregateStep, InterpolateStep, SentinelStep,
                      scan_sentinels, run_chain)
from theme import PALETTE, apply_light_theme, style_axes
from signal_filters import (DEFAULT_CUTOFF_MINUTES, DEFAULT_ORDER,
                            GHI_VALID_RANGE, TEMP_CUTOFF_MINUTES,
                            TEMP_VALID_RANGE, lowpass)
from widgets import (ACCENT, ACCENT_ACTIVE, MUTED_TEXT, ReportModal, StepCard,
                     accent_button, set_enabled)

PRESET_SPANS = {"1 Week": 7, "1 Day": 1}  # days; "Full Range" and "1 Month" are special
PRESETS = ("Full Range", "1 Month", "1 Week", "1 Day")
VIEW_OPTIONS = ("Time Series", "Distribution")

METHOD_ZSCORE = "Rolling Z-score"
DEFAULT_SIGMA = "3.0"
FILTERED_COLOR = "#1f2937"
OUTLIER_COLOR = "#dc2626"


class Stage(Enum):
    """Pipeline progress. Monotonic, except that Revert can walk it back."""

    EMPTY = "empty"
    LOADED = "loaded"
    SENTINELS_HANDLED = "sentinels_handled"
    AGGREGATED = "aggregated"


class Overlay(Enum):
    """Free, reversible analysis state. Never gates a pipeline step."""

    NONE = "none"
    OUTLIERS_MARKED = "outliers_marked"
    OUTLIERS_CLEANED = "outliers_cleaned"
    FILTERED = "filtered"


@dataclass(frozen=True)
class AppState:
    stage: Stage = Stage.EMPTY
    overlay: Overlay = Overlay.NONE
    busy: Optional[str] = None    # name of the running operation, or None
    modal: bool = False           # a report window holds grab_set()


class LoadProfileApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Load Data Preprocessing & Visualisation Pipeline")
        self.geometry("1500x880")
        self.minsize(1120, 700)
        self.option_add("*Font", "Helvetica 10")

        # Lock the light appearance BEFORE any widget is built. On macOS
        # the native ttk theme follows the system into dark mode, while
        # the report panes and the chart are drawn light - the result is
        # not a dark app but a broken one. See theme.py.
        self._theme = apply_light_theme(self)

        self.state_: AppState = AppState()
        self.stack: Optional[LayerStack] = None
        self._sentinel_scan: Optional[dict] = None
        self._stats_cache = stats_engine.StatsCache()

        self._cursor = None                 # keeps the mplcursors hover object alive
        self._outlier_scatter = None        # the red-circle overlay artist, if any
        self._worker_queue: "queue.Queue" = queue.Queue()
        self._last_calibration = None
        self._last_cleaning = None
        self._last_filter = None
        self._last_scaler = None            # most recent MinMaxScaler, for the split marker
        self._last_plot: Optional[dict] = None
        self._month_mode = False            # True while the window came from "1 Month"
        self._inspector_open = False
        self._pending_step = None           # a staged layer awaiting the modal's dismissal

        self._build_layout()
        self._refresh_enablement()

    # ================================================================ state
    @property
    def data(self) -> Optional[Dataset]:
        """The active layer. Exposes the LoadProfileData surface, so every
        analytical module keeps working unchanged."""
        return self.stack.dataset if self.stack is not None else None

    @property
    def kind(self) -> Optional[DatasetKind]:
        return self.data.kind if self.data is not None else None

    def _transition(self, **changes) -> None:
        """The ONLY way state changes. Enablement follows automatically."""
        self.state_ = replace(self.state_, **changes)
        self._refresh_enablement()
        self._refresh_status_strip()

    # =============================================================== layout
    def _build_layout(self) -> None:
        self._build_toolbar()

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 4))

        self.sidebar = ttk.Frame(body, width=312)
        body.add(self.sidebar, weight=0)
        self._build_sidebar()

        centre = ttk.Frame(body)
        body.add(centre, weight=1)
        self._build_centre(centre)

        self.inspector = ttk.Frame(body, width=320)
        self._build_inspector()
        self._body_pane = body

        opening = "Import a file to begin."
        if self._theme.get("system_dark") and self._theme.get("applied"):
            opening += "  (Your system is in dark mode; the app stays light on purpose.)"
        self.status_var = tk.StringVar(value=opening)
        ttk.Label(self, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=(8, 4)).pack(side="bottom", fill="x")

    # -- toolbar -----------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(side="top", fill="x")

        self.import_button = accent_button(bar, "Import File…", self._on_import)
        self.import_button.pack(side="left")

        self.file_label_var = tk.StringVar(value="No file loaded yet.")
        ttk.Label(bar, textvariable=self.file_label_var,
                  foreground=MUTED_TEXT).pack(side="left", padx=(14, 0))

        # Persistent across every state and every dataset kind, which is
        # why it lives on the toolbar rather than in the pipeline sidebar
        # - it is not a pipeline step.
        self.properties_button = ttk.Button(bar, text="ⓘ  Data Properties",
                                            command=self._toggle_inspector)
        self.properties_button.pack(side="right")

        self.history_button = ttk.Button(bar, text="Layers…", command=self._show_history)
        self.history_button.pack(side="right", padx=(0, 8))

        ttk.Separator(self, orient="horizontal").pack(side="top", fill="x")

    # -- sidebar -----------------------------------------------------------
    def _build_sidebar(self) -> None:
        ttk.Label(self.sidebar, text="PIPELINE", font=("Helvetica", 9, "bold"),
                  foreground=MUTED_TEXT).pack(side="top", anchor="w", padx=10, pady=(10, 2))

        # Five cards plus their explanatory summaries do not fit a short
        # window, and a clipped card is worse than no card - so the
        # column scrolls rather than silently losing step 5 off the
        # bottom.
        outer = ttk.Frame(self.sidebar)
        outer.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, width=300,
                           background=PALETTE["bg"])
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._cards_frame = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self._cards_frame, anchor="nw")
        self._cards_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(seq, lambda e, c=canvas: self._scroll_sidebar(e, c))
        self._sidebar_canvas = canvas

        def card(number: int, title: str) -> StepCard:
            holder = ttk.Frame(self._cards_frame, relief="solid", borderwidth=1)
            holder.pack(side="top", fill="x", padx=8, pady=4)
            c = StepCard(holder, number, title)
            c.pack(fill="both", expand=True)
            return c

        # ---- 1. Import ---------------------------------------------------
        self.card_import = card(1, "Import")
        self.card_import.set_state("available", "Excel or CSV. Household panel, aggregate series, or solar.")

        # ---- 2. Sentinels & interpolate ----------------------------------
        self.card_sentinels = card(2, "Sentinels & Interpolate")
        self.sentinel_var = tk.StringVar(value=", ".join(f"{s:g}" for s in DEFAULT_SENTINELS))
        self.max_gap_var = tk.StringVar(value=str(DEFAULT_MAX_GAP))
        self.card_sentinels.add_param("Codes", self.sentinel_var, width=9, row=0, column=0)
        self.card_sentinels.add_param("Max gap", self.max_gap_var, width=4, row=0, column=1)
        self.clean_sentinels_button = self.card_sentinels.set_action(
            "Clean Sentinels & Interpolate", self._on_clean_sentinels)

        # ---- 3. Aggregate ------------------------------------------------
        self.card_aggregate = card(3, "Aggregate")
        self.aggregate_button = self.card_aggregate.set_action("Aggregate", self._on_aggregate)

        # ---- 4. Outliers -------------------------------------------------
        self.card_outliers = card(4, "Repair  ·  outliers & gaps")
        self.window_var = tk.StringVar(value=str(OUTLIER_WINDOW))
        self.sigma_var = tk.StringVar(value=DEFAULT_SIGMA)
        self.gap_lambda_var = tk.StringVar(value=f"{DEFAULT_GAPFILL_LAMBDA:g}")
        self.gap_max_var = tk.StringVar(value=str(DEFAULT_GAP_LIMIT))
        self.card_outliers.add_param("Window", self.window_var, width=5, row=0, column=0)
        self.card_outliers.add_param("σ", self.sigma_var, width=5, row=0, column=1)
        self.card_outliers.add_param("Gap λ", self.gap_lambda_var, width=5, row=1, column=0)
        self.card_outliers.add_param("Max gap", self.gap_max_var, width=5, row=1, column=1)
        self.detect_outliers_button = self.card_outliers.set_action(
            "Detect Outliers", self._on_detect_outliers)
        self.clear_outliers_button = self.card_outliers.add_button("Clear", self._on_clear_outliers)
        # A fixed 2-column grid rather than a packed row: ttk themes size
        # buttons differently (clam runs noticeably wider than the macOS
        # default), and a packed row silently clipped "Revert" off the
        # edge under clam. A grid wraps predictably instead.
        row2 = ttk.Frame(self.card_outliers)
        row2.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        row2.columnconfigure(0, weight=1, uniform="act")
        row2.columnconfigure(1, weight=1, uniform="act")
        self.calibrate_button = ttk.Button(row2, text="Calibrate…", state="disabled",
                                           command=self._on_calibrate)
        self.calibrate_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=(0, 3))
        self.clean_button = ttk.Button(row2, text="Clean Data", state="disabled",
                                       command=self._on_clean_data)
        self.clean_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=(0, 3))
        self.revert_button = ttk.Button(row2, text="Revert", state="disabled",
                                        command=self._on_revert)
        self.revert_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        # Fills holes that are ALREADY in the series, with no detection
        # step - which is what irradiance needs, because a Z-score cannot
        # see an irradiance sensor fault at all. The filter's valid range
        # does the removing; this does the refilling.
        self.fill_gaps_button = ttk.Button(row2, text="Fill Gaps", state="disabled",
                                           command=self._on_fill_gaps)
        self.fill_gaps_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))

        # ---- 5. Low-pass filter ------------------------------------------
        self.card_filter = card(5, "Low-Pass Filter")
        self.cutoff_var = tk.StringVar(value=f"{DEFAULT_CUTOFF_MINUTES:g}")
        self.order_var = tk.StringVar(value=str(DEFAULT_ORDER))
        self.valid_min_var = tk.StringVar(value="")
        self.valid_max_var = tk.StringVar(value="")
        self.card_filter.add_param("Cutoff (min)", self.cutoff_var, width=5, row=0, column=0)
        self.card_filter.add_param("Order", self.order_var, width=3, row=0, column=1)
        self.card_filter.add_param("Valid", self.valid_min_var, width=6, row=1, column=0)
        self.card_filter.add_param("to", self.valid_max_var, width=6, row=1, column=1)
        self.apply_filter_button = self.card_filter.set_action("Apply Filter", self._on_apply_filter)
        prow = ttk.Frame(self.card_filter)
        prow.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(prow, text="Preset:", foreground=MUTED_TEXT,
                  font=("Helvetica", 8)).pack(side="left")
        self.ghi_button = ttk.Button(prow, text="GHI", width=5, state="disabled",
                                     command=self._set_ghi_range)
        self.ghi_button.pack(side="left", padx=(4, 0))
        self.temp_button = ttk.Button(prow, text="Temp", width=6, state="disabled",
                                      command=self._set_temp_range)
        self.temp_button.pack(side="left", padx=(4, 0))
        self.filter_revert_button = ttk.Button(prow, text="Revert", state="disabled",
                                               command=self._on_revert)
        self.filter_revert_button.pack(side="left", padx=(6, 0))

        # ---- 6. Normalise -------------------------------------------------
        self.card_normalise = card(6, "Normalise  ·  min-max")
        self.fit_fraction_var = tk.StringVar(value=f"{DEFAULT_FIT_FRACTION * 100:g}")
        self.target_min_var = tk.StringVar(value=f"{DEFAULT_RANGE[0]:g}")
        self.target_max_var = tk.StringVar(value=f"{DEFAULT_RANGE[1]:g}")
        self.card_normalise.add_param("Train split %", self.fit_fraction_var,
                                      width=5, row=0, column=0)
        self.card_normalise.add_param("Range", self.target_min_var, width=4, row=1, column=0)
        self.card_normalise.add_param("to", self.target_max_var, width=4, row=1, column=1)
        self.normalise_button = self.card_normalise.set_action("Normalise", self._on_normalise)
        self.normalise_revert_button = self.card_normalise.add_button(
            "Revert", self._on_revert)

    def _scroll_sidebar(self, event, canvas) -> None:
        """Wheel scrolling over the sidebar, platform differences flattened."""
        try:
            num = getattr(event, "num", None)
            if num == 4:
                canvas.yview_scroll(-1, "units")
            elif num == 5:
                canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                step = event.delta // 120 if abs(event.delta) >= 120 else event.delta
                canvas.yview_scroll(-step, "units")
        except tk.TclError:
            pass

    # -- centre ------------------------------------------------------------
    def _build_centre(self, parent: ttk.Frame) -> None:
        self.layer_var = tk.StringVar(value="")
        viewbar = ttk.Frame(parent, padding=(4, 6))
        viewbar.pack(side="top", fill="x")

        ttk.Label(viewbar, text="Series:").pack(side="left")
        self.household_var = tk.StringVar()
        self.household_combo = ttk.Combobox(viewbar, textvariable=self.household_var,
                                            state="disabled", width=20)
        self.household_combo.pack(side="left", padx=(6, 16))
        self.household_combo.bind("<<ComboboxSelected>>", self._on_channel_change)

        ttk.Label(viewbar, text="Start:").pack(side="left")
        if HAS_TKCALENDAR:
            self.start_picker = DateEntry(viewbar, width=11, state="disabled",
                                          date_pattern="yyyy-mm-dd")
        else:
            self.start_var = tk.StringVar()
            self.start_picker = ttk.Entry(viewbar, textvariable=self.start_var,
                                          width=12, state="disabled")
        self.start_picker.pack(side="left", padx=(6, 12))

        ttk.Label(viewbar, text="End:").pack(side="left")
        if HAS_TKCALENDAR:
            self.end_picker = DateEntry(viewbar, width=11, state="disabled",
                                        date_pattern="yyyy-mm-dd")
        else:
            self.end_var = tk.StringVar()
            self.end_picker = ttk.Entry(viewbar, textvariable=self.end_var,
                                        width=12, state="disabled")
        self.end_picker.pack(side="left", padx=(6, 0))

        # View and Plot live on the SECOND row, not squeezed onto the end
        # of the first. Under a wider ttk theme the first row ran out of
        # space and clipped the "View:" label down to a sliver; the
        # quick-range row has room to spare, and the split reads better
        # anyway - row 1 is what to look at, row 2 is how to look at it.
        rangebar = ttk.Frame(parent, padding=(4, 0, 4, 6))
        rangebar.pack(side="top", fill="x")

        self.plot_button = accent_button(rangebar, "Plot", self._on_plot)
        self.plot_button.pack(side="right")
        self.view_var = tk.StringVar(value=VIEW_OPTIONS[0])
        self.view_combo = ttk.Combobox(rangebar, textvariable=self.view_var, values=VIEW_OPTIONS,
                                       state="disabled", width=12, justify="center")
        self.view_combo.pack(side="right", padx=(0, 10))
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_change)
        ttk.Label(rangebar, text="View:", foreground=MUTED_TEXT).pack(side="right", padx=(12, 4))

        ttk.Label(rangebar, text="Quick range:", foreground=MUTED_TEXT).pack(side="left")
        self.preset_buttons: List[ttk.Button] = []
        for label in PRESETS:
            b = ttk.Button(rangebar, text=label, width=10, state="disabled",
                           command=lambda l=label: self._apply_preset(l))
            b.pack(side="left", padx=(6, 0))
            self.preset_buttons.append(b)
        ttk.Separator(rangebar, orient="vertical").pack(side="left", fill="y", padx=12)
        self.prev_button = ttk.Button(rangebar, text="◀ Prev", width=8, state="disabled",
                                      command=lambda: self._shift_range(-1))
        self.next_button = ttk.Button(rangebar, text="Next ▶", width=8, state="disabled",
                                      command=lambda: self._shift_range(1))
        self.prev_button.pack(side="left")
        self.next_button.pack(side="left", padx=(6, 0))

        # The layer breadcrumb goes on the first row, which now has the
        # free space that Plot and View gave up.
        ttk.Label(viewbar, textvariable=self.layer_var, foreground=MUTED_TEXT,
                  font=("Helvetica", 8)).pack(side="right", padx=(12, 0))

        self.plot_frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        self.plot_frame.pack(side="top", fill="both", expand=True)

        self.figure = plt.Figure(figsize=(9, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._style_empty_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=2, pady=2)

    # -- inspector ---------------------------------------------------------
    def _build_inspector(self) -> None:
        head = ttk.Frame(self.inspector, padding=(10, 10, 10, 4))
        head.pack(side="top", fill="x")
        ttk.Label(head, text="DATA PROPERTIES", font=("Helvetica", 9, "bold"),
                  foreground=MUTED_TEXT).pack(side="left")
        ttk.Button(head, text="✕", width=3, command=self._toggle_inspector).pack(side="right")

        self.inspector_scope_var = tk.StringVar(value="No window selected.")
        ttk.Label(self.inspector, textvariable=self.inspector_scope_var, foreground=MUTED_TEXT,
                  font=("Helvetica", 8), wraplength=290, justify="left").pack(
            side="top", fill="x", padx=10, pady=(0, 6))

        buttons = ttk.Frame(self.inspector, padding=(10, 6))
        buttons.pack(side="bottom", fill="x")
        ttk.Button(buttons, text="Copy", command=self._copy_stats).pack(side="left")
        ttk.Button(buttons, text="Save…", command=self._save_stats).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Refresh", command=self._refresh_inspector).pack(side="right")

        frame = ttk.Frame(self.inspector)
        frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 4))
        scroll = ttk.Scrollbar(frame, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.stats_text = tk.Text(frame, wrap="word", font=("Courier", 9), relief="flat",
                                  background=PALETTE["surface_alt"], foreground=PALETTE["fg"],
                                  insertbackground=PALETTE["fg"],
                                  selectbackground=PALETTE["select_bg"],
                                  selectforeground=PALETTE["select_fg"],
                                  yscrollcommand=scroll.set, width=36)
        self.stats_text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.stats_text.yview)
        self.stats_text.tag_configure("head", font=("Courier", 9, "bold"))
        self.stats_text.tag_configure("key", foreground=MUTED_TEXT)
        self.stats_text.tag_configure("big", font=("Courier", 11, "bold"), foreground=ACCENT)
        self.stats_text.config(state="disabled")

    def _style_empty_axes(self) -> None:
        self.ax.clear()
        style_axes(self.figure, self.ax)
        self.ax.set_title("Import a file and choose a series to plot", color=MUTED_TEXT)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        self.figure.tight_layout()

    # =========================================================== enablement
    def _refresh_enablement(self) -> None:
        """
        The ONLY place pipeline widget state is set.

        A pure function of (stage, kind, overlay, busy, modal). If a
        button is wrong, this is the one function to read - which is the
        whole point: the previous build set widget state at 48 sites
        across 14 methods, and that is where gating bugs live.
        """
        s = self.state_
        blocked = s.busy is not None or s.modal
        loaded = self.stack is not None
        kind = self.kind

        def gate(widget, ok: bool) -> None:
            set_enabled(widget, ok and not blocked)

        # ---- 1. Import: always available except while busy --------------
        gate(self.import_button, True)
        self.card_import.set_state(
            "done" if loaded else "available",
            self.data.summary() if loaded else
            "Excel or CSV. Household panel, aggregate series, or solar.")

        # ---- 2. Sentinels & interpolate ---------------------------------
        can_sentinel = loaded and kind is not DatasetKind.ENVIRONMENTAL and s.stage is Stage.LOADED
        gate(self.clean_sentinels_button, can_sentinel)
        if not loaded:
            self.card_sentinels.set_state("blocked", "Import a file first.")
        elif kind is DatasetKind.ENVIRONMENTAL:
            self.card_sentinels.set_state(
                "na", "Not applicable — solar data has no sentinel codes. "
                      "Use the low-pass filter's valid range instead.")
        elif s.stage in (Stage.SENTINELS_HANDLED, Stage.AGGREGATED):
            self.card_sentinels.set_state("done", self._sentinel_summary())
        else:
            self.card_sentinels.set_state("available", self._sentinel_summary())

        # ---- 3. Aggregate ------------------------------------------------
        can_aggregate = (loaded and kind is DatasetKind.HOUSEHOLD_PANEL
                         and s.stage is Stage.SENTINELS_HANDLED)
        gate(self.aggregate_button, can_aggregate)
        if not loaded:
            self.card_aggregate.set_state("blocked", "Import a file first.")
        elif kind is DatasetKind.ENVIRONMENTAL:
            self.card_aggregate.set_state("na", "Not applicable to solar data.")
        elif s.stage is Stage.AGGREGATED:
            self.card_aggregate.set_state(
                "done", f"Summed into one total-demand series ({self.data.households[0]}).")
        elif kind is DatasetKind.AGGREGATE_SERIES:
            self.card_aggregate.set_state("na", "Already aggregated — this file is a single series.")
        elif can_aggregate:
            self.card_aggregate.set_state("available", "Sum all channels into total demand.")
        else:
            self.card_aggregate.set_state("blocked", "Run step 2 first.")

        # ---- 4. Outliers --------------------------------------------------
        # Gated on the SELECTED CHANNEL being power, not on the pipeline
        # having reached the aggregate. Detection has always worked on a
        # single household column and it is a reasonable thing to do
        # while exploring; hard-gating it behind step 3 would have taken
        # away working functionality in the name of guidance. The card
        # still points at aggregation, because cleaning the total is what
        # the pipeline is for - it guides without removing the option.
        selected_role = (self.data.channel(self.household_var.get()).role
                         if loaded else ChannelRole.UNKNOWN)
        analysis_ready = loaded and selected_role is ChannelRole.POWER
        on_aggregate = loaded and kind is DatasetKind.AGGREGATE_SERIES
        showing_series = (self._last_plot is not None
                          and self.view_var.get() == VIEW_OPTIONS[0])
        gate(self.detect_outliers_button, analysis_ready and showing_series)
        gate(self.clear_outliers_button, analysis_ready and showing_series)
        gate(self.calibrate_button, analysis_ready)
        gate(self.clean_button, analysis_ready)
        gate(self.revert_button, loaded and self.stack.can_revert())
        # Gap filling needs holes, not a power channel: it is the half of
        # cleaning that irradiance actually needs, since the filter's
        # valid range has already screened the impossible readings out.
        has_holes = bool(loaded and self._selected_has_gaps())
        gate(self.fill_gaps_button, has_holes)
        if not loaded:
            self.card_outliers.set_state("blocked", "Import a file first.")
        elif kind is DatasetKind.ENVIRONMENTAL:
            self.card_outliers.set_state(
                "available" if has_holes else "na", self._gap_summary())
        elif not analysis_ready:
            self.card_outliers.set_state("blocked", "Select a power channel to detect outliers on.")
        elif s.overlay is Overlay.OUTLIERS_CLEANED:
            self.card_outliers.set_state("done", self._outlier_summary())
        else:
            note = self._outlier_summary()
            if not on_aggregate:
                note = ("Working on one household. Aggregate first (step 3) to clean the total "
                        "demand series — that is what the forecast is trained on. " + note)
            if not showing_series:
                note = "Plot a time series to circle outliers on it. " + note
            self.card_outliers.set_state("available", note)

        # ---- 5. Low-pass filter -------------------------------------------
        has_env = loaded and self.data.has_environmental
        gate(self.apply_filter_button, has_env)
        gate(self.ghi_button, has_env)
        gate(self.temp_button, has_env)
        gate(self.filter_revert_button, loaded and self.stack.can_revert())
        if not loaded:
            self.card_filter.set_state("blocked", "Import a file first.")
        elif not has_env:
            self.card_filter.set_state(
                "na", "No irradiance or temperature channels in this dataset.")
        elif s.overlay is Overlay.FILTERED:
            self.card_filter.set_state("done", "Filtered. Revert restores the previous layer.")
        else:
            self.card_filter.set_state("available", self._filter_summary())

        # ---- 6. Normalise --------------------------------------------------
        # The last step before the data leaves for a model, so it is
        # available whenever there is data - but it warns when the layer
        # is still Raw, because a surviving spike sets the maximum and
        # squashes every real reading into a narrow band.
        gate(self.normalise_button, loaded)
        gate(self.normalise_revert_button, loaded and self.stack.can_revert())
        if not loaded:
            self.card_normalise.set_state("blocked", "Import a file first.")
        elif self.data.layer_name == "Normalised":
            self.card_normalise.set_state("done", self._normalise_summary())
        else:
            self.card_normalise.set_state("available", self._normalise_summary())

        # ---- view controls -------------------------------------------------
        for w in self.preset_buttons + [self.prev_button, self.next_button]:
            gate(w, loaded)
        gate(self.plot_button, loaded)
        gate(self.properties_button, loaded)
        gate(self.history_button, loaded)
        try:
            self.household_combo.configure(state="readonly" if (loaded and not blocked) else "disabled")
            self.view_combo.configure(state="readonly" if (loaded and not blocked) else "disabled")
            self.start_picker.configure(state="normal" if (loaded and not blocked) else "disabled")
            self.end_picker.configure(state="normal" if (loaded and not blocked) else "disabled")
        except tk.TclError:
            pass

    def _sentinel_summary(self) -> str:
        if self._sentinel_scan is None:
            return "Scanning…"
        scan = self._sentinel_scan
        bits = []
        if scan["n_hits"]:
            bits.append(f"{scan['n_hits']:,} sentinel reading(s) found "
                        f"({scan['fraction'] * 100:.3f}%) across {scan['n_channels']} channel(s)")
        else:
            bits.append("No sentinel codes found")
        if scan["n_nan"]:
            bits.append(f"{scan['n_nan']:,} already-missing reading(s)")
        return ". ".join(bits) + "."

    # The two caveats below are the ones that actually bite on this data,
    # so they stay on the card rather than living in a tooltip nobody
    # opens: a big outlier inflates the very σ it is tested against, and
    # a sustained level shift is invisible to any point-wise test.
    _OUTLIER_CAVEATS = ("A large outlier inflates the σ of its own window (masking), and a "
                        "sustained level shift is invisible to this test entirely.")

    def _outlier_summary(self) -> str:
        try:
            window = int(float(self.window_var.get()))
            ceiling = max_achievable_zscore(window)
            return (f"Flags a reading more than σ rolling standard deviations from its rolling "
                    f"mean. Window {self.data.samples_to_duration(window)}; the highest "
                    f"reachable Z-score at this window is {ceiling:.2f}σ. "
                    + self._OUTLIER_CAVEATS)
        except (ValueError, AttributeError):
            return ("Flags a reading more than σ rolling standard deviations from its rolling "
                    "mean. " + self._OUTLIER_CAVEATS)

    def _selected_has_gaps(self) -> int:
        """How many readings are missing in the selected channel."""
        name = self.household_var.get()
        if self.stack is None or name not in self.data.df.columns:
            return 0
        return int(self.data.df[name].isna().sum())

    def _gap_summary(self) -> str:
        """The step-4 card text for a solar dataset."""
        n = self._selected_has_gaps()
        base = ("A Z-score cannot see an irradiance sensor fault — it catches none of them — so "
                "detection here is the low-pass filter's valid range, which screens impossible "
                "readings out to holes. ")
        if not n:
            return base + ("Nothing is missing in this channel yet. Run step 5 first; "
                           "Fill Gaps then repairs what it screened out.")
        return base + (f"{n:,} reading(s) are missing in this channel. Fill Gaps refits them "
                       "from a local B-spline; runs longer than Max gap are left alone.")

    def _normalise_summary(self) -> str:
        if self.data.layer_name == "Normalised":
            return ("Normalised. The saved JSON holds the exact min/max per channel — use it "
                    "to scale future data and to map model output back to real units.")
        base = ("Scales each channel to the target range. The min/max are fitted on the "
                "TRAIN SPLIT ONLY, so the test period cannot leak into the scaling; "
                "later values may therefore land outside the range, and the report says "
                "how often.")
        if self.stack.active_layer.step_name == "Raw":
            return ("⚠ This layer is still Raw. One surviving spike would set the maximum and "
                    "squash every real reading into a narrow band — clean first (steps 2–4). "
                    + base)
        return base

    def _filter_summary(self) -> str:
        roles = {c.role.value for c in self.data.channels.values() if c.is_environmental}
        return ("Zero-phase Butterworth. Screening runs first — a filter spreads an impossible "
                f"reading rather than removing it. Channels: {', '.join(sorted(roles))}.")

    def _refresh_status_strip(self) -> None:
        if self.stack is None:
            self.layer_var.set("")
            return
        lyr = self.stack.active_layer
        # Kept short: this label shares the row with the date pickers, and
        # a long breadcrumb is the first thing to get clipped.
        text = f"Layer: {self.stack.breadcrumb()}  ·  {len(self.data.households)} ch"
        if lyr.n_changed:
            text += f"  ·  {lyr.n_changed:,} modified"
        self.layer_var.set(text)

    # ================================================================ import
    def _on_import(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a data file (household panel, aggregate series, or solar)",
            filetypes=[
                ("Excel or CSV files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if self.stack is not None and self.stack.can_revert():
            if not messagebox.askokcancel(
                "Discard the current pipeline?",
                "Importing a new file discards the layers built from the current one "
                "(nothing on disk is affected). Continue?",
            ):
                return
        self.status_var.set(f"Loading {Path(path).name} …  "
                            "(a large .xlsx can take a minute to parse)")
        self._transition(busy="loading")
        threading.Thread(target=self._load_in_background, args=(path,), daemon=True).start()
        self.after(100, self._poll_worker)

    def _load_in_background(self, path: str) -> None:
        try:
            ds = load_dataset(path)
            scan = scan_sentinels(ds.df)   # auto-detect, per the brief - free at 0.017 s
        except Exception as exc:  # noqa: BLE001 - handed to the UI thread
            self._worker_queue.put(("load", "error", exc))
            return
        self._worker_queue.put(("load", "ok", (ds, scan)))

    def _on_load_success(self, data) -> None:
        """Accepts a Dataset or a LoadProfileData (which it classifies)."""
        ds = data if isinstance(data, Dataset) else from_load_profile(data)
        self.stack = LayerStack(ds)
        self._stats_cache.clear()
        self._last_plot = None
        self._last_cleaning = None
        self._last_filter = None
        self._last_calibration = None
        self._month_mode = False
        self._outlier_scatter = None
        if self._sentinel_scan is None or self._sentinel_scan.get("_for") != id(ds):
            try:
                self._sentinel_scan = scan_sentinels(ds.df)
            except Exception:  # noqa: BLE001
                self._sentinel_scan = None

        self.file_label_var.set(ds.summary())
        self.household_combo.config(values=ds.households)
        self.household_combo.current(0)
        self.view_var.set(VIEW_OPTIONS[0])

        if HAS_TKCALENDAR:
            self.start_picker.config(mindate=ds.start.date(), maxdate=ds.end.date())
            self.end_picker.config(mindate=ds.start.date(), maxdate=ds.end.date())
        self._set_date_range(ds.start.date(), ds.end.date())

        # Solar files get the cutoff and physical limits of whichever
        # channel is selected, rather than making the user find them.
        self._apply_channel_defaults()

        self._transition(stage=Stage.LOADED, overlay=Overlay.NONE, busy=None)
        notes = "  ".join(ds.notes)
        self.status_var.set(
            f"Loaded {ds.source_path.name} — {ds.kind_label().lower()}, "
            f"{len(ds.households)} channel(s) at {ds.interval_label()}. "
            + (notes + "  " if notes else "")
            + self._next_action_hint()
        )
        self._apply_preset("Full Range")

    def _next_action_hint(self) -> str:
        if self.kind is DatasetKind.HOUSEHOLD_PANEL:
            return "Next: step 2, Clean Sentinels & Interpolate."
        if self.kind is DatasetKind.ENVIRONMENTAL:
            return "Next: step 5, the low-pass filter."
        return "Next: step 4, outlier detection — this file is already a single series."

    def _on_load_error(self, exc: Exception) -> None:
        messagebox.showerror("Failed to load file", str(exc))
        self.status_var.set("Failed to load file.")
        self._transition(busy=None)

    # ============================================================ worker bus
    def _poll_worker(self) -> None:
        """
        Drain worker results on the MAIN thread.

        Workers only ever read layers and hand results back here; only
        this thread appends to the stack or touches a widget. Because
        layers are immutable there is nothing to race on.
        """
        again = True
        try:
            while True:
                tag, status, payload = self._worker_queue.get_nowait()
                if tag == "progress":
                    self.status_var.set(str(payload))
                    continue
                again = False
                if status == "error":
                    self._on_worker_error(tag, payload)
                else:
                    self._on_worker_done(tag, payload)
        except queue.Empty:
            pass
        if again and self.state_.busy is not None:
            self.after(100, self._poll_worker)

    def _on_worker_error(self, tag: str, exc: Exception) -> None:
        if tag == "load":
            self._on_load_error(exc)
            return
        messagebox.showerror(f"{tag.replace('_', ' ').title()} failed", str(exc))
        self.status_var.set(f"{tag.replace('_', ' ').title()} failed.")
        self._transition(busy=None)

    def _on_worker_done(self, tag: str, payload) -> None:
        if tag == "load":
            ds, scan = payload
            self._sentinel_scan = scan
            self._on_load_success(ds)
        elif tag == "sentinels":
            self._present_step(payload, Stage.SENTINELS_HANDLED,
                               "Sentinel & Interpolation Report", "sentinel_interpolation_report.txt")
        elif tag == "aggregate":
            self._present_step(payload, Stage.AGGREGATED,
                               "Aggregation Report", "aggregation_report.txt")
        elif tag == "calibrate":
            self._transition(busy=None)
            self._apply_calibration(*payload)

    # ========================================================= pipeline: 2/3
    def _on_clean_sentinels(self) -> None:
        """
        Step 2. Runs sentinel replacement and interpolation as a chain of
        two independent programs behind one button - see pipeline.py for
        why they are separate objects rather than one function.
        """
        if self.stack is None or self.state_.busy:
            return
        try:
            codes = tuple(float(c) for c in self.sentinel_var.get().replace(",", " ").split())
            if not codes:
                raise ValueError("Give at least one sentinel code, e.g. -999")
            max_gap = int(float(self.max_gap_var.get()))
            if max_gap <= 0:
                raise ValueError("Max gap must be a positive number of samples.")
        except ValueError as exc:
            messagebox.showerror(
                "Invalid settings",
                f"{exc}\n\nCodes is a list like '-999, -9999'. Max gap is in samples — "
                f"at this data's {self.data.interval_label()} step, "
                f"{DEFAULT_MAX_GAP} samples is "
                f"{DEFAULT_MAX_GAP * self.data.interval.total_seconds() / 3600:.0f} hours.")
            return

        span = len(self.data.df)
        if max_gap > 0.1 * span:
            if not messagebox.askokcancel(
                "That is a very long gap",
                f"{max_gap:,} samples is more than 10% of the record ({span:,} samples). "
                "Filling a gap that long invents data with no nearby evidence behind it, "
                "which is worse in a forecasting training set than a visible hole.\n\n"
                "Continue anyway?",
            ):
                return

        self.status_var.set("Replacing sentinel codes and interpolating …")
        self._transition(busy="sentinels")
        ds = self.data
        steps = [SentinelStep(codes), InterpolateStep(max_gap=max_gap)]
        threading.Thread(target=self._run_steps_in_background,
                         args=("sentinels", steps, ds), daemon=True).start()
        self.after(100, self._poll_worker)

    def _on_aggregate(self) -> None:
        """Step 3. Sum the panel into one total-demand series."""
        if self.stack is None or self.state_.busy:
            return
        self.status_var.set("Aggregating across channels …")
        self._transition(busy="aggregate")
        threading.Thread(target=self._run_steps_in_background,
                         args=("aggregate", [AggregateStep()], self.data), daemon=True).start()
        self.after(100, self._poll_worker)

    def _run_steps_in_background(self, tag: str, steps, ds: Dataset) -> None:
        try:
            result = run_chain(steps, ds)
        except Exception as exc:  # noqa: BLE001 - handed to the UI thread
            self._worker_queue.put((tag, "error", exc))
            return
        self._worker_queue.put((tag, "ok", result))

    def _present_step(self, result, next_stage: Stage, title: str, filename: str) -> None:
        """
        Stage the result, then open the modal.

        The layer is HELD, not committed: until the user acknowledges the
        report the canvas still shows the previous layer, so the data on
        screen and the numbers in the report can never disagree. The
        modal's dismissal is what commits it and refreshes the canvas.
        """
        layer = self.stack.stage(result.dataset, result.dataset.layer_name,
                                 result.report, result.ledger)
        self._pending_step = (layer, next_stage, result)
        self._transition(busy=None, modal=True)
        ReportModal(
            self, title, result.report,
            headline=result.headline,
            save_callback=lambda p: self._save_step_report(p, result),
            default_filename=filename,
            on_dismiss=self._commit_pending_step,
        )

    def _commit_pending_step(self) -> None:
        """Runs when the report modal closes - by Okay, Escape, or the window X."""
        if self._pending_step is None:
            self._transition(modal=False)
            return
        layer, next_stage, result = self._pending_step
        self._pending_step = None
        self.stack.commit(layer)
        self._stats_cache.clear()

        # Aggregation changes the channel list out from under the combo.
        current = self.household_var.get()
        self.household_combo.config(values=self.data.households)
        if current not in self.data.households:
            self.household_combo.current(0)

        self._clear_outlier_overlay()
        self._transition(stage=next_stage, overlay=Overlay.NONE, modal=False)
        self._on_plot(_reset_month_mode=False)
        self.status_var.set(f"{layer.step_name}: {result.headline}.  {self._next_action_hint()}")

    def _save_step_report(self, path: Path, result) -> None:
        """Write the report, and the change ledger beside it when there is one."""
        path.write_text(result.report)
        if len(result.ledger):
            ledger_path = path.with_name(f"{path.stem}_changes.csv")
            result.ledger.to_csv(ledger_path, index=False)
        data_path = path.with_name(f"{path.stem}_data.csv")
        result.dataset.df.to_csv(data_path)

    def _show_history(self) -> None:
        if self.stack is None:
            return
        body = ["LAYER STACK", "=" * 46, "",
                "Nothing is overwritten: each step appends a layer and Revert moves",
                "a pointer rather than restoring a backup. The arrow marks the layer",
                "the canvas is drawing from.", ""]
        body += self.stack.history()
        body += ["", f"Resident memory: {self.stack.memory_mb():.1f} MB",
                 f"Source file: {self.data.source_path}  (never modified)"]
        ledger = self.stack.full_ledger()
        if len(ledger):
            body += ["", f"CHANGE LEDGER — {len(ledger):,} reading(s) altered so far", ""]
            body.append(ledger.head(40).to_string(index=False))
            if len(ledger) > 40:
                body.append(f"… and {len(ledger) - 40:,} more (Save writes the full ledger).")
        ReportModal(self, "Layers", "\n".join(body), headline=self.stack.breadcrumb(),
                    save_callback=lambda p: self._save_history(p, ledger),
                    default_filename="layer_history.txt", modal=False)

    def _save_history(self, path: Path, ledger: pd.DataFrame) -> None:
        path.write_text("\n".join(self.stack.history()))
        if len(ledger):
            ledger.to_csv(path.with_name(f"{path.stem}_ledger.csv"), index=False)

    # ============================================================ date range
    def _set_date_range(self, start: date, end: date) -> None:
        if HAS_TKCALENDAR:
            self.start_picker.set_date(start)
            self.end_picker.set_date(end)
        else:
            self.start_var.set(start.isoformat())
            self.end_var.set(end.isoformat())

    def _get_date_range(self) -> Tuple[date, date]:
        if HAS_TKCALENDAR:
            return self.start_picker.get_date(), self.end_picker.get_date()
        return date.fromisoformat(self.start_var.get()), date.fromisoformat(self.end_var.get())

    def _apply_preset(self, label: str) -> None:
        """
        Quick-range buttons. "Full Range" is the whole file; the others
        are relative to whatever start date is showing, so they resize
        the window you are already looking at.

        "1 Month" is a real calendar month (28-31 days as the anchor
        month actually has), not a fixed 30-day block - which is what
        makes paging through a multi-month record land on month
        boundaries instead of drifting a day or two per page.
        """
        if self.stack is None:
            return
        data_start, data_end = self.data.start.date(), self.data.end.date()

        if label == "Full Range":
            new_start, new_end = data_start, data_end
            self._month_mode = False
        elif label in ("1 Month",) or label in PRESET_SPANS:
            try:
                anchor, _ = self._get_date_range()
            except ValueError:
                anchor = data_start
            anchor = min(max(anchor, data_start), data_end)
            new_start = anchor
            if label == "1 Month":
                new_end = min(self._month_end(anchor), data_end)
                self._month_mode = True
            else:
                new_end = min(anchor + timedelta(days=PRESET_SPANS[label] - 1), data_end)
                self._month_mode = False
        else:
            return

        self._set_date_range(new_start, new_end)
        self._on_plot(_reset_month_mode=False)

    @staticmethod
    def _month_end(start: date) -> date:
        """Last day (inclusive) of the calendar month starting at `start`."""
        return (pd.Timestamp(start) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()

    def _shift_range(self, direction: int) -> None:
        """Page by the window's own length - or by a true calendar month
        when the window came from the "1 Month" preset."""
        if self.stack is None:
            return
        try:
            start_d, end_d = self._get_date_range()
        except ValueError:
            return
        data_start, data_end = self.data.start.date(), self.data.end.date()

        if self._month_mode:
            if direction > 0:
                new_start = (pd.Timestamp(start_d) + pd.DateOffset(months=1)).date()
                if new_start > data_end:
                    self.status_var.set("Already at the end of the loaded data.")
                    return
                new_end = min(self._month_end(new_start), data_end)
            else:
                new_start = (pd.Timestamp(start_d) - pd.DateOffset(months=1)).date()
                new_end = min(self._month_end(new_start), data_end)
                if new_end < data_start:
                    self.status_var.set("Already at the start of the loaded data.")
                    return
                new_start = max(new_start, data_start)
        else:
            span_days = (end_d - start_d).days + 1
            if direction > 0:
                new_start = end_d + timedelta(days=1)
                if new_start > data_end:
                    self.status_var.set("Already at the end of the loaded data.")
                    return
                new_end = min(new_start + timedelta(days=span_days - 1), data_end)
            else:
                new_end = start_d - timedelta(days=1)
                if new_end < data_start:
                    self.status_var.set("Already at the start of the loaded data.")
                    return
                new_start = max(new_end - timedelta(days=span_days - 1), data_start)

        self._set_date_range(new_start, new_end)
        self._on_plot(_reset_month_mode=False)

    # =================================================================== plot
    def _on_view_change(self, _event=None) -> None:
        if self.stack is not None:
            self._on_plot(_reset_month_mode=False)

    def _on_channel_change(self, _event=None) -> None:
        self._apply_channel_defaults()
        if self.stack is not None:
            self._on_plot(_reset_month_mode=False)

    def _apply_channel_defaults(self) -> None:
        """Load the selected channel's own physical limits and cutoff, so
        the filter panel is right for irradiance vs temperature without
        the user having to remember which is which."""
        if self.stack is None:
            return
        ch = self.data.channel(self.household_var.get())
        if ch.valid_range is not None:
            self.valid_min_var.set(f"{ch.valid_range[0]:g}")
            self.valid_max_var.set(f"{ch.valid_range[1]:g}")
        if ch.default_cutoff_min is not None:
            self.cutoff_var.set(f"{ch.default_cutoff_min:g}")

    def _current_unit(self) -> str:
        return self.data.channel(self.household_var.get()).unit if self.stack else ""

    def _y_label(self) -> str:
        ch = self.data.channel(self.household_var.get())
        names = {ChannelRole.POWER: "Active Power", ChannelRole.IRRADIANCE: "Irradiance",
                 ChannelRole.TEMPERATURE: "Temperature", ChannelRole.UNKNOWN: "Value"}
        return f"{names[ch.role]}" + (f" ({ch.unit})" if ch.unit else "")

    def _on_plot(self, _reset_month_mode: bool = True) -> None:
        if self.stack is None:
            return
        if _reset_month_mode:
            self._month_mode = False
        channel = self.household_var.get()
        if not channel:
            messagebox.showwarning("No series selected", "Please choose a series.")
            return
        if channel not in self.data.df.columns:
            self.household_combo.config(values=self.data.households)
            self.household_combo.current(0)
            channel = self.household_var.get()

        try:
            start_d, end_d = self._get_date_range()
        except ValueError:
            messagebox.showerror("Invalid date", "Start/end dates must be valid (YYYY-MM-DD).")
            return
        if start_d > end_d:
            messagebox.showerror("Invalid range", "Start date must be on or before end date.")
            return

        start_ts = pd.Timestamp(start_d)
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59)
        series = self.data.series_for(channel, start_ts, end_ts)
        if series.empty:
            messagebox.showwarning("No data", "No readings fall in the selected range.")
            return

        if self.view_var.get() == "Distribution":
            self._draw_distribution(series, channel, start_ts, end_ts)
        else:
            self._draw(series, channel, start_ts, end_ts)

        self._last_plot = {"series": channel, "start": start_ts, "end": end_ts}
        self._refresh_enablement()
        self._refresh_status_strip()
        if self._inspector_open:
            self._refresh_inspector()

    def _draw(self, series, channel, start_ts, end_ts) -> None:
        self.ax.clear()
        style_axes(self.figure, self.ax)
        self._outlier_scatter = None
        (line,) = self.ax.plot(series.index, series.values, linewidth=1.0, color=ACCENT)
        self.ax.fill_between(series.index, series.values, color=ACCENT, alpha=0.08, linewidth=0)
        self.ax.set_title(
            f"{channel} — {self.stack.active_layer.step_name}  "
            f"({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})",
            fontsize=11, fontweight="bold", loc="left")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel(self._y_label())
        self.ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

        # On a normalised layer, mark where the scaler's fit window ended.
        # Everything right of this line was scaled with parameters that
        # never saw it - which is the whole point, and worth being able
        # to see rather than take on trust.
        self._draw_split_marker(start_ts, end_ts)

        self.figure.autofmt_xdate()
        self.figure.tight_layout()

        if HAS_MPLCURSORS:
            if self._cursor is not None:
                try:
                    self._cursor.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._cursor = mplcursors.cursor([line], hover=True)
            unit = self._current_unit()

            @self._cursor.connect("add")
            def _on_add(sel):  # noqa: ANN001
                x, y = sel.target
                ts = mdates.num2date(x)
                sel.annotation.set_text(f"{ts:%Y-%m-%d %H:%M}\n{y:.3f} {unit}")

        self.canvas.draw_idle()
        n_missing = int(series.isna().sum())
        self.status_var.set(
            f"Plotted {channel}: {len(series):,} readings "
            f"({start_ts:%Y-%m-%d %H:%M} to {end_ts:%Y-%m-%d %H:%M})"
            + (f"   ·   {n_missing:,} missing in this window" if n_missing else ""))

    def _draw_split_marker(self, start_ts, end_ts) -> None:
        """Vertical rule at the train/test boundary, when one applies."""
        if (self._last_scaler is None
                or self.stack is None
                or self.stack.active_layer.step_name != "Normalised"):
            return
        split = self._last_scaler.fit_end
        if split is None or not (start_ts <= split <= end_ts):
            return
        self.ax.axvline(split, color=OUTLIER_COLOR, linestyle="--", linewidth=1.2,
                        alpha=0.85, zorder=4)
        self.ax.annotate(
            f"scaler fitted up to here\n({split:%Y-%m-%d})",
            xy=(split, 0.97), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="top", fontsize=8, color=OUTLIER_COLOR)

    def _draw_distribution(self, series, channel, start_ts, end_ts) -> None:
        """Histogram plus a summary card, on the same chart area. Outlier
        circling is a time-series concept, so those controls switch off
        while this view is showing and come back automatically."""
        self.ax.clear()
        style_axes(self.figure, self.ax)
        self._clear_outlier_overlay()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

        values = series.dropna()
        n_missing = int(series.isna().sum())
        if values.empty:
            self.ax.set_title("No non-missing readings in this window", color=MUTED_TEXT)
            self.ax.set_xlabel(self._y_label())
            self.ax.set_ylabel("Count")
            for spine in ("top", "right"):
                self.ax.spines[spine].set_visible(False)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.status_var.set(f"{channel}: no non-missing readings to summarise.")
            return

        self.ax.hist(values, bins=50, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.4)
        self.ax.set_title(
            f"{channel} — Distribution  ({start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d})",
            fontsize=11, fontweight="bold", loc="left")
        self.ax.set_xlabel(self._y_label())
        self.ax.set_ylabel("Count")
        self.ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

        st = self._window_stats(series, channel)
        unit = self._current_unit()
        lines = [f"n = {len(values):,}",
                 f"mean = {st.mean:.2f} {unit}",
                 f"median = {st.median:.2f} {unit}",
                 f"std = {st.std:.2f} {unit}",
                 f"min = {st.minimum:.2f} {unit}",
                 f"max = {st.maximum:.2f} {unit}",
                 f"skew = {st.skew:.3f}",
                 f"kurtosis = {st.excess_kurtosis:.3f}"]
        if n_missing:
            lines.append(f"missing = {n_missing:,}")
        self.ax.text(0.98, 0.97, "\n".join(lines), transform=self.ax.transAxes,
                     ha="right", va="top", fontsize=9, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                               edgecolor="#d0d5dd", alpha=0.92))
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(
            f"Distribution of {channel}: {len(values):,} readings   ·   {st.distribution}")

    # ============================================================== inspector
    def _toggle_inspector(self) -> None:
        if self._inspector_open:
            self._body_pane.forget(self.inspector)
            self._inspector_open = False
            return
        self._body_pane.add(self.inspector, weight=0)
        self._inspector_open = True
        self._refresh_inspector()

    def _window_stats(self, series, channel: str):
        """Cached per (layer, channel, window) - immutable layers make the
        key safe, and the distribution fit is the priciest thing on the
        UI thread."""
        key = (self.stack.active_layer.id, channel,
               series.index.min(), series.index.max(), len(series))
        hit = self._stats_cache.get(key)
        if hit is not None:
            return hit
        ch = self.data.channel(channel)
        # Energy is only meaningful in real power units: totalling
        # normalised values would produce a confident, meaningless kWh.
        st = stats_engine.compute(series, channel, ch.unit, self.data.interval,
                                  is_power=(ch.role is ChannelRole.POWER and ch.unit == "kW"))
        return self._stats_cache.put(key, st)

    def _refresh_inspector(self) -> None:
        if self.stack is None or not self._inspector_open:
            return
        channel = self.household_var.get()
        if self._last_plot is not None:
            start_ts, end_ts = self._last_plot["start"], self._last_plot["end"]
        else:
            start_ts, end_ts = self.data.start, self.data.end
        series = self.data.series_for(channel, start_ts, end_ts)
        if series.empty:
            self.inspector_scope_var.set("No readings in the selected window.")
            return

        self.status_var.set(f"Computing statistics for {channel} …")
        self.update_idletasks()
        st = self._window_stats(series, channel)
        self._current_stats = st

        self.inspector_scope_var.set(
            f"{channel} · layer “{self.stack.active_layer.step_name}” · "
            f"{start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d} · {st.n:,} samples")

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        for key, value in st.required_lines():
            self.stats_text.insert("end", f"{key}\n", "key")
            self.stats_text.insert("end", f"{value}\n\n", "big")
        if st.verdict:
            self.stats_text.insert("end", st.verdict + "\n\n")
        for title, rows in st.detail_groups():
            self.stats_text.insert("end", f"{title.upper()}\n", "head")
            for key, value in rows:
                self.stats_text.insert("end", f"  {key:<20} {value}\n")
            self.stats_text.insert("end", "\n")
        self.stats_text.config(state="disabled")
        self.status_var.set(f"Data properties for {channel} over the plotted window.")

    def _copy_stats(self) -> None:
        st = getattr(self, "_current_stats", None)
        if st is None:
            return
        self.clipboard_clear()
        self.clipboard_append(st.as_text())
        self.status_var.set("Statistics copied to the clipboard.")

    def _save_stats(self) -> None:
        st = getattr(self, "_current_stats", None)
        if st is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save statistics", defaultextension=".txt",
            initialfile=f"{st.channel}_properties.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        Path(path).write_text(st.as_text())
        self.status_var.set(f"Saved statistics to {Path(path).name}.")

    # =============================================================== outliers
    def _on_detect_outliers(self) -> None:
        """
        Runs over the FULL series, then draws only the flagged points
        inside the plotted range - a short window would give the rolling
        statistics too little context at its edges, and the answer would
        change depending on how you happened to be scrolled.
        """
        if self.stack is None or self._last_plot is None:
            return
        try:
            sigmas = float(self.sigma_var.get())
            if sigmas <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid threshold", "Threshold (σ) must be a positive number.")
            return
        try:
            window = int(float(self.window_var.get()))
            if window < 3:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid window",
                "Window must be a whole number of samples, 3 or more.\n\n"
                f"At this data's {self.data.interval_label()} step, "
                f"12 samples is {12 * self.data.interval.total_seconds() / 60:.0f} minutes.")
            return

        # A reading helps set the mean and standard deviation of its own
        # window, so its Z-score can never exceed (w-1)/sqrt(w). Asking
        # for more flags nothing however corrupt the data is, and "no
        # outliers found" would be badly misleading - so say so instead.
        ceiling = max_achievable_zscore(window)
        if sigmas >= ceiling:
            messagebox.showwarning(
                "Threshold out of reach for this window",
                f"With a {window}-sample window, no reading can be more than {ceiling:.2f}σ "
                "from its own window's mean — a reading helps set that mean and standard "
                f"deviation itself.\n\nAt {sigmas:g}σ nothing can be flagged, however corrupt "
                f"the data is.\n\nEither lower the threshold below {ceiling:.2f}σ, or widen "
                f"the window to at least {min_window_for_zscore(sigmas)} samples.")
            return

        self._run_rolling(METHOD_ZSCORE, self._last_plot["series"],
                          self._last_plot["start"], self._last_plot["end"], window, sigmas)

    def _run_rolling(self, method, series_name, start_ts, end_ts, window, sigmas) -> None:
        full = self.data.df[[series_name]]
        mask, _, _ = rolling_zscore(full, window=window, n_sigmas=sigmas)
        flagged = mask[series_name]
        flagged_ts = flagged.index[flagged]
        visible = flagged_ts[(flagged_ts >= start_ts) & (flagged_ts <= end_ts)]

        self._clear_outlier_overlay()
        described = self.data.samples_to_duration(window)
        if len(visible) == 0:
            self.canvas.draw_idle()
            self.status_var.set(
                f"No outliers in this window at {sigmas:g}σ (Z-score, {described}).")
            self._transition(overlay=Overlay.NONE)
            return

        values = full.loc[visible, series_name]
        self._outlier_scatter = self.ax.scatter(
            visible, values, s=45, facecolors="none", edgecolors=OUTLIER_COLOR,
            linewidths=1.4, zorder=5, label=f"Outlier ({method})")
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.canvas.draw_idle()
        self._transition(overlay=Overlay.OUTLIERS_MARKED)
        self.status_var.set(
            f"Found {len(visible):,} outlier(s) in this window — Z-score, {sigmas:g}σ, "
            f"{described}. These are candidates, not confirmed errors; nothing has changed yet.")

    def _on_clear_outliers(self) -> None:
        self._clear_outlier_overlay()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        self.canvas.draw_idle()
        self._transition(overlay=Overlay.NONE)
        self.status_var.set("Outlier overlay cleared.")

    def _clear_outlier_overlay(self) -> None:
        if self._outlier_scatter is not None:
            try:
                self._outlier_scatter.remove()
            except Exception:  # noqa: BLE001
                pass
            self._outlier_scatter = None

    # ============================================================ calibration
    def _on_calibrate(self) -> None:
        """Pick window and sigma by synthetic fault injection: plant
        faults of known size and position, sweep the grid, score every
        cell by F1 against the known truth."""
        if self.stack is None or self.state_.busy:
            return
        series_name = self.household_var.get()
        if not series_name:
            messagebox.showwarning("No series selected", "Please choose a series.")
            return
        series = self.data.df[series_name].dropna()
        self.status_var.set("Calibrating by fault injection …")
        self._transition(busy="calibrate")
        threading.Thread(target=self._calibrate_in_background,
                         args=(series, "zscore", series_name), daemon=True).start()
        self.after(100, self._poll_worker)

    def _calibrate_in_background(self, series, method, series_name) -> None:
        put = self._worker_queue.put
        try:
            result = calibrate(
                series, method=method,
                progress=lambda done, total: put(
                    ("progress", "ok", f"Calibrating by fault injection … {done}/{total} sweeps")))
        except Exception as exc:  # noqa: BLE001
            put(("calibrate", "error", exc))
            return
        put(("calibrate", "ok", (result, series_name)))

    def _apply_calibration(self, result, series_name: str) -> None:
        self._last_calibration = result
        best = result.best
        self.window_var.set(str(int(best["window"])))
        self.sigma_var.set(f"{best['sigma']:g}")
        self.status_var.set(
            f"Calibrated on {series_name}: window {int(best['window'])} "
            f"({self.data.samples_to_duration(int(best['window']))}), "
            f"threshold {best['sigma']:g}σ — F1 {best['f1']:.3f} "
            f"(precision {best['precision']:.2f}, recall {best['recall']:.2f}) against "
            f"{result.n_faults} injected faults × {result.repeats} repeats. "
            "Click Detect Outliers to apply.")
        self._refresh_enablement()
        self._show_calibration_report(result)

    def _show_calibration_report(self, result) -> None:
        """The F1 surface over the whole grid - a sensitivity report, not
        just a recommendation."""
        win = tk.Toplevel(self)
        win.title(f"Calibration — {result.method} — {result.series_name}")
        win.geometry("1080x680")

        fig = plt.Figure(figsize=(7.2, 4.4), dpi=100)
        ax = fig.add_subplot(111)
        style_axes(fig, ax)
        pivot = result.grid.pivot(index="window", columns="sigma", values="f1")
        reach = result.grid.pivot(index="window", columns="sigma", values="reachable")
        data = pivot.to_numpy(dtype=float)
        mask = ~reach.to_numpy(dtype=bool)
        shown = np.where(mask, np.nan, data)

        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eceff3")
        im = ax.imshow(shown, cmap=cmap, aspect="auto", origin="lower", vmin=0,
                       vmax=max(0.01, np.nanmax(shown)))
        fig.colorbar(im, ax=ax).set_label("F1 against injected faults", fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Threshold (σ)")
        ax.set_ylabel("Window (samples)")
        ax.set_title("Sensitivity surface: which settings actually recover known faults",
                     fontsize=11, fontweight="bold", loc="left")

        best_r = list(pivot.index).index(int(result.best["window"]))
        best_c = list(pivot.columns).index(result.best["sigma"])
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if mask[r, c]:
                    ax.text(c, r, "n/a", ha="center", va="center", fontsize=7, color=MUTED_TEXT)
                    continue
                strong = shown[r, c] > 0.6 * np.nanmax(shown)
                ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if strong else "#1f2937")
        ax.add_patch(plt.Rectangle((best_c - 0.5, best_r - 0.5), 1, 1, fill=False,
                                   edgecolor=OUTLIER_COLOR, linewidth=2.5, zorder=5))
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=(10, 4))
        canvas.draw_idle()

        frame = ttk.Frame(win)
        frame.pack(side="top", fill="both", expand=False, padx=10, pady=(0, 10))
        scroll = ttk.Scrollbar(frame, orient="vertical")
        scroll.pack(side="right", fill="y")
        text = tk.Text(frame, height=14, wrap="word", font=("Courier", 9), relief="flat",
                       background=PALETTE["surface_alt"], foreground=PALETTE["fg"],
                       insertbackground=PALETTE["fg"],
                       selectbackground=PALETTE["select_bg"],
                       selectforeground=PALETTE["select_fg"],
                       yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.config(command=text.yview)
        text.insert("1.0", result.summary() +
                    "\n\nThe red box is the recommended cell, now loaded into the Window and "
                    "Threshold boxes. 'n/a' cells are thresholds a Z-score cannot reach at "
                    "that window size.")
        text.config(state="disabled")

    # ================================================================ cleaning
    def _on_clean_data(self) -> None:
        """Pop the detected outliers and refill the holes from a locally
        fitted B-spline. Applied to the WHOLE series, not the visible
        window - you are cleaning a dataset, not a view."""
        if self.stack is None or self.state_.busy:
            return
        series_name = self.household_var.get()
        if not series_name:
            messagebox.showwarning("No series selected", "Please choose a series.")
            return
        try:
            sigmas = float(self.sigma_var.get())
            window = int(float(self.window_var.get()))
            if sigmas <= 0 or window < 3:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Set a positive threshold and a window of at least 3 samples before cleaning.")
            return
        if sigmas >= max_achievable_zscore(window):
            messagebox.showwarning(
                "Threshold out of reach for this window",
                f"At a {window}-sample window a Z-score cannot exceed "
                f"{max_achievable_zscore(window):.2f}σ, so nothing would be flagged and "
                "cleaning would do nothing.\n\nLower the threshold or widen the window.")
            return

        self.status_var.set(f"Cleaning {series_name} — detecting, popping and refitting …")
        self.update_idletasks()
        try:
            result = clean_series(self.data.df[series_name], method="zscore", window=window,
                                  n_sigmas=sigmas, source=str(self.data.source_path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cleaning failed", str(exc))
            self.status_var.set("Cleaning failed.")
            return

        before = self.data.df
        after = before.copy()
        after[series_name] = result.cleaned
        touched = pd.DataFrame(False, index=before.index, columns=before.columns)
        touched[series_name] = result.flagged.reindex(before.index, fill_value=False)
        ledger = ledger_from_mask(before, after, touched,
                                  f"Z-score {sigmas:g}σ/{window} → B-spline")

        self._last_cleaning = result
        layer = self.stack.stage(self.data.with_frame(after, "Cleaned"), "Cleaned",
                                 result.report(), ledger)
        self._pending_step = (layer, Stage.AGGREGATED if self.state_.stage is Stage.AGGREGATED
                              else self.state_.stage, _Simple(result.report(), ledger, after,
                                                              f"{result.n_flagged} flagged · "
                                                              f"{result.n_imputed} imputed · "
                                                              f"{result.n_left_missing} left empty"))
        self._transition(modal=True, overlay=Overlay.OUTLIERS_CLEANED)
        ReportModal(self, f"Cleaning report — {series_name}", result.report(),
                    headline=f"{result.n_flagged} flagged  ·  {result.n_imputed} imputed  ·  "
                             f"{result.n_left_missing} left empty",
                    save_callback=lambda p: self._save_cleaning(p, result, series_name),
                    default_filename=f"{series_name}_cleaning_report.txt",
                    on_dismiss=self._commit_pending_step)

    def _on_fill_gaps(self) -> None:
        """
        Refill holes that are already in the series, with no detection.

        This is the half of Clean Data that irradiance needs. Clean Data
        runs a Z-score first, and a Z-score cannot see an irradiance
        sensor fault — measured on the real record, it flags none of the
        713 impossible readings, because they are not statistical
        outliers of a rolling window, they are physically impossible
        values. The low-pass filter's valid range is what removes them,
        leaving holes; this fills those holes.
        """
        if self.stack is None or self.state_.busy:
            return
        series_name = self.household_var.get()
        if not series_name:
            messagebox.showwarning("No series selected", "Please choose a series.")
            return
        series = self.data.df[series_name]
        if not series.isna().any():
            messagebox.showinfo(
                "Nothing to fill",
                f"{series_name} has no missing readings.\n\nOn solar data the holes are made by "
                "the low-pass filter's valid range screening out physically impossible values — "
                "run step 5 first.")
            return

        try:
            lam = float(self.gap_lambda_var.get())
            max_gap = int(float(self.gap_max_var.get()))
            if lam <= 0 or max_gap <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid gap settings",
                "λ must be a positive number and Max gap a positive whole number of samples.\n\n"
                "λ is the B-spline's smoothing penalty: lower follows the data more closely. "
                f"{DEFAULT_GAPFILL_LAMBDA:g} is the measured optimum for filtered 1-minute "
                "irradiance; 1.0 suits unfiltered 5-minute load.")
            return

        ch = self.data.channel(series_name)
        self.status_var.set(f"Filling gaps in {series_name} …")
        self.update_idletasks()
        try:
            result = impute_missing(
                series, lam=lam, max_gap=max_gap,
                valid_range=ch.valid_range,
                dark_threshold=(DEFAULT_DARK_THRESHOLD
                                if ch.role is ChannelRole.IRRADIANCE else None),
                unit=ch.unit, source=str(self.data.source_path))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Gap filling failed", str(exc))
            self.status_var.set("Gap filling failed.")
            return

        before = self.data.df
        after = before.copy()
        after[series_name] = result.filled
        touched = pd.DataFrame(False, index=before.index, columns=before.columns)
        touched[series_name] = result.imputed.reindex(before.index, fill_value=False)
        ledger = ledger_from_mask(before, after, touched, f"B-spline gap fill (λ={lam:g})")

        self._last_cleaning = result
        layer = self.stack.stage(self.data.with_frame(after, "Gaps Filled"), "Gaps Filled",
                                 result.report(), ledger)
        headline = (f"{result.n_missing:,} missing  ·  {result.n_imputed:,} filled"
                    + (f"  ·  {result.n_left_missing:,} left as holes"
                       if result.n_left_missing else "")
                    + (f"  ·  {result.n_night:,} night hole(s) set to zero"
                       if result.n_night else ""))
        self._pending_step = (layer, self.state_.stage,
                              _Simple(result.report(), ledger, after, headline))
        self._transition(modal=True)
        ReportModal(self, f"Gap filling report — {series_name}", result.report(),
                    headline=headline,
                    save_callback=lambda p: self._save_gap_fill(p, result, series_name),
                    default_filename=f"{series_name}_gap_fill_report.txt",
                    on_dismiss=self._commit_pending_step)

    def _save_gap_fill(self, path: Path, result, series_name: str) -> None:
        path.write_text(result.report())
        frame = result.filled.rename(series_name).reset_index()
        frame.columns = ["timestamp", series_name]
        frame.to_csv(path.with_name(f"{path.stem}_data.csv"), index=False)
        if len(result.regions):
            result.regions.to_csv(path.with_name(f"{path.stem}_gaps.csv"), index=False)

    def _save_cleaning(self, path: Path, result, series_name: str) -> None:
        path.write_text(result.report())
        frame = result.cleaned.rename(series_name).reset_index()
        frame.columns = ["timestamp", series_name]
        frame.to_csv(path.with_name(f"{path.stem}_data.csv"), index=False)

    def _on_revert(self) -> None:
        """Step back one layer. Nothing is deleted - the pointer moves."""
        if self.stack is None or not self.stack.can_revert():
            return
        gone = self.stack.active_layer.step_name
        self.stack.revert()
        self._stats_cache.clear()
        self._last_cleaning = None
        self._last_filter = None

        current = self.household_var.get()
        self.household_combo.config(values=self.data.households)
        if current not in self.data.households:
            self.household_combo.current(0)

        stage = {"Raw": Stage.LOADED, "Sentinels": Stage.LOADED,
                 "Interpolated": Stage.SENTINELS_HANDLED,
                 "Aggregated": Stage.AGGREGATED}.get(
            self.stack.active_layer.step_name, self.state_.stage)
        if self.kind is DatasetKind.AGGREGATE_SERIES and self.stack.active == 0:
            stage = Stage.LOADED

        self._clear_outlier_overlay()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        self._transition(stage=stage, overlay=Overlay.NONE)
        if self._last_plot is not None:
            self._on_plot(_reset_month_mode=False)
        self.status_var.set(
            f"Reverted the “{gone}” layer — now showing “{self.stack.active_layer.step_name}”. "
            "Nothing was deleted and nothing on disk was changed.")

    # ================================================================== filter
    def _set_ghi_range(self) -> None:
        """Irradiance preset: physical limits plus the cloud-band cutoff."""
        lo, hi = GHI_VALID_RANGE
        self.valid_min_var.set(f"{lo:g}")
        self.valid_max_var.set(f"{hi:g}")
        self.cutoff_var.set(f"{DEFAULT_CUTOFF_MINUTES:g}")
        self.status_var.set(
            f"Irradiance preset: valid range {lo:g}–{hi:g} W/m², "
            f"{DEFAULT_CUTOFF_MINUTES:g}-minute cutoff "
            "(cloud variability sits at 10–60 minute periods).")

    def _set_temp_range(self) -> None:
        """Temperature preset: wider limits and a shorter cutoff."""
        lo, hi = TEMP_VALID_RANGE
        self.valid_min_var.set(f"{lo:g}")
        self.valid_max_var.set(f"{hi:g}")
        self.cutoff_var.set(f"{TEMP_CUTOFF_MINUTES:g}")
        self.status_var.set(
            f"Temperature preset: valid range {lo:g}–{hi:g} °C, "
            f"{TEMP_CUTOFF_MINUTES:g}-minute cutoff "
            "(ambient temperature is already far smoother than irradiance).")

    def _read_valid_range(self) -> Optional[Tuple[float, float]]:
        lo_s, hi_s = self.valid_min_var.get().strip(), self.valid_max_var.get().strip()
        if not lo_s and not hi_s:
            return None
        lo = float(lo_s) if lo_s else -np.inf
        hi = float(hi_s) if hi_s else np.inf
        if lo >= hi:
            raise ValueError("The valid range's minimum must be below its maximum.")
        return (lo, hi)

    def _on_apply_filter(self) -> None:
        """Screen to physical limits, then low-pass. Screening runs first
        because a Butterworth cannot remove an impossible reading - it
        spreads it across the filter window."""
        if self.stack is None or self.state_.busy:
            return
        series_name = self.household_var.get()
        if not series_name:
            messagebox.showwarning("No series selected", "Please choose a series.")
            return
        try:
            cutoff = float(self.cutoff_var.get())
            order = int(float(self.order_var.get()))
            valid_range = self._read_valid_range()
        except ValueError as exc:
            messagebox.showerror(
                "Invalid filter settings",
                f"{exc}\n\nCutoff is in minutes, order is a whole number (1-10), and the "
                "valid range is optional — leave both boxes empty to skip screening.")
            return

        nyquist = 2 * self.data.interval.total_seconds() / 60
        if cutoff <= nyquist:
            messagebox.showerror(
                "Cutoff below the Nyquist limit",
                f"This data is sampled every {self.data.interval_label()}, so the shortest "
                f"period it can represent is {nyquist:g} minutes. A cutoff of {cutoff:g} "
                "minutes asks the filter to remove something the data cannot contain.\n\n"
                f"Use a cutoff above {nyquist:g} minutes.")
            return

        self.status_var.set(f"Filtering {series_name} …")
        self.update_idletasks()
        try:
            result = lowpass(self.data.df[series_name], cutoff_minutes=cutoff,
                             order=order, valid_range=valid_range)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Filter failed", str(exc))
            self.status_var.set("Filter failed.")
            return

        # The filter may have regularised the index, so rebuild on the
        # result's own grid rather than assuming they still align.
        before = self.data.df
        after = before.reindex(result.filtered.index).copy()
        after[series_name] = result.filtered
        touched = pd.DataFrame(False, index=after.index, columns=after.columns)
        aligned = before[series_name].reindex(after.index)
        touched[series_name] = ~np.isclose(aligned.to_numpy(dtype=float),
                                           result.filtered.to_numpy(dtype=float),
                                           rtol=0, atol=1e-9, equal_nan=True)
        ledger = ledger_from_mask(after.assign(**{series_name: aligned}), after, touched,
                                  f"Butterworth order {order}, {cutoff:g}-min cutoff")

        self._last_filter = result
        layer = self.stack.stage(self.data.with_frame(after, "Filtered"), "Filtered",
                                 result.report(), ledger)
        d_before = result.original.diff().dropna()
        d_after = result.filtered.diff().dropna()
        headline = (f"Butterworth order {order}, {cutoff:g}-minute cutoff, zero-phase  ·  "
                    + (f"{result.n_screened} screened as non-physical  ·  " if result.n_screened else "")
                    + f"sample-to-sample variation {d_before.std():.2f} → {d_after.std():.2f}")
        self._pending_step = (layer, self.state_.stage,
                              _Simple(result.report(), ledger, after, headline))
        self._transition(modal=True, overlay=Overlay.FILTERED)
        ReportModal(self, f"Filter report — {series_name}", result.report(),
                    headline=headline,
                    save_callback=lambda p: self._save_filter(p, result, series_name),
                    default_filename=f"{series_name}_filter_report.txt",
                    on_dismiss=self._commit_pending_step)

    # =============================================================== normalise
    def _on_normalise(self) -> None:
        """
        Min-max scale every channel, fitting the bounds on the training
        split only.

        Whole-frame, not per-channel: a model is trained on all the
        channels together, so they all have to be scaled by one
        consistent, saveable set of parameters.
        """
        if self.stack is None or self.state_.busy:
            return
        try:
            pct = float(self.fit_fraction_var.get())
            if not 0 < pct <= 100:
                raise ValueError("Train split must be a percentage above 0 and up to 100.")
            lo = float(self.target_min_var.get())
            hi = float(self.target_max_var.get())
            if hi <= lo:
                raise ValueError("The target range's maximum must be above its minimum.")
        except ValueError as exc:
            messagebox.showerror(
                "Invalid normalisation settings",
                f"{exc}\n\nTrain split is a percentage of the record (70 means the first 70% "
                "of rows fit the scaler). The range is the interval to scale into, "
                "normally 0 to 1.")
            return

        if self.stack.active_layer.step_name == "Raw":
            if not messagebox.askokcancel(
                "Normalise the raw layer?",
                "This layer has not been cleaned. Min-max is driven entirely by the two most "
                "extreme readings, so a single surviving spike or sentinel code will set the "
                "maximum and squash every real reading into a narrow band at the bottom of "
                "the range.\n\nRun steps 2–4 first, or continue if you know this data is "
                "already clean.",
            ):
                return

        ds = self.data
        self.status_var.set(f"Fitting min-max on the first {pct:g}% of the record …")
        self.update_idletasks()
        try:
            result = normalise(
                ds.df, fit_fraction=pct / 100.0, feature_range=(lo, hi),
                source=str(ds.source_path), layer=self.stack.active_layer.step_name,
                units={name: ch.unit for name, ch in ds.channels.items()})
        except Exception as exc:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Normalisation failed", str(exc))
            self.status_var.set("Normalisation failed.")
            return

        # Normalised channels keep their ROLE (so the pipeline still knows
        # what they are) but lose their unit and physical limits: a valid
        # range of 0-1400 W/m2 is meaningless once the data is in [0, 1],
        # and kWh computed from normalised values would be nonsense.
        channels = {
            name: replace(ch, unit="normalised", valid_range=None)
            for name, ch in ds.channels.items()
        }
        new_ds = ds.with_frame(result.normalised, "Normalised", channels=channels)

        self._last_scaler = result.scaler
        layer = self.stack.stage(new_ds, "Normalised", result.report(), empty_ledger())
        n_out = int(result.holdout_excursions()["n_outside"].sum()) \
            if not result.holdout_excursions().empty else 0
        headline = (f"{len(result.scaler.channels)} channel(s) scaled to "
                    f"[{lo:g}, {hi:g}]  ·  fitted on the first {pct:g}% "
                    f"(to {result.scaler.fit_end:%Y-%m-%d})"
                    + (f"  ·  {n_out:,} held-out reading(s) outside the range" if n_out else ""))
        self._pending_step = (layer, self.state_.stage,
                              _Simple(result.report(), empty_ledger(), result.normalised, headline))
        self._transition(modal=True)
        ReportModal(self, "Normalisation report", result.report(),
                    headline=headline,
                    save_callback=lambda p: self._save_normalisation(p, result),
                    default_filename="normalisation_report.txt",
                    on_dismiss=self._commit_pending_step)

    def _save_normalisation(self, path: Path, result) -> None:
        """
        Write the report, the scaled data, AND the fitted parameters.

        The JSON is the important one. A forecast trained on normalised
        data comes back in normalised units; without these exact numbers
        it cannot be mapped to kW, and refitting on new data would give
        a different scaling from the one the model learnt.
        """
        path.write_text(result.report())
        result.scaler.save_params(path.with_name(f"{path.stem}_scaler.json"))
        result.normalised.to_csv(path.with_name(f"{path.stem}_data.csv"))

    def _save_filter(self, path: Path, result, series_name: str) -> None:
        path.write_text(result.report())
        frame = result.filtered.rename(series_name).reset_index()
        frame.columns = ["timestamp", series_name]
        frame.to_csv(path.with_name(f"{path.stem}_data.csv"), index=False)


class _Simple:
    """Minimal stand-in for a StepResult, for the in-place steps (cleaning,
    filtering) that produce a report and a ledger without running a
    PipelineStep."""

    def __init__(self, report: str, ledger, df, headline: str):
        self.report = report
        self.ledger = ledger
        self.dataset = None
        self.headline = headline
        self._df = df


if __name__ == "__main__":
    app = LoadProfileApp()
    app.mainloop()
