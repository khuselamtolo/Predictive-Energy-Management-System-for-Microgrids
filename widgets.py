"""
The two reusable widgets the new layout is built from.

StepCard  - one pipeline step in the sidebar. Every step is the same
            widget with different content, which is what makes the
            sequence legible: five identical cards with a status glyph
            read as "one pipeline, you are at step 3", where five
            differently-shaped panels read as five unrelated tools.

            A card is never hidden. A step that does not apply to the
            loaded file stays on screen, greyed, and says WHY - a card
            that disappears reads as a bug and destroys the user's model
            of what the pipeline does.

ReportModal - the report window the brief requires: it halts the main
            UI, and it will not go away until the user chooses Save or
            Okay. Four details make that actually true rather than
            approximately true; see the class docstring.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

# Palette shared with app.py. Kept here too so this module stands alone.
from theme import PALETTE

ACCENT = "#2563eb"
ACCENT_ACTIVE = "#1d4ed8"
MUTED_TEXT = "#5b6472"
DONE_GREEN = "#15803d"
RUNNING_AMBER = "#b45309"
CARD_BG = "#ffffff"
SIDEBAR_BG = "#eef1f5"

# Status glyphs. The whole sidebar is readable at a glance from these.
GLYPHS = {
    "done": ("✓", DONE_GREEN),
    "available": ("●", ACCENT),
    "blocked": ("○", MUTED_TEXT),
    "na": ("—", MUTED_TEXT),
    "running": ("⟳", RUNNING_AMBER),
}


# A disabled button must still be READABLE - you need to know what the
# thing you cannot click is. The first attempt at this (a pale label on
# a mid grey) had a luminance difference of 0.16 and was effectively
# invisible; these two are at 0.34, muted enough to read as disabled and
# contrasty enough to read at all.
DISABLED_BG = "#d7dde5"
DISABLED_FG = "#7c8794"


def set_enabled(widget, enabled: bool) -> None:
    """
    Enable or disable a control, keeping an accent button's APPEARANCE
    honest.

    tk.Button keeps its background when disabled - only the foreground
    changes - so a greyed-out Aggregate still rendered solid blue and
    read as the next thing to click, which is precisely the affordance
    the accent colour exists to carry. ttk widgets handle this
    themselves, so they only need the state flag.
    """
    if widget is None:
        return
    try:
        widget.configure(state="normal" if enabled else "disabled")
        if isinstance(widget, tk.Button) and not isinstance(widget, ttk.Button):
            widget.configure(
                bg=ACCENT if enabled else DISABLED_BG,
                activebackground=ACCENT_ACTIVE if enabled else DISABLED_BG,
                fg="white" if enabled else DISABLED_FG,
                cursor="hand2" if enabled else "arrow",
            )
    except tk.TclError:
        pass


def accent_button(parent, text: str, command, **kw) -> tk.Button:
    """
    The primary-action button.

    Deliberately a plain tk.Button, not ttk: macOS's aqua ttk theme
    ignores background colours on ttk.Button, so an accent ttk button
    renders grey there and the "exactly one highlighted next action"
    affordance silently disappears on the platform this app runs on.
    """
    return tk.Button(
        parent, text=text, command=command,
        bg=ACCENT, fg="white", activebackground=ACCENT_ACTIVE, activeforeground="white",
        font=("Helvetica", 10, "bold"), relief="flat", padx=14, pady=4,
        disabledforeground="#c9d3e6", cursor="hand2", highlightthickness=0,
        **kw
    )


class StepCard(ttk.Frame):
    """One numbered pipeline step in the sidebar."""

    def __init__(self, parent, number: int, title: str, accent: bool = True):
        super().__init__(parent, padding=(10, 8, 10, 10))
        self.number = number
        self.title = title
        self._state = "blocked"

        self.columnconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(2, weight=1)

        self.glyph_var = tk.StringVar(value=GLYPHS["blocked"][0])
        self.glyph_label = ttk.Label(header, textvariable=self.glyph_var,
                                     font=("Helvetica", 12), foreground=MUTED_TEXT, width=2)
        self.glyph_label.grid(row=0, column=0, sticky="w")

        ttk.Label(header, text=f"{number}", font=("Helvetica", 9),
                  foreground=MUTED_TEXT, width=2).grid(row=0, column=1, sticky="w")
        self.title_label = ttk.Label(header, text=title.upper(), font=("Helvetica", 9, "bold"))
        self.title_label.grid(row=0, column=2, sticky="w")

        self.summary_var = tk.StringVar(value="")
        self.summary_label = ttk.Label(self, textvariable=self.summary_var, justify="left",
                                       foreground=MUTED_TEXT, font=("Helvetica", 8))
        self.summary_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self.bind("<Configure>",
                  lambda e: self.summary_label.config(wraplength=max(140, e.width - 26)))

        # Steps put their own parameter widgets in here.
        self.params = ttk.Frame(self)
        self.params.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.actions = ttk.Frame(self)
        self.actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.action: Optional[tk.Widget] = None
        self.extra_buttons: List[tk.Widget] = []
        self._accent = accent

    # -- construction helpers ---------------------------------------------
    def set_action(self, text: str, command: Callable) -> tk.Widget:
        if self._accent:
            self.action = accent_button(self.actions, text, command)
        else:
            self.action = ttk.Button(self.actions, text=text, command=command)
        self.action.pack(side="left")
        set_enabled(self.action, False)
        return self.action

    def add_button(self, text: str, command: Callable, width: int = 0) -> ttk.Button:
        btn = ttk.Button(self.actions, text=text, command=command,
                         **({"width": width} if width else {}))
        btn.pack(side="left", padx=(6, 0))
        btn.configure(state="disabled")
        self.extra_buttons.append(btn)
        return btn

    def add_param(self, label: str, variable: tk.StringVar, width: int = 6,
                  row: int = 0, column: int = 0) -> ttk.Entry:
        holder = ttk.Frame(self.params)
        holder.grid(row=row, column=column, sticky="w", padx=(0, 10), pady=1)
        ttk.Label(holder, text=label, foreground=MUTED_TEXT,
                  font=("Helvetica", 8)).pack(side="left")
        entry = ttk.Entry(holder, textvariable=variable, width=width, font=("Helvetica", 9))
        entry.pack(side="left", padx=(4, 0))
        return entry

    # -- state -------------------------------------------------------------
    def set_state(self, state: str, summary: str = "") -> None:
        """
        state is one of: done, available, blocked, na, running.

        The glyph and the summary always move together - a greyed button
        without a reason beside it is the thing this widget exists to
        prevent.
        """
        self._state = state
        glyph, colour = GLYPHS.get(state, GLYPHS["blocked"])
        self.glyph_var.set(glyph)
        self.glyph_label.configure(foreground=colour)
        self.summary_var.set(summary)

    @property
    def state(self) -> str:
        return self._state


class ReportModal(tk.Toplevel):
    """
    A modal report the user must acknowledge.

    Four details make this genuinely modal rather than merely on top:

      1. transient()  - rides with the main window, stays off the taskbar.
      2. grab_set()   - THIS is what halts the main UI. Without it the
                        user can click Aggregate while the sentinel
                        report is still open, which is exactly the gating
                        hole the modal exists to close.
      3. WM_DELETE_WINDOW is wired to the same handler as Okay. If the
                        title-bar X skipped the acknowledgement, the app
                        would land in a state where the work is done but
                        the UI still says it is not, with no way out but
                        re-running the step.
      4. wait_window() - the caller blocks here, so the handler that
                        opened the modal reads as a straight line and can
                        commit the layer on the line after.

    Save deliberately does NOT dismiss: someone who saves the report and
    then wants to re-read it should not have to re-run the step.
    """

    def __init__(self, parent, title: str, body: str,
                 headline: str = "",
                 save_callback: Optional[Callable[[Path], None]] = None,
                 default_filename: str = "report.txt",
                 on_dismiss: Optional[Callable[[], None]] = None,
                 geometry: str = "980x700",
                 modal: bool = True):
        super().__init__(parent)
        self.title(title)
        self.geometry(geometry)
        self.body = body
        self.headline = headline
        self.saved = False
        self.dismissed = False
        self._save_callback = save_callback
        self._default_filename = default_filename
        self._on_dismiss = on_dismiss

        if headline:
            bar = ttk.Frame(self, padding=(12, 10, 12, 0))
            bar.pack(side="top", fill="x")
            ttk.Label(bar, text=headline, font=("Helvetica", 10, "bold"),
                      wraplength=900, justify="left").pack(side="left")

        # Buttons live at the bottom and are packed BEFORE the text area,
        # so shrinking the window eats the text rather than hiding the
        # only controls that can close a modal window.
        buttons = ttk.Frame(self, padding=(12, 8))
        buttons.pack(side="bottom", fill="x")

        self.okay_button = accent_button(buttons, "Okay", self._dismiss)
        self.okay_button.pack(side="right")
        self.save_button = ttk.Button(buttons, text="Save report…", command=self._save)
        self.save_button.pack(side="right", padx=(0, 8))

        self.hint_var = tk.StringVar(
            value="Save writes the report to a file. Okay closes it and applies the result.")
        ttk.Label(buttons, textvariable=self.hint_var, foreground=MUTED_TEXT,
                  font=("Helvetica", 8)).pack(side="left")

        frame = ttk.Frame(self, padding=(12, 10))
        frame.pack(side="top", fill="both", expand=True)
        yscroll = ttk.Scrollbar(frame, orient="vertical")
        yscroll.pack(side="right", fill="y")
        xscroll = ttk.Scrollbar(frame, orient="horizontal")
        xscroll.pack(side="bottom", fill="x")
        # An explicit foreground is not optional here. With only a
        # background set, this Text inherits the system's default text
        # colour - white under macOS dark mode - and every report renders
        # white-on-white, i.e. blank.
        self.text = tk.Text(frame, wrap="none", font=("Courier", 9), relief="flat",
                            background=PALETTE["surface_alt"], foreground=PALETTE["fg"],
                            insertbackground=PALETTE["fg"],
                            selectbackground=PALETTE["select_bg"],
                            selectforeground=PALETTE["select_fg"],
                            yscrollcommand=yscroll.set,
                            xscrollcommand=xscroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        yscroll.config(command=self.text.yview)
        xscroll.config(command=self.text.xview)
        self.text.insert("1.0", body)
        self.text.config(state="disabled")

        self.protocol("WM_DELETE_WINDOW", self._dismiss)   # X behaves as Okay
        self.bind("<Escape>", lambda _e: self._dismiss())
        self.okay_button.focus_set()

        if modal:
            self.transient(parent)
            try:
                self.grab_set()
            except tk.TclError:
                pass  # headless/offscreen; the rest still behaves

    def _save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save report", defaultextension=".txt",
            initialfile=self._default_filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            if self._save_callback is not None:
                self._save_callback(Path(path))
            else:
                Path(path).write_text(self.body)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Could not save", str(exc))
            return
        self.saved = True
        self.hint_var.set(f"Saved to {Path(path).name}. Click Okay to continue.")

    def _dismiss(self) -> None:
        if self.dismissed:
            return
        self.dismissed = True
        try:
            self.grab_release()
        except tk.TclError:
            pass
        if self._on_dismiss is not None:
            self._on_dismiss()
        self.destroy()

    def wait(self) -> None:
        """Block the caller until the user acknowledges."""
        try:
            self.wait_window()
        except tk.TclError:
            pass
