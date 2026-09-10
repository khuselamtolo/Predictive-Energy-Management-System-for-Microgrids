"""
Lock the app to a light appearance.

The problem this solves: on macOS, Tk's native "aqua" ttk theme follows
the system appearance, so turning on Dark Mode turns every ttk widget
dark - but the app also sets explicit light colours in places (the
report and statistics panes use a near-white background, the plot is
drawn on white). The result is not a dark app; it is a broken one.
Worst of all, a tk.Text given `background="#f7f8fa"` and no `foreground`
inherits the system's *white* default text colour in dark mode, so the
reports render white-on-white and read as blank.

Two things are therefore needed, and both are done here:

  1. **Stop ttk from going dark.** The aqua theme ignores colour options
     - you cannot simply tell it to be light - so when the system is in
     dark mode we switch to the "clam" theme, whose colours we control
     completely. In light mode aqua is left alone, so the app still
     looks native on the platform it runs on.

  2. **Never rely on a default colour.** Every classic Tk widget
     (Text, Canvas, Listbox, Toplevel) gets an explicit foreground AND
     background through the option database, so nothing inherits a
     system colour that happens to be wrong.

If you would rather the app follow the system into dark mode, that is a
different job - a full dark palette for the charts and the report panes
- and is not what this module does. Set LOAD_PROFILE_THEME=system in the
environment to disable the override entirely and take whatever the
platform gives you.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tkinter as tk
from tkinter import ttk

# The light palette. Deliberately the same greys the report panes and the
# sidebar were already using, so forcing light changes nothing visually
# on a machine that was in light mode already.
PALETTE = {
    "bg": "#f0f2f5",          # window and frame ground
    "surface": "#ffffff",     # entries, text areas, canvas
    "surface_alt": "#f7f8fa", # report and statistics panes
    "fg": "#1a1f26",          # body text
    "fg_muted": "#5b6472",
    "border": "#c3cbd6",
    "select_bg": "#cfe0f5",
    "select_fg": "#101418",
    "disabled_fg": "#9aa4b2",
    "accent": "#2563eb",
}


def system_is_dark() -> bool:
    """
    True when the OS is currently in dark mode.

    macOS reports this through the global domain: the key is simply
    absent in light mode, which is why a non-zero exit is treated as
    "light" rather than an error. Other platforms have no single
    reliable signal, so they are reported as light and left alone.
    """
    if platform.system() != "Darwin":
        return False
    try:
        out = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=2,
        )
        return out.returncode == 0 and "dark" in out.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _configure_ttk(style: ttk.Style) -> None:
    """Paint every ttk widget class we actually use from the palette."""
    p = PALETTE
    style.configure(".", background=p["bg"], foreground=p["fg"],
                    fieldbackground=p["surface"], bordercolor=p["border"],
                    lightcolor=p["bg"], darkcolor=p["bg"],
                    troughcolor=p["bg"], focuscolor=p["accent"])
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["fg"])
    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["fg"])
    style.configure("TPanedwindow", background=p["bg"])
    style.configure("TSeparator", background=p["border"])

    style.configure("TButton", background=p["surface"], foreground=p["fg"],
                    bordercolor=p["border"], focusthickness=1, padding=(8, 3))
    style.map("TButton",
              background=[("pressed", p["select_bg"]), ("active", "#e8edf4"),
                          ("disabled", p["bg"])],
              foreground=[("disabled", p["disabled_fg"])])

    for cls in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(cls, fieldbackground=p["surface"], background=p["surface"],
                        foreground=p["fg"], bordercolor=p["border"],
                        insertcolor=p["fg"], arrowcolor=p["fg"])
        style.map(cls,
                  fieldbackground=[("readonly", p["surface"]), ("disabled", p["bg"])],
                  foreground=[("disabled", p["disabled_fg"])],
                  arrowcolor=[("disabled", p["disabled_fg"])])
    # The dropdown LIST of a Combobox is a classic Tk listbox living
    # outside the ttk theme, so it has to be coloured through the option
    # database or it stays dark on its own.
    style.map("TCombobox", selectbackground=[("readonly", p["select_bg"])],
              selectforeground=[("readonly", p["select_fg"])])

    style.configure("Vertical.TScrollbar", background=p["surface"],
                    troughcolor=p["bg"], bordercolor=p["border"], arrowcolor=p["fg"])
    style.configure("Horizontal.TScrollbar", background=p["surface"],
                    troughcolor=p["bg"], bordercolor=p["border"], arrowcolor=p["fg"])
    style.configure("TCheckbutton", background=p["bg"], foreground=p["fg"])
    style.configure("TRadiobutton", background=p["bg"], foreground=p["fg"])
    style.configure("TNotebook", background=p["bg"])
    style.configure("TNotebook.Tab", background=p["bg"], foreground=p["fg"])


def _configure_classic(root: tk.Misc) -> None:
    """
    Explicit colours for the non-ttk widgets, via the option database so
    Toplevels created later (the report modals) inherit them too.

    Every entry here pairs a background WITH a foreground. A background
    on its own is the bug: a Text told to be near-white while inheriting
    the system's white text renders blank in dark mode.
    """
    p = PALETTE
    o = root.option_add
    o("*background", p["bg"])
    o("*foreground", p["fg"])
    o("*Toplevel.background", p["bg"])
    o("*Canvas.background", p["bg"])
    o("*Canvas.highlightBackground", p["bg"])

    for cls in ("Text", "Listbox", "Entry", "Spinbox"):
        o(f"*{cls}.background", p["surface"])
        o(f"*{cls}.foreground", p["fg"])
        o(f"*{cls}.selectBackground", p["select_bg"])
        o(f"*{cls}.selectForeground", p["select_fg"])
        o(f"*{cls}.highlightBackground", p["border"])
    o("*Text.insertBackground", p["fg"])
    o("*Entry.insertBackground", p["fg"])

    # tk.Menu is what a Combobox drop-down and any context menu use.
    o("*Menu.background", p["surface"])
    o("*Menu.foreground", p["fg"])
    o("*Menu.activeBackground", p["select_bg"])
    o("*Menu.activeForeground", p["select_fg"])

    # The Combobox popdown listbox is addressed by widget path, not class.
    try:
        root.tk.eval(f"""
            option add *TCombobox*Listbox.background {p['surface']}
            option add *TCombobox*Listbox.foreground {p['fg']}
            option add *TCombobox*Listbox.selectBackground {p['select_bg']}
            option add *TCombobox*Listbox.selectForeground {p['select_fg']}
        """)
    except tk.TclError:
        pass


def apply_light_theme(root: tk.Misc) -> dict:
    """
    Force the light appearance. Call once, before building any widgets.

    Returns a small report of what it did, which the app puts in the
    status bar so the behaviour is visible rather than mysterious.
    """
    mode = os.environ.get("LOAD_PROFILE_THEME", "light").strip().lower()
    dark = system_is_dark()
    if mode == "system":
        return {"applied": False, "system_dark": dark, "theme": None,
                "note": "LOAD_PROFILE_THEME=system — following the OS appearance."}

    style = ttk.Style(root)
    available = set(style.theme_names())
    original = style.theme_use()

    # aqua ignores colour options outright, so when the OS is dark the
    # only way to get a light window is to leave aqua behind. In light
    # mode it is left alone, keeping the native look on macOS.
    switched = None
    if dark or mode == "force":
        for candidate in ("clam", "alt", "default"):
            if candidate in available:
                try:
                    style.theme_use(candidate)
                    switched = candidate
                    break
                except tk.TclError:
                    continue

    _configure_ttk(style)
    _configure_classic(root)
    try:
        root.configure(background=PALETTE["bg"])
    except tk.TclError:
        pass

    if switched:
        note = (f"System is in dark mode; using the '{switched}' widget theme so the app "
                "stays light. Set LOAD_PROFILE_THEME=system to follow the OS instead.")
    else:
        note = "Light appearance locked."
    return {"applied": True, "system_dark": dark, "theme": switched or original, "note": note}


def style_axes(figure, ax) -> None:
    """Keep matplotlib light too, whatever rcParams or style is in force."""
    figure.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["fg"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.xaxis.label.set_color(PALETTE["fg"])
    ax.yaxis.label.set_color(PALETTE["fg"])
    ax.title.set_color(PALETTE["fg"])
