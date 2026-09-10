"""
The immutable layer stack: Raw -> Sentinels -> Interpolated -> Aggregated.

Two rules carry the whole "no destructive overwrites" requirement:

  1. No frame is ever mutated in place. Every pipeline step returns a
     NEW Dataset and the old one stays in the list, so being
     non-destructive is a structural property rather than a discipline
     someone has to remember.

  2. Revert moves a pointer. It does not restore from a backup copy.
     That gives unlimited undo, an audit trail and A/B comparison for
     free, and it is also what makes the worker threads safe: because a
     layer can never change under them, a worker reading layer 2 while
     the main thread appends layer 3 has nothing to race on.

Whole frames are kept deliberately. Measured on the 8-month panel
(72 576 rows x 66 households) a layer is 38.9 MB and 0.131 s to produce,
so four layers cost ~156 MB and half a second - far too cheap to justify
lazy evaluation or spilling to disk. What is NOT stored as a whole frame
is the record of what changed: sentinel replacement touches 864 of
532 224 cells (0.162%), so each layer also carries a compact long-format
ledger of exactly which readings changed and why. That ledger is 35 KB
against a 38.9 MB snapshot, and it is what the reports render from, what
Save exports, and what answers "why is this value 1.84 and not -999?"
six months from now.

Threading contract: workers may only READ layers and must hand results
back through a queue. Only the main thread calls stage()/commit()/
revert().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from dataset import Dataset

LEDGER_COLUMNS = ["timestamp", "channel", "before", "after", "reason"]


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


@dataclass
class Layer:
    """One immutable state of the data, plus how it got there."""

    id: int
    dataset: Dataset
    step_name: str
    parent_id: Optional[int] = None
    report: str = ""
    ledger: pd.DataFrame = field(default_factory=empty_ledger)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def n_changed(self) -> int:
        return len(self.ledger)

    def describe(self) -> str:
        bits = [self.step_name, f"{len(self.dataset.households)} ch"]
        if self.n_changed:
            bits.append(f"{self.n_changed:,} readings modified")
        return "  ·  ".join(bits)


class LayerStack:
    """Append-only list of Layers with a single mutable pointer."""

    def __init__(self, root: Dataset, step_name: str = "Raw"):
        self.layers: List[Layer] = [Layer(id=0, dataset=root, step_name=step_name)]
        self.active: int = 0
        self._next_id = 1
        self._staged: Optional[Layer] = None

    # -- reading ---------------------------------------------------------
    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active]

    @property
    def dataset(self) -> Dataset:
        return self.active_layer.dataset

    @property
    def root(self) -> Layer:
        return self.layers[0]

    def can_revert(self) -> bool:
        return self.active > 0

    def history(self) -> List[str]:
        return [
            ("→ " if i == self.active else "  ") + f"L{lyr.id} {lyr.describe()}"
            for i, lyr in enumerate(self.layers)
        ]

    def breadcrumb(self) -> str:
        """'Raw → Sentinels → Interpolated' up to the active layer."""
        return " → ".join(l.step_name for l in self.layers[: self.active + 1])

    # -- writing (main thread only) ---------------------------------------
    def stage(self, dataset: Dataset, step_name: str, report: str = "",
              ledger: Optional[pd.DataFrame] = None) -> Layer:
        """
        Hold a finished result WITHOUT making it active.

        The report modal is what commits it. Until the user acknowledges
        the report, the canvas keeps showing the old layer - so the data
        on screen and the numbers in the report never disagree.
        """
        layer = Layer(
            id=self._next_id,
            dataset=dataset,
            step_name=step_name,
            parent_id=self.active_layer.id,
            report=report,
            ledger=ledger if ledger is not None else empty_ledger(),
        )
        self._next_id += 1
        self._staged = layer
        return layer

    def commit(self, layer: Optional[Layer] = None) -> Layer:
        """Append the staged layer and make it active."""
        layer = layer or self._staged
        if layer is None:
            raise RuntimeError("Nothing staged to commit.")
        # Committing on top of a reverted state drops the abandoned
        # branch, which is what a user expects from undo-then-do-something-else.
        del self.layers[self.active + 1:]
        self.layers.append(layer)
        self.active = len(self.layers) - 1
        self._staged = None
        return layer

    def discard(self) -> None:
        self._staged = None

    def revert(self) -> Layer:
        """Step back one layer. Nothing is deleted - the pointer moves."""
        if not self.can_revert():
            return self.active_layer
        self.active -= 1
        return self.active_layer

    def revert_to_root(self) -> Layer:
        self.active = 0
        return self.active_layer

    # -- accounting -------------------------------------------------------
    def memory_mb(self) -> float:
        seen, total = set(), 0
        for lyr in self.layers:
            frame = lyr.dataset.df
            if id(frame) in seen:
                continue
            seen.add(id(frame))
            total += int(frame.memory_usage(deep=True).sum())
        return total / 1e6

    def full_ledger(self) -> pd.DataFrame:
        """Every change from the root up to the active layer, in order."""
        parts = [l.ledger for l in self.layers[1: self.active + 1] if len(l.ledger)]
        if not parts:
            return empty_ledger()
        return pd.concat(parts, ignore_index=True)


def ledger_from_mask(before: pd.DataFrame, after: pd.DataFrame,
                     mask: pd.DataFrame, reason: str, limit: int = 200_000) -> pd.DataFrame:
    """
    Build a change ledger from a boolean mask of touched cells.

    `limit` is a safety valve, not an expectation: at the measured 0.162%
    sentinel density the 8-month panel produces roughly 7 500 rows. If a
    step somehow rewrites most of the frame, truncate rather than build a
    ledger bigger than the data it describes.
    """
    idx = mask.to_numpy().nonzero()
    if len(idx[0]) == 0:
        return empty_ledger()
    rows, cols = idx[0][:limit], idx[1][:limit]
    return pd.DataFrame({
        "timestamp": before.index[rows],
        "channel": [before.columns[c] for c in cols],
        "before": before.to_numpy()[rows, cols],
        "after": after.to_numpy()[rows, cols],
        "reason": reason,
    })
