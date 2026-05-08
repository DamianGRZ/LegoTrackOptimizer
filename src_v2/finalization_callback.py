"""Gating callback that toggles repair.finalization_active on the repair pipeline.

When ``algorithm.termination.perc > 0.9``, the repair pipeline begins
attempting branch completion on incomplete switches. Below 0.9, incomplete
switches are tolerated as soft-penalized intermediate states.

This stagger avoids a non-stationary feasibility shock during the
``LegoAdaptiveEpsilon`` relaxation window (t ∈ [0.2, 0.9]).
"""
from __future__ import annotations

from pymoo.core.callback import Callback


class FinalizationGatingCallback(Callback):
    """Sets ``repair.finalization_active`` based on termination progress."""

    def __init__(self, repair_pipeline, threshold: float = 0.9) -> None:
        super().__init__()
        self._repair = repair_pipeline
        self._threshold = threshold

    def notify(self, algorithm) -> None:
        perc = getattr(algorithm.termination, "perc", 0.0) or 0.0
        self._repair.finalization_active = perc > self._threshold
