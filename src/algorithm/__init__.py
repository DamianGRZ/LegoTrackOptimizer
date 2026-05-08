"""Algorithm package: GA assembly and run orchestration.

Wraps pymoo's NSGA-II with project-specific operators, callbacks,
and constraint handling so callers need only pass an
``OptimizationConfig`` and a ``TrackCatalog``.
"""

from src.algorithm.monitoring import ConvergenceMonitorCallback
from src.algorithm.runner import (
    LegoAdaptiveEpsilon,
    ProgressCallback,
    log_piece_usage,
    run_optimization,
    save_results,
)

__all__ = [
    "ConvergenceMonitorCallback",
    "LegoAdaptiveEpsilon",
    "ProgressCallback",
    "log_piece_usage",
    "run_optimization",
    "save_results",
]
