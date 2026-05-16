"""Algorithm package: GA assembly and run orchestration.

Wraps pymoo's R-NSGA-II (Deb & Sundar, AAAI 2006) with project-specific
operators, callbacks, and constraint handling so callers need only pass
an ``OptimizationConfig`` and a ``TrackCatalog``. Preference for high
utilization is expressed via a utopian reference point and asymmetric
modified-crowding weights (utilization weighted 0.85 vs speed 0.15).
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
