# LEGO Track Optimizer - Source Package
#
# NOTE: Imports reduced during BRKGA→NSGA-II encoding migration.
# Modules not yet migrated (sampling, operators, repair, survival, etc.)
# are imported lazily to avoid import errors.

from .data import (
    TrackCatalog,
    TrackPiece,
)
from .config import (
    OptimizationConfig,
)

__all__ = [
    # Data
    "TrackCatalog",
    "TrackPiece",
    # Config
    "OptimizationConfig",
]
