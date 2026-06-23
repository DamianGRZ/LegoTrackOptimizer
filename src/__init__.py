# LEGO Track Optimizer - Source Package
#
# Curated top-level re-exports; import submodules directly for everything else
# (e.g. `from src.encoding import ...`).

from .config import OptimizationConfig
from .catalog import TrackCatalog, TrackPiece

__all__ = [
    "OptimizationConfig",
    "TrackCatalog",
    "TrackPiece",
]
