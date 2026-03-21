# LEGO Track Optimizer - Source Package

# Re-export key classes for convenient imports
from .problem import (
    TrackOptimizationProblem,
    ConvergenceTracker,
    EpsilonTightening,
)
from .island_model import (
    IslandConfig,
    IslandOptimizer,
    run_island_optimization,
)
from .sampling import (
    MultiSegmentSampling,
    HeuristicSampling,
)
from .data import (
    TrackCatalog,
    TrackPiece,
)
from .config import (
    OptimizationConfig,
)

__all__ = [
    # Problem
    "TrackOptimizationProblem",
    "ConvergenceTracker",
    "EpsilonTightening",
    # Island Model
    "IslandConfig",
    "IslandOptimizer",
    "run_island_optimization",
    # Sampling
    "MultiSegmentSampling",
    "HeuristicSampling",
    # Data
    "TrackCatalog",
    "TrackPiece",
    # Config
    "OptimizationConfig",
]
