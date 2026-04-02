# LEGO Track Optimizer - Source Package

from .config import OptimizationConfig
from .data import TrackCatalog, TrackPiece
from .decoder import DecoderConfig, decode_chromosome
from .encoding import ChromosomeDimensions, compute_dimensions, generate_bounds
from .evaluation import SpeedProfile, compute_speed_profile
from .geometry import Layout, build_layout, compute_fk_chain
from .operators import TrackMutation, UniformNodeCrossover
from .problem import TrackOptimizationProblem
from .repair import TrackRepairPipeline
from .sampling import IntegerSampling

__all__ = [
    "OptimizationConfig",
    "TrackCatalog",
    "TrackPiece",
    "DecoderConfig",
    "decode_chromosome",
    "ChromosomeDimensions",
    "compute_dimensions",
    "generate_bounds",
    "SpeedProfile",
    "compute_speed_profile",
    "Layout",
    "build_layout",
    "compute_fk_chain",
    "TrackMutation",
    "UniformNodeCrossover",
    "TrackOptimizationProblem",
    "TrackRepairPipeline",
    "IntegerSampling",
]
