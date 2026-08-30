"""Train physics package — lateral stability, speed profiling, scoring.

Public API:
    TrainConfig — immutable locomotive physics parameters, loaded from a train YAML
    TrainConfigError — raised when that YAML is unreadable or invalid
    v_eff_array — vectorized speed cap over radius array
    available_accel — capped friction-circle longitudinal acceleration
    SpeedProfile — time-optimal speed profile result
    compute_speed_profile — 3-pass speed profiling algorithm
"""

from .physics import (
    TrainConfig,
    TrainConfigError,
    available_accel,
    v_eff_array,
)
from .scoring import SpeedProfile, compute_speed_profile
from .evaluation import PhysicalEvaluation, evaluate_layout

__all__ = [
    "TrainConfig",
    "TrainConfigError",
    "available_accel",
    "v_eff_array",
    "SpeedProfile",
    "compute_speed_profile",
    "PhysicalEvaluation",
    "evaluate_layout",
]
