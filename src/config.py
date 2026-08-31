"""Pydantic v2 configuration models for optimization and physics parameters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from .train import TrainConfig


class _StrictModel(BaseModel):
    """Base for every config model: an unknown key is an error, not a silent drop.

    Pydantic would otherwise discard a misspelled key and leave the field on its
    default, so the run would not be the one the file describes. Mutability stays:
    the loader sets _base_dir, and the ablation driver toggles component flags.
    """

    model_config = ConfigDict(extra="forbid")


class BoundaryConfig(_StrictModel):
    """Spatial boundary constraints for the track layout."""

    min_x: float = Field(default=-100.0, description="Minimum X coordinate in studs")
    max_x: float = Field(default=100.0, description="Maximum X coordinate in studs")
    min_y: float = Field(default=-100.0, description="Minimum Y coordinate in studs")
    max_y: float = Field(default=100.0, description="Maximum Y coordinate in studs")

    @property
    def width(self) -> float:
        """Width of the boundary in studs."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Height of the boundary in studs."""
        return self.max_y - self.min_y

    @property
    def diagonal(self) -> float:
        """Diagonal length for normalization."""
        return math.sqrt(self.width**2 + self.height**2)


class TerminationConfig(_StrictModel):
    """pymoo termination criteria."""

    n_max_gen: int = Field(default=1000, ge=1, description="Maximum generations")
    ftol: float = Field(default=1e-6, ge=0, description="Objective tolerance")
    xtol: float = Field(default=1e-6, ge=0, description="Variable tolerance")
    period: int = Field(
        default=0, ge=0,
        description="Stagnation window (generations) for improvement-based early "
                    "stop; 0 (default) disables it — the run uses the full n_gen "
                    "budget. Set >0 to stop after that many stagnant generations.",
    )


class SearchComponentsConfig(_StrictModel):
    """Ablation toggles for the search components added on top of pymoo.

    All-on is the production system. All-off is the ablation baseline: the same
    problem (encoding, decoder, objectives, constraints) solved with stock pymoo
    modules only -- IntegerRandomSampling, SBX/PM with RoundingRepair, and
    NSGA-II's own RankAndCrowding survival with unweighted CV.
    """

    heuristic_sampling: bool = Field(
        default=True, description="Heuristic seed sampling vs IntegerRandomSampling",
    )
    custom_operators: bool = Field(
        default=True, description="Partitioned crossover/mutation vs stock SBX/PM",
    )
    repair: bool = Field(
        default=True, description="TrackRepairPipeline vs no repair",
    )
    adaptive_epsilon: bool = Field(
        default=True, description="LegoAdaptiveEpsilon wrapper vs plain NSGA2",
    )
    elite_injection: bool = Field(
        default=True, description="Feasible + category elite re-injection callbacks",
    )
    constr_survival: bool = Field(
        default=True,
        description="ConstrRankAndCrowding (constraint-vector NDS for infeasibles) vs "
                    "NSGA-II's own default RankAndCrowding (scalar-CV truncation)",
    )


class AlgorithmConfig(_StrictModel):
    """NSGA-II algorithm parameters."""

    name: Literal["NSGA2"] = Field(
        default="NSGA2",
        description="Algorithm: NSGA2 (Deb's feasibility-first via ConstrRankAndCrowding)",
    )
    pop_size: int = Field(default=1000, ge=10, description="Population size")
    n_gen: int = Field(default=1000, ge=1, description="Number of generations")
    heuristic_ratio: float = Field(
        default=0.20, ge=0.0, le=0.5, description="Fraction of initial pop from heuristics",
    )
    crossover_prob: float = Field(default=0.2, ge=0, le=1, description="Crossover probability")
    mutation_prob: float = Field(default=0.8, ge=0, le=1, description="Mutation probability")
    eliminate_duplicates: bool = Field(default=True, description="Remove duplicate solutions")
    crowding_func: Literal["cd", "pcd", "ce", "mnn", "2nn"] = Field(
        default="cd",
        description="Crowding metric the survival operator sorts the splitting front by. "
                    "'cd' is NSGA-II's original; 'pcd' additionally scores objective-space "
                    "duplicates as zero, which pymoo recommends for two-objective problems.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    components: SearchComponentsConfig = Field(default_factory=SearchComponentsConfig)


class OptimizationConfig(_StrictModel):
    """Complete optimization configuration."""

    inventory: Dict[str, int] = Field(
        default_factory=dict, description="Available pieces {piece_id: count}",
    )
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    closure_tolerance: float = Field(
        default=4.0, ge=0.1, description="Position closure tolerance in studs",
    )
    angle_tolerance: float = Field(
        default=5.0, ge=0.5, description="Angle closure tolerance in degrees",
    )
    boundary_tolerance: float = Field(
        default=2.0, ge=0.0, description="Boundary overshoot tolerance in studs",
    )
    special_piece_weight: float = Field(
        default=3.0, ge=1.0,
        description="Utilization weight per special piece (switch pair / crossing / "
                    "double-crossover). >1 rewards multi-path topology so the GA does "
                    "not strip it as overhead.",
    )
    f0_objective: Literal["piece_score", "route_length"] = Field(
        default="piece_score",
        description="F[0] variant: 'piece_score' maximizes the weighted piece score; "
                    "'route_length' maximizes the summed length in studs of every "
                    "unique circuit, so special pieces earn the routes they open.",
    )
    f1_speed_model: Literal["physics", "constant"] = Field(
        default="physics",
        description="F[1] speed model: 'physics' profiles the train over every circuit "
                    "(3-pass profiler); 'constant' charges every segment at "
                    "f1_constant_speed instead.",
    )
    f1_constant_speed: float = Field(
        default=0.8, gt=0.0,
        description="Speed in m/s charged to every segment when "
                    "f1_speed_model='constant'. Unused under 'physics'.",
    )
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    train_config_path: str = Field(
        description="Path to the train physics YAML, relative to this config file. "
                    "Required: locomotive physics has no code-level fallback.",
    )
    n_workers: int = Field(default=1, ge=1, description="Parallelization workers")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @field_validator("inventory")
    @classmethod
    def validate_inventory(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure all inventory counts are non-negative."""
        for piece_id, count in v.items():
            if count < 0:
                raise ValueError(
                    f"Inventory count for {piece_id} must be non-negative, got {count}"
                )
        return v

    @property
    def total_inventory(self) -> int:
        """Total number of pieces available."""
        return sum(self.inventory.values())

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config = cls.model_validate(data)
        config._base_dir = path.parent
        return config

    @property
    def train_config_file(self) -> Path:
        """Resolved path of the train physics YAML, relative to this config's own file."""
        return self._base_dir / self.train_config_path

    def load_train_config(self) -> TrainConfig:
        """Load the train physics YAML named by train_config_path."""
        return TrainConfig.from_yaml(self.train_config_file)
