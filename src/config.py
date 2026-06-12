"""Pydantic v2 configuration models for optimization and physics parameters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .train import TrainConfig


class BoundaryConfig(BaseModel):
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


class TerminationConfig(BaseModel):
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


class AlgorithmConfig(BaseModel):
    """NSGA-II algorithm parameters."""

    name: Literal["NSGA2", "RNSGA2"] = Field(default="NSGA2", description="Algorithm: NSGA2 (Deb's feasibility-first via ConstrRankAndCrowding) or RNSGA2 (preference-guided)")
    pop_size: int = Field(default=1000, ge=10, description="Population size")
    n_gen: int = Field(default=1000, ge=1, description="Number of generations")
    heuristic_ratio: float = Field(default=0.20, ge=0.0, le=0.5, description="Fraction of initial pop from heuristics")
    crossover_prob: float = Field(default=0.9, ge=0, le=1, description="Crossover probability")
    mutation_prob: float = Field(default=0.1, ge=0, le=1, description="Mutation probability")
    eliminate_duplicates: bool = Field(default=True, description="Remove duplicate solutions")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    termination: TerminationConfig = Field(default_factory=TerminationConfig)


class OptimizationConfig(BaseModel):
    """Complete optimization configuration."""

    inventory: Dict[str, int] = Field(default_factory=dict, description="Available pieces {piece_id: count}")
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    closure_tolerance: float = Field(default=4.0, ge=0.1, description="Position closure tolerance in studs")
    angle_tolerance: float = Field(default=5.0, ge=0.5, description="Angle closure tolerance in degrees")
    boundary_tolerance: float = Field(default=2.0, ge=0.0, description="Boundary overshoot tolerance in studs")
    special_piece_weight: float = Field(
        default=3.0, ge=1.0,
        description="Utilization weight per special piece (switch pair / crossing / "
                    "double-crossover). >1 rewards multi-path topology so the GA does "
                    "not strip it as overhead.",
    )
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    train_config_path: Optional[str] = Field(
        default=None,
        description="Path to TrainConfig YAML, relative to this config file.",
    )
    n_workers: int = Field(default=1, ge=1, description="Parallelization workers")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @field_validator("inventory")
    @classmethod
    def validate_inventory(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure all inventory counts are non-negative."""
        for piece_id, count in v.items():
            if count < 0:
                raise ValueError(f"Inventory count for {piece_id} must be non-negative, got {count}")
        return v

    @property
    def total_inventory(self) -> int:
        """Total number of pieces available."""
        return sum(self.inventory.values())

    def calculate_max_layout_pieces(self, safety_margin: float = 10.0) -> int:
        """Calculate maximum chromosome size based on boundary dimensions.

        Args:
            safety_margin: Extra studs of clearance from boundary edges.

        Returns:
            Maximum number of pieces that could fit in the layout.
        """
        # Base circle requires 16 R40 curves (approx 80 studs diameter)
        base_circle_width = 80.0
        straight_length = 16.0
        min_curves_for_closure = 16

        # Calculate available space after margins
        available_width = self.boundary.width - 2 * safety_margin
        available_height = self.boundary.height - 2 * safety_margin

        # Check if base circle fits
        if available_width < base_circle_width or available_height < base_circle_width:
            return min_curves_for_closure

        # Calculate extra space for straights
        extra_space = max(0, min(available_width, available_height) - base_circle_width)
        max_straights = int((extra_space / straight_length) * 2)

        # Total pieces, capped by inventory
        total = min_curves_for_closure + max_straights
        total = min(total, self.total_inventory)

        # Add 10% buffer for mutations
        return int(total * 1.1)

    @property
    def n_var(self) -> int:
        """Number of decision variables for pymoo."""
        return self.calculate_max_layout_pieces()

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config = cls.model_validate(data)
        config._base_dir = path.parent
        return config

    def load_train_config(self) -> TrainConfig:
        """Load the TrainConfig referenced by train_config_path, or defaults."""
        if self.train_config_path is None:
            return TrainConfig()
        return TrainConfig.from_yaml(self._base_dir / self.train_config_path)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
