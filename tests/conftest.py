"""Shared pytest fixtures for LEGO Track Optimizer tests."""

from pathlib import Path

import pytest

from src.config import BoundaryConfig, OptimizationConfig
from src.catalog import TrackCatalog
from src.train import TrainConfig


@pytest.fixture
def catalog() -> TrackCatalog:
    """Load the track catalog (v2 port-centric schema)."""
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def train_config() -> TrainConfig:
    """Default train physics configuration."""
    return TrainConfig()


@pytest.fixture
def measured_train_config() -> TrainConfig:
    """Train physics from measured AFM SL+Cargo M0015TW consist (2026-05-06)."""
    return TrainConfig.from_yaml(
        Path(__file__).parent.parent / "configs/trains/measured_consist.yaml"
    )


@pytest.fixture
def boundary() -> BoundaryConfig:
    """Default boundary configuration."""
    return BoundaryConfig()


@pytest.fixture
def default_config() -> OptimizationConfig:
    """Load default optimization configuration."""
    return OptimizationConfig.load("configs/default.yaml")


@pytest.fixture
def inventory(default_config: OptimizationConfig) -> dict:
    """Inventory from default configuration."""
    return default_config.inventory


@pytest.fixture
def compact_config() -> OptimizationConfig:
    """Load compact optimization configuration."""
    return OptimizationConfig.load("configs/compact.yaml")


@pytest.fixture
def switches_config() -> OptimizationConfig:
    """Load with_switches optimization configuration."""
    return OptimizationConfig.load("configs/with_switches.yaml")


@pytest.fixture
def crossing_config() -> OptimizationConfig:
    """Load with_crossing optimization configuration."""
    return OptimizationConfig.load("configs/with_crossing.yaml")
