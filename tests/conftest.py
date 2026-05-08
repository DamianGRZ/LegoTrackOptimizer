"""Shared pytest fixtures for LEGO Track Optimizer tests."""

import pytest

from src.config import BoundaryConfig, OptimizationConfig
from src.catalog import TrackCatalog
from src.train import TrainConfig


@pytest.fixture
def catalog() -> TrackCatalog:
    """Load track catalog from YAML (v2 schema to avoid deprecation noise)."""
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture
def train_config() -> TrainConfig:
    """Default train physics configuration."""
    return TrainConfig()


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
