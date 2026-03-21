"""Shared pytest fixtures for LEGO Track Optimizer tests."""

import pytest

from src.config import BoundaryConfig, OptimizationConfig, PhysicsConfig
from src.data import TrackCatalog


@pytest.fixture
def catalog() -> TrackCatalog:
    """Load track catalog from YAML."""
    return TrackCatalog.load("data/track_pieces.yaml")


@pytest.fixture
def physics() -> PhysicsConfig:
    """Default physics configuration."""
    return PhysicsConfig()


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
