"""Phase 0 - measured-consist YAML loads with the expected physics.

Tests 0.1, 0.2, 0.3 from docs/PLAN.md Section 10.2.
Validates the measured AFM SL+Cargo Train consist YAML produced by Phase 0
loads cleanly through TrainConfig.from_yaml, yields the expected mass_total
and v_eff(R40), and pickles for use under StarmapParallelization workers
(Rule 11).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src_v2.train.physics import TrainConfig

MEASURED_CONSIST_YAML = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "trains"
    / "measured_consist.yaml"
)


@pytest.fixture(scope="module")
def measured() -> TrainConfig:
    return TrainConfig.from_yaml(MEASURED_CONSIST_YAML)


def test_phase0_yaml_loads_with_measured_mass(measured: TrainConfig) -> None:
    assert measured.mass_total > 0.7
    assert measured.mass_total == pytest.approx(0.820, abs=0.001)


def test_phase0_v_eff_r40_is_slide_bound(measured: TrainConfig) -> None:
    assert measured.v_eff(0.32) == pytest.approx(0.886, abs=0.001)


def test_phase0_train_config_pickles(measured: TrainConfig) -> None:
    blob = pickle.dumps(measured)
    assert pickle.loads(blob) == measured
