"""Tests for ``tests/fixtures/regression_baselines.py`` (Phase 17.C.5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.mini_problem import mini_optimization_run
from tests.fixtures.regression_baselines import (
    compare_to_baseline,
    extract_metrics,
    load_baseline,
    save_baseline,
)


# ---------------------------------------------------------------- RB.1
def test_extract_metrics_from_real_result(tmp_path):
    result = mini_optimization_run(tmp_path, n_gen=5, pop_size=20)
    metrics = extract_metrics(result)

    assert set(metrics) == {
        "feasibility_rate", "n_feasible", "n_pop",
        "mean_util_feasible", "max_util_feasible", "min_min_speed_feasible",
    }
    assert metrics["n_pop"] == 20
    assert 0 <= metrics["feasibility_rate"] <= 1
    assert isinstance(metrics["n_feasible"], int)


# ---------------------------------------------------------------- RB.2
def test_save_then_load_baseline_roundtrip(tmp_path):
    metrics = {
        "feasibility_rate": 0.78,
        "n_feasible": 50,
        "n_pop": 64,
        "mean_util_feasible": 0.62,
        "max_util_feasible": 0.78,
        "min_min_speed_feasible": 0.886,
    }
    save_baseline(metrics, tmp_path, "default")
    loaded = load_baseline(tmp_path, "default")

    assert loaded == metrics
    assert (tmp_path / "default.json").exists()


# ---------------------------------------------------------------- RB.3
def test_compare_within_tolerance_flags_within_band(tmp_path):
    baseline = {"feasibility_rate": 0.78, "mean_util_feasible": 0.62}
    current = {"feasibility_rate": 0.80, "mean_util_feasible": 0.61}  # +2.6% / -1.6%

    deltas = compare_to_baseline(current, baseline, tolerance=0.05)

    assert deltas["feasibility_rate"]["within_band"] is True
    assert deltas["mean_util_feasible"]["within_band"] is True


# ---------------------------------------------------------------- RB.4
def test_compare_out_of_tolerance_flags_out_of_band(tmp_path):
    baseline = {"feasibility_rate": 0.78, "mean_util_feasible": 0.62}
    current = {"feasibility_rate": 0.50, "mean_util_feasible": 0.40}  # -36% / -35%

    deltas = compare_to_baseline(current, baseline, tolerance=0.05)

    assert deltas["feasibility_rate"]["within_band"] is False
    assert deltas["mean_util_feasible"]["within_band"] is False
    assert deltas["feasibility_rate"]["rel_delta"] < -0.3


def test_compare_handles_none_values_gracefully():
    baseline = {"min_min_speed_feasible": None}
    current = {"min_min_speed_feasible": 0.886}
    deltas = compare_to_baseline(current, baseline)

    assert deltas["min_min_speed_feasible"]["within_band"] is None
