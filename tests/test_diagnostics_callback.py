"""Tests for ``src_v2.instrumentation.DiagnosticsCallback`` (Phase 17.A, PLAN Section 10.4).

Off-catalog tests authored under user approval; PLAN Section 10.2 has no entry
for the Pre-Phase-1 scaffolding deliverables.
"""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from src_v2.instrumentation import DiagnosticsCallback


T_CATALOG = 4
N_CONSTRAINTS = 11 + T_CATALOG


def _stub_algorithm(F: np.ndarray, G: np.ndarray, X=None, n_gen: int = 1):
    """Stub algorithm whose ``.pop.get(key)`` returns the right array, or None."""
    cv = np.maximum(G, 0).sum(axis=1, keepdims=True)
    data = {"F": F, "G": G, "CV": cv}
    if X is not None:
        data["X"] = X
    pop = SimpleNamespace(get=lambda key, _d=data: _d.get(key))
    return SimpleNamespace(pop=pop, n_gen=n_gen)


def _expected_columns(n_constraints: int) -> list[str]:
    return (
        ["gen", "n_feasible", "n_infeasible",
         "mean_cv", "median_cv", "p90_cv", "p99_cv"]
        + [f"constraint_{i}_mean" for i in range(n_constraints)]
        + [f"constraint_{i}_p90" for i in range(n_constraints)]
        + ["n_chromosomes_with_switches", "n_chromosomes_with_crossings",
           "mean_pre_repair_cv", "mean_post_repair_cv",
           "dedupe_rejection_rate",
           "mean_util_feasible", "max_util_feasible",
           "mean_min_speed_feasible", "max_min_speed_feasible"]
    )


# ---------------------------------------------------------------- 17A.1
def test_csv_header_matches_section_10_4_schema(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)
    F = np.array([[-0.5, -0.886], [-0.4, -0.886]])
    G = np.zeros((2, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    with (tmp_path / "diagnostics.csv").open() as f:
        header = next(csv.reader(f))

    assert header == _expected_columns(N_CONSTRAINTS)


# ---------------------------------------------------------------- 17A.2
def test_one_row_appended_per_notify_call(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, N_CONSTRAINTS))

    for gen in (1, 2, 3):
        cb.notify(_stub_algorithm(F, G, n_gen=gen))

    with (tmp_path / "diagnostics.csv").open() as f:
        rows = list(csv.DictReader(f))

    assert [int(r["gen"]) for r in rows] == [1, 2, 3]


# ---------------------------------------------------------------- 17A.3
def test_n_feasible_plus_n_infeasible_equals_pop_size(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)
    F = np.tile(np.array([-0.5, -0.886]), (4, 1))
    G = np.zeros((4, N_CONSTRAINTS))
    G[2, 0] = 0.5
    G[3, 5] = 1.0
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert int(row["n_feasible"]) == 2
    assert int(row["n_infeasible"]) == 2
    assert int(row["n_feasible"]) + int(row["n_infeasible"]) == F.shape[0]


# ---------------------------------------------------------------- 17A.4
def test_edge_cases_all_feasible_and_all_infeasible(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)

    F_feas = np.array([[-0.7, -0.886], [-0.5, -1.26]])
    G_feas = np.zeros((2, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F_feas, G_feas, n_gen=1))

    F_infeas = np.array([[-0.3, -0.886]])
    G_infeas = np.ones((1, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F_infeas, G_infeas, n_gen=2))

    with (tmp_path / "diagnostics.csv").open() as f:
        rows = list(csv.DictReader(f))

    feas_row, infeas_row = rows
    assert int(feas_row["n_feasible"]) == 2 and int(feas_row["n_infeasible"]) == 0
    assert float(feas_row["mean_util_feasible"]) == pytest.approx(0.6)
    assert float(feas_row["max_util_feasible"]) == pytest.approx(0.7)

    assert int(infeas_row["n_feasible"]) == 0 and int(infeas_row["n_infeasible"]) == 1
    assert infeas_row["mean_util_feasible"] == ""
    assert infeas_row["max_util_feasible"] == ""
    assert infeas_row["mean_min_speed_feasible"] == ""
    assert infeas_row["max_min_speed_feasible"] == ""


# ---------------------------------------------------------------- 17A.5
def test_per_constraint_mean_and_p90_match_numpy(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)
    rng = np.random.default_rng(seed=42)
    F = -rng.uniform(0.0, 1.0, size=(20, 2))
    G = rng.uniform(-0.5, 1.5, size=(20, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    expected_mean = G.mean(axis=0)
    expected_p90 = np.percentile(G, 90, axis=0)

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    for i in range(N_CONSTRAINTS):
        assert float(row[f"constraint_{i}_mean"]) == pytest.approx(expected_mean[i])
        assert float(row[f"constraint_{i}_p90"]) == pytest.approx(expected_p90[i])


# ---------------------------------------------------------------- 17A.6 (nested dir)
def test_creates_nested_output_dir(tmp_path):
    nested = tmp_path / "outputs" / "deep" / "nested"
    cb = DiagnosticsCallback(output_dir=nested, n_constraints=N_CONSTRAINTS)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    assert (nested / "diagnostics.csv").exists()


# ---------------------------------------------------------------- 17A.7 — switch chromosome count
def test_n_chromosomes_with_switches_counted_via_isin(tmp_path):
    n_max = 10
    switch_indices = np.array([3, 4], dtype=np.int16)
    crossing_indices = np.array([5], dtype=np.int16)
    cb = DiagnosticsCallback(
        output_dir=tmp_path, n_constraints=N_CONSTRAINTS,
        n_max=n_max,
        switch_piece_indices=switch_indices,
        crossing_piece_indices=crossing_indices,
    )

    # 4 individuals; chromosome's piece-slot region is X[:, :n_max].
    # ind 0 has piece 3 (switch), ind 1 has piece 4 (switch),
    # ind 2 has piece 5 (crossing), ind 3 has only piece 0 (neither).
    X = np.full((4, n_max + 5), -1, dtype=np.int16)
    X[0, 0] = 3
    X[1, 1] = 4
    X[2, 2] = 5
    X[3, 0] = 0

    F = np.tile(np.array([-0.5, -0.886]), (4, 1))
    G = np.zeros((4, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, X=X, n_gen=1))

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert int(row["n_chromosomes_with_switches"]) == 2
    assert int(row["n_chromosomes_with_crossings"]) == 1


# ---------------------------------------------------------------- 17A.8 — switch/crossing blank when X absent
def test_switch_crossing_blank_when_no_x_data(tmp_path):
    cb = DiagnosticsCallback(
        output_dir=tmp_path, n_constraints=N_CONSTRAINTS,
        n_max=10,
        switch_piece_indices=np.array([3, 4], dtype=np.int16),
        crossing_piece_indices=np.array([5], dtype=np.int16),
    )
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))  # no X

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert row["n_chromosomes_with_switches"] == ""
    assert row["n_chromosomes_with_crossings"] == ""


# ---------------------------------------------------------------- 17A.9 — dedupe rate from fresh stats
def test_dedupe_rate_from_fresh_dedupe_callback(tmp_path):
    dedupe_stub = SimpleNamespace(
        last_gen=1, last_pop_size=10,
        last_n_phenotypes=7, last_n_duplicates=3,
    )
    cb = DiagnosticsCallback(
        output_dir=tmp_path, n_constraints=N_CONSTRAINTS,
        dedupe_callback=dedupe_stub,
    )
    F = np.tile(np.array([-0.5, -0.886]), (10, 1))
    G = np.zeros((10, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert float(row["dedupe_rejection_rate"]) == pytest.approx(0.3)


# ---------------------------------------------------------------- 17A.10 — dedupe rate blank when stale
def test_dedupe_rate_blank_when_dedupe_stats_are_stale(tmp_path):
    dedupe_stub = SimpleNamespace(
        last_gen=1, last_pop_size=10,
        last_n_phenotypes=7, last_n_duplicates=3,
    )
    cb = DiagnosticsCallback(
        output_dir=tmp_path, n_constraints=N_CONSTRAINTS,
        dedupe_callback=dedupe_stub,
    )
    F = np.tile(np.array([-0.5, -0.886]), (10, 1))
    G = np.zeros((10, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=5))  # current=5 vs dedupe.last_gen=1 → stale

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert row["dedupe_rejection_rate"] == ""


# ---------------------------------------------------------------- 17A.11 — repair CV columns blank pre-Phase-1
def test_pre_post_repair_cv_blank_until_phase1(tmp_path):
    cb = DiagnosticsCallback(output_dir=tmp_path, n_constraints=N_CONSTRAINTS)
    F = np.array([[-0.5, -0.886]])
    G = np.zeros((1, N_CONSTRAINTS))
    cb.notify(_stub_algorithm(F, G, n_gen=1))

    with (tmp_path / "diagnostics.csv").open() as f:
        row = next(csv.DictReader(f))

    assert row["mean_pre_repair_cv"] == ""
    assert row["mean_post_repair_cv"] == ""
