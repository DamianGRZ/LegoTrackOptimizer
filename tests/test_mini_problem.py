"""Tests for ``tests/fixtures/mini_problem.py`` (Phase 17.C.1)."""
from __future__ import annotations

import inspect
import time

import numpy as np

from tests.fixtures.mini_problem import (
    DEFAULT_MINI_INVENTORY,
    mini_optimization_run,
)


def test_default_inventory_is_40_pieces_switch_bearing():
    assert sum(DEFAULT_MINI_INVENTORY.values()) == 40
    assert DEFAULT_MINI_INVENTORY["R40_SWITCH_LEFT"] >= 1
    assert DEFAULT_MINI_INVENTORY["R40_SWITCH_RIGHT"] >= 1
    assert "R40_CURVE" in DEFAULT_MINI_INVENTORY  # V2 piece id, not V1's R40_LEFT


def test_signature_has_expected_kwargs():
    sig = inspect.signature(mini_optimization_run)
    assert set(sig.parameters) == {
        "output_dir", "seed", "n_gen", "pop_size",
        "inventory", "n_workers", "heuristic_ratio",
    }


def test_runs_to_completion_returns_pymoo_result(tmp_path):
    result = mini_optimization_run(tmp_path, n_gen=10, pop_size=30)
    # pymoo Result attributes used by downstream phase tests
    assert result.pop is not None
    assert result.X is not None
    assert result.F is not None
    assert result.G is not None
    # Sidecar artifacts written by the chained callbacks
    assert (tmp_path / "diagnostics.csv").exists()
    assert (tmp_path / "snapshots" / "snapshots_metadata.json").exists()


def test_two_back_to_back_runs_both_complete(tmp_path):
    """Two consecutive runs both complete without state pollution.

    Budget is generous (60s for two runs) — Windows process scheduler noise
    can push a single run from ~10s nominal to ~16s under load, so 30s for
    one run / 60s for two is the realistic ceiling. The test's primary
    value is "two runs back-to-back don't break each other"; runtime is a
    sanity floor against runaway loops, not a tight benchmark.

    NOTE on determinism: same-seed runs are NOT byte-equal yet. Sampler and
    mutation are now seeded via ``config.algorithm.seed``; pymoo's
    ``minimize(seed=…)`` seeds numpy/python globals at start. But several
    operators still mix ``self.rng`` (seeded) with ``np.random.*`` (global),
    and ``DefaultMultiObjectiveTermination`` may early-terminate at
    slightly different generations. Full reproducibility is a project-level
    follow-up, out of scope for this fixture.
    """
    t0 = time.time()
    res1 = mini_optimization_run(tmp_path / "a", seed=42, n_gen=20, pop_size=40)
    res2 = mini_optimization_run(tmp_path / "b", seed=42, n_gen=20, pop_size=40)
    elapsed = time.time() - t0

    assert res1.pop is not None and len(res1.pop) > 0
    assert res2.pop is not None and len(res2.pop) > 0
    assert elapsed < 60.0, f"two runs took {elapsed:.1f}s; runaway-loop ceiling is 60s"
