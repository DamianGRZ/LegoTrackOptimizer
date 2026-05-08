"""Tests for ``tests/fixtures/multiprocessing_smoke.py`` (Phase 17.C.4).

Validates Rule 1's positive half: standard pymoo ``out`` keys (F, G)
round-trip correctly from worker → main under ``StarmapParallelization``.
A single ``Pool(2)`` run is shared across the 3 tests via a module-scoped
fixture so we pay the ~2 sec process-spawn overhead only once.
"""
from __future__ import annotations

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from tests.fixtures.mini_problem import CATALOG_PATH
from tests.fixtures.multiprocessing_smoke import multiprocessing_smoke_run


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multiprocessing_smoke")
    return multiprocessing_smoke_run(tmp, n_gen=5, pop_size=20)


@pytest.fixture(scope="module")
def expected_g_cols():
    return 11 + TrackCatalog.load(CATALOG_PATH).n_pieces


# ---------------------------------------------------------------- MP.1
def test_runs_with_2_workers_without_pickle_errors(smoke_result):
    assert smoke_result.pop is not None
    assert len(smoke_result.pop) > 0


# ---------------------------------------------------------------- MP.2
def test_F_round_trips_with_correct_shape(smoke_result):
    F = smoke_result.pop.get("F")
    assert F is not None
    assert F.shape[1] == 2  # 2 objectives: -util, -min_speed
    assert not np.isnan(F).any()


# ---------------------------------------------------------------- MP.3
def test_G_round_trips_with_correct_shape(smoke_result, expected_g_cols):
    G = smoke_result.pop.get("G")
    assert G is not None
    assert G.shape[1] == expected_g_cols
    assert not np.isnan(G).any()
