"""Tests for Phase 2/5 constraint reformulation in PortPairProblem.

Covers:
- ``n_constr == 11 + T`` (Phase 5: was ``9 + T`` after the closed-track
  + loose-port additions)
- ``incomplete_switch_ratio == 0`` for layout with no switches
- ``incomplete_switch_ratio == 1`` when every switch has unpaired ports
- ``branch_cycle_deficit == 0`` when ``min_branch_count`` is unset (default 0)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.encoding import (
    compute_port_pair_dimensions,
    create_empty_chromosome,
    set_piece_slot,
    set_port_pair,
)
from src_v2.problem import PortPairProblem


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"
CONFIG_DEFAULT = Path(__file__).parent.parent / "configs" / "default.yaml"
CONFIG_SWITCHES = Path(__file__).parent.parent / "configs" / "with_switches.yaml"


def _build_problem(config_path: Path) -> PortPairProblem:
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(config_path)
    return PortPairProblem(catalog, config)


def test_n_constr_is_11_plus_T():
    problem = _build_problem(CONFIG_DEFAULT)
    expected = 11 + problem.catalog.n_pieces
    assert problem.n_ieq_constr == expected


def test_n_constr_with_switches_config():
    problem = _build_problem(CONFIG_SWITCHES)
    expected = 11 + problem.catalog.n_pieces
    assert problem.n_ieq_constr == expected


def test_incomplete_switch_ratio_zero_for_layout_without_switches():
    """Pure-oval layout has no switches — completeness ratio is 0 trivially."""
    problem = _build_problem(CONFIG_DEFAULT)
    x = create_empty_chromosome(problem.dims)
    curve_idx = problem.catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, problem.dims, k, curve_idx)
    for k in range(16):
        set_port_pair(x, problem.dims, k, k, 1, (k + 1) % 16, 0)

    out = {}
    problem._evaluate(x, out)
    G = out["G"]
    # G[5+T] is incomplete_switch_ratio (T = catalog.n_pieces)
    T = problem.catalog.n_pieces
    assert G[5 + T] == 0.0


def test_incomplete_switch_ratio_one_when_all_switches_dangling():
    """Switch with no edges = paired set is empty != {A, B, C}, so incomplete."""
    problem = _build_problem(CONFIG_SWITCHES)
    x = create_empty_chromosome(problem.dims)
    sw_idx = problem.catalog.id_to_index["R40_SWITCH_LEFT"]
    # Place a single switch with NO edges; need at least one other slot
    # and one cycle to pass the n_slots / cycle constraints.
    curve_idx = problem.catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, problem.dims, k, curve_idx)
    for k in range(16):
        set_port_pair(x, problem.dims, k, k, 1, (k + 1) % 16, 0)
    # Add a dangling switch at slot 16 — no edges to it.
    set_piece_slot(x, problem.dims, 16, sw_idx)

    out = {}
    problem._evaluate(x, out)
    G = out["G"]
    T = problem.catalog.n_pieces
    # 1 switch, 1 incomplete → ratio 1.0
    assert G[5 + T] == 1.0


def test_branch_cycle_deficit_zero_when_min_branch_unset():
    """``min_branch_count`` is not in default config → constraint is 0.0 always."""
    problem = _build_problem(CONFIG_DEFAULT)
    x = create_empty_chromosome(problem.dims)
    curve_idx = problem.catalog.id_to_index["R40_CURVE"]
    for k in range(16):
        set_piece_slot(x, problem.dims, k, curve_idx)
    for k in range(16):
        set_port_pair(x, problem.dims, k, k, 1, (k + 1) % 16, 0)

    out = {}
    problem._evaluate(x, out)
    G = out["G"]
    T = problem.catalog.n_pieces
    # G[8+T] is branch_cycle_deficit
    assert G[8 + T] == 0.0


def test_cycle_count_constraint_is_at_index_7_plus_T():
    """G[7+T] is cycle_count = 1 - n_cycles. Open chain of 4+ pieces has
    a useful component but no cycle, so constraint = 1.0."""
    problem = _build_problem(CONFIG_DEFAULT)
    x = create_empty_chromosome(problem.dims)
    s_idx = problem.catalog.id_to_index["STRAIGHT_16"]
    for k in range(4):
        set_piece_slot(x, problem.dims, k, s_idx)
    # Open chain: 0.B → 1.A, 1.B → 2.A, 2.B → 3.A — no closing edge
    for k in range(3):
        set_port_pair(x, problem.dims, k, k, 1, k + 1, 0)

    out = {}
    problem._evaluate(x, out)
    G = out["G"]
    T = problem.catalog.n_pieces
    # No cycle → 1 - 0 = 1.0
    assert G[7 + T] == 1.0
