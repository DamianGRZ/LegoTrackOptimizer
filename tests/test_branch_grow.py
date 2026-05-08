"""Tests for the angular-budget A* in ``src_v2.branch_grow``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src_v2.branch_grow import BranchStep, find_branch_path
from src_v2.catalog import TrackCatalog


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(CATALOG_PATH)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_trivial_closure_no_pieces(catalog, rng):
    """Target equals start → empty path, no pieces consumed."""
    pose = (0.0, 0.0, 0.0)
    result = find_branch_path(
        start_pose=pose, target_pose=pose,
        inventory={"STRAIGHT_16": 4, "R40_CURVE": 4},
        catalog=catalog, max_depth=8, tolerance=0.01, rng=rng,
    )
    assert result == []


def test_single_straight_closure(catalog, rng):
    """Target = start + (16, 0, 0) → one STRAIGHT_16."""
    result = find_branch_path(
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(16.0, 0.0, 0.0),
        inventory={"STRAIGHT_16": 4, "R40_CURVE": 4},
        catalog=catalog, max_depth=8, tolerance=0.5, rng=rng,
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].piece_id == "STRAIGHT_16"


def test_two_straights_closure(catalog, rng):
    """Target = start + (32, 0, 0) → two STRAIGHT_16."""
    result = find_branch_path(
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(32.0, 0.0, 0.0),
        inventory={"STRAIGHT_16": 4, "R40_CURVE": 4},
        catalog=catalog, max_depth=8, tolerance=0.5, rng=rng,
    )
    assert result is not None
    assert len(result) == 2
    assert all(s.piece_id == "STRAIGHT_16" for s in result)


def test_inventory_exhausted_returns_none(catalog, rng):
    """Empty inventory → no closure possible (unless trivial)."""
    result = find_branch_path(
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(16.0, 0.0, 0.0),
        inventory={"STRAIGHT_16": 0, "R40_CURVE": 0},
        catalog=catalog, max_depth=8, tolerance=0.5, rng=rng,
    )
    assert result is None


def test_max_depth_respected(catalog, rng):
    """Target requiring 5 pieces with max_depth=2 → None."""
    result = find_branch_path(
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(80.0, 0.0, 0.0),  # needs 5 STRAIGHT_16
        inventory={"STRAIGHT_16": 16, "R40_CURVE": 0},
        catalog=catalog, max_depth=2, tolerance=0.5, rng=rng,
    )
    assert result is None


def test_curve_closure_quarter_circle(catalog, rng):
    """4 R40 left curves form a 90° quarter-arc.

    Start (0, 0, 0) → after 4 left curves end at (40, 40, π/2).
    Verify A* finds this path with curves only.
    """
    # Compute exact target by FK (avoids relying on hand-derived numbers).
    from src_v2.se2 import pose_compose
    pose = (0.0, 0.0, 0.0)
    delta_left = (15.307, 3.045, math.pi / 8)
    for _ in range(4):
        pose = pose_compose(pose, delta_left)

    result = find_branch_path(
        start_pose=(0.0, 0.0, 0.0),
        target_pose=pose,
        inventory={"STRAIGHT_16": 0, "R40_CURVE": 4},
        catalog=catalog, max_depth=4, tolerance=0.5, rng=rng,
    )
    assert result is not None
    assert len(result) == 4
    assert all(s.piece_id == "R40_CURVE" and s.flip == 0 for s in result)


def test_branchstep_namedtuple_shape():
    """Sanity: BranchStep has the contract the mutation operators expect."""
    step = BranchStep(piece_id="STRAIGHT_16", flip=0, rotate=0)
    assert step.piece_id == "STRAIGHT_16"
    assert step.flip == 0
    assert step.rotate == 0
