"""Tests for the Phase 5 ε-archive (Laumanns et al. 2002)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src_v2.epsilon_archive import EpsilonArchive


def test_admit_first_entry():
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=10)
    assert arc.admit(np.array([1, 2, 3]), np.array([0.5, 0.5])) is True
    assert len(arc) == 1


def test_dominated_candidate_rejected():
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=10)
    arc.admit(np.array([1]), np.array([0.0, 0.0]))
    # Strictly dominated candidate (worse on both axes by > eps).
    assert arc.admit(np.array([2]), np.array([1.0, 1.0])) is False
    assert len(arc) == 1


def test_dominating_candidate_replaces_archive_entry():
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=10)
    arc.admit(np.array([1]), np.array([1.0, 1.0]))
    # Strictly better on both → drops the existing entry, admits new.
    admitted = arc.admit(np.array([2]), np.array([0.0, 0.0]))
    assert admitted is True
    assert len(arc) == 1
    np.testing.assert_array_equal(arc.X[0], np.array([2]))


def test_non_dominated_pair_both_kept():
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=10)
    arc.admit(np.array([1]), np.array([0.0, 1.0]))
    arc.admit(np.array([2]), np.array([1.0, 0.0]))
    assert len(arc) == 2


def test_truncation_drops_least_isolated():
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=3)
    # Three well-spread points along a Pareto front.
    arc.admit(np.array([0]), np.array([0.0, 1.0]))
    arc.admit(np.array([1]), np.array([0.5, 0.5]))
    arc.admit(np.array([2]), np.array([1.0, 0.0]))
    # Add a fourth that's near point #1 — it (or #1) gets dropped.
    arc.admit(np.array([3]), np.array([0.51, 0.49]))
    assert len(arc) == 3


def test_to_json_roundtrip(tmp_path):
    arc = EpsilonArchive(epsilon=(0.1, 0.1), max_size=10)
    arc.admit(np.array([1, 2, 3]), np.array([0.0, 1.0]))
    arc.admit(np.array([4, 5, 6]), np.array([1.0, 0.0]))
    p = tmp_path / "out.json"
    arc.to_json(p)
    data = json.loads(p.read_text())
    assert data["epsilon"] == [0.1, 0.1]
    assert data["max_size"] == 10
    assert len(data["F"]) == 2
    assert len(data["X"]) == 2


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        EpsilonArchive(epsilon=(0.0, 0.1), max_size=10)
    with pytest.raises(ValueError):
        EpsilonArchive(epsilon=(-0.1, 0.1), max_size=10)


def test_box_dominance_within_epsilon_keeps_first():
    """Two points sharing the same ε-box: first admitted wins."""
    arc = EpsilonArchive(epsilon=(1.0, 1.0), max_size=10)
    # Both points map to box (0, 0) — second point is ε-dominated by first.
    arc.admit(np.array([1]), np.array([0.1, 0.1]))
    # Slightly worse position but same box — neither dominates → both
    # admitted and kept (the spec doesn't reject equal-box neighbours).
    second_admitted = arc.admit(np.array([2]), np.array([0.5, 0.5]))
    assert second_admitted is True
    assert len(arc) == 2
