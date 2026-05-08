"""Tests for ``intersection.count_uncovered_intersections`` — collision G[4]."""

from __future__ import annotations

from pathlib import Path

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.intersection import count_uncovered_intersections
from src_v2.types import PortEdge, PortGraph


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"


@pytest.fixture(scope="module")
def catalog():
    return TrackCatalog.load(CATALOG_PATH)


def _make_graph(slot_pieces, slot_poses, edges=()):
    return PortGraph(
        slot_pieces=dict(slot_pieces),
        slot_indices={s: 0 for s in slot_pieces},
        slot_flips={s: 0 for s in slot_pieces},
        slot_rotates={s: 0 for s in slot_pieces},
        edges=[PortEdge(*e) for e in edges],
        slot_poses=dict(slot_poses),
        closure_residuals=[],
        loose_ports=[],
        connected_components=[set(slot_pieces.keys())],
        dropped_edge_count=0,
    )


def test_no_intersections_for_distant_pieces(catalog):
    """Two straights far apart — no intersection."""
    graph = _make_graph(
        slot_pieces={0: "STRAIGHT_16", 1: "STRAIGHT_16"},
        slot_poses={0: (0, 0, 0), 1: (0, 100, 0)},
    )
    assert count_uncovered_intersections(graph, catalog) == 0


def test_two_perpendicular_straights_intersect(catalog):
    """Two straights crossing at right angles → 1 uncovered intersection."""
    graph = _make_graph(
        slot_pieces={0: "STRAIGHT_16", 1: "STRAIGHT_16"},
        # Slot 0: horizontal from (0,0) to (16,0)
        # Slot 1: vertical from (8,-8) to (8,8) — crosses slot 0 at (8,0)
        slot_poses={0: (0, 0, 0), 1: (8, -8, 1.5707963)},  # theta=π/2
    )
    assert count_uncovered_intersections(graph, catalog) == 1


def test_endpoint_touch_is_not_intersection(catalog):
    """Two straights end-to-end (port B → port A) → no intersection."""
    graph = _make_graph(
        slot_pieces={0: "STRAIGHT_16", 1: "STRAIGHT_16"},
        # Slot 0: (0,0) to (16,0); Slot 1: starts at (16, 0) goes east
        slot_poses={0: (0, 0, 0), 1: (16, 0, 0)},
    )
    assert count_uncovered_intersections(graph, catalog) == 0


def test_cross_90_excluded_from_segments(catalog):
    """A CROSS_90 slot is not counted as an ordinary segment."""
    graph = _make_graph(
        slot_pieces={0: "CROSS_90", 1: "STRAIGHT_16"},
        slot_poses={0: (0, 0, 0), 1: (8, -8, 1.5707963)},
    )
    # CROSS_90 is excluded; only one segment (slot 1), no pair to check.
    assert count_uncovered_intersections(graph, catalog) == 0


def test_double_crossover_excluded(catalog):
    """A DOUBLE_CROSSOVER slot is not counted as an ordinary segment."""
    graph = _make_graph(
        slot_pieces={0: "DOUBLE_CROSSOVER", 1: "STRAIGHT_16"},
        slot_poses={0: (0, 0, 0), 1: (24, -10, 1.5707963)},
    )
    assert count_uncovered_intersections(graph, catalog) == 0


def test_empty_graph_zero_intersections(catalog):
    graph = _make_graph(slot_pieces={}, slot_poses={})
    assert count_uncovered_intersections(graph, catalog) == 0


def test_three_straights_all_crossing(catalog):
    """Three straights, two of three pairs cross."""
    graph = _make_graph(
        slot_pieces={0: "STRAIGHT_16", 1: "STRAIGHT_16", 2: "STRAIGHT_16"},
        # 0: horizontal at y=0; 1: vertical crossing 0 at x=8; 2: parallel
        # to 0 but offset
        slot_poses={
            0: (0, 0, 0),
            1: (8, -8, 1.5707963),
            2: (0, 5, 0),  # parallel, no crossing
        },
    )
    # 0 vs 1 → cross at (8,0). 0 vs 2 → parallel, no cross. 1 vs 2 → 1 vertical
    # crossing 2 horizontal at (8, 5). Total: 2.
    assert count_uncovered_intersections(graph, catalog) == 2
