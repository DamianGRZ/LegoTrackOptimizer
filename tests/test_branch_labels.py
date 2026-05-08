"""Tests for ``src_v2.decoder._compute_branch_labels`` — cycle membership.

Tests construct port-edge sets directly and call the private helper, using a
real catalog spec loaded from ``data/track_pieces_v2.yaml`` so route names
match the production data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set, Tuple

import pytest

from src_v2.catalog import TrackCatalog
from src_v2.decoder import _compute_branch_labels
from src_v2.types import PortEdge


CATALOG_PATH = Path(__file__).parent.parent / "data" / "track_pieces_v2.yaml"


@pytest.fixture(scope="module")
def spec():
    """Real V2 catalog spec from data/track_pieces_v2.yaml."""
    return TrackCatalog.load(CATALOG_PATH).spec


def _edges(*tuples: Tuple[int, str, int, str]) -> list:
    return [PortEdge(*t) for t in tuples]


def _components(*sets: Iterable[int]) -> list:
    return [set(s) for s in sets]


# =============================================================================
# Trivial cases
# =============================================================================


def test_empty_graph(spec):
    assert _compute_branch_labels([], [], {}, spec) == {}


def test_open_chain_has_no_labels(spec):
    """Two straights in a row but no closure → no cycle → no labels."""
    edges = _edges((0, "B", 1, "A"))
    pieces = {0: "STRAIGHT_16", 1: "STRAIGHT_16"}
    components = _components({0, 1})
    assert _compute_branch_labels(components, edges, pieces, spec) == {}


# =============================================================================
# Single cycle
# =============================================================================


def test_oval_8_curves_all_main_zero(spec):
    """8 R40 curves in a closed loop — every slot labeled (slot, 'main') = 0."""
    edges = _edges(*((i, "B", (i + 1) % 8, "A") for i in range(8)))
    pieces = {i: "R40_CURVE" for i in range(8)}
    components = _components(set(range(8)))

    labels = _compute_branch_labels(components, edges, pieces, spec)

    assert labels == {(i, "main"): 0 for i in range(8)}


def test_mixed_oval(spec):
    """Mixed straights + curves in one closed loop — single cycle."""
    pieces = {0: "R40_CURVE", 1: "STRAIGHT_16", 2: "R40_CURVE", 3: "STRAIGHT_16"}
    edges = _edges(*((i, "B", (i + 1) % 4, "A") for i in range(4)))
    components = _components(set(range(4)))

    labels = _compute_branch_labels(components, edges, pieces, spec)

    assert labels == {(i, "main"): 0 for i in range(4)}


# =============================================================================
# Multiple cycles
# =============================================================================


def test_two_disjoint_cycles_get_distinct_ids(spec):
    """Two separate ovals share no slots → cycle_ids 0 and 1."""
    edges = _edges(
        *((i, "B", (i + 1) % 4, "A") for i in range(4)),
        *((i + 10, "B", ((i + 1) % 4) + 10, "A") for i in range(4)),
    )
    pieces = {i: "R40_CURVE" for i in [0, 1, 2, 3, 10, 11, 12, 13]}
    components = _components({0, 1, 2, 3}, {10, 11, 12, 13})

    labels = _compute_branch_labels(components, edges, pieces, spec)

    cycle_ids = {labels[k] for k in labels}
    assert cycle_ids == {0, 1}
    # Each component has its own id
    cycle_a = labels[0, "main"]
    cycle_b = labels[10, "main"]
    assert cycle_a != cycle_b
    assert all(labels[i, "main"] == cycle_a for i in range(4))
    assert all(labels[i + 10, "main"] == cycle_b for i in range(4))


# =============================================================================
# Switches — passing siding gets two labels per switch slot
# =============================================================================


def test_passing_siding_two_cycles_per_switch(spec):
    """Mainline of 4 straights + branch through 2 curves between two switches.

    Layout (slot indices in brackets):
        ── [0:STRAIGHT_16] ── [1:SWITCH_LEFT] ── [2:STRAIGHT_16] ──
                                    ↘                   ↗
                                  [4:R40_CURVE flipped] ─ [5:R40_CURVE]
                                    ↘                   ↗
        ── [3:STRAIGHT_16] ── [6:SWITCH_RIGHT] ── ...

    Topologically:
      - Main cycle: 0 - 1.A-B - 2 - 3 - 6.B-A - back to 0 (impossible without
        more pieces, but for unit test we close it manually).

    For test simplicity, we build an *abstract* graph that captures the
    topology even if it isn't physically realizable: two switches with
    A-B used in a 4-piece cycle and A-C used in a 2-curve branch back.
    """
    pieces = {
        0: "STRAIGHT_16",
        1: "R40_SWITCH_LEFT",
        2: "STRAIGHT_16",
        3: "STRAIGHT_16",
        4: "R40_CURVE",
        5: "R40_CURVE",
        6: "R40_SWITCH_RIGHT",
    }
    # Main cycle: 0-1.A, 1.B-2, 2-3, 3-6.B, 6.A-0
    # Branch:     1.C-4, 4-5, 5-6.C
    edges = _edges(
        (0, "B", 1, "A"),
        (1, "B", 2, "A"),
        (2, "B", 3, "A"),
        (3, "B", 6, "B"),
        (6, "A", 0, "A"),
        # Branch edges
        (1, "C", 4, "A"),
        (4, "B", 5, "A"),
        (5, "B", 6, "C"),
    )
    components = _components(set(pieces))

    labels = _compute_branch_labels(components, edges, pieces, spec)

    # Each switch should have BOTH 'through' and 'diverging' labels.
    assert (1, "through") in labels
    assert (1, "diverging") in labels
    assert (6, "through") in labels
    assert (6, "diverging") in labels

    # The two routes should be on DIFFERENT cycles for each switch
    assert labels[1, "through"] != labels[1, "diverging"]
    assert labels[6, "through"] != labels[6, "diverging"]

    # Two distinct cycle_ids
    assert len({labels[k] for k in labels}) == 2


def test_passing_siding_branch_pieces_share_cycle(spec):
    """Branch curves should share the diverging cycle_id with switch C-routes."""
    pieces = {
        0: "STRAIGHT_16", 1: "R40_SWITCH_LEFT", 2: "STRAIGHT_16",
        3: "STRAIGHT_16", 4: "R40_CURVE", 5: "R40_CURVE",
        6: "R40_SWITCH_RIGHT",
    }
    edges = _edges(
        (0, "B", 1, "A"), (1, "B", 2, "A"), (2, "B", 3, "A"),
        (3, "B", 6, "B"), (6, "A", 0, "A"),
        (1, "C", 4, "A"), (4, "B", 5, "A"), (5, "B", 6, "C"),
    )
    components = _components(set(pieces))

    labels = _compute_branch_labels(components, edges, pieces, spec)

    branch_id = labels[1, "diverging"]
    assert labels[4, "main"] == branch_id
    assert labels[5, "main"] == branch_id


# =============================================================================
# Crossings — two routes on the same slot
# =============================================================================


def test_crossing_has_two_routes(spec):
    """CROSS_90 with both horizontal and vertical paths closed → 2 labels."""
    pieces = {
        0: "R40_CURVE", 1: "CROSS_90", 2: "R40_CURVE",
        3: "R40_CURVE", 4: "R40_CURVE",
    }
    # Horizontal cycle: 0 - 1.A-B - 2 - back to 0 (close manually)
    # Vertical cycle:   3 - 1.C-D - 4 - back to 3 (close manually)
    edges = _edges(
        (0, "B", 1, "A"), (1, "B", 2, "A"), (2, "B", 0, "A"),  # horizontal cycle
        (3, "B", 1, "C"), (1, "D", 4, "A"), (4, "B", 3, "A"),  # vertical cycle
    )
    components = _components(set(pieces))

    labels = _compute_branch_labels(components, edges, pieces, spec)

    assert (1, "horizontal") in labels
    assert (1, "vertical") in labels
    assert labels[1, "horizontal"] != labels[1, "vertical"]