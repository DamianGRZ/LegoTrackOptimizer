"""Tests for ``src_v2.canonical`` — canonical PortGraph hashing.

Tests use hand-built :class:`PortGraph` instances rather than going through
the decoder, so we can isolate the hash function from any decoder bugs.
"""

from __future__ import annotations

import hashlib

from src_v2.canonical import canonical_graph_hash
from src_v2.types import PortEdge, PortGraph


def _make_graph(
    pieces, edges_tuples, *, flips=None, rotates=None, components=None,
):
    """Build a PortGraph from compact specs.

    Args:
        pieces: dict {slot_idx: piece_id}.
        edges_tuples: list of (slot_a, port_a, slot_b, port_b).
        flips, rotates: dicts {slot_idx: bit}, default 0.
        components: list of frozen-sets, or None to derive a single component
            covering all slots.
    """
    flips = flips or {s: 0 for s in pieces}
    rotates = rotates or {s: 0 for s in pieces}
    edges = [PortEdge(*t) for t in edges_tuples]
    if components is None:
        components = [set(pieces.keys())]
    return PortGraph(
        slot_pieces=dict(pieces),
        slot_indices={s: 0 for s in pieces},
        slot_flips=dict(flips),
        slot_rotates=dict(rotates),
        edges=edges,
        slot_poses={},
        closure_residuals=[],
        loose_ports=[],
        connected_components=[set(c) for c in components],
        dropped_edge_count=0,
    )


def _oval_8curves():
    """Closed loop of 8 R40_CURVE pieces, slots 0..7."""
    pieces = {i: "R40_CURVE" for i in range(8)}
    edges = [(i, "B", (i + 1) % 8, "A") for i in range(8)]
    return _make_graph(pieces, edges)


# =============================================================================
# Determinism + invariance
# =============================================================================


def test_hash_is_deterministic():
    h1 = canonical_graph_hash(_oval_8curves())
    h2 = canonical_graph_hash(_oval_8curves())
    assert h1 == h2


def test_hash_invariant_under_slot_permutation():
    g_low = _oval_8curves()

    pieces_high = {100 + i: "R40_CURVE" for i in range(8)}
    edges_high = [(100 + i, "B", 100 + ((i + 1) % 8), "A") for i in range(8)]
    g_high = _make_graph(
        pieces_high, edges_high,
        components=[{100 + i for i in range(8)}],
    )

    assert canonical_graph_hash(g_low) == canonical_graph_hash(g_high)


def test_hash_invariant_under_edge_row_order():
    g1 = _oval_8curves()

    pieces = {i: "R40_CURVE" for i in range(8)}
    edges_reversed = [(i, "B", (i + 1) % 8, "A") for i in range(8)]
    edges_reversed.reverse()
    g2 = _make_graph(pieces, edges_reversed)

    assert canonical_graph_hash(g1) == canonical_graph_hash(g2)


def test_hash_invariant_under_anchor():
    """Anchor lives in the chromosome; PortGraph already strips it.

    PortGraph carries no anchor field — slot_poses are computed from anchor
    in the decoder but the hash uses only slot_pieces/edges/flips/rotates.
    Two graphs with identical topology but different absolute world poses
    therefore hash equal as long as the topology is the same.
    """
    g1 = _oval_8curves()
    g2 = _oval_8curves()
    g2.slot_poses = {i: (1000.0, 2000.0, 1.5) for i in range(8)}
    assert canonical_graph_hash(g1) == canonical_graph_hash(g2)


# =============================================================================
# Discrimination
# =============================================================================


def test_hash_distinguishes_different_topologies():
    g_oval = _oval_8curves()

    # Two disjoint cycles of 4 pieces each.
    pieces = {i: "R40_CURVE" for i in range(8)}
    edges = []
    for i in range(4):
        edges.append((i, "B", (i + 1) % 4, "A"))
    for i in range(4):
        edges.append((4 + i, "B", 4 + ((i + 1) % 4), "A"))
    g_two_cycles = _make_graph(
        pieces, edges,
        components=[{0, 1, 2, 3}, {4, 5, 6, 7}],
    )

    assert canonical_graph_hash(g_oval) != canonical_graph_hash(g_two_cycles)


def test_hash_distinguishes_different_piece_kinds():
    g_curves = _oval_8curves()

    pieces = {i: "STRAIGHT_16" for i in range(8)}
    edges = [(i, "B", (i + 1) % 8, "A") for i in range(8)]
    g_straights = _make_graph(pieces, edges)

    assert canonical_graph_hash(g_curves) != canonical_graph_hash(g_straights)


def test_hash_distinguishes_flip_bit():
    g_default = _oval_8curves()

    flips_one_set = {i: 0 for i in range(8)}
    flips_one_set[0] = 1
    g_flipped = _make_graph(
        {i: "R40_CURVE" for i in range(8)},
        [(i, "B", (i + 1) % 8, "A") for i in range(8)],
        flips=flips_one_set,
    )

    assert canonical_graph_hash(g_default) != canonical_graph_hash(g_flipped)


def test_hash_distinguishes_rotate_bit():
    g_default = _oval_8curves()

    rotates_one_set = {i: 0 for i in range(8)}
    rotates_one_set[0] = 1
    g_rotated = _make_graph(
        {i: "R40_CURVE" for i in range(8)},
        [(i, "B", (i + 1) % 8, "A") for i in range(8)],
        rotates=rotates_one_set,
    )

    assert canonical_graph_hash(g_default) != canonical_graph_hash(g_rotated)


# =============================================================================
# Junk filtering
# =============================================================================


def test_hash_drops_junk_components():
    """A 2-piece junk component is filtered before hashing."""
    g_clean = _oval_8curves()

    pieces_with_junk = {i: "R40_CURVE" for i in range(8)}
    pieces_with_junk[100] = "STRAIGHT_16"
    pieces_with_junk[101] = "STRAIGHT_16"
    edges_with_junk = [(i, "B", (i + 1) % 8, "A") for i in range(8)]
    edges_with_junk.append((100, "B", 101, "A"))
    g_with_junk = _make_graph(
        pieces_with_junk, edges_with_junk,
        components=[{i for i in range(8)}, {100, 101}],
    )

    assert canonical_graph_hash(g_clean) == canonical_graph_hash(g_with_junk)


def test_empty_graph_has_stable_hash():
    g_empty = PortGraph()
    h1 = canonical_graph_hash(g_empty)
    h2 = canonical_graph_hash(PortGraph())
    assert h1 == h2
    assert len(h1) == 8


def test_all_junk_collapses_to_empty():
    """A graph composed entirely of sub-threshold components hashes as empty."""
    pieces = {0: "STRAIGHT_16", 1: "STRAIGHT_16"}
    edges = [(0, "B", 1, "A")]
    g = _make_graph(pieces, edges, components=[{0, 1}])

    assert canonical_graph_hash(g) == canonical_graph_hash(PortGraph())


# =============================================================================
# Threshold control
# =============================================================================


def test_min_useful_component_size_is_respected():
    """Lowering the threshold should make a 2-piece component count."""
    pieces = {0: "STRAIGHT_16", 1: "STRAIGHT_16"}
    edges = [(0, "B", 1, "A")]
    g = _make_graph(pieces, edges, components=[{0, 1}])

    assert canonical_graph_hash(g) == canonical_graph_hash(PortGraph())
    assert canonical_graph_hash(g, min_useful_component_size=2) != canonical_graph_hash(
        PortGraph(),
    )


# =============================================================================
# Hash format
# =============================================================================


def test_hash_is_8_bytes():
    h = canonical_graph_hash(_oval_8curves())
    assert isinstance(h, bytes)
    assert len(h) == 8


def test_empty_uses_blake2b_sentinel():
    expected = hashlib.blake2b(b"empty-port-graph", digest_size=8).digest()
    assert canonical_graph_hash(PortGraph()) == expected
