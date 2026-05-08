"""Phase 3 vertical-slice tests for src_v2/decoder.py.

These hand-crafted tests are the GATE for the port-pair encoding approach.
If any of them fails, the design needs to be re-evaluated before any further
downstream work (operators, problem, runner) is built.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.decoder import DecoderConfig, decode_chromosome
from src_v2.encoding import (
    PortPairDimensions,
    create_empty_chromosome,
    set_anchor,
    set_piece_slot,
    set_port_pair,
)


# Piece indices from _LEGACY_PIECE_INDEX in catalog.py
P_STRAIGHT_16 = 0
P_STRAIGHT_24 = 1
P_R40_LEFT = 2
P_R40_RIGHT = 3
P_CROSS_90 = 4
P_SWITCH_LEFT_IN = 5
P_SWITCH_LEFT_OUT = 6
P_SWITCH_RIGHT_IN = 7
P_SWITCH_RIGHT_OUT = 8
P_DOUBLE_CROSSOVER = 9


# Port name -> port index per V2 spec ordering
# A=0, B=1 for straights/curves; A=0, B=1, C=2, D=3 for switches/crossings
PA, PB, PC, PD = 0, 1, 2, 3


@pytest.fixture(scope="module")
def catalog():
    """Load V2 catalog so catalog.spec is populated for the decoder."""
    return TrackCatalog.load(Path("data/track_pieces_v2.yaml"))


@pytest.fixture
def small_dims():
    """Generous dimensions for hand-crafted test chromosomes (32 slots, 32 pairs)."""
    return PortPairDimensions(N_max=32, E_max=32)


@pytest.fixture
def decoder_config():
    return DecoderConfig(
        closure_position_tolerance=0.5,
        closure_angle_tolerance_deg=2.0,
    )


# =============================================================================
# Vertical Slice 1 — 16-piece R40 LEFT closed circle
# =============================================================================


class TestSixteenPieceCircle:
    def test_decodes_to_one_component(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        for k in range(16):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        # 16 chain edges + 1 closing edge = 16 edges total
        for k in range(16):
            set_port_pair(x, small_dims, k,
                          slot_a=k, port_a=PB,
                          slot_b=(k + 1) % 16, port_b=PA)
        set_anchor(x, small_dims, 0, 0, 0)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.n_slots == 16
        assert graph.n_edges == 16
        assert graph.n_components == 1
        assert graph.n_loose_ports == 0

    def test_closes_within_tolerance(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        for k in range(16):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 16, PA)
        set_anchor(x, small_dims, 0, 0, 0)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        # 16 R40 LEFT pieces sum to 16 * pi/8 = 2 pi (full circle).
        # Per-piece FK (15.307, 3.045, pi/8) is the V2 catalog value, which
        # is consistent with itself, so the cycle should close exactly.
        assert graph.max_closure_position < 0.5, (
            f"Closure position residual too large: {graph.max_closure_position}"
        )
        assert graph.max_closure_angle_deg < 2.0, (
            f"Closure angle residual too large: {graph.max_closure_angle_deg}"
        )

    def test_one_cycle_in_topology(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        for k in range(16):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 16, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        # V=16, E=16, cycles = E - V + 1 = 1
        assert graph.n_cycles == 1


# =============================================================================
# Vertical Slice 2 — Open chain (4 R40, no closing edge)
# =============================================================================


class TestOpenChain:
    def test_open_chain_has_no_cycles_and_loose_ports(
        self, catalog, small_dims, decoder_config,
    ):
        x = create_empty_chromosome(small_dims)
        for k in range(4):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        # 3 edges; chain is open (no edge from slot 3 back to slot 0)
        for k in range(3):
            set_port_pair(x, small_dims, k, k, PB, k + 1, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.n_components == 1
        assert graph.n_cycles == 0
        # Slot 0 port A and slot 3 port B are unconnected
        assert graph.n_loose_ports == 2
        loose_set = set(graph.loose_ports)
        assert (0, "A") in loose_set
        assert (3, "B") in loose_set


# =============================================================================
# Vertical Slice 3 — Two disconnected loops
# =============================================================================


class TestTwoDisconnectedLoops:
    def test_two_4_piece_cycles(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        # Loop 1: slots 0-3 (impossible to close 4 R40 pieces alone, but use
        # 4 R40 LEFT for now — closure will not be exact, but topology is what
        # we test here)
        for k in range(4):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(4):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 4, PA)
        # Loop 2: slots 16-19, 4 more R40 LEFT
        for k in range(16, 20):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(4):
            set_port_pair(x, small_dims, 16 + k,
                          16 + k, PB, 16 + (k + 1) % 4, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.n_slots == 8
        assert graph.n_edges == 8
        assert graph.n_components == 2
        # Each component has E=V=4, so 1 cycle each, total 2
        assert graph.n_cycles == 2

    def test_tiled_components_do_not_overlap(
        self, catalog, small_dims, decoder_config,
    ):
        """Two disconnected loops should be tiled side-by-side, not stacked."""
        x = create_empty_chromosome(small_dims)
        for k in range(16):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 16, PA)
        for k in range(16, 32):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, small_dims, 16 + k,
                          16 + k, PB, 16 + (k + 1) % 16, PA)
        set_anchor(x, small_dims, 0, 0, 0)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        # Pull bounding boxes per component
        comp1_xs = [graph.slot_poses[s][0] for s in range(16) if s in graph.slot_poses]
        comp2_xs = [graph.slot_poses[s][0] for s in range(16, 32) if s in graph.slot_poses]
        assert comp1_xs and comp2_xs
        # Component 2 must start strictly after component 1 ends
        assert min(comp2_xs) > max(comp1_xs), (
            f"Components overlap: comp1 max_x={max(comp1_xs):.2f}, "
            f"comp2 min_x={min(comp2_xs):.2f}"
        )


# =============================================================================
# Vertical Slice 4 — Invalid edges dropped
# =============================================================================


class TestInvalidEdgesDropped:
    def test_self_loop_dropped(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        # Valid edge
        set_port_pair(x, small_dims, 0, 0, PB, 1, PA)
        # Self-loop on slot 0 — should be dropped
        set_port_pair(x, small_dims, 1, 0, PA, 0, PB)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.dropped_edge_count >= 1
        # Surviving edges must not include any self-loop
        for edge in graph.edges:
            assert edge.slot_a != edge.slot_b

    def test_double_booked_port_dropped(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        for k in range(3):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        # First edge claims slot 0 port B
        set_port_pair(x, small_dims, 0, 0, PB, 1, PA)
        # Second edge ALSO claims slot 0 port B — should be dropped
        set_port_pair(x, small_dims, 1, 0, PB, 2, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.dropped_edge_count >= 1
        # Verify each port appears in at most one surviving edge
        seen = set()
        for edge in graph.edges:
            for key in [(edge.slot_a, edge.port_a), (edge.slot_b, edge.port_b)]:
                assert key not in seen, f"Port {key} double-booked in survivors"
                seen.add(key)

    def test_out_of_range_port_dropped(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_R40_LEFT)  # only ports 0, 1
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        # Edge references port 2 of slot 0 — but R40 LEFT has only A and B
        set_port_pair(x, small_dims, 0, 0, 2, 1, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        # The edge is dropped at port-name resolution, before _sanitize_edges,
        # so dropped_edge_count may not increment but the edge must not survive.
        assert graph.n_edges == 0


# =============================================================================
# Vertical Slice 5 — Anchor placement
# =============================================================================


class TestAnchor:
    def test_anchor_pose_propagates_to_first_slot(
        self, catalog, small_dims, decoder_config,
    ):
        x = create_empty_chromosome(small_dims)
        for k in range(16):
            set_piece_slot(x, small_dims, k, P_R40_LEFT)
        for k in range(16):
            set_port_pair(x, small_dims, k, k, PB, (k + 1) % 16, PA)
        set_anchor(x, small_dims, 50, -30, 90)  # 90 degrees

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        # Slot 0 (anchor slot in its component) should be at the anchor pose
        assert 0 in graph.slot_poses
        x0, y0, theta0 = graph.slot_poses[0]
        assert x0 == pytest.approx(50.0)
        assert y0 == pytest.approx(-30.0)
        assert theta0 == pytest.approx(math.radians(90), abs=1e-9)


# =============================================================================
# Vertical Slice 6 — Empty chromosome
# =============================================================================


class TestEmptyChromosome:
    def test_empty_decodes_to_empty_graph(
        self, catalog, small_dims, decoder_config,
    ):
        x = create_empty_chromosome(small_dims)
        graph = decode_chromosome(x, small_dims, catalog, decoder_config)
        assert graph.n_slots == 0
        assert graph.n_edges == 0
        assert graph.n_components == 0
        assert graph.n_cycles == 0


# =============================================================================
# Vertical Slice 7 — Switch + simple branch (basic multi-path)
# =============================================================================


class TestSwitchBranch:
    """A LEFT switch's diverging port connected to a small branch.

    This is the simplest non-trivial multi-path topology: a switch with two
    routes through it. The switch's port C (diverging) is connected to a
    branch piece's port A. The decoder should:
      - place all slots without dropping the edge
      - report 2 loose ports (the switch's port B and the branch's port B,
        which are the through-route exit and the branch end)
    """

    def test_switch_with_branch_decodes(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        # Slot 0: switch LEFT IN; Slot 1: a curve attached to port C
        set_piece_slot(x, small_dims, 0, P_SWITCH_LEFT_IN)
        set_piece_slot(x, small_dims, 1, P_R40_LEFT)
        # Connect switch port C (index 2) to curve port A
        set_port_pair(x, small_dims, 0, 0, PC, 1, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.n_slots == 2
        assert graph.n_edges == 1
        assert graph.n_components == 1
        # Loose ports: slot 0 port A (switch entry), slot 0 port B (through
        # exit), slot 1 port B (branch end) = 3
        assert graph.n_loose_ports == 3


# =============================================================================
# Vertical Slice 8 — CROSS_90 with two routes
# =============================================================================


class TestCross90BothRoutes:
    """A CROSS_90 with both horizontal (A-B) and vertical (C-D) routes
    connected to terminator-like pieces — all 4 ports paired."""

    def test_cross_90_with_4_neighbors(self, catalog, small_dims, decoder_config):
        x = create_empty_chromosome(small_dims)
        set_piece_slot(x, small_dims, 0, P_CROSS_90)
        # 4 surrounding pieces (use straights for simplicity)
        for k in range(1, 5):
            set_piece_slot(x, small_dims, k, P_STRAIGHT_16)
        # Pair each of CROSS_90's 4 ports to a neighbour's port A
        set_port_pair(x, small_dims, 0, 0, PA, 1, PA)
        set_port_pair(x, small_dims, 1, 0, PB, 2, PA)
        set_port_pair(x, small_dims, 2, 0, PC, 3, PA)
        set_port_pair(x, small_dims, 3, 0, PD, 4, PA)

        graph = decode_chromosome(x, small_dims, catalog, decoder_config)

        assert graph.n_slots == 5
        assert graph.n_edges == 4
        assert graph.n_components == 1
        # Loose: 4 neighbour port-Bs (port A is consumed by the edge)
        assert graph.n_loose_ports == 4
