"""Tests for src_v2/types.py PortGraph extension."""

import math

import pytest

from src_v2.types import CycleResidual, PortEdge, PortGraph


# =============================================================================
# PortEdge
# =============================================================================

class TestPortEdge:
    def test_construction(self):
        e = PortEdge(slot_a=0, port_a="B", slot_b=1, port_b="A")
        assert e.slot_a == 0
        assert e.port_b == "A"

    def test_involves(self):
        e = PortEdge(0, "B", 1, "A")
        assert e.involves(0)
        assert e.involves(1)
        assert not e.involves(2)

    def test_other_endpoint(self):
        e = PortEdge(0, "B", 1, "A")
        assert e.other_endpoint(0) == (1, "A")
        assert e.other_endpoint(1) == (0, "B")

    def test_other_endpoint_raises_on_unrelated(self):
        e = PortEdge(0, "B", 1, "A")
        with pytest.raises(ValueError):
            e.other_endpoint(99)

    def test_frozen(self):
        e = PortEdge(0, "B", 1, "A")
        with pytest.raises(Exception):
            e.slot_a = 5  # frozen dataclass


# =============================================================================
# CycleResidual
# =============================================================================

class TestCycleResidual:
    def test_construction(self):
        r = CycleResidual(slot_a=0, slot_b=15, dx=0.1, dy=-0.05, dtheta=0.001)
        assert r.dx == 0.1
        assert r.dtheta == 0.001


# =============================================================================
# PortGraph
# =============================================================================

class TestPortGraph:
    def test_default_construction(self):
        g = PortGraph()
        assert g.n_slots == 0
        assert g.n_edges == 0
        assert g.n_components == 0
        assert g.n_loose_ports == 0
        assert g.max_closure_position == 0.0
        assert g.max_closure_angle_rad == 0.0

    def test_n_cycles_simple_loop(self):
        # 4-slot cycle: V=4, E=4, cycles = 4 - 4 + 1 = 1
        g = PortGraph(
            slot_pieces={0: "R40", 1: "R40", 2: "R40", 3: "R40"},
            edges=[
                PortEdge(0, "B", 1, "A"),
                PortEdge(1, "B", 2, "A"),
                PortEdge(2, "B", 3, "A"),
                PortEdge(3, "B", 0, "A"),
            ],
            connected_components=[{0, 1, 2, 3}],
        )
        assert g.n_cycles == 1

    def test_n_cycles_open_chain(self):
        # 4-slot chain: V=4, E=3, cycles = 3 - 4 + 1 = 0
        g = PortGraph(
            slot_pieces={0: "STR", 1: "STR", 2: "STR", 3: "STR"},
            edges=[
                PortEdge(0, "B", 1, "A"),
                PortEdge(1, "B", 2, "A"),
                PortEdge(2, "B", 3, "A"),
            ],
            connected_components=[{0, 1, 2, 3}],
        )
        assert g.n_cycles == 0

    def test_n_cycles_two_separate_loops(self):
        # Two 4-cycles, total cycles = 2
        g = PortGraph(
            slot_pieces={i: "R40" for i in range(8)},
            edges=[
                PortEdge(0, "B", 1, "A"), PortEdge(1, "B", 2, "A"),
                PortEdge(2, "B", 3, "A"), PortEdge(3, "B", 0, "A"),
                PortEdge(4, "B", 5, "A"), PortEdge(5, "B", 6, "A"),
                PortEdge(6, "B", 7, "A"), PortEdge(7, "B", 4, "A"),
            ],
            connected_components=[{0, 1, 2, 3}, {4, 5, 6, 7}],
        )
        assert g.n_cycles == 2

    def test_n_cycles_figure_8(self):
        # Figure-8: 7 nodes (one shared CROSS_90), 8 edges → 2 cycles
        # Component is connected (one set), edges = 8, vertices = 7
        # cycles = 8 - 7 + 1 = 2
        g = PortGraph(
            slot_pieces={i: ("R40" if i != 3 else "CROSS_90") for i in range(7)},
            edges=[
                # Loop 1: 0-1-2-3-0 via port A↔B
                PortEdge(0, "B", 1, "A"),
                PortEdge(1, "B", 2, "A"),
                PortEdge(2, "B", 3, "A"),
                PortEdge(3, "B", 0, "A"),
                # Loop 2: 3-4-5-6-3 via vertical CROSS_90 ports
                PortEdge(3, "C", 4, "A"),
                PortEdge(4, "B", 5, "A"),
                PortEdge(5, "B", 6, "A"),
                PortEdge(6, "B", 3, "D"),
            ],
            connected_components=[{0, 1, 2, 3, 4, 5, 6}],
        )
        assert g.n_cycles == 2

    def test_max_closure_position(self):
        g = PortGraph(
            closure_residuals=[
                CycleResidual(0, 1, dx=3.0, dy=4.0, dtheta=0.0),  # mag = 5
                CycleResidual(2, 3, dx=0.1, dy=0.1, dtheta=0.0),  # mag ≈ 0.14
            ],
        )
        assert g.max_closure_position == pytest.approx(5.0)

    def test_max_closure_angle(self):
        g = PortGraph(
            closure_residuals=[
                CycleResidual(0, 1, dx=0.0, dy=0.0, dtheta=0.5),
                CycleResidual(2, 3, dx=0.0, dy=0.0, dtheta=-0.8),
            ],
        )
        assert g.max_closure_angle_rad == pytest.approx(0.8)
        assert g.max_closure_angle_deg == pytest.approx(math.degrees(0.8))

    def test_edges_at(self):
        g = PortGraph(
            slot_pieces={0: "R40", 1: "R40", 2: "R40"},
            edges=[
                PortEdge(0, "B", 1, "A"),
                PortEdge(1, "B", 2, "A"),
                PortEdge(2, "B", 0, "A"),
            ],
        )
        assert len(g.edges_at(0)) == 2  # touched by edges 0 and 2
        assert len(g.edges_at(1)) == 2  # touched by edges 0 and 1
        assert len(g.edges_at(2)) == 2  # touched by edges 1 and 2

    def test_loose_ports_tracked(self):
        g = PortGraph(
            slot_pieces={0: "R40_SWITCH_LEFT_IN"},
            edges=[],
            loose_ports=[(0, "A"), (0, "B"), (0, "C")],
        )
        assert g.n_loose_ports == 3

    def test_dropped_edge_count_default_zero(self):
        g = PortGraph()
        assert g.dropped_edge_count == 0
