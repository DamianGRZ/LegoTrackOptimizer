"""Shared domain types for the LEGO Track Optimizer.

Pure data containers with no logic beyond property accessors.
These form the shared vocabulary across module tiers:
  - Tier 1 (domain core): catalog, geometry, train
  - Tier 2 (EA layer): decoder, operators, problem
  - Tier 3 (infra): visualization, config, main
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# PIECE CLASSIFICATION AND ROUTING
# =============================================================================


class PieceClass(Enum):
    """Classify pieces by port topology."""

    SIMPLE_2PORT = "simple_2port"
    SWITCH_3PORT = "switch_3port"
<<<<<<< Updated upstream
    CROSSING_4PORT = "crossing_4port"
=======
    SWITCH_4PORT = "switch_4port"
    CROSSING_4PORT = "crossing_4port"
    BUMPER_1PORT = "bumper_1port"
>>>>>>> Stashed changes


@dataclass(frozen=True)
class FKRoute:
    """Single route through a piece (entry port -> exit port)."""

    entry_port: int
    exit_port: int
    dx: float
    dy: float
    dtheta: float
    arc_length: float = 0.0
    radius_mm: Optional[float] = None
    speed_limit: float = 1.57

    def to_array(self) -> NDArray[np.float64]:
        """Return [dx, dy, dtheta] as numpy array."""
        return np.array([self.dx, self.dy, self.dtheta], dtype=np.float64)

    @property
    def is_straight(self) -> bool:
        """Check if this is a straight route (no turning)."""
        return abs(self.dtheta) < 0.01


@dataclass(frozen=True)
class PieceTopology:
    """Complete metadata for a piece (all phases)."""

    piece_id: str
    piece_index: int
    piece_class: PieceClass
    num_ports: int
    routes: Tuple[FKRoute, ...]
    default_route_idx: int = 0
    port_positions: Tuple[Tuple[float, float], ...] = ()
    port_headings: Tuple[float, ...] = ()

    def get_default_fk(self) -> FKRoute:
        """Get default route FK."""
        return self.routes[self.default_route_idx]

    def get_route(self, entry_port: int, exit_port: int) -> Optional[FKRoute]:
        """Find route by entry and exit ports."""
        for route in self.routes:
            if route.entry_port == entry_port and route.exit_port == exit_port:
                return route
        return None

    def get_route_by_index(self, route_idx: int) -> FKRoute:
        """Get route by index with fallback to default."""
        if 0 <= route_idx < len(self.routes):
            return self.routes[route_idx]
        return self.routes[self.default_route_idx]

    def is_simple_through(self) -> bool:
        """Check if single route through piece."""
        return len(self.routes) == 1

    def is_switch(self) -> bool:
        """Check if piece is a switch."""
<<<<<<< Updated upstream
        return self.piece_class == PieceClass.SWITCH_3PORT
=======
        return self.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)
>>>>>>> Stashed changes

    def is_crossing(self) -> bool:
        """Check if piece is a crossing."""
        return self.piece_class == PieceClass.CROSSING_4PORT

<<<<<<< Updated upstream
=======
    def is_terminator(self) -> bool:
        """Check if piece is a bumper/terminator."""
        return self.piece_class == PieceClass.BUMPER_1PORT

>>>>>>> Stashed changes

# =============================================================================
# SWITCH PAIR AND TRAVERSAL PATH
# =============================================================================


@dataclass
class SwitchPair:
    """Defines a paired IN/OUT switch creating a branch point.

    A switch pair creates two possible routes:
    1. Straight-through: main loop path (both switches use default route)
    2. Branch: diverge at IN, follow branch pieces, merge at OUT
    """

    pair_id: int
    in_position: int
    out_position: int
    in_switch_idx: int = -1
    out_switch_idx: int = -1
    branch_pieces: List[int] = field(default_factory=list)
    absorbed_positions: List[int] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if switch pair is properly defined."""
        return (
            self.in_position >= 0
            and self.out_position > self.in_position
            and self.in_switch_idx >= 0
            and self.out_switch_idx >= 0
        )

    @property
    def span(self) -> int:
        """Number of main loop positions between IN and OUT."""
        return self.out_position - self.in_position - 1


@dataclass
class TraversalPath:
    """A specific continuous route through the track topology."""

    path_id: int
    route_choices: Tuple[int, ...]
    piece_sequence: List[int] = field(default_factory=list)
    states: NDArray[np.float64] = field(default_factory=lambda: np.zeros((1, 3)))
    closure_error: float = 0.0
    angle_error: float = 0.0

    @property
    def n_pieces(self) -> int:
        """Number of pieces in this path."""
        return len(self.piece_sequence)

    @property
    def is_closed(self) -> bool:
        """Check if path forms a closed loop (within typical tolerances)."""
        return self.closure_error < 1.0 and self.angle_error < 5.0

    def describe_route(self) -> str:
        """Human-readable description of route choices."""
        if not self.route_choices:
            return "main"
        parts = []
        for i, choice in enumerate(self.route_choices):
            parts.append(f"SW{i}:{'branch' if choice else 'main'}")
        return ", ".join(parts)


@dataclass
class MultiPathLayout:
    """Complete track topology with all possible traversal paths.

    Represents a track layout where switch pairs create multiple valid
    routes. Each path is a continuous FK chain that must form a closed loop.
    """

    main_loop_pieces: List[int] = field(default_factory=list)
    switch_pairs: List[SwitchPair] = field(default_factory=list)
    paths: List[TraversalPath] = field(default_factory=list)
    start_position: Tuple[float, float] = (0.0, 0.0)
    loose_port_count: int = 0
    secondary_closure_error: float = 0.0

    @property
    def n_switch_pairs(self) -> int:
        return len(self.switch_pairs)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    @property
    def all_paths_closed(self) -> bool:
        return all(path.is_closed for path in self.paths)

    @property
    def max_closure_error(self) -> float:
        if not self.paths:
            return 0.0
        return max(path.closure_error for path in self.paths)

    @property
    def max_angle_error(self) -> float:
        if not self.paths:
            return 0.0
        return max(path.angle_error for path in self.paths)

    @property
    def total_pieces(self) -> int:
        all_pieces = set(self.main_loop_pieces)
        for sp in self.switch_pairs:
            all_pieces.add(sp.in_switch_idx)
            all_pieces.add(sp.out_switch_idx)
            all_pieces.update(sp.branch_pieces)
        all_pieces.discard(-1)
        return len(all_pieces)

    def get_path_by_choices(self, choices: Tuple[int, ...]) -> Optional[TraversalPath]:
        for path in self.paths:
            if path.route_choices == choices:
                return path
        return None

    def get_main_path(self) -> Optional[TraversalPath]:
        choices = tuple(0 for _ in self.switch_pairs)
        return self.get_path_by_choices(choices)

    # Backward compatibility with Layout interface
    @property
    def n_pieces(self) -> int:
        main = self.get_main_path()
        n = main.n_pieces if main else len(self.main_loop_pieces)
        for sp in self.switch_pairs:
            n += len(sp.branch_pieces)
        return n

    @property
    def indices(self) -> NDArray[np.int32]:
        main = self.get_main_path()
        if main:
            return np.array(main.piece_sequence, dtype=np.int32)
        return np.array(self.main_loop_pieces, dtype=np.int32)

    @property
    def states(self) -> NDArray[np.float64]:
        main = self.get_main_path()
        return main.states if main else np.zeros((1, 3))

    @property
    def closure_error(self) -> float:
        return self.max_closure_error

    @property
    def angle_error(self) -> float:
        return self.max_angle_error

    def is_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        return self.max_closure_error < pos_tol and self.max_angle_error < angle_tol

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        all_x = []
        all_y = []
        for path in self.paths:
            if len(path.states) > 0:
                all_x.extend(path.states[:, 0])
                all_y.extend(path.states[:, 1])
        if not all_x:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(all_x), min(all_y), max(all_x), max(all_y))

    @property
    def area(self) -> float:
        min_x, min_y, max_x, max_y = self.bounding_box
        return (max_x - min_x) * (max_y - min_y)


# =============================================================================
# PORT-GRAPH (V2 decoder intermediate representation)
# =============================================================================
#
# These types describe the port-graph view of a chromosome: piece slots as
# nodes, port pairs as edges. The decoder produces a PortGraph from a
# chromosome, then derives a MultiPathLayout from it for downstream consumers
# (problem evaluation, visualization).
#
# Conventions:
#   - Slot indices match positions in the chromosome's piece_slots region.
#   - Port names match catalog spec keys ("A", "B", "C", "D"); the decoder
#     converts integer port indices from the chromosome into names at parse
#     time.
#   - Poses are stored as (x, y, theta) with theta in RADIANS. Conversion to
#     degrees happens only at the MultiPathLayout boundary, where the existing
#     visualization code expects degrees.
# =============================================================================


@dataclass(frozen=True)
class PortEdge:
    """One connection between two slot ports.

    A connection is undirected; the (slot_a, port_a) / (slot_b, port_b)
    ordering is canonicalized in repair (e.g. lexicographic by slot then port)
    so that hashing and equality behave predictably.
    """

    slot_a: int
    port_a: str
    slot_b: int
    port_b: str

    def involves(self, slot_idx: int) -> bool:
        return self.slot_a == slot_idx or self.slot_b == slot_idx

    def other_endpoint(self, slot_idx: int) -> Tuple[int, str]:
        """Return the (slot, port) on the other side of this edge from ``slot_idx``."""
        if self.slot_a == slot_idx:
            return (self.slot_b, self.port_b)
        if self.slot_b == slot_idx:
            return (self.slot_a, self.port_a)
        raise ValueError(f"Edge does not involve slot {slot_idx}")


@dataclass(frozen=True)
class CycleResidual:
    """Closure residual on an edge that closed a cycle during BFS pose propagation.

    When BFS reaches a slot via an edge but that slot already has a propagated
    pose from a different walk, the difference is the residual. dx and dy are
    in studs; dtheta is in radians.
    """

    slot_a: int
    slot_b: int
    dx: float
    dy: float
    dtheta: float


@dataclass
class PortGraph:
    """Decoder's intermediate representation: a port-graph plus FK results."""

    slot_pieces: Dict[int, str] = field(default_factory=dict)
    """slot_idx -> piece_id (catalog key)."""

    slot_indices: Dict[int, int] = field(default_factory=dict)
    """slot_idx -> piece_index (numeric chromosome index)."""

<<<<<<< Updated upstream
    slot_flips: Dict[int, int] = field(default_factory=dict)
    """slot_idx -> flip bit (0 or 1). Always 0 for asymmetric pieces; the
    decoder enforces this regardless of what the chromosome says, so a
    visualization consumer can trust this dict directly."""

    slot_rotates: Dict[int, int] = field(default_factory=dict)
    """slot_idx -> 180° rotate bit (0 or 1). Always 0 for non-rotatable
    pieces; analogous to slot_flips."""

=======
>>>>>>> Stashed changes
    edges: List[PortEdge] = field(default_factory=list)
    """Sanitized connection list. Each port appears in at most one edge."""

    slot_poses: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    """slot_idx -> (x_world, y_world, theta_rad) at port A."""

    closure_residuals: List[CycleResidual] = field(default_factory=list)
    """One residual per cycle-closing edge encountered during BFS."""

    loose_ports: List[Tuple[int, str]] = field(default_factory=list)
    """Ports not present in any edge: (slot_idx, port_name)."""

    connected_components: List[Set[int]] = field(default_factory=list)
    """Slot-index sets, one per connected component."""

    dropped_edge_count: int = 0
    """Number of raw chromosome port-pair rows that failed validation
    (out-of-range slot, port_idx exceeding piece, double-booking, self-loop)."""

<<<<<<< Updated upstream
    branch_labels: Dict[Tuple[int, str], int] = field(default_factory=dict)
    """``(slot_idx, route_name) -> cycle_id``. Cycle membership labels.

    Derived from topology at decode time. Keys cover every slot that lies on
    at least one cycle, indexed by the route name (from the catalog spec)
    whose ports are both touched by edges of that cycle.

    For 2-port pieces (straights, curves) a single key per slot, ``"main"``.
    For switches, two keys: ``"through"`` (A-B route, usually the main loop)
    and ``"diverging"`` (A-C route, usually a siding branch). For crossings,
    ``"horizontal"`` (A-B) and ``"vertical"`` (C-D), each on a distinct
    cycle.

    Slots on no cycle (open chain ends, isolated singletons) are absent.
    Slots on cycles via a port pair that doesn't match any catalog route
    (e.g. switch port B-C edge usage, physically invalid) are also absent.
    """

=======
>>>>>>> Stashed changes
    @property
    def n_slots(self) -> int:
        return len(self.slot_pieces)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_components(self) -> int:
        return len(self.connected_components)

    @property
    def n_loose_ports(self) -> int:
        return len(self.loose_ports)

    @property
    def n_cycles(self) -> int:
        """Number of closed cycles, derived from V - E for each component.

        A connected component with V slots and E edges has ``E - V + 1`` cycles
        (the cyclomatic complexity). Open chains have 0 cycles.
        """
        cycles = 0
        for component in self.connected_components:
            v = len(component)
            e = sum(1 for edge in self.edges
                    if edge.slot_a in component and edge.slot_b in component)
            cycles += max(0, e - v + 1)
        return cycles

    @property
    def max_closure_position(self) -> float:
        """Largest position residual magnitude in studs."""
        if not self.closure_residuals:
            return 0.0
        return max(math.hypot(r.dx, r.dy) for r in self.closure_residuals)

    @property
    def max_closure_angle_rad(self) -> float:
        """Largest angle residual magnitude in radians."""
        if not self.closure_residuals:
            return 0.0
        return max(abs(r.dtheta) for r in self.closure_residuals)

    @property
    def max_closure_angle_deg(self) -> float:
        return math.degrees(self.max_closure_angle_rad)

    def edges_at(self, slot_idx: int) -> List[PortEdge]:
        """Return all edges touching the given slot."""
        return [e for e in self.edges if e.involves(slot_idx)]
