"""Data structures for forward-compatible multi-phase topology representation.

Supports:
- Phase 1: Single loop, default routes only
- Phase 2: Branches (dead-end or passing sidings)
- Phase 3: Multi-route switches (explicit route selection)
- Phase 4: Multiple loops with crossing connections

Key Concepts:
- SwitchPair: A pair of IN/OUT switches defining a branch point
- TraversalPath: A specific route through the track (continuous FK chain)
- MultiPathLayout: All possible traversal paths through the track topology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# SWITCH PAIR AND TRAVERSAL PATH STRUCTURES
# =============================================================================


@dataclass
class SwitchPair:
    """Defines a paired IN/OUT switch creating a branch point.

    A switch pair creates two possible routes:
    1. Straight-through: main loop path (both switches use default route)
    2. Branch: diverge at IN, follow branch pieces, merge at OUT

    Attributes:
        pair_id: Unique identifier for this switch pair
        in_position: Main loop position of IN switch (diverge point)
        out_position: Main loop position of OUT switch (merge point)
        in_switch_idx: Piece index of IN switch
        out_switch_idx: Piece index of OUT switch
        branch_pieces: Piece indices forming the branch
    """

    pair_id: int
    in_position: int
    out_position: int
    in_switch_idx: int = -1  # Determined from switch mask
    out_switch_idx: int = -1  # Determined from switch mask
    branch_pieces: List[int] = field(default_factory=list)
    # Positions absorbed by switches (their FK is included in the switch FK)
    # Used to skip these positions during path FK computation to preserve closure
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
    """A specific continuous route through the track topology.

    Each path represents one way a train could traverse the track.
    All paths must form closed loops.

    Attributes:
        path_id: Unique identifier for this path
        route_choices: Binary mask indicating route at each switch pair
                      (0 = straight-through, 1 = branch)
        piece_sequence: Ordered list of piece indices for this path
        states: Computed FK states (n+1, 3) for [x, y, theta]
        closure_error: Distance from end to start position
        angle_error: Deviation from 360 degrees
    """

    path_id: int
    route_choices: Tuple[int, ...]  # Binary mask for switch pair choices
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

    Attributes:
        main_loop_pieces: Base piece sequence (without switches)
        switch_pairs: List of switch pairs defining branch points
        paths: All possible traversal paths (2^N for N switch pairs)
        start_position: Starting position offset (x, y)
    """

    main_loop_pieces: List[int] = field(default_factory=list)
    switch_pairs: List[SwitchPair] = field(default_factory=list)
    paths: List[TraversalPath] = field(default_factory=list)
    start_position: Tuple[float, float] = (0.0, 0.0)
    loose_port_count: int = 0  # Unconnected switch/crossing port count

    @property
    def n_switch_pairs(self) -> int:
        """Number of switch pairs."""
        return len(self.switch_pairs)

    @property
    def n_paths(self) -> int:
        """Number of traversal paths."""
        return len(self.paths)

    @property
    def all_paths_closed(self) -> bool:
        """Check if all paths form closed loops."""
        return all(path.is_closed for path in self.paths)

    @property
    def max_closure_error(self) -> float:
        """Maximum closure error across all paths."""
        if not self.paths:
            return 0.0
        return max(path.closure_error for path in self.paths)

    @property
    def max_angle_error(self) -> float:
        """Maximum angle error across all paths."""
        if not self.paths:
            return 0.0
        return max(path.angle_error for path in self.paths)

    @property
    def total_pieces(self) -> int:
        """Total unique pieces in the layout."""
        all_pieces = set(self.main_loop_pieces)
        for sp in self.switch_pairs:
            all_pieces.add(sp.in_switch_idx)
            all_pieces.add(sp.out_switch_idx)
            all_pieces.update(sp.branch_pieces)
        # Remove -1 (inactive) if present
        all_pieces.discard(-1)
        return len(all_pieces)

    def get_path_by_choices(self, choices: Tuple[int, ...]) -> Optional[TraversalPath]:
        """Find path with specific route choices."""
        for path in self.paths:
            if path.route_choices == choices:
                return path
        return None

    def get_main_path(self) -> Optional[TraversalPath]:
        """Get the all-straight-through path (main loop)."""
        choices = tuple(0 for _ in self.switch_pairs)
        return self.get_path_by_choices(choices)

    # Backward compatibility with Layout interface
    @property
    def n_pieces(self) -> int:
        """Total pieces including main loop and all branch pieces.

        Counts main loop pieces plus branch pieces from switch pairs.
        This ensures switch-containing layouts get credit for all pieces used.
        """
        main = self.get_main_path()
        n = main.n_pieces if main else len(self.main_loop_pieces)
        # Add branch pieces from switch pairs (not double-counted)
        for sp in self.switch_pairs:
            n += len(sp.branch_pieces)
        return n

    @property
    def indices(self) -> NDArray[np.int32]:
        """Piece indices (from main path for compatibility)."""
        main = self.get_main_path()
        if main:
            return np.array(main.piece_sequence, dtype=np.int32)
        return np.array(self.main_loop_pieces, dtype=np.int32)

    @property
    def states(self) -> NDArray[np.float64]:
        """States (from main path for compatibility)."""
        main = self.get_main_path()
        return main.states if main else np.zeros((1, 3))

    @property
    def closure_error(self) -> float:
        """Closure error (max across all paths)."""
        return self.max_closure_error

    @property
    def angle_error(self) -> float:
        """Angle error (max across all paths)."""
        return self.max_angle_error

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Combined bounding box across all paths."""
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
        """Bounding box area."""
        min_x, min_y, max_x, max_y = self.bounding_box
        return (max_x - min_x) * (max_y - min_y)


class PieceClass(Enum):
    """Classify pieces by port topology."""

    SIMPLE_2PORT = "simple_2port"  # Straights, curves
    SWITCH_3PORT = "switch_3port"  # Standard switches (IN/OUT)
    SWITCH_4PORT = "switch_4port"  # Wye/3-way switches (Phase 3)
    CROSSING_4PORT = "crossing_4port"  # 90 deg crossings, double crossover
    BUMPER_1PORT = "bumper_1port"  # Dead-end terminators (Phase 2)


@dataclass(frozen=True)
class FKRoute:
    """Single route through a piece (entry port -> exit port)."""

    entry_port: int
    exit_port: int
    dx: float
    dy: float
    dtheta: float
    arc_length: float = 0.0
    radius_mm: Optional[float] = None  # None for straights
    speed_limit: float = 1.57  # Default to motor top speed

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
        return self.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)

    def is_crossing(self) -> bool:
        """Check if piece is a crossing."""
        return self.piece_class == PieceClass.CROSSING_4PORT

    def is_terminator(self) -> bool:
        """Check if piece is a bumper/terminator."""
        return self.piece_class == PieceClass.BUMPER_1PORT


# =============================================================================
# GENOTYPE - Chromosome Representation
# =============================================================================


@dataclass
class RouteSpec:
    """Specifies route selection at piece position."""

    piece_position: int
    route_idx: int = 0


@dataclass
class BranchDef:
    """Definition of branch (dead-end or passing siding)."""

    branch_id: int
    parent_loop_id: int
    diverge_position: int  # Position in parent loop where branch starts
    diverge_route: RouteSpec  # Route spec at divergence point
    piece_indices: List[int] = field(default_factory=list)
    route_specs: List[RouteSpec] = field(default_factory=list)
    is_dead_end: bool = True  # False for passing siding that rejoins
    rejoin_position: Optional[int] = None  # Position where branch rejoins (if not dead-end)
    rejoin_route: Optional[RouteSpec] = None
    connects_to_loop: Optional[int] = None  # Target loop ID for Phase 4

    def num_pieces(self) -> int:
        """Count branch pieces."""
        return len(self.piece_indices)

    def get_route_idx(self, position: int) -> int:
        """Get route index at position in branch."""
        for spec in self.route_specs:
            if spec.piece_position == position:
                return spec.route_idx
        return 0


@dataclass
class LoopDef:
    """Definition of single closed loop with branches."""

    loop_id: int
    main_sequence: List[int] = field(default_factory=list)  # Piece indices in loop
    route_specs: List[RouteSpec] = field(default_factory=list)
    branches: List[BranchDef] = field(default_factory=list)
    priority: int = 0  # Loop priority for Phase 4

    def num_main_pieces(self) -> int:
        """Count pieces in main loop."""
        return len(self.main_sequence)

    def num_branch_pieces(self) -> int:
        """Count pieces in all branches."""
        return sum(b.num_pieces() for b in self.branches)

    def num_total_pieces(self) -> int:
        """Count all pieces including branches."""
        return self.num_main_pieces() + self.num_branch_pieces()

    def get_all_piece_indices(self) -> List[int]:
        """Flat list of all pieces (main + branches)."""
        result = list(self.main_sequence)
        for branch in self.branches:
            result.extend(branch.piece_indices)
        return result

    def get_route_idx(self, position: int) -> int:
        """Get route index at position in main loop."""
        for spec in self.route_specs:
            if spec.piece_position == position:
                return spec.route_idx
        return 0


@dataclass
class LayoutChromosome:
    """Complete layout genotype (supports all phases)."""

    loops: List[LoopDef] = field(default_factory=list)
    crossing_connections: Dict[Tuple[int, int], Tuple[int, int, int]] = field(default_factory=dict)
    # Key: (loop1_id, loop2_id), Value: (crossing_piece_idx, loop1_position, loop2_position)

    def num_loops(self) -> int:
        """Count number of loops."""
        return len(self.loops)

    def num_pieces_total(self) -> int:
        """Total pieces (crossings counted once)."""
        total = sum(loop.num_total_pieces() for loop in self.loops)
        # Subtract duplicate crossings
        total -= len(self.crossing_connections)
        return total

    def get_all_piece_indices(self) -> List[int]:
        """Flat list of all pieces."""
        result = []
        for loop in self.loops:
            result.extend(loop.get_all_piece_indices())
        return result

    def get_main_loop(self) -> Optional[LoopDef]:
        """Get first loop (main loop for Phase 1)."""
        return self.loops[0] if self.loops else None

    def to_flat_array(self, max_length: int) -> NDArray[np.int32]:
        """Convert to pymoo flat array representation.

        Args:
            max_length: Maximum array length (padded with -1).

        Returns:
            Array of piece indices, -1 for empty slots.
        """
        indices = self.get_all_piece_indices()
        result = np.full(max_length, -1, dtype=np.int32)
        result[: len(indices)] = indices
        return result

    @classmethod
    def from_flat_array(cls, arr: NDArray) -> "LayoutChromosome":
        """Create from flat array (Phase 1 only - single loop).

        Args:
            arr: Array of piece indices (-1 for empty).

        Returns:
            LayoutChromosome with single loop.
        """
        valid_indices = arr[arr >= 0].tolist()
        loop = LoopDef(loop_id=0, main_sequence=valid_indices)
        return cls(loops=[loop])

    @classmethod
    def from_simple_sequence(cls, piece_indices: List[int]) -> "LayoutChromosome":
        """Create from simple piece list (Phase 1)."""
        loop = LoopDef(loop_id=0, main_sequence=piece_indices)
        return cls(loops=[loop])


# =============================================================================
# PHENOTYPE - Computed Geometry
# =============================================================================


@dataclass
class BranchGeometry:
    """Computed branch geometry (phenotype)."""

    branch_id: int
    parent_loop_id: int
    piece_indices: NDArray[np.int32]
    route_indices: NDArray[np.int32]
    states: NDArray[np.float64]  # (n+1, 3) trajectory [x, y, theta]
    arc_lengths: NDArray[np.float64]
    radii: NDArray[np.float64]
    speed_limits: NDArray[np.float64]
    is_dead_end: bool = True
    rejoin_error: Optional[float] = None  # Position error at rejoin
    rejoin_angle_error: Optional[float] = None  # Angle error at rejoin

    @property
    def n_pieces(self) -> int:
        """Number of pieces in branch."""
        return len(self.piece_indices)

    @property
    def total_length(self) -> float:
        """Total arc length in studs."""
        return float(np.sum(self.arc_lengths))

    @property
    def start_state(self) -> NDArray[np.float64]:
        """Starting state [x, y, theta]."""
        return self.states[0]

    @property
    def end_state(self) -> NDArray[np.float64]:
        """Ending state [x, y, theta]."""
        return self.states[-1]


@dataclass
class LoopGeometry:
    """Computed loop geometry (phenotype)."""

    loop_id: int
    piece_indices: NDArray[np.int32]
    route_indices: NDArray[np.int32]
    states: NDArray[np.float64]  # (n+1, 3) trajectory [x, y, theta]
    arc_lengths: NDArray[np.float64]
    radii: NDArray[np.float64]
    speed_limits: NDArray[np.float64]
    closure_error: float = 0.0  # Position error at closure
    angle_error: float = 0.0  # Angle error at closure
    bbox_min: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    bbox_max: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    branches: List[BranchGeometry] = field(default_factory=list)

    @property
    def n_pieces(self) -> int:
        """Number of pieces in main loop."""
        return len(self.piece_indices)

    @property
    def n_states(self) -> int:
        """Number of states (n_pieces + 1)."""
        return len(self.states)

    @property
    def total_length(self) -> float:
        """Total arc length in studs."""
        return float(np.sum(self.arc_lengths))

    @property
    def final_state(self) -> NDArray[np.float64]:
        """Final state [x, y, theta]."""
        return self.states[-1]

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounding box (min_x, min_y, max_x, max_y)."""
        return (
            float(self.bbox_min[0]),
            float(self.bbox_min[1]),
            float(self.bbox_max[0]),
            float(self.bbox_max[1]),
        )

    @property
    def area(self) -> float:
        """Bounding box area in studs^2."""
        width = self.bbox_max[0] - self.bbox_min[0]
        height = self.bbox_max[1] - self.bbox_min[1]
        return float(width * height)

    def is_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        """Check if loop is closed within tolerances."""
        return self.closure_error <= pos_tol and self.angle_error <= angle_tol


@dataclass
class LayoutGeometry:
    """Complete computed geometry (phenotype)."""

    loops: List[LoopGeometry] = field(default_factory=list)
    collisions: List[Tuple[int, int, float, float]] = field(default_factory=list)
    # List of (loop1_id, loop2_id, x, y) collision points

    def num_loops(self) -> int:
        """Count number of loops."""
        return len(self.loops)

    def total_pieces(self) -> int:
        """Total pieces in all loops."""
        return sum(loop.n_pieces for loop in self.loops)

    def total_branch_pieces(self) -> int:
        """Total pieces in all branches."""
        total = 0
        for loop in self.loops:
            for branch in loop.branches:
                total += branch.n_pieces
        return total

    def all_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        """Check if all loops are closed."""
        return all(loop.is_closed(pos_tol, angle_tol) for loop in self.loops)

    def max_closure_error(self) -> float:
        """Maximum closure error across all loops."""
        if not self.loops:
            return 0.0
        return max(loop.closure_error for loop in self.loops)

    def max_angle_error(self) -> float:
        """Maximum angle error across all loops."""
        if not self.loops:
            return 0.0
        return max(loop.angle_error for loop in self.loops)

    def combined_bounding_box(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combined bounding box for all loops."""
        if not self.loops:
            return np.zeros(2), np.zeros(2)

        all_min = np.array([loop.bbox_min for loop in self.loops])
        all_max = np.array([loop.bbox_max for loop in self.loops])
        return np.min(all_min, axis=0), np.max(all_max, axis=0)

    def combined_bounding_box_area(self) -> float:
        """Combined bounding box area."""
        bbox_min, bbox_max = self.combined_bounding_box()
        width = bbox_max[0] - bbox_min[0]
        height = bbox_max[1] - bbox_min[1]
        return float(width * height)

    def has_collisions(self) -> bool:
        """Check if any collisions detected."""
        return len(self.collisions) > 0

    def get_main_loop(self) -> Optional[LoopGeometry]:
        """Get first loop (main loop for Phase 1)."""
        return self.loops[0] if self.loops else None

    # =========================================================================
    # Backward Compatibility Properties (Phase 1)
    # =========================================================================

    @property
    def indices(self) -> NDArray[np.int32]:
        """Piece indices from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.piece_indices if main_loop else np.array([], dtype=np.int32)

    @property
    def states(self) -> NDArray[np.float64]:
        """States from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.states if main_loop else np.zeros((1, 3))

    @property
    def n_pieces(self) -> int:
        """Number of pieces in main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.n_pieces if main_loop else 0

    @property
    def closure_error(self) -> float:
        """Closure error from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.closure_error if main_loop else 0.0

    @property
    def angle_error(self) -> float:
        """Angle error from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.angle_error if main_loop else 0.0

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounding box from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.bounding_box if main_loop else (0.0, 0.0, 0.0, 0.0)

    @property
    def area(self) -> float:
        """Area from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.area if main_loop else 0.0

    @property
    def final_state(self) -> NDArray[np.float64]:
        """Final state from main loop (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.final_state if main_loop else np.zeros(3)

    def is_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        """Check if main loop is closed (Phase 1 compatibility)."""
        main_loop = self.get_main_loop()
        return main_loop.is_closed(pos_tol, angle_tol) if main_loop else False
