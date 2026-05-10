"""Shared domain types for the LEGO Track Optimizer.

Pure data containers with no logic beyond property accessors.
These form the shared vocabulary across module tiers:
  - Tier 1 (domain core): catalog, geometry, train
  - Tier 2 (EA layer): decoder, operators, problem
  - Tier 3 (infra): visualization, config, main
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# PIECE CLASSIFICATION AND ROUTING
# =============================================================================


class PieceClass(Enum):
    """Classify pieces by port topology."""

    SIMPLE_2PORT = "simple_2port"
    SWITCH_3PORT = "switch_3port"
    SWITCH_4PORT = "switch_4port"
    CROSSING_4PORT = "crossing_4port"
    BUMPER_1PORT = "bumper_1port"


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
        return self.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)

    def is_crossing(self) -> bool:
        """Check if piece is a crossing."""
        return self.piece_class == PieceClass.CROSSING_4PORT

    def is_terminator(self) -> bool:
        """Check if piece is a bumper/terminator."""
        return self.piece_class == PieceClass.BUMPER_1PORT


# =============================================================================
# SWITCH PAIR AND TRAVERSAL PATH
# =============================================================================


@dataclass
class SwitchPair:
    """A passing siding occupying two positions in the main loop.

    The two switches are opposite-handed (one LEFT, one RIGHT). The entry
    switch matches `handedness`; the exit switch is opposite handedness and
    is installed REVERSED on the main loop. The decoder derives concrete
    switch piece indices from `handedness` when building the FK chain.

    Two route modes per pair:
      - Through: main loop traversal — both switches as 32-stud straights.
      - Branch: enter siding via the entry switch's diverge route, follow
        `branch_pieces`, return via the exit switch's reversed merge.
    """

    pair_id: int
    in_position: int                # main-loop position of the entry switch
    out_position: int               # main-loop position of the exit switch
    handedness: str = "LEFT"        # "LEFT" or "RIGHT" — siding diverges to this side
    branch_pieces: List[int] = field(default_factory=list)
    absorbed_positions: List[int] = field(default_factory=list)
    # Train-frame FK applied at the exit (reversed-install) switch on a branch
    # path. Populated from PassingSidingTemplate.merge_fk at decode time
    # because catalog routes can't express reversed-installation traversal.
    merge_fk: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def is_valid(self) -> bool:
        return (
            self.in_position >= 0
            and self.out_position > self.in_position
            and self.handedness in ("LEFT", "RIGHT")
        )

    @property
    def span(self) -> int:
        """Number of main loop positions between IN and OUT."""
        return self.out_position - self.in_position - 1


@dataclass
class CrossJunction:
    """4-switch + CROSS_90 junction in the layout.

    The 4 switches sit on the main loop (replacing pieces at four specific
    positions). The CROSS_90 + 4 R40 spurs are an internal sub-structure.

    Per junction, train traversal at the junction has 3 modes:
      0: bypass — train uses each switch's port B, stays on main loop
      1: cross_we — train enters via switch_for_W, traverses cross W↔E,
                     exits via switch_for_E (or reverse direction)
      2: cross_ns — train enters via switch_for_N, traverses cross N↔S,
                     exits via switch_for_S (or reverse direction)
    """

    junction_id: int
    handedness: str  # "LEFT" or "RIGHT" — handedness of all 4 switches
    # Main-loop positions where the 4 switches are injected, keyed by which
    # cross port they connect to. Each position is an index in main_loop_pieces.
    switch_positions: Dict[str, int] = field(default_factory=dict)  # "W"/"E"/"N"/"S" -> position
    # Cross center in main frame (computed from switch positions during inject)
    cross_center: Tuple[float, float] = (0.0, 0.0)
    cross_idx: int = 4  # CROSS_90 piece index

    def is_valid(self) -> bool:
        return (
            len(self.switch_positions) == 4
            and self.handedness in ("LEFT", "RIGHT")
            and all(p >= 0 for p in self.switch_positions.values())
        )


@dataclass
class TraversalPath:
    """A specific continuous route through the track topology."""

    path_id: int
    route_choices: Tuple[int, ...]
    piece_sequence: List[int] = field(default_factory=list)
    states: NDArray[np.float64] = field(default_factory=lambda: np.zeros((1, 3)))
    closure_error: float = 0.0
    angle_error: float = 0.0
    # Map sorted-pair index (matching route_choices indexing) -> (start, end_inclusive)
    # piece_sequence/states slice covering that pair's branch section (IN switch
    # through OUT switch, inclusive on both ends). Populated only for pair indices
    # where route_choices[i] == 1.
    divergent_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)

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
    cross_junctions: List["CrossJunction"] = field(default_factory=list)
    paths: List[TraversalPath] = field(default_factory=list)
    start_position: Tuple[float, float] = (0.0, 0.0)
    loose_port_count: int = 0
    secondary_closure_error: float = 0.0

    @property
    def n_cross_junctions(self) -> int:
        return len(self.cross_junctions)

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
        # main_loop_pieces already includes both switches (injected at
        # in_position and out_position by the decoder). Just add branch pieces.
        all_pieces = set(self.main_loop_pieces)
        for sp in self.switch_pairs:
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

    def get_branch_segments(self) -> List[Tuple[List[int], NDArray[np.float64]]]:
        """Return one (piece_sequence_slice, states_slice) per branched switch pair.

        Iterates paths to find the first occurrence of each branched pair and
        extracts the branch geometry from that path's piece_sequence and states.
        Each pair appears at most once (deduplicated across the 2^N enumeration).
        """
        seen: set = set()
        segments: List[Tuple[List[int], NDArray[np.float64]]] = []
        for path in self.paths:
            for pair_idx, (start, end) in path.divergent_ranges.items():
                if pair_idx in seen:
                    continue
                seen.add(pair_idx)
                pieces = path.piece_sequence[start : end + 1]
                states = path.states[start : end + 2]  # +2 to include exit state
                segments.append((pieces, states))
        return segments

    def get_post_merge_drift(self) -> List[Tuple[int, NDArray[np.float64]]]:
        """Return (path_idx, drift_states) for each path with FK drift.

        Drift starts after the FIRST branched OUT switch in the path's
        sequence. Pieces past that point are physically real (drawn solid in
        the main path / branch templates), but their FK positions here have
        drifted because the branch's merge route did not align with the main
        loop heading. Renderers should display this as a diagnostic indicator
        (dotted line), not as solid track.
        """
        drifts: List[Tuple[int, NDArray[np.float64]]] = []
        for path_idx, path in enumerate(self.paths):
            if not path.divergent_ranges:
                continue
            first_merge_end = min(end for _, end in path.divergent_ranges.values())
            drift_states = path.states[first_merge_end + 1 :]
            if len(drift_states) >= 2:
                drifts.append((path_idx, drift_states))
        return drifts

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

