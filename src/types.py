"""Shared domain types for the LEGO Track Optimizer.

Pure data containers (dataclasses / enums) with no logic beyond property
accessors — the common vocabulary depended on by the catalog, geometry,
train, decoder, operator, problem, and visualization modules.
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

from src.encoding import CROSS_90 as CROSS_90_INDEX


class PieceClass(Enum):
    """Classify pieces by port topology."""

    SIMPLE_2PORT = "simple_2port"
    SWITCH_3PORT = "switch_3port"
    CROSSING_4PORT = "crossing_4port"    # routes intersect, no way between them
    CROSSOVER_4PORT = "crossover_4port"  # diagonals carry a train between tracks


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

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.dx, self.dy, self.dtheta], dtype=np.float64)

    @property
    def is_straight(self) -> bool:
        return abs(self.dtheta) < 0.01


@dataclass(frozen=True)
class PieceTopology:
    """All FK/port metadata for one catalog piece."""

    piece_id: str
    piece_index: int
    piece_class: PieceClass
    num_ports: int
    routes: Tuple[FKRoute, ...]
    default_route_idx: int = 0
    port_positions: Tuple[Tuple[float, float], ...] = ()
    port_headings: Tuple[float, ...] = ()

    def is_switch(self) -> bool:
        return self.piece_class == PieceClass.SWITCH_3PORT

    def is_crossing(self) -> bool:
        """CROSS_90: two routes cross, a train cannot pass between them."""
        return self.piece_class == PieceClass.CROSSING_4PORT

    def is_crossover(self) -> bool:
        """DOUBLE_CROSSOVER: diagonal routes carry a train from track to track."""
        return self.piece_class == PieceClass.CROSSOVER_4PORT


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
    # Per-slot flip for the branch piece sequence. R40_CURVE consults this to
    # negate dy/dtheta when flip=1; symmetric pieces (straights, switches) ignore.
    branch_flips: List[int] = field(default_factory=list)
    absorbed_positions: List[int] = field(default_factory=list)
    # Train-frame FK applied at the exit (reversed-install) switch on a branch
    # path. Populated from PassingSidingTemplate.merge_fk at decode time
    # because catalog routes can't express reversed-installation traversal.
    merge_fk: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DblCrossover:
    """A DOUBLE_CROSSOVER physical piece traversed twice by the loop.

    The piece occupies two main-loop slots — pos_1 (with route_1) and pos_2
    (with route_2) — whose routes together cover all 4 ports {A,B,C,D} so no
    port dangles. The injection algorithm sets both slots in
    main_loop_pieces to DOUBLE_CROSSOVER and records the route used at each
    so the FK chain pulls the correct catalog route at each traversal.
    """

    slot: int
    positions: Tuple[int, int]
    routes: Tuple[int, int]
    origin: Tuple[float, float, float]  # world pose of port A


@dataclass
class CrossJunction:
    """A CROSS_90 physical piece traversed twice by the loop.

    The piece occupies two main-loop slots — pos_1 and pos_2 — whose FK states
    coincide in world space and cross at ~90 deg, so all four ports are used and
    none dangles. The injection algorithm sets both slots in main_loop_pieces to
    CROSS_90. Both passes are straight-through (CROSS_90 FK == STRAIGHT_16 FK), so
    no per-slot route map is needed. Mirrors DblCrossover. ``slot`` is the
    committing descriptor's index, or -1 when the record comes from the emergent
    self-intersection repair.

    NOTE: this models a *bare self-crossing* (figure-8), NOT the 4-switch routing
    cross-junction (4 switches around a central cross). The latter is not
    deliberately simulated.
    """

    slot: int
    positions: Tuple[int, int]
    origin: Tuple[float, float, float]  # world pose of the crossing center

    def is_valid(self) -> bool:
        return (
            self.positions[0] >= 0
            and self.positions[1] >= 0
            and self.positions[0] != self.positions[1]
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
    # Catalog route index per piece in piece_sequence (parallel list). 0 = through /
    # default route; switch diverging = 1; DOUBLE_CROSSOVER diagonals = 2, 3. The
    # speed profiler uses it so branch and diagonal segments are curve-speed-limited
    # instead of scored at through-route physics.
    route_indices: List[int] = field(default_factory=list)
    # Physical-piece identity per piece in piece_sequence (parallel list):
    # ("main", slot, 0) for a main-loop slot, ("branch", pair_idx, k) for the
    # k-th branch piece of switch pair pair_idx. Paths sharing a physical piece
    # carry the same uid, so per-piece quantities can be aggregated across the
    # 2^J routes without double-counting. Note a descriptor CROSS_90 /
    # DOUBLE_CROSSOVER is ONE physical piece on TWO main slots (two uids);
    # consumers unify them via the layout's junction records.
    piece_uids: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def n_pieces(self) -> int:
        return len(self.piece_sequence)

    @property
    def is_closed(self) -> bool:
        """Check if path forms a closed loop (within typical tolerances)."""
        return self.closure_error <= 1.0 and self.angle_error <= 5.0

    def describe_route(self) -> str:
        """Human-readable description of route choices."""
        if not self.route_choices:
            return "main"
        return ", ".join(
            f"SW{i}:{'branch' if choice else 'main'}"
            for i, choice in enumerate(self.route_choices)
        )


@dataclass
class MultiPathLayout:
    """Complete track topology with all possible traversal paths.

    Represents a track layout where switch pairs create multiple valid
    routes. Each path is a continuous FK chain that must form a closed loop.
    """

    main_loop_pieces: List[int] = field(default_factory=list)
    switch_pairs: List[SwitchPair] = field(default_factory=list)
    cross_junctions: List["CrossJunction"] = field(default_factory=list)
    dbl_crossovers: List["DblCrossover"] = field(default_factory=list)
    # Map: main_loop position -> catalog route index (for DOUBLE_CROSSOVER slots
    # whose train traversal uses a non-default route). Used by _compute_path_fk
    # to pick the right FK row at each occurrence.
    main_loop_routes: Dict[int, int] = field(default_factory=dict)
    paths: List[TraversalPath] = field(default_factory=list)
    start_position: Tuple[float, float] = (0.0, 0.0)
    loose_port_count: int = 0
    # Human-readable reasons for descriptors the decoder skipped (empty when
    # everything committed). Consumed by the per-category run report.
    drop_log: List[str] = field(default_factory=list)

    @property
    def n_cross_junctions(self) -> int:
        return len(self.cross_junctions)

    @property
    def n_physical_pieces(self) -> int:
        """Physical pieces used, including siding branches.

        CROSS_90 / DOUBLE_CROSSOVER records occupy TWO traversal slots in
        ``main_loop_pieces`` but are one physical piece, so subtract one per
        record (descriptor and emergent crossings share this shape).
        """
        n_paired = len(self.cross_junctions) + len(self.dbl_crossovers)
        branch = sum(len(sp.branch_pieces) for sp in self.switch_pairs)
        return len(self.main_loop_pieces) - n_paired + branch

    @property
    def n_cross_pieces(self) -> int:
        """Physical CROSS_90 pieces in the loop, regardless of origin.

        Every committed crossing — descriptor or emergent — marks BOTH its
        slots CROSS_90 and carries a CrossJunction record, so physical pieces
        are CROSS_90 slots minus one per record.
        """
        cross_slots = sum(1 for p in self.main_loop_pieces if p == CROSS_90_INDEX)
        return cross_slots - len(self.cross_junctions)

    @property
    def n_dbl_crossovers(self) -> int:
        return len(self.dbl_crossovers)

    @property
    def n_switch_pairs(self) -> int:
        return len(self.switch_pairs)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

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

    # These properties expose a geometry.Layout-shaped view for the renderer
    # and train evaluation.
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
        return self.max_closure_error <= pos_tol and self.max_angle_error <= angle_tol

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        all_x = []
        all_y = []
        for path in self.paths:
            if len(path.states) > 0:
                all_x.extend(path.states[:, 0])
                all_y.extend(path.states[:, 1])
        if not all_x:
            return 0.0, 0.0, 0.0, 0.0
        return min(all_x), min(all_y), max(all_x), max(all_y)

    @property
    def area(self) -> float:
        min_x, min_y, max_x, max_y = self.bounding_box
        return (max_x - min_x) * (max_y - min_y)
