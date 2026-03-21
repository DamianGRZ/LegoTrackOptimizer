"""Unified track layout optimization problem.

Simplified approach where topology emerges naturally:
- Evolve piece sequence (straights, curves, switches)
- Decoder validates switch connections via self-intersection
- Constraint penalizes loose switch ports
- GA discovers valid topologies (sidings, figure-8, loop-to-loop)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.problem import ElementwiseProblem

from .data import TrackCatalog
from .geometry import Layout, build_layout, compute_fk_chain
from .intersection import analyze_switch_connections, IntersectionResult
from .encoding import (
    INACTIVE,
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
    SWITCH_INDICES,
)


# =============================================================================
# Simplified Encoding Constants
# =============================================================================

# Just piece sequence + start position
U_MAX = 120      # Max pieces in sequence
U_POSITION = 2   # start_x, start_y
U_N_VAR = U_MAX + U_POSITION  # 122 total genes

# Gene indices
U_PIECES_START = 0
U_PIECES_END = U_MAX
U_POS_START = U_MAX
U_POS_END = U_N_VAR


@dataclass
class UnifiedConfig:
    """Configuration for unified track problem."""

    # Closure tolerances
    position_tolerance: float = 2.0   # studs
    angle_tolerance: float = 5.0      # degrees

    # Boundary
    boundary_min: float = -200.0
    boundary_max: float = 200.0

    # Switch connection tolerance
    switch_position_tol: float = 4.0  # studs
    switch_angle_tol: float = 10.0    # degrees

    # Fitness weights
    utilization_weight: float = 1.0
    speed_weight: float = 0.5
    switch_bonus: float = 0.1  # Bonus per connected switch pair


@dataclass
class UnifiedLayout:
    """Result of decoding a unified chromosome."""

    # Basic layout
    indices: NDArray[np.int32]
    states: NDArray[np.float64]

    # Closure metrics
    closure_error: float
    angle_error: float

    # Bounding box
    bounding_box: Tuple[float, float, float, float]

    # Switch analysis
    intersection_result: IntersectionResult

    # Piece counts
    n_pieces: int
    n_switches: int
    n_connected_pairs: int
    n_loose_ports: int

    @property
    def is_closed(self) -> bool:
        """Check if main loop closes."""
        return self.closure_error < 2.0 and self.angle_error < 5.0

    @property
    def has_valid_switches(self) -> bool:
        """Check if all switches have valid connections."""
        return self.n_loose_ports == 0


def decode_unified(
    chromosome: NDArray,
    catalog: TrackCatalog,
    config: Optional[UnifiedConfig] = None,
) -> UnifiedLayout:
    """Decode unified chromosome into layout with switch analysis.

    Args:
        chromosome: Array of length U_N_VAR.
        catalog: Track catalog for FK lookup.
        config: Optional configuration.

    Returns:
        UnifiedLayout with geometry and switch connection analysis.
    """
    if config is None:
        config = UnifiedConfig()

    # Extract pieces (filter inactive)
    pieces = chromosome[U_PIECES_START:U_PIECES_END].astype(np.int32)
    valid_mask = pieces >= 0
    indices = pieces[valid_mask]

    n_pieces = len(indices)

    if n_pieces == 0:
        return _empty_layout()

    # Compute FK chain
    fk_deltas = catalog.get_fk(indices)
    states = compute_fk_chain(fk_deltas)

    # Compute closure metrics BEFORE applying offset
    # (closure is relative to origin of FK chain)
    final = states[-1]
    closure_error = float(np.sqrt(final[0]**2 + final[1]**2))

    # Apply start position offset for visualization/boundary
    start_x = chromosome[U_POS_START]
    start_y = chromosome[U_POS_START + 1]
    states[:, 0] += start_x
    states[:, 1] += start_y
    total_angle = abs(final[2])
    angle_error = min(total_angle % 360, 360 - (total_angle % 360)) if total_angle > 0 else 360.0

    # Compute bounding box
    min_x, max_x = float(np.min(states[:, 0])), float(np.max(states[:, 0]))
    min_y, max_y = float(np.min(states[:, 1])), float(np.max(states[:, 1]))
    bounding_box = (min_x, min_y, max_x, max_y)

    # Analyze switch connections
    intersection_result = analyze_switch_connections(
        states, indices,
        position_tolerance=config.switch_position_tol,
        angle_tolerance=config.switch_angle_tol,
    )

    # Count switches
    n_switches = sum(1 for idx in indices if idx in SWITCH_INDICES)
    n_connected_pairs = len(intersection_result.connected_switches)
    n_loose_ports = intersection_result.loose_port_count

    return UnifiedLayout(
        indices=indices,
        states=states,
        closure_error=closure_error,
        angle_error=angle_error,
        bounding_box=bounding_box,
        intersection_result=intersection_result,
        n_pieces=n_pieces,
        n_switches=n_switches,
        n_connected_pairs=n_connected_pairs,
        n_loose_ports=n_loose_ports,
    )


def _empty_layout() -> UnifiedLayout:
    """Create empty layout for invalid chromosomes."""
    return UnifiedLayout(
        indices=np.array([], dtype=np.int32),
        states=np.zeros((1, 3)),
        closure_error=1000.0,
        angle_error=360.0,
        bounding_box=(0, 0, 0, 0),
        intersection_result=IntersectionResult(
            opportunities=[],
            switch_count=0,
            connected_switches=[],
            loose_port_count=0,
        ),
        n_pieces=0,
        n_switches=0,
        n_connected_pairs=0,
        n_loose_ports=0,
    )


# =============================================================================
# Unified Problem Class
# =============================================================================

class UnifiedTrackProblem(ElementwiseProblem):
    """Unified track layout optimization problem.

    Chromosome: [piece_indices × U_MAX] + [start_x, start_y]

    Objectives (minimized):
        F[0]: -utilization (maximize piece usage)
        F[1]: -topology_score (maximize connected switches + closure quality)

    Constraints (g <= 0 feasible):
        G[0]: closure_error - tolerance
        G[1]: angle_error - tolerance
        G[2]: boundary_violation
        G[3]: inventory_excess
        G[4]: loose_port_count (switches must connect)
    """

    def __init__(
        self,
        catalog: TrackCatalog,
        inventory: Dict[str, int],
        config: Optional[UnifiedConfig] = None,
    ):
        """Initialize unified track problem.

        Args:
            catalog: Track catalog with piece properties.
            inventory: Available pieces {piece_id: count}.
            config: Problem configuration.
        """
        self.catalog = catalog
        self.inventory = inventory
        self.config = config or UnifiedConfig()

        # Convert inventory to index-based
        self.inventory_by_idx = self._convert_inventory(inventory, catalog)
        self.total_inventory = sum(inventory.values())

        # Get max piece index
        self.max_piece_idx = catalog.n_pieces - 1

        # Bounds
        xl = np.full(U_N_VAR, -1.0)  # -1 = inactive
        xu = np.full(U_N_VAR, float(self.max_piece_idx))

        # Position bounds
        xl[U_POS_START:U_POS_END] = self.config.boundary_min
        xu[U_POS_START:U_POS_END] = self.config.boundary_max

        super().__init__(
            n_var=U_N_VAR,
            n_obj=2,
            n_ieq_constr=5,
            xl=xl,
            xu=xu,
            vtype=float,
        )

    def _convert_inventory(
        self,
        inventory: Dict[str, int],
        catalog: TrackCatalog,
    ) -> Dict[int, int]:
        """Convert string-keyed inventory to index-keyed."""
        result = {}
        id_to_idx = catalog.id_to_index
        for piece_id, count in inventory.items():
            if piece_id in id_to_idx:
                result[id_to_idx[piece_id]] = count
        return result

    def _evaluate(self, x: NDArray, out: Dict, *args, **kwargs) -> None:
        """Evaluate single chromosome."""
        # Decode chromosome
        layout = decode_unified(x, self.catalog, self.config)

        # Compute objectives
        utilization = layout.n_pieces / max(1, self.total_inventory)

        # Topology score: reward connected switches and good closure
        closure_quality = max(0, 1.0 - layout.closure_error / 10.0)
        switch_score = layout.n_connected_pairs * self.config.switch_bonus
        topology_score = closure_quality + switch_score

        out["F"] = [
            -utilization,      # Maximize utilization
            -topology_score,   # Maximize topology quality
        ]

        # Compute constraints
        # G[0]: Position closure
        pos_tol = self.config.position_tolerance
        g_closure = (layout.closure_error - pos_tol) / max(pos_tol, 1.0)

        # G[1]: Angle closure
        ang_tol = self.config.angle_tolerance
        g_angle = (layout.angle_error - ang_tol) / max(ang_tol, 1.0)

        # G[2]: Boundary violation
        min_x, min_y, max_x, max_y = layout.bounding_box
        boundary_violation = max(
            0,
            self.config.boundary_min - min_x,
            self.config.boundary_min - min_y,
            max_x - self.config.boundary_max,
            max_y - self.config.boundary_max,
        )
        diagonal = np.sqrt(2) * (self.config.boundary_max - self.config.boundary_min)
        g_boundary = boundary_violation / max(diagonal, 1.0)

        # G[3]: Inventory excess
        inventory_excess = self._compute_inventory_excess(layout.indices)
        g_inventory = inventory_excess

        # G[4]: Loose switch ports
        g_loose = float(layout.n_loose_ports)

        out["G"] = [g_closure, g_angle, g_boundary, g_inventory, g_loose]

    def _compute_inventory_excess(self, indices: NDArray) -> float:
        """Compute how much inventory is exceeded."""
        usage = {}
        for idx in indices:
            usage[idx] = usage.get(idx, 0) + 1

        excess = 0
        for idx, count in usage.items():
            available = self.inventory_by_idx.get(idx, 0)
            if count > available:
                excess += count - available

        return float(excess)


# =============================================================================
# Sampling for Unified Problem
# =============================================================================

class UnifiedSampling:
    """Sampling strategies for unified problem."""

    def __init__(self, catalog: TrackCatalog, inventory: Dict[str, int]):
        self.catalog = catalog
        self.inventory = inventory

    def sample_circle(self) -> NDArray:
        """Sample a simple R40 circle (16 curves)."""
        x = np.full(U_N_VAR, INACTIVE, dtype=np.float64)

        # 16 left curves = full circle
        from .encoding import R40_LEFT
        for i in range(16):
            x[i] = R40_LEFT

        return x

    def sample_oval(self, n_straights: int = 4) -> NDArray:
        """Sample an oval with straights on long sides."""
        x = np.full(U_N_VAR, INACTIVE, dtype=np.float64)

        from .encoding import R40_LEFT, R40_RIGHT, STRAIGHT_16

        pos = 0
        # First semicircle (8 left curves)
        for _ in range(8):
            x[pos] = R40_LEFT
            pos += 1
        # Straight section
        for _ in range(n_straights):
            x[pos] = STRAIGHT_16
            pos += 1
        # Second semicircle
        for _ in range(8):
            x[pos] = R40_LEFT
            pos += 1
        # Straight section back
        for _ in range(n_straights):
            x[pos] = STRAIGHT_16
            pos += 1

        return x

    def sample_oval_with_siding_opportunity(self, n_straights: int = 6) -> NDArray:
        """Sample oval with enough straights for potential siding.

        Places more straights to create opportunities for switches.
        The GA can then evolve switches into these positions.
        Both straight sections must be equal length for closure.
        """
        x = np.full(U_N_VAR, INACTIVE, dtype=np.float64)

        from .encoding import R40_LEFT, STRAIGHT_16

        pos = 0
        # First curve section (8 curves = 180 degrees)
        for _ in range(8):
            x[pos] = R40_LEFT
            pos += 1
        # Long straight section (siding opportunity)
        for _ in range(n_straights):
            x[pos] = STRAIGHT_16
            pos += 1
        # Second curve section (8 curves = 180 degrees)
        for _ in range(8):
            x[pos] = R40_LEFT
            pos += 1
        # Return straight section (MUST match first for closure)
        for _ in range(n_straights):
            x[pos] = STRAIGHT_16
            pos += 1

        return x

    def sample_figure_8_base(self) -> NDArray:
        """Sample base for figure-8 (two circles meeting at a point)."""
        x = np.full(U_N_VAR, INACTIVE, dtype=np.float64)

        from .encoding import R40_LEFT, R40_RIGHT

        pos = 0
        # First circle (16 left curves)
        for _ in range(16):
            x[pos] = R40_LEFT
            pos += 1
        # Second circle opposite direction (16 right curves)
        # This creates a figure-8 shape
        for _ in range(16):
            x[pos] = R40_RIGHT
            pos += 1

        return x

    def random_sample(self, rng: np.random.Generator) -> NDArray:
        """Generate random chromosome with valid piece distribution."""
        x = np.full(U_N_VAR, INACTIVE, dtype=np.float64)

        # Build piece pool from inventory
        piece_pool = []
        id_to_idx = self.catalog.id_to_index
        for piece_id, count in self.inventory.items():
            if piece_id in id_to_idx:
                piece_pool.extend([id_to_idx[piece_id]] * count)

        if not piece_pool:
            return x

        # Shuffle and place
        rng.shuffle(piece_pool)
        n = min(len(piece_pool), U_MAX)
        x[:n] = piece_pool[:n]

        return x
