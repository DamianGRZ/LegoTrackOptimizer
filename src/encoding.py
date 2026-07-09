"""Partitioned chromosome encoding for track layout optimization.

A fixed-length int16 vector partitioned into six contiguous segments
(see PartitionedDimensions):

  [main-loop types | main-loop flips | siding junctions | cross-junctions |
   double-crossovers | start position]

All segment sizes scale dynamically from inventory/config — n_var is never
hardcoded. Switches, crossings and double-crossovers enter via their descriptor
blocks (not as main-loop alleles); branch content is template-determined for
closure by construction.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

INACTIVE = -1
GENES_PER_JUNCTION = 4  # passing siding: (active, position, handedness, n_straights)
GENES_PER_CROSS_JUNCTION = 3  # cross junction: (active, pos_1, pos_2)
GENES_PER_DBL_CROSSOVER = 5  # double crossover: (active, pos_1, route_1, pos_2, route_2)

# Passing-siding junction gene offsets within a 4-gene descriptor
JUNC_ACTIVE = 0
JUNC_POSITION = 1
JUNC_HANDEDNESS = 2
JUNC_N_STRAIGHTS = 3

# Cross-junction descriptor gene offsets. Each active descriptor models ONE
# physical CROSS_90 traversed twice by the loop (both passes straight, the two
# passes intersecting at ~90 deg so all four ports are used). pos_1 and pos_2
# are main-loop positions of the two traversals.
CJ_ACTIVE = 0
CJ_POSITION_1 = 1
CJ_POSITION_2 = 2

# Double-crossover descriptor gene offsets. Each active descriptor models ONE
# physical DOUBLE_CROSSOVER traversed twice by the loop, with disjoint port
# pairs so all four ports are used (no dangling). pos_1 and pos_2 are main-loop
# positions; route_1/route_2 are catalog-route indices (0..3) per yaml order:
#   0 = track1_through (A->B)
#   1 = track2_through (C->D)
#   2 = cross_1_to_2   (A->D)
#   3 = cross_2_to_1   (C->B)
DC_ACTIVE = 0
DC_POSITION_1 = 1
DC_ROUTE_1 = 2
DC_POSITION_2 = 3
DC_ROUTE_2 = 4

# Catalog route indices for DOUBLE_CROSSOVER (yaml order in track_pieces_v2.yaml)
DC_ROUTE_TRACK1_THROUGH = 0
DC_ROUTE_TRACK2_THROUGH = 1
DC_ROUTE_CROSS_1_TO_2 = 2
DC_ROUTE_CROSS_2_TO_1 = 3
DC_N_ROUTES = 4


# =============================================================================
# Piece Index Constants — must match _CANONICAL_PIECE_INDEX in src/catalog/catalog.py
# =============================================================================

class PieceIndex(IntEnum):
    """Track piece indices from catalog.

    R40 is ONE physical piece; handedness is selected per placement via the
    chromosome's parallel flip array (flip=0 → LEFT +22.5°, flip=1 → RIGHT -22.5°).
    Switches keep separate LEFT/RIGHT entries — their port/route geometry is
    structurally different (not a simple mirror).
    """
    STRAIGHT_16 = 0
    STRAIGHT_24 = 1
    R40_CURVE = 2
    CROSS_90 = 3
    SWITCH_LEFT = 4
    SWITCH_RIGHT = 5
    DOUBLE_CROSSOVER = 6


# Convenience aliases
STRAIGHT_16 = PieceIndex.STRAIGHT_16
STRAIGHT_24 = PieceIndex.STRAIGHT_24
R40_CURVE = PieceIndex.R40_CURVE
CROSS_90 = PieceIndex.CROSS_90
SWITCH_LEFT = PieceIndex.SWITCH_LEFT
SWITCH_RIGHT = PieceIndex.SWITCH_RIGHT
DOUBLE_CROSSOVER = PieceIndex.DOUBLE_CROSSOVER

# Piece categories
SWITCH_INDICES = {SWITCH_LEFT, SWITCH_RIGHT}

# Main loop piece types: simple pieces only (switches/crossings enter via descriptors).
MAIN_LOOP_PIECE_INDICES = {STRAIGHT_16, STRAIGHT_24, R40_CURVE}
MAX_MAIN_LOOP_PIECE = max(MAIN_LOOP_PIECE_INDICES)


# =============================================================================
# Chromosome Dimensions (Dynamic, From Inventory)
# =============================================================================

@dataclass(frozen=True)
class PartitionedDimensions:
    """Chromosome layout dimensions, computed from inventory at runtime.

    Segments:
        [0, n_main)                                    — main loop piece types
        [n_main, 2*n_main)                             — main loop flip bits (per slot)
        [flip_end, junc_end)                           — passing-siding descriptors (J × 4)
        [junc_end, cross_junc_end)                     — cross-junction descriptors (K × 3)
        [cross_junc_end, dbl_crossover_end)            — double-crossover descriptors (D × 5)
        [dbl_crossover_end, dbl_crossover_end + 2)     — start position (x, y)

    Flip bits ∈ {0, 1}: 0 = catalog default direction (LEFT for R40_CURVE),
    1 = mirrored (negate dy, dtheta). For non-R40 slots the flip is ignored at
    decode time but kept in {0, 1} so chromosome bounds stay simple.
    """
    n_main: int              # Max main loop pieces (non-switch inventory)
    max_junctions: int       # Max passing-siding junction slots
    max_cross_junctions: int  # Max cross-junction slots (from CROSS_90 inventory)
    max_double_crossovers: int  # Max DOUBLE_CROSSOVER pieces (from inventory)
    n_straights_16: int      # STRAIGHT_16 pieces in inventory
    n_straights_24: int      # STRAIGHT_24 pieces in inventory
    boundary_min_x: float
    boundary_max_x: float
    boundary_min_y: float
    boundary_max_y: float

    @property
    def total_straights(self) -> int:
        """Combined straight inventory (both lengths)."""
        return self.n_straights_16 + self.n_straights_24

    @property
    def main_flips_start(self) -> int:
        return self.n_main

    @property
    def main_flips_end(self) -> int:
        return 2 * self.n_main

    @property
    def junc_start(self) -> int:
        return self.main_flips_end

    @property
    def junc_end(self) -> int:
        return self.junc_start + self.max_junctions * GENES_PER_JUNCTION

    @property
    def cross_junc_start(self) -> int:
        return self.junc_end

    @property
    def cross_junc_end(self) -> int:
        return self.cross_junc_start + self.max_cross_junctions * GENES_PER_CROSS_JUNCTION

    @property
    def dbl_crossover_start(self) -> int:
        return self.cross_junc_end

    @property
    def dbl_crossover_end(self) -> int:
        return self.dbl_crossover_start + self.max_double_crossovers * GENES_PER_DBL_CROSSOVER

    @property
    def start_pos_start(self) -> int:
        return self.dbl_crossover_end

    @property
    def n_var(self) -> int:
        return self.dbl_crossover_end + 2


def compute_dimensions(config, catalog) -> PartitionedDimensions:
    """Compute chromosome dimensions from inventory configuration.

    Args:
        config: OptimizationConfig with inventory and boundary.
        catalog: TrackCatalog for piece_index lookup.

    Returns:
        PartitionedDimensions with all segment sizes derived from inventory.
    """
    inv = config.inventory

    # Main loop: all non-switch pieces
    n_main = 0
    for piece_id, count in inv.items():
        idx = catalog.id_to_index.get(piece_id)
        if idx is not None and idx not in SWITCH_INDICES:
            n_main += count
    n_main = max(1, n_main)

    # Each passing siding is opposite-handed: consumes 1 LEFT + 1 RIGHT switch.
    # So available passing-siding junctions = min of the two counts.
    left_count = inv.get("R40_SWITCH_LEFT", 0)
    right_count = inv.get("R40_SWITCH_RIGHT", 0)
    max_junctions = min(left_count, right_count)

    # Cross-junctions: each descriptor is one physical CROSS_90 traversed twice
    # by the loop (both passes straight, intersecting at ~90 deg). One descriptor
    # per physical CROSS_90, so the slot count is the CROSS_90 inventory directly.
    max_cross_junctions = inv.get("CROSS_90", 0)

    # One descriptor per physical DOUBLE_CROSSOVER. Each descriptor occupies
    # TWO main-loop positions (the two complementary traversals of the same
    # physical piece). The decoder enforces no-dangling-ports by validating
    # the descriptor's two routes union to {A,B,C,D}.
    max_double_crossovers = inv.get("DOUBLE_CROSSOVER", 0)

    return PartitionedDimensions(
        n_main=n_main,
        max_junctions=max_junctions,
        max_cross_junctions=max_cross_junctions,
        max_double_crossovers=max_double_crossovers,
        n_straights_16=inv.get("STRAIGHT_16", 0),
        n_straights_24=inv.get("STRAIGHT_24", 0),
        boundary_min_x=config.boundary.min_x,
        boundary_max_x=config.boundary.max_x,
        boundary_min_y=config.boundary.min_y,
        boundary_max_y=config.boundary.max_y,
    )


def chromosome_csv_header(dims: PartitionedDimensions) -> str:
    """Segment-aware column names for chromosomes.csv, one per gene.

    Order mirrors the chromosome partition exactly, so the header length
    always equals ``dims.n_var``.
    """
    names = [f"main_{i}" for i in range(dims.n_main)]
    names += [f"flip_{i}" for i in range(dims.n_main)]
    for k in range(dims.max_junctions):
        names += [f"junc{k}_active", f"junc{k}_pos", f"junc{k}_hand",
                  f"junc{k}_nstr"]
    for k in range(dims.max_cross_junctions):
        names += [f"cross{k}_active", f"cross{k}_p1", f"cross{k}_p2"]
    for k in range(dims.max_double_crossovers):
        names += [f"dc{k}_active", f"dc{k}_p1", f"dc{k}_r1",
                  f"dc{k}_p2", f"dc{k}_r2"]
    names += ["start_x", "start_y"]
    return ",".join(names)


# =============================================================================
# Bounds Generation
# =============================================================================

def generate_bounds(dims: PartitionedDimensions) -> Tuple[NDArray, NDArray]:
    """Generate per-gene lower and upper bounds.

    Returns:
        Tuple of (xl, xu) integer bound arrays of length n_var.
    """
    xl = np.full(dims.n_var, INACTIVE, dtype=np.int16)
    xu = np.full(dims.n_var, INACTIVE, dtype=np.int16)

    xl[:dims.n_main] = INACTIVE
    xu[:dims.n_main] = MAX_MAIN_LOOP_PIECE

    # Main loop flips: [0, 1] — irrelevant for symmetric pieces but kept legal
    xl[dims.main_flips_start:dims.main_flips_end] = 0
    xu[dims.main_flips_start:dims.main_flips_end] = 1

    # Junction descriptors (passing siding)
    for k in range(dims.max_junctions):
        base = dims.junc_start + k * GENES_PER_JUNCTION
        xl[base + JUNC_ACTIVE] = 0
        xu[base + JUNC_ACTIVE] = 1
        xl[base + JUNC_POSITION] = 0
        xu[base + JUNC_POSITION] = dims.n_main - 1
        xl[base + JUNC_HANDEDNESS] = 0
        xu[base + JUNC_HANDEDNESS] = 1  # LEFT, RIGHT
        xl[base + JUNC_N_STRAIGHTS] = 0
        xu[base + JUNC_N_STRAIGHTS] = dims.total_straights

    # Cross-junction descriptors: one physical CROSS_90, two main-loop traversal
    # positions whose FK states must coincide perpendicular (validated in decoder).
    for k in range(dims.max_cross_junctions):
        base = dims.cross_junc_start + k * GENES_PER_CROSS_JUNCTION
        xl[base + CJ_ACTIVE] = 0
        xu[base + CJ_ACTIVE] = 1
        xl[base + CJ_POSITION_1] = 0
        xu[base + CJ_POSITION_1] = dims.n_main - 1
        xl[base + CJ_POSITION_2] = 0
        xu[base + CJ_POSITION_2] = dims.n_main - 1

    # Double-crossover descriptors: one per physical piece, two main-loop
    # traversal positions + their catalog-route choices.
    for k in range(dims.max_double_crossovers):
        base = dims.dbl_crossover_start + k * GENES_PER_DBL_CROSSOVER
        xl[base + DC_ACTIVE] = 0
        xu[base + DC_ACTIVE] = 1
        xl[base + DC_POSITION_1] = 0
        xu[base + DC_POSITION_1] = dims.n_main - 1
        xl[base + DC_ROUTE_1] = 0
        xu[base + DC_ROUTE_1] = DC_N_ROUTES - 1
        xl[base + DC_POSITION_2] = 0
        xu[base + DC_POSITION_2] = dims.n_main - 1
        xl[base + DC_ROUTE_2] = 0
        xu[base + DC_ROUTE_2] = DC_N_ROUTES - 1

    # Start position
    xl[dims.start_pos_start] = int(dims.boundary_min_x)
    xu[dims.start_pos_start] = int(dims.boundary_max_x)
    xl[dims.start_pos_start + 1] = int(dims.boundary_min_y)
    xu[dims.start_pos_start + 1] = int(dims.boundary_max_y)

    return xl, xu


# =============================================================================
# Main Loop Gene Access
# =============================================================================

def get_main_loop_types(x: NDArray, dims: PartitionedDimensions) -> NDArray:
    """Read main loop piece types as array. Values: -1 (inactive) or 0-2."""
    return x[:dims.n_main].copy()


def set_main_loop_type(x: NDArray, dims: PartitionedDimensions,
                       pos: int, piece_type: int) -> None:
    """Set a single main loop piece type (in-place)."""
    x[pos] = piece_type


def get_active_main_pieces(x: NDArray, dims: PartitionedDimensions) -> NDArray:
    """Get main loop piece types, filtering out INACTIVE."""
    types = x[:dims.n_main]
    return types[types != INACTIVE].copy()


def get_main_loop_flips(x: NDArray, dims: PartitionedDimensions) -> NDArray:
    """Per-slot flip bits for the main loop."""
    return x[dims.main_flips_start:dims.main_flips_end].copy()


def set_flip(x: NDArray, dims: PartitionedDimensions, pos: int, value: int) -> None:
    """Write flip bit for main-loop slot ``pos`` (clamped to {0, 1})."""
    x[dims.main_flips_start + pos] = 1 if value else 0


def get_active_main_pieces_with_flips(
    x: NDArray, dims: PartitionedDimensions,
) -> Tuple[NDArray, NDArray]:
    """Return (active_types, active_flips) for slots where type != INACTIVE."""
    types = x[:dims.n_main]
    flips = x[dims.main_flips_start:dims.main_flips_end]
    mask = types != INACTIVE
    return types[mask].copy(), flips[mask].copy()


def fk_rows_with_flips(fk_table: NDArray, types, flips=None) -> NDArray:
    """FK rows for ``types`` with R40 handedness applied.

    The catalog row for R40_CURVE encodes the LEFT turn; a set flip bit
    mirrors it (negates dy and dtheta). Other piece types ignore their flip
    bit. ``types`` must be valid catalog indices. Fancy indexing copies, so
    the returned array is safe to mutate. ``flips=None`` skips mirroring.
    """
    types = np.asarray(types, dtype=np.int32)
    deltas = fk_table[types]
    if flips is None:
        return deltas
    # flip=1 mirrors the stored LEFT turn into RIGHT about the forward axis:
    # negate dy and dtheta, keep dx.
    negate = (types == int(R40_CURVE)) & (np.asarray(flips, dtype=np.int32) == 1)
    deltas[negate, 1] *= -1.0  # dy: mirror lateral offset
    deltas[negate, 2] *= -1.0  # dtheta: mirror heading change
    return deltas


# =============================================================================
# Junction Gene Access
# =============================================================================

def get_junction(x: NDArray, dims: PartitionedDimensions,
                 slot: int) -> Tuple[int, int, int, int]:
    """Read a junction descriptor.

    Returns:
        (active, position, handedness, n_straights) tuple.
    """
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    return (
        int(x[base + JUNC_ACTIVE]),
        int(x[base + JUNC_POSITION]),
        int(x[base + JUNC_HANDEDNESS]),
        int(x[base + JUNC_N_STRAIGHTS]),
    )


def set_junction(x: NDArray, dims: PartitionedDimensions, slot: int,
                 active: int, position: int, handedness: int,
                 n_straights: int) -> None:
    """Write a junction descriptor (in-place)."""
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + JUNC_ACTIVE] = active
    x[base + JUNC_POSITION] = position
    x[base + JUNC_HANDEDNESS] = handedness
    x[base + JUNC_N_STRAIGHTS] = n_straights


def get_active_junctions(
    x: NDArray, dims: PartitionedDimensions,
) -> List[Tuple[int, int, int, int, int]]:
    """Get active junction descriptors, sorted by position.

    Returns:
        List of (slot, active, position, handedness, n_straights) tuples,
        sorted by position ascending.
    """
    junctions = []
    for k in range(dims.max_junctions):
        active, position, handedness, n_straights = get_junction(x, dims, k)
        if active:
            junctions.append((k, active, position, handedness, n_straights))
    junctions.sort(key=lambda j: j[2])  # Sort by position
    return junctions


# =============================================================================
# Cross-Junction Gene Access
# =============================================================================

def get_cross_junction(x: NDArray, dims: PartitionedDimensions,
                       slot: int) -> Tuple[int, int, int]:
    """Read a cross-junction descriptor.

    Returns:
        (active, pos_1, pos_2) tuple.
    """
    base = dims.cross_junc_start + slot * GENES_PER_CROSS_JUNCTION
    return (
        int(x[base + CJ_ACTIVE]),
        int(x[base + CJ_POSITION_1]),
        int(x[base + CJ_POSITION_2]),
    )


def set_cross_junction(x: NDArray, dims: PartitionedDimensions, slot: int,
                       active: int, pos_1: int, pos_2: int) -> None:
    """Write a cross-junction descriptor (in-place)."""
    base = dims.cross_junc_start + slot * GENES_PER_CROSS_JUNCTION
    x[base + CJ_ACTIVE] = active
    x[base + CJ_POSITION_1] = pos_1
    x[base + CJ_POSITION_2] = pos_2


def get_active_cross_junctions(
    x: NDArray, dims: PartitionedDimensions,
) -> List[Tuple[int, int, int, int]]:
    """Get active cross-junction descriptors, sorted by pos_1.

    Returns:
        List of (slot, active, pos_1, pos_2) tuples.
    """
    junctions = []
    for k in range(dims.max_cross_junctions):
        active, pos_1, pos_2 = get_cross_junction(x, dims, k)
        if active:
            junctions.append((k, active, pos_1, pos_2))
    junctions.sort(key=lambda j: j[2])  # sort by pos_1
    return junctions


# =============================================================================
# Double-Crossover Gene Access
# =============================================================================

def get_double_crossover(
    x: NDArray, dims: PartitionedDimensions, slot: int,
) -> Tuple[int, int, int, int, int]:
    """Read a double-crossover descriptor.

    Returns:
        (active, pos_1, route_1, pos_2, route_2) tuple.
    """
    base = dims.dbl_crossover_start + slot * GENES_PER_DBL_CROSSOVER
    return (
        int(x[base + DC_ACTIVE]),
        int(x[base + DC_POSITION_1]),
        int(x[base + DC_ROUTE_1]),
        int(x[base + DC_POSITION_2]),
        int(x[base + DC_ROUTE_2]),
    )


def set_double_crossover(
    x: NDArray, dims: PartitionedDimensions, slot: int,
    active: int, pos_1: int, route_1: int, pos_2: int, route_2: int,
) -> None:
    """Write a double-crossover descriptor (in-place)."""
    base = dims.dbl_crossover_start + slot * GENES_PER_DBL_CROSSOVER
    x[base + DC_ACTIVE] = active
    x[base + DC_POSITION_1] = pos_1
    x[base + DC_ROUTE_1] = route_1
    x[base + DC_POSITION_2] = pos_2
    x[base + DC_ROUTE_2] = route_2


def get_active_double_crossovers(
    x: NDArray, dims: PartitionedDimensions,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Get active DBL_CROSSOVER descriptors, sorted by pos_1.

    Returns:
        List of (slot, active, pos_1, route_1, pos_2, route_2) tuples.
    """
    out = []
    for k in range(dims.max_double_crossovers):
        active, pos_1, route_1, pos_2, route_2 = get_double_crossover(x, dims, k)
        if active:
            out.append((k, active, pos_1, route_1, pos_2, route_2))
    out.sort(key=lambda d: d[2])
    return out


# =============================================================================
# Start Position Access
# =============================================================================

def get_start_position(x: NDArray, dims: PartitionedDimensions) -> Tuple[float, float]:
    """Read start position (x, y)."""
    return (float(x[dims.start_pos_start]), float(x[dims.start_pos_start + 1]))


# =============================================================================
# Chromosome Construction
# =============================================================================

def create_empty_chromosome(dims: PartitionedDimensions) -> NDArray:
    """Create a chromosome with all genes inactive/zero."""
    x = np.full(dims.n_var, INACTIVE, dtype=np.int16)
    # Flip array is part of the main-loop region but holds bits, not types — zero it.
    x[dims.main_flips_start:dims.main_flips_end] = 0
    for k in range(dims.max_junctions):
        set_junction(x, dims, k, active=0, position=0, handedness=0, n_straights=0)
    for k in range(dims.max_cross_junctions):
        set_cross_junction(x, dims, k, active=0, pos_1=0, pos_2=0)
    for k in range(dims.max_double_crossovers):
        set_double_crossover(x, dims, k, active=0,
                             pos_1=0, route_1=0, pos_2=0, route_2=0)
    x[dims.start_pos_start] = 0
    x[dims.start_pos_start + 1] = 0
    return x


def create_chromosome_from_pieces(
    dims: PartitionedDimensions,
    main_loop_pieces: List[int],
    main_loop_flips: Optional[List[int]] = None,
    junctions: Optional[List[Tuple[int, int, int, int]]] = None,
    cross_junctions: Optional[List[Tuple[int, int, int]]] = None,
    double_crossovers: Optional[List[Tuple[int, int, int, int, int]]] = None,
) -> NDArray:
    """Create a chromosome from a piece sequence and optional descriptors.

    Args:
        dims: Partitioned dimensions.
        main_loop_pieces: Piece type indices for the main loop.
        main_loop_flips: Per-slot flip bits parallel to ``main_loop_pieces``.
            Defaults to all-zero (catalog direction).
        junctions: Optional (active, position, handedness, n_straights) tuples
            for passing-siding descriptors.
        cross_junctions: Optional (active, pos_1, pos_2) tuples for
            cross-junction descriptors.
        double_crossovers: Optional (active, pos_1, route_1, pos_2, route_2)
            tuples for DOUBLE_CROSSOVER descriptors.

    Returns:
        Chromosome array.
    """
    x = create_empty_chromosome(dims)
    n = min(len(main_loop_pieces), dims.n_main)
    x[:n] = main_loop_pieces[:n]

    if main_loop_flips is not None:
        m = min(len(main_loop_flips), dims.n_main)
        x[dims.main_flips_start:dims.main_flips_start + m] = main_loop_flips[:m]

    if junctions:
        for k, junc in enumerate(junctions):
            if k >= dims.max_junctions:
                break
            set_junction(x, dims, k, *junc)

    if cross_junctions:
        for k, cj in enumerate(cross_junctions):
            if k >= dims.max_cross_junctions:
                break
            set_cross_junction(x, dims, k, *cj)

    if double_crossovers:
        for k, dc in enumerate(double_crossovers):
            if k >= dims.max_double_crossovers:
                break
            set_double_crossover(x, dims, k, *dc)

    return x


# =============================================================================
# Validation
# =============================================================================

def _range_errors(specs):
    """Yield 'name=value out of [lo, hi]' for each spec whose value is out of range."""
    return (
        f"{name}={value} out of [{lo}, {hi}]"
        for name, value, lo, hi in specs
        if not lo <= value <= hi
    )


def _junction_specs(x, dims):
    """Per-slot junction (name, value, lo, hi) range specs."""
    n_main_hi = dims.n_main - 1
    str_hi = dims.total_straights
    for k in range(dims.max_junctions):
        active, pos, hand, n_str = get_junction(x, dims, k)
        yield (f"junction[{k}].active", active, 0, 1)
        yield (f"junction[{k}].position", pos, 0, n_main_hi)
        yield (f"junction[{k}].handedness", hand, 0, 1)
        yield (f"junction[{k}].n_straights", n_str, 0, str_hi)


def _cross_junction_specs(x, dims):
    """Per-slot cross-junction (name, value, lo, hi) range specs."""
    n_main_hi = dims.n_main - 1
    for k in range(dims.max_cross_junctions):
        active, p1, p2 = get_cross_junction(x, dims, k)
        yield (f"cross_junction[{k}].active", active, 0, 1)
        yield (f"cross_junction[{k}].pos_1", p1, 0, n_main_hi)
        yield (f"cross_junction[{k}].pos_2", p2, 0, n_main_hi)


def _dbl_crossover_specs(x, dims):
    """Per-slot DOUBLE_CROSSOVER (name, value, lo, hi) range specs."""
    n_main_hi = dims.n_main - 1
    route_hi = DC_N_ROUTES - 1
    for k in range(dims.max_double_crossovers):
        active, p1, r1, p2, r2 = get_double_crossover(x, dims, k)
        yield (f"dbl_crossover[{k}].active", active, 0, 1)
        yield (f"dbl_crossover[{k}].pos_1", p1, 0, n_main_hi)
        yield (f"dbl_crossover[{k}].route_1", r1, 0, route_hi)
        yield (f"dbl_crossover[{k}].pos_2", p2, 0, n_main_hi)
        yield (f"dbl_crossover[{k}].route_2", r2, 0, route_hi)


def _main_loop_errors(x, dims):
    types = x[: dims.n_main].astype(int)
    bad = np.where(
        (types != INACTIVE) & ((types < 0) | (types > MAX_MAIN_LOOP_PIECE))
    )[0]
    return (
        f"main[{i}]={int(types[i])} out of [-1, {MAX_MAIN_LOOP_PIECE}]"
        for i in bad
    )


def validate_chromosome(x: NDArray, dims: PartitionedDimensions) -> List[str]:
    """Validate chromosome gene values are within bounds.

    Returns:
        List of error messages (empty if valid).
    """
    if len(x) != dims.n_var:
        return [f"Length {len(x)} != expected {dims.n_var}"]

    return [
        *_main_loop_errors(x, dims),
        *_range_errors(_junction_specs(x, dims)),
        *_range_errors(_cross_junction_specs(x, dims)),
        *_range_errors(_dbl_crossover_specs(x, dims)),
    ]
