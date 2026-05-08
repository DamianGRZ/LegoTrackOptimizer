"""Port-pair chromosome encoding for the LEGO Track Optimizer (v2).

Layout
------
<<<<<<< Updated upstream
    [ piece_slots:  N_max int16 ]
    [ flip_flags:   N_max int16 ]
    [ rotate_flags: N_max int16 ]
    [ port_pairs:   E_max * 4 int16 ]
    [ anchor:       3 int16 ]

- Piece slot: piece_id index from catalog ordering, or -1 (INACTIVE).
- Flip flag (0 or 1): per-slot Y-mirror orientation bit. Honoured only when
  the slot's piece is marked ``symmetric`` in the catalog (e.g. R40_CURVE —
  one SKU, flip selects left vs right turn). For asymmetric pieces the bit
  is forced to 0 by the repair pipeline and ignored by the decoder.
- Rotate flag (0 or 1): per-slot 180° in-plane rotation bit. Honoured only
  when the slot's piece is marked ``rotatable`` (switches: rotate=1 selects
  the OUT placement of the same physical brick — frog at throat-end instead
  of far-end). Forced to 0 by repair on non-rotatable pieces.
=======
    [ piece_slots: N_max int16 ]  [ port_pairs: E_max * 4 int16 ]  [ anchor: 3 int16 ]

- Piece slot: piece_id index from catalog ordering, or -1 (INACTIVE).
>>>>>>> Stashed changes
- Port-pair row: (slot_a, port_idx_a, slot_b, port_idx_b), or all -1 (INACTIVE).
- Anchor: (start_x, start_y, start_theta_deg) as int16.

Slot indices are stable positional references (no compaction). Port indices
follow tuple(spec.ports) order: A=0, B=1, C=2, D=3.

Sentinel -1 marks inactive piece slots and inactive port-pair rows. The decoder
counts sentinels at decode time; there is no separate active-count gene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

INACTIVE: int = -1
"""Sentinel for inactive piece slots and inactive port-pair rows."""

<<<<<<< Updated upstream
ENCODING_VERSION: int = 3
"""Bumps whenever the chromosome layout, decoder semantics, or persisted
artifact format changes incompatibly with prior versions. History:

- 1 -> 2 (Phase 3): anchor reinterpreted as +/-5% offset; decoder auto-centers.
- 2 -> 3 (Phase 4): chromosome gains a junction segment of 5*J_max genes
  inserted between port-pair edges and anchor.

Per Rule 13, every persisted artifact stamps this and refuses to load
mismatched files (see :class:`EpsilonArchive.from_json`)."""

ANCHOR_OFFSET_FRACTION: float = 0.05
"""Fraction of boundary width/height that bounds the anchor (off_x, off_y)
genes in Phase 3+. The decoder auto-centers the layout, then translates by
this small offset so the GA can still nudge placement off boundary center."""

GENES_PER_PAIR: int = 4
"""Each port-pair row occupies (slot_a, port_a, slot_b, port_b) = 4 ints."""

JUNCTION_GENES: int = 5
"""Each junction descriptor occupies (active, anchor_slot, kind, param_a, param_b)
= 5 ints. Phase 4 adds the segment with all junctions inactive (decoder ignores
them); Phase 5+ wires templates that consume the descriptors."""

# Junction kind enum -- canonical ints for the chromosome.
JUNCTION_KIND_PASSING_SIDING: int = 0
JUNCTION_KIND_FIGURE_8_CROSS: int = 1
JUNCTION_KIND_PARALLEL_DC_BRIDGE: int = 2
JUNCTION_KIND_MAX: int = 2  # update when new templates land

# Junction gene offsets relative to row base
JUNC_ACTIVE_OFFSET: int = 0
JUNC_ANCHOR_OFFSET: int = 1
JUNC_KIND_OFFSET: int = 2
JUNC_PARAM_A_OFFSET: int = 3
JUNC_PARAM_B_OFFSET: int = 4

# Phase 4: conservative param range; templates own the actual semantics.
# Junction params are int16-stored and bounded so repair / validate can clamp.
JUNCTION_PARAM_MAX: int = 255

=======
GENES_PER_PAIR: int = 4
"""Each port-pair row occupies (slot_a, port_a, slot_b, port_b) = 4 ints."""

>>>>>>> Stashed changes
ANCHOR_GENES: int = 3
"""Anchor occupies (x, y, theta_deg) = 3 ints."""

# Port-pair row gene offsets relative to row base
PAIR_SLOT_A_OFFSET: int = 0
PAIR_PORT_A_OFFSET: int = 1
PAIR_SLOT_B_OFFSET: int = 2
PAIR_PORT_B_OFFSET: int = 3

# Anchor gene offsets relative to anchor_start
ANCHOR_X_OFFSET: int = 0
ANCHOR_Y_OFFSET: int = 1
ANCHOR_THETA_OFFSET: int = 2

DEFAULT_EDGE_MARGIN_STUDS: float = 20.0
DEFAULT_BRANCH_OVERLAY_FACTOR: float = 1.5
DEFAULT_FALLBACK_AXIAL: float = 16.0

DTYPE = np.int16
"""Locked-in storage dtype for the chromosome (decided in Phase 0)."""

MAX_PORT_IDX: int = 3
"""Catalog ports per piece: max 4 (CROSS_90, DOUBLE_CROSSOVER); indices 0..3."""


# =============================================================================
# Dimensions
# =============================================================================

@dataclass(frozen=True)
class PortPairDimensions:
<<<<<<< Updated upstream
    """Chromosome region offsets and capacities.

    Layout (in gene index order):
        [slot_start,   slot_end)    piece-id per slot         (length N_max)
        [flip_start,   flip_end)    per-slot flip bit         (length N_max)
        [rotate_start, rotate_end)  per-slot rotate bit       (length N_max)
        [pair_start,   pair_end)    port-pair rows            (length 4*E_max)
        [junc_start,   junc_end)    junction descriptors      (length 5*J_max)
        [anchor_start, n_var)       anchor x,y,theta_deg      (length 3)

    ``J_max == 0`` collapses the junction segment to a zero-length slice;
    decoder/operators traverse it as a no-op without special-casing.
    """

    N_max: int
    E_max: int
    J_max: int = 0

    def __post_init__(self) -> None:
        # Rule 2: int16 overflow guard. ``n_var`` indexes int16 arrays, and
        # very large inventories combined with a J_max > 0 push n_var toward
        # the 32767 ceiling. Fail loud rather than silently truncate.
        int16_max = int(np.iinfo(DTYPE).max)
        if self.n_var >= int16_max:
            raise ValueError(
                f"PortPairDimensions n_var={self.n_var} exceeds int16 max "
                f"({int16_max}); reduce N_max/E_max/J_max."
            )
=======
    """Chromosome region offsets and capacities."""

    N_max: int
    E_max: int
>>>>>>> Stashed changes

    @property
    def slot_start(self) -> int:
        return 0

    @property
    def slot_end(self) -> int:
        return self.N_max

    @property
<<<<<<< Updated upstream
    def flip_start(self) -> int:
        return self.slot_end

    @property
    def flip_end(self) -> int:
        return self.flip_start + self.N_max

    @property
    def rotate_start(self) -> int:
        return self.flip_end

    @property
    def rotate_end(self) -> int:
        return self.rotate_start + self.N_max

    @property
    def pair_start(self) -> int:
        return self.rotate_end

    @property
    def pair_end(self) -> int:
        return self.pair_start + GENES_PER_PAIR * self.E_max

    @property
    def junc_start(self) -> int:
        return self.pair_end

    @property
    def junc_end(self) -> int:
        return self.junc_start + JUNCTION_GENES * self.J_max

    @property
    def anchor_start(self) -> int:
        return self.junc_end
=======
    def pair_start(self) -> int:
        return self.N_max

    @property
    def pair_end(self) -> int:
        return self.N_max + GENES_PER_PAIR * self.E_max

    @property
    def anchor_start(self) -> int:
        return self.pair_end
>>>>>>> Stashed changes

    @property
    def n_var(self) -> int:
        return self.anchor_start + ANCHOR_GENES


# =============================================================================
# Catalog adapters (V1 TrackCatalog and V2 TrackCatalogSpec)
# =============================================================================

def _iter_pieces(catalog) -> Iterator:
    """Yield piece records from either V1 or V2 catalog."""
    if hasattr(catalog, "pieces") and not hasattr(catalog, "_pieces"):
        # V2 TrackCatalogSpec
        yield from catalog.pieces
    else:
        # V1 TrackCatalog
        for piece_id in catalog.id_to_index:
            piece = catalog._pieces.get(piece_id)
            if piece is not None:
                yield piece


def _piece_by_id(catalog, piece_id: str):
    """Look up a piece by id from either V1 or V2 catalog."""
    if hasattr(catalog, "by_id"):
        return catalog.by_id.get(piece_id)
    return catalog._pieces.get(piece_id)


def _axial_extent(piece) -> float:
    """Forward extent of a piece for path-density calculation.

    Handles both V1 TrackPiece (has ``arc_length`` property) and V2
    TrackPieceSpec (has ``length_studs`` / ``body_length_studs`` /
    ``radius_studs`` + ``sector_angle_rad``).
    """
    arc_attr = getattr(piece, "arc_length", None)
    if arc_attr is not None:
        return float(arc_attr)

    length_studs = getattr(piece, "length_studs", None)
    if length_studs is not None:
        return float(length_studs)

    body = getattr(piece, "body_length_studs", None)
    if body is not None:
        return float(body)

    radius = getattr(piece, "radius_studs", None)
    sector = getattr(piece, "sector_angle_rad", None)
    if radius is not None and sector is not None:
        return float(radius * abs(sector))

    return DEFAULT_FALLBACK_AXIAL


# =============================================================================
# Dimension formula
# =============================================================================

def compute_port_pair_dimensions(
    boundary,
    catalog,
    inventory: Optional[Dict[str, int]] = None,
    *,
    edge_margin_studs: float = DEFAULT_EDGE_MARGIN_STUDS,
    branch_overlay_factor: float = DEFAULT_BRANCH_OVERLAY_FACTOR,
) -> PortPairDimensions:
    """Compute N_max and E_max from boundary, catalog, and (optional) inventory.

    Both caps derive from physical reality:

    - ``geometric_cap``: pieces fitting in the box, computed as
      ``perimeter * branch_overlay_factor / min_axial_extent``.
    - ``inventory_cap``: physical pieces owned (when ``inventory`` is supplied).

    The smaller of the two binds. The chromosome length is
    ``n_var = N_max + 4 * E_max + 3``.
    """
    avail_w = max(0.0, boundary.width - 2 * edge_margin_studs)
    avail_h = max(0.0, boundary.height - 2 * edge_margin_studs)
    max_path_length = 2.0 * (avail_w + avail_h) * branch_overlay_factor

    pieces = list(_iter_pieces(catalog))
    if not pieces:
        raise ValueError("Catalog has no registered pieces")

    axial_values = [_axial_extent(p) for p in pieces]
    axial_values = [a for a in axial_values if a > 0]
    if not axial_values:
        raise ValueError("Catalog pieces have no positive axial extent")
    min_axial = min(axial_values)

    geometric_cap = max(1, int(max_path_length / min_axial))

    if inventory:
        inventory_cap = sum(inventory.values())
    else:
        inventory_cap = geometric_cap  # geometry binds when no inventory supplied

    N_max = max(1, min(geometric_cap, int(inventory_cap)))

    if inventory:
        total_ports = 0
        for piece_id, count in inventory.items():
            piece = _piece_by_id(catalog, piece_id)
            if piece is not None:
                total_ports += len(piece.ports) * count
    else:
        avg_ports = sum(len(p.ports) for p in pieces) / len(pieces)
        total_ports = int(N_max * avg_ports)

    E_max = max(1, total_ports // 2)

<<<<<<< Updated upstream
    J_max = _compute_junction_capacity(inventory or {})

    return PortPairDimensions(N_max=N_max, E_max=E_max, J_max=J_max)


def _compute_junction_capacity(inventory: Dict[str, int]) -> int:
    """Conservative upper bound on simultaneously-active junction descriptors.

    Each PASSING_SIDING uses 2 switches (1 LEFT + 1 RIGHT). Each FIGURE_8_CROSS
    uses 1 CROSS_90. PARALLEL_DC_BRIDGE uses 2 CROSS_90 + 4 switches but every
    piece can only be in ONE junction, so the worst-case count is bounded by
    ``floor(switches/2) + crossings`` -- mixing the cheap-per-junction
    templates. Phase 4 stores all junctions inactive so this only sizes the
    chromosome; later phases consume from real inventory.
    """
    switches = (
        inventory.get("R40_SWITCH_LEFT", 0)
        + inventory.get("R40_SWITCH_RIGHT", 0)
    )
    crossings = (
        inventory.get("CROSS_90", 0)
        + inventory.get("DOUBLE_CROSSOVER", 0)
    )
    return (switches // 2) + crossings
=======
    return PortPairDimensions(N_max=N_max, E_max=E_max)
>>>>>>> Stashed changes


# =============================================================================
# Bounds
# =============================================================================

def generate_bounds(
    dims: PortPairDimensions,
    boundary,
    *,
    max_piece_id: int,
    max_port_idx: int = MAX_PORT_IDX,
) -> Tuple[NDArray, NDArray]:
    """Generate per-gene lower and upper bounds (xl, xu) as int16 arrays.

    Args:
        dims: Chromosome dimensions.
        boundary: Object with ``min_x``, ``max_x``, ``min_y``, ``max_y`` attrs.
        max_piece_id: Largest valid piece_id (catalog n_pieces - 1).
        max_port_idx: Largest valid port index (default 3 for 4-port pieces).
    """
    xl = np.full(dims.n_var, INACTIVE, dtype=DTYPE)
    xu = np.full(dims.n_var, INACTIVE, dtype=DTYPE)

    # Piece slots: [-1, max_piece_id]
    xl[dims.slot_start:dims.slot_end] = INACTIVE
    xu[dims.slot_start:dims.slot_end] = max_piece_id

<<<<<<< Updated upstream
    # Flip flags: 0 or 1 (binary). Inactive slots ignore their flip; we still
    # bound to [0,1] so the GA never produces nonsense values.
    xl[dims.flip_start:dims.flip_end] = 0
    xu[dims.flip_start:dims.flip_end] = 1

    # Rotate flags: 0 or 1 (binary). Same treatment as flip.
    xl[dims.rotate_start:dims.rotate_end] = 0
    xu[dims.rotate_start:dims.rotate_end] = 1

=======
>>>>>>> Stashed changes
    # Port-pair rows
    for k in range(dims.E_max):
        base = dims.pair_start + k * GENES_PER_PAIR
        xl[base + PAIR_SLOT_A_OFFSET] = INACTIVE
        xu[base + PAIR_SLOT_A_OFFSET] = max(0, dims.N_max - 1)
        xl[base + PAIR_PORT_A_OFFSET] = INACTIVE
        xu[base + PAIR_PORT_A_OFFSET] = max_port_idx
        xl[base + PAIR_SLOT_B_OFFSET] = INACTIVE
        xu[base + PAIR_SLOT_B_OFFSET] = max(0, dims.N_max - 1)
        xl[base + PAIR_PORT_B_OFFSET] = INACTIVE
        xu[base + PAIR_PORT_B_OFFSET] = max_port_idx

<<<<<<< Updated upstream
    # Junction segment (Phase 4): one descriptor per junction slot, 5 genes each.
    for j in range(dims.J_max):
        base = dims.junc_start + j * JUNCTION_GENES
        xl[base + JUNC_ACTIVE_OFFSET] = 0
        xu[base + JUNC_ACTIVE_OFFSET] = 1
        xl[base + JUNC_ANCHOR_OFFSET] = 0
        xu[base + JUNC_ANCHOR_OFFSET] = max(0, dims.N_max - 1)
        xl[base + JUNC_KIND_OFFSET] = 0
        xu[base + JUNC_KIND_OFFSET] = JUNCTION_KIND_MAX
        xl[base + JUNC_PARAM_A_OFFSET] = 0
        xu[base + JUNC_PARAM_A_OFFSET] = JUNCTION_PARAM_MAX
        xl[base + JUNC_PARAM_B_OFFSET] = 0
        xu[base + JUNC_PARAM_B_OFFSET] = JUNCTION_PARAM_MAX

    # Anchor (Phase 3+): x and y are small +/-ANCHOR_OFFSET_FRACTION offsets
    # around the auto-centered layout, NOT absolute world positions.
    off_x = max(1, int(boundary.width * ANCHOR_OFFSET_FRACTION))
    off_y = max(1, int(boundary.height * ANCHOR_OFFSET_FRACTION))
    xl[dims.anchor_start + ANCHOR_X_OFFSET] = -off_x
    xu[dims.anchor_start + ANCHOR_X_OFFSET] = off_x
    xl[dims.anchor_start + ANCHOR_Y_OFFSET] = -off_y
    xu[dims.anchor_start + ANCHOR_Y_OFFSET] = off_y
=======
    # Anchor: x and y in boundary range; theta in [0, 359]
    xl[dims.anchor_start + ANCHOR_X_OFFSET] = int(boundary.min_x)
    xu[dims.anchor_start + ANCHOR_X_OFFSET] = int(boundary.max_x)
    xl[dims.anchor_start + ANCHOR_Y_OFFSET] = int(boundary.min_y)
    xu[dims.anchor_start + ANCHOR_Y_OFFSET] = int(boundary.max_y)
>>>>>>> Stashed changes
    xl[dims.anchor_start + ANCHOR_THETA_OFFSET] = 0
    xu[dims.anchor_start + ANCHOR_THETA_OFFSET] = 359

    return xl, xu


# =============================================================================
# Piece-slot accessors
# =============================================================================

def get_piece_slot(x: NDArray, dims: PortPairDimensions, slot_idx: int) -> int:
    """Read piece_id at a slot, or -1 if inactive."""
    return int(x[dims.slot_start + slot_idx])


def set_piece_slot(
    x: NDArray, dims: PortPairDimensions, slot_idx: int, piece_id: int,
) -> None:
    """Write piece_id at a slot."""
    x[dims.slot_start + slot_idx] = piece_id


def iter_active_slots(
    x: NDArray, dims: PortPairDimensions,
) -> Iterator[Tuple[int, int]]:
    """Yield (slot_idx, piece_id) for each non-INACTIVE slot."""
    for k in range(dims.N_max):
        v = int(x[dims.slot_start + k])
        if v != INACTIVE:
            yield k, v


# =============================================================================
<<<<<<< Updated upstream
# Flip-bit accessors
# =============================================================================

def get_slot_flip(x: NDArray, dims: PortPairDimensions, slot_idx: int) -> int:
    """Read flip bit at a slot. Returns 0 or 1."""
    return int(x[dims.flip_start + slot_idx]) & 1


def set_slot_flip(
    x: NDArray, dims: PortPairDimensions, slot_idx: int, flip: int,
) -> None:
    """Write flip bit at a slot. Coerces non-{0,1} values to 0."""
    x[dims.flip_start + slot_idx] = 1 if flip else 0


def iter_active_flips(
    x: NDArray, dims: PortPairDimensions,
) -> Iterator[Tuple[int, int]]:
    """Yield (slot_idx, flip_bit) for each non-INACTIVE slot."""
    for slot_idx, _ in iter_active_slots(x, dims):
        yield slot_idx, get_slot_flip(x, dims, slot_idx)


def get_slot_rotate(x: NDArray, dims: PortPairDimensions, slot_idx: int) -> int:
    """Read rotate bit at a slot. Returns 0 or 1."""
    return int(x[dims.rotate_start + slot_idx]) & 1


def set_slot_rotate(
    x: NDArray, dims: PortPairDimensions, slot_idx: int, rotate: int,
) -> None:
    """Write rotate bit at a slot. Coerces non-{0,1} values to 0."""
    x[dims.rotate_start + slot_idx] = 1 if rotate else 0


def iter_active_rotates(
    x: NDArray, dims: PortPairDimensions,
) -> Iterator[Tuple[int, int]]:
    """Yield (slot_idx, rotate_bit) for each non-INACTIVE slot."""
    for slot_idx, _ in iter_active_slots(x, dims):
        yield slot_idx, get_slot_rotate(x, dims, slot_idx)


# =============================================================================
=======
>>>>>>> Stashed changes
# Port-pair accessors
# =============================================================================

def get_port_pair(
    x: NDArray, dims: PortPairDimensions, row: int,
) -> Tuple[int, int, int, int]:
    """Read one port-pair row as (slot_a, port_a, slot_b, port_b)."""
    base = dims.pair_start + row * GENES_PER_PAIR
    return (
        int(x[base + PAIR_SLOT_A_OFFSET]),
        int(x[base + PAIR_PORT_A_OFFSET]),
        int(x[base + PAIR_SLOT_B_OFFSET]),
        int(x[base + PAIR_PORT_B_OFFSET]),
    )


def set_port_pair(
    x: NDArray, dims: PortPairDimensions, row: int,
    slot_a: int, port_a: int, slot_b: int, port_b: int,
) -> None:
    """Write one port-pair row in place."""
    base = dims.pair_start + row * GENES_PER_PAIR
    x[base + PAIR_SLOT_A_OFFSET] = slot_a
    x[base + PAIR_PORT_A_OFFSET] = port_a
    x[base + PAIR_SLOT_B_OFFSET] = slot_b
    x[base + PAIR_PORT_B_OFFSET] = port_b


def clear_port_pair(x: NDArray, dims: PortPairDimensions, row: int) -> None:
    """Mark a port-pair row as inactive (all four genes set to INACTIVE)."""
    set_port_pair(x, dims, row, INACTIVE, INACTIVE, INACTIVE, INACTIVE)


def iter_active_pairs(
    x: NDArray, dims: PortPairDimensions,
) -> Iterator[Tuple[int, int, int, int, int]]:
    """Yield (row, slot_a, port_a, slot_b, port_b) for fully-populated rows.

    A row is considered active iff none of its four genes is INACTIVE.
    Partially-INACTIVE rows are treated as inactive here; ``repair`` is
    responsible for canonicalizing them.
    """
    for k in range(dims.E_max):
        sa, pa, sb, pb = get_port_pair(x, dims, k)
        if INACTIVE not in (sa, pa, sb, pb):
            yield k, sa, pa, sb, pb


# =============================================================================
# Anchor accessors
# =============================================================================

def get_anchor(
    x: NDArray, dims: PortPairDimensions,
) -> Tuple[int, int, int]:
    """Read (x, y, theta_deg) anchor."""
    return (
        int(x[dims.anchor_start + ANCHOR_X_OFFSET]),
        int(x[dims.anchor_start + ANCHOR_Y_OFFSET]),
        int(x[dims.anchor_start + ANCHOR_THETA_OFFSET]),
    )


def set_anchor(
    x: NDArray, dims: PortPairDimensions,
    ax: int, ay: int, atheta: int,
) -> None:
    """Write anchor."""
    x[dims.anchor_start + ANCHOR_X_OFFSET] = ax
    x[dims.anchor_start + ANCHOR_Y_OFFSET] = ay
    x[dims.anchor_start + ANCHOR_THETA_OFFSET] = atheta


# =============================================================================
<<<<<<< Updated upstream
# Junction accessors (Phase 4)
# =============================================================================

def get_junction(
    x: NDArray, dims: PortPairDimensions, j: int,
) -> Tuple[int, int, int, int, int]:
    """Read (active, anchor_slot, kind, param_a, param_b) for junction ``j``."""
    base = dims.junc_start + j * JUNCTION_GENES
    return (
        int(x[base + JUNC_ACTIVE_OFFSET]),
        int(x[base + JUNC_ANCHOR_OFFSET]),
        int(x[base + JUNC_KIND_OFFSET]),
        int(x[base + JUNC_PARAM_A_OFFSET]),
        int(x[base + JUNC_PARAM_B_OFFSET]),
    )


def set_junction(
    x: NDArray,
    dims: PortPairDimensions,
    j: int,
    *,
    active: int,
    anchor: int,
    kind: int,
    param_a: int,
    param_b: int,
) -> None:
    """Write all 5 genes of junction ``j``."""
    base = dims.junc_start + j * JUNCTION_GENES
    x[base + JUNC_ACTIVE_OFFSET] = active
    x[base + JUNC_ANCHOR_OFFSET] = anchor
    x[base + JUNC_KIND_OFFSET] = kind
    x[base + JUNC_PARAM_A_OFFSET] = param_a
    x[base + JUNC_PARAM_B_OFFSET] = param_b


def iter_junctions(
    x: NDArray, dims: PortPairDimensions,
) -> Iterator[Tuple[int, Tuple[int, int, int, int, int]]]:
    """Yield ``(j, descriptor)`` for every active junction."""
    for j in range(dims.J_max):
        descriptor = get_junction(x, dims, j)
        if descriptor[0] == 1:
            yield j, descriptor


# =============================================================================
=======
>>>>>>> Stashed changes
# Construction helpers
# =============================================================================

def create_empty_chromosome(dims: PortPairDimensions) -> NDArray:
<<<<<<< Updated upstream
    """Create a chromosome with all slots / pairs inactive and zero anchor.

    Flip and rotate bits start at 0; they are only meaningful for the
    matching piece-spec flag (symmetric / rotatable) and repair will
    override them anyway, but we keep them well-defined here.
    """
    x = np.full(dims.n_var, INACTIVE, dtype=DTYPE)
    x[dims.flip_start:dims.flip_end] = 0
    x[dims.rotate_start:dims.rotate_end] = 0
    # Junctions start fully inactive (active=0, anchor=0, kind=0, param_a=0, param_b=0).
    x[dims.junc_start:dims.junc_end] = 0
=======
    """Create a chromosome with all genes inactive and zero anchor."""
    x = np.full(dims.n_var, INACTIVE, dtype=DTYPE)
>>>>>>> Stashed changes
    x[dims.anchor_start + ANCHOR_X_OFFSET] = 0
    x[dims.anchor_start + ANCHOR_Y_OFFSET] = 0
    x[dims.anchor_start + ANCHOR_THETA_OFFSET] = 0
    return x


# =============================================================================
# Validation
# =============================================================================

def validate_chromosome(
    x: NDArray, dims: PortPairDimensions,
) -> List[str]:
    """Validate chromosome shape and gene-value bounds.

    Returns:
        List of error messages, empty if the chromosome is structurally valid.

    Notes:
        Does not check piece_id ranges (would require catalog) or per-piece
        port_idx bounds (would require per-slot piece kind). Those are
        enforced in repair.
    """
    errors: List[str] = []

    if len(x) != dims.n_var:
        errors.append(f"Length {len(x)} != expected {dims.n_var}")
        return errors

    for k in range(dims.N_max):
        v = int(x[dims.slot_start + k])
        if v < INACTIVE:
            errors.append(f"Slot[{k}]: {v} < INACTIVE")

<<<<<<< Updated upstream
    for k in range(dims.N_max):
        f = int(x[dims.flip_start + k])
        if f not in (0, 1):
            errors.append(f"Flip[{k}]={f} not in {{0, 1}}")

    for k in range(dims.N_max):
        r = int(x[dims.rotate_start + k])
        if r not in (0, 1):
            errors.append(f"Rotate[{k}]={r} not in {{0, 1}}")

=======
>>>>>>> Stashed changes
    for k in range(dims.E_max):
        sa, pa, sb, pb = get_port_pair(x, dims, k)
        for label, slot in (("slot_a", sa), ("slot_b", sb)):
            if slot != INACTIVE and (slot < 0 or slot >= dims.N_max):
                errors.append(
                    f"Pair[{k}].{label}={slot} out of [0, {dims.N_max - 1}]"
                )
        for label, port in (("port_a", pa), ("port_b", pb)):
            if port != INACTIVE and (port < 0 or port > MAX_PORT_IDX):
                errors.append(
                    f"Pair[{k}].{label}={port} out of [0, {MAX_PORT_IDX}]"
                )

<<<<<<< Updated upstream
    # Junction descriptors: active in {0,1}; anchor in [0, N_max-1]; kind /
    # params bounded. Phase 4 carries them inert through every operator.
    for j in range(dims.J_max):
        active, anchor, kind, pa_, pb_ = get_junction(x, dims, j)
        if active not in (0, 1):
            errors.append(f"Junction[{j}].active={active} not in {{0, 1}}")
        if not 0 <= anchor < max(1, dims.N_max):
            errors.append(
                f"Junction[{j}].anchor={anchor} out of [0, {dims.N_max - 1}]"
            )
        if not 0 <= kind <= JUNCTION_KIND_MAX:
            errors.append(
                f"Junction[{j}].kind={kind} out of [0, {JUNCTION_KIND_MAX}]"
            )
        if not 0 <= pa_ <= JUNCTION_PARAM_MAX:
            errors.append(
                f"Junction[{j}].param_a={pa_} out of [0, {JUNCTION_PARAM_MAX}]"
            )
        if not 0 <= pb_ <= JUNCTION_PARAM_MAX:
            errors.append(
                f"Junction[{j}].param_b={pb_} out of [0, {JUNCTION_PARAM_MAX}]"
            )

=======
>>>>>>> Stashed changes
    return errors


# =============================================================================
# Statistics
# =============================================================================

def chromosome_stats(x: NDArray, dims: PortPairDimensions) -> Dict:
    """Summary statistics for diagnostics and tests."""
    n_active_slots = sum(1 for _ in iter_active_slots(x, dims))
    n_active_pairs = sum(1 for _ in iter_active_pairs(x, dims))

    piece_counts: Dict[int, int] = {}
    for _, piece_id in iter_active_slots(x, dims):
        piece_counts[piece_id] = piece_counts.get(piece_id, 0) + 1

    return {
        "n_var": dims.n_var,
        "N_max": dims.N_max,
        "E_max": dims.E_max,
        "n_active_slots": n_active_slots,
        "n_active_pairs": n_active_pairs,
        "piece_counts": piece_counts,
        "anchor": get_anchor(x, dims),
    }
