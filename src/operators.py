"""Segment-aware genetic operators for partitioned chromosome encoding.

Crossover:
- PartitionedCrossover: One-point crossover on main loop, uniform swap on
  junction slots, uniform swap on start position.

Mutation:
- PartitionedMutation: Weighted sub-operator selection with separate mutation
  strategies for main loop genes (80%) and junction genes (20%).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from .catalog import TrackCatalog
from .encoding import (
    CROSS_90,
    DC_N_ROUTES,
    GENES_PER_DBL_CROSSOVER,
    GENES_PER_JUNCTION,
    INACTIVE,
    MAIN_LOOP_PIECE_INDICES,
    MAX_MAIN_LOOP_PIECE,
    R40_CURVE,
    STRAIGHT_16,
    PartitionedDimensions,
    get_active_double_crossovers,
    get_double_crossover,
    set_double_crossover,
)
from .geometry import compute_fk_chain
from .intersection import find_crossing_pairs

# Valid main-loop piece types as a sorted array for fast random selection
_MAIN_LOOP_TYPES = np.array(sorted(MAIN_LOOP_PIECE_INDICES), dtype=np.int16)


def _protected_positions(x: NDArray, dims: PartitionedDimensions) -> set:
    """Main-loop positions that mutation must NOT disturb.

    The whole figure-8 / two-layer-loop main loop is precisely tuned for FK
    closure around its DOUBLE_CROSSOVER. Mutating ANY of its slots — not just
    the DC entry/exit positions — silently breaks the geometry and the
    decoder drops the DC from the layout. So when an active DC descriptor is
    present, we treat every active main-loop slot as sticky; the GA can
    still explore via DC mutations and uniform crossover slot swap.

    Layouts without any active DC keep the legacy behaviour (empty set), so
    non-DC heuristics still mutate normally.
    """
    if not get_active_double_crossovers(x, dims):
        return set()
    return {i for i in range(dims.n_main) if x[i] != INACTIVE}


# =============================================================================
# Crossover
# =============================================================================

class PartitionedCrossover(Crossover):
    """Segment-aware crossover respecting chromosome partition boundaries.

    - Main loop [0, n_main): one-point crossover (random cut, swap tails).
    - Junction slots [junc_start, junc_end): uniform per-slot swap (each
      4-gene junction descriptor taken from one parent at random).
    - Start position [start_pos_start, end): uniform from either parent.

    Args:
        dims: Chromosome partition dimensions.
        prob: Crossover probability per mating.
    """

    def __init__(self, dims: PartitionedDimensions, prob: float = 0.9) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.dims = dims

    def _do(self, problem, X, **kwargs) -> NDArray:
        # pymoo convention: X shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        dims = self.dims

        for k in range(n_matings):
            p1 = X[0, k]
            p2 = X[1, k]
            c1 = p1.copy()
            c2 = p2.copy()

            # --- Main loop + DC descriptors: tied together ---
            # The DC descriptor refers to specific main-loop positions; swapping
            # the descriptor alone from a different parent's main loop produces
            # a "ghost" descriptor that can't decode. So when either parent has
            # an active DC, we preserve the full main-loop AND DC-descriptor
            # block together — children get one parent's main loop with that
            # same parent's DC descriptors, no mixing. When neither parent has
            # DC, do classic one-point on the main loop and uniform swap on DC
            # slots (which are all-zero either way, so neutral).
            p1_has_dc = bool(get_active_double_crossovers(p1, dims))
            p2_has_dc = bool(get_active_double_crossovers(p2, dims))
            if p1_has_dc or p2_has_dc:
                # Preserve parent-to-child main loop + DC descriptors.
                pass
            else:
                if dims.n_main > 1:
                    cut = np.random.randint(1, dims.n_main)
                    c1[cut:dims.n_main] = p2[cut:dims.n_main]
                    c2[cut:dims.n_main] = p1[cut:dims.n_main]
                    # Mirror the cut on the parallel flip array so each child
                    # keeps the piece-and-flip pair from the same parent.
                    flip_cut = dims.main_flips_start + cut
                    c1[flip_cut:dims.main_flips_end] = p2[flip_cut:dims.main_flips_end]
                    c2[flip_cut:dims.main_flips_end] = p1[flip_cut:dims.main_flips_end]
                for j in range(dims.max_double_crossovers):
                    if np.random.random() < 0.5:
                        base = dims.dbl_crossover_start + j * GENES_PER_DBL_CROSSOVER
                        end = base + GENES_PER_DBL_CROSSOVER
                        c1[base:end] = p2[base:end]
                        c2[base:end] = p1[base:end]

            # --- Junctions: uniform per-slot swap ---
            for j in range(dims.max_junctions):
                if np.random.random() < 0.5:
                    base = dims.junc_start + j * GENES_PER_JUNCTION
                    end = base + GENES_PER_JUNCTION
                    c1[base:end] = p2[base:end]
                    c2[base:end] = p1[base:end]

            # --- Start position: uniform from either parent ---
            if np.random.random() < 0.5:
                sp = dims.start_pos_start
                c1[sp:sp + 2] = p2[sp:sp + 2]
                c2[sp:sp + 2] = p1[sp:sp + 2]

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


# =============================================================================
# Mutation Sub-operators (Main Loop)
# =============================================================================

def _assign_piece(x: NDArray, dims: PartitionedDimensions, pos: int, piece: int) -> None:
    """Write piece type at ``pos`` and align the flip slot with that piece's
    symmetry: R40_CURVE gets a random flip bit; symmetric pieces get flip=0."""
    x[pos] = piece
    if int(piece) == int(R40_CURVE):
        x[dims.main_flips_start + pos] = int(np.random.randint(0, 2))
    else:
        x[dims.main_flips_start + pos] = 0


def _mutate_piece_type(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Change a random active non-sticky main-loop position to a different piece
    type. CROSS_90 (placed by the decoder's aggressive repair) and slots claimed
    by an active DOUBLE_CROSSOVER descriptor are sticky."""
    main = x[:dims.n_main]
    protected = _protected_positions(x, dims)
    candidates = [
        i for i in range(dims.n_main)
        if main[i] != INACTIVE and int(main[i]) != int(CROSS_90) and i not in protected
    ]
    if not candidates:
        return
    pos = candidates[np.random.randint(len(candidates))]
    old = x[pos]
    choices = _MAIN_LOOP_TYPES[_MAIN_LOOP_TYPES != old]
    if len(choices) == 0:
        return
    _assign_piece(x, dims, pos, int(choices[np.random.randint(len(choices))]))


def _activate_position(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Activate a random inactive main-loop slot with a random piece type."""
    protected = _protected_positions(x, dims)
    inactive = [
        i for i in range(dims.n_main)
        if x[i] == INACTIVE and i not in protected
    ]
    if not inactive:
        return
    pos = inactive[np.random.randint(len(inactive))]
    _assign_piece(x, dims, pos, int(_MAIN_LOOP_TYPES[np.random.randint(len(_MAIN_LOOP_TYPES))]))


def _deactivate_position(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Set a random active non-sticky main-loop position to INACTIVE. CROSS_90
    and DC-claimed slots are sticky."""
    main = x[:dims.n_main]
    protected = _protected_positions(x, dims)
    candidates = [
        i for i in range(dims.n_main)
        if main[i] != INACTIVE and int(main[i]) != int(CROSS_90) and i not in protected
    ]
    if len(candidates) <= 4:  # keep minimum for closure
        return
    pos = candidates[np.random.randint(len(candidates))]
    x[pos] = INACTIVE


def _swap_positions(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Swap two random main-loop slots — both piece type AND its flip move together."""
    if dims.n_main < 2:
        return
    protected = _protected_positions(x, dims)
    i, j = np.random.choice(dims.n_main, size=2, replace=False)
    if int(x[i]) == int(CROSS_90) or int(x[j]) == int(CROSS_90):
        return
    if i in protected or j in protected:
        return
    x[i], x[j] = x[j], x[i]
    fi = dims.main_flips_start + i
    fj = dims.main_flips_start + j
    x[fi], x[fj] = x[fj], x[fi]


def _flip_position(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Toggle the flip bit on a random active R40_CURVE main-loop slot."""
    r40 = int(R40_CURVE)
    candidates = [i for i in range(dims.n_main) if int(x[i]) == r40]
    if not candidates:
        return
    pos = candidates[np.random.randint(len(candidates))]
    fp = dims.main_flips_start + pos
    x[fp] = 0 if int(x[fp]) else 1


def _straighten_near_unresolved_crossing(
    x: NDArray, dims: PartitionedDimensions, catalog: Optional[TrackCatalog] = None,
) -> None:
    """Replace a curve near an unresolved self-intersection with STRAIGHT_16.

    Decodes the chromosome's main loop, runs find_crossing_pairs on the FK
    chain, and looks for crossings the existing _apply_crossing_repair would
    refuse — pairs where at least one piece is non-STRAIGHT_16. For one such
    pair, swaps the curve-side piece to STRAIGHT_16 so subsequent decode
    passes have a chance of producing a STR-on-STR perpendicular crossing
    that the repair will convert to CROSS_90.

    Falls through to _mutate_piece_type when:
      - the catalog is unavailable (operator called outside a Problem context)
      - no main-loop pieces are active
      - find_crossing_pairs returns nothing
      - all detected crossings are already STR-on-STR (the existing repair
        already handles those; nudging won't help)
    """
    if catalog is None:
        _mutate_piece_type(x, dims)
        return

    main_pieces: List[int] = []
    chrom_indices: List[int] = []
    for i in range(dims.n_main):
        v = int(x[i])
        if v == INACTIVE:
            continue
        main_pieces.append(v)
        chrom_indices.append(i)

    if len(main_pieces) < 4:
        _mutate_piece_type(x, dims)
        return

    indices = np.asarray(main_pieces, dtype=np.int32)
    states = compute_fk_chain(catalog.get_fk(indices))
    pairs = find_crossing_pairs(states, main_pieces)

    unresolved = [
        (pi, pj) for pi, pj, _ang in pairs
        if main_pieces[pi] != STRAIGHT_16 or main_pieces[pj] != STRAIGHT_16
    ]
    if not unresolved:
        _mutate_piece_type(x, dims)
        return

    pi, pj = unresolved[np.random.randint(len(unresolved))]
    target = pi if main_pieces[pi] != STRAIGHT_16 else pj
    chrom_pos = chrom_indices[target]
    x[chrom_pos] = int(STRAIGHT_16)
    x[dims.main_flips_start + chrom_pos] = 0


# =============================================================================
# Mutation Sub-operators (Junction)
# =============================================================================

def _toggle_active(x: NDArray, dims: PartitionedDimensions) -> None:
    """Flip the active flag on a random junction slot."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base] = 1 - x[base]  # 0 -> 1, 1 -> 0


def _reposition_junction(x: NDArray, dims: PartitionedDimensions) -> None:
    """Shift position of a random junction by a small delta."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    delta = np.random.randint(-5, 6)  # -5 to +5, excluding 0 is unlikely but fine
    new_pos = int(np.clip(x[base + 1] + delta, 0, dims.n_main - 1))
    x[base + 1] = new_pos


def _change_handedness(x: NDArray, dims: PartitionedDimensions) -> None:
    """Set handedness of a random junction to a valid template index."""
    if dims.max_junctions == 0:
        return
    from .templates import TEMPLATES
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    x[base + 2] = np.random.randint(0, len(TEMPLATES))


def _adjust_straights(x: NDArray, dims: PartitionedDimensions) -> None:
    """Adjust n_straights of a random junction by +-1..3."""
    if dims.max_junctions == 0:
        return
    slot = np.random.randint(dims.max_junctions)
    base = dims.junc_start + slot * GENES_PER_JUNCTION
    delta = np.random.choice([-3, -2, -1, 1, 2, 3])
    new_val = int(np.clip(x[base + 3] + delta, 0, dims.total_straights))
    x[base + 3] = new_val


# =============================================================================
# Mutation Sub-operators (Double Crossover)
# =============================================================================

# Catalog routes paired into valid 2-route covers — used by _rotate_dc_route_pair
# to keep mutation inside the no-dangling family rather than dropping a
# descriptor into infeasibility.
_DC_BOTH_THROUGH = (0, 1)
_DC_BOTH_CROSS = (2, 3)
_DC_VALID_PAIRS = (_DC_BOTH_THROUGH, _DC_BOTH_CROSS)


def _toggle_dc_active(x: NDArray, dims: PartitionedDimensions) -> None:
    """Flip the active flag on a random DBL_CROSSOVER descriptor."""
    if dims.max_double_crossovers == 0:
        return
    slot = np.random.randint(dims.max_double_crossovers)
    active, p1, r1, p2, r2 = get_double_crossover(x, dims, slot)
    set_double_crossover(x, dims, slot, 1 - active, p1, r1, p2, r2)


def _shift_dc_position(x: NDArray, dims: PartitionedDimensions) -> None:
    """Nudge pos_1 or pos_2 on a random DBL_CROSSOVER descriptor by ±1..5."""
    if dims.max_double_crossovers == 0 or dims.n_main < 2:
        return
    slot = np.random.randint(dims.max_double_crossovers)
    active, p1, r1, p2, r2 = get_double_crossover(x, dims, slot)
    delta = int(np.random.choice([-5, -3, -1, 1, 3, 5]))
    if np.random.random() < 0.5:
        p1 = int(np.clip(p1 + delta, 0, dims.n_main - 1))
    else:
        p2 = int(np.clip(p2 + delta, 0, dims.n_main - 1))
    set_double_crossover(x, dims, slot, active, p1, r1, p2, r2)


def _rotate_dc_route_pair(x: NDArray, dims: PartitionedDimensions) -> None:
    """Rotate routes on a random descriptor to the OTHER valid 2-route cover.

    Keeps the descriptor in the no-dangling family — flips both-through to
    both-cross or vice versa. Standalone route changes that break the cover
    are pointless: the decoder skips invalid descriptors so the chromosome
    just loses the piece.
    """
    if dims.max_double_crossovers == 0:
        return
    slot = np.random.randint(dims.max_double_crossovers)
    active, p1, _r1, p2, _r2 = get_double_crossover(x, dims, slot)
    other_pair = _DC_VALID_PAIRS[np.random.randint(len(_DC_VALID_PAIRS))]
    set_double_crossover(x, dims, slot, active, p1, other_pair[0], p2, other_pair[1])


def _swap_dc_traversals(x: NDArray, dims: PartitionedDimensions) -> None:
    """Swap (pos_1, route_1) with (pos_2, route_2) on a random descriptor.

    Reorders which traversal happens first along the loop; both end states
    are still on the same physical piece, so the no-dangling property is
    preserved.
    """
    if dims.max_double_crossovers == 0:
        return
    slot = np.random.randint(dims.max_double_crossovers)
    active, p1, r1, p2, r2 = get_double_crossover(x, dims, slot)
    set_double_crossover(x, dims, slot, active, p2, r2, p1, r1)


# =============================================================================
# Combined Mutation
# =============================================================================

# Main loop sub-operators with equal weights. The crossing-aware nudge has a
# small weight so its decode + intersection cost (O(n_main^2)) doesn't dominate
# total mutation cost, while still firing often enough to bias the population
# toward repairable straight-on-straight crossings.
_MAIN_LOOP_OPS = [
    _mutate_piece_type,
    _activate_position,
    _deactivate_position,
    _swap_positions,
    _flip_position,
    _straighten_near_unresolved_crossing,
]
# _flip_position carries 0.10 — it's the only operator that explores R40 handedness,
# so giving it < 10% would leave LEFT/RIGHT diversity entirely to the initial seeds.
_MAIN_LOOP_WEIGHTS = np.array([0.25, 0.22, 0.18, 0.20, 0.10, 0.05])
_MAIN_LOOP_WEIGHTS /= _MAIN_LOOP_WEIGHTS.sum()

# Junction sub-operators with equal weights
_JUNCTION_OPS = [_toggle_active, _reposition_junction, _change_handedness, _adjust_straights]
_JUNCTION_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])
_JUNCTION_WEIGHTS /= _JUNCTION_WEIGHTS.sum()

# Double-crossover sub-operators with equal weights
_DBL_CROSSOVER_OPS = [
    _toggle_dc_active,
    _shift_dc_position,
    _rotate_dc_route_pair,
    _swap_dc_traversals,
]
_DBL_CROSSOVER_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])
_DBL_CROSSOVER_WEIGHTS /= _DBL_CROSSOVER_WEIGHTS.sum()


class PartitionedMutation(Mutation):
    """Segment-aware mutation with weighted sub-operator selection.

    Each individual that passes the probability gate receives exactly one
    mutation drawn from two categories:

    - **Main loop mutations** (80% when junctions exist, 100% otherwise):
      piece_type_change, activate_position, deactivate_position, swap_positions.
    - **Junction mutations** (20% when junctions exist):
      toggle_active, reposition, change_handedness, adjust_straights.

    Args:
        dims: Chromosome partition dimensions.
        prob: Per-individual mutation probability.
    """

    def __init__(self, dims: PartitionedDimensions, prob: float = 0.3) -> None:
        super().__init__(prob=prob)
        self.dims = dims

    def _do(self, problem, X, **kwargs) -> NDArray:
        has_junctions = self.dims.max_junctions > 0
        catalog = getattr(problem, "catalog", None)

        # Mutation budget per individual:
        #   10% junction (passing-siding) when present, remainder on the main
        #   loop. DBL_CROSSOVER descriptors are deliberately NOT mutated:
        #   heuristic seeds are the only safe source. Any mutation on a
        #   working DC descriptor (shift/rotate/swap/toggle) reliably breaks
        #   the geometric pair, drops the DC from the layout, and selection
        #   then eliminates that lineage. Reintroducing DC via mutation alone
        #   essentially never validates against random geometry.
        junc_thresh = 0.10 if has_junctions else 0.0

        for i in range(len(X)):
            r = np.random.random()
            if r < junc_thresh:
                op_idx = np.random.choice(len(_JUNCTION_OPS), p=_JUNCTION_WEIGHTS)
                _JUNCTION_OPS[op_idx](X[i], self.dims)
            else:
                op_idx = np.random.choice(len(_MAIN_LOOP_OPS), p=_MAIN_LOOP_WEIGHTS)
                _MAIN_LOOP_OPS[op_idx](X[i], self.dims, catalog)

        return X


# =============================================================================
# Convenience: NoOpCrossover (kept for backward compatibility)
# =============================================================================

class NoOpCrossover(Crossover):
    """Identity crossover -- returns parents unchanged."""

    def __init__(self, **kwargs) -> None:
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, X, **kwargs) -> NDArray:
        return X
