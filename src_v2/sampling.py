"""Integer sampling for partitioned chromosome encoding.

PORT-PAIR ADAPTATION REQUIRED.
This module currently emits V1 partitioned chromosomes (main loop + junction
descriptors). It will not import successfully in src_v2 until either:
  (a) port_pair encoding/templates equivalents exist as siblings, or
  (b) the IntegerSampling class is rewritten to emit port-pair chromosomes.
The pattern generators (_gen_simple_loop, _gen_oval, _gen_racetrack,
_gen_oval_with_siding, _gen_oval_two_sidings) carry boundary/inventory-aware
sizing logic worth preserving — port them to emit (piece_slots, port_pairs)
tuples instead of (List[int], List[Junction]).

Generates initial population with:
- Heuristic seeds: boundary-aware, inventory-scaling closed-loop patterns
  (simple loops, ovals, racetracks, single- and two-siding ovals)
- Random chromosomes: random piece types with partial fill and random junctions

All patterns scale from inventory AND boundary — no hardcoded piece counts
and no hardcoded dimensional limits.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

from .config import OptimizationConfig
from .catalog import TrackCatalog
from .encoding import (
    INACTIVE,
    MAIN_LOOP_PIECE_INDICES,
    R40_LEFT,
    R40_RIGHT,
    STRAIGHT_16,
    STRAIGHT_24,
    SWITCH_INDICES,
    PartitionedDimensions,
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    generate_bounds,
    set_junction,
)
from .templates import (
    LEFT_SIDING,
    RIGHT_SIDING,
    TEMPLATES,
    check_siding_inventory,
    get_siding_inventory_requirements,
)


# =============================================================================
# Sizing helpers: every pattern dimension derives from boundary + inventory
# =============================================================================

Junction = Tuple[int, int, int, int]
Pattern = Tuple[List[int], Optional[List[Junction]]]

_PIECE_LEN = 16       # Stud length of STRAIGHT_16 (also a proxy for axial step)
_END_CAP = 80         # ~2 * R40 diameter: half-circle end cap along axial dir
_CORNER_SPAN = 40     # ~90° corner footprint (4 R40 curves)
_MARGIN = 20          # Boundary margin so layouts don't hug the edge
_MAX_BRANCH = 8       # Cap branch straights at physically plausible length


def _boundary_wh(dims: PartitionedDimensions) -> Tuple[float, float]:
    return (
        dims.boundary_max_x - dims.boundary_min_x,
        dims.boundary_max_y - dims.boundary_min_y,
    )


def _fit_oval_straights_per_section(
    dims: PartitionedDimensions,
    n_str_avail: int,
    piece_len: int = _PIECE_LEN,
    end_cap: float = _END_CAP,
    margin: float = _MARGIN,
    extra_axial: float = 0.0,
) -> int:
    """Per-section straight count for an oval, bounded by boundary and inventory.

    ``extra_axial`` accounts for post-decoder additions that stretch a section
    beyond its pre-injection length — most importantly the 32-stud surplus each
    siding contributes (switch FK 32 − straight FK 16 = 16 per switch × 2 switches).
    Symmetric oval seeds pass 0; single-siding and two-siding seeds pass 32.
    """
    w, _ = _boundary_wh(dims)
    m_fit = max(0, int((w - 2 * margin - end_cap - extra_axial) // piece_len))
    m_inv = n_str_avail // 2
    return max(1, min(m_fit, m_inv))


def _fit_racetrack_runs(
    dims: PartitionedDimensions,
    n_str_avail: int,
    piece_len: int = _PIECE_LEN,
    corner_span: float = _CORNER_SPAN,
    margin: float = _MARGIN,
) -> Tuple[int, int]:
    """Long/short straight run counts for a racetrack, split inventory evenly across 4 runs."""
    w, h = _boundary_wh(dims)
    long_fit = max(0, int((max(w, h) - 2 * margin - 2 * corner_span) // piece_len))
    short_fit = max(0, int((min(w, h) - 2 * margin - 2 * corner_span) // piece_len))
    budget = n_str_avail // 4
    return (
        max(1, min(long_fit, budget)),
        max(1, min(short_fit, budget)),
    )


def _fit_siding_branch_straights(
    n_str_remaining: int,
    m_mainline: int,
    max_branch: int = _MAX_BRANCH,
) -> int:
    """Branch straight count that fits inventory and leaves mainline room for the OUT switch."""
    return max(1, min(max_branch, n_str_remaining, max(1, m_mainline - 1)))


# =============================================================================
# Inventory utilities
# =============================================================================


def _count_pieces(pieces: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for p in pieces:
        counts[p] = counts.get(p, 0) + 1
    return counts


def _pieces_fit_inventory(counts: Dict[int, int], inv: Dict[int, int]) -> bool:
    return all(inv.get(idx, 0) >= c for idx, c in counts.items())


# =============================================================================
# Family generators — each emits boundary/inventory-aware size variants
# =============================================================================


def _gen_simple_loop(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """16 same-direction R40 curves = a tight closed circle."""
    variants: List[Pattern] = []
    if inv.get(R40_LEFT, 0) >= 16:
        variants.append(([int(R40_LEFT)] * 16, None))
    if inv.get(R40_RIGHT, 0) >= 16:
        variants.append(([int(R40_RIGHT)] * 16, None))
    return variants


def _gen_oval(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """2 directions × up to 4 size variants (m_max, 3/4, 1/2, 1/4)."""
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    m_max = _fit_oval_straights_per_section(dims, n_str)
    sizes = sorted(
        {max(1, int(m_max * f)) for f in (1.0, 0.75, 0.5, 0.25)},
        reverse=True,
    )
    for curve in (R40_LEFT, R40_RIGHT):
        if inv.get(curve, 0) < 16:
            continue
        for m in sizes:
            pieces = (
                [int(curve)] * 8 + [int(STRAIGHT_16)] * m
                + [int(curve)] * 8 + [int(STRAIGHT_16)] * m
            )
            if _pieces_fit_inventory(_count_pieces(pieces), inv):
                variants.append((pieces, None))
    return variants


def _gen_racetrack(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """2 directions × up to 3 aspect ratios."""
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    long_fit, short_fit = _fit_racetrack_runs(dims, n_str)
    seen: set = set()
    aspects: List[Tuple[int, int]] = []
    for pair in ((long_fit, short_fit), (long_fit, long_fit), (short_fit, short_fit)):
        if pair not in seen:
            seen.add(pair)
            aspects.append(pair)
    for curve in (R40_LEFT, R40_RIGHT):
        if inv.get(curve, 0) < 16:
            continue
        corner = [int(curve)] * 4
        for L, S in aspects:
            pieces = (
                corner + [int(STRAIGHT_16)] * L
                + corner + [int(STRAIGHT_16)] * S
                + corner + [int(STRAIGHT_16)] * L
                + corner + [int(STRAIGHT_16)] * S
            )
            if _pieces_fit_inventory(_count_pieces(pieces), inv):
                variants.append((pieces, None))
    return variants


def _gen_oval_with_siding(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Large oval + 1 passing siding (up to 2 dirs × 2 branch sizes).

    Main loop is **asymmetric** on purpose: first straight section has ``m + 2``
    straights, second section has ``m`` straights. Injecting 2 switches into the
    second section (each switch FK 32 vs each straight FK 16) adds +32 studs,
    balancing the axial length back to ``16 * (m + 2)`` and closing the loop.
    A symmetric ``[S]*m + [S]*m`` pre-injection seed is NOT feasible — the post-
    injection second section becomes 32 studs longer than the first.

    Handedness matches main direction (LEFT main + LEFT_SIDING, RIGHT main +
    RIGHT_SIDING) so the branch curves cooperate with the mainline curvature.

    Variants emitted only when the second section is long enough to absorb the
    required siding span (``compute_required_main_distance`` plus one guard
    straight so ``_find_out_position`` can land cleanly inside the section).
    """
    if dims.max_junctions < 1:
        return []
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    # Switches stretch the siding section by 32 studs post-injection; reserve that.
    m = _fit_oval_straights_per_section(dims, n_str, extra_axial=32.0)
    for curve, hand in ((R40_LEFT, 0), (R40_RIGHT, 1)):
        if inv.get(curve, 0) < 16:
            continue
        # Need enough straights for (m + 2) + m = 2m + 2 in the main loop.
        if 2 * m + 2 > n_str:
            continue
        main_pieces = (
            [int(curve)] * 8 + [int(STRAIGHT_16)] * (m + 2)
            + [int(curve)] * 8 + [int(STRAIGHT_16)] * m
        )
        main_counts = _count_pieces(main_pieces)
        if not _pieces_fit_inventory(main_counts, inv):
            continue
        n_str_remaining = max(0, n_str - (2 * m + 2))
        n_br_max = _fit_siding_branch_straights(n_str_remaining, m)
        branch_sizes = sorted(
            {n_br_max, max(1, n_br_max // 2)}, reverse=True,
        )
        template = TEMPLATES[hand]
        section_start = m + 18  # Start of second straight section in the asymmetric oval.
        for n_br in branch_sizes:
            # Guard: the second section must be long enough for the siding span.
            if not _siding_fits_in_section(template, n_br, m):
                continue
            if not check_siding_inventory(
                template, n_br,
                available_inventory=inv, used_inventory=main_counts,
            ):
                continue
            # Emit 1-3 distinct IN positions inside the section. More positions
            # give more seed diversity without changing closure (section length
            # is fixed pre-injection, swap locations don't affect axial sum).
            max_offset = m - _siding_walk_slots(template, n_br)
            offset_candidates = sorted({0, max_offset // 3, (2 * max_offset) // 3})
            for offset in offset_candidates:
                if offset < 0 or offset > max_offset:
                    continue
                junctions: List[Junction] = [
                    (1, section_start + offset, hand, n_br),
                ]
                variants.append((main_pieces, junctions))
    return variants


def _siding_walk_slots(template, n_br: int) -> int:
    """Number of straight slots the decoder must walk from IN through OUT."""
    from .templates import compute_required_main_distance

    required = compute_required_main_distance(template, n_br)
    return 2 + int(np.ceil(required / 16.0))


def _siding_fits_in_section(template, n_br: int, m: int) -> bool:
    """True iff the siding's required main distance fits within m straights.

    ``compute_required_main_distance`` gives the axial span of the branch.
    The decoder's ``_find_out_position`` walks straight pieces at 16 studs each
    looking for a matching position. We need room for IN + walk + OUT inside
    ``m`` straights; a one-straight guard keeps OUT strictly inside the section.
    """
    return m >= _siding_walk_slots(template, n_br)


def _gen_oval_two_sidings(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Large oval + 2 passing sidings (up to 4 hand-pairs × 2 branch sizes)."""
    if dims.max_junctions < 2:
        return []
    if inv.get(R40_LEFT, 0) < 16:
        return []
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    # Both sections absorb a siding; both stretch by 32 studs post-injection.
    m = _fit_oval_straights_per_section(dims, n_str, extra_axial=32.0)
    main_pieces = (
        [int(R40_LEFT)] * 8 + [int(STRAIGHT_16)] * m
        + [int(R40_LEFT)] * 8 + [int(STRAIGHT_16)] * m
    )
    main_counts = _count_pieces(main_pieces)
    if not _pieces_fit_inventory(main_counts, inv):
        return []
    n_str_remaining = max(0, n_str - 2 * m)
    # Split remaining straights across the two sidings
    n_br_max = _fit_siding_branch_straights(n_str_remaining // 2, m)
    branch_sizes = sorted(
        {n_br_max, max(1, n_br_max // 2)}, reverse=True,
    )
    hand_pairs = [(0, 0), (1, 1), (0, 1), (1, 0)]
    for h1, h2 in hand_pairs:
        for n_br in branch_sizes:
            used = dict(main_counts)
            t1 = TEMPLATES[h1]
            if not check_siding_inventory(
                t1, n_br, available_inventory=inv, used_inventory=used,
            ):
                continue
            for idx, need in get_siding_inventory_requirements(t1, n_br).items():
                used[idx] = used.get(idx, 0) + need
            t2 = TEMPLATES[h2]
            if not check_siding_inventory(
                t2, n_br, available_inventory=inv, used_inventory=used,
            ):
                continue
            junctions: List[Junction] = [
                (1, 8, h1, n_br),
                (1, 16 + m, h2, n_br),
            ]
            variants.append((main_pieces, junctions))
    return variants


# =============================================================================
# Sampling operator
# =============================================================================


class IntegerSampling(Sampling):
    """Sampling operator for partitioned chromosome encoding.

    Generates a mix of boundary/inventory-aware heuristic closed-loop seeds
    and random chromosomes. Heuristic fraction defaults to 0.20 so the seeded
    topologies (including switch-bearing sidings) form a meaningful bridgehead
    for the GA rather than a handful of noise points.
    """

    def __init__(
        self,
        catalog: TrackCatalog,
        config: OptimizationConfig,
        heuristic_ratio: float = 0.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.catalog = catalog
        self.config = config
        self.dims = compute_dimensions(config, catalog)
        self.heuristic_ratio = heuristic_ratio

        self.inventory_by_index: Dict[int, int] = {}
        for piece_id, count in config.inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None:
                self.inventory_by_index[idx] = (
                    self.inventory_by_index.get(idx, 0) + count
                )

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        """Generate initial population."""
        dims = self.dims
        rng = np.random.default_rng()
        X = np.full((n_samples, dims.n_var), INACTIVE, dtype=np.int16)

        n_heuristic = max(1, int(n_samples * self.heuristic_ratio))
        patterns = self._get_heuristic_patterns(rng)

        for i in range(n_heuristic):
            if patterns:
                pieces, junctions = patterns[i % len(patterns)]
                x = create_chromosome_from_pieces(dims, pieces, junctions)
                self._set_random_start(x, rng)
                X[i, :] = x
            else:
                X[i, :] = self._random_chromosome(rng)

        for i in range(n_heuristic, n_samples):
            X[i, :] = self._random_chromosome(rng)

        return X

    # =========================================================================
    # Heuristic pattern generation
    # =========================================================================

    def _get_heuristic_patterns(self, rng: np.random.Generator) -> List[Pattern]:
        """Build boundary/inventory-valid heuristic patterns, shuffled for variety."""
        inv = self.inventory_by_index
        dims = self.dims
        patterns: List[Pattern] = []
        patterns += _gen_simple_loop(inv, dims)
        patterns += _gen_oval(inv, dims)
        patterns += _gen_racetrack(inv, dims)
        patterns += _gen_oval_with_siding(inv, dims)
        patterns += _gen_oval_two_sidings(inv, dims)
        rng.shuffle(patterns)
        return patterns

    # =========================================================================
    # Random chromosome generation
    # =========================================================================

    def _random_chromosome(self, rng: Optional[np.random.Generator] = None) -> NDArray:
        """Random chromosome with partial fill and random junctions."""
        dims = self.dims
        x = create_empty_chromosome(dims)
        if rng is None:
            rng = np.random.default_rng()

        available = [
            idx for idx, count in self.inventory_by_index.items()
            if count > 0 and idx in MAIN_LOOP_PIECE_INDICES
        ]
        if not available:
            self._set_random_start(x, rng)
            return x

        fill_ratio = rng.uniform(0.5, 0.8)
        n_fill = max(1, int(dims.n_main * fill_ratio))
        positions = rng.choice(dims.n_main, size=n_fill, replace=False)
        positions.sort()

        inv_remaining = dict(self.inventory_by_index)
        for pos in positions:
            candidates = [idx for idx in available if inv_remaining.get(idx, 0) > 0]
            if not candidates:
                break
            idx = candidates[rng.integers(len(candidates))]
            x[pos] = idx
            inv_remaining[idx] -= 1

        for k in range(dims.max_junctions):
            active = int(rng.integers(0, 2))
            position = int(rng.integers(0, dims.n_main))
            handedness = int(rng.integers(0, 2))
            n_straights = int(rng.integers(0, min(dims.total_straights + 1, 6)))
            set_junction(x, dims, k, active, position, handedness, n_straights)

        self._set_random_start(x, rng)
        return x

    # =========================================================================
    # Helpers
    # =========================================================================

    def _set_random_start(
        self, x: NDArray, rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Set a small random fine-tuning offset on top of decoder auto-centering.

        The decoder's ``_auto_center`` already anchors each layout at the boundary
        center; the start_pos genes are a perturbation ON TOP of that. Sampling
        the full boundary range here puts large seeded layouts outside the
        boundary box (the root cause of the post-rewrite feasibility collapse).
        5 % of the boundary extent keeps large oval/racetrack seeds inside the
        boundary while still giving ``eliminate_duplicates`` enough room to keep
        cycled seeds as distinct individuals.
        """
        dims = self.dims
        if rng is None:
            rng = np.random.default_rng()
        w = dims.boundary_max_x - dims.boundary_min_x
        h = dims.boundary_max_y - dims.boundary_min_y
        off_x = max(1, int(w * 0.05))
        off_y = max(1, int(h * 0.05))
        x[dims.start_pos_start] = int(rng.integers(-off_x, off_x + 1))
        x[dims.start_pos_start + 1] = int(rng.integers(-off_y, off_y + 1))


# Backward-compatible aliases
MultiSegmentSampling = IntegerSampling
HeuristicSampling = IntegerSampling
