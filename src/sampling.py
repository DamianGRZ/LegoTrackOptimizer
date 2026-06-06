"""Integer sampling for partitioned chromosome encoding.

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
    CROSS_90,
    DC_ROUTE_CROSS_1_TO_2,
    DC_ROUTE_CROSS_2_TO_1,
    DC_ROUTE_TRACK1_THROUGH,
    DC_ROUTE_TRACK2_THROUGH,
    DOUBLE_CROSSOVER,
    INACTIVE,
    MAIN_LOOP_PIECE_INDICES,
    R40_CURVE,
    STRAIGHT_16,
    STRAIGHT_24,
    SWITCH_INDICES,
    SWITCH_LEFT,
    SWITCH_RIGHT,
    PartitionedDimensions,
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    generate_bounds,
    set_cross_junction,
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
CrossJunctionDescriptor = Tuple[int, int, int]      # (active, pos_1, pos_2)
DblCrossoverDescriptor = Tuple[int, int, int, int, int]  # (active, p1, r1, p2, r2)
# (main_pieces, main_flips, junctions, cross_junctions, dbl_crossovers)
Pattern = Tuple[
    List[int],
    List[int],
    Optional[List[Junction]],
    Optional[List[CrossJunctionDescriptor]],
    Optional[List[DblCrossoverDescriptor]],
]


def _curve_flips(pieces: List[int], default_flip: int) -> List[int]:
    """Per-slot flip array. R40_CURVE slots take ``default_flip``; everything else 0."""
    r40 = int(R40_CURVE)
    return [default_flip if p == r40 else 0 for p in pieces]

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
    """16 same-direction R40 curves = a tight closed circle. Emits LEFT and RIGHT."""
    variants: List[Pattern] = []
    if inv.get(R40_CURVE, 0) < 16:
        return variants
    pieces = [int(R40_CURVE)] * 16
    for flip_value in (0, 1):
        variants.append((pieces, [flip_value] * 16, None, None, None))
    return variants


def _gen_oval(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """2 directions × up to 4 size variants. Direction is a flip bit on the R40 slots."""
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    if inv.get(R40_CURVE, 0) < 16:
        return variants
    m_max = _fit_oval_straights_per_section(dims, n_str)
    sizes = sorted(
        {max(1, int(m_max * f)) for f in (1.0, 0.75, 0.5, 0.25)},
        reverse=True,
    )
    for direction_flip in (0, 1):
        for m in sizes:
            pieces = (
                [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * m
                + [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * m
            )
            if _pieces_fit_inventory(_count_pieces(pieces), inv):
                variants.append((pieces, _curve_flips(pieces, direction_flip), None, None, None))
    return variants


def _gen_racetrack(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """2 directions × up to 3 aspect ratios."""
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    if inv.get(R40_CURVE, 0) < 16:
        return variants
    long_fit, short_fit = _fit_racetrack_runs(dims, n_str)
    seen: set = set()
    aspects: List[Tuple[int, int]] = []
    for pair in ((long_fit, short_fit), (long_fit, long_fit), (short_fit, short_fit)):
        if pair not in seen:
            seen.add(pair)
            aspects.append(pair)
    corner = [int(R40_CURVE)] * 4
    for direction_flip in (0, 1):
        for L, S in aspects:
            pieces = (
                corner + [int(STRAIGHT_16)] * L
                + corner + [int(STRAIGHT_16)] * S
                + corner + [int(STRAIGHT_16)] * L
                + corner + [int(STRAIGHT_16)] * S
            )
            if _pieces_fit_inventory(_count_pieces(pieces), inv):
                variants.append((pieces, _curve_flips(pieces, direction_flip), None, None, None))
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
    if inv.get(R40_CURVE, 0) < 16:
        return variants
    for hand in (0, 1):
        if 2 * m + 2 > n_str:
            continue
        # LEFT siding uses LEFT-curving mainline (flip=0); RIGHT siding uses RIGHT (flip=1).
        main_pieces = (
            [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * (m + 2)
            + [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * m
        )
        main_flips = _curve_flips(main_pieces, hand)
        main_counts = _count_pieces(main_pieces)
        if not _pieces_fit_inventory(main_counts, inv):
            continue
        n_str_remaining = max(0, n_str - (2 * m + 2))
        n_br_max = _fit_siding_branch_straights(n_str_remaining, m)
        branch_sizes = sorted(
            {n_br_max, max(1, n_br_max // 2)}, reverse=True,
        )
        template = TEMPLATES[hand]
        section_start = m + 18
        for n_br in branch_sizes:
            if not _siding_fits_in_section(template, n_br, m):
                continue
            if not check_siding_inventory(
                template, n_br,
                available_inventory=inv, used_inventory=main_counts,
            ):
                continue
            max_offset = m - _siding_walk_slots(template, n_br)
            offset_candidates = sorted({0, max_offset // 3, (2 * max_offset) // 3})
            for offset in offset_candidates:
                if offset < 0 or offset > max_offset:
                    continue
                junctions: List[Junction] = [
                    (1, section_start + offset, hand, n_br),
                ]
                variants.append((main_pieces, main_flips, junctions, None, None))
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


def _gen_figure_eight(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Single-loop figure-8 wrapped around one CROSS_90.

    Construction (from cross center at (8, 0) with 2-STR extensions on each
    port and 12-R40 three-quarter arcs on diagonally opposite quadrants):

      Variant A (TL + BR arcs):
        N ext (S-bound) -> cross V (N->S) -> S ext (S-bound)
          -> 12 R40_L (bottom-right arc) -> E ext (W-bound)
          -> cross H (E->W) -> W ext (W-bound)
          -> 12 R40_R (top-left arc) -> back to N ext start

      Variant B (TR + BL arcs):
        N ext (S-bound) -> cross V (N->S) -> S ext (S-bound)
          -> 12 R40_R (bottom-left arc) -> W ext (E-bound)
          -> cross H (W->E) -> E ext (E-bound)
          -> 12 R40_L (top-right arc) -> back to N ext start

    Each variant is a 34-piece chromosome (10 STR + 12 R40_L + 12 R40_R).
    The two STR slots that physically pass through the cross center (8, 0)
    cross at 90 deg; the self-intersection repair (decoder Step 4)
    converts one slot to CROSS_90 -> net inventory 9 STR + 12 R40_L +
    12 R40_R + 1 CROSS_90.

    Bounding box ~160 x 160 stud (x in [-72, 88], y in [-80, 80] before
    auto-centering), so a boundary of at least ~200 stud per axis is
    required for the seed to sit cleanly inside.
    """
    if (
        inv.get(STRAIGHT_16, 0) < 10
        or inv.get(R40_CURVE, 0) < 12
        or inv.get(R40_CURVE, 0) < 12
        or inv.get(CROSS_90, 0) < 1
    ):
        return []

    # Boundary guard: figure-8 needs ~160 stud in each dimension; reject if the
    # bounding box would not fit with a small margin on each side.
    w, h = _boundary_wh(dims)
    if w < 180 or h < 180:
        return []

    # Both variants share the same piece layout post-collapse; handedness is
    # encoded entirely in the flip array. Variant A: first arc LEFT, second RIGHT.
    # Variant B (mirror): first arc RIGHT, second LEFT.
    pieces = (
        [int(STRAIGHT_16)] * 5
        + [int(R40_CURVE)] * 12
        + [int(STRAIGHT_16)] * 5
        + [int(R40_CURVE)] * 12
    )

    flips_a = [0] * 5 + [0] * 12 + [0] * 5 + [1] * 12  # first arc LEFT, second RIGHT
    flips_b = [0] * 5 + [1] * 12 + [0] * 5 + [0] * 12  # mirror

    variants: List[Pattern] = []
    for flips in (flips_a, flips_b):
        if _pieces_fit_inventory(_count_pieces(pieces), inv):
            variants.append((pieces, flips, None, None, None))
    return variants


def _figure_eight_main_loop(k: int) -> Tuple[List[int], List[int]]:
    """Main-loop sequence + flips for a figure-8 around one DBL_CROSSOVER.

    Each lobe consumes ``3 + 2*k`` straights between two half-circles. The two
    lobes' R40_CURVE arcs use opposite handedness so the figure-8 closes:
    first lobe uses flip=0 (LEFT), second uses flip=1 (RIGHT).
    """
    s_p = lambda n: [int(STRAIGHT_16)] * n
    s_f = lambda n: [0] * n
    arc_p = lambda n: [int(R40_CURVE)] * n
    pieces = (
        [int(STRAIGHT_16)]
        + s_p(k) + arc_p(8) + s_p(2 * k + 3) + arc_p(8) + s_p(k)
        + [int(STRAIGHT_16)]
        + s_p(k) + arc_p(8) + s_p(2 * k + 3) + arc_p(8) + s_p(k)
    )
    flips = (
        [0]
        + s_f(k) + [0] * 8 + s_f(2 * k + 3) + [0] * 8 + s_f(k)
        + [0]
        + s_f(k) + [1] * 8 + s_f(2 * k + 3) + [1] * 8 + s_f(k)
    )
    return pieces, flips


def _gen_figure_eight_dbl_crossover(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Single-loop figure-8 wrapped around one DOUBLE_CROSSOVER.

    Train passes through ONE physical DBL_CROSSOVER twice via cross_1_to_2 +
    cross_2_to_1 (the both-cross 2-route cover of {A,B,C,D}) — no port dangles.

    Emits up to ``len(_K_SIZES)`` size variants. Each variant inflates the
    figure-8 along the east-west axis: ``k=0`` is the tight 40-piece base,
    higher ``k`` adds 4 STR_16 per step. Larger variants compete on
    utilisation with non-DC seeds, so the GA actually keeps them around.
    """
    if dims.max_double_crossovers < 1:
        return []
    base = {R40_CURVE: 16, R40_CURVE: 16, DOUBLE_CROSSOVER: 1}
    if any(inv.get(idx, 0) < n for idx, n in base.items()):
        return []

    n_str = inv.get(STRAIGHT_16, 0)
    # k extra east-bound straights per lobe cost (4k + 8) straights total.
    # The 40-piece minimum already needs 8 STR_16; bigger variants grow from there.
    k_inv = max(0, (n_str - 8) // 4)
    w, h = _boundary_wh(dims)
    # Each k increment widens the bounding box by 32 stud (16 stud per side);
    # base width at k=0 is 128 stud, base height ~176 stud.
    k_fit_w = max(0, int((w - 128) // 32))
    # Scale purely from inventory + boundary (project invariant: no hardcoded size
    # caps). The old hardcoded `6` capped the figure-8 at ~88 pieces (42% util);
    # uncapped it reaches the boundary-limited k (~11 -> 128 pieces, ~61%), giving
    # the GA a DC-bearing seed that competes with the racetrack instead of being
    # bred out as too small. Each variant is still oracle-feasible by construction.
    k_max = max(0, min(k_inv, k_fit_w))

    variants: List[Pattern] = []
    for k in sorted({k_max, k_max // 2, 0}, reverse=True):
        if h < 200 or w < 128 + 32 * k:
            continue
        pieces, flips = _figure_eight_main_loop(k)
        if not _pieces_fit_inventory(_count_pieces(pieces), inv):
            continue
        pos_2 = len(pieces) // 2
        descriptors: List[DblCrossoverDescriptor] = [
            (1, 0, DC_ROUTE_CROSS_1_TO_2, pos_2, DC_ROUTE_CROSS_2_TO_1),
        ]
        variants.append((pieces, flips, None, None, descriptors))
    return variants


def _gen_two_layer_loop_dbl_crossover(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Disabled: the naive two-layer pattern has an unavoidable self-crossing
    on the east side of the DOUBLE_CROSSOVER.

    Train geometry: exiting port B of track 1 heading east, the first 4
    R40_CURVE arc passes through (~72, 8) on its way up; exiting port D of
    track 2 heading east, the first 4 R40_CURVE arc passes through (~72, 8)
    on its way down. The two arcs cross at one external point. Repairing the
    crossing needs a CROSS_90 piece, but the ``with_double_crossover``
    inventory carries no CROSS_90s and a heuristic seed that consumes a
    CROSS_90 it doesn't own is not useful.

    Left as a stub so the seeder pipeline retains the slot for a future
    redesign that uses extra straights/curves to lift one arc clear of the
    other (or that ships a config with CROSS_90 inventory).
    """
    return []


def _gen_figure_eight_cross(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Closed figure-8 with one bare CROSS_90 self-crossing at ~90 deg.

    Two opposite-handed lobes (12 R40 curves each) joined by 5-straight runs.
    The loop crosses itself perpendicular at main-loop slots 2 and 19; verified
    closed by construction (closure 0.0, angle 0.0, bounding box ~160x160 studs).
    Consumes 10 STRAIGHT_16 + 24 R40_CURVE + 1 CROSS_90. The descriptor
    (active, pos_1, pos_2) = (1, 2, 19) is committed by _inject_cross_junctions.

    Unlike the retired 4-switch soft seed, this geometry validates immediately —
    the seed lands a feasible CROSS_90 layout in the initial population.
    """
    if dims.max_cross_junctions < 1 or inv.get(CROSS_90, 0) < 1:
        return []
    if inv.get(STRAIGHT_16, 0) < 10 or inv.get(R40_CURVE, 0) < 24:
        return []
    w, h = _boundary_wh(dims)
    if w < 170 or h < 170:  # figure-8 spans ~160 studs each axis
        return []

    pieces = (
        [int(STRAIGHT_16)] * 5 + [int(R40_CURVE)] * 12
        + [int(STRAIGHT_16)] * 5 + [int(R40_CURVE)] * 12
    )
    flips = [0] * 5 + [0] * 12 + [0] * 5 + [1] * 12
    descriptors: List[CrossJunctionDescriptor] = [(1, 2, 19)]
    return [(pieces, flips, None, descriptors, None)]


def _gen_oval_two_sidings(
    inv: Dict[int, int], dims: PartitionedDimensions,
) -> List[Pattern]:
    """Large oval + 2 passing sidings (up to 4 hand-pairs × 2 branch sizes)."""
    if dims.max_junctions < 2:
        return []
    if inv.get(R40_CURVE, 0) < 16:
        return []
    variants: List[Pattern] = []
    n_str = inv.get(STRAIGHT_16, 0)
    # Both sections absorb a siding; both stretch by 32 studs post-injection.
    m = _fit_oval_straights_per_section(dims, n_str, extra_axial=32.0)
    main_pieces = (
        [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * m
        + [int(R40_CURVE)] * 8 + [int(STRAIGHT_16)] * m
    )
    main_counts = _count_pieces(main_pieces)
    if not _pieces_fit_inventory(main_counts, inv):
        return []
    n_str_remaining = max(0, n_str - 2 * m)
    n_br_max = _fit_siding_branch_straights(n_str_remaining // 2, m)
    branch_sizes = sorted(
        {n_br_max, max(1, n_br_max // 2)}, reverse=True,
    )
    hand_pairs = [(0, 0), (1, 1), (0, 1), (1, 0)]
    for h1, h2 in hand_pairs:
        # Mainline curve direction matches the first siding's handedness so the
        # branch curves cooperate with mainline curvature on at least one siding.
        main_flips = _curve_flips(main_pieces, h1)
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
            variants.append((main_pieces, main_flips, junctions, None, None))
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
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.catalog = catalog
        self.config = config
        self.dims = compute_dimensions(config, catalog)
        self.heuristic_ratio = heuristic_ratio
        self._seed = seed

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
        rng = np.random.default_rng(self._seed)
        X = np.full((n_samples, dims.n_var), INACTIVE, dtype=np.int16)

        n_heuristic = max(1, int(n_samples * self.heuristic_ratio))
        patterns = self._get_heuristic_patterns(rng)

        for i in range(n_heuristic):
            if patterns:
                pieces, flips, junctions, cross_junctions, dbl_crossovers = (
                    patterns[i % len(patterns)]
                )
                x = create_chromosome_from_pieces(
                    dims, pieces,
                    main_loop_flips=flips,
                    junctions=junctions,
                    cross_junctions=cross_junctions,
                    double_crossovers=dbl_crossovers,
                )
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
        patterns += _gen_figure_eight(inv, dims)
        patterns += _gen_figure_eight_cross(inv, dims)
        patterns += _gen_figure_eight_dbl_crossover(inv, dims)
        patterns += _gen_two_layer_loop_dbl_crossover(inv, dims)
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
        r40_curve = int(R40_CURVE)
        for pos in positions:
            candidates = [idx for idx in available if inv_remaining.get(idx, 0) > 0]
            if not candidates:
                break
            idx = candidates[rng.integers(len(candidates))]
            x[pos] = idx
            inv_remaining[idx] -= 1
            # Random handedness for R40_CURVE; symmetric pieces stay flip=0.
            if idx == r40_curve:
                x[dims.main_flips_start + pos] = int(rng.integers(0, 2))

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
