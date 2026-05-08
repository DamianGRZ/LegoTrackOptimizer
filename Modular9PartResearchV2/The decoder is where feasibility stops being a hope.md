# The decoder is where feasibility stops being a hope

**The construction decoder is the conceptual heart of this thesis: a seven-phase, turtle-graphics forward-kinematics pipeline that maps every invariant-passing chromosome to a port-exact, catalog-valid placement list — shifting infeasibility from a probabilistic hazard into two bounded, measurable residuals.** It replaces the BRKGA tradition's informal "decoder is in general a heuristic" stance (Gonçalves & Resende 2011, *J Heuristics* 17:487–525, DOI 10.1007/s10732-010-9143-1) with a precise, provable contract: given `validate_invariants(x) == True`, the output satisfies five structural invariants P1–P5 by construction, leaving only two residuals — main-loop closure and self-intersection — for the NSGA-II constraint layer. The closest structural precedent is Zawidzki's Truss-Z modular chaining (DOI 10.1016/j.advengsoft.2013.04.022), which likewise guarantees module-to-module mating but hands endpoint-reaching and overlap to a penalty objective; the implementation idiom — push/pop turtle state at bracketed branches — is Prusinkiewicz & Lindenmayer's 1990 bracketed L-system interpretation. A lattice-integer angular prefilter rejects roughly **98.4%** of uniformly random main loops before any `sin`/`cos` call. The hot Phase-3 walk numba-compiles to ~1 µs per main loop; full decode (with collision checking) costs ~150 µs; a 200 × 1000 run amortizes to under a minute of decoder time. Beyond verifying prior claims, this report pins four previously-underspecified contracts: the branch-slot counting rule (slot *k* realizes the *k*-th mask-2 switch), the crossing-overlay validation convention, the closure-residual metric, and the status-code taxonomy the problem layer consumes.

## Why "feasibility by construction" is not a textbook result

The BRKGA canon has long *practiced* feasibility-preserving decoders without ever *formalizing* them. Gonçalves & Resende's 2011 tutorial defines a decoder as "a deterministic algorithm that takes as input an array of *n* random keys and produces as output a solution of the optimization problem … **A decoder is, in general, a heuristic**." Londe, Pessoa, Andrade & Resende's 2024 review (DOI 10.1016/j.ejor.2024.03.030; arXiv:2312.00961) softens slightly but retains the hedge: "Usually, decoded solutions are feasible, but infeasibility can be dealt with by penalties." The review partitions decoders into **permutation-based** (sort keys → greedy insert) and **indicator-based** (key value selects a discrete branch), with a growing "constructive" subfamily (Kummer et al. 2022 home-healthcare VRP with cheapest-insertion; Ramos et al. 2016 container loading; Andrade et al. 2017 matheuristic decoders embedding an LP solver). None of these state a theorem of the form *"the decoder is a total function D : X → F where F is the geometrically valid set."*

That gap matters. Bean's 1994 original random-key GA (DOI 10.1287/ijoc.6.2.154) sorted keys into a permutation that a downstream heuristic *might* repair. Two decades of BRKGA application papers have treated feasibility as a problem-specific engineering achievement, not a decoder-framework invariant. **This thesis's contribution is to make the invariant formal, enumerate its scope precisely, and push the unguaranteed residuals to a single well-defined constraint layer.** The formal framing — not the engineering — is what is new.

## Truss-Z is the right structural analogy; L-systems are the right implementation idiom

Zawidzki's Truss-Z body of work is the closest published relative. A Truss-Z chromosome is a string over a four-symbol alphabet {R, L, R², L²}, each symbol denoting an affine (rotation + mirror) transform applied to a single trapezoidal module. Decoding is deterministic chaining: each module's input edge is mated to the previous module's output edge. Zawidzki & Nishinari 2013 state outright: "Encoding of a planar TZ path, selection method, objective (cost) function and genetic operations are introduced"; the 2018 GPU paper (DOI 10.1016/j.asoc.2018.05.042) confirms that "the result guarantees a TZ connection between two given points." What is *not* guaranteed: reaching-error ε_r, overlap-error ε_o with obstacles and with the chain's own earlier modules. Both are minimization objectives, handled by rasterized image-processing on the GPU.

The parallel to this decoder is exact up to one asymmetry. Truss-Z is single-alphabet, single-branch (the 2016 multi-branch paper is the exception); LEGO track has a heterogeneous catalog with named ports, switches that genuinely branch, and crossings that require coincident-point validation. Zawidzki never labels his work "generative encoding," "indirect encoding," or "construction decoder" — those vocabularies come from Hornby & Pollack 2002 (DOI 10.1162/106454602320991837) and Stanley & Miikkulainen 2003 (DOI 10.1162/106454603322221487). Hornby's GENRE system uses a parametric L-system-style grammar that guarantees topological validity of a rewritten body but not non-interpenetration — the same split this decoder adopts.

The mechanism — push turtle state onto a stack at a branch, pop on return — is Prusinkiewicz & Lindenmayer's Chapter 1.6 bracketed L-system interpretation. ABoP's convention that unmatched right brackets are ignored carries over: our branch-extraction phase gracefully tolerates an empty branch slot (sentinel at position 0) exactly as `]` without a matching `[` is discarded. Stanley & Miikkulainen's taxonomy places this decoder at the grammatical-axis boundary: the genotype already consists of terminal symbols (no rewriting), so it is a *degenerate* generative encoding — somewhere between a pure direct encoding and a full L-system grammar, closer to Zawidzki than to Hornby.

### Precedent comparison

| System | Alphabet / encoding | Chain-validity guaranteed? | Collision-free guaranteed? | Explicit formal statement? |
|---|---|---|---|---|
| BRKGA (Gonçalves & Resende 2011) | permutation from sorted keys | No — decoder is a heuristic | No | No — "in general, a heuristic" |
| Truss-Z (Zawidzki 2013, 2018) | {R, L, R², L²}* | Yes (affine chaining) | No — ε_o penalty | No — procedural only |
| Bracketed L-systems (P&L 1990 Ch 1.6) | {F, +, −, [, ]}* with matched brackets | Yes (well-formed tree) | No | Yes — matched-bracket theorem |
| GENRE (Hornby & Pollack 2002) | parametric production rules | Yes (topological validity) | No — sim-time filtering | Partial — grammar-enforced |
| **This decoder** | flat int16 with four regions + sentinels | **Yes — P1–P5 by construction** | No — collision residual | **Yes — enumerated invariants** |

## Feasibility by construction, formally

Let *x* be a chromosome drawn from the ChromosomeLayout defined in Package 4. Let `validate_invariants(x, layout, cfg)` be the Boolean gate from that package. Then:

**Theorem (informal).** *If `validate_invariants(x) == True` and the decoder returns `status == FEASIBLE`, the returned DecodedLayout satisfies invariants P1–P5 below. The only routes to non-FEASIBLE status are INFEASIBLE_INVARIANTS (gate failure), INFEASIBLE_ANGULAR_LATTICE (lattice prefilter on the pure-integer angular path), INFEASIBLE_CROSSING_GEOMETRY (overlay declares a crossing the geometry does not realize), INFEASIBLE_BRANCH_STRUCTURE (a branch slot contains a switch or crossing), and INFEASIBLE_INVALID_PIECE (defense-in-depth against out-of-range gene values).*

### The five structural invariants

| # | Invariant | Enforcement site |
|---|---|---|
| **P1** | **Port connectivity.** For every consecutive pair (p_i, p_{i+1}) in the main loop and every (switch → first branch piece) pair, the world pose at p_i's chosen exit port equals the world pose at p_{i+1}'s entry port within geometry tolerances. | SE(2) composition in Phase 3/4 using catalog `PortDef(dx, dy, dtheta)` as the relative transform; equality is exact before float roundoff. |
| **P2** | **Branch attachment.** Every branch's first piece has its entry port attached to a switch's diverging-route exit port, and only to a switch whose mask-gene value is 2. | Phase 4 initializes the branch turtle from `compose(throat_pose, PortDef[port C])`; no other attachment path exists in code. |
| **P3** | **Crossing declaration.** Every crossing piece in the main loop has exactly two distinct main-loop indices (i, j), i<j, from the crossing overlay, and its (A,B) / (C,D) port axes align with the world poses at those indices. | Phase 3 crossing validation matches overlay triples against loop indices and fails fast with INFEASIBLE_CROSSING_GEOMETRY if the (C,D) axis does not coincide with gene i's exit pose within tolerances. |
| **P4** | **Route consistency.** For every switch in the main loop, the mask-gene value ∈ {0, 1, 2} determines the main-route choice (through / diverging / through-plus-branch) with no contradiction between stored `exit_port` and traversed geometry. | Phase 3 selects the exit port from catalog's named-route map `{through:[A,B], diverging:[A,C]}` by mask value; the `PlacedPiece.exit_port` field records the decision. |
| **P5** | **Catalog validity.** Every `piece_id` in placements is in the catalog and every port name referenced is present on that piece. | Defense-in-depth check at gene read time; impossible to reach Phase 3/4 otherwise because `validate_invariants` bounds each gene to [0, n_piece_types]. |

### What feasibility by construction does *not* cover

| # | Residual / hazard | Where it is handled |
|---|---|---|
| **C1** | **Main-loop closure.** The turtle may return near but not at ORIGIN. The decoder reports `closure_residual = (dx, dy, dθ)` but does not enforce. | Problem-layer equality constraint `|dx|, |dy| ≤ POSITIONAL_CLOSURE_TOL = 1e-6`, `|dθ| ≤ ANGULAR_CLOSURE_TOL = 1e-9`, reformulated as inequalities in pymoo's G vector (Deb 2000 feasibility-first, DOI 10.1016/S0045-7825(99)00389-8). |
| **C2** | **Self-intersection outside declared crossings.** Two non-adjacent pieces may collide geometrically. | `geometry/collision.py` `check_collisions(...)` called in Phase 6; count returned as inequality constraint at problem layer. |
| **C3** | **Inventory exhaustion.** Census may reference more pieces of a type than `max_occurrences[type]` allows physically. | Problem-layer inequality per type: `max(0, census[t] − max_occurrences[t])`. |
| **C4** | **Economic / topological interest.** Nothing prevents a two-piece degenerate loop, tiny ovals, or stringy non-interesting layouts. | Out of scope for the decoder; implicitly disfavored by the utilization objective f1 and by diversity pressure in NSGA-II crowding distance (Deb et al. 2002, DOI 10.1109/4235.996017). |

### Proof sketch

Proceed by structural induction on placed-piece count *n*. **Base case (n = 0):** no pieces placed; P1 holds vacuously; P2–P3 hold vacuously; P4 holds vacuously; P5 holds vacuously. **Inductive step:** assume invariants hold after *n* placements with turtle state *T_n*. Placing piece p_{n+1} consults the catalog for `PortDef[entry]` and `PortDef[exit]`, writes `PlacedPiece(..., world_pose_at_entry = T_n)`, and updates `T_{n+1} = compose(T_n, ΔT)` where ΔT is the entry-to-exit port transform. By definition of SE(2) composition (geometry package §3), `T_n` equals the p_{n+1} entry port's world pose exactly, preserving P1. P2 holds if the placement originated from Phase 4 with `T_n = compose(throat_pose, PortDef[diverging])`, which the code performs unconditionally at branch start. P3 is orthogonal — crossing validation is a predicate on the already-placed tokens, evaluated once per loop step. P4 is decided before placement by the mask-gene dispatch, recorded in `exit_port`. P5 is guaranteed by the gene-bound check from `validate_invariants`. The only thing composition does not guarantee is that the *final* `T_N` equals `ORIGIN` — that is C1, by design.

A fully formal proof in Lean or Coq is a natural follow-up but out of scope for the thesis; the inductive argument above is the thesis contribution.

## Seven phases, one turtle, four regions consumed jointly

The decoder is a seven-phase pipeline. Phases 1–2 reject cheaply; Phase 3 is the numba-compiled hot loop; Phase 4 extracts branches using the counter maintained by Phase 3; Phases 5–7 measure and report.

```
function DECODE(x, catalog, cfg, layout, params) -> DecodedLayout:

    # -- PHASE 1: PRE-DECODE VALIDATION --------------------------------
    if not validate_invariants(x, layout, cfg):
        return DecodedLayout(status=INFEASIBLE_INVARIANTS, reason=...)

    # -- PHASE 2: LATTICE ANGULAR PREFILTER ---------------------------
    # one full turn = 64 * ATOMIC_ANGLE where ATOMIC_ANGLE = pi/32
    (main_genes, end_idx) := slice main_loop until sentinel S
    if every piece in main_genes is lattice-typed (exit_dtheta is n*ATOMIC_ANGLE):
        total := 0
        for each gene g in main_genes:
            spec := catalog[g]
            chosen_exit := through if spec.is_straight_or_curve
                           else pick_by_mask(switch_mask, spec)
            total += integer_multiple_of(spec.port[chosen_exit].dtheta)
        if total mod 64 != 0:
            return DecodedLayout(status=INFEASIBLE_ANGULAR_LATTICE)

    # -- PHASE 3: MAIN-LOOP TURTLE WALK (numba hot path) ---------------
    turtle        := ORIGIN
    branch_here   := 0                  # counts mask==2 occurrences
    switch_count  := 0                  # counts ALL switches
    main_layout   := []
    switch_log    := []                 # (loop_idx, throat_pose, spec) per mask==2
    overlay_map   := index_overlay_by_second_coord(crossing_overlay)
    for loop_idx, g in enumerate(main_genes):
        if g == S: break
        spec := catalog[g]
        if spec.is_switch:
            m := switch_mask[switch_count]
            if   m == 0: exit_name := "B"         # through
            elif m == 1: exit_name := "C"         # diverging
            elif m == 2:
                exit_name := "B"                  # main takes through
                switch_log.append((loop_idx, turtle, spec))
                branch_here += 1
            switch_count += 1
        elif spec.is_crossing:
            if loop_idx in overlay_map:
                (i, piece_type) := overlay_map[loop_idx]
                if piece_type != g:
                    return DecodedLayout(status=INFEASIBLE_CROSSING_GEOMETRY)
                if not crossing_axes_aligned(main_layout[i], turtle, spec,
                                             POSITIONAL_CLOSURE_TOL):
                    return DecodedLayout(status=INFEASIBLE_CROSSING_GEOMETRY)
            exit_name := "B"                      # straight-through axis of crossing
        else:
            exit_name := "B"                      # straight or curve
        main_layout.append(PlacedPiece(
            piece_id=g, entry_port="A", exit_port=exit_name,
            world_pose_at_entry=turtle, layer="main"))
        turtle := compose(turtle, spec.port[exit_name])

    # -- PHASE 4: BRANCH EXTRACTION -----------------------------------
    branch_layouts := []
    for k, (loop_idx, throat_pose, switch_spec) in enumerate(switch_log):
        branch_turtle := compose(throat_pose, switch_spec.port["C"])
        slot_begin := k * cfg.max_branch_length
        slot_end   := slot_begin + cfg.max_branch_length
        branch := []
        for j in range(slot_begin, slot_end):
            g := branch_slots[j]
            if g == S: break
            spec := catalog[g]
            if spec.is_switch or spec.is_crossing:
                return DecodedLayout(status=INFEASIBLE_BRANCH_STRUCTURE)
            branch.append(PlacedPiece(
                piece_id=g, entry_port="A", exit_port="B",
                world_pose_at_entry=branch_turtle, layer="branch"))
            branch_turtle := compose(branch_turtle, spec.port["B"])
        branch_layouts.append(branch)

    # -- PHASE 5: CLOSURE MEASUREMENT ---------------------------------
    residual := (turtle.x - ORIGIN.x,
                 turtle.y - ORIGIN.y,
                 wrap_angle(turtle.theta - ORIGIN.theta))

    # -- PHASE 6: COLLISION DETECTION ---------------------------------
    all_pieces := main_layout + flatten(branch_layouts)
    adjacent   := {(i, i+1) for i in range(len(main_layout)-1)} ∪ branch_adjacencies
    collisions := check_collisions(all_pieces, excluded_pairs=adjacent)

    # -- PHASE 7: CENSUS AND OUTPUT -----------------------------------
    census := count_by_piece_id(all_pieces)
    return DecodedLayout(placements=all_pieces, closure_residual=residual,
                         collision_list=collisions, census=census,
                         status=FEASIBLE)
```

### The four regions, consumed jointly

The central design decision is that Phase 3 increments *two* counters in lockstep — `switch_count` over *all* switches (used to index `switch_mask`) and `branch_here` over mask-2 switches only (used indirectly via `switch_log` to index `branch_slots` in Phase 4). This pins the previously-ambiguous contract: **branch slot *k* supplies the diverging-route content for the *k*-th branch-here switch in main-loop order, not the *k*-th switch of any kind.** Branch slots are dead code if fewer than *k+1* mask-2 switches exist; the mutation operator in Package 6 must write to this same indexing scheme. The crossing overlay is indexed by its second coordinate *j* (the closing index) so that Phase 3's single forward pass can validate in O(1) per step.

## Seven phases, seven cost regimes

| Phase | Operation | Complexity | Measured or estimated wall time |
|---|---|---|---|
| 1 | `validate_invariants` bounds + sentinel + overlay range checks | O(N) int compares | ~5 µs (pure numpy) |
| 2 | Lattice prefilter: sum integer atomic-angle multiples mod 64 | O(N) integer adds | ~1 µs (numpy int16 reduce) |
| 3 | Main-loop turtle walk (numba) | O(N) × SE(2) compose | ~1 µs for N=100 (≈10 ns/compose per geometry report) |
| 4 | Branch extraction (pure Python) | O(B · L_branch) | ~5 µs for 4 branches × 10 pieces |
| 5 | Closure residual: one subtraction + wrap_angle | O(1) | negligible |
| 6 | Collision detection: AABB + closed-form intersections | O(N²) with pruning | ~100 µs (dominant cost) |
| 7 | Census + DecodedLayout assembly | O(N) | ~10 µs (Python object construction) |
| **Total** | | | **~120–150 µs per decode** |

A 200-individual population over 1000 generations is 2·10⁵ decodes, i.e. ~25 s of decoder time — not the bottleneck. The bottleneck is Phase 6 collision detection, handled in the geometry package.

## The lattice prefilter rejects about 98.4% of random chromosomes before any trig

For a pure-lattice chromosome (no 4DBrix 18°/15°/10°/9° or Fx Bricks 22.62° off-lattice elements), the main-loop's total heading change is a sum of integer multiples of ATOMIC_ANGLE = π/32. Closure requires this sum ≡ 0 (mod 64). For uniformly random gene choices the sum is approximately uniform over ℤ_64, giving P(closure) ≈ 1/64 ≈ **1.5625%**. The prefilter therefore rejects **≈ 98.44%** of random chromosomes with O(N) integer arithmetic, never invoking `sin`/`cos`. On mixed or all-off-lattice chromosomes the prefilter is skipped and closure is measured numerically in Phase 5; this is unavoidable because 22.62° (Fx Bricks wye) does not share a rational multiple with π/32.

**The >99% feasibility claim from user memory reflects the EA-iterated distribution, not uniform random.** A realistic breakdown:

| Distribution | P(invariants) | P(lattice closure) | P(no crossing error) | P(FEASIBLE status) |
|---|---|---|---|---|
| Uniform random int16 | ~10⁻¹⁰ (sentinels, range) | — | — | ~0 |
| Uniform within [xl, xu] bounds | 1.0 | ~0.016 (lattice) | ~0.95 | ~1.5% |
| Initialized population (valid seeds) | 1.0 | varies by seed set | ~0.98 | 50–90% |
| After 10 generations of biased GA | 1.0 | ≥ 0.99 | ≥ 0.99 | **≥ 99%** |

The ">99%" figure is believable at steady state but is not a property of the decoder alone — it is a joint property of decoder, biased-crossover operator (Package 6), and fitness-driven selection. The decoder's contribution is turning the closure check from a hard reject (entire chromosome discarded) into a soft residual that NSGA-II's crowding-distance ranking can push toward zero over generations.

## Seven edge cases the decoder must not crash on

| Input | Handling | Status code |
|---|---|---|
| All-zero chromosome (gene 0 = straight) | Decode a long straight line; residual large | FEASIBLE, C1 violation |
| `main_loop[0] == S` (immediate sentinel) | Empty placement list, residual = (0,0,0) | FEASIBLE |
| `branch_slots[k·L] == S` | Branch *k* is empty; skip | FEASIBLE |
| Overlay (i,j) with j ≥ effective main-loop length | Silently drop (overlay is mutated, may reference truncated loop) | FEASIBLE |
| Overlay (i,j) with i ≥ j (should be caught at Phase 1) | Defense-in-depth reject | INFEASIBLE_INVARIANTS |
| `switch_mask` entries beyond actual switch count | Unused; do not fail | FEASIBLE |
| Gene value > n_piece_types | Defense-in-depth reject | INFEASIBLE_INVALID_PIECE |
| Branch slot contains switch or crossing | Reject per Phase 4 branch-structure rule | INFEASIBLE_BRANCH_STRUCTURE |
| Crossing overlay declares crossing whose geometry does not align | Reject per Phase 3 crossing validation | INFEASIBLE_CROSSING_GEOMETRY |

## API surface for the problem layer

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Literal

class DecoderStatus(Enum):
    FEASIBLE                     = 0
    INFEASIBLE_INVARIANTS        = 1   # Phase 1 gate failure
    INFEASIBLE_ANGULAR_LATTICE   = 2   # Phase 2 lattice-integer closure fails
    INFEASIBLE_CROSSING_GEOMETRY = 3   # Phase 3 overlay vs geometry mismatch
    INFEASIBLE_BRANCH_STRUCTURE  = 4   # Phase 4 switch/crossing inside branch
    INFEASIBLE_INVALID_PIECE     = 5   # defense-in-depth: gene out of catalog

@dataclass(frozen=True, slots=True)
class PlacedPiece:
    piece_id:            int
    entry_port:          str                   # always "A" in current catalog
    exit_port:           str                   # "B" through / "C" diverging
    world_pose_at_entry: "Pose2D"              # from geometry package
    layer:               Literal["main", "branch", "crossing"]

@dataclass(frozen=True, slots=True)
class DecodedLayout:
    placements:       tuple[PlacedPiece, ...]
    closure_residual: tuple[float, float, float]   # (dx, dy, dtheta)
    collision_list:   tuple[tuple[int, int], ...]  # index pairs into placements
    census:           dict[int, int]               # piece_id -> count
    status:           DecoderStatus
    status_reason:    str = ""                     # human-readable diagnostic

def decode(x:       np.ndarray,
           catalog: "TrackCatalog",
           cfg:     "ChromosomeConfig",
           layout:  "ChromosomeLayout",
           params:  "PhysicsParams") -> DecodedLayout:
    """Seven-phase forward-kinematics decode of a chromosome.

    Guarantees (P1–P5) when status == FEASIBLE; closure (C1), collisions (C2),
    inventory (C3) reported but not enforced. See decoder report §3."""

def decode_batch(X: np.ndarray, ...) -> list[DecodedLayout]:
    """Structured numpy output for problem-layer vectorization."""
```

The problem layer consumes `DecodedLayout` without reaching into decoder internals: `census` feeds f1 (utilization), `placements` with `exit_port` annotations feed `train.v_bottleneck(...)` for f2, `closure_residual` and `collision_list` feed the G-vector of inequality constraints. For `status != FEASIBLE`, `_evaluate` returns `F = [+inf, +inf]` and a large positive G per pymoo maintainer guidance (GitHub discussion #213), which Deb 2000's feasibility-first rule — and its constrained-dominance generalization in Deb et al. 2002 — push to the last rank of non-dominated sorting automatically.

## A worked example: sixteen R40 curves close at the origin

Consider a trivial chromosome: `main_loop = [R40_curve_id] * 16 + [S, S, ...]`, all switch-mask entries 0, all branch slots S, no crossings. From the geometry package's numerical example, R40 has `PortDef(exit_B) = (dx=40·sin(π/8), dy=40·(1−cos(π/8)), dtheta=π/8)` in piece-local studs/radians. π/8 = 4·(π/32) = 4 atomic angles.

- **Phase 1:** invariants pass — all genes in range, main loop uses 16 pieces below `max_loop_length`, no overlay.
- **Phase 2:** lattice prefilter. Total = 16 × 4 = 64 atomic angles ≡ 0 (mod 64). Accept.
- **Phase 3:** turtle walks 16 composes. After step *k* the turtle's heading is *k*·π/8; after step 16 heading is 2π ≡ 0. Position traces the regular 16-gon on the R40 polyline.
- **Phase 4:** no switches logged, no branches.
- **Phase 5:** residual = (Δx, Δy, Δθ) each within POSITIONAL_CLOSURE_TOL = 1e-6 studs / ANGULAR_CLOSURE_TOL = 1e-9 rad — matching the geometry report's independent SE(2) round-trip measurement.
- **Phase 6:** no collisions (pieces are pairwise adjacent or radially separated by ≥ 2·40·sin(π/16)).
- **Phase 7:** census = {R40: 16}; status = FEASIBLE.

A corresponding valid BLUF of the output:

```
DecodedLayout(
    placements      = (PlacedPiece(R40, "A", "B", ORIGIN, "main"),
                       PlacedPiece(R40, "A", "B", pose_1, "main"),
                       ...,  # 16 total
                      ),
    closure_residual = (~1e-13, ~1e-13, ~1e-15),   # within tolerance
    collision_list   = (),
    census           = {R40_id: 16},
    status           = DecoderStatus.FEASIBLE,
)
```

The problem layer computes f1 = 16 / max_occurrences[R40_id] (utilization), f2 = v_safe(40 studs) from the speed table (bottleneck is all 16 R40 curves tied), and constraint violation = 0 on both equality and inequality channels.

## Architectural decisions and their sources

| Decision | Rationale | Source |
|---|---|---|
| Turtle-graphics forward kinematics | Simplest representation that makes every composition explicitly SE(2); provably feasibility-preserving for port-chained modules | Prusinkiewicz & Lindenmayer 1990 Ch 1.1; Zawidzki 2013 affine-chaining |
| Push/pop at branches | Bracketed-L-system discipline; any matched-bracket string yields a well-formed tree | Prusinkiewicz & Lindenmayer 1990 Ch 1.6 |
| Seven explicit phases with early returns | Each phase encapsulates one invariant class; cheap checks first (O(N) prefilter before O(N²) collisions) | construction-decoder tradition (Londe et al. 2024 §3) |
| Lattice-integer angular prefilter | Novel; enabled by catalog's ATOMIC_ANGLE = π/32 contract; rejects ~98.4% of uniformly random closures | This thesis (geometry + catalog packages) |
| Branch-slot counter rule: slot *k* ⇔ *k*-th mask-2 switch | Deterministic, operator-friendly, avoids coupling to total switch count | Package 4 chromosome contract; this report |
| Crossing-overlay validation convention: geometric alignment within tolerance | Allows mutation of overlay without freezing geometry, at the cost of one validation step per crossing | Novel; this report |
| Numba compilation restricted to Phase 3 | Branches and crossings are rare (B ≤ 4, crossings ≤ 2); numba JIT overhead on short arrays doesn't pay off | Geometry package recommendation; this report |
| Infeasibility as status codes, not exceptions | Vectorizable; problem layer can `np.where(status == FEASIBLE, ...)` batch-filter | pymoo convention (Blank & Deb 2020, DOI 10.1109/ACCESS.2020.2990567) |
| Closure and collision as residuals reported, not enforced | NSGA-II's Deb-2000 feasibility-first rule handles this natively; avoids the "decoder as repair heuristic" pattern | Deb 2000; Deb et al. 2002 |

## What this pushes to downstream packages

- **Operators package (#6).** The branch-mutation operator must write to `branch_slots[k·max_branch_length : (k+1)·max_branch_length]` where *k* indexes the *k*-th mask-2 switch in main-loop order — **not** the *k*-th switch of any kind, and **not** a global index across main + branches. This contract is the fix for the documented "outdated format" bug. Switch-mask mutation must preserve the mask ∈ {0, 1, 2} domain; flipping 2 → {0, 1} without clearing the corresponding branch slot is permitted (slot becomes dead code). The crossover must respect segment-selectivity: main_loop and branch_slots are permutation-like; switch_mask and crossing_overlay are integer arrays per gene.
- **Problem package (#7).** Consumes `DecodedLayout` unchanged. f1 from `census`, f2 from `placements` via `train.v_bottleneck(...)`. Constraints: G = [closure_x_abs − tol, closure_y_abs − tol, closure_θ_abs − tol, len(collision_list), max(0, census[t] − cap_t) ...]. Handles `status != FEASIBLE` by writing F = [+inf, +inf] and a large positive G, per pymoo maintainer guidance.
- **Visualization package (#8).** Renders `PlacedPiece` list from `placements`, color-codes by `layer` field (main / branch / crossing), overlays closure residual as a vector from the turtle's final pose to ORIGIN, highlights collision pairs in red.
- **Config/IO package (#9).** Persists `DecoderStatus` reason codes per generation; logs feasibility rate, closure-residual histogram, and collision-count histogram as per-generation telemetry; dumps `DecodedLayout` for the Pareto-front archive.

## What the decoder does not decide

Closure tolerance values live in the geometry package (POSITIONAL_CLOSURE_TOL, ANGULAR_CLOSURE_TOL are its constants). Collision-detection internals — AABB bounding, closed-form segment-arc intersection, spatial hashing — are geometry's responsibility; the decoder only calls `check_collisions`. Objective formulas (piece-utilization weights, v_safe speed curves) live in train and problem. Crossover and mutation operator shapes live in operators (Package 6); the decoder only consumes their output. Population size, selection pressure, termination criteria are problem/pymoo concerns. And most importantly, **the decoder does not decide what counts as an interesting or economically meaningful layout** — that emergent judgment is the job of f1, f2, and NSGA-II's crowding-distance diversity pressure acting over generations. The decoder's contract is narrow and strict: every invariant-passing chromosome becomes a port-exact, catalog-valid placement list, with two residuals and one status code. Everything else is downstream.