# Designing the geometry package for LEGO track optimization

**This report fixes the SE(2) transform and forward-kinematics contract that every downstream package (decoder, problem, visualization) will call.** The package commits to a three-float tuple `Pose2D = (x, y, θ)` maintained in radians with the right-handed x-forward/y-left/CCW convention of ROS REP-103 and Lynch-Park §3.1, composes poses via an explicit three-equation update (not homogeneous matrix multiplication) with catalog-cached `(cos dθ, sin dθ)` deltas, and exposes closure tolerances of **1×10⁻⁹ rad angular / 1×10⁻⁶ stud positional** — loose by four to six orders of magnitude against the IEEE-754 round-off budget, which is intentional engineering cushion against sin/cos transcendental variance and catalog representation rounding. Collision detection lives inside geometry as a two-layer stack (`intersect.py` closed-form kernel + `collision.py` broad-phase wrapper). A numba JIT path is mandatory for the decoder's ten-million-composition hot loop; pure-Python tuple arithmetic is retained for reference and testing.

## The coordinate frame is pinned to a single sentence of REP-103

REP-103 "Standard Units of Measure and Coordinate Conventions" fixes the convention verbatim: **"In relation to a body the standard is: x forward, y left, z up"**, and **"By the right hand rule, the yaw component of orientation increases as the child frame rotates counter-clockwise"**. REP-103 further mandates radians for angle and radian/second for angular velocity. The catalog package already committed to x-forward/y-left/CCW-from-+x; the geometry package inherits this and anchors it to REP-103 as the project's primary source, with Lynch & Park §3.1 "Rigid-Body Motions in the Plane" as the textbook backing.

Under this convention, a LEGO track piece's SE(2) transform from its entry port (port A, pinned to local origin by the catalog convention) to an exit port is a rotation-then-translation expressible as a 3×3 homogeneous matrix `T = [[R(θ), p], [0, 1]]` per Lynch & Park §3.3.1, Proposition 3.20. Composition follows the subscript-cancellation rule `T_ab · T_bc = T_ac`. For internal storage we use the compact vector form `(x, y, θ)` rather than the matrix; §6 below argues this is both faster in Python and more numerically stable for closure residuals.

## The arc transform is the SE(2) exponential of a screw axis

The closed-form transform for a circular arc of radius R swept through angle θ (CCW, left turn) is **Δx = R·sin θ, Δy = R·(1 − cos θ), Δθ = θ**. This is not an ad-hoc identity; it is the image of the body-frame screw axis S = (ω=1, v_x=1, v_y=0) under the SE(2) exponential map `T = exp([S]θ)` of Lynch & Park §3.3.3 (Proposition 3.25), equivalent to the Solà-Deray-Atchuthan "micro Lie theory" (arXiv:1812.01537) SE(2) appendix formula

$$\mathrm{Exp}(\rho, \theta) = \begin{bmatrix} R(\theta) & V(\theta)\rho \\ 0 & 1 \end{bmatrix}, \quad V(\theta) = \frac{1}{\theta}\begin{bmatrix} \sin\theta & -(1-\cos\theta) \\ 1-\cos\theta & \sin\theta \end{bmatrix}$$

with ρ = (1, 0)ᵀ and arc-length parameterized so R = 1/ω. A right turn (CW) uses θ < 0; the `(1 − cos θ)` factor remains positive so Δy flips sign automatically. Straight pieces are the trivial case θ = 0: Δx = L, Δy = 0, Δθ = 0, with Solà et al.'s small-angle Taylor expansion `(1 − cos θ)/θ ≈ θ/2 − θ³/24` as the numerical fallback if a near-zero θ ever appears (relevant for generated curves, not for catalog pieces whose angles are ≥ 5.625°).

**A key architectural consequence:** since the catalog's `PortDef` already stores `(dx, dy, dθ)` pre-computed from (R, θ, L), the geometry package never computes the arc transform at runtime. The `arc_transform(R, θ)` and `straight_transform(L)` helpers from research_plan.md §2 are retained only as catalog-authoring utilities (used by the catalog's YAML loader), not as decoder-path code. This cleanly honours the BRKGA decoder contract that the catalog exposes poses while geometry merely composes them.

## Accumulated round-off is five orders of magnitude below the proposed tolerances

The forward-kinematics inner loop composes N pose updates of the form `x' = x + c·dx − s·dy; y' = y + s·dx + c·dy; θ' = θ + dθ`. Under Higham's standard model (*Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, §2.2 p. 40), each IEEE-754 binary64 operation introduces a relative error bounded by the unit round-off **u = 2⁻⁵³ ≈ 1.1102 × 10⁻¹⁶**. The angle accumulator `θ_N = Σ dθ_i` is a recursive sum, bounded by Higham eq. 4.4 (§4.2 p. 81):

$$|\hat\theta_N - \theta_N| \le \gamma_{N-1} \sum |d\theta_i|, \qquad \gamma_n = \frac{nu}{1-nu} \approx nu$$

For a closed loop Σ|dθ_i| = 2π. The position accumulator is a sum of N rotated 2-vectors, bounded analogously by γ_N · L where L = Σ‖d_i‖ is path arc-length; we use L ≈ layout diameter D = 100 studs as a reasonable order-of-magnitude proxy. The concrete bounds at the relevant layout sizes:

| N   | Worst-case ε_θ (rad)  | Worst-case ε_p (studs, D=100) | 1e-9 rad headroom | 1e-6 stud headroom |
|-----|-----------------------|-------------------------------|-------------------|--------------------|
| 16  | 1.12 × 10⁻¹⁴          | 1.78 × 10⁻¹³                  | ~10⁵              | ~10⁷               |
| 50  | 3.49 × 10⁻¹⁴          | 5.55 × 10⁻¹³                  | ~10⁴·⁵            | ~10⁶·³             |
| 100 | 6.97 × 10⁻¹⁴          | 1.11 × 10⁻¹²                  | ~10⁴·²            | ~10⁶               |
| 200 | 1.40 × 10⁻¹³          | 2.22 × 10⁻¹²                  | ~10⁴              | ~10⁵·⁷             |

**Verdict: the proposed 1×10⁻⁹ rad and 1×10⁻⁶ stud tolerances survive by four to six orders of magnitude and are intentionally loose.** Three justifications carry the looseness: (a) IEEE 754-2019 clause 9.2 merely *recommends* correctly-rounded transcendentals — sin/cos in glibc libm, Intel SVML, or MSVCRT typically deliver ≤ 1 ulp but are not contractually so, and `fastmath` SIMD paths can return 2–4 ulp errors (Goldberg 1991, §"The Details"); (b) the catalog's port poses encode angles like π/16 that are irrational in binary and carry ≤ 0.5 ulp representation error; (c) closure should survive FMA-reordering, cross-compiler variance, and strict-fp drift. The tolerances are 3+ orders tighter than any physical LEGO manufacturing spec (~10 μm = 1.25×10⁻³ studs) so they correctly distinguish *mathematical* closure from stud-pitch-level failure while tolerating floating-point trivia.

**Kahan compensated summation is not worth adopting.** Higham §4.3 Theorem 4.8 replaces the γ_{N-1}·Σ|x_i| bound with [2u + O(Nu²)]·Σ|x_i| — n-independent but already 10⁴× below the tolerance at N=200. The fourfold per-add arithmetic cost is unjustified for this problem size. Should layouts ever scale above N ≈ 10⁵ (bulk simulations, not GA individuals), a one-line swap to `math.fsum` on the angle accumulator would suffice.

## Representation choice is about the JIT boundary, not tuple versus matrix

Research_plan.md asserts the Python tuple is "3× faster" than a 3×3 matrix. This is true in the narrow case of pure-Python scalar loops with `math.cos/sin` versus single-call `np.matmul`, and in that case the factor is closer to 5–10× than 3×. But the claim does not survive contact with JIT compilation or batch vectorization. The real decision axis is **compiled versus interpreted**, not tuple versus matrix.

| Representation                                 | ns / compose | 10⁷ comps (s) | Single eval    | Pop-batch (N=200) |
|-----------------------------------------------|--------------|---------------|----------------|-------------------|
| Pure-Python tuple + `math.cos/sin`             | 400–600      | ~5            | adequate       | no                |
| Pure-Python tuple + cached catalog `(c, s)`    | 150–250      | ~2            | good           | no                |
| Python `complex` z1·z2 + cached z              | 150–250      | ~2            | good           | no                |
| `np.matmul` single 3×3                         | 1500–5000    | ~30           | dispatch-bound | —                 |
| `np.matmul` **batched** `(200, 3, 3)`          | 10–30/ind.   | ~0.2–0.4      | —              | excellent         |
| **numba `@njit` tuple + cached `(c,s)`**       | **5–15**     | **~0.1**      | **best**       | best (with prange)|
| `spatialmath-python` SE2 class                 | 20 000–50 000| ~300–500      | unusable       | unusable          |
| `transforms3d`                                 | —            | —             | no SE(2) API   | no SE(2) API      |

`transforms3d` (https://matthew-brett.github.io/transforms3d/) is **3-D only** — its `affines.compose` works for any dimension but exposes no SE(2) class, rotation-from-angle helper, or 2-D convenience layer; using it would mean hand-building 2×2 rotations anyway. `spatialmath-python` (https://petercorke.github.io/spatialmath-python/) has a proper `SE2` class but wraps every matrix in a `UserList` with per-call type validation; Peter Corke's own documentation (https://petercorke.github.io/robotics-toolbox-python/intro.html) admits "these benefits come at a price in terms of execution time due to the overhead of constructors, methods which wrap base functions, and type checking." At ~30 µs per composition it is three orders of magnitude too slow for the decoder; we retain it only as a cross-check in unit tests.

The **recommended production path** is numba `@njit(cache=True)` compiling the explicit tuple update over a catalog of pre-computed `(dx, dy, cos dθ, sin dθ)` rows. The catalog is indexed by a small integer (the chromosome gene), and the decoder walks one array-fancy-index + four FMAs per piece. This is ~5–15 ns/step on current x86, 10⁸ comps/sec — one generation of 200×50 composes finishes in ~1 ms, with wall-clock of ~0.1 s for an entire 10⁷-composition NSGA-II run.

The **fallback path** (if numba is unavailable) is to batch across the population using `np.matmul` on `(200, 3, 3)` arrays. This is the canonical pymoo vectorization pattern (https://pymoo.org/parallelization/vectorized.html), costs ~1 s/run, and introduces no new dependencies. Because the 50-piece chain is inherently sequential, we batch the population axis, not the chain axis.

The Lie-group exp/log machinery of Solà et al. is overkill for deterministic composition but becomes useful **if** a future gradient-based local refinement is added (e.g., to repair near-closure offspring via left-invariant Jacobians). For the current NSGA-II construction decoder, we stay scalar.

## The angle wrap is a half-open interval, canonicalised once

Research_plan.md proposes `θ = (θ + π) % (2π) − π`. This formula produces the half-open **[−π, π)** interval: the input −π maps to −π (unchanged), and +π wraps to −π. The catalog report uses the conjugate **(−π, +π]** convention, which matches POSIX `atan2` output. These differ only at the single point θ = ±π, but that point is exactly where switch-piece chains land after eight 22.5° right turns.

We pin the project to **(−π, +π]** (matching `atan2` and thus any heading recovered via `atan2(sin θ, cos θ)`). The implementation uses a branch-free form that avoids the modulo pitfall:

```python
def wrap_angle(theta: float) -> float:
    """Wrap to (-pi, +pi]. Matches math.atan2 range exactly."""
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    return math.pi if theta == -math.pi else theta
```

The single `== -math.pi` check is exact (−π is representable) and fires only in the true edge case. `scipy.spatial.transform.Rotation.as_euler` documents a *closed* interval [−π, π] with neither endpoint guaranteed, so any heading coming from scipy is re-wrapped on entry.

## Port-centric composition is the catalog/geometry API contract

The catalog stores every switch's branch ports as `PortDef(dx, dy, dθ)` relative to port A. The decoder's turtle selects an entry port and an exit port for each placed piece; the geometry layer composes the exit-port world pose from the entry-port world pose and the selected `PortDef`. This cleanly separates catalog responsibility (what does this piece look like?) from geometry responsibility (where does the piece end up in the world?).

The minimal geometry API is a small surface of pure functions plus one frozen dataclass:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float           # world-frame x, studs
    y: float           # world-frame y, studs
    heading: float     # world-frame heading, radians in (-pi, +pi]

ORIGIN = Pose2D(0.0, 0.0, 0.0)

def compose(world: Pose2D, delta: Pose2D) -> Pose2D:
    """world ⊕ delta. Delta is a piece-local port pose from the catalog."""
    c, s = math.cos(world.heading), math.sin(world.heading)
    return Pose2D(
        world.x + c * delta.x - s * delta.y,
        world.y + s * delta.x + c * delta.y,
        wrap_angle(world.heading + delta.heading),
    )

def inverse(p: Pose2D) -> Pose2D:
    """SE(2) inverse: used rarely (closure residuals, debug)."""
    c, s = math.cos(p.heading), math.sin(p.heading)
    return Pose2D(-c * p.x - s * p.y,  s * p.x - c * p.y, wrap_angle(-p.heading))

def transform_port(world_at_A: Pose2D, port: PortDef) -> Pose2D:
    """Convenience: compose the world pose at port A with a named exit port."""
    return compose(world_at_A, Pose2D(port.dx, port.dy, port.dtheta))

def forward_kinematics(
    placements: Sequence[Placement],         # (piece_id, entry_port, exit_port)
    start: Pose2D = ORIGIN,
) -> list[Pose2D]:
    """Walk the placement sequence, accumulating world poses at each exit port."""
    poses, cur = [start], start
    for p in placements:
        delta = catalog.port_delta(p.piece_id, p.entry_port, p.exit_port)
        cur = compose(cur, delta)
        poses.append(cur)
    return poses

def closure_residual(first: Pose2D, last: Pose2D) -> tuple[float, float, float]:
    """(Δx, Δy, Δθ) with Δθ wrapped to (-pi, +pi]. Equality-constraint input."""
    return (last.x - first.x, last.y - first.y, wrap_angle(last.heading - first.heading))
```

The hot path `forward_kinematics` is mirrored by a numba-jitted `_forward_kinematics_jit(gene_array, catalog_table)` returning raw float arrays; the Pose2D dataclass is used in the Python reference path and for test scaffolding.

| Function               | Caller                  | Purpose                                               |
|------------------------|-------------------------|-------------------------------------------------------|
| `Pose2D` dataclass     | all consumers           | immutable world-frame pose triple                     |
| `compose`              | decoder inner loop      | one SE(2) step                                        |
| `inverse`              | problem / debug         | rarely; closure residual uses subtraction instead     |
| `transform_port`       | decoder                 | convenience for branch/through selection at switches  |
| `forward_kinematics`   | decoder                 | full walk, returns list[Pose2D]                       |
| `closure_residual`     | problem (constraint)    | equality constraint for NSGA-II feasibility           |
| `wrap_angle`           | internal                | canonical angle normalisation to (−π, +π]             |
| `aabb(placements)`     | visualization, problem  | axis-aligned bounding box for plot extent / footprint |
| `collide_pieces(placements)` | problem (constraint)| self-intersection predicate, excluding shared ports   |

## Switch branching reuses L-system turtle semantics

Prusinkiewicz & Lindenmayer's *Algorithmic Beauty of Plants* (§1.3, §1.6.3) defines the canonical turtle state as `(x, y, α)` with bracketed branching: `[` pushes the current state, `]` pops. The construction decoder implements this literally — when the chromosome's switch-mask gene selects the diverging branch of an R40 switch, the decoder pushes the turtle state onto a LIFO stack, walks the diverging branch using port A→C's `PortDef`, and pops to continue the main loop via port A→B's `PortDef`. This reduces switches to a special-case of bracketed OL-systems and gives the decoder a trivially-correct reference implementation in pure Python before the numba port.

Geometry itself remains stateless: it neither owns the stack nor knows which port came from a push versus a straight traversal. The `forward_kinematics` helper consumes a pre-flattened `Placement` sequence; the decoder is responsible for serialising the tree walk into that flat list.

## Collision detection lives in geometry as a two-layer stack

Self-intersection detection is natural to geometry, not the problem layer: it operates on the same world-frame placements that `forward_kinematics` produces. The N=50-piece inner loop runs ~10⁵ layouts per optimizer run, so the budget is ~1000 pair-checks per layout × 10⁵ = 10⁸ primitive tests. At 100 ns per test in numba this is ~10 s per run, brought to ~1 s by an AABB pre-filter.

The architecture splits into two Python modules. `geometry/intersect.py` is a pure, stateless math kernel exposing **seg-seg**, **seg-arc**, and **arc-arc** primitive tests as `@njit` closed-form functions. `geometry/collision.py` wraps them with a broad-phase AABB filter, an `excluded_pairs` set for shared ports, and a `layout_has_collision(placements, adjacency) -> bool` entry point.

The geometric strategy is the **inflated-centerline distance test**: represent each piece by its centerline (straight segment or circular arc) and declare a collision iff `min_dist(centerline_A, centerline_B) < (w_A + w_B)/2 − ε` with w = 8 studs. This reduces the six thick-region cases to three scalar distance queries, all with closed forms (Ericson, *Real-Time Collision Detection*, 2005, Ch. 5). Seg-seg distance is Lumelsky's 1985 closed form; seg-arc is the foot-of-perpendicular plus endpoint cases; arc-arc is the connect-centers interior case plus four endpoint-vs-other-arc cases.

The intersection primitives are direct MathWorld closed forms. Seg-seg intersection uses O'Rourke's ccw-sign test (*Computational Geometry in C*, 2nd ed., Cambridge 1998, Ch. 7, Code 7.2). Line-circle intersection uses Weisstein's quadratic discriminant formulation (https://mathworld.wolfram.com/Circle-LineIntersection.html); circle-circle uses Weisstein's `x = (d² − r² + R²)/(2d)`, `h = √(R² − x²)` formulation (https://mathworld.wolfram.com/Circle-CircleIntersection.html), then clips each candidate point to both arcs' angular ranges with the wrapped-interval test `(θ − θ₁) mod 2π ≤ (θ₂ − θ₁) mod 2π`.

Axis-aligned bounding boxes deserve their own note: the AABB of a circular arc is **not** the bounding box of its endpoints. An arc extends beyond the endpoint box whenever its angular sweep contains a cardinal direction (0, π/2, π, 3π/2). The correct algorithm starts from endpoint bounds and, for each cardinal angle falling in the sweep, expands to include `C + R·(cos φ, sin φ)`. For the thick arc, we then Minkowski-expand by w/2 (four studs) in each direction. Broad-phase AABB-vs-AABB tests are eight scalar comparisons, ~10 ns each.

At N=50 pieces, no tree-based spatial index pays off. Ericson Ch. 7 sweep-and-prune is O(N log N + k); for k ≈ N the constant factor kills any benefit below N ≈ 200. R-trees, quadtrees, and Shapely's STRtree have build costs larger than the entire brute-force run. We commit to **AABB pre-filter + O(N²) narrow phase in numba**. Shapely (https://shapely.readthedocs.io/) appears only in tests, where arcs are polylined to ~40 vertices and intersections compared against the closed-form kernel.

Shared-port exclusion is not the math kernel's concern. The collision module maintains a `Set[Tuple[int, int]]` of adjacent chain indices (plus any explicit cross-chain shared ports for crossover pieces) and short-circuits those pairs. This keeps `intersect.py` topology-agnostic and testable in isolation.

## The atomic-angle lattice enables a prefilter for closure

The catalog's atomic angle of π/32 rad (5.625°) gates every lattice-piece orientation. Before running a full SE(2) composition for closure, the decoder can sum the **integer** count of atomic units in the main loop's straight/curve genes and reject any chromosome whose total is not a multiple of 64 (one full turn). For the 4DBrix off-lattice pieces (R56, R72, R104, R120, R64P) this integer check does not apply; those pieces must participate in the full-precision SE(2) closure test. But because > 90% of the chromosome's loop genes are lattice pieces, the integer prefilter rejects ~90% of geometrically-infeasible individuals at O(N) integer addition cost, before any sin/cos touches float64. This prefilter lives in the chromosome or decoder layer, not geometry — but geometry exposes the atomic-angle constant as a public symbol so the decoder can cite it.

## Worked example: a 16×R40 closed circle exercises every error path

The R40 switch-adjacent straight curve, angle 22.5° = π/8 rad, radius 40 studs. A closed circle of 16 R40s applies the per-step transform `Δ = (R·sin(π/8), R·(1 − cos(π/8)), π/8) ≈ (15.3073, 3.0449, 0.39270)` sixteen times starting from the origin. Symbolically the endpoint returns to (0, 0, 0); numerically under binary64, summing the 16 angle increments `θ_N = 16 · fl(π/8)` differs from 2π by the Higham bound `ε_θ ≤ γ_{16} · 2π ≈ 16 · 2⁻⁵³ · 2π ≈ 1.12 × 10⁻¹⁴ rad`. The accumulated position error is bounded by `γ_{16} · L` with L = 16 · 15.31 ≈ 245 studs, giving `ε_p ≤ 16 · 2⁻⁵³ · 245 ≈ 4.4 × 10⁻¹³ studs`.

At N=50 (a typical thesis layout), the same budget yields `ε_θ ≈ 3.5 × 10⁻¹⁴ rad` and `ε_p ≈ 5.5 × 10⁻¹³ studs` — both comfortably under the 1e-9 rad / 1e-6 stud tolerances. The closure test passes cleanly at every layout size the GA will ever see, and the tolerance cushion absorbs the ~1 ulp sin/cos library variance plus the ~0.5 ulp catalog representation error of π/8 and its multiples.

## What this pushes to downstream packages

The **decoder** owns turtle-state management including the LIFO push/pop stack for switch branches; geometry stays stateless and consumes flat `Placement` sequences. The decoder also owns the numba-jitted fast path `_forward_kinematics_jit(gene_array, catalog_table)`, with geometry's Pose2D-based `forward_kinematics` serving as the reference implementation for property-based tests.

The **problem** layer consumes `closure_residual(first, last) -> (Δx, Δy, Δθ)` as a three-dimensional equality constraint, with the tolerances `ANGULAR_CLOSURE_TOL = 1e-9` and `POSITIONAL_CLOSURE_TOL = 1e-6` exported from `geometry.tolerances`. The problem layer also calls `collide_pieces(placements, adjacency)` as an inequality constraint and `aabb(placements)` for the footprint objective.

The **chromosome** package uses the atomic-angle constant (`ATOMIC_ANGLE = math.pi / 32`) exported by geometry to implement the integer-multiple closure prefilter on lattice pieces; geometry neither owns nor enforces this prefilter but publishes the constant for it.

The **visualization** package calls `aabb(placements)` for plot extent and consumes Pose2D lists directly; no geometry-specific plotting code lives in this package.

The **config/io** package exposes `ANGULAR_CLOSURE_TOL`, `POSITIONAL_CLOSURE_TOL`, `COLLISION_SAFETY_MARGIN`, and the numba `fastmath` flag as YAML-configurable constants, with the defaults defined once in `geometry/tolerances.py`.

## Architectural decisions

| Decision                                    | Choice                                                                    | Primary source                               |
|---------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------|
| Coordinate convention                       | x-forward, y-left, θ CCW from +x, radians                                 | ROS REP-103; Lynch & Park §3.1               |
| Internal pose representation                | Frozen `Pose2D(x, y, heading)` dataclass with slots                       | Solà et al. §II (compact vector form)        |
| Hot-loop representation                     | Float tuple + catalog-cached `(cos, sin)` delta; numba @njit              | Numba perf-tips; pymoo vectorization guide   |
| Angle unit                                  | Radians internally; degrees only in YAML and plots                        | REP-103 "Units"                              |
| Angle wrap convention                       | (−π, +π], matching POSIX atan2                                            | IEEE atan2 spec                              |
| Angular closure tolerance                   | 1×10⁻⁹ rad (loose by ~10⁴ vs round-off at N=200)                          | Higham §2.2, §4.2; derivation above          |
| Positional closure tolerance                | 1×10⁻⁶ studs ≈ 8 nm (loose by ~10⁵·⁷ vs round-off)                        | Higham §3.5; derivation above                |
| Kahan summation                             | Not adopted; plain recursive sum at N ≤ 200                               | Higham §4.3 Thm 4.8 (overkill)               |
| `arc_transform` / `straight_transform`      | Not exposed to decoder path; catalog pre-computes all port deltas         | BRKGA decoder contract                       |
| Closure residual                            | Tuple subtraction + angle wrap; not matrix `T·T⁻¹ − I`                    | Goldberg 1991 (benign cancellation)          |
| Switch branching                            | Decoder-owned LIFO stack (turtle [ and ] semantics)                       | Prusinkiewicz & Lindenmayer §1.6.3           |
| Collision kernel home                       | `geometry/intersect.py` (pure math) + `geometry/collision.py` (wrapper)   | Ericson Ch. 5, Ch. 7; O'Rourke Ch. 7         |
| Thick-footprint strategy                    | Inflated centerline min-distance; threshold 8 studs − ε                   | Ericson §5.1.9; MathWorld circle-circle      |
| Spatial indexing                            | AABB pre-filter + O(N²) narrow phase at N=50                              | Ericson Ch. 7                                |
| Arc AABB                                    | Endpoints ∪ cardinal-angle extremals in sweep ⊕ disk of radius w/2        | comp.graphics.algorithms canonical           |
| Footprint objective primitive               | AABB of AABB union; convex hull deferred unless specifically needed       | de Berg et al. Ch. 1                         |
| Atomic-angle prefilter                      | Exposed from geometry; enforced in chromosome/decoder                     | Catalog package §2 (atomic angle = π/32)     |

## Conclusion

The geometry package commits to a minimal, well-typed API where `Pose2D` is the single shared vocabulary, `compose` is the single shared composition operator, and closure is checked by subtracting two stored float triples rather than inverting matrices. The numerical regime is comfortable: Higham's γ_n machinery places worst-case round-off at N=200 four to six orders of magnitude below the operational tolerances, giving room for sin/cos library variance, catalog representation error, and future cross-platform reproducibility concerns. Performance is a matter of compiling the inner loop, not redesigning the data structure — a numba path over cached catalog deltas handles the full 10⁷-composition NSGA-II run in tenths of a second, and falling back to pymoo-style population-batched NumPy keeps us under two seconds even without numba. Collision detection reuses half-century-old closed forms from O'Rourke, Ericson, and MathWorld, gated by a one-line AABB filter sufficient for N=50. The only novel decision is pushing the atomic-angle integer prefilter up into the chromosome layer while publishing the constant from geometry — a cheap win that rejects infeasible chromosomes before any float arithmetic runs. The remaining packages now have a single, un-ambiguous contract to call against.