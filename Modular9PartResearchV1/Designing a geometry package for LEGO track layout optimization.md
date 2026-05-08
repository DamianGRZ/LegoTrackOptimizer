# Designing a geometry package for LEGO track layout optimization

**The optimal architecture uses SE(2) rigid-body transforms in compact (x, y, θ) representation, composed via a stateless pure-function API, with a frozen `Pose2D` dataclass at its core.** This approach outperforms 3×3 matrix representations on every metric that matters for this system: memory (3 floats vs. 9), clarity (direct access to position and heading), numerical stability (no orthogonality drift — only angle wrapping needed), and performance. A 16-piece R40 closure loop achieves **~1e-13 stud** position error in double precision, well within the ~1e-6 tolerance target. The critical architectural insight is that forward kinematics across an entire NSGA-II population can be vectorized by looping over pieces (20–100 iterations) while vectorizing across layouts (100–1,000 simultaneously) — turning O(50,000) Python iterations into O(50) NumPy-accelerated steps.

---

## SE(2) composition is the mathematical backbone

SE(2), the Special Euclidean group in 2D, is the group of orientation-preserving isometries of ℝ² — rotations combined with translations. It has **3 degrees of freedom** and is topologically homeomorphic to ℝ² × S¹. Every track piece maps to a fixed SE(2) element describing the rigid-body transform from its entry port to its exit port.

The **homogeneous 3×3 matrix representation** encodes an SE(2) element as:

```
    ┌ cos θ  -sin θ   x ┐     ┌ R(θ)   p ┐
X = │ sin θ   cos θ   y │  =  │ 0ᵀ     1 │
    └   0       0      1 ┘
```

However, the **compact (x, y, θ) tuple** is the superior representation for this domain. The composition formula — verified from van Goor's SE(2) tutorial, LaValle's *Planning Algorithms* §4.2, and Eade's Lie group reference — computes the product of two poses (x₁, y₁, θ₁) ∘ (x₂, y₂, θ₂) as:

```
x_new = x₁ + x₂·cos(θ₁) − y₂·sin(θ₁)
y_new = y₁ + x₂·sin(θ₁) + y₂·cos(θ₁)
θ_new = θ₁ + θ₂
```

This follows directly from expanding **p₁ + R(θ₁)·p₂** in the matrix product. The inverse of (x, y, θ) is **(-x·cos θ − y·sin θ, x·sin θ − y·cos θ, −θ)**, derived from −R(−θ)p. Both formulas require only 2 trigonometric evaluations and a handful of multiplies/adds — faster than a full 3×3 matrix multiply.

**The Lie algebra se(2)** — the tangent space at the identity — consists of 3×3 matrices with an angular velocity ω and translational velocity v. The exponential map from se(2) to SE(2) uses the **V matrix** (sometimes called the left Jacobian):

```
exp(ω, v) = (R(ω), V·v)  where  V = (1/ω)·[sin ω, -(1-cos ω); (1-cos ω), sin ω]
```

When ω = 0, V reduces to I₂ (pure translation). The logarithmic map inverts this. These maps become relevant only if you need optimization on the SE(2) manifold, interpolation between poses, or uncertainty propagation — none of which the forward-kinematics layer requires. For this package, the Lie perspective is mathematically elegant context but **practically unnecessary**; the trig-based composition formula is simpler, faster, and equally correct.

| Criterion | 3×3 matrices | (x, y, θ) tuples |
|---|---|---|
| Memory per pose | 9 floats (72 bytes) | 3 floats (24 bytes) |
| Composition cost | 9 muls + 6 adds | 2 trig + 4 muls + 4 adds |
| Normalization needed | Re-orthogonalize R | Wrap θ to [−π, π) |
| Numerical stability | Orthogonality drift accumulates | Only angle magnitude grows |
| API clarity | Requires indexing into matrix | Direct `.x`, `.y`, `.heading` |
| Batch vectorization | `np.einsum` / `@` | Vectorized trig formulas |

The tuple representation wins across the board for this application.

## Arc transforms derive from constant-curvature integration

Every curved track piece traces a circular arc of radius R through sweep angle θ. The closed-form SE(2) delta for each arc type is derived by integrating the body-frame velocity along the arc. For a body starting at the origin facing +x, the instantaneous center of curvature (ICC) for a **left turn** sits at (0, +R). Tracing the circle:

```
Δx = R·sin(θ)
Δy = R·(1 − cos(θ))        ← left turn (positive, heading increases)
Δheading = +θ

Δy = −R·(1 − cos(θ))       ← right turn (negative, heading decreases)
Δheading = −θ
```

These assume the coordinate convention x = forward, y = left, heading measured CCW from the x-axis — consistent with ROS REP 103 and standard mobile robotics. **Arc length** is simply L = R·|θ| by definition of radian measure, and curvature is κ = 1/R (constant).

A **straight piece** of length L is the degenerate case R → ∞ with Rθ = L held constant: Δx = L, Δy = 0, Δheading = 0. This means the same compose function handles both straights and curves uniformly — piece type determines only the delta values, not the composition logic.

For **visualization discretization**, the number of interpolation points along an arc should be `max(2, ceil(|θ| / step))` where step ≈ π/32 (~5.6°). This yields 64 segments per full circle — the chord-to-arc sag at R = 40 studs is only R·(1 − cos(π/64)) ≈ **0.048 studs** (~0.38 mm), sub-pixel for any reasonable render scale. CAD systems typically use a sag tolerance, maximum chord length, or maximum angular step; π/32 per segment satisfies all three for train track radii.

The **tangent direction** at parameter t along an arc is simply the current heading: unit tangent = (cos(heading₀ + t), sin(heading₀ + t)). Position at parameter t in the local frame is (R·sin(t), ±R·(1 − cos(t))).

## Forward kinematics is serial-chain composition with a tree extension

Each track piece is a "link" with a fixed SE(2) transform, and assembling a layout is the forward-kinematics problem from robotics — directly analogous to the Denavit-Hartenberg formulation, but simpler since all "joints" are frozen (no variable parameters). The global pose of piece *i* is:

```
T_global[i] = T_global[i−1] ∘ T_local[i]
```

The most natural Python implementation uses **`itertools.accumulate`** with the compose function:

```python
from itertools import accumulate

poses = list(accumulate(deltas, compose, initial=Pose2D(0, 0, 0)))
```

This yields all intermediate poses in a single O(N) pass — exactly what visualization needs. For closure detection (only the final pose matters), `functools.reduce` computes the endpoint directly.

**Switch pieces create tree-structured layouts.** The robotics analogy shifts from serial manipulators to humanoid robots — kinematic trees where the torso is the root and limbs are branches. The standard approach, borrowed from **L-systems** in procedural graphics, uses stack-based traversal: encountering a switch pushes the current pose onto an explicit stack, traversal follows one branch to completion, then pops to follow the diverging branch. This is iterative (no recursion-depth limits), O(1) memory per branch point, and naturally compatible with the stateless composition model — each `pop` simply restores a previously computed immutable pose.

Robotics libraries like **Pinocchio** (Carpentier et al., IEEE SII 2019) and **KDL** (Orocos project, used in ROS) implement tree-structured forward kinematics by traversing from root to each leaf, composing transforms along each path. The same pattern applies here: each branch shares a common prefix of composed transforms up to the switch point.

## Floating-point error stays well within tolerance for realistic chains

The critical question: after composing N SE(2) transforms in double precision (IEEE 754, machine epsilon ε ≈ 2.22 × 10⁻¹⁶), how much position error accumulates?

**Error bounds for the 16-piece R40 closure test** (16 × 22.5° = 360°): Each piece's delta involves R = 40 studs, θ = π/8. The circle traced has diameter ~2R = 80 studs. Each composition step performs ~6 floating-point operations with per-operation error ~ε × operand magnitude. The accumulated error over 16 steps:

- **Worst-case position error**: ~N · ε · max_coordinate ≈ 16 × 2.22 × 10⁻¹⁶ × 80 ≈ **2.8 × 10⁻¹³ studs**
- **Typical (RMS) position error**: ~√N · ε · max_coordinate ≈ 4 × 1.8 × 10⁻¹⁴ ≈ **7 × 10⁻¹⁴ studs**
- **Angle accumulation error**: 16 additions of π/8, error ≈ 16 · ε · π/8 ≈ **1.4 × 10⁻¹⁵ rad**

The target tolerances of ~1e-6 studs and ~1e-9 radians sit **6–7 orders of magnitude** above the actual errors. Double precision is more than sufficient; no compensated arithmetic is needed for chains under a few hundred pieces.

**Catastrophic cancellation** in the expression (1 − cos θ) is a well-known numerical hazard: for small θ, cos θ ≈ 1, and the subtraction loses significant digits. The fix uses the identity **1 − cos θ = 2·sin²(θ/2)**, which avoids subtracting near-equal quantities. The cancellation becomes significant when θ < √(2ε) ≈ 2.1 × 10⁻⁸ radians. For standard track pieces (θ ≥ 22.5° ≈ 0.393 rad), this is irrelevant — but the `2·sin²(θ/2)` form should be used defensively in the arc-delta computation to handle any future near-straight custom pieces correctly:

```python
def arc_delta(R: float, theta: float, sign: int) -> Pose2D:
    """Numerically stable arc SE(2) delta. sign: +1 left, -1 right."""
    half = theta / 2
    dx = R * math.sin(theta)
    dy = sign * R * 2.0 * math.sin(half) ** 2   # avoids 1-cos(θ)
    return Pose2D(dx, dy, sign * theta)
```

**Kahan summation** reduces accumulated rounding in long sums from O(N·ε) to O(ε) — effectively independent of chain length. It applies directly to **angle accumulation** (θ_total = Σ Δθᵢ) and can be adapted for x and y coordinate accumulation. For chains under ~200 pieces, the benefit is marginal (standard summation already yields ~10⁻¹³ errors), but Kahan summation costs almost nothing and provides a safety margin for pathological layouts.

**Angle wrapping to [−π, π)** is the recommended convention: it matches `math.atan2` output, naturally represents signed turns (left = positive, right = negative), and simplifies closure detection (check if final heading ≈ 0 mod 2π). The robust implementation is `atan2(sin θ, cos θ)` rather than the modular arithmetic version `(θ + π) % (2π) − π`, as `atan2` handles edge cases at ±π correctly.

## The Pose2D frozen dataclass is the atomic type

The immutable value type pattern anchors the entire API:

```python
@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    heading: float
```

`frozen=True` makes instances immutable and auto-generates `__hash__`, enabling use as dictionary keys and in sets — critical for memoization caches. `slots=True` (Python 3.10+) eliminates per-instance `__dict__`, reducing memory ~35% and accelerating attribute access. One caveat: frozen dataclass construction is ~2.4× slower than mutable due to the custom `__setattr__` used during `__init__`. For hot paths creating millions of poses, the batch NumPy path (operating on arrays rather than individual Pose2D objects) avoids this overhead entirely.

The **stateless functional design** — `pose_new = compose(pose_old, delta)` — is strongly preferred over a mutable turtle for this optimization context. Pure functions guarantee determinism (identical genomes always produce identical layouts), enable memoization (`@lru_cache` on frozen-dataclass inputs), are trivially testable (no state setup/teardown), and are inherently thread-safe for potential future parallelization. The turtle metaphor remains valuable as a *mental model* for how track assembly works — it just shouldn't be implemented as mutable state.

## Batch vectorization uses the "loop over pieces, vectorize over layouts" pattern

The NSGA-II population requires computing forward kinematics for hundreds to thousands of layouts simultaneously. The key insight is that the chain length (20–100 pieces) is small and inherently sequential, but the batch dimension (100–1,000 layouts) is large and embarrassingly parallel. The optimal strategy: **a Python loop of ~50 iterations, where each iteration performs fully vectorized NumPy operations across all layouts**.

```python
def cumulative_compose_batch(deltas: np.ndarray) -> np.ndarray:
    """
    deltas: shape (N_layouts, N_pieces, 3) — per-piece (dx, dy, dθ)
    Returns: poses shape (N_layouts, N_pieces+1, 3) — cumulative global poses
    """
    N_layouts, N_pieces, _ = deltas.shape
    poses = np.zeros((N_layouts, N_pieces + 1, 3))
    for i in range(N_pieces):
        c = np.cos(poses[:, i, 2])          # shape (N_layouts,)
        s = np.sin(poses[:, i, 2])
        poses[:, i+1, 0] = poses[:, i, 0] + c * deltas[:, i, 0] - s * deltas[:, i, 1]
        poses[:, i+1, 1] = poses[:, i, 1] + s * deltas[:, i, 0] + c * deltas[:, i, 1]
        poses[:, i+1, 2] = poses[:, i, 2] + deltas[:, i, 2]
    return poses
```

Each loop body executes 2 trig calls and 8 elementwise operations on arrays of length N_layouts — all dispatched to optimized C/BLAS code inside NumPy. For 1,000 layouts × 50 pieces, this replaces 50,000 Python-level iterations with 50 vectorized steps, yielding roughly **100–500× speedup** over a naive per-layout Python loop.

The delta arrays themselves are computed from the piece catalog via advanced indexing:

```python
# piece_indices: shape (N_layouts, N_pieces), integer type IDs
dx_all = catalog_dx[piece_indices]     # shape (N_layouts, N_pieces)
dy_all = catalog_dy[piece_indices]
dh_all = catalog_dheading[piece_indices]
deltas = np.stack([dx_all, dy_all, dh_all], axis=-1)
```

Memory layout uses `(N_layouts, N_pieces, 3)` with the last axis storing (x, y, θ) — contiguous in C order, cache-friendly for the sequential access pattern. This is **3× more memory-efficient** than the matrix alternative `(N_layouts, N_pieces, 3, 3)`.

For the matrix representation path (if ever needed), batched matrix multiplication uses either `np.matmul` with broadcasting or `np.einsum('...ij,...jk->...ik', A, B)`. There is no built-in `np.cumprod` for matrices; the same loop-with-vectorization pattern applies. For GPU/TPU acceleration, `jax.lax.associative_scan` implements a **parallel prefix scan** that computes cumulative matrix products in O(log N) parallel depth, but this is unnecessary for chain lengths under 100.

## Multi-port pieces and bounding boxes complete the geometry layer

**Switch pieces** have 3+ ports, each defined as an SE(2) transform relative to the piece origin. The catalog stores these as a dictionary:

```python
@dataclass(frozen=True, slots=True)
class PieceType:
    name: str
    ports: dict[str, Pose2D]   # "entry" → Pose2D(0,0,0), "exit" → Pose2D(dx,dy,dθ), etc.
```

Connection logic when piece B's entry attaches to piece A's exit: `global_pose_B = compose(global_exit_A, inverse(entry_port_B))`. For the common convention where entry is always at the origin with heading 0, the inverse is the identity and this simplifies to direct composition.

**Tree traversal** for layouts with switches uses an explicit stack, directly paralleling L-system bracket notation (`[` = push, `]` = pop):

```python
def assemble_tree(pieces, catalog):
    stack = []
    current = Pose2D(0, 0, 0)
    poses = []
    for spec in pieces:
        ptype = catalog[spec.type_id]
        if spec.is_branch_start:
            stack.append(current)
        elif spec.is_branch_return:
            current = stack.pop()
        poses.append(current)
        current = compose(current, ptype.ports[spec.exit_port])
    return poses
```

**Bounding boxes** for arc pieces require checking not just endpoints but also cardinal-direction extrema within the arc sweep. The arc center in global coordinates is offset perpendicularly from the piece origin by ±R. The algorithm tests whether any of the four cardinal angles (0°, 90°, 180°, 270°) fall within the arc's angular sweep; if so, the corresponding extremal point (center ± R along that axis) is included. For batch computation during fitness evaluation, a fast approximation using just piece-center positions (ignoring local extents) gives layout width/height via `np.min`/`np.max` along the piece axis.

## The library decision: roll your own with NumPy + math

The Python ecosystem offers several SE(2)-capable libraries: **manif** (artivis, C++/pybind11, MIT license) is the most complete — full SE(2) with analytical Jacobians, exp/log, adjoint, well-documented with SE(2) localization examples. **PyMLG** (decargroup, successor to liegroups) offers a stateless static-class API with NumPy/JAX/PyTorch backends. **Sophus** (strasdat, ~2.4k GitHub stars) is in maintenance mode with SE(2) support via third-party Python bindings (SophusPy). **SciPy's `spatial.transform.RigidTransform`** is 3D-only (SE(3)) with no 2D support. **pylie** (van Goor) and **jaxlie** (brentyi) offer research-oriented SE(2) implementations.

For this geometry package, **none of these libraries are needed**. The SE(2) operations required — compose, inverse, transform_point, arc_delta — total roughly 30 lines of pure Python/NumPy. Adding a library dependency for this would introduce build complexity, version coupling, and conceptual overhead without meaningful benefit. The Lie algebra operations (exp, log, adjoint, Jacobians) that justify libraries like manif are irrelevant to forward kinematics of fixed-transform chains.

The recommended approach: implement SE(2) composition directly in ~50 lines, test rigorously against known invariants (composition associativity, inverse correctness, 16-piece loop closure), and keep the code readable enough that the mathematical formulas are transparent in the source.

## Conclusion

The geometry package should be built on three pillars: a **frozen `Pose2D` dataclass** as the immutable pose type, a **stateless `compose` function** implementing the SE(2) composition formula, and a **batch `cumulative_compose_batch` function** using the loop-over-pieces/vectorize-over-layouts pattern. The arc-delta computation should use `2·sin²(θ/2)` for defensive numerical stability, angles should be wrapped to [−π, π), and closure tolerance of **1e-6 studs / 1e-9 radians** provides a comfortable 6-order-of-magnitude margin above actual double-precision errors. No external Lie group library is needed — the mathematical operations are simple enough that a self-contained implementation with good test coverage is the right engineering choice. The entire pure-math layer can be implemented in under 200 lines of Python with zero dependencies beyond NumPy and the standard library.