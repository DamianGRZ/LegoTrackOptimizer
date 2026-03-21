# Physics model for LEGO locomotive derailment and speed optimization

**A LEGO City locomotive weighing 330–400 g on standard R40 curved track (320 mm radius) hits a lateral-friction derailment limit near 1.0 m/s—roughly two-thirds of its 1.57 m/s no-load top speed.** This means derailment physics actively constrains performance on any layout with standard curves, making a quasi-static model both necessary and highly informative for genetic algorithm optimization. The binding constraint is lateral sliding (not tip-over), aerodynamic drag is negligible, and the forward-backward pass algorithm computes time-optimal speed profiles in O(N) time. Below is every parameter, formula, and integration pattern needed to build this system.

---

## LEGO train physical parameters: the measured data

The foundation of the physics model rests on empirical values gathered primarily from Philo's motor testing (philohome.com), the L-Gauge community standard (l-gauge.org), and component weight measurements.

**Mass.** No community source has directly weighed an assembled LEGO City locomotive (sets 60197, 60198, 60336, 60337), but component-level data yields a reliable estimate. The Powered Up hub (88009) with 6×AAA batteries weighs approximately **150–165 g**, the Powered Up train motor (88011) weighs **57 g**, and a typical City locomotive brick body (100–150 pieces) adds **120–170 g**. Total assembled locomotive mass: **330–400 g**, with ~370 g as a good central estimate for modeling. The hub-plus-batteries represent roughly 40–45% of total mass and sit at the bottom of the assembly, which matters for center-of-gravity estimation.

**Track gauge and wheel geometry.** The L-Gauge standard defines the centerline gauge at **5 studs = 40 mm** and the inner rail-to-rail gauge at **37.5 mm**. Overall track width including sleepers is 8 studs (64 mm). The powered bogie uses 55423-type Technic train wheels with a rubber friction band: **flange diameter 24 mm, tread (rolling) diameter 17.0 mm, tread width 2.5 mm**. Unpowered rolling stock uses 2878-type wheels with a slightly smaller rolling diameter of 16.5 mm. Each locomotive has two 4-wheel bogies (8 wheels total), with one bogie motorized.

**Center of gravity height.** No direct measurement exists in community literature. Geometric estimation based on the mass distribution—heavy hub and motor at the base (~20 mm above rail), lighter brick superstructure extending to ~110 mm—places the CG at approximately **30 mm above the rail head** (roughly 3 bricks plus a plate). This estimate has significant uncertainty (plausible range: 25–40 mm) and is the single most impactful parameter to measure empirically if precision matters.

**Track cant.** Standard LEGO curved track pieces have **zero banking angle**. The track sits perfectly flat. Some enthusiasts manually bank curves by placing plates under the outer rail (1 plate = 3.2 mm over 48 mm track width ≈ 3.8° cant), but this is non-standard and actually increases derailment risk at typical LEGO speeds. For modeling purposes, assume **0° superelevation** on all standard pieces.

### Motor torque-speed characteristics from Philo's testing

Philo's motor comparison page provides the most authoritative data, measured at regulated 9V DC. The train motors contain **internal gearing** (they are not direct drive despite common misconception—iFixit teardowns confirm internal spur gears).

| Parameter | PUP Train Motor (88011) | PF Train Motor (88002) |
|---|---|---|
| **No-load RPM** (9V) | **1,760 rpm** | 1,900 rpm |
| **No-load current** (9V) | 100 mA | 90 mA |
| **Stall torque** (9V, extrapolated) | **29 mNm** | 36 mNm |
| **Stall current** (9V, extrapolated) | 1.1 A | 1.3 A |
| **Max efficiency** (9V) | ~31% | ~38% |
| **Max mechanical power** (9V) | ~1.3 W | ~1.7 W |
| **Weight** | 57 g | 57 g |

Stall values are extrapolated because a thermistor inside the motor trips before true stall can be measured. Philo notes the PUP motor shows "higher internal friction" than the PF predecessor, though this may reflect individual sample variation. At the Powered Up hub's actual operating voltage of **7–9V** (depending on battery state), expect roughly 10–20% lower performance than the 9V bench figures.

The DC motor linear torque-speed relationship gives: **τ(ω) = τ_stall × (1 − ω/ω_no-load)**. Converting to tractive force at the wheel: **F(v) = (τ_stall / r_wheel) × (1 − v / v_max)**, where r_wheel = 0.0085 m and v_max = 1.57 m/s. This yields a **stall tractive force of 3.41 N**, dropping linearly to zero at no-load speed.

### Friction coefficient: the critical uncertainty

This is the hardest parameter to pin down and the most impactful on the model. Modern LEGO train wheels (post-2019) use **bio-based HDPE** running on **ABS** rails. Philo's traction experiments confirm Coulomb friction behavior (force proportional to weight, independent of contact area), but no direct wheel-on-rail μ measurement exists in the community.

| Condition | Estimated μ |
|---|---|
| HDPE/ABS wheel on ABS rail, clean and dry | **0.2–0.4** |
| Best estimate for modeling (static) | **~0.3** |
| Kinetic/sliding friction | **~0.2** |
| Dusty or dirty track | 0.1–0.2 |
| Wheel with rubber friction band (powered bogie) | 0.4–0.6 |

The powered bogie's 55423 wheels include a **rubber friction band** in their groove, which significantly increases traction. This means the driven wheels may have μ ≈ 0.4–0.5 while unpowered wheels have μ ≈ 0.2–0.3. For the **derailment model** (lateral sliding of all wheels), use the lower value of **μ ≈ 0.25–0.30** as the conservative estimate, since derailment occurs when *all* wheels slide laterally simultaneously.

---

## Complete track piece catalog with geometry

### Official LEGO City track pieces

| Piece | Part # | Centerline Length (mm) | Radius (mm) | Angle (°) | Notes |
|---|---|---|---|---|---|
| **Standard Straight** | 53401 | 128 (16 studs) | ∞ | 0 | Fits 16×16 grid |
| **Standard Curve (R40)** | 53400 | ~126 | 320 (40 studs) | 22.5 | 16 per circle, 4 per 90° |
| **Left Switch** | 53407 | 256 (32 studs) | Complex S-curve | 22.5 (diverging) | Crossing vee at 32.5° |
| **Right Switch** | 53404 | 256 (32 studs) | Complex S-curve | 22.5 (diverging) | Mirror of 53407 |
| **Flex Track** | 88492c00 | 32 (4 studs) per segment | Variable | ~5.6° max per segment | Fragmented outer rail when bent |

The standard curve has an **inner rail radius of ~288 mm**, **centerline radius of 320 mm**, and **outer rail radius of ~352 mm**. A full circle requires 16 pieces with an outer diameter of **704 mm**. The switch diverging route has an effective radius **tighter than R40** due to its S-curve geometry, making switches the tightest curve point in any standard layout. One switch plus one R40 curve produces parallel tracks at the standard **16-stud (128 mm) center-to-center** spacing.

### Third-party compatible curves (key radii)

Adjacent radii differ by 16 studs (128 mm), enabling perfectly parallel curved sections. The major manufacturers are Trixbrix (3D printed and injection molded), 4DBrix (3D printed), ME Models, BrickTracks (injection molded), and Fx Bricks (injection molded with 9V metal rails).

| Radius | Studs | mm | Angle per piece | Pieces/circle | Sources |
|---|---|---|---|---|---|
| R24 | 24 | 192 | 22.5° | 16 | Trixbrix (tight; many trains derail) |
| R32 | 32 | 256 | 22.5° | 16 | Trixbrix (compatible with all LEGO trains) |
| **R40** | 40 | 320 | 22.5° | 16 | **LEGO official** + all third parties |
| R56 | 56 | 448 | 11.25–22.5° | 16–32 | Trixbrix, 4DBrix, ME Models, BlueBrixx |
| R72 | 72 | 576 | 11.25–22.5° | 16–32 | Trixbrix, 4DBrix, ME Models |
| R88 | 88 | 704 | 11.25° | 32 | Trixbrix, 4DBrix, BrickTracks |
| R104 | 104 | 832 | 11.25° | 32 | Trixbrix, 4DBrix, BrickTracks |
| R120 | 120 | 960 | 11.25° | 32 | Trixbrix, 4DBrix |

Third-party switches are available at R40 (matching LEGO), R56, R104, and R148 radii. BrickTracks and Trixbrix offer R104 switches with significantly gentler diverging geometry than the LEGO R40 switch. Flex track is available from LEGO (88492c00) in 4-stud segments that hinge at a center pivot, but **community consensus strongly discourages extended use** due to derailment-prone rail gaps.

### Derailment risk by piece type

**Switches** present the highest derailment risk due to their tighter-than-R40 effective radius on the diverging route and the rail gap at the crossing vee. **S-curves** (back-to-back opposing curves with no intervening straight) are a major derailment source; club standards like NILTC/PennLUG require a minimum of 1 straight (16 studs / 128 mm) between opposing curves and 20 inches (~500 mm) of straight before S-curves. **Flex track** at tight radii produces a fragmented outer rail that increases derailment probability. For the physics model, apply a **speed penalty factor of 0.7–0.8** on switch diverging routes and flex-track curves relative to standard curves of equivalent radius.

---

## Quasi-static derailment physics at LEGO scale

### Lateral sliding: the binding constraint

On flat (unsuperelevated) track, the train stays on the rails only if lateral friction can supply the required centripetal force. The quasi-static force balance yields the **critical sliding speed**:

**v_slide = √(μ · g · R)**

This is the maximum speed before the entire vehicle slides laterally off the track. For LEGO parameters:

| Curve | R (m) | v_slide at μ=0.25 | v_slide at μ=0.30 | v_slide at μ=0.40 |
|---|---|---|---|---|
| R40 | 0.320 | 0.89 m/s | 0.97 m/s | 1.12 m/s |
| R56 | 0.448 | 1.05 m/s | 1.15 m/s | 1.33 m/s |
| R72 | 0.576 | 1.19 m/s | 1.30 m/s | 1.51 m/s |
| R88 | 0.704 | 1.31 m/s | 1.44 m/s | 1.66 m/s |
| R104 | 0.832 | 1.43 m/s | 1.57 m/s | 1.81 m/s |
| R120 | 0.960 | 1.54 m/s | 1.68 m/s | 1.94 m/s |
| Straight | ∞ | ∞ | ∞ | ∞ |

At the standard R40 radius with μ = 0.30, the critical speed of **0.97 m/s is 62% of the motor's 1.57 m/s no-load top speed**. This confirms that derailment physics genuinely constrains performance—the train cannot safely run at full speed on standard curves.

### Tip-over: secondary constraint, rarely binding

The tip-over speed comes from moment balance about the outer rail contact point. When the centripetal-force moment (m·v²·h / R) exceeds the gravitational restoring moment (m·g·d):

**v_overturn = √(g · R · d / h)**

where d = half the inner rail gauge = **18.75 mm** and h = CG height ≈ **30 mm**. The ratio v_overturn / v_slide = √(d / (h·μ)). For d = 0.01875 m, h = 0.030 m, μ = 0.30: ratio = √(0.01875 / 0.009) = **1.44**. The train slides off **44% before it would tip**, making lateral sliding the binding constraint for all standard LEGO trains. Tip-over only becomes binding for unusually tall constructions where h > d/μ ≈ 63 mm (roughly 8 bricks above the rail).

| Curve | v_slide (μ=0.30) | v_overturn (h=30mm) | Binding |
|---|---|---|---|
| R40 | 0.97 m/s | 1.40 m/s | Sliding |
| R88 | 1.44 m/s | 2.08 m/s | Sliding |
| R104 | 1.57 m/s | 2.26 m/s | Sliding |

### Lateral load transfer quantification

On curves, weight shifts from inner to outer wheels. The fractional load transfer is:

**LLT = v² · h / (g · R · d)**

At the sliding limit (v² = μgR): LLT = μh/d = 0.30 × 0.030 / 0.01875 = **0.48** (48% load transfer). The inner wheel still carries 52% of its static load—confirming the train slides well before the inner wheel lifts.

### Nadal's formula: not practically relevant

Nadal's wheel-climb criterion gives the critical lateral-to-vertical force ratio: (L/V)_crit = (tan δ − μ) / (1 + μ·tan δ), where δ is the flange contact angle. LEGO wheel flanges are **nearly vertical** (δ ≈ 80–85°), giving (L/V)_crit > 1.5—far higher than the L/V ratio at the sliding limit (~0.3). **Wheel-climb derailment is not a concern for LEGO trains**; the wheels slide laterally long before the flange climbs over the rail.

### Aerodynamic drag: confirmed negligible

At v = 1.5 m/s, the Reynolds number for a LEGO locomotive (characteristic length ~0.2 m) is Re ≈ **20,000**—firmly in the low-speed regime. The aerodynamic drag force is F_drag = ½ρv²C_D·A ≈ 0.5 × 1.225 × 2.25 × 1.0 × 0.0016 ≈ **0.002 N**. Rolling resistance (F_roll ≈ μ_roll·mg ≈ 0.01 × 0.37 × 9.81 ≈ 0.036 N) is already small, and drag is **less than 6% of rolling resistance**. Omit aerodynamic drag entirely from the model.

---

## Speed profile computation: the forward-backward pass algorithm

The time-optimal speed profile along a track with piece-by-piece speed limits, acceleration constraints, and braking constraints can be computed in **O(N) time** using a three-pass algorithm. This is mathematically equivalent to the TOPP-RA (Time-Optimal Path Parameterization) framework from Pham & Pham (2018, IEEE T-RO).

### Algorithm overview

Discretize the track into N points at uniform spacing Δs. Each point has a curvature-based speed limit, and the vehicle has maximum acceleration a_max and maximum braking deceleration a_brake.

**Pass 1 — Curvature speed limits.** For each point i with curve radius R[i]:

```
v_limit[i] = SF × sqrt(μ × g × R[i])
```

where SF is a safety factor (0.7–0.8) accounting for friction uncertainty, track imperfections, and dynamic effects not captured by the quasi-static model. On straight sections, v_limit = v_motor_max (≈1.5 m/s).

**Pass 2 — Forward pass (acceleration).** Starting from the initial speed, propagate forward ensuring the vehicle cannot accelerate faster than physics allows:

```
v_fwd[0] = v_initial
for i = 1 to N-1:
    v_fwd[i] = min(v_limit[i], sqrt(v_fwd[i-1]² + 2 × a_max × Δs))
```

**Pass 3 — Backward pass (braking).** From the end, propagate backward ensuring the vehicle can decelerate in time for upcoming speed limits:

```
v_bwd[N-1] = v_fwd[N-1]
for i = N-2 down to 0:
    v_bwd[i] = min(v_fwd[i], sqrt(v_bwd[i+1]² + 2 × a_brake × Δs))
```

The final speed profile is v[i] = v_bwd[i], which is the element-wise minimum of all three constraints. This is **provably time-optimal** under the given constraints.

### Handling closed loops

For a closed track where point 0 = point N, the boundary condition v[0] = v[N] must be satisfied. Two approaches work:

**Double-unroll method (recommended for simplicity).** Concatenate the track with itself to form a doubled path of length 2N. Run the three-pass algorithm on this doubled path. Extract the profile from positions N through 2N−1. The wrap-around constraints are automatically handled because the first copy's backward pass constrains the second copy's initial speed.

**Iterative fixed-point method.** Initialize v[0] = v_limit[0]. Run forward+backward passes. If v_bwd[0] < v[0], update v[0] = v_bwd[0] and repeat. Convergence typically occurs in **2–3 iterations** for well-behaved tracks.

### Acceleration and braking parameters

The maximum acceleration is the minimum of motor-limited and adhesion-limited values:

**a_motor(v) = F_motor(v) / m = (τ_stall / (r_wheel × m)) × (1 − v/v_max)**

At low speeds this exceeds the adhesion limit, so: **a_max = min(a_motor(v), μ_traction × g)**

For μ_traction = 0.4 (rubber-banded driven wheels): a_adhesion = **3.92 m/s²**. The motor force exceeds the adhesion limit for all speeds below ~1.3 m/s, so practical acceleration is **adhesion-limited** through most of the speed range.

For braking, LEGO trains brake by reversing motor polarity or coasting. With friction braking only: **a_brake = μ × g ≈ 0.25 × 9.81 = 2.45 m/s²** (using the lower kinetic friction for unpowered wheel sliding). Braking distance from speed v: **d_brake = v² / (2 × a_brake)**.

| v (m/s) | d_brake (μ=0.25) | d_brake (μ=0.40) | Context |
|---|---|---|---|
| 0.5 | 51 mm | 32 mm | ~0.4 straight pieces |
| 1.0 | 204 mm | 127 mm | ~1.6 straight pieces |
| 1.5 | 459 mm | 287 mm | ~3.6 straight pieces |

### Dynamic feasibility check

A layout is **dynamically feasible** if the speed profile algorithm produces v[i] > 0 everywhere. In practice, *every* LEGO layout with positive-radius curves is feasible because the curve speed limit is always positive—the train can always crawl through slowly enough. The meaningful question is whether the layout achieves acceptable **average speed**, which the speed profile algorithm directly computes.

**Average speed** over a closed loop of total length L: v_avg = L / T, where T = Σ(Δs / v[i]). **Lap time** is T directly. These become the objective functions for the genetic algorithm.

---

## pymoo integration: constraints, encoding, and optimization

### Constraint formulation (g ≤ 0 convention)

In pymoo ≥0.6, inequality constraints use `n_ieq_constr` and are written to `out["G"]` in the `_evaluate` method. A value ≤ 0 is feasible; > 0 is infeasible. The constraint violation (CV) is computed as **CV = Σ max(0, g_j)**, and pymoo's default **feasibility-first** selection always prefers feasible solutions over infeasible ones regardless of objective values. When both solutions are infeasible, the one with lower CV wins.

For the derailment constraint, the natural formulation is:

```python
# g_derail ≤ 0 means no derailment
# v_actual_max = max speed anywhere on the profile
# v_critical_min = minimum critical speed across all curves
g_derail = (max_speed_ratio - 1.0)  # where max_speed_ratio = v_actual / v_critical at worst point
```

Alternatively, embed the speed profile algorithm *inside* the constraint evaluation: if the speed profile is computable with v > 0 everywhere, the constraint is automatically satisfied (derailment prevented by construction). The constraint then becomes a **soft objective** to maximize average speed.

### Recommended problem structure

```python
class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, piece_catalog, inventory, max_pieces=50, **kwargs):
        super().__init__(
            n_var=max_pieces, n_obj=2, n_ieq_constr=4,
            xl=-1, xu=len(piece_catalog)-1, vtype=int, **kwargs
        )
        self.catalog = piece_catalog
        self.inventory = inventory

    def _evaluate(self, x, out, *args, **kwargs):
        active = x[x >= 0].astype(int)
        track = build_track([self.catalog[i] for i in active])
        profile = compute_speed_profile(track, mu=0.30, safety_factor=0.8)

        out["F"] = [-profile.avg_speed, len(active)]  # maximize speed, minimize pieces
        out["G"] = [
            (profile.max_derail_ratio - 1.0),                    # derailment (normalized)
            (track.closure_error - CLOSURE_TOL) / CLOSURE_TOL,   # closure
            float(track.n_disconnects),                           # connectivity
            inventory_excess(active, self.inventory),             # inventory
        ]
```

Use **`ElementwiseProblem`** rather than `Problem` because each evaluation runs an O(N) physics simulation that cannot be easily vectorized across the population. Enable parallelization with `StarmapParallelization`:

```python
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
import multiprocessing

pool = multiprocessing.Pool(8)
problem = TrackLayoutProblem(
    piece_catalog=CATALOG, inventory=INV,
    elementwise_runner=StarmapParallelization(pool.starmap)
)
```

### Variable-length chromosome strategies

pymoo requires fixed `n_var`, so variable-length track layouts need one of three workarounds:

**Strategy 1: Sentinel-padded fixed-length array** (simplest, recommended to start). Set n_var = MAX_PIECES with values in [-1, N_catalog-1]. Value -1 means "no piece at this position." Standard pymoo integer operators (SBX + polynomial mutation discretized) work out of the box, though the search space includes many degenerate solutions.

**Strategy 2: Custom variable type with object encoding.** Set n_var=1 where x[0] is a Python list of arbitrary length. This requires implementing custom `Sampling`, `Crossover`, `Mutation`, and `DuplicateElimination` operators. Much more expressive but significantly more implementation work.

**Strategy 3: Indirect/BRKGA-style encoding.** Use a fixed-length real-valued chromosome [0, 1]^n that gets decoded into a variable-length track via a deterministic decoder (e.g., threshold-based activation of piece slots, or a constructive heuristic guided by the chromosome values). This works well with pymoo's standard real-valued operators.

### Mixing hard and soft constraints effectively

**Normalize all constraints** to approximately the same scale ([-1, 1] range) by dividing each by its characteristic violation magnitude. This ensures the CV metric treats all constraints comparably. Keep constraints as **separate g values** rather than combining them—pymoo's selection operators use the sum of violations, and separate constraints provide better gradient signal.

For constraints that are **easy to repair programmatically** (like inventory limits or connectivity), implement a pymoo `Repair` operator that fixes solutions before evaluation:

```python
from pymoo.core.repair import Repair

class TrackRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = fix_inventory(X[i], problem.inventory)
            X[i] = fix_connectivity(X[i], problem.catalog)
        return X

algorithm = NSGA2(pop_size=200, repair=TrackRepair())
```

This concentrates the GA's search effort on the hard-to-satisfy physics constraints rather than wasting evaluations on trivially fixable violations. For problems where the feasible region is very small, consider pymoo's **ε-constraint handling** (`AdaptiveEpsilonConstraintHandling`), which gradually tightens constraint enforcement over the run.

---

## Putting it all together: parameter summary and model equations

### Complete parameter table for the physics model

| Parameter | Symbol | Value | Confidence | Source |
|---|---|---|---|---|
| Locomotive mass | m | **0.37 kg** (range: 0.33–0.40) | Medium | Component weights, Philo/Keybrick |
| Track centerline gauge | 2d_c | **40 mm** | Definitive | L-Gauge Standard |
| Inner rail gauge | 2d | **37.5 mm** | Definitive | L-Gauge / Wikipedia |
| Half inner gauge | d | **18.75 mm** | Definitive | Derived |
| Powered wheel rolling diameter | D | **17.0 mm** | Definitive | L-Gauge (55423 wheel) |
| Wheel radius | r | **8.5 mm** | Definitive | Derived |
| CG height above rail | h | **~30 mm** (range: 25–40) | Low | Geometric estimation |
| Motor no-load RPM (9V) | ω₀ | **1,760 rpm** | High | Philo (88011) |
| Motor stall torque (9V) | τ_s | **29 mNm** | High (extrapolated) | Philo (88011) |
| Max tractive force (stall) | F_s | **3.41 N** | High | τ_s / r |
| No-load top speed | v₀ | **1.57 m/s** | High | ω₀ × r |
| Lateral friction coeff. (static) | μ | **0.25–0.35** (best: 0.30) | Medium | ABS literature + Philo traction |
| Traction friction (rubber band) | μ_t | **0.4–0.5** | Medium | Rubber-on-ABS estimates |
| Rolling resistance coeff. | μ_r | **~0.01** | Low | Standard polymer estimate |
| Track banking angle | θ | **0°** | Definitive | L-Gauge, community consensus |
| R40 curve radius | R₄₀ | **320 mm** | Definitive | L-Gauge |
| Straight piece length | L_s | **128 mm** | Definitive | L-Gauge |
| Safety factor | SF | **0.8** (recommended) | Engineering judgment | — |

### Complete equation set

**Curve speed limit:**
v_curve(R) = SF × √(μ × g × R)

**Tip-over speed** (secondary check):
v_tip(R) = √(g × R × d / h)

**Binding constraint:**
v_max(R) = min(v_curve, v_tip, v_motor_max)

**Motor tractive force:**
F(v) = (τ_s / r) × (1 − v / v₀) = 3.41 × (1 − v / 1.57) [N]

**Maximum acceleration:**
a(v) = min(F(v)/m, μ_t × g) = min(9.22 × (1 − v/1.57), 3.92) [m/s²]

**Braking deceleration:**
a_brake = μ × g ≈ 2.5–3.0 m/s²

**Braking distance:**
d_brake(v₁ → v₂) = (v₁² − v₂²) / (2 × a_brake)

**Lap time for closed track:**
T = Σᵢ (Δs / v[i])

**Average speed:**
v_avg = L_total / T

The forward-backward pass algorithm with these equations produces the time-optimal speed profile in O(N) operations, and the resulting average speed and feasibility flag serve directly as pymoo objective and constraint values. The entire physics evaluation per candidate layout takes microseconds—fast enough for population sizes of 200+ across hundreds of generations.

## Conclusion

The LEGO train derailment model is dominated by a single phenomenon: **lateral sliding on curves**, with v_max = √(μgR). Tip-over, wheel climb (Nadal), and aerodynamic drag are all secondary or negligible at this scale. The critical design insight is that standard R40 curves limit speed to about **1.0 m/s** (at μ = 0.30)—meaningfully below the motor's **1.57 m/s** capability—so layouts with wider-radius third-party curves (R56, R72, R88+) can achieve substantially higher average speeds. The three-pass speed profile algorithm is both optimal and trivially fast to compute, making it ideal as an inner-loop evaluation inside pymoo's NSGA-II. The key modeling risk is **uncertainty in the friction coefficient** (μ = 0.25–0.40), which propagates as a ±15% uncertainty in critical speeds; empirical measurement of μ for your specific wheel/rail combination would be the single highest-value calibration experiment to perform.