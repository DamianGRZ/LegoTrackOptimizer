# Objective Function — Full Specification

The problem is bi-objective and **both objectives are minimized**, as pymoo requires. A layout
that uses more of the kit scores better on the first; a layout the train covers faster scores
better on the second. The two conflict — more track means more time — which is what gives the
Pareto front something to spread along.

```
F₀ = − weighted utilization      (maximize piece usage, negated for minimization)
F₁ = expected traversal time     (seconds; minimized directly)
```

Everything below is computed per chromosome in `TrackOptimizationProblem._evaluate`, after the
decoder has turned genes into a concrete layout.

`src/problem.py:171` — `_evaluate`, the entry point for both objectives and all constraints

---

## 1. F₀ — weighted utilization

```
              n_phys + (W − 1) · n_spec
F₀  =  −  ───────────────────────────────
                     N_inv
```

| Symbol | Meaning |
|---|---|
| `n_phys` | physical pieces in the layout: main-loop slots, minus one for every two-slot piece (CROSS_90, DOUBLE_CROSSOVER), plus all siding-branch pieces |
| `n_spec` | multi-path elements: switch pairs + crossings (descriptor **and** emergent) + double-crossovers |
| `W` | `special_piece_weight`, default **3.0** |
| `N_inv` | total pieces available in the config's inventory |

The weight exists because a switch pair or a crossing costs geometry and closure effort while
occupying few slots. Counted plainly, the GA strips them as overhead; counted at `W = 3` each,
branching topology raises the score and survives selection.

`src/problem.py:290` — `_weighted_utilization`
`src/types.py:235` — `n_physical_pieces`, where the two-slot correction is applied

---

## 2. F₁ — expected traversal time

Not a lap time: the **expected time to cover every physical piece of the network once**, when
the train picks its route at random. A layout with sidings must pay for the track it owns,
including the parts a single lap would skip.

### 2.1 Routes

A layout with `J` sidings has `2^J` routes — each siding is independently taken or bypassed.
Route `r` is a sequence of segments `k = 1 … n_r`; each segment carries its traversed-route
radius `R`, its traversed-route arc length `L`, and the identity of the physical piece it
belongs to.

### 2.2 Speed profile of one route (three passes)

**Pass 1 — per-segment speed cap.** Three derailment modes plus the motor, then a safety margin:

```
v_slide = √(μ · g · R)                            lateral sliding
v_tip   = √(g · R · (b/2) / h)                    tip-over
v_nadal = √(g · R · (tanδ − μ) / (1 + μ · tanδ))  Nadal wheel-climb

v̄ = 0.95 · min(v_slide, v_tip, v_nadal, v_motor)
```

Straights have `R = ∞`, so all three derailment caps go to infinity and the motor binds. On R40
the sliding cap binds. The margin 0.95 (`SPEED_SAFETY_MARGIN`) keeps the operating speed
strictly below every derailment threshold.

**Passes 2 and 3 — acceleration and braking.** Forward, then backward:

```
v_k^fwd = min( v̄_k ,  √( (v_{k−1}^fwd)² + 2 · a_acc · L_{k−1} ) )
v_k     = min( v_k^fwd ,  √( v_{k+1}² + 2 · a_brk · L_k ) )
```

The available longitudinal acceleration comes from a **capped friction circle**. Combined tyre
force stays inside the circle of radius `μ·g`, and the longitudinal component is additionally
bounded by the drive or brake cap:

```
a_lat = v² / R
a     = min( a_cap , √( (μ · g)² − a_lat² ) )        a_cap = a_max  or  a_brk
```

The cap is a motor-torque limit, not a grip limit, so lateral demand does not scale it down
until combined demand actually reaches the circle. When accelerating with trailing vehicles, one
correction iteration adds the coupler's destabilising lateral force:

```
φ       = d_coupler / (2 · R)
a_lat′  = a_lat + (m_trail / m_loco) · a · sin φ
a       = min( a_cap , √( (μ · g)² − a_lat′² ) )
```

Braking skips this correction — coupler compression is stabilising there, so omitting it is
conservative.

**Closed loops.** A route that closes within the config's tolerances (4.0 studs, 5.0°) is
profiled on **three concatenated copies of itself, and the middle copy is returned**. The
forward pass's start transient stays in the first copy and the backward pass's missing
wrap-around braking stays in the third, so the reported lap has a full lap of history on both
sides. The result does not depend on which piece is numbered first. A route that does not close
is profiled as open track.

`src/train/physics.py:96` — `derailment_caps`, the three formulas
`src/train/physics.py:144` — `available_accel`, the capped friction circle
`src/train/scoring.py:119` — `_compute_speeds_triple_unroll`

### 2.3 Arc length of a route

Length is a property of the **route taken**, not of the piece: a switch entered on its diverging
leg is longer than the same switch run straight through. Each route's length is derived from its
own endpoint pose in the catalog (`c` = chord between entry and exit ports, `Δθ` = heading
change, `R` = route radius):

```
L = c                                straight route
L = c · (|Δθ|/2) / sin(|Δθ|/2)       single circular arc
L = 4R · arcsin( c / (4R) )          symmetric S-curve, zero net turn
```

| Route | Arc length (studs) |
|---|---|
| STRAIGHT_16 | 16.000 |
| R40_CURVE | 15.708 |
| Switch, through | 32.000 |
| Switch, diverging | 35.463 |
| Double-crossover, through | 48.000 |
| Double-crossover, diagonal | 51.480 |

`src/catalog/catalog.py:507` — `_route_arc_length_studs`

### 2.4 From segments to the objective

Segment time is `t = L / v`. Two chromosome slots belonging to one physical piece (a crossing,
a double-crossover) are unified to a single identity, so the piece is counted once no matter how
many times the train passes over it. For every physical piece `p`:

```
T_p = Σ  t          summed over all passages of p, on all routes
m_p = number of those passages

        T_p
F₁ = Σ  ───
    p   m_p
```

Each piece is charged the **mean** of its traversal times — the expected per-piece cost under a
uniform random route choice — and the objective sums over distinct pieces.

Three consequences worth stating:

- A plain loop has one route covering each piece exactly once, so `F₁` **is** that loop's lap time.
- A self-crossing loop comes out **below** its lap time: the crossing is one physical piece the
  lap drives over twice, so it is charged once, at the mean of the two passages.
- A layout with sidings comes out **above** every single route's lap time, because the branch and
  the bypassed straights are both owned and both charged.

`src/problem.py:47` — `_expected_traversal_time`

### 2.5 Edge cases

| Situation | Result |
|---|---|
| No route carries any piece | `F₁ = +∞` — zero time would rank an unusable layout best |
| Profiler stalled a segment at `v = 0` | speed floored at 0.001 m/s, so the piece costs ~100 s |
| Layout decodes to zero pieces | `F = [+∞, +∞]`, `G = 10⁶` — never NaN, which would break dominance |

---

## 3. Constraints

The objectives rank layouts; these decide which layouts count. All are inequalities, feasible at
`g ≤ 0`, handled by Deb's feasibility-first rules.

```
G₀ = |dx| / tol_pos − 1                                    closure in x
G₁ = |dy| / tol_pos − 1                                    closure in y
G₂ = |dθ| / tol_ang − 1                                    closure in heading
G₃ = (boundary_violation − tol_boundary) / diagonal        fits the table
G₄ = n_unresolved / 5 + n_dangling_cross + n_dangling_dc   collisions
G₅₊ₜ = max(0, census_t − max_occ_t) / max(1, max_occ_t)    inventory, per piece type
```

Closure is an inequality, not an equality: a loop is closed when its gap is under tolerance, and
the residual is normalized by that tolerance so all three closure entries share one scale.

`G₄` mixes two failure modes deliberately. An unresolved self-crossing is a mild penalty (`/5`) —
the track crosses itself but could be re-routed. A dangling port on a crossing or
double-crossover contributes a full 1.0 each, because such a layout is structurally unbuildable.

`G₀…G₃` are **soft** — the adaptive-epsilon schedule may relax them early in a run. `G₄` and the
inventory entries are **hard**: their contribution to the constraint violation is weighted ×1000
so no epsilon can relax them.

`src/problem.py:230` — closure, boundary and collision assembly
`src/problem.py:305` — `_compute_per_type_inventory_violation`
`src/algorithm/runner.py:440` — `SOFT_CONSTRAINT_COUNT`, the soft/hard split

---

## 4. Physical parameters

From `configs/trains/measured_consist.yaml` (AFM SL+Cargo M0015TW, measured 2026-05-06).

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Design friction | μ | 0.25 | assumed, pessimistic; every cap formula reads this |
| Gravity | g | 9.81 m/s² | standard |
| Motor top speed | v_motor | 1.26 m/s | measured, full consist |
| Acceleration cap | a_max | 0.68 m/s² | measured; train is torque-limited, not grip-limited |
| Braking cap | a_brk | 2.45 m/s² | assumed |
| Track gauge | b | 0.0375 m | LEGO L-gauge |
| Centre-of-gravity height | h | 0.030 m | assumed |
| Flange angle | δ | 50° | assumed |
| Locomotive mass | m_loco | 0.493 kg | measured, batteries included |
| Trailing mass | m_trail | 0.327 kg | measured, two cars |
| Coupler offset | d_coupler | 0.106 m | measured |

Derived, at R40 (radius 0.320 m):

```
v_slide = 0.8859 m/s   ← binding cap on curves
v_tip   = 1.4007 m/s
v_nadal = 1.5092 m/s
v_motor = 1.2600 m/s   ← binding cap on straights

operating speed:  0.8416 m/s on R40,  1.1970 m/s on straights   (after ×0.95)
friction circle radius:  μ · g = 2.4525 m/s²
```

Objective-side constants: `special_piece_weight = 3.0`, `closure_tolerance = 4.0` studs,
`angle_tolerance = 5.0°`, `boundary_tolerance = 2.0` studs, `SPEED_SAFETY_MARGIN = 0.95`.

---

## 5. Worked anchors

Two layouts computed end to end, useful for checking any reimplementation.

**16 × R40_CURVE circle** (`all_pieces` inventory, 218 pieces):

```
distance   16 × 15.708 studs × 8 mm = 2.0106 m
speed      curve-capped everywhere: 0.8416 m/s
F₁         2.0106 / 0.8416 = 2.389058 s      (= lap time, single route, each piece once)
F₀         −16 / 218 = −0.073394             (no special pieces)
```

**Figure-eight with one CROSS_90** (`with_crossing` inventory, 34 slots):

```
n_phys     33 physical pieces (34 slots − 1 for the two-slot crossing)
n_spec     1 crossing  →  F₀ = −(33 + 2·1) / 202 = −0.173267
lap time   4.774904 s
F₁         4.659366 s      strictly below the lap time: the crossing is driven
                           twice and charged once
```

The same physical layout scores identically whether its crossing was named by a descriptor gene
or discovered by the decoder's self-intersection repair — same `F₀`, same `F₁`, same piece count.

---

## 6. Where it lives

| File | Responsibility |
|---|---|
| `src/problem.py` | both objectives, all constraints, the per-piece averaging |
| `src/train/scoring.py` | three-pass speed profile, triple unroll, lap time |
| `src/train/physics.py` | derailment caps, capped friction circle, coupler correction |
| `src/catalog/catalog.py` | per-route arc lengths and radii |
| `src/decoder/construction.py` | genes → layout, including piece identity for two-slot pieces |

Tests that pin the definitions: `tests/test_problem.py` (whole-graph property, shape
sensitivity), `tests/test_cross90_objective.py` (crossing charged once, descriptor/emergent
parity), `tests/test_scoring.py` (rotation invariance), `tests/test_train.py` (friction circle),
`tests/test_catalog.py` (route arc lengths).
