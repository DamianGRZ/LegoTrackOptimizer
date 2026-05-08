---
date: 2026-05-08
status: draft
authors: Damian Grzesło (with Claude Opus 4.7)
related-files:
  - src/train/physics.py
  - src/train/scoring.py
  - src/problem.py
  - configs/trains/measured_consist.yaml
  - docs/Lateral stability model for LEGO train layout optimization.md
  - docs/Locomotive_dynamics.md
---

# Comprehensive Physical Evaluation Model

## Goal

Replace the current single-scalar physics signal (`F[1] = -min_speed`) with a structured, comprehensive physical evaluation model that:

1. Computes all relevant physics in a single pass per chromosome.
2. Returns a documented dataclass with provenance for every field.
3. Decouples physics computation from optimizer-specific F[]/G[] mapping.
4. Produces thesis-defensible metrics grounded in the user's measured train (`configs/trains/measured_consist.yaml`).
5. Operates at a configurable safety margin below derailment limits (default: 95%).

## Why

The current `_evaluate` reaches into the physics stack and extracts only `min_speed`, which is binary-ish in practice (any-R40-layout → 0.886 m/s, all-straight-layout → 1.26 m/s under the measured consist). All other physics in `physics.py` and `scoring.py` (friction ellipse, coupler correction, lap time, max/avg speed) is computed but discarded.

A comprehensive model exposes that computation as a structured object, enabling:

- **Thesis defensibility**: every reported metric traces to an explicit physical formula and to a measured input.
- **Diagnostic clarity**: when a layout is bad, the optimizer can answer *why* via per-segment fields (was it slide-bound? coupler-φ violation? a specific motor saturation?).
- **Future extension**: new objectives or constraints (energy, jerk, safety margins, coupler-φ as G[]) become a 1-line `_evaluate` change without touching physics.
- **Test isolation**: the model is a pure function of `(layout, catalog, train_config)`, independently testable against hand-calculations.

## Scope

### In scope

- New module `src/train/evaluation.py` exposing `evaluate_layout(layout, catalog, train_config, safety_margin) -> PhysicalEvaluation`.
- `PhysicalEvaluation` frozen dataclass with five sub-domains (geometry, stability, kinematics, dynamics, energy).
- Configurable safety margin in the kinematics layer (default `safety_margin = 0.95`, i.e. operate at 95% of `v_eff`).
- Initial optimizer integration: replace `F[1] = -min_speed` with `F[1] = lap_time` (minimized).
- Per-domain unit tests with hand-computed expected values (using the measured consist).
- Documentation: every metric has a docstring with formula + citation + units.

### Out of scope (deferred to future iterations)

- Adding new G[] hard constraints from the model (e.g. coupler-φ ≤ 12°). The model **computes** all such metrics; the optimizer initially does **not** consume them as constraints.
- Battery-decay model (motor exponential τ_m, voltage drop across N laps). Flagged as pre-thesis methodology gap; defaults retained.
- Multi-train throughput / scheduling.
- Empirical rolling-resistance measurement. A literature-derived μ_roll = 0.05 is used as a `TrainConfig` field with default value.
- Lateral RMS / jerk smoothness metrics consumed as F[]. Per-segment a_lat is exposed in the dataclass; aggregations like RMS or jerk integrals are deferred until consumed.

## Architecture

### Five physical domains

| Domain | Question it answers | Inputs (already available) | Outputs |
|---|---|---|---|
| **Geometry** | Where is the train and where do components hinge? | Catalog FK chain, `coupler_offset`, switch radii | Per-segment R(s), per-switch φ, max φ across loop |
| **Stability** | What speed derails the train? | `μ_design`, `gauge_b`, `cog_height_h`, `flange_angle_deg`, `v_motor_max` | Per-segment `v_slide`, `v_tip`, `v_nadal`, `v_eff`, binding-cap label |
| **Kinematics** | What speed does the train *actually* do, time-optimally, with safety margin? | `v_eff(s)`, `max_accel`, `brake_decel`, `safety_margin` | Time-optimal `v(s)`, `lap_time`, max/min/avg, `total_distance` |
| **Dynamics** | What forces does the motor / track / coupler see? | `v(s)`, `R(s)`, `m_loco`, `m_trailing`, `μ_design` | Per-segment `a_lat`, `a_long`, grip-utilization, lateral coupler reaction |
| **Energy** | How much work does the motor do per lap? | `v(s)`, `m_total`, `μ_roll`, motor-work integral | `motor_work_per_lap`, rolling dissipation (diagnostic), KE round-trips (diagnostic) |

### Module layout

```
src/train/
├── __init__.py            # public API: TrainConfig, PhysicalEvaluation, evaluate_layout, ...
├── physics.py             # KEEP unchanged: TrainConfig, v_eff_array, available_accel
├── scoring.py             # KEEP API: SpeedProfile, compute_speed_profile (consumed by evaluation.py)
└── evaluation.py          # NEW: evaluate_layout, PhysicalEvaluation, per-domain compute fns
```

`evaluation.py` orchestrates the five domains. Geometry, dynamics, and energy are NEW pure functions added inside `evaluation.py`. Stability re-uses `v_eff_array` (no API change). Kinematics calls `compute_speed_profile` and applies the safety margin to its `v_limit`.

### `PhysicalEvaluation` dataclass

```python
@dataclass(frozen=True)
class PhysicalEvaluation:
    """Full physical evaluation of a layout under a given train consist.

    Computed in a single pass via evaluate_layout(). Optimizer-agnostic.

    Conventions:
        Lengths in meters (catalog studs converted via stud_mm/1000).
        Angles in radians.
        Speeds in m/s, accelerations in m/s².
        Energies in joules.
        Per-segment arrays have length == n_pieces of the layout's main path.
    """

    # ---- Geometry ----
    coupler_phi_per_segment: NDArray[np.float64]
    """Per-segment coupler hinge angle: phi(R) = coupler_offset / (2·R) at curves;
    0.0 on straights. Reference: small-angle bogie kinematics, Pacejka §1.4."""

    coupler_phi_per_switch: dict[int, float]
    """Per-switch coupler angle, keyed by switch position in main loop.
    R is the diverging-route radius from the switch's catalog spec."""

    max_coupler_phi: float
    """Worst hinge angle across coupler_phi_per_segment ∪ coupler_phi_per_switch."""

    # ---- Stability ----
    v_slide_per_segment: NDArray[np.float64]
    """sqrt(μ_design · g · R). Reference: lateral force balance (no banking)."""

    v_tip_per_segment: NDArray[np.float64]
    """sqrt(g · R · b/(2h)). Reference: moment balance about outer rail head."""

    v_nadal_per_segment: NDArray[np.float64]
    """sqrt(g · R · L/V_crit), L/V_crit = (tan δ - μ)/(1 + μ·tan δ).
    Reference: Nadal 1908; Wickens 2003 §5.6."""

    v_eff_per_segment: NDArray[np.float64]
    """min(v_slide, v_tip, v_nadal, v_motor_max). Speed at which derailment is
    imminent. Operating speed (kinematics) stays below this by safety_margin."""

    binding_cap_per_segment: NDArray[np.str_]
    """Per-segment label: 'slide' | 'tip' | 'nadal' | 'motor'. Diagnostic."""

    # ---- Kinematics ----
    speed_profile: SpeedProfile
    """3-pass time-optimal profile (existing scoring.py output). The Pass-1
    cap is safety_margin · v_eff_per_segment. Passes 2-3 unchanged."""

    safety_factor_min: float
    """min(operating_speed / v_eff) across the loop. ≥ safety_margin by
    construction (equality on segments where the cap binds)."""

    safety_factor_mean: float
    """Distance-weighted mean of operating_speed / v_eff."""

    # ---- Dynamics ----
    a_lat_per_segment: NDArray[np.float64]
    """v² / R per segment. 0.0 on straights."""

    a_long_per_segment: NDArray[np.float64]
    """Per-segment longitudinal accel/decel from speed_profile finite differences:
    a_long[i] = (v[i+1]² - v[i]²) / (2·arc_length[i]). For closed loops,
    v[n] wraps to v[0]. Positive = accel, negative = brake."""

    grip_utilization_per_segment: NDArray[np.float64]
    """sqrt((a_lat / (μ·g))² + (a_long / cap)²) where cap = max_accel if
    a_long ≥ 0 else brake_decel. Friction-ellipse fraction; ∈ [0, 1] when
    feasible. Reference: Kapania & Gerdes 2015 §III."""

    coupler_force_lat_per_segment: NDArray[np.float64]
    """Lateral coupler reaction (N): m_trailing · a_long · sin(coupler_phi).
    The same force `available_accel` uses internally for its 1-iteration
    correction; here exposed per-segment for diagnostic visibility."""

    # ---- Energy ----
    motor_work_per_lap: float
    """Σ max(0, m_total · a_long[i]) · arc_length[i] over the loop.
    Joules. Includes both fighting friction and rebuilding kinetic energy.
    For a steady-speed closed loop, motor_work ≈ rolling_dissipation.
    For a brake-then-respin loop, motor_work > rolling_dissipation by
    the KE round-trip amount."""

    rolling_dissipation_per_lap: float
    """μ_roll · m_total · g · total_distance. Diagnostic breakdown of
    motor_work; not added on top of it. Default μ_roll = 0.05 (literature)."""

    ke_roundtrip_per_lap: float
    """Σ ½·m_total·(v_high² - v_low²) over each brake-then-respin pair
    (local minimum in the speed profile, sandwiched by higher values).
    Diagnostic breakdown; identity: motor_work ≈ rolling_diss + ke_roundtrip
    on closed loops in steady-state."""

    # ---- Provenance ----
    train_config: TrainConfig
    """The TrainConfig used for evaluation. Embedded for reproducibility."""

    safety_margin: float
    """Margin used in kinematics (0.95 = 95% of v_eff, the recommended default)."""

    catalog_signature: str
    """Hash or version tag of the catalog. Future-proofs reproducibility
    when catalog content changes."""
```

### Public API

```python
def evaluate_layout(
    layout: Layout | MultiPathLayout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    safety_margin: float = 0.95,
) -> PhysicalEvaluation:
    """Comprehensive physical evaluation. Pure function, no side effects.

    Returns a fully-populated PhysicalEvaluation. Cost: dominated by the
    3-pass speed profile (already O(n) per chromosome). New geometry,
    dynamics, and energy passes add O(n) each, so total stays O(n).
    """
```

### Optimizer integration (initial, conservative)

Modify `src/problem.py:TrackOptimizationProblem._evaluate`:

```python
def _evaluate(self, x, out, *args, **kwargs):
    layout = decode_chromosome(x, self.catalog, ...)

    if layout.n_pieces == 0:
        out["F"] = np.array([np.inf, np.inf])
        out["G"] = np.full(self.n_ieq_constr, 1e6)
        return

    phys = evaluate_layout(
        layout, self.catalog, self._train_config, safety_margin=0.95,
    )

    utilization = layout.n_pieces / self.total_inventory
    out["F"] = [-utilization, phys.speed_profile.lap_time]   # NEW: lap_time minimized
    # G[] unchanged (closure x/y/theta, boundary, collisions, per-type inventory)
```

Other `PhysicalEvaluation` fields (coupler-φ, dynamics, energy) are computed but **not consumed** by F[] or G[] in this iteration. They are accessible via the `_evaluate`-returned object (or via re-running `evaluate_layout` post-hoc) for diagnostics, post-hoc analysis, and future iterations.

## Calculation details

### Geometry: coupler-φ

For a curve of radius R and a vehicle of length L between coupling points (small-angle approximation):

```
phi = L / (2R)
```

- `L` = `coupler_offset` (TrainConfig field; m). For the measured consist, `L = 0.106 m` (the short car immediately behind the loco).
- `R` per segment:
  - **curves**: catalog `radius_studs` × stud_to_m
  - **switches**: catalog `diverging_radius_studs` × stud_to_m (the diverging branch is the smaller-radius leg, so it's the binding one)
  - **straights**: ∞ → φ = 0

`coupler_phi_per_switch` is a `dict[int, float]` keyed by `position_in_main_loop` of each switch. For multi-path layouts, the through-route is also evaluated for completeness but the diverging-route value is the one stored as the per-switch φ (worst-case).

### Stability: per-segment caps

Existing `v_eff_array(config, radii_m)` already computes `min(v_slide, v_tip, v_nadal, v_motor_max)`. The new evaluation layer exposes the four caps separately (small redundant compute) and adds a `binding_cap_per_segment` label via `argmin` across the four.

For straights (R = ∞): `v_slide = v_tip = v_nadal = ∞`, so `v_eff = v_motor_max`, label `"motor"`.

For tight curves where slide binds (the typical R40 case): label `"slide"`.

For configs where `flange_angle_deg` is small or `μ_design` is high, Nadal can become the binding cap; for unusual cog_height_h vs gauge_b ratios, tip can. The label captures the regime.

### Kinematics: safety margin

Pass 1 of `compute_speed_profile` currently sets `v_limit[i] = v_eff[i]`. The new behavior:

```python
v_limit[i] = safety_margin * v_eff_per_segment[i]
```

**Injection mechanism**: extend `compute_speed_profile` with a `safety_margin: float = 1.0` keyword. Internally it scales the Pass-1 cap. `evaluate_layout` calls it with `safety_margin=0.95`. Existing callers (if any pass through other paths) are unchanged because the default is 1.0. Alternative — pre-compute `v_limit` outside and inject it — is rejected as it would fork the closed-loop double-unroll logic.

Passes 2-3 unchanged (forward acceleration, backward braking with friction-ellipse-corrected `available_accel`). Result: every layout operates strictly below derailment by `(1 - safety_margin) × 100 %` (default 5%) on every cap-bound segment, and may operate further below where accel/brake constraints force it.

`safety_factor_min` is computed as `np.min(speed_profile.speeds / v_eff_per_segment)`. By construction this equals `safety_margin` exactly on segments where the operating speed is at the (margined) cap, and exceeds it on segments where accel/brake forces the train below.

### Algorithm walk-through (worked example: STR–STR–R40–R40)

Concrete trace using `measured_consist.yaml` values
(`v_motor_max = 1.26`, `μ_design = 0.25`, `max_accel = 0.68`, `brake_decel = 2.45`,
arc lengths: `STRAIGHT_16` = 0.128 m, R40 = 0.126 m).

**Pass 1 — per-piece derailment cap** (with `safety_margin = 0.95`):

| i | Piece | R (m) | v_slide | v_tip | v_nadal | v_motor | `v_eff` | `v_limit = 0.95·v_eff` | binding |
|---|---|---|---|---|---|---|---|---|---|
| 0 | STR | ∞ | ∞ | ∞ | ∞ | 1.26 | 1.26 | **1.197** | motor |
| 1 | STR | ∞ | ∞ | ∞ | ∞ | 1.26 | 1.26 | **1.197** | motor |
| 2 | R40 | 0.32 | 0.886 | 1.401 | 1.509 | 1.26 | 0.886 | **0.842** | slide |
| 3 | R40 | 0.32 | 0.886 | 1.401 | 1.509 | 1.26 | 0.886 | **0.842** | slide |

Pass 1 produces "the speed at which derailment is imminent (with margin)." Passes 2 and 3 will never overshoot it.

**Pass 2 — forward acceleration**: `v[i] = min(v_limit[i], √(v[i-1]² + 2·a_avail·arc_length[i-1]))`.

Starting from rest (open path), the train CAN'T reach 1.197 across two STR pieces — `√(2·0.68·0.128) ≈ 0.418` after piece 0, `√(0.418² + 2·0.68·0.128) ≈ 0.591` after piece 1. Accel binds **before** motor cap binds; layouts with too-short straights between curves never reach motor speed and lose lap time.

For closed loops via double-unroll, the wrap-around delivers piece 3's exit speed (0.842) into piece 0's "previous", so v[0] ≤ √(0.842² + 2·a_avail·0.126) ≈ 0.928 — the loop runs at sub-motor speed everywhere because the curves never let the train re-accelerate to 1.197.

**Pass 3 — backward braking**: `v[i] = min(v_fwd[i], √(v[i+1]² + 2·a_brake·arc_length[i]))`.

Brake distance from 1.197 → 0.842:

```
d_brake = (1.197² - 0.842²) / (2 · 2.45) = (1.433 - 0.709) / 4.9 ≈ 0.148 m
```

Slightly more than one STRAIGHT_16 arc (0.128 m). So a single STR before an R40 is **not** enough to coast at 1.197 then slow — the train must already be slowing on the straight. Pass 3 enforces this by computing `√(0.842² + 2·2.45·0.128) ≈ 1.156` for piece 1, capping it below the 1.197 motor cap. Result: piece 1 runs at **1.156 m/s**, not 1.197.

**Output `v[i]` per piece (closed loop):**

| i | Pass 1 cap | After Pass 2 (fwd) | After Pass 3 (bwd) |
|---|---|---|---|
| 0 | 1.197 | 0.928 | 0.928 |
| 1 | 1.197 | 1.034 | 0.928 |
| 2 | 0.842 | 0.842 | 0.842 |
| 3 | 0.842 | 0.842 | 0.842 |

(Numbers are first-order estimates that ignore the friction-ellipse cornering reduction inside `available_accel`. Exact values are slightly lower because cornering shrinks the longitudinal accel headroom from 0.68 m/s² down to ≈0.29 m/s² at v_slide. The qualitative pattern — loop never reaches motor cap, R40s bind at margined-slide — is robust regardless.)

**Lap time:**
```
lap_time = 0.128/0.928 + 0.128/0.928 + 0.126/0.842 + 0.126/0.842
         ≈ 0.138 + 0.138 + 0.150 + 0.150
         ≈ 0.576 s
```

**What this discriminates between:**

- A 16-R40 closed circle (no straights at all): every piece runs at 0.842 → lap_time = 16·0.126/0.842 ≈ 2.39 s.
- A 4-R40 + 12-STR oval (long straights between corners): straight sections actually reach 1.197 before braking → lap_time ≈ (4·0.126/0.842) + (12·0.128/1.197) ≈ 0.60 + 1.28 ≈ 1.88 s.
- A 4-R40 + 4-STR oval (short straights, accel-bound): straight pieces never reach 1.197 because there isn't enough length to spin up → lap_time penalised relative to long-straight oval, even though piece counts are similar.

This is what makes lap_time a **layout-discriminating** signal where `min_speed` was binary (0.842 vs 1.197).

### Dynamics: per-segment force diagnostics

```
a_lat[i]  = v[i]² / R[i]                                       # 0 on straights
a_long[i] = (v[(i+1) mod n]² - v[i]²) / (2 · arc_length[i])    # finite difference, wraps for closed loops
cap[i]    = max_accel  if a_long[i] >= 0 else brake_decel
grip_util[i] = sqrt( (a_lat[i] / (μ_design · g))² + (a_long[i] / cap[i])² )
F_coupler_lat[i] = m_trailing · a_long[i] · sin(coupler_phi[i])
```

For open paths (non-closed layouts), `a_long[n-1]` is set to 0 (no wrap).

`grip_utilization` should always be ≤ 1 by construction because `available_accel` was derived from the same friction ellipse. Spec asserts this in tests; if violated for any segment under any layout, that's a bug in the kinematics-dynamics consistency (unit test `test_grip_utilization_below_one`).

### Energy: motor work and breakdowns

```
motor_work_per_lap         = Σ max(0, m_total · a_long[i]) · arc_length[i]
rolling_dissipation_per_lap = μ_roll · m_total · g · total_distance
ke_roundtrip_per_lap        = Σ ½ · m_total · (v_high² - v_low²)
                              over each brake-then-respin pair
```

**Simplification: rotational inertia ignored.** The motor-work formula treats the train as a point mass `m_total`. Wheel rotational inertia and motor armature inertia are neglected. For LEGO scale (~1 kg total mass, small wheels), this is a < 5% effect on KE; far smaller than the rolling-resistance uncertainty. Documented for thesis appendix; revisitable if measurements demand.

**Identifying brake-then-respin pairs**: scan the speed profile for local minima sandwiched by higher values. A local minimum at index `i` means `v[i] < v[i-1]` AND `v[i] < v[(i+1) mod n]`. The pair contributes `½·m·(max(v[i-1], v[(i+1) mod n])² - v[i]²)`. This corresponds to the kinetic energy lost at the brake event and rebuilt by the motor immediately after.

**Energy identity check**: for a closed loop in steady state (after profile convergence), `motor_work ≈ rolling_dissipation + ke_roundtrip`. Slight discrepancies are acceptable due to discretization. Unit test `test_energy_identity` asserts `|motor_work - (rolling_diss + ke_roundtrip)| / motor_work < 0.05`.

`μ_roll` is added as a `TrainConfig` field with default `0.05`. The user's `measured_consist.yaml` notes "Rolling resistance ~0.03–0.10 m/s² not deducted" — the 0.05 default is the literature-typical midpoint. Future measurement can override.

## Validation strategy

### Unit tests (per domain)

For each domain, the test fixes a small layout, fixes `train_config = measured_consist.yaml`, and asserts hand-computed values:

| Test | Layout | Expected (using measured consist) |
|---|---|---|
| `test_geometry_phi_R40` | One R40 curve, R = 0.32 m | `phi == 0.106 / (2·0.32) ≈ 0.1656 rad ≈ 9.49°` |
| `test_geometry_phi_straight` | One STRAIGHT_16 | `phi == 0.0` |
| `test_stability_v_slide_R40` | One R40 curve | `v_slide ≈ √(0.25 · 9.81 · 0.32) ≈ 0.886 m/s` (matches user's sanity check) |
| `test_stability_binding_cap` | R40 + STRAIGHT_16 | `binding_cap == ['slide', 'motor']` |
| `test_kinematics_safety_margin` | 16-R40 closed circle, `safety_margin=0.95` | `v_actual[i] ≈ 0.95 · 0.886 ≈ 0.842 m/s` everywhere; `safety_factor_min == 0.95` |
| `test_kinematics_lap_time_circle` | 16-R40 closed circle | `lap_time ≈ 2π·0.32 / 0.842 ≈ 2.39 s` |
| `test_dynamics_a_lat_R40` | One R40 curve at v=0.842 m/s | `a_lat ≈ 0.842² / 0.32 ≈ 2.215 m/s²` (≈ 0.95² · μg ≈ 0.9025 · 2.45) |
| `test_dynamics_grip_util_below_one` | Any layout | `np.all(grip_utilization ≤ 1.0 + 1e-9)` |
| `test_energy_rolling_constant_speed` | All-straight closed loop at v_motor_max · 0.95 | `motor_work ≈ rolling_diss = 0.05·0.82·9.81·distance`; `ke_roundtrip ≈ 0` |
| `test_energy_identity` | 16-R40 + 16-STR layout | `\|motor_work − rolling_diss − ke_roundtrip\| / motor_work < 0.05` |

### Integration test

Run optimizer with `safety_margin=0.95` on a known-good handcrafted chromosome (16-R40 circle) and assert:

- `phys.speed_profile.lap_time` matches the unit-test value within ±1%
- `out["F"][1] == phys.speed_profile.lap_time`
- All other fields populated (no NaN, no None)

### Manual cross-check vs measurement (thesis appendix)

When the user logs a real lap time on a built layout, the model should predict it within ±5%, allowing for un-modeled rolling-resistance variance and the uniform-accel approximation. Documented in the thesis appendix as a validation check.

## Implementation phases

| Phase | Deliverable | Tests |
|---|---|---|
| 1 | `evaluation.py` skeleton: `PhysicalEvaluation` dataclass, `evaluate_layout` signature with empty domains; Geometry domain wired | `test_geometry_*` |
| 2 | Stability domain wired (re-expose existing physics + binding-cap labels) | `test_stability_*` |
| 3 | Kinematics: `safety_margin` parameter into `compute_speed_profile` (or wrapper); `safety_factor_min/mean` computed | `test_kinematics_*` |
| 4 | Dynamics domain (a_lat, a_long, grip_util, coupler_force) | `test_dynamics_*` |
| 5 | Energy domain (motor_work, rolling_diss, ke_roundtrip) | `test_energy_*` |
| 6 | Wire `lap_time` into `_evaluate` as F[1]; integration test; remove old `min_speed` reference | end-to-end test |

Each phase is independently committable. Phases 1–5 do not change optimizer behavior. Phase 6 is the only F[]-changing commit.

## Open questions

- **Coupler-φ at switches: which radius?** Spec assumes the diverging-route radius (smaller, binding). Confirm against `data/track_pieces_v2.yaml` switch geometry — the catalog should already expose `diverging_radius_studs` per the V2 schema.
- **`μ_roll` placement**: Spec adds it as a `TrainConfig` field with default 0.05. Alternative: a module-level constant in `evaluation.py`. TrainConfig keeps it user-tunable per consist, which is the more thesis-friendly choice.
- **`binding_cap_per_segment` storage**: numpy string array (as specced) vs `list[CapLabel]` enum vs int codes. String is easiest to debug but uses more memory; profile if it shows up in optimizer hot path.
- **`catalog_signature`**: simple option is `hashlib.sha256(yaml_content).hexdigest()[:12]`; more elaborate is a semantic version field in the catalog YAML. Spec assumes SHA-prefix for now.
- **`safety_margin` in `TrainConfig` or as `evaluate_layout` parameter?** Currently both (parameter overrides config; config provides default). This means `_evaluate` can pin `0.95` explicitly without depending on the YAML.

## Future work (out of scope for this spec)

- Replace uniform-accel-then-coast model with motor exponential dynamics (`v(t) = v_∞ · (1 − e^(−t/τ_m))`); requires Tracker software re-analysis (user's deferred methodology gap).
- Battery voltage decay across N laps; recompute v_motor_max each lap.
- Add coupler-φ as G[] hard constraint (model already produces it; just wire into optimizer).
- Add total_energy_per_lap as F[2] (multi-objective NSGA-III) once Pareto fronts demand it.
- Add lateral RMS / jerk smoothness as F[].
- Multi-train throughput modeling.

## References

- Kapania, N. R. & Gerdes, J. C. (2015). "Path tracking of highly dynamic autonomous vehicle trajectories via iterative learning control." *Vehicle System Dynamics*. (Friction ellipse formulation.)
- Wickens, A. H. (2003). *Fundamentals of Rail Vehicle Dynamics*. §5.6. (Nadal criterion.)
- Pacejka, H. B. (2012). *Tire and Vehicle Dynamics*. §1.4. (Small-angle bogie kinematics.)
- Nadal, M. J. (1908). "Locomotives à vapeur." *Collection Encyclopédie Scientifique, Bibliothèque de Mécanique Appliquée et Génie*. (Original wheel-climb derivation.)
- `docs/Lateral stability model for LEGO train layout optimization.md` — in-house derivation of the v_slide/v_tip/v_nadal triple in LEGO scale.
- `docs/Locomotive_dynamics.md` — in-house physics summary, friction ellipse derivation.
- `configs/trains/measured_consist.yaml` — validated parameter set, dated 2026-05-06; user's authoritative train physics.