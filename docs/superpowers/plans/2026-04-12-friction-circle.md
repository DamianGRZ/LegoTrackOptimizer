# Friction Circle + Coupler Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the speed profile physically honest for a consist accelerating/braking on curves by coupling lateral and longitudinal friction demands via the friction ellipse, and adding a coupler lateral force correction for the leading vehicle.

**Architecture:** Add three mass/geometry fields to `TrainConfig` (`mass_loco`, `mass_trailing`, `coupler_offset`) plus a `mass_total` property. Add a scalar `available_accel` function to `src/train.py` implementing the friction ellipse with one-iteration coupler correction. Modify `_forward_pass` and `_backward_pass` in `src/evaluation.py` to call `available_accel` per-step instead of using flat constants. All changes preserve O(N) complexity.

**Tech Stack:** Python 3.x, numpy, pytest. No new dependencies.

**Research basis:** Kapania & Gerdes 2015 (forward-backward solver with friction constraint), TUM/FTM `trajectory_planning_helpers` (reference Python implementation), Delft TU quasi-static coupler force paper.

---

## Key physics

**Friction ellipse.** The wheels have different friction limits for lateral vs longitudinal:
- Lateral limit: `a_lat_max = mu_design × g = 0.25 × 9.81 = 2.45 m/s²` (all wheels, plain friction)
- Longitudinal limit: `a_long_max = max_accel` (driven wheels, rubber-banded traction, user-settable per consist)

The combined constraint is an ellipse:
```
(a_lat / a_lat_max)² + (a_long / a_long_max)² ≤ 1
```

Available longitudinal acceleration at lateral demand `a_lat`:
```
a_long_avail = a_long_max × √(max(0, 1 − (a_lat / a_lat_max)²))
```

At `a_lat = 0` (straights): full `max_accel`. At `a_lat = a_lat_max` (v_slide on curve): zero.

For braking: `a_long_max = brake_decel = mu_design × g`, making the ellipse a circle. Same formula applies but with `brake_decel` as the cap.

**Coupler correction.** During acceleration on a curve, the first trailing car's coupler transmits a lateral destabilising force on the locomotive:
```
φ = coupler_offset / (2 × R)         # coupler angle (rad)
a_coupler_lat = (m_trailing / m_loco) × a_long × sin(φ)
```

This adds to the lateral demand, reducing the available longitudinal budget. Applied during acceleration only — during braking, coupler compression is stabilising (omitting it is conservative).

**One-iteration scheme.** Compute `a_long` without coupler, use it to estimate `a_coupler_lat`, recompute `a_long` with the corrected lateral demand. Converges within 2.5% of exact in one step.

---

## Task 1: Add mass/coupler fields and `available_accel` to `src/train.py`

**Files:**
- Modify: `tests/test_train.py`
- Modify: `src/train.py`

- [ ] **Step 1: Append failing tests to `tests/test_train.py`**

Add two new test classes at the end of the file:

```python
class TestConsistFields:
    """TrainConfig carries mass and coupler geometry for consist modeling."""

    def test_default_mass_loco(self):
        assert TrainConfig().mass_loco == pytest.approx(0.370)

    def test_default_mass_trailing(self):
        assert TrainConfig().mass_trailing == pytest.approx(0.0)

    def test_default_coupler_offset(self):
        assert TrainConfig().coupler_offset == pytest.approx(0.100)

    def test_mass_total_bare_loco(self):
        assert TrainConfig().mass_total == pytest.approx(0.370)

    def test_mass_total_with_cars(self):
        tc = TrainConfig(mass_loco=0.370, mass_trailing=0.600)
        assert tc.mass_total == pytest.approx(0.970)


class TestFrictionCircle:
    """available_accel implements friction ellipse + coupler correction."""

    def test_straight_full_accel(self):
        """On a straight (R=inf), full max_accel is available."""
        from src.train import available_accel
        tc = TrainConfig()
        assert available_accel(tc, v=0.5, radius_m=math.inf) == pytest.approx(tc.max_accel)

    def test_straight_full_brake(self):
        """On a straight (R=inf), full brake_decel is available."""
        from src.train import available_accel
        tc = TrainConfig()
        assert available_accel(tc, v=0.5, radius_m=math.inf, is_braking=True) == pytest.approx(
            tc.brake_decel
        )

    def test_r40_at_vslide_zero_accel(self):
        """At v_slide on R40, lateral demand saturates friction — zero accel."""
        from src.train import available_accel
        tc = TrainConfig()
        v_slide = tc.v_slide(0.320)
        assert available_accel(tc, v=v_slide, radius_m=0.320) == pytest.approx(0.0, abs=0.01)

    def test_r40_at_half_speed_partial_accel(self):
        """At half v_slide on R40, some accel budget remains."""
        from src.train import available_accel
        tc = TrainConfig()
        v_half = tc.v_slide(0.320) / 2.0
        accel = available_accel(tc, v=v_half, radius_m=0.320)
        assert 0.0 < accel < tc.max_accel

    def test_r40_at_half_speed_partial_brake(self):
        """At half v_slide on R40, some brake budget remains."""
        from src.train import available_accel
        tc = TrainConfig()
        v_half = tc.v_slide(0.320) / 2.0
        brake = available_accel(tc, v=v_half, radius_m=0.320, is_braking=True)
        assert 0.0 < brake < tc.brake_decel

    def test_coupler_reduces_accel_on_curve(self):
        """Trailing mass reduces available accel on curves via coupler force."""
        from src.train import available_accel
        tc_bare = TrainConfig(mass_trailing=0.0)
        tc_consist = TrainConfig(mass_trailing=0.600)
        v = 0.5  # well below v_slide so there IS accel budget to reduce
        a_bare = available_accel(tc_bare, v=v, radius_m=0.320)
        a_consist = available_accel(tc_consist, v=v, radius_m=0.320)
        assert a_consist < a_bare

    def test_coupler_no_effect_on_straight(self):
        """Coupler has no effect on straights (coupler angle is zero)."""
        from src.train import available_accel
        tc = TrainConfig(mass_trailing=0.600)
        assert available_accel(tc, v=0.5, radius_m=math.inf) == pytest.approx(tc.max_accel)

    def test_coupler_not_applied_during_braking(self):
        """Coupler correction skipped during braking (conservative)."""
        from src.train import available_accel
        tc_bare = TrainConfig(mass_trailing=0.0)
        tc_consist = TrainConfig(mass_trailing=0.600)
        b_bare = available_accel(tc_bare, v=0.5, radius_m=0.320, is_braking=True)
        b_consist = available_accel(tc_consist, v=0.5, radius_m=0.320, is_braking=True)
        assert b_bare == pytest.approx(b_consist)

    def test_zero_speed_full_accel_on_curve(self):
        """At v=0 on any curve, no lateral demand — full max_accel available."""
        from src.train import available_accel
        tc = TrainConfig()
        assert available_accel(tc, v=0.0, radius_m=0.320) == pytest.approx(tc.max_accel)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_train.py::TestConsistFields tests/test_train.py::TestFrictionCircle -v`

Expected: all 15 tests fail. `TestConsistFields` with `AttributeError` (missing fields). `TestFrictionCircle` with `ImportError` (missing function).

- [ ] **Step 3: Add the three fields and property to `TrainConfig` in `src/train.py`**

In the `TrainConfig` dataclass body, after the `brake_decel` field (which should currently be the last field), add:

```python
    # --- Consist mass and coupler geometry ---
    mass_loco: float = 0.370         # locomotive mass (kg)
    mass_trailing: float = 0.0       # total trailing vehicle mass (kg), 0.0 = bare loco
    coupler_offset: float = 0.100    # vehicle length for coupler angle calc (m)

    @property
    def mass_total(self) -> float:
        """Total consist mass: locomotive + trailing vehicles."""
        return self.mass_loco + self.mass_trailing
```

- [ ] **Step 4: Add `available_accel` function to `src/train.py`**

Add this function after `v_eff_array` and BEFORE the `DEFAULT_TRAIN_CONFIG = TrainConfig()` line, at module level:

```python
def available_accel(
    config: TrainConfig,
    v: float,
    radius_m: float,
    is_braking: bool = False,
) -> float:
    """Available longitudinal accel/brake at speed v on a curve of radius_m.

    Implements the friction ellipse (Kapania & Gerdes 2015, TUM/FTM pattern):
    lateral and longitudinal friction limits may differ, giving an elliptical
    combined-grip boundary rather than a circular one.

    Lateral limit: mu_design * g (all-wheel sliding friction).
    Longitudinal limit: max_accel (driven-wheel traction) or brake_decel.

    During acceleration, a one-iteration coupler correction adds the lateral
    destabilising force from trailing vehicles to the lateral demand.
    During braking, coupler compression is stabilising; omitting it is
    conservative, so no correction is applied.

    Args:
        config: Train physics configuration.
        v: Current speed (m/s).
        radius_m: Curve radius (m). math.inf for straights.
        is_braking: If True, use brake_decel as longitudinal cap.

    Returns:
        Available longitudinal acceleration/deceleration (m/s^2), >= 0.
    """
    a_lat_max = config.mu_design * config.g          # lateral friction limit
    cap = config.brake_decel if is_braking else config.max_accel  # longitudinal cap

    # No lateral demand on straights
    if math.isinf(radius_m) or radius_m <= 0:
        return cap

    a_lat = v * v / radius_m

    # Friction ellipse: a_long = cap * sqrt(1 - (a_lat / a_lat_max)^2)
    ratio = a_lat / a_lat_max
    if ratio >= 1.0:
        return 0.0
    a_long = cap * math.sqrt(1.0 - ratio * ratio)

    # Coupler correction: acceleration only, one iteration
    if not is_braking and config.mass_trailing > 0:
        phi = config.coupler_offset / (2.0 * radius_m)
        a_coupler_lat = (config.mass_trailing / config.mass_loco) * a_long * math.sin(phi)
        a_lat_total = a_lat + a_coupler_lat
        ratio_total = a_lat_total / a_lat_max
        if ratio_total >= 1.0:
            return 0.0
        a_long = cap * math.sqrt(1.0 - ratio_total * ratio_total)

    return a_long
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_train.py::TestConsistFields tests/test_train.py::TestFrictionCircle -v`

Expected: all 15 passed.

- [ ] **Step 6: Run full suite — expect no regressions**

Run: `pytest --tb=short -q`

Expected: same baseline. `available_accel` is additive — nothing in the evaluation pipeline calls it yet.

*(No commit — user controls commit timing.)*

---

## Task 2: Modify speed profile passes to use friction ellipse

**Files:**
- Modify: `src/evaluation.py`

This is the core change. The `_forward_pass` and `_backward_pass` functions currently take flat `a_max` / `a_brake` floats. They need to compute per-step friction-ellipse-limited acceleration using `available_accel` from `src/train.py`.

- [ ] **Step 1: Add `available_accel` to the imports in `src/evaluation.py`**

Find the existing import line:

```python
from .train import DEFAULT_TRAIN_CONFIG, TrainConfig, v_eff_array
```

Replace with:

```python
from .train import DEFAULT_TRAIN_CONFIG, TrainConfig, available_accel, v_eff_array
```

- [ ] **Step 2: Thread `radii_m` through the dispatch helpers**

The current `_compute_speeds_double_unroll` and `_compute_speeds_open` functions receive `train_config` but only pass `train_config.max_accel` / `train_config.brake_decel` to the inner passes. They need to also pass `radii_m` so the inner passes can compute per-step friction ellipse limits.

Replace `_compute_speeds_double_unroll`:

```python
def _compute_speeds_double_unroll(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for closed loop using double-unroll method."""
    n = len(v_limit)
    v_limit_double = np.concatenate([v_limit, v_limit])
    arc_lengths_double = np.concatenate([arc_lengths, arc_lengths])
    radii_double = np.concatenate([radii_m, radii_m])

    v_fwd = _forward_pass(v_limit_double, arc_lengths_double, radii_double, train_config)
    v_bwd = _backward_pass(v_fwd, arc_lengths_double, radii_double, train_config)

    return v_bwd[n : 2 * n]
```

Replace `_compute_speeds_open`:

```python
def _compute_speeds_open(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for open track (no wrap-around)."""
    v_fwd = _forward_pass(v_limit, arc_lengths, radii_m, train_config)
    return _backward_pass(v_fwd, arc_lengths, radii_m, train_config)
```

- [ ] **Step 3: Replace `_forward_pass` with friction-ellipse version**

Replace the entire `_forward_pass` function:

```python
def _forward_pass(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Forward pass: apply acceleration limits with friction ellipse.

    At each step, the available longitudinal acceleration is reduced by the
    lateral demand from cornering via the friction ellipse constraint.
    """
    n = len(v_limit)
    v_fwd = np.zeros(n)
    v_fwd[0] = v_limit[0]

    for i in range(1, n):
        a_max = available_accel(train_config, float(v_fwd[i - 1]), float(radii_m[i - 1]))
        v_accel = np.sqrt(v_fwd[i - 1] ** 2 + 2 * a_max * arc_lengths[i - 1])
        v_fwd[i] = min(v_limit[i], v_accel)

    return v_fwd
```

- [ ] **Step 4: Replace `_backward_pass` with friction-ellipse version**

Replace the entire `_backward_pass` function:

```python
def _backward_pass(
    v_fwd: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Backward pass: apply braking limits with friction ellipse.

    At each step, the available braking deceleration is reduced by the
    lateral demand from cornering via the friction ellipse constraint.
    """
    n = len(v_fwd)
    v_bwd = np.zeros(n)
    v_bwd[-1] = v_fwd[-1]

    for i in range(n - 2, -1, -1):
        a_brake = available_accel(
            train_config, float(v_bwd[i + 1]), float(radii_m[i]), is_braking=True
        )
        v_brake = np.sqrt(v_bwd[i + 1] ** 2 + 2 * a_brake * arc_lengths[i])
        v_bwd[i] = min(v_fwd[i], v_brake)

    return v_bwd
```

- [ ] **Step 5: Update `compute_speed_profile` to pass `radii_m` to the dispatch**

In `compute_speed_profile`, find the dispatch block (currently around the `is_closed` check). Replace:

```python
    # Apply 3-pass algorithm
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = (
        _compute_speeds_double_unroll(v_limit, arc_lengths, train_config)
        if is_closed
        else _compute_speeds_open(v_limit, arc_lengths, train_config)
    )
```

With:

```python
    # Apply 3-pass algorithm (friction ellipse reduces accel/brake on curves)
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = (
        _compute_speeds_double_unroll(v_limit, arc_lengths, radii_m, train_config)
        if is_closed
        else _compute_speeds_open(v_limit, arc_lengths, radii_m, train_config)
    )
```

`radii_m` is already computed earlier in the function body (around `radii_m = catalog.get_radii(layout.indices) / 1000.0`). No new computation needed — just pass the existing variable.

- [ ] **Step 6: Run the full test suite**

Run: `pytest --tb=short -q`

Expected: same baseline (110 passed, 2 failed pre-existing, 1 error pre-existing). The friction ellipse does NOT change results for:
- Pure R40 circle (no speed transitions → a_long = 0 everywhere → same profile)
- Pure straights (R = inf → full max_accel → same profile)
- Any layout at zero speed (a_lat = 0 → full budget)

The only layouts where results change are mixed-curvature (ovals, racetracks) where the old model allowed acceleration on curved segments and the new model restricts it. The existing tests do not assert specific avg_speed values for ovals, so they pass.

If any test fails, it means the friction-ellipse formula is producing different results for a case the plan assumed was unchanged. Stop and diagnose — read the test's layout, compute expected a_lat and a_long by hand, and compare against the function output.

*(No commit — user controls commit timing.)*

---

## Task 3: Add friction-circle-specific speed profile test

**Files:**
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Append a new test to `TestSpeedProfile` in `tests/test_evaluation.py`**

Add this test method to the existing `TestSpeedProfile` class:

```python
    def test_friction_circle_limits_accel_on_curve(self, catalog, train_config):
        """On a mixed layout, friction ellipse reduces avg_speed vs unconstrained model.

        An oval (8xR40_LEFT + 8xSTRAIGHT_16) has transitions where the old model
        allowed acceleration on curved segments. With the friction ellipse, accel
        on R40 at v near v_slide is near zero, forcing all speed changes onto the
        straights. This produces a lower avg_speed.
        """
        # Build an oval: 8 R40_LEFT curves + 8 straights
        chromosome = np.array([2] * 8 + [0] * 8, dtype=np.int32)
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        # The profile should have a meaningful speed (layout is not degenerate)
        assert profile.avg_speed > 0.5

        # The R40 curve cap is 0.886 m/s, straights cap at 1.10 m/s.
        # With friction ellipse, the transition ramps are steeper (all accel on straights).
        # avg_speed should be BELOW the motor cap because curves slow the train.
        assert profile.avg_speed < train_config.v_motor_max

        # max_speed should not exceed motor cap
        assert profile.max_speed <= train_config.v_motor_max + 0.01
```

This test does NOT compare against the old model's value (we don't have a reference). It verifies the profile is physically reasonable: positive, below motor cap, and the curve segments actually slow the train.

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_evaluation.py::TestSpeedProfile::test_friction_circle_limits_accel_on_curve -v`

Expected: PASS. The oval layout builds and the speed profile is computed with the friction ellipse active. The assertions are loose bounds that should hold for any physically reasonable profile.

- [ ] **Step 3: Run full suite — expect no regressions**

Run: `pytest --tb=short -q`

Expected: same baseline + 1 new pass = 111 passed.

*(No commit — user controls commit timing.)*

---

## Task 4: Update `configs/trains/default.yaml` for the user's consist

**Files:**
- Modify: `configs/trains/default.yaml`

- [ ] **Step 1: Add the user's consist parameters**

The user has: 1 locomotive (~370g) + 1 coal tender + 2 cargo cars (~200g each estimated, totaling ~600g trailing).

Replace the current comment-only content of `configs/trains/default.yaml` with:

```yaml
# Default LEGO train config: 1 locomotive + 3 trailing cars
#
# Friction ellipse parameters:
#   max_accel is the straight-line adhesion-limited acceleration for this consist.
#   On curves, the friction ellipse further reduces it based on lateral demand.
#   With 4 vehicles total (~0.970 kg), max_accel = mu_traction * g * m_loco / m_total
#   ≈ 0.4 * 9.81 * 0.370 / 0.970 ≈ 1.49 m/s^2.

mass_loco: 0.370
mass_trailing: 0.600
coupler_offset: 0.100
max_accel: 1.49
```

Note: `brake_decel` stays at the default (2.45) because braking friction is the same for all wheels (mass-independent). The other fields (`mu_nominal`, `mu_design`, `g`, `v_motor_max`, `gauge_b`, `cog_height_h`, `flange_angle_deg`) all keep defaults.

- [ ] **Step 2: Verify the YAML loads correctly**

Run:

```bash
python -c "
from src.train import TrainConfig
tc = TrainConfig.from_yaml('configs/trains/default.yaml')
print('mass_loco:', tc.mass_loco)
print('mass_trailing:', tc.mass_trailing)
print('mass_total:', tc.mass_total)
print('coupler_offset:', tc.coupler_offset)
print('max_accel:', tc.max_accel)
print('brake_decel:', tc.brake_decel)
"
```

Expected:
```
mass_loco: 0.37
mass_trailing: 0.6
mass_total: 0.97
coupler_offset: 0.1
max_accel: 1.49
brake_decel: 2.45
```

- [ ] **Step 3: Run full suite**

Run: `pytest --tb=short -q`

Expected: same as Task 3 result. The YAML change affects `TrainConfig.from_yaml("configs/trains/default.yaml")` which is called by `OptimizationConfig.load_train_config()`. Any test that loads the default config and runs speed profiles will now use the new values. The `test_from_yaml_default_file` test in `test_train.py` currently asserts `tc == TrainConfig()` — this will **FAIL** because the YAML now overrides defaults.

Fix: update `test_from_yaml_default_file` in `tests/test_train.py::TestYamlRoundTrip`:

```python
    def test_from_yaml_default_file(self):
        tc = TrainConfig.from_yaml("configs/trains/default.yaml")
        assert tc.mass_loco == pytest.approx(0.370)
        assert tc.mass_trailing == pytest.approx(0.600)
        assert tc.max_accel == pytest.approx(1.49)
```

Also update `test_straight_track_max_speed` in `tests/test_evaluation.py` if it uses the default config's train_config (check if it does — the `train_config` fixture returns `TrainConfig()` with defaults, NOT from YAML, so it should be unaffected).

*(No commit — user controls commit timing.)*

---

## Task 5: End-to-end verification

**Files:** no source changes.

- [ ] **Step 1: Full pytest suite**

Run: `pytest --tb=short -v 2>&1 | tail -40`

Expected: 111+ passed (110 original + 1 new friction-circle test + 15 new TrainConfig/friction tests - 1 updated YAML test), same 2 pre-existing failures, 1 collection error.

- [ ] **Step 2: Physics sanity — friction ellipse behavior**

Run:

```bash
python -c "
from src.train import TrainConfig, available_accel
import math

tc = TrainConfig.from_yaml('configs/trains/default.yaml')
print('=== User consist (1 loco + 3 cars) ===')
print(f'mass_total: {tc.mass_total:.3f} kg')
print(f'max_accel (straight): {tc.max_accel:.2f} m/s^2')
print()

for v in [0.0, 0.3, 0.5, 0.7, 0.886]:
    a_accel = available_accel(tc, v, 0.320)
    a_brake = available_accel(tc, v, 0.320, is_braking=True)
    print(f'R40 v={v:.3f}: accel={a_accel:.3f}, brake={a_brake:.3f} m/s^2')

print()
for v in [0.0, 0.5, 1.0]:
    a = available_accel(tc, v, math.inf)
    print(f'Straight v={v:.1f}: accel={a:.3f} m/s^2')
"
```

Expected: acceleration drops toward zero as v approaches v_slide (0.886 m/s) on R40. On straights, full max_accel (1.49) is available at all speeds.

- [ ] **Step 3: Optimizer end-to-end**

Run: `python main.py --config configs/with_switches.yaml 2>&1 | grep -E "Best|feasible|Done"`

Expected: completes without error. avg_speed will be LOWER than pre-friction-circle runs because the friction ellipse constrains acceleration on curves.

- [ ] **Step 4: Grep sanity — no stale flat-accel references**

Grep for `physics.max_accel` or `physics.brake_decel` in `src/` → zero matches.
Grep for `a_max` parameter name in `_forward_pass` signature → zero matches (replaced by `train_config`).

*(No commit — user controls commit timing.)*

---

## Summary of working-tree changes

**Modified files:**
- `src/train.py` — 3 new fields, 1 property, `available_accel` function (~40 lines)
- `src/evaluation.py` — `_forward_pass`, `_backward_pass`, `_compute_speeds_double_unroll`, `_compute_speeds_open` signatures updated; per-step logic uses `available_accel`
- `configs/trains/default.yaml` — user's consist mass/coupler values + `max_accel` override
- `tests/test_train.py` — `TestConsistFields` (5 tests) + `TestFrictionCircle` (10 tests)
- `tests/test_evaluation.py` — `test_friction_circle_limits_accel_on_curve` (1 test)

**Behavioral changes:**
- On curved segments at speed near v_slide: available accel drops to zero (was: full max_accel)
- On curved segments at lower speeds: accel reduced by friction ellipse (was: full max_accel)
- Straights: no change (lateral demand = 0, full max_accel available)
- Pure R40 circles: no change (constant speed, zero accel, friction ellipse not triggered)
- Ovals/racetracks: lower avg_speed due to restricted acceleration on curved transitions
- Consist with trailing mass: further reduced accel on curves via coupler correction

**Algorithm complexity:** unchanged — O(N) per pass, ~10 extra FLOPs per segment step.
