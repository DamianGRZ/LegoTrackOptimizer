# Design: Train physics consolidation and friction calibration band

**Date:** 2026-04-11
**Scope:** Retire `PhysicsConfig`. Make `src/train.py` the sole physics object. Express the friction coefficient as a nominal/design pair so the optimiser evaluates speed caps against the pessimistic end of the μ uncertainty band.
**Status:** Approved for implementation planning.

## Background

The previous refactor moved lateral-stability speed caps into `src/train.py` (`TrainConfig`, `v_eff_array`), leaving `src/config.py::PhysicsConfig` with seven fields of which four are dead reads (`safety_factor`, `friction_coeff`, `gravity`, `motor_top_speed`) and three are still live (`max_accel`, `brake_decel`, `stud_mm`). This split made the codebase less portable, not more: every evaluation call still has to take both `physics: PhysicsConfig` and `train_config: TrainConfig`, and a reader has to know which knob lives where.

In parallel, `docs/Locomotive_dynamics.md` documents μ as a calibration band (0.25–0.40, central 0.30), noting that μ "propagates as a ±15% uncertainty in critical speeds". The current code uses a single central value and offers the GA no way to prefer layouts that stay feasible at the pessimistic end.

## Goals

1. **Single physics object.** `TrainConfig` is the only place the rest of the codebase reads train physics from. `PhysicsConfig` no longer exists.
2. **Explicit friction band.** `TrainConfig` carries both a nominal μ (for diagnostics and documentation) and a design μ (used by every formula). The optimiser evaluates at the design μ by default, producing layouts that stay feasible under pessimistic friction.
3. **Dedupe `stud_mm`.** The LEGO stud constant lives on `TrackCatalog`, where the rest of the track-geometry data lives. It is removed from the physics object entirely.
4. **Preserve portability.** `src/train.py` must continue to import only stdlib + numpy + yaml. No new cross-module dependencies.

## Non-goals

- **No multi-car / consist modelling.** One fixed locomotive configuration, as agreed.
- **No velocity-dependent acceleration.** `max_accel` stays as a flat float; the motor torque curve is out of scope.
- **No per-piece risk penalty.** Per-piece speed limits remain in `data/track_pieces.yaml`.
- **No new GA objectives or constraints.** Number and shape of `F` / `G` vectors are unchanged.

## Architecture

After this change, the physics data flow is:

```
configs/*.yaml (no physics: block)
    └─> OptimizationConfig
          └─> train_config_path: trains/default.yaml
                └─> TrainConfig.from_yaml()
                      ├─ mu_nominal, mu_design, g, v_motor_max,
                      │  gauge_b, cog_height_h, flange_angle_deg,
                      │  max_accel, brake_decel
                      └─> consumed by compute_speed_profile()

data/track_pieces.yaml
    └─> TrackCatalog
          └─ stud_mm (new attribute, read from YAML line 9)
                └─> consumed by compute_speed_profile() for stud→metre conversion
```

Every evaluation call takes `(catalog, train_config)` where it used to take `(catalog, physics)`. No callsite takes both.

## `TrainConfig` field changes

| Field | Action | Default | Purpose |
|---|---|---|---|
| `mu` | **renamed** → `mu_nominal` | 0.30 | Informational only. Reference central estimate from `docs/Locomotive_dynamics.md`. No code consumer. |
| (new) `mu_design` | **added** | 0.25 | Pessimistic μ. Every scalar and vectorised formula reads this. |
| `g` | unchanged | 9.81 | |
| `v_motor_max` | unchanged | 1.10 | |
| `gauge_b` | unchanged | 0.0375 | |
| `cog_height_h` | unchanged | 0.030 | |
| `flange_angle_deg` | unchanged | 50.0 | |
| (new) `max_accel` | **added** | 3.92 | Moved from `PhysicsConfig`. |
| (new) `brake_decel` | **added** | 2.45 | Moved from `PhysicsConfig`. |

**Scalar methods** (`v_slide`, `v_tip`, `v_nadal`, `v_max`, `v_eff`) read `self.mu_design` where they previously read `self.mu`. No method-level μ override parameter — portability rule: "no optional arguments just in case". A future caller that wants to evaluate at `mu_nominal` can construct a second `TrainConfig` instance.

**Vectorised function** `v_eff_array(config, radii_m)` reads `config.mu_design`.

**YAML loading.** `from_yaml` keeps its tolerant semantics: missing fields keep the defaults above. An empty YAML file continues to yield `TrainConfig()`.

## `PhysicsConfig` removal

Delete `class PhysicsConfig` from `src/config.py` entirely. Delete the `physics: PhysicsConfig = Field(default_factory=PhysicsConfig)` field from `OptimizationConfig`. Remove the now-unused import in `src/evaluation.py`.

**Migration of the three live fields:**

| Old path | New path |
|---|---|
| `physics.max_accel` | `train_config.max_accel` |
| `physics.brake_decel` | `train_config.brake_decel` |
| `physics.stud_mm` | `catalog.stud_mm` |

The four dead fields (`safety_factor`, `friction_coeff`, `gravity`, `motor_top_speed`) are simply dropped — no new home.

## `TrackCatalog` gets `stud_mm`

`data/track_pieces.yaml:9` already contains `stud_mm: 8.0` but nothing currently reads it (confirmed by grep). The YAML loader in `src/data.py::TrackCatalog` gains one line: read the top-level `stud_mm` key and store it as `self.stud_mm: float` (default 8.0 if missing, for resilience).

Usage in `src/evaluation.py` line 64:
```python
# before
stud_to_m = physics.stud_mm / 1000.0
# after
stud_to_m = catalog.stud_mm / 1000.0
```

## `compute_speed_profile` signature change

```python
# before
def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    physics: PhysicsConfig,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> SpeedProfile: ...

# after
def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> SpeedProfile: ...
```

The same signature shape applies to the internal helpers `_run_speed_profile` and `_run_speed_profile_double_unroll` (currently at `evaluation.py:95` and `evaluation.py:111`): the `physics: PhysicsConfig` parameter is dropped, `train_config: TrainConfig` takes its place, and `max_accel` / `brake_decel` reads come from `train_config` instead.

## Config YAML changes

Every `configs/*.yaml` loses its entire `physics:` block (lines 35-37 in `default.yaml`, 34-36 in `compact.yaml`, 49-51 in `with_switches.yaml`, 46-48 in `with_crossing.yaml`). The `train_config_path: trains/default.yaml` line added in the previous refactor remains.

`configs/trains/default.yaml` stays empty (comment-only), which loads as `TrainConfig()` defaults. The existing `configs/trains/with_car.yaml` scaffolding is **deleted** — the "one train configuration" decision makes per-config overrides unnecessary and the file is unused.

## Test impact

**`tests/conftest.py` (line 14-18):** the `physics` fixture is renamed to `train_config` and returns `TrainConfig()` instead of `PhysicsConfig()`. The `PhysicsConfig` import is removed.

**`tests/test_evaluation.py`:** every callsite using the `physics` fixture (lines 13, 23, 34, 44, 60, 88 — eight `compute_speed_profile` calls total) renames the parameter to `train_config`. The line-81 special case (`default_config.physics`) switches to using the `train_config` fixture directly, dropping the `default_config` dependency at that callsite.

**Numeric assertion update:** `test_r40_circle_speed_limit` (line 23) currently asserts `profile.max_speed <= 0.97 + 0.01`. At `mu_design = 0.25` the R40 cap becomes `sqrt(0.25 · 9.81 · 0.320) = 0.886 m/s`. Update the assertion to `<= 0.89`. The existing `avg_speed < 1.0` bound on the same test stays valid at `mu_design = 0.25` and is left unchanged.

**Unchanged:** `test_straight_track_max_speed` — still caps at `v_motor_max = 1.10 m/s`.

## Accepted behavioural changes

- **R40 curve cap:** 0.970 → 0.886 m/s (−8.7%). This is the design intent of E1.
- **`avg_speed` objective values drop** on layouts dominated by R40 curves. Pareto fronts shift toward wider-radius solutions.
- **Straight-track speeds unchanged** (motor-capped at 1.10).
- **Third-party scripts importing `PhysicsConfig` break loudly.** No silent fallback, no deprecation stub.

## Implementation order

1. Edit `src/train.py`: rename `mu` → `mu_nominal`, add `mu_design`, add `max_accel`, add `brake_decel`. Update all scalar methods and `v_eff_array` to read `mu_design`.
2. Edit `src/data.py`: `TrackCatalog` gains `stud_mm: float` attribute, loaded from `track_pieces.yaml` with default 8.0.
3. Edit `src/config.py`: delete `PhysicsConfig` class, delete `physics` field on `OptimizationConfig`.
4. Edit `src/evaluation.py`: drop `physics` parameter from `compute_speed_profile`, `_run_speed_profile`, `_run_speed_profile_double_unroll`; switch reads to `train_config.max_accel`, `train_config.brake_decel`, `catalog.stud_mm`; remove `PhysicsConfig` import.
5. Edit `src/problem.py`: update `_evaluate` call to drop `self.config.physics`.
6. Delete `physics:` block from `configs/default.yaml`, `configs/compact.yaml`, `configs/with_switches.yaml`, `configs/with_crossing.yaml`.
7. Delete `configs/trains/with_car.yaml`.
8. Edit `tests/conftest.py`: rename `physics` fixture → `train_config`, drop `PhysicsConfig` import, return `TrainConfig()`.
9. Edit `tests/test_evaluation.py`: rename fixture parameter at eight callsites; update the R40 numeric bound for the 0.886 m/s cap.
10. Run `/test` full suite. Fix any callsite misses. Do not modify production code to keep old test numbers passing — the numeric shift is the point.
11. Run `/optimize -c default` and `/diag` end-to-end to confirm the pipeline still produces feasible layouts and the reported `avg_speed` reflects the lower cap.

## Verification

**Unit-level physics sanity** (replaces the checks in the previous refactor):

```python
from src.train import TrainConfig, v_eff_array
import numpy as np

tc = TrainConfig()
assert tc.mu_nominal == 0.30
assert tc.mu_design == 0.25
assert tc.max_accel == 3.92
assert tc.brake_decel == 2.45

# R40 radius = 0.320 m, now evaluated at mu_design = 0.25
assert abs(tc.v_slide(0.320) - 0.8862) < 1e-3
assert abs(tc.v_eff(0.320)   - 0.8862) < 1e-3
assert tc.v_eff(float("inf")) == 1.10

out = v_eff_array(tc, np.array([0.320, 0.448, np.inf]))
# 0.448 m radius: v_slide = sqrt(0.25 * 9.81 * 0.448) = 1.049 -> still below motor cap
assert np.allclose(out, [0.8862, 1.0488, 1.10], atol=1e-3)
```

**Integration checks:**

- `/test` — full suite, zero new failures (pre-existing `test_decoder.py` collection error and `compute_dimensions` failures are out of scope).
- `/optimize -c default` — completes, writes outputs/.
- `/diag` — reports sensible closure error and an `avg_speed` lower than the pre-consolidation run for R40-dominant layouts.
- Grep confirms zero references to `PhysicsConfig` remain anywhere in `src/` or `tests/`.

**Portability smoke test:** from a clean Python session, `import src.train; TrainConfig()` works with only stdlib + numpy + yaml installed. No `src.config` or `src.data` import is pulled in transitively.
