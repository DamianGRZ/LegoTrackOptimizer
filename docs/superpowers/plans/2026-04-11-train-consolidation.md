# Train Physics Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire `PhysicsConfig`, make `src/train.py::TrainConfig` the sole physics object, and encode μ as a `(mu_nominal, mu_design)` pair so the optimiser evaluates speed caps against the pessimistic (0.25) end of the friction uncertainty band.

**Architecture:** Expand `TrainConfig` with four new fields (`mu_nominal`, `mu_design`, `max_accel`, `brake_decel`); move `stud_mm` onto `TrackCatalog` (read from `metadata.stud_mm` in `track_pieces.yaml`); delete `PhysicsConfig` entirely from `src/config.py`; thread `train_config: TrainConfig` through `compute_speed_profile` and its two helpers, dropping the `physics` parameter; update `src/problem.py`, `tests/conftest.py`, `tests/test_evaluation.py`, and all four `configs/*.yaml` files accordingly.

**Tech Stack:** Python 3.x, pymoo 0.6.1.6, pydantic v2, numpy ≥1.24, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-04-11-train-consolidation-design.md`

---

## Reference: current source-of-truth line numbers

These are the pre-implementation line numbers the plan references. Do not trust them blindly after you start editing — reads will shift as you go. Use them as starting points, not authorities.

| File | Location | Current content |
|---|---|---|
| `src/train.py:22-29` | `TrainConfig` field block | `mu`, `g`, `v_motor_max`, `gauge_b`, `cog_height_h`, `flange_angle_deg` |
| `src/train.py:33-61` | scalar methods | `v_slide`, `v_tip`, `v_nadal`, `v_max`, `v_eff` — all read `self.mu` |
| `src/train.py:77-90` | `v_eff_array` function | reads `config.mu` |
| `src/config.py:39-48` | `class PhysicsConfig` | 7 fields, delete whole class |
| `src/config.py:82` | `physics: PhysicsConfig = Field(...)` | on `OptimizationConfig` |
| `src/config.py:12` | `from .train import TrainConfig` | already present |
| `src/data.py:101-111` | `TrackCatalog.__init__` | add `self.stud_mm = 8.0` default here |
| `src/data.py:125-143` | `TrackCatalog._parse_yaml` | add one line to read `metadata.stud_mm` |
| `src/evaluation.py:9` | `from .config import BoundaryConfig, PhysicsConfig` | drop `PhysicsConfig` |
| `src/evaluation.py:27-32` | `compute_speed_profile` signature | drop `physics: PhysicsConfig` |
| `src/evaluation.py:64` | `stud_to_m = physics.stud_mm / 1000.0` | switch to `catalog.stud_mm` |
| `src/evaluation.py:75` | dispatch to helpers | pass `train_config` instead of `physics` |
| `src/evaluation.py:92-96` | `_compute_speeds_double_unroll` | drop `physics` param |
| `src/evaluation.py:102-103` | `physics.max_accel` / `physics.brake_decel` | switch to `train_config` |
| `src/evaluation.py:108-112` | `_compute_speeds_open` | drop `physics` param |
| `src/evaluation.py:114-115` | `physics.max_accel` / `physics.brake_decel` | switch to `train_config` |
| `src/problem.py:107-110` | `compute_speed_profile(layout, catalog, self.config.physics, train_config=...)` | drop the positional `physics` arg |
| `tests/conftest.py:5` | `from src.config import BoundaryConfig, OptimizationConfig, PhysicsConfig` | drop `PhysicsConfig` |
| `tests/conftest.py:15-18` | `physics` fixture | rename to `train_config`, return `TrainConfig()` |
| `tests/test_evaluation.py:6` | `from src.evaluation import ...` | unchanged |
| `tests/test_evaluation.py:13,23,34,44,60,88` | six methods with `physics` fixture param | rename param |
| `tests/test_evaluation.py:18,28,39,49,64,92` | six `compute_speed_profile(...physics)` calls | rename arg |
| `tests/test_evaluation.py:77,81` | `test_utilization_range` using `default_config.physics` | add `train_config` fixture, drop `default_config.physics` |
| `configs/default.yaml:35-37` | `physics:` block (max_accel, brake_decel, stud_mm) | delete block |
| `configs/compact.yaml:34-36` | `physics:` block | delete block |
| `configs/with_switches.yaml:49-51` | `physics:` block | delete block |
| `configs/with_crossing.yaml:46-48` | `physics:` block | delete block |
| `data/track_pieces.yaml:9` | `  stud_mm: 8.0` under `metadata:` | read in `_parse_yaml` |

---

## Execution ordering rationale

**Why this order?** The refactor has two kinds of change: (1) structural additions to `TrainConfig` and `TrackCatalog` that are backward-compatible and leave tests green, and (2) a breaking signature change on `compute_speed_profile` that must happen atomically across evaluation, problem, conftest, test_evaluation, and config.py. Tasks 1–4 do the additions with their own tests. Task 5 deletes YAML blocks that would otherwise become orphaned. Task 6 is the big atomic refactor. Tasks 7–8 are cleanup and verification.

Between any two tasks the test suite must pass. Inside task 6, intermediate file states are inconsistent on purpose — only the final commit needs a green suite.

---

## Task 1: Create failing tests for TrainConfig expansion

**Files:**
- Create: `tests/test_train.py`

Goal: drive the field additions in Task 2. This is a pure test-file creation. `tests/test_train.py` does not currently exist (verified by listing `tests/`).

- [ ] **Step 1: Create `tests/test_train.py` with the full target surface**

```python
"""Tests for the portable TrainConfig physics module."""

import math

import numpy as np
import pytest

from src.train import DEFAULT_TRAIN_CONFIG, TrainConfig, v_eff_array


class TestTrainConfigFields:
    """TrainConfig expands with mu_nominal/mu_design/max_accel/brake_decel."""

    def test_default_mu_nominal(self):
        assert TrainConfig().mu_nominal == 0.30

    def test_default_mu_design(self):
        assert TrainConfig().mu_design == 0.25

    def test_default_max_accel(self):
        assert TrainConfig().max_accel == pytest.approx(3.92)

    def test_default_brake_decel(self):
        assert TrainConfig().brake_decel == pytest.approx(2.45)

    def test_legacy_mu_field_removed(self):
        assert not hasattr(TrainConfig(), "mu")


class TestSpeedCapsUseMuDesign:
    """Scalar and vectorised speed caps read mu_design, not mu_nominal."""

    def test_v_slide_r40_at_mu_design(self):
        # R40 = 0.320 m, mu_design = 0.25 -> sqrt(0.25*9.81*0.320) = 0.8862
        assert TrainConfig().v_slide(0.320) == pytest.approx(0.8862, abs=1e-3)

    def test_v_eff_r40_equals_v_slide(self):
        # Sliding binds below motor cap
        tc = TrainConfig()
        assert tc.v_eff(0.320) == pytest.approx(tc.v_slide(0.320), abs=1e-6)

    def test_v_eff_straight_equals_motor_cap(self):
        assert TrainConfig().v_eff(math.inf) == pytest.approx(1.10, abs=1e-6)

    def test_v_eff_array_vectorised(self):
        tc = TrainConfig()
        out = v_eff_array(tc, np.array([0.320, 0.448, np.inf]))
        assert np.allclose(out, [0.8862, 1.0488, 1.10], atol=1e-3)


class TestYamlRoundTrip:
    """YAML loader still tolerates empty files and ignores unknown keys."""

    def test_from_yaml_default_file(self):
        tc = TrainConfig.from_yaml("configs/trains/default.yaml")
        assert tc == TrainConfig()

    def test_default_singleton(self):
        assert DEFAULT_TRAIN_CONFIG == TrainConfig()
```

- [ ] **Step 2: Run the new test file to confirm everything fails**

Run: `pytest tests/test_train.py -v`

Expected: the first run errors or reports failures on every test in `TestTrainConfigFields` and `TestSpeedCapsUseMuDesign`. Specifically, `test_default_mu_nominal` and `test_default_mu_design` fail with `AttributeError: 'TrainConfig' object has no attribute 'mu_nominal'`, and the R40 sanity check fails against the current `mu = 0.30` output of 0.970 m/s.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_train.py
git commit -m "test(train): add failing tests for TrainConfig expansion"
```

---

## Task 2: Implement TrainConfig expansion

**Files:**
- Modify: `src/train.py`

Goal: make the tests from Task 1 pass. Rename `mu` → `mu_nominal`, add `mu_design`, `max_accel`, `brake_decel`, and flip every formula to read `mu_design`.

- [ ] **Step 1: Replace the TrainConfig field block**

Edit `src/train.py:18-29`. Replace the existing `TrainConfig` dataclass body header and fields with:

```python
@dataclass(frozen=True)
class TrainConfig:
    """Immutable lateral-stability physical parameters for a LEGO train.

    The friction coefficient is expressed as a (nominal, design) pair:
    mu_nominal is the central estimate for diagnostics; mu_design is the
    pessimistic value that every speed cap formula actually reads. This
    keeps the friction-uncertainty band explicit in the data model while
    making the optimiser design for the worst plausible friction.
    """

    # --- Friction (nominal = reference, design = the value used by formulas) ---
    mu_nominal: float = 0.30         # central wheel-rail friction estimate (diagnostics only)
    mu_design: float = 0.25          # pessimistic friction used by v_slide / v_nadal / v_eff_array
    # --- Environment ---
    g: float = 9.81                  # gravitational acceleration (m/s^2)
    # --- Motor ---
    v_motor_max: float = 1.10        # Powered Up drive-train top speed (m/s)
    # --- Bogie / wheel geometry ---
    gauge_b: float = 0.0375          # inner rail-to-rail gauge (m)
    cog_height_h: float = 0.030      # CoG above rail head (m)
    flange_angle_deg: float = 50.0   # effective flange contact angle (deg)
    # --- Speed-profile dynamics (moved from PhysicsConfig) ---
    max_accel: float = 3.92          # maximum acceleration (m/s^2)
    brake_decel: float = 2.45        # braking deceleration (m/s^2)
```

- [ ] **Step 2: Flip scalar methods to read `mu_design`**

Edit `src/train.py:33-53`. Replace the three derailment-mode scalar methods:

```python
    def v_slide(self, r_m: float) -> float:
        """Lateral sliding cap: sqrt(mu_design * g * R). Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        return math.sqrt(self.mu_design * self.g * r_m)

    def v_tip(self, r_m: float) -> float:
        """Tip-over cap: sqrt(g * R * (b/2) / h). Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        return math.sqrt(self.g * r_m * (self.gauge_b / 2.0) / self.cog_height_h)

    def v_nadal(self, r_m: float) -> float:
        """Nadal wheel-climb cap via L/V criterion at mu_design. Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        tan_d = math.tan(math.radians(self.flange_angle_deg))
        lv_crit = (tan_d - self.mu_design) / (1.0 + self.mu_design * tan_d)
        if lv_crit <= 0:
            return math.inf
        return math.sqrt(self.g * r_m * lv_crit)
```

`v_tip` is unchanged but included for clarity. `v_max` and `v_eff` at `src/train.py:55-61` need no edits — they already call the scalar helpers.

- [ ] **Step 3: Flip `v_eff_array` to read `mu_design`**

Edit `src/train.py:77-90`. Replace the whole function body:

```python
def v_eff_array(config: TrainConfig, radii_m: np.ndarray) -> np.ndarray:
    """Vectorized effective speed cap over an array of radii (in metres).

    Uses config.mu_design (pessimistic friction). Handles inf radii: np.sqrt(inf)
    = inf, and np.minimum with v_motor_max collapses straights to the motor cap.
    """
    r = np.asarray(radii_m, dtype=np.float64)
    v_slide = np.sqrt(config.mu_design * config.g * r)
    v_tip = np.sqrt(config.g * r * (config.gauge_b / 2.0) / config.cog_height_h)
    tan_d = math.tan(math.radians(config.flange_angle_deg))
    lv_crit = (tan_d - config.mu_design) / (1.0 + config.mu_design * tan_d)
    v_nadal = np.sqrt(config.g * r * lv_crit) if lv_crit > 0 else np.full_like(r, np.inf)
    v_max = np.minimum.reduce([v_slide, v_tip, v_nadal])
    return np.minimum(v_max, config.v_motor_max)
```

- [ ] **Step 4: Run Task 1's tests — expect PASS**

Run: `pytest tests/test_train.py -v`

Expected: all tests in `TestTrainConfigFields`, `TestSpeedCapsUseMuDesign`, and `TestYamlRoundTrip` pass. Numeric tolerances: `v_slide(0.320) ≈ 0.8862`, `v_eff(inf) = 1.10`, `v_eff_array([0.320, 0.448, inf]) ≈ [0.8862, 1.0488, 1.10]`.

- [ ] **Step 5: Run the full test suite — expect PASS**

Run: `pytest --tb=short -q`

Expected: the existing `test_evaluation.py` tests continue to pass. The R40 numeric shift (0.970 → 0.886 m/s) does not break `test_r40_circle_speed_limit` because the current assertion `profile.max_speed <= 0.97 + 0.01` is loose enough to absorb the lower new value, and `profile.avg_speed < 1.0` also stays valid at 0.886. The signature of `compute_speed_profile` has not been touched yet — nothing else was disturbed. Pre-existing unrelated failures (`test_decoder.py` collection error and any `compute_dimensions`-signature failures in `test_problem.py` / `test_sampling.py`) are out of scope and should be ignored for this run.

- [ ] **Step 6: Commit**

```bash
git add src/train.py
git commit -m "feat(train): expand TrainConfig with mu band and dynamics fields"
```

---

## Task 3: Add failing test for `TrackCatalog.stud_mm`

**Files:**
- Modify: `tests/test_data.py`

Goal: drive the `stud_mm` attribute addition in Task 4. Append a small test class — do not touch existing tests.

- [ ] **Step 1: Append a `TestStudMm` class at the end of `tests/test_data.py`**

```python
class TestStudMm:
    """TrackCatalog exposes stud_mm read from track_pieces.yaml metadata."""

    def test_catalog_has_stud_mm_attribute(self, catalog):
        assert hasattr(catalog, "stud_mm")

    def test_stud_mm_value_from_yaml(self, catalog):
        # data/track_pieces.yaml -> metadata.stud_mm = 8.0
        assert catalog.stud_mm == pytest.approx(8.0)
```

If the existing file does not already import `pytest`, add `import pytest` at the top. Check first — it probably already does.

- [ ] **Step 2: Run only the new tests — expect FAIL**

Run: `pytest tests/test_data.py::TestStudMm -v`

Expected: `test_catalog_has_stud_mm_attribute` fails because `TrackCatalog` has no `stud_mm` attribute yet.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_data.py
git commit -m "test(data): add failing test for TrackCatalog.stud_mm"
```

---

## Task 4: Implement `TrackCatalog.stud_mm`

**Files:**
- Modify: `src/data.py`

Goal: make Task 3's test pass by reading `metadata.stud_mm` out of `track_pieces.yaml` into a catalog attribute.

- [ ] **Step 1: Initialise the attribute with a safe default in `__init__`**

Edit `src/data.py:101-111` (inside `TrackCatalog.__init__`). After the line `self._topologies: Dict[int, PieceTopology] = {}`, add one line:

```python
        self.stud_mm: float = 8.0
```

The attribute is typed `float` with an 8.0 fallback so that catalogs built from YAML files missing a `metadata` block still load without raising. This tolerant default is consistent with `TrainConfig.from_yaml`'s "missing keys keep defaults" semantics.

- [ ] **Step 2: Populate `stud_mm` from the YAML in `_parse_yaml`**

Edit `src/data.py:125`. At the very start of `_parse_yaml`, before the `piece_index = data.get("piece_index", {})` line, insert:

```python
        metadata = data.get("metadata", {}) or {}
        self.stud_mm = float(metadata.get("stud_mm", 8.0))
```

Rationale: `track_pieces.yaml:5-12` puts `stud_mm` under `metadata:`, not at the top level. Reading with `.get("metadata", {}).get("stud_mm", 8.0)` tolerates both the current file structure and a degenerate file with no `metadata` block. The `or {}` handles the edge case where `metadata:` is present as an explicit `null` or empty block.

- [ ] **Step 3: Run Task 3's tests — expect PASS**

Run: `pytest tests/test_data.py::TestStudMm -v`

Expected: both `test_catalog_has_stud_mm_attribute` and `test_stud_mm_value_from_yaml` pass.

- [ ] **Step 4: Run the full test suite — expect PASS**

Run: `pytest --tb=short -q`

Expected: no new failures. `evaluation.py` still reads `physics.stud_mm`, not `catalog.stud_mm` — the catalog attribute is additive and nothing consumes it yet.

- [ ] **Step 5: Commit**

```bash
git add src/data.py
git commit -m "feat(data): expose stud_mm on TrackCatalog from metadata.stud_mm"
```

---

## Task 5: Delete `physics:` blocks from optimizer config YAMLs

**Files:**
- Modify: `configs/default.yaml:35-37`
- Modify: `configs/compact.yaml:34-36`
- Modify: `configs/with_switches.yaml:49-51`
- Modify: `configs/with_crossing.yaml:46-48`

Goal: remove the `physics:` blocks so that Task 6's deletion of `PhysicsConfig` does not orphan them. Pydantic v2 defaults to `extra='ignore'`, so these blocks are currently silently deserialised into `PhysicsConfig` defaults but they encode no information the optimizer will read. Each block contains exactly three lines: `max_accel: 3.92`, `brake_decel: 2.45`, `stud_mm: 8.0`.

- [ ] **Step 1: Delete the `physics:` block from `configs/default.yaml`**

Remove lines 35-37 (the `physics:` header and its three child entries). Leave surrounding blocks untouched. If there's a blank line before or after the block, preserve existing blank-line spacing so the YAML remains visually tidy.

- [ ] **Step 2: Delete the `physics:` block from `configs/compact.yaml`**

Remove the equivalent three-line block at lines 34-36.

- [ ] **Step 3: Delete the `physics:` block from `configs/with_switches.yaml`**

Remove the equivalent three-line block at lines 49-51.

- [ ] **Step 4: Delete the `physics:` block from `configs/with_crossing.yaml`**

Remove the equivalent three-line block at lines 46-48.

- [ ] **Step 5: Grep check — zero `physics:` blocks remain in configs**

Run (use the Grep tool, not raw grep/rg):

Pattern: `^physics:|^  physics:|physics:\s*$`, path: `configs/`

Expected: zero matches under `configs/` (matches inside `docs/` or `src/` are fine and expected).

- [ ] **Step 6: Run the full test suite — expect PASS**

Run: `pytest --tb=short -q`

Expected: no new failures. `OptimizationConfig.load` still deserialises these YAMLs successfully — the `physics` field on `OptimizationConfig` just falls back to `default_factory=PhysicsConfig` since the YAML no longer supplies a value.

- [ ] **Step 7: Commit**

```bash
git add configs/default.yaml configs/compact.yaml configs/with_switches.yaml configs/with_crossing.yaml
git commit -m "chore(configs): remove physics: blocks from optimizer YAML configs"
```

---

## Task 6: Atomic refactor — drop `physics` from `compute_speed_profile` and delete `PhysicsConfig`

**Files:**
- Modify: `src/evaluation.py`
- Modify: `src/problem.py:107-110`
- Modify: `src/config.py:39-48, 82`
- Modify: `tests/conftest.py:5, 15-18`
- Modify: `tests/test_evaluation.py` (6 method signatures + 6 call sites + 1 special case)

Goal: atomically change the public signature of `compute_speed_profile` from `(layout, catalog, physics, *, train_config=...)` to `(layout, catalog, train_config=...)`, update every consumer, and delete `PhysicsConfig` class and field. This is one logical unit: intermediate file states are broken on purpose, only the final commit needs a green test suite. Execute the steps in the order below — they are ordered so that each downstream edit has the types it needs when you finally run the tests.

- [ ] **Step 1: Update `compute_speed_profile` signature and body**

Edit `src/evaluation.py:27-89`. Replace lines 27 through the end of `compute_speed_profile`'s `return` block. Key changes: drop `physics: PhysicsConfig` from the parameter list, read `stud_to_m` from `catalog.stud_mm`, pass `train_config` (not `physics`) to the internal helpers, update the docstring.

```python
def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> SpeedProfile:
    """Compute time-optimal speed profile using 3-pass algorithm.

    Derailing prevention is built into Pass 1 via the TrainConfig speed cap.
    See src/train.py for the portable lateral-stability physics model.

    Algorithm:
    1. Pass 1: Curvature limits via TrainConfig.v_eff(R) at mu_design
    2. Pass 2: Forward acceleration - respect train_config.max_accel
    3. Pass 3: Backward braking - respect train_config.brake_decel
    4. Double-unroll method for closed loops

    Args:
        layout: Track layout with geometry
        catalog: Track catalog for piece properties (also provides stud_mm)
        train_config: Portable locomotive physics (default: DEFAULT_TRAIN_CONFIG)

    Returns:
        SpeedProfile with speeds, avg_speed, lap_time, etc.
    """
    if layout.n_pieces == 0:
        return SpeedProfile(
            speeds=np.array([]),
            avg_speed=0.0,
            lap_time=0.0,
            total_distance=0.0,
            max_speed=0.0,
            min_speed=0.0,
        )

    # Get piece properties and convert units
    stud_to_m = catalog.stud_mm / 1000.0
    arc_lengths = catalog.get_arc_lengths(layout.indices) * stud_to_m  # meters
    radii_m = catalog.get_radii(layout.indices) / 1000.0  # meters
    speed_limits = catalog.get_speed_limits(layout.indices)  # m/s

    # Pass 1: Curvature speed limits (vectorized) - THIS PREVENTS DERAILING
    v_curve = v_eff_array(train_config, radii_m)
    v_limit = np.minimum(v_curve, speed_limits)

    # Apply 3-pass algorithm
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = (
        _compute_speeds_double_unroll(v_limit, arc_lengths, train_config)
        if is_closed
        else _compute_speeds_open(v_limit, arc_lengths, train_config)
    )

    # Compute metrics
    total_distance = float(np.sum(arc_lengths))
    lap_time = _compute_lap_time(speeds, arc_lengths)
    avg_speed = total_distance / lap_time if lap_time > 0 else 0.0

    return SpeedProfile(
        speeds=speeds,
        avg_speed=avg_speed,
        lap_time=lap_time,
        total_distance=total_distance,
        max_speed=float(np.max(speeds)) if len(speeds) > 0 else 0.0,
        min_speed=float(np.min(speeds)) if len(speeds) > 0 else 0.0,
    )
```

- [ ] **Step 2: Update the two private helpers to take `train_config`**

Edit `src/evaluation.py:92-115`. Replace both helpers:

```python
def _compute_speeds_double_unroll(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for closed loop using double-unroll method."""
    n = len(v_limit)
    v_limit_double = np.concatenate([v_limit, v_limit])
    arc_lengths_double = np.concatenate([arc_lengths, arc_lengths])

    v_fwd = _forward_pass(v_limit_double, arc_lengths_double, train_config.max_accel)
    v_bwd = _backward_pass(v_fwd, arc_lengths_double, train_config.brake_decel)

    return v_bwd[n : 2 * n]


def _compute_speeds_open(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    """Compute speeds for open track (no wrap-around)."""
    v_fwd = _forward_pass(v_limit, arc_lengths, train_config.max_accel)
    return _backward_pass(v_fwd, arc_lengths, train_config.brake_decel)
```

`_forward_pass` and `_backward_pass` below them at `evaluation.py:118-149` are already parametrised by raw `a_max` / `a_brake` floats and need no changes.

- [ ] **Step 3: Drop the `PhysicsConfig` import from `src/evaluation.py`**

Edit `src/evaluation.py:9`. Change:

```python
from .config import BoundaryConfig, PhysicsConfig
```

to:

```python
from .config import BoundaryConfig
```

`BoundaryConfig` is still needed by `compute_constraints` lower in the file.

- [ ] **Step 4: Update the `compute_speed_profile` call in `src/problem.py`**

Edit `src/problem.py:107-110`. Replace:

```python
        # F[1]: -avg_speed
        speed_profile = compute_speed_profile(
            layout, self.catalog, self.config.physics,
            train_config=self._train_config,
        )
```

with:

```python
        # F[1]: -avg_speed
        speed_profile = compute_speed_profile(
            layout, self.catalog, train_config=self._train_config,
        )
```

`self._train_config` was already populated at `problem.py:68` in the previous refactor, so no other problem.py lines need touching.

- [ ] **Step 5: Rename the `physics` fixture in `tests/conftest.py`**

Edit `tests/conftest.py:5` to drop the `PhysicsConfig` import:

```python
from src.config import BoundaryConfig, OptimizationConfig
from src.data import TrackCatalog
from src.train import TrainConfig
```

Edit `tests/conftest.py:15-18` to rename the fixture:

```python
@pytest.fixture
def train_config() -> TrainConfig:
    """Default train physics configuration."""
    return TrainConfig()
```

No other conftest fixtures need touching.

- [ ] **Step 6: Rename fixture parameter at six `test_evaluation.py` method signatures**

Edit `tests/test_evaluation.py` method signatures on lines 13, 23, 34, 44, 60, 88. At each line, replace the `physics` parameter with `train_config`:

- Line 13: `def test_straight_track_max_speed(self, catalog, train_config):`
- Line 23: `def test_r40_circle_speed_limit(self, catalog, train_config):`
- Line 34: `def test_double_unroll_closure(self, catalog, train_config):`
- Line 44: `def test_empty_layout(self, catalog, train_config):`
- Line 60: `def test_objectives_correct_signs(self, catalog, train_config, default_config):`
- Line 88: `def test_speed_objective_negative(self, catalog, train_config, default_config):`

- [ ] **Step 7: Update six `compute_speed_profile` calls inside those methods**

Edit the six call sites at `test_evaluation.py:18, 28, 39, 49, 64, 92`. At each call, replace the positional `physics` argument with a keyword `train_config=train_config`. Concretely each call becomes:

```python
profile = compute_speed_profile(layout, catalog, train_config=train_config)
```

- [ ] **Step 7b: Tighten the R40 numeric assertion to reflect `mu_design = 0.25`**

Edit `test_evaluation.py:30-32`. The current assertions inside `test_r40_circle_speed_limit` are:

```python
        # R40 curves have catalog speed limit of 0.97 m/s
        assert profile.avg_speed < 1.0
        assert profile.max_speed <= 0.97 + 0.01  # Small tolerance
```

Replace them with:

```python
        # R40 curves cap at sqrt(mu_design * g * R) = sqrt(0.25 * 9.81 * 0.320) ≈ 0.886 m/s
        assert profile.avg_speed < 1.0
        assert profile.max_speed <= 0.89
```

The `avg_speed < 1.0` upper bound is retained unchanged; at `mu_design = 0.25` the R40 circle produces `avg_speed ≈ 0.886` which still satisfies the old loose bound, and tightening the `avg_speed` assertion would couple this test to the forward-backward pass wrap-around behaviour in a way that adds no useful coverage. Only the `max_speed` assertion is tightened because that one is directly anchored to `v_slide`.

- [ ] **Step 8: Update `test_utilization_range` which uses `default_config.physics`**

Edit `tests/test_evaluation.py:77-86`. The current method signature is `def test_utilization_range(self, catalog, default_config):` and line 81 reads `profile = compute_speed_profile(layout, catalog, default_config.physics)`. Since `default_config.physics` no longer exists after Step 10, and since the test only needs a TrainConfig (not a full OptimizationConfig), replace the method body so it uses the `train_config` fixture directly:

```python
    def test_utilization_range(self, catalog, train_config, default_config):
        """Utilization in [0, 1] range."""
        chromosome = np.array([0] * 10 + [-1] * 54, dtype=np.int32)  # 10 pieces
        layout = build_layout(chromosome, catalog)
        profile = compute_speed_profile(layout, catalog, train_config=train_config)

        F = compute_objectives(layout, profile, catalog, default_config.total_inventory)

        utilization = -F[0]  # Flip sign to get actual utilization
        assert 0.0 <= utilization <= 1.0
```

`default_config` is still needed for `total_inventory`, so it stays in the parameter list.

- [ ] **Step 9: Delete the `PhysicsConfig` class from `src/config.py`**

Edit `src/config.py:39-48`. Delete the entire `class PhysicsConfig(BaseModel):` block (all ten lines from the class header through the `stud_mm` field line). Leave the blank lines around it for readability.

- [ ] **Step 10: Delete the `physics` field from `OptimizationConfig`**

Edit `src/config.py:82`. Delete the single line:

```python
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
```

The `train_config_path` field on line 83 and the `_base_dir` PrivateAttr on line 89 are retained.

- [ ] **Step 11: Run the full test suite — expect PASS**

Run: `pytest --tb=short -q`

Expected: the entire suite passes, modulo pre-existing unrelated failures documented in the session summary (`test_decoder.py` collection error from a missing `ChromosomeDimensions` import and `compute_dimensions`-signature failures in `test_problem.py` / `test_sampling.py`). Those are out of scope and their count should not increase. If you see any NEW failure — a test that was passing before Task 6 but is now failing — stop and fix the root cause before committing. Do not commit a red suite.

Likely failure modes to diagnose:
- `TypeError: compute_speed_profile() got unexpected keyword argument 'physics'` → you missed a callsite in `test_evaluation.py`. Grep for `physics` under `tests/` to find it.
- `AttributeError: 'OptimizationConfig' object has no attribute 'physics'` → a call outside `problem.py` still reads `config.physics`. Grep for `config.physics` and `.physics` under `src/` and `tests/`.
- `AttributeError: ... 'stud_mm'` on a `TrackCatalog` built from a non-default YAML → your `_parse_yaml` edit in Task 4 did not run for that file; check whether Step 2 of Task 4 lands before `_parse_yaml` returns.

- [ ] **Step 12: Grep sanity check — zero `PhysicsConfig` references remain**

Use the Grep tool:

Pattern: `PhysicsConfig`, path: `src/` → expect zero matches
Pattern: `PhysicsConfig`, path: `tests/` → expect zero matches
Pattern: `\.physics\b`, path: `src/` → expect zero matches
Pattern: `\.physics\b`, path: `tests/` → expect zero matches

If any match comes back, fix it before committing. It means something still reads the deleted type.

- [ ] **Step 13: Commit the whole atomic refactor**

```bash
git add src/evaluation.py src/problem.py src/config.py tests/conftest.py tests/test_evaluation.py
git commit -m "refactor: drop PhysicsConfig, thread TrainConfig through evaluation"
```

---

## Task 7: Delete `configs/trains/with_car.yaml`

**Files:**
- Delete: `configs/trains/with_car.yaml`

Goal: remove the now-obsolete scaffolding file. The "one train configuration" decision made per-config overrides unnecessary, and this file currently has no consumer. `configs/trains/default.yaml` is retained — it is referenced from every optimizer config via `train_config_path: trains/default.yaml`.

- [ ] **Step 1: Confirm nothing references `with_car.yaml`**

Grep for `with_car` under the repository root (Grep tool, not raw grep).

Pattern: `with_car`, path: (repository root) → expect zero matches

If there are matches outside this plan document itself, stop and investigate before deleting.

- [ ] **Step 2: Remove the file via git**

```bash
git rm configs/trains/with_car.yaml
```

This removes the file both from disk and from the index in one step. This is the only destructive filesystem action in the plan and it is authorised by the spec's "Delete configs/trains/with_car.yaml" line under "Config YAML changes".

- [ ] **Step 3: Run the full test suite — expect PASS**

Run: `pytest --tb=short -q`

Expected: no change from Task 6's green state. No test loads `with_car.yaml`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(configs): remove obsolete trains/with_car.yaml scaffolding"
```

---

## Task 8: End-to-end verification

**Files:** no source changes. This task runs the full verification suite and produces literal evidence that the refactor is complete.

- [ ] **Step 1: Full pytest suite, verbose**

Run: `pytest --tb=short -v 2>&1 | tail -80`

Expected: suite completes. Zero test failures relative to Task 6 baseline. Pre-existing failures (`test_decoder.py` collection error, `compute_dimensions` signature failures) may still appear — capture their count and confirm it has not increased. Paste the literal tail output under this step before ticking it off.

- [ ] **Step 2: Unit-level physics sanity (ad-hoc)**

Run:

```bash
python -c "
from src.train import TrainConfig, v_eff_array
import numpy as np
tc = TrainConfig()
print('mu_nominal:', tc.mu_nominal)
print('mu_design: ', tc.mu_design)
print('max_accel: ', tc.max_accel)
print('brake_decel:', tc.brake_decel)
print('v_slide(0.320):', round(tc.v_slide(0.320), 4))
print('v_eff(0.320): ', round(tc.v_eff(0.320), 4))
print('v_eff(inf):   ', tc.v_eff(float('inf')))
out = v_eff_array(tc, np.array([0.320, 0.448, np.inf]))
print('v_eff_array:  ', np.round(out, 4).tolist())
"
```

Expected literal output:
```
mu_nominal: 0.3
mu_design:  0.25
max_accel:  3.92
brake_decel: 2.45
v_slide(0.320): 0.8862
v_eff(0.320):  0.8862
v_eff(inf):    1.1
v_eff_array:   [0.8862, 1.0488, 1.1]
```

Paste the literal output under this step before ticking it off.

- [ ] **Step 3: Portability smoke test**

Run:

```bash
python -c "
import sys
# Block any accidental src.config or src.data import
class Blocker:
    def find_module(self, name, path=None):
        if name.startswith('src.') and name not in ('src', 'src.train'):
            return self
    def load_module(self, name):
        raise ImportError(f'blocked: {name}')
sys.meta_path.insert(0, Blocker())
from src.train import TrainConfig, v_eff_array
tc = TrainConfig()
print('v_eff(0.320):', round(tc.v_eff(0.320), 4))
print('portable OK')
"
```

Expected output:
```
v_eff(0.320): 0.8862
portable OK
```

This confirms `src/train.py` loads with zero transitive imports from anywhere else in `src/`. If the import blocker triggers, the portability invariant is broken and the refactor has a bug (something snuck an import into `src/train.py`).

- [ ] **Step 4: Optimizer end-to-end — default config**

Run: `/optimize -c default` (via the Skill tool)

Expected: optimisation completes, writes outputs under `outputs/`. `avg_speed` on R40-dominated layouts is noticeably lower than pre-refactor runs (0.886 cap instead of 0.970 cap).

- [ ] **Step 5: Diagnostic report**

Run: `/diag` (via the Skill tool)

Expected: closure error ≤ tolerance, orphan switches = 0, avg_speed reported. Paste the literal diagnostic summary under this step before ticking it off.

- [ ] **Step 6: Switches config end-to-end**

Run: `/optimize -c with_switches` (via the Skill tool)

Expected: completes without error. Switch pairs produce branches in the best layout.

- [ ] **Step 7: Final grep sanity**

Use the Grep tool to confirm the consolidation is complete:

- Pattern: `PhysicsConfig` → expect zero matches anywhere in the repo (including `docs/` — mentions in `docs/superpowers/specs/2026-04-11-train-consolidation-design.md` are acceptable, note and ignore).
- Pattern: `physics\.(max_accel|brake_decel|stud_mm|safety_factor|friction_coeff|gravity|motor_top_speed)` → zero matches in `src/` and `tests/`.
- Pattern: `self\.config\.physics` → zero matches in `src/`.

If anything slipped through, fix it and commit before ticking this step off.

- [ ] **Step 8: Final commit (docs update if needed)**

If any literal outputs from Steps 1, 2, 3, or 5 revealed a divergence from expectations that the plan did not predict, document it in the spec file `docs/superpowers/specs/2026-04-11-train-consolidation-design.md` under a new "Implementation observations" section and commit:

```bash
git add docs/superpowers/specs/2026-04-11-train-consolidation-design.md
git commit -m "docs(spec): record train consolidation implementation observations"
```

If there are no observations worth recording, skip the commit. Do not force an empty commit.

---

## Summary of commits this plan produces

1. `test(train): add failing tests for TrainConfig expansion` — Task 1
2. `feat(train): expand TrainConfig with mu band and dynamics fields` — Task 2
3. `test(data): add failing test for TrackCatalog.stud_mm` — Task 3
4. `feat(data): expose stud_mm on TrackCatalog from metadata.stud_mm` — Task 4
5. `chore(configs): remove physics: blocks from optimizer YAML configs` — Task 5
6. `refactor: drop PhysicsConfig, thread TrainConfig through evaluation` — Task 6
7. `chore(configs): remove obsolete trains/with_car.yaml scaffolding` — Task 7
8. (optional) `docs(spec): record train consolidation implementation observations` — Task 8

Seven guaranteed commits, one conditional. The test suite is green at the tip of each commit.
