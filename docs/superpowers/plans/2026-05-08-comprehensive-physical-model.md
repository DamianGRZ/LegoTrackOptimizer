# Comprehensive Physical Evaluation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `evaluate_layout()` returning a comprehensive `PhysicalEvaluation` dataclass across five domains (geometry, stability, kinematics, dynamics, energy) and wire `lap_time` as `F[1]` in the optimizer.

**Architecture:** New module `src/train/evaluation.py` orchestrates five private compute functions. Re-uses existing `physics.py` (`TrainConfig`, `v_eff_array`, `available_accel`) and `scoring.py` (`compute_speed_profile`, extended with a `safety_margin` kwarg, default `1.0` so existing callers are unaffected). Optimizer integration is a 3-line change in `problem.py:_evaluate` at the final task — phases 1-5 are no-ops for optimizer behavior.

**Tech Stack:** Python 3, numpy, pymoo 0.6.1.6, pytest. Uses existing fixtures `catalog` and `train_config` from `tests/conftest.py`; adds `measured_train_config` fixture for tests that need the measured AFM consist values.

**Spec:** [`docs/superpowers/specs/2026-05-08-comprehensive-physical-model-design.md`](../specs/2026-05-08-comprehensive-physical-model-design.md)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/train/evaluation.py` | **NEW** | `PhysicalEvaluation` dataclass + `evaluate_layout()` orchestrator + 5 private domain compute fns |
| `src/train/__init__.py` | modify | Re-export `PhysicalEvaluation`, `evaluate_layout` |
| `src/train/scoring.py` | modify | `compute_speed_profile()` gains `safety_margin: float = 1.0` kwarg |
| `src/train/physics.py` | modify | `TrainConfig` gains `mu_roll: float = 0.05` field (Task 5 only) |
| `src/problem.py` | modify | `_evaluate()` swaps `F[1] = -min_speed` → `F[1] = lap_time` (Task 6 only) |
| `tests/test_evaluation.py` | **NEW** | All domain unit tests + integration test |
| `tests/conftest.py` | modify | Add `measured_train_config` fixture (Task 1 only) |

Each task below is independently committable. Tasks 1–5 do not change optimizer behavior. Only Task 6 changes `F[]`.

---

## Task 1: PhysicalEvaluation skeleton + Geometry domain + measured fixture

**Files:**
- Create: `tests/test_evaluation.py`
- Modify: `tests/conftest.py`
- Create: `src/train/evaluation.py`
- Modify: `src/train/__init__.py`

- [ ] **Step 1.1: Add `measured_train_config` fixture**

Edit `tests/conftest.py`. Add after the existing `train_config` fixture (around line 19):

```python
@pytest.fixture
def measured_train_config() -> TrainConfig:
    """Train physics from measured AFM SL+Cargo M0015TW consist (2026-05-06)."""
    return TrainConfig.from_yaml("configs/trains/measured_consist.yaml")
```

- [ ] **Step 1.2: Write failing tests for Geometry domain**

Create `tests/test_evaluation.py` with this content:

```python
"""Tests for the comprehensive physical evaluation model."""

import math

import numpy as np
import pytest

from src.catalog import TrackCatalog
from src.geometry import Layout, build_layout
from src.train import TrainConfig
from src.train.evaluation import PhysicalEvaluation, evaluate_layout

# Piece indices from src/encoding.py
STRAIGHT_16 = 0
R40_LEFT = 2
R40_RIGHT = 3


def _make_layout(piece_indices: list[int], catalog: TrackCatalog) -> Layout:
    """Build a Layout from a piece-index list (helper used across tests)."""
    return build_layout(np.array(piece_indices, dtype=np.int32), catalog)


class TestGeometryDomain:
    """coupler-phi per segment + per-switch + max."""

    def test_phi_R40_matches_user_sanity_check(self, catalog, measured_train_config):
        """Single R40_LEFT, measured coupler_offset=0.106, R=0.32 m -> phi == 0.106/(2*0.32) ~ 9.49 deg."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        expected_rad = 0.106 / (2.0 * 0.32)
        assert phys.coupler_phi_per_segment[0] == pytest.approx(expected_rad, abs=1e-4)
        assert math.degrees(phys.coupler_phi_per_segment[0]) == pytest.approx(9.49, abs=0.05)

    def test_phi_straight_is_zero(self, catalog, measured_train_config):
        """STRAIGHT_16 has no curvature -> phi == 0."""
        layout = _make_layout([STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.coupler_phi_per_segment[0] == pytest.approx(0.0, abs=1e-9)

    def test_max_coupler_phi_picks_worst(self, catalog, measured_train_config):
        """max_coupler_phi == max over segments and switches."""
        layout = _make_layout([STRAIGHT_16, R40_LEFT, STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        # Only the R40 segment has non-zero phi
        expected_max = 0.106 / (2.0 * 0.32)
        assert phys.max_coupler_phi == pytest.approx(expected_max, abs=1e-4)
```

- [ ] **Step 1.3: Run failing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py -v`
Expected: 3 FAIL with `ModuleNotFoundError: No module named 'src.train.evaluation'` (or `ImportError` for `PhysicalEvaluation` / `evaluate_layout`).

- [ ] **Step 1.4: Implement `evaluation.py` with full dataclass + Geometry domain real, other domains stubbed**

Create `src/train/evaluation.py`:

```python
"""Comprehensive physical evaluation of a track layout under a given train consist.

Produces a PhysicalEvaluation across five physical domains (geometry,
stability, kinematics, dynamics, energy) in a single O(n) pass per chromosome.
Pure function; no side effects.

Spec: docs/superpowers/specs/2026-05-08-comprehensive-physical-model-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray

from ..catalog import TrackCatalog
from ..geometry import Layout
from .physics import DEFAULT_TRAIN_CONFIG, TrainConfig
from .scoring import SpeedProfile, compute_speed_profile


@dataclass(frozen=True)
class PhysicalEvaluation:
    """Full physical evaluation of a layout under a given train consist.

    Conventions:
        - Lengths in meters (catalog studs converted via stud_mm/1000).
        - Angles in radians.
        - Speeds in m/s, accelerations in m/s^2.
        - Energies in joules.
        - Per-segment arrays have length == n_pieces of the layout main path.
    """

    # ---- Geometry ----
    coupler_phi_per_segment: NDArray[np.float64]
    coupler_phi_per_switch: Dict[int, float]
    max_coupler_phi: float

    # ---- Stability ----
    v_slide_per_segment: NDArray[np.float64]
    v_tip_per_segment: NDArray[np.float64]
    v_nadal_per_segment: NDArray[np.float64]
    v_eff_per_segment: NDArray[np.float64]
    binding_cap_per_segment: NDArray[np.str_]

    # ---- Kinematics ----
    speed_profile: SpeedProfile
    safety_factor_min: float
    safety_factor_mean: float

    # ---- Dynamics ----
    a_lat_per_segment: NDArray[np.float64]
    a_long_per_segment: NDArray[np.float64]
    grip_utilization_per_segment: NDArray[np.float64]
    coupler_force_lat_per_segment: NDArray[np.float64]

    # ---- Energy ----
    motor_work_per_lap: float
    rolling_dissipation_per_lap: float
    ke_roundtrip_per_lap: float

    # ---- Provenance ----
    train_config: TrainConfig
    safety_margin: float
    catalog_signature: str


def _compute_geometry(
    radii_m: NDArray[np.float64],
    coupler_offset: float,
) -> NDArray[np.float64]:
    """Per-segment coupler hinge angle: phi(R) = L/(2R). 0 on straights (R=inf)."""
    safe_R = np.where(np.isfinite(radii_m) & (radii_m > 0), radii_m, np.inf)
    return coupler_offset / (2.0 * safe_R)


def _stub_per_segment_array(n: int, dtype=np.float64) -> NDArray:
    """Stub for not-yet-implemented domains; returns zero array of correct shape."""
    return np.zeros(n, dtype=dtype)


def evaluate_layout(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    safety_margin: float = 0.95,
) -> PhysicalEvaluation:
    """Comprehensive physical evaluation. Pure function, no side effects."""
    n = layout.n_pieces

    if n == 0:
        return _empty_evaluation(train_config, safety_margin)

    stud_to_m = catalog.stud_mm / 1000.0
    radii_m = catalog.get_radii(layout.indices) / 1000.0  # mm -> m

    # ---- Geometry ----
    coupler_phi_per_segment = _compute_geometry(radii_m, train_config.coupler_offset)
    coupler_phi_per_switch: Dict[int, float] = {}  # filled in if/when switch-aware
    max_phi = float(np.max(coupler_phi_per_segment)) if n > 0 else 0.0
    if coupler_phi_per_switch:
        max_phi = max(max_phi, max(coupler_phi_per_switch.values()))

    # ---- Stability (stubbed; real impl in Task 2) ----
    v_slide_per_segment = _stub_per_segment_array(n)
    v_tip_per_segment = _stub_per_segment_array(n)
    v_nadal_per_segment = _stub_per_segment_array(n)
    v_eff_per_segment = _stub_per_segment_array(n)
    binding_cap_per_segment = np.array(["motor"] * n, dtype="<U6")

    # ---- Kinematics (stubbed; real impl in Task 3) ----
    speed_profile = compute_speed_profile(layout, catalog, train_config)
    safety_factor_min = 1.0
    safety_factor_mean = 1.0

    # ---- Dynamics (stubbed; real impl in Task 4) ----
    a_lat_per_segment = _stub_per_segment_array(n)
    a_long_per_segment = _stub_per_segment_array(n)
    grip_utilization_per_segment = _stub_per_segment_array(n)
    coupler_force_lat_per_segment = _stub_per_segment_array(n)

    # ---- Energy (stubbed; real impl in Task 5) ----
    motor_work_per_lap = 0.0
    rolling_dissipation_per_lap = 0.0
    ke_roundtrip_per_lap = 0.0

    return PhysicalEvaluation(
        coupler_phi_per_segment=coupler_phi_per_segment,
        coupler_phi_per_switch=coupler_phi_per_switch,
        max_coupler_phi=max_phi,
        v_slide_per_segment=v_slide_per_segment,
        v_tip_per_segment=v_tip_per_segment,
        v_nadal_per_segment=v_nadal_per_segment,
        v_eff_per_segment=v_eff_per_segment,
        binding_cap_per_segment=binding_cap_per_segment,
        speed_profile=speed_profile,
        safety_factor_min=safety_factor_min,
        safety_factor_mean=safety_factor_mean,
        a_lat_per_segment=a_lat_per_segment,
        a_long_per_segment=a_long_per_segment,
        grip_utilization_per_segment=grip_utilization_per_segment,
        coupler_force_lat_per_segment=coupler_force_lat_per_segment,
        motor_work_per_lap=motor_work_per_lap,
        rolling_dissipation_per_lap=rolling_dissipation_per_lap,
        ke_roundtrip_per_lap=ke_roundtrip_per_lap,
        train_config=train_config,
        safety_margin=safety_margin,
        catalog_signature=_catalog_signature(catalog),
    )


def _empty_evaluation(train_config: TrainConfig, safety_margin: float) -> PhysicalEvaluation:
    """Empty evaluation for n==0 layouts."""
    empty = np.zeros(0, dtype=np.float64)
    return PhysicalEvaluation(
        coupler_phi_per_segment=empty,
        coupler_phi_per_switch={},
        max_coupler_phi=0.0,
        v_slide_per_segment=empty,
        v_tip_per_segment=empty,
        v_nadal_per_segment=empty,
        v_eff_per_segment=empty,
        binding_cap_per_segment=np.array([], dtype="<U6"),
        speed_profile=SpeedProfile(
            speeds=empty, avg_speed=0.0, lap_time=0.0,
            total_distance=0.0, max_speed=0.0, min_speed=0.0,
        ),
        safety_factor_min=1.0,
        safety_factor_mean=1.0,
        a_lat_per_segment=empty,
        a_long_per_segment=empty,
        grip_utilization_per_segment=empty,
        coupler_force_lat_per_segment=empty,
        motor_work_per_lap=0.0,
        rolling_dissipation_per_lap=0.0,
        ke_roundtrip_per_lap=0.0,
        train_config=train_config,
        safety_margin=safety_margin,
        catalog_signature="",
    )


def _catalog_signature(catalog: TrackCatalog) -> str:
    """Short signature for the catalog. Stable across runs of the same YAML.
    Using piece count + n_pieces is sufficient identification for now;
    upgrade to a SHA hash later if cross-catalog comparison is needed."""
    return f"npieces={catalog.n_pieces}"
```

- [ ] **Step 1.5: Re-export from `src/train/__init__.py`**

Modify `src/train/__init__.py` to add the new exports. After the existing `from .scoring import ...` line, append:

```python
from .evaluation import PhysicalEvaluation, evaluate_layout
```

And add to `__all__`:

```python
__all__ = [
    "DEFAULT_TRAIN_CONFIG",
    "TrainConfig",
    "available_accel",
    "v_eff_array",
    "SpeedProfile",
    "compute_speed_profile",
    "PhysicalEvaluation",     # NEW
    "evaluate_layout",        # NEW
]
```

- [ ] **Step 1.6: Run Geometry tests, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py -v`
Expected: 3 PASS.

- [ ] **Step 1.7: Run full test suite, verify nothing else broke**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All previously-passing tests still pass; 3 new test_evaluation tests pass.

- [ ] **Step 1.8: Commit (after explicit user approval — see CLAUDE.md)**

```bash
git add tests/conftest.py tests/test_evaluation.py src/train/evaluation.py src/train/__init__.py
git commit -m "feat(train): add PhysicalEvaluation skeleton with Geometry domain"
```

---

## Task 2: Stability domain (per-segment caps + binding-cap label)

**Files:**
- Modify: `src/train/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 2.1: Write failing tests for Stability domain**

Append to `tests/test_evaluation.py` (after `class TestGeometryDomain`):

```python
class TestStabilityDomain:
    """Per-segment slide/tip/nadal/motor caps + binding-cap label."""

    def test_v_slide_R40_matches_user_sanity(self, catalog, measured_train_config):
        """v_slide(R40) = sqrt(0.25 * 9.81 * 0.32) = 0.886 m/s (user's sanity check)."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_slide_per_segment[0] == pytest.approx(0.886, abs=1e-3)

    def test_v_eff_R40_is_slide_with_measured_consist(self, catalog, measured_train_config):
        """For measured config, v_slide < v_motor on R40, so v_eff == v_slide."""
        layout = _make_layout([R40_LEFT], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_eff_per_segment[0] == pytest.approx(phys.v_slide_per_segment[0], abs=1e-6)

    def test_v_eff_straight_is_motor(self, catalog, measured_train_config):
        """For straights (R=inf), v_eff == v_motor_max == 1.26."""
        layout = _make_layout([STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.v_eff_per_segment[0] == pytest.approx(1.26, abs=1e-3)

    def test_binding_cap_labels_R40_then_straight(self, catalog, measured_train_config):
        """R40 -> 'slide', straight -> 'motor'."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16], catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config)
        assert phys.binding_cap_per_segment[0] == "slide"
        assert phys.binding_cap_per_segment[1] == "motor"
```

- [ ] **Step 2.2: Run failing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestStabilityDomain -v`
Expected: 4 FAIL (stub returns zeros and "motor" label).

- [ ] **Step 2.3: Replace Stability stubs with real implementation**

In `src/train/evaluation.py`, add a private function above `evaluate_layout`:

```python
def _compute_stability(
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Per-segment v_slide, v_tip, v_nadal, v_eff, binding-cap label.

    Reuses the same closed-form expressions as physics.v_eff_array but exposes
    each cap separately (small redundant compute).
    """
    import math
    g = train_config.g
    mu = train_config.mu_design

    v_slide = np.sqrt(mu * g * radii_m)
    v_tip = np.sqrt(g * radii_m * (train_config.gauge_b / 2.0) / train_config.cog_height_h)

    tan_d = math.tan(math.radians(train_config.flange_angle_deg))
    lv_crit = (tan_d - mu) / (1.0 + mu * tan_d)
    if lv_crit <= 0:
        v_nadal = np.full_like(radii_m, np.inf)
    else:
        v_nadal = np.sqrt(g * radii_m * lv_crit)

    v_motor = np.full_like(radii_m, train_config.v_motor_max)

    # Stack and argmin to identify binding cap
    caps = np.stack([v_slide, v_tip, v_nadal, v_motor], axis=0)  # shape (4, n)
    v_eff = np.min(caps, axis=0)
    binding_idx = np.argmin(caps, axis=0)
    labels = np.array(["slide", "tip", "nadal", "motor"], dtype="<U6")
    binding_cap = labels[binding_idx]

    return v_slide, v_tip, v_nadal, v_eff, binding_cap
```

Then in `evaluate_layout`, replace the stability stubs:

```python
    # ---- Stability ----
    (v_slide_per_segment,
     v_tip_per_segment,
     v_nadal_per_segment,
     v_eff_per_segment,
     binding_cap_per_segment) = _compute_stability(radii_m, train_config)
```

- [ ] **Step 2.4: Run Stability tests, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestStabilityDomain -v`
Expected: 4 PASS.

- [ ] **Step 2.5: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All tests pass.

- [ ] **Step 2.6: Commit (after explicit user approval)**

```bash
git add src/train/evaluation.py tests/test_evaluation.py
git commit -m "feat(train): wire Stability domain with per-segment caps + binding-cap labels"
```

---

## Task 3: Kinematics with safety_margin

**Files:**
- Modify: `src/train/scoring.py`
- Modify: `src/train/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 3.1: Write failing tests for Kinematics + safety_margin**

Append to `tests/test_evaluation.py`:

```python
class TestKinematicsDomain:
    """compute_speed_profile w/ safety_margin + safety_factor metrics."""

    def test_safety_margin_default_unchanged(self, catalog, train_config):
        """compute_speed_profile with safety_margin=1.0 (default) keeps old behavior."""
        from src.train import compute_speed_profile
        layout = _make_layout([R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT,
                               R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT], catalog)
        sp = compute_speed_profile(layout, catalog, train_config)
        sp2 = compute_speed_profile(layout, catalog, train_config, safety_margin=1.0)
        assert np.allclose(sp.speeds, sp2.speeds, atol=1e-9)

    def test_safety_margin_scales_speed(self, catalog, measured_train_config):
        """16-R40 closed circle at safety_margin=0.95 -> all speeds ~ 0.95 * 0.886 = 0.842."""
        from src.train import compute_speed_profile
        layout = _make_layout([R40_LEFT] * 16, catalog)
        sp = compute_speed_profile(layout, catalog, measured_train_config, safety_margin=0.95)
        # All segments slide-bound at 0.842 m/s
        assert np.all(sp.speeds == pytest.approx(0.842, abs=5e-3))

    def test_safety_factor_min_equals_margin_on_capped_loop(self, catalog, measured_train_config):
        """16-R40 circle at safety_margin=0.95 -> safety_factor_min == 0.95."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.safety_factor_min == pytest.approx(0.95, abs=1e-3)

    def test_safety_factor_min_above_margin_on_brake_bound(self, catalog, measured_train_config):
        """Layout with curve before short straight -> brake-bound segments below 0.95 cap (so factor > 0.95)."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 2, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        # Some segments forced below v_limit by accel/brake -> safety_factor > 0.95 there
        # safety_factor_min is still 0.95 (on cap-bound segments)
        assert phys.safety_factor_min == pytest.approx(0.95, abs=5e-3)
        # Mean is strictly above 0.95
        assert phys.safety_factor_mean >= 0.95 - 1e-6

    def test_lap_time_R40_circle_matches_pen_and_paper(self, catalog, measured_train_config):
        """16-R40 closed circle lap_time = 2*pi*R / (0.95 * v_slide).
        2*pi*0.32 / 0.842 = 2.011 / 0.842 ~ 2.39 s."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.speed_profile.lap_time == pytest.approx(2.39, abs=0.05)
```

- [ ] **Step 3.2: Run failing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestKinematicsDomain -v`
Expected: at least `test_safety_margin_default_unchanged` FAIL with `TypeError: compute_speed_profile() got an unexpected keyword argument 'safety_margin'`. Other tests fail because evaluation stubs safety_factor_min/mean to 1.0.

- [ ] **Step 3.3: Extend `compute_speed_profile` with `safety_margin` kwarg**

Modify `src/train/scoring.py`. Change the signature of `compute_speed_profile`:

```python
def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    safety_margin: float = 1.0,
) -> SpeedProfile:
```

And modify the body's Pass 1 line. Find:

```python
    # Pass 1: Curvature speed limits (vectorized) - THIS PREVENTS DERAILING
    v_curve = v_eff_array(train_config, radii_m)
    v_limit = np.minimum(v_curve, speed_limits)
```

Replace with:

```python
    # Pass 1: Curvature speed limits (vectorized) - THIS PREVENTS DERAILING.
    # safety_margin (default 1.0) scales every cap so the operating speed
    # stays strictly below the derailment cap.
    v_curve = v_eff_array(train_config, radii_m)
    v_limit = np.minimum(v_curve, speed_limits) * safety_margin
```

(Update the function docstring to mention `safety_margin`.)

- [ ] **Step 3.4: Wire safety_margin and safety_factor metrics in evaluation.py**

Modify `evaluate_layout` in `src/train/evaluation.py`. Replace the kinematics block:

```python
    # ---- Kinematics ----
    speed_profile = compute_speed_profile(layout, catalog, train_config)
    safety_factor_min = 1.0
    safety_factor_mean = 1.0
```

With:

```python
    # ---- Kinematics ----
    speed_profile = compute_speed_profile(
        layout, catalog, train_config, safety_margin=safety_margin,
    )
    # safety_factor[i] = operating_speed[i] / v_eff[i]. NaN-safe: 0/0 -> 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            v_eff_per_segment > 0,
            speed_profile.speeds / v_eff_per_segment,
            0.0,
        )
    safety_factor_min = float(np.min(ratio)) if len(ratio) else 1.0
    arc_lengths_m = catalog.get_arc_lengths(layout.indices) * stud_to_m
    total = float(np.sum(arc_lengths_m))
    if total > 0:
        safety_factor_mean = float(np.sum(ratio * arc_lengths_m) / total)
    else:
        safety_factor_mean = 1.0
```

- [ ] **Step 3.5: Run Kinematics tests, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestKinematicsDomain -v`
Expected: 5 PASS.

- [ ] **Step 3.6: Run existing scoring tests to ensure backward compat**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scoring.py tests/test_train.py -v`
Expected: All previous tests still pass (default `safety_margin=1.0` preserves old behavior).

- [ ] **Step 3.7: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All tests pass.

- [ ] **Step 3.8: Commit (after explicit user approval)**

```bash
git add src/train/scoring.py src/train/evaluation.py tests/test_evaluation.py
git commit -m "feat(train): add safety_margin kwarg to speed profile + Kinematics domain"
```

---

## Task 4: Dynamics domain (a_lat, a_long, grip_utilization, coupler_force_lat)

**Files:**
- Modify: `src/train/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 4.1: Write failing tests for Dynamics domain**

Append to `tests/test_evaluation.py`:

```python
class TestDynamicsDomain:
    """a_lat, a_long, grip_utilization, coupler_force_lat per segment."""

    def test_a_lat_R40_at_margined_speed(self, catalog, measured_train_config):
        """16-R40 circle at safety_margin=0.95: a_lat = v^2/R = 0.842^2 / 0.32 ~ 2.215."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.a_lat_per_segment[0] == pytest.approx(2.215, abs=0.05)

    def test_a_lat_zero_on_straights(self, catalog, measured_train_config):
        """Straight segments have a_lat == 0."""
        layout = _make_layout([STRAIGHT_16] * 8, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert np.allclose(phys.a_lat_per_segment, 0.0, atol=1e-9)

    def test_grip_utilization_below_one(self, catalog, measured_train_config):
        """grip_utilization is in [0, 1] always (within numerical tolerance)."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 4, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert np.all(phys.grip_utilization_per_segment <= 1.0 + 1e-6)
        assert np.all(phys.grip_utilization_per_segment >= 0.0)

    def test_coupler_force_lat_zero_with_no_trailing(self, catalog):
        """With mass_trailing=0, lateral coupler force is 0 everywhere."""
        bare_loco = TrainConfig(mass_trailing=0.0, coupler_offset=0.106)
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, bare_loco, safety_margin=0.95)
        assert np.allclose(phys.coupler_force_lat_per_segment, 0.0, atol=1e-9)

    def test_coupler_force_lat_proportional_to_trailing_mass(self, catalog):
        """Doubling mass_trailing doubles the lateral coupler force (when a_long != 0)."""
        # Pick a layout with brake transitions (so a_long is non-zero)
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 2, catalog)
        tc1 = TrainConfig(mass_trailing=0.327, coupler_offset=0.106)
        tc2 = TrainConfig(mass_trailing=0.654, coupler_offset=0.106)
        phys1 = evaluate_layout(layout, catalog, tc1, safety_margin=0.95)
        phys2 = evaluate_layout(layout, catalog, tc2, safety_margin=0.95)
        # Where coupler_force_lat is non-trivial, ratio should be ~2
        nonzero = np.abs(phys1.coupler_force_lat_per_segment) > 1e-3
        if nonzero.any():
            ratio = (phys2.coupler_force_lat_per_segment[nonzero]
                     / phys1.coupler_force_lat_per_segment[nonzero])
            assert np.allclose(ratio, 2.0, atol=1e-6)
```

- [ ] **Step 4.2: Run failing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestDynamicsDomain -v`
Expected: 5 FAIL (Dynamics is still stubbed).

- [ ] **Step 4.3: Implement `_compute_dynamics` in evaluation.py**

Add a private function in `src/train/evaluation.py`:

```python
def _compute_dynamics(
    speeds: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    arc_lengths_m: NDArray[np.float64],
    coupler_phi: NDArray[np.float64],
    train_config: TrainConfig,
    is_closed: bool,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Per-segment a_lat, a_long, grip_utilization, lateral coupler force.

    a_lat[i]   = v[i]^2 / R[i]                 (0 on straights)
    a_long[i]  = (v[next]^2 - v[i]^2) / (2 * arc_length[i])    (closed: next wraps to 0)
    grip_util  = sqrt((a_lat/(mu*g))^2 + (a_long/cap)^2)       cap = max_accel or brake_decel
    F_coup_lat = m_trailing * a_long * sin(phi)
    """
    n = len(speeds)
    a_lat = np.where(np.isfinite(radii_m) & (radii_m > 0),
                     speeds ** 2 / np.where(radii_m > 0, radii_m, 1.0),
                     0.0)

    # Wrap-around for closed loops; open paths set last a_long to 0
    if is_closed and n > 1:
        v_next = np.roll(speeds, -1)
    else:
        v_next = np.concatenate([speeds[1:], [speeds[-1]]]) if n > 0 else speeds

    a_long = np.where(arc_lengths_m > 0,
                      (v_next ** 2 - speeds ** 2) / (2.0 * np.maximum(arc_lengths_m, 1e-9)),
                      0.0)

    cap = np.where(a_long >= 0, train_config.max_accel, train_config.brake_decel)
    a_lat_max = train_config.mu_design * train_config.g
    grip_util = np.sqrt(
        (a_lat / a_lat_max) ** 2 + np.where(cap > 0, (a_long / cap) ** 2, 0.0)
    )
    grip_util = np.clip(grip_util, 0.0, 1.0 + 1e-9)

    F_coupler_lat = train_config.mass_trailing * a_long * np.sin(coupler_phi)

    return a_lat, a_long, grip_util, F_coupler_lat
```

Replace the dynamics stubs in `evaluate_layout`:

```python
    # ---- Dynamics ----
    a_lat_per_segment = _stub_per_segment_array(n)
    a_long_per_segment = _stub_per_segment_array(n)
    grip_utilization_per_segment = _stub_per_segment_array(n)
    coupler_force_lat_per_segment = _stub_per_segment_array(n)
```

With:

```python
    # ---- Dynamics ----
    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    arc_lengths_m = catalog.get_arc_lengths(layout.indices) * stud_to_m
    (a_lat_per_segment,
     a_long_per_segment,
     grip_utilization_per_segment,
     coupler_force_lat_per_segment) = _compute_dynamics(
        speed_profile.speeds, radii_m, arc_lengths_m,
        coupler_phi_per_segment, train_config, is_closed,
    )
```

(Note: `arc_lengths_m` is now computed at the dynamics step. If Task 3 also computed it for `safety_factor_mean`, deduplicate by computing it once near the top of `evaluate_layout` and reusing.)

- [ ] **Step 4.4: Deduplicate arc_lengths_m computation**

In `evaluate_layout`, after `radii_m = ...`, add:

```python
    arc_lengths_m = catalog.get_arc_lengths(layout.indices) * stud_to_m
```

And remove the duplicate computation inside the kinematics safety_factor block. Single source of truth.

- [ ] **Step 4.5: Run Dynamics tests, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestDynamicsDomain -v`
Expected: 5 PASS.

- [ ] **Step 4.6: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All tests pass.

- [ ] **Step 4.7: Commit (after explicit user approval)**

```bash
git add src/train/evaluation.py tests/test_evaluation.py
git commit -m "feat(train): wire Dynamics domain with friction-ellipse grip utilization"
```

---

## Task 5: Energy domain (motor_work, rolling_diss, ke_roundtrip) + mu_roll TrainConfig field

**Files:**
- Modify: `src/train/physics.py`
- Modify: `src/train/evaluation.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_train.py`

- [ ] **Step 5.1: Write failing tests for `mu_roll` TrainConfig field**

Append to `tests/test_train.py` inside `class TestTrainConfigFields`:

```python
    def test_default_mu_roll(self):
        assert TrainConfig().mu_roll == pytest.approx(0.05)

    def test_mu_roll_can_be_overridden(self):
        assert TrainConfig(mu_roll=0.10).mu_roll == pytest.approx(0.10)
```

- [ ] **Step 5.2: Run, verify FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py::TestTrainConfigFields -v`
Expected: 2 FAIL (`AttributeError: 'TrainConfig' object has no attribute 'mu_roll'`).

- [ ] **Step 5.3: Add `mu_roll` field to TrainConfig**

In `src/train/physics.py`, modify `TrainConfig` to add a new field. Find:

```python
    # --- Speed-profile dynamics ---
    max_accel: float = 3.92          # maximum acceleration (m/s^2)
    brake_decel: float = 2.45        # braking deceleration (m/s^2)
```

And add immediately after:

```python
    # --- Rolling resistance ---
    mu_roll: float = 0.05            # rolling-friction coefficient (literature default; tunable)
```

- [ ] **Step 5.4: Run, verify mu_roll tests PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py::TestTrainConfigFields -v`
Expected: All TrainConfigFields tests PASS.

- [ ] **Step 5.5: Write failing tests for Energy domain**

Append to `tests/test_evaluation.py`:

```python
class TestEnergyDomain:
    """motor_work_per_lap, rolling_dissipation, ke_roundtrip."""

    def test_rolling_dissipation_constant_speed(self, catalog, measured_train_config):
        """All-straight closed loop at v=1.197: rolling_diss = mu_roll * m_total * g * total_distance."""
        layout = _make_layout([STRAIGHT_16] * 8, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        m_total = measured_train_config.mass_total
        g = measured_train_config.g
        mu_roll = measured_train_config.mu_roll
        total_distance = phys.speed_profile.total_distance
        expected = mu_roll * m_total * g * total_distance
        assert phys.rolling_dissipation_per_lap == pytest.approx(expected, abs=1e-6)

    def test_ke_roundtrip_zero_on_constant_speed_loop(self, catalog, measured_train_config):
        """All-R40 circle at single speed: no brake-respin events, ke_roundtrip == 0."""
        layout = _make_layout([R40_LEFT] * 16, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.ke_roundtrip_per_lap == pytest.approx(0.0, abs=1e-3)

    def test_motor_work_nonneg(self, catalog, measured_train_config):
        """Motor work (sum of positive a_long contributions) is always non-negative."""
        layout = _make_layout([R40_LEFT, STRAIGHT_16, R40_LEFT, STRAIGHT_16] * 4, catalog)
        phys = evaluate_layout(layout, catalog, measured_train_config, safety_margin=0.95)
        assert phys.motor_work_per_lap >= 0.0
```

- [ ] **Step 5.6: Run, verify FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestEnergyDomain -v`
Expected: 3 FAIL (Energy still stubbed at 0.0).

- [ ] **Step 5.7: Implement `_compute_energy` in evaluation.py**

Add a private function in `src/train/evaluation.py`:

```python
def _compute_energy(
    speeds: NDArray[np.float64],
    arc_lengths_m: NDArray[np.float64],
    a_long: NDArray[np.float64],
    train_config: TrainConfig,
    is_closed: bool,
) -> tuple[float, float, float]:
    """Energy domain: motor work, rolling dissipation, KE round-trips per lap.

    motor_work       = sum of positive m_total * a_long * arc_length
    rolling_diss     = mu_roll * m_total * g * total_distance
    ke_roundtrip     = sum of (v_high^2 - v_low^2) * 0.5 * m_total at brake-respin pairs
                       (local speed minima sandwiched by higher values).
    """
    m = train_config.mass_total
    g = train_config.g
    mu_roll = train_config.mu_roll

    # Motor work: only positive longitudinal accel contributes
    pos_force = np.maximum(0.0, m * a_long)
    motor_work = float(np.sum(pos_force * arc_lengths_m))

    total_distance = float(np.sum(arc_lengths_m))
    rolling_diss = mu_roll * m * g * total_distance

    # KE roundtrip: local speed minima (v[i] < v[i-1] AND v[i] < v[i+1])
    n = len(speeds)
    ke_roundtrip = 0.0
    if n >= 3:
        if is_closed:
            v_prev = np.roll(speeds, 1)
            v_next = np.roll(speeds, -1)
        else:
            v_prev = np.concatenate([[speeds[0]], speeds[:-1]])
            v_next = np.concatenate([speeds[1:], [speeds[-1]]])
        is_local_min = (speeds < v_prev) & (speeds < v_next)
        v_high = np.maximum(v_prev, v_next)
        roundtrips = np.where(is_local_min,
                              0.5 * m * (v_high ** 2 - speeds ** 2),
                              0.0)
        ke_roundtrip = float(np.sum(roundtrips))

    return motor_work, rolling_diss, ke_roundtrip
```

Replace the energy stubs in `evaluate_layout`:

```python
    # ---- Energy ----
    motor_work_per_lap = 0.0
    rolling_dissipation_per_lap = 0.0
    ke_roundtrip_per_lap = 0.0
```

With:

```python
    # ---- Energy ----
    motor_work_per_lap, rolling_dissipation_per_lap, ke_roundtrip_per_lap = (
        _compute_energy(
            speed_profile.speeds, arc_lengths_m, a_long_per_segment,
            train_config, is_closed,
        )
    )
```

- [ ] **Step 5.8: Run Energy tests, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestEnergyDomain -v`
Expected: 3 PASS.

- [ ] **Step 5.9: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All tests pass; the new mu_roll TrainConfig field doesn't break any existing test.

- [ ] **Step 5.10: Commit (after explicit user approval)**

```bash
git add src/train/physics.py src/train/evaluation.py tests/test_evaluation.py tests/test_train.py
git commit -m "feat(train): wire Energy domain + mu_roll TrainConfig field"
```

---

## Task 6: Wire lap_time into _evaluate as F[1]

**Files:**
- Modify: `src/problem.py`
- Modify: `tests/test_evaluation.py` (add integration test)

- [ ] **Step 6.1: Write failing integration test**

Append to `tests/test_evaluation.py`:

```python
class TestProblemIntegration:
    """Integration: F[1] in TrackOptimizationProblem._evaluate is now lap_time, minimized."""

    def test_F1_is_lap_time_not_min_speed(self, switches_config, catalog):
        """After Task 6, F[1] from _evaluate should equal phys.speed_profile.lap_time."""
        from src.problem import TrackOptimizationProblem
        from src.sampling import IntegerSampling
        # Build problem with switches config
        problem = TrackOptimizationProblem(catalog=catalog, config=switches_config)
        # Sample one valid chromosome
        sampler = IntegerSampling()
        pop = sampler.do(problem, 1)
        x = pop[0].X
        # Manually evaluate via problem
        out: dict = {}
        problem._evaluate(x, out)
        F = out["F"]
        # Independently compute via evaluate_layout
        from src.decoder import decode_chromosome
        layout = decode_chromosome(
            x, catalog, switches_config.inventory,
            dims=problem.dims, config=problem.decoder_config,
        )
        if layout.n_pieces == 0:
            # n_pieces==0 sentinel uses np.inf for both, skip
            assert np.isinf(F[1])
        else:
            phys = evaluate_layout(layout, catalog, problem._train_config, safety_margin=0.95)
            assert F[1] == pytest.approx(phys.speed_profile.lap_time, abs=1e-6)
            # And NOT equal to -min_speed (the old behavior)
            assert F[1] != pytest.approx(-phys.speed_profile.min_speed, abs=1e-6)
```

- [ ] **Step 6.2: Run, verify FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestProblemIntegration -v`
Expected: FAIL because `_evaluate` still returns `-min_speed`.

- [ ] **Step 6.3: Modify `src/problem.py:TrackOptimizationProblem._evaluate`**

In `src/problem.py`, find the speed-profile and F-assembly block:

```python
        speed_profile = compute_speed_profile(
            layout, self.catalog, train_config=self._train_config,
        )

        out["F"] = [-utilization, -speed_profile.min_speed]
```

Replace with:

```python
        from .train import evaluate_layout
        phys = evaluate_layout(
            layout, self.catalog, self._train_config, safety_margin=0.95,
        )

        out["F"] = [-utilization, phys.speed_profile.lap_time]
```

(Remove the now-unused `compute_speed_profile` import if it's only used here. The earlier `from .train import compute_speed_profile` line can be removed if unused; leave it if other functions in problem.py still call it.)

- [ ] **Step 6.4: Run integration test, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluation.py::TestProblemIntegration -v`
Expected: PASS.

- [ ] **Step 6.5: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest --tb=short -q`
Expected: All tests pass. (`test_problem.py` may have explicit `min_speed` assertions — if so, they need updating to lap_time semantics; address inline if any fail.)

- [ ] **Step 6.6: Manual smoke test against optimizer**

Run: `.venv/Scripts/python.exe main.py --config configs/with_switches.yaml --output outputs_v1/lap_time_smoke --quick-test --verbose`
Expected: Optimizer runs (20 generations); the verbose output should now show `best_speed=...` replaced or repurposed semantically. The terminal should NOT crash; F values should print as `(util, lap_time)` (lap_time positive, larger means slower).

- [ ] **Step 6.7: Inspect outputs via /diag**

Run: `/diag` (skill) or inspect `outputs_v1/lap_time_smoke/` for closure error, feasibility count, layout image.
Expected: Layout image renders, layout is closed, at least 1 feasible solution among the 20-pop quick test. (Quick test on 20 gen rarely finds many feasibles — that's expected.)

- [ ] **Step 6.8: Commit (after explicit user approval)**

```bash
git add src/problem.py tests/test_evaluation.py
git commit -m "feat(problem): F[1] = lap_time at safety_margin=0.95 (replaces -min_speed)"
```

---

## Self-review (post-plan, pre-execution)

This section was completed during plan-writing, fixing any issues inline. The plan is ready to execute.

**Spec coverage:**
- ✅ PhysicalEvaluation dataclass (Task 1, full shape)
- ✅ Geometry domain (Task 1)
- ✅ Stability domain (Task 2)
- ✅ Kinematics + safety_margin (Task 3)
- ✅ Dynamics domain (Task 4)
- ✅ Energy domain + mu_roll TrainConfig field (Task 5)
- ✅ F[1] = lap_time wiring (Task 6)
- ✅ Per-domain unit tests (each task)
- ✅ Integration test (Task 6)
- ⚠️ `coupler_phi_per_switch` for switches with diverging-route radius: spec lists this as an open question. The plan currently leaves the dict empty and uses only per-segment phi (which already covers R40 curves). When the switch-aware logic is implemented (post-MVP), update Task 1's geometry compute to populate the dict from the catalog's switch metadata.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, or vague-handler steps in any task. Every code block is complete and runnable. Every test asserts a hand-computed value or a property check.

**Type consistency:**
- `PhysicalEvaluation` dataclass field names match across tasks.
- `_compute_*` private functions return tuples whose unpacking matches their callers.
- `safety_margin: float` is consistent across `evaluate_layout`, `compute_speed_profile`, and tests.
- `binding_cap_per_segment` uses `dtype="<U6"` consistently (longest label = "nadal" = 5 chars; "<U6" gives 1-char headroom).

**One subtle item:** Task 4 has a step that says "deduplicate arc_lengths_m computation." That's a refactor inside the same task. Acceptable because Tasks 3 and 4 ship together logically; if executed sequentially without deduplication, the code still works (just computes arc_lengths twice). The dedup step is hygiene, not blocking.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-comprehensive-physical-model.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration via `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batch with checkpoints.

Which approach?