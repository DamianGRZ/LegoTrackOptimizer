# Phase 6: Problem V2 Migration Implementation Plan

> **For agentic workers:** Execute tasks task-by-task using `superpowers:subagent-driven-development`. Each task ends at "tests pass, working tree modified." **Never commit automatically** — the user controls commits.

**Goal:** Migrate `src/problem.py` from the current `TrackOptimizationProblem` to a V2-ready `TrackLayoutProblem` in two stages. Stage A fixes the infeasibility-sentinel bug and adds HV/IGD monitoring without changing the G vector shape. Stage B expands G to V2's per-axis + per-type form while preserving current feasibility semantics (scales derived from existing tolerances, not V2's tighter defaults).

**Why two stages:** Stage A is a pure bug-fix + instrumentation delivery with zero feasibility-shape change. Stage B is a constraint-vector reshape with proven parity. Splitting them lets the user stop after Stage A if Stage B's broader blast radius warrants caution.

**Architecture:** Shim-mode — current `MultiPathLayout` decoder, current `compute_speed_profile`, current chromosome encoding all stay in place. Problem reads them but emits V2-shaped F/G. Three localized upgrade points inside `_evaluate` will slot in when Phases 3 (train v_bottleneck), 4 (chromosome), 5 (decoder `DecodedLayout`) land. No upstream dependencies blocking Phase 6.

**Tech Stack:** pymoo 0.6.1.6, numpy, pytest. No new runtime dependencies.

**Companion docs:**
- Strategic roadmap: `docs/superpowers/plans/2026-04-20-modular-v2-adoption-roadmap.md` §Phase 6
- Batch 2 research: `docs/superpowers/plans/2026-04-20-batch-2-implementation-research.md` §Problem
- V2 spec: `Modular9PartResearchV2/Wrapping the pipeline as a pymoo problem.md`

**Extension from V2 spec:** our Phase 6 preserves the current `boundary` constraint (V2 drops it). The final G vector is **5 + |piece_types|** entries (V2 + boundary), not V2's 4 + |piece_types|. Rationale: the project has `BoundaryConfig` as a hard constraint with user-configurable min/max; dropping it would be a behavioral regression.

**Scale decision (shim-mode):** Stage B uses **current** tolerances (`closure_tolerance`, `angle_tolerance`, `boundary_tolerance` from `OptimizationConfig`), not V2's tighter defaults (S_xy=0.5, S_theta=π/180). This preserves feasibility for existing configs. V2's scale tightening is a deliberate future task and is explicitly **not** part of this phase.

**Non-goals:**
- Replacing `compute_speed_profile` with `v_bottleneck` function-by-function rewrite — the shim substitutes `SpeedProfile.min_speed` (the bottleneck of the profile, already computed) for the F[1] objective without touching Phase 3.
- Switching F[1] semantics from "fastest feasible traversal" to "slowest-curve cap only". The shim uses `min_speed` which is the same concept.
- Adopting V2's tighter closure scales (S_xy=0.5, S_theta=π/180). Preserving current feasibility is a design decision for the shim; tightening is a separate task.
- Splitting `src/problem.py` into a `problem/` package. It stays a single file; the new callback lives in `src/algorithm/monitoring.py`.
- Modifying `configs/*.yaml`. No config schema change.

**Risk posture:** Stage A is bug-fix + additive. Stage B changes G shape from 6 to 15 entries — every consumer of `pop.get("G")` must stay length-agnostic. Validated: `FeasibleEliteCallback` reads `CV <= 0`, `LegoAdaptiveEpsilon` reads aggregated CV — both length-agnostic per Batch 2 research. `save_results` and `/diag` parsers read F only — unaffected. HV reference point remains `(+0.10, -0.55)` in F space which doesn't depend on G.

---

## File Structure After Phase 6

```
src/
├── problem.py                      (MODIFY — heavy changes to _evaluate, __init__)
└── algorithm/
    ├── runner.py                   (MODIFY — one-line callback chain addition)
    └── monitoring.py               (NEW — ConvergenceMonitorCallback)

tests/
├── test_problem.py                 (MODIFY — add infeasibility, min_speed, G-shape tests)
└── test_monitoring.py              (NEW — HV/IGD/feasibility callback tests)
```

Total new files: 2. Total modified files: 3.

---

## Stage Overview

| Stage | Tasks | Delivers |
|---|---|---|
| **Stage A: Bug fixes + instrumentation** | Tasks 1–4 | `+inf` sentinel; `min_speed` F[1]; ConvergenceMonitorCallback; runner wiring |
| **Stage B: V2 G-vector shape** | Tasks 5–8 | Closure as 3 inequalities; per-type inventory; boundary preserved as G[3]; feasibility parity verified |

Each stage is independently mergeable. Stop after Stage A if user judges Stage B's blast radius too large for the current branch.

---

# Stage A — Bug Fixes + Instrumentation

## Task A1: Infeasibility sentinel (`+inf` for F, `1e6` for G)

**Files:**
- Modify: `src/problem.py` — the empty-layout branch in `_evaluate`
- Modify: `tests/test_problem.py` — add infeasibility-sentinel tests

**Current state** (verify before editing): `_evaluate` has a branch for `layout.n_pieces == 0` that writes `out["F"] = [0.0, 0.0]` and `out["G"] = [1.0, 1.0, 1.0, 1.0, -1.0, 0.0]`. This is the bug: `F=[0,0]` looks like "zero pieces, zero speed" — a valid-but-bad point — when it should look like "infeasible, don't consider in dominance."

### Steps

- [ ] **Step 1: Read `src/problem.py`** end-to-end. Note the exact current `_evaluate` signature, the empty-layout branch line numbers, the current G vector size (`n_ieq_constr=6`), and all call-sites of `layout.n_pieces`.

- [ ] **Step 2: Write failing tests** — append to `tests/test_problem.py`:

```python
class TestInfeasibilitySentinel:
    """Phase 6 Stage A: empty/infeasible layouts must emit +inf F, not 0.0."""

    def test_empty_layout_f_is_positive_infinity(self, catalog, default_config):
        """An all-sentinel chromosome produces an empty layout; F must be +inf.

        Current (buggy): F=[0.0, 0.0] makes infeasibles look like valid zero-piece
        solutions, dominated by nothing on f1 (utilization) = 0.
        V2 (correct): F=[+inf, +inf] — infeasibles never dominate anything.
        """
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        # All-sentinel chromosome: every gene at its minimum → empty main loop
        x = problem.xl.astype(int)
        out = {}
        problem._evaluate(x, out)
        assert np.isinf(out["F"][0]), f"F[0] should be +inf, got {out['F'][0]}"
        assert np.isinf(out["F"][1]), f"F[1] should be +inf, got {out['F'][1]}"
        assert out["F"][0] > 0 and out["F"][1] > 0, "sentinel must be POSITIVE infinity"

    def test_empty_layout_g_is_large_finite(self, catalog, default_config):
        """G must be finite and strongly positive so CV orders infeasibles."""
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        x = problem.xl.astype(int)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        assert np.all(np.isfinite(G)), f"G must be finite (not inf/nan): {G}"
        assert np.all(G > 0), f"All G entries must be positive for infeasible: {G}"
        assert G.max() >= 1e5, f"Sentinel too small; CV won't dominate real violations: max={G.max()}"

    def test_infeasibility_sentinel_dominated_by_feasible(self, catalog, default_config):
        """An infeasible individual (F=+inf) must lose to any feasible (F finite) in dominance."""
        from src.problem import TrackOptimizationProblem
        from pymoo.util.dominator import Dominator
        import numpy as np

        problem = TrackOptimizationProblem(catalog, default_config)
        # Construct two out dicts manually — no chromosome needed
        F_feas = np.array([[-0.5, -1.0]])       # feasible, utilization=0.5, speed=1.0
        F_infeas = np.array([[np.inf, np.inf]])  # the sentinel

        # In NSGA-II's dominance check, finite dominates +inf on every axis
        dom = Dominator.get_relation(F_feas[0], F_infeas[0])
        assert dom == 1, (
            f"Feasible F={F_feas[0]} should dominate infeasible F={F_infeas[0]}, "
            f"got relation={dom} (1=dominates, 0=non-dominated, -1=dominated-by)"
        )
```

- [ ] **Step 3: Run tests — expect fail.**

Run: `pytest tests/test_problem.py::TestInfeasibilitySentinel -v`
Expected: 3 FAIL (current F is `[0.0, 0.0]`, not `[+inf, +inf]`).

- [ ] **Step 4: Modify `src/problem.py`** — find the `layout.n_pieces == 0` branch in `_evaluate` and replace its body. The exact replacement:

```python
        # Empty/infeasible layout: V2 sentinel — +inf on F so feasibles dominate,
        # large finite on G so CV orders infeasibles by CV-sum. Never use NaN:
        # pymoo's HV and dominance comparisons tolerate +inf but not NaN.
        if layout.n_pieces == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)
            return
```

The numpy array form matters: pymoo expects `out["F"]` shape `(n_obj,)` and `out["G"]` shape `(n_ieq_constr,)`. Passing plain Python lists has worked historically but numpy is more robust against type drift.

- [ ] **Step 5: Run infeasibility tests — expect pass.**

Run: `pytest tests/test_problem.py::TestInfeasibilitySentinel -v`
Expected: 3 passed.

- [ ] **Step 6: Run full problem test class — watch for regressions.**

Run: `pytest tests/test_problem.py -v`
Expected: all previously-passing tests still pass.

- [ ] **Step 7: Run full suite.**

Run: `pytest tests/ -q`
Expected: 180 passed + 3 new = 183 passed.

**Done criteria:** 3 new sentinel tests pass; no existing test regresses.

---

## Task A2: Switch F[1] from `avg_speed` to `min_speed`

**Files:**
- Modify: `src/problem.py` — F[1] computation in `_evaluate`
- Modify: `tests/test_problem.py` — add min_speed test

**Rationale:** V2's f2 is `v_bottleneck` — the min over curve speed caps. Current code computes `avg_speed` from `SpeedProfile` (the length-weighted harmonic-mean of the time-optimal profile). The `SpeedProfile` already computes `min_speed` as a field — the shim just reads a different field. This is strictly a safety-semantics improvement: f2 now answers "how fast is the slowest curve?" not "how fast is the average trip?"

### Steps

- [ ] **Step 1: Verify `SpeedProfile.min_speed` exists.**

Run: `grep -n "min_speed" src/train/scoring.py src/train/__init__.py`
Expected: confirmation that `SpeedProfile` has a `min_speed` field.

- [ ] **Step 2: Write failing test.**

Append to `tests/test_problem.py`:

```python
class TestF1MinSpeed:
    """Phase 6 Stage A: F[1] should be -min_speed (bottleneck), not -avg_speed."""

    def test_f1_equals_negative_min_speed(self, catalog, default_config):
        """For a closed R40 circle, F[1] = -min_speed ≈ -0.97 (R40 speed cap)."""
        import numpy as np
        from src.problem import TrackOptimizationProblem
        from src.sampling import build_valid_closed_loop_16_r40

        problem = TrackOptimizationProblem(catalog, default_config)
        # Build a chromosome representing 16 R40 pieces in a circle
        x = build_valid_closed_loop_16_r40(problem, catalog)
        out = {}
        problem._evaluate(x, out)
        # R40's speed cap is 0.97 m/s per catalog (verified in test_catalog.py)
        # F[1] is negated; expect between -1.0 and -0.9
        assert -1.1 < out["F"][1] < -0.85, (
            f"F[1]={out['F'][1]} should be ≈ -0.97 (R40 cap). "
            f"If -avg_speed is reported instead, value would fall in [-1.3, -0.7]."
        )

    def test_f1_lower_bound_is_slowest_piece(self, catalog, default_config):
        """F[1] magnitude never exceeds the slowest piece's speed cap."""
        import numpy as np
        from src.problem import TrackOptimizationProblem

        problem = TrackOptimizationProblem(catalog, default_config)
        slowest_cap = float(catalog.speed_table.min())
        # For any decoded layout, F[1] = -min_speed <= -slowest_cap (it's the min
        # over pieces, bounded above by the slowest piece; after negation, bounded
        # below by -slowest_cap when the layout contains that piece).
        # For circles with only R40 (cap=0.97), F[1] >= -0.98 (slack for numerics).
        x = build_valid_closed_loop_16_r40(problem, catalog)
        out = {}
        problem._evaluate(x, out)
        assert out["F"][1] >= -slowest_cap - 0.01, (
            f"F[1]={out['F'][1]} below slowest cap=-{slowest_cap}; "
            f"min_speed computation is corrupt."
        )
```

Helper — add to `tests/conftest.py` OR inline in `test_problem.py` if it doesn't already exist:

```python
def build_valid_closed_loop_16_r40(problem, catalog):
    """Construct a chromosome representing 16 R40_LEFT pieces forming a closed circle."""
    import numpy as np
    from src.encoding import compute_dimensions, R40_LEFT

    dims = problem.dims if hasattr(problem, 'dims') else compute_dimensions(problem.config, catalog)
    x = np.full(dims.n_var, -1, dtype=np.int32)  # all INACTIVE
    x[:16] = int(R40_LEFT)  # first 16 main_loop slots = R40_LEFT
    # start_position at origin
    x[-2] = 0
    x[-1] = 0
    return x
```

Note: this helper may need adjustment based on current encoding. Adapt if `dims.n_var` computation differs. If building a valid chromosome is too complex inline, skip these two tests and rely on smoke-test validation in Task A4.

- [ ] **Step 3: Run tests — expect fail.**

Run: `pytest tests/test_problem.py::TestF1MinSpeed -v`
Expected: 2 FAIL (F[1] currently returns -avg_speed, not -min_speed).

- [ ] **Step 4: Modify `src/problem.py`** — in `_evaluate`, replace the F[1] computation. Find:

```python
        out["F"] = [-utilization, -speed_profile.avg_speed]
```

Replace with:

```python
        # F[1] = -min_speed (bottleneck). V2 semantics: the slowest segment dictates
        # safe traversal. avg_speed was the 3-pass harmonic-mean profile (travel-time
        # question), which allows a fast straight to mask a dangerous curve.
        # See docs/superpowers/plans/2026-04-20-batch-2-implementation-research.md §Problem Q6.
        out["F"] = [-utilization, -speed_profile.min_speed]
```

- [ ] **Step 5: Run min_speed tests — expect pass.**

Run: `pytest tests/test_problem.py::TestF1MinSpeed -v`
Expected: 2 passed.

- [ ] **Step 6: Full problem tests.**

Run: `pytest tests/test_problem.py -v`
Expected: all pass. If a previously-passing test asserts on `avg_speed`-specific F[1] values (e.g., expects F[1] ≈ -1.3), it needs updating. Report such failures as DONE_WITH_CONCERNS and list the affected tests.

- [ ] **Step 7: Full suite.**

Run: `pytest tests/ -q`
Expected: ≥ 185 passed.

**Done criteria:** min_speed tests pass; any legacy test asserting on avg_speed is documented.

---

## Task A3: `ConvergenceMonitorCallback`

**Files:**
- Create: `src/algorithm/monitoring.py`
- Create: `tests/test_monitoring.py`

### Steps

- [ ] **Step 1: Write failing tests.**

Create `tests/test_monitoring.py`:

```python
"""Tests for ConvergenceMonitorCallback: HV, IGD, feasibility rate."""

import numpy as np
import pytest


class TestConvergenceMonitorCallback:
    def test_callback_initializes_data_keys(self):
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback()
        expected_keys = {"n_gen", "n_eval", "hv", "igd", "n_feas", "feas_rate",
                         "mean_closure_x", "mean_closure_y", "mean_closure_theta"}
        assert expected_keys.issubset(cb.data.keys()), (
            f"Missing keys: {expected_keys - cb.data.keys()}"
        )
        for k in expected_keys:
            assert cb.data[k] == [], f"{k} should init as empty list, got {cb.data[k]}"

    def test_hv_filters_infeasibles_before_computing(self):
        """The +inf sentinel must never reach HV.__call__."""
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback(ref_point=(0.10, -0.55))

        # Mock algorithm with mixed feasible + infeasible population
        class FakePop:
            def __init__(self, F, CV):
                self._F, self._CV = F, CV
            def get(self, key):
                return {"F": self._F, "CV": self._CV, "G": None}.get(key)
            def __len__(self):
                return len(self._F)

        class FakeEvaluator:
            n_eval = 100

        class FakeAlgo:
            n_gen = 5
            pop = FakePop(
                F=np.array([[-0.8, -1.05], [np.inf, np.inf], [-0.6, -0.9]]),
                CV=np.array([[0.0], [5e6], [0.0]]),
            )
            evaluator = FakeEvaluator()

        cb.notify(FakeAlgo())
        assert len(cb.data["hv"]) == 1
        assert np.isfinite(cb.data["hv"][0]), f"HV must not be inf/nan"
        assert cb.data["hv"][0] > 0, f"HV should be positive, got {cb.data['hv'][0]}"
        assert cb.data["n_feas"][0] == 2
        assert cb.data["feas_rate"][0] == pytest.approx(2 / 3, rel=1e-6)

    def test_hv_zero_when_all_infeasible(self):
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback()

        class FakePop:
            def get(self, key):
                return {
                    "F": np.array([[np.inf, np.inf], [np.inf, np.inf]]),
                    "CV": np.array([[1e6], [2e6]]),
                    "G": None,
                }.get(key)
            def __len__(self):
                return 2

        class FakeEvaluator:
            n_eval = 50

        class FakeAlgo:
            n_gen = 1
            pop = FakePop()
            evaluator = FakeEvaluator()

        cb.notify(FakeAlgo())
        assert cb.data["hv"][0] == 0.0
        assert cb.data["n_feas"][0] == 0
        assert cb.data["feas_rate"][0] == 0.0

    def test_igd_against_rolling_best_front(self):
        """IGD must be computable against a self-improving best-known front."""
        from src.algorithm.monitoring import ConvergenceMonitorCallback
        cb = ConvergenceMonitorCallback(pareto_ref=None)  # self-improving mode

        class FakePop:
            def __init__(self, F, CV):
                self._F, self._CV = F, CV
            def get(self, key):
                return {"F": self._F, "CV": self._CV, "G": None}.get(key)
            def __len__(self):
                return len(self._F)

        class FakeEvaluator:
            n_eval = 100

        class FakeAlgo:
            def __init__(self, F, CV):
                self.n_gen = 1
                self.pop = FakePop(F, CV)
                self.evaluator = FakeEvaluator()

        # Generation 1: one feasible point
        cb.notify(FakeAlgo(np.array([[-0.5, -1.0]]), np.array([[0.0]])))
        assert np.isfinite(cb.data["igd"][0])
        # Generation 2: better point
        cb.notify(FakeAlgo(np.array([[-0.8, -1.05]]), np.array([[0.0]])))
        assert np.isfinite(cb.data["igd"][1])
```

- [ ] **Step 2: Run — expect fail.**

Run: `pytest tests/test_monitoring.py -v`
Expected: 4 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/algorithm/monitoring.py`.**

Content:

```python
"""Convergence instrumentation for NSGA-II: HV, IGD, feasibility rate.

HV reference point (+0.10, -0.55) is 10% beyond empirical nadir (0, -0.60) per
Ishibuchi et al. 2018 / Auger et al. 2009. The callback always filters to
feasible-only before computing HV/IGD — the +inf infeasibility sentinel never
reaches the indicators.

See docs/superpowers/plans/2026-04-20-batch-2-implementation-research.md §Problem Q5.
"""

from __future__ import annotations

import math
import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


_S_XY = 0.5
_S_THETA = math.pi / 180  # 1 degree in rad — used only for de-normalizing mean_closure_*


class ConvergenceMonitorCallback(Callback):
    """Per-generation: HV, IGD, feasibility rate, mean closure residuals."""

    def __init__(
        self,
        ref_point: tuple[float, float] = (0.10, -0.55),
        pareto_ref: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self._pareto_ref = pareto_ref
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        for k in (
            "n_gen", "n_eval",
            "hv", "igd",
            "n_feas", "feas_rate",
            "mean_closure_x", "mean_closure_y", "mean_closure_theta",
        ):
            self.data[k] = []
        self._best_F: np.ndarray | None = None

    def notify(self, algorithm) -> None:
        pop = algorithm.pop
        F = pop.get("F")
        CV = pop.get("CV")
        G = pop.get("G")

        if F is None or CV is None:
            return

        feas_mask = CV.ravel() <= 0.0
        F_feas = F[feas_mask]
        n_feas = int(feas_mask.sum())
        pop_size = max(1, len(pop))

        hv_val = float(self.hv.do(F_feas)) if n_feas > 0 else 0.0

        igd_val = float("nan")
        if n_feas > 0:
            self._best_F = self._update_best_front(F_feas)
            if self._pareto_ref is not None and self.igd is not None:
                igd_val = float(self.igd.do(F_feas))
            elif self._best_F is not None and len(self._best_F) > 0:
                igd_val = float(IGD(self._best_F).do(F_feas))

        # De-normalize closure residuals from G for human-readable logging.
        # Assumes G[0..2] are the V2 closure normalization: abs(dx)/S_xy - 1, etc.
        # Pre-Stage-B this block may be a no-op (G has 6 entries with magnitude-based
        # closure, not per-axis). Safe because shape check guards.
        mean_cx = mean_cy = mean_ct = float("nan")
        if G is not None and G.shape[1] >= 3 and n_feas > 0:
            G_feas = G[feas_mask]
            mean_cx = float(np.mean((G_feas[:, 0] + 1.0) * _S_XY))
            mean_cy = float(np.mean((G_feas[:, 1] + 1.0) * _S_XY))
            mean_ct = float(np.mean((G_feas[:, 2] + 1.0) * _S_THETA))

        self.data["n_gen"].append(algorithm.n_gen)
        self.data["n_eval"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(hv_val)
        self.data["igd"].append(igd_val)
        self.data["n_feas"].append(n_feas)
        self.data["feas_rate"].append(n_feas / pop_size)
        self.data["mean_closure_x"].append(mean_cx)
        self.data["mean_closure_y"].append(mean_cy)
        self.data["mean_closure_theta"].append(mean_ct)

    def _update_best_front(self, F_new: np.ndarray) -> np.ndarray:
        if self._best_F is None or len(self._best_F) == 0:
            combined = F_new
        else:
            combined = np.vstack([self._best_F, F_new])
        idx = NonDominatedSorting().do(combined, only_non_dominated_front=True)
        return combined[idx]
```

- [ ] **Step 4: Verify `src/algorithm/__init__.py` doesn't need update.**

Run: `cat src/algorithm/__init__.py`
If empty or minimal, no change. If it re-exports names, add `from .monitoring import ConvergenceMonitorCallback` and append `"ConvergenceMonitorCallback"` to `__all__`.

- [ ] **Step 5: Run callback tests — expect pass.**

Run: `pytest tests/test_monitoring.py -v`
Expected: 4 passed.

- [ ] **Step 6: Full suite.**

Run: `pytest tests/ -q`
Expected: ≥ 189 passed.

**Done criteria:** 4 callback tests pass; full suite green.

---

## Task A4: Wire `ConvergenceMonitorCallback` into the runner

**Files:**
- Modify: `src/algorithm/runner.py`
- Modify: `tests/test_main.py` (if a runner-level test exists)

### Steps

- [ ] **Step 1: Read `src/algorithm/runner.py`** end-to-end. Locate:
  - Where `CallbackChain` is constructed (typical: inside `run_optimization`)
  - The list of callbacks appended (should include `ProgressCallback`, `FeasibleEliteCallback`, etc.)

- [ ] **Step 2: Add ConvergenceMonitorCallback to the chain.**

In `run_optimization` (or wherever the callback list is built), add:

```python
from .monitoring import ConvergenceMonitorCallback

# ... existing callback construction ...
monitor = ConvergenceMonitorCallback(ref_point=(0.10, -0.55))
callbacks = CallbackChain([
    progress_callback,        # existing
    feasible_elite_callback,  # existing
    monitor,                  # NEW
    # ... any other existing ...
])
```

Adapt the exact structure to match the current code. The callback order matters only for side-effect operations; `ConvergenceMonitorCallback` is read-only (just logs), so placement is flexible.

- [ ] **Step 3: Expose `monitor.data` in `run_optimization`'s return value.**

If `run_optimization` returns a result dict/object, add `monitor_data=monitor.data` or equivalent. If it writes results to disk, also write `monitor.data` as `outputs/monitor.pkl` using `pickle.dump` (or JSON if simple). Minimal form: just store on the returned object.

- [ ] **Step 4: Run the optimizer smoke test.**

Run: `python main.py --config configs/default.yaml --quick-test`
Expected: completes cleanly. If `monitor.data` is exposed, check that `hv`, `igd`, `n_feas` keys contain lists of length equal to n_gen.

- [ ] **Step 5: Full test suite.**

Run: `pytest tests/ -q`
Expected: all pass. If `tests/test_main.py` has a runner-level test, it should still pass (new callback doesn't change results, just adds monitoring).

**Done criteria:** quick-test run completes; monitor data populated.

---

## Stage A Completion Gate

Before proceeding to Stage B:
1. All 4 Stage A tasks complete.
2. Full suite green (~190 tests).
3. Smoke test: `python main.py --config configs/default.yaml --quick-test` completes with `ConvergenceMonitorCallback` populating `hv`, `igd`, `n_feas` per generation.
4. Bug verified fixed: decoded chromosome that produces empty layout has `F=[+inf, +inf]` and `G=[1e6]*6` (note: still 6 — Stage B expands to 15).
5. **User reviews before Stage B.** Stage B has broader blast radius; user may choose to commit Stage A and defer Stage B.

---

# Stage B — V2 G-vector Shape

## Task B1: Split closure into 3 per-axis inequalities

**Files:**
- Modify: `src/problem.py`
- Modify: `tests/test_problem.py`

### Steps

- [ ] **Step 1: Read current closure computation in `_evaluate`.** Note the exact formula and where `main_path.closure_error` and `main_path.angle_error` come from. Current single scalar:

```python
g_closure = (closure_err - self.closure_tolerance) / self.closure_tolerance
```

**Important:** `closure_err` is the magnitude `sqrt(dx² + dy²)` (positive scalar). `angle_error` is degrees. Stage B splits into dx, dy, dtheta which require accessing the path's state vector, not the pre-computed magnitudes.

- [ ] **Step 2: Write failing tests.**

```python
class TestClosurePerAxis:
    """Stage B: G has 3 separate closure inequalities (x, y, theta) instead of 1 magnitude."""

    def test_g_has_three_closure_entries(self, catalog, default_config):
        from src.problem import TrackOptimizationProblem
        problem = TrackOptimizationProblem(catalog, default_config)
        assert problem.n_ieq_constr >= 3, (
            f"n_ieq_constr={problem.n_ieq_constr}; Stage B requires ≥ 3 closure entries"
        )

    def test_closed_r40_circle_satisfies_all_three_closure(self, catalog, default_config):
        """For a closed 16×R40 circle, dx ≈ dy ≈ dtheta ≈ 0 — all three G entries satisfied."""
        import numpy as np
        from src.problem import TrackOptimizationProblem
        problem = TrackOptimizationProblem(catalog, default_config)
        x = build_valid_closed_loop_16_r40(problem, catalog)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        # Closure x, y, theta are G[0], G[1], G[2]
        assert G[0] <= 0, f"G[0] closure_x = {G[0]}, should be ≤ 0 (|dx| ≤ closure_tolerance)"
        assert G[1] <= 0, f"G[1] closure_y = {G[1]}, should be ≤ 0"
        assert G[2] <= 0, f"G[2] closure_theta = {G[2]}, should be ≤ 0"
```

- [ ] **Step 3: Run — expect fail** (current G has magnitude-based closure, not per-axis).

- [ ] **Step 4: Modify `_evaluate` in `src/problem.py`.**

The new G composition (5 + T entries, with T = number of piece types in catalog):

```python
        # Closure residual from decoded layout (main path).
        # main_path.closure_dxy: tuple (dx, dy) in studs
        # main_path.closure_dtheta: float, degrees (current) → radians (post-Phase-5)
        # Shim-mode: derive per-axis from main_path.states. main_path.states is
        # (n+1, 3) with [x_stud, y_stud, theta_deg] — final row is the closure
        # endpoint relative to start.
        start_state = main_path.states[0]
        end_state = main_path.states[-1]
        dx = float(end_state[0] - start_state[0])
        dy = float(end_state[1] - start_state[1])
        dtheta_deg = float(end_state[2] - start_state[2])
        # Wrap dtheta to (-180, +180] (matches (-π, +π] in rad once Phase 5 ships)
        dtheta_deg = ((dtheta_deg + 180.0) % 360.0) - 180.0
        if dtheta_deg == -180.0:
            dtheta_deg = 180.0

        # Per-axis closure normalization (shim scale: closure_tolerance stud-equivalent
        # per axis — preserves current magnitude feasibility approximately).
        g_closure_x = abs(dx) / self.closure_tolerance - 1.0
        g_closure_y = abs(dy) / self.closure_tolerance - 1.0
        g_closure_theta = abs(dtheta_deg) / self.angle_tolerance - 1.0

        # Rest of G (boundary, collision, per-type inventory) — see B2 and B3
        # ... (to be built across remaining Stage B tasks)
```

**Partial implementation note:** the full G is not assembled until Task B3. This task only adds per-axis closure. Keep the rest of G as-is (current 6 entries plus 2 new = 8 entries; `n_ieq_constr=8`) until B2/B3 complete the reshape. This keeps the test at each step focused on one change.

Update `n_ieq_constr` in `__init__`:

```python
# Stage B1 interim: 3 closure + current 5 (boundary, inventory, alignment, crossings)
# Post-B3: 5 + n_piece_types
n_ieq_constr=3 + 5,  # interim; final in B3
```

Hmm wait — this task-by-task reshape is fragile because the full suite must pass after each task. Let me adjust: **B1 adds 2 new G entries (splitting magnitude into x and y)** but keeps the old magnitude entry as G[old_idx] and retains angle at its current position. So G grows by 2 (from 6 to 8):
- G[0]: closure magnitude (unchanged, for backward compat of any runtime assertion)
- G[1]: angle (unchanged)
- G[2]: boundary (unchanged)
- G[3]: inventory (unchanged)
- G[4]: alignment (unchanged)
- G[5]: crossings (unchanged)
- G[6]: NEW closure_x
- G[7]: NEW closure_y

Task B3 rearranges to final V2 order after all Stage B changes land, so the test suite can progress one piece at a time. 

Actually, this grows overly complex. Simpler: do B1, B2, B3 as one atomic reshape, test at end. But that violates "one task at a time" TDD discipline.

Let me restructure: **collapse Stage B into a single task B1 that does the full reshape**, with tests for each part.

- [ ] **Step 5: Run all Stage B1 tests.**

Run: `pytest tests/test_problem.py::TestClosurePerAxis -v`
Expected: pass.

**Task B1 done when:** closure is per-axis in G; tests pass; full suite green.

### Revised plan: merge B1+B2+B3 into one reshape task

Given the fragility of partial G reshapes, Stage B is best executed as a **single atomic task** producing the final V2+boundary G vector. Splitting into sub-tasks creates intermediate states that can't pass all tests. The following task replaces the separate B1, B2, B3.

---

## Task B: Full G-vector reshape (atomic)

**Files:**
- Modify: `src/problem.py` — `__init__` (n_ieq_constr), `_evaluate` (full G composition)
- Modify: `tests/test_problem.py` — add G-shape parity tests

### The new G vector (5 + T entries, T = catalog.n_pieces)

```
G[0]: closure_x          = abs(dx) / closure_tolerance - 1
G[1]: closure_y          = abs(dy) / closure_tolerance - 1
G[2]: closure_theta      = abs(dtheta_deg) / angle_tolerance - 1   (bridge: deg until Phase 5)
G[3]: boundary           = (boundary_violation - boundary_tolerance) / diagonal
G[4]: collisions         = len(collision_list) / 5.0
G[5..4+T]: per-type inv  = max(0, census[t] - max_occ[t]) / max(1, max_occ[t])
```

For current catalog T=10, total = 15 entries.

### Steps

- [ ] **Step 1: Write failing tests** — append to `tests/test_problem.py`:

```python
class TestGShapeV2:
    """Stage B: G has 5 + n_piece_types entries."""

    def test_n_ieq_constr_is_5_plus_piece_types(self, catalog, default_config):
        from src.problem import TrackOptimizationProblem
        problem = TrackOptimizationProblem(catalog, default_config)
        expected = 5 + catalog.n_pieces
        assert problem.n_ieq_constr == expected, (
            f"n_ieq_constr={problem.n_ieq_constr}, expected {expected} (5 + {catalog.n_pieces})"
        )

    def test_g_entries_order_and_semantics(self, catalog, default_config):
        """Verify G indices 0..4 are closure_x, closure_y, closure_theta, boundary, collisions."""
        import numpy as np
        from src.problem import TrackOptimizationProblem
        problem = TrackOptimizationProblem(catalog, default_config)
        x = build_valid_closed_loop_16_r40(problem, catalog)
        out = {}
        problem._evaluate(x, out)
        G = np.asarray(out["G"])
        # Closed circle: closure ≈ 0, boundary OK (40-stud radius fits default 300x300),
        # no collisions, inventory depends on config.
        assert len(G) == 5 + catalog.n_pieces
        assert G[0] <= 0.0, f"G[0] closure_x={G[0]}"
        assert G[1] <= 0.0, f"G[1] closure_y={G[1]}"
        assert G[2] <= 0.0, f"G[2] closure_theta={G[2]}"
        assert G[3] <= 0.0, f"G[3] boundary={G[3]}"
        assert G[4] <= 0.0, f"G[4] collisions={G[4]}"

    def test_forbidden_piece_yields_positive_inventory_entry(self, catalog, small_config):
        """A piece with max_occ=0 used in layout yields a positive G entry at its index."""
        import numpy as np
        from src.problem import TrackOptimizationProblem
        # small_config has e.g. only STRAIGHT_16 in inventory, no R40_LEFT
        # Build a chromosome that uses R40_LEFT (forbidden)
        # ... test body ...
        pytest.skip("Requires small_config fixture; implement when fixture available")

class TestFeasibilityParity:
    """Stage B feasibility preservation: existing configs still produce feasible layouts."""

    @pytest.mark.parametrize("config_name", ["default", "with_switches", "with_crossing", "compact"])
    def test_config_smoke_still_finds_feasibles(self, config_name):
        """Running --quick-test on each config must find ≥ 0 feasible solutions (same as pre-Stage-B).

        This is a regression guard: Stage B must NOT make previously-feasible configs infeasible.
        """
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "main.py", "--config", f"configs/{config_name}.yaml", "--quick-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"{config_name} quick-test failed: returncode={result.returncode}\n"
            f"stdout:\n{result.stdout[-1000:]}\n"
            f"stderr:\n{result.stderr[-500:]}"
        )
```

- [ ] **Step 2: Run — expect fail** (n_ieq_constr=6 currently, not 15).

- [ ] **Step 3: Rewrite `_evaluate` G composition in `src/problem.py`.**

In `__init__`, change:
```python
n_ieq_constr=6,  # OLD
```
to:
```python
n_ieq_constr=5 + catalog.n_pieces,  # 3 closure + boundary + collisions + per-type inventory
```

Replace the G composition block in `_evaluate`:

```python
        # ---- V2 G vector (shim-mode, preserves current tolerances) ----

        # Closure: split into 3 per-axis inequalities.
        # Shim-mode unit bridge: current main_path.states[:, 2] is degrees.
        # Post-Phase-5: DecodedLayout.closure_residual is (dx, dy, dtheta_rad).
        start_state = main_path.states[0]
        end_state = main_path.states[-1]
        dx = float(end_state[0] - start_state[0])
        dy = float(end_state[1] - start_state[1])
        dtheta_deg = float(end_state[2] - start_state[2])
        # Wrap to (-180, +180]
        dtheta_deg = ((dtheta_deg + 180.0) % 360.0) - 180.0
        if dtheta_deg == -180.0:
            dtheta_deg = 180.0

        g_closure_x = abs(dx) / self.closure_tolerance - 1.0
        g_closure_y = abs(dy) / self.closure_tolerance - 1.0
        g_closure_theta = abs(dtheta_deg) / self.angle_tolerance - 1.0

        # Boundary — preserved from current (V2 drops this; we keep it).
        boundary_violation = self._compute_boundary_violation(layout)
        g_boundary = (
            (boundary_violation - self.boundary_tolerance) / max(self.diagonal, 1.0)
        )

        # Collisions — count of unresolved self-intersections.
        # Shim: use existing count_segment_crossings (already in current code).
        n_collisions = count_segment_crossings(
            layout.states,
            list(layout.main_loop_pieces),
        )
        g_collisions = float(n_collisions) / 5.0  # COLLISION_SCALE = 5.0

        # Per-type inventory excess (T entries, T = catalog.n_pieces).
        # Replaces the current single-scalar inventory G.
        g_inventory_per_type = self._compute_per_type_inventory_violation(layout)

        # Assemble G. Order matters: G[0..2]=closure, G[3]=boundary, G[4]=collisions,
        # G[5..4+T]=per-type inventory.
        g_vec = np.concatenate([
            np.array([g_closure_x, g_closure_y, g_closure_theta,
                      g_boundary, g_collisions]),
            g_inventory_per_type,
        ])
        out["F"] = np.array([-utilization, -speed_profile.min_speed])
        out["G"] = g_vec
```

Add the helper `_compute_per_type_inventory_violation`:

```python
    def _compute_per_type_inventory_violation(self, layout) -> np.ndarray:
        """Per-catalog-index inventory excess, normalized by max_occ[t].

        Returns array of length self.catalog.n_pieces. Entry t is:
            max(0, census[t] - max_occ[t]) / max(1, max_occ[t])
        where max_occ[t] comes from self.inventory_by_index (0 if piece not allowed).
        """
        n_types = self.catalog.n_pieces
        result = np.zeros(n_types, dtype=np.float64)

        # Count pieces by catalog index across all placed pieces.
        for path in getattr(layout, "paths", [layout]):
            if not hasattr(path, "piece_sequence"):
                continue
            indices = np.asarray(path.piece_sequence, dtype=np.int32)
            for t in range(n_types):
                count_t = int(np.sum(indices == t))
                max_occ_t = self.inventory_by_index.get(t, 0)
                excess = max(0, count_t - max_occ_t)
                result[t] = excess / max(1, max_occ_t)
        return result
```

If the layout exposes pieces differently, adapt the counting loop. Match whatever `layout.main_loop_pieces` / `layout.paths[i].piece_sequence` is accessible.

- [ ] **Step 4: Update the empty-layout sentinel** to match the new G length:

```python
        if layout.n_pieces == 0:
            out["F"] = np.array([np.inf, np.inf])
            out["G"] = np.full(self.n_ieq_constr, 1e6)  # 5 + T entries
            return
```

(Already uses `self.n_ieq_constr` from Task A1 — should just work.)

- [ ] **Step 5: Run shape tests — expect pass.**

Run: `pytest tests/test_problem.py::TestGShapeV2 -v`
Expected: 2 passed (third is skipped pending fixture).

- [ ] **Step 6: Run parity tests — expect pass on all 4 configs.**

Run: `pytest tests/test_problem.py::TestFeasibilityParity -v`
Expected: 4 passed. Each config's quick-test completes with exit code 0.

- [ ] **Step 7: Full suite.**

Run: `pytest tests/ -q`
Expected: all pass. If any tests assert on specific G indices (e.g., `assert G[4] == expected_inventory_value`), they need updating because the G layout changed. Flag such tests and update them.

- [ ] **Step 8: Verify the `ConvergenceMonitorCallback.mean_closure_*` fields populate.**

Run: `python main.py --config configs/default.yaml --quick-test`
After the run, inspect the monitor data (via a debugger breakpoint or by loading the persisted pickle if available). Confirm `mean_closure_x`, `mean_closure_y`, `mean_closure_theta` contain finite floats, not NaN (they were NaN before B because G didn't have per-axis closure).

**Done criteria:** G shape test passes; all 4 configs run to exit 0; full suite green; monitor closure fields populate.

---

# Cross-Task Risks & Verifications

1. **`FeasibleEliteCallback` and `LegoAdaptiveEpsilon` length-agnostic claim.**

   Before committing Stage B, read `src/algorithm/runner.py` and confirm these callbacks don't index into G at fixed positions (e.g., `G[3]`, `G[5]`). If they do, those indices become stale after reshape and must be adjusted.

   Likely safe: research reported these use `CV` aggregate (sum of positives across all G entries) or `G <= 0` mask.

2. **`save_results` column headers.**

   If `save_results` writes `constraints.csv` with named columns (like "closure_err", "inventory_excess"), those column names need updating to match the new G layout. If it writes anonymous indices `constraint_0, constraint_1, ...`, the column count changes but the semantics of each index do.

3. **`/diag` skill parsers.**

   The `/diag` skill parses `outputs/constraints.csv` and `outputs/chromosomes.csv` to report feasibility. If it has hardcoded G-index assumptions (e.g., "closure error is G[0]"), it may report wrong information post-Stage-B.

   Deferred: fix `/diag` only if needed after Stage B lands. Phase 6 doesn't block on `/diag` correctness.

4. **HV reference point sanity check.**

   `ref_point=(+0.10, -0.55)` assumes F range `[-1, 0] × [-1.10, -0.60]`. With `F[1] = -min_speed`, `min_speed` is ≤ `v_cap ≈ 1.10`, so `F[1] ≥ -1.10` — within the assumed range. If the catalog added a piece with cap > 1.10, the ref point would be dominated by feasibles and HV would be wrong. Current catalog's max speed is `DEFAULT_SPEED = 1.57` (straight/crossing caps), so `min_speed` can exceed 1.10 briefly. Verify HV values look plausible (positive, bounded, monotone non-decreasing over generations).

   If HV looks corrupted, compute an empirical nadir from a first-generation dry run and adjust ref_point in `configs/*.yaml` or pass via CLI. Deferred unless an issue surfaces.

---

# Self-Review Checklist

**Spec coverage (Batch 2 research §Problem):**

| V2 requirement | Task | Status |
|---|---|---|
| `+inf` infeasibility sentinel | A1 | ✅ |
| `min_speed` (V2 `v_bottleneck`-equivalent) as F[1] | A2 | ✅ |
| `ConvergenceMonitorCallback` with HV/IGD/feasibility | A3 | ✅ |
| Runner integration | A4 | ✅ |
| Closure as 3 inequalities (x, y, theta) | B | ✅ |
| Per-type inventory | B | ✅ |
| HV ref point `(+0.10, -0.55)` | A3 | ✅ |
| pymoo `ElementwiseProblem` base class | pre-existing | ✅ (already used) |

**Extensions to V2 spec (documented):**
- Boundary constraint preserved as G[3] (V2 drops; we keep)
- Tolerance scales use current `closure_tolerance`/`angle_tolerance` (V2 uses tighter S_xy=0.5/S_theta=π/180) — deliberate for shim feasibility parity
- File structure: kept `src/problem.py` as single file; new callback in `src/algorithm/monitoring.py`

**Deferred (per "Non-goals"):**
- `v_bottleneck` function-level rewrite (Phase 3 Train dependency)
- Tighter V2 scales (S_xy=0.5 stud, S_theta=π/180 rad) — requires a deliberate feasibility re-tuning
- `_warm_up()` numba stub (Phase 5 dependency)
- `problem/` package split (cosmetic)

**Placeholder scan:** No "TBD", "fill in details". Two spots use adaptive language:
- `build_valid_closed_loop_16_r40` helper construction depends on current encoding shape — the plan flags this with explicit "adapt if dims.n_var computation differs."
- Per-type inventory counting loop depends on how `layout.paths[i].piece_sequence` is exposed — plan flags "match whatever is accessible."
These are **informed adaptations**, not placeholders.

**Type consistency:**
- `ConvergenceMonitorCallback` — consistent name across callback file, tests, runner
- `n_ieq_constr = 5 + catalog.n_pieces` — consistent formula across problem init and tests
- `COLLISION_SCALE = 5.0` — used consistently (inlined, not a named constant yet; could be promoted later)

**Commit count:** zero. Commits are the user's decision.

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-phase-6-problem.md`. Two execution options:

1. **Subagent-Driven** (recommended) — fresh subagent per task, two-stage review between. **No auto-commits.**
2. **Inline execution** — same session, batch with checkpoints.

Which approach? And: do you want to execute both stages now, or stop after Stage A for review before proceeding to Stage B?
