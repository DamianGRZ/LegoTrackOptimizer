# Train Physics Disconnect — Remediation Plan

> **Companion plan to [docs/PLAN.md](PLAN.md).** Where PLAN.md governs the multi-phase optimizer architecture, this document addresses a multi-layer integration bug discovered post-Phase-8: the measured locomotive physics in `configs/trains/measured_consist.yaml` are loaded by no optimization run. F[1] objective values are computed against hardcoded V2 catalog defaults (1.57 / 0.97 m/s), not the user's measured AFM SL+Cargo Train physics (1.26 / 0.886 m/s).

---

## Quick Index

**Current state**: Bug confirmed by 4-agent parallel audit. All Phases 0-8 implementation tested green, but the test suite never validated the loading path because no test asserts `problem.train_config.v_motor_max == measured_value`. Real-LEGO test is BLOCKED on Tier 1.

### Status

- [ ] **HARD BLOCKER**: do NOT run real-LEGO test until Tier 1 lands
- [ ] Tier 1.1 — Repoint all 8 configs from `trains/default.yaml` to `trains/measured_consist.yaml`
- [ ] Tier 1.2 — Wire `TrainConfig` from `OptimizationConfig.load_train_config()` into `PortPairProblem.__init__`
- [ ] Tier 1.3 — Replace `catalog.get_speed_for_route()` lookup in `problem.py` F[1] computation with `train_config.v_eff(radius_m)`
- [ ] Tier 1 verification — run full pytest suite (212 tests must pass) + 50-gen mini-opt sanity
- [ ] Tier 2.1 — Make `TrackCatalog.load()` accept `train_config` OR remove `speed_table` (architectural decision)
- [ ] Tier 2.2 — Deprecate `_V2_DEFAULT_PHYSICS`, `_V2_ROUTE_PHYSICS`, `DEFAULT_SPEED`
- [ ] Tier 2.3 — Add startup log line announcing loaded train physics
- [ ] Tier 2.4 — Add wiring smoke test (`tests/test_train_physics_wiring.py`)
- [ ] Tier 3 — `compute_speed_profile()` integration (DEFERRED until Phase 9 lap_time objective lands)
- [ ] Real-LEGO physical test — runs only after Tier 1 verification passes

### How to find a section

| Want | Section | Approx. line |
|---|---|---|
| The 4-agent audit findings | §2 — Findings | ~50 |
| The disconnect chain (visual) | §3 — Disconnect Chain | ~110 |
| Tier 1 fix specs (code-level) | §5 — Tier 1 | ~150 |
| Tier 2 cleanup specs | §6 — Tier 2 | ~250 |
| Verification commands | §7 — Verification | ~310 |
| Cross-references to PLAN.md | §8 — Cross-References | ~360 |
| Why this wasn't caught earlier | §9 — Root Cause | ~380 |

---

## §1 — Executive Summary

**The problem**: Phase 0 measurements (locomotive mass 0.493 kg, trailing 0.327 kg, motor cap 1.26 m/s, max acceleration 0.68 m/s², coupler offset 0.106 m) were measured, written to `configs/trains/measured_consist.yaml`, and then **never loaded by any optimization run**. F[1] = -min_speed is computed against catalog-static hardcoded values (`1.57` for through-routes, `0.97` for diverging) instead of the measured train's actual physics.

**Practical impact for real-LEGO testing**:

| Layout shape | F[1] reports | Physical reality | Discrepancy |
|---|---|---|---|
| R40-curve-rich (any closed cycle) | 0.886 m/s (slide-bound on R40) | 0.886 m/s | None — slide-bound is correct regardless of `v_motor_max` |
| Straight-rich (long straights, few curves) | 1.57 m/s (catalog DEFAULT_SPEED) | 1.26 m/s (measured motor cap) | **~24 % over-reported** |
| Pareto-front F[1] upper bound | 1.57 m/s | 1.26 m/s | **~24 % inflation** in hypervolume |

**Why it's not a feasibility bug**: F[1] is an objective, not a constraint. Feasibility (closure, boundary, inventory, multi-component, loose-port) is unaffected by speed values. The optimizer's exploration **trajectory** is identical with or without measured physics — but the **objective values it reports** are systematically inflated for non-slide-bound layouts.

**Why it matters for the thesis**: any quantitative F[1] claim against a straight-rich layout is fictitious. The methodology chapter would have to document the inflation if not fixed. Hypervolume convergence plots are wrong on the F[1] axis.

---

## §2 — Findings (4-agent parallel audit, 2026-05-08)

Four general-purpose agents (Sonnet) ran in parallel, each scoped to one slice of the repo. All four converged on the same conclusion via independent file inspection.

### Finding F1 — Wrong YAML pointer (8/8 configs)

Agent 1 confirmed via Glob + per-config Read:

| Config | `train_config_path:` field | Actual loaded values |
|---|---|---|
| `default.yaml` | `trains/default.yaml` | mass_loco=0.370, no v_motor_max → falls to TrainConfig default 1.10 |
| `default_only_mutation.yaml` | `trains/default.yaml` | same |
| `with_switches.yaml` | `trains/default.yaml` | same |
| `with_switches_only_mutation.yaml` | `trains/default.yaml` | same |
| `with_crossing.yaml` | `trains/default.yaml` | same |
| `with_crossing_only_mutation.yaml` | `trains/default.yaml` | same |
| `compact.yaml` | `trains/default.yaml` | same |
| `compact_only_mutation.yaml` | `trains/default.yaml` | same |

`configs/trains/default.yaml` content (verified against the file):
- `mass_loco: 0.370` (V2 default, NOT measured 0.493)
- `mass_trailing: 0.600` (a stale guess, NOT measured 0.327)
- `coupler_offset: 0.100` (V2 default, NOT measured 0.106)
- `max_accel: 1.49` (yet another stale value, neither V2 default 3.92 nor measured 0.68)
- **No `v_motor_max` field at all** → silently falls back to TrainConfig dataclass default (1.10)
- **No `gauge_b` field at all** → falls back to default 0.0375 (this happens to match measured by coincidence)

### Finding F2 — Dead config-loading code path

Agent 1 + Agent 4 cross-verified:

`OptimizationConfig.load_train_config()` exists at `src_v2/config.py:145-149`. The implementation is correct:
- Reads `self.train_config_path`, joins with `self._base_dir`
- Calls `TrainConfig.from_yaml(resolved_path)`
- Falls back to `TrainConfig()` (V2 defaults) if `train_config_path` is None

But:
- `src_v2/runner.py:run_optimization()` **never calls `config.load_train_config()`**
- `run_v2.py` and `run_v2_all_configs.py` (CLI entries) never call it
- `PortPairProblem.__init__()` has no `train_config` parameter and no `self.train_config` attribute
- The loaded `TrainConfig` object would have **zero consumers** even if it were instantiated

### Finding F3 — Catalog speed_table is hardcoded, decoupled from TrainConfig

Agent 3 confirmed by reading `src_v2/catalog/catalog.py` fully:

Module-level constants:

```python
# catalog.py:55-58
_V2_DEFAULT_PHYSICS: Dict[str, Tuple[Optional[float], float]] = {
    "R40_CURVE": (320.0, 0.97),
}

# catalog.py:62-74
_V2_ROUTE_PHYSICS = {
    "R40_SWITCH_LEFT":  {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "R40_SWITCH_RIGHT": {"through": (None, 1.57), "diverging": (320.0, 0.97)},
    "CROSS_90":         {"horizontal": (None, 1.57), "vertical": (None, 1.57)},
    "DOUBLE_CROSSOVER": {
        "track1_through": (None, 1.57), "track2_through": (None, 1.57),
        "cross_1_to_2":   (320.0, 0.97), "cross_2_to_1":   (320.0, 0.97),
    },
}

# catalog.py:95
DEFAULT_SPEED = 1.57

# catalog.py:113
@classmethod
def load(cls, path):  # ← takes NO train_config parameter
```

`get_speed_for_route(piece_idx, route_name)` at `catalog.py:412-432` reads the static dict. **No code path connects `TrainConfig.v_motor_max` to these values.**

The `1.57` constant is independent of (and higher than) both V2 default `v_motor_max=1.10` AND measured `v_motor_max=1.26`. It appears to be a stale catalog-era estimate predating the `TrainConfig` system, frozen at module-import time.

### Finding F4 — `compute_speed_profile()` is orphaned

Agent 4 grep-verified across 6 files (`problem.py`, `decoder.py`, `repair.py`, `operators.py`, `junction_materializer.py`, `templates.py`): **zero call sites** of `compute_speed_profile()`. The function exists in `src_v2/train/scoring.py`, is exported from `src_v2/train/__init__.py`, has tests, and is invoked by no production code.

The full physics pipeline that PLAN.md Rule 35 was supposed to wire up exists in isolation. Rule 35 states "F[1] aggregates over (slot, route) pairs from `branch_labels`" — which the implementation satisfies — but the rule did not specify that the SOURCE of speeds must be `TrainConfig`-derived.

---

## §3 — Disconnect Chain (visual)

```
configs/trains/measured_consist.yaml   [ measured 1.26, 0.493, 0.327, 0.106, 0.68 ]
         │
         ↓ (BROKEN POINTER — F1)
configs/*.yaml all point to "trains/default.yaml"   [ stale: 0.370, 0.600, 0.100, 1.49 ]
         │
         ↓ (DEAD CODE PATH — F2)
OptimizationConfig.load_train_config()   [ correct impl, never called ]
         │
         ↓
PortPairProblem.__init__   [ no train_config param, no self.train_config ]
         │
         ↓ (HARDCODED FALLBACK — F3)
problem.py:271 → catalog.get_speed_for_route()   [ static dict, returns 1.57 / 0.97 ]
         │
         ↓
F[1] = -min_speed   [ inflated by 1.57/1.26 ≈ 1.24x for straight-rich layouts ]

         (compute_speed_profile() exists but is NEVER CALLED — F4)
```

Three sequential disconnections, each one independent. Fixing only F1 (repoint configs) leaves F2/F3 broken. Fixing F1+F2 leaves F3 broken. **All three must land for measured values to reach F[1]**.

---

## §4 — Why bi-objective NSGA-II survives the bug

The thesis story doesn't fall apart, but the numbers are wrong:

1. **Feasibility is unaffected**: speed values appear only in the F[1] objective. Constraints (closure, boundary, inventory, multi-component, loose-port) use no speed.

2. **Pareto front shape is approximately correct**: F[1] is dominated by R40 slide-bound 0.886 m/s on every layout that contains an R40 curve (which is every closed cycle in the inventory). The minimum across cycle slots is 0.886 regardless of whether catalog says 1.57 or train_config says 1.26 for the through routes.

3. **F[1] upper bound is inflated**: only when computing **average** F[1] or hypervolume reference points, the catalog's 1.57 inflates the apparent F[1] range. For thesis hypervolume reporting this is a real issue.

4. **Pure-curve layouts are honest**: 16 R40 oval reports F[1] = -0.886, train physically does ~0.886 — no discrepancy.

This explains why all 212 tests passed despite the bug: no test asserts F[1] matches measured-physics-derived values. Tests assert F[1] matches catalog-derived values, which is internally consistent but disconnected from the user's hardware.

---

## §5 — Tier 1: Minimum Fix (~30 minutes, unblocks real-LEGO test)

### Tier 1.1 — Repoint all 8 configs

**Files**: `configs/default.yaml`, `configs/with_switches.yaml`, `configs/with_crossing.yaml`, `configs/compact.yaml`, plus their 4 `_only_mutation` siblings.

**Edit** (each file):

```yaml
# Before:
train_config_path: trains/default.yaml

# After:
train_config_path: trains/measured_consist.yaml
```

Optional: delete `configs/trains/default.yaml` entirely OR overwrite it to be a symlink/duplicate of `measured_consist.yaml`. Recommendation: keep `default.yaml` as-is for fallback, just stop pointing configs at it.

### Tier 1.2 — Wire `TrainConfig` into `PortPairProblem`

**File**: `src_v2/runner.py`, in `run_optimization()`.

Add before constructing the problem:

```python
train_config = config.load_train_config()
logger.info(
    f"Train physics: v_motor_max={train_config.v_motor_max:.2f} m/s, "
    f"mass_total={train_config.mass_total:.3f} kg, "
    f"coupler_offset={train_config.coupler_offset:.3f} m, "
    f"max_accel={train_config.max_accel:.2f} m/s²"
)
```

Pass to problem constructor:

```python
problem = PortPairProblem(
    catalog, config,
    train_config=train_config,
    elementwise_runner=runner,
)
```

**File**: `src_v2/problem.py`, `PortPairProblem.__init__()`.

Add parameter and store:

```python
def __init__(
    self,
    catalog: TrackCatalog,
    config: OptimizationConfig,
    train_config: TrainConfig | None = None,   # NEW
    closure_tolerance: float = None,
    angle_tolerance: float = None,
    **kwargs,
) -> None:
    # ... existing init ...
    from .train import TrainConfig, DEFAULT_TRAIN_CONFIG
    self.train_config = train_config or DEFAULT_TRAIN_CONFIG
```

Add the import at the top:

```python
from .train import TrainConfig, DEFAULT_TRAIN_CONFIG
```

### Tier 1.3 — Replace catalog speed lookup in F[1] computation

**File**: `src_v2/problem.py:_evaluate`, around line 271.

Read the current code first:

```python
# Current (line ~271):
speed = self.catalog.get_speed_for_route(piece_idx, route_name)
```

Replace with train-config-aware lookup. Two options:

**Option A — minimal: bypass speed_table entirely, compute v_eff per piece**

```python
# Get the geometric radius from the catalog (this IS catalog data — geometry, not physics)
radius_m = self.catalog.get_radius_m_for_route(piece_idx, route_name)
# Speed cap is min(curvature limit, motor limit) — both from train_config
import math
speed = self.train_config.v_eff(radius_m if radius_m is not None else math.inf)
```

This requires adding `get_radius_m_for_route()` to `TrackCatalog`:

```python
# In src_v2/catalog/catalog.py
def get_radius_m_for_route(self, piece_idx: int, route_name: str) -> Optional[float]:
    """Geometric curve radius for a (piece, route) pair, in meters.

    None for straight routes (motor-bound only). Distinct from
    get_speed_for_route — this returns geometry; speed comes from
    TrainConfig.v_eff().
    """
    piece = self._index_to_piece.get(piece_idx)
    if piece is None:
        return None
    route_physics = _V2_ROUTE_PHYSICS.get(piece.id, {})
    route_entry = route_physics.get(route_name)
    if route_entry is None:
        # Fall back to default-route physics
        radius_mm, _speed = _V2_DEFAULT_PHYSICS.get(piece.id, (None, 1.57))
    else:
        radius_mm, _speed = route_entry
    return radius_mm / 1000.0 if radius_mm else None
```

**Option B — surgical: edit speed_table at catalog-load time**

(Tier 2 architectural decision; not Tier 1 minimum.)

### Tier 1 Verification

```bash
cd S:\Programming\Repo\DamianGRZ\LegoTrackOptimizer
.venv/Scripts/python.exe -m pytest tests/ --tb=short -q
```

Expected: 212/212 pass.

```bash
.venv/Scripts/python.exe run_v2.py with_switches_only_mutation --gen 50
```

Expected: log line shows `Train physics: v_motor_max=1.26 m/s, mass_total=0.820 kg, coupler_offset=0.106 m, max_accel=0.68 m/s²`. Best feasible F[1] should be -0.886 (slide-bound R40), unchanged from before — verifying that for curve-rich layouts the fix is a no-op as expected.

```bash
.venv/Scripts/python.exe -c "from src_v2.config import OptimizationConfig; from pathlib import Path; cfg = OptimizationConfig.load(Path('configs/with_switches.yaml')); tc = cfg.load_train_config(); print(f'v_motor_max={tc.v_motor_max}, mass_total={tc.mass_total}'); assert abs(tc.v_motor_max - 1.26) < 0.001, 'WRONG'; assert abs(tc.mass_total - 0.820) < 0.001, 'WRONG'; print('OK')"
```

Expected: `v_motor_max=1.26, mass_total=0.82` then `OK`. If `WRONG`, Tier 1.1 didn't land cleanly.

---

## §6 — Tier 2: Architectural Cleanup (~2 hours, post-real-LEGO test)

### Tier 2.1 — Decide catalog speed_table fate

Two clean options:

**(a) Delete `speed_table` and `get_speed_for_route()` entirely.** Force all callers to use `train_config.v_eff(radius_m)`. The catalog returns geometry (radii, FK deltas, port positions); physics lives in `TrainConfig`. This is the right architectural separation and matches PLAN.md Rule 23 ("use measured-consist YAML as single source of train physics").

**(b) Keep `speed_table` but make it train-config-aware.** Pass `train_config` to `TrackCatalog.load()`, recompute `_speed_table[i] = train_config.v_eff(radius_m_for_piece_i)` at load time. Backward-compatible with any existing callers but couples catalog instances to specific train configs.

**Recommendation**: (a). Cleaner, surfaces hidden coupling, removes the stale `_V2_ROUTE_PHYSICS` constants. Migration cost: every `catalog.get_speed_for_route()` call site (currently only `problem.py:271` per Agent 4's grep) gets replaced with the Tier 1.3 v_eff pattern.

### Tier 2.2 — Deprecate hardcoded physics constants

After Tier 2.1 lands, delete from `src_v2/catalog/catalog.py`:

- `_V2_DEFAULT_PHYSICS` (line 55-58)
- `_V2_ROUTE_PHYSICS` (line 62-74)
- `DEFAULT_SPEED = 1.57` (line 95)
- `speed_limit_ms` field in `TrackPiece` (`src_v2/catalog/pieces.py:51`)

Keep only the geometry portions (radii, route names, FK deltas). All speed-derived behavior moves to `TrainConfig.v_eff(radius)`.

### Tier 2.3 — Startup log line

Already specified in Tier 1.2; lift if not already in place. Every run announces:

```
Train physics: v_motor_max=1.26 m/s, mass_total=0.820 kg, coupler_offset=0.106 m, max_accel=0.68 m/s²
```

Catches future regressions instantly — if the value is `1.10` you know the wiring broke again.

### Tier 2.4 — Wiring smoke test

New file: `tests/test_train_physics_wiring.py`.

```python
"""Verify the chain: config.yaml → load_train_config → PortPairProblem → F[1].

Catches the disconnect bug discovered 2026-05-08: measured physics in
configs/trains/measured_consist.yaml were silently bypassed because the
loading path was a dead code branch.
"""
def test_problem_has_measured_train_config():
    """F[1] for a 16-R40 oval should be 0.886 m/s (slide-bound), and
    train_config.v_motor_max must be 1.26 (measured), not 1.10 (V2 default)."""
    cfg = OptimizationConfig.load(Path("configs/with_switches.yaml"))
    catalog = TrackCatalog.load(Path("data/track_pieces_v2.yaml"))
    train_cfg = cfg.load_train_config()
    assert abs(train_cfg.v_motor_max - 1.26) < 0.001, \
        "v_motor_max not loaded from measured_consist.yaml"
    assert abs(train_cfg.mass_total - 0.820) < 0.001, \
        "mass_total not loaded from measured_consist.yaml"
    problem = PortPairProblem(catalog, cfg, train_config=train_cfg)
    assert problem.train_config is train_cfg, "problem doesn't store train_config"
    # Optional: evaluate a known fixture chromosome and assert F[1] ≈ -0.886
```

---

## §7 — Tier 3: `compute_speed_profile()` (DEFERRED)

The 3-pass profile (forward accel + backward brake + curvature limit + double-unroll) is overkill for `F[1] = -min_speed`. PLAN.md Rule 22 documents this:

> "For min_speed, skip the 3-pass profile and use Pass 1 directly: passes 2 and 3 only lower speeds; they cannot raise the per-piece curvature cap."

Tier 1.3's `train_config.v_eff(radius)` per piece IS Pass 1. The 3-pass is needed only for `lap_time` or `avg_speed` reporting — which is a Phase 9 future-work scope.

**Tier 3 lands only if**:
- A future Phase 9 adds `lap_time` as a third objective (Rule 33 currently forbids this — bi-objective discipline)
- OR the thesis methodology chapter wants per-segment speed profiles in figures

Otherwise, `compute_speed_profile()` remains exported but uncalled, and that's fine.

---

## §8 — Cross-References to docs/PLAN.md

| Topic in PLAN.md | Section | Relevance to this remediation |
|---|---|---|
| Rule 23 — measured-consist YAML as single source | Part 8 §"Phase 0 deliverable" / Part 7 | This remediation IS the missing wire-up Rule 23 implies |
| Rule 22 — Pass 1 only for min_speed | Part 8 §"Bottom line" / Part 7 | Justifies Tier 3 deferral |
| Rule 35 — F[1] aggregates over (slot, route) pairs | Part 7 / Part 9 §9.5 | Currently satisfied syntactically but with wrong source — Tier 1.3 fixes the source |
| Phase 0 measurement protocol | Part 8 | The measured values exist correctly in `measured_consist.yaml`; problem is downstream |
| Coupling E — F[1] refactor methodology | Part 10 §10.3 | Tangentially related — Coupling E was about per-piece vs per-route migration; this disconnect is one layer deeper (per-route vs train-config) |
| Bi-objective discipline (Rule 33) | Part 9 §9.5 | Preserved by Tiers 1-2; Tier 3 deferral honors it |

This plan does NOT supersede any rule in PLAN.md. It implements the wire-up that Rules 23 + 35 imply but were never explicitly tested for.

---

## §9 — Root Cause Analysis (why this wasn't caught earlier)

The bug survived multiple review passes:

1. **Phase 0 measurement audit** (Part 8 of PLAN.md): focused on the *physics correctness* of measured values. Did not assert the values flow through to runtime.

2. **Post-Phase-4 review** (python-pymoo-reviewer): reviewed code against per-phase specs. Spec said "TrainConfig defaults are tunable via YAML"; reviewer verified `from_yaml()` works in isolation. Did not trace the loading path end-to-end.

3. **Post-Phase-5-8 review** (current session): focused on Phase 5a's per-route F[1] refactor. Verified `get_speed_for_route()` exists and is called from `problem.py:271`. Did not question whether `get_speed_for_route()` itself reads measured physics or hardcoded constants.

4. **§9.7 sanity run** (60.6% feasibility verified): the run completed and logged "best_min_speed=0.97 m/s" — a value that's slide-bound on R40 anyway. No comparison to expected measured-physics-derived value happened, because the test framework had no fixture for "F[1] should be X given measured train Y."

5. **212 tests pass**: every existing test asserts F[1] matches the catalog's static values. Internally consistent. Disconnected from hardware reality.

**The lesson for thesis methodology**: a "loading path smoke test" (Tier 2.4) should exist for every YAML-driven configuration parameter. PLAN.md's testing pyramid (Part 10 §10.1) defines T1-T5 tiers but does not have a "loading-path integration tier" that asserts YAML values reach hot code.

This is documented as a methodology lesson in the thesis chapter; the test in Tier 2.4 prevents recurrence.

---

## §10 — Recommended Execution Order

**Today**:

1. Apply Tier 1.1 (8 YAML edits) — ~5 minutes
2. Apply Tier 1.2 (`runner.py` + `problem.py` wiring) — ~10 minutes
3. Apply Tier 1.3 (catalog `get_radius_m_for_route` + `problem.py` v_eff replacement) — ~15 minutes
4. Run Tier 1 verification commands — ~5 minutes
5. If green: real-LEGO test is unblocked

**Same session if time**:

6. Tier 2.4 (smoke test) — ~30 minutes

**After real-LEGO test**:

7. Tier 2.1 (decide catalog speed_table fate)
8. Tier 2.2 (deprecate stale constants)
9. Tier 2.3 (already in place from Tier 1.2)
10. Update CLAUDE.md or PLAN.md TOC to reference this remediation as completed

**Phase 9 / future**:

11. Tier 3 (compute_speed_profile integration) — only if `lap_time` objective is added

---

## Document History

- **2026-05-08**: Bug discovered via 4-agent parallel audit (config + train + catalog + fitness paths). All 4 agents converged on the disconnect chain. This plan written immediately to capture findings + fix path before momentum is lost.

---

## Appendix A — Agent Audit Summary

| Agent | Scope | Verdict |
|---|---|---|
| A1 | `configs/*.yaml` + `src_v2/config.py` + `src_v2/runner.py` | All 8 configs point to wrong YAML; `load_train_config()` is a dead code path |
| A2 | `src_v2/train/*.py` | `TrainConfig.from_yaml()` correct in isolation; `compute_speed_profile()` defaults to V2 if caller omits param |
| A3 | `src_v2/catalog/*.py` + `data/track_pieces_v2.yaml` | Catalog speed_table is hardcoded module-level dicts, completely decoupled from `TrainConfig` |
| A4 | `src_v2/problem.py` + decoder + repair + operators + materializer + templates | F[1] reads `catalog.get_speed_for_route()` which returns 1.57/0.97; `compute_speed_profile()` has zero call sites |

All four reports cite literal file:line. The disconnect chain in §3 is the synthesis.

---

## Appendix B — Specific Hardcoded Values to Watch

When applying Tier 2.2, search for these as the regression-prevention sweep:

| Value | Where it appears | Fix |
|---|---|---|
| `1.57` | `catalog.py:51,63-73,95`, `pieces.py:51`, `types.py:45` | Delete; replace consumers with `train_config.v_eff(inf)` |
| `0.97` | `catalog.py:57,63-74` | Delete; replace consumers with `train_config.v_eff(0.32)` |
| `1.10` | `train/physics.py:34` (TrainConfig default) | Keep as fallback default; Tier 1.1 ensures YAML overrides it |
| `0.370`, `0.0`, `0.100`, `3.92` | `train/physics.py:40-46` (TrainConfig defaults) | Keep as fallback; Tier 1.1 ensures YAML overrides |
| `0.886` | (computed, not hardcoded) | No action — derived from `mu_design × g × R` |

After Tier 2 cleanup, the only hardcoded physics defaults should live in `TrainConfig`'s dataclass field declarations, accessible only when no YAML is loaded (an explicit fallback), and visible in the startup log line.
