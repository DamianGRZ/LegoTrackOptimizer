# Selective Epsilon — Keep Collisions/Inventory Hard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the adaptive-epsilon constraint handler relax only closure + boundary (SOFT); keep collisions + per-type inventory HARD, so closed self-crossing / over-inventory individuals can never be treated as feasible and stop dominating the population for hundreds of generations.

**Architecture:** `LegoAdaptiveEpsilon` currently sets a single scalar `cv_eps` (≤30) that relaxes the *summed* CV, which accidentally shields tiny collision violations (a closed self-crosser has CV ≈ 0.42 ≤ 30). Switch to pymoo's per-constraint `cv_ieq["eps"]` array: schedule epsilon on the first `n_soft` constraints (closure G[0:3] + boundary G[3]) and 0 on the rest (collisions G[4] + inventory G[5:]), with top-level `cv_eps = 0`. Then `FEAS = (Σ_i max(0, G_i − eps_i) ≤ 0)`.

**Tech Stack:** Python, pymoo 0.6.1.6 (`pymoo.constraints.eps.AdaptiveEpsilonConstraintHandling`, `pymoo.core.individual.calc_cv`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-selective-epsilon-hard-collisions-design.md`

---

## File Structure

- **Modify** `src/algorithm/runner.py`
  - add module constant `SOFT_CONSTRAINT_COUNT = 4`
  - `LegoAdaptiveEpsilon.__init__`: add required `n_ieq_constr` and `n_soft=SOFT_CONSTRAINT_COUNT`
  - `LegoAdaptiveEpsilon._adapt_constraint_handling`: write a per-constraint `cv_ieq["eps"]` array + `cv_eps = 0` instead of the scalar `cv_eps`
  - caller at `run_optimization` (~line 448): pass `n_ieq_constr=problem.n_ieq_constr`
- **Create** `tests/test_selective_epsilon.py` — unit tests of the FEAS semantics produced by the real `_adapt_constraint_handling`.
- **Create** `scripts_verify_selective_epsilon.py` (temporary verification, deleted after Task 2) — short GA run proving self-crossing champions no longer dominate while feasibles are still found.

Only one production caller exists (`runner.py:448`); `src/algorithm/__init__.py` re-exports the name but does not construct it; no test constructs it. So the signature change is fully covered by updating that single call site.

---

### Task 1: Per-constraint epsilon (SOFT closure+boundary, HARD collisions+inventory)

**Files:**
- Modify: `src/algorithm/runner.py` (class `LegoAdaptiveEpsilon` ~313-372; caller ~448)
- Test: `tests/test_selective_epsilon.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_selective_epsilon.py`:

```python
"""Selective epsilon: closure + boundary are epsilon-relaxable (SOFT); collisions
and per-type inventory are HARD (never relaxed). A closed self-crossing layout
must therefore stay infeasible even at full epsilon, while a near-closed clean
siding is still relaxed. Tests exercise the real LegoAdaptiveEpsilon._adapt_
constraint_handling and pymoo's own calc_cv (no reimplementation of CV)."""
import types

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual, calc_cv

from src.algorithm.runner import LegoAdaptiveEpsilon, SOFT_CONSTRAINT_COUNT


def _adapted_config(perc, n_ieq=12, n_soft=SOFT_CONSTRAINT_COUNT, max_cv=30.0,
                    hold_until=0.2, perc_eps_until=0.9):
    """The config LegoAdaptiveEpsilon writes at schedule position ``perc``."""
    h = LegoAdaptiveEpsilon(NSGA2(pop_size=10), n_ieq_constr=n_ieq,
                            hold_until=hold_until, perc_eps_until=perc_eps_until)
    h.max_cv = max_cv
    h.termination = types.SimpleNamespace(perc=perc)
    cfg = {}
    h._adapt_constraint_handling(cfg)
    return cfg


def _feas(G, cfg):
    full = {**Individual.default_config(), **cfg}
    cv = float(calc_cv(G=np.asarray(G, dtype=float), config=full))
    return bool(cv <= cfg["cv_eps"])


def _G(closure=(-1.0, -1.0, -1.0), boundary=0.0, collisions=0.0, inv=0.0):
    # n_ieq=12 layout: [closure_x, closure_y, closure_theta, boundary,
    #                   collisions, inv_0..inv_6]
    return list(closure) + [boundary, collisions] + [inv] * 7


def test_closed_self_crosser_infeasible_even_at_full_epsilon():
    cfg = _adapted_config(perc=0.0)  # hold phase, alpha=1, full epsilon
    G = _G(closure=(-1, -1, -1), boundary=0.02, collisions=0.4)
    assert not _feas(G, cfg)


def test_near_closed_clean_siding_stays_epsilon_feasible():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(4.0, 4.0, 0.5), boundary=0.01, collisions=0.0)
    assert _feas(G, cfg)


def test_standalone_feasible_circle_is_feasible():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(-1, -1, -1), boundary=-0.1, collisions=0.0)
    assert _feas(G, cfg)


def test_over_inventory_infeasible_even_at_full_epsilon():
    cfg = _adapted_config(perc=0.0)
    G = _G(closure=(-1, -1, -1), collisions=0.0, inv=0.3)
    assert not _feas(G, cfg)


def test_strict_phase_relaxes_nothing():
    cfg = _adapted_config(perc=0.95)  # past perc_eps_until -> alpha=0
    assert cfg["cv_eps"] == 0.0
    G = _G(closure=(4.0, 0.0, 0.0), collisions=0.0)
    assert not _feas(G, cfg)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_selective_epsilon.py -q`
Expected: FAIL at import/collection — `ImportError: cannot import name 'SOFT_CONSTRAINT_COUNT'` (and `LegoAdaptiveEpsilon.__init__` does not yet accept `n_ieq_constr`).

- [ ] **Step 3: Add the SOFT_CONSTRAINT_COUNT constant**

In `src/algorithm/runner.py`, immediately above `class LegoAdaptiveEpsilon(...)`:

```python
# Closure (G[0:3]) and boundary (G[3]) are SOFT: a near-closed / near-fitting loop
# can evolve toward satisfying them, so the epsilon schedule may relax them.
# Collisions (G[4]) and per-type inventory (G[5:]) are HARD: structurally
# unbuildable, never relaxed. See docs/superpowers/specs/
# 2026-06-07-selective-epsilon-hard-collisions-design.md.
SOFT_CONSTRAINT_COUNT = 4
```

- [ ] **Step 4: Update `__init__` to take the constraint count**

Replace `LegoAdaptiveEpsilon.__init__` with:

```python
    def __init__(self, algorithm, n_ieq_constr, hold_until=0.2,
                 perc_eps_until=0.7, n_soft=SOFT_CONSTRAINT_COUNT):
        super().__init__(algorithm, perc_eps_until=perc_eps_until)
        self.hold_until = hold_until
        self.n_ieq_constr = int(n_ieq_constr)
        self.n_soft = int(n_soft)
        self._logger = logging.getLogger(__name__)
```

- [ ] **Step 5: Rewrite `_adapt_constraint_handling` to set a per-constraint eps array**

Replace `LegoAdaptiveEpsilon._adapt_constraint_handling` with:

```python
    def _adapt_constraint_handling(self, config, **kwargs):
        t = self.termination.perc
        if t < self.hold_until:
            alpha = 1.0
        elif t < self.perc_eps_until:
            alpha = 1.0 - (t - self.hold_until) / (self.perc_eps_until - self.hold_until)
        else:
            alpha = 0.0
        eps_value = alpha * self.max_cv

        # Epsilon relaxes ONLY the soft constraints (closure + boundary). The
        # hard constraints (collisions, per-type inventory) get eps 0, so any
        # violation there keeps the individual infeasible no matter the schedule
        # -- a closed self-crosser (collisions > 0) can never be treated feasible.
        eps = np.zeros(self.n_ieq_constr, dtype=float)
        eps[: self.n_soft] = eps_value
        config["cv_ieq"] = dict(scale=None, eps=eps, pow=None, func=np.sum)
        config["cv_eps"] = 0.0
```

- [ ] **Step 6: Update the only caller to pass `n_ieq_constr`**

In `run_optimization` (~`runner.py:448`), replace:

```python
    algorithm = LegoAdaptiveEpsilon(
        base_algorithm,
        hold_until=0.2,
        perc_eps_until=0.9,
    )
```

with:

```python
    algorithm = LegoAdaptiveEpsilon(
        base_algorithm,
        n_ieq_constr=problem.n_ieq_constr,
        hold_until=0.2,
        perc_eps_until=0.9,
    )
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_selective_epsilon.py -q`
Expected: PASS (5 passed).

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: the 5 new tests pass; the only pre-existing failures remain exactly `tests/test_catalog.py::TestV1Deprecation::test_loading_v1_yaml_warns` (missing `data/track_pieces.yaml`) and `tests/test_dbl_crossover_inject.py::TestInjectionHappyPaths::test_two_layer_loop_closes` (unrelated decoder geometry). No other test changes status.

- [ ] **Step 9: Commit**

```bash
git add tests/test_selective_epsilon.py src/algorithm/runner.py
git commit -m "feat(epsilon): relax only closure+boundary; keep collisions/inventory hard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Integration verification — self-crossing champions no longer dominate

**Files:**
- Create (temporary): `scripts_verify_selective_epsilon.py`

Goal: prove behaviourally that the change de-thrones the persistent self-crossing champion while preserving the epsilon benefit (feasibles still found). Baseline (pre-fix, measured earlier): population-with-self-crossing-circle grows to ~1000/1000 by gen 15.

- [ ] **Step 1: Write the verification script**

Create `scripts_verify_selective_epsilon.py`:

```python
"""Short GA run on all_pieces, asserting selective epsilon de-thrones the
self-crossing champion (circle-bearing self-crossers no longer take over) while
feasibles are still produced. Prints per-gen counts; exits 0 on PASS, 1 on FAIL."""
import sys
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.problem import TrackOptimizationProblem
from src.sampling import IntegerSampling
from src.repair import TrackRepairPipeline
from src.operators import PartitionedCrossover, PartitionedMutation
from src.algorithm.runner import LegoAdaptiveEpsilon
from src.decoder import decode_chromosome
from src.intersection import find_crossing_pairs

cat = TrackCatalog.load("data/track_pieces_v2.yaml")
cfg = OptimizationConfig.load("configs/all_pieces.yaml")
prob = TrackOptimizationProblem(cat, cfg)
dims = prob.dims
sampler = IntegerSampling(cat, cfg, heuristic_ratio=cfg.algorithm.heuristic_ratio)
repair = TrackRepairPipeline(dims=dims, inventory_by_index=prob._convert_inventory(cfg.inventory),
                             catalog_fk_table=cat._fk_table, enable_closure_repair=True)
base = NSGA2(pop_size=cfg.algorithm.pop_size, sampling=sampler,
             crossover=PartitionedCrossover(dims, prob=cfg.algorithm.crossover_prob),
             mutation=PartitionedMutation(dims, prob=cfg.algorithm.mutation_prob),
             repair=repair, survival=ConstrRankAndCrowding(),
             eliminate_duplicates=cfg.algorithm.eliminate_duplicates)
algo = LegoAdaptiveEpsilon(base, n_ieq_constr=prob.n_ieq_constr, hold_until=0.2, perc_eps_until=0.9)

TOL = 1e-9
last = {}

def self_crossers_and_feas(alg):
    X = alg.pop.get("X"); F = alg.pop.get("F"); G = alg.pop.get("G")
    util = -F[:, 0]
    cv = np.maximum(G, 0).sum(1)
    feas = (cv <= TOL).sum()
    sc = 0
    for r in range(len(X)):
        L = decode_chromosome(X[r], cat, cfg.inventory, dims=dims)
        if len(find_crossing_pairs(L.states, list(L.main_loop_pieces))) > 0 and util[r] >= 0.6:
            sc += 1
    return sc, int(feas), float(util[cv <= TOL].max()) if feas else 0.0

class CB(Callback):
    def notify(self, alg):
        g = alg.n_gen
        if g not in (5, 10, 15):
            return
        sc, feas, bestfeas = self_crossers_and_feas(alg)
        last.update(gen=g, self_crossers=sc, feasible=feas, best_feas_util=bestfeas)
        print(f"gen {g:2d}: self-crossing-champions(util>=60%)={sc}/1000  feasible={feas}/1000  best_feas_util={bestfeas:.1%}")

minimize(prob, algo, get_termination("n_gen", 15), seed=42, verbose=False, callback=CB())

# PASS criteria: by gen 15 the self-crossing champions do NOT dominate (baseline was ~1000),
# AND the run still produces feasible solutions.
ok = last.get("self_crossers", 1000) <= 100 and last.get("feasible", 0) >= 1
print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
      f"(self_crossers={last.get('self_crossers')}, feasible={last.get('feasible')})")
sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Run the verification**

Run: `.venv/Scripts/python.exe scripts_verify_selective_epsilon.py`
Expected: prints per-gen lines; by gen 15 `self-crossing-champions` is far below the ~1000/1000 baseline (criterion `<= 100`) and `feasible >= 1`; final line `RESULT: PASS`; exit code 0.

If it prints `RESULT: FAIL` because `feasible == 0` (epsilon over-tightened — population can no longer close), this means the per-constraint closure budget is too generous-then-strict; the fix is to confirm `eps_value` is applied per soft constraint (Task 1 Step 5) and re-run. Do not loosen collisions. If `self_crossers` is still high (~1000), re-check that `config["cv_eps"] = 0.0` and the eps array has 0 on indices `>= n_soft` (Task 1 Step 5).

- [ ] **Step 3: Delete the temporary verification script**

Run: `rm scripts_verify_selective_epsilon.py`
(It is a throwaway harness, not a committed test — the committed coverage is `tests/test_selective_epsilon.py`.)

- [ ] **Step 4: Commit (nothing to commit if only the deleted script changed)**

No commit needed — the script was never committed. Task 1's commit is the deliverable; Task 2 is a behavioural gate.

---

## Self-Review

**Spec coverage:**
- SOFT/HARD split (closure+boundary vs collisions+inventory) → Task 1 Steps 3–5. ✓
- pymoo hook = per-constraint `cv_ieq["eps"]` + `cv_eps=0` → Task 1 Step 5. ✓
- Indices derived, not hardcoded (constraint count from `problem.n_ieq_constr`) → Task 1 Steps 4, 6. ✓
- Unit tests (self-crosser infeasible, near-closed-clean relaxed, standalone circle feasible) → Task 1 Step 1 (plus over-inventory and strict-phase cases). ✓
- Integration (~15 gens, champions don't dominate, feasibles preserved) → Task 2. ✓
- Open question (per-constraint vs shared budget) → resolved deterministically: full `eps_value` per soft constraint; Task 2 gates that closure relaxation neither over- nor under-shoots (feasibles still found). ✓
- Out of scope (seeding, circle excision, objective) → untouched by every task. ✓

**Placeholder scan:** none — all steps carry exact code/commands and expected output.

**Type consistency:** `SOFT_CONSTRAINT_COUNT` (constant), `n_ieq_constr`/`n_soft` (`__init__` params), `eps` (np array len `n_ieq_constr`), `config["cv_ieq"]`/`config["cv_eps"]` (pymoo config keys), `problem.n_ieq_constr` (attribute confirmed in `problem.py`) — all names match across tasks and the pymoo source.
