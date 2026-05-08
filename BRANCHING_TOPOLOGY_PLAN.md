# Branching Topology Implementation Plan

> **Hand-off plan for the PyCharm AI assistant.** Self-contained — everything needed to implement Phases 1–5 is in this document.

---

## 0. Project context

LEGO Track Optimizer V2: pymoo NSGA-II producing closed LEGO/4DBrix layouts from a fixed inventory. Current bug: layouts containing **multi-port pieces** (3-port switches, 4-port CROSS_90 / DOUBLE_CROSSOVER) collapse to **1/500 feasible** because the default crossover destroys structural fragments faster than mutation rebuilds, and switches with dangling port C survive as decorative pieces that never form branch cycles.

**Goal**: produce layouts with ≥ 2 route-complete switches and ≥ 1 closed branch cycle in the best feasible individual on `with_switches.yaml`; match or beat the documented mutation-only baseline (4–6 switches, 3–4 cycles, 500/500 feasible) on hypervolume.

---

## 1. Environment

**Use the project venv for everything**:

```cmd
.venv\Scripts\python.exe run_v2.py <config>
.venv\Scripts\python.exe -m pytest tests/...
```

Why: the venv runs Python 3.11.9 with pymoo 0.6.1.6 (which has `pymoo.parallelization.StarmapParallelization` that public PyPI 0.6.1.5 lacks) plus `ruamel.yaml`, `pydantic 2.x`, `eval_type_backport`. The system `python` (3.9.13) chokes on PEP 604 unions and `itertools.pairwise`.

If you need to install extra dev deps:

```cmd
.venv\Scripts\pip.exe install <package>
```

Do NOT install into the system Python.

---

## 2. What NOT to touch

- **`configs/*.yaml` runtime parameters** — `crossover_prob`, `mutation_prob`, `heuristic_ratio`, `pop_size`, `n_gen`, `seed`. The user sets these manually through the web UI's **tweaks panel** ([web/tweaks-panel.jsx](web/tweaks-panel.jsx)) — `python server.py` + browser is the live development workflow per [CLAUDE.md](CLAUDE.md). YAML values are just initial defaults the UI exposes. **Don't pre-bake parameter sweeps into YAML.**
- **`src_v2/decoder.py` geometry / FK propagation** — settled. Per memory `feedback_geometry_is_done.md`. All branching-topology improvements happen on the GA side (operators, problem, repair, runner, new callbacks), not in pose composition.
- **The Pareto-survival operator** — `survival=ConstrRankAndCrowding()` at [src_v2/runner.py:235](src_v2/runner.py) stays. NSGA-II's combined-population (µ+λ) selection already implements elitism implicitly; the redundant `FeasibleEliteCallback` is *retired* in Phase 5, not replaced.

YAML changes ARE allowed for **new structural settings** that aren't exposed in the tweaks panel: e.g., `min_branch_count`, `branch_completion_max_pieces`, `branch_completion_time_budget_ms`, `epsilon_archive.epsilon`, `hv_reference_offset`. Add those when their owning phase needs them.

---

## 3. Research grounding (one-line per technique)

- **Cellular Encoding** (Gruau 1994) / **L-systems** with bracket notation — branch growth grammars with closure verification. Used in Phase 3's A* search.
- **Bond graphs** — multi-port junction completeness invariants. Used in Phase 2's repair rules.
- **CLLP loop encoding** (Cheng, Gen & Tosawa 1996, doi:10.1016/S0360-8352(96)00276-8) — domain-specific edge-insertion repair for closed loops. Used in Phase 4's Lamarckian closure repair.
- **Lamarckian repair** (Orvosh & Davis 1993) — empirically optimal 5% rate. Phase 4.
- **Memetic EA** (Neri & Cotta 2012, doi:10.1016/j.swevo.2011.11.003) — 10–30% gains from local search per offspring. Phase 4.
- **NSGA-II** (Deb et al. 2002, doi:10.1109/4235.996017) — already in use; correct for m=2.
- **ε-archive** (Laumanns, Thiele, Deb & Zitzler 2002, doi:10.1162/106365602760234108). Phase 5.
- **ε-constraint relaxation** (Takahama & Sakai 2010, doi:10.1109/CEC.2010.5586484) — already at [src_v2/runner.py:133](src_v2/runner.py).
- **Diversity via selection, not objective shaping** (Goldberg & Richardson 1987; Deb 2001) → Phase 1 drops `DIVERSITY_BONUS`.
- **HV reference-point spec** (Ishibuchi-Imada-Setoguchi-Nojima 2018, doi:10.1162/evco_a_00226) → per-config empirical ref point in verification.
- **ALNS / FRRMAB** (Pisinger & Ropke 2007; Li, Fialho, Kwong & Zhang 2014, doi:10.1109/TEVC.2013.2239648) → Phase 5.
- **Heuristic seeding** (Surry & Radcliffe 1996, doi:10.1007/BFb0032789; Kazimipour, Li & Qin 2014, doi:10.1109/CEC.2014.6900618) — 10–50% peak band → Phase 4.
- **Mutation-only justification** — empirical (memory `project_crossover_destroys_feasibility.md`: 1/500 vs 500/500); theoretical context Jansen & Wegener 2002 + Miller 2020. NOT Hansen 2006 (CMA-ES is continuous; uses recombination).
- **Phenotype-level deduplication** (Hildebrandt & Branke 2015, doi:10.1162/EVCO_a_00126; Goldman & Punch 2014 P3, doi:10.1145/2576768.2598350). Phase 5.

---

## 4. Phase 0 — Baseline (DONE — for reference only)

Already executed. Results stored in `outputs_v2/baseline_*/`. Reference numbers:

| Config | Wall | Feasible/500 | Best feasible | Best infeasible | Cycles |
|---|---|---|---|---|---|
| `default` | 346s | **500/500** | 34 pc, 54.9% util, 0.97 m/s | — | 1 |
| `with_switches` | 480s | **1/500** | 42 pc, 25.8% util | 106 pc, 67.3%, CV=0.01 | 1 |
| `with_crossing` | 513s | **1/500** | 46 pc, 47.5% util | 38 pc, 40.5%, CV=0.03 | 1 |

Speed-axis spread observed: 0.97–1.57 m/s (narrow). Used to set ε-archive ε[1] = 0.01 in Phase 5.

---

## 5. Phase 1 — Drop `DIVERSITY_BONUS` from objective

### 5.1 Why

Per audit Recommendation 1: diversity in NSGA-II should be enforced through **selection** (crowding distance + ε-archive), not by *shaping the utilization objective* with a kind-count multiplier. The current `F[0] = -utilization × (1 + 0.10·kind_frac)` distorts hypervolume comparisons (Goldberg & Richardson 1987; Deb 2001).

### 5.2 Files to modify

**[src_v2/problem.py](src_v2/problem.py)**:

1. **Replace docstring (lines 1–31)**:

```python
"""pymoo bi-objective problem for port-pair encoded track layouts.

Objectives (both minimized for pymoo, so return negated values):

- ``F[0] = -utilization`` — fraction of inventory placed in useful components
  (component size >= MIN_USEFUL_COMPONENT_SIZE). Excludes 2-/3-piece side-cycles
  the GA otherwise spawns as a utilization-inflation loophole. Diversity is
  enforced via NSGA-II's native crowding distance plus the external ε-archive
  (Laumanns et al. 2002), not by shaping this objective with a kind-count
  bonus — see Goldberg & Richardson 1987 and Deb 2001 on keeping objectives
  clean and pushing diversity into selection.
- ``F[1] = -min_speed`` — slowest piece speed limit across useful components,
  recovering V1's bottleneck semantics. Pieces in junk components do not
  contribute to either objective.

Constraints (g <= 0 feasible) — see Phase 2 for the reformulation."""
```

2. **Delete the `DIVERSITY_BONUS` and `DIVERSITY_KINDS` blocks (lines 59–70)**. Keep `MIN_USEFUL_COMPONENT_SIZE` (line 50), `BRANCH_ROUTES`, `COLLISION_NORMALIZER`.

3. **Delete the diversity-bonus block in `_evaluate` (lines 207–223)**:

```python
# DELETE these lines:
# Diversity bonus on utilization: count distinct piece KINDS in the
# useful components. Linear scaling from 1 kind (factor=1.0) to all
# kinds (factor=1+DIVERSITY_BONUS).
useful_kinds: set = set()
for slot in useful_slots:
    piece_id = graph.slot_pieces.get(slot)
    if piece_id is None:
        continue
    ps = self.spec.by_id.get(piece_id)
    if ps is not None and ps.kind in DIVERSITY_KINDS:
        useful_kinds.add(ps.kind)
max_kinds = len(DIVERSITY_KINDS)
if useful_kinds and max_kinds > 1:
    kind_frac = (len(useful_kinds) - 1) / (max_kinds - 1)
else:
    kind_frac = 0.0
diversity_factor = 1.0 + DIVERSITY_BONUS * kind_frac
```

4. **Change line 225** from:

```python
out["F"] = np.array([-utilization * diversity_factor, -min_speed])
```

to:

```python
out["F"] = np.array([-utilization, -min_speed])
```

### 5.3 Verification

- Run `with_switches` from the web UI (set `crossover_prob=0.0` in the tweaks panel — that's where this knob lives, NOT in YAML).
- Expect: feasibility ratio jumps from 1/500 → ≥ 0.95 of pop (matches memory baseline). Best feasible should now have 4–6 switches (the GA can preserve them once crossover isn't shredding port-pair rows).
- HV is now measurable cleanly: `F[0] ∈ [-1, 0]` is unbiased utilization.

### 5.4 No tests to add yet

Phase 1 just removes a feature; the existing `_evaluate` shape is unchanged. No unit-test churn.

---

## 6. Phase 2 — Repair completeness + constraint reformulation

### 6.1 Why

Switches with port C unpaired survive as decorative pieces. They never form branch cycles, get visualized as bare lines, and inflate switch counts dishonestly. Per user decision: tolerate them throughout exploration; *attempt* completion only at the **finalization phase** (`t > 0.9`); commit on success, revert on failure (next generation retries).

DOUBLE_CROSSOVER routes are mutually exclusive (`{A↔B, C↔D}` parallel-through OR `{A↔D, C↔B}` crossover-engaged) — the union case `{A↔B, C↔D, A↔D, C↔B}` port-double-books and is physically invalid (real 4DBrix part 210.1 is a route selector, not a 4-route piece).

### 6.2 New file: `src_v2/finalization_callback.py`

```python
"""Gating callback that toggles repair.finalization_active on the repair pipeline.

When ``algorithm.termination.perc > 0.9``, the repair pipeline begins
attempting branch completion on incomplete switches. Below 0.9, incomplete
switches are tolerated as soft-penalized intermediate states.

This stagger avoids a non-stationary feasibility shock during the
``LegoAdaptiveEpsilon`` relaxation window (t ∈ [0.2, 0.9]).
"""
from __future__ import annotations

from pymoo.core.callback import Callback


class FinalizationGatingCallback(Callback):
    """Sets ``repair.finalization_active`` based on termination progress."""

    def __init__(self, repair_pipeline, threshold: float = 0.9) -> None:
        super().__init__()
        self._repair = repair_pipeline
        self._threshold = threshold

    def notify(self, algorithm) -> None:
        perc = getattr(algorithm.termination, "perc", 0.0) or 0.0
        self._repair.finalization_active = perc > self._threshold
```

### 6.3 Modify [src_v2/repair.py](src_v2/repair.py)

**6.3.1 Add the `finalization_active` attribute** to `PortPairRepairPipeline.__init__`:

```python
self.finalization_active: bool = False  # Toggled by FinalizationGatingCallback
```

**6.3.2 Add `_enforce_route_completeness` pass** (called between `_sanitize_edges` and `_enforce_inventory` in the existing fixed-point loop):

```python
def _enforce_route_completeness(self, x: np.ndarray) -> bool:
    """Bond-graph completeness pass.

    During exploration (``self.finalization_active == False``): no-op apart
    from validating that crossing pair-sets match catalog routes (drops
    pair-rows that don't form a valid route).

    During finalization (``self.finalization_active == True``): for each
    incomplete switch, *attempt* to grow a closing branch (Section A.1 of
    plan). On success commit; on failure revert (leave incomplete; next
    generation retries — that's the "try in the next turn" semantics).

    Returns True iff x changed.
    """
    changed = False
    # See spec in plan Section A.1–A.2 for the per-kind handler registry.
    # Use the Mapping[str, tuple[Callable, Callable]] (check, repair) registry
    # pattern, NOT an if/elif chain.
    handlers = self._completeness_handlers()
    for slot_idx, piece_index in iter_active_slots(x, self.dims):
        piece_id = self.catalog.index_to_id[piece_index]
        ps = self.spec.by_id[piece_id]
        check, repair = handlers.get(ps.kind, (None, None))
        if check is None:
            continue  # straight/curve already handled by edge sanitization
        if not check(x, slot_idx, ps):
            continue  # already complete
        if self.finalization_active and repair is not None:
            if repair(x, slot_idx, ps):
                changed = True
    # Always validate crossing pair-sets even during exploration.
    if self._validate_crossing_pair_sets(x):
        changed = True
    return changed


def _completeness_handlers(self):
    """Registry: kind → (is_incomplete_fn, attempt_repair_fn).

    Avoids if/elif chains. Functions accept (chromosome, slot_idx, piece_spec)
    and return bool. ``attempt_repair_fn`` returns True iff x mutated.
    """
    return {
        "switch": (_switch_is_incomplete, self._attempt_switch_completion),
        "crossing": (_crossing_is_incomplete, self._attempt_crossing_completion),
    }
```

**6.3.3 Implement `_attempt_switch_completion`** — reads as: try Phase 3's `mutate_grow_branch`; revert on failure:

```python
def _attempt_switch_completion(self, x, slot_idx, piece_spec) -> bool:
    """Try to close port C of this switch via branch growth.

    Calls :func:`src_v2.branch_grow.find_branch_path` with a target switch
    on the same `through` cycle. On success, install pieces and edges; on
    failure, leave the switch incomplete (G[5+T] soft penalty applies).
    """
    from .branch_grow import find_branch_path  # avoid import cycle

    # Decode current state to find candidate target switches.
    graph = decode_chromosome(x, self.dims, self.catalog, self.decoder_config)
    # ... pick (s_start, t) on same through cycle, both with C unpaired ...
    # ... budget = self._branch_local_budget(graph, max_branch_pieces=16) ...
    # ... result = find_branch_path(start_pose, target_pose, budget,
    #                                self.catalog, max_depth=16, tolerance=8.0,
    #                                rng=self.rng) ...
    # if result is None: return False  # revert
    # else: install pieces + edges, decrement inventory, return True
    raise NotImplementedError("Implement after Phase 3 ships branch_grow.py")
```

**6.3.4 Implement `_attempt_crossing_completion`** — analogous; tries to close the second route through ports C↔D when only A↔B is paired (rare success; symmetric with switch path).

**6.3.5 Validate crossing pair-sets in `_sanitize_edges`** — extend the existing pass to drop pair-rows on `kind == "crossing"` slots whose port pair doesn't appear in `piece_spec.routes`. Kills `{A↔C}` on CROSS_90 (no such route); kills the union case on DOUBLE_CROSSOVER (4 ports paired but more pair-rows than the 2 valid pair-sets allow).

### 6.4 Modify [src_v2/problem.py](src_v2/problem.py) — constraint reformulation

**6.4.1 Update the constraints docstring** (after Phase 1's docstring revision) to reflect the new layout:

```
- ``G[0..2]`` — per-axis closure residual / tolerance.
- ``G[3]`` — boundary violation.
- ``G[4]`` — uncovered intersections (collisions not mediated by crossing).
- ``G[5..4+T]`` — per-type inventory excess.
- ``G[5+T]`` — incomplete switches / total switches (route-completeness).
- ``G[6+T]`` — incomplete crossings / total crossings (route-aligned check).
- ``G[7+T]`` — 1 - n_cycles (require ≥ 1 closed cycle).
- ``G[8+T]`` — (min_branch_count - n_branch_cycles) / max(1, min_branch_count).

Total: ``9 + T`` inequality constraints. (T = catalog.n_pieces.)
```

**6.4.2 Change line 164** from `n_constr = 7 + catalog.n_pieces` to `n_constr = 9 + catalog.n_pieces`.

**6.4.3 Replace the constraint block (lines 261–270)** with:

```python
# Route-completeness ratios (audit Recommendation: replaces generic
# loose-port aggregate). Soft-penalized during warmup via ε-relaxation;
# only fully active post-warmup when finalization_active flips on.
total_switches = sum(
    1 for pid in graph.slot_pieces.values()
    if pid in self.spec.by_id and self.spec.by_id[pid].kind == "switch"
)
incomplete_switches = self._count_incomplete_switches(graph)
G.append(incomplete_switches / max(1, total_switches))

total_crossings = sum(
    1 for pid in graph.slot_pieces.values()
    if pid in self.spec.by_id and self.spec.by_id[pid].kind == "crossing"
)
incomplete_crossings = self._count_incomplete_crossings(graph)
G.append(incomplete_crossings / max(1, total_crossings))

# Cycle presence (renumbered from previous G[6+T])
G.append(1.0 - graph.n_cycles)

# Branch-cycle target (always emitted; YAML-absent → min_branch_count = 0
# → constraint always 0.0 → no pressure).
min_branch_count = getattr(self.config, "min_branch_count", 0) or 0
n_branch_cycles = self._count_branch_cycles(graph)
G.append(
    (min_branch_count - n_branch_cycles) / max(1, min_branch_count)
    if min_branch_count > 0 else 0.0
)
```

**6.4.4 Add helper methods** to `PortPairProblem`:

```python
def _count_incomplete_switches(self, graph) -> int:
    """Switches whose paired ports != {A, B, C}."""
    incomplete = 0
    used = self._port_use_set(graph)
    for slot, pid in graph.slot_pieces.items():
        ps = self.spec.by_id.get(pid)
        if ps is None or ps.kind != "switch":
            continue
        paired = {p for (s, p) in used if s == slot}
        if paired != set(ps.ports.keys()):
            incomplete += 1
    return incomplete

def _count_incomplete_crossings(self, graph) -> int:
    """Crossings whose pair-set isn't route-aligned or fully populated."""
    # See plan Section A row 4 for valid pair-sets per piece_id.
    ...

def _count_branch_cycles(self, graph) -> int:
    """Cycles containing ≥ 1 (slot, "diverging") in branch_labels."""
    diverging_cycle_ids = {
        cid for (_s, route), cid in graph.branch_labels.items()
        if route in BRANCH_ROUTES
    }
    return len(diverging_cycle_ids)

def _port_use_set(self, graph):
    """Set of (slot, port_name) pairs that appear in any active edge."""
    used = set()
    for edge in graph.edges:
        used.add((edge.slot_a, edge.port_a))
        used.add((edge.slot_b, edge.port_b))
    return used
```

### 6.5 Modify [src_v2/runner.py](src_v2/runner.py)

**6.5.1 Wire `FinalizationGatingCallback` first in the `CallbackChain`**:

```python
from .finalization_callback import FinalizationGatingCallback
# ...
callbacks = CallbackChain([
    FinalizationGatingCallback(repair_pipeline, threshold=0.9),  # FIRST
    ProgressLoggerCallback(...),                                  # existing
    FeasibleEliteCallback(...),                                   # existing — retire in Phase 5
])
```

**6.5.2 Update the CSV header at lines 308–318** atomically with the new `n_constr`. Replace `loose_ports,cycle_count` with `incomplete_switch_ratio,incomplete_crossing_ratio,cycle_count,branch_cycle_deficit`.

### 6.6 Tests to add

`tests/test_repair_completeness.py`:

- `test_switch_completion_during_exploration_no_op`: `finalization_active = False` → incomplete switch survives.
- `test_switch_completion_during_finalization_attempts_grow`: `finalization_active = True` → A* called.
- `test_switch_completion_revert_on_failure`: A* returns None → x byte-identical.
- `test_double_crossover_union_case_rejected`: chromosome with all 4 routes paired → drops to one valid pair-set.
- `test_cross90_invalid_pair_set_dropped`: chromosome with `{A↔C}` only on CROSS_90 → edge dropped.

`tests/test_constraint_reformulation.py`:

- `test_n_constr_is_9_plus_T`.
- `test_incomplete_switch_ratio_zero_for_complete_layout`.
- `test_incomplete_switch_ratio_one_for_all_dangling`.
- `test_branch_cycle_deficit_zero_when_min_branch_unset`.

Run: `.venv\Scripts\python.exe -m pytest tests/test_repair_completeness.py tests/test_constraint_reformulation.py -v --tb=short`.

### 6.7 Verification

Run `with_switches` from web UI with `crossover_prob=0.0`. Expect:

- Feasibility ratio ≥ 0.5 (was already ~0.95 from Phase 1; completeness gating shouldn't regress it).
- `incomplete_switch_ratio` and `incomplete_crossing_ratio` columns appear in `constraints.csv`.
- During the last 10% of generations (`t > 0.9`), best feasible should start showing route-complete switches as A* succeeds.

---

## 7. Phase 3 — Branch growth (correctness foundation)

### 7.1 Why

Current `_introduce_switch_pair` ([src_v2/structural_mutations.py:130–323](src_v2/structural_mutations.py)) hardcodes 2 R40_CURVE for the branch and never verifies angular-budget compatibility between IN and OUT. Memory issue #1: branches don't close geometrically.

Replace with angular-budget-aware A* search over inventory.

### 7.2 New file: `src_v2/branch_grow.py`

```python
"""Angular-budget-aware A* search for closing a switch's port C back to
another switch's port C on the same `through` cycle.

Cellular-Encoding-style branch growth grammar with closure verification.
A* state space: append one of {STRAIGHT_16, R40_CURVE flip=0, R40_CURVE flip=1}.
Heuristic: euclidean(pose, target) / 16 (admissible, monotone).
Closed set keyed on (rounded_pose, sorted_inventory_tuple, depth).

Pure function — no chromosome mutation. The mutation operator wraps it.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from .catalog import TrackCatalog
from .se2 import pose_compose

# Branch-step gene tuple: piece_id, flip bit, rotate bit. Hashable for closed set.
class BranchStep(NamedTuple):
    piece_id: str
    flip: int
    rotate: int


# Internal A* state. Hashable for closed set; ordered by (f, tiebreak) for heapq.
class _State(NamedTuple):
    f: float        # g + h
    g: int          # depth (= pieces placed so far)
    pose: tuple     # (x, y, theta) — quantized for hashing
    inventory: tuple  # sorted ((piece_id, count), ...) — hashable
    path: tuple     # tuple of BranchStep so far


def find_branch_path(
    start_pose: tuple[float, float, float],
    target_pose: tuple[float, float, float],
    inventory: Mapping[str, int],
    catalog: TrackCatalog,
    *,
    max_depth: int,
    tolerance: float,
    rng: np.random.Generator,
) -> list[BranchStep] | None:
    """Find a sequence of pieces whose placed end-pose ≈ target_pose.

    Args:
        start_pose: pose of the start port (in world frame).
        target_pose: pose where the chain must end (port_C of OUT switch,
            inverted so the branch's outgoing direction mates with it).
        inventory: piece-id → remaining count budget.
        catalog: TrackCatalog (used for piece port offsets).
        max_depth: hard cap on branch piece count (typically 16).
        tolerance: stud distance under which goal is considered reached.
        rng: for tie-breaking on equal-f successors.

    Returns:
        List of BranchSteps, or None if no closure found within budget.
    """
    succ_specs = _build_successor_specs(catalog)  # 3 successors with FK deltas

    start_state = _State(
        f=_h(start_pose, target_pose),
        g=0,
        pose=_quantize_pose(start_pose),
        inventory=_sorted_inv(inventory),
        path=(),
    )
    heap = [(start_state.f, rng.random(), start_state)]  # (f, tiebreak, state)
    closed: set[tuple] = set()

    while heap:
        _, _, state = heapq.heappop(heap)
        key = (state.pose, state.inventory, state.g)
        if key in closed:
            continue
        closed.add(key)

        # Goal test: euclidean from state.pose to target_pose ≤ tolerance.
        if _euclidean(state.pose, target_pose) <= tolerance:
            return list(state.path)

        if state.g >= max_depth:
            continue

        for spec in succ_specs:
            if state.inventory_dict.get(spec.piece_id, 0) <= 0:
                continue
            new_pose = pose_compose(state.pose, spec.delta)
            new_inv = _decrement_inv(state.inventory, spec.piece_id)
            new_path = state.path + (BranchStep(spec.piece_id, spec.flip, spec.rotate),)
            new_g = state.g + 1
            new_h = _h(new_pose, target_pose)
            new_state = _State(
                f=new_g + new_h,
                g=new_g,
                pose=_quantize_pose(new_pose),
                inventory=new_inv,
                path=new_path,
            )
            new_key = (new_state.pose, new_state.inventory, new_state.g)
            if new_key not in closed:
                heapq.heappush(heap, (new_state.f, rng.random(), new_state))
    return None


# --- helpers ---

def _h(pose, target) -> float:
    """Admissible heuristic: euclidean / 16 (one straight closes 16 studs)."""
    return _euclidean(pose, target) / 16.0


def _euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _quantize_pose(pose) -> tuple:
    """Round to quarter-stud precision so float poses become hashable."""
    return (round(pose[0] * 4) / 4, round(pose[1] * 4) / 4, round(pose[2], 4))


def _sorted_inv(inv: Mapping[str, int]) -> tuple:
    return tuple(sorted((pid, c) for pid, c in inv.items() if c > 0))


def _decrement_inv(inv_tuple: tuple, piece_id: str) -> tuple:
    return tuple(
        (pid, c - 1) if pid == piece_id else (pid, c)
        for pid, c in inv_tuple
        if not (pid == piece_id and c <= 1)
    )


def _build_successor_specs(catalog: TrackCatalog):
    """Return a list of successor specs: (piece_id, flip, rotate, FK delta).

    Three successors: STRAIGHT_16, R40_CURVE flip=0, R40_CURVE flip=1.
    Each spec provides the SE(2) delta from input port A to output port B
    in the piece-local frame.
    """
    # Implementation: read piece spec from catalog, compute port_B - port_A
    # delta with the flip applied.
    raise NotImplementedError("Compute FK deltas from catalog.spec.by_id[...]")
```

> **Note**: `_build_successor_specs` and `_State.inventory_dict` (a memoized property) need fleshing out. The skeleton shows the algorithm; the catalog FK lookup is straightforward.

### 7.3 Modify [src_v2/operators.py](src_v2/operators.py) — register two new sub-operators in `OP_WEIGHTS`

```python
OP_WEIGHTS = {
    # ... existing entries ...
    "grow_branch": 0.05,
    "insert_switch_with_branch": 0.05,
}
```

And add the dispatch hooks in `PortPairMutation._do`:

```python
elif op == "grow_branch":
    success = self._mutate_grow_branch(x_copy)
elif op == "insert_switch_with_branch":
    success = self._mutate_insert_switch_with_branch(x_copy)
```

### 7.4 Modify [src_v2/structural_mutations.py](src_v2/structural_mutations.py)

**7.4.1 Add `mutate_grow_branch` and `mutate_insert_switch_with_branch`** — wrappers around `find_branch_path`:

```python
def mutate_grow_branch(x, dims, catalog, inventory, rng) -> bool:
    """Pick an incomplete switch; A* search for a closing branch.

    Commit-on-success with rollback (caller passes ``x`` in-place; we mutate
    a shadow copy and only write back if A* succeeds).
    """
    x_shadow = x.copy()
    # ... decode, pick s_start with port C unpaired ...
    # ... find candidate t on same through cycle ...
    # ... compute start_pose, target_pose ...
    # ... call find_branch_path(...) ...
    # ... if None: return False (no commit) ...
    # ... else: install pieces, wire pair rows, commit ...
    raise NotImplementedError


def mutate_insert_switch_with_branch(x, dims, catalog, inventory, rng) -> bool:
    """Replace a STRAIGHT_16-STRAIGHT_16 edge with switch_in + branch + switch_out.

    Uses the existing IN/OUT placement scaffolding from _introduce_switch_pair
    (lines 202-308) up through chain rerouting. Then forces s_start = in_switch,
    t = out_switch and calls mutate_grow_branch logic. Rolls back the IN/OUT
    placement on A* failure.
    """
    raise NotImplementedError
```

**7.4.2 Replace the hardcoded branch-stitching at lines 310–321 of `_introduce_switch_pair`** with a call to the angular-budget A*. The IN/OUT placement scaffolding stays.

### 7.5 Tests

`tests/test_branch_grow.py`:

- `test_trivial_closure_no_pieces`: target == start → empty path.
- `test_single_straight_closure`: target = start + (16, 0, 0) → one STRAIGHT_16.
- `test_two_curve_passing_siding`: closes a 6-stud-y-residual passing siding with 2 curves.
- `test_inventory_exhausted_returns_none`: inventory = {} → None.
- `test_max_depth_respected`: target requires 17 pieces with `max_depth=16` → None.

Run: `.venv\Scripts\python.exe -m pytest tests/test_branch_grow.py -v`.

### 7.6 Verification

Web UI run of `with_switches` (still `crossover_prob=0.0`, with Phases 1–2 in place):

- Best feasible at end of run should now have ≥ 1 closed branch cycle.
- `incomplete_switch_ratio` in best feasible should be ≤ 0.05.
- `n_cycles ≥ 2` regularly (mainline + at least one branch).

---

## 8. Phase 4 — Diversification mutations + improved seeding + Lamarckian closure repair

### 8.1 Mutations to add (in [src_v2/structural_mutations.py](src_v2/structural_mutations.py))

All commit-on-success with rollback. All registered in `OP_WEIGHTS`.

**E.3 `mutate_swap_switch_hand`** — toggle `R40_SWITCH_LEFT ↔ R40_SWITCH_RIGHT`. After the slot rewrite, BFS-flip the `flip` bit on every R40_CURVE along the branch. Then **chain into `mutate_grow_branch`** with the OUT switch fixed to verify closure post-mirror; rollback on A* failure.

```python
def mutate_swap_switch_hand(x, dims, catalog, inventory, rng) -> bool:
    x_shadow = x.copy()
    # 1. Pick a switch slot s. Inventory check for the opposite hand.
    # 2. Rewrite slot s's piece to opposite hand.
    # 3. BFS from (s, C) along edges; flip the flip bit on every R40_CURVE
    #    encountered; stop at the OUT switch without flipping it.
    # 4. Call mutate_grow_branch with s_start = s, t = OUT_switch forced.
    # 5. On A* failure, rollback to original x (don't write x_shadow back).
    raise NotImplementedError
```

**E.4 `mutate_split_loop_with_crossing`** — pick a cycle with ≥ 6 slots; pick two non-adjacent edges with mid-segment poses ≤ 32 studs apart and ≈ 90° relative angle; place CROSS_90 at midpoint; wire e1 through ports A↔B, e2 through C↔D. Distinct from existing `introduce_crossing` ([src_v2/structural_mutations.py:46–127](src_v2/structural_mutations.py)) which only resolves *existing* perpendicular self-intersections.

**E.5 `mutate_branch_extend` / `mutate_branch_shrink`** — local edits on existing branches:

```python
def mutate_branch_extend(x, dims, catalog, inventory, rng) -> bool:
    """Pick a non-switch edge on a branch cycle; insert a STRAIGHT_16 in-line."""
    raise NotImplementedError

def mutate_branch_shrink(x, dims, catalog, inventory, rng) -> bool:
    """Pick a STRAIGHT_16 on a branch cycle; drop it and splice adjacent edges."""
    raise NotImplementedError
```

**E.6 `mutate_reverse_switch_pairing`** — swap `(C, C)` → `(C, A)`, `(B, C)` etc. on a switch pair. Memory issue #3: enables OUT→IN reverse sidings. Delegates re-growth to `mutate_grow_branch`.

**E.7 `mutate_closure_repair_lamarckian`** — memetic local search at 5% offspring probability (Orvosh & Davis 1993; Neri & Cotta 2012). Invoked only when `closure_gap > τ`:

```python
def mutate_closure_repair_lamarckian(x, dims, catalog, inventory, rng,
                                      *, gap_threshold: float = 4.0) -> bool:
    """Window-based piece replacement on a failing cycle.

    1. Decode → graph. Compute closure_gap per cycle.
    2. If max(closure_gap) ≤ gap_threshold: return False.
    3. Pick the failing cycle. Pick a random length-K=3 window of slots on it.
    4. For each window-substitution candidate (try a few thousand):
       - Substitute the K pieces with K alternates from inventory.
       - Recompute closure_gap.
       - If improved: keep substitution, commit.
    5. If no improvement found in budget: return False (revert).
    """
    raise NotImplementedError
```

CLLP loop-repair lineage (Cheng, Gen & Tosawa 1996): the window-substitution idiom is standard for closed-loop layouts.

### 8.2 Improved seeding (in [src_v2/operators.py](src_v2/operators.py))

**8.2.1 Resurrect `_emit_simple_oval_with_siding`** — currently disabled at lines 421–429 with a comment about R40+STRAIGHT_16 sidings not closing exactly. With Phase 3's `mutate_grow_branch` adapting branch length post-seed, this restriction lifts. Re-enable.

**8.2.2 Add new emitters**:

- `_emit_dog_bone` — two 8-curve end-caps + long straight middle. ~28 pieces, no switches.
- `_emit_dumbbell` — two ovals connected by a straight chord with one CROSS_90 dividing it. ~36 pieces, 2 cycles.
- `_emit_two_loops_with_figure8_2crossings` — clean stadium with 2 CROSS_90 at opposite ends.
- `_emit_parallel_tracks_with_crossover` — gated on `inventory["DOUBLE_CROSSOVER"] ≥ 1`; **silently skip** when absent.

### 8.3 Operator dispatch performance

Replace `np.random.choice(len(ops), p=self.OP_WEIGHTS)` per individual at [src_v2/operators.py:739](src_v2/operators.py) with **CDF + `np.searchsorted`**, recomputed only when ALNS updates weights:

```python
def _build_cdf(self) -> None:
    """Cache CDF for fast operator dispatch."""
    weights = np.array(list(self.OP_WEIGHTS.values()), dtype=np.float64)
    self._cdf = np.cumsum(weights / weights.sum())

def _pick_op(self, rng) -> str:
    """O(log N) operator pick via searchsorted."""
    u = rng.random()
    idx = int(np.searchsorted(self._cdf, u, side="right"))
    return list(self.OP_WEIGHTS.keys())[idx]
```

### 8.4 RNG seeding

Replace global `np.random` calls (offenders at [src_v2/operators.py:497, 845](src_v2/operators.py)) with `self.rng = np.random.default_rng(seed)` in `PortPairMutation.__init__` and propagate `seed` from config. Apply to all sub-operators.

### 8.5 Tests

`tests/test_diversification_mutations.py` (one test per mutation: success path + rollback path).
`tests/test_lamarckian_closure_repair.py` (window substitution improves closure; no improvement → revert).

### 8.6 Verification

`with_switches` run from web UI:

- Best feasible should show ≥ 2 distinct branch cycles regularly (E.3, E.4 contribute).
- E.7 firing rate ≈ 5% on individuals with closure_gap > 4 studs.

---

## 9. Phase 5 — ALNS + ε-archive + phenotype dedupe + retire `FeasibleEliteCallback`

### 9.1 New file: `src_v2/alns_callback.py`

```python
"""Adaptive Large Neighborhood Search callback for operator weight tuning.

Pisinger & Ropke 2007; Li, Fialho, Kwong & Zhang 2014 (FRRMAB).

Tracks per-operator success rate via mutation._last_op_indices and
algorithm.off CV. Reweights every K=40 generations.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from pymoo.core.callback import Callback


class ALNSCallback(Callback):
    LAMBDA = 0.4
    EXPLORATION_FLOOR = 0.02
    K = 40  # reweight cadence (generations)

    def __init__(self) -> None:
        super().__init__()
        self._mutation = None
        self._reward_sum: dict[str, float] = defaultdict(float)
        self._call_count: dict[str, int] = defaultdict(int)
        self._gen = 0

    def attach_to(self, mutation) -> None:
        """Store reference to the mutation instance so we can read
        ``mutation._last_op_indices`` and write ``mutation.OP_WEIGHTS``.
        """
        self._mutation = mutation

    def notify(self, algorithm) -> None:
        if self._mutation is None or algorithm.off is None:
            return

        last_ops = list(self._mutation._last_op_indices or ())
        cv = algorithm.off.get("CV").flatten()  # (n_off,)

        for op_idx, cv_val in zip(last_ops, cv):
            op_name = list(self._mutation.OP_WEIGHTS.keys())[op_idx]
            self._call_count[op_name] += 1
            # Reward: feasibility (binary) + small inverse-CV continuous term.
            reward = 1.0 if cv_val <= 0 else 1.0 / (1.0 + cv_val)
            self._reward_sum[op_name] += reward

        self._gen += 1
        if self._gen % self.K == 0:
            self._reweight()

    def _reweight(self) -> None:
        weights_old = self._mutation.OP_WEIGHTS
        rewards = {
            name: self._reward_sum[name] / max(1, self._call_count[name])
            for name in weights_old
        }
        total = sum(rewards.values()) or 1.0
        norm_rewards = {n: r / total for n, r in rewards.items()}
        weights_new = {
            n: max(self.EXPLORATION_FLOOR,
                   (1 - self.LAMBDA) * weights_old[n]
                   + self.LAMBDA * norm_rewards[n])
            for n in weights_old
        }
        self._mutation.OP_WEIGHTS = weights_new
        self._mutation._build_cdf()  # re-cache CDF
        self._reward_sum.clear()
        self._call_count.clear()
```

### 9.2 New file: `src_v2/epsilon_archive.py`

```python
"""External ε-archive (Laumanns, Thiele, Deb & Zitzler 2002).

Pure-data ``EpsilonArchive`` + thin ``EpsilonArchiveCallback`` glue.
Internal storage: numpy 2D ``_F`` matrix + parallel ``_X`` list.
Vectorized box-coordinate ε-dominance for O(N) admission test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymoo.core.callback import Callback


class EpsilonArchive:
    """Bounded, well-spread ε-non-dominated set.

    Each candidate ``F = (F0, F1)`` admitted iff no archive box ε-dominates
    it; on admission, drop archive entries that the new candidate's box
    ε-dominates.
    """

    def __init__(self, epsilon: tuple[float, float], max_size: int = 200):
        self._eps = np.array(epsilon, dtype=np.float64)
        self._max_size = max_size
        self._F = np.empty((0, 2), dtype=np.float64)
        self._X: list[np.ndarray] = []

    def admit(self, X_row: np.ndarray, F_row: np.ndarray) -> bool:
        """Try to admit. Returns True iff admitted."""
        if self._F.shape[0] == 0:
            self._F = F_row.reshape(1, 2).copy()
            self._X = [X_row.copy()]
            return True

        new_box = np.floor(F_row / self._eps).astype(np.int64)
        old_boxes = np.floor(self._F / self._eps).astype(np.int64)

        # Existing entry ε-dominates new? (all dims ≤ AND any dim <)
        dominates_new = np.all(old_boxes <= new_box, axis=1) & np.any(
            old_boxes < new_box, axis=1
        )
        if np.any(dominates_new):
            return False

        # New entry ε-dominates existing? (all dims ≤ AND any dim <)
        new_dominates = np.all(new_box <= old_boxes, axis=1) & np.any(
            new_box < old_boxes, axis=1
        )
        keep_mask = ~new_dominates
        self._F = np.vstack([self._F[keep_mask], F_row.reshape(1, 2)])
        self._X = [self._X[i] for i, k in enumerate(keep_mask) if k] + [X_row.copy()]

        # Truncate to max_size by dropping least-isolated members.
        if self._F.shape[0] > self._max_size:
            self._truncate_to_max_size()
        return True

    def _truncate_to_max_size(self) -> None:
        # Drop the entry with smallest minimum-distance-to-any-other.
        # ... implementation ...
        raise NotImplementedError

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({
            "F": self._F.tolist(),
            "X": [x.tolist() for x in self._X],
            "epsilon": self._eps.tolist(),
        }, indent=2))


class EpsilonArchiveCallback(Callback):
    def __init__(self, epsilon: tuple[float, float], max_size: int,
                 output_path: Path):
        super().__init__()
        self._archive = EpsilonArchive(epsilon, max_size)
        self._output_path = output_path

    def notify(self, algorithm) -> None:
        opt = algorithm.opt
        if opt is None or len(opt) == 0:
            return
        for ind in opt:
            if ind.feasible[0]:
                self._archive.admit(ind.X, ind.F)

    def finalize(self) -> None:
        self._archive.to_json(self._output_path)
```

### 9.3 New file: `src_v2/phenotype_dedupe.py`

```python
"""Phenotype-level deduplication callback (Hildebrandt & Branke 2015;
Goldman & Punch 2014 P3).

Hash population by structural-summary tuple; replace duplicates with
random offspring at start of next generation's sampling.

Phenotype is computed during ``problem._evaluate``'s decode and stored
on the individual (avoids 500k duplicate decodes).
"""
from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

from pymoo.core.callback import Callback


class Phenotype(NamedTuple):
    n_switches: int
    n_crossings: int
    n_cycles: int
    n_branch_cycles: int
    max_component_size: int
    piece_histogram: tuple[tuple[str, int], ...]


class PhenotypeDedupeCallback(Callback):
    """Replace phenotype-duplicates with random offspring next generation."""

    def __init__(self, sampling) -> None:
        super().__init__()
        self._sampling = sampling

    def notify(self, algorithm) -> None:
        pop = algorithm.pop
        if pop is None:
            return
        buckets: dict[Phenotype, list[int]] = defaultdict(list)
        for i, ind in enumerate(pop):
            phen = ind.data.get("phenotype")
            if phen is not None:
                buckets[phen].append(i)
        # For each bucket with >1, retain best CV; mark others for replacement.
        replace_indices = []
        for indices in buckets.values():
            if len(indices) <= 1:
                continue
            best = min(indices, key=lambda i: pop[i].CV[0])
            replace_indices.extend(j for j in indices if j != best)
        if not replace_indices:
            return
        # Generate fresh offspring via sampling; pymoo will re-evaluate next gen.
        new_X = self._sampling.do(algorithm.problem, len(replace_indices))
        for slot, j in enumerate(replace_indices):
            pop[j].X = new_X[slot]
            # CV/F invalidated; pymoo's next eval cycle handles it.
```

Phenotype piggyback: in [src_v2/problem.py](src_v2/problem.py) `_evaluate`, after decoding `graph`, compute the `Phenotype` tuple from the same `graph` and store it via `out["data"] = {"phenotype": phen}` (or a per-individual data dict per pymoo's API).

### 9.4 Retire `FeasibleEliteCallback`

In [src_v2/runner.py](src_v2/runner.py) — remove the entry from `CallbackChain`. The class definition can stay in the file (tagged `# DEPRECATED — replaced by EpsilonArchiveCallback in Phase 5`) for one release cycle, then deleted.

### 9.5 Wire everything in [src_v2/runner.py](src_v2/runner.py)

```python
from .alns_callback import ALNSCallback
from .epsilon_archive import EpsilonArchiveCallback
from .phenotype_dedupe import PhenotypeDedupeCallback

# ... in run_optimization ...
mutation = PortPairMutation(...)
sampling = PortPairSampling(...)

alns = ALNSCallback()
alns.attach_to(mutation)

eps_eps = config.epsilon_archive.epsilon if hasattr(config, "epsilon_archive") else (0.005, 0.01)
eps_max = config.epsilon_archive.max_size if hasattr(config, "epsilon_archive") else 200
epsilon_archive = EpsilonArchiveCallback(
    epsilon=eps_eps,
    max_size=eps_max,
    output_path=Path(f"outputs_v2/{config_name}/epsilon_archive.json"),
)

dedupe = PhenotypeDedupeCallback(sampling)

callbacks = CallbackChain([
    FinalizationGatingCallback(repair_pipeline, threshold=0.9),
    ProgressLoggerCallback(...),
    alns,
    epsilon_archive,
    dedupe,
    # FeasibleEliteCallback removed.
])
```

After `minimize` returns, call `epsilon_archive.finalize()` to write the archive JSON.

### 9.6 Tests

- `tests/test_alns_callback.py`: weights drift toward operators that produce feasibility gains; floor=0.02 always respected.
- `tests/test_epsilon_archive.py`: admission test correctness on hand-crafted F-pairs; truncation when `len > max_size`.
- `tests/test_phenotype_dedupe.py`: collision detection; replacement happens; non-collisions untouched.

### 9.7 Verification

`with_switches` run from web UI:

- Final hypervolume ≥ 110% of Phase 0 baseline (with Ishibuchi-spec'd ref point: per-config `nadir + 0.05`).
- `outputs_v2/with_switches/epsilon_archive.json` contains a bounded, well-spread Pareto set.
- ALNS log shows a few operators consistently above floor; no operator stuck at floor for 500 generations.

---

## 10. Cross-cutting concerns

### 10.1 Python style (per [CLAUDE.md](CLAUDE.md) and memory)

- Vectorized numpy. Early returns. Functional decomposition.
- No nested for-then-if helpers; use comprehensions, `defaultdict`, generators.
- No `import random` inside method bodies. Use `self.rng = np.random.default_rng(seed)` at class init.
- No `Any` in new files — only at YAML boundaries.
- No hardcoded `N_max`/`E_max` — always derive from inventory at runtime.

### 10.2 Commit-on-success with rollback

Every Phase 3–4 mutation operator:

```python
def mutate_xxx(x, dims, catalog, inventory, rng) -> bool:
    x_shadow = x.copy()
    # ... attempt mutation on x_shadow ...
    if success:
        x[:] = x_shadow  # commit
        return True
    return False  # x byte-identical to caller
```

### 10.3 Anti-patterns to avoid

- Don't edit YAML for runtime parameters (covered above).
- Don't add `else: pass` after `if` — use early return or guard clause.
- Don't use `Optional[X]` — prefer `X | None` (project requires Python 3.10+).
- Don't catch broad `Exception` to swallow errors. If A* fails, explicitly check the None return; don't try/except.

### 10.4 Verification protocol per phase

After each phase, run from web UI (or via venv CLI):

```cmd
.venv\Scripts\python.exe run_v2.py default
.venv\Scripts\python.exe run_v2.py with_switches
.venv\Scripts\python.exe run_v2.py with_crossing
```

(Remember: `crossover_prob=0.0` for switch/crossing configs is set in the **web UI tweaks panel**, not in YAML.)

For each, capture from `outputs_v2/<config>/`:

- `result.json` — feasibility ratio, best feasible structure.
- `constraints.csv` — column-by-column trajectory.
- `epsilon_archive.json` (Phase 5+) — final archive.
- The runner log's structural inventory line:

  ```
  INFO: Best feasible structure: switches=N, crossings=M, cycles=C, branch_cycles=B, components=K
  ```

Compare each phase's metrics against the Phase 0 baseline table in §4.

### 10.5 Run tests after every phase

```cmd
.venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

Do NOT use `--quick` or `-k <subset>` (per memory `feedback_no_quick_tests.md`).

---

## 11. Phase ordering and rollback

| Phase | Code change scope | Rollback if regression |
|---|---|---|
| 1 | `problem.py` (drop DIVERSITY_BONUS) | Re-add the kind_frac block; one revert. |
| 2 | `repair.py`, `problem.py`, `runner.py` callback chain, new `finalization_callback.py` | Set `repair.finalization_active = False` permanently → completeness rules are no-ops → behavior reverts to Phase 1. |
| 3 | New `branch_grow.py`, two new mutations in `structural_mutations.py`, `OP_WEIGHTS` entries | Set the two new operators' weights to 0 → never fire → behavior reverts to Phase 2. |
| 4 | E.3–E.7 mutations, new emitters, CDF dispatch, RNG seeding | Same: zero out the operators' weights. |
| 5 | ALNS, ε-archive, dedupe, retire FeasibleEliteCallback | Comment out the new callbacks; restore FeasibleEliteCallback in `CallbackChain`. |

The universal safety net: in the web UI, set `crossover_prob=0.0`. That alone matches the documented mutation-only baseline (memory `project_crossover_destroys_feasibility.md` — 500/500 feasible with 4–6 switches).

---

## 12. Reference: existing files to read before each phase

- Phase 1: [src_v2/problem.py](src_v2/problem.py).
- Phase 2: [src_v2/repair.py](src_v2/repair.py), [src_v2/problem.py](src_v2/problem.py), [src_v2/runner.py](src_v2/runner.py), [src_v2/decoder.py](src_v2/decoder.py) (for `branch_labels`), [src_v2/types.py](src_v2/types.py) (for `PortGraph`, `PortEdge`).
- Phase 3: [src_v2/structural_mutations.py](src_v2/structural_mutations.py), [src_v2/se2.py](src_v2/se2.py), [src_v2/catalog/specs.py](src_v2/catalog/specs.py), [data/track_pieces_v2.yaml](data/track_pieces_v2.yaml), [src_v2/encoding.py](src_v2/encoding.py) (gene accessors).
- Phase 4: [src_v2/operators.py](src_v2/operators.py) (emitters region: lines 86–437), [src_v2/intersection.py](src_v2/intersection.py).
- Phase 5: [src_v2/runner.py](src_v2/runner.py) (CallbackChain construction).

---

## 13. Open questions resolved by the user

1. **DOUBLE_CROSSOVER part 210.1** — confirmed real (https://www.4dbrix.com/products/train/2-04-tracks/set-210-1/). Kept in catalog; Phase 2 row 4 (mutually-exclusive route pair-sets) stands.
2. **Switch completeness** — tolerate during exploration; try-then-revert at finalization (`t > 0.9`). Phase 2 implements this.
3. **Speed-axis range** — narrow (0.97–1.57 m/s observed). Phase 5 uses ε[1] = 0.01.
4. **`FeasibleEliteCallback` retirement** — drop atomically in Phase 5 alongside ε-archive ship.

---

## 14. Future direction (NOT in current scope)

**BRKGA reframing** of the encoding (Gonçalves & Resende 2011 doi:10.1007/s10732-010-9143-1; Londe et al. 2025 doi:10.1016/j.ejor.2024.03.030). The slot+port-pair+anchor representation is structurally a BRKGA decoder over a fixed-length chromosome. Parameterised uniform crossover with ρ ≈ 0.7 would absorb validity through the decoder and resolve the "crossover destroys feasibility" empirical finding without disabling crossover entirely. Defer to a separate architectural plan.
