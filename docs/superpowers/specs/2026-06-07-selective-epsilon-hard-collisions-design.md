# Design: Selective epsilon — keep collisions (and inventory) hard

**Date:** 2026-06-07
**Status:** approved (design), pending implementation plan
**Area:** `src/algorithm/runner.py` (`LegoAdaptiveEpsilon`), `src/problem.py` (constraint index map)

## Problem

The optimizer gets permanently stuck on high-utilization **infeasible** champions
that **self-cross** and can never be made feasible. The recurring shape is a
redundant full **16×R40 = 360° circle** embedded in the main loop (a "utilization
pump") that crosses neighbouring track at **oblique angles** (22.5° multiples,
e.g. 67°, 23°) — angles no piece in the kit can legalize (`CROSS_90` is 90°-only;
`DOUBLE_CROSSOVER` is a parallel-track scissors). Measured: `best_infeas` holds
~69–76% util for hundreds of generations without ever converging; the structure
is closed (closure ≈ 0) but infeasible on the **collision** constraint.

### Root cause (measured, not assumed)

- `Individual.FEAS = (CV ≤ cv_eps)` where `CV = Σ_i max(0, G_i − eps_i)/scale_i`
  (pymoo 0.6.1.6, `core/individual.py`).
- `LegoAdaptiveEpsilon` sets only the **scalar** `cv_eps = alpha·max_cv`,
  `max_cv ≤ 30`; `cv_ieq.eps` stays 0.
- A closed self-crossing champion has tiny aggregate CV (e.g. boundary 0.02 +
  collisions 0.40 ≈ **0.42**) which is `≤ cv_eps` (up to 30) → treated **feasible**
  for selection. So util reward + epsilon shield let it dominate genuine
  feasibles, and it never has to become buildable.
- Not a seeding artifact: with the `_gen_simple_loop` circle seed fully removed,
  circles still grow from 6 → 1000/1000 of the population within 15 generations
  (operators manufacture them; selection amplifies them). So the lever is the
  **incentive/feasibility test**, applied every generation — not seeding, and not
  a pattern-specific repair.

## Goal

Stop self-intersecting (and over-inventory) individuals from being treated as
feasible, so selection drops them — **without**:
- disabling adaptive epsilon (confirmed worse without it),
- forbidding circles as a pattern (a standalone circle, collisions = 0, is a
  legitimate feasible loop and must remain reachable),
- changing the objective `F[0]` (utilization) or adding a special-case repair.

## Design

Split the inequality constraints into two classes and relax them differently:

| Class | Constraints | Treatment |
|-------|-------------|-----------|
| **SOFT** | closure `G[0:3]`, boundary `G[3]` | epsilon-relaxed (as today) — "almost buildable, evolution can close/shrink it" |
| **HARD** | collisions `G[4]`, per-type inventory `G[5:]` | **never** relaxed — structurally unbuildable |

### Mechanism (pymoo hook, no magic factors)

In `LegoAdaptiveEpsilon._adapt_constraint_handling`, instead of setting the scalar
`cv_eps`, set a **per-constraint** epsilon array on the managed config:

- `config["cv_ieq"]["eps"]` = array of length `n_ieq_constr`:
  scheduled epsilon (`alpha·max_cv`, possibly normalized — see Open question) on
  SOFT indices, **0** on HARD indices.
- `config["cv_eps"] = 0` (top-level FEAS threshold).

Then `FEAS = (Σ_i max(0, G_i − eps_i) ≤ 0)`: an individual is feasible iff its
closure/boundary are within epsilon **and** collisions/inventory are zero.

This uses pymoo's own per-constraint `eps` (broadcast element-wise in
`constr_to_cv`) — no arbitrary scale/weight constant.

### Constraint indices are derived, not hardcoded

`n_ieq_constr = 5 + catalog.n_pieces` (`problem.py`). SOFT = indices `{0,1,2,3}`
(3 closure + boundary); HARD = `{4, …, n_ieq_constr-1}` (collisions + per-type
inventory). The split is computed from the problem at setup, so it tracks
inventory size automatically (project invariant: no hardcoded constraint counts).

## Components / data flow

- `LegoAdaptiveEpsilon` gains knowledge of the SOFT index set (passed in or
  derived from `problem.n_ieq_constr`). `_adapt_constraint_handling` writes the
  per-constraint `eps` array each generation per the existing 3-phase schedule
  (hold → decay → strict). `_initialize_advance` (epsilon_0 from 10th-percentile
  CV, capped 30) is unchanged.
- Nothing else changes: objective, repair pipeline, operators, sampling, decoder
  all stay as-is.

## Testing

- **Unit (feasibility semantics):** build three layouts and assert FEAS under the
  adapted config:
  1. closed self-crossing "circle champion" (collisions > 0, closure ≈ 0) →
     `FEAS = False` even at full epsilon.
  2. near-closed clean siding (closure error within epsilon, collisions = 0) →
     `FEAS = True` (still relaxed).
  3. standalone feasible circle (collisions = 0) → `FEAS = True`.
- **Integration (~15 gens, seed-fixed):** population-with-circle and
  closed-self-crossing champions do **not** dominate; feasible-front util does not
  regress vs current behaviour; run still produces feasible solutions (epsilon
  benefit for closure preserved). Compare against a baseline run.

## Open question (resolve in implementation, verify integrationally)

Per-constraint `eps` changes the closure relaxation budget from **shared** (one
`cv_eps` across the summed CV) to **per-constraint** (each soft G_i gets its own
eps). Three soft constraints each receiving the full epsilon may over-relax
closure. Options: keep full epsilon per constraint, or normalize to `eps/n_soft`.
Pick by integration test — must not collapse the population the other way (back to
trivial feasibles only).

## Out of scope (explicitly)

- Seeding changes (proven not the source).
- Pattern-specific circle excision in repair (too specific; would also kill
  legitimate circles).
- Objective changes.
- The larger "serpentine / non-crossing space-filling for high util" effort — a
  separate follow-up once self-crossers stop dominating.

## Implementation note (2026-06-07, after first attempt)

The per-constraint `cv_ieq["eps"]` hook above was implemented and unit-tested, but
**crashed at runtime** (caught by the integration gate). Root cause: it makes
pymoo's `Individual.CV` the *relaxed* CV, and NSGA2's `binary_tournament` routes
feasible/infeasible by **raw** `CV[0] > 0` every generation; because the epsilon
schedule updates *after* survival, a relaxed individual could be infeasible at
survival (no crowding assigned) yet `CV==0` at the next tournament, reaching the
crowding comparison with `crowding=None` → `TypeError`.

**Shipped fix (same design intent, safe hook):** keep CV **schedule-independent**.
Restore the scalar `cv_eps = alpha·max_cv` (relaxes soft constraints via the FEAS
threshold, as originally) and instead weight the HARD constraints far above the
epsilon cap via `cv_ieq["scale"]` (`HARD_CONSTRAINT_WEIGHT = 1000`, scale
`1/1000` on collisions+inventory, `1` on closure+boundary). Then CV is a fixed
function of G, the tournament's raw-CV routing is unchanged, and any hard
violation (≥0.2) lands at ≥200 ≫ any `cv_eps` (≤30) so it is never relaxed.
Integration gate: self-crossing champions 0/1000 (was ~998), feasible 996/1000,
no crash. The "open question" (per-constraint vs shared budget) is moot — the
shipped fix keeps the original shared `cv_eps` budget on closure.

## References (memory)

`project_adaptive_epsilon_harmful`, `project_ga_crossings_are_oblique`,
`project_cross90_util_ceiling`, `project_boundary_util_ceiling`.
