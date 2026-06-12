# Design: Category elite archive — preserve & report element-bearing solutions

**Date:** 2026-06-11
**Status:** approved (user), implementation via TDD
**Area:** `src/problem.py`, `src/decoder/construction.py`, `src/types.py`,
`src/algorithm/runner.py`

## Problem

Solutions containing special elements (passing sidings / CROSS_90 / DOUBLE_
CROSSOVER) are bred out of the population within a few generations: with the
speed objective degenerate, utilization is the only effective selection
pressure, and element-bearing layouts are structurally lower-util than plain
racetracks (measured ceilings: figure-8-cross ~17%, DC weave ≤ ~61% vs
racetrack ~62%+). Seeds exist and validate — the failure is **retention**, not
generation. Result: every run reports `CROSS_90: 0/2`, `DC: 0/2`, and there is
no artifact showing how good an element-bearing solution *could* be, nor why
it lost.

## Goal (user-approved scope)

1. **Diversity floor:** keep the best **feasible** individual containing each
   element alive in the population (1 injected elite per category), so each
   category participates in mating and never goes extinct.
2. **Per-category reporting:** at run end, render and diagnose the best
   solution per category (`best_with_switch.png`, `best_with_cross.png`,
   `best_with_dc.png`) with computed explanations of the utilization gap and —
   when a category never materialized — the decoder's reasons for dropping its
   descriptors.

Categories: `switch` (n_switch_pairs > 0), `cross` (committed CROSS_90 > 0),
`dc` (committed DOUBLE_CROSSOVER > 0). No combination categories in v1.

## Design

### 1. Category capture — free, via pymoo custom out-keys (verified)

`TrackOptimizationProblem._evaluate` already decodes every chromosome. It
additionally writes:

```python
out["n_sw_pairs"] = len(layout.switch_pairs)
out["n_cross_comm"] = len(layout.cross_junctions)
out["n_dc_comm"] = len(layout.dbl_crossovers)
```

pymoo's Evaluator transports custom out-keys onto the Population
(`pop.get("n_sw_pairs")`) — **verified empirically for both sequential and
StarmapParallelization evaluation** on pymoo 0.6.1.6. This also captures
*emergent* CROSS_90 placements (decoder repair path) that are invisible in the
genotype. The empty-layout sentinel branch writes zeros.

### 2. `CategoryEliteArchive(Callback)` — capture + injection

Generalizes the proven `FeasibleEliteCallback` pattern:

- Per generation, for each category: among **feasible** individuals with
  element count > 0, find max-utilization; deep-copy into the archive when it
  beats the archived elite.
- **Injection (1 per category):** if the current population has no feasible
  member of the category with utilization ≥ the archived elite's, replace the
  worst individual (worst-CV infeasible first, else lowest-util feasible),
  excluding slots already used by this notify pass. Runs AFTER
  `FeasibleEliteCallback` in the chain so the global elite slot is never
  clobbered (it is no longer "worst" once injected).
- Also tracks best **infeasible** per category — **report-only, never
  injected** (feeds the "why it could not fit" explanation).
- Archive exposed on the result as `res.category_elites`.

### 3. Decoder drop log — the "why it didn't fit" signal

`MultiPathLayout.drop_log: list[str]` (default empty). The three injection
paths append a human-readable reason whenever an active descriptor is skipped
(position not STRAIGHT_16 / occupied, geometry validation failed, inventory
unavailable, out of range). Strings are built only on drop — no hot-path cost.

### 4. Reporting — `save_results` extension

For each category with an archived elite: decode, render
`best_with_<cat>.png` (correct renderer dispatch — multi-path vs plain — per
the two-render-paths rule), and write `category_report.md` containing per
category:

- utilization, piece count, speed, **gap vs global best feasible**;
- element counts (e.g. 2 switch pairs / 1 CROSS_90);
- binding-constraint margins and bbox span vs boundary;
- for a category with **no feasible elite**: best-infeasible numbers (CV,
  which G binds) and aggregated `drop_log` reasons from decoding up to 5
  final-population genomes whose descriptor block is active for that element;
- one short static context line per category (general geometric fact, e.g.
  "a CROSS_90 figure-8 spends ≥24 R40 on two turning-circle lobes").

## Out of scope (explicitly)

- MAP-Elites / bin-local competition (B2) — escalation path if injected
  categories stagnate at seed quality.
- Objective changes (third objective / replacing degenerate F[1]).
- Combination categories (switch+cross etc.).
- Any change to inventory, boundary, piece types, or the epsilon schedule.

## Testing

- Unit (TDD, red→green per piece): out-key capture incl. empty-layout zeros;
  archive capture (feasible-only, max-util, deep-copy isolation); injection
  semantics (replaces worst, no clobber of better in-pop member, slot
  exclusivity); drop_log reasons for each skip path; report file generation.
- Integration gate: full pytest suite (no new failures vs 2 pre-existing),
  quick smoke run, then a full `with_crossing` run — expectation: cross elite
  present in `category_report.md` with a rendered PNG, run quality not
  regressed.

## Risks

- Injected elites are dominated and mostly lose tournaments — presence is
  guaranteed, *improvement* is not (accepted; B2 is the escalation).
- 3 injected individuals per generation marginally reduce selection pressure
  (3/1000 ≈ negligible; FeasibleEliteCallback already does 1).
- Custom out-keys ride pymoo's documented mechanism; verified on 0.6.1.6.
