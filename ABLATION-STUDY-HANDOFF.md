# Ablation Study — Results

**Status: EXECUTED 2026-08-10; leave-one-out arms added 2026-08-14.** This document
reports what was run and what came out. It replaces the earlier design draft, several of
whose assumptions were measured to be wrong (§7). Where it disagrees with older notes,
this document wins.

Questions answered: *does stock pymoo solve this problem?* and *what is each custom
component worth — on its own, and inside the working system?*

---

## 1. Verdict

**Stock pymoo solves the problem — barely.** It produced a feasible closed layout in
**13 of 63** cells (21 configs x 3 seeds), and where it succeeded the Pareto front
collapsed to a median of **1 point**. That is enough for a baseline to exist, so the study
is a component ablation against it (no fallback to a parameter study was needed).

**The full system beats that baseline by ~12.9x in mean hypervolume and wins all 63
blocks against it.** The leave-one-out arms then show that the advantage rests almost
entirely on ONE component, and that two others cannot be shown to do anything at all.

| arm | components | cells solved | mean HV | median HV | HV wins | median front |
|---|---|---|---|---|---|---|
| `baseline` | none (stock pymoo) | 13/63 | 0.055 | 0.000 | 0 | 1 |
| `operators` | + partitioned crossover/mutation | 10/63 | 0.041 | 0.000 | 0 | 2 |
| `repair` | + repair pipeline | **63/63** | 0.558 | 0.536 | 0 | 41 |
| `seeding` | + heuristic sampling | **63/63** | 0.642 | 0.643 | 0 | 19 |
| `full_minus_seeding` | all but heuristic sampling | **63/63** | 0.571 | 0.565 | 1 | 41 |
| `full_minus_operators` | all but custom operators | **63/63** | 0.676 | 0.678 | 0 | 41 |
| `full_minus_repair` | all but repair pipeline | **63/63** | 0.702 | 0.704 | 13 | 37 |
| `full_minus_elites` | all but elite injection | **63/63** | 0.710 | 0.709 | 13 | 51 |
| `full_minus_epsilon` | all but adaptive epsilon | **63/63** | **0.711** | **0.711** | **24** | 55 |
| `full` | everything | **63/63** | 0.710 | 0.710 | 9 | 52 |

"HV wins" counts blocks an arm took **outright**. Three further blocks are exact ties
between `full` and `full_minus_elites` and are awarded to neither — breaking them would
make the ranking depend on iteration order. 60 sole winners + 3 ties = 63.

All ten arms are scored in ONE common ideal/nadir box per config, over every arm and seed
of that config (§3). Mixing boxes across tables would make the numbers incomparable, and
the four leave-one-out arms moved those boxes, so every figure here differs from the
six-arm version of this table. **Rescore, never append** — and note that the movement runs
both ways: on `cross_figure8_tall` `full_minus_seeding` contributed a slower extreme that
pushed the nadir out and lifted everyone's normalized HV (`full` 0.746 -> 0.805), while on
`all_pieces_rect` `full_minus_elites` found a *better* extreme that pulled the ideal in and
pushed everyone down (`full` 0.733 -> 0.690). Per-config HV is only comparable inside one
scoring run.

**Paired over the 63 (config, seed) blocks**, one-sided sign test:

| comparison | full wins | losses | ties | mean delta | p |
|---|---|---|---|---|---|
| full > baseline | 63 | 0 | 0 | +0.654 | 1.1e-19 |
| full > operators | 63 | 0 | 0 | +0.669 | 1.1e-19 |
| full > repair | 62 | 1 | 0 | +0.151 | 6.9e-18 |
| full > seeding | 63 | 0 | 0 | +0.067 | 1.1e-19 |
| full > full_minus_seeding | 60 | 3 | 0 | +0.138 | 4.5e-15 |
| full > full_minus_operators | 62 | 1 | 0 | +0.033 | 6.9e-18 |
| full > full_minus_repair | 43 | 20 | 0 | +0.007 | 0.0026 |
| full > full_minus_elites | 21 | 23 | **19** | +0.000 | 0.67 |
| full > full_minus_epsilon | 29 | 34 | 0 | −0.001 | 0.78 |

Component contribution, measured two ways. The isolated column adds the component alone
to the stock baseline; the leave-one-out column removes it from the production system.
The two readings can disagree in sign and in magnitude, which is why both exist:

| component | added alone to baseline | removed from the full system | verdict |
|---|---|---|---|
| heuristic seeding | **+0.587 HV** | **+0.138 HV** (60/3, p=5e-15) | carries the system |
| custom operators | −0.014 HV (harmful) | **+0.033 HV** (62/1, p=7e-18) | small but real |
| repair pipeline | **+0.503 HV** | **+0.007 HV** (43/20, p=0.003) | near-redundant |
| elite injection | not measured | 0.000 HV (21/23, p=0.67) | **no measurable effect** |
| adaptive epsilon | not measured | −0.001 HV (29/34, p=0.78) | **no measurable effect** |

Three results overturn earlier readings of this study and are developed in §5.2, §5.4
and §5.8:

- **Repair is near-redundant inside the system.** It looked like the second pillar when
  measured in isolation (+0.503); its marginal worth is 70x smaller (+0.007) and it is
  the most expensive component per unit of quality bought.
- **Adaptive epsilon and elite injection are indistinguishable from no-ops.** Epsilon adds
  31.6 s per run — 24% on top of the system without it, a fifth of `full`'s total — for
  −0.001 HV.
- **"`full` wins 62 of 63 blocks" was an artifact** of a field made only of crippled
  arms. Against arms that differ from it by one component, `full` takes 9 blocks outright
  and `full_minus_epsilon` takes the most at 24. What survives untouched is the comparison
  against stock pymoo: 63/0, p = 1.1e-19.

And one result that needs no leave-one-out arm to read off §1's table: **heuristic seeding
bolted onto stock pymoo (0.642) beats the entire rest of the system without it (0.571)** —
52 of 63 blocks, mean +0.071. Every other component combined does not match what seeding
does alone.

---

## 2. What "baseline" means here

The problem definition is NOT ablatable and is identical in every arm: the chromosome
encoding, `decode_chromosome`, both objectives, and all `5 + n_piece_types` constraints.
pymoo cannot infer LEGO geometry; without the decoder there is nothing to optimize.

Ablated (all `True` in production, all `False` in the baseline — `SearchComponentsConfig`
in `src/config.py`):

| flag | on | off (stock pymoo) |
|---|---|---|
| `heuristic_sampling` | `IntegerSampling` seed families | `IntegerRandomSampling` |
| `custom_operators` | `PartitionedCrossover` / `PartitionedMutation` | `SBX(prob=0.9, eta=15)` / `PM(eta=20)`, both `vtype=float, repair=RoundingRepair()` |
| `repair` | `TrackRepairPipeline` | no repair operator |
| `adaptive_epsilon` | `LegoAdaptiveEpsilon` | plain NSGA2, unweighted CV |
| `elite_injection` | feasible + category elite callbacks | none |
| `constr_survival` | `ConstrRankAndCrowding` | `RankAndCrowding` (NSGA-II's own default) |

The stock operator values are NSGA-II's **constructor defaults**, verified on the
installed pymoo 0.6.1.6 — not a strawman. `IntegerRandomSampling` + `RoundingRepair` is
the integer adaptation pymoo documents in `customization/discrete`.

`survival` is a flag because `ConstrRankAndCrowding` is **not** what `NSGA2()` gives you;
its default is `RankAndCrowding`. Leaving the project's survival in every arm would have
quietly strengthened the "stock" baseline on a 12-constraint problem.

---

## 3. Method

- **10 arms x 21 configs x 3 seeds = 630 runs**, all completed, no crashes. Seeds {1,2,3}
  are identical across arms, so comparisons are blocked on (config, seed). They are not
  common random numbers — arms with different samplers consume the RNG stream differently
  — so this is blocking, not true pairing.
- Every arm ran with the survival crowding metric at its default `cd`. The metric is an
  algorithm parameter rather than a component toggle, so it is a separate campaign axis
  (§8.1); nothing here is a `pcd` measurement.
- Every config ran at **its own `pop_size` and `n_gen`** — budgets are never shortened.
  `termination.period` is forced to 0 so an improvement-based early stop cannot give
  arms unequal budgets (this only ever lengthens a run).
- Driver: `run_ablation.py` (repo root). Arms are declared in its `ARMS` table; the
  resolved flags are read back off the loaded config into `<run_dir>/arm.json`, because
  pydantic silently ignores a misspelled key and the intended patch is not evidence of
  what ran.
- Configs are loaded from their original path so the relative `train_config_path`
  resolves. Materializing a patched copy next to the run breaks it (`FileNotFoundError`),
  and dropping the key silently substitutes default physics (`v_motor_max` 1.1 instead of
  the measured 1.26).
- Artifacts: `outputs/ablation/<config>/<arm>/s<seed>/`.

Reproduce:

```bash
python run_ablation.py --arms baseline,seeding,operators,repair,full,full_minus_operators --seeds 1,2,3
python run_ablation.py --arms full_minus_seeding,full_minus_repair,full_minus_epsilon,full_minus_elites --seeds 1,2,3
python score_ablation.py --arms baseline,seeding,operators,repair,full,full_minus_operators,full_minus_seeding,full_minus_repair,full_minus_epsilon,full_minus_elites
```

**Pin the arm list when scoring.** Bare `score_ablation.py` scores every arm directory it
finds, so a probe run of a future arm would join the ideal/nadir boxes and move these
tables. The scorer warns when an arm lacks three seeds, but the pin is what makes the
numbers above reproducible.

The driver is resume-safe (it skips any cell whose `run_info.md` already carries a
`## Run Summary`), so any of these may be interrupted and re-issued freely. The whole
campaign is ~19 h sequentially on 16-32 workers.

**Hypervolume protocol**, implemented in `score_ablation.py` — every number in §1, §4 and
§5.5 comes out of that one command, so the tables are reproducible rather than
hand-assembled. Per config, `ideal`/`nadir` over the union of **all ten arms'** archives
(`pareto_archive.csv`, the run-cumulative feasible front), `has_extent` guard, then pymoo
`HV(ref_point=HV_REF_POINT, norm_ref_point=False, zero_to_one=True, ideal=, nadir=)`.
Never pool raw HV across configs: `special_piece_weight` is 6.0 on `dc_figure8_wide` and
3.0 elsewhere, so F[0] is not on a common scale. A run with no feasible solution scores
0; a run without `## Run Summary` is skipped rather than scored 0, so a crash cannot be
charged to an arm as a bad result (§7). Per-cell output lands in
`outputs/ablation/scores.csv`.

**"Front size" means the archive**, i.e. rows in `pareto_archive.csv`. It is not the count
of distinct objective vectors in the final population, which is a diversity measure and
runs 10-15x larger (§5.3). Earlier notes mixed the two.

---

## 4. Results per config

Median hypervolume over the three seeds. Ranking is by HV, never by utilization or front
size alone (§5.3). `−x` columns are the leave-one-out arms; **bold** marks the row's best.

| config | base | ops | rep | seed | −seed | −ops | −rep | −elite | −eps | full |
|---|---|---|---|---|---|---|---|---|---|---|
| `all_pieces` | 0.000 | 0.000 | 0.496 | 0.678 | 0.509 | 0.705 | 0.710 | 0.714 | **0.718** | 0.710 |
| `all_pieces_150x150` | 0.000 | 0.000 | 0.536 | 0.690 | 0.579 | 0.699 | 0.714 | 0.716 | **0.717** | 0.716 |
| `all_pieces_350x350` | 0.195 | 0.155 | 0.498 | 0.690 | 0.468 | 0.706 | 0.720 | 0.721 | **0.722** | 0.722 |
| `all_pieces_rect` | 0.000 | 0.000 | 0.456 | 0.660 | 0.459 | 0.651 | 0.684 | **0.691** | 0.689 | 0.690 |
| `compact` | 0.000 | 0.000 | 0.645 | 0.618 | **0.659** | 0.642 | 0.619 | 0.657 | 0.657 | 0.657 |
| `cross_dc_rect` | 0.000 | 0.000 | 0.547 | 0.644 | 0.543 | 0.670 | 0.708 | 0.677 | **0.711** | 0.678 |
| `cross_figure8_tall` | 0.000 | 0.000 | 0.636 | 0.747 | 0.616 | 0.786 | 0.795 | 0.805 | **0.805** | 0.805 |
| `cross_figure8_wide` | 0.000 | 0.000 | 0.636 | 0.708 | 0.670 | 0.729 | 0.748 | 0.750 | **0.756** | 0.750 |
| `dc_figure8_large` | 0.000 | 0.000 | 0.515 | 0.622 | 0.528 | 0.642 | **0.695** | 0.695 | 0.682 | 0.682 |
| `dc_figure8_wide` | 0.000 | 0.000 | 0.526 | 0.589 | 0.519 | 0.630 | 0.663 | 0.691 | **0.691** | 0.691 |
| `default` | 0.348 | 0.290 | 0.697 | 0.605 | 0.735 | 0.712 | 0.704 | 0.747 | **0.751** | 0.746 |
| `plain_wide_racetrack` | 0.000 | 0.000 | 0.617 | 0.641 | 0.636 | 0.709 | **0.736** | 0.719 | 0.720 | 0.720 |
| `switch_cross_rect` | 0.000 | 0.000 | 0.518 | 0.642 | 0.542 | 0.634 | 0.682 | **0.709** | 0.703 | 0.707 |
| `switch_one_siding_wide` | 0.000 | 0.000 | 0.534 | 0.611 | 0.566 | 0.638 | **0.699** | 0.678 | 0.687 | 0.678 |
| `switch_two_sidings_tall` | 0.293 | 0.310 | 0.504 | 0.642 | 0.531 | 0.646 | 0.667 | 0.681 | **0.682** | 0.681 |
| `with_crossing` | 0.000 | 0.000 | 0.565 | 0.659 | 0.637 | 0.718 | 0.696 | 0.724 | 0.723 | **0.724** |
| `with_double_crossover` | 0.224 | 0.231 | 0.605 | 0.633 | 0.573 | 0.691 | 0.685 | 0.691 | 0.721 | **0.721** |
| `with_double_crossover_narrow` | 0.000 | 0.000 | 0.516 | 0.499 | 0.632 | 0.613 | 0.677 | 0.692 | **0.693** | 0.692 |
| `with_double_crossover_small` | 0.000 | 0.000 | 0.516 | 0.630 | 0.539 | 0.654 | **0.716** | 0.707 | 0.708 | 0.707 |
| `with_switches` | 0.000 | 0.000 | 0.613 | 0.667 | 0.703 | 0.678 | 0.720 | 0.726 | 0.722 | **0.726** |
| `with_switches_and_crossing` | 0.207 | 0.000 | 0.529 | 0.644 | 0.560 | 0.684 | **0.713** | 0.700 | 0.689 | 0.701 |

**No arm dominates any more.** The per-config winner is `full_minus_epsilon` in 10 configs,
`full_minus_repair` in 5, `full` in 3, `full_minus_elites` in 2 and `full_minus_seeding`
in 1 — and in most rows the top five sit within 0.01 of each other, i.e. inside the noise
that three seeds can resolve. The earlier claim that `full` leads every config was true
only against the crippled arms it was then compared with.

What the table still separates cleanly is the two tiers below: every arm carrying
heuristic seeding lands in the 0.65-0.81 band, every arm without it (`baseline`,
`operators`, `repair`, `full_minus_seeding`) lands in 0.00-0.70 and is beaten in 20 of 21
configs. `compact` is the single exception: a 24-piece kit that every capable arm drives
to 100% utilization on a front ~8 points wide, so the block is saturated and
`full_minus_seeding` edges it.

---

## 5. Findings

### 5.1 Stock pymoo's failure mode is the boundary, not closure

In **12 of the 16** configs it failed, the closest individual violates exactly ONE
constraint block, and it is almost always the boundary — closure is already satisfied.
Denormalized from `G[3] = (violation - boundary_tolerance) / diagonal`:

```
all_pieces              closure OK, ~90 studs outside the box
dc_figure8_large        closure OK, ~125 studs
switch_one_siding_wide  closure OK, ~57 studs
with_switches           closure OK, ~55 studs
dc_figure8_wide         closure OK, ~34 studs
```

**Stock pymoo can close loops; it cannot size them to the box.** `compact` is the lone
exception — there closure binds (~16 studs off), which fits a 24-piece kit where the
curve count must land exactly.

### 5.2 Seeding and repair are redundant, and the redundancy resolves in seeding's favour

Either one **alone** lifts coverage from 5/21 to 21/21. Isolated HV gains sum to +1.090
while the full system only reaches +0.654 over baseline — heavily sub-additive. They solve
the same blocker (§5.1: sizing a closed loop to the box) by different routes, so the
second one to arrive finds little left to do.

The leave-one-out arms say which one is load-bearing:

| | isolated | leave-one-out | median utilization |
|---|---|---|---|
| heuristic seeding | +0.587 | **+0.138** (60/3, p=5e-15) | 55.6% alone, 58.9% in `full` |
| repair pipeline | +0.503 | **+0.007** (43/20, p=0.003) | 39.6% alone — lowest of any arm |

**Removing repair from the working system costs almost nothing** (+0.007 HV, and it still
solves 63/63 with median utilization 58.9%, identical to `full`). Removing seeding costs
20x more and drops median utilization to 40.9%, i.e. all the way back to what repair
achieves on its own. Repair reaches feasibility but not quality; seeding reaches both, and
once it is present repair has almost nothing left to contribute.

The effect is nevertheless real, not zero: p=0.003 over 63 blocks, and repair still leads
5 configs outright (§4). It is small, not absent — unlike epsilon and elites (§5.8).

### 5.3 Front size is a misleading metric; rank by hypervolume

On `with_crossing` seed 1 the `repair` arm's final population holds **827 distinct
objective vectors against `full`'s 70** — and it loses on HV, 0.565 vs 0.724. Its median
archive is more than double `seeding`'s (41 vs 19) and it still loses (0.558 vs 0.642).
The extra points sit under the competitor's front. Reporting spread alone inverts the
conclusion.

**Two different quantities were conflated in earlier notes and must not be:**

| metric | `with_crossing` s1, `repair` | same cell, `full` |
|---|---|---|
| distinct F in the final population (diversity) | 827 | 70 |
| rows in `pareto_archive.csv` (the front) | 52 | 71 |

Read as diversity, `repair` is 12x wider; read as a front, it is *narrower* than `full`.
The 827 figure that appeared next to "median front 45" in the six-arm write-up was the
population measure, not a front. This document uses the archive throughout (§3).

### 5.4 Custom operators are complementary, not independently useful

In isolation they are **harmful**: 3/21 configs solved, worse than stock pymoo's 5/21,
losing `all_pieces_350x350` and `with_switches_and_crossing` that pymoo solved without
them. Measured leave-one-out inside the working system they are **worth +0.033 mean HV
and win 62 of 63 blocks** (p=7e-18) — the only component besides seeding with an
unambiguous positive marginal value.

Their value concentrates on **special-piece configs**, matching the geometry-aware
sub-operators in `src/operators.py` (`_grow_dc_figure_eight`, `_compensated_pair_grow`,
crossing-aware straighten). Median utilization, `full` vs `full_minus_operators`:

```
switch_cross_rect             58.2% vs 45.1%   (+13.1pp)
with_double_crossover         69.8% vs 59.3%   (+10.5pp)
with_double_crossover_narrow  42.2% vs 35.6%    (+6.6pp)
switch_two_sidings_tall       67.7% vs 61.3%    (+6.4pp)
with_switches                 42.7% vs 36.6%    (+6.1pp)
switch_one_siding_wide        59.1% vs 53.0%    (+6.1pp)
```

`full` leads utilization in 9 of 21 configs and **ties in the other 12** — where the
operators do not raise the peak they still improve the front's interior (median archive
52 vs 41). Keep them: they are second only to seeding in marginal value, and they are no
longer the worst value-per-second component in the system — repair is (§5.5).

### 5.5 What each component costs, and whether it pays for itself

More components means more compute per generation — expected, not a confound. The question
is whether the spend buys quality. Cost is the sum of `gen_seconds` over a run's
`convergence.csv`, available for all 630 cells; validated against the driver's own
`elapsed_s` on the 454 cells where both exist (mean ratio 0.986, range 0.906-0.998 — the
remainder is setup and artifact writing).

| arm | mean s/run | vs baseline | mean HV | HV gain vs baseline |
|---|---|---|---|---|
| `baseline` | 68.4 | 1.00x | 0.055 | — |
| `seeding` | 70.3 | 1.03x | 0.642 | +0.587 |
| `operators` | 89.5 | 1.31x | 0.041 | −0.014 |
| `repair` | 102.8 | 1.50x | 0.558 | +0.503 |
| `full_minus_repair` | 117.2 | 1.71x | 0.702 | +0.647 |
| `full_minus_operators` | 120.3 | 1.76x | 0.676 | +0.621 |
| `full_minus_epsilon` | 131.5 | 1.92x | **0.711** | +0.656 |
| `full_minus_seeding` | 138.4 | 2.02x | 0.571 | +0.516 |
| `full_minus_elites` | 153.2 | 2.24x | 0.710 | +0.655 |
| `full` | 163.1 | 2.38x | 0.710 | +0.654 |

**What each component costs inside the working system**, from the leave-one-out gap. The
seconds are not the component's own runtime alone: removing it also changes what the
search builds, and the self-intersection scan is quadratic in piece count.

| component | extra s/run | marginal HV | HV per extra second |
|---|---|---|---|
| heuristic seeding | +24.7 (+18%) | +0.138 | **0.0056** |
| custom operators | +42.8 (+36%) | +0.033 | 0.00077 |
| repair pipeline | +45.9 (+39%) | +0.007 | 0.00015 |
| elite injection | +9.9 (+6%) | 0.000 | 0 |
| adaptive epsilon | +31.6 (+24%) | −0.001 | negative |

Seeding is **7x** more compute-efficient than the operators and **37x** more than repair,
and that understates it: bolted onto the bare baseline it returns 0.310 HV per second, two
orders of magnitude beyond anything else here.

**The three cheapest wins available are all subtractions.** Dropping adaptive epsilon
saves 31.6 s per run (a fifth of `full`'s total) for −0.001 HV; dropping elite injection
saves 9.9 s for 0.000; dropping repair saves 45.9 s for −0.007. Together they are 88 s of
the 163 s `full` spends, against a combined HV cost inside the resolution of three seeds.
That is a hypothesis about their sum, not a measurement — no arm removed more than one
component at a time, and their effects need not add.

The full system's 12.9x HV advantage over stock pymoo is bought with 2.38x the CPU, but
the component responsible for most of it is effectively free.

### 5.6 Generation 1 runs undersized — a one-generation effect, not a lost factor level

pymoo attaches repair in two places: `Initialization` repairs the whole initial population
and then deduplicates it **without refilling to `pop_size`**
(`pymoo/core/initialization.py:39,42`), whereas `Mating` refills in a loop
(`pymoo/core/infill.py:33`). Audited with the production sampler and repair pipeline at
sampler seed 1, then independently reproduced by three reviews.

What happens: **19 of 21 configs start generation 1 below `pop_size`**, by up to 8%
(`switch_one_siding_wide` 600 -> 552, `all_pieces_rect` 800 -> 753, `all_pieces_350x350`
1000 -> 955). Every missing individual is a heuristic seed; the random block never
collapses in any config. Repair causes the loss in 17 of those 19 — in `default` and
`compact` the duplicates come from the sampler itself and repair changes nothing.

**Mechanism, established with a negative control.** `_get_heuristic_patterns` yields far
fewer patterns than seed rows, and the seed loop deals them round-robin
(`src/sampling.py:646`) — about 7 copies of each variant. The only thing distinguishing
the copies is a random start offset, which `BoundaryAwareRepair` zeroes on any layout that
would leave the box (`src/repair.py:611`). Every collapsed pair differs *only* in those two
genes, and with `enable_boundary_repair=False` the collapse disappears entirely (600/600).

**Why this is nevertheless not a problem — three measurements:**

1. **`pop_size` is delivered for the whole run.** `n_offsprings` defaults to `pop_size` and
   survival is called with `n_survive=self.pop_size`, so the merged pool is truncated back
   to size at the first generational step. Across 379 archived runs the recovered
   per-generation population deviates at generation 1 only — **zero deviations at any
   generation >= 2** — and all 377 runs with final artifacts wrote exactly `pop_size` rows.
2. **No diversity is lost.** Dedup keeps the first occurrence and matches at
   `epsilon=1e-16`, so each deleted row was bit-identical to a retained one. Over 30
   populations (5 configs x 6 sampler seeds) **no seed family and no decoded phenotype was
   ever eliminated**; every generated variant survives at least once, the DC and crossing
   bridgeheads included.
3. **The raw seeds work on their own.** The `seeding` arm applies no repair anywhere and
   still solved 21/21 with fronts of 5-66 points, so the patterns close unaided — the worry
   that seeds succeed only because repair fixes them does not hold.

What remains is a mild reallocation of *initial mating weight*: the large-footprint
families (`racetrack`, `oval_with_siding`) enter the first tournament with fewer duplicate
tickets than the round-robin intended.

Two design smells surfaced by the audit, worth separate attention:

- **The sampler and the repair pipeline work against each other.** `src/sampling.py:747`
  documents the random start offset as existing specifically to "give `eliminate_duplicates`
  enough room to keep cycled seeds as distinct individuals" — and `src/repair.py:611`
  erases exactly that offset, on exactly those seeds.
- **Seed budget is allocated per variant, not per family.** In `all_pieces`,
  `oval_two_sidings` contributes 16 variants and claims 34% of the seed rows while
  `figure_eight` contributes 2 and claims 4%. That is a far larger lever on which
  topologies the search bootstraps from than the deduplication is.

One caveat on all numbers here: they are measured at sampler seed 1. Production configs
ship `seed: null`, so the exact counts move run to run; the mechanism does not.

### 5.7 The heuristic sampler, not the GA, produces the champion in half the configs

From a separate full-system sweep (one run per config, production settings): a feasible
layout exists in **generation 1 in 21/21** configs, and in **10 of 21** the best
utilization never improves for the rest of the run. The GA broadened the front in 21/21
(e.g. 20 -> 138 distinct vectors). Where it does improve utilization it is worth up to
+17.4pp (`all_pieces_rect`), and those are the configs whose boundary does not match a
seed pattern. No elite was ever lost — the final best always equals the run best.

### 5.8 Adaptive epsilon and elite injection cannot be shown to do anything

Both leave-one-out arms come out level with the production system, and the epsilon arm
comes out marginally ahead:

| removed | wins | losses | ties | mean delta | p | utilization vs `full` |
|---|---|---|---|---|---|---|
| `adaptive_epsilon` | 29 | 34 | 0 | −0.001 | 0.78 | ties 15/21, `full` leads 4/21 |
| `elite_injection` | 21 | 23 | **19** | 0.000 | 0.67 | ties 16/21, `full` leads 3/21 |

Neither test comes close to significance, and `full_minus_epsilon` has the highest mean HV
of any arm in the study (0.711) while costing 31.6 s per run less — 24% on top of what the
system costs without it, a fifth of `full`'s total.

**The elite callbacks are sometimes a literal no-op.** In 19 of the 63 blocks `full` and
`full_minus_elites` land on *exact* the same HV (3 of those 19 are also the block's top
score, hence §1's tie note), and on `with_crossing` seed 1 the two runs' `fitness.csv` are
**byte-identical**: with the seed fixed, removing the callbacks changed nothing whatsoever,
because the elite was already in the population and the re-injection had nothing to do.
`FeasibleEliteCallback` and `CategoryEliteArchive` consume no RNG, so a no-op leaves the
stream untouched and the run replays exactly.

**Where they are not a no-op, the evidence points the wrong way.** On `all_pieces_rect`,
`full` returns the *identical* champion on all three seeds — 119 pieces, 58.5%, 14.87 s,
zero switch pairs — while `full_minus_elites` seed 1 found **131 pieces, 68.9%, two switch
pairs**, the best layout any arm produced there:

```
all_pieces_rect   full              s1/s2/s3   119 pcs  58.5%  0 switch pairs  (identical)
                  full_minus_elites s1         131 pcs  68.9%  2 switch pairs
```

A system that converges to the same point regardless of seed is not exploring, and the
branched layout it never reaches is exactly the topology the special-piece weighting is
supposed to reward. This is one cell, so it is a lead rather than a finding — but it is a
mechanism for the null result above: re-injecting elites every generation may be costing
diversity rather than protecting quality. Worth pairing with §8.1, where over 90% of the
final population is already a duplicate in objective space.

Two things this does **not** establish:

- **It does not price epsilon alone.** `adaptive_epsilon=False` also drops the x1000
  hard-constraint CV weighting — both are set inside `LegoAdaptiveEpsilon`
  (`src/algorithm/runner.py`, `cv_ieq` scale). What is neutral is the pair. Splitting them
  into two flags is the prerequisite for any decision to remove either.
- **It does not say epsilon was never useful.** A plausible reconciliation with the older
  observation that runs were worse without it: the translational closure repair landed
  after that observation and removed the feasible-collapse epsilon was introduced to
  survive. On this instance family, at these budgets, whatever epsilon rescues is now
  rescued earlier by something else. That is a hypothesis — no arm tests it.

The actionable reading is narrower than "delete them": these two components are carrying
31.6 s and 9.9 s per run of unproven weight, and they are the first place to look for
compute that could fund more generations or more seeds.

---

## 6. Limitations

- **Three seeds per cell.** Enough for the pooled sign tests in §1, not enough for
  per-config inference: with three observations the exact-test floor is p = 0.125 for any
  single config. Per-config numbers are descriptive; only the pooled comparisons carry
  statistical weight — and §4's per-config winners, separated by <0.01 HV, are noise.
  Seeds are matched by index across arms, which is blocking, not common random numbers —
  arms with different samplers consume the RNG stream differently.
- **The two null results are "not detected", not "proven zero".** 63 blocks resolve the
  effects that matter here — seeding (+0.138) and operators (+0.033) clear significance
  easily, and repair does at +0.007 — so epsilon and elites are certainly not large. But a
  true effect of a few thousandths of HV would not be separable from noise at this sample
  size, and the sign test ignores magnitude entirely.
- Configs are the project's own development set, several co-designed with the heuristic
  seeds (`configs/dc_figure8_wide.yaml` documents its box being sized so the DC seed
  fires). Results do not generalize beyond this instance family. **This is the largest
  remaining weakness**: the arm that wins hardest is the one whose seed patterns those
  boxes were shaped around.
- Arms are compared at **equal generations**, and therefore not at equal wall-clock:
  adding components costs compute, and arms that succeed build larger layouts whose
  cost per generation is higher (the self-intersection scan is quadratic). This is
  reported rather than corrected. Equalizing wall-clock was rejected on purpose: the
  cost ratio is an *outcome* of each arm's behaviour, not a parameter of the method, so
  any scaling factor would have to be chosen after seeing the runs — a researcher degree
  of freedom, not a control. Cost per arm is instead reported next to quality in §5.5, so
  the question "did the extra compute buy anything" is answered directly.
- `adaptive_epsilon` off also removes the x1000 hard-constraint CV weighting — the two
  are set together inside `LegoAdaptiveEpsilon`. The baseline is unaffected (it wants
  pymoo's unweighted CV) but the two mechanisms cannot currently be priced separately.
- **`constr_survival` is the one component with no leave-one-out arm.** It is off in the
  baseline and on in every `full_minus_*` arm, so its marginal value inside the system is
  unmeasured. Everything else now has both readings, and they disagree badly enough
  (operators flip sign; repair shrinks 70x) that no isolated number should be read as a
  marginal one.
- **Only one component is removed at a time.** §5.5 notes that epsilon, elites and repair
  together account for 88 of `full`'s 163 s at a combined HV cost that looks negligible —
  but that is addition of three separate measurements, and interactions are exactly what
  this study keeps finding. A `full_minus_{epsilon,elites,repair}` arm would test it.

---

## 7. Corrections to the earlier design draft

Each was measured, not argued:

- **`RoundingRepair` returns `np.around(X).astype(int)`** — stock-arm populations carry
  int64, never float. The "required int16 cast at the decoder entry" was unnecessary and
  its acceptance test could not have failed. Not implemented.
- **`ConstrRankAndCrowding` is not NSGA-II's default** (`RankAndCrowding` is), and on a
  near-fully-infeasible population it admits individuals with ~48% higher CV. Now a flag.
- **"Orphan switches" does not exist** anywhere in `src/`, and `G[4]` has no switch term —
  it is `unresolved_crossings/5 + dangling_cross + dangling_dc`. The draft's gate
  criterion cited a metric that cannot be computed.
- **`max_closure_error` is euclidean while `G[0..2]` are per-axis**, so the draft's
  closure check rejected 20–30% of genuinely feasible layouts (214/1000 in one run).
- **A missing `pareto_archive.csv` does not mean zero feasible** — it also means the run
  crashed before `save_results`. Scoring it as a failure would bias whichever arm crashes
  more. Disambiguate on `## Run Summary` in `run_info.md`.
- **`category_report.md` exists iff `elite_injection` is on**, not iff the run found
  something feasible.
- **`n_eval` in `convergence.csv` reads 0 under the adaptive-epsilon wrapper** and
  non-zero without it — an artifact that looks like a real cross-arm difference. Use
  `n_gen x pop_size` as the budget axis.
- The stock `SBX(prob=0.9, eta=15)` / `PM(eta=20)` values are **NSGA-II's constructor
  defaults**, not the discrete-customization doc (which uses `prob=1.0, eta=3.0`).

One research thread was opened and closed: pymoo ships `MixedVariableGA` /
`MixedVariableMating`, which supply type-correct operators (`UX`, `ChoiceRandomMutation`,
`BFM`) for heterogeneous genomes like this one, and would have been a stronger stock
baseline. It was not needed — the flat integer baseline already cleared the gate, so
strengthening it would not have changed the branch.

---

## 8. Open work

Done and folded into the sections above: seeds 2-3 for every arm, the four leave-one-out
arms, the generation-0 seed & repair audit (§8.2), the cost-vs-quality table (§5.5), and
the `pcd`-versus-`cd` comparison on every arm (§8.1 — answered, no difference).

| item | cost | what it buys |
|---|---|---|
| split `adaptive_epsilon` into schedule and hard-CV-weighting flags, then one arm per half | small code change + ~2.5 h | prices the two mechanisms apart — the prerequisite for acting on §5.8, which currently only shows the *pair* is neutral |
| `full_minus_survival` arm | ~2.5 h | `constr_survival` is the only component with no leave-one-out reading (§6) |
| combined removal, `full_minus_{epsilon,elites,repair}` | ~2 h | tests whether three separately-null components are jointly null, or whether they cover for each other (§5.5) |
| held-out instances generated by a stated rule | ~2 h | an external-validity claim the development config set cannot support |

### 8.2 Seed & repair audit — DONE, and it came out mostly negative

**Executed and cross-checked by three independent reviews. The alarming reading below was
refuted; read §5.5 for what actually holds.** Summary of the correction, kept here because
this section previously told the parameter study to treat `pop_size` and `heuristic_ratio`
as unreliable:

- **`pop_size` IS delivered.** The dedup shortfall lasts exactly **one generation**:
  `n_offsprings` defaults to `pop_size` and survival is called with
  `n_survive=self.pop_size`, so the merged parent+offspring pool is truncated back to the
  configured size from generation 2 on. Verified across 379 archived runs — 113 show a
  generation-1 shortfall, **zero deviate at any generation >= 2**, and all 377 runs with
  final artifacts wrote exactly `pop_size` rows to `fitness.csv`.
- **No diversity is lost.** `DefaultDuplicateElimination` keeps the first occurrence and
  matches at `epsilon=1e-16`, so every deleted individual was bit-identical to one that
  stayed. Across 30 populations (5 configs x 6 sampler seeds) no seed family and no
  decoded phenotype was ever wiped out.
- **The parameter study is therefore NOT affected.** Factor levels for `pop_size` and
  `heuristic_ratio` are what the config says. Disregard the earlier warning.

### 8.3 Why the audit was run first (historical)

pymoo attaches the repair operator in **two** places, not one — the offspring path and the
initialization path (`pymoo/algorithms/base/genetic.py:56` and `:64`). `Initialization`
repairs the whole starting population and then removes duplicates **without refilling to
`pop_size`** (`pymoo/core/initialization.py:39,42`).

This bears directly on two claims already made above and must be checked before either is
treated as settled:

- **§5.2's "seeding and repair are redundant" may be measuring something narrower.** The
  `seeding` arm has `repair=False`, so it evaluates raw seed patterns; the `repair` arm
  evaluates repaired random chromosomes. If a raw seed does not close on its own, then
  "seeding alone lifts coverage to 21/21" is really "seeding-as-filtered-by-repair does",
  and the two components are not independent at generation 0 at all.
- **`pop_size` and `heuristic_ratio` may be nominal, not delivered.** The nine seed
  families are dealt cyclically (`src/sampling.py:647`), so the same pattern is written
  many times differing only in its start-position genes — exactly what
  `BoundaryAwareRepair` normalizes. Collapsed duplicates are deleted, not replaced, so
  both the delivered population size and the delivered seed share can come out below the
  configured values. That would also make the parameter study's factor levels nominal.

The audit is sampling plus repair only — no evaluation, no process pool, seconds on one
core — but it needs an idle machine, because wall-clock per run is a reported metric here.

### 8.1 `pcd` versus `cd` — run on every arm, and there is no difference

`RankAndCrowding` and `ConstrRankAndCrowding` both take `crowding_func='cd'` by default
and this project never overrides it (`src/algorithm/runner.py:739`), so every arm above
ran on the original NSGA-II crowding distance. pymoo's survival documentation recommends
**`'pcd'` (pruning crowding distance) for two-objective problems** and `'mnn'` beyond two;
this problem has `n_obj=2`.

The mechanism matters more here than the recommendation. `cd` is built with
`filter_out_duplicates=False` and `pcd` with `True`, and that flag filters duplicates in
**objective space** — duplicates are assigned a crowding score of zero and lose the
in-front comparison. `eliminate_duplicates=True` on the algorithm only removes duplicate
**genotypes**, and in this encoding many distinct genomes decode to the same phenotype.
Measured on the `full` arm, seed 1:

```
config              unique F at gen 1 -> final     final population distinct
all_pieces                    823 -> 72                        7.2%
with_crossing                 810 -> 70                        7.0%
switch_cross_rect             581 -> 79                        9.9%
dc_figure8_wide               431 -> 58                        9.7%
```

Over 90% of the final population is a clone in objective space, which is exactly the
regime where a crowding metric that does not filter duplicates misreports density.

**Design:** a new *arm* per metric on the existing seeds {1,2,3}, never a new seed — a seed
is a replication, and running `pcd` under a fresh draw would confound the operator change
with the draw. `crowding_func` lives in `AlgorithmConfig` (an algorithm parameter, not a
component toggle); the driver crosses it with the arms as a second axis (`--crowding`),
innermost in the loop so both members of a pair run back to back, and cells at `cd` keep
the bare arm directory name so the archived runs stay resumable and comparable.

**Executed in full:** 10 arms x 21 configs x 3 seeds x {cd, pcd} = **1260 cells**, zero
crashes. Sign test of each variant against its own `cd` arm, blocked on (config, seed):

| arm | pcd wins | losses | ties | mean HV delta | p |
|---|---:|---:|---:|---:|---:|
| `seeding` | 31 | 32 | 0 | +0.003 | 0.60 |
| `repair` | 31 | 32 | 0 | -0.004 | 0.60 |
| `full` | 29 | 34 | 0 | -0.000 | 0.78 |
| `full_minus_epsilon` | 28 | 35 | 0 | -0.001 | 0.84 |
| `full_minus_seeding` | 27 | 36 | 0 | -0.016 | 0.90 |
| `full_minus_repair` | 25 | 38 | 0 | -0.003 | 0.96 |
| `full_minus_elites` | 22 | 41 | 0 | -0.002 | 0.99 |
| `full_minus_operators` | 16 | 46 | 1 | -0.003 | 1.00 |
| `operators` | 5 | 4 | 54 | -0.001 | 0.50 |
| `baseline` | 1 | 0 | 62 | +0.000 | 0.50 |

**`pcd` never wins.** It loses more blocks than it wins in eight arms of ten, every mean
delta sits inside -0.016..+0.003, and no arm reaches `p < 0.5`. The largest single effect
is against it. Cost is equal too: 19.7 h of compute for `pcd` against 20.2 h for `cd` over
the same 630 cells, so there is no speed argument either way. **`cd` stays the default.**

The duplicate-density hypothesis above is therefore not refuted so much as shown to be
irrelevant: objective-space cloning is real, but which clone survives does not move the
hypervolume. Two further readings support that. In `baseline` and `operators` — the arms
that solve 13/63 and 10/63 cells — 62 and 54 blocks are exact ties, because
`ConstrRankAndCrowding` applies the crowding metric only to the feasible subset and there
is barely a feasible subset to apply it to. Across the whole campaign 118 of 630 pairs
produced bit-identical hypervolume.

**Implementation note, load-bearing:** selecting `pcd` must go through
`_pruning_crowding` (`src/algorithm/runner.py`). pymoo 0.6.1.6's *compiled* `calc_pcd`
writes past its output buffer once `n_remove >= n_distinct - n_obj`, which surfaces as an
intermittent SIGSEGV or a corrupted neighbouring array (`IndexError` inside
`metrics.py`). The Python reference clamps at `n - n_obj` and the compiled kernel still
faults on that exact value, so the wrapper stops one step below. This is not a corner
case here: survival computes `n_remove` on the full front while the metric sees only the
distinct rows, and a 1000-individual population of this problem holds ~25 distinct `F`.