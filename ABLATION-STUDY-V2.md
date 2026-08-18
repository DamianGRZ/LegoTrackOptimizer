# Ablation Study — What Each Custom Component Is Worth

**10 arms x 21 configs x 4 seeds = 840 runs, all completed, no crashes.**

Question: *does stock pymoo solve this problem, and what is each custom component worth — on
its own, and inside the working system?*

Every number below was re-derived from the raw `pareto_archive.csv` files by two independent
audits.

---

## 1. Verdict

**Stock pymoo reaches a feasible closed layout in 16 of 84 cells**, and where it succeeds the
front collapses to a median of 1 point. The full system solves 84/84.

| arm | components | solved | mean HV | median HV | HV wins | median front |
|---|---|---|---|---|---|---|
| `baseline` | none (stock pymoo) | 16/84 | 0.051 | 0.000 | 0 | 1 |
| `operators` | + partitioned crossover/mutation | 13/84 | 0.042 | 0.000 | 0 | 2 |
| `repair` | + repair pipeline | **84/84** | 0.557 | 0.536 | 0 | 40 |
| `seeding` | + heuristic sampling | **84/84** | 0.643 | 0.643 | 0 | 18 |
| `full_minus_seeding` | all but heuristic sampling | **84/84** | 0.574 | 0.564 | 1 | 41 |
| `full_minus_operators` | all but custom operators | **84/84** | 0.677 | 0.680 | 0 | 41 |
| `full_minus_repair` | all but repair pipeline | **84/84** | 0.702 | 0.705 | 16 | 37 |
| `full_minus_elites` | all but elite injection | **84/84** | 0.710 | 0.710 | 18 | 51 |
| `full_minus_epsilon` | all but adaptive epsilon | **84/84** | **0.711** | 0.711 | **33** | 55 |
| `full` | everything | **84/84** | **0.711** | **0.712** | 12 | 52 |

"HV wins" counts blocks an arm took outright; a further 4 blocks are exact ties between `full`
and `full_minus_elites` and are awarded to neither.

**Read the mean-HV gap carefully.** The smallest non-zero HV anywhere in the campaign is
0.155 — there is no continuum between "found nothing" (0.000) and "found something", so mean
HV is mostly a coverage rate:

| | mean HV, all cells | mean HV, cells actually solved |
|---|---|---|
| `baseline` | 0.051 (n=84) | 0.270 (n=16) |
| `full` | 0.711 (n=84) | 0.711 (n=84) |

**13.9x unconditional, 2.64x conditional.** The headline advantage is ~80% coverage and ~20%
front quality.

One-sided sign test, blocked on (config, seed) — and the same test collapsed to a per-config
majority, because four seeds of a config are not independent observations:

| comparison | wins/losses/ties | mean delta | p (84 blocks) | p (21 configs) |
|---|---|---|---|---|
| full > baseline | 84/0/0 | +0.660 | 5.2e-26 | 4.8e-07 |
| full > operators | 84/0/0 | +0.670 | 5.2e-26 | 4.8e-07 |
| full > repair | 83/1/0 | +0.154 | 4.4e-24 | 4.8e-07 |
| full > seeding | 84/0/0 | +0.068 | 5.2e-26 | 4.8e-07 |
| full > full_minus_seeding | 81/3/0 | +0.138 | 5.1e-21 | 9.5e-07 |
| full > full_minus_operators | 83/1/0 | +0.034 | 4.4e-24 | 4.8e-07 |
| **full > full_minus_repair** | 58/26/0 | +0.009 | 0.00031 | **0.058** |
| full > full_minus_elites | 29/32/**23** | +0.001 | 0.70 | 0.70 |
| full > full_minus_epsilon | 36/48/0 | −0.000 | 0.92 | 0.95 |

Ties are excluded from the sign-test denominator (Dixon–Mood), so the `full_minus_elites` row
is n=61, not 84; `mean delta` is averaged over all blocks including ties. Nine simultaneous
comparisons, no multiplicity correction (Bonferroni x9 leaves repair at 0.0028 on blocks).

Component value, measured two ways:

| component | added alone to baseline | removed from the full system | verdict |
|---|---|---|---|
| heuristic seeding | **+0.592 HV** | **+0.138 HV** (81/3, p=5e-21) | carries the system |
| custom operators | −0.010 HV | **+0.034 HV** (83/1, p=4e-24) | real, but see §5.2 |
| repair pipeline | **+0.506 HV** | +0.009 HV (58/26, p=0.0003 → **0.058** by config) | see §5.3 |
| elite injection | no arm | +0.001 HV (29/32, p=0.70) | not measurable, see §5.5 |
| adaptive epsilon | no arm | −0.000 HV (36/48, p=0.92) | never engaged, see §5.4 |
| constr survival | no arm | **no arm** | unmeasured, see §6 |

One result needs no leave-one-out arm: **heuristic seeding bolted onto stock pymoo (0.643)
beats the entire rest of the system without it (0.574)**.

---

## 2. What was ablated — and what stayed on in every arm

Six flags, `SearchComponentsConfig` in `src/config.py`:

| flag | on | off |
|---|---|---|
| `heuristic_sampling` | `IntegerSampling` seed families | `IntegerRandomSampling` |
| `custom_operators` | `PartitionedCrossover` / `PartitionedMutation` | `SBX(prob=0.9, eta=15)` / `PM(eta=20)`, `vtype=float, repair=RoundingRepair()` |
| `repair` | `TrackRepairPipeline` | no repair operator |
| `adaptive_epsilon` | `LegoAdaptiveEpsilon` | plain NSGA2, unweighted CV |
| `elite_injection` | feasible + category elite callbacks | none |
| `constr_survival` | `ConstrRankAndCrowding` | `RankAndCrowding` (NSGA-II's own default) |

The arm families are exactly: none (`baseline`), one component (`seeding`, `operators`,
`repair`), all (`full`), all-but-one (`full_minus_*`).

Verified per cell, three ways. The flags in `<run_dir>/arm.json` are read back off the
**loaded** config and match the arm table in 840/840 cells. Each cell's `run.log` states the
six flags and names the six objects the run actually built — sampler, crossover, mutation,
repair, survival, algorithm — plus its callback chain; 210 logs checked against the arm table,
0 mismatches. And an instrumented full run counts the invocations: a disabled component is
called exactly **0** times, an enabled one thousands.

```
arm                    IntSampling  IntRandomSampl  PartCross  PartMut   SBX    PM   RepairPipeline
baseline                         0               1          0        0   998  1008                0
full                             1               0       2546     2581     0     0             2582
full_minus_repair                1               0       1345     1364     0     0                0
full_minus_operators             1               0          0        0  1977  2006             2007
```

**What is NOT ablated, and runs in every arm including `baseline`.** The problem definition is
not inert — `decode_chromosome` is a repair operator with a validator inside it:

`src/decoder/construction.py:181` — inventory clamp: a piece whose type is exhausted is skipped
`src/decoder/construction.py:221` — junction descriptors clamped, then dropped if unaffordable
`src/decoder/construction.py:673` — `_apply_crossing_repair`, emergent self-crossings → CROSS_90
`src/decoder/construction.py:955` — `_auto_center`, the translate branch of `BoundaryAwareRepair`

Measured on 300 stock-sampled chromosomes for `all_pieces`: the decoder drops **28.5% of the
active main-loop genes** on inventory grounds (158.6 → 113.4 pieces) and 3.5 descriptors per
decode. Consequences run through the whole study and are developed in §5.3 and §6.

---

## 3. Method

- Seeds {1,2,3,4} identical across arms, so comparisons are **blocked** on (config, seed). Arms
  with different samplers consume the RNG stream differently, so this is blocking, not common
  random numbers.
- Every config ran at **its own `pop_size` and `n_gen`**. `termination.period` is forced to 0
  so an improvement-based early stop cannot hand arms unequal budgets. Verified: all 840 cells
  executed exactly their planned generations, reason `max-generations`.
- Driver `run_ablation.py`; artifacts in `outputs/ablation/<config>/<arm>/s<seed>/`.

```bash
python run_ablation.py --arms baseline,seeding,operators,repair,full,full_minus_operators,full_minus_seeding,full_minus_repair,full_minus_epsilon,full_minus_elites --seeds 1,2,3,4
python score_ablation.py --arms baseline,seeding,operators,repair,full,full_minus_operators,full_minus_seeding,full_minus_repair,full_minus_epsilon,full_minus_elites
```

The driver is resume-safe (skips any cell whose `run_info.md` carries a `## Run Summary`), so
these may be re-issued freely; against the current archive they re-run nothing. The scorer
prints a completeness warning hardcoded to three seeds, so it flags every (config, arm) pair as
short — the grid is complete at four.

**Hypervolume protocol.** Per config, `ideal`/`nadir` over the union of the scored arms'
`pareto_archive.csv` (the run-cumulative feasible front), `has_extent` guard, then pymoo
`HV(ref_point=HV_REF_POINT, norm_ref_point=False, zero_to_one=True, ideal=, nadir=)`. A run
with no feasible solution scores 0; a run without `## Run Summary` is skipped rather than
scored 0, so a crash cannot be charged to an arm as a bad result.

**Never pool raw HV across configs**: `special_piece_weight` is 6.0 in **9 of the 21 configs**
and 3.0 in the other 12, so F[0] is not on a common scale.

**The box depends on which arms are scored, so pin `--arms`.** Adding an arm can move the
per-config ideal/nadir and hence every normalized HV; a box only moves if the new arm extends a
per-axis extreme, so some cells shift and the rest stay bit-identical. Rescore the whole
campaign in one pass rather than appending. `write_scores` records no arm set, so
`outputs/ablation/scores.csv` cannot describe its own box — it reproduces these tables only
when produced by the command above.

**"Front size" means the archive** — rows in `pareto_archive.csv` — not the count of distinct
objective vectors in the final population, which is a diversity measure and runs 10–15x larger.

**Cost is reported from seeds 1–3 only** (§5.6): those three ran as one uninterrupted session,
while seed 4 was measured on a different day across several stop/resume cycles. HV is unaffected
— it is deterministic given (config, seed) — but wall-clock is not comparable across sessions.

---

## 4. Results per config

Median hypervolume over four seeds; **bold** marks the row's best. `−x` columns are
leave-one-out arms.

| config | base | ops | rep | seed | −seed | −ops | −rep | −elite | −eps | full |
|---|---|---|---|---|---|---|---|---|---|---|
| `all_pieces` | 0.000 | 0.000 | 0.501 | 0.678 | 0.528 | 0.705 | 0.710 | **0.716** | 0.715 | 0.712 |
| `all_pieces_150x150` | 0.000 | 0.000 | 0.514 | 0.689 | 0.523 | 0.699 | 0.714 | **0.720** | 0.720 | 0.716 |
| `all_pieces_350x350` | 0.188 | 0.172 | 0.497 | 0.690 | 0.488 | 0.706 | 0.720 | 0.721 | **0.722** | 0.722 |
| `all_pieces_rect` | 0.000 | 0.000 | 0.444 | 0.641 | 0.453 | 0.633 | 0.663 | 0.669 | 0.667 | **0.669** |
| `compact` | 0.000 | 0.000 | 0.639 | 0.603 | 0.652 | 0.644 | 0.619 | 0.655 | **0.656** | 0.655 |
| `cross_dc_rect` | 0.000 | 0.000 | 0.544 | 0.644 | 0.545 | 0.670 | 0.691 | 0.694 | **0.711** | 0.694 |
| `cross_figure8_tall` | 0.000 | 0.000 | 0.631 | 0.747 | 0.625 | 0.786 | 0.792 | 0.804 | **0.805** | 0.804 |
| `cross_figure8_wide` | 0.000 | 0.000 | 0.645 | 0.720 | 0.679 | 0.741 | 0.761 | 0.761 | **0.769** | 0.761 |
| `dc_figure8_large` | 0.000 | 0.000 | 0.513 | 0.622 | 0.529 | 0.642 | 0.690 | **0.694** | 0.682 | 0.681 |
| `dc_figure8_wide` | 0.000 | 0.000 | 0.520 | 0.589 | 0.518 | 0.628 | 0.667 | 0.691 | **0.691** | 0.691 |
| `default` | 0.334 | 0.312 | 0.696 | 0.605 | 0.732 | 0.697 | 0.704 | 0.747 | **0.751** | 0.747 |
| `plain_wide_racetrack` | 0.000 | 0.000 | 0.617 | 0.641 | 0.630 | 0.709 | **0.736** | 0.719 | 0.720 | 0.720 |
| `switch_cross_rect` | 0.000 | 0.000 | 0.520 | 0.644 | 0.548 | 0.637 | 0.688 | **0.714** | 0.705 | 0.712 |
| `switch_one_siding_wide` | 0.000 | 0.000 | 0.536 | 0.611 | 0.565 | 0.639 | **0.700** | 0.671 | 0.690 | 0.676 |
| `switch_two_sidings_tall` | 0.289 | 0.155 | 0.496 | 0.643 | 0.522 | 0.646 | 0.668 | **0.683** | 0.676 | **0.683** |
| `with_crossing` | 0.000 | 0.000 | 0.576 | 0.671 | 0.646 | 0.733 | 0.711 | 0.737 | 0.737 | **0.737** |
| `with_double_crossover` | 0.112 | 0.116 | 0.592 | 0.633 | 0.579 | 0.691 | 0.691 | 0.691 | **0.721** | 0.720 |
| `with_double_crossover_narrow` | 0.000 | 0.000 | 0.510 | 0.499 | 0.575 | 0.613 | 0.677 | 0.692 | **0.693** | 0.692 |
| `with_double_crossover_small` | 0.000 | 0.000 | 0.513 | 0.630 | 0.558 | 0.658 | **0.716** | 0.707 | 0.708 | 0.707 |
| `with_switches` | 0.000 | 0.000 | 0.619 | 0.670 | 0.679 | 0.681 | 0.725 | **0.730** | 0.726 | 0.730 |
| `with_switches_and_crossing` | 0.230 | 0.000 | 0.532 | 0.658 | 0.559 | 0.684 | **0.701** | 0.695 | 0.689 | 0.696 |

**No arm dominates.** Per-config winners: `full_minus_epsilon` 9, `full_minus_elites` 6,
`full_minus_repair` 4, `full` 3 — and in most rows the top four sit within 0.01 of each other,
i.e. inside the noise four seeds can resolve. What the table does separate cleanly is the tier
below: **every arm carrying heuristic seeding lands in 0.50–0.81, every arm without it in
0.00–0.73**, beaten in 20 of 21 configs. `compact` is the exception — a 24-piece kit every
capable arm drives to 100% utilization, so the block is saturated.

---

## 5. Findings

### 5.1 Seeding carries the system; repair reaches the same place, later

Either one **alone** lifts coverage from 5/21 configs to 21/21. Isolated gains sum to +1.098
while the full system only reaches +0.660 over baseline — heavily sub-additive. They solve the
same blocker (§5.7: sizing a closed loop to the box) by different routes.

| | isolated | leave-one-out | median utilization |
|---|---|---|---|
| heuristic seeding | +0.592 | **+0.138** (81/3, p=5e-21) | 55.6% alone, 58.9% in `full` |
| repair pipeline | +0.506 | +0.009 (58/26) | 39.6% alone — lowest of any arm |

Removing seeding costs 15x more than removing repair and drops median utilization to 40.8%,
back to what repair achieves on its own. Repair reaches feasibility but not quality.

**The two are interchangeable in outcome, not in mechanism.** The operators do break closed
layouts; what differs is who puts them back. Mean feasible share of the population:

| arm | operators / repair | gen 1 | gen 2 | gen 10 | final |
|---|---|---|---|---|---|
| `full` | custom / **yes** | 22.8% | 33.1% | **94.0%** | 100% |
| `full_minus_repair` | custom / **no** | 22.5% | 32.5% | **63.3%** | 100% |
| `seeding` | stock / no | 22.5% | 23.7% | 53.5% | 100% |
| `repair` | stock / yes | 0.0% | 0.0% | 6.3% | 97.4% |
| `baseline` | stock / no | 0.0% | 0.0% | 0.0% | 16.7% |

With identical operators, repair lifts generation-10 feasibility from 63.3% to 94.0% — that gap
is breakage being mended. Without it the broken offspring are simply infeasible and die under
feasibility-first survival while the closed parents carry on, so the population reaches 100%
either way. **Repair buys speed to feasibility, and at 200–500 generations that speed has
stopped mattering** — which is the whole of its +0.009. Under a short budget the same
measurement would come out very differently.

The gen-2 column also qualifies "seeding alone is enough": stock operators barely move the
seeded population (22.5 → 23.7) while the custom ones move it (22.5 → 32.5). Part of why the
seeds survive untouched is that the stock operators have almost no power to disturb them
(§5.2).

One more caveat on the isolated reading: the `seeding` arm has `repair=False`, so its seeds are
never repaired — whereas in `full`, pymoo repairs the **initial** population too
(`pymoo/core/initialization.py:36`). "Seeding alone" and "seeding inside the system" are not
the same seeds.

### 5.2 Custom operators are complementary — and the comparator is weak

In isolation they are harmful: 4/21 configs solved by median HV against stock pymoo's 5/21
(7/21 each counting configs with at least one solved seed). Measured leave-one-out inside the
working system they are worth **+0.034 mean HV and win 83 of 84 blocks** (p=4e-24), concentrated
on special-piece configs — median utilization, `full` vs `full_minus_operators`:

```
switch_cross_rect             58.5% vs 45.1%   (+13.4pp)
with_double_crossover         69.8% vs 59.3%   (+10.5pp)
switch_two_sidings_tall       68.6% vs 61.3%    (+7.3pp)
default                       76.4% vs 69.5%    (+7.0pp)
with_double_crossover_narrow  42.2% vs 35.6%    (+6.6pp)
switch_one_siding_wide        59.5% vs 53.0%    (+6.5pp)
```

`full` leads utilization in 9 of 21 configs and ties in the other 12; where it does not raise
the peak it still improves the front's interior (median archive 52 vs 41).

**But +0.034 is measured against a partly inert comparator.** `SBX(eta=15)` / `PM(eta=20)` are
NSGA-II's *constructor* defaults — continuous-problem settings. pymoo's own discrete recipe is
`eta=3.0, prob=1.0`. On this 454-gene genome:

```
PM eta=20 : 0.0402 genes changed per offspring (eta=3 would change 32.9% of genes)
  main_flips  212 genes, span 1 -> 0.00000 changes.  Never.
  descriptor active bits, span 1 -> likewise zero
SBX eta=15: 5.4 novel gene values per child, of which main_types 0.03, flips 0.0
```

PM cannot cross the threshold on a binary gene, and SBX between parents holding 0 and 1
produces no third value. So **the 212-gene R40 handedness array and every descriptor `active`
bit are recombination-only for the whole run** — diversity there can only decay. Handedness is
the gene that decides whether a loop closes.

This also explains the study's oddest result. `PartitionedMutation` has no sub-operator for the
start-position genes (`src/operators.py:137` — they are only swapped by crossover), so the
`operators` arm freezes a start offset drawn from ~U[−250,250] for the entire run, with nothing
able to move it; stock PM at least nudges it 0.4% of the time. **`operators` scoring below
`baseline` measures a sampler/operator interaction, not the operators.**

### 5.3 "Repair" is the wrong name for what the flag removes

Three of `TrackRepairPipeline`'s four stages have equivalents inside the decoder, which runs in
every arm (§2). What `repair=False` actually removes is:

- `MainLoopClosureRepair` (angular + translational),
- `BoundaryAwareRepair`'s shrink branch,
- the **Lamarckian write-back** of clamps into the genome.

So the isolated +0.506 is the value of *closure repair, shrink and inheritance*, not of
"repair" as a category. The marginal +0.009 says those three add almost nothing **by the end of
a long run** — §5.1 shows they are worth 31 percentage points of feasibility at generation 10.
`full_minus_repair` still finishes at 84/84 and median utilization 58.8%, against `full`'s 58.9%.

**The effect is small and borderline.** Over 84 blocks it is significant (58/26, p=0.0003), but
those blocks are not independent — four seeds share a config. Collapsed to a per-config majority
it is **14/6, p=0.058**: short of conventional significance, and closer to it than the
three-seed reading (p=0.095) was. Repair leads 4 configs outright.

### 5.4 Adaptive epsilon: the adaptation never engaged

`full_minus_epsilon` ties `full` on mean HV (0.711 both), takes the most blocks outright of any
arm (33), and costs 31.6 s per run less. But the null result does not say what it appears to.

`src/algorithm/runner.py:572` — `epsilon_0` is hard-capped at `SOFT_CONSTRAINT_COUNT = 4`:

```python
    return min(max(order_stat, floor), float(self.n_soft))
```

Measured across the archive: in **420 of 420** cells with epsilon on, `cv_eps` starts at exactly
4.000 — the cap, never the population percentile. Initial-population CVs run 26–765. So the
"adaptive" calibration is saturated in every run: what shipped is a constant 4 on a decay
schedule, not a schedule fitted to the population.

Two consequences. `full_minus_epsilon` measured the epsilon relaxation **together with** the
x1000 hard-constraint CV weighting (both are set in the same method,
`src/algorithm/runner.py:618`), and the epsilon half was pinned at its ceiling throughout.
Splitting the flag in two will therefore not price the schedule either — the cap has to be
raised, or the soft-G normalization rescaled, before it is measurable at all.

### 5.5 Elite injection is not measurable under this metric

The scored quantity is `pareto_archive.csv` — the **run-cumulative** front of every feasible
point ever seen, written by a monitor attached in every arm, outside the `elite_injection`
guard. That metric already preserves what the elite callbacks exist to preserve inside the
population, so the component has little left to buy. p=0.70 should be read as *not measurable
this way*, not as *does nothing*.

The callbacks are sometimes a literal no-op: **22 of the 84 blocks have byte-identical
`fitness.csv` between `full` and `full_minus_elites`** — with the seed fixed, removing them
changed nothing, because the elite was already in the population. Neither callback consumes
RNG, so a no-op leaves the stream untouched and the run replays exactly.

**A fourth seed retired the one mechanism this study had proposed.** On three seeds
`all_pieces_rect` looked like evidence that re-injection costs exploration: `full` returned the
identical champion on all three seeds while `full_minus_elites` found a much better one. The
fourth seed reverses the roles:

```
all_pieces_rect   full              s1/s2/s3  119 pcs 58.5% 0 sw    s4  143 pcs 74.5% 2 sw
                  full_minus_elites s1        131 pcs 68.9% 2 sw    s2/s3/s4  119 pcs 58.5% 0 sw
```

Both arms mostly land on the same 119-piece layout and each produced one better outlier on one
seed — and `full`'s outlier is the better of the two, and the best layout any arm found on that
config. What looked like "the full system is not exploring" was a three-draw artifact. There is
now **no evidence of a mechanism** behind the elite null, only the null itself.

### 5.6 Cost, and whether it pays for itself

Sum of `gen_seconds` per run over **seeds 1–3**; seed 4 is excluded because it was measured in a
different session (§3). Validated against the driver's own `elapsed_s` on the 453 cells where
both exist (mean ratio 0.986).

| arm | mean s/run | vs baseline | mean HV (4 seeds) |
|---|---|---|---|
| `baseline` | 68.4 | 1.00x | 0.051 |
| `seeding` | 70.3 | 1.03x | 0.643 |
| `operators` | 89.5 | 1.31x | 0.042 |
| `repair` | 102.8 | 1.50x | 0.557 |
| `full_minus_repair` | 117.2 | 1.71x | 0.702 |
| `full_minus_operators` | 120.3 | 1.76x | 0.677 |
| `full_minus_epsilon` | 131.5 | 1.92x | **0.711** |
| `full_minus_seeding` | 138.4 | 2.02x | 0.574 |
| `full_minus_elites` | 153.2 | 2.24x | 0.710 |
| `full` | 163.1 | 2.38x | **0.711** |

| component | extra s/run | marginal HV | HV per extra second |
|---|---|---|---|
| heuristic seeding | +24.7 (+18%) | +0.138 | **0.0056** |
| custom operators | +42.8 (+36%) | +0.034 | 0.00079 |
| repair pipeline | +45.9 (+39%) | +0.009 | 0.00020 |
| elite injection | +9.9 (+6%) | +0.001 | 0.00010 |
| adaptive epsilon | +31.6 (+24%) | −0.000 | negative |

Seeding is 7x more compute-efficient than the operators and 28x more than repair; bolted onto
the bare baseline it returns 0.31 HV per second, two orders of magnitude beyond anything else.
Epsilon, elites and repair together account for 88 s of `full`'s 163 s. That their removal is
jointly harmless is a hypothesis, not a measurement — no arm removed more than one component.

### 5.7 Stock pymoo's failure mode is the boundary, and it is barely doing multi-objective search

Where `baseline` fails, the closest individual usually violates one constraint block and it is
almost always the boundary — closure is already satisfied. Denormalized from
`G[3] = (violation − boundary_tolerance) / diagonal`, seed 1:

```
all_pieces              closure OK,  90.5 studs outside the box
dc_figure8_large        closure OK, 125.2
switch_one_siding_wide  closure OK,  56.7
with_switches           closure OK,  54.4
dc_figure8_wide         closure OK,  34.2
```

Counting configs where the *fewest-violated-blocks* individual violates exactly one, that is 12
of the 16 failed configs — 10 boundary-only, 2 closure-only (`compact`, `with_crossing`). Under
a *lowest-CV* reading it is 8 of 16. Both readings are given because the count depends on which
one is used.

**Stock pymoo can close loops; it cannot size them to the box.** Part of that is a bounds trap
rather than a search failure: start-position genes are bounded by the whole box
(`src/encoding.py:313`), and the custom sampler deliberately draws them from ±5% of the extent
instead. Stock sampling at ~U[−250,250] costs mean boundary CV +0.19.

Worse, with zero feasible individuals `RankAndCrowding` (`filter_infeasible=True`) degenerates
to CV-ascending truncation — `F` plays no part in survival — and NSGA-II's tournament also
routes by CV whenever either parent is infeasible. Since `baseline` never reached feasibility
in 68 of 84 cells, **most baseline runs were single-objective constraint minimizers**, not
NSGA-II. Against `ConstrRankAndCrowding` this differs by 17% of survivors.

---

## 6. Threats to validity

- **The decoder repairs in every arm** (§2). The baseline is flattered by inventory clamping,
  descriptor validation, auto-centering and crossing repair; the `repair` component is
  correspondingly understated. This is the largest unstated confound in the study.
- **Component value is budget-dependent** (§5.1). Repair's contribution is concentrated in the
  early generations; measured at the end of a 200–500 generation run it looks negligible. Every
  marginal number here is "worth this much at this budget", not "worth this much".
- **Seven of the twelve constraints are structurally dead.** The decoder enforces inventory
  before `_evaluate` runs, so `G[5..11]` are identically zero — verified across all 840 cells.
  This is a 5-constraint problem; `InventoryRepair` can never reduce a violation, and the
  x1000 hard weighting has exactly one live target, collisions.
- **The stock operator arm cannot mutate ~47% of the genome** (§5.2). Claims of the form
  "stock pymoo cannot solve this" should read "stock *continuous* operators at NSGA-II's
  default `eta` cannot move this genome".
- **`constr_survival` has no arm of either kind.** It is off in `baseline`/`seeding`/
  `operators`/`repair` and on in every `full`-family arm, so the isolated and leave-one-out
  columns are measured under different survival regimes, and the whole `full > baseline` gap
  silently contains it.
- **Blocks are not independent.** Four seeds share a config; the config-level column in §1 is
  the conservative reading. Only `full > full_minus_repair` changes verdict under it.
- **Four seeds per cell.** Enough for the pooled sign tests, not for per-config inference: with
  four observations the exact-test floor is p = 0.0625 for any single config. §4's per-config
  winners, separated by <0.01 HV, are noise — and §5.5 shows what a fourth draw can do to a
  single-cell story.
- **Null results are "not detected", not "proven zero".** No equivalence test was run.
- **Wall-clock spans two measurement sessions.** Seeds 1–3 ran uninterrupted; seed 4 ran on a
  different day across several stop/resume cycles, and its per-arm means differ by up to
  +44.5 s (`full_minus_elites`) with no corresponding HV change. Cost therefore comes from
  seeds 1–3 alone. Per-cell logging, added for seed 4, is not the cause: ~22 records and 2 KB
  per run, emitted outside the generation loop that `gen_seconds` measures.
- **Configs are the project's own development set**, several co-designed with the heuristic
  seeds. The arm that wins hardest is the one whose seed patterns those boxes were shaped
  around. Results do not generalize beyond this instance family.
- **Arms are compared at equal generations, not equal wall-clock.** Reported rather than
  corrected: the cost ratio is an outcome of each arm's behaviour, so any equalizing factor
  would have to be chosen after seeing the runs. Cost sits next to quality in §5.6 instead.
- **`eliminate_duplicates` retries mating up to 100 times per generation** in every arm. Since
  stock PM changes 0.04 genes per offspring, the baseline burns far more retries than `full` —
  an uncounted diversity mechanism whose effective strength differs per arm. Instrumentation
  shows repair nearly doubles the mating attempts (2546 vs 1345 crossover calls), which is part
  of what its +45.9 s buys.

### Provenance caveats

- **Seeds 1–3 ran on an uncommitted working tree that changed mid-campaign**, and `arm` is the
  outermost loop in the driver, so each arm's cells ran consecutively and arm ≡ source state ≡
  time block. Three source states cover those seeds; the deltas were recovered from the diffstat
  embedded in each `run_info.md` and are inert. One cell (`compact/baseline/s1`) ran on a state
  with no snapshot.
- **One cell does not reproduce byte-for-byte** under the current tree
  (`compact/full_minus_repair/s1`): generations 1–3 are identical, the populations diverge from
  generation 4, and the final optimum is the same (100% utilization, 2.389 s). Unexplained. A
  full-component cell re-run under a fixed tree does reproduce exactly, so the pipeline is a
  pure function of (config, seed) when the source does not move.
- **Seed 4 carries a per-cell `run.log`**; seeds 1–3 do not, because the driver had no file
  handler then. For those seeds the component state is recoverable from `arm.json` plus artifact
  traces: `cv_eps` and `n_eval` for epsilon, `category_report.md` for elites, generation-1
  feasibility for seeding.
- **`outputs/ablation/manifest.json` is a lossy append log** — written once at the end of each
  invocation, so interrupted runs lose their rows. Its key set reconciles with the archive, and
  the scorer never reads it.

---

## 7. Open work

| item | cost | what it buys |
|---|---|---|
| re-run the stock arm at pymoo's documented discrete settings (`eta=3.0, prob=1.0`) | ~3 h | a fair comparator; every claim about `baseline`/`operators`/`full_minus_operators` currently rests on a partly inert one |
| score the existing archive at generations 25/50/100 as well as the end | scoring change only | prices every component at short budgets, where §5.1 shows repair is doing real work |
| an arm that disables the decoder's internal clamps | design + ~3 h | separates "repair" from "the decoder repairs anyway" (§2, §5.3) |
| raise the `epsilon_0` cap or rescale soft-G, then re-measure | small change + ~3 h | the schedule is saturated in 100% of runs, so it has never been tested (§5.4) |
| `full_minus_survival` arm | ~3 h | the only component with no reading of either kind |
| a population-based elite metric, or an arm scored on the final population | scoring change | `elite_injection` is unmeasurable against a cumulative archive (§5.5) |
| combined removal, `full_minus_{epsilon,elites,repair}` | ~2.5 h | tests whether three separately-null components are jointly null |
| held-out instances generated by a stated rule | ~2.5 h | an external-validity claim the development config set cannot support |