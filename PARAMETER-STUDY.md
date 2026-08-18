# Parameter Study — Design, and the Parameters It Can Reach

The component ablation (`ABLATION-STUDY.md`) asked what each custom component is worth. This
asks a different question: **how sensitive is the full system to its own parameters?** Every
component flag stays `True`; `AlgorithmConfig` fields are what varies.

It is not a fallback. The original spec had two branches — measure the components if stock pymoo
works, compare method parameters if it does not — and stock pymoo does work (5/21 configs). So
this is an optional second study, not a replacement, and the component ablation is not re-run.

The inventory below comes from reading the repository and the installed pymoo 0.6.1.6, not from
runs. It says what **can** be varied and what each value **means**, never which value is better.

---

## 1. Three findings that change how any operator result reads

### 1.1 Mutation is far weaker than the config number suggests

`src/operators.py:674` — the probability pymoo receives applies **per individual**
`src/operators.py:692` — each selected individual gets **at most one** operation

`mutation_prob: 0.8` means "80% of individuals receive one small edit" — one piece type swapped,
one curve flipped — not "80% of genes mutate". As genetic algorithms go that is very gentle,
which is probably why some configs push it to 0.8 while others drop to 0.05. If the slider
enters a grid, report it as the share of individuals receiving a single edit, never as mutation
strength.

### 1.2 Crossover and mutation are set as one slider

All 21 configs set `crossover_prob + mutation_prob = 1.0`, which pymoo does not require:

| values | configs |
|---|---|
| 0.2 / 0.8 | `default`, `with_switches`, `with_double_crossover_narrow`, `with_double_crossover_small` |
| 0.8 / 0.2 | `all_pieces_rect`, `switch_cross_rect`, `switch_one_siding_wide`, `switch_two_sidings_tall`, `with_switches_and_crossing` |
| 0.9 / 0.1 | `all_pieces`, `all_pieces_350x350`, `compact`, `cross_dc_rect`, `dc_figure8_large`, `dc_figure8_wide`, `plain_wide_racetrack`, `with_double_crossover` |
| 0.95 / 0.05 | `all_pieces_150x150`, `cross_figure8_tall`, `cross_figure8_wide`, `with_crossing` |

pymoo treats the two probabilities independently. Someone adopted the convention, and the spread
— from "almost only crossover" to "almost only mutation" — has no measurement behind it. That
makes the pair the strongest remaining candidate for a factor: already in the config, no code
change, and the disagreement between configs demands an explanation.

### 1.3 The seven-operator portfolio never fires for switch or DC configs

Mutation branches three ways:

`src/operators.py:698` — DC descriptor active: 50% compensated grow, 25% figure-8 regrow, 25% untouched
`src/operators.py:709` — siding active: 10% a junction operation, otherwise always the compensated grow
`src/operators.py:717` — everything else: 10% junction, otherwise one of seven main-loop operators by weight

The seven weights in `_MAIN_LOOP_WEIGHTS` (`src/operators.py:634`) therefore apply to the third
case only. For `with_switches`, `switch_*` or `dc_figure8_*`, tuning them changes nothing — the
single number steering mutation there is that 10%. The narrowing is deliberate and documented in
the code: the other operators break loop closure around a switch, and editing a DC descriptor
almost always ejects it from the layout.

---

## 2. Parameter inventory

### 2.1 Available now — in the config, no code change

| parameter | what it does | note |
|---|---|---|
| `pop_size` | how many layouts the GA holds at once | in the grid (§3.4) |
| `n_gen` | how many rounds of improvement it runs | in the grid |
| `crossover_prob` / `mutation_prob` | the slider between crossover and mutation | see §1.1, §1.2 |
| `heuristic_ratio` | share of the initial population that is a seed pattern | hard ceiling 0.5 (`src/config.py:93`, `le=0.5`) |
| `eliminate_duplicates` | drop repeated solutions | always on today |
| `termination.period` | stop early on stagnation | campaigns force 0 for equal budgets |

Where the runner reads them: `src/algorithm/runner.py:675` (`heuristic_ratio`), `:681`
(`crossover_prob`), `:682` (`mutation_prob`), `:740` (`eliminate_duplicates`).

`heuristic_ratio = 0.0` is **not** `heuristic_sampling = False`: it keeps the custom partial-fill
random generator and removes only the seed patterns.

### 2.2 Hardcoded — one or two lines of pass-through to expose

`src/sampling.py:706` — random-individual fill, `rng.uniform(0.5, 0.8)`: a random layout occupies
50–80% of the available slots, and nobody has checked whether that range is right
`src/sampling.py:725` — each switch slot in a random individual is a coin flip
`src/sampling.py:728` — siding length in a random individual: 0 to at most 5 straights
`src/operators.py:690` — `junc_thresh = 0.10`, the share of mutation aimed at junctions rather
than the main loop; for switch configs this is the only number steering mutation
`src/operators.py:121` / `:129` / `:136` — three 50% rates deciding how strongly children mix the
parents' DC descriptors, junction descriptors and start position. Never changed.
`src/algorithm/runner.py:691` / `:692` — `enable_closure_repair` and `enable_boundary_repair`,
both wired `True`. The ablation switches repair on and off as a whole; splitting it costs two
lines and answers a question nobody has asked — which stage of repair actually works.
`pymoo/algorithms/base/genetic.py:41` — `n_offsprings` defaults to `pop_size`. A smaller value
gives steady-state renewal instead of full replacement, but it changes evaluations per
generation, so the budget stops being `pop_size x n_gen`.

The epsilon schedule rests on five numbers, none of them in a config:
`src/algorithm/runner.py:756` — `hold_until=0.2`, `perc_eps_until=0.9`
`src/algorithm/runner.py:536` — `theta=0.2`, `ratchet_trigger=0.25`, `ratchet_cooldown=5`
`src/algorithm/runner.py:449` — `HARD_CONSTRAINT_WEIGHT = 1000.0`

Note that the epsilon calibration is saturated in every archived run (`ABLATION-STUDY.md` §5.4),
so exposing these numbers is a prerequisite for measuring the schedule at all, not a tuning
opportunity on top of a working mechanism.

### 2.3 Not settable today

**Seed composition, as opposed to seed count.** Patterns from all nine families are shuffled and
then dealt cyclically, so every family gets an equal share whether or not it suits the config —
a DC config spends most of its seed budget on ovals and racetracks.

`src/sampling.py:647` — `patterns[i % len(patterns)]`

"More patterns of the right family" is a different question from `heuristic_ratio` and needs a
new config field plus a change in the generator. `ABLATION-STUDY.md` §5.9 measures how skewed the
current allocation is: in `all_pieces`, `oval_two_sidings` contributes 16 variants and claims 34%
of the seed rows while `figure_eight` contributes 2 and claims 4%.

**The seven mutation weights.** Seven numbers summing to one do not make a grid factor. Disabling
one operator at a time and watching what breaks is the sensible design — a small operator
ablation, not the tuning of a single number.

### 2.4 Do not touch

`src/problem.py:33` — `SPEED_SAFETY_MARGIN = 0.95`, together with `special_piece_weight` and the
closure / angle / boundary tolerances, defines the **problem**, not the method. Changing any of
them moves the objectives and the feasibility line, so cells stop sharing one scale and their
hypervolumes stop being comparable (§4). Inventory and boundary are externally given constraints.

Selection pressure looks like a parameter but is not available: pymoo's tournament is written for
a two-way comparison and raises on anything else.

`pymoo/algorithms/moo/nsga2.py:25` — `raise ValueError("Only implemented for binary tournament!")`

`crowding_func` is **settled, not swept**: pymoo recommends `pcd` for two-objective problems, it
was run as a paired second axis over every arm, and it never wins (`ABLATION-STUDY.md` §5.8). A
grid corner would only re-ask an answered question on fewer configs.

---

## 3. Design

### 3.1 The budget confound — decided

**`n_gen` is a swept factor.** The spec names "population size, max iterations etc." as the
parameters to compare. The standing rule against shortening runs is about making a campaign
cheaper; it does not apply when the generation count is itself the variable. Holding
`pop_size x n_gen` constant is therefore allowed.

Report, do not hide, the coupling: both factors scale the evaluation budget, so print total
evaluations in every table and read a `pop_size` result as "this many evaluations arranged this
way", never as "a bigger population is better".

`n_gen` is **not** a pure budget knob here. The epsilon schedule's phase boundaries are fractions
of the *planned* generation count (`hold_until=0.2`, `perc_eps_until=0.9`, passed at
`src/algorithm/runner.py:756`), driven by `n_gen_planned` rather than elapsed generations, and
snapshot targets rescale the same way. Halving `n_gen` halves the absolute length of the strict
phase while keeping its relative position — a short-`n_gen` cell is not a truncated long one.

### 3.2 One factor at a time, or factorial

OFAT is a defensible screening design and is cheap, but it cannot see interactions, and two are
guaranteed here: `pop_size` x `n_gen` through the budget, and `heuristic_ratio` x `pop_size`
through seed count (seeds = ratio x pop_size). A 2^3 factorial on the three most interesting
factors costs 8 cells x 3 seeds = 24 runs per config and yields main effects **and** all two-way
interactions — often cheaper and stronger than a wide OFAT grid.

### 3.3 The grid — two knobs, four corners

Population size and generations, each at half and double that config's own production value. For
`configs/default.yaml` (`pop_size: 1000`, `n_gen: 200`): 500 or 2000 layouts, 100 or 400 rounds.

| population | generations | cost vs one production run |
|---|---|---|
| 500 | 100 | 0.25x |
| 500 | 400 | 1x |
| 2000 | 100 | 1x |
| 2000 | 400 | 4x |

6.25x one production run per config per seed, 18.75x over three seeds. Levels are per config,
never global: `configs/dc_figure8_wide.yaml` runs 600 x 250, so its corners are 300/1200 and
125/500.

**Why four corners rather than one knob at a time.** Sweeping population alone tells you a bigger
population helps, but not whether it helps on its own or only when it also gets enough rounds.
The corners separate the two: if 2000 x 100 does poorly while 2000 x 400 does well, a large
population is useless without the time to work it.

**Why the corners are not cost-equalised.** Keeping only the equal-work pairs (500 x 400 and
2000 x 100 are both 200k evaluations) would make generations a consequence of the population
choice rather than a knob of their own. Let the cost differ and print total evaluations.

**The centre is already computed.** Production settings sit at the centre, and the ablation's
`full` arm ran exactly that point on all 21 configs and every seed, with `period = 0` and every
component on. Reuse those archives as centre cells under two conditions: every other setting must
match (check the resolved `arm.json` the driver writes, do not assume), and all cells of a config
must be scored in one shared box (§4).

**`pop_size` and `heuristic_ratio` are trustworthy levels — audited.** pymoo deduplicates the
repaired starting population without refilling, so generation 1 runs below `pop_size` in 19 of 21
configs. That is where it ends: the population is back to size from generation 2, verified across
379 archived runs, and the deleted individuals are bit-identical duplicates
(`ABLATION-STUDY.md` §5.9). Use the configured values as levels; do not build the grid around a
"delivered vs nominal" distinction that does not exist past the first generation.

### 3.4 Config selection and cost

Do not use all 21 configs — the component ablation covers breadth, this study needs depth. Pick a
set spanning the behaviours in `ABLATION-STUDY.md` §5: at least one config where the GA genuinely
improves on its seed (`all_pieces_rect`, `default`, `switch_cross_rect`) and one where the seed
is already the champion (`dc_figure8_wide`, `cross_figure8_wide`).

Measured single-run wall-clock at production settings — use these to budget, do not guess:

```
default 0.5 min   compact 0.7   plain_wide_racetrack 1.1   switch_one_siding_wide 1.5
switch_two_sidings_tall 1.7   dc_figure8_wide 2.0   cross_figure8_wide 2.0
cross_figure8_tall 2.1   all_pieces 4.0   with_crossing 4.0   with_switches 11.1
dc_figure8_large 40.6
```

`dc_figure8_large` has a measured pathology: seconds per generation grow ~15x **within a single
run** (0.71 → 10.66) as layouts grow, because the self-intersection scan is quadratic in piece
count. Avoid it unless the study is specifically about cost.

### 3.5 Order to extend in

1. **The crossover/mutation slider** — already in the config, the largest unexplained spread, and
   free to expose.
2. **`heuristic_ratio`** — seeding is the dominant component, so seed quantity matters; note that
   seeds = ratio x `pop_size`, so it is coupled to a factor already in the grid.
3. **The random fill range** (`rng.uniform(0.5, 0.8)`) — one line, and nobody knows whether the
   range came from analysis or from habit.

For switch and DC configs swap the third for `junc_thresh`, since the other mutation operators
never run there (§1.3).

---

## 4. Metrics and analysis

- **Primary metric: hypervolume.** Per config, `ideal`/`nadir` over the union of **all cells of
  that config**, `has_extent` guard, then
  `HV(ref_point=HV_REF_POINT, norm_ref_point=False, zero_to_one=True, ideal=, nadir=)`. Helpers
  live in `src/normalization.py`. Do not skip the `has_extent` guard — a single-point archive
  self-normalized against `[1.1, 1.1]` scores **1.21**, the maximum possible, and single-point
  archives are common here.
- **Never pool raw HV across configs**: `special_piece_weight` differs, so F[0] is on different
  scales. Across configs pool signs and ranks only.
- **Never mix normalization boxes in one table.** Adding cells later means scoring them against
  the frozen box or recomputing and re-reporting everything.
- **Front size is not a quality metric.** Measured: one arm produced 827 distinct points against
  a competitor's 70 and still lost on HV.
- **Report cost next to quality.** Do not build "equal-CPU" cells by scaling a budget with a
  ratio measured from the runs — the ratio is an outcome of behaviour, not a parameter, and
  choosing which one to use is a researcher degree of freedom.
- **Statistics.** Unit of analysis = one run; block = (config, seed index). Three seeds put the
  exact-test floor at p = 0.125 per config, so no per-config comparison can reach significance —
  pool blocks across configs or make no inferential claim. Seeds are matched by index, not common
  random numbers: different cells consume the RNG stream differently, so this is blocking, not
  pairing.

---

## 5. Implementation notes

**Reuse `run_ablation.py`'s shape, do not extend it.** Its `ARMS` table maps arm names to
component flags, which is the wrong axis for parameters. What it already solves and a parameter
driver needs: loading each config from its **original path** so the relative
`train_config_path: trains/measured_consist.yaml` resolves (materializing a patched copy raises
`FileNotFoundError`; dropping the key silently substitutes default physics, `v_motor_max` 1.1
instead of the measured 1.26, and corrupts F[1] with no error); writing the **resolved** settings
to `<run_dir>/arm.json` read back off the loaded model, because pydantic silently ignores a
misspelled key; the resume-safe skip on `"## Run Summary"`; forcing `termination.period = 0`; and
the `if __name__ == "__main__":` guard, mandatory on Windows with spawn plus `Pool`.

**Output isolation is required.** Write to `outputs/parameters/...`, never `outputs/ablation/...`,
with your own manifest — `run_ablation.py` does a read-modify-write of its manifest at the end of
a run and would clobber shared rows. Keep everything under the single `outputs/` tree.

**Our operators satisfy pymoo's contracts**, checked against the installed 0.6.1.6 base classes
rather than the docs, which show usage but not requirements. Crossover returns the expected shape
(`pymoo/core/crossover.py:51`) and pairs that fail the probability draw are copied through
unchanged (`:66`). Mutation is drawn once per individual (`pymoo/core/mutation.py:32`), and our
in-place edit is safe because the library hands over a copy (`pymoo/core/population.py:72`).
Repair conforms: the chain calls each stage's internal method rather than the public one, which
is correct — the public method expects a whole population, the chain passes the array
(`pymoo/core/repair.py:13`).

**At low mutation probability most of that work is discarded.** pymoo mutates the **whole**
population and only then draws which individuals keep the change. That is cheap for vectorized
library operators; ours walk individuals one at a time and some are expensive — the costliest
straightens track near a self-crossing, which means building the layout and comparing every
segment with every other, quadratic in piece count. At `mutation_prob: 0.05` (`with_crossing`,
both `cross_figure8_*`) roughly 95% of that work is done and thrown away. It can be avoided by
setting the probability to 1.0 and moving the draw inside our operator — statistically identical,
but it changes the meaning of the number in every config, so not during a parameter study.

---

## 6. Traps already measured

Verified with literal output during the component ablation; `ABLATION-STUDY.md` §3 lists the ones
that bite when reading its archive. Three more matter to anyone writing a driver or a scorer:

- `RoundingRepair._do` returns `np.around(X).astype(int)`, so stock-operator populations carry
  **int64, never float** — no int16 cast is needed at the decoder entry.
- `compute_closure_metrics` returns `angle_error = 360.0` when total turning is exactly zero, so
  `max_angle_error` must never be used as a closure criterion.
- Archives written before commit `69e04b7` store a **negated speed** in F[1], not seconds. Reject
  any archive row with `f1 <= 0` so a stale directory cannot poison a normalization box.

---

## 7. Limitations of this inventory

- It is code analysis, not measurement. None of these numbers was checked by a run; the report
  says what can be varied and what a value means, never which value is better.
- Extreme levels are unvalidated — whether `crossover_prob = 0` stalls the population, for
  instance. Run each extreme once before it enters a grid.
- The wall-clock figures come from existing runs. Configs missing from that table have no
  measurement, and it must not be guessed.