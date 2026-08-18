# Brief — Parameter Study (second ablation track)

**For a fresh instance working in parallel with an in-flight campaign.** Read this
whole file before touching anything. `ABLATION-STUDY-HANDOFF.md` (same directory) is the
completed component ablation and is your main source of measured facts; `CLAUDE.md` holds
the process rules, all of which apply.

---

## 0. HARD CONSTRAINTS — read first

**Do NOT start any optimization run.** A campaign of 252 runs (6 arms x 21 configs x
seeds {2,3}) started 18:52 on 2026-08-10 and was **PAUSED at 21:00 with 99 of 252 cells
done**, to be resumed the next day with the same command:

```powershell
python run_ablation.py --arms baseline,seeding,operators,repair,full,full_minus_operators --seeds 2,3
```

**An idle machine does NOT mean the campaign is over.** It is paused, not finished, and
every constraint below still applies through the resumed run.

Count completed cells (a cell is done only once its `run_info.md` contains a summary):

```powershell
Get-ChildItem outputs/ablation -Recurse -Filter run_info.md |
  Where-Object { (Get-Content $_.FullName -Raw) -match '## Run Summary' } | Measure-Object
```

The campaign is finished when that count reaches 378 (126 seed-1 cells + 252 for seeds
2 and 3). Until then:

- **No runs.** The machine has **20 physical / 28 logical cores** and configs request
  `n_workers` of 16 or 32 — `with_switches` alone already oversubscribes. A second
  campaign would not find free cores.
- **This is not just about speed.** Wall-clock per run (`elapsed_s` in the manifest) is a
  *reported metric* of the component ablation — cost is presented next to quality instead
  of running an artificial equal-budget arm. Concurrent load would invalidate that metric
  for **both** studies.
- **Do not edit anything under `src/`** while the campaign runs — **including while it is
  paused.** `run_info.md` records the git state and working-tree diff per run; an edit
  made during the pause would mean the 153 remaining cells execute different code from
  the 99 already done, which cannot be repaired afterwards in analysis. Verify before
  resuming that the tree still matches the launch snapshot:

  ```powershell
  git diff HEAD | Out-File -Encoding utf8 tmp.patch
  Compare-Object (Get-Content tmp.patch) (Get-Content outputs/ablation/code_state_at_launch.patch)
  ```

  No output means the code is unchanged. Tracked files are the ones that matter here;
  editing an *untracked* file (this brief, `ABLATION-STUDY-HANDOFF.md`, `run_ablation.py`)
  changes neither `git diff HEAD` nor the porcelain line that already lists it, so those
  edits are provenance-neutral and allowed.
- **Do not commit, and do not change `HEAD` in any way** (no commit, checkout, branch,
  reset, stash) until the campaign finishes. This is stricter than the edit rule and for
  a worse failure: an edit only perturbs the "uncommitted changes" section of later
  `run_info.md` files, whereas a commit changes the recorded **base SHA**, so cells run
  before and after it would claim different source revisions while having executed
  identical code. Nothing in this tree is committed and the user has not authorised
  committing.
- **Batch your test runs.** The suite is ~34 s across all cores. Once is harmless;
  running it every few minutes for hours perturbs `elapsed_s` of whichever campaign cell
  is executing, and that column is a reported metric. Accumulate changes and verify in
  larger batches while the campaign is live.

Writing code, designing, and researching are fine — they cost no CPU. Queue the runs.

**Ground truth for what the campaign actually executed** is captured in
`outputs/ablation/code_state_at_launch.md` and `.patch` (HEAD, `git status --porcelain`,
and the full `git diff HEAD` taken just after launch). The campaign process imported its
modules at start, so that snapshot — not whatever `run_info.md` records later — describes
the code behind every cell. Do not delete it.

---

## 1. What this study is, and what it is not

The user's original spec had two branches: (a) if stock pymoo can solve the problem, it
is the baseline and we measure how much the custom components improve on it; (b) if it
cannot and produces garbage, compare method parameters — population size, max iterations
etc. — because that is also an ablation.

**Branch (a) was measured and it works** (stock pymoo solved 5/21 configs). So branch (b)
is **no longer a fallback** — it is an optional, separate study: *how sensitive is the
full system to its own parameters?* Do not frame it as a replacement for, or a competitor
to, the component ablation. Do not re-run the component ablation.

---

## 2. What already exists — reuse, do not rebuild

- `src/config.py` — `SearchComponentsConfig`: six booleans (`heuristic_sampling`,
  `custom_operators`, `repair`, `adaptive_epsilon`, `elite_injection`,
  `constr_survival`), all defaulting to `True`, reached as
  `config.algorithm.components`. For a parameter study every flag stays `True` (the
  full system) — you vary `AlgorithmConfig` fields instead.
- `run_ablation.py` — the component driver. **Read it, copy its shape, do not extend
  it.** Its `ARMS` table maps arm name to component flags, which is the wrong axis for
  parameters. Reuse these solved problems from it:
  - loads each config from its **original path** so the relative
    `train_config_path: trains/measured_consist.yaml` resolves. Materializing a patched
    copy next to the run raises `FileNotFoundError`; dropping the key silently
    substitutes default physics (`v_motor_max` 1.1 instead of the measured 1.26) and
    corrupts F[1] with no error.
  - writes the **resolved** settings to `<run_dir>/arm.json`, read back off the loaded
    model — pydantic silently ignores a misspelled key, so the intended patch is not
    evidence of what ran.
  - resume-safe skip on `"## Run Summary" in run_info.md` (a header alone means the run
    died).
  - forces `termination.period = 0` so an improvement-based early stop cannot hand cells
    unequal budgets. Three configs ship `period: 100`.
  - `if __name__ == "__main__":` guard — mandatory on Windows (spawn + `Pool`).

**Output isolation is required.** Write to `outputs/parameters/...`, never
`outputs/ablation/...`, and use your own manifest file. `run_ablation.py` does a
read-modify-write of `outputs/ablation/manifest.json` at the end of its run; sharing it
would clobber rows. Keep everything under the single `outputs/` tree — parallel
top-level output directories are explicitly disallowed.

---

## 3. Design — factors and grid

Center of every sweep = the config's own production values. Run the center **once** per
config and reuse it across factors.

| factor | why it is interesting | trap |
|---|---|---|
| `pop_size` | exploration vs exploitation | **see §3.1 — this is the one to get right** |
| `n_gen` | is the budget saturated? | interacts with `pop_size` |
| `heuristic_ratio` | how much does seed quantity matter, given seeding is the dominant component | **coupled to `pop_size`**: seed count = ratio x pop_size, so the two are not independent factors |
| (`crossover_prob`, `mutation_prob`) | already treated as a pair in the configs | production values differ wildly per config: `all_pieces` 0.9/0.1, `with_switches` 0.2/0.8, `with_crossing` 0.95/0.05 — so "the center" is per-config, not global |

`heuristic_ratio = 0.0` is **not** the same as `heuristic_sampling = False`: it keeps the
custom partial-fill random generator and removes only the seed patterns. That distinction
is worth a sentence in the write-up.

**`pop_size` and `heuristic_ratio` are trustworthy factor levels — audited.** pymoo's
`Initialization` deduplicates the repaired starting population without refilling, so
generation 1 runs below `pop_size` in 19 of 21 configs (up to 8%). That is where it ends:
`n_offsprings` defaults to `pop_size` and survival takes `n_survive=self.pop_size`, so the
population is back to size from generation 2, verified across 379 archived runs (zero
deviations at any generation >= 2). Deleted individuals are bit-identical duplicates, so no
seed family or phenotype is lost. **Use the configured values as your levels**; if you want
belt and braces, note the generation-1 count, but do not build the grid around a
"delivered vs nominal" distinction — it does not exist past the first generation.
See `ABLATION-STUDY-HANDOFF.md` §5.5.

**Out of scope — settled by the component-ablation track:** `crowding_func` on the
survival operator. pymoo's own survival docs recommend `'pcd'` for two-objective problems
(this problem has `n_obj=2`), so it was run as a paired second axis over every arm — 10
arms x 21 configs x 3 seeds x {cd, pcd}, 1260 cells. **`pcd` is not better:** it loses
more blocks than it wins in eight arms of ten, mean HV deltas span -0.016..+0.003, no arm
reaches `p < 0.5`, and compute cost is equal. `cd` remains the production default. See
`ABLATION-STUDY-HANDOFF.md` §8.1. **Do not put it in the parameter grid** — the question
is answered, and a grid corner would only re-ask it on fewer configs.

### 3.1 The budget confound — DECIDED, do not re-open

**`n_gen` is a swept factor.** The user's spec names "population size, max iterations
etc." as the parameters to compare, and confirmed it directly: max iterations is an
ablation parameter like any other. The standing rule against shortening runs is about
making a campaign cheaper — it does not apply when the generation count is itself the
variable under study. Holding `pop_size x n_gen` constant is therefore allowed.

Report, do not hide, the budget coupling: `pop_size` and `n_gen` both scale the
evaluation budget, so print total evaluations in every table and read a `pop_size`
result as "this many evaluations arranged this way", never as "a bigger population is
better".

`n_gen` is **not** a pure budget knob in this system. The epsilon schedule's phase
boundaries are *fractions* of the planned generation count (`hold_until=0.2`,
`perc_eps_until=0.9`, passed at `src/algorithm/runner.py:756`), and the schedule is
driven by `n_gen_planned`, not by elapsed generations. Halving `n_gen` therefore halves
the absolute length of the strict-feasibility phase while keeping its relative position,
and snapshot targets rescale the same way. A short-`n_gen` cell is not a truncated
long one — say so in the limitations section.

### 3.2 OFAT vs factorial

One-factor-at-a-time is defensible as a screening design and is cheap. It cannot see
interactions, and at least two are guaranteed here (`pop_size` x `n_gen` through the
budget; `heuristic_ratio` x `pop_size` through seed count). A 2^3 factorial on the three
most interesting factors costs 8 cells x 3 seeds = 24 runs per config and yields main
effects **and** all two-way interactions — often cheaper *and* stronger than a wide OFAT
grid. Recommend one; do not present a menu.

### 3.3 Config selection and cost

Do not use all 21 configs — the component ablation already covers breadth, and this study
needs depth. Pick a small set spanning the behaviours found in §5 of the results report:
at least one config where the GA genuinely improves on its seed (`all_pieces_rect`,
`default`, `switch_cross_rect`) and one where the seed is already the champion
(`dc_figure8_wide`, `cross_figure8_wide`).

Measured single-run wall-clock at production settings (from existing artifacts — use
these to budget, do not guess):

```
default 0.5 min   compact 0.7   plain_wide_racetrack 1.1   switch_one_siding_wide 1.5
switch_two_sidings_tall 1.7   dc_figure8_wide 2.0   cross_figure8_wide 2.0
cross_figure8_tall 2.1   all_pieces 4.0   with_crossing 4.0   with_switches 11.1
dc_figure8_large 40.6
```

`dc_figure8_large` has a measured pathology: seconds-per-generation grows ~15x **within a
single run** (0.71 -> 10.66) as layouts grow, because the self-intersection scan is
quadratic in piece count. Avoid it here unless the study is specifically about cost.

### 3.4 The grid — two knobs, decided with the user

Two numbers get swept:

- **population size** — how many candidate layouts the GA holds at once, i.e. how wide
  the search is.
- **generations** — how many rounds of improvement it runs before stopping, i.e. how
  long the search is.

Each is tried at half and at double that config's own production value. For
`configs/default.yaml` (`pop_size: 1000`, `n_gen: 200`) that means 500 or 2000 layouts
and 100 or 400 rounds — four combinations:

| population | generations | cost vs one production run |
|---|---|---|
| 500 | 100 | 0.25x |
| 500 | 400 | 1x |
| 2000 | 100 | 1x |
| 2000 | 400 | 4x |

That is 6.25x one production run per config per seed, 18.75x over three seeds. Convert
to hours with the measured wall-clock table above; do not guess for configs missing from
it. Levels are per config, never global: `configs/dc_figure8_wide.yaml` runs 600 x 250,
so its corners are 300/1200 layouts and 125/500 rounds.

**Why all four corners instead of one knob at a time.** Sweeping population alone tells
you that a bigger population helps, but not whether it helps on its own or only when it
also gets enough rounds. The corners separate the two: if 2000 x 100 does poorly while
2000 x 400 does well, a large population is useless without the time to work it. That is
the budget-saturation question, and one-factor-at-a-time cannot see it.

**Why the corners are not cost-equalised.** It is tempting to keep only the equal-work
pairs — 500 x 400 and 2000 x 100 are both 200k evaluations. But then generations stop
being a knob of their own and become a consequence of the population choice, so they
cannot be judged separately. Let the cost differ, and print total evaluations in every
cell (§3.1).

**The centre is already computed.** Production settings sit at the centre of this grid,
and the component ablation's `full` arm ran exactly that point on all 21 configs x seeds
{1,2,3}, with `period = 0` and every component on. Reuse those archives as the centre
cells instead of re-running them, under two conditions. First, every other setting must
match: check the resolved `arm.json` the driver writes per run, do not assume it. Second,
all cells of a config must be scored on one shared scale — the quality metric needs a
common reference box, and cells scored in different boxes are not comparable even though
the numbers look comparable (§4).

`crowding_func` is **not** swept here, and no longer needs to be. The plain question —
does `pcd` beat `cd`, and if so does it become the production default — was answered by a
straight A/B at production settings on every arm and all 21 configs: it does not, so the
default stays `cd` (`ABLATION-STUDY-HANDOFF.md` §8.1).

### 3.5 Further factors, if the study is extended

Full inventory and reasoning: `PARAMETER-RESEARCH.md` at the repo root (written in
Polish, for the user). What a grid builder needs from it:

**Three findings that change how any operator result is read.**

- `mutation_prob` is **per individual**, and an individual that passes the gate gets
  **exactly one** edit. It is not a per-gene rate: `mutation_prob: 0.8` means "80% of
  individuals receive one small change", which is a very gentle mutation for a GA.
  `src/operators.py:674` — the probability is per individual
  `src/operators.py:692` — at most one operator is applied per individual
- All 21 configs set `crossover_prob + mutation_prob = 1.0`, which pymoo does not
  require — the two are being used as one slider. Its position ranges from 0.2 to 0.95
  across configs with no measurement behind the spread, which is exactly why this pair
  is the strongest remaining candidate.
- The seven-operator main-loop portfolio only ever fires for chromosomes with no siding
  and no DC. Siding-bearing individuals get a junction op 10% of the time and the
  compensated grow otherwise; DC-bearing ones get 50% compensated grow, 25% figure-8
  regrow, 25% untouched. Tuning those weights would change nothing for `with_switches`,
  `switch_*` or `dc_figure8_*`.
  `src/operators.py:634` — `_MAIN_LOOP_WEIGHTS`, the seven weights
  `src/operators.py:698` — DC branch
  `src/operators.py:709` — siding branch

**Tier A — already in the config, no code change.** The crossover/mutation slider,
`heuristic_ratio` (hard-capped at 0.5 by `src/config.py:93`), `eliminate_duplicates`,
`termination.period`.

**Tier B — hardcoded, one or two lines of pass-through.**

`src/sampling.py:706` — random-individual fill, `rng.uniform(0.5, 0.8)`
`src/operators.py:690` — `junc_thresh = 0.10`, the mutation budget split for sidings
`src/operators.py:121` — 50% swap rate for DC descriptor blocks in crossover
`src/operators.py:129` — 50% swap rate for junction blocks
`src/operators.py:136` — 50% swap rate for the start position
`src/algorithm/runner.py:691` — `enable_closure_repair`, wired True
`src/algorithm/runner.py:692` — `enable_boundary_repair`, wired True
`pymoo/algorithms/base/genetic.py:41` — `n_offsprings` defaults to `pop_size`; lowering
it changes evaluations per generation, so the budget stops being `pop_size x n_gen`

**Tier C — not a parameter yet.** Seed *composition*, as opposed to seed count: the nine
pattern families are shuffled and then dealt out cyclically, so every family gets an
equal share whether or not it suits the config — a DC config spends most of its seed
budget on ovals and racetracks. That is a different question from `heuristic_ratio` and
needs a new config field.

`src/sampling.py:647` — `patterns[i % len(patterns)]`

**Do not sweep** anything that redefines the problem rather than the method:
`SPEED_SAFETY_MARGIN` (`src/problem.py:33`), `special_piece_weight`, the closure / angle
/ boundary tolerances, inventory, boundary. They move the objectives and the feasibility
line, so cells stop sharing one scale and their hypervolumes stop being comparable (§4).
Selection pressure is not available either — pymoo's tournament raises on anything but a
two-way comparison.

`pymoo/algorithms/moo/nsga2.py:25` — `raise ValueError("Only implemented for binary tournament!")`

**Order to extend in:** the crossover/mutation slider first (free, biggest unexplained
spread), then `heuristic_ratio` — noting that seeds = ratio x `pop_size`, so it is
coupled to a factor already in the grid — then the random fill range. For siding and DC
configs swap that third one for `junc_thresh`, since the other mutation operators never
run there.

---

## 4. Metrics and analysis — copy this, it is already settled

- **Primary metric: hypervolume.** Per config, `ideal`/`nadir` over the union of **all
  cells of that config**, `has_extent` guard, then
  `HV(ref_point=HV_REF_POINT, norm_ref_point=False, zero_to_one=True, ideal=, nadir=)`.
  Helpers live in `src/normalization.py` (`ideal_nadir`, `normalize`, `has_extent`,
  `HV_REF_POINT`). Do not reimplement; do not skip the `has_extent` guard — a
  single-point archive self-normalized against `[1.1, 1.1]` scores **1.21**, the maximum
  possible, and single-point archives are common here.
- **Never pool raw HV across configs.** `special_piece_weight` is 6.0 in
  `configs/dc_figure8_wide.yaml` and 3.0 elsewhere, so F[0] is on different scales.
  Across configs, pool only signs and ranks.
- **Never mix normalization boxes in one table.** If you add cells later, either score
  them against the frozen box or recompute and re-report everything.
- **Front size is not a quality metric.** Measured: one arm produced 827 distinct points
  against a competitor's 70 and still lost on HV. Rank by HV, always.
- **Report cost next to quality** — mean `elapsed_s` per cell, from your manifest. Do
  **not** build "equal-CPU" cells by scaling a budget with a ratio measured from the runs:
  the ratio is an outcome of behaviour, not a parameter, and choosing which ratio to use
  is a researcher degree of freedom. The user rejected that approach explicitly.
- **Statistics.** Unit of analysis = one run; block = (config, seed index). Three seeds
  per cell puts the exact-test floor at p = 0.125 **per config**, so no per-config
  comparison can reach significance — pool blocks across configs or make no inferential
  claim. Seeds are matched by index, **not** common random numbers: different cells
  consume the RNG stream differently. Do not call it "paired".

---

## 5. Measured corrections — do not rediscover these

Every item below was verified with literal output during the component ablation. They
cost hours to find.

- `RoundingRepair._do` returns `np.around(X).astype(int)`. Stock-arm populations carry
  **int64, never float**. No int16 cast is needed at the decoder entry.
- `NSGA2`'s default survival is `RankAndCrowding`, **not** `ConstrRankAndCrowding`. The
  project's choice is now behind the `constr_survival` flag.
- **"Orphan switches" does not exist** in `src/` or `tests/`, and `G[4]` has no switch
  term — it is `unresolved_crossings/5 + dangling_cross + dangling_dc`.
- `max_closure_error` is **euclidean** while `G[0..2]` are **per-axis**, so using the
  former as a feasibility check falsely rejects 20-30% of feasible layouts.
- `compute_closure_metrics` returns `angle_error = 360.0` when total turning is exactly
  zero — never use `max_angle_error` as a closure criterion.
- **A missing `pareto_archive.csv` does not mean zero feasible**; it also means the run
  crashed before `save_results`. Disambiguate on `"## Run Summary"` in `run_info.md`.
  Scoring a crash as a failure biases whichever cell crashes more.
- `category_report.md` exists **iff `elite_injection` is on**, not iff anything feasible
  was found.
- **`n_eval` in `convergence.csv` reads 0** whenever the adaptive-epsilon wrapper is
  attached (it is a pymoo `Meta` algorithm and the callback sees the wrong evaluator) and
  non-zero without it. Never use it. Budget axis = `pop_size x n_gen`.
- Archives written before commit `69e04b7` store a **negated speed** in F[1], not seconds.
  Reject any archive row with `f1 <= 0` so a stale directory cannot poison a
  normalization box.

---

## 6. Process rules that bind this work

From `CLAUDE.md` and standing user preferences:

- Full test suite (`python -m pytest --tb=short -q`, baseline **434 passed**) and
  `python -m pycodestyle src tests main.py run_v1_all_configs.py run_ablation.py`
  (exit 0) after every code change.
- **No assertion without evidence** — never claim something works without pasting the
  literal command output.
- **Never `--quick-test`**, for any purpose, including smoke tests.
- **Never edit a config's inventory or boundary** — they are externally given constraints,
  not knobs.
- **Never shorten a run's budget to make a campaign affordable.** Manage cost by choosing
  which runs happen. This does not restrict `n_gen` as a swept factor — see §3.1.
- Prose to the user in **Polish**; code, identifiers and documents in English.
- Explain a non-trivial change **before** making it; narrate each edit in 1-2 sentences;
  small thematic batches.
- Git: nothing in this working tree is committed. Do **not** commit, push, or create
  branches without explicit permission. If asked, stage an explicit file list (never
  `git add -A`) and write plain messages with no AI attribution footer. A foreign
  GitHub Desktop stash exists — never `git stash pop`/`apply`.
- Verify pymoo APIs against the installed package or context7 (`/anyoptimization/pymoo`)
  before asserting them. Prefer pymoo-native facilities over hand-rolled ones.

---

## 7. Deliverables

1. A grid built on the §3.1 decision (`n_gen` is a factor; equal-budget cells allowed).
2. A driver at repo root (own file, own manifest, `outputs/parameters/` tree).
3. Unit tests for whatever new code you add; suite stays green.
4. An analysis script and a results document written the way
   `ABLATION-STUDY-HANDOFF.md` is: verdict first, method, per-cell table, findings,
   an explicit limitations section, and corrections to your own earlier assumptions.
5. Runs queued behind the in-flight campaign — check §0 before launching.