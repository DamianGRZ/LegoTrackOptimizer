# Session Handoff — Objective Rewire (F[1]) + Normalization + Wiring Audit

**Purpose:** load this into a fresh chat so a new instance has the FULL state of this work
without re-deriving it. Read together with `CLAUDE.md` (auto-loaded) and the memory index.
Date of session: 2026-07-30.

**Status: SUPERSEDED IN PART — read §10 FIRST.** Sections 1-9 are the original design session
(2026-07-30, first pass). A second pass on the same day executed Etap 1, rewired the train
config, researched the physics layer, and stopped **mid-refactor with a broken working tree**.
§10 has the current state, the new findings, and the exact next steps.

---

## 1. TL;DR

Two threads, one goal — make the multi-objective problem actually multi-objective and clean:

1. **Fix the degenerate 2nd objective.** `F[1]` will change from `-avg_speed` to
   **total traversal time through all pieces** (minimize). Decision LOCKED (§3).
2. **Add Part-3 normalization** for post-processing (HV / Pareto plot / compromise pick).
   Decision LOCKED. Do this FIRST, before the F[1] change, then run tests and observe.
3. A 3-agent **wiring audit** ran and found the live-vs-dead map (§6). Fix the one real bug
   (`count_pieces`) and decide fate of the dead physics layer as part of this work.

Working language: user wants prose/discussion in **Polish**, code + identifiers in English.
User dislikes verbosity — be terse and decisive.

---

## 2. The core problem (why we are doing this)

- `F[0] = -weighted_utilization` — healthy, range ~[0,1].
- `F[1] = -_slowest_route_speed` — **degenerate**. It is the `min` over the `2^J` take/skip-siding
  routes of each route's `avg_speed` (whole-lap pace, NOT per-segment min).
- The train motor caps speed at `v_motor_max = 1.10 m/s` (`src/train/physics.py:33`), so the
  catalog straight limit 1.57 never binds. Curves cap ~0.97. With `safety_margin=0.95` the
  achievable `avg_speed` band is ~**[0.92, 1.05] m/s** — tiny dynamic range.
- `avg_speed` also normalizes out length (`dist/time`), so it barely varies and does **not
  conflict** with utilization. Result: the Pareto front collapses toward a single criterion
  (utilization); NSGA-II, category archive, HV monitoring all run on a near-1D problem.

**Objectives are not normalized** (util in [0,1], speed ~0.96): "apples vs oranges." Per pymoo
Part 3 this matters for **post-processing / decision-making distance calcs**, NOT for NSGA-II
selection itself (see §4).

---

## 3. LOCKED decisions

1. **`F[1]` = total traversal time through ALL pieces in the layout** (minimize).
   - Definition = sum over each **distinct physical piece** (main loop + all siding-internal +
     crossings + DC), counted **once**, of that piece's traversal time.
   - User chose this over `ke_roundtrip` (ride-quality/energy) and total track length.
   - **Accepted trade-off (eyes open):** favors short pieces (S16 cheaper than S24), DC is the
     most expensive piece (big + slow); the metric partly behaves like "inverse utilization" so
     the front is thin near max-util. Specials are NOT stripped because a high-util+high-time DC
     layout is non-dominated vs a fast low-util layout (both stay on the front), and
     `CategoryEliteArchive` guarantees switch/cross/DC elites survive.

2. **Normalization = pymoo Part 3, POST-PROCESSING only.**
   - Compute on the feasible non-dominated front: `nF = (F - ideal)/(nadir - ideal)`,
     `ideal=F.min(0)`, `nadir=F.max(0)`.
   - Use `nF` for: Pareto plot axes (0–1), HV (on `nF`), and a compromise-solution pick (ASF /
     pseudo-weights) reported next to the max-utilization champion. Print per-objective scale.
   - `_evaluate` keeps RAW `F` (NSGA-II normalizes internally; normalizing the F fed to
     `minimize()` is a per-axis monotonic no-op — dominance ignores it, crowding re-normalizes).
   - **Order: do normalization FIRST (on the current util/speed objectives), run the full test
     suite, observe the change, THEN do the F[1]→time change.**

3. **Recommended implementation choices (proposed, not yet user-confirmed):**
   - Per-piece time = **free-flow**: `arc_length(p) / v_eff(p)`, route-aware speed for
     switch-diverge / DC segments, at `SPEED_SAFETY_MARGIN` — reuses `v_eff_array` + catalog
     route speeds; order-independent; matches the "S16<S24, DC most expensive" mental model.
     (NOT the full 3-pass accel/brake profile — that needs a single traversal, undefined over a
     branched union.)
   - **`evaluate_layout` wired into REPORTING** (champions in `save_results`/`run_info`) to revive
     the dead physics stack for diagnostics, NOT into the per-chromosome hot loop (cost).

---

## 4. Facts established this session (do not re-derive)

- **How F[1] is computed today:** `problem.py:_slowest_route_speed` builds a per-route `Layout`
  view, calls `scoring.py:compute_speed_profile(...).avg_speed`, returns `min` across routes.
- **Physics is real and wired via `scoring.py`:** 3-pass time-optimal profiler (Pass 1 caps,
  Pass 2 forward accel, Pass 3 backward brake), friction ellipse in `available_accel`
  (`physics.py:121`), derailment caps slide/tip/nadal + motor cap, double-unroll for closed loops.
  `compute_speed_profile` already returns `lap_time` — currently discarded (`scoring.py:23,93`).
- **NSGA-II + scale (verified against pymoo docs via context7):** dominance compares each
  objective separately (scale-invariant); crowding is classic Deb-2002 Manhattan distance
  normalized per-objective by `(f_max - f_min)` within the front. So raw scale does NOT bias
  selection. pymoo Part 3 normalization is explicitly a **post-processing / MCDM** step
  ("Handling different scales ... is an inherent part of any multi-objective algorithms, thus we
  need to do the same for post-processing").
- **DC-stripping fear is unfounded under NSGA-II:** non-dominance + category archive keep DC/switch
  solutions on the front regardless of their time cost.

---

## 5. Implementation plan (batches, each behind a review gate)

**Review gate after EVERY batch (user-mandated, rigorous):**
`/review` (or `python-pymoo-reviewer` agent, model Opus 4.8) + a connectivity grep-proof (every
NEW symbol has a live runtime consumer, not tests-only; every changed contract leaves no stale
consumer — quote the hits) + `pycodestyle` exit 0 + targeted tests + **human checkpoint (wait for OK)**.

- **Etap 1 — Normalization (Part 3, post-processing) on current objectives.**
  Add `nF` helper on feasible front; wire into Pareto plot (0–1 axes), HV (on `nF`, replaces the
  hand-tuned `ref_point=(0.10,-0.55)`), and log per-objective scale. Do NOT touch `_evaluate`.
- **Etap 2 — Tests + observe.** Full `/test`; full run on `all_pieces` config; `/diag`. Confirm the
  scale readout shows util spread 0–1 vs speed pinned ~0.96.
- **Etap 3 — F[1] → total traversal time.** New helper `_total_traversal_time(layout, catalog,
  train_config)`; `out["F"] = [-utilization, +total_time]`. Then fix every site assuming the old
  `-speed` sign / scale.

**All sign/scale touch-points to update in Etap 3** (verified by grep this session):
- `src/problem.py:175` (out["F"]), docstrings `4-6, 80-92, 41-74`.
- `src/algorithm/runner.py:` 85,90,96 (speeds=-F[:,1] + max→needs min), 369-372, 721 (log text),
  785, 880, 934 & 954 (CSV headers `neg_slowest_route_speed`/`f1_neg_speed`), 981/990, 1005/1010,
  1029/1035, and `702` (ref_point).
- `src/algorithm/monitoring.py:52,59,90,98` (ref_point + HV; now `nF`-based).
- `src/visualization/pareto_plot.py:62,66,81` (no negation, axis label → time/normalized).
- `src/run_info.py:186-189,277,291`.
- Tests: `tests/test_problem.py:100-406` (hard-asserts `F[1]==-avg_speed` → rewrite to time),
  `tests/test_monitoring.py:47,55-71`, `tests/test_replication_harness.py:36`.
- HV `ref_point` under time units needs recalibration from a real run (no guessing).
- Sync `CLAUDE.md` Architecture F[1] description + baseline test count last.

---

## 6. Wiring audit — 3-agent results (read-only, nothing edited)

Live pipeline: `main.py -> runner.run_optimization -> problem._evaluate -> _slowest_route_speed
-> scoring.compute_speed_profile -> physics`. Anything off that chain (or a live callback / out-key /
artifact reader) is flagged. `problem.py`, `scoring.py`, `monitoring.py` are fully wired.

### TIER 1 — real bug (produces wrong numbers) — FIX
- `src/run_info.py:157` `count_pieces` adds +1 per `main_loop_pieces` slot, so CROSS_90 and
  DOUBLE_CROSSOVER (2 slots each) are **double-counted**. Feeds `run_info.md` "Piece usage" and the
  end-of-run log (`runner.py:37,50`). Contradicts the inventory census in `problem.py:291-301` and
  `MultiPathLayout.n_physical_pieces`. A single physical DC vs `DOUBLE_CROSSOVER: 2` reports `2/2`.

### TIER 2 — large dead layers (decide: revive via reporting, or delete)
- `src/train/evaluation.py` — **entire module** (`PhysicalEvaluation` + domains
  geometry/stability/dynamics/energy, `evaluate_layout`, helpers) is **tests-only**; runtime bypasses
  it. `CLAUDE.md` Code Map mislabels it the runtime "orchestrator" (stale).
- `TrainConfig` members touched only by that dead module or tests: `mu_roll` (`physics.py:42`),
  `mass_total` (`:50`), `mu_nominal` (`:28`, pure orphan), and the **scalar** cap methods
  `v_slide/v_tip/v_nadal/v_max/v_eff/_derailment_caps` (`:58-80`) — dead mirror of the vectorized
  `v_eff_array` used by the profiler.
- `src/operators.py:504-554` — **four dead DC mutation sub-operators** (`_toggle_dc_active`,
  `_shift_dc_position`, `_rotate_dc_route_pair`, `_swap_dc_traversals`) + `_DBL_CROSSOVER_OPS/_WEIGHTS`
  (`:643,649`): defined, weighted, normalized, but `PartitionedMutation._do` never selects them (DC
  gets only grow ops). Dead-by-design but leftover; a reader would wrongly think DC descriptors mutate.

### TIER 3 — silent no-ops (controls that do not control)
- `TerminationConfig.ftol/xtol` (`config.py:43-44`) — read only when `period>0`; no shipped config
  sets them → always default 1e-6.
- `MultiPathLayout.loose_port_count` (`types.py:218`) — always 0, never read.
- `res.monitor_data` (`runner.py:756`), `res.snapshots` (`:757`) — set, never read (docstring even
  promises `res.snapshots`).
- Pydantic models lack `extra="forbid"` → a typo'd yaml key is silently ignored (latent trap).

### TIER 4 — orphan / tests-only balast (cleanup batch; re-grep each symbol immediately before delete)
- `encoding.py`: `validate_chromosome`+5 helpers (`:671,616-660`), `get_active_main_pieces_with_flips`
  (`:352`), `DC_ROUTE_TRACK1/2_THROUGH` (`:61-62`, tests-only).
- `types.py`: `SwitchPair.pair_id/handedness/absorbed_positions` (`:101,104,109`),
  `DblCrossover.slot/origin` (`:127,131`), `CrossJunction.is_valid/slot/origin` (`:152,149,150`),
  `n_cross_junctions` (`:224`), `n_paths` (`:260`).
- `decoder/types.py`: `ValidatedJunction.handedness` (`:107`), `InventoryTracker.used` (`:92`).
- `geometry.py`: `build_layout` (`:121`), `Layout.total_angle` (`:93`) — tests-only.
- `catalog.py`: `speed_table/radius_table/arc_length_table/index_to_id` (`:391-413`) orphan;
  `fk_table/__getitem__/get_topology/get_fk_with_routes/classify_pieces/get_simple_pieces/get_switch_pieces`
  tests-only.
- `specs.py`: `atomic_angle_rad` (`:110`), `unit/angle_unit` (`:108-109`),
  `manufacturer/part_numbers` (`:37-38`), and `by_id/by_kind/by_manufacturer/n_types/piece_ids/on_angle_lattice`
  tests-only.
- `config.py:33` `BoundaryConfig.diagonal` (unused; formula duplicated inline in `problem.py:117-120`);
  `templates.py:179` `DC_LENGTH_STUDS` (renderer keeps its own copy).

### TIER 5 — `CLAUDE.md` map is stale
- `:269` "v1 fallback" — false; `catalog.py:90-101 load()` raises on non-v2 (kit is v2-only).
- `:252` "every config" — `run_v1_all_configs.py:16` runs only 3 of 21 (`default/with_switches/with_crossing`).
- `:283-287` test counts stale (34/335/360 → now 38/381/**401**).
- `:275` "patch BOTH render paths" — now both dispatch through shared `_draw_piece`; over-stated.
- Code Map calls `evaluation.py` the "orchestrator" — it orchestrates nothing at runtime.
- `problem.py:90-91` local docstring for `G[4]` is stale (CLAUDE.md Architecture is correct:
  `unresolved/5 + dangling_cross + dangling_dc`).

---

## 7. Steering / context7 changes already made this session

- `CLAUDE.md` MCP-Servers row rewritten: **always verify pymoo AND Python with context7, never from
  memory.** Verified library ids pinned: pymoo `/anyoptimization/pymoo`, Python/stdlib
  `/python/cpython`, numpy `/numpy/numpy`, scipy `/websites/scipy_doc_scipy`. All spot-checked good
  (pymoo covers `AdaptiveEpsilonConstraintHandling`/`ConstrRankAndCrowding`; cpython is official docs).
- No better GA framework than pymoo exists on context7 (DEAP is a different framework, only a secondary
  reference; the "genetic algorithm" search returns Rust libs / PyTorch toys). context7 is a DOCS
  server, not a runtime GA engine — nothing to "connect."
- Advice given: do NOT create auto-loaded `pymoo.md`/`GA.md` (context bloat + goes stale vs context7);
  if separation is wanted, use an on-demand skill instead.

---

## 8. Open decisions for the next instance to resolve (before coding)

1. Confirm the two implementation choices in §3.3 (free-flow per-piece time; `evaluate_layout` in
   reporting only) — or the user picks otherwise.
2. Which audit tiers to action and in what order. Recommended: TIER 1 (`count_pieces` bug) + TIER 5
   (map sync) now; TIER 2 decided alongside the F[1] work (energy domain / `mu_roll` won't be needed
   under total-time F1 — revive via report or delete); TIER 3 cheap fixes; TIER 4 separate cleanup batch.
3. Whether to run `code-map-audit` as a baseline before edits.

## 9. Session-specific gotchas (project-wide rules live in CLAUDE.md + memory, auto-loaded)

- Never edit `configs/*.yaml` to clear a constraint; boundary box + inventory are external constraints
  — optimize in `src/`.
- Inventory is immutable (no new piece types); kit is R40-only, catalog is v2-only.
- Adaptive epsilon has a collapse failure mode but do NOT disable it (worse without) — fix via
  repair/schedule/diversity.
- No git branch changes without explicit permission. Explicit file staging (no `git add -A`).
  PowerShell for shell commands. Verify-before-delete (re-grep + quote hits).
- "Test this" after a decoder/optimizer change = a real full run with `configs/all_pieces.yaml`, not
  pytest-only. Full runs only (never `--quick`).

---

# 10. Session state — 2026-07-30, second pass

## 10.0 Tree state: GREEN

The mid-refactor breakage recorded here earlier (`NameError: _default_radius_for`) is resolved —
batch 2a is finished. Last verified: `419 passed`, `pycodestyle exit: 0`.

Do NOT `git checkout .` at any point — Etap 1, the config rewire and batch 2a are all finished
and wanted, and none of them is committed yet.

## 10.1 Done and verified this pass

**Etap 1 — Part-3 normalization (post-processing only). COMPLETE.**
Evidence at the time of completion: `419 passed in 36.99s`, `pycodestyle exit: 0`.

- `src/normalization.py` (NEW, top-level on purpose — inside `src/algorithm/` it closes an import
  cycle `src/algorithm/__init__.py` → runner → `src/visualization/__init__.py` → pareto_plot).
  Exports `ideal_nadir`, `has_extent`, `normalize`, `compromise_index`, `HV_REF_POINT`.
  `normalize` wraps pymoo's own `ZeroToOneNormalization` so the plot, the ASF pick and HV all
  apply the identical transform.
- `src/algorithm/monitoring.py` — `_hv` builds `HV(ref_point=(1.1,1.1), norm_ref_point=False,
  zero_to_one=True, ideal=…, nadir=…)` per generation from the **cumulative archive**'s
  ideal/nadir. Returns NaN until the archive spans both objectives. The `ref_point` ctor
  parameter is gone.
  *Why the archive and not the current front*: a front normalized by its own spread fills the
  unit box by construction, so HV would measure front shape, not growth. User decision, verbatim:
  "Zrób tak aby faktycznie liczyć objętość frontu".
- `src/algorithm/runner.py` — `log_front_scale_and_compromise` logs the raw span of both
  objectives plus the equal-weight ASF compromise pick; called from `save_results`.
- `src/visualization/pareto_plot.py` — axes normalized 0-1 and plotted as `1 - nF`, so up-right
  still means better. Side benefit: normalization erases the sign convention, so Etap 3 needs no
  sign fixes here, only the axis label.
- `tests/test_normalization.py` (NEW, 9 tests) and `tests/test_monitoring.py` (+2 tests:
  NaN-until-extent, and HV measuring archive coverage rather than shape).
- `README.md` — the `pareto_front.png` bullet now says the axes are normalized.

**Train config rewired to measured physics. COMPLETE.**
All 21 `configs/*.yaml` now carry `train_config_path: trains/measured_consist.yaml` (was
`trains/default.yaml`). User decision: use measured physics only.
`configs/trains/default.yaml` is now referenced by nothing — left in place deliberately, not deleted.

## 10.2 Physics research — what is actually wired

The premise "practically nothing is connected" is **half true**. The physics model itself runs on
every chromosome; the *diagnostic* layer above it is what is dead.

LIVE, per chromosome:
`problem.py:_slowest_route_speed` → `scoring.compute_speed_profile` → Pass 1 `v_eff_array` →
`derailment_caps` (slide / tip / Nadal) → Passes 2-3 `available_accel` (friction ellipse +
coupler correction, active because `mass_trailing > 0`).
`TrainConfig` fields read on that path: `mu_design`, `g`, `v_motor_max`, `gauge_b`,
`cog_height_h`, `flange_angle_deg`, `max_accel`, `brake_decel`, `mass_loco`, `mass_trailing`,
`coupler_offset`.

DEAD (re-grepped this pass, not taken from §6 on faith):
- `src/train/evaluation.py` — whole module, tests-only. Exported at `src/train/__init__.py:19`.
- `TrainConfig.mu_nominal` — read by *nothing*, not even the dead module (see §10.3 for why it
  exists at all).
- `TrainConfig.mu_roll`, `TrainConfig.mass_total` — energy domain of the dead module only.
- Scalar caps `v_slide/v_tip/v_nadal/v_max/v_eff/_derailment_caps` — dead mirror of `v_eff_array`.
- Of `SpeedProfile`, runtime reads **only** `.avg_speed`. `lap_time`, `total_distance`,
  `max_speed`, `min_speed`, `speeds` are computed every chromosome and thrown away.
  (`lap_time` is exactly the quantity Etap 3 wants — it is being discarded at `scoring.py:99`.)

## 10.3 Where the catalog speed numbers came from (answered)

```
sqrt(0.25 * 9.81 * 0.320) = 0.885889 m/s   <- mu_design, what the profiler uses
sqrt(0.30 * 9.81 * 0.320) = 0.970443 m/s   <- mu_nominal, hence the catalog's 0.97
```

So `_DEFAULT_PHYSICS["R40_CURVE"] = (320.0, 0.97)` was the **same sliding formula frozen at the
optimistic friction**, and `DEFAULT_SPEED = 1.57` (`pieces.py:50` comment: "Motor top speed
default") is a superseded motor guess. Neither is measured. Neither binds, because the train
module recomputes both more conservatively — 0.886 vs 0.97 on curves, 1.10/1.26 vs 1.57 on
straights. This is also the only reason `mu_nominal` exists.

Proven inert — catalog limits neutralized to `inf`, lap times identical to 6 decimals:

| layout | with catalog | without | delta |
|---|---|---|---|
| loop +0 S16/side | 2.389058 s | 2.389058 s | 0.00e+00 |
| loop +3 S16/side | 3.977575 s | 3.977575 s | 0.00e+00 |
| loop +8 S16/side | 6.149185 s | 6.149185 s | 0.00e+00 |
| loop +20 S16/side | 11.282017 s | 11.282017 s | 0.00e+00 |

## 10.4 R40 curve speed — derived, not measurable-by-hand

For R = 320 mm, at `mu_design = 0.25`:

| mechanism | formula | value |
|---|---|---|
| lateral slide | `sqrt(mu*g*R)` | **0.8859 m/s** ← binds |
| tip-over | `sqrt(g*R*(b/2)/h)` | 1.4007 m/s |
| Nadal wheel-climb | `sqrt(g*R*L/V)`, L/V = 0.7256 | 1.5092 m/s |

After the 0.95 margin: **0.8416 m/s**, which is exactly the `min_v` observed in every profile run.
`v_slide` depends only on mu, g and R — not on mass, motor or acceleration — so switching the
train config did not move it at all.

Consequence for "measured physics only": the curve cap reduces to **one unmeasured number, mu**.
A lateral pull test (break-away force / weight) would close it. `cog_height_h` and
`flange_angle_deg` are also assumptions but do not matter: tip-over only starts binding above
**mu > 0.625**, unreachable for plastic wheel on plastic rail.

Measured in `configs/trains/measured_consist.yaml`: `v_motor_max` 1.26, `max_accel` 0.68, masses,
`coupler_offset`. Assumed by its own comments: `mu_design` 0.25 ("pull-test deferred"),
`cog_height_h` 0.030 ("tilt-test optional"), `flange_angle_deg` 50, `brake_decel` 2.45
("passive coasting unmeasured").

Effect of the rewire (same layouts, default → measured): curves unchanged, straights faster
(1.197 vs 1.045 m/s), acceleration much weaker (0.68 vs 1.49). Small loops ~1% slower, the
+20-straight racetrack 8.1% faster.

## 10.5 Batch 2a — DONE

The catalog holds geometry only; `src/train/physics.py` is the sole authority on speed.

Removed: `DEFAULT_SPEED`, `_speed_table`, `get_speed_limits`, `get_speed_route`,
`get_route_speeds`, the `speed_table` property, `TrackPiece.speed_limit_ms`,
`FKRoute.speed_limit`. `_DEFAULT_PHYSICS` / `_ROUTE_PHYSICS` became `_DEFAULT_RADIUS_MM` /
`_ROUTE_RADIUS_MM` (radius only), `_default_physics_for` became `_default_radius_for`.
`scoring.py` Pass 1 is now `v_eff_array(train_config, radii_m) * safety_margin` — the only
behavioural change in the batch.

Tests restated rather than deleted: `tests/test_catalog.py` `test_catalog_carries_no_speed`
(guards that no speed accessor comes back), `tests/test_scoring.py`
`test_r40_circle_capped_by_lateral_slide` (asserts the exact 0.8859 m/s slide cap).

Gate evidence: `419 passed`, `pycodestyle exit: 0`, and lap times identical to the pre-refactor
baseline to < 1e-6 s on all four reference loops — the removal is provably a no-op numerically.

## 10.6 Remaining plan, in order

- **2a** — finish the above; tree green again.
- **2b** — provenance. User wants physics in one place *with each value marked measured /
  assumed / standard, and derived constants showing how they are computed*. Design sketch:
  a `provenance:` block in the train YAML, a matching field on `TrainConfig` (`from_yaml`
  currently filters unknown keys out silently — an unread block would be an orphan control),
  printed into `run_info.md` so it has a live consumer.
- **3** — revive `evaluate_layout` in reporting only (user chose this over deleting it):
  per-champion binding cap, safety factors, grip utilization into `save_results` / `run_info`.
  Do NOT put it in the per-chromosome loop.
- **Etap 2** — one full run on `configs/all_pieces.yaml` + `/diag`, now that the physics is
  settled, so there is a single meaningful observation run rather than several.
- **Etap 3** — `F[1]` → total traversal time. Decided this pass: use the **3-pass profile**
  (segment time = `arc_length[i] / speeds[i]`), summed over **unique** pieces, NOT the free-flow
  `arc/v_eff` of §3.3 — free-flow would disconnect `available_accel`, killing the friction
  ellipse and coupler correction. User requirement: the time must stay capped by the train's
  max speed; the profile already does this (`max_v = 1.0450 = 1.10 * 0.95` pre-rewire).
  For a switchless layout this equals `lap_time`, already computed and discarded today.

## 10.7 Why F[1] must change — measured, not argued

`configs/all_pieces.yaml`, `trains/default.yaml`, margin 0.95:

| layout | pcs | distance | time | avg_v |
|---|---|---|---|---|
| pure R40 circle | 16 | 2.011 m | 2.389 s | 0.8416 |
| oval +1 S16/side | 20 | 2.523 m | 2.961 s | 0.8520 |
| oval +3 S16/side | 28 | 3.547 m | 3.941 s | 0.9000 |
| racetrack +8 S16 | 48 | 6.107 m | 6.391 s | 0.9556 |
| racetrack +20 S16 | 96 | 12.251 m | 12.270 s | 0.9984 |
| racetrack +20 S24 | 96 | 17.371 m | 17.211 s | 1.0093 |

The problem is worse than §2 assumed. It is not only the narrow band — `avg_speed` **rises with
piece count**, i.e. it moves the same direction as utilization. The two objectives are not in
conflict, so there is no trade-off for a front to form on. Time moves the opposite way
(2.39 → 17.21 s), which is exactly the conflict the rewire is after.

## 10.8 Stale docs found, not yet fixed

- `configs/trains/measured_consist.yaml:52` — claims `F[1] = min_speed` and that `max_accel` is
  unused. Both false: the forward pass reads it through the friction ellipse. (An edit fixing
  this was drafted and not applied.)
- `CLAUDE.md` baseline "401 passed" was already stale before this pass (408 before Etap 1's
  +11 tests). Also §6 TIER 5 items are all still open.
- `CLAUDE.md` Code Map still calls `src/train/evaluation.py` the runtime "orchestrator".
