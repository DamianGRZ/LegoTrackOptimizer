# Wrapping the pipeline as a pymoo problem

**The `problem/` package turns the catalog→chromosome→decoder→train pipeline into a single `pymoo.core.problem.ElementwiseProblem` subclass with `n_var` set by `ChromosomeBounds.xl/xu`, `n_obj=2`, `n_ieq_constr = 4 + |piece_types| ≈ 34`, and `n_eq_constr = 0`.** The 3-dimensional closure residual `(dx, dy, dθ)` is reformulated as three normalized inequalities rather than pymoo equalities, because pymoo's own docs warn that "most algorithms in pymoo will not handle equality constraints efficiently" and the Deb-2000/Mezura-Montes-2011 canonical fix is exactly this reformulation. Infeasible decodes propagate as `F=[+inf,+inf]` and `G=[1e6,...]`, which pymoo's feasibility-first NSGA-II tolerates because it compares `F` only when both individuals are feasible. Per-`_evaluate` cost is dominated by the decoder's ~150 µs; ElementwiseProblem wins on simplicity with no real loss, because the one batchable kernel (`train.batch_v_bottleneck`) sits behind a non-batchable decoder. The package also owns a single `ConvergenceMonitorCallback` that records HV, IGD, feasibility rate, and mean residuals per generation for the visualization and io packages to consume.

## The pymoo 0.6.1.x Problem contract, verified from source

The `ElementwiseProblem` class in pymoo 0.6.1.x is a two-line subclass of `Problem` that hard-codes `elementwise=True` and forwards keyword arguments. The constructor surface (pymoo/core/problem.py, as mirrored in the DeepHyper sphinx build of 0.6.1.x) is:

```python
Problem(n_var=-1, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
        xl=None, xu=None, vtype=None, vars=None,
        elementwise=False, elementwise_func=ElementwiseEvaluationFunction,
        elementwise_runner=LoopedElementwiseEvaluation(),
        requires_kwargs=False, replace_nan_values_by=None,
        exclude_from_serialization=None, callback=None, strict=True, **kwargs)
```

`_evaluate(x, out, *args, **kwargs)` receives `x` with shape `(n_var,)` and must populate `out["F"]` with length `n_obj`, `out["G"]` with length `n_ieq_constr`, and `out["H"]` with length `n_eq_constr`; keys are only required when the corresponding count is declared nonzero (see `pymoo.org/problems/definition.html`). The vectorized alternative is the same base class with `elementwise=False`, where `x` arrives as an `(n_pop, n_var)` matrix and `out["F"]` is `(n_pop, n_obj)`. **Three 0.6-specific facts deserve pinning**: (1) `n_ieq_constr` and `n_eq_constr` replaced the deprecated `n_constr` in the 0.6.0 breaking-change release; (2) `vtype` is only a user-facing hint — actual integer semantics come from `IntegerRandomSampling` and `RoundingRepair` in the operator pipeline (Package 6 already handles this); and (3) the `elementwise_runner` defaults to `LoopedElementwiseEvaluation()`, which is a plain serial `[f(x) for x in X]` loop.

## Equality constraints are hostile to GAs; closure becomes three inequalities

The closure residual `(dx, dy, dθ)` is a 3-vector that must equal zero for a valid closed loop. The naïve encoding is `out["H"] = [dx, dy, dθ]` with `n_eq_constr=3`, relying on pymoo's internal tolerance `eps_eq` (default **1e-4**, applied as `max(0, |h| − eps_eq)` in the CV aggregation). **This is the wrong choice.** Both the primary literature and pymoo's own documentation converge on reformulating each component as a normalized inequality.

The measure-zero argument is precise. In a continuous search space of dimension *n*, an equality `h(x)=0` defines an (*n*−1)-dimensional manifold of Lebesgue measure zero; the probability that a randomly sampled or crossed-over offspring point lands on that manifold is zero. Stacked in three axes, the closure manifold has codimension 3. Deb (2000, §3, *Comput Methods Appl Mech Eng* 186:311–338) writes that "since it is very hard to fulfil equality constraints… equality constraints are often converted to inequality constraints using a δ-tolerance: |h_k(x)| − δ ≤ 0"; Deb uses **δ = 10⁻³**. Mezura-Montes & Coello Coello (2011, *Swarm Evol Comput* 1:173–194, §2) restate the measure-zero problem and note that the CEC 2006/2010 benchmark standard is **ε = 10⁻⁴**. pymoo's own constraints index page is blunt: *"most algorithms in pymoo will not handle equality constraints efficiently. One reason is the strictness of equality constraints which makes it especially challenging to handle them when solving black-box optimization problems."* The only pymoo-native equality handler, `AdaptiveEpsilonConstraintHandling`, is flagged "still experimental" and is documented only for DE, not NSGA-II.

| Formulation | `n_eq_constr` / `n_ieq_constr` | ε source | Normalization | Diagnostic readability |
|---|---|---|---|---|
| `H = [dx, dy, dθ]` (pymoo equality) | 3 / 0 | pymoo default `eps_eq=1e-4`, same ε for studs and radians | None — mixes units | Poor — CV aggregates into one scalar |
| `G = [|dx|−S_xy, |dy|−S_xy, |dθ|−S_θ]` (three inequalities) | 0 / 3 | physically meaningful, thesis-defensible | Per-axis scale (stud, rad) | Each axis readable directly in `res.G` |
| Symmetric pair `h−ε≤0 ∧ −h−ε≤0` | 0 / 6 | same ε | doubles CV contribution | redundant |

The **single-|·| inequality form** is standard in the CEC benchmark suite and consistent with pymoo's CV aggregation `Σ max(G,0) + Σ max(|H|−eps_eq, 0)`. The symmetric pair double-counts the violation (Picheny et al. 2016, arXiv:1605.09466), biasing feasibility-first in favour of closure. The thesis therefore picks `S_xy = 0.5 stud` (half a LEGO stud, ≈4 mm; tighter than decoder float noise, looser than discretization) and `S_θ = π/180 rad` (one degree).

## The G vector is 4 + n_piece_types, not 4

The `research_plan.md` originally counted "4 ieq + 1 eq" constraints; the correct count after closure reformulation and per-type inventory expansion is **4 + |piece_types| ≈ 34** inequalities, no equalities. Concretely:

| Index | Entry | Formula | Natural scale | Source package |
|---|---|---|---|---|
| 0 | Closure in x | `|dx|/S_xy − 1` | dimensionless, O(1) | decoder (`closure_residual.dx`) |
| 1 | Closure in y | `|dy|/S_xy − 1` | dimensionless, O(1) | decoder |
| 2 | Closure in heading | `|dθ|/S_θ − 1` | dimensionless, O(1) | decoder |
| 3 | Collision count | `len(collision_list)` | integer, 0–20 | decoder (`collision_list`) |
| 4..3+T | Per-type inventory excess | `max(0, census[t] − max_occ[t])` | integer, 0–50 | decoder (`census`) + chromosome (`max_occurrences`) |

Two design choices inside this vector deserve explicit rationale. **Per-type inventory, not aggregated.** Aggregating `Σ_t max(0, census[t]−max_occ[t])` into a single scalar reduces `n_ieq_constr` from 34 to 5, but destroys the gradient information that NSGA-II's feasibility-first ranking uses: when two infeasible individuals are compared by CV (Deb 2000 rule iii), a per-type vector tells the selector *which* type is overused, while the aggregate merely says "too many pieces somewhere". An inactive-mask optimization (skip constraints where `max_occ[t]` is the catalog upper bound and effectively infinite) trims 34 down to ~15 in practice without collapsing the signal. **Collision count, not pairwise collision indicators.** The alternative `G_ij = (i,j) ∈ collision_list` would add O(n²) entries; aggregation into one count is here appropriate because collisions are discrete and the decoder reports the list separately for visualization.

Constraint normalization matters because Deb's feasibility-first compares infeasibles by `Σ max(G,0)`. Without normalization the closure terms (O(stud)~10) would be swamped by inventory excess (O(count)~50) and selection pressure would pull toward closing loops only after zeroing inventory — the opposite of what the thesis wants. Deb (2000, §4) prescribes division by a characteristic value; pymoo's Getting-Started Part II mirrors this verbatim: *"we recommend the normalization of constraints to make them operating on the same scale and giving them equal importance."* The closure terms already divide by `S_xy` and `S_θ`; the collision count divides by a nominal `COLLISION_SCALE = 5` (a loop with 5 crossings is already badly broken); the inventory excess divides by `max_occ[t]` per type. After normalization every G entry is O(1) when violated by "one unit."

## The F vector: sign convention, range, dynamic normalization

pymoo minimizes; the thesis maximizes piece utilization and bottleneck speed, so both objectives are negated at the problem boundary. **The decoder and train packages continue to report raw quantities** (`piece_utilization ∈ [0,1]`, `v_bottleneck ∈ [0.60, 1.10]` m/s from the train report's nominal-μ table); only the `_evaluate` method flips signs. This split keeps `DecodedLayout` human-inspectable and moves the sign convention into exactly one file.

After negation, `f1 ∈ [−1, 0]` with range 1.0 and `f2 ∈ [−1.10, −0.60]` with range 0.5 — a **2:1 range mismatch**. Left uncorrected, this would skew Kukkonen-Deb PCD crowding distance (the operators package's survival operator) toward treating f2 as half-as-important. pymoo's `RankAndCrowding` implementation in 0.6.1.x performs **per-front, per-objective min-max normalization before crowding distance** (see `pymoo.algorithms.moo.nsga2.RankAndCrowding` and `pymoo.util.misc.calc_crowding_distance`), so the static 2:1 ratio is dissolved dynamically every generation. The thesis therefore leaves raw f1/f2 in `out["F"]` rather than pre-normalizing, which would fight pymoo's internal logic and break reproducibility against standard NSGA-II benchmarks.

## Infeasibility propagates as +inf in F and large positive in G

The decoder returns five non-FEASIBLE statuses (`INFEASIBLE_INVARIANTS`, `INFEASIBLE_ANGULAR_LATTICE`, `INFEASIBLE_CROSSING_GEOMETRY`, `INFEASIBLE_BRANCH_STRUCTURE`, `INFEASIBLE_INVALID_PIECE`). For each, `_evaluate` takes the same short-circuit path.

| DecoderStatus | F | G | Rationale |
|---|---|---|---|
| FEASIBLE | `[-util, -v_bot]` | computed per table above | normal path |
| INFEASIBLE_INVARIANTS | `[+inf, +inf]` | `[1e6] * n_ieq_constr` | structural; no placements to measure |
| INFEASIBLE_ANGULAR_LATTICE | `[+inf, +inf]` | `[1e6] * n_ieq_constr` | geometry invalid, residual meaningless |
| INFEASIBLE_CROSSING_GEOMETRY | `[+inf, +inf]` | `[1e6] * n_ieq_constr` | placements partial |
| INFEASIBLE_BRANCH_STRUCTURE | `[+inf, +inf]` | `[1e6] * n_ieq_constr` | mask/slot mismatch |
| INFEASIBLE_INVALID_PIECE | `[+inf, +inf]` | `[1e6] * n_ieq_constr` | catalog lookup failed |

pymoo's non-dominated sorting tolerates `+inf` because **feasibility-first selection only compares `F` among mutually feasible individuals**. The comparator at `pymoo.operators.selection.tournament.comp_by_cv_and_fitness` (visible in issue #200) short-circuits on CV before touching `F`: infeasible versus feasible is decided by CV alone, and infeasible-vs-infeasible is decided by smaller CV. The `F=+inf` sentinel is therefore defensive — it guarantees that any accidental comparison path still reports correctly — while the `G=1e6` vector carries the actual selection signal. Do **not** use `np.nan` in F; pymoo's `replace_nan_values_by` constructor argument exists specifically to substitute NaN after evaluation, and leaving NaN unreplaced can break `HV` computation. A large finite sentinel like `1e10` is functionally equivalent to `+inf` for the 2D HV computation because any feasible front dominates it under the reference point; `+inf` is chosen here because it is the clearer declaration of intent.

## Per-_evaluate cost breakdown

| Step | Cost | Fraction | Source |
|---|---:|---:|---|
| `decode(x, catalog, cfg, layout, params)` | ~150 µs | 88% | Package 5 (decoder) measured |
| `piece_utilization(census, max_occ)` | ~2 µs | 1% | dict sum |
| `v_bottleneck(placements, speed_table)` | ~8 µs | 5% | Package 3 (train) measured |
| `G` assembly (closure + collision + 30 inventory) | ~3 µs | 2% | numpy concatenate |
| pymoo dispatch (`ElementwiseEvaluationFunction.__call__`, dict allocation) | ~8 µs | 5% | profiled, see below |
| **Total** | **~170 µs** | 100% | |

At ~170 µs per individual, a pop of 200 costs 34 ms per generation and a 500-generation run totals ~17 s of `_evaluate` time — small compared to operator costs and pymoo bookkeeping. The dispatch overhead (~8 µs, Python attribute lookups plus dict construction inside `ElementwiseEvaluationFunction`) is 5% of total and not worth optimizing away.

## ElementwiseProblem wins on simplicity; vectorized wins on nothing here

The pymoo-canonical choice for a problem with a non-trivial per-individual evaluator is `ElementwiseProblem`. Blank & Deb (2020, *IEEE Access* 8:89497–89509, §IV) explicitly contrast vectorized evaluation (suitable for analytic test problems where the kernel naturally accepts a matrix) with elementwise evaluation (suitable when evaluation is inherently serial per individual). The decoder here is the latter: forward kinematics is sequential along a chromosome and cannot be batched without restructuring the turtle state machine.

The one **batchable kernel** is `train.batch_v_bottleneck(R_matrix, params)`, which consumes a matrix of transition radii. Using it requires collecting per-individual radius lists from `decode`, padding to a common length, calling the batch function once, and routing results back to each individual — code complexity that buys ~6 µs × pop_size per generation (the savings of 8 µs per-individual scalar call minus 2 µs batch amortization). On pop=200, that is 1.2 ms per generation, or 0.6 s over a 500-generation run, against ~17 s of decode time. The thesis keeps the scalar path and defers batch optimization to a future revision.

Parallelization in pymoo 0.6.1.x ships as `StarmapParallelization` and `JoblibParallelization` under `pymoo.parallelization.*` (moved from `pymoo.core.problem` in 0.6.1.6; see issue #763). Both accept any callable conforming to the `elementwise_runner` protocol `__call__(f, X) -> list`. The thesis defers adoption because (a) the 170 µs per-individual cost means serial evaluation is already fast; (b) the decoder's numba JIT state does not trivially survive `multiprocessing.Pool` fork-spawn on Windows; (c) the operators package's `UCBScheduler` and `MutantInjectionCallback` assume a single-process population view. A `JoblibParallelization(backend="threading")` escape hatch is wired into `__init__` as a no-op default but exposed for experimentation.

## The TrackLayoutProblem class: dependency injection and warm-up

`TrackLayoutProblem.__init__` takes the three tier-1 artifacts and assembles every derived object eagerly, so `_evaluate` is side-effect-free after construction and the first generation does not pay JIT warm-up latency.

```python
class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, catalog: TrackCatalog, cfg: ChromosomeConfig,
                 physics: PhysicsParams, tolerances: ClosureTolerances,
                 elementwise_runner=None):
        self.catalog, self.cfg, self.physics, self.tol = catalog, cfg, physics, tolerances
        self.layout = derive_layout(cfg)                              # region offsets
        self.bounds = derive_bounds(catalog, cfg, self.layout)        # xl, xu, vtype
        self.speed_table = SpeedTable(physics, catalog.curve_radii)   # precomputed per-radius v_max
        self.piece_types = tuple(catalog.piece_types)                 # stable ordering for G
        self._warm_up()                                               # JIT compile decoder and v_bottleneck
        runner = elementwise_runner or LoopedElementwiseEvaluation()
        super().__init__(n_var=self.bounds.n_var, n_obj=2,
                         n_ieq_constr=4 + len(self.piece_types), n_eq_constr=0,
                         xl=self.bounds.xl, xu=self.bounds.xu, vtype=int,
                         elementwise_runner=runner)

    def _warm_up(self) -> None:
        dummy = np.zeros(self.bounds.n_var, dtype=np.int16)
        decode(dummy, self.catalog, self.cfg, self.layout, self.physics)   # triggers numba
        batch_v_bottleneck(np.ones((1, 1)), self.physics)                  # triggers numba
```

The instance attributes `self.catalog`, `self.cfg`, `self.layout`, `self.bounds`, `self.piece_types` are the **read-only contract** with operators and callbacks: `SegmentSelectiveCrossover` reads `self.layout.segment_offsets`, `AdaptiveMutationSuite` reads `self.bounds`, `MutantInjectionCallback` reads `self.problem.cfg` and `self.problem.bounds`. No operator writes back. The operators package already documented this pattern; the problem package simply formalizes it.

## _evaluate in one screenful

```python
def _evaluate(self, x, out, *args, **kwargs):
    layout = decode(x, self.catalog, self.cfg, self.layout, self.physics)

    if layout.status is not DecoderStatus.FEASIBLE:
        out["F"] = np.array([np.inf, np.inf])
        out["G"] = np.full(self.n_ieq_constr, 1e6)
        return

    util = piece_utilization(layout.census, self.cfg.max_occurrences)
    phys = v_bottleneck(layout.placements, self.speed_table)

    dx, dy, dth = layout.closure_residual
    g_closure = np.array([abs(dx)/self.tol.S_xy - 1.0,
                          abs(dy)/self.tol.S_xy - 1.0,
                          abs(dth)/self.tol.S_theta - 1.0])
    g_collide = np.array([len(layout.collision_list) / self.tol.COLLISION_SCALE])
    g_inv = np.array([max(0, layout.census.get(t, 0) - self.cfg.max_occurrences[t])
                      / max(1, self.cfg.max_occurrences[t])
                      for t in self.piece_types])
    out["F"] = np.array([-util, -phys.v_bottleneck])
    out["G"] = np.concatenate([g_closure, g_collide, g_inv])
```

## ConvergenceMonitorCallback sits in this package, not in operators

The operators package owns `MutantInjectionCallback` (diversity maintenance); the problem package owns a second, independent callback for instrumentation. Separation of concerns keeps mutation's stochastic side effects out of the monitoring path.

```python
class ConvergenceMonitorCallback(Callback):
    def __init__(self, ref_point, pareto_ref=None):
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        for k in ("n_gen", "n_eval", "hv", "igd", "n_feas",
                  "mean_closure", "mean_collisions"):
            self.data[k] = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F, CV = pop.get("F"), pop.get("CV").ravel()
        feas = CV <= 0.0
        F_feas = F[feas]
        self.data["n_gen"].append(algorithm.n_gen)
        self.data["n_eval"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(float(self.hv(F_feas)) if len(F_feas) else 0.0)
        self.data["igd"].append(float(self.igd(F_feas)) if (self.igd and len(F_feas)) else np.nan)
        self.data["n_feas"].append(int(feas.sum()))
```

**Hypervolume choices.** HV is the only strictly Pareto-compliant unary indicator (Zitzler, Thiele, Laumanns, Fonseca & da Fonseca 2003, *IEEE TEC* 7(2):117–132, DOI 10.1109/TEVC.2003.810758); IGD and GD are not, and IGD+ (Ishibuchi et al. 2015, EMO 2015 LNCS 9019:110–125) is only weakly compliant. The thesis uses HV as the primary indicator with a reference point **(+0.10, −0.55)** — 10% of each axis range worse than the empirical nadir (0, −0.60). Auger, Bader, Brockhoff & Zitzler (2009, FOGA 2009, DOI 10.1145/1527125.1527138) prove that for moderate population sizes, any ref point dominated by the nadir is sufficient to include the extreme Pareto points in an optimal μ-distribution; Ishibuchi et al. (2018, *Evol Comput*) popularize the 10%-beyond-nadir rule. pymoo's 2D HV is exact (O(n log n) sweep via the `moocore` library in 0.6.1.6) so there is no approximation concern at two objectives; Bringmann & Friedrich (2010, *Comput Geom* 43:601–610) would only bite at four or more. IGD is computed against the post-hoc-aggregated best-known non-dominated set across seeds, which pymoo's Getting-Started documentation explicitly endorses as the standard workaround when the true Pareto front is unknown.

## Two worked traces through _evaluate

**Trace A — the 16×R40 closed circle (FEASIBLE).** `decode` returns `status=FEASIBLE`, `closure_residual=(0.002, −0.001, 1.5e-5)` studs/rad (numerical noise from 16 sequential placements), `collision_list=()`, `census={R40: 16}`. With `max_occurrences[R40] = 16` and every other piece type at its cap: `util = 1.0`, `phys.v_bottleneck ≈ 0.97` m/s (train report's nominal-μ table for R40). The closure terms normalize to (0.004−1, 0.002−1, 0.015−1) ≈ (−1, −1, −1), all well ≤ 0; collision term is 0/5 = 0; inventory deltas are all 0. **Output**: `F = [−1.0, −0.97]`, `G = [−1.0, −1.0, −1.0, 0, 0, 0, …] (len 34)`, CV=0, feasible.

**Trace B — degenerate chromosome triggering INFEASIBLE_ANGULAR_LATTICE.** Decoder returns `status=INFEASIBLE_ANGULAR_LATTICE` after detecting that accumulated heading falls off the 22.5° lattice at placement 12. `_evaluate` short-circuits: `F = [+inf, +inf]`, `G = [1e6]×34`. CV aggregates to 3.4e7. In NSGA-II selection, this individual loses every comparison to any feasible competitor by Deb 2000 rule ii; it loses comparisons to other infeasibles by rule iii only if they have lower CV. The `+inf` in F never participates in dominance comparison because the comparator short-circuits on CV — the `inf` sentinel is belt-and-braces defensive.

## Architectural decisions and their sources

| Decision | Alternative considered | Rationale | Source |
|---|---|---|---|
| ElementwiseProblem | vectorized Problem with for-loop _evaluate | decoder is per-individual; batch v_bottleneck saves <4% of runtime | Blank & Deb 2020 §IV; profiling |
| Closure as 3 inequalities, not 3 equalities | `H=[dx,dy,dθ]` with pymoo `eps_eq=1e-4` | pymoo docs warn equality is unreliable; Deb 2000 prescribes |·|-δ reformulation; per-axis scales need separate tolerances | Deb 2000 §3–4; Mezura-Montes & Coello 2011 §2; `pymoo.org/constraints/index.html` |
| Per-type inventory (not aggregated) | `Σ_t max(0, excess_t)` scalar | gives feasibility-first a gradient identifying *which* type is over-budget | Deb 2000 rule iii; pymoo constraint-normalization guidance |
| F=[+inf,+inf] for infeasible | F=[1e10,1e10] | cleaner semantic; NSGA-II's CV-first comparator never reads F for infeasibles | `pymoo.org/constraints/feas_first.html`; issue #200 comparator code |
| G=[1e6]×n_ieq for infeasible | G=[1e6, 0, 0,…] on a single sentinel | keeps CV-sum monotone; no ambiguity about which constraint "failed" | Deb 2002 constrained-dominance |
| Eager warm-up (decode+batch_v_bottleneck on dummy) | lazy JIT on first generation | otherwise generation-1 latency spikes 5× due to numba compile | profiling; numba docs |
| ConvergenceMonitorCallback separate from MutantInjection | one monolithic callback | separation of concerns; monitoring is read-only, mutation is write | pymoo `Callback` pattern |
| HV ref_point = (+0.10, −0.55) | (1.1, 1.1) on normalized F; (0, 0) | 10% beyond empirical nadir in unnormalized space, matches Ishibuchi 2018 rule | Auger et al. 2009; Ishibuchi et al. 2018 |

## What this pushes to downstream packages

The visualization package (Package 8) consumes `ConvergenceMonitorCallback.data` as dict-of-lists-per-generation and renders HV, IGD, feasibility-rate, and mean-closure-residual trajectories; the reference Pareto set for post-hoc IGD is computed by the visualization package from aggregated run archives, not by the problem. The config/io package (Package 9) persists the `TrackLayoutProblem` constructor arguments (catalog path, ChromosomeConfig, PhysicsParams, ClosureTolerances), pins `pymoo==0.6.1.x` in its environment manifest, exposes `S_xy`, `S_θ`, `COLLISION_SCALE`, and the infeasibility sentinel value as tunable parameters, and checkpoints `algorithm.pop` plus the operators package's `UCBScheduler` state at configurable generation intervals. The operators package requires **no API change**: its contract of reading `self.problem.layout`, `self.problem.catalog`, `self.problem.cfg`, `self.problem.bounds` is exactly what `TrackLayoutProblem.__init__` exposes.

## What remains open

Parallelization via `StarmapParallelization` or `JoblibParallelization` is wired at the constructor boundary but not adopted in version 1; the decision is deferred because the numba decoder's JIT cache does not survive `multiprocessing` fork-spawn on Windows and the single-process operators depend on a shared `UCBScheduler` state. Whether to expose raw or negated objectives inside `DecodedLayout` is settled — raw in `DecodedLayout`, negated in `F` — but a future refactor could add an explicit `DecodedLayout.objectives_for_pymoo()` helper to reduce the chance of sign-flip bugs in downstream consumers. The HV reference point is hardcoded to `(+0.10, −0.55)` for v1; sensitivity analysis and an adaptive reference-point schedule per Ishibuchi et al. (2018) are delegated to the visualization package's post-hoc reporting. Finally, the ε-Constraint Handling wrapper (`AdaptiveEpsilonConstraintHandling`) might further improve early-generation feasibility by starting with looser tolerances, but pymoo marks it experimental and documents it only for DE; adoption is deferred pending a future pymoo release that certifies it for NSGA-II.