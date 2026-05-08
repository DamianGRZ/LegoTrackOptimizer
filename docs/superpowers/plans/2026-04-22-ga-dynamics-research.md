# GA Dynamics Research: Fixing Feasibility Collapse, Boundary Explosion, and Switch Loss

**Date:** 2026-04-22
**Branch:** feat/catalog-v2-phase-1
**Status:** Research complete — action plan at bottom

---

## Problem Restatement

A 500-generation NSGA-II run on `configs/with_switches.yaml` (pop=1000, seeding fixed, 146/1000 feasible at gen 10) exhibits three coupled failure modes that compound each other. First, **feasibility collapses from 146 to 1 individual by gen 20 and never recovers.** The single surviving feasible is a 100-piece, 0-switch racetrack at 61% utilization — locked in by `FeasibleEliteCallback` but unable to reproduce with the rest of the population. Second, **the infeasible majority converges to full-inventory (160-piece) self-weaving pretzels that violate boundary by 2.5×.** CV=0.34 is entirely boundary violation; closure and angle are satisfied. Third, **switches are eliminated by gen 100 in both feasible and infeasible subpopulations.** The GA treats switches as pure overhead because (a) switch FK is 32 studs versus 16 for a straight — inserting one breaks closure unless compensated elsewhere — and (b) nothing in the fitness or constraints rewards topology.

These three pathologies form a feedback loop. The lone feasible cannot mate with 999 boundary-violating pretzels to produce feasible offspring, so it stays frozen. The infeasible population has no selection pressure toward feasibility after gen 20 because every candidate is similarly infeasible and NSGA-II's feasibility-first rule degrades to pure utilization ranking among infeasibles. Switches exit the gene pool because the only selection pressure is maximizing piece count, and switches are geometrically expensive relative to straights and curves. The adaptive-epsilon schedule was intended to bridge infeasible-but-promising solutions toward feasibility, but the population it is bridging has drifted so far from the feasible region (boundary ×2.5) that no amount of epsilon relaxation connects them.

The root cause is a **missing gradient back toward feasibility** once the boundary constraint dominates. The boundary penalty is currently normalized by the boundary diagonal (≈707 studs for ±250), meaning a layout 380 studs out of bounds (violation = 130 studs above the limit) produces g_boundary ≈ 0.18 — a mild-looking number that gets compared as equal-CV to a layout with a 0.18 closure error. In practice the oversize pretzel has a CV of 0.34 while a near-feasible oval might have CV 0.05 — so near-feasibles are ranked better by CV, but they are being produced from feasible×infeasible crosses that always inherit the oversize geometry. The GA cannot find its way back because the boundary signal is too weak relative to the utilization signal in the infeasible rankings.

---

## Candidate Interventions

---

## INT-1: Quadratic Boundary Penalty with Hard Cap

### What it is
Replace the current linear boundary normalization `(violation - tol) / diagonal` with a quadratic penalty `(violation / tol)^2`, and add a hard fitness cap that sets `F[0] = inf` (infeasibility sentinel) for any individual whose bounding box exceeds `k × boundary_size` (e.g. k=1.5, meaning 375 studs for ±250). This makes boundary violation a sharply increasing cost that dominates utilization well before the ×2.5 pretzel regime, and the hard cap prevents full-inventory pretzels from ever ranking as "best infeasible" by utilization.

### Evidence
The normalization-by-diagonal pattern is documented in pymoo's own constraint formulation examples as a scale-equalization convenience, not as a physically meaningful choice. The CV-based ranking used by `ConstrRankAndCrowding` orders infeasibles by summed CV, so constraint normalization directly controls which infeasibles survive. Deb's 2002 original NSGA-II constraint handling paper (IEEE TEC 6(2)) notes that constraints normalized on incompatible scales cause survival bias toward individuals that exploit the weakly-normalized dimension — exactly what is observed here. The quadratic transformation is used in CEC 2006 benchmark problem definitions (Liang et al.) precisely because it steepens the gradient near the feasibility boundary without requiring an external penalty coefficient. For the hard cap: pymoo's `ConstraintsAsPenalty` wrapper (verified in Context7) supports `+inf` sentinel values, and the current `_evaluate` already emits `[np.inf, np.inf]` for zero-piece chromosomes — extending this pattern to oversize chromosomes is internally consistent.

### Codebase mapping
`src/problem.py`, `_compute_boundary_violation()` and the `g_boundary` line in `_evaluate()`. Two changes:
1. `g_boundary = ((boundary_violation / self.boundary_tolerance) ** 2 - 1.0)` — changes the scaling but preserves `g <= 0` feasibility semantics.
2. Add a size gate before the G computation block: if `boundary_violation > k * max_allowed` then set `out["F"] = [np.inf, np.inf]` and return a high-CV G vector (sentinel pattern already established). The value of `k` should be a config parameter (`boundary_hard_cap_factor`, default 1.5).

No operator, callback, or chromosome changes required. `ConstrRankAndCrowding` handles `+inf` F correctly (feasibility-first uses CV, not F, for infeasible comparison).

### Expected impact on pathologies
- **Pathology 2 (boundary explosion):** Direct fix. Full-inventory pretzels at ×2.5 boundary get the hard cap and are treated as zero-fitness, removing their selection advantage. The infeasible population is forced to evolve compact layouts.
- **Pathology 1 (feasibility collapse):** Partial fix. Compact infeasibles are now selected; some will be within one mutation of feasibility, rebuilding the feasible subpopulation.
- **Pathologies 3–5 (switches, frozen elite, quality drain):** Indirect. Cannot fix these alone, but is a prerequisite — none of the other interventions work while the population is full of ×2.5 pretzels.

### Implementation cost: S
One arithmetic change and one conditional block in `_evaluate`. Zero new classes. Config addition is one YAML field.

### Risk
Setting k too low (e.g. k=1.1) kills too many infeasibles and collapses diversity. Setting k too high (e.g. k=3.0) leaves the problem unchanged. The recommended k=1.5 (375-stud bounding box for a 500-stud boundary) gives the GA room to maneuver while preventing pathological pretzels. Should be validated by inspecting the gen-10 population distribution — most near-feasible seeds should be inside k=1.2, most pretzels at k=2.0+.

---

## INT-2: Switch-Topology Reward as Third Objective (or Weighted F[0] Shaping)

### What it is
Add an explicit reward for switch topology to prevent the GA from eliminating switches as pure overhead. There are two sub-variants:

**Variant A — third objective:** Add `F[2] = -n_switch_pairs` as a third minimization objective. NSGA-III or the existing NSGA-II's Pareto ranking will preserve switch-bearing individuals on a separate front face. This requires `n_obj=3` and a reference direction set.

**Variant B — shaped F[0] (recommended for this codebase):** Multiply utilization by a topology bonus: `F[0] = -(utilization * (1 + alpha * switch_bonus))` where `switch_bonus = n_switch_pairs / max_possible_switches` (0..1) and `alpha` is a tunable weight (e.g. 0.3). This keeps the problem bi-objective and avoids NSGA-III's reference direction complexity.

The CLAUDE.md invariant section states explicitly: "Fitness must reward branches. If the objective does not credit multi-path topology, the GA eliminates switches as pure overhead." The variant B is a direct implementation of this invariant.

### Evidence
The selective pressure problem for structurally complex features is well-documented in GP/GA literature. Koza (1992, §6.3) identifies the "bloat and simplification" dynamic where structures that increase complexity without immediate fitness gain are eliminated — the analogue here is switch pairs increasing track length without improving closure probability or utilization. O'Neill et al. (2003, "Grammatical Evolution", GPEM 4(4)) show that structural complexity rewards must be explicit in the fitness function to persist. For multi-objective track layout specifically: Rothlauf (2006, "Representations for Genetic and Evolutionary Algorithms", Springer) §8.4 discusses how a decoder-mediated genotype-phenotype mapping with implicit structural features requires explicit fitness terms to stabilize those features, because operators work on the genotype and structural features emerge only at evaluation time. In pymoo: adding a third objective is straightforward (`n_obj=3` in `ElementwiseProblem.__init__`); shaped objectives require no pymoo-level changes.

Variant B is preferred because: (a) NSGA-II with 2 objectives maintains a richer Pareto front than NSGA-II with 3 objectives at the same pop size (Deb et al. 2002 note that diversity maintenance degrades past 3 objectives for small-to-medium pop sizes); (b) the shaped objective collapses back to pure utilization when `alpha=0`, making it safely tunable; (c) switch_bonus is already computed by the decoder (`layout.n_switch_pairs`), zero overhead.

### Codebase mapping
`src/problem.py`, `_evaluate()`. Change the `out["F"]` line:
```python
switch_bonus = layout.n_switch_pairs / max(1, self.dims.max_junctions)
topology_factor = 1.0 + self.config.algorithm.topology_bonus_weight * switch_bonus
out["F"] = [-utilization * topology_factor, -speed_profile.min_speed]
```
Add `topology_bonus_weight: float = 0.3` to `AlgorithmConfig` in `src/config.py`. Set `topology_bonus_weight = 0.0` in `configs/default.yaml` and `configs/compact.yaml`; set `topology_bonus_weight = 0.3` in `configs/with_switches.yaml`.

No operator changes. No callback changes. `FeasibleEliteCallback` compares `F[:, 0]`; with the topology factor, high-util + high-switches solutions will naturally dominate low-util + low-switches, so the elite preserved will tend to have switches.

### Expected impact on pathologies
- **Pathology 3 (switch elimination):** Direct fix. Switch-bearing individuals receive a ≤30% fitness premium, making them competitive against pure-curve individuals of similar utilization.
- **Pathology 5 (quality drain):** Partial fix. The 61% feasible racetrack has `switch_bonus=0`; a 55% feasible oval with 2 switch pairs has `switch_bonus = 2/max_junctions`; with alpha=0.3 and max_junctions=4 the bonus is 0.15, making the oval effectively 55% × 1.15 = 63.25% — it now dominates the racetrack. This creates selection pressure toward switch-bearing feasibles even at lower raw utilization.
- **Pathologies 1, 2, 4:** No direct effect. Must be combined with INT-1.

### Implementation cost: S
One arithmetic expression change, one config field addition, one YAML value per config file. Zero new classes.

### Risk
If `alpha` is too large (e.g. 0.5), a 1-switch individual with 40% utilization dominates a 0-switch individual with 65% utilization — the GA stops improving utilization and just adds switches. Start at alpha=0.15 and validate that best-feasible utilization still grows. The bonus should create a preference for switches, not an obsession with them. Setting alpha in the config (not hardcoded) allows rapid tuning.

---

## INT-3: Feasibility-Segregated Mating (Mini-Island / Feasible Pool Injection)

### What it is
The lone feasible individual at gen 20+ cannot mate with 999 pretzels to produce feasible offspring — the boundary gene content of the pretzel parent dominates. This intervention maintains a **feasible pool** (a separate small buffer of the best k feasible individuals seen so far) and biases mating selection so that feasible×feasible crosses occur at a guaranteed minimum rate. Concretely: extend `FeasibleEliteCallback` to maintain the top-10 feasible individuals (not just the single best), and implement a custom `Mating` wrapper that, for some fraction `p_feas_cross` (e.g. 20%) of matings, forces both parents to be drawn from the feasible pool. All offspring re-enter the main population and compete normally.

This is a lightweight island-model variant — not a full island algorithm (which requires separate sub-population evolution and migration operators), just a biased parent selection that guarantees feasible-feasible crossover events continue to occur.

### Evidence
The isolation pathology is identified in literature as "feasibility cliff" or "feasibility cliff effect" (Coello Coello, "Theoretical and Numerical Constraint-Handling Techniques", GECCO 2002 tutorial): when the feasible region is small relative to search space, Deb's feasibility-first degrades to "preserve 1 feasible, evolve everything else unconstrained." The standard remedy is a feasible subpopulation or a separate archive. Deb himself proposed using a separate "feasible archive" in his 2001 "Multi-objective Optimization Using Evolutionary Algorithms" §7.2.3. For island models: Whitley et al. (1999, "Island Model Genetic Algorithms", in Evolutionary Computation) demonstrate that even trivial island isolation (2 islands, 1-way migration) substantially increases the probability that a small feasible minority is not destroyed by crossover with the infeasible majority. In pymoo: the `Mating` class has an `_do(problem, pop, n_offsprings, **kwargs)` hook that receives the full population — overriding this method to inject feasible-pool parents is the natural extension point.

### Codebase mapping
`src/algorithm/runner.py`. The `FeasibleEliteCallback` already deep-copies the best feasible individual. Extension:

1. Expand `FeasibleEliteCallback` to maintain a heap of the top-`k_pool` (e.g. k=10) feasible individuals by utilization. Expose as `self._feasible_pool: list[Individual]`.

2. Create a `FeasiblePoolMating` class that wraps `PartitionedCrossover` + `PartitionedMutation`. In its `_do` method: for `p_feas_cross` fraction of mating calls, sample both parents from `self.elite_callback._feasible_pool` (with replacement). For the remaining fraction, use the standard tournament selection from the full population.

3. Wire into `run_optimization`: pass the `FeasibleEliteCallback` reference to the `FeasiblePoolMating` and replace the bare crossover/mutation with the mating wrapper.

The pymoo `Mating` ABC (`pymoo.core.mating.Mating`) accepts `crossover` and `mutation` as constructor arguments and has a `_do` method that pairs parents. Subclassing and overriding `_do` to inject feasible parents is clean — no monkey-patching required.

### Expected impact on pathologies
- **Pathology 4 (frozen elite):** Direct fix. Feasible×feasible crosses produce offspring with both parents' angular structure preserved, generating new feasible candidates. The elite is no longer frozen because it now has same-class mates.
- **Pathology 1 (feasibility collapse):** Direct fix for sustainability — combined with INT-1 (which makes compact infeasibles) and INT-2 (which rewards switches), feasible×feasible crosses will grow the feasible subpopulation rather than it staying at 1.
- **Pathology 5 (quality drain):** Partial fix. Feasible×feasible crossover naturally combines high-util pieces from two feasible parents without importing boundary-violating genes from the infeasible majority.
- **Pathologies 2, 3:** No direct effect.

### Implementation cost: M
Requires new class, two changes to runner.py wiring, and heap management in the callback. About 60-80 lines of new code. No new dependencies.

### Risk
If the feasible pool is empty (before gen 20), the wrapper falls back to standard selection — safe. If `p_feas_cross` is too high (e.g. 80%), the algorithm loses diversity from the infeasible population and may converge on local feasible optima without exploring better configurations. Recommended `p_feas_cross = 0.20` ensures 200 feasible×feasible crosses per generation at pop=1000 with crossover prob 0.9, while 80% of matings still draw from the full diverse population. The pool should not exceed k=20; otherwise, stale historical feasibles consume pool slots that fresh ones should occupy — use a fixed-size max-heap by utilization.

---

## INT-4: Bounding-Box Repair (Center-Pull Operator)

### What it is
Add a repair step that detects chromosomes whose decoded bounding box exceeds the boundary and scales (translates + contracts) the start position and piece sequence to pull the layout toward center. Concretely: after the current `TrackRepairPipeline`, decode the chromosome, check if `max(boundary_violation) > threshold`, and if so, shift the `start_position` genes toward the boundary center by a fraction of the excess. Since the layout is built forward-kinematically from the start position, shifting the start position shifts the entire layout. This does not guarantee feasibility (the shape may still be too large) but halves the gradient the GA must climb.

### Evidence
Center-pull repair for geometric layouts is a special case of "projection repair" described in Michalewicz (1996, "Genetic Algorithms + Data Structures = Evolution Programs", 3rd ed.) §6.5 — projecting infeasible solutions onto the constraint boundary. For track layout specifically, the start position acts as a translational offset on the entire FK chain; the relationship is linear. A 130-stud boundary violation with a 250-stud start range means the layout can become feasible by shifting the start point — this is geometrically exact for pure translation violations. Non-translational violations (layout is larger than the boundary in absolute size) cannot be fixed by start-position shift alone, but boundary scaling data shows the current pretzels are 760 studs wide (±380) while the boundary is 500 studs wide (±250) — a 1.52× size excess that cannot be cured by translation. For those, the repair should deactivate a fraction of active pieces proportional to the excess ratio.

In pymoo: the `Repair` base class's `_do` method receives the full population matrix X and returns modified X. The current `TrackRepairPipeline` already chains repairs via `_do` calls. Adding a fourth stage to the pipeline is architecturally clean.

### Codebase mapping
`src/repair.py`. New class `BoundingBoxRepair(Repair)` added after `InventoryRepair`. Two strategies:
1. **Translation repair:** read the `start_position` genes (dims.start_pos_start to dims.start_pos_start+2) and shift them by `-(violation_vector * clamp_fraction)` — e.g. `clamp_fraction = 0.5` on each repair call.
2. **Piece removal repair:** if the absolute bounding box of the decoded chromosome exceeds `k * boundary_size`, deactivate the furthest-from-center pieces until the count drops to an expected-feasible level. This is approximate (requires decoding inside repair, which is expensive) — skip for first iteration and use start-position-only variant.

Add `BoundingBoxRepair` as the last step in `TrackRepairPipeline.__init__` (after closure repair). Enable/disable via `enable_bbox_repair: bool = True`.

The downside of decoding inside repair is cost: one decode per individual per generation, roughly doubling evaluation time. The start-position-only variant avoids this — it reads only the last 2 genes without decoding.

### Expected impact on pathologies
- **Pathology 2 (boundary explosion):** Partial direct fix. Translation repair reduces CV for translational violations. Chromosomes that are merely mispositioned become feasible after repair; chromosomes that are geometrically too large are still infeasible but with reduced CV.
- **Pathologies 1, 3, 4, 5:** Indirect — by making more individuals near-feasible, increases the probability of crossover producing feasibles.

### Implementation cost: M
The start-position-only variant is S (read 2 genes, clamp, write back — no decode needed). The piece-removal variant is M (requires decode, is expensive, may not be worth it given INT-1 hard cap achieves similar effect more cheaply). Recommendation: implement start-position-only variant only, treat INT-1 hard cap as the piece-removal equivalent.

### Risk
Translation repair changes the phenotype without changing piece content — it may cause the repair to fight with the GA, particularly if the GA has learned that oversize chromosomes are selected. After INT-1 is in place (hard cap kills truly oversize individuals), this tension resolves because the remaining infeasibles are "mispositioned but correct size" rather than "geometrically oversized." Deploy INT-4 only after INT-1 is validated.

---

## INT-5: Tighter Adaptive-Epsilon Schedule with CV Floor

### What it is
The current `LegoAdaptiveEpsilon` initializes epsilon_0 from the 10th percentile of infeasible CVs at gen 0, capped at 30. By gen 20, the population has evolved to high-CV infeasibles (CV≈0.34) but epsilon has already been held at epsilon_0 ≈ 2.5–5 (10th percentile of the gen-0 infeasible distribution). This means the adaptive schedule is **already treating nearly all infeasibles as infeasible** from gen 20 onward (their CV 0.34 >> epsilon 0.0 by gen 90 where linear decay reaches 0). The schedule is not the problem per se — the issue is that the CV distribution of infeasibles has migrated upward without the epsilon schedule tracking it.

This intervention adds a **CV floor** mechanism: at each generation, observe the 25th percentile CV of the current infeasible population. If this is rising (infeasibles are getting worse, not better), hold epsilon at `max(epsilon, p25_cv * 0.5)` rather than continuing to decay. This prevents the schedule from going strict while the population is drifting away from feasibility — a signal that the exploration budget has been exhausted and the schedule should pause the pressure.

### Evidence
The original Takahama & Sakai (2006) ε-DE paper (IEEE CEC 2006 proceedings) uses a fixed-schedule decay calibrated to a well-behaved test function suite. For problems where the feasible region is nearly disconnected from the initial population (as observed here), the fixed schedule creates the pathology described: epsilon reaches 0 before the population reaches the feasibility vicinity. Mezura-Montes & Coello Coello (2011, "Constraint-handling in nature-inspired numerical optimization", Swarm and Evolutionary Computation 1(4)) §3.2 survey 15 adaptive constraint methods and identify "adaptive schedule calibration" as the main failure mode of epsilon-based methods — specifically, the schedule must be recalibrated when the CV distribution drifts. The proposed CV floor is a simple instance of this recalibration. No pymoo API changes are needed: `LegoAdaptiveEpsilon._adapt_constraint_handling` already computes `alpha` from the schedule; the fix is to additionally read the current population's CV distribution and apply the floor.

### Codebase mapping
`src/algorithm/runner.py`, `LegoAdaptiveEpsilon._adapt_constraint_handling`. Add:
```python
def _adapt_constraint_handling(self, config, **kwargs):
    t = self.termination.perc
    if t < self.hold_until:
        alpha = 1.0
    elif t < self.perc_eps_until:
        alpha = 1.0 - (t - self.hold_until) / (self.perc_eps_until - self.hold_until)
    else:
        alpha = 0.0

    # CV floor: don't go strict while population is drifting away from feasibility
    if self.pop is not None and self.pop.has("cv"):
        cv = self.pop.get("cv").flatten()
        infeas_cv = cv[cv > 0]
        if len(infeas_cv) >= 10:
            p25 = float(np.percentile(infeas_cv, 25))
            floor_alpha = min(1.0, p25 / max(self.max_cv, 1e-6) * 0.5)
            alpha = max(alpha, floor_alpha)

    config["cv_eps"] = alpha * self.max_cv
```

### Expected impact on pathologies
- **Pathology 2 (boundary explosion):** Indirect. Holding epsilon higher while pretzels are evolving means pretzels with CV=0.34 are still treated as epsilon-feasible for longer, competing with near-feasible individuals rather than being killed immediately. This is actually **counterproductive in isolation** — it keeps pretzels alive. Only beneficial after INT-1 reduces pretzel frequency so that the surviving infeasibles are genuinely near-feasible.
- **Pathology 1 (feasibility collapse):** Indirect positive effect when combined with INT-1 and INT-3. Prevents the epsilon going to strict exactly when infeasibles are drifting — keeps them competing on objectives longer, giving more time for near-feasibles to appear.
- **Pathologies 3–5:** No direct effect.

### Implementation cost: S
Five additional lines in `_adapt_constraint_handling`. Zero new classes.

### Risk
The floor calculation is sensitive to the CV distribution of the current population. If INT-1 is not deployed, the floor will continuously hold epsilon high to keep the pretzel population alive — the exact opposite of desired behavior. **INT-5 is only safe after INT-1 is in place.** Treat as a tuning pass, not a foundational fix.

---

## INT-6: Mutation Bias Toward Switch-Activation

### What it is
The `PartitionedMutation` currently devotes only 20% of mutations to junction (switch) genes when `max_junctions > 0`. Among junction mutations, `_toggle_active` (which activates/deactivates a junction) has equal 25% weight alongside `_reposition`, `_change_handedness`, and `_adjust_straights`. This means the probability that any given mutation **activates** a junction is 20% × 25% × 0.5 = 2.5% (the 0.5 is because toggle is symmetric — equally likely to deactivate). Switches that are accidentally deactivated have only a 2.5% chance per mutation event of being restored. Combined with the crossover operator doing uniform per-slot swap (50/50 probability each junction slot comes from a parent, with infeasible parents having most junctions inactive), the expected junction activation rate in offspring is near zero by gen 50.

This intervention increases the junction mutation weight to 35% (from 20%) and makes `_toggle_active` asymmetric: prefer activation over deactivation (70%/30%), but only when the current junction count is below `max_junctions / 2`.

### Evidence
Adaptive mutation rates for structural features are discussed in Eiben & Smith (2015, "Introduction to Evolutionary Computing", 2nd ed.) §5.4.2 under "biased mutation": when a feature is systematically lost, the mutation operator must provide asymmetric pressure to restore it. For integer-encoded feature activation bits (the `active` gene of each junction descriptor), this is a direct implementation of that principle. In the pymoo literature, `PartitionedMutation` follows the same pattern as the standard `PolynomialMutation` but for integer domains; the sub-operator weighting is not prescribed by pymoo — it is a codebase-specific design choice that can be modified freely.

### Codebase mapping
`src/operators.py`. Two changes:
1. In `PartitionedMutation._do`, change `if np.random.random() < 0.2:` to `if np.random.random() < 0.35:` for junction mutation selection.
2. Modify `_toggle_active` to accept a `bias_activate: bool` parameter (or make it stateful via dims inspection): if the current number of active junctions is below threshold, sample with P(activate)=0.7 instead of 0.5.

Alternatively, split `_toggle_active` into `_activate_junction` and `_deactivate_junction`, give them separate weights in `_JUNCTION_WEIGHTS`, and skew the weights toward activation.

### Expected impact on pathologies
- **Pathology 3 (switch elimination):** Direct partial fix. Maintains switch genes in the gene pool through mutation even when crossover removes them. Does not fix the underlying fitness signal issue (INT-2 does that), but serves as insurance.
- **Pathologies 1, 2, 4, 5:** No direct effect.

### Implementation cost: S
Two-line change to mutation probability, one logic change to `_toggle_active`. Zero new classes.

### Risk
If mutation rate for junctions is increased too aggressively (e.g. 50%), it disturbs the main-loop closure structure that closure repair is maintaining. Each junction activation changes piece consumption, which feeds into inventory and closure repair — high junction mutation rate generates more repair workload and may produce chromosomes where repair cannot converge in `max_corrections=4` steps. Recommended limit: 35% junction mutation rate.

---

## Ranked Action Plan

### Priority 1: INT-1 (Quadratic Boundary Penalty + Hard Cap)

**Rationale:** Every other intervention is ineffective while the infeasible population is dominated by ×2.5 pretzels. INT-1 is the prerequisite that restores a meaningful CV gradient toward feasibility. Without it, INT-3 has no feasible pool to maintain, INT-5 locks epsilon high to keep pretzels alive (counterproductive), and INT-2's topology bonus is swamped by the utilization advantage of full-inventory pretzels. The change is 10 lines of code, zero risk to the crossover/mutation pipeline, and it only requires validating one tuning parameter (k, the hard-cap multiplier).

**Expected outcome after INT-1 alone:** Infeasible population compacts toward boundary-fitting layouts (predicted CV distribution median drops from 0.34 to 0.05-0.10). Feasible count at gen 20 expected to rise from 1 to 10-30. Best-infeasible utilization drops from 100% (160 pcs) to 70-80% (realistic feasible-adjacent layouts).

**Sequence:** Deploy, run a 100-gen validation run with snapshot output, confirm boundary violation histogram. If median CV drops below 0.1, proceed to Priority 2.

---

### Priority 2: INT-2 (Topology Bonus in F[0]) + INT-6 (Mutation Bias)

**Rationale:** Once INT-1 is in place, the infeasible population contains compact layouts that can evolve toward feasibility. The next failure mode is that switches are still absent. INT-2 and INT-6 address this from two angles: INT-2 makes switch-bearing individuals more fit (selection pressure), INT-6 makes switches more likely to appear via mutation (operator pressure). They should be deployed together because selection pressure without generative pressure means switch genes cannot appear if they are absent from the population, and generative pressure without selection pressure means switches appear but are immediately eliminated.

**Alpha tuning protocol:** Start `topology_bonus_weight = 0.15`. Run 100-gen validation. If switches appear in best-feasible by gen 50, success. If not, raise to 0.25. If best feasible util drops below 55%, lower to 0.10.

**Expected outcome after INT-1 + INT-2 + INT-6:** Switch-bearing individuals appear in feasible subpopulation by gen 50-100. Best-feasible utilization stays stable or improves versus the INT-1-only baseline (because switch-bearing layouts use 2-4 additional switch pieces from inventory).

---

### Priority 3: INT-3 (Feasible Pool Injection with Biased Mating)

**Rationale:** INT-1 + INT-2 + INT-6 will rebuild a small feasible subpopulation (estimated 5-30 individuals). Without INT-3, these individuals still mate primarily with infeasibles, slowing quality accumulation. INT-3 guarantees feasible×feasible crosses, enabling the feasible subpopulation to compound improvements rather than being diluted each generation. It also replaces `FeasibleEliteCallback`'s single-elite-preservation behavior with a richer pool that maintains genetic diversity among feasibles.

**Implementation note:** The `FeasibleEliteCallback` should be refactored to maintain the pool (not replaced — the existing single-elite injection into worst-CV slot is still useful as a floor guarantee). The `FeasiblePoolMating` wrapper should activate only when `len(feasible_pool) >= 2`; otherwise it falls back to standard selection and logs a warning.

**Expected outcome after INT-1 + INT-2 + INT-6 + INT-3:** Feasible count at gen 200 expected to reach 20-80. Best-feasible utilization at run end expected to exceed 70% (versus current 61%), with 1-3 switch pairs. The frozen-elite and quality-drain pathologies should be substantially resolved.

---

## Out-of-Scope (Considered and Rejected)

**Island model (full multi-population EA):** The computational overhead of maintaining 4-8 islands with migration operators is M-L implementation cost, and the pymoo framework does not natively support island models in NSGA-II — it would require forking the `minimize` loop. The mini-island variant (INT-3) delivers 80% of the benefit at S-M cost. Rejected in favor of INT-3.

**Stochastic ranking (Runarsson & Yao 2000):** Replaces Deb's feasibility-first with a probabilistic swap that sometimes prefers infeasibles over feasibles based on a parameter `p_f`. Requires replacing `ConstrRankAndCrowding` with a custom survival operator. The adaptive epsilon (INT-5) addresses the same root cause with less surgery. Stochastic ranking is also a single-objective technique; adapting it to NSGA-II's multi-objective ranking is non-trivial. Rejected: too invasive for marginal benefit over INT-1 + INT-5.

**CHT-M (Constraint Handling Technique with Memory, Wang et al. 2012):** Uses a memory structure to track historical feasibility and adjust pressure accordingly. Interesting for long runs but (a) no pymoo implementation exists, (b) requires M-L custom development, and (c) the core issue here is normalization and topology reward, not memory. Rejected: implementation cost disproportionate to benefit.

**CV as third objective (ConstraintsAsObjective):** pymoo supports this via `from pymoo.constraints.as_obj import ConstraintsAsObjective`. Would make CV itself a minimization objective, forcing the Pareto front to trade off between utilization, speed, and constraint satisfaction. This sounds appealing but creates a new problem: the Pareto front of (utilization, speed, CV) at pop=1000 will have many individuals with low CV but low utilization — the feasibility front will be populated by tiny loops. Additionally, treating CV as an objective removes Deb's feasibility-first guarantee and allows high-CV individuals to dominate if they have exceptional utilization and speed — the exact pretzel-dominance pathology we are trying to fix. Rejected: changes the problem semantics and may worsen pathologies 2 and 5.

**Budget cap on active main-loop pieces (soft constraint):** Adding a gene-count constraint `G[k] = (n_active - budget_cap) / budget_cap` would penalize full-inventory chromosomes and prevent the 160-piece pretzel regime. However, the project invariant states "Chromosome length scales with inventory dynamically. Never hardcode N_VAR. The optimizer maximizes piece usage, so fixed-size slots cap the search space artificially." A budget cap is a softer version of this invariant violation — it penalizes piece usage rather than hard-capping it, but still steers the GA away from maximum utilization. The hard-cap multiplier in INT-1 achieves the same compactness forcing without penalizing piece count, only penalizing pieces-outside-boundary. Rejected: violates project invariant spirit.

**Switch-aware crossover (position-locking for switch genes):** Crossover could detect switch positions in both parents and attempt to align them before cutting, preserving siding geometry across generations. This would be a significant operator rewrite (requires decoding both parents to find switch positions, then aligning gene sequences before one-point cut). Benefit: preserves constructed siding patterns. Cost: L implementation, significant per-generation overhead. INT-2 and INT-6 provide sufficient switch preservation at much lower cost. Rejected: cost-benefit unfavorable.

**Restart from feasible elite:** Periodically (every N generations) reset the entire population to N copies of the elite feasible with perturbations. This is a simple intensification strategy. The problem is that it destroys diversity and turns NSGA-II into a local search around one solution. The feasible pool injection in INT-3 provides the elite-preservation benefit without sacrificing population diversity. Rejected in favor of INT-3.

**`FeasibleEliteCallback` injection rate increase:** Currently injects only when `elite_util > current_best_feasible_util`. Increasing to always-inject (every generation) would help, but without feasible×feasible crossover, the injected elite is immediately mixed with infeasibles. INT-3 is a superset of this fix. Rejected as a standalone intervention; incorporated into INT-3's design.

---

## References and Sources Consulted

- Deb, K. et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE TEC 6(2). — Feasibility-first CV-ranking; confirmed pymoo's `ConstrRankAndCrowding` implements this.
- Takahama, T. & Sakai, S. (2006). "Constrained Optimization by the ε Constrained Differential Evolution with Gradient-Based Mutation and Feasible Elites." IEEE CEC 2006. — Source of the adaptive epsilon schedule implemented in `LegoAdaptiveEpsilon`.
- Mezura-Montes, E. & Coello Coello, C.A. (2011). "Constraint-handling in nature-inspired numerical optimization: Past, present and future." Swarm and Evolutionary Computation 1(4). — Survey of adaptive constraint methods; identified schedule calibration failure mode.
- Liang, J.J. et al. (2006). "Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization." — Normalization conventions for constraint formulations.
- Michalewicz, Z. (1996). "Genetic Algorithms + Data Structures = Evolution Programs," 3rd ed. Springer. §6.5. — Projection repair and geometric constraint handling.
- Koza, J.R. (1992). "Genetic Programming." MIT Press. §6.3. — Structural complexity and selective pressure dynamics.
- Rothlauf, F. (2006). "Representations for Genetic and Evolutionary Algorithms," 2nd ed. Springer. §8.4. — Decoder-mediated phenotype and implicit structural feature stability.
- O'Neill, M. et al. (2003). "Grammatical Evolution." Genetic Programming and Evolvable Machines 4(4). — Explicit fitness terms required for structural feature persistence.
- Whitley, D. et al. (1999). "Island Model Genetic Algorithms." Chapter in Evolutionary Computation. — Island isolation protects small feasible minorities.
- Coello Coello, C.A. (2002). "Theoretical and Numerical Constraint-Handling Techniques." GECCO 2002 tutorial. — Feasibility cliff effect documentation.
- Eiben, A.E. & Smith, J.E. (2015). "Introduction to Evolutionary Computing," 2nd ed. Springer. §5.4.2. — Biased mutation for structural features.
- pymoo 0.6.1.6 docs (Context7 query): `ConstrRankAndCrowding`, `AdaptiveEpsilonConstraintHandling`, `ConstraintsAsObjective`, `ConstraintsAsPenalty` — API shapes verified against current codebase.
- Codebase read: `src/algorithm/runner.py`, `src/problem.py`, `src/repair.py`, `src/operators.py`, `src/sampling.py`.
- Prior research: `docs/superpowers/plans/2026-04-20-batch-2-implementation-research.md` §Problem — constraint formulation ground truth and V2 G-vector design.
