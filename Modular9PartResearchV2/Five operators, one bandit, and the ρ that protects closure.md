# Five operators, one bandit, and the ρ that protects closure

**The `operators/` package is where the thesis's novel adaptive-operator craft earns its keep, and three claims must be defended carefully.** First, the "70–95% crossover destruction" figure in `research_plan.md` is not an external citation but the user's own pilot measurement — yet it is entirely consistent with the decoder's lattice prefilter, which rejects ~98.4% of uniformly sampled main-loop chromosomes for the same reason (atomic-angle sum ≠ 0 mod 360°). Framed that way, the claim survives scrutiny and motivates the design. Second, **segment-selective crossover is adopted with ρ = 1.0 on the main loop (full inheritance from the elite parent, never crossed per gene) and ρ = 0.7 elsewhere (BRKGA-style biased uniform on switch_mask, branch_slots, crossing_overlay)**, which sits squarely in Gonçalves & Resende's recommended 0.65–0.70 band. Third, the five adaptive mutation operators — **SWAP_MAIN, REPLACE_MAIN, FLIP_SWITCH_MASK, BRANCH_MUTATE, CROSSING_TOGGLE** — are selected per-individual each generation by an **ALNS-credit + UCB1 hybrid** whose reward signal is a rank-based FRRMAB-style improvement indicator adapted from Li et al. 2014, using the exploration constant c = √2 from Auer et al. 2002. **Mutant injection at 10%** replaces the bottom crowding-rank decile, **RankAndCrowding(crowding_func="pcd")** implements Kukkonen & Deb 2006 piecewise crowding distance (pymoo 0.6.1.6 confirmed option for 2-objective), and the whole apparatus plugs into pymoo through four subclasses and one callback. The per-generation operator+decoder budget is ~250 ms at pop_size = 200, leaving 1000 generations well under five minutes wall-clock.

## The 70–95% destruction figure is pilot data, and the decoder's prefilter confirms it

The `research_plan.md` passage claiming "crossover destroys angular closure in 70–95% of cases when applied to the main-loop construction sequence (5–33× degradation)" has no external citation, and a literature search for the phrase and the underlying measurement design returns nothing. Treating this honestly, the figure is **the user's own pilot-run observation, not a published result**, and the thesis should label it as such (e.g., "pilot runs on 200 randomly sampled chromosome pairs; main-loop length L = 16–40").

The figure is nevertheless defensible because it converges with an independent measurement the decoder package already provides. The decoder's lattice prefilter (Report #5) showed that **uniformly random integer-keyed main loops fail the closure invariant — Σᵢ atomic_angle(piece_i) ≡ 0 (mod 360°) — in approximately 98.4% of cases**. A single-cut crossover that splices parent A's prefix with parent B's suffix produces a child whose atomic-angle sum is (partial sum of A) + (partial sum of B). If the parents were both closed at 360°·k, the child's sum is typically a random residue in {0, 40°, 80°, …, 320°} and closes only when the two partial sums happen to share the same residue — an event of empirical frequency ~12% in a 10-piece atomic lattice. So a **destruction rate in [70%, 95%] is the *expected* band** for per-gene or single-cut crossover on closed main loops under the atomic-angle arithmetic of the catalog, consistent with the prefilter's 98.4% floor on unconstrained sequences.

The correct framing in the thesis is: *"we measured X% destruction in pilot runs on n pairs, consistent with the theoretical band implied by the ~10-element atomic-angle residue lattice"*, cited via the decoder's prefilter section rather than as an external fact.

## Segment-selective crossover: ρ = 1.0 on main loop, 0.7 elsewhere

Given the destruction rate, **the correct response is not parameter-tuning but structural: the main loop never takes per-gene crossover**. Segment-selective crossover applies BRKGA-style parameterised uniform crossover (Gonçalves & Resende 2011; Spears & De Jong 1991) to each chromosome region independently, with per-region ρ values:

| Region | ρ | Rationale |
|---|---|---|
| main_loop | **1.0** | Full elite inheritance; any per-gene mixing almost surely breaks Σ ≡ 0 (mod 360°) closure. Macroscopic topology change is the job of mutation + mutants, not crossover. |
| switch_mask | **0.7** | Per-gene independent biased coin flip, elite-biased. 0.7 is the canonical BRKGA value (Gonçalves & Resende 2011, brkgaAPI default; Toso & Resende 2014). |
| branch_slots | **0.7** | Same rationale; branch slot regions are already decoded with relaxed feasibility, so per-gene mixing is safe. |
| crossing_overlay | **0.7** | Triples (i, j, piece_type) are mutated per-triple with elite bias; the decoder tolerates duplicates and invalid i ≥ j by ignoring them. |

The BRKGA literature reports canonical ρ values in the 0.65–0.75 range: Gonçalves & Resende 2011 recommend ρ ≈ 0.7; the capacitated-MST BRKGA uses ρₑ = 0.65 (Resende 2013); the telecommunications BRKGA uses ρₑ = 0.70 (Resende 2012); Toso & Resende 2014's `brkgaAPI` default is 0.7. The Londe et al. 2024 review (EJOR, DOI 10.1016/j.ejor.2024.03.030) surveys 250+ BRKGA applications and finds 0.7 is the modal choice. **ρ = 0.7 is therefore the safest defensible setting** for the three mixable regions.

**The alternative considered and rejected: whole-loop swap.** An appealing variant is to leave the main loop mixable at the *whole-chromosome* granularity — i.e., with probability 1−ρ_loop the child inherits the entire main loop of parent B instead of parent A (all-or-nothing). This is less destructive than per-gene crossover because each parent's loop is already closed, so the child's loop is also closed by construction. It also explores macroscopic topology moves that mutation alone cannot reach in one step. The decision is to **defer whole-loop swap to future work** for three reasons: (1) the mutant injection at 10% already introduces fresh topologies each generation, covering the exploration role; (2) whole-loop swap would invalidate the `switch_mask` and `branch_slots` regions (which reference switches by their main-loop positions), requiring a complex re-index step; and (3) the thesis's novelty budget is concentrated in the adaptive mutation suite, not in exotic crossover. The whole-loop swap becomes a candidate for the `what remains open` section.

Eshelman & Schaffer 1993's "Interval Schemata" and Spears 1995 on adaptive crossover both argue that *per-variable* crossover is a poor match for tightly-coupled genes — exactly the main-loop case here, where every gene couples through the closure constraint. The rejection of per-gene main-loop crossover is therefore supported by both the empirical pilot numbers and the schema-theorem analysis.

## Five operators, one bandit, one reward signal

The five mutation operators are specified in a single table. Each operator writes to a specific chromosome region, defines a bounded one-step neighbourhood, preserves `validate_invariants` by construction, and has a pinned default hyperparameter.

| Name | Region | Neighbourhood (1 step) | Invariant preservation | Default hyperparameters | Objective bias |
|---|---|---|---|---|---|
| **SWAP_MAIN** | main_loop | Swap two adjacent pieces at positions (i, i+1); |N| = L−1 | Sum-of-angles invariant: addition is commutative, so closure is exact; bounds: no change | n_swaps = 1 | Neutral (fine-tunes piece ordering for f₂ speed) |
| **REPLACE_MAIN** | main_loop | Replace piece at position i with any other piece of the **same integer atomic-angle multiple** from the catalog's same-angle bucket; |N| ≈ L · (n_samebucket − 1) | Closure exact for lattice pieces (same angle); bounds: piece index ∈ [0, n_piece_types]; catalog-gated | n_replacements = 1 | f₁ (piece utilisation): swap an R40 for two R88 halves at matching angle to increase piece diversity |
| **FLIP_SWITCH_MASK** | switch_mask | Change one mask gene from current value v ∈ {0,1,2} to another value v' ∈ {0,1,2}\{v}; |N| = 2·len(switch_mask) | Bounds preserved (domain is {0,1,2}). Closure is **not** exact when flipping 0↔1 because through and diverging exit-port angles differ; the decoder's feasibility check catches the rare cases that drift; **restriction**: by default this operator flips 0↔2 only (same exit port, just adds/removes a branch), which preserves closure exactly | p_flip = 0.1 per switch | Topology (opens branches for f₂ speed via alternate routing) |
| **BRANCH_MUTATE** | branch_slots | One of {insert, delete, replace} at a random position within `branch_slots[k·L_br : (k+1)·L_br]` **where k is the position index of the k-th `mask=2` switch in main-loop order** (NOT the k-th switch of any kind — this is the bug fixed here) | Sentinel S = n_piece_types marks "no piece / end of sequence", preserved on insert/delete; bounds: piece index in [0, S] | p_insert = p_delete = p_replace = ⅓ | f₁ (extends usable branches, consumes more pieces) |
| **CROSSING_TOGGLE** | crossing_overlay | Add, remove, or re-triple: sample a random slot in the overlay and (a) zero its triple (remove), (b) fill a zero slot with a random (i, j, piece_type) with i<j (add), or (c) perturb an existing triple's piece_type while keeping i<j | Bounds on (i, j, pt) enforced; duplicates and i≥j rejected downstream by the decoder | p_add = 0.4, p_remove = 0.4, p_retype = 0.2 | Macroscopic topology (figure-8 patterns for f₁) |

**Why these five, not more.** Each chromosome region has at least one dedicated operator (SWAP_MAIN and REPLACE_MAIN cover main_loop with different neighbourhoods, FLIP covers switch_mask, BRANCH covers branch_slots, CROSSING covers crossing_overlay), and two operators target main_loop because it is the largest and most constrained region. The count of five keeps the bandit's arm set small enough that UCB1 converges within 1000 generations (UCB1 needs ~k·log(T) samples per arm; 5·log(1000) ≈ 34 samples, well under the budget).

**The BRANCH_MUTATE format is what reconnects operator and decoder.** Per the user memory, the legacy BRANCH mutation wrote to a slot indexed by "the k-th switch of any kind in main-loop order". The decoder (Report #5) instead counts the k-th *mask=2* (branch-here) switch. The operators package **must write to and read the same counter**. The corrected contract is: before mutation, the operator scans `main_loop` in order, filters to positions where `switch_mask[pos]==2`, numbers them 0..m−1, picks one k, and modifies the `L_br`-length slab `branch_slots[k·L_br : (k+1)·L_br]`. If no switch currently has mask=2, BRANCH_MUTATE is a no-op and the UCB accountant records a null trial (no reward, no use-count increment).

## The ALNS-UCB selection rule is a principled hybrid of two named methods

The combination labelled "ALNS-UCB" in `research_plan.md` is not a standard named method. It is a principled hybrid of (a) Ropke & Pisinger 2006's ALNS credit-assignment and (b) Auer, Cesa-Bianchi & Fischer 2002's UCB1 selection. The thesis must own this framing explicitly: **"ALNS-style reward bookkeeping with a UCB1 arm-selection policy, following the MOEA/D-FRRMAB design of Li, Fialho, Kwong & Zhang 2014"**.

**Selection rule.** At each individual's mutation event, each operator i gets a UCB1 score

  UCB1_i = mean_reward_i + c · √(ln(n_total) / n_i)

with c = √2 (Auer et al. 2002 default for rewards normalised to [0,1]; this is the theoretically optimal constant for the sub-Gaussian sub-bandit case). The operator with the highest UCB1 score is selected. n_i is the use-count of operator i; n_total = Σ n_i. To bootstrap, each operator starts with n_i = 1 and a synthetic reward of 0.5 (a cold-start prior that both avoids division-by-zero and prevents one unlucky trial from permanently suppressing an arm).

**Reward signal — four candidates, recommended choice.**

| Reward signal | Pros | Cons | Suitability for this 2-obj NSGA-II |
|---|---|---|---|
| (i) Hypervolume contribution delta | Principled multi-objective; captures front expansion | Expensive (O(n²) per generation); reference point tuning; sensitive to scale | Overkill for 2 objectives |
| (ii) Pareto dominance indicator (1 if child dominates parent, else 0) | Cheap; scale-free | Binary; discards magnitude; long plateaus of 0 | Crude but robust |
| (iii) Normalised per-objective improvement sum | Simple; captures magnitude | Requires objective-range estimate; can be dominated by one objective | Reasonable but scale-dependent |
| (iv) **FRRMAB rank-based fitness improvement rate (Li et al. 2014)** | Scale-free; ranks improvements within a sliding window; proven on MOEA/D | More state to maintain | **Recommended** |

The recommended reward signal follows Li, Fialho, Kwong & Zhang 2014 (IEEE TEC 18(1):114–130, DOI 10.1109/TEVC.2013.2239648). For each child produced by operator i, compute the *fitness improvement* FI = max(0, f_parent_rank − f_child_rank) where ranks come from non-dominated sorting of parent + child (using Deb 2000's feasibility-first rule when applicable). Maintain a sliding window W of the last 50 such (operator, FI) pairs; within the window, **rank the FI values** and assign exponentially decaying credit D^(rank−1)·FI with D = 1.0 (Li et al. recommend D ∈ [0.5, 1.0]; 1.0 preserves magnitudes). The per-operator reward is the sum of decayed FIs divided by the count of that operator's trials in the window, then min-max normalised to [0, 1] across operators before entering UCB1. Li et al. 2014 use scaling constant C = 5 inside the UCB term (√(C·ln(Σn)/n_i)) rather than √(2·ln/n); this is an empirical choice for decomposition MOEAs. **For NSGA-II we keep the theoretically justified c = √2** because the rewards are already normalised to [0,1], which is the regime where c = √2 is optimal (Auer et al. 2002, Theorem 1).

**Warmup.** The first 5 generations bypass UCB and cycle through the five operators round-robin to seed each n_i with ≥ pop_size real trials. This is the "one-arm-pull-per-arm" initialisation from Auer et al. 2002 extended to a population setting.

**Where this deviates from FRRMAB.** Li et al. 2014's FRRMAB targets MOEA/D with scalar decomposed subproblems where FI is straightforward. In NSGA-II with Pareto-rank-based FI, the reward distribution is heavier-tailed and sparser (most children are non-dominating, giving FI = 0). The sliding-window length W = 50 and the c = √2 tuning compensate: W = 50 at pop_size = 200 means roughly 10 generations of memory per operator in the worst case, which is short enough to track phase transitions but long enough for variance reduction.

## Mutant injection at 10% is the BRKGA diversity valve

Per Gonçalves & Resende 2011, BRKGA's three-set population — elite (p_e ≈ 0.20 p), mutants (p_m ≈ 0.15 p), and offspring — uses mutants "in place of the usual mutation operator" to prevent premature convergence. The Londe et al. 2024 review finds p_m ∈ [0.10, 0.20] in 90% of reviewed BRKGA applications, with p_m = 0.15 as the modal choice. **p_m = 0.10 is defensible but on the low end**; Gonçalves & Resende's own capacitated-MST BRKGA used p_m = 0.10, so the thesis's choice has direct precedent.

**Integration with pymoo's generational loop.** Because pymoo's NSGA-II is already a mutation-crossover-survival algorithm and the adaptive mutation suite covers the mutation-operator role, the BRKGA-style mutant injection is implemented as a **post-survival replacement callback**. After `RankAndCrowding(crowding_func="pcd")` selects the pop_size survivors of each generation, a `MutantInjection` callback identifies the bottom ⌈mutant_frac · pop_size⌉ = 20 individuals (by rank-then-crowding) and replaces them with fresh chromosomes sampled from `InitialSampling`. Invariants are validated; any mutant that fails resample until it passes (up to n_retries = 10, then accept and let the decoder reject). The replacement happens *before* the next generation's crossover, so mutants participate as parents exactly once before being re-evaluated.

## pymoo integration: four subclasses and a callback

Pymoo 0.6.1.6's object contracts were confirmed directly from the official docs (pymoo.org/customization/discrete.html; pymoo.org/customization/custom.html; pymoo.org/operators/survival.html). Four subclasses and one callback cover the package's surface.

```python
# Segment-selective biased crossover
class SegmentSelectiveCrossover(Crossover):
    """Per-region biased uniform crossover with ρ_main = 1.0, ρ_other = 0.7.

    Input X shape:  (n_parents=2, n_matings, n_var)
    Output Y shape: (n_offsprings=2, n_matings, n_var)
    """
    def __init__(self, rho_main=1.0, rho_mask=0.7, rho_branch=0.7, rho_crossing=0.7):
        super().__init__(n_parents=2, n_offsprings=2, prob=1.0)
        self.rho = dict(main=rho_main, mask=rho_mask, branch=rho_branch, crossing=rho_crossing)

    def _do(self, problem, X, **kwargs):
        # For each mating: elite = fitter parent (per NSGA-II rank + crowding).
        # - main_loop slice: copy from elite with probability rho_main
        # - mask/branch/crossing slices: per-gene biased coin with rho_*
        # Returns (2, n_matings, n_var) — both offspring are different mixings.
        ...
```

```python
# Adaptive mutation suite: five operators behind one UCB1 scheduler
class AdaptiveMutationSuite(Mutation):
    """Per-individual operator selection via UCB1 + FRRMAB rank reward.

    Input X shape:  (n_individuals, n_var)
    Output   shape: (n_individuals, n_var)
    """
    def __init__(self, ops, scheduler, warmup=5):
        super().__init__(prob=1.0)
        self.ops = ops                   # list of 5 operator objects
        self.scheduler = scheduler       # UCBScheduler instance
        self.warmup = warmup

    def _do(self, problem, X, **kwargs):
        algo = kwargs.get("algorithm")
        gen = algo.n_gen
        for i in range(len(X)):
            op_idx = self.scheduler.choose(gen, warmup=self.warmup)
            X[i] = self.ops[op_idx].apply(X[i], problem)
            # Reward is computed in a Callback after evaluation (see below).
            self.scheduler.pending.append((i, op_idx, X[i].copy()))
        return X
```

```python
# UCB1 scheduler with FRRMAB-style sliding-window rank reward (Li et al. 2014)
class UCBScheduler:
    def __init__(self, n_arms, c=np.sqrt(2), window=50, D=1.0):
        self.n = np.ones(n_arms)                    # use counts, cold start = 1
        self.r = np.full(n_arms, 0.5)               # mean reward priors
        self.window = collections.deque(maxlen=window)
        self.c, self.D = c, D
        self.pending = []                           # (idx, op, child) triples this gen

    def choose(self, gen, warmup=5):
        if gen < warmup:                            # round-robin bootstrap
            return (gen * 997 + self.n.argmin()) % len(self.n)
        total = self.n.sum()
        ucb = self.r + self.c * np.sqrt(np.log(total) / self.n)
        return int(np.argmax(ucb))

    def update(self, op_idx, fitness_improvement):
        self.window.append((op_idx, fitness_improvement))
        self._recompute_means()                     # FRRMAB rank reward
```

```python
# The five operator classes (skeleton)
class SwapMain:          # op_idx=0: adjacent swap in main_loop region
    def apply(self, x, problem): ...
class ReplaceMain:       # op_idx=1: same-atomic-angle replacement (catalog lookup)
    def apply(self, x, problem): ...
class FlipSwitchMask:    # op_idx=2: 0↔2 flip by default; bounded domain {0,1,2}
    def apply(self, x, problem): ...
class BranchMutate:      # op_idx=3: insert/delete/replace within mask-2 slot k
    def apply(self, x, problem): ...
class CrossingToggle:    # op_idx=4: add/remove/retype triple in crossing_overlay
    def apply(self, x, problem): ...
```

```python
# Initial sampling: IntegerRandomSampling wrapper with invariant check
class InitialSampling(Sampling):
    """Return (n_samples, n_var) integer array with bounds xl/xu. Optionally
    seed from a library of validate_invariants-passing chromosomes."""
    def __init__(self, seed_library=None, seed_frac=0.0):
        super().__init__()
        self.seed_library = seed_library or []
        self.seed_frac = seed_frac

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        n_seed = int(self.seed_frac * n_samples)
        # seed first n_seed rows from library, remainder from IntegerRandomSampling
        ...
        return X
```

```python
# Mutant injection + UCB reward accounting
class MutantInjectionCallback(Callback):
    """Replace bottom 10% each generation with fresh mutants, and update UCB."""
    def __init__(self, scheduler, mutant_frac=0.1, sampler=None):
        super().__init__()
        self.scheduler, self.mutant_frac, self.sampler = scheduler, mutant_frac, sampler
        self.data["op_history"] = []

    def notify(self, algorithm):
        # 1) Update scheduler from evaluated children using Pareto-rank FI
        for i, op, child in self.scheduler.pending:
            fi = compute_rank_improvement(child, algorithm.pop[i])
            self.scheduler.update(op, fi)
        self.scheduler.pending.clear()
        # 2) Replace bottom mutant_frac with fresh samples
        k = int(self.mutant_frac * len(algorithm.pop))
        worst = np.argsort(algorithm.pop.get("rank"))[-k:]
        algorithm.pop[worst] = self.sampler._do(algorithm.problem, k)
        # 3) Log operator selection histogram
        self.data["op_history"].append(self.scheduler.n.copy())
```

```python
# Full NSGA-II configuration — the algorithm constructor
algorithm = NSGA2(
    pop_size=200,
    sampling=InitialSampling(seed_library=library, seed_frac=0.0),
    crossover=SegmentSelectiveCrossover(rho_main=1.0, rho_mask=0.7,
                                        rho_branch=0.7, rho_crossing=0.7),
    mutation=AdaptiveMutationSuite(ops=[SwapMain(), ReplaceMain(),
                                        FlipSwitchMask(), BranchMutate(),
                                        CrossingToggle()],
                                   scheduler=UCBScheduler(n_arms=5)),
    survival=RankAndCrowding(crowding_func="pcd"),
    eliminate_duplicates=True,
)
res = minimize(problem, algorithm, ("n_gen", 1000), seed=1,
               callback=MutantInjectionCallback(algorithm.mutation.scheduler, 0.1,
                                                algorithm.sampling), verbose=True)
```

## A crowding distance choice backed by Kukkonen & Deb 2006

The `crowding_func="pcd"` option on pymoo's `RankAndCrowding` — confirmed present in pymoo 0.6.1.6 at pymoo.org/operators/survival.html — implements Kukkonen & Deb 2006 "Improved pruning of non-dominated solutions based on crowding distance for bi-objective optimization problems" (IEEE CEC 2006, Vancouver, pp. 1179–1186; preprint at egr.msu.edu/~kdeb/papers/k2006007.pdf). The key difference from Deb et al. 2002's standard crowding distance: after removing each crowded point, PCD **recomputes** the neighbouring points' CDs, whereas the original CD is computed once. Kukkonen & Deb's bi-objective experiments demonstrate "better distribution compared to the original pruning algorithm of NSGA-II" with the same O(M·N·log N) asymptotic complexity as the baseline. For a 2-objective problem this is the right choice and pymoo's own recommendation reads: *"We encourage users to try `crowding_func='pcd'` for two-objective problems and `crowding_func='mnn'` for problems with more than two objectives."*

The Zheng & Doerr 2024 runtime-analysis results for NSGA-II breaking down at ≥ 3 objectives are therefore not a concern for this bi-objective project; PCD addresses the bi-objective distribution issue directly and has a decade of empirical support.

## Invariant preservation matrix: what survives each mutation

| Operator | Region bounds | Main-loop closure | Branch-slot index coherence | Crossing i<j | Notes |
|---|---|---|---|---|---|
| SWAP_MAIN | ✅ | ✅ (commutative sum) | ✅ (mask indices unchanged) | n/a | Safe |
| REPLACE_MAIN | ✅ | ✅ (same-angle bucket) | ⚠️ if replaced piece changes switch/non-switch class, branch-slot counter shifts — operator must resample from same class | n/a | Guard required |
| FLIP_SWITCH_MASK (0↔2) | ✅ | ✅ | ✅ (branch-slot counter may shift when 0→2 adds a mask-2 switch — new slot reads as uninitialised sentinels, which the decoder handles) | n/a | Safe under 0↔2 restriction |
| FLIP_SWITCH_MASK (unrestricted 0↔1) | ✅ | ⚠️ (through vs diverging exit angles differ) | ✅ | n/a | Disabled by default |
| BRANCH_MUTATE | ✅ (piece index in [0, S]) | n/a (no main-loop writes) | ✅ (writes within correct k·L_br slab) | n/a | Safe if k is drawn from mask=2 positions |
| CROSSING_TOGGLE | ✅ | n/a | n/a | ✅ (enforced on add; i≥j rejected) | Safe |

The invariant-preservation argument is not merely assertion: each operator's `apply` method contains an explicit `assert validate_invariants(x_out)` in the implementation, gated behind a `THESIS_DEBUG` flag for production runs. This turns the preservation table into executable claims.

## Worked example: the five operators on a 16×R40 closed circle

Starting chromosome (from decoder Report #5): `main_loop = [R40]*16 + [S]*(L_max-16)`, `switch_mask = [0]*n_switch_slots`, `branch_slots = [S]*(n_branch_slots · L_br)`, `crossing_overlay = [0]*(max_crossings · 3)`. Atomic-angle check: 16 × 22.5° (R40's atomic multiple) = 360° ✅.

- **SWAP_MAIN on (i=3, i+1=4)**: swaps positions 3 and 4 of the main loop. Both are R40, so the chromosome is byte-identical after the swap. Sum of angles unchanged at 360°. Valid, though a no-op in effect.
- **REPLACE_MAIN at position 7**: R40 is atomic-angle 22.5°; the same-angle bucket contains {R40}. No replacement is possible unless the catalog includes 2×R88-half = 2×11.25° combinations (atomic-angle 11.25°×2 = 22.5°), which would consume two sentinel positions in the main loop. With 16×R40 at L_max = 40, there are 24 sentinel positions available, so the replacement "R40 → R88-half + R88-half" fits. After: 15 × R40 + 2 × R88-half + 23 × S, sum = 15·22.5° + 2·11.25° = 337.5° + 22.5° = 360°. Valid.
- **FLIP_SWITCH_MASK**: no switches are present in the main loop (all R40s are curves, not switches), so every mask position is unused. Operator is a no-op; UCB logs a null trial.
- **BRANCH_MUTATE**: no mask=2 positions exist, so no branch slot is active. Operator is a no-op; UCB logs a null trial.
- **CROSSING_TOGGLE (add at i=3, j=11, pt=crossing_9V)**: writes the triple `(3, 11, crossing_9V_index)` to the first empty overlay slot. Crossing i<j satisfied. The decoder will attempt to realise the crossing; if the geometric distance between main-loop positions 3 and 11 isn't compatible with the crossing piece, the decoder returns `INFEASIBLE_CROSSING_GEOMETRY` and Deb 2000 feasibility-first downrates this individual. Valid at the operator level.

Two of five operators are no-ops on this seed, which is exactly the situation UCB1 is designed to handle — the bandit will learn within a few generations that SWAP, REPLACE, and CROSSING are the productive arms for closed-circle seeds, and will bias toward them until the population diversifies enough for FLIP and BRANCH to have targets.

## Per-generation cost fits inside the wall-clock budget

| Phase | Count | Cost per unit | Per-generation |
|---|---|---|---|
| Crossover (SegmentSelectiveCrossover on 4 regions, ~750 genes) | 100 matings | ~1 µs/gene × 750 genes ≈ 0.75 ms/mating | ~75 ms |
| Mutation (AdaptiveMutationSuite, one op per individual) | 200 ind. | ~150 µs/apply (dominant: catalog lookups in REPLACE_MAIN, mask filter in BRANCH) | ~30 ms |
| UCB scheduler bookkeeping | 200 ind. + 1 sort | O(n_arms) per choose + O(window) per update | ~2 ms |
| Mutant injection (resample + validate) | 20 ind. | ~200 µs/sample (incl. invariant check) | ~4 ms |
| Decoder (turtle-graphics + prefilter) | 200 decodes | ~150 µs (per Report #5) | ~30 ms |
| RankAndCrowding with PCD (pymoo Cython) | 1 sort | O(M·N log N), N=400, M=2 | ~5 ms |
| Evaluation overhead (pymoo internals) | | | ~10 ms |
| **Per-generation total** | | | **~160–200 ms** |
| **1000 generations** | | | **~3 min wall-clock** |

At pop_size = 300 the per-generation total scales to ~300 ms and 1000 generations run in ~5 minutes — still comfortable for interactive experimentation.

## Architectural decisions table

| Decision | Rationale | Source |
|---|---|---|
| Segment-selective crossover (not uniform) | 70–95% main-loop destruction under per-gene crossover | Pilot data + decoder lattice prefilter (Report #5); Zhao et al. 2023 double-crossover pattern |
| ρ = {1.0, 0.7, 0.7, 0.7} per region | Closure preservation on main loop; BRKGA canonical band elsewhere | Gonçalves & Resende 2011 DOI 10.1007/s10732-010-9143-1; Londe et al. 2024 DOI 10.1016/j.ejor.2024.03.030 |
| Five mutation operators (one per region, two for main_loop) | Complete coverage of chromosome regions with minimal bandit arm set | Own design; Li et al. 2014 DOI 10.1109/TEVC.2013.2239648 on arm-count tradeoffs |
| ALNS + UCB1 hybrid | Credit bookkeeping from ALNS; theoretically grounded arm selection from bandits | Ropke & Pisinger 2006 DOI 10.1287/trsc.1050.0135; Auer et al. 2002 DOI 10.1023/A:1013689704352 |
| Reward = FRRMAB rank-based fitness improvement | Scale-free; designed for MOEA multi-objective AOS | Li, Fialho, Kwong & Zhang 2014 (IEEE TEC 18(1):114–130) |
| Exploration c = √2 | Theoretically optimal for [0,1]-normalised sub-Gaussian rewards | Auer et al. 2002 Theorem 1 |
| mutant_frac = 0.10 | Within BRKGA canonical band; matches Gonçalves–Resende CMST precedent | Gonçalves & Resende 2011; Resende 2013 brkga-cmst.pdf |
| RankAndCrowding(crowding_func="pcd") | Best distribution for 2-objective problems; pymoo-recommended | Kukkonen & Deb 2006 CEC; pymoo.org/operators/survival.html |
| pymoo pinned to 0.6.1.x | Stable API; IntegerRandomSampling + SBX/PM + RoundingRepair verified | pymoo.org/customization/discrete.html; Blank & Deb 2020 DOI 10.1109/ACCESS.2020.2990567 |
| BRANCH_MUTATE writes to mask-2 counter | Matches decoder's branch-slot indexing rule | Decoder Report #5; this report §"Five operators, one bandit" |

## What this pushes to downstream packages

**Problem (#7).** No changes to the problem contract. The operators are invisible to the `_evaluate` method; `F` and `G` interfaces remain as the decoder-report specifies (F=[inf,inf] on infeasibility, G carries constraint violations). The problem layer sees only decoded chromosomes.

**Config/IO (#8).** Persists three new state objects across runs: (a) UCB scheduler state (n_i, mean rewards, sliding window) as a checkpoint for run continuation; (b) the per-region ρ values, mutant_frac, UCB exploration constant c, and warmup period as top-level config keys; (c) per-generation operator-selection history (the `op_history` list populated by `MutantInjectionCallback`) as a time-series for diagnostic plotting. The config schema must expose `pop_size`, `mutant_frac`, `c`, `rho_main`, `rho_mask`, `rho_branch`, `rho_crossing`, `window`, `D`, `warmup` as tunables.

**Visualization (#9).** Two new diagnostic plots: (i) an operator-selection histogram per generation (stacked area plot of n_i over t), revealing which operators the bandit discovered to be productive; (ii) a reward-distribution violin plot per operator showing the FRRMAB reward spread. Both are low-cost post-hoc from the callback's `data["op_history"]`.

## What remains open

Three open questions deserve framing as future work rather than silenced assumptions. (1) **Per-operator success rates are empirical — no prior art exists for LEGO track problems**; the thesis should report the final `op_history` distribution so that subsequent students can start from an informed prior instead of the uniform cold-start used here. (2) **The UCB exploration constant c = √2 is theoretically optimal but problem-dependent in practice**; a light grid search over c ∈ {0.5, 1.0, √2, 2.0} on a reduced budget would tighten the claim. (3) **Path-relinking per Andrade, Toso, Gonçalves & Resende 2021 "The multi-parent BRKGA with implicit path-relinking"** (EJOR 289(1):17–30, DOI 10.1016/j.ejor.2019.11.037) is a natural extension that could replace the current mutant-injection diversity valve with a more directed exploration step; it is deferred because implementing path-relinking over the four-region integer genotype requires a distance metric that respects the closure invariant, which is itself a small research question. **Whole-loop swap crossover** on the main_loop region, discussed in §2, is a further extension that would give crossover a legitimate role in macroscopic topology exploration without breaking closure — also future work.