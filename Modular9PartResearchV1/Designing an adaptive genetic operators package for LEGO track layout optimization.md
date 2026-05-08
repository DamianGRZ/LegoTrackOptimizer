# Designing an adaptive genetic operators package for LEGO track layout optimization

**A segment-selective BRKGA crossover combined with a five-operator ALNS-UCB mutation suite provides a principled, literature-grounded architecture for optimizing modular train track layouts via NSGA-II in pymoo.** The crossover freezes the main-loop chromosome region (ρ=1.0) to preserve angular closure while allowing standard biased recombination (ρ=0.7) on switch, branch, and crossing regions. The mutation suite uses UCB1 with a sliding window of W≈50 generations to adaptively select among five region-targeted operators, with reward feedback flowing through a pymoo Callback that observes pre/post fitness. This report synthesizes the AOS literature, UCB theory, ALNS principles, pymoo integration patterns, and concrete implementation designs needed to build the complete `operators/` package.

---

## Segment-selective crossover preserves closure while enabling diversity

Standard BRKGA biased uniform crossover (Gonçalves & Resende, 2011, *Journal of Heuristics* 17(5):487–525) inherits each gene from the elite parent with probability ρ_e and from the non-elite parent with probability 1−ρ_e. The canonical setting is **ρ_e ∈ [0.6, 0.8]**, with ρ_e≈0.7 as the most common choice. The key innovation for track layouts is making ρ position-dependent via a probability mask vector.

**Why ρ=1.0 on the main loop.** The main-loop region encodes a sequence of angular decisions that must collectively satisfy a geometric closure constraint—the track must return to its starting point. This constraint couples *all* main-loop genes together: mixing genes from Parent A's closed loop with Parent B's closed loop destroys closure in **70–95% of offspring** because the combined angular sequence almost certainly fails to sum to a valid closed path. Setting ρ=1.0 means `rand < 1.0` is always true for uniform random values in [0, 1), so the elite parent's entire main-loop region is inherited intact. This is analogous to how TSP crossover operators (PMX, OX, GPX) avoid naive gene swapping that destroys tour validity.

**Why ρ=0.7 on other regions.** Switch mask, branch slot, and crossing overlay genes encode decisions that are relatively independent—each switch state or branch configuration can be mixed freely without violating structural constraints. Standard biased crossover at ρ=0.7 provides the exploitation/exploration balance that makes BRKGA effective on these regions.

The implementation constructs a static probability mask from the SegmentMap and applies it via vectorized NumPy operations:

```python
class SegmentSelectiveCrossover(Crossover):
    def __init__(self, rho_main=1.0, rho_other=0.7):
        super().__init__(n_parents=2, n_offsprings=2, prob=0.9)
        self.rho_main = rho_main
        self.rho_other = rho_other

    def _do(self, problem, X, **kwargs):
        seg = problem.segment_map
        _, n_matings, n_var = X.shape
        Y = np.empty((2, n_matings, n_var), dtype=np.int16)

        # Build probability mask: 1.0 for main loop, 0.7 elsewhere
        rho = np.full(n_var, self.rho_other)
        rho[seg.main_loop_start:seg.main_loop_end] = self.rho_main

        rand = np.random.random((n_matings, n_var))
        elite_mask = rand < rho[np.newaxis, :]  # broadcast

        # Offspring 1: elite-biased
        Y[0] = np.where(elite_mask, X[0], X[1])
        # Offspring 2: complementary on non-frozen regions
        frozen = np.zeros(n_var, dtype=bool)
        frozen[seg.main_loop_start:seg.main_loop_end] = True
        Y[1] = np.where(frozen[np.newaxis, :], X[0],
                        np.where(~elite_mask, X[0], X[1]))
        return Y.astype(np.int16)
```

Both offspring inherit the elite parent's main loop identically. They differ only in the non-main-loop regions, where Offspring 2 receives the complement of Offspring 1's gene selections. This follows Bean's (1994) random-key GA tradition while extending it with the segment-selective mask—a design that has clear parallels to Whitley's **partition crossover** (GECCO 2009, Best Paper), which decomposes the chromosome into independent subproblems and crossovers within each partition separately. Both approaches recognize that genes participating in coupled constraints should be inherited as a unit.

---

## Three-component AOS architecture drives adaptive mutation

Adaptive Operator Selection decomposes into three components (Fialho, 2010, PhD thesis, Université Paris-Sud XI; Pei et al., 2025, *IEEE Trans. AI* 6(8):1991–2012):

- **Operator pool**: the five mutation operators (arms of the bandit)
- **Credit assignment**: the reward signal measuring operator effectiveness
- **Operator selection rule**: the bandit algorithm balancing exploration and exploitation

The FRRMAB framework (Li, Fialho, Kwong & Zhang, 2014, *IEEE TEVC* 18(1):114–130, DOI: 10.1109/TEVC.2013.2239648) demonstrated that UCB-based selection with ranked Fitness Improvement Rates inside a sliding window significantly improves MOEA/D. Gonçalves & Almeida (2015, EMO, DOI: 10.1007/978-3-319-15934-8_28) showed UCB-Tuned outperforms both probability matching and FRRMAB by incorporating observed variance. Thierens (2005, GECCO, DOI: 10.1145/1068009.1068251) proposed Adaptive Pursuit as an alternative: a winner-takes-all scheme with learning rate α and pursuit rate β that converges faster to the best operator than probability matching, but lacks the exploration guarantees of UCB.

For the track layout optimizer, **UCB1 with a sliding window** is the recommended starting point: it has formal regret bounds, requires minimal tuning, and handles the five-operator pool well. UCB-Tuned is the upgrade path when arm variances differ substantially.

---

## UCB1 and UCB-Tuned formulas for operator selection

The foundational paper is Auer, Cesa-Bianchi & Fischer (2002, *Machine Learning* 47(2–3):235–256, DOI: 10.1023/A:1013689704352). The formulas below govern operator selection in the `UCBSelector` class.

**UCB1** selects the operator j maximizing:

> UCB1_j = x̄_j + c · √(ln n / n_j)

where x̄_j is the empirical mean reward of operator j, n is total plays, n_j is plays of arm j, and **c = √2** is theoretically optimal for [0,1]-bounded rewards. The regret bound is O(K · ln n), logarithmic in time. In practice, AOS implementations typically tune **c ∈ [0.5, 1.0]** because the theoretical √2 over-explores in finite-horizon settings.

**UCB-Tuned** replaces the fixed confidence bound with a variance-aware term:

> UCB-Tuned_j = x̄_j + √( (ln n / n_j) · min(1/4, V_j) )

where V_j = σ̂²_j + √(2 · ln n / n_j) is an upper confidence bound on the variance, and σ̂²_j is the sample variance of arm j's rewards. The min(1/4, ·) clips the variance estimate since [0,1] rewards have maximum variance 1/4. UCB-Tuned has no formal regret proof but consistently outperforms UCB1 empirically.

**UCB-V** (Audibert, Munos & Szepesvári, 2009, *Theoretical Computer Science* 410(19):1876–1902) provides formal variance-dependent bounds:

> UCB-V_j = x̄_j + √(2ε · σ̂²_j · ln n / n_j) + 3b · c · ln n / n_j

with regret O(Σ σ²_a · ln n / Δ_a), tighter than UCB1 when variances are small. The extra hyperparameters (ε, c, b) make it less practical for the five-operator setting where UCB-Tuned suffices.

| Algorithm | Best for | Formal guarantees | Complexity |
|-----------|----------|-------------------|------------|
| UCB1 | General-purpose, unknown variance | O(K ln n) regret | Minimal—one parameter c |
| UCB-Tuned | Heterogeneous arm variances | Empirical only | Tracks per-arm variance |
| UCB-V | Variance-sensitive with guarantees | O(Σ σ² ln n / Δ) | Three hyperparameters |

---

## Sliding windows handle non-stationary operator rewards

Operator reward distributions shift as the population evolves—an operator effective for early exploration becomes less useful during convergence. Three approaches handle this non-stationarity:

**Sliding window (SLMAB)** computes statistics using only the last W applications across all operators. With **W ≈ 50 generations** (approximately 50 × population_size total applications), the selector forgets stale information while retaining enough history for stable estimates. The tradeoff is fundamental: small W (≈25) reacts quickly but produces noisy estimates; large W (≈200) is stable but slow to adapt. FRRMAB (Li et al., 2014) recommends W proportional to 0.5 × population size × generations per window.

**Dynamic MAB (DMAB)** by Da Costa, Fialho, Schoenauer & Sebag (GECCO 2008) combines UCB with the **Page-Hinkley change detection test**. During stationary periods, standard UCB runs normally. When the Page-Hinkley test detects a distribution shift (cumulative deviation exceeds threshold γ), all statistics are reset and exploration restarts. The Page-Hinkley test tracks cumulative deviations: a change is detected when M_t − |m_t| > γ, where m_t = Σ(r_i − r̄_i + δ) and M_t = max|m_i|. DMAB excels at abrupt regime changes but is overly aggressive for the gradual shifts typical in evolutionary search.

**Discounted UCB** applies exponential decay: x̄_j^(γ) = Σ γ^(n_j−s) · x_{j,s} / Σ γ^(n_j−s) with discount factor γ ∈ [0.9, 0.99]. This naturally forgets old information without abrupt restarts, providing smoother adaptation than sliding windows.

For the track layout optimizer, **sliding window with W ≈ 50 generations** is recommended as the simplest effective approach. The window stores tuples of (operator_index, reward) and computes per-operator statistics on demand. Discounted UCB is the natural upgrade if the sliding window proves too coarse.

---

## Credit assignment via Fitness Improvement Rate

The reward signal must measure how much each operator application contributed to search progress. The **Fitness Improvement Rate (FIR)** is the most widely validated signal:

> FIR = (f_parent − f_offspring) / |f_parent|

where positive FIR indicates improvement (minimization). For multi-objective problems with NSGA-II, FIR can be computed per-objective using a scalarization (e.g., Tchebycheff), or alternatively using **Pareto rank change** (reward = 1 if offspring enters a better non-dominated front). Hypervolume contribution change (ΔHV) is theoretically appealing but computationally prohibitive.

Fialho et al. (2008, 2009) demonstrated that **extreme value credit assignment** (using the maximum FIR in the window rather than the average) outperforms average-based assignment because "rare but highly beneficial jumps matter as much as frequent small improvements." The FRRMAB framework takes this further by ranking FIR values within the window and using ranks as rewards, which provides invariance to fitness scaling—critical as the scale of improvements shrinks during convergence.

For the implementation, FIR values should be **min-max normalized** within the sliding window to [0, 1] before feeding to UCB1, since the UCB1 regret bound assumes bounded rewards. The `OperatorTrackingCallback` observes pre-mutation fitness (from `algorithm.pop`) and post-mutation fitness (from `algorithm.off`), computes FIR per individual, and calls `ucb_selector.update(operator_index, reward)`.

---

## Five mutation operators span the exploration-exploitation spectrum

Each operator targets a specific chromosome region with a distinct search behavior, following the ALNS destroy/repair paradigm (Ropke & Pisinger, 2006, *Transportation Science* 40(4):455–472, DOI: 10.1287/trsc.1050.0135). In ALNS, competing destroy/repair heuristics are selected adaptively based on historical performance—this maps directly to our UCB-selected mutation pool.

**Operator 1: Random segment swap** exchanges gene segments between non-main-loop regions. Two random positions within switch/branch/crossing regions are selected, and a segment of length 1–max_len is swapped between them. This is a pure **exploration** operator: it rearranges existing genetic material without introducing new values, creating novel combinations of track features. Cost is O(swap_len). Best in early generations.

```python
def _random_segment_swap(self, x, seg):
    child = x.copy()
    regions = [(seg.switch_start, seg.switch_end),
               (seg.branch_start, seg.branch_end),
               (seg.crossing_start, seg.crossing_end)]
    r1, r2 = [regions[i] for i in np.random.choice(3, 2)]
    p1 = np.random.randint(r1[0], r1[1])
    p2 = np.random.randint(r2[0], r2[1])
    L = min(r1[1]-p1, r2[1]-p2, np.random.randint(1, 5))
    child[p1:p1+L], child[p2:p2+L] = x[p2:p2+L].copy(), x[p1:p1+L].copy()
    return child
```

**Operator 2: Worst removal + greedy repair** identifies genes in high-penalty regions and regenerates them with random valid values. This is the most ALNS-faithful operator: "destroy" removes ~20% of non-main-loop genes weighted by per-region constraint violation, and "repair" fills them with random values within valid bounds. It is an **exploitation** operator that directly targets constraint violations. Requires per-region penalty tracking (stored on the Problem or passed via the Callback).

**Operator 3: Switch toggle** flips k random switch states in the switch mask region via XOR (`child[idx] ^= 1`). With k ∈ {1, 2, 3}, this is a minimal-perturbation **exploitation** operator ideal for fine-tuning connectivity in late generations. Cost is O(k).

**Operator 4: Branch perturbation** modifies branch slot genes either by small offsets (±1 to ±3, clipped to bounds) or full regeneration. Offset mode provides exploitation; regeneration mode provides exploration. A **mixed** operator useful across all generation phases.

**Operator 5: Curve radius shift** adjusts main-loop gene values by ±1 or ±2 within per-gene bounds to change curve radius selections. This is the **highest-risk** operator because it can break angular closure, so only 1–2 genes are shifted per application. It must be paired with post-mutation validation. If closure is violated, the mutation is reverted. Pure exploitation for late-stage geometry refinement.

| Operator | Region | Type | Phase | Risk | Cost |
|----------|--------|------|-------|------|------|
| Random segment swap | Switch/Branch/Crossing | Exploration | Early | Low | O(L) |
| Worst removal + repair | Switch/Branch/Crossing | Exploitation | Mid-late | Low | O(n+k) |
| Switch toggle | Switch mask | Exploitation | Late | Very low | O(k) |
| Branch perturbation | Branch slots | Mixed | All | Low | O(k) |
| Curve radius shift | Main loop | Exploitation | Late | Medium-high | O(k) |

---

## pymoo integration requires subclassing Crossover, Mutation, and Callback

The pymoo framework (v0.6+) provides clean extension points. The critical API shapes are:

**Crossover._do(self, problem, X, \*\*kwargs)**: receives X with shape **(n_parents, n_matings, n_var)** and must return shape **(n_offsprings, n_matings, n_var)**. The `problem` argument carries custom attributes like `segment_map`. The `algorithm` object is accessible via `kwargs.get('algorithm')`, passed from `GeneticAlgorithm._infill()`.

**Mutation._do(self, problem, X, \*\*kwargs)**: receives X with shape **(n_individuals, n_var)** and returns the same shape. Mutation probability is handled by the base class; `_do` receives only the individuals selected for mutation.

**Callback.notify(self, algorithm)**: called once per generation with full access to `algorithm.pop` (current population), `algorithm.off` (offspring), `algorithm.opt` (Pareto front), `algorithm.n_gen` (generation), and `algorithm.problem`. Data persists in `self.data` dict.

The SegmentMap is stored on the Problem subclass and accessed as `problem.segment_map` inside both crossover and mutation operators. This pattern is idiomatic in pymoo—the Problem is the natural carrier for problem-specific metadata.

**Per-individual operator dispatch** is the key integration challenge. pymoo assumes one mutation operator per generation by default. The solution is to implement `ALNSMutationSuite` as a single Mutation subclass that internally dispatches to one of five sub-operators per individual:

```python
class ALNSMutationSuite(Mutation):
    def __init__(self, ucb_selector):
        super().__init__(prob=1.0)
        self.ucb_selector = ucb_selector
        self.operators = [
            random_segment_swap,
            worst_removal_repair,
            switch_toggle,
            branch_perturbation,
            curve_radius_shift
        ]
        self._last_selections = []  # track which op was used per individual

    def _do(self, problem, X, **kwargs):
        seg = problem.segment_map
        n_ind, n_var = X.shape
        self._last_selections = []

        for i in range(n_ind):
            op_idx = self.ucb_selector.select()
            X[i] = self.operators[op_idx](X[i], seg)
            self._last_selections.append(op_idx)
        return X.astype(np.int16)
```

The `_last_selections` list is read by the `OperatorTrackingCallback` after fitness evaluation to assign credit. This avoids the need for a custom Mating class—the existing NSGA-II pipeline works unmodified.

---

## UCBSelector and OperatorTrackingCallback close the feedback loop

The `UCBSelector` class is the only stateful component. It maintains per-operator counts and reward histories within a sliding window:

```python
class UCBSelector:
    def __init__(self, n_operators=5, c=1.0, window_size=50):
        self.K = n_operators
        self.c = c
        self.W = window_size
        self.history = deque(maxlen=window_size * n_operators)
        self.total_plays = 0

    def select(self):
        # Ensure each arm played at least once
        counts = np.zeros(self.K)
        rewards = np.zeros(self.K)
        for op_idx, reward in self.history:
            counts[op_idx] += 1
            rewards[op_idx] += reward

        unplayed = np.where(counts == 0)[0]
        if len(unplayed) > 0:
            return np.random.choice(unplayed)

        means = rewards / counts
        n_total = counts.sum()
        ucb_values = means + self.c * np.sqrt(np.log(n_total) / counts)
        return int(np.argmax(ucb_values))

    def update(self, op_idx, reward):
        self.history.append((op_idx, reward))
        self.total_plays += 1
```

The `OperatorTrackingCallback` bridges pymoo's generation loop and the UCB selector:

```python
class OperatorTrackingCallback(Callback):
    def __init__(self, mutation_suite):
        super().__init__()
        self.mutation_suite = mutation_suite
        self.data["op_history"] = []

    def notify(self, algorithm):
        if algorithm.off is None:
            return
        parent_F = algorithm.pop.get("F")
        off_F = algorithm.off.get("F")
        selections = self.mutation_suite._last_selections

        for i, op_idx in enumerate(selections):
            if i < len(off_F):
                # Scalarize for multi-objective (e.g., sum of objectives)
                f_parent = parent_F[i % len(parent_F)].sum()
                f_off = off_F[i].sum()
                fir = max(0, (f_parent - f_off) / (abs(f_parent) + 1e-12))
                self.mutation_suite.ucb_selector.update(op_idx, fir)
                self.data["op_history"].append(
                    (algorithm.n_gen, op_idx, fir))
```

The complete wiring in the NSGA-II constructor:

```python
ucb = UCBSelector(n_operators=5, c=0.8, window_size=50)
mutation = ALNSMutationSuite(ucb)
crossover = SegmentSelectiveCrossover(rho_main=1.0, rho_other=0.7)
callback = OperatorTrackingCallback(mutation)

algorithm = NSGA2(
    pop_size=100,
    crossover=crossover,
    mutation=mutation,
)
res = minimize(problem, algorithm, ('n_gen', 300),
               seed=42, callback=callback)
```

Thread safety is not a concern for operators: pymoo parallelizes only fitness evaluation (`_evaluate`), while all genetic operators run single-threaded in the main process. The UCBSelector's shared state is only accessed sequentially.

---

## Connections to partition crossover and gray-box optimization

The segment-selective crossover shares deep conceptual roots with Whitley's **partition crossover** (GECCO 2009, Best Paper; PPSN 2010). Partition crossover forms the union graph of two parent solutions, removes common edges, identifies connected components, and independently selects each component from one parent—effectively decomposing the problem into independent subproblems. Both approaches respect structural boundaries: segment-selective freezes tightly coupled regions; partition crossover keeps connected components together.

| Aspect | Segment-selective crossover | Partition crossover |
|--------|----------------------------|---------------------|
| Decomposition | Static, from SegmentMap | Dynamic, per parent pair |
| Problem knowledge | Region structure (gray-box) | Variable interaction graph (gray-box) |
| Boundary respect | Freezes coupled main loop | Keeps connected components intact |
| Offspring quality | Stochastic on free regions | Optimal over 2^q combinations |
| Complexity | O(n) vectorized | O(n) graph decomposition |

The key distinction is that segment-selective crossover uses **fixed** regions defined by the chromosome encoding, while partition crossover discovers regions **dynamically** from the parent pair. For the track layout problem, the fixed-region approach is appropriate because the structural constraints (angular closure on main loop) are inherent to the encoding, not dependent on specific solution values.

---

## Empirical validation requires controlled comparison and ablation

**Crossover comparison.** Run segment-selective (ρ_main=1.0, ρ_other=0.7) against standard BRKGA (uniform ρ=0.7) over 30+ independent runs. Track median final fitness, convergence speed (generation to reach 90% of best), angular closure violation rate, and population diversity (entropy per region). The hypothesis is that segment-selective crossover dramatically reduces closure violations while maintaining diversity in non-main-loop regions.

**Ablation study.** Run all-five-operators baseline against five remove-one variants plus uniform-random-selection and single-best-operator baselines. Use **Kruskal-Wallis H test** (omnibus) followed by pairwise **Wilcoxon rank-sum tests with Bonferroni correction**. Report Vargha-Delaney A₁₂ effect sizes. If removing Worst Removal + Repair degrades performance significantly while removing Switch Toggle does not, the adaptive selector should learn this—and the ablation confirms the UCB is working correctly.

**Hyperparameter sensitivity.** The three critical parameters are:

- **c** (UCB exploration constant): sweep [0.1, 0.5, 0.8, 1.0, 1.5, √2]. Theory suggests √2; practice favors 0.5–1.0.
- **W** (sliding window): sweep [25, 50, 100, 200]. Small W is responsive but noisy; large W is stable but slow.
- **ρ_other** (non-main-loop crossover bias): sweep [0.5, 0.6, 0.7, 0.8]. Values below 0.5 make BRKGA unbiased; above 0.8 risks premature convergence.

Interaction plots (2D heatmaps of best fitness vs. parameter pairs) reveal whether parameters are independent or coupled. The FRRMAB literature (Li et al., 2014) suggests W and c interact: larger W needs smaller c because more data reduces the need for exploration.

---

## Conclusion

The `operators/` package architecture rests on three pillars. First, the **segment-selective crossover** extends BRKGA's biased uniform crossover with a position-dependent probability mask, preserving angular closure on the main loop while enabling recombination elsewhere—a design well-grounded in partition crossover theory. Second, the **ALNS-UCB mutation suite** adapts the destroy/repair paradigm from operations research to evolutionary mutation, with five region-targeted operators spanning exploration to exploitation, selected via UCB1 with sliding-window non-stationarity handling. Third, the **pymoo integration** is clean: Crossover and Mutation subclasses receive the SegmentMap via the Problem reference, per-individual operator dispatch happens inside a single Mutation wrapper, and a Callback closes the reward feedback loop. The key insight from the AOS literature is that UCB-based selection with FIR-ranked rewards in a sliding window (FRRMAB pattern) outperforms both static operator allocation and simple probability matching. The five operators are not arbitrary—each targets a specific region and search phase, and the UCB selector learns their relative value as the population evolves.