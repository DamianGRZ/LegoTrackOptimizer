# Evolutionary computation techniques for constrained LEGO track optimization

This report synthesizes academic research across seven critical areas for improving a genetic algorithm that optimizes closed LEGO railway track layouts using random-key encoding and a greedy construction decoder within pymoo 0.6.1.6. **BRKGA with adaptive ε-constraint handling, sliding-window adaptive operator selection, Lamarckian local search on RK vectors, and island-model diversity preservation emerge as the strongest ensemble of techniques** for this 218-variable, 5-constraint problem. Each topic below provides a specific recommended approach with parameters calibrated for a population of 1000 and up to 1000 generations.

---

## Topic 1: BRKGA vs Standard GA for random-key construction problems

### Recommended approach

**Biased Random-Key Genetic Algorithm (BRKGA)** — Gonçalves & Resende, 2011.

### Mechanism

BRKGA partitions the population into three groups every generation. The **elite set** (top *p_e* individuals by fitness) is copied directly to the next generation. **Offspring** are produced by parameterized uniform crossover between one elite parent and one non-elite parent, where each gene is inherited from the elite parent with probability ρ_e > 0.5. **Mutants** are entirely new random vectors in [0,1]^n injected to maintain diversity. No traditional mutation operator exists — mutants replace it entirely.

Step-by-step per generation:

1. Evaluate all *p* individuals via the construction decoder.
2. Rank by fitness (incorporating constraint handling — see Topic 2).
3. Copy the top *p_e* individuals unchanged to the next generation.
4. Generate *p_o* offspring: for each, select one parent uniformly from the elite set and one from the non-elite set. For each of the 218 gene positions, inherit the elite parent's value with probability ρ_e, otherwise inherit the non-elite parent's value.
5. Generate *p_m* mutants as fresh random vectors drawn uniformly from [0,1]^218.
6. The new population of size *p = p_e + p_o + p_m* replaces the old one completely.

### Why it fits this problem

BRKGA was purpose-built for random-key encoded problems with construction decoders. The elite-biased crossover preserves approximately **70% of gene values** from known-good construction sequences, maintaining the positional building blocks that produce effective track layouts. Since the greedy decoder reads RK values as priorities for sequential piece placement, gene-position-specific inheritance directly preserves learned construction orderings. The universal crossover operator requires zero problem-specific design — only the decoder needs domain knowledge. Mutant injection provides continuous diversity without the parameter sensitivity of mutation rate tuning.

The double-elitism mechanism (guaranteed survival plus biased inheritance) converges faster than tournament-based standard GAs on RK problems. Gonçalves, Resende & Toso (2014) demonstrated BRKGA converges faster than both Bean's original RKGA and a modified RKGA with bias across set covering problems.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Population *p* | 1000 | Given; adequate for 218 variables |
| Elite *p_e* | **200 (20%)** | Standard BRKGA default from Toso & Resende (2015) API paper |
| Offspring *p_o* | **700 (70%)** | Remainder after elite + mutants |
| Mutants *p_m* | **100 (10%)** | Baseline diversity; increase to 150 if premature convergence observed |
| Bias ρ_e | **0.70** | Standard starting point; ~153 of 218 genes from elite parent |

```python
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

algorithm = BRKGA(
    n_elites=200,
    n_offsprings=700,
    n_mutants=100,
    bias=0.70,
    eliminate_duplicates=MyDuplicateElimination()
)
```

**Tuning levers:** If <5% of population is feasible after generation 100, temporarily increase mutants to 200 and decrease offspring to 600. If diversity collapses (average gene-wise σ < 0.05), reduce ρ_e to 0.65. If convergence is too slow, increase ρ_e to 0.75.

### Pitfalls

BRKGA's double-elitism creates strong convergence pressure that can cause **premature convergence** — the most commonly reported weakness in the 2025 review by Londe et al. The elite set can become genetically homogeneous within 100–200 generations. Mitigation requires the shaking/restart mechanisms detailed in Topic 7. Additionally, **custom crossover and mutation operators are incompatible** with standard BRKGA architecture — the existing switch-preserving crossover and 5 mutation operators would need to be abandoned or adapted. The current algorithm's structural niching (33% reserved for switch/crossing pieces) also needs reimplementation, potentially through decoder-side logic rather than population-level partitioning.

Repair operators that modify the RK vector *are* compatible with BRKGA — Toso & Resende (2015) explicitly support read-write decoders where the chromosome is modified during evaluation. Repaired chromosomes entering the elite carry corrected genetic information that improves future crossover effectiveness.

### Alternatives considered

**Standard GA with tournament selection:** Provides more operator flexibility (custom crossover, multiple mutations) but lacks BRKGA's clean elitist structure. Tournament selection applies no structural bias toward preserving elite genes position-by-position. For RK+decoder problems, this means worse inheritance of positional building blocks. Rejected as primary approach but could serve as an island variant (see Topic 5).

**BRKGA-MP-IPR** (Andrade et al., 2021): Multi-parent crossover with 3 elite and 5 total parents, plus Implicit Path-Relinking in [0,1]^n space. Demonstrated consistent improvement over standard BRKGA across three real-world problems. However, pymoo does not implement BRKGA-MP; it requires the separate `brkga-mp-ipr` Python package. **Worth pursuing as a Phase 2 upgrade** if standard BRKGA plateaus, as IPR adds free problem-independent local search. The additional parameters (number of parents, bias function) add tuning complexity.

### Key references

- Bean, J.C. (1994). "Genetic algorithms and random keys for sequencing and optimization." *ORSA Journal on Computing*, 6(2), 154–160.
- Gonçalves, J.F. & Resende, M.G.C. (2011). "Biased random-key genetic algorithms for combinatorial optimization." *Journal of Heuristics*, 17(5), 487–525.
- Toso, R.F. & Resende, M.G.C. (2015). "A C++ application programming interface for biased random-key genetic algorithms." *Optimization Methods and Software*, 30(1), 81–93.
- Andrade, C.E., Toso, R.F., Gonçalves, J.F. & Resende, M.G.C. (2021). "The Multi-Parent Biased Random-Key Genetic Algorithm with Implicit Path-Relinking and its real-world applications." *European Journal of Operational Research*, 289(1), 17–30.
- Londe, M.A., Pessoa, L.S., Andrade, C.E. & Resende, M.G.C. (2025). "Biased random-key genetic algorithms: A review." *European Journal of Operational Research*, 321(1), 1–22.

---

## Topic 2: Adaptive constraint handling for construction-decoder problems

### Recommended approach

**Hybrid adaptive ε-constrained method** combining Takahama & Sakai's ε-level comparison (2006, 2010) with Fan et al.'s (2019) feasibility-ratio feedback, augmented by per-constraint normalization from Farmani & Wright (2003).

### Mechanism

The ε-constrained method modifies Deb's feasibility-first comparison rules by introducing a relaxation parameter ε that gradually tightens from permissive to strict.

**Comparison rules at generation *t*:** Given two individuals with objectives *f₁, f₂* and total constraint violations *φ₁, φ₂*:

1. If both φ₁ ≤ ε(t) and φ₂ ≤ ε(t): compare by objective *f* (both are "ε-feasible").
2. If one is ε-feasible and the other is not: prefer the ε-feasible one.
3. If both φ₁ > ε(t) and φ₂ > ε(t): prefer lower constraint violation.

When ε = ∞, constraints are ignored. When ε = 0, this reduces exactly to Deb's feasibility-first rules.

**Initialization:** Sort the initial population by total constraint violation. Set **ε₀ = CV of the θ-th individual**, where θ = 0.2 × N. For pop=1000, this is the 200th most-feasible individual's CV.

**Decay schedule (Takahama 2010):**
```
ε(t) = ε₀ × (1 − t/Tc)^cp    for t < Tc
ε(t) = 0                        for t ≥ Tc
```

**Fan et al. feasibility-ratio feedback:** Every 10 generations, compute RFS = (number feasible) / (population size). If RFS drops below a target ratio, slow the ε decay (or temporarily increase ε). If RFS exceeds twice the target, accelerate decay. This bidirectional adjustment prevents population collapse to all-infeasible — critical for construction decoders where feasibility is fragile.

**Per-constraint normalization (Farmani & Wright):** Normalize each constraint violation by its maximum violation in the current population: φ_normalized = Σⱼ wⱼ × max(0, gⱼ(x)) / max_pop(max(0, gⱼ)). This ensures easy constraints (inventory, boundary) and hard constraints (closure) contribute proportionally.

### Why it fits this problem

The construction decoder's greedy nature means feasible closed-loop solutions are rare initially — potentially <1% of a random population. Pure Deb's feasibility-first would select almost exclusively on constraint violation for hundreds of generations, ignoring objective quality entirely. The ε-relaxation allows **promising-but-infeasible solutions to survive**, maintaining objective-driven selection pressure even when few solutions are feasible. The per-constraint normalization directly addresses the heterogeneous constraint difficulty: position/angle closure errors operate on geometric scales (mm, degrees) while inventory excess is a piece count — raw aggregation would be meaningless.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| θ (initial percentile) | **0.20** (200th individual) | Standard Takahama default |
| Tc (cutoff generation) | **600** | 60% of 1000 gens; leaves 400 gens for strict feasibility refinement |
| cp (decay exponent) | **Start at 5, minimum 3** | Slow initial decay for hard closure constraints |
| Target feasibility ratio | **0.25** | 25% feasible maintains exploration/exploitation balance |
| Feedback interval | **10 generations** | Responsive but not noisy |
| Constraint weights | closure: 3.0, angle: 3.0, loose ports: 2.0, boundary: 1.0, inventory: 1.0 | Prioritize hard constraints |

**Feasibility ratio schedule across generations:**

- **Generations 0–200:** Target FR = 5–15% (exploration, ε high)
- **Generations 200–600:** Target FR = 20–40% (gradual tightening)
- **Generations 600–800:** Target FR = 50–80% (rapid tightening)
- **Generations 800–1000:** Target FR = 95–100% (strict feasibility, ε = 0)

### Pitfalls

The ε schedule requires **careful calibration of Tc** — too early and the search hasn't explored enough infeasible space; too late and insufficient generations remain for feasible-space exploitation. If the feasibility ratio crashes to zero during tightening and the feedback mechanism increases ε, this can cause oscillatory behavior where ε bounces between values. Impose a **monotonic constraint**: ε can only increase at most once per 50 generations, and never above 50% of its previous maximum. Additionally, per-constraint ε levels (different Tc per constraint) add substantial implementation complexity — start with a single global ε and per-constraint normalization before attempting independent schedules.

### Alternatives considered

**Stochastic ranking (Runarsson & Yao, 2000):** Uses a modified bubble sort where adjacent individuals are compared by objective with probability *Pf* = 0.45, otherwise by constraint violation. Elegant and well-validated, but the O(N²) sorting cost is significant at pop=1000 (~10⁶ comparisons per generation). Could be used as an alternative ranking within one island of an island model. An adaptive variant with Pf decaying from 0.45 to 0 over generations is promising but less studied.

**Farmani & Wright (2003) self-adaptive penalty alone:** Requires zero parameters and auto-normalizes heterogeneous constraints. However, it lacks the explicit feasibility-ratio control needed for problems where feasibility is initially very rare. Better as a component (the normalization) than a standalone method.

**Mallipeddi & Suganthan (2010) ECHT ensemble:** Splits the population into 4 sub-populations each using a different constraint handling technique (Deb's rules, ε-constrained, stochastic ranking, self-adaptive penalty). Robust but adds implementation complexity and reduces effective population size per sub-strategy to 250. Consider if single-method results plateau.

### Key references

- Deb, K. (2000). "An efficient constraint handling method for genetic algorithms." *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.
- Runarsson, T.P. & Yao, X. (2000). "Stochastic ranking for constrained evolutionary optimization." *IEEE Transactions on Evolutionary Computation*, 4(3), 284–294.
- Takahama, T. & Sakai, S. (2006). "Constrained optimization by the ε constrained differential evolution with gradient-based mutation and feasible elites." *IEEE CEC 2006*, 308–315.
- Takahama, T. & Sakai, S. (2010). "Constrained optimization by the ε constrained differential evolution with an archive and gradient-based mutation." *IEEE CEC 2010*, 1–9.
- Farmani, R. & Wright, J.A. (2003). "Self-adaptive fitness formulation for constrained optimization." *IEEE Transactions on Evolutionary Computation*, 7(5), 445–455.
- Fan, Z. et al. (2019). "An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions." *Soft Computing*, 23, 12491–12510.
- Mallipeddi, R. & Suganthan, P.N. (2010). "Ensemble of constraint handling techniques." *IEEE Transactions on Evolutionary Computation*, 14(4), 561–579.

---

## Topic 3: Local search for constrained layout/routing problems

### Recommended approach

**Lamarckian memetic algorithm with RK-space neighborhood operators** — drawing from Chaves, Resende et al.'s (2024) Random-Key Optimizer (RKO) framework and Merz & Freisleben's (1997) memetic GA methodology. Apply selectively to the top 20% of the population.

### Mechanism

Local search operates directly on the continuous RK vector, exploiting four neighborhood moves defined in the RKO framework:

1. **SWAP:** Select two random gene positions *i, j*; swap their values χ[i] ↔ χ[j]. Changes the relative priority ordering of two construction steps. Cost: 1 decoder evaluation.
2. **CHANGE:** Select a random position *i*; replace χ[i] with a new random value in [0,1). Completely re-randomizes one construction decision. Cost: 1 decoder evaluation.
3. **SHIFT:** Remove gene from position *i* and reinsert at position *j*, shifting intermediate genes. Analogous to or-opt in permutation space. Cost: 1 decoder evaluation.
4. **INVERSION:** Reverse the subsequence of RK values between positions *i* and *j*. Reverses priority ordering of a construction segment. Cost: 1 decoder evaluation.

**Selective application per generation:**

1. After offspring generation, rank the full population by constrained fitness.
2. Apply LS to the top **200 individuals (20%)** using first-improvement strategy.
3. For each selected individual: attempt up to **5 LS steps** using randomly ordered neighborhoods (Randomized Variable Neighborhood Descent).
4. At each step, generate one neighbor via a randomly chosen move. If the neighbor improves the penalized objective (objective + weighted CV), accept it and continue. If no improvement after trying all 4 neighborhoods at a position, terminate LS for that individual.
5. **Lamarckian writeback:** Replace the individual's RK vector with the improved vector.

### Why it fits this problem

The RK encoding creates a natural continuous neighborhood structure where small perturbations (swap, change) produce local variations in the construction sequence. Unlike permutation-based problems where 2-opt requires careful feasibility maintenance, **RK-space moves automatically produce valid decoder inputs** — any [0,1]^218 vector is decodable. The greedy construction decoder's sensitivity to gene values means single-gene changes can cascade through the construction sequence, producing both local and occasionally global structural changes.

Lamarckian learning is strongly preferred here because the decoder is computationally expensive. Baldwinian learning (evaluate improvement but discard the improved genotype) wastes decoder evaluations — the improvement information is used only for selection, not propagated to offspring. Lamarckian writeback ensures improved construction priorities are inherited, making the evaluation cost worthwhile. El-Mihoub et al. (2020) confirm Lamarckian superiority for static optimization problems with expensive evaluation.

Selective application to the top 20% balances evaluation budget against benefit. The decoder is the bottleneck: 200 individuals × 5 LS steps = **1000 additional evaluations per generation**, effectively doubling the per-generation cost. This is acceptable if LS improves the best-feasible solution more than an equivalent number of random offspring would.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| LS type | **Lamarckian** (100% writeback) | Expensive decoder makes Baldwinian wasteful |
| Application scope | **Top 20% (200 individuals)** | Balance cost vs. benefit |
| Steps per individual | **5 maximum, first-improvement** | Diminishing returns after 3–5 steps per Merz & Freisleben |
| Steps for top 1% (10 best) | **15 maximum** | Best solutions deserve deeper refinement |
| Neighborhood selection | **RVND (random order)** | Prevents bias toward any single operator |
| Move probabilities | SWAP: 35%, CHANGE: 35%, SHIFT: 20%, INVERSION: 10% | Cheap moves prioritized |
| Total LS budget per generation | **~1000 evaluations** | Doubles generation cost; cap at 1500 |
| Integration point | **After crossover/mutation, before survival selection** | LS-improved individuals compete for elite spots |

### Pitfalls

**Cascading effects from early genes:** In the greedy construction decoder, changing a gene at position 5 can completely alter all subsequent construction decisions. SWAP and CHANGE on early-position genes produce near-random layouts rather than local improvements. Mitigate by biasing LS moves toward **later gene positions** (e.g., positions 50–218) where changes affect fewer downstream decisions. Alternatively, use a weighted probability where position *i* is selected with probability proportional to *i/n*.

**Premature convergence acceleration:** Lamarckian LS applied to elite individuals can homogenize the elite set, accelerating convergence toward a local optimum. Mitigate with the mixed strategy: apply Lamarckian to 80% of selected individuals and Baldwinian to 20%. Monitor gene-wise variance in the elite set; if it drops below 0.02, suspend LS for 20 generations.

**Reverse mapping challenge:** If LS operates in phenotype space (actual track layout) rather than RK space, mapping improved layouts back to RK vectors is non-trivial. The RK-space approach avoids this entirely — all moves operate directly on gene values.

### Alternatives considered

**Pure Baldwinian learning:** Preserves diversity better but wastes 50% of LS evaluation cost (improvements not inherited). Rejected as primary strategy for this expensive-decoder problem.

**Full 2-opt-style exhaustive local search:** Evaluating all O(n²) ≈ 23,000 SWAP neighbors per individual is prohibitively expensive. First-improvement with a budget cap is essential.

**Problem-specific geometric LS (move a track piece):** Requires deep decoder integration and phenotype-to-genotype mapping. Higher potential payoff but significantly more implementation effort. Consider as Phase 2 enhancement.

### Key references

- Moscato, P. (1989). "On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms." Technical Report C3P 826, Caltech.
- Merz, P. & Freisleben, B. (1997). "Genetic local search for the TSP: New results." *IEEE ICEC*, 159–163.
- Neri, F. & Cotta, C. (2012). "Memetic algorithms and memetic computing optimization: A literature review." *Swarm and Evolutionary Computation*, 2, 1–14.
- Prins, C. (2004). "A simple and effective evolutionary algorithm for the vehicle routing problem." *Computers & Operations Research*, 31(12), 1985–2002.
- Chaves, A.A., Resende, M.G.C. et al. (2024). "A Random-Key Optimizer for Combinatorial Optimization." *Journal of Heuristics* / arXiv:2411.04293.
- El-Mihoub, T.A., Hopgood, A. & Nolle, L. (2020). "Self-adaptive learning for hybrid genetic algorithms." *Evolutionary Intelligence*, Springer.

---

## Topic 4: Adaptive operator selection for multi-operator mutation

### Recommended approach

**Sliding-Window Multi-Armed Bandit (SlMAB) with extreme-value credit assignment** — Fialho, Da Costa, Schoenauer & Sebag, 2010.

### Mechanism

SlMAB combines the Upper Confidence Bound (UCB1) bandit algorithm with a sliding window to handle the non-stationarity of operator quality during evolution. It selects among *K* = 5 mutation operators.

**Step-by-step algorithm:**

```
INITIALIZE:
  K = 5 mutation operators
  W = 200 (sliding window size)
  C = 1.0 (UCB exploration constant)
  window = empty FIFO queue of (operator_id, reward) tuples

PHASE 1: COLD START (generations 0–2):
  Select each operator with uniform probability 1/K = 0.2
  Collect reward statistics into window

PHASE 2: FULL AOS (generation 3+):
  FOR each mutation event:
    1. FOR each operator i = 1..K:
         q̂_i = max reward for operator i in window    [extreme-value credit]
         n_i = count of operator i applications in window
         N = total window entries
    2. SELECT operator: i* = argmax_i [ q̂_i + C × √(2·ln(N) / n_i) ]
    3. APPLY operator i* to individual → offspring
    4. COMPUTE reward r:
         IF both parent and offspring feasible:
           r = max(0, (f_offspring - f_parent) / max(|f_parent|, ε))
         IF offspring feasible AND parent infeasible:
           r = 1.0 + f_offspring / f_max      [large bonus]
         IF both infeasible:
           r = max(0, (CV_parent - CV_offspring) / CV_parent)
         IF parent feasible AND offspring infeasible:
           r = 0
    5. ADD (i*, r) to window; remove oldest if |window| > W
```

The extreme-value credit (reward = maximum improvement in the window, not average) is critical. Operators that produce occasional large jumps — even if most applications fail — are properly rewarded. This matches the reality of mutation operators where most applications are neutral but rare applications find breakthrough improvements.

### Why it fits this problem

The 5 mutation operators likely have very different success profiles: some may frequently produce small improvements (Gaussian perturbation) while others rarely help but occasionally produce large fitness jumps (segment shuffle). **Average credit would undervalue the high-variance operators.** Extreme-value credit ensures the "occasionally brilliant" operator gets selected proportionally to its potential.

With pop=1000, each generation produces ~700–1000 mutation events (depending on how many offspring undergo mutation). A window of W=200 covers roughly **0.2–0.3 generations**, providing responsive adaptation. The UCB1 exploration term ensures every operator gets tested regularly even if its recent performance is poor — critical because operator effectiveness changes as the population evolves from infeasible exploration to feasible exploitation.

The constrained credit assignment handles the transition between constraint-satisfaction and objective-improvement phases. Early in the search, operators that reduce constraint violation get high rewards. Later, only fitness improvement matters.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| K (operators) | **5** | Given |
| Window W | **200** | ~40 samples per operator; responsive to changes |
| UCB constant C | **1.0** | Standard for normalized [0,1] rewards |
| Cold-start duration | **2 generations** (2000 mutation events) | ~400 samples per operator; statistically reliable |
| Minimum operator probability | **Implicit via UCB** | UCB naturally explores undersampled operators; no explicit p_min needed |
| Reward normalization | **[0, 1] clipping** | Prevents outlier rewards from dominating |

### Pitfalls

**Reward scale sensitivity:** If constraint violations and fitness improvements operate on different scales, the reward signal becomes unreliable. The multi-level credit assignment (feasible-to-feasible vs. infeasible-to-feasible) mitigates this, but test with a simpler binary reward (1 if offspring constrained-dominates parent, 0 otherwise) as a robust fallback.

**Interaction with BRKGA:** Standard BRKGA has no mutation — only biased crossover and mutant injection. If adopting BRKGA, AOS applies to: (a) selecting among different crossover bias values (ρ_e ∈ {0.6, 0.65, 0.7, 0.75, 0.8}), (b) controlling mutant fraction, or (c) post-crossover perturbation operators applied to offspring before evaluation. Option (a) is most natural: treat each ρ_e value as a "crossover operator" and use AOS to select which bias to apply per offspring.

**Window too small:** With W=200 and 5 operators, some operators may have only 20–30 samples in the window if their selection probability drops. UCB's exploration term handles this by boosting selection of undersampled operators, but extreme-value credit from very few samples is noisy. If operator performance estimates oscillate excessively, increase W to 500.

### Alternatives considered

**Probability Matching with Adaptive Pursuit (Thierens, 2005):** Simpler implementation — maintains explicit probability vector over operators. Adaptive Pursuit pushes the best operator toward p_max and others toward p_min. Performs well but slightly inferior to SlMAB in Fialho's (2010) comprehensive comparison. Good as a fallback if UCB tuning proves difficult.

**Dynamic MAB (DMAB, Da Costa et al., 2008):** Couples UCB1 with Page-Hinkley change detection, resetting statistics when operator quality shifts are detected. The γ threshold parameter is very sensitive to problem characteristics, making it harder to tune than SlMAB's window size. Offers potential benefits for long runs where operator utility changes dramatically, but adds complexity without proportional gain for 1000 generations.

**Thompson Sampling:** Bayesian approach using Beta distributions. Elegant and naturally handles exploration-exploitation. For binary rewards (success/failure), requires fewer samples than UCB. However, for continuous rewards (fitness improvement magnitude), requires Normal-Inverse-Gamma priors, adding implementation complexity. Promising but less validated for AOS specifically.

### Key references

- Fialho, Á., Da Costa, L., Schoenauer, M. & Sebag, M. (2010). "Analyzing bandit-based adaptive operator selection mechanisms." *Annals of Mathematics and Artificial Intelligence*, 60(1), 25–64.
- Thierens, D. (2005). "An adaptive pursuit strategy for allocating operator probabilities." *GECCO 2005*, 1539–1546.
- Da Costa, L., Fialho, Á., Schoenauer, M. & Sebag, M. (2008). "Adaptive operator selection with dynamic multi-armed bandits." *GECCO 2008*, 913–920.
- Maturana, J. & Saubion, F. (2008). "A compass to guide genetic algorithms." *PPSN X*, LNCS 5199, 256–265.
- Li, K., Fialho, Á., Kwong, S. & Zhang, Q. (2014). "Adaptive operator selection with bandits for a multiobjective evolutionary algorithm based on decomposition." *IEEE Transactions on Evolutionary Computation*, 18(1), 114–130.
- Fialho, Á. (2010). *Adaptive Operator Selection for Optimization*. PhD thesis, Université Paris-Sud XI.

---

## Topic 5: Population diversity in constrained single-objective optimization

### Recommended approach

**Island model with phenotypic diversity tracking and MAP-Elites sidecar archive** — combining Whitley et al.'s island model framework (1999) with Mouret & Clune's (2015) quality-diversity concept, using Vidal et al.'s (2012) biased fitness incorporating diversity contribution.

### Mechanism

**Island model:** Partition the population of 1000 into 5 islands of 200 individuals each, connected via ring topology. Each island runs an independent BRKGA (or GA variant) with occasional migration of elite individuals.

**Per-generation island operation:**
1. Each island evolves independently for one generation.
2. Every 40 generations, migrate the **3 best individuals** from each island to its ring neighbor. Replace the **3 worst non-elite individuals** in the receiving island.
3. Islands can have different configurations to promote structural diversity:
   - Islands 1–2: Standard BRKGA (ρ_e = 0.70)
   - Island 3: Exploratory BRKGA (ρ_e = 0.60, mutants = 20%)
   - Island 4: Exploitative BRKGA (ρ_e = 0.80, mutants = 5%)
   - Island 5: GA with the existing custom crossover and 5 mutation operators

**Diversity tracking:** Use a combined metric:

- **Primary (phenotypic):** Piece-type histogram distance. Each solution's phenotype is a 10-element vector counting track pieces by type. Pairwise distance via L1 norm. Average pairwise distance across 500 random pairs per island serves as the diversity indicator.
- **Secondary (genotypic):** Average gene-wise standard deviation across the 218 genes. Initial value ≈ 0.289 (std dev of Uniform[0,1]). Collapse threshold: **0.05**.

**MAP-Elites sidecar archive** (zero additional evaluation cost):
- Define a behavioral descriptor grid: **(n_switches: 4 bins) × (n_straights: 5 bins) × (bbox_area: 4 bins)** = 80 cells.
- After each evaluation, attempt to archive the solution in its descriptor cell. A new solution replaces the current occupant only if it has better constrained fitness.
- During parent selection, 10% of parents are drawn from the MAP-Elites archive, injecting structural diversity into the evolutionary process.

### Why it fits this problem

The LEGO track problem has natural structural diversity axes: layouts with switches vs. without, compact vs. sprawling, simple loops vs. complex networks. The island model allows different solution topologies to evolve in partial isolation, preventing the switch-using solutions from being dominated by simpler (often fitter) non-switch solutions before they have time to mature. This addresses the existing structural niching concern (33% reserved for switch/crossing pieces) more naturally than quota-based reservation.

Phenotypic diversity is more meaningful than genotypic diversity for RK-encoded construction decoders because **many different RK vectors decode to identical or near-identical track layouts** (many-to-one mapping). Two genotypically distant RK vectors may produce the same piece sequence. Tracking piece-type histograms captures functional diversity directly.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Islands | **5 × 200 individuals** | Sufficient sub-population size per Cantú-Paz |
| Migration rate | **3 individuals per event** | 1.5% of island — small enough to preserve island identity |
| Migration interval | **Every 40 generations** | 25 migration events over 1000 generations |
| Topology | **Unidirectional ring** | Slowest information propagation; best diversity preservation |
| Diversity collapse threshold | **Avg gene σ < 0.05** | ~17% of initial diversity |
| MAP-Elites grid | **80 cells** (4 × 5 × 4) | Low overhead, meaningful descriptors |
| Archive selection fraction | **10% of parents** | Enough to inject diversity without disrupting convergence |

### Pitfalls

**Island overhead:** 5 islands × 200 individuals evaluates the same total 1000 individuals, but the reduced sub-population size may slow convergence within each island. Partially mitigated by migration and heterogeneous configurations. If total convergence speed drops significantly, reduce to 4 islands of 250.

**Migration disrupting constraint-adapted populations:** An infeasible-but-promising migrant from an exploratory island can disrupt a converging island's feasibility ratio, interacting badly with ε-tightening. Mitigate by migrating only **feasible individuals**, or only individuals whose CV is below the receiving island's current ε.

**MAP-Elites archive staleness:** Early archive entries may be low-quality but lock cells for many generations. Periodically (every 100 generations) clear the archive and rebuild it from the current population to prevent stale entries from biasing parent selection.

### Alternatives considered

**Fitness sharing (Goldberg & Richardson, 1987):** Requires computing pairwise distances for all N² pairs, expensive at pop=1000. The sharing radius σ in 218-dimensional space is poorly defined — the curse of dimensionality makes Euclidean distance uninformative. Rejected as too expensive and poorly suited to high-dimensional RK space.

**Deterministic crowding (Mahfoud, 1995):** Offspring compete against their most similar parent. Low overhead and parameter-free. However, it cannot *introduce* new diversity once lost — it only preserves existing diversity. Better as a within-island mechanism than a standalone diversity strategy.

**Clearing (Pétrowski, 1996):** More aggressive than sharing — within each niche, only the best individual retains fitness. Effective but requires a well-defined niche radius, which suffers the same high-dimensional issues as sharing.

### Key references

- Goldberg, D.E. & Richardson, J. (1987). "Genetic algorithms with sharing for multimodal function optimization." *ICGA*, 41–49.
- Sareni, B. & Krahenbuhl, L. (1998). "Fitness sharing and niching methods revisited." *IEEE Transactions on Evolutionary Computation*, 2(3), 97–106.
- Mouret, J.B. & Clune, J. (2015). "Illuminating search spaces by mapping elites." arXiv:1504.04909.
- Črepinšek, M., Liu, S.-H. & Mernik, M. (2013). "Exploration and exploitation in evolutionary algorithms: A survey." *ACM Computing Surveys*, 45(3), Article 35.
- Whitley, D., Rana, S. & Heckendorn, R.B. (1999). "The island model genetic algorithm: On separability, population size and convergence." *J. Computing and Information Technology*.
- Vidal, T., Crainic, T.G., Gendreau, M. & Prins, C. (2013). "A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows." *Computers & Operations Research*, 40(1), 475–489.

---

## Topic 6: Convergence and termination criteria

### Recommended approach

**Multi-criteria termination with a three-phase stagnation response** (intensify → shake → restart), informed by gene-wise variance tracking and feasibility-ratio stability. Budget allocation follows the **bet-and-run hybrid strategy** from Fischetti & Monaci (2014).

### Mechanism

**Convergence detection** uses three concurrent signals:

1. **Best-feasible fitness plateau:** Track the best-ever feasible solution's objective value. Stagnation = no improvement exceeding **0.1% of current best** for *W_f* = **100 consecutive generations**.
2. **Diversity collapse:** Compute average gene-wise standard deviation σ̄ across all 218 genes. Initial σ̄ ≈ 0.289. Collapse = σ̄ drops below **0.05** (17% of initial).
3. **Feasibility ratio stability:** After ε reaches 0, track FR = (feasible count) / 1000. Stability = FR has been above **0.30** and unchanged (±5%) for 20 generations.

**True convergence** is declared when: ε has reached its final target (0) AND best-feasible unchanged for 100 generations AND σ̄ < 0.05.

**Three-phase stagnation response:**

- **Phase 1 — Intensify** (triggered at 50 generations without improvement): Apply local search to the top 5% of solutions with increased budget (15 LS steps instead of 5). Reduce mutant fraction temporarily from 10% to 5%. Duration: 50 generations.
- **Phase 2 — Shake** (triggered at 100 generations without improvement, i.e., Phase 1 failed): Apply BRKGA shaking — perturb 20% of elite genes with Gaussian noise (σ = 0.1, clipped to [0,1]), completely re-randomize non-elite solutions. Keep the single best-ever solution untouched. Duration: 50 generations.
- **Phase 3 — Restart** (triggered at 150 generations without improvement): Discard all but the best-ever feasible solution and 10 diverse elites. Regenerate 989 individuals randomly. Reset ε to 50% of initial ε₀. Duration: until next stagnation or budget exhaustion.

**Budget allocation — hybrid strategy:**

- **Primary run:** 600 generations (600K evaluations).
- **Restart run 1:** 200 generations with fresh random population + 5% seeds from primary run's elites.
- **Restart run 2:** 200 generations with fresh random population + 5% different seeds.
- Return the best feasible solution across all runs.

### Why it fits this problem

The ε-tightening schedule creates artificial oscillations in population fitness — the best solution can appear to "worsen" when ε decreases because previously ε-feasible solutions become infeasible. Tracking best-ever-feasible separately from current population best prevents false stagnation signals during ε transitions. The three-phase response avoids both premature termination (by trying intensification first) and wasted computation (by escalating to restart only after gentler interventions fail).

The bet-and-run allocation ensures the primary run gets **60% of the budget** for sustained constraint-landscape navigation, which is essential for hard closure constraints that require many generations to satisfy. Restart runs with elite seeds provide fresh diversity while preserving learned building blocks.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Stagnation window | **100 generations** | Balances patience with efficiency for complex constraints |
| Diversity threshold (avg σ̄) | **0.05** | ~17% of initial; below this, population is effectively converged |
| Phase 1 trigger | **50 gens** without improvement | Early intervention |
| Phase 2 trigger | **100 gens** cumulative | Escalation after intensification fails |
| Phase 3 trigger | **150 gens** cumulative | Full restart as last resort |
| Max Phase 3 restarts | **3** | Diminishing returns beyond 3 |
| Primary run budget | **600 generations** | 60% of total |
| Restart runs | **2 × 200 generations** | 40% split equally |
| Per-gene entropy bins | **10** | For entropy-based convergence (alternative metric) |

### Pitfalls

**False convergence from homogeneous infeasible population:** If all individuals converge to a similar infeasible region (low gene variance but nonzero CV), the diversity metric triggers convergence while no feasible solution has been found. Always check that at least one feasible solution exists before declaring convergence. If no feasible solution exists after 500 generations, this signals a fundamental problem with the constraint handling or problem formulation rather than convergence.

**Restart overhead with expensive decoder:** Each restart reinitializes 989 random individuals that must be evaluated. At ~1ms per decoder call, this costs ~1 second — negligible. But if the decoder is slower (100ms), restarts cost ~100 seconds. Budget restarts based on actual decoder cost.

**ε reset after restart:** Resetting ε to 50% of initial allows the restarted population to explore infeasible space, but if the ε schedule expects to reach 0 by generation 600, the restart at generation 650 has only 350 generations with ε > 0 before the hard stop. Adjust Tc proportionally for restart runs.

### Alternatives considered

**Single long run with 1000 generations:** Simpler but vulnerable to premature convergence with no recovery mechanism. Friedrich et al. (2018) show restart strategies consistently outperform single runs on multimodal landscapes.

**Many short restarts (10 × 100 generations):** Too short for constrained problems where 100 generations barely reaches the ε-feasibility transition. Constrained combinatorial problems need sustained search to navigate the feasibility boundary.

**CMA-ES-style IPOP restarts (doubling population size):** Elegant for continuous optimization but incompatible with fixed-budget discrete construction decoder problems. Population size increases are undesirable when the decoder is the bottleneck.

### Key references

- Eiben, A.E. & Smith, J.E. (2015). *Introduction to Evolutionary Computing*, 2nd ed. Springer.
- Črepinšek, M., Liu, S.-H. & Mernik, M. (2013). "Exploration and exploitation in evolutionary algorithms: A survey." *ACM Computing Surveys*, 45(3), Article 35.
- Karafotias, G., Hoogendoorn, M. & Eiben, A.E. (2015). "Parameter control in evolutionary algorithms: Trends and challenges." *IEEE Transactions on Evolutionary Computation*, 19(2), 167–187.
- Fischetti, M. & Monaci, M. (2014). "Exploiting erraticism in search." *Operations Research*, 62(1), 114–122.
- Zielinski, K. & Laur, R. (2008). "Stopping criteria for differential evolution in constrained single-objective optimization." In *Advances in Differential Evolution*, Springer, 111–138.

---

## Topic 7: Warm-starting and solution injection

### Recommended approach

**Reduced-ratio diverse seeding (5–8%) with noise injection and stagnation-triggered BRKGA shaking** — synthesizing BRKGA literature best practices (Londe et al., 2025; Andrade et al., 2019) with Friedrich & Wagner's (2015) seeding analysis.

### Mechanism

**Initial seeding (generation 0):**

1. Generate **50–80 seed solutions** (5–8% of pop=1000) using the heuristic constructor with different random tie-breaking parameters. Ensure each seed uses a different random seed or heuristic parameter setting to maximize diversity among seeds.
2. **Add Gaussian noise** to each seed: for each gene position, perturb the value by N(0, 0.10²), clipping to [0,1]. This creates a "cloud" around each heuristic solution rather than exact copies.
3. Verify pairwise diversity among seeds: if any two seeds have Euclidean distance < **2.0** in the 218-dimensional RK space, replace one with a more heavily perturbed variant (σ = 0.20).
4. Fill the remaining **920–950 positions** with uniform random vectors from [0,1]^218.
5. For the first **10 generations**, increase the mutant fraction to **20%** (200 mutants instead of 100) to counteract seed dominance in the elite partition.

**Cross-instance transfer (when inventory changes):**

When track piece inventory changes (e.g., adding switch types), transform prior solutions:

1. If new genes are appended (encoding length increases from 218 to N'): **pad** old solutions with random values for new positions: x' = [x₁,...,x₂₁₈, U₁,...,U_{N'-218}].
2. If gene semantics change: establish a position mapping between old and new decoder semantics. Copy gene values for positions whose meaning is preserved; randomize positions with changed meanings.
3. **Gene-frequency analysis:** For each gene position, compute the mean μ_j and standard deviation σ_j across the top 10% of elite solutions from the old run. For the new run, initialize 20% of the random population from N(μ_j, max(σ_j, 0.10)) instead of Uniform(0,1). This biases initialization toward learned structure without copying specific solutions.

**Stagnation-triggered injection:**

When the best-feasible fitness has not improved for **100 generations**:

1. Replace **10% of non-elite individuals** (80 individuals) with a mix:
   - **20 freshly generated heuristic solutions** (2% of population) — expensive but high quality.
   - **30 perturbed versions of the best-ever solution** (3%) — add noise σ = 0.15 per gene.
   - **30 uniform random vectors** (3%) — maximum diversity.
2. Reset the AOS statistics window (cold restart for operator selection).
3. If using ε-tightening, temporarily increase ε by 10% for 20 generations to re-explore near the feasibility boundary.

### Why it fits this problem

The current 15% heuristic seeding (150 solutions) is likely **too aggressive** for pop=1000. With BRKGA's elite fraction at 20% (200 individuals), 150 high-quality seeds can fill 75% of the elite partition immediately, leaving only 50 elite slots for randomly generated individuals. This dramatically reduces genetic diversity in the mating pool from generation 1. Friedrich & Wagner (2015) confirm that 100% seeding fails, and the BRKGA literature warns that "excessive use of heuristic initialization functions could decrease the exploration capacity, trapping the population in local optimums quickly."

Reducing to 5–8% (50–80 seeds) with noise injection provides the benefit of heuristic knowledge (good construction priorities, feasible solutions) without overwhelming the population's genetic diversity. The noise clouds around seeds create a diverse neighborhood of near-heuristic solutions that BRKGA's biased crossover can recombine productively.

The stagnation-triggered injection mirrors BRKGA's native shaking mechanism from Andrade et al. (2019). Shaking perturbs elite genes with probability ρ_shake while completely re-randomizing non-elite solutions — a more effective restart than full population replacement because it preserves the structural knowledge encoded in elite genes.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Initial seed ratio | **5–8% (50–80 solutions)** | Down from 15%; prevents elite domination |
| Seed noise σ | **0.10 per gene** | Creates diverse cloud; maintains ~90% of original structure |
| Seed pairwise distance threshold | **2.0** (Euclidean in ℝ^218) | Ensures structural diversity among seeds |
| Boosted mutant rate (first 10 gens) | **20%** (200 mutants) | Counteracts seed dominance |
| Stagnation trigger | **100 generations** without improvement | Consistent with Topic 6 framework |
| Injection amount | **10% of non-elite** (80 individuals) | Balance between disruption and impact |
| Injection composition | 25% heuristic, 37.5% perturbed elite, 37.5% random | Balanced quality-diversity injection |
| Shaking perturbation on elites | **ρ_shake = 0.20** | Perturb 20% of elite genes during shake events |
| Gene-frequency transfer (cross-instance) | Top 10% elite statistics → N(μ_j, σ_j) initialization for 20% of new population | Transfers learned structure without overfitting |

### Pitfalls

**Seed quality variance:** If heuristic seeds vary widely in quality (e.g., 10 excellent feasible seeds and 40 mediocre infeasible ones), the excellent seeds dominate the elite immediately while mediocre seeds provide no benefit over random individuals. Pre-filter seeds: only inject solutions that are either **feasible** or have **CV below the median CV of a random sample**. Replace filtered-out seeds with additional random individuals.

**Cross-instance transfer overfitting:** Gene-frequency biasing from prior runs can lock in suboptimal construction strategies if the inventory change fundamentally alters the optimal layout structure (e.g., adding switches enables completely new topologies). Apply gene-frequency bias to only **20% of the new population** and keep 80% uniform random to preserve exploration capacity.

**Injection timing interaction with ε:** If stagnation-triggered injection occurs during the ε = 0 phase, injected random individuals are almost certainly infeasible and immediately discarded by feasibility-first selection. The temporary ε increase (10% for 20 generations) addresses this by creating space for infeasible injected individuals to survive and contribute genetic material.

### Alternatives considered

**Heavy seeding (20–30%):** Commonly recommended for warm-starting but excessive for BRKGA with aggressive elitism. Hernandez-Diaz et al. found 10–30% optimal, but BRKGA's elite copying mechanism amplifies seed dominance beyond what tournament-based GAs experience. The lower end (5–10%) is safer for BRKGA.

**Delayed/gradual injection:** Inject 20% of seeds at gen 0, 20% at gen 10, 20% at gen 20, etc. Prevents early elite domination but adds implementation complexity and delays the benefit of heuristic knowledge. The noise-injection approach achieves a similar diversity-preserving effect more simply.

**Full population restart on stagnation:** Discards all evolutionary progress. The shaking + partial injection approach preserves elite knowledge while restoring diversity, consistently outperforming full restarts in BRKGA applications (Andrade et al., 2019).

### Key references

- Friedrich, T. & Wagner, M. (2015). "Seeding the initial population of multi-objective evolutionary algorithms: A computational study." *Applied Soft Computing*, 33, 223–230.
- Gonçalves, J.F. & Resende, M.G.C. (2011). "Biased random-key genetic algorithms for combinatorial optimization." *Journal of Heuristics*, 17(5), 487–525.
- Londe, M.A. et al. (2025). "Biased random-key genetic algorithms: A review." *European Journal of Operational Research*, 321(1), 1–22.
- Andrade, C.E. et al. (2019/2021). "The Multi-Parent Biased Random-Key Genetic Algorithm with Implicit Path-Relinking." *European Journal of Operational Research*, 289(1), 17–30.
- Sörensen, K. & Glover, F. (2013). "Metaheuristics." In *Encyclopedia of Operations Research and Management Science*, Springer.
- Czarn, A. et al. (2004). "Statistical exploratory analysis of genetic algorithms." *IEEE Transactions on Evolutionary Computation*, 8(4), 405–421.

---

## Conclusion: an integrated architecture

These seven topics form a coherent system when combined. The recommended architecture is a **BRKGA-based island model with adaptive ε-constraint handling, sliding-window AOS for crossover/mutation selection, selective Lamarckian local search, and conservative seeding with stagnation-triggered shaking**.

Three integration points are critical. First, the ε-constraint handling (Topic 2) must feed into the AOS credit assignment (Topic 4) — operators that improve constraint satisfaction deserve reward proportional to the search phase. Second, the island model (Topic 5) interacts with the convergence framework (Topic 6): stagnation detection and restart decisions should operate per-island rather than globally, allowing healthy islands to continue while stagnant ones are shaken. Third, local search budget (Topic 3) must be balanced against population evaluation cost — the combined overhead of 1000 LS evaluations + 1000 population evaluations per generation doubles the computation, and the bet-and-run budget allocation (Topic 6) must account for this.

The single highest-impact change from the current implementation is likely **replacing the standard GA with BRKGA** (Topic 1). The RK encoding with greedy construction decoder is exactly the problem class BRKGA was designed for, and pymoo provides a ready-to-use implementation. The second highest-impact change is the **adaptive ε-constraint handling** (Topic 2), which directly addresses the challenge of navigating between easy constraints (inventory, boundary) and hard constraints (closure, loose ports). Together, these two changes establish a robust foundation on which local search, AOS, diversity management, and warm-starting can layer incrementally.