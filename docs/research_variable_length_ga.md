# Variable-Length GA Research: Chaotic Map-Coded Metaheuristics for LEGO Track Optimization

## Source Paper

**"Chaotic map-coded metaheuristics for metameric variable-length problems"**
Yang, Li, Yang, Chiba, Kagami, Hashimoto, Nagata (2025)
*Genetic Programming and Evolvable Machines*, 26:19
Published: June 4, 2025

---

## Paper Summary

### The Core Problem: Metameric Variable-Length Optimization

Many real-world problems require optimizing systems where **the number of components is itself unknown**. Traditional metaheuristics require fixed-dimension solution vectors, which forces the user to predefine the component count — leading to suboptimal solutions when that guess is wrong.

The paper defines **metameric variable-length problems** (borrowing from biology: organisms composed of structurally similar but not identical segments, like vertebrae or arthropod segments). In these problems:
- The solution is valid regardless of the number of components
- Each component is defined by the same parameter set (e.g., coordinates)
- The optimal number of components is unknown a priori
- As component count changes, the **dimensionality of the solution space changes**

### The Proposed Solution: Chaotic Transcription

The paper introduces a **dual-population architecture** inspired by the biological central dogma (DNA → RNA → Protein):

**DNA Layer** (fixed-length, used for evolutionary operations):
```
D_i = [m_i, z_{i,1}, y_i]    (3 dimensions, always fixed)
```
- `m_i` — integer: number of components in this solution
- `z_{i,1}` — float in (0,1): initial condition for chaotic map
- `y_i` — float in (0,1): control parameter for chaotic map

**PWLCM (Piecewise Linear Chaotic Map)** transcription:
```
z_{i,(j+1)} = z_{i,j} / y_i           if z_{i,j} ∈ (0, y_i)
z_{i,(j+1)} = (1 - z_{i,j}) / (1-y_i) if z_{i,j} ∈ [y_i, 1)
```
Starting from `z_{i,1}` with parameter `y_i`, iterate `m_i` times to generate `m_i` values.

**RNA Layer** (variable-length, used for fitness evaluation):
```
R_i = [z_{i,1}, z_{i,2}, ..., z_{i,m_i}]    (m_i dimensions, varies per individual)
```

**Key mechanism**: Search and selection happen on the fixed-length DNA. Evaluation happens on the variable-length RNA. The chaotic map provides a **deterministic, reversible compression** from a low-dimensional seed to an arbitrarily long sequence.

### Properties of Chaotic Transcription

1. **Sensitivity to initial conditions**: Tiny changes in `z_{i,1}` or `y_i` produce exponentially divergent RNA sequences — automatic diversity amplification
2. **Ergodicity**: Over sufficient iterations, PWLCM covers the entire (0,1) interval uniformly — ensures full exploration
3. **Determinism**: Given initial conditions, the output is fully reproducible — compatible with elite preservation and selection
4. **Dimensionality compression**: An entire variable-length solution is encoded by just 3 numbers — metaheuristic operators work in 3D regardless of actual solution complexity

---

## Real-World Examples From the Paper

### 1. Dendritic Neuron Model (DNM) Optimization — Continuous Problem

**Problem**: A neuron model has `m` dendritic branches, each with weights and thresholds. The optimal number of branches `m` is unknown. Each branch adds 2×I parameters (weights + thresholds for I inputs). With m=1 the search space is ~10D, with m=20 it's ~200D.

**How chaotic transcription works here**:
- DNA individual: `[m, z_initial, y_param]` — 3 genes
- PWLCM generates `m` weight values and `m` threshold values from these 3 genes
- Different `m` values create neurons of different complexity
- The algorithm simultaneously searches across all possible neuron sizes

**Results** (Section 5.2, Tables 2-4):
- Tested on 12 UCI classification datasets (Iris, Wine, Seeds, Breast Cancer, etc.)
- Chaotic-coded SHADE (CSHADE) achieves **mean accuracy 96.31%** vs traditional VGA's **95.32%** across all datasets
- CSHADE wins 26/36 comparisons (Wilcoxon test, α=0.05)
- On complex datasets (Heart, Parkinsons), chaotic methods significantly outperform VGA
- On simpler datasets (Iris, Seeds), performance is comparable
- **Key finding**: Chaotic transcription excels in continuous parameter spaces where the search landscape is complex and high-dimensional

### 2. Wind Farm Layout Optimization (WFLOP) — Discrete Problem

**Problem**: Place wind turbines on a grid to maximize total power output while minimizing wake effects. The optimal number of turbines is unknown. A farm with 30 turbines has >10^44 possible layouts.

**How chaotic transcription works here**:
- DNA individual: `[m, z_initial, y_param]` — 3 genes
- PWLCM generates `m` position values, rounded to grid coordinates
- Different `m` values = different numbers of turbines
- One optimization run explores all turbine-count possibilities simultaneously

**Results** (Section 5.3, Tables 5-9):

*Fixed turbine count (testing chaotic encoding alone)*:
- Tested across 13 wind farm layouts × 4 wind conditions × 12 turbine counts = 624 scenarios
- CSIS (chaotic SIS) wins **608/624** comparisons vs traditional SIS
- CGLPSO wins 367/624 vs GLPSO
- CSHADE wins 309/624 vs SHADE (weakest improvement)
- **Finding**: Chaotic encoding improves even fixed-length problems, with strongest gains for simpler algorithms

*Variable turbine count (full metameric problem)*:
- CSHADE finds the **highest total power output with the fewest turbines** in P1, P3, P4 wind conditions
- In complex wind scenario P4 (12 directions, variable speeds), CSHADE performs **comparably to dedicated VGA** (p=0.2969, not significant)
- As wind complexity increases (P1→P4), CSHADE's performance gap vs VGA narrows and eventually reaches parity
- **Finding**: Chaotic methods are competitive on discrete problems and improve with problem complexity

### Paper's Overall Assessment

- Chaotic transcription **outperforms VGA on continuous problems** (DNM optimization)
- On discrete problems (WFLOP), it **performs comparably to VGA** under complex conditions but is **inferior under simple conditions**
- The method **always outperforms fixed-length baselines**, meaning the variable-length capability comes at no cost when the dimension is fixed
- The core advantage is **exploration capability**: chaotic maps cover the search space more uniformly than VGA's insert/delete operators

---

## Application to LEGO Track Optimization

### Why This Problem Is Metameric Variable-Length

The LEGO track layout problem fits the metameric definition precisely:
- **Homogeneous components**: Each track piece is selected from the same catalog with the same parameter structure (piece type, orientation)
- **Unknown optimal count**: We don't know a priori how many pieces the best closed loop uses. A simple circle needs 16 R40 curves. A complex layout with switches might use 40+ pieces. The current system pre-allocates 100 main loop slots and uses `RK_INACTIVE_THRESHOLD` to disable unused ones
- **Variable solution dimensionality**: A 16-piece circle lives in a fundamentally different solution space than a 50-piece racetrack with passing sidings
- **Closure constraint couples piece count to geometry**: Adding or removing one piece changes whether the track closes. This tight coupling between component count and feasibility is exactly the challenge metameric methods address

### Current Fixed-Length Approach and Its Limitations

The current encoding (`encoding.py`) uses:
```
N_VAR = 218 genes (all in [0,1])
- Main loop:     100 piece selection keys (genes 0-99)
- Priority keys: 100 port priority keys (genes 100-199)
- Branch slots:  16 genes for 4 branch templates (genes 200-215)
- Start position: 2 genes (genes 216-217)
```

**Problems with the fixed 100-slot approach**:
1. **Wasted search space**: Most good solutions use 16-50 pieces, but the GA must search over all 100 slots. The `RK_INACTIVE_THRESHOLD` (e.g., values < 0.1 mean "skip this slot") introduces a discontinuity in the fitness landscape
2. **Inactive slot noise**: During crossover, inactive slots exchange genetic material that has no phenotypic effect, diluting the biased crossover's ability to preserve good building blocks
3. **No incentive to find the right length**: The objective (maximize utilization) implicitly prefers longer solutions, but the constraint system (closure) makes longer solutions exponentially harder to close. This creates a fitness landscape conflict that a fixed-length encoding handles poorly
4. **Curse of dimensionality**: 218 genes is a large search space. The paper argues that chaotic compression can reduce this dramatically

### How Chaotic Transcription Could Be Applied

#### Option A: Full Chaotic Encoding (Replace Current RK Encoding)

**DNA layer** (per individual, fixed 5 dimensions):
```
D_i = [n_pieces, z_piece, y_piece, z_priority, y_priority]
```
- `n_pieces`: integer in [8, 100] — how many main loop pieces to place
- `z_piece, y_piece`: PWLCM seed and parameter for generating piece selection keys
- `z_priority, y_priority`: PWLCM seed and parameter for generating port priority keys

**Chaotic transcription**:
```
PWLCM(z_piece, y_piece, n_pieces) → [piece_key_1, ..., piece_key_n]
PWLCM(z_priority, y_priority, n_pieces) → [priority_key_1, ..., priority_key_n]
```

**RNA layer** (variable-length):
```
R_i = [piece_key_1, ..., piece_key_n, priority_key_1, ..., priority_key_n, start_x, start_y]
```
Length = 2 × n_pieces + 2 (varies per individual)

**Advantages**:
- Search space collapses from 218D to 5D (or 7D with branch parameters)
- `n_pieces` is directly optimized as part of the chromosome
- PWLCM's ergodicity ensures piece selection keys are uniformly distributed in (0,1)
- Small changes in PWLCM seeds produce correlated but different layouts — smooth fitness landscape
- Compatible with any metaheuristic (BRKGA, DE, PSO, SHADE) since DNA is fixed-length

**Risks**:
- PWLCM produces **deterministically correlated** sequences — piece selections are not independent. This means you can't independently optimize piece 5 without affecting piece 12
- The paper's results show **weaker performance on discrete problems** (WFLOP) vs continuous ones (DNM). Track piece selection is inherently discrete
- Branch slot encoding (switches) may not map well to PWLCM sequences
- The existing decoder expects independent RK values; correlated inputs may reduce decoder effectiveness

#### Option B: Hybrid Approach (Chaotic Length + RK Content)

**DNA layer** (per individual, fixed 5 dimensions):
```
D_i = [n_pieces, z_piece, y_piece, start_x, start_y]
```

**Chaotic transcription generates ONLY the piece selection keys** using PWLCM. The priority keys are derived from the piece keys by a fixed transformation (e.g., priority = 1 - piece_key), eliminating half the search dimensions.

**RK decoding** proceeds as normal: the decoder receives `n_pieces` piece selection keys and `n_pieces` priority keys, and constructs the layout.

**Advantages**: Combines variable-length benefits with existing decoder infrastructure.

**Risks**: Same correlation issue — less flexible than independent RK values.

#### Option C: Variable-Length BRKGA (Direct Encoding, No Chaos)

Instead of chaotic transcription, implement a **direct variable-length encoding** within BRKGA:

**Chromosome**: Variable-length array of [0,1] values
```
[n_pieces, rk_1, rk_2, ..., rk_n, priority_1, ..., priority_n, branch_genes..., start_x, start_y]
```

**Modified BRKGA operators**:
- **Biased crossover**: For genes present in both parents, apply standard biased crossover. For genes present in one parent only (different lengths), include/exclude with probability based on elite bias
- **Mutants**: Generate with random length in [8, 100]
- **Elite selection**: Rank by fitness regardless of length

This is closer to the VGA (Variable-Length GA) that the paper compares against. The paper shows VGA outperforms chaotic methods on discrete problems (WFLOP), suggesting this may be the better choice for the LEGO problem.

**Advantages**:
- Each gene is independently optimizable (no chaotic correlation)
- Compatible with existing decoder (just truncate piece arrays to actual length)
- BRKGA's biased crossover preserves building blocks across lengths
- No PWLCM complexity

**Risks**:
- Length-changing crossover generally degrades fitness (paper Section 1, p.4: "altering the length of a solution... generally has a negative impact on fitness")
- The fitness landscape shifts when solution length changes — building blocks may not transfer across lengths
- pymoo's BRKGA implementation assumes fixed-length chromosomes

---

## Comparison of Approaches

| Criterion | Current Fixed-218 BRKGA | Chaotic (Option A) | Hybrid (Option B) | VGA (Option C) |
|-----------|------------------------|--------------------|--------------------|----------------|
| Search space | 218D | 5-7D | 5D + decoder | Variable (16-200D) |
| Length optimization | Implicit (via RK_INACTIVE_THRESHOLD) | Explicit (n_pieces gene) | Explicit | Explicit |
| Gene independence | Full | None (chaotic correlation) | Partial | Full |
| pymoo compatibility | Native BRKGA | Any metaheuristic (SHADE, PSO, etc.) | Custom | Custom (no pymoo support) |
| Discrete problem fit | Good (paper: BRKGA + RK is standard) | Weak (paper: chaotic methods weaker on discrete) | Medium | Good (paper: VGA wins on WFLOP) |
| Implementation effort | Already done | High (new framework) | Medium | High (custom GA framework) |
| Building block preservation | Strong (BRKGA biased crossover) | Weak (correlated sequences) | Medium | Medium (length-change disrupts) |
| Thesis novelty | Low (already implemented) | High (first application of chaotic transcription to layout problems) | Medium | Medium |

---

## Recommendations

### For Best Optimization Performance: Stay with BRKGA + Improve

The paper's own results show that on **discrete problems** (which track piece selection is), VGA outperforms chaotic methods in 3 of 4 wind conditions (Table 9). The current BRKGA+RK approach is the established gold standard for discrete combinatorial problems (per the BRKGA review paper). The highest-leverage improvements are those from the BRKGA research document (elite local search, shake operator, flat landscape fix).

### For Thesis Novelty: Implement Chaotic Transcription as a Comparison

The paper explicitly states (Section 7, p.48): "future research will explore various chaotic maps for improving performance across problem types." Applying chaotic transcription to LEGO track layout optimization would be a **novel application** — the paper only tests DNM (continuous) and WFLOP (discrete grid). The LEGO problem is a third category: **constrained discrete construction with geometric closure** — never tested with chaotic methods.

**Recommended experimental design**:
1. **Baseline**: Current BRKGA (218D fixed-length RK)
2. **Treatment 1**: Chaotic SHADE (Option A, 5D DNA)
3. **Treatment 2**: Variable-length BRKGA (Option C, direct encoding)
4. **Treatment 3**: Hybrid chaotic (Option B, 5D DNA + RK decoder)
5. **Metrics**: Best feasible piece count, generation to first feasible, convergence speed, solution diversity
6. **Statistical tests**: Wilcoxon rank-sum (30 runs per config), Friedman test for ranking
7. **Inventory variations**: Minimal (16 R40 curves), default, with_switches — tests scalability

This experimental design directly addresses the thesis requirement: *"investigate the influence of algorithm parameters on solution quality, convergence, and scalability depending on the number and variety of available elements."*

### Variable-Length Elements Worth Adopting Regardless of Framework

Even without switching to chaotic transcription or VGA, the paper's core insight — **let the algorithm discover the optimal number of pieces** — can improve the current BRKGA:

1. **Add an explicit "active length" gene**: Instead of RK_INACTIVE_THRESHOLD, add a single gene that encodes the number of active main loop positions. Values above this threshold are ignored by the decoder. This makes length a first-class optimizable parameter.

2. **Length-aware fitness**: Currently, a 16-piece circle and a 50-piece failed loop get very different fitness. The transition is cliff-like. A smoother fitness gradient that rewards longer partial loops (even infeasible) would help the optimizer find longer feasible solutions.

3. **Length mutation**: When injecting fresh individuals (stagnation), vary the implicit length (number of active pieces) deliberately. Current uniform random injection produces random length distributions.

---

## Key Takeaways for the LEGO Problem

1. **The LEGO track problem IS a metameric variable-length problem** — it fits the definition exactly: homogeneous modular components, unknown optimal count, closure constraint that couples count to feasibility.

2. **Chaotic transcription's main strength is exploration diversity**, not solution quality. It excels when the search space is large and continuous. The LEGO problem is discrete and constrained, where VGA and BRKGA have historically performed better.

3. **The paper's most actionable finding**: Fixed-length metaheuristics that pre-define component count require running N separate optimizations (one per possible piece count). Variable-length methods do it in one run. Currently, the LEGO optimizer handles this via the RK_INACTIVE_THRESHOLD, which is a weaker form of variable-length encoding — it works but wastes search capacity on inactive slots.

4. **The chaotic compression idea (3D → ND) is intriguing but risky** for this problem. The deterministic correlation between chaotic map outputs means you cannot independently optimize individual piece positions — a change to one piece's RK value cascades to all subsequent pieces. For construction-based decoders that place pieces sequentially, this could be catastrophic or beneficial depending on the problem structure.

5. **For the thesis, implementing chaotic transcription as a comparison algorithm** would provide high novelty while the BRKGA baseline provides reliability. The experimental comparison between fixed-length BRKGA, variable-length VGA, and chaotic metaheuristics on the LEGO problem would be a genuine research contribution.

---

## Paper References for Key Claims

| Claim | Section | Page |
|-------|---------|------|
| Metameric variable-length definition | 1, citing [16] Ryerkerk et al. 2017 | 3 |
| Direct vs indirect encoding tradeoffs | 1, citing [21] Ryerkerk et al. 2019 | 3-4 |
| Length change degrades fitness | 1, citing [21] | 4 |
| PWLCM ergodicity and uniformity | 4, citing [76] Baranovsky & Daems 1995 | 13 |
| CSHADE outperforms VGA on DNM (continuous) | 5.2, Tables 2-4 | 18-25 |
| VGA outperforms CSHADE on WFLOP P1-P3 (discrete) | 5.3.3, Table 9 | 44-47 |
| CSHADE matches VGA on WFLOP P4 (complex discrete) | 5.3.3, Table 9 | 44 |
| CSIS wins 608/624 vs SIS on fixed-length WFLOP | 5.3.1, Table 5 | 31 |
| Chaotic exploration vs exploitation tradeoff | 6 | 48-49 |
| Future work: test more chaotic maps (Logistic, Gaussian) | 6 | 49 |