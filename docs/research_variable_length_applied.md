# Variable-Length GA Applied to LEGO Track Optimization

## Source

Based on "Chaotic map-coded metaheuristics for metameric variable-length problems" (Yang et al., 2025, Genetic Programming and Evolvable Machines 26:19).

---

## Problem Statement

**Goal**: Maximize the number of LEGO track pieces used in a closed layout within a bounded area, subject to:

1. **Closure**: The track must form a closed loop — the train returns to its starting position and heading. No loose ends.
2. **Physical feasibility**: The LEGO City Trains locomotive must be able to traverse the track without derailing. This means:
   - No excessively tight turns (minimum radius constraint from R40 curves)
   - No "snaky" sequences — rapid left-right-left-right alternation can cause derailment on real track
   - Speed limits vary by piece type (curves are slower than straights)
   - The physics model computes speed profiles: `v = SF * sqrt(mu * g * R)` where R is curve radius
3. **Boundary**: The entire layout must fit within a defined rectangular area (e.g., a table)
4. **Inventory**: Each piece type has a limited count (e.g., 40 R40_LEFT curves, 80 straights, 2 switches)

**The fundamental tension**: Using MORE pieces scores higher (maximizes utilization), but longer tracks are exponentially harder to close geometrically AND more likely to create physically infeasible sequences for the locomotive.

---

## Why This Problem Is Metameric Variable-Length

The paper defines **metameric variable-length problems** as those where:
- The solution consists of **homogeneous components** (same parameter structure, different values)
- The **optimal number of components is unknown** a priori
- The solution is valid at any component count, but quality varies

The LEGO track problem fits this definition precisely:

| Metameric Property | LEGO Track Instance |
|---|---|
| Homogeneous components | Track pieces — each is a piece type selected from the same catalog |
| Unknown optimal count | Could be 16 (simple circle) or 200+ (complex multi-branch layout) |
| Valid at any count | 16 curves form a valid circle; 80 pieces can form a valid racetrack |
| Quality varies with count | More pieces = higher utilization score, IF the track still closes |

**Nested metameric structure**: The track has variable-length at THREE levels:
1. **Main loop**: variable number of pieces (16 to hundreds)
2. **Branches**: variable number of passing sidings (0 to many), each with variable piece count
3. **Multiple loops**: crossings can create additional independent loops (future)

---

## The Physical Feasibility Problem and Variable Length

### Why Fixed-Length Encoding Makes Physics Harder

With the current 100-slot fixed encoding, the optimizer has no direct control over track length. It activates/deactivates individual gene slots via a threshold. This creates two physics problems:

**Problem 1: No length-aware smoothness control**

When the optimizer activates genes scattered through the 100 slots, the resulting piece sequence can have arbitrary ordering. The decoder places pieces left-to-right, and the physical smoothness of the track depends on which piece types end up adjacent. With 100 slots and ~30 active, the "active" pieces are determined by which RK values happen to be above threshold — there's no mechanism to ensure smooth transitions between curves and straights.

A snaky sequence like `[R40_LEFT, R40_RIGHT, R40_LEFT, R40_RIGHT, ...]` is physically dangerous for the locomotive:
- Each direction change causes lateral force reversal
- The locomotive's coupler slack amplifies oscillation
- At R40 radius (40 stud ≈ 320mm), speed limit is 0.97 m/s
- Rapid alternation at this speed causes derailment on real track

**Problem 2: Closure becomes harder with arbitrary length**

With 100 fixed slots, the optimizer must simultaneously:
1. Choose which ~30 genes to activate (push above threshold)
2. Choose piece types for those ~30 positions (RK value selection)
3. Ensure the resulting sequence closes geometrically
4. Ensure the sequence is physically traversable

The threshold mechanism means adding one piece (activating one gene) changes the track length by 1 piece at an arbitrary position — disrupting both closure and smoothness.

### Why Variable-Length Encoding Makes Physics Easier

With true variable-length, the chromosome LENGTH is the track length. Adding a piece means the chromosome grows by one gene AT THE END (or at a specific position). This gives the optimizer:

**Direct control over length**: The crossover and mutation operators can explicitly grow or shrink the track. A 30-piece closed loop can grow to 31 pieces by inserting a straight into a straight section — maintaining both closure (a straight in a straight section doesn't change the angle) and smoothness (no direction change added).

**Sequential semantics**: Piece at position i is always followed by piece at position i+1 in the physical track. Crossover between parents preserves contiguous subsequences that are already physically smooth. In the fixed encoding, pieces at positions 5 and 47 might be adjacent in the decoded track — crossover has no way to preserve this adjacency.

**Smoothness as a building block**: In variable-length GA, a subsequence like `[R40_LEFT, R40_LEFT, R40_LEFT, R40_LEFT]` (a 90° curve arc) is a building block that crossover can preserve. It's a contiguous gene segment. In fixed encoding, these four curves might be at slots 12, 34, 67, 89 — crossover cannot preserve them as a unit.

---

## Real-World Examples from the Paper

### 1. Wind Farm Layout (WFLOP) — Closest Analogy

**Problem**: Place N wind turbines on a grid to maximize total power output. Wake effects mean downstream turbines produce less power.

| Property | WFLOP | LEGO Tracks |
|---|---|---|
| Components | Turbines (position on grid) | Track pieces (type + sequential order) |
| Unknown count | Optimal N unknown | Optimal piece count unknown |
| Physical constraint | Wake effect (proximity penalty) | Derailment risk (curvature, smoothness) |
| Spatial constraint | Farm boundary | Table boundary |
| Objective | Max total power | Max piece utilization |
| Feasibility | Always feasible | Must close loop (hard constraint) |

**Results** (Tables 5-9):
- CSIS (chaotic map) wins **608/624** comparisons vs fixed-length SIS on WFLOP
- Chaotic SHADE finds **maximum total power with fewest turbines** (Table 8) — exactly our goal of maximizing utilization
- On variable turbine count: CSHADE matches VGA under complex wind conditions (P4, p=0.30)

**Key analogy**: WFLOP traditionally runs 100 separate optimizations (one per possible turbine count). Variable-length does it in ONE run. Similarly, our fixed-100-slot encoding implicitly searches all lengths 0-100 simultaneously but inefficiently. True variable-length would search length and content jointly.

### 2. Dendritic Neuron Model (DNM) — Structural Analogy

**Problem**: A neuron has M dendritic branches, each with parameters. Optimal M is unknown.

This directly mirrors our branch structure:
```
DNM neuron:                          LEGO track layout:
├── Branch 0 (weights, threshold)    ├── Passing siding 0 (position, length)
├── Branch 1 (weights, threshold)    ├── Passing siding 1 (position, length)
└── Branch 2 (weights, threshold)    └── Passing siding 2 (position, length)
Number of branches: UNKNOWN          Number of sidings: UNKNOWN
```

**Results** (Tables 2-4):
- CSHADE achieves **96.31% accuracy** vs VGA's **95.32%** across 12 datasets
- Chaotic transcription outperforms VGA on 26/36 comparisons (continuous problem)
- Key finding: chaotic map's **extreme sensitivity to initial conditions** generates diverse neuron architectures, preventing premature convergence to simple structures

**Analogy to our problem**: Just as DNM optimization must find the right number of branches without settling for too-simple neurons, our optimizer must find layouts with the right number of passing sidings without settling for branchless circles.

### 3. References Supporting Variable-Length for Layout Problems

From the paper's citations:
- **Wireless sensor networks** [17]: Variable number of transmitters with position optimization. Multi-objective variable-length NSGA-II used successfully.
- **Structural optimization** [23]: Variable-length genome for transmission tower topology. Gene segments represent physical structural members — direct analogy to track pieces.
- **Cloud service composition** [26]: Variable-length chain of services. Each service is a homogeneous component — chain length optimized alongside service selection.

---

## The Variable-Length Chromosome for LEGO Tracks

### Architecture

The chromosome IS the track. Its length equals the number of pieces plus structural metadata:

```
Chromosome = [
    piece_key_1, piece_key_2, ..., piece_key_N,    # N main loop pieces
    branch_0_pos, branch_0_hand, branch_0_str,      # Branch 0 params
    branch_1_pos, branch_1_hand, branch_1_str,      # Branch 1 params
    ...                                              # M branches total
    start_x, start_y                                 # Position
]

Total length = N + 3M + 2    (varies per individual)
```

**Every gene is active. Every gene affects fitness. No wasted capacity.**

A 16-piece circle: 20 genes. A 200-piece mega-layout with 8 branches: 226 genes.

### How Length Relates to Physical Feasibility

The locomotive physics model constrains which tracks are buildable:

**Speed profile**: The evaluation computes speed at each piece: `v = 0.8 * sqrt(0.30 * 9.81 * R)`. For R40 curves (R ≈ 320mm), v_max ≈ 0.97 m/s. For straights, v_max ≈ 1.57 m/s. The forward-backward pass ensures the locomotive can accelerate and decelerate safely.

**Smoothness**: Rapid curvature changes (left-right-left-right) cause derailment. In a variable-length encoding, the optimizer can:
- Insert straights between opposing curves (grow chromosome by 1-2 genes) to add transition zones
- Group curves of the same direction into arcs (contiguous subsequences preserved by crossover)
- The chromosome's sequential order IS the physical track order — smoothness is directly encoded in gene adjacency

**Closure at different lengths**:
- 16 R40_LEFT curves → 16 × 22.5° = 360° → perfect circle closure
- 8 R40_LEFT + 2 STRAIGHT + 8 R40_LEFT + 2 STRAIGHT → oval closure
- The angle budget (360° for a closed loop) constrains which lengths are feasible
- Variable-length lets the optimizer discover that, e.g., 24 pieces close better than 23 or 25 for a given piece mix

### Variable-Length Operators That Respect Physics

**Insert mutation** (chromosome grows by 1):
```
Before: [..., R40_LEFT, R40_LEFT, STRAIGHT, ...]
Insert STRAIGHT between two curves:
After:  [..., R40_LEFT, STRAIGHT, R40_LEFT, STRAIGHT, ...]

This IMPROVES smoothness (adds transition between curves)
```

**Remove mutation** (chromosome shrinks by 1):
```
Before: [..., STRAIGHT, STRAIGHT, STRAIGHT, STRAIGHT, ...]
Remove one straight from a long straight section:
After:  [..., STRAIGHT, STRAIGHT, STRAIGHT, ...]

This maintains smoothness (straight section is still smooth)
```

**Segment-preserving crossover**:
```
Parent A: [CURVE_ARC(4), STRAIGHT_RUN(3), CURVE_ARC(4), STRAIGHT_RUN(2)]  (13 pieces)
Parent B: [CURVE_ARC(8), STRAIGHT_RUN(7)]                                   (15 pieces)

Identify physically-smooth segments, exchange complete segments:
Child: [CURVE_ARC(4) from A, STRAIGHT_RUN(7) from B, CURVE_ARC(4) from A]   (15 pieces)

The child inherits COMPLETE smooth building blocks, not random gene fragments.
```

**Branch insert mutation** (3 genes appended):
```
Before: [pieces... | start_x, start_y]                                      (N+2 genes)
After:  [pieces... | branch_pos, branch_hand, branch_str | start_x, start_y] (N+5 genes)

The main loop is unchanged. A new passing siding is added.
The locomotive can traverse either the main route or the branch.
```

---

## Chaotic Transcription Applied to Track Layout

### The DNA/RNA Architecture for Tracks

**DNA** (fixed 7 genes, used for evolution):
```
D = [n_pieces, z_piece, y_piece, n_branches, z_branch, y_branch, z_pos]
```

**PWLCM chaotic map** generates variable-length RNA:
```
PWLCM(z_piece, y_piece, n_pieces) → [rk_1, rk_2, ..., rk_n]
PWLCM(z_branch, y_branch, 3 × n_branches) → [branch_params...]
```

**RNA** (variable-length, used for evaluation):
```
R = [rk_1, ..., rk_n, branch_params..., start_x, start_y]
```

### How Chaotic Properties Help Track Physics

**Temporal autocorrelation**: PWLCM generates sequences where adjacent values are correlated (determined by the map's trajectory). For track layouts, this means adjacent pieces tend to be similar types — creating natural "runs" of curves or straights. This inherently produces smoother tracks than independent random values.

```
PWLCM trajectory: 0.23, 0.31, 0.42, 0.38, 0.29, 0.15, 0.72, 0.85, 0.91, 0.88, 0.76
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   Low values → select curves              High values → select straights

This naturally creates: [curve, curve, curve, curve, curve, straight, straight, straight, straight]
                        = a physically smooth 90° arc followed by a straight run
```

**Ergodicity**: Over sufficient length, PWLCM covers [0,1] uniformly. This means all piece types get represented in long tracks — maximizing utilization while maintaining the temporal correlation that ensures smoothness.

**Sensitivity to initial conditions**: Small changes in z_piece produce very different track layouts. This provides natural diversity in the population without explicit diversity operators.

### The 7-Gene Compression

The entire variable-length track layout — main loop of any length, any number of branches, each of any length — is encoded in just 7 numbers. The metaheuristic operates in 7D space regardless of track complexity.

**For a 200-piece layout with 8 branches**: 7 genes (DNA) vs 226 genes (VGA) vs 218 genes (current fixed).

The tradeoff: changing z_piece changes ALL piece selections simultaneously. You cannot independently optimize piece 5 without affecting piece 50. This is the fundamental limitation of chaotic transcription for discrete problems.

---

## Experimental Design

### Comparison Framework

| Algorithm | Encoding | Length Control | Physics Handling |
|-----------|----------|---------------|-----------------|
| Current BRKGA | Fixed 218 genes | Implicit (threshold) | Post-hoc speed profile |
| Variable-Length GA | N+3M+2 genes | Explicit (chromosome length) | Segment-preserving crossover |
| Chaotic BRKGA | Fixed 7 genes (DNA) | n_pieces gene | Chaotic autocorrelation |

### Metrics

1. **Piece utilization**: n_pieces_used / total_inventory (primary objective)
2. **Closure error**: Euclidean distance from end to start (must be < tolerance)
3. **Physical feasibility**: Can a real LEGO locomotive traverse without derailing?
   - Minimum curve radius satisfied?
   - No snaky sequences (rapid direction alternation)?
   - Speed profile within limits at every piece?
4. **Convergence speed**: Generations to first feasible solution
5. **Scalability**: Performance vs inventory size (16 pieces → 200+ pieces)

### Required Experiments

**Experiment 1**: Fixed-length BRKGA (baseline) vs Variable-Length GA on default config
- 30 runs, Wilcoxon rank-sum test
- Hypothesis: VGA finds higher utilization because it searches length explicitly

**Experiment 2**: Variable-Length GA vs Chaotic Transcription on default config
- 30 runs, Wilcoxon rank-sum test
- Hypothesis: VGA wins on piece selection quality; Chaotic wins on convergence speed
- Paper reference: VGA wins on WFLOP P1-P3 (simple discrete), ties on P4 (complex)

**Experiment 3**: All three algorithms on with_switches config (branches required)
- Tests variable-length branch handling
- Hypothesis: Variable-length handles branch count optimization better than fixed 4-slot encoding

**Experiment 4**: Scalability test — increase inventory from 40 to 200 total pieces
- The fixed-100-slot encoding cannot handle >100 pieces
- VGA chromosome grows naturally to 200+
- Chaotic transcription handles any length via n_pieces gene
- Tests the paper's core claim: variable-length scales where fixed-length cannot

**Experiment 5**: Physical verification
- Build top-3 solutions from each algorithm on real LEGO track
- Test with LEGO City Trains locomotive at various speeds
- Record: derailments, smooth operation, aesthetic quality
- This directly addresses the thesis requirement for physical verification

---

## Key Takeaways

1. **The LEGO track problem IS a metameric variable-length problem** — homogeneous components (track pieces) with unknown optimal count, exactly as defined in the paper.

2. **Variable-length encoding eliminates the 100-piece cap and the 4-branch cap** that currently limit solution diversity. The chromosome grows and shrinks to match the solution.

3. **Physical feasibility (no derailment) benefits from variable-length** because the sequential gene order matches physical track order, enabling segment-preserving crossover that maintains smooth building blocks.

4. **The paper's chaotic transcription provides an alternative** that compresses the entire layout to 7 genes via PWLCM. Its temporal autocorrelation naturally produces smoother tracks. But its coupling (changing one gene affects all pieces) is problematic for the discrete piece selection sub-problem.

5. **The paper's experimental results suggest VGA is better for discrete problems** (WFLOP: VGA wins P1-P3) while chaotic methods are better for continuous problems (DNM: chaotic wins). Our problem is primarily discrete (piece selection from catalog) but has continuous aspects (branch positioning, start position), suggesting a **hybrid approach** may be optimal.

6. **The nested metameric structure (pieces within branches within loops) extends beyond the paper's scope** — the paper only tests flat metameric problems. Our multi-level variable-length problem would be a novel research contribution.
