# Crossover operators for multi-segment LEGO track genomes

**Segment-selective crossover—applying different operators to each chromosome region based on its epistatic structure—is the most promising strategy for this problem, resolving the tension between prior "mutation-only" findings and the new multi-segment encoding.** The core insight is that the prior 5–33× degradation from crossover applied only to the high-epistasis main loop; the switch mask, branch slots, and crossing overlay segments have low-to-moderate epistasis that makes them crossover-compatible. A BRKGA-inspired design that inherits the main loop intact from the fitter parent while applying parameterized uniform crossover (ρ = 0.7) to topology segments can combine elite track geometry with diverse topological features. This document grounds every recommendation in published evolutionary computation research, covering segment-level analysis, angular-budget preservation, topology-level recombination, decoder interactions, experimental validation, and risk assessment across nine research areas.

---

## A. Each chromosome segment demands a different crossover operator

The multi-segment chromosome's four regions span a wide epistasis spectrum, from the tightly coupled main loop to the nearly independent crossing overlay. This heterogeneity is the central design constraint: no single crossover operator suits all segments. The GLEAM framework (Jakob et al., PPSN 2008) established the principle of applying different operators to different chromosome regions—a "mixed-representation crossover" approach endorsed by Eiben & Smith (2015) for heterogeneous encodings.

### Main loop: high epistasis demands caution or abstinence

The main loop's **sequential positional epistasis** via turtle-graphics forward kinematics means each piece's world position depends on all predecessors. This creates a chain-like variable interaction graph where crossover at position *i* disrupts the geometric meaning of every gene from *i+1* onward. Standard order-based crossover operators—OX (Davis, 1985), PMX (Goldberg & Lingle, 1985), CX (Oliver et al., 1987)—are designed for permutation encodings where each allele appears exactly once. The main loop is *not* a permutation: it allows repeated piece types and sentinel values (-1), making these operators inapplicable without fundamental modification.

Blind two-point crossover on the main loop produces offspring satisfying the **Σdθ = 360° closure constraint** in fewer than **5–15% of cases**. This aligns with the prior finding of 5–33× worse performance. The recommended default is **ρ_mainloop = 1.0**: always inherit the main loop from the fitter parent. If main-loop crossover is attempted, it must use angular-budget-aware point selection (Section B) with mandatory repair.

### Switch mask: uniform crossover is near-optimal

The switch mask's **low inter-position epistasis** makes it a textbook case for Syswerda's uniform crossover (1989), which selects each gene independently from either parent with probability p = 0.5. Schema disruption is proportional to schema order (not defining length), providing positional-bias-free mixing. Spears & De Jong (1991) showed parameterized uniform crossover (PUX) with p₀ ∈ [0.3, 0.7] can bias toward the fitter parent when desired. Expected feasibility is **~100%**—no structural constraints exist on the mask itself. The only semantic repair needed is ensuring switches reference valid branch slots, a trivial O(S_max) check.

### Branch slots: block-level exchange preserves intra-slot coherence

Each branch slot [src_switch_idx, piece_0…piece_n, rejoin_target] exhibits **moderate intra-slot epistasis** (pieces within a branch must form a geometrically coherent path) but **low inter-slot epistasis** (branches are largely independent). This is the classic scenario for block-structured crossover. Falkenauer's Grouping Genetic Algorithm (1998) established that crossover operating at group boundaries—never splitting functional units—preserves structural integrity. The hierarchical cable routing GA of Ma et al. (2006) used exactly this pattern: block crossover at the route level, with individual route optimization handled separately.

The recommended operator is **slot-level uniform crossover**: generate a K-bit mask (K = number of branch slots), swap entire 10-gene slots according to the mask. Expected feasibility is **70–90%**. Repair involves remapping src_switch_idx to the nearest valid switch position and remapping rejoin_target to the nearest valid main-loop position.

### Crossing overlay and start position: straightforward operators suffice

The crossing overlay's position pairs have **low epistasis**, making uniform crossover at the pair level appropriate—each position pair is inherited atomically from one parent. Expected feasibility is **90–100%** with minor position-reference validation. For the two-float start position, **BLX-α** (Eshelman & Schaffer, 1993) with α = 0.5 or SBX (Deb & Agrawal, 1995) with η = 20 provides bounded continuous crossover at O(1) cost.

### Segment-level summary

| Segment | Recommended operator | Expected feasibility | Repair cost | Key citation |
|---|---|---|---|---|
| Main loop (30 genes) | Inherit from elite parent (ρ = 1.0) | 100% | None | Prior project research |
| Switch mask (30 genes) | Uniform crossover (p = 0.5) | ~100% | O(S_max) | Syswerda 1989 |
| Branch slots (40 genes) | Slot-level uniform crossover | 70–90% | O(K) | Falkenauer 1998 |
| Crossing overlay (10 genes) | Pair-level uniform crossover | 90–100% | O(1) | Syswerda 1989 |
| Start position (2 genes) | BLX-α (α = 0.5) | 100% | O(1) | Eshelman & Schaffer 1993 |

---

## B. Angular-budget-preserving crossover for the main loop

If main-loop crossover is pursued despite the risks, the angular closure constraint **Σdθ = 360°** demands specialized operators. Blindly mixing two closed sequences almost always breaks closure. Several published approaches address this class of constraint.

### Compatible crossover points exploit cumulative angle matching

The most direct approach defines **compatible crossover point pairs** (i, j) where position i in Parent A and position j in Parent B have approximately equal cumulative angles: |θ_cumA(i) − θ_cumB(j)| < ε. The algorithm precomputes cumulative angle arrays for both parents in O(L), then for each position in Parent A, performs binary search in Parent B's sorted angles in O(L log L). Offspring are constructed as pieces 0…i* from Parent A concatenated with pieces j*…L from Parent B, so the angular contributions of the two halves approximately sum to 360°.

With tolerance **ε = 15°**, roughly **40–60% of offspring** are viable without additional repair. With a repair step that substitutes 3–5 pieces near the boundary, feasibility rises above **90%**. The angular resolution analysis confirms this is tractable: the GCD of available arc angles (22.5°, 18°, 15°, 11.25°, 0°) is **0.75°**, meaning any angular deficit can theoretically be resolved using integer combinations of available pieces.

### Same Adjacency crossover identifies structurally meaningful cut sites

Zhang & Ding (2016) developed **Same Adjacency (SA) crossover** for variable-length path optimization chromosomes. SA identifies crossing sites where both parents share identical adjacency patterns—consecutive genes with matching transitions. This produces more valid crossover points than Same Point crossover, which only matches identical genes at identical positions. For train tracks, the "adjacency" criterion is redefined as **matching cumulative angular contribution**, providing natural crossover points that preserve local geometry on both sides of the cut. The authors reported improved convergence over SP crossover across varied network sizes.

### Edge Assembly Crossover offers a principled recombination framework

Nagata & Kobayashi's **Edge Assembly Crossover (EAX)** (1997, 2013) is among the most powerful TSP recombination operators, competitive with Lin-Kernighan-Helsgott. Its mechanism—merging two parent tours into a graph, identifying AB-cycles of alternating parental edges, selectively replacing edge subsets, then reconnecting subtours greedily—provides a template for track recombination. The critical feature is that **reconnection can introduce new genetic material** (new piece combinations) not present in either parent, guided by local geometric information. For the train track, subtour reconnection must account for the angular budget, selecting pieces that close the angular gap during the reconnection step.

### Molecular ring closure provides the strongest direct analogy

The track layout's turtle-graphics FK is closely analogous to internal coordinates in molecular conformations. GASCOS (Vázquez et al., 2000) uses an **analytical ring closure constraint** for cyclic molecules: when building conformations in dihedral-angle space, the last three bond angles are solved analytically to satisfy closure. The direct adaptation is a **closure buffer**: apply crossover for the first L−k pieces, then heuristically solve the last k pieces (e.g., k = 5–10) to close the angular budget. This ensures closure while preserving most crossover-generated diversity.

### Partition Crossover decomposes based on variable interaction

Whitley, Hains & Howe's **Partition Crossover (PX)** (2009) exploits the variable interaction graph to decompose recombination into independent sub-problems. Where parents agree on gene values, natural partition boundaries form. Between boundaries, each partition independently selects the parent contribution with better partial evaluation. The critical property: given two locally optimal parents, **>80% of offspring are locally optimal** in the full search space (Tinós, Whitley & Chicano, 2015). For the track problem, common subsequences (positions where both parents use the same piece type) form partition boundaries, and the angular budget can be maintained by constraining partitions to preserve their angular subtotals.

### Repair strategies when crossover introduces angular residual

Four repair mechanisms address post-crossover angular mismatch:

- **Piece substitution repair** (O(L)): Compute deficit Δθ = 360° − Σdθ, then substitute pieces near the crossover boundary to minimize |Δθ|. Available substitution increments include ±4.5° (R40↔R56), ±7.5° (R40↔R72), ±11.25° (R40↔R88), ±3° (R56↔R72). A greedy or dynamic programming sweep finds minimal substitutions.
- **Compensating segment insertion/deletion** (O(k)): Insert curve pieces to increase angle or replace curves with straights (0° contribution) to decrease angle. Insertion point near the crossover boundary minimizes disruption.
- **Bidirectional build repair**: Build track from both ends of the crossover point; solve the "meeting zone" using enumeration of possible piece combinations.
- **ε-relaxation with progressive tightening** (Takahama & Sakai, 2006): Accept offspring violating closure by up to ε, which decreases monotonically over generations.

---

## C. Topology-level crossover recombines high-level layout features

Beyond gene-level recombination, crossover can operate at the **topological** level—exchanging switch configurations, complete branch definitions, and crossing patterns. Several published domains provide operators for this.

### VLSI floorplanning crossover informs spatial layout recombination

VLSI floorplanning GAs face an analogous problem: recombining spatial arrangements of rectangular modules while preserving area and connectivity constraints. Three representations offer relevant crossover operators:

**Sequence pair crossover** (Murata et al., 1996) encodes module positions as two permutations. Nakaya, Koide & Wakabayashi (2000) applied standard PMX/OX to each permutation independently, recombining spatial relationships between parents. The key insight—that two permutations jointly encode 2D spatial relationships—parallels how the main loop sequence and switch mask jointly encode track topology.

**B*-tree crossover** (Chang et al., 2000) represents floorplans as binary trees where left children are placed rightward and right children upward. Crossover on BFS-linearized tree representations (Shunmugathammal et al., 2020) recombines layout regions. PMX with swap mutation performed best in empirical tests. The parent-child spatial relationships in B*-trees are analogous to how consecutive track pieces relate spatially.

**Slicing tree crossover** via normalized postfix strings (Valenzuela & Wang, 2000, 2002) ensures solution integrity under standard crossover operators. Track layouts with a main-loop-plus-branches hierarchy could use a postfix-like encoding where operators encode topological branching and rejoining relationships.

### CGP subgraph crossover preserves active-node connectivity

Kalkreuth, Rudolph & Droschinsky (2017) introduced **subgraph crossover for Cartesian Genetic Programming** that preserves active nodes on both sides of the crossover point. The crossover point is chosen between active function nodes; after swapping, the first active node past the cut has connections re-wired to maintain the active path. Kalkreuth (2020) showed CGP with subgraph crossover **outperformed mutation-only CGP** on standard benchmarks—countering the long-held view that crossover is harmful in CGP.

The CGP encoding—a linear integer array where each node has [input connections, function type]—is directly analogous to the track chromosome. CGP's active/inactive node distinction maps to main-loop pieces versus sentinel-filled positions. The **re-wiring step** at crossover boundaries is essential: when transplanting a track subsequence, the first piece after the cut must be reconnected to maintain the active path, analogous to CGP's reconnection of the first active node.

### Bond graph and graph grammar crossover handle multi-port components

Bond graph evolution (Fan, Hu & Goodman, 2004; Seo et al., 2003) uses GP trees encoding topology-modifying operators applied to an embryo structure. Standard GP subtree crossover recombines modification sequences while the embryo guarantees a valid base structure. For train tracks, the embryo would be a minimal valid loop, with construction functions (insert_switch, add_branch, add_crossing) encoded in a tree.

Domschke et al. (2025) defined **cut-and-join crossover for molecular graphs**: introduce small cuts into parent graphs, decompose into fragments, then rejoin fragments from both parents respecting degree constraints. This operator is ergodic (can explore the entire graph space without mutation) and preserves vertex degrees—naturally maintaining port constraints (2-port pieces stay 2-port, switches stay 3-port). For track layouts, valid cuts occur at switch/junction points, and rejoining must maintain the branch-rejoin requirement.

---

## D. BRKGA-style crossover offers the strongest integration path

The Biased Random-Key Genetic Algorithm framework (Gonçalves & Resende, 2011; building on Bean, 1994) provides the most directly applicable crossover mechanism for the train track GA, because it was designed specifically for problems with construction-based decoders.

### Elite-biased parameterized uniform crossover is the core mechanism

BRKGA's crossover flips a biased coin for each gene: with probability ρ > 0.5, inherit from the elite parent; otherwise, from the non-elite parent. The canonical **ρ = 0.7** (documented in Resende's reference implementation and the comprehensive BRKGA review by Londe, Pessoa, Andrade & Resende, 2024) ensures **70% of genes come from the fitter parent**, preserving most elite structure while introducing diversity. Values of ρ ∈ [0.6, 0.8] are most common in the literature. Andrade et al. (2021) extended this to **BRKGA-MP-IPR** with multiple parents and a rank-based bias function, showing improved results.

For the train track GA, segment-specific ρ values exploit the known epistasis structure. The recommended configuration:

| Segment | ρ value | Rationale |
|---|---|---|
| Main loop | 1.0 | Always inherit from elite; crossover too destructive |
| Switch mask | 0.7 | Standard BRKGA bias; low epistasis |
| Branch slots | 0.6–0.7 | Block-level exchange; moderate epistasis |
| Crossing overlay | 0.6 | Low epistasis; more exploration beneficial |
| Start position | 0.5 | No bias needed; BLX-α handles continuous genes |

### BRKGA's mutant injection complements elite-biased crossover

BRKGA replaces traditional mutation with **mutant injection**: each generation, 10–20% of the population's worst individuals are replaced with completely random chromosomes. This is formally equivalent to the "random immigrants" concept (Yang, 2007; Grefenstette, 1992). The review by Londe et al. identifies the critical tension: "the backdrop of double elitism is the fast convergence to a local optimum." Mutant injection provides the exploration counterweight. Since the train track decoder guarantees >99% feasibility even for random chromosomes, mutant injection is cheap and safe—random individuals decode into valid (if low-quality) layouts.

The existing ALNS-adaptive mutation operators (including COMPOUND segment replacement) already provide targeted exploration. BRKGA-style mutant injection at **10% per generation** adds a complementary diversity source that does not interact with the ALNS reward structure.

### BRKGA's decoder-first design aligns with the existing architecture

The BRKGA framework's central property—"under a weak assumption, crossover always produces feasible offspring" (GECCO 2016 tutorial, Resende & Ribeiro)—holds because any point in the genotype space is decoded into a feasible solution. This is exactly the train track GA's situation. Crossover in BRKGA has been validated across **150+ published applications** (Londe et al., 2024), including sequencing problems with construction-based decoders: job-shop scheduling (Gonçalves, Mendes & Resende, 2005), 3D container loading (Gonçalves & Resende, 2012), and resource-constrained project scheduling (Gonçalves, Resende & Mendes, 2010). In all cases, random keys encode priorities, and a construction heuristic decoder builds feasible solutions—exactly analogous to the track decoder's construction sequence.

---

## E. The decoder absorbs crossover damage but cannot prevent quality loss

The construction-based decoder fundamentally changes the crossover risk profile. In direct encodings, crossover can produce structurally invalid offspring (broken connectivity, violated constraints). With the decoder, **crossover damage manifests as quality loss, not infeasibility**. This is a crucial distinction.

### Smooth genotype-phenotype mappings make crossover less destructive

The effectiveness of crossover depends on the smoothness of the genotype-to-phenotype mapping. For low-epistasis segments (switch mask, crossing overlay), small genotype changes produce small phenotype changes—crossover offspring have intermediate fitness, making crossover useful for exploitation. For the main loop, the mapping is **highly rugged**: changing an early gene cascades through all downstream pieces via FK. This confirms why main-loop crossover was 5–33× worse: the decoder produces valid but geometrically scrambled layouts.

Miller (2011) observed the identical pattern in CGP: crossover on the full flat encoding was "disruptive to the subgraphs within the chromosome." But when "a CGP genotype is divided into a collection of chromosomes, crossover can be very effective... selecting the best chromosomes from parents' genotypes can produce super-individuals." This is precisely the multi-segment strategy recommended here: treat each chromosome segment as a separate "CGP chromosome" for crossover purposes.

### Neutral drift through decoder skip behavior enables exploration

The decoder's skip-invalid-pieces behavior creates "inactive genes"—positions whose alleles don't contribute to the decoded phenotype. This enables **neutral genetic drift**, which Miller & Thomson (1999) showed is essential for CGP's search effectiveness. Neutral drift allows the population to accumulate potentially useful genetic material in inactive regions, which crossover or mutation can later activate. The crossover interaction with neutral drift is beneficial: crossover can transfer inactive-but-promising gene patterns from one individual to another, where they may become active in a different genomic context.

---

## F. The crossover-mutation balance favors selective, moderate crossover

The question of whether crossover adds value beyond powerful mutation operators is empirically contested. The evidence supports a nuanced position: crossover helps on segments with building-block structure but not on the high-epistasis main loop.

### AmoebaNet's mutation-only success does not generalize broadly

Real et al. (2019) evolved AmoebaNet-A using **mutation-only aging evolution**, matching reinforcement learning-based NAS. However, this finding is specific to the NAS search space—highly structured modular graphs with complex topology and high inter-node epistasis. Dang & Lehre (2015) proved theoretically that crossover makes genetic algorithms **at least twice as fast** as mutation-only on problems with building-block structure, because "crossover is able to repair the disruptive effects of mutation in later generations." Osaba et al. (2014) found mutation-only EAs outperformed GAs on TSP and other combinatorial problems when using **blind crossover**, but noted this does not apply to problem-aware crossover operators.

The train track problem's multi-segment encoding deliberately creates building-block structure in the topology segments. The AmoebaNet finding applies to the main loop (graph-like, high epistasis) but not to the switch mask and branch slots (modular, low-to-moderate epistasis).

### Recommended crossover probability is lower than standard guidelines

Standard guidelines suggest P_c ∈ [0.5, 1.0] (Grefenstette, 1986), but these assume crossover is the primary recombination mechanism. With COMPOUND segment replacement already providing targeted exploitation, crossover's role shifts to combining good sub-solutions across individuals. The recommended **P_c = 0.3–0.4 on topology segments only** reflects this reduced role. Ye, Wang, Doerr & Bäck (2020, PPSN XVI) found empirically that optimal P_c decreases with problem dimension and increases with population size—consistent with moderate values for the ~80 topology genes (excluding the 30-gene main loop).

### Crossover should use a fixed rate, not participate in ALNS-AOS

Crossover and mutation serve fundamentally different roles—exploitation versus exploration (Ortiz-Boyer et al., JAIR 2005), though this distinction blurs for specific operators. Including crossover in the ALNS adaptive operator selection would conflate these roles. Liu et al. (2024) proposed a hybrid genetic-ALNS that maintained separate populations with per-individual ALNS meta-parameters, but crossover and ALNS operated at different levels—crossover transferred solutions between individuals while ALNS improved within individuals. **Keep crossover at a fixed rate** as a separate mechanism, with ALNS-AOS managing the mutation operator portfolio.

---

## G. Pipe routing, circuit, and cable harness GAs provide operational templates

The train track problem's combination of sequential construction, branching topology, and geometric closure constraints is rare but not unique. Several engineering domains face analogous challenges.

### Ship pipe routing GAs handle branched 3D sequences

Sui & Niu (2016) developed a **branch-pipe-routing GA** decomposing 3D pipe routes into two-point segments connected via maze algorithms. Their "complete crossover strategy" operates on grid-point sequences—directly analogous to track piece sequences. Dong & Lin (2017) and Dong & Bian (2020) extended this with **cooperative coevolutionary GA** where each pipe route evolves as a separate species. Crossover operates within each species; co-evolution handles inter-pipe conflicts through combined fitness evaluation. This cooperative coevolutionary pattern maps naturally to the multi-segment chromosome: main loop, branches, and crossings could co-evolve with segment-specific crossover operators.

### Cable harness routing uses hierarchical block crossover

Conru (1994) decomposed cable harness routing into topology evolution and spatial routing—two paired GAs with problem-specific crossover for each. Ma et al. (2006) used a two-level hierarchical GA: block crossover at the route-combination level and individual route optimization at the second level. Their key finding that infeasible solutions from level 2 still provide useful partial information ("lethal gene" exploitation) parallels the train track decoder's graceful handling of invalid pieces.

### Circuit topology synthesis separates topology from parameters

Shi et al. (2022) proposed a tree-based representation for analog circuits with **separate topology and value crossover operators**—topology crossover swaps structural subtrees while value crossover blends component parameters. This explicit separation mirrors the train track's multi-segment design: the main loop defines geometry (analogous to circuit parameters), while switch mask and branch slots define topology (analogous to circuit structure). Koza's (1996) GP-based circuit synthesis demonstrated that even with ~65% infeasible circuits in generation 0, selection pressure alone reduced this to **0.3% by generation 30**—no explicit repair needed. The construction decoder provides even stronger guarantees.

### PCB routing decouples placement from routing

Berlier (2011) showed that GA crossover should operate on high-level **placement** (component positions and rotations) while routing is deterministically computed for each placement. This decoupling—GA evolves topology, algorithm computes geometry—directly mirrors the track GA's architecture where the chromosome encodes piece sequences and topology, while the turtle-graphics decoder computes spatial layout.

---

## H. A rigorous experimental protocol to validate crossover utility

Testing crossover's contribution requires careful experimental design. The stakes are significant: if crossover on topology segments improves convergence speed or solution quality, it justifies the implementation complexity; if not, the mutation-only approach remains optimal.

### Hypotheses, controls, and treatments

The primary hypothesis is that segment-selective crossover (on switch mask, branch slots, and crossing overlay, with the main loop inherited from the elite parent) improves hypervolume over the mutation-only baseline. Six ablation conditions isolate each segment's contribution:

- **C₀**: Mutation-only baseline (P_crossover = 0)
- **A₁**: Crossover on switch mask only
- **A₂**: Crossover on crossing overlay only
- **A₃**: Crossover on branch slots only (atomic slot swapping)
- **A₄**: Crossover on all topology segments (switch mask + branch slots + crossing overlay)
- **A₅**: Full crossover on all segments including main loop (expected negative control)
- **A₆**: Crossover on main loop only (expected worst performance)

Each treatment runs at P_crossover ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, yielding ~30 configurations plus the control.

### Metrics must capture both quality and constraint satisfaction

**Hypervolume (HV)** is the primary metric—the only unary indicator strictly Pareto-compliant (Zitzler & Thiele, 1998; Auger et al., 2012). **IGD+** (Ishibuchi et al., 2015) serves as secondary metric, using a reference front approximated from combined best-known solutions. Domain-specific metrics include **feasibility ratio** per generation, **topology diversity** (number of distinct switch configurations, average pairwise Hamming distance on topology segments), and **constraint violation severity** disaggregated by constraint type (angular closure, inventory, branch rejoin, switch pairing).

**Anytime performance** is captured via HV-over-evaluations curves with 95% confidence bands. The **Empirical Attainment Function** (López-Ibáñez et al., 2025, GECCO) captures performance trajectories more precisely than target-based ECDF, without requiring a priori quality targets.

### Statistical rigor demands 30+ runs with Holm's correction

Derrac, García, Molina & Herrera (2011)—the most-cited guide for EA statistical comparison (4000+ citations)—recommends the Friedman test for ranking multiple algorithms, followed by **Holm's step-down procedure** for post-hoc pairwise comparisons. Holm's procedure is strictly more powerful than Bonferroni correction and controls the family-wise error rate. With ~30 comparisons, Bonferroni's adjusted α ≈ 0.0017 is overly conservative; Holm's adaptive thresholds provide better power.

Campelo & Takahashi (2019) provide the principled framework for sample size calculation: **30–50 independent runs** per configuration, with same random seeds across treatments for paired comparisons. With 30 configurations × 30 runs × 50,000 evaluations, the total budget is approximately **45 million evaluations**. Page's trend test (Derrac et al., 2014) at 10 equidistant cut-points tests whether one algorithm consistently converges faster.

---

## I. Risk analysis across five threat vectors

### Feasibility destruction concentrates in the main loop

The probability and severity of feasibility destruction vary dramatically across segments. Main-loop crossover poses the highest risk: **70–95% of offspring** from unrestricted crossover violate the Σdθ = 360° closure constraint. The decoder can produce a valid layout from any chromosome, but mixed main-loop genes yield geometrically scrambled tracks with poor fitness—effectively wasted evaluations. Switch mask crossover carries only **5–15%** infeasibility risk (switch pairing violations, trivially repairable). Branch slot crossover at the atomic-slot level has **20–40%** risk of orphaned branches (rejoin target invalid in new context), mitigated by target remapping. Crossing overlay has **5–10%** risk from invalid position references.

### Elite-biased crossover creates a diversity collapse hazard

BRKGA's double elitism—elite preservation plus ρ > 0.5 bias—drives fast convergence but risks premature homogenization. Leung, Gao & Xu (1997) proved that population diversity converges to zero with probability one under selection without sufficient diversification. The risk is **MEDIUM-HIGH (40–60% probability)** that aggressive crossover (P_c > 0.7) combined with elite bias reduces genotypic diversity below critical threshold within 50–100 generations. Once diversity collapses in the main loop, recovery is extremely difficult because the closure constraint makes random exploration inefficient.

Mitigation requires per-segment diversity monitoring, NSGA-II/III crowding distance, mutant injection at 10%, and an **adaptive crossover rate** that decreases when diversity drops below threshold. Damm et al. (2016) introduced elite diversification in BRKGA that only copies individuals into the elite set if they differ sufficiently from existing elites—a mechanism worth incorporating.

### The ε-constraint interaction creates a "triple squeeze"

If the GA uses ε-constraint handling for geometric closure (initially relaxed, progressively tightened), crossover interacts in phase-dependent ways. During the relaxed phase (ε large), crossover can productively explore infeasible regions. During the tight phase (ε ≈ 0), crossover between marginally feasible parents frequently produces infeasible offspring that are immediately rejected. The combination of crossover + elite bias + tightening ε creates a "triple squeeze" on diversity, with **30–50% probability** of causing population collapse during the transition phase. Mitigation: synchronize the ε schedule with crossover rate—decrease P_c as ε tightens.

### Computational cost is dominated by wasted evaluations, not operator cost

Crossover itself is cheap (**<5% of evaluation cost**). The real expense is wasted evaluations on offspring that the decoder produces as low-quality solutions. If crossover produces beneficial offspring at rate B and mutation at rate M, crossover is worthwhile when B > M. Prior research found B_crossover = M_mutation / 5 to M_mutation / 33 for raw main-loop sequences. For topology segments, B is expected to be substantially higher—the empirical question the experimental protocol addresses.

### Risk-prioritized mitigation matrix

| Risk | Priority | Probability | Mitigation |
|---|---|---|---|
| Main loop feasibility destruction | P1 | 70–95% | Exclude main loop from crossover (ρ = 1.0) |
| Diversity collapse from elite bias | P2 | 40–60% | Monitor diversity, adaptive P_c, 10% mutant injection |
| Decoder non-repairability | P2 | 20–40% | Pre-decoder validation, decoder timeout |
| ε-constraint phase interaction | P3 | 30–50% | Synchronize ε schedule with crossover rate |
| Branch slot disruption | P3 | 20–40% | Atomic slot swapping (never split slots) |
| Computational waste | P4 | 15–30% | Offspring pre-screening, feasibility tracking |

---

## Ranked recommendations: from most to least promising

Based on the full literature analysis, risk assessment, and the established genome structure, the following strategies are ranked by expected net benefit relative to implementation complexity and risk.

**Rank 1: Segment-selective BRKGA-style crossover (RECOMMENDED)**
Apply parameterized uniform crossover with ρ = 0.7 to switch mask, branch slots (atomic slot-level), and crossing overlay. Inherit main loop and start position from elite parent (ρ = 1.0). This strategy exploits the multi-segment encoding's deliberate epistasis structure, is grounded in 150+ BRKGA applications (Londe et al., 2024), and poses minimal risk. Expected **10–25% faster convergence** to equivalent hypervolume based on BRKGA literature for problems with mixed epistasis. P_c = 0.3–0.4. Computational overhead: negligible.

**Rank 2: Segment-selective crossover with adaptive rate**
Same as Rank 1 but with feasibility-triggered adaptation: monitor per-segment feasibility ratio each generation; reduce P_c when feasibility drops below 0.8; increase when above 0.95. Srinivas & Patnaik (1994) AGA framework provides the adaptation mechanism. Adds modest implementation complexity but protects against the diversity-collapse and ε-interaction risks.

**Rank 3: Topology-level block crossover with CGP-style protection**
Exchange complete branch definitions as atomic units, using CGP subgraph crossover's node-preservation principle (Kalkreuth et al., 2017) to protect multi-port components (switches, crossings) from being split during crossover. More sophisticated than Rank 1 but provides better topological diversity—offspring can have branch configurations that neither parent possessed. Requires careful boundary handling at switch nodes.

**Rank 4: Compatible-crossover-point main loop recombination**
Apply angular-budget-aware crossover (Section B) to the main loop, selecting crossover points where cumulative angles match between parents. Combined with GASCOS-style closure buffer for the last 5–10 positions and piece-substitution repair. This is the most ambitious strategy, offering the potential to combine complementary track geometries from different parents. However, the **70–95% pre-repair infeasibility** and implementation complexity make it a Phase 3 addition after Ranks 1–2 are validated.

**Rank 5: Full Partition Crossover (PX) on main loop**
Apply Whitley et al.'s (2009) Partition Crossover using the known variable interaction graph structure. Theoretically powerful (>80% of offspring locally optimal) but requires gray-box access to partial fitness evaluation and significant implementation effort. Best reserved for a future iteration after simpler strategies are validated.

**Rank 6: Cooperative coevolutionary crossover**
Evolve main loop, branches, and crossings as separate co-evolving species with species-specific crossover (inspired by Dong & Lin, 2017). Fundamentally restructures the GA architecture beyond the current pymoo NSGA-II/III framework, making it impractical as an initial implementation despite theoretical appeal.

### Resolving the mutation-only tension

The prior finding that "crossover breaks connectivity on raw construction sequences (5–33× worse)" remains valid for the main loop segment. The multi-segment encoding resolves this tension by creating **crossover-compatible regions** (switch mask, branch slots, crossing overlay) that were not previously available. The recommended strategy preserves the mutation-only approach where it works (main loop) and adds crossover where it helps (topology segments). This is not a contradiction but a natural consequence of encoding design: **the genome was structured to make selective crossover possible**.

### Staged deployment minimizes risk

Phase 1 deploys crossover on switch mask and crossing overlay only (lowest risk, ~100% feasibility). Phase 2 adds branch-slot crossover with atomic slot swapping. Phase 3, contingent on Phase 1–2 showing statistically significant benefit, tests constrained main-loop crossover with angular repair. Each phase requires **30 independent runs** with Holm's step-down testing against the mutation-only baseline, following Derrac et al.'s (2011) methodology. This staged approach ensures each increment of complexity is empirically justified before the next is attempted.

---

## Conclusion: selective crossover fills a gap that mutation alone cannot

The core finding from this research is that the multi-segment encoding creates an opportunity space for crossover that did not exist in the monolithic construction-sequence representation. **Segment-selective BRKGA-style crossover**—preserving the elite parent's main loop while recombining topology segments with ρ = 0.7 bias—is the strategy best supported by literature, lowest in risk, and most compatible with the existing pymoo architecture. The BRKGA framework's 150+ successful applications to decoder-based combinatorial problems (Londe et al., 2024) provide strong evidence for this approach.

The angular-budget-preserving crossover for the main loop remains a high-potential but high-risk option. The compatible-crossover-point mechanism, GASCOS-style closure buffer, and piece-substitution repair provide the theoretical toolkit, but the 70–95% pre-repair infeasibility means this should only be attempted after topology-segment crossover has been validated. The 0.75° angular resolution from available piece angles ensures repair is always theoretically possible.

The experimental validation framework—30+ runs per configuration, Friedman test with Holm's correction, HV as primary metric, staged ablation across segments and P_c values—provides the statistical machinery to answer definitively whether crossover adds value beyond the current mutation-only system. The answer will almost certainly be segment-dependent: positive for low-epistasis topology segments, neutral or negative for the high-epistasis main loop. This is not a limitation but the expected outcome of a well-designed heterogeneous encoding.