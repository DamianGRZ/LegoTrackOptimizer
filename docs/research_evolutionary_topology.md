# Evolving graph topologies: why your LEGO track optimizer needs a paradigm shift

**The hybrid encoding (Option I) decisively outperforms all other approaches for this problem, and pymoo is structurally mismatched to variable-topology graph construction—but workable with significant custom engineering.** This matters because the LEGO train track layout problem sits at the intersection of three hard subproblems—variable graph topology, multi-port component assembly, and discrete inventory constraints—that collectively break the assumptions underlying standard fixed-length evolutionary optimization. No published work directly addresses evolutionary optimization of toy train track layouts, but mature techniques from circuit topology synthesis, bond graph evolution, and modular robot design transfer directly. The most important insight from the literature is that **construction-based encodings that grow valid structures from seeds achieve >99% feasibility**, while random fixed-length encodings produce <1% valid individuals—a difference that dominates all other design choices.

## Five paradigms for evolving variable-topology graphs

### NEAT: powerful principles, no track precedent

NEAT (Stanley & Miikkulainen, 2002) solves the variable-topology evolution problem through three innovations: **innovation numbers** that assign globally unique IDs to each new structural gene, enabling alignment-based crossover between genomes of different lengths; **speciation** via compatibility distance that protects novel topological innovations from premature elimination; and **complexification from minimal structure**, starting with the simplest viable topology and growing complexity only when beneficial. These mechanisms are domain-independent in principle—Hu et al. (GECCO 2002) explicitly recognized the parallel between NEAT's speciation and their Structure Fitness Sharing for bond graph evolution.

However, a systematic review by Papavasileiou et al. (2021, *Evolutionary Computation*) catalogued 61 NEAT successors across 232 papers. **Every single one targeted neural network evolution.** No published work applies NEAT-style innovation numbers to physical layout, circuit design, or assembly graph problems. This represents a genuine research gap. Adapting NEAT for track layouts would require encoding pieces as node genes with port metadata and connections as port-to-port edge genes—feasible but entirely unexplored.

The Analog Genetic Encoding (Mattiussi & Floreano, 2007, *IEEE Trans. Evol. Comp.*) addresses the same alignment problem differently, using string similarity to determine connectivity in general analog networks. AGE has been demonstrated on circuits and neural networks, making it more domain-portable than NEAT itself.

### Grammatical Evolution: the strongest paradigm match

Grammatical Evolution maps integer codons to BNF grammar production rules, guaranteeing that **all decoded phenotypes are syntactically valid by construction**. This is not merely a theoretical advantage—Castejón & Carmona (2018, *Applied Soft Computing*) demonstrated GE generating circuit netlists where each component type (2-terminal resistors, 3-terminal BJTs, 4-terminal MOSFETs) maps to a grammar rule specifying exactly the correct number of node connections. This is structurally identical to the LEGO track problem: 2-port straights, 3-port switches, 4-port crossings.

The grammar can enforce inventory constraints by restricting terminal alternatives to available piece types, enforce port-count correctness through production rule structure, and handle variable topology through recursive/optional productions. Fenton et al. (2014, *IEEE Trans. Evol. Comp.*) used GE for truss optimization with **real-world standard construction elements from a discrete catalog**, directly addressing inventory constraints. The main limitation is that BNF grammars are context-free and cannot directly enforce context-dependent geometric constraints like "cumulative angle must return to zero." This requires either attribute grammar extensions or a post-processing geometric validation step.

Graph grammars extend BNF to operate directly on graph structures. NetGAP (2023, arXiv:2306.07778) used graph grammars with Monte Carlo Tree Search to synthesize avionics network topologies with multiple hardware module types—routers, switches, end systems—a strong structural parallel to multi-port track pieces.

### Graph-based GAs: powerful but crossover is the Achilles' heel

Direct graph encodings (adjacency matrices, edge lists, node-depth encodings) represent track layouts naturally but face a fundamental challenge: **graph crossover with cycles is qualitatively harder than string or tree crossover**. Globus et al. (NASA JavaGenes) identified five specific difficulties—cycle structure prevents clean splitting, fragments have multiple attachment points, fragment sizes are asymmetric, connectivity preservation is not guaranteed, and domain validity constraints must be maintained simultaneously.

Despite these challenges, practical solutions exist. Zeng et al. (2023, *Applied Energy*) developed GBMOGA for pipe network layout optimization with custom crossover that identifies common substructures between parents and recombines them while preserving connectivity—achieving **37% pressure drop reduction and 10% cost savings** versus expert designs. The bond graph + GP framework (Seo, Fan, Hu, Goodman, Rosenberg, 2001–2004, GECCO/Engineering Optimization) is the closest analogue to the track problem: bond graph components have explicit ports connected by power bonds, GP trees encode construction sequences that grow topologies from seeds, and dense modular primitives pre-assemble common functional blocks.

### Indirect encodings: elegant theory, poor problem fit

L-systems with standard turtle interpretation produce **tree structures only—no loops or cycles**—because the push/pop branching mechanism always returns to the branch point. Bielefeldt et al. (2019, AIAA SciTech) developed SPIDRS, a graph-based L-system interpretation that can create loops, but this is non-standard and adds significant complexity.

CPPNs and HyperNEAT (Stanley, 2007; Stanley et al., 2009) generate continuous spatial patterns and excel at capturing symmetry, repetition, and regularity. However, they are **fundamentally mismatched to discrete combinatorial assembly with strict geometric constraints**. Discretizing CPPN outputs to a fixed inventory of piece types destroys the smooth gradients that make these encodings effective. Cheney et al. (2014, GECCO) used CPPN-NEAT for multi-material voxel robots with 4+ material types, which superficially resembles piece-type assignment, but the continuous spatial substrate bears no resemblance to the port-connectivity graph of a track layout.

### Variable-length GAs: native in DEAP, workaround in pymoo

DEAP (Fortin et al., 2012, *JMLR*) natively supports variable-length list chromosomes through its creator/toolbox architecture—individuals are simply Python lists of any length. DEAP includes NSGA-II selection (`tools.selNSGA2`) but its multi-objective infrastructure is less polished than pymoo's. Specialized crossover operators for variable-length chromosomes include Same Adjacency crossover (2016, *Expert Systems with Applications*), Synapsing Variable Length Crossover (Hutt & Warwick, 2007, *IEEE Trans. Evol. Comp.*), and messy GA cut-and-splice (Goldberg et al., 1989). Yu (2022, MIT Thesis) explicitly combined variable-length chromosomes with NSGA-II for vehicle maneuver planning, demonstrating that the combination works.

pymoo's official workaround for non-standard representations is the **custom variable type pattern with `n_var=1`**, where the single "variable" is a complex object (graph, tree, variable-length list). The documentation explicitly states this supports "other data structures such as trees or lists with variable lengths." This effectively reduces pymoo to an evolutionary loop skeleton—users must implement all sampling, crossover, mutation, and duplicate elimination operators. pymoo still provides NSGA-II's non-dominated sorting and crowding distance, its visualization suite, and its constraint-handling infrastructure.

## Encoding strategy evaluation reveals a clear winner

The five encoding strategies span a spectrum from trivially pymoo-compatible but expressively limited (Option E) to maximally expressive but evolutionarily fragile (Option H). The critical discriminator across all encodings is **feasibility ratio**—what fraction of randomly generated or crossover-produced individuals represent valid, connected track layouts.

### Options E through H share a common failure mode

**Option E (fixed segments)** divides the chromosome into 8 branches × 14 pieces = 112 slots, each holding a piece type or NO_OP sentinel. This maps trivially to pymoo's integer variables, and the NO_OP padding finds theoretical support in Cartesian Genetic Programming literature—Miller & Smith (2006, *IEEE Trans. Evol. Comp.*) showed that ~95% inactive nodes enable beneficial neutral genetic drift. However, the encoding imposes hard topology ceilings, cannot represent cross-track inter-branch dependencies, and produces a feasibility ratio near zero for random chromosomes.

**Option F (switch-first + path-fill)** decomposes the problem hierarchically but assumes a single main loop exists—fatal for topologies where two independent loops connect via cross-tracks. Crossover on Phase 1 (switch positions) invalidates all Phase 2 (path fill) genes through epistatic coupling.

**Option G (path-tagged inventory)** assigns each piece a (type, branch_tag) tuple. This suffers from the **competing conventions problem**—identical phenotypes have exponentially many genotypic representations (any permutation of tag labels). NEAT's literature extensively documents how competing conventions destroy crossover effectiveness. Additionally, 4-port cross-track pieces inherently belong to two paths simultaneously, which a single tag cannot represent.

**Option H (construction-sequence actions)** achieves maximum expressiveness—any topology is representable—but exhibits **catastrophic crossover fragility**. Each gene's meaning depends on all preceding genes (positional epistasis). Cutting a construction sequence mid-execution and appending another parent's tail produces nonsensical builder states. This violates Holland's Building Block Hypothesis and is well-documented as the central weakness of linear genetic programming (Brameier & Banzhaf, 2007). The encoding essentially requires abandoning crossover entirely, relying on mutation-only search as CGP does with its (1+λ) strategy.

### Option I dominates through principled decomposition

**The hybrid encoding (Option I) scores 3.9/5 overall versus 2.5–2.9 for alternatives** across eight evaluation criteria. Its key advantages are:

The GA evolves only topology-level decisions—switch count, switch positions, cross-track placements, main loop shape—comprising roughly **20–30 variables** instead of 112+. Deterministic algorithms (A*/constraint propagation) then compute optimal branch fills between specified endpoints. This reduces the search space from approximately 29^112 ≈ 10^163 to roughly 29^30 ≈ 10^44—a reduction of **~120 orders of magnitude**.

The deterministic solver guarantees valid branch fills by construction, achieving a **near-100% feasibility ratio** versus <1% for random chromosomes in Options E, G, and H. Coello Coello's comprehensive survey (2002, *Comp. Methods Appl. Mech. Eng.*) documents that construction/repair approaches dramatically outperform penalty-based constraint handling for highly constrained problems.

At the topology level, crossover combines meaningful building blocks—switch configurations, loop shapes—satisfying Holland's Building Block Hypothesis. This is impossible in Options G and H where crossover destroys structure. The approach is formally a **memetic algorithm** (Moscato, 1989), and Sudholt & Zarges (2019, *Artificial Intelligence*) proved that memetic algorithms outperform pure evolutionary approaches on problems with "big valley" structure.

The main risk is **per-evaluation cost**: the deterministic solver must execute for every fitness evaluation, potentially thousands of times per generation. Caching, incremental solving, and early termination of clearly infeasible topologies can mitigate this. If branch-fill solving takes >1 second, this becomes the computational bottleneck.

## NSGA-II is correct, but pymoo fights the problem structure

**NSGA-II remains the right algorithm.** Its non-dominated sorting and crowding distance operate exclusively on objective values, making it entirely representation-agnostic. The algorithm does not need to "know" whether the decision vector is a fixed-length array or a variable-topology graph. CMA-ES is definitively unsuitable—it estimates covariance matrices in continuous ℝⁿ and requires fixed dimensionality, with known stagnation on integer variables even with recent margin adaptations (Hamano et al., GECCO 2022). Differential evolution shares CMA-ES's continuous-space assumptions. Genetic programming (tree-based) is a strong alternative for construction-sequence encodings, as demonstrated by Koza's circuit synthesis work, but sacrifices NSGA-II's mature multi-objective machinery.

**MAP-Elites deserves serious consideration** as a complementary approach. For track layout design, diversity of solutions is inherently valuable—users want to see varied layouts (ovals, figure-eights, spirals, branching networks). MAP-Elites with feature descriptors (footprint area, switch count, loop count, symmetry score) could illuminate the design space before NSGA-II refines promising topologies. Alvarez et al. (IEEE CoG 2019) used Interactive Constrained MAP-Elites for dungeon/level design with playability constraints—a structural parallel to track layout with connectivity constraints.

**pymoo is architecturally mismatched but functionally adequate.** The framework's requirement for fixed `n_var` is deeply embedded—populations are stored as NumPy arrays of shape `(pop_size, n_var)` for vectorized evaluation. The `n_var=1` object-encoding workaround is officially documented and functional, but it transforms pymoo into a thin wrapper around NSGA-II's selection mechanism while the user reimplements everything else: sampling, crossover, mutation, duplicate elimination, and repair. At this point, pymoo provides non-dominated sorting, crowding distance, constraint handling, visualization, and performance indicators—valuable but not irreplaceable.

DEAP offers native variable-length chromosome support with less multi-objective polish. A pragmatic architecture would use **pymoo for NSGA-II's selection and analysis infrastructure with the `n_var=1` object pattern**, accepting the engineering cost of custom operators. Alternatively, a custom framework built around pymoo's `NonDominatedSorting` utility class—which can be used standalone—could provide both representation flexibility and state-of-the-art Pareto front approximation without pymoo's population-matrix constraints.

## The architecture that the literature actually supports

Synthesizing across circuit evolution, bond graph design, LEGO assembly, modular robotics, and network optimization, the strongest architecture for this specific problem combines elements from multiple paradigms:

- **Encoding I (hybrid)** as the core strategy, with the GA evolving a compact topology descriptor (switch positions, cross-track placements, loop shape parameters) and a deterministic solver computing branch fills
- **NEAT-style speciation** to protect novel topological innovations (e.g., first cross-track usage, first nested switch) from premature elimination during early generations
- **Constructive initialization** seeding the population with known-good basic layouts (simple ovals, figure-eights) following Koza's embryonic circuit principle—start simple, complexify incrementally
- **Bond-graph-inspired port semantics** where each piece type declares its port count, positions, and orientations, and the decoder enforces port compatibility at connection time
- **Grammar-constrained topology mutations** drawn from GE principles, where topology modifications (add switch, remove switch, swap switch type, add cross-track) are guaranteed to produce syntactically valid topologies by grammar rule design
- **Repair operators** for crossover outputs, following Coello Coello's recommendation and Moon & Yoon's (2025) demonstration that repair preserves genetic characteristics while restoring feasibility

The closest published precedents are Lohn & Colombano's linear circuit primitives (>99% validity), the bond graph + GP framework (multi-port components with port-aware operators), and Peysakhov & Regli's messy GA for LEGO assembly graphs. None of these solve the exact LEGO track problem, but collectively they provide a validated toolkit for every component of the architecture.

## Conclusion

Three findings stand out from this analysis. First, **feasibility ratio is the dominant design factor**—the difference between <1% and ~100% valid individuals overwhelms all other encoding considerations, making the hybrid approach (Option I) the clear theoretical winner. Second, **pymoo's fixed-length constraint is a friction point, not a wall**—the `n_var=1` object pattern preserves access to NSGA-II's selection machinery while delegating representation entirely to the user, which is acceptable given that all competitive encodings require custom operators anyway. Third, **this problem is genuinely novel in the literature**: no prior work combines multi-port component assembly, variable graph topology evolution, discrete inventory constraints, and closed-loop geometric validity. The gap between circuit topology synthesis (which lacks geometric embedding) and LEGO brick assembly (which lacks multi-port graph topology) is precisely where the train track problem sits—and where the most impactful methodological contribution could be made.