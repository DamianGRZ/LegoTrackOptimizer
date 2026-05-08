# Chromosome encodings for multi-loop branching track layout optimization

**A BRKGA-style random-key chromosome with hierarchical constructive decoder is the strongest architecture for evolving branching LEGO train layouts under NSGA-II.** The literature on hub-location routing, multi-cycle vehicle routing, and SONET ring design provides three proven encoder–decoder patterns that map almost perfectly onto the multi-loop track problem: the Andrade et al. (2013) k-interconnected multi-depot multi-TSP, the Prins (2004) giant-tour-plus-Split paradigm, and the Chagas et al. (2021) NDS-BRKGA hybrid that fuses BRKGA crossover with NSGA-II survival selection. No prior work directly addresses automated layout optimization for discrete-piece branching track from a fixed inventory, confirming this as a genuine research gap at the intersection of combinatorial optimization, computational geometry, and evolutionary algorithms. The Kia et al. (2014) multi-floor manufacturing analogy holds structurally but diverges at the geometric-closure constraint — the hardest aspect of the LEGO problem — for which the Prins Split decoder and turtle-graphics forward-kinematics provide the necessary machinery.

---

## The three closest structural analogues in the literature

The most structurally similar solved problem is the **k-interconnected multi-depot multi-traveling-salesman problem** (k-IMDMTSP). Andrade, Miyazawa & Resende (2013) evolved solutions consisting of an *inner cycle* connecting k terminal vertices plus k *outer cycles* — one per terminal — using a single BRKGA chromosome. The decoder first selects which vertices become terminals (using indicator keys), then assigns non-terminals to terminals, then determines tour ordering within each cycle. The mapping to LEGO track is direct: the inner cycle is the backbone loop connected by switch pieces; the outer cycles are branch loops extending from those switches. The chromosome encodes both topology decisions (which pieces act as junctions) and routing decisions (piece ordering within each loop).

The **two-level hub location routing problem** with directed tours (Freitas, Aloise, Fontes, Santos & Menezes, 2023; *OR Spectrum* 45:903–924, DOI: 10.1007/s00291-023-00718-y) adds geometric realism. Hubs are connected by an oriented backbone ring; each hub anchors a cluster of spokes forming its own oriented cycle. The BRKGA decoder maps random keys to hub selection, spoke-to-hub assignments, and cycle orderings at both levels — almost identical to a two-level LEGO track with a main loop and branch loops. A third analogue is the **SONET Ring Assignment Problem** solved by Duhamel, Aloise, Santos & Oliveira (2016) via BRKGA with a *Split procedure*: a permutation of vertices is optimally partitioned into capacity-constrained rings using dynamic programming, with each partition forming a cycle in the optical network.

Beyond hub-and-ring problems, two other domains contribute critical ideas. The **MinMax k-Chinese Postman Problem** (Liu et al., 2019) decomposes edge traversal into k balanced closed tours sharing depot vertices — structurally identical to multi-loop track where junction nodes serve as cycle splitting points. And **racing track generation** for games (Loiacono et al., 2011, *IEEE Trans. Computational Intelligence and AI in Games*) evolves closed racing circuits from sequences of curved and straight segments with an explicit 360° angular closure constraint, which is the discrete-piece analog of the LEGO geometric-closure requirement. The difference is continuous parameters versus discrete fixed-geometry pieces.

---

## Four candidate chromosome architectures

The literature reveals four viable encoding patterns for multi-loop branching track layout, each with distinct trade-offs in locality, redundancy, and decoder complexity.

**Architecture A — Permutation + Split.** Encode all track pieces as a single permutation (via random keys sorted to produce a priority ordering). A dynamic-programming Split algorithm then partitions this permutation into feasible closed sub-loops at switch/junction genes, analogous to Prins's (2004) VRP decoder (*Computers & Operations Research* 31:1985–2002, DOI: 10.1016/S0305-0548(03)00158-8). The Split builds an auxiliary DAG where arc (i, j) represents a candidate sub-loop using pieces i+1 through j; the shortest path through this DAG gives the optimal partition. **Key advantage**: standard permutation crossover operators (OX, PMX) work directly on the chromosome — no trip delimiters, no repair needed at the chromosome level. All feasibility logic is encapsulated in the decoder. Vidal (2016) proved this Split runs in O(n) for capacitated VRP; a geometric-closure version would need to verify cumulative angle and position constraints per candidate arc, raising complexity but remaining tractable for inventories of ~100 pieces. **Key risk**: the geometric closure constraint (Δθ = ±360°, Δx = Δy = 0 per loop) is harder than a simple capacity check, so the Split's feasibility oracle may reject many arcs, producing thin feasible regions in the permutation space.

**Architecture B — Multi-segment chromosome with indicator + permutation keys.** Divide the chromosome into segments with different decoding semantics, following the Andrade et al. (2013) k-IMDMTSP pattern. Segment 1 (n_switches keys, indicator-decoded): determines which pieces become junctions and how branches connect. Segment 2 (n_pieces keys, permutation-decoded): determines piece ordering within each branch. This mirrors the three-layer encoding of Wen et al. (2023, *Expert Systems with Applications*, DOI: 10.1016/j.eswa.2023.120217) for production-inventory-distribution scheduling, where each layer has its own crossover semantics and the layers interact only through the decoder. **Key advantage**: clean separation of topology decisions from sequencing decisions improves locality — changing a topology key restructures connectivity without scrambling piece orderings, and vice versa. **Key risk**: the segment boundary is a design choice; poor boundary placement creates interdependencies that the crossover cannot navigate.

**Architecture C — Kia-style matrix chromosome with hierarchical decoder.** Following Kia et al. (2014, *Journal of Manufacturing Systems* 33:218–232, DOI: 10.1016/j.jmsy.2013.12.005), encode the solution as a set of interrelated matrices: branch topology, piece-to-branch assignment, junction positioning, intra-branch piece sequencing, and inventory status. The hierarchical decoder builds solutions level-by-level: (1) determine branch topology and junction positions, (2) assign pieces to branches respecting inventory counts, (3) sequence pieces within each branch, (4) verify geometric loop closure, (5) compute objectives. This five-ingredient structure was validated on an NP-hard integrated cell-formation and multi-floor layout problem. Wu, Chu, Wang & Yan (2007, *European Journal of Operational Research* 181:156–167) established the two-layer hierarchical GA for correlated cell-formation and layout decisions; Feng et al. (2018, *International Journal of Production Economics*) extended it to three layers for flexible routing. **Key advantage**: the matrix structure preserves interpretability and maps each decision dimension to distinct genes. **Key risk**: matrix-based chromosomes are harder to implement in pymoo's vector-oriented framework and may suffer from high redundancy if many matrix entries are structurally constrained.

**Architecture D — Grammar-based / L-system generative encoding.** Encode the track layout as a string in a formal grammar, decoded via turtle-graphics interpretation. L-systems (Prusinkiewicz & Lindenmayer, 1990) support branching via push/pop bracket notation — the turtle saves state at `[`, explores a branch, then returns at `]`. Hornby & Pollack (2001, *Computers & Graphics* 25:1041–1048) demonstrated that generative L-system encodings scale better than direct encodings for complex structures, producing creatures with hundreds of parts from small genomes. For track layout, a grammar could define production rules like `<loop> ::= <segment>+ | <segment>+ <junction> <loop> <segment>+`, where `<segment>` maps to track pieces. **Critical limitation**: Rothlauf & Oetzel (2006, EuroGP, LNCS 3905) measured that **82–94% of genotypic neighbors in Grammatical Evolution map to entirely different phenotypes** — catastrophic locality failure. Standard L-systems also produce tree structures, not closed loops; achieving closure requires constrained L-systems where cumulative angular turn must sum to 360° per loop, which is extremely difficult to guarantee grammatically. This architecture is theoretically elegant but practically problematic for the LEGO problem.

---

## Decoder design: from Rothlauf's principles to constructive track decoders

Franz Rothlauf's framework (*Representations for Genetic and Evolutionary Algorithms*, Springer, 2006, DOI: 10.1007/3-540-32444-5) establishes that **locality is the single most critical property** of any EA representation. High locality means small genotypic changes produce small phenotypic changes, preserving the fitness landscape's structure for gradient-like search. Low locality reduces the EA to near-random search. Rothlauf demonstrated this empirically on network design: the Prüfer number encoding for spanning trees has catastrophically low locality — a single-digit change can completely restructure the tree — while NetKeys (random-key-based) encoding exhibited high locality. This finding directly favors random-key-based encodings (Architectures A and B) over grammar-based encodings (Architecture D) for the track layout problem.

A constructive decoder for branching track layouts must maintain **turtle-graphics state** — position (x, y) and heading angle θ — at every active branch tip during incremental construction. Each track piece contributes a fixed local-frame transformation (Δx, Δy, Δθ) determined by its geometry: a straight piece advances the turtle by its length along the heading; a curve piece of radius R and arc angle α rotates the heading by ±α and displaces the position accordingly. At a switch piece, the turtle forks: the through-route continues with one transformation, and the diverging route pushes a new branch tip onto a stack. Each branch must independently close — the cumulative transformation from its start junction must return to the identity (or to its end junction's port geometry). Aickelin & Dowsland (2000, *Journal of Scheduling* 3:139–153) identified three critical decoder properties: **computational efficiency** (called ~100,000 times per GA run), **determinism** (same chromosome always yields same solution), and **completeness** (ability to reach all feasible solutions). For a track decoder processing ~100 pieces, O(n) per call is achievable, and the turtle-state approach is inherently deterministic.

The Prins Split analogy provides the key mechanism for partitioning a piece sequence into multiple closed loops. A "track Split" algorithm would construct an auxiliary DAG where arc (i, j) represents a candidate sub-loop using pieces at positions i+1 through j in the permutation. The arc's feasibility requires that the cumulative geometric transformation of those pieces — computed via forward kinematics — returns to the origin (or to a valid junction port). Arc cost is the negative of the sub-loop's contribution to the two objectives. A shortest-path computation through this DAG yields the optimal partition. The crucial design decision is **where junctions sit**: either junction positions are pre-determined by indicator keys in the chromosome (Architecture B), or the Split algorithm treats switch pieces encountered in the permutation as mandatory split points. The latter is simpler but less flexible.

For the LEGO problem's specific geometry, loop closure verification reduces to checking that the sum of all piece transformations in the loop yields Σ(Δθ) = ±360° and Σ(Δx, Δy) = (0, 0) in global coordinates. Since piece angles are rational multiples of common angles (22.5° for R40, 11.25° for R56, 9° for R72/R88/R104/R120, etc.), the angular closure is checkable in exact arithmetic. The positional closure requires either floating-point tolerance or exact computation using the known piece geometries.

---

## Integrating BRKGA-style encoding with NSGA-II in pymoo

The canonical method for combining BRKGA's representation advantages with NSGA-II's multi-objective selection is the **NDS-BRKGA** framework of Chagas, Blank, Wagner, Souza & Deb (2021, *Journal of Heuristics* 27:267–301, DOI: 10.1007/s10732-020-09457-7). This landmark paper applied non-dominated sorting with crowding distance as the survival mechanism while retaining BRKGA's biased crossover — the elite parent contributes each allele with probability ρ_e > 0.5. Tested on the bi-objective traveling thief problem (combining TSP route optimization with knapsack item selection — structurally similar to combining track layout with inventory selection), NDS-BRKGA matched or outperformed all prior approaches. The paper confirms that NSGA-II's crowding distance operates entirely in objective space and is **completely agnostic to chromosome structure** — it works identically for flat vectors, multi-segment chromosomes, or any other encoding.

In pymoo, the implementation path has two options. **Option 1 (recommended)**: define the problem with `n_var` equal to the total chromosome length, all variables as `Real` in [0, 1], and implement a custom decoder in `_evaluate()` that maps the random-key vector to a track layout and computes both objectives. Use pymoo's `NSGA2` algorithm with `RankAndCrowdingSurvival()` and a custom `Crossover` operator implementing BRKGA's biased mating. This requires minimal framework customization. **Option 2**: use pymoo's `MixedVariableGA` with `survival=RankAndCrowdingSurvival()` and define variable types per chromosome segment (Real for indicator keys, Integer for categorical decisions). The `MixedVariableCrossover` automatically applies type-specific operators. A custom `Repair` class can enforce inventory constraints post-crossover.

The state-of-the-art **BRKGA-MP-IPR** framework (Andrade, Toso, Gonçalves & Resende, 2021, *European Journal of Operational Research* 289:17–30, DOI: 10.1016/j.ejor.2019.11.037) extends basic BRKGA with multi-parent crossover and implicit path-relinking in the random-key space. Path-relinking generates intermediate solutions between two good solutions — operating entirely problem-independently in [0,1]^n — and has proven effective for complex network design problems. The **Random-Key Optimizer** (Chaves, Resende et al., 2025, *Journal of Heuristics*, DOI: 10.1007/s10732-025-09568-z) goes further: implement the track-layout decoder once, and automatically access GA, simulated annealing, iterated local search, VNS, GRASP, PSO, and large neighborhood search — all operating in the same random-key space. This modular design is ideal for novel problem domains where the best metaheuristic is unknown a priori.

MOEA/D is the principal algorithmic alternative to NSGA-II. Li & Zhang (2009, *IEEE Trans. Evolutionary Computation* 13:284–302) showed MOEA/D outperforms NSGA-II on problems with complicated Pareto set shapes, and Peng, Zhang & Li (2009) demonstrated its major advantage: easy integration of single-objective local search per subproblem. For the bi-objective LEGO problem, NSGA-II is the safer default — **two objectives is NSGA-II's sweet spot**, and pymoo's infrastructure is most mature for it — but MOEA/D becomes attractive once effective local search operators (piece-swap, sub-loop rotation, junction relocation) are developed.

---

## Modeling the two objectives and their interaction

**Piece utilization** from a fixed inventory is structurally identical to the knapsack component in the traveling thief problem. The decoder counts pieces placed versus pieces available. The Kia et al. (2014) machine-depot concept maps directly: unplaced pieces remain in the "depot." A decoder-enforced approach greedily places pieces from remaining inventory at each construction step, using random-key values as priorities; pieces not placed by construction's end are unused. This avoids the need for a separate penalty or repair mechanism for inventory feasibility. The constraint is that each piece type cannot be used more times than its inventory count — naturally enforced by decrementing a counter during constructive decoding.

**Safe traversal speed** uses the quasi-static lateral force model v_max = √(μ·g·R) per curve segment, where μ is the friction coefficient, g is gravitational acceleration, and R is the curve radius. Straight segments have unlimited speed (or a practical maximum). Switch/junction pieces have their own speed limit based on the diverging-route geometry. The length-weighted harmonic mean of segment speeds gives the **space-mean speed**, which is the physically correct average for a vehicle traversing segments at different speeds:

v̄ = L_total / Σ(L_i / v_i)

This metric naturally penalizes layouts with many tight-radius curves (which create bottleneck segments), rewarding the use of high-radius curves (R104, R120, R168) that permit faster traversal.

The two objectives exhibit a **partial synergy**: using more pieces (especially large-radius curves) increases both utilization and speed. However, the relationship becomes adversarial when switch pieces enter. Switches add branches (increasing utilization by enabling more pieces to be placed) but reduce speed through the junction's tight diverging geometry. A Pareto front is expected where one extreme is a simple large oval (few pieces, high speed) and the other extreme is a complex multi-branch layout (many pieces, lower speed due to numerous junctions and forced tight curves). The interesting solutions lie between these extremes — moderate branching with preferential use of high-radius curves in the main loop.

---

## Recommended architecture: multi-segment BRKGA with hierarchical constructive decoder

Based on the full literature analysis, the recommended chromosome architecture combines Architecture B's multi-segment random-key encoding with the Prins Split mechanism and a turtle-graphics constructive decoder, deployed under NDS-BRKGA (NSGA-II survival + BRKGA crossover).

**Chromosome structure**: A vector of N = n_topo + n_pieces real-valued keys in [0, 1].

- **Topology segment** (first n_topo keys, indicator-decoded): Determines the number and connectivity of sub-loops. Specifically, these keys decide (a) which switch pieces from inventory are activated, (b) how they connect the loops — e.g., a key > 0.5 activates a switch, and its value partitions into connectivity options. With k activated switches, the topology defines a multigraph of k+1 sub-loops.
- **Piece-sequencing segment** (remaining n_pieces keys, permutation-decoded): Sorted to produce a priority ordering of all non-switch track pieces. This ordering feeds into the constructive decoder, which assigns and sequences pieces within each sub-loop.

**Decoder algorithm** (hierarchical, constructive, O(n)):

1. **Topology pass**: Read topology keys to activate switches and determine the loop connectivity graph. Place switch pieces at their fixed-geometry junction configurations.
2. **Assignment pass**: For each sub-loop, the decoder draws pieces from the permutation ordering, one by one, using turtle-graphics forward kinematics to extend the current branch. At each step, it checks whether the next piece in priority order (a) fits geometrically without collision, (b) is available in inventory, and (c) does not make loop closure impossible (angle/position reachability check). If the piece fails, skip it and try the next.
3. **Closure pass**: When a sub-loop's cumulative angle approaches ±360°, the decoder attempts closure by selecting from remaining inventory the piece sequence that returns the turtle to the start junction's port. If closure is impossible, the sub-loop is truncated and remaining pieces recycled to other loops.
4. **Evaluation**: Compute piece utilization = pieces_placed / pieces_available. Compute space-mean speed = L_total / Σ(L_i / v_max_i) across all sub-loops.

**Crossover**: BRKGA biased crossover — each allele inherited from the elite parent with probability ρ_e ≈ 0.7. This is representation-agnostic and preserves both topology and sequencing patterns from good solutions. No problem-specific crossover operator needed.

**Local search (optional enhancement)**: Per the MOEA/D literature and Vidal et al.'s Hybrid Genetic Search, integrate light local search moves: (a) swap two adjacent pieces within a sub-loop, (b) relocate a piece from one sub-loop to another, (c) toggle a switch piece on/off (topology perturbation). Apply with probability p_LS after decoding, accepting improvements in either objective.

**pymoo implementation skeleton**:

```python
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
import numpy as np

class TrackLayoutProblem(Problem):
    def __init__(self, inventory, n_topo, n_pieces):
        super().__init__(n_var=n_topo + n_pieces,
                         n_obj=2, xl=0.0, xu=1.0)
        self.inventory = inventory
        self.n_topo = n_topo

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2))
        for i, chromosome in enumerate(X):
            layout = self.decode(chromosome)
            F[i, 0] = -layout.piece_utilization  # minimize negative
            F[i, 1] = -layout.space_mean_speed
        out["F"] = F

    def decode(self, chromosome):
        topo_keys = chromosome[:self.n_topo]
        piece_keys = chromosome[self.n_topo:]
        # 1. Topology pass: activate switches, build loop graph
        # 2. Assignment pass: turtle-graphics construction
        # 3. Closure pass: attempt loop closure
        # 4. Return layout with objectives
        ...

class BiasedCrossover(Crossover):
    def __init__(self, bias=0.7):
        super().__init__(n_parents=2, n_offsprings=1)
        self.bias = bias

    def _do(self, problem, X, **kwargs):
        n_matings, _, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < self.bias
        # Elite parent (first) contributes where M is True
        return X[:, 0] * M + X[:, 1] * (~M)
```

---

## Key citations with assessed relevance

The following table organizes the most important references by their contribution to the LEGO track problem, ordered by direct applicability:

| Relevance | Citation | Contribution |
|-----------|----------|-------------|
| ★★★★★ | Andrade, Miyazawa & Resende (2013), "Evolutionary algorithm for the k-interconnected multi-depot multi-TSP," GECCO 2013, pp. 463–470 | Inner + outer cycles from one BRKGA chromosome; closest structural match |
| ★★★★★ | Prins (2004), "A simple and effective evolutionary algorithm for the VRP," *Comput. Oper. Res.* 31:1985–2002, DOI: 10.1016/S0305-0548(03)00158-8 | Giant-tour + Split decoder; enables delimiter-free multi-loop partition |
| ★★★★★ | Chagas, Blank, Wagner, Souza & Deb (2021), "A non-dominated sorting based customized random-key GA for the bi-objective TTP," *J. Heuristics* 27:267–301, DOI: 10.1007/s10732-020-09457-7 | NDS-BRKGA: canonical BRKGA + NSGA-II hybrid for bi-objective problems |
| ★★★★☆ | Gonçalves & Resende (2011), "Biased random-key GAs for combinatorial optimization," *J. Heuristics* 17:487–525, DOI: 10.1007/s10732-010-9143-1 | Foundational BRKGA framework; three decoder types defined |
| ★★★★☆ | Freitas, Aloise, Fontes, Santos & Menezes (2023), "BRKGA for the two-level hub location routing problem," *OR Spectrum* 45:903–924, DOI: 10.1007/s00291-023-00718-y | Two-level hub ring + spoke cycles; near-identical structure to branching track |
| ★★★★☆ | Kia, Khaksar-Haghani, Javadian & Tavakkoli-Moghaddam (2014), "Solving a multi-floor layout design model of a DCMS," *J. Manuf. Syst.* 33:218–232, DOI: 10.1016/j.jmsy.2013.12.005 | Matrix chromosome with hierarchical decoder; floor ↔ branch analogy |
| ★★★★☆ | Rothlauf (2006), *Representations for Genetic and Evolutionary Algorithms*, Springer, DOI: 10.1007/3-540-32444-5 | Locality/redundancy framework; proves random-key encodings superior for network problems |
| ★★★★☆ | Londe, Pessoa, Andrade & Resende (2025), "Biased random-key GAs: A review," *Eur. J. Oper. Res.* 321:1–22, DOI: 10.1016/j.ejor.2024.03.030 | Comprehensive BRKGA taxonomy; 150+ applications cataloged |
| ★★★★☆ | Andrade, Toso, Gonçalves & Resende (2021), "BRKGA-MP-IPR," *Eur. J. Oper. Res.* 289:17–30, DOI: 10.1016/j.ejor.2019.11.037 | Multi-parent crossover + implicit path-relinking; state-of-the-art BRKGA |
| ★★★☆☆ | Loiacono et al. (2011), "Automatic track generation for racing games using evolutionary computation," *IEEE Trans. CI & AI in Games* | Closed racing track from curved/straight segments with 360° closure |
| ★★★☆☆ | Duhamel, Aloise, Santos & Oliveira (2016), "Split procedure for SONET Ring Problem with BRKGA," IFAC MIM 2016 | Permutation + Split → ring partition; DP-based cycle decomposition |
| ★★★☆☆ | Vidal, Crainic, Gendreau, Lahrichi & Rei (2012), "A hybrid genetic algorithm for MDVRP," *Oper. Res.* 60:611–624 | Multi-depot VRP with Split; depot ↔ junction analogy |
| ★★★☆☆ | Liu et al. (2019), "A GA for MinMax k-Chinese Postman Problem," SHMII-9 | Multi-cycle edge coverage from shared junctions |
| ★★★☆☆ | Chaves, Resende et al. (2025), "A Random-Key Optimizer for Combinatorial Optimization," *J. Heuristics*, DOI: 10.1007/s10732-025-09568-z | RKO: one decoder, seven metaheuristics; maximum flexibility |
| ★★★☆☆ | Wu, Chu, Wang & Yan (2007), "A genetic algorithm for cellular manufacturing design and layout," *Eur. J. Oper. Res.* 181:156–167 | Two-layer hierarchical chromosome; correlated-decision precedent |
| ★★☆☆☆ | Gonçalves & Resende (2015), "BRKGA for unequal area facility layout," *Eur. J. Oper. Res.* 246:86–107, DOI: 10.1016/j.ejor.2015.04.029 | Multi-stage layout decoder with LP refinement |
| ★★☆☆☆ | Hornby & Pollack (2001), "Evolving L-systems to generate virtual creatures," *Comput. Graph.* 25:1041–1048 | Generative encoding scalability; L-system branching |
| ★★☆☆☆ | Wen et al. (2023), "Solving bi-objective integrated scheduling with modified NSGA-II," *Expert Syst. Appl.*, DOI: 10.1016/j.eswa.2023.120217 | Three-layer encoding with per-layer crossover operators |
| ★★☆☆☆ | Kim & de Weck (2005), "Variable chromosome length GA for topology optimization," *Struct. Multidisc. Optim.* 29:445–456 | Progressive refinement: coarse topology → fine detail |
| ★★☆☆☆ | Blank & Deb (2020), "pymoo: Multi-Objective Optimization in Python," *IEEE Access* 8:89497–89509 | Implementation platform; custom operators, mixed variables |

---

## Conclusion: what this analysis changes about the design problem

The central insight from this literature synthesis is that the LEGO branching track problem is **not** best modeled as a layout problem with geometric constraints — it is best modeled as a **multi-cycle routing problem with inventory constraints and geometric feasibility**. The hub-location routing and k-IMDMTSP literature provides a mature, validated chromosome architecture (multi-segment BRKGA with hierarchical decoder) that handles both the topology decision (how many loops, where junctions sit) and the routing decision (which pieces go where, in what order) within a single random-key vector.

The practical recommendation diverges from the Kia et al. (2014) matrix-chromosome analogy despite the conceptual appeal of the floor ↔ branch mapping. The matrix structure introduces high redundancy and poor locality for the geometric-closure constraint — the hardest subproblem — while the BRKGA random-key approach delegates all constraint satisfaction to a turtle-graphics constructive decoder that naturally tracks geometric state across branch forks. The Prins Split mechanism provides the missing link: a principled, DP-based method for partitioning a piece permutation into multiple closed loops without requiring explicit loop delimiters in the chromosome.

Three unresolved design questions remain for implementation. First, the geometric-closure feasibility oracle within the Split decoder must be fast — precomputing reachable closure sets for common angle/position states could amortize this cost. Second, the interaction between topology keys and sequencing keys creates a partially epistatic landscape; adaptive operator selection (as in BRKGA-MP-IPR's multi-parent crossover) may be needed to manage this. Third, the speed objective's harmonic-mean formulation creates a non-smooth fitness landscape with plateau regions where adding a single tight curve dramatically reduces the mean speed — surrogate-assisted evaluation or decomposition-based search (MOEA/D) may help navigate these discontinuities.