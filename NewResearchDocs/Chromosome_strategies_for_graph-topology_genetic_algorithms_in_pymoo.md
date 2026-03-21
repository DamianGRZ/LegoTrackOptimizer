# Chromosome strategies for graph-topology genetic algorithms in pymoo

**A linear integer chromosome cannot represent branching LEGO track topologies — but pymoo fully supports arbitrary Python objects (including NetworkX graphs) as genotypes, provided you implement custom genetic operators.** The most promising encoding strategies are a constructive decoder with random-key chromosomes (guaranteeing validity by construction), a NEAT-inspired node+connection genome with innovation numbers (enabling meaningful crossover between topologically different layouts), and an L-system generative encoding (compactly representing complex branching patterns). No existing GA-based LEGO track optimizer exists — this is genuinely novel territory, but decades of research on evolving graph topologies in pipe networks, circuit routing, and neural architecture search provide proven techniques directly applicable to this problem.

## Why linear chromosomes break and what replaces them

The fundamental problem is one of **representational mismatch**. A linear integer sequence `[straight, curve, curve, straight, ...]` implies a single path traced sequentially — it cannot express the moment a switch splits flow into two independent branches that later rejoin. The solution space for branching track layouts is a graph, not a sequence, and the chromosome must encode graph structure.

Six established encoding families address this, each with distinct trade-offs:

**Adjacency matrix encoding** flattens an n×n connectivity matrix into a fixed-length binary chromosome. Standard GA operators (bit-flip mutation, one-point crossover) work directly. The penalty is **O(n²) scaling** — a layout with 50 track pieces requires a 2,500-bit chromosome, most configurations of which produce disconnected or physically impossible graphs. Miller, Todd & Hegde (1989) pioneered this for neural networks; Sid et al. (2007) adapted it for structural topology optimization. For LEGO tracks with modest piece counts (<100), this remains viable if paired with aggressive repair operators.

**Edge list encoding** stores only existing connections as (source, target) pairs, scaling with edge count rather than node count squared. Variable-length chromosomes require specialized operators — Munemoto's "Same Point" crossover (1998) and Wang et al.'s "Same Adjacency" crossover (2016) address alignment between parents of different lengths. This encoding naturally fits sparse track networks where each piece connects to only 2–4 neighbors.

**NEAT-style node+connection encoding** assigns each structural innovation (adding a piece or connection) a globally unique **innovation number**. During crossover, genes with matching innovation numbers align automatically, enabling meaningful recombination even between layouts of radically different complexity. Stanley & Miikkulainen (2002) demonstrated this for neural networks, but the mechanism is purely graph-theoretic — it works for any evolving topology. For LEGO tracks, node genes would encode piece type and position; connection genes would encode which ports link to which, plus an enabled/disabled flag.

**L-system encoding** represents the layout as a set of rewriting rules rather than the layout itself. The chromosome encodes production rules; a deterministic interpreter expands these rules into a track network. Kitano (1990) showed **order-of-magnitude speedup** over direct encoding because L-systems compactly capture repetition and modularity. The `[` and `]` branching symbols in turtle-graphics L-systems map directly to switch semantics: `[` pushes the turtle state (saves position at switch), traces one branch, then `]` pops back to trace the diverging branch. Bielefeldt et al. (2019, Cambridge University Press) published a complete L-System + GA topology optimization framework with 52 design variables generating complex 2D branched structures.

**Constructive/priority decoder encoding** uses a fixed-length real-valued chromosome interpreted by a deterministic builder algorithm. Each gene value guides a construction decision (which piece to place, which open port to extend next). Because the decoder only places geometrically valid pieces and only connects compatible ports, **every chromosome maps to a valid layout by construction** — no repair needed. Bean's random-keys approach (1994) established this paradigm. For track layouts, the decoder maintains a queue of open connection points, and at each step the chromosome selects which point to extend and what piece to place.

**Graph grammar encoding** evolves production rules that generate track networks through iterative subgraph replacement. Rules like "replace a straight segment with a switch + two branches" encode topological transformations. Kitano's graph generation system (1990) and Luerssen & Powers' hypergraph grammar evolution (2007) both demonstrated compact, scalable representations. NetGAP (2023) combined recursive graph grammars with Monte-Carlo Tree Search for network topology design.

| Encoding | Chromosome length | Handles branching | Guarantees validity | Operator complexity | Best for |
|---|---|---|---|---|---|
| Adjacency matrix | O(n²) fixed | Yes | No — needs repair | Low (standard ops) | Small layouts, rapid prototyping |
| Edge list | O(m) variable | Yes | No — needs repair | Medium | Sparse networks |
| NEAT-style | Variable (grows) | Via add-node mutation | Incremental growth helps | Medium-high | Topology exploration |
| L-system | Fixed (compact rules) | Native via `[`/`]` | Mostly — needs closure check | Medium | Repetitive/symmetric layouts |
| Constructive decoder | Fixed (real-valued) | Via decoder logic | **Yes, by construction** | Low (standard on keys) | General-purpose, robust |
| Graph grammar | Fixed (compact rules) | Via production rules | Rule-dependent | High | Structured, modular networks |

## How pymoo accommodates graph-based genotypes

pymoo explicitly supports arbitrary Python objects as decision variables — the official documentation states that "different kinds of variable types can be used (also more complicated ones such as tree, graph, …)." The mechanism uses **`n_var=1` with a numpy array of `dtype=object`**, where each individual's single "variable" is an entire Python object such as a NetworkX graph, a custom layout class, or a tree structure.

The critical requirement is implementing **four custom operator classes** plus an optional repair:

```python
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.duplicate import ElementwiseDuplicateElimination

class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        layout = x[0]  # x[0] is your layout object
        out["F"] = [total_track_length(layout), -route_complexity(layout)]
        out["G"] = [
            nx.number_connected_components(layout.graph) - 1,  # ≤0 = connected
            count_dangling_ports(layout),  # ≤0 = no dangling ends
        ]
```

The `Sampling` class creates initial random layouts (returning a `(n_samples, 1)` object array). `Crossover.__init__(2, 2)` declares two parents producing two offspring, with `_do()` receiving shape `(n_parents, n_matings, n_var)`. `Mutation._do()` receives `(n_individuals, n_var)`. The `Repair` operator runs after crossover/mutation but before evaluation, making it the natural place to enforce connectivity and close dangling ends.

**Constraint handling** in pymoo offers five strategies. For topological constraints, the recommended approach combines a **Repair operator** as primary defense (fixing connectivity and dangling ends structurally) with **inequality constraints in `_evaluate()`** as secondary defense (penalizing any remaining violations via `out["G"]`). pymoo's default feasibility-first ranking ensures feasible layouts always dominate infeasible ones in selection.

For the standard algorithm setup, `NSGA2` works with all custom operators:

```python
algorithm = NSGA2(
    pop_size=100,
    sampling=TrackSampling(),
    crossover=TrackCrossover(),
    mutation=TrackMutation(),
    repair=TrackConnectivityRepair(),
    eliminate_duplicates=TrackDuplicateElimination()
)
```

pymoo's `MixedVariableGA` supports `Real`, `Integer`, `Binary`, and `Choice` types out of the box but cannot directly accommodate graph structures — the custom object approach with a standard algorithm is required instead.

## Crossover operators that preserve track connectivity

The central challenge of graph-based crossover is producing offspring that remain valid connected networks. Three operator families solve this with different guarantees.

**The Globus/JavaGenes graph crossover** (NASA Ames) is the most thoroughly proven general-purpose approach and **guarantees connectivity by construction** for undirected graphs. The division algorithm repeatedly finds shortest paths between two endpoints of a chosen edge and removes random edges from these paths until a cut-set is found, producing two connected fragments with identified "broken edges" at the cut boundary. The combination algorithm merges one fragment from each parent by reconnecting their broken edges — matching broken edges from different parents are joined, while unmatched ones are either attached to random nodes in the other fragment or discarded. For LEGO tracks, broken edges correspond to track connection points at the subgraph boundary, and the matching step would enforce connector-type compatibility (male-to-female stud pairing).

**NEAT innovation-number crossover** aligns parent genomes by historical markers rather than position. Genes with matching innovation numbers (representing the same structural feature discovered independently or inherited from a common ancestor) are randomly selected from either parent. Disjoint and excess genes (structural innovations present in only one parent) inherit from the fitter parent. This enables meaningful recombination between layouts of completely different topologies — a simple oval can cross with a complex figure-eight, and the offspring inherits the best-proven components from each. For track layouts, each track piece and connection receives an innovation number when first created through mutation, and these numbers persist through generations.

**Decoder-based crossover** sidesteps the graph-validity problem entirely. If the genotype is a fixed-length real-valued vector interpreted by a constructive decoder, then standard one-point or uniform crossover on the genotype always produces a valid genotype, and the decoder always produces a valid layout. This is the simplest approach to implement and the most robust, at the cost of an indirect genotype-phenotype relationship that may create a more rugged fitness landscape.

For mutation, a **graduated operator set** balances exploration and exploitation: type mutation changes a piece while preserving connection count (30% probability), segment replacement swaps a path section for an alternative route between the same endpoints (20%), switch insertion converts a straight segment into a branching point with a new diverging path (15%), and piece insertion/deletion adds or removes individual pieces with subsequent repair (10%). NEAT's add-node mutation — splitting an existing connection by inserting a new node — is minimally disruptive because the new node initially passes through without changing layout behavior, giving subsequent evolution time to optimize the new branch.

## The constructive decoder approach in detail

The **constructive decoder** deserves special attention as the most practically robust encoding for this problem. The chromosome is a fixed-length array of real values in [0,1], and a deterministic decoder algorithm interprets these values to build a track layout piece by piece:

1. Initialize with a starting track piece and its open connection points
2. For each gene in the chromosome, the decoder selects the highest-priority open connection point (priority determined by the gene value), chooses a compatible piece type (mapped from another gene value to the available piece catalog), and places it
3. When a switch is placed, its two output ports are both added to the open-port queue — the decoder naturally handles branching by processing these ports in subsequent steps
4. When an open port is geometrically close to another open port, the decoder attempts to connect them (closing a loop), guided by a gene value threshold
5. After all genes are consumed, remaining open ports trigger a closure heuristic

**Every chromosome produces a valid, connected layout.** Standard SBX crossover and polynomial mutation on the real-valued genes work without modification. The decoder embeds all physical constraints — piece dimensions, connector compatibility, spatial collision avoidance — so the evolutionary search operates in an unconstrained genotype space while the phenotype space is always feasible. This mirrors the proven architecture of pipe-network GAs like GENOME and GAnet, where the GA optimizes pipe selections while an embedded hydraulic solver (EPANET) handles physics.

The main design decision is the **decoder's construction order**: depth-first (finish one branch before starting another) versus breadth-first (extend all branches simultaneously). Breadth-first tends to produce more balanced layouts; depth-first tends to produce layouts with one dominant main line and shorter sidings. The gene values can also encode this choice.

## What related domains teach us about this problem

**Pipe network optimization** is the closest analog. Water distribution network design shares the core challenge: optimizing a branching physical network with discrete component choices, connectivity constraints, and flow physics. The GBMOGA approach (Graph-Based Multi-Objective GA) directly encodes spanning trees as chromosomes with custom tree-preserving crossover. Dandy et al.'s "adjacency mutation" (1996) — mutating a pipe to an adjacent diameter rather than a random one — maps to changing a curve radius one step rather than randomly, preserving geometric coherence. The field universally uses **embedded simulation** (EPANET hydraulic solver) within fitness evaluation, paralleling the need for geometric validation in track layout fitness.

**PCB routing** contributes the concept of **grid-cell encoding** — discretizing the layout area into cells where each cell may contain a track segment with a specific orientation. Lienig's channel-routing GA (1993) used Inter-Cluster Mutation, operating on groups of related connections rather than random individual genes, recognizing that local changes to one track connection cascade to neighboring connections. The PCB domain also demonstrates that **multi-terminal nets** (connections linking more than two pins) require Steiner-tree decomposition — analogous to how a switch creates a three-way connection point.

**No existing GA-based LEGO track optimizer exists.** Monty's Trains blog explicitly confirms: "a common theme of the question is whether software exists that will perform this action automatically. No, such software does not exist." Current tools (BlueBrick, SCARM, 3D Train Studio) are all manual drag-and-drop editors. The geometric constraints are well-documented: **16 curves form a full circle** (22.5° each), straight segments are 16 studs, standard parallel spacing is 8 studs, and switches have asymmetric geometry distinct from curves.

## Phenotype mapping must enforce LEGO-specific geometry

The genotype-to-phenotype decoder must respect LEGO track physics regardless of encoding choice. Each track piece is a rigid body with precisely positioned connection points:

- **Straight piece**: 2 ports, collinear, 16 studs apart
- **Curve piece**: 2 ports, 22.5° angular offset, on a circle of radius R40 (standard) or R56/R72/R88/R104 (third-party)
- **Switch/turnout**: 3 ports — 1 input, 1 through-output (straight), 1 diverging-output (at switch angle), 32 studs long
- **90° crossing**: 4 ports forming two perpendicular through-paths
- **Crossover**: 4 ports forming two paths crossing at a non-90° angle

The decoder must track each port's **absolute position (x, y) and heading angle (θ)** in world coordinates. Two ports can connect only if they are collinear (same position, opposite headings, within manufacturing tolerance). The connection-point graph — where each piece is a node with typed ports and edges represent physical connections between compatible ports — is the natural internal representation regardless of chromosome encoding.

For closure detection (essential because train tracks must form loops), the decoder should maintain a spatial index of all open ports and check for near-matches after each piece placement. When two open ports are within connection tolerance, they can be joined — either automatically (in a constructive decoder) or via fitness reward (in a direct encoding). The **closure constraint** distinguishes this problem from tree-like domains (botanical L-systems, circuit trees): every branch that starts at a switch must eventually reconnect to the network, forming a loop.

## A recommended three-tier architecture

The strongest practical approach combines ideas from multiple encoding families into a layered system:

**Tier 1 — Genotype**: A fixed-length real-valued chromosome using random-keys encoding. Each gene is a float in [0,1]. Standard pymoo operators (SBX crossover, polynomial mutation) work without modification. The chromosome length is set to accommodate the maximum expected layout complexity (e.g., 200 genes for layouts of up to ~80 pieces with branches).

**Tier 2 — Constructive decoder**: A deterministic algorithm that reads the chromosome and builds a track layout step by step. Gene values map to piece-type selections, port-priority rankings, and closure-attempt thresholds. The decoder maintains a priority queue of open connection points and processes them in gene-driven order. Switches create multiple open ports; the decoder handles branching naturally by queuing both outputs. Physical constraints (piece geometry, collision avoidance, connector compatibility) are hard-coded into the decoder, ensuring every chromosome produces a valid layout.

**Tier 3 — Phenotype evaluation**: The resulting track layout is evaluated for multiple objectives — total usable track length, route diversity (number of distinct loops), aesthetic metrics (symmetry, compactness), and space utilization within the available area. Constraint violations that slip past the decoder (if any) are reported as inequality constraints in `_evaluate()`.

This architecture gives you **guaranteed validity** (every individual is feasible), **standard pymoo operators** (no custom crossover/mutation needed), and **rich expressiveness** (the decoder can produce any topology the construction grammar permits). The main implementation effort goes into the decoder, which is a well-defined engineering task rather than an open research problem.

For users wanting maximum topological exploration — where the optimal number of switches and branches is unknown — augmenting this with **NEAT-style complexification** is worthwhile: start evolution with simple single-loop layouts and gradually introduce switch-insertion mutations that increase branching complexity, protecting novel topologies via speciation. This prevents the premature convergence that can occur when the initial population contains layouts of wildly different complexity competing directly against each other.

## Conclusion

The path from linear chromosomes to branching track topologies requires rethinking representation, not just adding genes. **The constructive decoder with random-key chromosomes is the pragmatic first choice** — it guarantees validity, works with pymoo's built-in operators, and sidesteps the need for custom graph crossover entirely. For more sophisticated topology exploration, a NEAT-inspired encoding with innovation-number-aligned crossover offers principled handling of variable-complexity layouts but demands significantly more implementation effort. L-system encoding is compelling for layouts with repetitive structure (parallel sidings, symmetric yards) but requires a closure mechanism foreign to traditional L-systems.

The key insight from related domains is that **embedded constraint satisfaction inside the decoder** (as pipe-network GAs embed EPANET) is more effective than post-hoc repair or penalty functions. Build the physics into the decoder, let the GA optimize freely in genotype space, and let the decoder translate every genotype into a valid track layout. pymoo's `dtype=object` support and custom operator architecture make this cleanly implementable — define your layout class, write your decoder, plug in the four operator classes, and evolve.