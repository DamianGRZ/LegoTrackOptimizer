# Encoding 4DBrix tracks for evolutionary layout optimization

**The 4DBrix track system's non-standard, radius-dependent arc angles create a unique encoding challenge for genetic algorithms.** Unlike every other LEGO-compatible manufacturer that standardizes on 11.25° per curve, 4DBrix uses angles ranging from 22.5° (R40) down to 9° (R120), chosen so each radius produces clean integer piece counts per 90° turn. This report catalogs the full 4DBrix piece library, then evaluates three representation strategies—port-centric, parametric/arc-based, and graph-edge—for pymoo-based evolutionary layout generation. The parametric approach emerges as the strongest fit for mutation-only closed-loop evolution, though a hybrid strategy borrowing the port-centric model for piece definitions delivers the most complete solution.

---

## The complete 4DBrix track piece library

4DBrix organizes its train track catalog into five categories: **Straights & Curves**, **R40 Switches**, **R56 Switches**, **Ultimate Railroader (R148)**, and **Train Track Sets**. All pieces are 8 studs wide, fully anti-studded (for baseplate mounting), and manufactured as 3D-printed PLA available either as STL downloads or through authorized printers like OKBrickWorks.

### Standard curves with their critical non-standard angles

The defining feature of 4DBrix geometry is that **arc angle varies inversely with radius** such that each radius yields a clean integer number of pieces per quarter-turn. Adjacent radii differ by exactly 16 studs (center-to-center parallel track spacing), except R148 which breaks this pattern for Pythagorean-triple crossover geometry.

| Radius | Arc Angle | Pieces / 90° | Pieces / Circle | Chord (studs) | Arc Length (studs) | Part Number |
|--------|-----------|--------------|-----------------|----------------|-------------------|-------------|
| R40 | 22.5° | 4 | 16 | 15.61 | 15.71 | Standard LEGO |
| R56 | **18°** | 5 | 20 | 17.52 | 17.59 | 2.04.030 |
| R72 | **15°** | 6 | 24 | 18.80 | 18.85 | 2.04.031 |
| R88 | 11.25° | 8 | 32 | 17.25 | 17.28 | ~2.04.032 |
| R104 | **10°** | 9 | 36 | 18.13 | 18.15 | — |
| R120 | **9°** | 10 | 40 | 18.83 | 18.85 | — |
| R148 | **≈18.93°** | — | — | 48.66 | 48.90 | 2.04.074 |

R40 (standard LEGO radius) is not sold separately by 4DBrix but appears in switch component geometry. R148's angle derives from the **(12, 35, 37) × 4 Pythagorean triple**: 148 × sin(arctan(12/35)) = 48 studs forward, 148 × (1 − cos(arctan(12/35))) = 8 studs lateral—producing 96-stud crossovers with exactly 16-stud track spacing.

### Straight tracks

Three lengths serve as gap-fillers alongside standard LEGO 16-stud straights:

| Product | Length | Part Number |
|---------|--------|-------------|
| Quarter Straight | **4 studs** | 2.04.005 |
| Half Straight | **8 studs** | 2.04.001 |
| Full Straight | 16 studs | 2.04.003 |

### Switch components and the modular system

4DBrix's R40 switch system is radically modular: **22 individual components** combine into assembled configurations. Each component is approximately 16 studs long and connects via standard rail clips.

**R40 switch components** break into five functional groups:

- **Switch tracks** (#018, #018S, #019, #021, #021S, #022): the moving-rail split point, 16 studs long, available with ground throws on inside or outside, left or right hand
- **Divergent tracks** (#020, #023, #027, #028, #052, #053): the curved diverging route following the switch
- **Coupling tracks** (#025 at 14.4° R40, #026 at 8.1° R40, #029): reconnect divergent routes back to mainline or parallel track
- **Intersection tracks** (#024, #047): diamond crossing segments for crossovers
- **Three-way and wye tracks** (#043, #044, #045, #046, #051): specialized junction components

These assemble into standard configurations: the **R40 Parallel Track Switch** (32 studs total, 45° divergence, 16-stud track spacing), **R40 Double Crossover** (48 × 24 studs, four independent switches), **R40 Wye Switch** (16 studs, symmetric Y-junction), and **R40 Yard Ladder** sets for multi-track sidings.

**R56 switches** use a gentler **17.4° diverging angle** (Part 2.04.058, 16 studs long). The R56 system has fewer documented components than R40 but follows the same modular philosophy.

**R148 Ultimate Railroader crossovers** are the premium offering: 96 × 24 studs, 16-stud track spacing, all four switches independently settable. Available as single or double crossover in left/right variants.

### Special pieces and crossings

The catalog includes a **90° Cross Track** (2-piece diamond crossing), a **Track End Buffer** (4 studs, bumper stop), and a **Train Decoupler** (Part 2.04.017). No flex track, adapter pieces, or non-90° standalone crossings exist—4DBrix pieces connect directly to standard LEGO track via compatible rail geometry.

---

## How variable geometry reshapes the encoding problem

The non-standard angle-per-radius system has three consequences for genetic algorithm encoding that don't arise with uniform-angle manufacturers.

First, **piece interchangeability is severely constrained**. An R72 curve (15°) cannot substitute for an R104 curve (10°) without changing the total angular budget—a mutation that swaps one radius for another always changes the angular closure equation. With uniform 11.25° systems, swapping radii preserves angles while only affecting lateral displacement.

Second, **closure arithmetic becomes radius-dependent**. A closed loop requires total angular change of exactly 360° (or a multiple). With 4DBrix, achieving 360° requires **16 R40, 20 R56, 24 R72, 32 R88, 36 R104, or 40 R120 curves** (or combinations summing to 360°). Mixed-radius loops must satisfy: Σ(n_i × θ_i) = 360° where each θ_i is different. This is a constrained integer partition problem layered onto the spatial feasibility constraint.

Third, **switch geometry introduces non-catalog arc segments**. The coupling tracks (#025 at 14.4°, #026 at 8.1°, #062 at 2°) are specialized arcs that exist only within switch assemblies. The encoding must decide whether to treat assembled switches as atomic macro-pieces or decompose them into primitive arcs.

---

## Approach 1: port-centric representation

In the port-centric model, each piece is defined by **ports**—connection points with local position (x, y) and heading angle (θ) relative to the piece's reference frame. A straight has two ports at its endpoints. A curve has two ports offset by its arc geometry. A switch has three ports: entry, through, and diverge.

### Piece definition and catalog mapping

Every 4DBrix piece maps to a port configuration computed from its geometric parameters:

```
Curve (R, θ):
  Port 0: (0, 0, 0°)                              — entry
  Port 1: (R·sin(θ), R·(1−cos(θ)), θ)             — exit

Straight (L):
  Port 0: (0, 0, 0°)
  Port 1: (L, 0, 0°)

Switch (R, θ_through, θ_diverge):
  Port 0: (0, 0, 0°)                              — entry
  Port 1: (L_through, 0, 0°)                      — through exit
  Port 2: (R·sin(θ_d), R·(1−cos(θ_d)), θ_d)      — diverging exit
```

Variable-angle curves are naturally accommodated: each radius produces a different Port 1 position and angle. The piece catalog is a lookup table of pre-computed port arrays. Adding new radii means adding rows to this table.

### Connection model and forward kinematics

Layout state is a list of placed pieces with connection records: piece A's Port 1 connects to piece B's Port 0. FK proceeds by **graph traversal**: starting from a root piece, compute each child piece's world transform by aligning the child's connecting port to the parent's matching port. This requires a rotation-translation composition at each step—more expensive than turtle stepping but geometrically exact.

For switches, the three-port structure naturally creates branch points. The traversal pushes unvisited branches onto a stack and processes them depth-first. **This is the port-centric model's primary strength**: multi-port elements are first-class citizens, not awkward extensions to a linear chain.

### pymoo integration

The chromosome is a variable-length list of (piece_type, connection_source, connection_port) tuples stored in a `dtype=object` NumPy array. This forfeits all vectorized NumPy operations—population-level evaluation requires Python loops. Custom `Sampling`, `Mutation`, and `DuplicateElimination` operators must subclass pymoo's base classes.

Mutation operators check **port compatibility** at every step: an ADD operation must verify that the new piece has a port geometrically compatible with the available open port. A MUTATE operation (swapping piece type) must verify all existing connections remain valid. This compatibility checking adds per-mutation overhead but prevents structurally invalid layouts from entering the population.

**Closure detection** requires O(P²) pairwise distance checks across all open ports to find potential loop-closing connections, where P is the count of unconnected ports. For layouts with many open ports (branching topologies), this becomes expensive.

### Assessment

The port-centric model excels at **topological correctness** and **multi-port handling** but pays heavily in computational efficiency. It is the right choice for the **piece catalog definition layer** (how pieces are described internally) even if the chromosome representation uses a different approach.

---

## Approach 2: parametric arc-based representation

The parametric model treats each piece as a **turtle graphics command** that transforms a moving reference frame (x, y, θ). Curves contribute (Δx, Δy, Δθ) computed from their arc parameters; straights contribute (L·cos(θ), L·sin(θ), 0). The layout chromosome is a flat integer array of piece-type indices.

### Why this maps naturally to 4DBrix geometry

Each piece type in the catalog precomputes three values: `dx_local`, `dy_local`, and `dtheta`. For a curve with radius R and arc angle θ at the current heading, the local-frame deltas are:

```
dx_local = R · sin(θ)
dy_local = R · (1 − cos(θ))   [or its negative for left-hand curves]
dtheta   = ±θ
```

4DBrix's variable angles produce different `dtheta` values per radius—but **this is invisible to the FK engine**, which simply indexes into a precomputed table. An R56 curve (dtheta = 18°) and an R104 curve (dtheta = 10°) are just different table rows. The FK loop is identical regardless of how many distinct angles exist in the catalog.

The chromosome is a **1D integer array** of shape `(max_pieces,)` with sentinel values for empty slots:

```python
chromosome = np.array([5, 3, 12, 7, 1, 1, 1, 8, -1, -1, ...], dtype=np.int32)
# catalog[5] = R72 left curve → (dx=18.76, dy=2.46, dtheta=+15°)
# catalog[3] = half straight → (dx=8.0, dy=0.0, dtheta=0°)
```

### Population-level vectorization

This representation achieves **maximum pymoo compatibility**. The population is a 2D array of shape `(pop_size, max_pieces)` using standard integer dtype. Inventory counting is a single `np.apply_along_axis(np.bincount, 1, X)` call. Piece-type lookups use advanced indexing: `catalog_dtheta[X]` returns a `(pop_size, max_pieces)` array of angular deltas.

FK remains **sequential within each individual** (each step depends on the cumulative angle), but the inner loop can be JIT-compiled with Numba or implemented in Cython. Across the population, individuals are independent—embarrassingly parallel via `multiprocessing` or vectorized batch processing.

**Closure evaluation is trivially cheap**: after the FK pass, closure error = `sqrt((x_final − x_0)² + (y_final − y_0)²) + α · |θ_final − θ_0 mod 360°|`. No graph traversal, no port matching—just compare the turtle's final state to its initial state.

### The branching problem

The parametric model's **critical weakness** is multi-port elements. A switch creates a fork that the linear turtle chain cannot naturally represent. Three mitigation strategies exist:

**Strategy A — L-system branch stack**: Insert PUSH and POP pseudo-commands into the sequence. On PUSH, save the turtle state; on POP, restore it. A switch becomes [PUSH, through-route-pieces..., POP, diverge-route-pieces...]. This preserves the linear sequence but adds structural complexity to mutations—every PUSH must have a matching POP, and mutations must maintain this invariant.

**Strategy B — Atomic macro-pieces**: Treat assembled switch configurations (R40 Parallel Track Switch, R40 Double Crossover) as single "macro-pieces" with through-route turtle deltas. The diverging route is implicit metadata, not part of the main FK chain. This is valid when the GA optimizes the **mainline loop** and treats sidings as decoration, but it cannot optimize branch placement.

**Strategy C — Multi-sequence chromosome**: The chromosome contains a primary loop sequence plus indexed sub-sequences for branches. Each branch records its attachment point (index into the main sequence) and diverging piece sequence. Mutations on the main loop are standard; branch mutations operate on sub-sequences. This is **the recommended strategy** as it preserves the efficient integer-array main loop while enabling branch optimization.

### Mutation operators on integer arrays

ADD inserts a random piece type at a random position (O(N) array shift, or O(1) with gap-buffer). MUTATE swaps a piece type index for another valid type—**O(1), the fastest possible mutation**. DELETE removes a position and compacts (O(N) shift). All three are trivially implementable as pymoo custom `Mutation` operators operating on integer arrays.

The **butterfly effect** is the main concern: mutating piece 5 in a 30-piece sequence repositions pieces 6–30 in world space, potentially invalidating collision and boundary constraints for the entire downstream chain. This is inherent to sequential FK and affects convergence—but it also enables exploration of radically different layouts from small mutations, which can be beneficial in early evolutionary stages.

---

## Approach 3: graph-edge representation

The graph model represents the layout as a **connectivity graph** where nodes are connection points and edges are track pieces. Each edge carries attributes (piece_type, arc_angle, radius, length). Graph-theoretic operations—cycle detection, connectivity checking, degree analysis—become native operations rather than derived computations.

### Encoding track pieces as edges

A two-port piece (straight, curve) is a simple edge between two nodes. A three-port switch requires decomposition into a **subgraph**: an internal node with three incident edges (entry-to-internal, internal-to-through, internal-to-diverge). A 90° crossing has four ports and becomes a subgraph with a central node and four edges. Variable-angle curves are just edges with different attribute values—the graph structure is topology-agnostic to geometry.

The chromosome can be encoded as three parallel integer arrays:

```python
edge_src  = np.array([0, 1, 2, 3, 3, ...], dtype=np.int32)
edge_dst  = np.array([1, 2, 3, 4, 5, ...], dtype=np.int32)
edge_type = np.array([5, 3, 10, 11, 12, ...], dtype=np.int32)
```

This preserves some NumPy compatibility for attribute-level operations (edge type counting, attribute lookups) while the topological structure requires graph algorithms.

### Where graphs excel: topology constraints

**Connectivity** is a standard BFS/DFS check—O(V + E), constant-factor fast. **Cycle detection** is equally native, and finding all cycles reveals the loop structure. **Eulerian circuit detection** (does a continuous train path exist that traverses every edge?) reduces to checking that all nodes have even degree—a single `np.bincount` operation on the edge arrays. These are precisely the constraints that are **hardest to evaluate** in the parametric model and merely adequate in the port-centric model.

For complex layouts with multiple interconnected loops (figure-eights, passing loops, wye junctions), the graph model provides **correct-by-construction topology management** that neither alternative offers. Graph grammars—production rules that transform subgraphs (e.g., "replace a straight edge with a passing-loop subgraph")—enable high-level structural mutations that explore topology space efficiently.

### Forward kinematics on graphs

FK requires choosing a **traversal order** (DFS from a root node), then applying piece transforms edge-by-edge. This is functionally identical to the parametric turtle chain but with the overhead of maintaining the traversal. At branch points, the traversal pushes unexplored branches. The result is the same world-space geometry as the port-centric approach, with the same computational cost.

**Closure checking** on a graph is more nuanced: traverse a cycle and verify that the accumulated geometric transform returns to the starting state within tolerance. Multiple cycles require independent closure checks. This is more expensive than the parametric model's single final-state comparison but handles multi-loop layouts that the parametric model cannot.

### pymoo integration challenges

Using `dtype=object` arrays (one graph per individual), the representation loses all vectorized evaluation. Graph mutations—adding edges, removing edges, splitting/merging nodes—are inherently pointer-manipulation operations with **poor cache locality**. For populations of 100+ individuals each containing 30–100 edges, the Python-level overhead of graph operations significantly impacts evaluation throughput.

**Duplicate elimination** is particularly expensive: comparing two layouts for structural equivalence is a **graph isomorphism** problem. While practical for small graphs (< 100 nodes) using canonical labeling or the Weisfeiler-Leman algorithm, it adds per-generation overhead that the other approaches avoid.

The graph model is best suited to a **two-phase optimization** strategy: use graph-level evolution to find promising topologies (loop structures, switch placements), then switch to parametric optimization to refine geometry within each topology.

---

## Head-to-head comparison across critical dimensions

| Dimension | Port-Centric | Parametric/Arc | Graph-Edge |
|-----------|:---:|:---:|:---:|
| Variable-angle curve handling | Good | **Excellent** | Good |
| Switch/multi-port handling | **Excellent** | Poor (needs extensions) | Good |
| NumPy vectorization | Poor | **Excellent** | Poor |
| FK computation speed | Moderate | **Fast** | Moderate |
| Closure evaluation | O(P²) port matching | **O(1)** final state check | O(cycle length) |
| Connectivity checking | Explicit (free) | Implicit for linear (free), expensive for branches | **Native graph op** |
| Mutation operator simplicity | Moderate | **Trivial** (integer ops) | Complex |
| Memory per individual | High (port objects) | **Minimal** (int array) | Moderate (edge arrays) |
| Multi-loop layout support | **Native** | Awkward | **Native** |
| pymoo integration effort | High | **Low** | High |
| Duplicate elimination | Moderate | **Easy** (array comparison) | Hard (isomorphism) |
| Scalability (100+ pieces) | Moderate | **Excellent** | Moderate |

---

## Recommended hybrid architecture

The strongest implementation combines all three models at different architectural layers, exploiting each one's strengths while avoiding its weaknesses.

**Catalog layer (port-centric)**: Define every 4DBrix piece using port geometry. Each piece type stores an array of ports with local (x, y, θ). This is the single source of truth for geometric parameters—FK tables, collision bounds, and connection constraints all derive from port definitions. The port model handles switches and crossovers cleanly because it treats every piece uniformly regardless of port count.

**Chromosome layer (parametric)**: The evolutionary chromosome is a **padded integer array** of shape `(pop_size, max_pieces)` encoding the mainline closed loop as a sequence of piece-type indices. Pre-computed FK lookup tables (indexed by piece type) store `dx_local`, `dy_local`, `dtheta` derived from the port-centric catalog. For switches, use Strategy C (multi-sequence): the chromosome is `[main_loop | branch_0_attach, branch_0_pieces... | branch_1_attach, branch_1_pieces... ]`, with a fixed maximum number of branch slots. This preserves the integer-array property while supporting branching layouts.

**Constraint layer (graph-derived)**: During evaluation, reconstruct a lightweight connectivity graph from the chromosome to perform topology checks—cycle validation, connectivity, Eulerian path detection. This graph is **ephemeral** (built during evaluation, discarded after), so it doesn't burden the chromosome encoding. NetworkX handles this for prototyping; a custom Cython implementation handles production loads.

This layered approach maps to pymoo as follows: the `Problem` subclass accepts integer-array decision variables (`xl=0, xu=catalog_size-1, vtype=int`). Custom `Mutation._do()` operates on the integer population matrix. `Problem._evaluate()` runs the FK chain from precomputed tables, builds the ephemeral graph for topology checks, and computes all objectives and constraints. **Deb's feasibility-first ranking** (pymoo's native `CV` constraint handling) penalizes open loops, disconnected segments, boundary violations, and collisions in strict priority order.

This architecture scales cleanly as 4DBrix releases new radii or switch types: add a row to the port-centric catalog, recompute the FK lookup table, and the evolutionary engine requires zero code changes.

---

## Conclusion

4DBrix's variable arc angles (22.5° at R40 down to 9° at R120) are the library's defining characteristic and its primary encoding challenge. The parametric/arc-based approach handles this most efficiently—variable angles simply become different lookup-table entries consumed by an identical FK engine. However, pure parametric encoding fails at switches, which demand the topological awareness that port-centric and graph models provide. **The optimal architecture layers all three**: port-centric piece definitions feeding parametric integer-array chromosomes, with ephemeral graph construction for topology constraints.

Three novel insights emerged from this analysis. First, 4DBrix's angle choices create a **constrained integer partition** closure problem (Σ n_i × θ_i = 360°) that is more combinatorially rich than uniform-angle systems—this actually benefits evolutionary search by creating more diverse feasible solutions. Second, the R148 Pythagorean-triple geometry (12, 35, 37) × 4 suggests that crossovers should be treated as **atomic macro-pieces** in the GA rather than decomposed into primitive arcs, since their internal geometry is precisely engineered for a specific spatial result. Third, the butterfly-effect sensitivity of sequential FK is both a curse (constraint violations cascade) and an opportunity—coupling early-sequence mutations with a pymoo `Repair` operator that adjusts downstream pieces can convert this sensitivity into an efficient local-search mechanism within the evolutionary loop.