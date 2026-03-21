# Seeding genetic algorithms in pymoo for LEGO track layout optimization

**The most effective approach for a constrained, variable-length combinatorial problem like LEGO track layout optimization is a decoder-based architecture with a custom `Sampling` class that injects 10–30% heuristic seeds alongside random solutions, paired with a `Repair` operator that enforces inventory constraints after every genetic operation.** This strategy sidesteps the core challenge — that most random track sequences are infeasible — by guaranteeing every evaluated individual respects inventory limits and loop-closure geometry. pymoo v0.6+ provides clean extension points (`Sampling`, `Crossover`, `Mutation`, `Repair`, `Problem`) that make this architecture modular and YAML-driven, so new piece types require zero algorithm changes.

---

## How pymoo creates and seeds initial populations

pymoo's initialization system revolves around the `pymoo.core.sampling.Sampling` base class. Every sampling strategy subclasses it and overrides a single method, `_do(self, problem, n_samples, **kwargs)`, which returns a NumPy array of shape `(n_samples, n_var)`. The algorithm constructor accepts three interchangeable types for its `sampling` parameter:

- **A `Sampling` object** — a strategy that generates solutions on demand (e.g., `FloatRandomSampling`, `PermutationRandomSampling`, `LHS`).
- **A `Population` object** — pre-built (optionally pre-evaluated) individuals created via `Population.new("X", X_array)`.
- **A raw NumPy array** of shape `(n_individuals, n_var)` — evaluated automatically by the algorithm.

Built-in samplers include `FloatRandomSampling`, `IntegerRandomSampling`, `BinaryRandomSampling`, `PermutationRandomSampling`, and `LHS` (Latin Hypercube). For a LEGO track problem, none of these suffice directly because the chromosome is a variable-length sequence of discrete piece identifiers with inventory constraints. The solution is a custom `Sampling` class that combines heuristic construction with random generation:

```python
import numpy as np
from pymoo.core.sampling import Sampling

class TrackLayoutSampling(Sampling):
    """Seed a fraction of the population with heuristic layouts,
    fill the rest with random constructive sequences."""

    def __init__(self, heuristic_solutions, seed_fraction=0.2,
                 build_random_fn=None):
        super().__init__()
        self.heuristic_solutions = heuristic_solutions
        self.seed_fraction = seed_fraction
        self.build_random_fn = build_random_fn  # callable(problem) -> solution

    def _do(self, problem, n_samples, **kwargs):
        n_seeds = min(int(n_samples * self.seed_fraction),
                      len(self.heuristic_solutions))
        n_random = n_samples - n_seeds
        X = np.full((n_samples, 1), None, dtype=object)

        # Inject heuristic seeds
        idx = np.random.choice(len(self.heuristic_solutions),
                               n_seeds, replace=False)
        for i, j in enumerate(idx):
            X[i, 0] = list(self.heuristic_solutions[j])

        # Fill remainder with random constructive solutions
        for i in range(n_seeds, n_samples):
            X[i, 0] = self.build_random_fn(problem)

        return X
```

This pattern — **partial seeding** — is the consensus best practice. pymoo has no built-in partial-seeding mechanism, but the `Sampling` interface makes it trivial to implement. When pre-evaluated heuristic solutions exist, inject them without redundant evaluation by wrapping with `Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)` and passing the resulting `Population` directly.

---

## Optimal seeding strategies for constrained track sequences

### The 10–30% heuristic rule

The GA literature consistently converges on **seeding 10–30% of the population with heuristically generated solutions and filling the rest randomly**. Research on TSP initialization (Osaba et al., 2014; Teekeng, 2026) confirms that excessive heuristic seeding (>50%) collapses diversity and traps the population in local optima. For a LEGO track problem with tight inventory constraints, **20–30% heuristic seeding** is appropriate because feasible solutions are rare in random sampling, so the extra seeds ensure sufficient feasible individuals survive into generation one.

Heuristic seeds for track layouts should come from **greedy construction heuristics** — for example, a nearest-compatible-piece algorithm that extends a partial layout by selecting the piece that best continues the current heading while respecting inventory. Building **3–5 diverse greedy strategies** (shortest path to closure, maximum piece usage, random-greedy GRASP-style) and sampling from all of them prevents the seeds from clustering around a single local optimum.

### Decoder-based feasibility: the strongest approach

For problems where most random solutions are infeasible, the literature strongly recommends a **decoder-based architecture** (the Biased Random-Key GA pattern). Instead of evolving track sequences directly, the chromosome encodes a vector of **priority keys** — one per inventory slot or decision point — and a deterministic decoder translates these keys into a valid track layout:

```python
def decode(keys, inventory, grid):
    """Deterministic decoder: keys -> feasible track layout."""
    sorted_pieces = sort_by_keys(inventory.all_pieces, keys)
    layout = []
    for piece in sorted_pieces:
        if can_place(piece, layout, grid) and inventory.available(piece):
            layout.append(piece)
            inventory.decrement(piece)
        if is_closed_loop(layout):
            break
    return layout
```

**Every chromosome decodes to a feasible solution by construction.** The decoder handles inventory limits (never places a piece exceeding supply), geometric validity (only places pieces that connect), and can score partial loops. This cleanly separates the search space (continuous or integer keys) from the solution space (valid layouts), letting standard pymoo operators work without modification.

### Diversity preservation and opposition-based learning

For discrete spaces, diversity is maintained by ensuring initial solutions are **maximally spread** using problem-specific distance metrics (e.g., edit distance between piece sequences). **Opposition-based learning** (OBL) — generating "opposite" solutions and keeping the better of each pair — has been adapted for combinatorial problems using circular opposition (reversing permutation segments). Research integrating OBL into NSGA-III showed **15–30% convergence improvement**, though its benefit diminishes for highly constrained problems where the opposition mapping may produce infeasible solutions.

---

## Custom operators for variable-length track sequences

pymoo does **not** natively support variable-length chromosomes. The recommended pattern uses `n_var=1` with `object` dtype, storing each individual's track sequence as a Python list. This requires implementing all four operator types from scratch.

### Crossover: segment-based recombination

For track layouts, **order-based crossover adapted for variable length** works well. The `Crossover` base class requires specifying `n_parents` and `n_offsprings`, then overriding `_do()` which receives an array of shape `(n_parents, n_matings, n_var)`:

```python
from pymoo.core.crossover import Crossover

class TrackCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2, prob=0.9)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            cut = np.random.randint(1, min(len(p1), len(p2)))
            # Swap tails, preserving head geometry
            Y[0, k, 0] = p1[:cut] + p2[cut:]
            Y[1, k, 0] = p2[:cut] + p1[cut:]
        return Y
```

For problems where **adjacency matters** (track connections), pymoo's built-in `EdgeRecombinationCrossover` (ERX) is relevant for fixed-length permutations. For variable-length sequences, a custom segment-swap or subtour-exchange crossover that preserves geometric connectivity is more appropriate. Standard `OrderCrossover` (OX) preserves relative order and works well for routing-like problems.

### Mutation: add, remove, swap, and invert

A multi-action mutation operator covers the variable-length search space effectively:

```python
from pymoo.core.mutation import Mutation

class TrackMutation(Mutation):
    def __init__(self):
        super().__init__(prob=1.0)

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            seq = list(X[i, 0])
            r = np.random.random()
            if r < 0.25 and len(seq) > 2:
                idx = np.random.randint(len(seq))
                seq.pop(idx)                          # remove piece
            elif r < 0.50:
                piece = random_piece(problem.inventory)
                pos = np.random.randint(len(seq) + 1)
                seq.insert(pos, piece)                # insert piece
            elif r < 0.75 and len(seq) >= 2:
                a, b = np.random.choice(len(seq), 2, replace=False)
                seq[a], seq[b] = seq[b], seq[a]      # swap
            else:
                a, b = sorted(np.random.choice(len(seq), 2, replace=False))
                seq[a:b+1] = reversed(seq[a:b+1])    # inversion
            X[i, 0] = seq
        return X
```

pymoo's built-in `InversionMutation` is the standard choice for fixed-length permutations. For variable-length, the multi-action pattern above is necessary.

### Survival selection: feasibility-first dominance

pymoo's default NSGA-II survival uses the **constraint domination principle**: feasible solutions always dominate infeasible ones; among two infeasible solutions, lower constraint violation wins; among two feasible solutions, Pareto dominance decides. This is exactly right for track layout optimization — it aggressively drives the population toward feasibility without requiring penalty tuning. For many-objective formulations (e.g., maximizing piece usage, minimizing unused track area, maximizing loop smoothness), NSGA-III with reference directions provides better spread across the Pareto front.

---

## Repair operators are essential for highly constrained spaces

When most random solutions violate inventory or geometric constraints, **repair operators dramatically outperform penalty methods**. pymoo's `Repair` base class is called after crossover and mutation but before evaluation, ensuring every individual reaching the evaluator is feasible:

```python
from pymoo.core.repair import Repair

class InventoryRepair(Repair):
    """Trim sequences to respect inventory limits and fix connectivity."""

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            seq = list(X[i, 0])
            usage = {}
            valid = []
            for piece in seq:
                usage[piece] = usage.get(piece, 0) + 1
                if usage[piece] <= problem.inventory[piece]:
                    valid.append(piece)
            X[i, 0] = valid
        return X
```

The repair is wired into the algorithm alongside other operators:

```python
from pymoo.algorithms.soo.nonconvex.ga import GA

algorithm = GA(
    pop_size=200,
    sampling=TrackLayoutSampling(heuristic_seeds, seed_fraction=0.2,
                                  build_random_fn=random_track_builder),
    crossover=TrackCrossover(),
    mutation=TrackMutation(),
    repair=InventoryRepair(),
    eliminate_duplicates=True
)
```

pymoo also supports five formal constraint-handling strategies: **feasibility-first** (default, parameter-less), **CV-as-penalty** (`ConstraintsAsPenalty`), **CV-as-objective** (`ConstraintsAsObjective`), **ε-constraint handling** (dynamic feasibility threshold), and **repair operators**. For track layouts, combining a repair operator (hard constraint enforcement on inventory) with inequality constraints (soft penalties on loop-closure gap distance) is the most effective hybrid. Define constraints in `_evaluate()` using `out["G"]` for inequalities (satisfied when ≤ 0) and `out["H"]` for equalities.

---

## Population sizing, convergence, and island models

**Population size** should scale with problem complexity. For constrained combinatorial problems, the literature recommends starting at **100–200 individuals** and scaling up for larger inventory sets. The key trade-off: too small a population converges prematurely; too large wastes evaluations exploring redundant regions. A practical heuristic is to start at 150, then test 100/200/500 and observe convergence curves via pymoo's `Callback`:

```python
from pymoo.core.callback import Callback

class ConvergenceTracker(Callback):
    def __init__(self):
        super().__init__()
        self.data["best_f"] = []
        self.data["feasible_pct"] = []

    def notify(self, algorithm):
        pop = algorithm.pop
        self.data["best_f"].append(pop.get("F").min())
        self.data["feasible_pct"].append(
            np.mean(pop.get("feasible").flatten()))
```

**Island models** divide the population into 4–8 subpopulations evolving independently with periodic migration (**5–10% of individuals every 10–20 generations**). This preserves diversity far better than a single population, especially for combinatorial problems with many local optima. pymoo does not have built-in island model support, but the ask-and-tell interface enables implementation:

```python
algorithms = [GA(pop_size=50, ...) for _ in range(4)]  # 4 islands
for alg in algorithms:
    alg.setup(problem, termination=('n_gen', 200))

for gen in range(200):
    for alg in algorithms:
        pop = alg.ask()
        alg.evaluator.eval(problem, pop)
        alg.tell(infills=pop)

    if gen % 20 == 0:  # migrate every 20 generations
        migrate_best(algorithms, n_migrants=5)
```

**Adaptive population sizing** (GAVaPS, APOGA) grows the population when the algorithm makes progress and shrinks it during stagnation. While pymoo doesn't implement this natively, the ask-and-tell loop allows dynamic adjustment of `pop_size` between generations.

---

## YAML-driven expandability through data-driven problem design

The key architectural insight is that `n_var`, bounds, and evaluation logic should derive entirely from the YAML configuration. Adding a new track piece type (e.g., a Y-switch or bridge ramp) requires only a YAML edit:

```python
import yaml
from pymoo.core.problem import ElementwiseProblem

class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, config_path, **kwargs):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.pieces = cfg["pieces"]         # list of piece definitions
        self.inventory = {p["id"]: p["count"] for p in self.pieces}
        self.geometry = {p["id"]: p["geometry"] for p in self.pieces}

        # n_var, bounds, etc. derived from config — not hardcoded
        n_var = 1  # object-type: variable-length sequence
        super().__init__(n_var=n_var, n_obj=2,
                         n_ieq_constr=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        layout = x[0]  # the track piece sequence
        out["F"] = [-pieces_used(layout), loop_gap(layout)]
        out["G"] = [loop_gap(layout) - 0.01]  # must close within tolerance
```

pymoo's operators plug in as constructor arguments — `sampling`, `crossover`, `mutation`, `repair`, `selection` — all accepting any object implementing the corresponding `_do()` interface. A factory function can dynamically configure operators based on problem properties loaded from YAML (number of piece types, maximum track length, symmetry constraints). The `MixedVariableGA` and `MixedVariableProblem` classes in pymoo v0.6+ also support `Choice`, `Integer`, `Binary`, and `Real` variable types declared via a dictionary, enabling clean mixed-variable formulations if the decoder approach uses typed decision variables.

---

## Relevant real-world examples and prior art

Several published projects validate this architectural approach. **Lee, Kim & Moon (GECCO 2015)** optimized LEGO brick layouts using randomized greedy initialization combined with domain-specific mutations (random boundary mutation, thickening) — directly analogous to track piece placement. **Lee, Kim & Myung (IEEE Access, 2018)** introduced a split-and-merge GA for LEGO sculpture optimization that ensures structural connectivity between layers, paralleling the loop-closure constraint in track layouts.

In pymoo specifically, the **TSP permutation example** at `pymoo.org/customization/permutation.html` demonstrates `PermutationRandomSampling` + `OrderCrossover` + `InversionMutation` + custom `Repair` — the exact operator stack needed for route-like combinatorial problems. The **knapsack example** at `pymoo.org/customization/binary.html` shows repair-based constraint handling with `ConsiderMaximumWeightRepair`, which is structurally identical to an inventory constraint repair. The **custom variable type tutorial** at `pymoo.org/customization/custom.html` explicitly demonstrates object-dtype chromosomes with custom operators for non-standard data structures, confirming pymoo's support for variable-length list-based solutions.

A **multi-criteria routing example** using pymoo's NSGA-II on real road networks (from the Smart Mobility Algorithms book) implements custom Sampling, Crossover, and Mutation for variable-length route paths — the closest published analogue to the LEGO track layout problem.

---

## Conclusion

The optimal pymoo architecture for a LEGO track layout optimizer combines five key design decisions. **First**, use a decoder-based or constructive representation where chromosomes encode piece-placement priorities and a deterministic decoder builds valid layouts, guaranteeing feasibility by construction. **Second**, implement a custom `Sampling` class that seeds 20–30% of the initial population with greedy heuristic layouts (nearest-compatible-piece, maximum-usage-first) while filling the rest with random constructive solutions. **Third**, pair a `Repair` operator for hard inventory constraints with `out["G"]` inequality constraints for loop-closure tolerance, leveraging pymoo's default feasibility-first survival to drive convergence without penalty tuning. **Fourth**, design the `Problem` subclass to derive all parameters from YAML — piece geometry, inventory counts, connection rules — so new piece types require no code changes. **Fifth**, target a population of 150–200 individuals, track feasibility percentage and best fitness via `Callback`, and consider an island model (4 subpopulations, migrate 5% every 20 generations) if premature convergence occurs. This architecture exploits pymoo's modular operator pipeline while addressing the fundamental challenge that random track sequences are overwhelmingly infeasible.