# Designing the `problem/` package for a LEGO track optimizer

**The `TrackLayoutProblem` class should subclass `ElementwiseProblem`, use pymoo v0.6+'s split constraint API (`n_ieq_constr`/`n_eq_constr`), and implement a thin four-stage pipeline — decode, physics, objectives, constraints — with normalized constraint values and negated objectives.** This design leverages pymoo's automatic Constraint Domination Principle implementation, keeps the class free of domain logic, and meets the <1 ms evaluation budget through pre-computation and NumPy vectorization within each single-solution call. The theoretical basis is strong: Zheng & Doerr (2024) proved NSGA-II fails for ≥3 objectives in sub-exponential time, rigorously justifying the two-objective formulation, while Deb et al.'s (2002) parameter-free CDP eliminates the need for penalty tuning entirely.

---

## pymoo v0.6 API and the ElementwiseProblem contract

pymoo's `ElementwiseProblem` receives a **single solution** (`x` as 1D array of length `n_var`) per `_evaluate` call. Internally, pymoo wraps this in a `LoopedElementwiseEvaluation` that simply iterates `[f(x) for x in X]` across the population. The `__init__` constructor signature for v0.6+ is:

```python
super().__init__(
    n_var=int,           # total chromosome length
    n_obj=int,           # number of objectives (2 for this system)
    n_ieq_constr=int,    # inequality constraints (g ≤ 0)
    n_eq_constr=int,     # equality constraints  (h = 0)
    xl=np.ndarray,       # lower bounds per variable
    xu=np.ndarray,       # upper bounds per variable
    vtype=int,           # advisory type hint
    **kwargs             # elementwise_runner for parallelization
)
```

The `_evaluate` method writes to an output dictionary with three keys: **`out["F"]`** for objectives (shape `(n_obj,)`), **`out["G"]`** for inequality constraints (shape `(n_ieq_constr,)`), and **`out["H"]`** for equality constraints (shape `(n_eq_constr,)`). The critical v0.5→v0.6 breaking change replaced the single `n_constr` parameter with `n_ieq_constr` and `n_eq_constr`. Other migration points include module paths changing from `pymoo.model.*` to `pymoo.core.*`, factory methods being deprecated in favor of direct imports, and the parallelization interface shifting from tuple-based (`parallelization=("starmap", pool.starmap)`) to runner-based (`elementwise_runner=StarmapParallelization(pool.starmap)`).

pymoo computes constraint violation (CV) automatically per solution:

```
CV = Σ max(0, gᵢ) + Σ max(0, |hₖ| − ε)
```

A solution is **feasible** when CV = 0. This CV feeds directly into NSGA-II's crowded-comparison operator via the Constraint Domination Principle — no user configuration required.

---

## Why ElementwiseProblem beats vectorized Problem here

The decoder is inherently sequential: it constructs track geometry piece-by-piece, maintaining cumulative position and heading. This cannot be vectorized across the population. Two implementation strategies exist:

**ElementwiseProblem** (recommended): Each `_evaluate` call runs the full decode→physics→constraints→objectives pipeline for one chromosome. pymoo handles the population loop. The per-call overhead is ~1–5 μs (one dict allocation, one function-call wrapper), negligible against the evaluation budget. Parallelization is trivially added via `elementwise_runner=StarmapParallelization(pool.starmap)`.

**Vectorized Problem** (alternative): The `_evaluate` receives the full `(N, n_var)` matrix. The decoder must still loop internally (`for i in range(N): layout = decode(X[i])`), but physics and constraint evaluation could be batched into NumPy operations across all N layouts simultaneously. This pattern is worth considering only if profiling reveals the Python loop overhead dominates — unlikely given that decoder computation will dominate.

**Recommendation: start with `ElementwiseProblem`** for simplicity and correctness. The sequential decoder makes full vectorization impossible anyway, and pymoo's `LoopedElementwiseEvaluation` adds negligible overhead. A community benchmark (pymoo GitHub issue #610) confirmed that single-core `LoopedElementwiseEvaluation` outperforms parallelized runners for cheap evaluations, because IPC and pickling overhead exceeds computation time. Only switch to multiprocessing if per-evaluation cost grows beyond ~10 ms.

---

## Constraint Domination Principle: parameter-free constraint handling

Deb et al. (2002) defined the CDP with three rules that pymoo implements automatically:

1. **Both feasible** → standard Pareto dominance on objectives
2. **One feasible, one infeasible** → feasible solution always wins
3. **Both infeasible** → smaller CV wins; objectives are ignored entirely

This "feasibility-first" approach requires **zero penalty parameters**, avoiding the most problematic aspect of penalty methods identified by Coello Coello (2002). Penalty methods suffer when the penalty is too high (search trapped inside the feasible region, unable to reach boundary optima) or too low (excessive exploration of infeasible space). The CDP sidesteps this by never numerically comparing constraints with objectives — feasibility is a categorical property. Coello Coello's taxonomy classifies four approaches (penalty functions, decoders/repair, feasibility-first, hybrids), and the CDP is preferred for NSGA-II because it maps naturally onto binary tournament selection and adds no computational overhead to non-dominated sorting.

For this system, the angular closure constraint deserves special treatment. As a strict equality (`Σθ = 360°`), the feasible set is measure-zero in continuous space, making it extremely hard for evolutionary search. pymoo's own documentation warns that "most algorithms will not handle equality constraints efficiently." Two options:

- **pymoo native equality** via `out["H"]`: pymoo converts to `|h(x)| − ε ≤ 0` internally
- **Inequality pair** (recommended): `g = |Σθ − 360| / 360 − ε ≤ 0`, which is a single normalized inequality constraint that enlarges the feasible region to a band of width ±ε·360° around exact closure

**Use the inequality form** with `ε = 0.01`, corresponding to ±3.6° tolerance. Standard LEGO curves are **22.5°** each (16 form a full circle), so ±3.6° is roughly 1/6 of a curve piece — physically meaningful while permitting evolutionary search to converge.

---

## Constraint normalization prevents CV distortion

Without normalization, constraint violations on wildly different scales — degrees (0–360), collision counts (0–N²), boundary distance in studs (0–diagonal) — cause the CV aggregation to be dominated by whichever constraint has the largest raw scale. pymoo's documentation explicitly recommends normalizing constraints "to make them operating on the same scale."

The recommended normalization divides each constraint by its maximum possible violation, placing all constraints in the [0, 1] range when violated:

| Constraint | Raw formula | Normalization | Feasible when |
|---|---|---|---|
| Angular closure | \|Σθ − 360\| | ÷ 360 | ≤ ε (0.01) |
| Boundary | max(point violations) | ÷ layout diagonal | ≤ 0 |
| Inventory (per-type) | used_i − available_i | ÷ available_i | ≤ 0 |
| Collision | collision_count | ÷ n_segments | ≤ 0 |
| Port connectivity | unconnected_ports | ÷ total_ports | ≤ 0 |

This ensures that a 50% violation of *any* constraint contributes equally to CV, preventing degenerate ranking behavior where the optimizer satisfies easy large-scale constraints while ignoring hard small-scale ones. The collision normalization uses `n_segments` rather than the theoretical maximum `n·(n−1)/2` because the latter inflates the denominator, making even many collisions appear trivially small.

---

## Two-objective formulation with negated maximization

pymoo minimizes all objectives. Both targets — piece utilization and space-mean safe speed — are maximization goals, so they must be negated:

```python
f1 = -(pieces_used / pieces_available)      # utilization: range [-1, 0]
f2 = -(Σ Lₖ / Σ(Lₖ / v_safe_k))           # space-mean speed (negated)
```

The space-mean speed is the **length-weighted harmonic mean** of safe speeds, the standard metric in traffic flow analysis (Wardrop, 1952; FHWA Travel Time Handbook). It penalizes slow segments more aggressively than an arithmetic mean — a single tight curve drastically reduces the overall score — and is physically meaningful as the average speed a train achieves traversing the entire layout.

**These objectives genuinely conflict.** Using more pieces (especially curves) creates denser layouts with tighter turns, reducing safe speed. Conversely, high-speed layouts favor long straights with gentle curves, leaving pieces unused. The Pareto front is expected to exhibit an **L-shaped** topology: a cooperative region (adding straights improves both utilization and speed) transitioning to a conflicting region (adding curves to increase utilization degrades speed). The "knee" of this front — where the marginal cost of utilization in speed terms is minimized — typically represents the best engineering compromise.

An alternative rarity-weighted utilization `f1 = −Σ(wᵢ · nᵢ_used / nᵢ_avail) / Σwᵢ` where `wᵢ = 1/nᵢ_avail` encourages using scarce pieces preferentially. An alternative `f2 = −min(v_safe_k)` focuses on the bottleneck segment but discards distribution information. The harmonic mean is recommended as the primary formulation.

---

## Runtime theory justifies two objectives and sizes the population

Zheng & Doerr (2024, *Artificial Intelligence* 325:104016) provided the first rigorous runtime analysis of NSGA-II. Their key results directly inform this system's configuration:

**Population size ≥ 4× Pareto front size.** With this sizing, NSGA-II achieves O(Nn log n) evaluations on standard benchmarks — matching the asymptotic guarantees of simpler MOEAs like SEMO. Below this threshold, NSGA-II provably fails: with N equal to the Pareto front size, the population permanently misses a constant fraction of the front, even after exponentially many iterations. For the track optimizer, if the expected Pareto front contains ~25–50 solutions, the population should be **≥100–200**.

**NSGA-II is provably inefficient for ≥3 objectives.** Zheng & Doerr (2024, *IEEE TEVC* 28:1442–1454) proved that NSGA-II cannot compute full Pareto fronts in sub-exponential time when m ≥ 3. The root cause is crowding distance: for two objectives, sorting by one objective automatically gives inverse sorting by the other, so crowding distance correctly identifies extreme solutions. For ≥3 objectives, sortings along different objectives are uncorrelated, and crowding distance systematically loses Pareto-optimal solutions. This rigorously justifies restricting the system to **exactly two objectives** and using NSGA-III or SMS-EMOA only if a third objective becomes unavoidable.

---

## Complete TrackLayoutProblem implementation skeleton

```python
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class TrackLayoutProblem(ElementwiseProblem):
    """Thin orchestration: chromosome → decoder → physics → constraints → objectives.
    
    No domain logic lives here. All computation is delegated to decoder, physics,
    and constraint modules. This class is the bridge between pymoo and domain code.
    """
    
    def __init__(self, catalog, segment_map, physics_config, constraints_config,
                 **kwargs):
        self.catalog = catalog
        self.segment_map = segment_map
        self.physics_config = physics_config
        self.constraints_config = constraints_config
        
        # Derive chromosome length from SegmentMap
        n_var = segment_map.total_length
        
        # Pre-compute static lookup tables for <1ms evaluation
        self._piece_angles = np.array([p.angle for p in catalog], dtype=np.float64)
        self._piece_lengths = np.array([p.length for p in catalog], dtype=np.float64)
        self._available_counts = np.array(
            [catalog.available(i) for i in range(len(catalog))], dtype=np.int32
        )
        self._layout_diagonal = constraints_config.get("layout_diagonal", 1.0)
        self._epsilon_angular = constraints_config.get("epsilon_angular", 0.01)
        
        # Bounds: each gene in [0, catalog_size - 1] (integer index)
        n_catalog = len(catalog)
        xl = np.zeros(n_var, dtype=np.int16)
        xu = np.full(n_var, n_catalog - 1, dtype=np.int16)
        
        # Constraint counts (all inequality — angular closure as normalized |error|)
        n_piece_types = len(catalog)
        # 1 angular + 1 boundary + n_piece_types inventory + 1 collision + 1 connectivity
        n_ieq = 1 + 1 + n_piece_types + 1 + 1
        
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=n_ieq,
            n_eq_constr=0,         # angular closure handled as normalized inequality
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs               # accepts elementwise_runner for parallelization
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Phase 1: Decode chromosome → track layout
            layout = decode(x.astype(np.int16), self.catalog, self.segment_map)
            
            # Phase 2: Physics evaluation → safe speeds per segment
            physics = evaluate_physics(layout, self.physics_config)
            
            # Phase 3: Objectives (negated for pymoo minimization)
            pieces_used = layout.piece_count
            pieces_available = self._available_counts.sum()
            utilization = pieces_used / max(pieces_available, 1)
            
            lengths = physics.segment_lengths        # shape (n_segments,)
            safe_speeds = physics.safe_speeds         # shape (n_segments,)
            total_length = lengths.sum()
            travel_time = (lengths / np.maximum(safe_speeds, 1e-6)).sum()
            space_mean_speed = total_length / max(travel_time, 1e-9)
            
            out["F"] = np.array([-utilization, -space_mean_speed])
            
            # Phase 4: Constraints (g ≤ 0 is feasible, normalized to ~[0,1])
            G = np.empty(self.n_ieq_constr, dtype=np.float64)
            idx = 0
            
            # Angular closure: |Σθ - 360| / 360 - ε
            G[idx] = abs(layout.angle_sum - 360.0) / 360.0 - self._epsilon_angular
            idx += 1
            
            # Boundary: max violation / diagonal
            G[idx] = layout.max_boundary_violation / self._layout_diagonal
            idx += 1
            
            # Piece inventory: (used_i - available_i) / available_i per type
            used_per_type = layout.used_per_type      # shape (n_piece_types,)
            G[idx:idx+len(self._available_counts)] = (
                (used_per_type - self._available_counts) /
                np.maximum(self._available_counts, 1)
            )
            idx += len(self._available_counts)
            
            # Collision: collision_count / n_segments
            n_seg = max(layout.n_segments, 1)
            G[idx] = layout.collision_count / n_seg
            idx += 1
            
            # Port connectivity: unconnected_ports / total_ports
            G[idx] = layout.unconnected_ports / max(layout.total_ports, 1)
            
            out["G"] = G
            
        except Exception:
            # Penalty: worst objectives + maximum constraint violation
            out["F"] = np.array([0.0, 0.0])                     # 0 utilization, 0 speed
            out["G"] = np.full(self.n_ieq_constr, 1e6)          # all heavily violated
```

Key design decisions in this skeleton:

- **Pre-computation in `__init__`** — piece angles, lengths, and available counts are computed once and stored as NumPy arrays. The layout diagonal and angular tolerance come from configuration.
- **NumPy vectorization within `_evaluate`** — inventory constraints are computed as a single vectorized operation across all piece types rather than a Python loop.
- **Exception guard** — the `try/except` assigns worst-case objectives (0 utilization, 0 speed after negation = 0.0) and high positive constraint violations (1e6 >> any normalized constraint). Under CDP, these solutions are always ranked last among infeasible solutions.
- **Angular closure as inequality** — the single normalized expression `|Σθ − 360| / 360 − ε` avoids pymoo's documented weakness with strict equality constraints.

---

## Operators access problem attributes through the `problem` parameter

All pymoo operator `_do` methods receive `problem` as their first parameter. This is the standard, documented mechanism for operators to access problem-specific data:

```python
class SegmentAwareCrossover(Crossover):
    def _do(self, problem, X, **kwargs):
        # Access SegmentMap for crossover point selection
        segment_map = problem.segment_map
        catalog = problem.catalog
        # ... segment-respecting crossover logic ...
        return Y
```

This pattern is confirmed in pymoo's official custom variable type example, where `problem.n_characters` and `problem.ALPHABET` are accessed from crossover and mutation operators. Custom attributes stored as `self.xxx` in the problem's `__init__` are directly accessible. The algorithm also stores a reference via `algorithm.problem` after `setup()`, but operators should use the `problem` parameter passed to `_do`.

For integer handling in v0.6+, the recommended pattern combines `vtype=int` (advisory) with `RoundingRepair` on operators and `IntegerRandomSampling`:

```python
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

algorithm = NSGA2(
    pop_size=200,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, repair=RoundingRepair()),
    mutation=PM(eta=3.0, repair=RoundingRepair()),
    eliminate_duplicates=True,
)
```

---

## Convergence monitoring via callbacks and hypervolume

pymoo's `Callback` class provides per-generation hooks. For a bi-objective constrained problem, three metrics matter most:

```python
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume

class TrackOptCallback(Callback):
    def __init__(self, hv_ref_point):
        super().__init__()
        self.data["hv"] = []
        self.data["feasible_frac"] = []
        self.data["best_f1"] = []
        self.data["best_f2"] = []
    
    def notify(self, algorithm):
        pop = algorithm.pop
        cv = pop.get("CV").flatten()
        feasible_mask = cv <= 0
        self.data["feasible_frac"].append(feasible_mask.mean())
        
        F = pop.get("F")
        if feasible_mask.any():
            F_feas = F[feasible_mask]
            hv = Hypervolume(ref_point=self.hv_ref_point).do(F_feas)
            self.data["hv"].append(hv)
            self.data["best_f1"].append(F_feas[:, 0].min())
            self.data["best_f2"].append(F_feas[:, 1].min())
        else:
            self.data["hv"].append(0.0)
            self.data["best_f1"].append(np.nan)
            self.data["best_f2"].append(np.nan)
```

The **hypervolume reference point** should be set to slightly worse than the worst expected objective values — for negated objectives in [-1, 0] and [-v_max, 0], a reference point of `[0.1, 0.1]` ensures all feasible solutions contribute positive hypervolume. The **feasible fraction** tracks when the population transitions from mostly infeasible (early generations) to mostly feasible — a critical diagnostic for constraint calibration. If the feasible fraction stays near zero, constraints are too tight or the representation makes feasible solutions too rare.

An important gotcha: access callback data via `res.algorithm.callback.data`, not the original callback object, because `minimize()` deep-copies everything.

---

## Conclusion

The `problem/` package design rests on three citation-backed pillars. First, **pymoo's ElementwiseProblem** provides the simplest correct implementation path — the sequential decoder makes full vectorization impossible, and the per-call overhead (~1–5 μs) is negligible against the evaluation budget. Second, **Deb's CDP** (automatically implemented by pymoo) eliminates penalty parameter tuning, with normalized constraints ensuring fair CV aggregation across heterogeneous constraint scales. Third, **Zheng & Doerr's runtime proofs** mandate exactly two objectives (NSGA-II is provably inefficient for ≥3) and population size ≥ 4× Pareto front size.

The most consequential implementation decision is treating angular closure as a **normalized inequality** rather than a strict equality, since pymoo's own documentation acknowledges poor equality constraint handling in evolutionary algorithms. The `ε = 0.01` tolerance (±3.6°) balances physical fidelity with evolutionary searchability. The thin-orchestration pattern — where `TrackLayoutProblem._evaluate` is five lines of delegation plus constraint assembly — ensures the problem class remains a pure pipeline coordinator that can be tested, profiled, and modified independently of domain logic.