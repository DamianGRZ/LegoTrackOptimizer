Imple# Implementation Details

This document describes the implementation architecture of the LEGO Track Optimizer: how visualization, logging, and pymoo optimization work.

---

## 1. Visualization (High Level)

### Core Functions (`src/visualization.py`)

| Function | Purpose |
|----------|---------|
| `plot_layout()` | Simple line-based rendering of Layout objects |
| `plot_layout_geometry()` | Phase 1+ rendering with branches, switches|
| `create_results_figure()` | 2x2 grid showing top 4 layouts |
| `plot_pareto_front()` | 2D scatter plot of objectives (F[0], F[1]) |

### Color Coding

- **Blue**: Curves (R40_LEFT, R40_RIGHT)
- **Black**: Straights (STRAIGHT_16, STRAIGHT_24)
- **Red**: Switches (R40_SWITCH_*)
- **Green**: Crossings (CROSS_90)

### Rendering Approach

1. Extract piece states (x, y, theta) from FK chain
2. Draw line segments between consecutive positions
3. Color by piece type using catalog lookup
4. Add boundary rectangle if configured
5. Mark closure point (start/end overlap)

### Output Files

```
outputs/
  layout_1.png       # Best by utilization
  layout_2.png       # 2nd best
  layout_3.png       # 3rd best
  layout_4.png       # 4th best
  pareto_front.png   # 3D Pareto front
  summary.png        # Combined figure
```

---

## 2. Logging

### Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('track_optimizer.log')
    ]
)
```

### Log Levels

| Level | Usage |
|-------|-------|
| INFO | Generation progress, best solutions, final results |
| WARNING | Stagnation detected, no feasible solutions |
| ERROR | Configuration errors, file I/O failures |
| DEBUG | Detailed FK computation, constraint values |

### Progress Callback (`OptimizationProgressCallback`)

Logs every 10 generations (configurable):

```
2024-01-15 10:30:00 - INFO - Generation 10: Best F[0]=-0.85, F[1]=-1.20, Feasible: 450/1000
2024-01-15 10:30:15 - INFO - Generation 20: Best F[0]=-0.92, F[1]=-1.25, Feasible: 650/1000
...
```

### Output File Structure

| File | Format | Content |
|------|--------|---------|
| `chromosome_N.txt` | Human-readable | Raw sequence, piece breakdown |
| `chromosome_N.json` | Machine-readable | Full objectives, geometry, constraints |
| `track_optimizer.log` | Text | Full session log |
| `results.npz` | NumPy | Compressed arrays for post-processing |

### JSON Chromosome Structure

```json
{
  "raw_array": [2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, ...],
  "valid_indices": [2, 2, 2, 2, 0, 0, ...],
  "piece_breakdown": {
    "STRAIGHT_16": 4,
    "R40_LEFT": 12
  },
  "objectives": {
    "utilization": 0.92,
    "avg_speed": 1.15,
    "bounding_box_area": 12500
  },
  "geometry": {
    "closure_error": 0.1,
    "angle_error": 0.5,
    "bounding_box": [-50, -30, 80, 60]
  }
}
```

---

## 3. Pymoo Optimization Flow (Steps)

### Step 1: Load Configuration and Catalog

```python
config = OptimizationConfig.from_yaml('configs/default.yaml')
catalog = TrackCatalog.from_yaml('data/track_pieces.yaml')
```

- Parse YAML configuration (inventory, boundary, algorithm params)
- Build FK lookup tables from track piece definitions
- Validate piece indices match between config and catalog

### Step 2: Create TrackLayoutProblem

```python
problem = TrackLayoutProblem(
    catalog=catalog,
    config=config,
)
# Internally sets:
#   n_var = total_inventory + 2  # Piece indices + start position
#   n_obj = 2                    # Objectives: utilization, speed
#   n_ieq_constr = 5             # Constraints: closure, angle, boundary, inventory, orphan switches
```

The chromosome is a flat array:
- `x[0:n_piece_vars]`: Piece indices (-1 for empty, 0-9 for piece types)
- `x[n_piece_vars]`: start_x (starting X position)
- `x[n_piece_vars+1]`: start_y (starting Y position)

### Step 3: Initialize NSGA-II with HeuristicSampling

```python
algorithm = NSGA2(
    pop_size=config.algorithm.pop_size,
    sampling=HeuristicSampling(catalog, config),
    crossover=SBX(prob=config.algorithm.crossover_prob, eta=config.algorithm.crossover_eta),
    mutation=PM(prob=config.algorithm.mutation_prob, eta=config.algorithm.mutation_eta),
    repair=TrackRepair(catalog, config.inventory, problem.n_piece_vars),
    eliminate_duplicates=True
)
```

**HeuristicSampling (51% seeded patterns)**:
- Simple circles: 16 R40 curves = 360 deg
- Symmetric ovals: Curves + straights on 2 axes
- Racetracks: 4 corners with straight segments
- Validates patterns against available inventory

**Standard pymoo operators** (SBX crossover, PM mutation) are used with TrackRepair to fix inventory violations.

### Step 4: Evaluation Loop (Per Generation)

For each individual in population:

```python
def _evaluate(self, x, out, *args, **kwargs):
    # 4a. Extract piece indices and starting position from flat array
    pieces = np.round(x[:self.n_piece_vars]).astype(np.int32)
    start_x = float(x[self.n_piece_vars])
    start_y = float(x[self.n_piece_vars + 1])

    # 4b. Build layout from piece indices
    layout = build_layout(pieces, self.catalog)

    # 4c. Translate layout to starting position
    layout.states[:, 0] += start_x
    layout.states[:, 1] += start_y

    # 4d. Compute speed profile (3-pass algorithm with derailing prevention)
    speed_profile = compute_speed_profile(layout, self.catalog, self.config.physics)

    # 4e. Compute objectives F[] (2 objectives, both minimized)
    out["F"] = [
        -utilization,              # F[0]: Maximize pieces used
        -speed_profile.avg_speed   # F[1]: Maximize average speed
    ]

    # 4f. Compute constraints G[] (g <= 0 feasible)
    out["G"] = [
        (layout.closure_error - closure_tol) / closure_tol,   # G[0]
        (layout.angle_error - angle_tol) / angle_tol,         # G[1]
        boundary_violation / diagonal,                         # G[2]
        inventory_excess,                                      # G[3]
        orphan_switch_count                                    # G[4]
    ]
```

### Step 5: Apply Genetic Operators

**Crossover** (`SBX` - Simulated Binary Crossover):
- Standard pymoo SBX operator with rounding for integer piece indices
- prob=0.9, eta=15 (configurable)

**Mutation** (`PM` - Polynomial Mutation):
- Standard pymoo polynomial mutation
- prob=0.1, eta=20 (configurable)

**Repair** (`TrackRepair`):
- Removes pieces not in inventory (invalid piece types)
- Preserves starting position variables

### Step 6: Check Termination

```python
termination = PrimaryObjectiveStagnationTermination(
    max_gen=1000,
    patience=100,      # Stop if no improvement for 100 generations
    delta=0.001        # Minimum improvement threshold
)
```

Conditions:
- Maximum generations reached (1000 default)
- Primary objective (utilization) stagnant for 100 generations
- All constraints satisfied for best solution

### Step 7: Output Results

```python
# Extract Pareto front
pareto = result.F

# Get best solutions
best_indices = np.argsort(pareto[:, 0])[:5]  # Top 5 by utilization

# Save visualizations
for i, idx in enumerate(best_indices):
    save_layout_figure(layouts[idx], f'outputs/layout_{i+1}.png')

# Save Pareto front
plot_pareto_front(pareto, 'outputs/pareto_front.png')

# Save chromosome data
for i, idx in enumerate(best_indices):
    save_chromosome(chromosomes[idx], f'outputs/chromosome_{i+1}.json')
```

---

## 4. Architecture Assessment

### Strengths

**Phase-Based Design**:
- Phase 1: Single loop, default routes only (current)
- Phase 2: Branches (dead-end or rejoin sidings)
- Phase 3: Multi-route pieces (switches select routes)
- Phase 4: Multiple loops with collision detection

All structures pre-support Phase 4 allowing incremental feature additions.

**Genotype/Phenotype Separation**:
- `LayoutChromosome`: What pieces in what order (genotype)
- `LayoutGeometry`: Computed positions and metrics (phenotype)

Clean separation enables different encodings without changing evaluation.

**Vectorized Operations**:
- NumPy arrays for FK tables, states, objectives
- Batch lookup for speed limits, radii, arc lengths
- Efficient for populations of 1000+ individuals

**pymoo Operators**:
- `HeuristicSampling`: Custom sampling with 51% heuristic patterns
- `SBX`: Standard simulated binary crossover
- `PM`: Standard polynomial mutation
- `TrackRepair`: Custom repair for inventory validation

**Topology-Aware Repairs**:
- Switch pairing: Ensure IN/OUT balance
- Bumper placement: Valid termination points
- Closure repair: Add curves to close gaps

### Design Patterns Used

| Pattern | Usage |
|---------|-------|
| Factory | `TrackCatalog.from_yaml()`, `OptimizationConfig.from_yaml()` |
| Strategy | Interchangeable sampling, crossover, mutation operators |
| Composite | `LayoutGeometry` aggregates `LoopGeometry` + `BranchGeometry` |
| Frozen dataclasses | Immutable `FKDeltas`, `Port`, `TrackPiece` |

### Module Dependencies

```
main.py
  |-- config.py (Pydantic models)
  |-- data.py (TrackCatalog)
  |-- problem.py
        |-- geometry.py (Layout, LayoutGeometry)
        |-- evaluation.py (SpeedProfile, objectives, constraints)
        |-- topology.py (LayoutChromosome, PieceTopology)
  |-- sampling.py (HeuristicSampling)
  |-- operators.py (crossover, mutation, repair)
  |-- visualization.py (plotting)
```

### Areas for Improvement

1. **Parallelization**: Current single-threaded evaluation limits scalability
2. **Caching**: FK chains could be memoized for repeated sub-sequences
3. **Adaptive operators**: Mutation rates could adapt based on population diversity
4. **Multi-objective selection**: Consider MAP-Elites for diverse feature exploration

---

## References

- **Locomotive_dynamics.md**: Physics model derivation
- **track_geometry.md**: FK formulas and closure detection
- **technical_design.md**: Full architecture specification
- **research_evolutionary_topology.md**: Literature review on topology evolution