# Implementation Checkpoint - Before Deletion

This document captures the complete implementation state of the LEGO Track Optimizer before deleting src/, tests/, and main.py for a clean rewrite.

---

## Files to Delete

```
src/
  problem.py
  geometry.py
  evaluation.py
  data.py
  config.py
  sampling.py
  visualization.py
  topology.py
  operators.py

tests/
  test_data.py
  test_geometry.py
  test_evaluation.py
  test_operators.py

main.py
```

---

## Files to Keep

```
CLAUDE.md                    # Updated project reference
docs/
  implementation.md          # Architecture documentation
  technical_design.md        # Design specification
  Locomotive_dynamics.md     # Physics model
  Logging_and_Output.md      # Output formats
  Piece_geometry_3ways_exploration.md
  Raw_trackPieces_info.md
  research_evolutionary_topology.md
  Track_geometry.md
  checkpoint_before_rewrite.md  # This file

data/
  track_pieces.yaml          # Track catalog (needs DOUBLE_CROSSOVER added)

configs/
  default.yaml
  compact.yaml
  with_switches.yaml
  with_crossing.yaml

.claude/
  agents/                    # All agent definitions
  settings.local.json
```

---

## Source Files Blueprint

### 1. problem.py

**Classes:**
- `TrackLayoutProblem(ElementwiseProblem)` - n_var=1, n_obj=3, n_ieq_constr=5
- `TrackRepair(Repair)` - Inventory constraint repair
- `ClosureRepair(Repair)` - Adds curves to close gaps

**Key Methods:**
```python
def _evaluate(self, x, out, *args, **kwargs):
    chromosome = x[0]  # n_var=1 pattern
    layout_chr = flat_array_to_chromosome(chromosome.pieces)
    geometry = compute_layout_geometry(layout_chr, catalog)
    speed_profile = compute_layout_speed_profile(geometry, physics)
    out["F"] = [-utilization, -avg_speed, bbox_area]
    out["G"] = [closure, angle, boundary, inventory, switch_pairing]
```

---

### 2. geometry.py

**Classes:**
- `Layout` - Legacy single-loop (indices, states)

**Functions:**
```python
def compute_fk_chain(fk_deltas: np.ndarray) -> np.ndarray:
    # Returns (n+1, 3) states [x, y, theta]
    theta_rad = np.radians(states[i, 2])
    x_new = x + dx*cos(theta) - dy*sin(theta)
    y_new = y + dx*sin(theta) + dy*cos(theta)
    theta_new = theta + dtheta

def build_layout(chromosome, catalog) -> Layout
def compute_layout_geometry(chromosome, catalog) -> LayoutGeometry
def compute_loop_geometry(loop_def, catalog, initial_pose) -> LoopGeometry
def compute_branch_geometry(branch_def, catalog, diverge_pose, ...) -> BranchGeometry
```

---

### 3. evaluation.py

**Classes:**
- `SpeedProfile` - speeds, avg_speed, lap_time, total_distance

**Functions:**
```python
def compute_speed_limit(radius_mm, physics) -> float:
    return SF * sqrt(mu * g * R)  # SF=0.8, mu=0.30

def compute_speed_profile(layout, catalog, physics) -> SpeedProfile:
    # 3-pass: curve limits, forward accel, backward brake

def evaluate_closure(layout, tolerance) -> float  # G[0]
def evaluate_angle(layout, tolerance) -> float    # G[1]
def evaluate_boundary(layout, boundary) -> float  # G[2]
def evaluate_inventory(chromosome, inventory, catalog) -> float  # G[3]
def evaluate_switch_pairing(chromosome) -> float  # G[4]

def compute_objectives_from_geometry(geometry, profile, chromosome, total_inv) -> np.ndarray
def compute_constraints_from_geometry(geometry, chromosome, ...) -> np.ndarray
```

**Constants:**
```python
_SWITCH_PAIRS = {24: 25, 25: 24, 26: 27, 27: 26}
_SWITCH_IN_INDICES = {24, 26}
_SWITCH_OUT_INDICES = {25, 27}
_BUMPER_INDEX = 23
```

---

### 4. data.py

**Classes:**
- `FKDeltas` (frozen) - dx, dy, dtheta
- `Port` (frozen) - x, y, heading, gender
- `TrackPiece` - id, name, piece_type, fk, ports, index, physics
- `TrackCatalog` - Vectorized piece access with multi-route support

**Key Methods:**
```python
def load(path) -> TrackCatalog  # classmethod
def get_fk(indices) -> np.ndarray  # (n, 3)
def get_radii(indices) -> np.ndarray
def get_arc_lengths(indices) -> np.ndarray
def get_speed_limits(indices) -> np.ndarray
def get_fk_with_routes(piece_indices, route_indices) -> np.ndarray
def get_topology(piece_idx) -> PieceTopology
```

---

### 5. config.py

**Classes (Pydantic v2):**
- `BoundaryConfig` - min_x, max_x, min_y, max_y
- `TerminationConfig` - n_max_gen, ftol, xtol, period
- `AlgorithmConfig` - pop_size, n_gen, crossover/mutation params, seed
- `PhysicsConfig` - safety_factor=0.8, friction_coeff=0.30, gravity=9.81, motor_top_speed=1.57
- `OptimizationConfig` - inventory, boundary, tolerances, algorithm, physics

**Key Properties:**
```python
@property
def n_var(self) -> int:
    return self.calculate_max_layout_pieces()
```

---

### 6. sampling.py

**Classes:**
- `HeuristicSampling(Sampling)` - 51% heuristic, 49% random

**Patterns Generated:**
- Simple circles: 16 R40 curves
- Symmetric ovals: curves + straights
- Racetracks: 4 corners + straights
- Switch patterns: switch pairs replacing straights
- Max-piece pattern: boundary-aware maximum utilization

**Key Methods:**
```python
def _do(problem, n_samples, **kwargs) -> np.ndarray
def _build_valid_patterns()  # Validates against inventory
def _fits_inventory(pattern) -> bool
def _create_random_individual(max_len) -> np.ndarray
def _create_max_piece_pattern(safety_margin=10.0) -> np.ndarray
```

---

### 7. visualization.py

**Functions:**
```python
def plot_layout(layout, catalog, ax=None, ...) -> plt.Axes
def plot_layout_geometry(geometry, catalog, ax=None, ...) -> plt.Axes
def plot_pareto_front(F, ax=None, ...) -> plt.Axes
def plot_pareto_3d(F, ax=None, ...) -> plt.Axes
def create_results_figure(layouts, F, catalog, n_best=4) -> plt.Figure
def save_layout_geometry_figure(geometry, catalog, path, ...) -> None
```

**Color Coding:**
- Straights: dark gray
- Curves: blue
- Switch IN: red
- Switch OUT: orange
- Crossing: purple
- Bumper: dark red
- Branches: green dashed

---

### 8. topology.py

**Enums:**
- `PieceClass` - SIMPLE_2PORT, SWITCH_3PORT, SWITCH_4PORT, CROSSING_4PORT, BUMPER_1PORT

**Dataclasses (Genotype):**
- `FKRoute` (frozen) - entry/exit ports, dx/dy/dtheta, arc_length, radius, speed_limit
- `PieceTopology` (frozen) - piece_id, piece_index, piece_class, num_ports, routes
- `RouteSpec` - piece_position, route_idx
- `BranchDef` - diverge_position, piece_indices, is_dead_end, rejoin_position
- `LoopDef` - main_sequence, route_specs, branches
- `LayoutChromosome` - loops, crossing_connections

**Dataclasses (Phenotype):**
- `BranchGeometry` - states, arc_lengths, radii, rejoin_error
- `LoopGeometry` - states, closure_error, angle_error, bounding_box, branches
- `LayoutGeometry` - loops, collisions (backward compatible properties)

---

### 9. operators.py

**Classes:**
- `TrackChromosome` - start_x, start_y, pieces array
- `TrackSampling(Sampling)` - Uses HeuristicSampling internally
- `TrackCrossover(Crossover)` - Single-point on pieces
- `TrackMutation(Mutation)` - Position (1%) + piece (10%) mutation
- `TrackDuplicateElimination` - Object equality
- `SwitchPairRepair(Repair)` - Balance IN/OUT pairs
- `BumperRepair(Repair)` - Remove invalid bumpers
- `TopologyAwareRepair(Repair)` - Combined: inventory, switches, bumpers, branches, closure

**Functions:**
```python
def detect_switch_pairs(piece_indices) -> List[Tuple[int, int, str]]
def create_branch_from_switch_pair(...) -> BranchDef
def flat_array_to_chromosome_with_branches(x, detect_branches=True) -> LayoutChromosome
def identify_protected_segments(piece_indices) -> List[Tuple[int, int]]
def validate_bumper_placement(piece_indices) -> List[int]
```

**Constants:**
```python
R40_SWITCH_LEFT_IN = 24
R40_SWITCH_LEFT_OUT = 25
R40_SWITCH_RIGHT_IN = 26
R40_SWITCH_RIGHT_OUT = 27
BUMPER_INDEX = 23
SWITCH_PAIRS = {'left': (24, 25), 'right': (26, 27)}
IN_TO_OUT = {24: 25, 26: 27}
OUT_TO_IN = {25: 24, 27: 26}
```

---

## main.py Blueprint

### Command-Line Arguments
```python
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--catalog', default='data/track_pieces.yaml')
parser.add_argument('--output', default='outputs')
parser.add_argument('--verbose', action='store_true')
```

### Custom Termination
```python
class PrimaryObjectiveStagnationTermination:
    # Stops on max_gen OR no improvement for 100 generations
    # Tracks best feasible utilization (F[0])
```

### Progress Callback
```python
class OptimizationProgressCallback:
    # Logs every 10 generations
    # Shows: generation, feasible count, best objectives
```

### Optimization Flow
```python
1. Load catalog and config
2. Create TrackLayoutProblem (n_var=1, n_obj=3, n_ieq_constr=5)
3. Setup NSGA2:
   - TrackSampling (51% heuristic)
   - TrackCrossover (prob=0.9)
   - TrackMutation (position=0.01, piece=0.1)
   - TrackDuplicateElimination
4. Run minimize() with:
   - Custom termination
   - Progress callback
   - Seed from config
5. Filter feasible solutions (G.max <= 0)
6. Sort by F[0] (utilization)
7. Generate outputs:
   - chromosome_N.txt/json
   - layout_N.png
   - pareto_front.png
   - summary.png
   - summary.txt
   - results.npz
```

---

## Test Files Blueprint

### test_data.py (107 lines)
- `TestTrackCatalog` - Loading, FK table shape, piece lookup
- `TestVectorizedLookup` - get_fk, get_radii, get_speed_limits
- `TestFKDeltas` - to_array conversion
- `TestTrackPiece` - arc_length calculations

### test_geometry.py (124 lines)
- `TestComputeFKChain` - Empty, single straight, 90 deg turn
- `TestBuildLayout` - From chromosome, empty chromosome
- `TestLayout` - R40 circle closure, bounding box, area
- `TestMixedLayout` - Straight + curve combinations

### test_evaluation.py (289 lines)
- Speed limits: R40, R56, straight, larger radius faster
- Speed profiles: straight max, curved slower, empty zero
- Closure constraint: closed loop passes, open path fails
- Boundary constraint: within passes, exceeds fails
- Inventory constraint: within passes, exceeds fails
- Switch pairing: no switches passes, paired passes, orphan fails

### test_operators.py (338 lines)
- Switch detection: no switches, left/right pairs, orphans
- Protected segments: none, single segment
- Bumper validation: none, in main loop (invalid)
- SwitchPairRepair: orphan repair, preserve valid
- BumperRepair: remove from loop, preserve others
- TopologyAwareRepair: full pipeline

### Test Fixtures
```python
@pytest.fixture
def catalog():
    return TrackCatalog.load("data/track_pieces.yaml")

@pytest.fixture
def physics():
    return PhysicsConfig()

@pytest.fixture
def inventory():
    return {"STRAIGHT_16": 10, "R40_LEFT": 16, ...}
```

---

## Track Piece Index Mapping (New)

| Index | Piece ID | Description |
|-------|----------|-------------|
| 0 | STRAIGHT_16 | 16-stud straight |
| 1 | STRAIGHT_24 | 24-stud straight |
| 2 | R40_LEFT | 22.5 deg left curve |
| 3 | R40_RIGHT | 22.5 deg right curve |
| 4 | CROSS_90 | 90 deg crossing |
| 5 | R40_SWITCH_LEFT_IN | Left switch IN |
| 6 | R40_SWITCH_LEFT_OUT | Left switch OUT |
| 7 | R40_SWITCH_RIGHT_IN | Right switch IN |
| 8 | R40_SWITCH_RIGHT_OUT | Right switch OUT |
| 9 | DOUBLE_CROSSOVER | **TO IMPLEMENT** - 48x24 studs |

---

## Implementation Notes

### DOUBLE_CROSSOVER (Index 9) - Needs Implementation

From Raw_trackPieces_info.md:
- 4 independent switches in modular design
- 48x24 stud footprint
- 8-stud parallel track spacing
- All switches can be actuated independently
- LEGO-style ground throws, motorizable

Required implementation:
1. Add to track_pieces.yaml with FK routes for all path combinations
2. Define ports (likely 4-8 ports depending on modeling)
3. Calculate speed limits for diverging routes
4. Update sampling.py patterns if needed

### Phase Support

- **Phase 1** (current): Single loop, default routes
- **Phase 2**: Branches (dead-end, passing sidings)
- **Phase 3**: Multi-route switches (explicit route selection)
- **Phase 4**: Multiple loops with crossing connections

All data structures already support Phase 4; implementation can be incremental.

### Key Physics Constants

```python
SF = 0.8           # Safety factor
mu = 0.30          # Friction coefficient
g = 9.81           # Gravity (m/s^2)
top_speed = 1.57   # Motor top speed (m/s)
max_accel = 3.92   # Max acceleration (m/s^2)
brake_decel = 2.45 # Braking deceleration (m/s^2)
stud_mm = 8.0      # Stud size (mm)
```

### Speed Limits by Piece

| Piece | Speed Limit |
|-------|-------------|
| R40 curves | 0.97 m/s |
| Straights | 1.57 m/s |
| CROSS_90 | 1.57 m/s |
| Switches (diverge) | 0.87 m/s |

---

## Ready for Deletion

You can now safely delete:
- `src/` (all 9 files)
- `tests/` (all 4 files)
- `main.py`

Reference this document and `docs/implementation.md` when rebuilding.