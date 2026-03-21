# Technical Design Document

## Architecture Overview

The LEGO Track Optimizer implements a multi-objective genetic algorithm for generating feasible closed railway layouts within fixed inventory constraints. The system integrates locomotive physics (derailment limits, speed profiles) with track geometry (port alignment, closure detection) to evaluate candidate layouts against multiple competing objectives.

### Design Principles

1. **Separation of Concerns**: Geometry computation, physics simulation, and optimization logic reside in distinct modules
2. **Catalog-Driven Extensibility**: New track pieces require only data additions, not code changes
3. **Constraint-First Evaluation**: Feasibility checks (closure, connectivity) precede expensive physics simulation
4. **Vectorized Operations**: NumPy arrays for FK tables and batch lookups

---

## Module Architecture

### Core Modules

```
src/
├── problem.py      # pymoo Problem class, TrackRepair operator
├── geometry.py     # Layout class, FK chain computation
├── evaluation.py   # Objectives, constraints, speed profile
├── data.py         # TrackCatalog, TrackPiece, YAML loader
├── config.py       # Pydantic configuration models
├── sampling.py     # HeuristicSampling for seeding population
└── visualization.py # Layout plots, Pareto fronts
```

### Module Responsibilities

| Module | Primary Responsibility | Key Classes/Functions |
|--------|----------------------|----------------------|
| `problem.py` | pymoo problem definition | `TrackLayoutProblem`, `TrackRepair`, `create_problem()` |
| `geometry.py` | FK computation, layout building | `Layout`, `build_layout()`, `compute_fk_chain()` |
| `evaluation.py` | Objectives and constraints | `SpeedProfile`, `compute_speed_profile()`, `compute_objectives()`, `compute_constraints()` |
| `data.py` | Track catalog management | `TrackCatalog`, `TrackPiece`, `FKDeltas`, `Port` |
| `config.py` | Configuration validation | `OptimizationConfig`, `PhysicsConfig`, `BoundaryConfig`, `AlgorithmConfig` |
| `sampling.py` | Population initialization | `HeuristicSampling`, `create_sampling()` |
| `visualization.py` | Results visualization | `plot_layout()`, `plot_pareto_front()`, `create_results_figure()` |

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│ configs/*.yaml  │────▶│   config.py     │────▶ OptimizationConfig
└─────────────────┘     └─────────────────┘
                                                       │
┌─────────────────┐     ┌─────────────────┐           ▼
│track_pieces.yaml│────▶│    data.py      │────▶ TrackCatalog
└─────────────────┘     └─────────────────┘           │
                                                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   problem.py    │◀────│    main.py      │
                        └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │geometry.py│ │evaluation │ │visualization│
            └───────────┘ └───────────┘ └───────────┘
```

---

## Class Structures

### FKDeltas (data.py)

Forward kinematics deltas for a track piece.

```python
@dataclass(frozen=True)
class FKDeltas:
    dx: float       # X displacement in local frame (studs)
    dy: float       # Y displacement in local frame (studs)
    dtheta: float   # Heading change (degrees)

    def to_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dtheta])
```

### Port (data.py)

Connection point on a track piece.

```python
@dataclass(frozen=True)
class Port:
    x: float        # X position in piece-local coords
    y: float        # Y position in piece-local coords
    heading: float  # Heading in degrees
    gender: str     # 'M' or 'F'
```

### TrackPiece (data.py)

A track piece definition from the catalog.

```python
@dataclass
class TrackPiece:
    id: str                 # Unique identifier (e.g., "R56_LEFT")
    name: str               # Display name
    piece_type: str         # 'straight', 'curve', 'switch', 'crossing', 'bumper'
    fk: FKDeltas            # Forward kinematics deltas
    ports: Tuple[Port, ...] # Connection points
    index: int              # Position in piece_index mapping (-1 if unmapped)

    # Geometry
    length: float = 0.0     # Straight length (studs)
    radius: float = 0.0     # Curve radius (studs)
    angle: float = 0.0      # Arc angle (degrees)
    direction: str = ""     # 'left' or 'right' for curves

    # Physics
    radius_mm: float = np.inf     # Radius in mm for speed calculation
    speed_limit_ms: float = 1.57  # Max speed (m/s)
    is_terminator: bool = False   # True for bumpers
```

### TrackCatalog (data.py)

Manages the complete inventory of available track pieces with vectorized access.

```python
class TrackCatalog:
    SECTION_TYPES: ClassVar[Dict[str, str]] = {
        'straights': 'straight',
        'curves': 'curve',
        'special_curves': 'curve',
        'crossings': 'crossing',
        'bumpers': 'bumper',
        'r40_switch_components': 'switch',
        'r56_switch_components': 'switch',
    }

    # Numpy lookup tables
    _fk_table: np.ndarray        # (n, 3) [dx, dy, dtheta]
    _speed_table: np.ndarray     # (n,) speed limits in m/s
    _radius_table: np.ndarray    # (n,) radii in mm
    _arc_length_table: np.ndarray # (n,) arc lengths in studs

    # Methods
    def get_fk(indices: np.ndarray) -> np.ndarray      # Vectorized FK lookup
    def get_radii(indices: np.ndarray) -> np.ndarray   # Vectorized radius lookup
    def get_speed_limits(indices: np.ndarray) -> np.ndarray
```

### Layout (geometry.py)

A track layout represented as cumulative FK states.

```python
@dataclass
class Layout:
    indices: np.ndarray     # Piece indices used (valid entries from chromosome)
    states: np.ndarray      # (n+1, 3) array of [x, y, theta] states

    @property
    def n_pieces(self) -> int
    @property
    def final_state(self) -> np.ndarray
    @property
    def closure_error(self) -> float      # Distance from final to origin
    @property
    def angle_error(self) -> float        # Deviation from 360 degrees
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]
    @property
    def area(self) -> float               # Bounding box area

    def is_closed(pos_tol: float, angle_tol: float) -> bool
```

### SpeedProfile (evaluation.py)

Speed profile result from forward-backward pass algorithm.

```python
@dataclass
class SpeedProfile:
    speeds: np.ndarray      # Speed at each segment (m/s)
    avg_speed: float        # Mean speed over layout
    lap_time: float         # Total traversal time (s)
    total_distance: float   # Total distance (m)
```

### OptimizationConfig (config.py)

Complete optimization configuration using Pydantic.

```python
class OptimizationConfig(BaseModel):
    inventory: Dict[str, int]               # Piece ID -> count
    boundary: BoundaryConfig                # Spatial limits
    closure_tolerance: float = 0.5          # Position tolerance (studs)
    angle_tolerance: float = 5.0            # Angle tolerance (degrees)
    algorithm: AlgorithmConfig              # NSGA-II parameters
    physics: PhysicsConfig                  # Locomotive physics
    n_workers: int = 1                      # Parallelization

    @property
    def total_inventory(self) -> int        # Sum of all piece counts
    @property
    def n_var(self) -> int                  # Chromosome length = total_inventory
```

---

## Encoding Scheme

### Chromosome Structure

The chromosome is a fixed-length array combining piece indices and starting position:

```
chromosome: np.ndarray, shape=(n_piece_vars + 2,), dtype=float64

Structure:
  [0..n_piece_vars-1]  = piece indices (rounded to int for evaluation)
                         -1 = empty slot (piece not used)
                         0..9 = piece type index from piece_index mapping
  [n_piece_vars]       = start_x (starting X position within boundary)
  [n_piece_vars + 1]   = start_y (starting Y position within boundary)
```

The piece portion length equals `total_inventory` (sum of all available pieces). The starting position allows the layout to be placed anywhere within the boundary constraints.

### Piece Index Mapping

Defined in `data/track_pieces.yaml` under `piece_index`:

```yaml
piece_index:
  # Straights
  0: STRAIGHT_16
  1: STRAIGHT_24
  # Curves
  2: R40_LEFT
  3: R40_RIGHT
  # Crossings
  4: CROSS_90
  9: DOUBLE_CROSSOVER
  # Switches
  5: R40_SWITCH_LEFT_IN
  6: R40_SWITCH_LEFT_OUT
  7: R40_SWITCH_RIGHT_IN
  8: R40_SWITCH_RIGHT_OUT
  # Sentinel
  -1: null
```

### Forward Kinematics Lookup Table

FK deltas are precomputed and stored in `TrackCatalog._fk_table`:

```python
# Shape: (n_piece_types, 3)
# Columns: [dx_local, dy_local, dtheta]

# FK computation in geometry.py:
theta_rad = np.radians(state[2])
cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
x_new = x + dx * cos_t - dy * sin_t
y_new = y + dx * sin_t + dy * cos_t
theta_new = theta + dtheta
```

---

## pymoo Integration

### Problem Definition

```python
class TrackLayoutProblem(ElementwiseProblem):
    def __init__(self, catalog: TrackCatalog, config: OptimizationConfig):
        # Chromosome: [piece_indices..., start_x, start_y]
        n_piece_vars = config.total_inventory
        n_position_vars = 2
        n_total_vars = n_piece_vars + n_position_vars

        super().__init__(
            n_var=n_total_vars,          # Piece indices + starting position
            n_obj=2,                      # Utilization, speed
            n_ieq_constr=5,              # Closure, angle, boundary, inventory, orphan switches
            xl=xl,                        # -1 for pieces, boundary min for position
            xu=xu,                        # max_index for pieces, boundary max for position
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pieces = np.round(x[:n_piece_vars]).astype(np.int32)
        start_x, start_y = x[n_piece_vars], x[n_piece_vars + 1]

        layout = build_layout(pieces, self.catalog)
        layout.states[:, 0] += start_x  # Translate to starting position
        layout.states[:, 1] += start_y

        profile = compute_speed_profile(layout, self.catalog, self.config.physics)

        out["F"] = compute_objectives(layout, profile, self.catalog, self.total_inventory)
        out["G"] = compute_constraints(layout, pieces, self.inventory, ...)
```

### Why ElementwiseProblem

The evaluation involves sequential FK computation that cannot be vectorized across the population. `ElementwiseProblem` enables:

1. Clear single-individual evaluation logic
2. Built-in parallelization via `StarmapParallelization`
3. Simpler debugging and profiling

### Repair Operator

```python
class TrackRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = self._repair_inventory(X[i])
        return X

    def _repair_inventory(self, x):
        # Replace excess pieces with -1 (empty)
```

### HeuristicSampling

Custom sampling operator that seeds population with known closed loop patterns:

```python
class HeuristicSampling(Sampling):
    HEURISTIC_RATIO = 0.51  # 51% heuristic, 49% random

    def _do(self, problem, n_samples, **kwargs):
        # Generate patterns validated against inventory
        # Simple circles, ovals, racetracks
```

---

## Multi-Objective Formulation

### Objectives (2, all minimized)

| # | Objective | Direction | Implementation |
|---|-----------|-----------|----------------|
| 0 | Piece Utilization | Maximize | `-utilization` (pieces_used / total_inventory) |
| 1 | Average Speed | Maximize | `-profile.avg_speed` |

**Note:** Area objective was removed - it caused circles to dominate ovals on the Pareto front despite lower utilization/speed. With 2 objectives, ovals properly dominate circles on both dimensions.

### Constraints (5, g ≤ 0 is feasible)

| # | Constraint | Normalization |
|---|------------|---------------|
| 0 | Closure Position | `(error - tolerance) / tolerance` |
| 1 | Closure Angle | `(error - tolerance) / tolerance` |
| 2 | Boundary | `violation / diagonal` |
| 3 | Inventory | Sum of excess pieces |
| 4 | Orphan Switches | Count of unpaired IN/OUT switches |

### Physics Integration

Speed profile computation follows the forward-backward pass algorithm:

```python
# Pass 1: Curve speed limits
v_limit[i] = SF × sqrt(μ × g × R[i])

# Pass 2: Forward pass (acceleration)
v_fwd[i] = min(v_limit[i], sqrt(v_fwd[i-1]² + 2 × a_max × Δs))

# Pass 3: Backward pass (braking)
v_bwd[i] = min(v_fwd[i], sqrt(v_bwd[i+1]² + 2 × a_brake × Δs))
```

Physics parameters from research:
- Safety factor SF: 0.8
- Friction coefficient μ: 0.30
- Max acceleration: 3.92 m/s² (adhesion-limited)
- Braking deceleration: 2.45 m/s²

---

## Algorithm Configuration

### Default NSGA-II Setup

```yaml
algorithm:
  name: NSGA2
  pop_size: 100
  n_gen: 200

  crossover_prob: 0.9
  crossover_eta: 15.0
  mutation_eta: 20.0

  eliminate_duplicates: true
  seed: null
```

### Operators Used

- **Sampling**: `HeuristicSampling` (custom, 51% heuristic patterns)
- **Crossover**: `SBX` with `RoundingRepair`
- **Mutation**: `PM` with `RoundingRepair`
- **Repair**: `TrackRepair` (fix inventory violations)

---

## Performance Considerations

### Evaluation Bottlenecks

| Operation | Complexity | Optimization |
|-----------|------------|--------------|
| FK chain | O(N) per individual | Sequential loop (inherent) |
| Closure check | O(1) final state comparison | Direct array access |
| Speed profile | O(N) forward-backward pass | NumPy arrays |
| Inventory check | O(N) bincount | NumPy vectorized |

### Memory Layout

- FK table: (n_piece_types, 3) float64 ≈ 1 KB
- Population: (pop_size, n_var) int32 ≈ 40 KB for 100×100
- Per-evaluation Layout: ~10 KB transient

---

## Extension Points

### Adding New Track Pieces

1. Add piece definition to `data/track_pieces.yaml` with `fk`, `ports`, `physics`
2. Add entry to `piece_index` mapping with next available index
3. Update `INDEX_TO_ID` in `src/sampling.py`
4. Run validation: `pytest tests/test_data.py`

### Adding New Objectives

1. Implement objective function in `evaluation.py`
2. Update `n_obj` in `TrackLayoutProblem.__init__()`
3. Add to `compute_objectives()` return array
4. Update visualization if needed

### Adding New Constraints

1. Implement constraint function in `evaluation.py` (return g ≤ 0 format)
2. Update `n_ieq_constr` in `TrackLayoutProblem.__init__()`
3. Add to `compute_constraints()` return array
4. Optionally add corresponding repair logic

---

## References

- pymoo documentation: https://pymoo.org/customization/problem.html
- Deb et al. (2002): NSGA-II algorithm
- Pham & Pham (2018): TOPP-RA time-optimal path parameterization
- L-Gauge standard: https://l-gauge.org

---

**Version**: 2.0 (Updated to match implementation)
**Status**: Implementation complete
