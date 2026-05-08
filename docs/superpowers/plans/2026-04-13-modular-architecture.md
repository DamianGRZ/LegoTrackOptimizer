# Modular Architecture Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the codebase into clean module tiers with correct dependency direction, removing ~1,350 lines of dead code and fixing broken tests.

**Architecture:** Hybrid sub-package approach — `train/` and `catalog/` become packages (preparing for growth), everything else stays flat. A new `src/types.py` holds the shared domain vocabulary. `topology.py` is dissolved, `evaluation.py` moves into `train/`, dead Phase 3/4 code is deleted.

**Tech Stack:** Python 3.x, pymoo 0.6.1.6, numpy, pydantic, pytest

---

## File Map

### New files
- `src/types.py` — shared domain types (SwitchPair, TraversalPath, MultiPathLayout, PieceClass, FKRoute, PieceTopology)
- `src/train/__init__.py` — re-exports TrainConfig, v_eff_array, available_accel, DEFAULT_TRAIN_CONFIG
- `src/train/physics.py` — current train.py content (TrainConfig, v_eff_array, available_accel)
- `src/train/scoring.py` — SpeedProfile + compute_speed_profile (moved from evaluation.py)
- `src/catalog/__init__.py` — re-exports TrackCatalog, TrackPiece, FKDeltas
- `src/catalog/pieces.py` — FKDeltas, Port, TrackPiece dataclasses (from data.py lines 1-78)
- `src/catalog/catalog.py` — TrackCatalog class (from data.py lines 80-576)

### Deleted files
- `src/topology.py` — dissolved into types.py, dead code deleted
- `src/evaluation.py` — live code moves to train/scoring.py, dead code deleted

### Modified files
- `src/geometry.py` — remove dead functions (lines 159-475), update imports
- `src/intersection.py` — remove dead code (~250 lines)
- `src/data.py` — remove dead methods, becomes catalog/ package
- `src/templates.py` — remove dead templates/functions (~60 lines)
- `src/lego_track_models.py` — remove dead functions/constants (~150 lines)
- `src/problem.py` — update imports (evaluation → train.scoring, topology → types, data → catalog)
- `src/decoder.py` — update imports (topology → types, data → catalog)
- `src/operators.py` — no changes (only imports from encoding)
- `src/repair.py` — no changes (only imports from encoding)
- `src/sampling.py` — update imports (data → catalog)
- `src/config.py` — update imports (train → train.physics)
- `src/visualization.py` — update imports (topology → types, data → catalog)
- `src/__init__.py` — update imports (data → catalog)
- `main.py` — update imports (data → catalog)
- `tests/conftest.py` — update imports (data → catalog)
- `tests/test_data.py` — update imports (topology → types, data → catalog)
- `tests/test_evaluation.py` — delete dead test classes, keep TestSpeedProfile, update imports
- `tests/test_geometry.py` — delete TestLayoutGeometry + TestAngularDeficit, remove dead imports
- `tests/test_problem.py` — fix broken imports (ChromosomeDimensions → PartitionedDimensions), fix compute_dimensions call, fix constraint count assertion
- `tests/test_decoder.py` — fix broken imports + compute_dimensions call
- `tests/test_sampling.py` — fix broken compute_dimensions call

---

## Task 1: Fix broken tests (pre-refactor baseline)

The 3 broken test files must pass before any restructuring. This establishes a green baseline.

**Files:**
- Modify: `tests/test_problem.py`
- Modify: `tests/test_decoder.py`
- Modify: `tests/test_sampling.py`

- [ ] **Step 1: Fix test_problem.py**

```python
# tests/test_problem.py — fix imports and assertions

# Line 7-12: Replace the import block
from src.encoding import (
    compute_dimensions,
    create_chromosome_from_pieces,
    create_empty_chromosome,
    PartitionedDimensions,  # was ChromosomeDimensions
    PieceIndex,
)

# Line 21: Fix compute_dimensions call (needs config + catalog, not int)
#   OLD: dims = compute_dimensions(default_config.total_inventory)
#   NEW: (delete this line — problem already computes dims)
# The test_problem_dimensions test should use problem.dims directly:
def test_problem_dimensions(self, catalog, default_config):
    """Verify problem has correct dimensions."""
    problem = TrackOptimizationProblem(catalog, default_config)

    assert problem.n_var == problem.dims.n_var
    assert problem.n_obj == 2  # utilization + speed
    assert problem.n_ieq_constr == 6  # was 5, now 6

# Line 25: Fix constraint count assertion
#   OLD: assert problem.n_ieq_constr == 5
#   NEW: assert problem.n_ieq_constr == 6
```

- [ ] **Step 2: Fix test_decoder.py**

```python
# tests/test_decoder.py — fix import on line 10
# OLD: from src.encoding import (ChromosomeDimensions, compute_dimensions, ...
# NEW: from src.encoding import (PartitionedDimensions, compute_dimensions, ...

# Every occurrence of compute_dimensions(default_config.total_inventory) must change to:
#   compute_dimensions(default_config, catalog)
# This occurs on lines 32, 41, 55, 71, 87, 102 (approximately)
# Find-and-replace: compute_dimensions(default_config.total_inventory)
#   → compute_dimensions(default_config, catalog)

# Every occurrence of ChromosomeDimensions must change to PartitionedDimensions
```

- [ ] **Step 3: Fix test_sampling.py**

```python
# tests/test_sampling.py line 16
# OLD: dims = compute_dimensions(default_config.total_inventory)
# NEW: dims = compute_dimensions(default_config, catalog)

# The test_sampling_shape test needs catalog fixture:
def test_sampling_shape(self, catalog, default_config):
    """Returns (n_samples, n_var) array."""
    dims = compute_dimensions(default_config, catalog)
    # ... rest unchanged
```

- [ ] **Step 4: Run full test suite to verify baseline**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass (some may skip, but no ERRORS or import failures).

- [ ] **Step 5: Commit**

```bash
git add tests/test_problem.py tests/test_decoder.py tests/test_sampling.py
git commit -m "fix: repair broken test imports and stale API references"
```

---

## Task 2: Delete dead code from live files

Remove dead functions, classes, and constants from files that will remain. This reduces noise before restructuring.

**Files:**
- Modify: `src/geometry.py:159-475` — delete 6 dead functions
- Modify: `src/intersection.py` — delete dead classes and functions
- Modify: `src/templates.py` — delete dead templates and functions
- Modify: `src/lego_track_models.py` — delete dead functions and constants
- Modify: `src/data.py` — delete dead methods on TrackCatalog

- [ ] **Step 1: Clean geometry.py**

Delete these functions/code blocks (lines 159-475):
- `estimate_angular_deficit` (lines 159-174)
- `compute_layout_geometry` (lines 177-203)
- `compute_loop_geometry` (lines 206-286)
- `compute_branch_geometry` (lines 289-378)
- `_apply_initial_pose` (lines 381-408)
- `_normalize_angle_error` (lines 411-424)
- `build_layout_from_chromosome` (lines 427-447)
- `layout_to_geometry` (lines 450-475)

Remove the dead topology imports (lines 12-19):
```python
# DELETE these imports:
from .topology import (
    BranchDef,
    BranchGeometry,
    LayoutChromosome,
    LayoutGeometry,
    LoopDef,
    LoopGeometry,
)
```

After cleanup, geometry.py should contain only:
- `compute_fk_chain`
- `Layout` class
- `build_layout`
- `compute_closure_metrics`

- [ ] **Step 2: Clean intersection.py**

Delete these dead items:
- `SwitchOpportunity` dataclass (lines 25-54)
- `IntersectionResult` dataclass (lines 57-76)
- `find_self_intersections` function (starts line 79, ~80 lines)
- `analyze_switch_connections` function (starts ~line 159, ~80 lines)
- `compute_port2_pose` function (starts ~line 242)
- `_is_in_switch` function (starts ~line 293)
- `_is_out_switch` function (starts ~line 298)
- `_get_divergent_port_pose` function (starts ~line 329)
- `_get_merge_port_pose` function (starts ~line 341)
- `count_path_crossings` function (starts ~line 496)

Keep only:
- `SWITCH_DIVERGE_ANGLE` constant
- `CROSS_90_INDEX` constant
- `find_crossing_pairs` function
- `count_segment_crossings` function
- `_segments_intersect` helper
- `_normalize_angle` helper
- `_is_switch` helper
- `_transform_local_to_world` helper

- [ ] **Step 3: Clean templates.py**

Delete:
- `LEFT_SIDING_REVERSE` (line ~110) and `RIGHT_SIDING_REVERSE` (line ~122) template definitions
- Remove entries for template indices 2 and 3 from the `TEMPLATES` dict
- `compute_branch_piece_count` function (line ~173)
- `is_valid_siding` function (line ~356)

Keep:
- `PassingSidingTemplate` dataclass
- `LEFT_SIDING`, `RIGHT_SIDING` templates and `TEMPLATES[0]`, `TEMPLATES[1]`
- `compute_branch_pieces`, `compute_required_main_distance`, `apply_fk`, `compute_branch_endpoint`
- `compute_out_switch_alignment_error` (called by `compute_required_main_distance`)
- `check_siding_inventory`, `get_siding_inventory_requirements`

- [ ] **Step 4: Clean lego_track_models.py**

Delete these dead items:
- All color constants not imported by visualization.py: `COL_STRAIGHT_BED`, `COL_CURVE_BED`, `COL_SWITCH_MAIN`, `COL_BRANCH_BED`, `COL_PORT_A`, `COL_PORT_B`, `COL_PORT_C`, `COL_DIM` (lines ~40-48)
- `FancyArrowPatch` import (line 26)
- Functions: `draw_track_bed`, `draw_rails`, `draw_port`, `draw_dim_line`, `setup_subplot`, `draw_straight`, `draw_curve_r40`, `draw_switch`
- The `if __name__ == "__main__"` block at the end

Keep only what visualization.py imports:
- `R40`, `RAIL_OFFSET`, `CURVE_ANGLE`, `SWITCH_LEN`, `ARC1_ANGLE`, `ARC2_ANGLE`
- `COL_RAIL`, `BED_LW`, `RAIL_LW`, `N_ARC_PTS`
- `arc_points`, `offset_path` functions

- [ ] **Step 5: Clean data.py dead methods**

Delete these unused methods from TrackCatalog:
- `validate_inventory` method
- `get_piece_class` method
- `get_topologies` method (returns all topologies — never called externally)
- `get_piece_role` method
- `can_pair` method
- `get_alternate_route` method
- `_build_role_tables` method and its output tables (`_fork_indices`, `_merge_indices`, `_crossing_indices`, `_compatible_pairs`)
- Remove `_build_role_tables()` call from `_build_tables()`

- [ ] **Step 6: Clean dead code from evaluation.py**

In `src/evaluation.py`:
- Delete `compute_objectives` function (lines 177-209)
- Delete `compute_constraints` function (lines 212-249)
- Delete `_compute_boundary_violation` helper (lines 252-265)
- Delete `_compute_inventory_excess` helper (lines 268-283)
- Delete `_compute_orphan_switches` helper (lines 286-296)
- Delete `from .config import BoundaryConfig` import (no longer needed)
- Keep only: `SpeedProfile`, `compute_speed_profile`, and the 4 helper functions

- [ ] **Step 7: Clean dead test code**

In `tests/test_evaluation.py`:
- Delete `TestObjectives` class (lines 81-120)
- Delete `TestConstraints` class (lines 123-199)
- Remove `compute_constraints, compute_objectives` from the import line
- Keep only: `TestSpeedProfile` class, `compute_speed_profile` import

In `tests/test_geometry.py`:
- Delete `TestLayoutGeometry` class (lines 200-232)
- Delete `TestAngularDeficit` class (lines 182-197)
- Remove dead imports: `compute_layout_geometry`, `estimate_angular_deficit` from geometry import
- Remove `from src.topology import LayoutChromosome, LoopDef`

- [ ] **Step 8: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass. Fewer tests than before (dead tests removed).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove ~1350 lines of dead code from src/ and tests/"
```

---

## Task 3: Create `src/types.py` and dissolve `topology.py`

Extract the live shared types from topology.py into a new types.py, then delete topology.py.

**Files:**
- Create: `src/types.py`
- Delete: `src/topology.py`
- Modify: `src/data.py` — change `from .topology import ...` to `from .types import ...`
- Modify: `src/decoder.py` — change `from .topology import ...` to `from .types import ...`
- Modify: `src/visualization.py` — change `from .topology import ...` to `from .types import ...`
- Modify: `tests/test_data.py` — change `from src.topology import ...` to `from src.types import ...`
- Modify: `tests/test_decoder.py` — change `from src.topology import ...` to `from src.types import ...`

- [ ] **Step 1: Create src/types.py**

Extract from topology.py (keeping only live types, all dead Phase 3/4 types are already deleted in Task 2... wait, they weren't deleted yet — topology.py wasn't touched in Task 2). Extract only the live types:

```python
"""Shared domain types for the LEGO Track Optimizer.

Pure data containers with no logic beyond property accessors.
These form the shared vocabulary across module tiers:
  - Tier 1 (domain core): catalog, geometry, train
  - Tier 2 (EA layer): decoder, operators, problem
  - Tier 3 (infra): visualization, config, main
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# PIECE CLASSIFICATION
# =============================================================================


class PieceClass(Enum):
    """Classify pieces by port topology."""

    SIMPLE_2PORT = "simple_2port"
    SWITCH_3PORT = "switch_3port"
    SWITCH_4PORT = "switch_4port"
    CROSSING_4PORT = "crossing_4port"
    BUMPER_1PORT = "bumper_1port"


@dataclass(frozen=True)
class FKRoute:
    """Single route through a piece (entry port -> exit port)."""

    entry_port: int
    exit_port: int
    dx: float
    dy: float
    dtheta: float
    arc_length: float = 0.0
    radius_mm: Optional[float] = None
    speed_limit: float = 1.57

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.dx, self.dy, self.dtheta], dtype=np.float64)

    @property
    def is_straight(self) -> bool:
        return abs(self.dtheta) < 0.01


@dataclass(frozen=True)
class PieceTopology:
    """Complete metadata for a piece (all phases)."""

    piece_id: str
    piece_index: int
    piece_class: PieceClass
    num_ports: int
    routes: Tuple[FKRoute, ...]
    default_route_idx: int = 0
    port_positions: Tuple[Tuple[float, float], ...] = ()
    port_headings: Tuple[float, ...] = ()

    def get_default_fk(self) -> FKRoute:
        return self.routes[self.default_route_idx]

    def get_route(self, entry_port: int, exit_port: int) -> Optional[FKRoute]:
        for route in self.routes:
            if route.entry_port == entry_port and route.exit_port == exit_port:
                return route
        return None

    def get_route_by_index(self, route_idx: int) -> FKRoute:
        if 0 <= route_idx < len(self.routes):
            return self.routes[route_idx]
        return self.routes[self.default_route_idx]

    def is_simple_through(self) -> bool:
        return len(self.routes) == 1

    def is_switch(self) -> bool:
        return self.piece_class in (PieceClass.SWITCH_3PORT, PieceClass.SWITCH_4PORT)

    def is_crossing(self) -> bool:
        return self.piece_class == PieceClass.CROSSING_4PORT

    def is_terminator(self) -> bool:
        return self.piece_class == PieceClass.BUMPER_1PORT


# =============================================================================
# SWITCH PAIR AND TRAVERSAL PATH
# =============================================================================


@dataclass
class SwitchPair:
    """Defines a paired IN/OUT switch creating a branch point."""

    pair_id: int
    in_position: int
    out_position: int
    in_switch_idx: int = -1
    out_switch_idx: int = -1
    branch_pieces: List[int] = field(default_factory=list)
    absorbed_positions: List[int] = field(default_factory=list)

    def is_valid(self) -> bool:
        return (
            self.in_position >= 0
            and self.out_position > self.in_position
            and self.in_switch_idx >= 0
            and self.out_switch_idx >= 0
        )

    @property
    def span(self) -> int:
        return self.out_position - self.in_position - 1


@dataclass
class TraversalPath:
    """A specific continuous route through the track topology."""

    path_id: int
    route_choices: Tuple[int, ...]
    piece_sequence: List[int] = field(default_factory=list)
    states: NDArray[np.float64] = field(default_factory=lambda: np.zeros((1, 3)))
    closure_error: float = 0.0
    angle_error: float = 0.0

    @property
    def n_pieces(self) -> int:
        return len(self.piece_sequence)

    @property
    def is_closed(self) -> bool:
        return self.closure_error < 1.0 and self.angle_error < 5.0

    def describe_route(self) -> str:
        if not self.route_choices:
            return "main"
        parts = []
        for i, choice in enumerate(self.route_choices):
            parts.append(f"SW{i}:{'branch' if choice else 'main'}")
        return ", ".join(parts)


@dataclass
class MultiPathLayout:
    """Complete track topology with all possible traversal paths."""

    main_loop_pieces: List[int] = field(default_factory=list)
    switch_pairs: List[SwitchPair] = field(default_factory=list)
    paths: List[TraversalPath] = field(default_factory=list)
    start_position: Tuple[float, float] = (0.0, 0.0)
    loose_port_count: int = 0
    secondary_closure_error: float = 0.0

    @property
    def n_switch_pairs(self) -> int:
        return len(self.switch_pairs)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    @property
    def all_paths_closed(self) -> bool:
        return all(path.is_closed for path in self.paths)

    @property
    def max_closure_error(self) -> float:
        if not self.paths:
            return 0.0
        return max(path.closure_error for path in self.paths)

    @property
    def max_angle_error(self) -> float:
        if not self.paths:
            return 0.0
        return max(path.angle_error for path in self.paths)

    @property
    def total_pieces(self) -> int:
        all_pieces = set(self.main_loop_pieces)
        for sp in self.switch_pairs:
            all_pieces.add(sp.in_switch_idx)
            all_pieces.add(sp.out_switch_idx)
            all_pieces.update(sp.branch_pieces)
        all_pieces.discard(-1)
        return len(all_pieces)

    def get_path_by_choices(self, choices: Tuple[int, ...]) -> Optional[TraversalPath]:
        for path in self.paths:
            if path.route_choices == choices:
                return path
        return None

    def get_main_path(self) -> Optional[TraversalPath]:
        choices = tuple(0 for _ in self.switch_pairs)
        return self.get_path_by_choices(choices)

    @property
    def n_pieces(self) -> int:
        main = self.get_main_path()
        n = main.n_pieces if main else len(self.main_loop_pieces)
        for sp in self.switch_pairs:
            n += len(sp.branch_pieces)
        return n

    @property
    def indices(self) -> NDArray[np.int32]:
        main = self.get_main_path()
        if main:
            return np.array(main.piece_sequence, dtype=np.int32)
        return np.array(self.main_loop_pieces, dtype=np.int32)

    @property
    def states(self) -> NDArray[np.float64]:
        main = self.get_main_path()
        return main.states if main else np.zeros((1, 3))

    @property
    def closure_error(self) -> float:
        return self.max_closure_error

    @property
    def angle_error(self) -> float:
        return self.max_angle_error

    def is_closed(self, pos_tol: float = 0.5, angle_tol: float = 5.0) -> bool:
        return self.max_closure_error < pos_tol and self.max_angle_error < angle_tol

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        all_x = []
        all_y = []
        for path in self.paths:
            if len(path.states) > 0:
                all_x.extend(path.states[:, 0])
                all_y.extend(path.states[:, 1])
        if not all_x:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(all_x), min(all_y), max(all_x), max(all_y))

    @property
    def area(self) -> float:
        min_x, min_y, max_x, max_y = self.bounding_box
        return (max_x - min_x) * (max_y - min_y)
```

- [ ] **Step 2: Update all imports from topology → types**

In `src/data.py`:
```python
# OLD: from .topology import FKRoute, PieceClass, PieceTopology
# NEW:
from .types import FKRoute, PieceClass, PieceTopology
```

In `src/decoder.py`:
```python
# OLD: from .topology import MultiPathLayout, SwitchPair, TraversalPath
# NEW:
from .types import MultiPathLayout, SwitchPair, TraversalPath
```

In `src/visualization.py`:
```python
# OLD: from .topology import MultiPathLayout
# NEW:
from .types import MultiPathLayout
```

In `tests/test_data.py`:
```python
# OLD: from src.topology import PieceClass
# NEW:
from src.types import PieceClass
```

In `tests/test_decoder.py`:
```python
# OLD: from src.topology import MultiPathLayout
# NEW:
from src.types import MultiPathLayout
```

- [ ] **Step 3: Delete topology.py**

```bash
rm src/topology.py
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/types.py src/data.py src/decoder.py src/visualization.py tests/test_data.py tests/test_decoder.py
git rm src/topology.py
git commit -m "refactor: dissolve topology.py into types.py shared domain vocabulary"
```

---

## Task 4: Create `train/` package

Convert flat `train.py` into a package and move speed profile computation from `evaluation.py` into it.

**Files:**
- Create: `src/train/__init__.py`
- Rename: `src/train.py` → `src/train/physics.py`
- Create: `src/train/scoring.py` (from evaluation.py live code)
- Delete: `src/evaluation.py`
- Modify: `src/config.py` — update import
- Modify: `src/problem.py` — update import
- Modify: `tests/conftest.py` — update import
- Modify: `tests/test_train.py` — update import
- Modify: `tests/test_evaluation.py` — update import, becomes test_scoring.py

**Important:** When converting `src/train.py` to `src/train/physics.py`, first rename (or copy+delete) the file, then create `src/train/__init__.py`. Git can't have both `src/train.py` and `src/train/` simultaneously.

- [ ] **Step 1: Create the train/ package structure**

First, save train.py content, delete it, create directory:
```bash
# On Windows, train.py must be removed before train/ directory can exist
mv src/train.py src/_train_tmp.py
mkdir -p src/train
mv src/_train_tmp.py src/train/physics.py
```

- [ ] **Step 2: Create src/train/scoring.py**

Move `SpeedProfile` and `compute_speed_profile` (plus the 4 helper functions) from `evaluation.py`:

```python
"""Speed profile computation for track layouts.

Time-optimal speed profiling using a 3-pass algorithm with friction ellipse
constraints. This is the physics scoring module — it turns geometry into speed.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .physics import TrainConfig, available_accel, v_eff_array, DEFAULT_TRAIN_CONFIG

# These imports come from sibling Tier 1 modules:
from ..catalog import TrackCatalog
from ..geometry import Layout


@dataclass
class SpeedProfile:
    """Time-optimal speed profile along track layout."""

    speeds: NDArray[np.float64]
    avg_speed: float
    lap_time: float
    total_distance: float
    max_speed: float
    min_speed: float


def compute_speed_profile(
    layout: Layout,
    catalog: TrackCatalog,
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> SpeedProfile:
    """Compute time-optimal speed profile using 3-pass algorithm.

    Args:
        layout: Track layout with geometry
        catalog: Track catalog for piece properties
        train_config: Locomotive physics (default: DEFAULT_TRAIN_CONFIG)

    Returns:
        SpeedProfile with speeds, avg_speed, lap_time, etc.
    """
    if layout.n_pieces == 0:
        return SpeedProfile(
            speeds=np.array([]),
            avg_speed=0.0,
            lap_time=0.0,
            total_distance=0.0,
            max_speed=0.0,
            min_speed=0.0,
        )

    stud_to_m = catalog.stud_mm / 1000.0
    arc_lengths = catalog.get_arc_lengths(layout.indices) * stud_to_m
    radii_m = catalog.get_radii(layout.indices) / 1000.0
    speed_limits = catalog.get_speed_limits(layout.indices)

    v_curve = v_eff_array(train_config, radii_m)
    v_limit = np.minimum(v_curve, speed_limits)

    is_closed = layout.is_closed(pos_tol=1.0, angle_tol=10.0)
    speeds = (
        _compute_speeds_double_unroll(v_limit, arc_lengths, radii_m, train_config)
        if is_closed
        else _compute_speeds_open(v_limit, arc_lengths, radii_m, train_config)
    )

    total_distance = float(np.sum(arc_lengths))
    lap_time = _compute_lap_time(speeds, arc_lengths)
    avg_speed = total_distance / lap_time if lap_time > 0 else 0.0

    return SpeedProfile(
        speeds=speeds,
        avg_speed=avg_speed,
        lap_time=lap_time,
        total_distance=total_distance,
        max_speed=float(np.max(speeds)) if len(speeds) > 0 else 0.0,
        min_speed=float(np.min(speeds)) if len(speeds) > 0 else 0.0,
    )


def _compute_speeds_double_unroll(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    n = len(v_limit)
    v_limit_double = np.concatenate([v_limit, v_limit])
    arc_lengths_double = np.concatenate([arc_lengths, arc_lengths])
    radii_double = np.concatenate([radii_m, radii_m])

    v_fwd = _forward_pass(v_limit_double, arc_lengths_double, radii_double, train_config)
    v_bwd = _backward_pass(v_fwd, arc_lengths_double, radii_double, train_config)

    return v_bwd[n : 2 * n]


def _compute_speeds_open(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    v_fwd = _forward_pass(v_limit, arc_lengths, radii_m, train_config)
    return _backward_pass(v_fwd, arc_lengths, radii_m, train_config)


def _forward_pass(
    v_limit: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    n = len(v_limit)
    v_fwd = np.zeros(n)
    v_fwd[0] = v_limit[0]

    for i in range(1, n):
        a_max = available_accel(train_config, float(v_fwd[i - 1]), float(radii_m[i - 1]))
        v_accel = np.sqrt(v_fwd[i - 1] ** 2 + 2 * a_max * arc_lengths[i - 1])
        v_fwd[i] = min(v_limit[i], v_accel)

    return v_fwd


def _backward_pass(
    v_fwd: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    train_config: TrainConfig,
) -> NDArray[np.float64]:
    n = len(v_fwd)
    v_bwd = np.zeros(n)
    v_bwd[-1] = v_fwd[-1]

    for i in range(n - 2, -1, -1):
        a_brake = available_accel(
            train_config, float(v_bwd[i + 1]), float(radii_m[i]), is_braking=True
        )
        v_brake = np.sqrt(v_bwd[i + 1] ** 2 + 2 * a_brake * arc_lengths[i])
        v_bwd[i] = min(v_fwd[i], v_brake)

    return v_bwd


def _compute_lap_time(speeds: NDArray[np.float64], arc_lengths: NDArray[np.float64]) -> float:
    safe_speeds = np.where(speeds > 0, speeds, 0.001)
    return float(np.sum(arc_lengths / safe_speeds))
```

**Note on imports:** `scoring.py` imports from `..catalog` and `..geometry` which won't exist yet (catalog/ package is Task 5). For now, use the current paths:
```python
# Temporary until catalog/ package exists (Task 5):
from ..data import TrackCatalog
from ..geometry import Layout
```

- [ ] **Step 3: Create src/train/__init__.py**

```python
"""Train physics package — lateral stability, speed profiling, scoring.

Public API:
    TrainConfig — immutable locomotive physics parameters
    DEFAULT_TRAIN_CONFIG — default TrainConfig instance
    v_eff_array — vectorized speed cap over radius array
    available_accel — friction-ellipse longitudinal acceleration
    SpeedProfile — time-optimal speed profile result
    compute_speed_profile — 3-pass speed profiling algorithm
"""

from .physics import (
    DEFAULT_TRAIN_CONFIG,
    TrainConfig,
    available_accel,
    v_eff_array,
)
from .scoring import SpeedProfile, compute_speed_profile

__all__ = [
    "DEFAULT_TRAIN_CONFIG",
    "TrainConfig",
    "available_accel",
    "v_eff_array",
    "SpeedProfile",
    "compute_speed_profile",
]
```

- [ ] **Step 4: Delete src/evaluation.py**

After moving speed profile code to `train/scoring.py`, `evaluation.py` is empty (dead code already removed in Task 2). Delete it:

```bash
rm src/evaluation.py
```

- [ ] **Step 5: Update all imports**

In `src/config.py`:
```python
# OLD: from .train import TrainConfig
# NEW:
from .train import TrainConfig
# (No change needed — train/ package __init__.py re-exports TrainConfig)
```

In `src/problem.py`:
```python
# OLD: from .evaluation import compute_speed_profile
# NEW:
from .train import compute_speed_profile
```

In `tests/conftest.py`:
```python
# OLD: from src.train import TrainConfig
# NEW:
from src.train import TrainConfig
# (No change needed — train/ package re-exports)
```

In `tests/test_train.py`:
```python
# OLD: from src.train import DEFAULT_TRAIN_CONFIG, TrainConfig, v_eff_array
# NEW:
from src.train import DEFAULT_TRAIN_CONFIG, TrainConfig, v_eff_array
# (No change needed — train/ package re-exports)
```

In `tests/test_evaluation.py` (rename to `tests/test_scoring.py`):
```python
# OLD: from src.evaluation import compute_speed_profile
# NEW:
from src.train import compute_speed_profile
```
Also rename the file:
```bash
mv tests/test_evaluation.py tests/test_scoring.py
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/train/ tests/test_scoring.py src/problem.py
git rm src/train.py src/evaluation.py tests/test_evaluation.py
git commit -m "refactor: create train/ package with physics and scoring modules"
```

---

## Task 5: Create `catalog/` package

Convert flat `data.py` into a `catalog/` package with pieces and catalog modules.

**Files:**
- Create: `src/catalog/__init__.py`
- Create: `src/catalog/pieces.py` (FKDeltas, Port, TrackPiece from data.py)
- Create: `src/catalog/catalog.py` (TrackCatalog class from data.py)
- Delete: `src/data.py`
- Modify: `src/train/scoring.py` — update import from `..data` to `..catalog`
- Modify: `src/geometry.py` — update import
- Modify: `src/decoder.py` — update import
- Modify: `src/problem.py` — update import
- Modify: `src/sampling.py` — update import
- Modify: `src/visualization.py` — update import
- Modify: `src/__init__.py` — update import
- Modify: `main.py` — update import
- Modify: `tests/conftest.py` — update import
- Modify: `tests/test_data.py` — update import (rename to test_catalog.py)
- Modify: `tests/test_decoder.py` — update import
- Modify: `tests/test_geometry.py` — update import
- Modify: `tests/test_visualization.py` — update import (in tests/ dir)
- Modify: `test_visualization.py` — update import (repo root copy)

- [ ] **Step 1: Create catalog/ directory and split data.py**

```bash
mv src/data.py src/_data_tmp.py
mkdir -p src/catalog
```

Create `src/catalog/pieces.py` — extract the dataclasses (lines 1-78 of data.py):

```python
"""Track piece data types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..types import PieceClass, PieceTopology


@dataclass(frozen=True)
class FKDeltas:
    """Forward kinematics deltas for a single piece."""

    dx: float
    dy: float
    dtheta: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.dx, self.dy, self.dtheta], dtype=np.float64)


@dataclass(frozen=True)
class Port:
    """Connection point on a track piece."""

    x: float
    y: float
    heading: float
    gender: str


@dataclass
class TrackPiece:
    """Single track piece definition."""

    id: str
    name: str
    piece_type: str
    fk: FKDeltas
    ports: Tuple[Port, ...]
    index: int
    length: float = 16.0
    radius: Optional[float] = None
    angle: Optional[float] = None
    direction: Optional[str] = None
    radius_mm: Optional[float] = None
    speed_limit_ms: float = 1.57
    is_terminator: bool = False
    routes_data: Optional[List[Dict[str, Any]]] = None

    @property
    def is_straight(self) -> bool:
        return self.piece_type == "straight"

    @property
    def is_curve(self) -> bool:
        return self.piece_type == "curve"

    @property
    def arc_length(self) -> float:
        if self.is_straight:
            return self.length
        elif self.is_curve and self.radius and self.angle:
            return self.radius * math.radians(abs(self.angle))
        else:
            return self.length
```

Create `src/catalog/catalog.py` — the TrackCatalog class (the bulk of data.py starting at line 80):

```python
"""Track catalog: registry of track pieces with vectorized access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from numpy.typing import NDArray

from ..types import FKRoute, PieceClass, PieceTopology
from .pieces import FKDeltas, Port, TrackPiece


class TrackCatalog:
    """Manages track piece inventory with vectorized access."""
    # ... (entire TrackCatalog class from data.py, unchanged except import paths)
```

Copy the entire TrackCatalog class body from `src/_data_tmp.py` into this file unchanged.

- [ ] **Step 2: Create src/catalog/__init__.py**

```python
"""Track piece catalog package.

Public API:
    TrackCatalog — piece registry with vectorized FK/radius/speed lookup
    TrackPiece — single piece definition
    FKDeltas — forward kinematics delta values
    Port — connection point on a piece
"""

from .catalog import TrackCatalog
from .pieces import FKDeltas, Port, TrackPiece

__all__ = [
    "TrackCatalog",
    "TrackPiece",
    "FKDeltas",
    "Port",
]
```

- [ ] **Step 3: Delete src/_data_tmp.py**

```bash
rm src/_data_tmp.py
```

- [ ] **Step 4: Update all imports across codebase**

Every file that did `from .data import ...` or `from src.data import ...` needs updating.

In `src/` files — change `.data` to `.catalog`:
- `src/geometry.py`: `from .data import TrackCatalog` → `from .catalog import TrackCatalog`
- `src/decoder.py`: `from .data import TrackCatalog` → `from .catalog import TrackCatalog`
- `src/problem.py`: `from .data import TrackCatalog` → `from .catalog import TrackCatalog`
- `src/sampling.py`: `from .data import TrackCatalog` → `from .catalog import TrackCatalog`
- `src/visualization.py`: `from .data import TrackCatalog` → `from .catalog import TrackCatalog`
- `src/train/scoring.py`: `from ..data import TrackCatalog` → `from ..catalog import TrackCatalog`

In `src/__init__.py`:
```python
from .catalog import TrackCatalog, TrackPiece
```

In `main.py`:
```python
# OLD: from src.data import TrackCatalog
# NEW:
from src.catalog import TrackCatalog
```

In test files:
- `tests/conftest.py`: `from src.data import TrackCatalog` → `from src.catalog import TrackCatalog`
- `tests/test_data.py`: `from src.data import FKDeltas, TrackCatalog, TrackPiece` → `from src.catalog import FKDeltas, TrackCatalog, TrackPiece`
- `tests/test_decoder.py`: `from src.data import TrackCatalog` → `from src.catalog import TrackCatalog`
- `tests/test_geometry.py`: `from src.data import TrackCatalog` → `from src.catalog import TrackCatalog`
- `tests/test_visualization.py`: `from src.data import TrackCatalog` → `from src.catalog import TrackCatalog`
- `test_visualization.py` (repo root): `from src.data import TrackCatalog` → `from src.catalog import TrackCatalog`

Rename test file:
```bash
mv tests/test_data.py tests/test_catalog.py
```

**Note:** `Archive/export/` files also import `src.data` but are archived — leave them as-is (they'll break but are not part of the live codebase).

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/catalog/ tests/test_catalog.py src/geometry.py src/decoder.py src/problem.py src/sampling.py src/visualization.py src/__init__.py src/train/scoring.py main.py tests/conftest.py tests/test_decoder.py tests/test_geometry.py
git rm src/data.py tests/test_data.py
git commit -m "refactor: create catalog/ package from data.py"
```

---

## Task 6: Final verification and CLAUDE.md update

Verify everything works end-to-end, update documentation.

**Files:**
- Modify: `CLAUDE.md` — update Project Structure section

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass.

- [ ] **Step 2: Run optimizer smoke test (default config)**

Run: `python main.py --config configs/default.yaml --verbose` with reduced generations (add `--n-gen 5` or edit config temporarily).

Verify it starts, runs a few generations, and produces output without import errors.

- [ ] **Step 3: Verify import structure**

Run a quick dependency check:
```bash
# Verify train/ has no pymoo imports
grep -r "import pymoo\|from pymoo" src/train/

# Verify catalog/ has no pymoo imports
grep -r "import pymoo\|from pymoo" src/catalog/

# Verify types.py has no internal imports (only stdlib + numpy)
grep "from \.\|import src" src/types.py

# Verify no remaining references to old modules
grep -r "from.*topology import\|from.*evaluation import\|from.*data import" src/ tests/ main.py
```

Expected: No matches for any of these.

- [ ] **Step 4: Update CLAUDE.md Project Structure section**

Update the file table to reflect the new structure:

```markdown
### Core Code (`/src`)

| File/Package | Purpose |
|------|---------|
| `types.py` | Shared domain types: `SwitchPair`, `TraversalPath`, `MultiPathLayout`, `PieceClass`, `FKRoute`, `PieceTopology` |
| `catalog/` | `TrackCatalog`, `TrackPiece`, `FKDeltas`, YAML loader |
| `train/` | `TrainConfig`, `SpeedProfile`, `compute_speed_profile`, lateral stability physics |
| `geometry.py` | `Layout`, `build_layout()`, `compute_fk_chain()` |
| `problem.py` | `TrackOptimizationProblem` — bi-objective NSGA-II problem |
| `decoder.py` | Construction-based decoder for partitioned chromosomes |
| `encoding.py` | Partitioned chromosome encoding, `PartitionedDimensions` |
| `operators.py` | `PartitionedCrossover`, `PartitionedMutation` |
| `repair.py` | `TrackRepairPipeline` for fixing chromosomes |
| `sampling.py` | `IntegerSampling` — seeds with valid closed loops |
| `templates.py` | Template-based passing siding definitions |
| `intersection.py` | Crossing detection for self-intersecting layouts |
| `config.py` | Pydantic models: `OptimizationConfig`, `BoundaryConfig` |
| `visualization.py` | `plot_layout()`, `plot_multi_path_layout()`, `plot_pareto_front()` |
| `lego_track_models.py` | Geometry constants for visualization rendering |
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new modular architecture"
```

---

## Summary of structural changes

### Before
```
src/
├── topology.py          695 lines (catch-all types, half dead)
├── data.py              576 lines (catalog + pieces)
├── evaluation.py        296 lines (mixed physics + objectives + constraints)
├── train.py             172 lines (physics only)
├── geometry.py          474 lines (half dead Phase 3/4)
├── intersection.py      571 lines (half dead)
├── ... (11 more flat files)
```

### After
```
src/
├── types.py             ~250 lines (shared domain vocabulary)
├── catalog/
│   ├── __init__.py      (re-exports)
│   ├── pieces.py        ~70 lines (FKDeltas, Port, TrackPiece)
│   └── catalog.py       ~400 lines (TrackCatalog)
├── train/
│   ├── __init__.py      (re-exports)
│   ├── physics.py       ~170 lines (TrainConfig, v_eff_array)
│   └── scoring.py       ~150 lines (SpeedProfile, compute_speed_profile)
├── geometry.py          ~160 lines (FK chain, Layout, build_layout)
├── intersection.py      ~250 lines (crossing detection only)
├── ... (8 more flat files, unchanged)
```

### Dependency tiers (enforced)
```
Tier 1 (no pymoo): types.py → catalog/ → geometry.py → train/
Tier 2 (pymoo):    encoding.py, decoder.py, operators.py, repair.py, sampling.py, problem.py
Tier 3 (infra):    config.py, visualization.py, main.py
```

### Dead code removed: ~1,350 lines
### Broken tests fixed: 3 files (test_problem, test_decoder, test_sampling)
### Dead tests removed: TestObjectives, TestConstraints, TestLayoutGeometry, TestAngularDeficit
