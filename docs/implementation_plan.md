# Implementation Plan: Multi-Segment GA Architecture

## Overview

This plan transforms the LEGO Track Optimizer from a flat chromosome encoding to a multi-segment CGP-inspired architecture with a 4-phase construction decoder.

**Key Design Decisions (from user clarification):**
- Segment sizes: Use document defaults (n_var = 112)
- Phase 1 scope: Main loop only, phases 2-4 as stubs
- Start position: Keep 2 position floats at chromosome end
- Module structure: Create new `decoder.py` module

---

## Phase 1: Core System

### 1.1 Define Segment Constants and Helpers

**File: `src/encoding.py` (NEW)**

```python
# Segment sizes (from research doc)
L_MAX = 30      # Main loop positions
S_MAX = 30      # Switch mask positions (mirrors main loop)
B_MAX = 4       # Number of branch slots
B_SLOT = 10     # Genes per branch slot: [src_switch, piece×8, rejoin_target]
C_MAX = 5       # Crossing overlay pairs (2 genes each = 10)

# Computed dimensions
N_PIECE_GENES = L_MAX + S_MAX + (B_MAX * B_SLOT) + (C_MAX * 2)  # 110
N_POSITION_GENES = 2  # start_x, start_y
N_VAR = N_PIECE_GENES + N_POSITION_GENES  # 112

# Segment slicing functions
def get_main_loop(x): return x[0:L_MAX]
def get_switch_mask(x): return x[L_MAX:L_MAX+S_MAX]
def get_branch_slots(x): return x[L_MAX+S_MAX:L_MAX+S_MAX+(B_MAX*B_SLOT)]
def get_crossing_overlay(x): return x[L_MAX+S_MAX+(B_MAX*B_SLOT):N_PIECE_GENES]
def get_start_position(x): return x[N_PIECE_GENES:N_VAR]
```

**Tasks:**
- [ ] Create `src/encoding.py` with segment constants
- [ ] Add segment slicing functions
- [ ] Add segment-specific bounds generation (`xl`, `xu` arrays)
- [ ] Add helper to validate chromosome structure

---

### 1.2 Implement 4-Phase Decoder (Main Loop Only for Phase 1)

**File: `src/decoder.py` (NEW)**

The decoder translates an integer chromosome into a `Layout` object. Phase 1 implements main loop construction; phases 2-4 are stubs.

**Phase 1 - Main Loop Construction:**
1. Process genes in `main_loop[0:L_MAX]` left-to-right
2. For each gene:
   - Skip if gene == -1 (inactive)
   - Skip if piece type exhausted (inventory check)
   - Skip if cumulative angle exceeds budget (angular budget check)
   - Apply FK: turtle advances by (dx, dy, dθ)
3. Check closure when cumulative angle ≈ 360° and position ≈ origin
4. Return `Layout` object with states and closure metrics

```python
def decode_chromosome(
    chromosome: NDArray[np.int16],
    catalog: TrackCatalog,
    inventory: Dict[str, int],
    closure_tol: float = 8.0,
    angle_tol: float = 15.0,
) -> Layout:
    """Decode chromosome into Layout via 4-phase construction.

    Phase 1 only: Main loop construction.
    Phases 2-4: Stubs returning empty lists.
    """
```

**Tasks:**
- [ ] Create `src/decoder.py`
- [ ] Implement `decode_chromosome()` with Phase 1 logic
- [ ] Implement inventory tracking during construction
- [ ] Implement angular budget checking
- [ ] Add collision detection (optional for Phase 1)
- [ ] Add stub functions for phases 2-4
- [ ] Unit tests for decoder

---

### 1.3 Implement Basic Mutation Operators

**File: `src/operators.py` (NEW)**

Create `TrackLayoutMutation(Mutation)` class with three operators:

**MUTATE (p=0.30):** Replace a random active gene with another valid piece index.
```python
def _mutate_swap(self, x: NDArray) -> NDArray:
    # Find active positions (gene >= 0) in main loop
    # Select random position, replace with valid piece index
    # Use angle-budget-aware candidate selection
```

**ADD (p=0.20):** Insert a new piece, shift downstream genes right.
```python
def _mutate_add(self, x: NDArray) -> NDArray:
    # Select insertion position (bias toward end)
    # Shift genes right (truncate last if necessary)
    # Insert random piece from available inventory
```

**DELETE (p=0.10):** Remove a piece, shift genes left.
```python
def _mutate_delete(self, x: NDArray) -> NDArray:
    # Select active position to remove
    # Shift downstream genes left
    # Fill last position with -1
```

**BRANCH (p=0.20):** Stub returning unchanged chromosome.

**COMPOUND (p=0.20):** Stub returning unchanged chromosome.

**Tasks:**
- [ ] Create `src/operators.py`
- [ ] Implement `TrackLayoutMutation(Mutation)` class
- [ ] Implement MUTATE operator
- [ ] Implement ADD operator
- [ ] Implement DELETE operator
- [ ] Add probability-weighted operator selection
- [ ] Add BRANCH stub
- [ ] Add COMPOUND stub
- [ ] Unit tests for mutation operators

---

### 1.4 Implement NoOpCrossover

**File: `src/operators.py`**

```python
class NoOpCrossover(Crossover):
    """Identity crossover returning parents unchanged.

    Used in Phase 1 mutation-only evolution.
    """
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, X, **kwargs):
        return X.copy()
```

**Tasks:**
- [ ] Implement `NoOpCrossover(Crossover)` class
- [ ] Unit test for NoOpCrossover

---

### 1.5 Update TrackLayoutProblem

**File: `src/problem.py`**

Changes:
1. Set `n_var = 112` (from encoding constants)
2. Set `n_obj = 1` (single fitness with soft penalties)
3. Set `n_ieq_constr = 2` (position closure, angle closure)
4. Use segment-specific bounds from `encoding.py`
5. Call `decode_chromosome()` instead of `build_layout()`
6. Compute fitness: `F = pieces_used × 1000 - soft_penalties`
7. Compute constraints: `G_pos`, `G_angle` with ε-relaxation

```python
def _evaluate(self, x, out, *args, **kwargs):
    # Decode chromosome
    layout = decode_chromosome(x, self.catalog, self.inventory)

    # Compute single objective with soft penalties
    pieces_used = layout.n_pieces
    boundary_penalty = compute_boundary_penalty(layout, self.boundary)
    collision_penalty = compute_collision_penalty(layout)

    fitness = pieces_used * 1000 - boundary_penalty - collision_penalty
    out["F"] = [-fitness]  # Minimize negative = maximize

    # Compute closure constraints with ε-relaxation
    out["G"] = [
        max(0, layout.closure_error - 8.0),   # G_pos
        max(0, layout.angle_error - 15.0),    # G_angle
    ]
```

**Tasks:**
- [ ] Update `TrackLayoutProblem.__init__()` with new n_var, n_obj, n_ieq_constr
- [ ] Generate segment-specific bounds
- [ ] Update `_evaluate()` to use decoder
- [ ] Implement single-objective fitness computation
- [ ] Implement 2-constraint closure checks with ε-relaxation
- [ ] Add soft penalty functions for boundary/collision

---

### 1.6 Update HeuristicSampling

**File: `src/sampling.py`**

Changes:
1. Generate chromosomes in new multi-segment format
2. Reduce heuristic ratio from 51% to 15%
3. Heuristic patterns fill main_loop segment only
4. Random chromosomes decoded by construction decoder

```python
class HeuristicSampling(Sampling):
    HEURISTIC_RATIO = 0.15  # 15% heuristic, 85% random

    def _do(self, problem, n_samples, **kwargs) -> NDArray:
        n_heuristic = int(n_samples * self.HEURISTIC_RATIO)
        n_random = n_samples - n_heuristic

        heuristic = self._generate_heuristic_chromosomes(n_heuristic)
        random_samples = self._generate_random_chromosomes(n_random)

        return np.vstack([heuristic, random_samples])
```

**Tasks:**
- [ ] Update chromosome format to multi-segment
- [ ] Reduce heuristic ratio to 15%
- [ ] Update pattern generation for main_loop segment
- [ ] Update random chromosome generation
- [ ] Add random start position generation

---

### 1.7 Implement Basic Repair Operators

**File: `src/repair.py` (NEW)**

**Closure Repair:** Attempt to fix unclosed layouts.
```python
def closure_repair(x: NDArray, catalog: TrackCatalog) -> NDArray:
    """Swap last N pieces to achieve closure.

    Simple implementation for Phase 1:
    1. Compute angular deficit
    2. Try swapping last 1-5 pieces with closing sequence
    3. Return modified chromosome (Baldwinian: only evaluate repaired)
    """
```

**Inventory Repair:** Remove excess pieces.
```python
def inventory_repair(x: NDArray, inventory: Dict[str, int]) -> NDArray:
    """Remove pieces exceeding inventory limits.

    For each piece type over limit:
    - Find positions using that piece
    - Replace excess with -1 (prefer later positions)
    """
```

**Tasks:**
- [ ] Create `src/repair.py`
- [ ] Implement basic closure repair
- [ ] Implement inventory repair
- [ ] Create `TrackRepairPipeline(Repair)` combining repairs
- [ ] Unit tests for repair operators

---

### 1.8 Update main.py

**File: `main.py`**

Changes:
1. Import new modules (encoding, decoder, operators)
2. Replace MixedTypeCrossover with NoOpCrossover
3. Replace PM mutation with TrackLayoutMutation
4. Update algorithm creation
5. Update result interpretation for single objective

**Tasks:**
- [ ] Update imports
- [ ] Replace crossover with NoOpCrossover
- [ ] Replace mutation with TrackLayoutMutation
- [ ] Update algorithm configuration
- [ ] Update result saving for single objective
- [ ] Add brief test run capability (20 gen, pop=20)

---

### 1.9 Integration Testing

**Tasks:**
- [ ] Create test config with small inventory
- [ ] Run 20 generations with pop_size=20
- [ ] Verify: no crashes
- [ ] Verify: Layout objects created correctly
- [ ] Verify: Fitness values computed
- [ ] Verify: Constraints reported correctly
- [ ] Verify: At least some feasible solutions found

---

## File Summary: Phase 1

| File | Action | Lines (est.) |
|------|--------|--------------|
| `src/encoding.py` | NEW | ~100 |
| `src/decoder.py` | NEW | ~300 |
| `src/operators.py` | NEW | ~400 |
| `src/repair.py` | NEW | ~150 |
| `src/problem.py` | MODIFY | ~50 changed |
| `src/sampling.py` | MODIFY | ~100 changed |
| `main.py` | MODIFY | ~50 changed |

---

## Phase 2: Branches and ε-Constraints (Future)

After Phase 1 verification:
- Implement decoder phases 2-3 (switch placement, branch construction)
- Implement BRANCH operator (4 sub-operators)
- Implement branch rejoin repair and switch pairing repair
- Add `AdaptiveEpsilonConstraintHandling`
- Implement decoder phase 4 (crossing overlay)
- Implement COMPOUND operator
- Extend visualization for branches

---

## Phase 3: Adaptive Selection and Crossover (Future)

After Phase 2 verification:
- Implement AOS controller (`TrackMutationAOS`)
- Implement segment-selective crossover (`SegmentSelectiveCrossover`)
- Add BRKGA mutant injection
- Add aging evolution
- Add external elite archive
- Add stagnation detection and response

---

## Critical Constraints

1. **Reuse FK engine** - `geometry.py`'s `compute_fk_chain()` is unchanged
2. **Reuse TrackCatalog** - `data.py` consumed as-is
3. **Reuse physics** - `evaluation.py`'s speed profile unchanged
4. **pymoo compatibility** - All operators subclass pymoo base classes
5. **Integer chromosome** - `int16` dtype, bounds via `xl`/`xu`
6. **No breaking data files** - `track_pieces.yaml` and configs unchanged
7. **Each phase runs independently** - Phase 1 must work before Phase 2

---

## Success Criteria for Phase 1

1. Optimization runs without crashes
2. Layouts are generated with correct FK geometry
3. Single objective fitness computed correctly
4. Position/angle closure constraints reported
5. Mutation operators modify chromosomes correctly
6. At least 10% feasible solutions in final population
7. Results saved and visualized correctly

