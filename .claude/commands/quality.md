---
description: Deep Python & pymoo quality review. Rewrites code to be beautiful, idiomatic, and production-grade against this project's standards.
---

# /quality - Python & pymoo Quality Gate

Deep code quality review against this project's established standards. Not a checklist — this skill reads the code like a senior engineer would and proposes rewrites that make it beautiful, idiomatic, and correct.

## Arguments

| Argument | Description |
|----------|-------------|
| `<file>` | Review a specific file (e.g., `src/problem.py`) |
| `<file1> <file2>` | Review multiple files |
| `--fix` | Auto-apply fixes (edit files directly) |
| `--diff` | Show proposed changes as diffs without applying |

## Examples

```
/quality src/survival.py              # Review one file
/quality src/operators.py src/repair.py  # Review multiple
/quality src/decoder.py --fix         # Review and auto-fix
/quality src/evaluation.py --diff     # Show diffs only
```

## Execution

### 1. Read the target files fully

Read each file completely. You need full context — not just diffs.

### 2. Evaluate against this project's quality bar

This codebase follows specific patterns. Review code against ALL of these dimensions:

---

#### A. Python Craftsmanship

**Structure & Flow**
- Early returns to reduce nesting — no deeply nested if/else pyramids
- Single responsibility: each function does ONE thing
- Functional decomposition: large functions broken into composable parts
- No dead code, commented-out blocks, or TODO placeholders left behind

**Naming**
- Functions: `verb_noun` — `compute_fk_chain`, `decode_chromosome`, `count_balanced_switch_pairs`
- Classes: descriptive nouns — `SpeedProfile`, `TrackCatalog`, `StructuralNichingSurvival`
- Variables: meaningful, not abbreviated — `closure_error` not `ce`, `feasible_mask` not `fm`
- Boolean variables: `is_*`, `has_*` — `is_closed`, `has_complex_pieces`
- Constants: `UPPER_SNAKE` — `RK_INACTIVE_THRESHOLD`, `MAIN_LOOP_START`

**Type Annotations**
- All public function signatures must have type hints
- Use `NDArray[np.float64]` for numpy arrays (from `numpy.typing`)
- Use `tuple[...]` / `list[...]` / `dict[...]` for collections
- Return types always specified

**Docstrings (Google style)**
```python
def function(param1: int, param2: NDArray) -> float:
    """Brief one-line description.

    Longer explanation if needed. Reference algorithms or papers.

    Args:
        param1: Description.
        param2: Description with shape info like (n, 3).

    Returns:
        Description of return value.

    Raises:
        ValueError: When and why.
    """
```

**Imports**
- Organized: stdlib, then third-party, then local (relative `.` imports)
- No wildcard imports
- No unused imports
- Group related imports from same module

---

#### B. NumPy / Scientific Computing

**Vectorization is mandatory**
- Replace Python loops over arrays with numpy operations
- Use `np.where`, `np.maximum`, `np.minimum` instead of element-wise if/else
- Use boolean indexing: `values[mask]` not `[v for v, m in zip(values, mask) if m]`
- Broadcasting over explicit loops for pairwise operations

**Array discipline**
- Always `.copy()` when mutating a slice that shouldn't affect the original
- Specify `dtype` when creating arrays: `np.zeros(n, dtype=np.float64)`
- Use `np.errstate` for operations that may produce warnings (divide by zero, log of zero)
- Shape comments on non-obvious arrays: `# (n_pop, n_var)`

**Performance patterns**
- Pre-allocate arrays, don't append in loops
- Use `out=` parameter for in-place operations in hot paths
- Avoid creating temporary arrays in tight loops
- `np.linalg.norm` over manual `sqrt(sum(x**2))`

---

#### C. pymoo Framework Compliance

**Problem definition**
- `ElementwiseProblem` for single-solution evaluation with `_evaluate(self, x, out, *args, **kwargs)`
- `Problem` (vectorized) for batch evaluation with `_evaluate(self, X, out, *args, **kwargs)`
- `out["F"]` for objectives — ALL minimized (negate for maximization)
- `out["G"]` for inequality constraints — `g <= 0` is feasible
- `super().__init__(n_var=, n_obj=, n_ieq_constr=, xl=, xu=)` — all params explicit

**Genetic operators**
- `Crossover._do(self, problem, X, **kwargs)` — X shape `(n_matings, n_parents, n_var)`, return `(n_matings, n_offsprings, n_var)`
- `Mutation._do(self, problem, X, **kwargs)` — X shape `(n_solutions, n_var)`, return same shape
- `Sampling._do(self, problem, n_samples, **kwargs)` — return `(n_samples, n_var)`
- `Repair._do(self, problem, X, **kwargs)` — X shape `(n_solutions, n_var)`, return repaired
- `Survival._do(self, problem, pop, *args, n_survive=None, **kwargs)` — return Population

**Constraint normalization (this project's convention)**
```python
G[0] = (closure_error - tolerance) / tolerance      # Normalized to [-1, +inf)
G[1] = (angle_error - angle_tolerance) / angle_tolerance
G[2] = boundary_violation / diagonal                 # Normalized by scale
G[3] = inventory_excess                              # Already 0 or positive int
G[4] = loose_port_count                              # 0 = all connected
```

---

#### D. Project-Specific Patterns

**Random-Key encoding**
- All chromosome genes in `[0.0, 1.0]`
- Decode via `rk_to_piece_index(gene, available_pieces)`
- Inactive if `gene < RK_INACTIVE_THRESHOLD` (~0.0714)
- Encode via `piece_index_to_rk(piece_idx, available_pieces)`

**Forward kinematics state**
- `[x, y, theta]` — always degrees for theta
- FK chain: cumulative states `(n+1, 3)` including start state

**Decode-Modify-Re-encode pattern for repairs**
- Never directly manipulate RK values as piece indices
- Decode to get piece sequence, modify, re-encode

**Module docstrings**
- Every file starts with a module-level docstring explaining purpose and key contents
- Lists main classes/functions and their roles

---

### 3. Classify findings

For each issue found, classify as:

- **REWRITE**: Code that works but is ugly, non-idiomatic, or hard to follow. Propose a cleaner version.
- **BUG**: Incorrect behavior — wrong shapes, missing copies, broken interfaces.
- **PERF**: Working code that's unnecessarily slow — loops that should be vectorized, redundant allocations.
- **STYLE**: Naming, imports, docstrings, type hints — things that affect readability.

### 4. Output

```
## Quality Review: <file(s)>

### Summary
One paragraph: overall quality assessment and the most impactful change.

### Findings

#### [REWRITE] description (file:line)
**Before:**
```python
# existing code
```
**After:**
```python
# proposed rewrite
```
**Why:** explanation of what makes this better.

#### [BUG] description (file:line)
...

#### [PERF] description (file:line)
...

### Verdict
PASS — production quality, no changes needed
PASS WITH NOTES — good quality, minor improvements suggested
NEEDS WORK — significant issues that should be addressed
```

### 5. If `--fix` flag is set

Apply all REWRITE, BUG, and PERF fixes directly using the Edit tool. Do NOT apply STYLE-only changes unless they're in code you're already editing. After applying, run `/test` to verify nothing broke.

### 6. If `--diff` flag is set

Show the proposed edits but don't apply them. Format as before/after blocks.

## Quality Bar

This project's code reads like a textbook on evolutionary optimization in Python. The bar is:
- A pymoo contributor would approve this code
- A numpy veteran would find the vectorization clean
- A Python developer would find the structure intuitive
- Variable names tell you what's happening without reading the docstring

Code that "works" but reads like it was written by someone unfamiliar with numpy/pymoo idioms does NOT pass.