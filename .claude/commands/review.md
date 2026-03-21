---
description: Compact code review for Python/pymoo. Checks changed files against project conventions.
---

# /review - Inline Code Review

Review code for Python best practices and pymoo conventions without spawning an agent.

## Arguments

Parse arguments after `/review`:

| Argument | Description |
|----------|-------------|
| (none) | Review all staged/unstaged changes (`git diff`) |
| `<file>` | Review specific file (e.g., `src/problem.py`) |
| `--staged` | Review only staged changes |
| `--last` | Review changes in last commit |

## Examples

```
/review                     # Review current changes
/review src/evaluation.py   # Review specific file
/review --staged            # Only staged changes
/review --last              # Last commit's changes
```

## Execution

### 1. Get the diff

- No args: `git diff` (unstaged + staged)
- `--staged`: `git diff --cached`
- `--last`: `git diff HEAD~1`
- Specific file: `git diff -- <file>`, falling back to reading the full file if no diff

If there is no diff (clean working tree), read the specified file or ask what to review.

### 2. Review against this checklist

Check ONLY these items (keep it focused):

**pymoo interface** (critical):
- `_evaluate` signature: `(self, x, out, *args, **kwargs)`
- Objectives in `out["F"]`, all minimized (negate for maximization)
- Constraints in `out["G"]`, `g <= 0` feasible
- Operator `_do` returns correct shapes
- `super().__init__()` called with correct params

**numpy/vectorization**:
- Loops that could be vectorized
- Missing `.copy()` on arrays that get mutated
- Incorrect broadcasting shapes

**Project conventions**:
- Constraint normalization: `(error - tolerance) / tolerance`
- Piece indices: -1 = inactive, 0-9 = piece index
- Random-key encoding: all genes in [0.0, 1.0]
- FK chain: `[x, y, theta]` state convention

**Python quality**:
- Mutable default arguments
- Bare `except:` clauses
- Unused imports or variables

### 3. Report

Output a compact review:

```
## Review: <file(s)>

**Lines changed**: ~N

### Issues
- **[CRITICAL]** description (file:line)
- **[WARN]** description (file:line)
- **[STYLE]** description (file:line)

### OK
Brief note on what looks correct.
```

If no issues found, just say "No issues found." in one line. Do NOT pad with praise or filler.