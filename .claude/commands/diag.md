---
description: Quick diagnostics on optimization output files. Parses fitness, constraints, chromosomes, and layout images.
---

# /diag - Optimization Diagnostics

Parse the `outputs/` directory and produce a one-shot diagnostic report. No agent needed.

## Arguments

| Argument | Description |
|----------|-------------|
| (none) | Diagnose default `outputs/` directory |
| `<dir>` | Diagnose a custom output directory |

## Examples

```
/diag                   # Default outputs/
/diag outputs_switches  # Custom directory
```

## Execution

### 1. Check output directory exists

List files in `outputs/` (or custom dir). If missing, report "No output directory found."

### 2. Parse fitness.csv

- Load the file (single column = single-objective, two columns = bi-objective)
- Compute: count, min, max, mean of each objective column
- For single-objective: utilization = `-min(F[:,0])`
- For bi-objective: utilization = `-min(F[:,0])`, speed = `-min(F[:,1])`

### 3. Parse constraints.csv

- 5 columns: `[closure, angle, boundary, inventory, loose_ports]`
- Count feasible solutions: rows where ALL columns <= 0
- For infeasible solutions, report which constraint is most commonly violated
- Show worst violation per constraint type

### 4. Parse chromosomes.csv (best feasible)

- Find the row index of the best feasible solution (lowest F value among feasible)
- Read that row from chromosomes.csv
- Report chromosome summary:
  - Total genes, how many piece keys are active (> RK_INACTIVE_THRESHOLD ~0.07)
  - Whether branch slots appear active (genes [200-216], check if active_key > 0.5)
  - Start position keys (genes [216-218])

### 5. Check best_layout.png

- Read the image file and describe what you see:
  - Is the track closed (loop returns to start)?
  - Are there branches/sidings visible?
  - Does it stay within the boundary box?
  - General shape (circle, oval, figure-8, complex)

### 6. Report

```
## Optimization Diagnostics

**Output dir**: outputs/
**Files found**: fitness.csv, constraints.csv, chromosomes.csv, best_layout.png

### Fitness
- **Mode**: Single-objective / Bi-objective
- **Best utilization**: XX.X%
- **Best speed**: X.XX m/s (if bi-objective)
- **Population size**: N solutions

### Feasibility
- **Feasible**: N / M (XX%)
- **Most violated constraint**: closure (N violations, worst: X.XX)
- **Constraint breakdown**: closure=N, angle=N, boundary=N, inventory=N, loose_ports=N

### Best Chromosome
- **Active pieces**: N / 100 slots
- **Branch slots active**: N / 4
- **Start position**: (X.XX, Y.XX)

### Layout Visual
- **Closed**: Yes/No
- **Branches**: N visible sidings
- **Shape**: description
- **Boundary**: Within bounds / Exceeds bounds
```

Keep report concise. If a file is missing, skip that section and note it.