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

- Two columns: `[-utilization_score, -avg_speed]` (both maximized, stored negated)
- Best score = `-min(F[:,0])`, best speed = `-min(F[:,1])`
- **The score is WEIGHTED** (`special_piece_weight` per switch pair / crossing /
  double-crossover) — never present it as a raw inventory fraction
- Count DISTINCT F rows and their multiplicities
  (`np.unique(np.round(F, 6), axis=0, return_counts=True)`): a terminal
  population is typically a monoculture (hundreds of copies of the champion
  plus injected category elites) — report the multiplicity, it is normal

### 3. Parse constraints.csv

- `5 + n_piece_types` columns: `G[0..2]` per-axis closure (|dx|, |dy|, |dθ|),
  `G[3]` boundary, `G[4]` collisions, `G[5..]` per-type inventory excess
- Feasible = ALL columns <= 0; report violations-per-column and worst value
  for the infeasible rows

### 4. Parse chromosomes.csv (best feasible)

- int16 partitioned genome (NOT random keys): main-loop piece types in
  `[0, n_main)` with `-1` = INACTIVE (0=STRAIGHT_16, 1=STRAIGHT_24,
  2=R40_CURVE; 3+ enter via descriptors/decoder), parallel flip bits,
  descriptor blocks, last 2 genes = start position
- `n_main` comes from `compute_dimensions(config, catalog)` — **never
  hardcode gene ranges**, they scale with inventory
- Best feasible row = lowest F[:,0] among feasible; report active-slot count
  and per-type census (`!= -1` mask)
- For a full phenotype decode and mismatch hunting, hand off to
  `/inspect-genome <dir>`

### 4b. Check run artifacts beyond the basics

- `category_report.md` — all three sections (switch / cross / dc) present?
  Which categories have a feasible elite, and how far below the global best?
- `pareto_archive.csv` — run-cumulative feasible front (distinct trade-off
  points discovered across ALL generations); if absent, the run predates it
- `pareto_front.png` — distinct points carry ×N multiplicity labels;
  `best_with_*.png` titles show `N/total pcs` and `score X%`

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
- **Best score**: XX.X% (weighted) | **Best speed**: X.XX m/s
- **Population**: N solutions, K distinct F points (top multiplicity ×M)

### Feasibility
- **Feasible**: N / M (XX%)
- **Most violated**: closure_x/closure_y/closure_theta/boundary/collisions/
  inventory[type] (N violations, worst: X.XX)

### Best Chromosome
- **Active main slots**: N / n_main, census {S16: a, S24: b, R40: c, ...}
- **Start position**: (X, Y)

### Categories & Front
- **Category elites**: switch X% / cross Y% / dc Z% (or "missing")
- **Run front (pareto_archive.csv)**: K points

### Layout Visual
- **Closed**: Yes/No
- **Branches**: N visible sidings
- **Shape**: description
- **Boundary**: Within bounds / Exceeds bounds
```

Keep report concise. If a file is missing, skip that section and note it.