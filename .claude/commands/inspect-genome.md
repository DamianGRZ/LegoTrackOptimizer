---
description: Decode chromosomes from a run's chromosomes.csv and cross-check the phenotype against out-keys, PNG titles, and piece-usage claims. Use when a layout image and a report disagree (e.g. a crossing is visible but counted 0), when verifying which pieces a champion actually uses, or when auditing STRAIGHT_24/special-piece usage across the final population.
---

# /inspect-genome - Decode & Cross-Check Chromosomes

Answer "what does this individual ACTUALLY contain?" with a decode, not by
trusting CSVs, PNG titles, or report lines. `/diag` parses artifacts;
this skill re-derives the phenotype from the genome and flags mismatches.

## Arguments

| Argument | Description |
|----------|-------------|
| `<dir>` | Run output dir (e.g. `outputs/verify_all_pieces_3`). **Required.** |
| `--row N` | Inspect population row N instead of the best feasible. |
| `--piece <ID>` | Also census that piece type across the whole population. |

## Execution

### 1. Resolve the run's config

`grep "Config file" <dir>/run_info.md`. **Use the verbatim config block inside
run_info.md if the config file changed since the run** (inventory edits change
`n_main`/dims and would mis-decode old chromosomes).

### 2. Decode

```python
import csv, collections
import numpy as np
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import compute_dimensions

cat = TrackCatalog.load("data/track_pieces_v2.yaml")
cfg = OptimizationConfig.load("<config from run_info>")
dims = compute_dimensions(cfg, cat)
d = "<dir>/"
F = np.loadtxt(d + "fitness.csv", delimiter=",", skiprows=1)
G = np.loadtxt(d + "constraints.csv", delimiter=",", skiprows=1)
feas = np.all(G <= 0, axis=1)
with open(d + "chromosomes.csv") as f:
    X = np.array(list(csv.reader(f))[1:], dtype=float).astype(np.int16)

row = np.where(feas)[0][np.argmin(F[feas, 0])]  # best feasible (or --row N)
lay = decode_chromosome(X[row], cat, cfg.inventory, dims=dims)
names = {v: k for k, v in cat._id_to_index.items()}  # print NAMES, not indices
print("row", row, "| F", F[row], "| feasible", bool(feas[row]))
print("slot census:", {names[t]: c
                       for t, c in collections.Counter(lay.main_loop_pieces).items()})
print("switch_pairs:", lay.n_switch_pairs,
      "| cross records:", lay.n_cross_junctions,
      "| cross pieces:", lay.n_cross_pieces,
      "| dc:", lay.n_dbl_crossovers)
print("physical:", lay.n_physical_pieces, "| traversal slots:", lay.n_pieces)
print("paths:", len(lay.paths), "| drop_log:", lay.drop_log)
```

Population census for `--piece`: `(X[:, :dims.n_main] == IDX).sum(axis=1)` —
report how many individuals carry it and the max per genome.

### 3. Cross-check & explain

Compare against the run log (`Piece usage:` block), PNG titles, and
`category_report.md`. Known non-bugs vs bugs:

| Observation | Meaning |
|---|---|
| CROSS_90 slot present, `n_cross_junctions == 0` | Emergent Step-4 crossing (one slot, no record). Counted by `n_cross_pieces` — NOT a bug. |
| `n_pieces > n_physical_pieces` | Descriptor cross/DC occupy two traversal slots per physical piece. Use `n_physical_pieces` for inventory claims. |
| Same score %, different piece counts | Score is weighted (`special_piece_weight` per switch pair/cross/DC); compare `n_physical/total_inventory` separately. |
| Gene present but element missing from layout | Read `lay.drop_log` — the decoder dropped the descriptor and says why. |
| Archive/report % impossible for the piece count | Upper bound check: `(physical + 2*max_specials) / total_inventory`. If report exceeds it, suspect stale bookkeeping — re-derive from the individual's own F. |

### 4. Report

State row, F, feasibility, censuses, and an explicit verdict per mismatch:
**explained** (mechanism above) or **bug** (file + suspected code path).
Visual doubts → `/inspect-layout`. Whole-run stats → `/diag`.
