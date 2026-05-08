# Modular Architecture Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt the high-value subset of the 9-package modular architecture from `Modular9PartResearchV1/` — split the three largest monolithic files (`visualization.py`, `decoder.py`, `main.py`) into focused packages, and add static enforcement of layer boundaries with `import-linter`.

**Architecture:** The research document proposes nine packages in three tiers (domain / EA / infrastructure). `catalog/` and `train/` are already packages that satisfy the proposal. This plan addresses the three biggest flat-file offenders — `visualization.py` (866 lines), `decoder.py` (773 lines), and `main.py` (392 lines containing algorithm runner logic) — and leaves alone the small flat files (`geometry.py`, `problem.py`, `config.py`, `encoding.py`, `operators.py`) and the already-packaged modules (`catalog/`, `train/`). Finer sub-splits proposed in the research (train's 5-way physics split, operators' UCB split, chromosome rename) are explicitly deferred as over-engineering for the current scope.

**Tech Stack:** Python 3.x, pymoo 0.6.1.6, pytest, import-linter (new dev dependency).

**Out of scope (explicitly deferred):**
- `train/` five-way split (`lateral_force`, `motor_model`, `speed_table`, `scoring`, `engine`) — current two-file split is sufficient until a motor model is added
- `operators/` split into `crossover`/`mutation`/`ucb` — defer to ALNS-UCB implementation
- `chromosome/segment_map.py` rename of `encoding.py` — cosmetic
- `geometry/` package split — file is only 4.6KB
- `problem/` package split — file is only 6.6KB
- `io/` package creation — no checkpointing implemented
- `config/` package split — pydantic models and `yaml.safe_load` do not warrant splitting

**Risk posture:** These are non-behavioural refactors. The existing test suite is the regression net. Every task ends with `/test` verification and a commit so the plan can be paused or rolled back between tasks.

---

## File Structure After Plan

```
src/
├── catalog/              (unchanged)
├── train/                (unchanged)
├── algorithm/            (NEW — from main.py)
│   ├── __init__.py
│   └── runner.py
├── decoder/              (NEW — from decoder.py)
│   ├── __init__.py
│   ├── types.py
│   └── construction.py
├── visualization/        (NEW — from visualization.py)
│   ├── __init__.py
│   ├── track_renderer.py
│   ├── pareto_plot.py
│   └── convergence.py
├── geometry.py           (unchanged)
├── problem.py            (unchanged)
├── config.py             (unchanged)
├── encoding.py           (unchanged)
├── operators.py          (unchanged)
├── repair.py             (unchanged)
├── sampling.py           (unchanged)
├── templates.py          (unchanged)
├── intersection.py       (unchanged)
├── types.py              (unchanged)
└── lego_track_models.py  (unchanged)

main.py                   (thinned to CLI + dispatch)

.importlinter             (NEW — layer contracts)
pyproject.toml            (NEW, minimal — only if needed for import-linter discovery)
```

Backward compatibility: each new package's `__init__.py` re-exports the public API under the original module name, so `from src.visualization import plot_layout` and `from src.decoder import decode_chromosome, DecoderConfig` keep working at every external callsite (`main.py`, `tests/`, `src/problem.py`).

---

## Phase 1: Visualization Split

**Files:**
- Create: `src/visualization/__init__.py`
- Create: `src/visualization/track_renderer.py`
- Create: `src/visualization/pareto_plot.py`
- Create: `src/visualization/convergence.py`
- Delete: `src/visualization.py`
- Modify: none (callsites use `from src.visualization import X` which the `__init__.py` preserves)

**Source mapping** (from `grep -n "^def \|^class " src/visualization.py`):

| Current location in `visualization.py` | Target module |
|---|---|
| Module-level constants (`BED_LW`, `RAIL_OFFSET`, color palette, etc.) | `track_renderer.py` |
| `get_piece_color` (line 95), `get_piece_short_name` (line 383) | `track_renderer.py` |
| `_draw_track_bed`, `_draw_rails` (lines 102–114) | `track_renderer.py` |
| `_draw_straight_piece`, `_draw_curve_piece`, `_draw_switch_piece`, `_draw_cross90_piece`, `_draw_double_crossover_piece`, `_draw_piece` (lines 116–382) | `track_renderer.py` |
| `plot_layout` (line 388) | `track_renderer.py` |
| `plot_multi_path_layout`, `_plot_combined_paths`, `_plot_single_path` (lines 551–753) | `track_renderer.py` |
| `plot_pareto_front` (line 754) | `pareto_plot.py` |
| `plot_convergence` (line 817) | `convergence.py` |

### Task 1.1: Create `track_renderer.py`

- [ ] **Step 1: Create the new file with copied content**

Read `src/visualization.py` lines 1–753 (everything up to the end of `_plot_single_path`, inclusive of its trailing blank line — verify with your editor that the last included line is the final line of `_plot_single_path` before `def plot_pareto_front`).

Write `src/visualization/track_renderer.py` with:
- The module docstring from the original file header
- The exact imports block from the original (matplotlib, numpy, MultiPathLayout, etc.)
- All constants, helper functions, and the public functions `get_piece_color`, `get_piece_short_name`, `plot_layout`, `plot_multi_path_layout`

Do **not** include `plot_pareto_front` or `plot_convergence` here.

- [ ] **Step 2: Verify the new module imports cleanly**

```bash
python -c "from src.visualization.track_renderer import plot_layout, plot_multi_path_layout, get_piece_color, get_piece_short_name; print('OK')"
```
Expected: `OK`. If it fails with `ModuleNotFoundError`, the old `src/visualization.py` is still shadowing the new package — that is expected at this step; skip ahead to Task 1.4.

### Task 1.2: Create `pareto_plot.py`

- [ ] **Step 1: Create the file**

Read `src/visualization.py` lines 754–816 (the `plot_pareto_front` function).

Write `src/visualization/pareto_plot.py`:

```python
"""Pareto-front scatter plots for multi-objective results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# [paste the exact body of plot_pareto_front from src/visualization.py lines 754–816]
```

Only add imports that `plot_pareto_front` actually uses — check the function body and the original file's import block; do not blanket-copy unused imports.

### Task 1.3: Create `convergence.py`

- [ ] **Step 1: Create the file**

Read `src/visualization.py` lines 817–866 (the `plot_convergence` function and any trailing helpers).

Write `src/visualization/convergence.py`:

```python
"""Convergence-history plots for optimization runs."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# [paste the exact body of plot_convergence from src/visualization.py lines 817–end]
```

### Task 1.4: Create `__init__.py` re-export and delete the old module

- [ ] **Step 1: Create `src/visualization/__init__.py`**

```python
"""Visualization package.

Re-exports the public API from the submodules so callsites using
``from src.visualization import plot_layout`` keep working.
"""

from src.visualization.track_renderer import (
    get_piece_color,
    get_piece_short_name,
    plot_layout,
    plot_multi_path_layout,
)
from src.visualization.pareto_plot import plot_pareto_front
from src.visualization.convergence import plot_convergence

__all__ = [
    "get_piece_color",
    "get_piece_short_name",
    "plot_layout",
    "plot_multi_path_layout",
    "plot_pareto_front",
    "plot_convergence",
]
```

- [ ] **Step 2: Delete the old flat file**

```bash
git rm src/visualization.py
```

Use `git rm`, not plain `rm`, so the removal is staged in the same commit as the new files.

- [ ] **Step 3: Verify backward-compatible imports**

```bash
python -c "from src.visualization import plot_layout, plot_multi_path_layout, plot_pareto_front, plot_convergence; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: all previously passing tests still pass. Pay particular attention to `tests/test_visualization.py` and any test that imports from `src.visualization`.

- [ ] **Step 5: Run the dedicated visualization smoke test**

```bash
python test_visualization.py
```
Expected: the script produces `test_layout.png` without errors.

- [ ] **Step 6: Commit**

```bash
git add src/visualization/
git commit -m "refactor: split visualization.py into focused package

- track_renderer.py: piece drawing + plot_layout/plot_multi_path_layout
- pareto_plot.py: plot_pareto_front
- convergence.py: plot_convergence
- __init__.py re-exports public API for backward compatibility

Part of modular architecture adoption (Modular9PartResearchV1)."
```

---

## Phase 2: Decoder Split

**Files:**
- Create: `src/decoder/__init__.py`
- Create: `src/decoder/types.py`
- Create: `src/decoder/construction.py`
- Delete: `src/decoder.py`

**Source mapping** (from `grep -n "^def \|^class \|^@dataclass" src/decoder.py`):

| Current | Target |
|---|---|
| `DecoderConfig` (line 53–71), `InventoryTracker` (line 72–128), `ValidatedJunction` (line 216–228) | `types.py` |
| `decode_chromosome` (line 130) + all `_read_*`, `_inject_*`, `_find_*`, `_compute_*`, `_release_*`, `_apply_*`, `_build_*`, `_auto_center`, `_empty_layout` helpers | `construction.py` |

### Task 2.1: Create `src/decoder/types.py`

- [ ] **Step 1: Create the file**

Read `src/decoder.py` header + lines 53–71 (`DecoderConfig`) + lines 72–128 (`InventoryTracker`) + lines 216–228 (`ValidatedJunction`). Note: `InventoryTracker` may be a regular class rather than a dataclass — preserve its exact definition.

Write `src/decoder/types.py`:

```python
"""Decoder-internal dataclasses and helper types.

These types are produced and consumed by the construction decoder.
They are separated from construction logic so that test code, the
problem class, and (potentially) alternative decoders can import
types without pulling in the full construction algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# [paste DecoderConfig dataclass exactly as in decoder.py]
# [paste InventoryTracker class exactly as in decoder.py]
# [paste ValidatedJunction dataclass exactly as in decoder.py]
```

Preserve the original `@dataclass(frozen=...)` / `field(default_factory=...)` decorators exactly — these affect hashability and mutation semantics.

- [ ] **Step 2: Verify import**

```bash
python -c "from src.decoder.types import DecoderConfig, InventoryTracker, ValidatedJunction; print('OK')"
```
Expected: `OK` (assuming `src/decoder.py` is already renamed/deleted — if not, this step will shadow; acceptable to skip until Task 2.4).

### Task 2.2: Create `src/decoder/construction.py`

- [ ] **Step 1: Create the file**

Read `src/decoder.py` lines 1–52 (header + imports) and lines 130–773 (all decode logic).

Write `src/decoder/construction.py`:

```python
"""Construction-based decoder: chromosome → MultiPathLayout.

Algorithm:
1. Read main loop piece types, filter INACTIVE → active_pieces
2. Read active junctions, sort by position, compute branch pieces from templates
3. Inject switches into main loop copy (replace, not insert)
4. Self-intersection repair (inject CROSS_90 at ~90° crossings)
5. Compute FK for augmented main loop
6. Enumerate 2^J traversal paths
7. Auto-center within boundary, return MultiPathLayout
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Pull types from the sibling module instead of redefining them:
from src.decoder.types import DecoderConfig, InventoryTracker, ValidatedJunction

# [paste the exact bodies of the following, in order]:
# - decode_chromosome
# - _read_main_loop
# - _read_junctions
# - _inject_switches
# - _find_out_position
# - _compute_state_at_position
# - _release_junction_inventory
# - _apply_crossing_repair
# - _build_multi_path_layout
# - _compute_single_path
# - _compute_path_fk
# - _auto_center
# - _empty_layout
```

Remove the local class definitions of `DecoderConfig`, `InventoryTracker`, and `ValidatedJunction` from this file — they live in `types.py` now. All other helpers move verbatim.

### Task 2.3: Create `src/decoder/__init__.py`

- [ ] **Step 1: Create the file**

```python
"""Decoder package: chromosome → MultiPathLayout.

Public API re-exported here so callers can keep using
``from src.decoder import decode_chromosome, DecoderConfig``.
"""

from src.decoder.construction import decode_chromosome
from src.decoder.types import DecoderConfig

__all__ = ["decode_chromosome", "DecoderConfig"]
```

### Task 2.4: Delete old flat file and verify

- [ ] **Step 1: Delete `src/decoder.py`**

```bash
git rm src/decoder.py
```

- [ ] **Step 2: Verify backward-compatible imports**

```bash
python -c "from src.decoder import decode_chromosome, DecoderConfig; print('OK')"
```
Expected: `OK`

Also verify the internal types are reachable through the subpath for any test that needs them:

```bash
python -c "from src.decoder.types import InventoryTracker, ValidatedJunction; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: all previously passing tests still pass. Decoder tests (`tests/test_decoder.py`) are the primary regression check.

- [ ] **Step 4: Run a quick optimization smoke test**

```bash
python main.py --config configs/default.yaml --quick-test 2>&1 | tail -20
```
Expected: optimization completes, reports a best solution, writes an output file. Any `AttributeError` or `ImportError` here indicates a missing re-export.

- [ ] **Step 5: Commit**

```bash
git add src/decoder/
git commit -m "refactor: split decoder.py into types + construction

- src/decoder/types.py: DecoderConfig, InventoryTracker, ValidatedJunction
- src/decoder/construction.py: decode_chromosome and private helpers
- src/decoder/__init__.py re-exports decode_chromosome, DecoderConfig

Part of modular architecture adoption (Modular9PartResearchV1)."
```

---

## Phase 3: Algorithm Runner Extraction

**Goal:** Move the GA assembly and run logic out of `main.py` into a reusable `src/algorithm/runner.py`. This makes the run callable from notebooks and experiment scripts without shelling out to a CLI.

**Files:**
- Create: `src/algorithm/__init__.py`
- Create: `src/algorithm/runner.py`
- Modify: `main.py` (thin to CLI parsing + dispatch)

**Source mapping** (from `grep -n "^def \|^class " main.py`):

| Currently in `main.py` | Destination |
|---|---|
| `setup_logging` (line 30) | stays in `main.py` (CLI concern) |
| `log_piece_usage` (line 42) | `src/algorithm/runner.py` (used during run) |
| `ProgressCallback` (line 74) | `src/algorithm/runner.py` |
| `LegoAdaptiveEpsilon` (line 113) | `src/algorithm/runner.py` |
| `run_optimization` (line 172) | `src/algorithm/runner.py` |
| `save_results` (line 273) | `src/algorithm/runner.py` |
| `main` (line 355) | stays in `main.py` |

### Task 3.1: Create `src/algorithm/__init__.py`

- [ ] **Step 1: Create the file**

```python
"""Algorithm package: GA assembly and run orchestration.

Wraps pymoo's NSGA-II with project-specific operators, callbacks,
and constraint handling so callers need only pass an
``OptimizationConfig`` and a ``TrackCatalog``.
"""

from src.algorithm.runner import (
    LegoAdaptiveEpsilon,
    ProgressCallback,
    log_piece_usage,
    run_optimization,
    save_results,
)

__all__ = [
    "LegoAdaptiveEpsilon",
    "ProgressCallback",
    "log_piece_usage",
    "run_optimization",
    "save_results",
]
```

### Task 3.2: Create `src/algorithm/runner.py` by moving functions from `main.py`

- [ ] **Step 1: Create the file with the moved content**

Read `main.py` lines 42–354 (everything from `log_piece_usage` through the end of `save_results`, not including `main()`).

Write `src/algorithm/runner.py`:

```python
"""NSGA-II runner for the track optimization problem."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.core.callback import Callback
from pymoo.operators.survival.rank_and_crowding import ConstrRankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import compute_dimensions
from src.operators import PartitionedCrossover, PartitionedMutation
from src.problem import TrackOptimizationProblem
from src.repair import TrackRepairPipeline
from src.sampling import IntegerSampling
from src.visualization import plot_layout, plot_multi_path_layout, plot_pareto_front

# [paste log_piece_usage, ProgressCallback, LegoAdaptiveEpsilon,
#  run_optimization, save_results — verbatim from main.py]
```

Carry over every import the moved functions actually use — do not copy imports that were only used by `main()`.

### Task 3.3: Thin `main.py` to a CLI wrapper

- [ ] **Step 1: Replace the body of `main.py`**

The new `main.py` should contain only:
- the original module docstring
- `argparse` / `logging` imports
- `setup_logging`
- `main()` (which now imports `run_optimization` and `save_results` from `src.algorithm`)
- the `if __name__ == "__main__":` guard

```python
"""Main entry point for LEGO Track Optimizer.

NSGA-II multi-objective optimization with partitioned chromosome encoding.
Maximizes piece utilization and train speed with template-based passing sidings.
"""

import argparse
import logging
from pathlib import Path

from src.algorithm import run_optimization, save_results
from src.catalog import TrackCatalog
from src.config import OptimizationConfig


def setup_logging(verbose: bool = False) -> None:
    # [paste existing setup_logging body verbatim]


def main() -> None:
    # [paste existing main body verbatim — it already calls run_optimization and save_results]


if __name__ == "__main__":
    main()
```

Remove the now-unused imports (`NSGA2`, `Callback`, `minimize`, `ConstrRankAndCrowding`, `AdaptiveEpsilonConstraintHandling`, `get_termination`, `IntegerSampling`, `PartitionedCrossover`, `PartitionedMutation`, `TrackOptimizationProblem`, `TrackRepairPipeline`, `decode_chromosome`, `compute_dimensions`, `plot_*`, `numpy`) from `main.py` — they live in `src/algorithm/runner.py` now.

- [ ] **Step 2: Verify the CLI still runs**

```bash
python main.py --config configs/default.yaml --quick-test 2>&1 | tail -20
```
Expected: quick optimization completes without import errors. Verify the same output structure as before the split (generation progress, best solution summary, output files written).

- [ ] **Step 3: Verify programmatic use works**

```bash
python -c "
from src.algorithm import run_optimization
from src.config import OptimizationConfig
from src.catalog import TrackCatalog
cfg = OptimizationConfig.load('configs/default.yaml')
cat = TrackCatalog.load('data/track_pieces.yaml')
print('imports OK')
"
```
Expected: `imports OK`

- [ ] **Step 4: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: all previously passing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/algorithm/ main.py
git commit -m "refactor: extract algorithm runner from main.py

- src/algorithm/runner.py: run_optimization, save_results, callbacks
- main.py: thinned to CLI argument parsing + dispatch
- enables programmatic runs from notebooks and experiment scripts

Part of modular architecture adoption (Modular9PartResearchV1)."
```

---

## Phase 4: Static Layer-Contract Enforcement

**Goal:** Make the dependency table from the research document machine-checkable so future changes cannot silently introduce layer violations (e.g. `train/` accidentally importing pymoo, or `visualization/` importing `decoder/`).

**Tool choice:** `import-linter` — it is the tool the research document recommends, supports "layers" and "forbidden" contracts declaratively, and runs as a standalone command that can be added to `/test` or CI later.

**Files:**
- Create: `.importlinter` (the contract file)
- Create: `pyproject.toml` (minimal — only if `lint-imports` cannot discover the project without it)
- Modify: `requirements.txt` (add `import-linter` to dev deps)
- Modify: `CLAUDE.md` (document the new check)

### Task 4.1: Add `import-linter` as a dev dependency

- [ ] **Step 1: Install the tool locally and add it to requirements**

```bash
pip install import-linter
```

Append to `requirements.txt`:

```
# Architecture enforcement (dev)
import-linter>=2.0
```

- [ ] **Step 2: Verify installation**

```bash
lint-imports --help 2>&1 | head -5
```
Expected: usage text containing `lint-imports`.

### Task 4.2: Define the contracts

- [ ] **Step 1: Create `.importlinter` at the repo root**

```ini
[importlinter]
root_packages =
    src

[importlinter:contract:layers]
name = Three-tier architecture (domain → EA → infrastructure)
type = layers
layers =
    src.algorithm
    src.problem | src.operators | src.decoder | src.encoding | src.repair | src.sampling
    src.train | src.geometry | src.catalog | src.types | src.templates | src.intersection | src.lego_track_models | src.config
exhaustive = false

[importlinter:contract:train-isolation]
name = train/ must not import EA or infrastructure code
type = forbidden
source_modules =
    src.train
forbidden_modules =
    pymoo
    src.decoder
    src.operators
    src.problem
    src.algorithm
    src.visualization
    src.sampling
    src.repair
    src.encoding

[importlinter:contract:catalog-isolation]
name = catalog/ must not import framework code
type = forbidden
source_modules =
    src.catalog
forbidden_modules =
    pymoo
    matplotlib
    src.decoder
    src.operators
    src.problem
    src.algorithm
    src.visualization

[importlinter:contract:visualization-isolation]
name = visualization/ is a consumer of domain types, not of EA internals
type = forbidden
source_modules =
    src.visualization
forbidden_modules =
    pymoo
    src.operators
    src.sampling
    src.repair
    src.algorithm
```

- [ ] **Step 2: Run the contracts against the current tree**

```bash
lint-imports
```

Expected: all four contracts KEPT. If any is BROKEN, the violation is genuine — fix it before proceeding. Most likely false-alarm candidate: `visualization/` pulling from `decoder/` through `MultiPathLayout` (which actually lives in `src.types`, not `src.decoder`, so this should be fine). Genuine violations should be investigated, not silenced — the research doc's isolation claim is a real invariant.

- [ ] **Step 3: Create a minimal `pyproject.toml` only if needed**

Run `lint-imports` without a `pyproject.toml`. If the tool exits cleanly, skip this step entirely. If it complains about project discovery, create:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "lego-track-optimizer"
version = "0.1.0"
requires-python = ">=3.10"
```

Do not put the contracts themselves here — keep them in `.importlinter` so the tool can find them via its default lookup.

- [ ] **Step 4: Document the check in `CLAUDE.md`**

Append under the "Essential Commands" section:

```markdown
# Architecture enforcement
lint-imports                        # Verify layer contracts (import-linter)
```

- [ ] **Step 5: Run the full test suite one more time**

```bash
pytest --tb=short -q && lint-imports
```
Expected: all tests pass AND all contracts are kept.

- [ ] **Step 6: Commit**

```bash
git add .importlinter requirements.txt CLAUDE.md
# Include pyproject.toml in the add only if Task 4.2 Step 3 created it.
git commit -m "build: add import-linter layer contracts

Enforces the three-tier architecture from Modular9PartResearchV1:
- train/ and catalog/ cannot import pymoo or matplotlib
- visualization/ cannot import EA internals
- layered order: infrastructure → EA → domain (no upward imports)

Run locally with: lint-imports"
```

---

## Verification Checklist (run after all phases)

- [ ] `pytest --tb=short -q` — all tests pass
- [ ] `python main.py --config configs/default.yaml --quick-test` — CLI still works
- [ ] `python test_visualization.py` — visualization smoke test renders `test_layout.png`
- [ ] `lint-imports` — all layer contracts kept
- [ ] Invoke the `config-test-runner` agent to run all four configs end-to-end and confirm layouts are closed, switches produce branches, and crossings produce crossings.
- [ ] `git log --oneline -10` — expect four new commits (visualization split, decoder split, algorithm extraction, import-linter) in that order.

---

## Self-Review Notes

**Spec coverage:** Every high-value split identified in the comparison against the research document is covered (viz, decoder, algorithm, import-linter). Deliberately-deferred items (train five-way, operators/UCB, chromosome rename, io/, geometry/, problem/, config/) are explicitly listed as out of scope with reasons.

**Placeholder scan:** Steps that say "paste the exact body of X" are acceptable here because the source file is the authoritative reference — re-quoting 700 lines of `decoder.py` inside the plan would be counterproductive. Every task names the exact source line range.

**Type consistency:** `DecoderConfig`, `MultiPathLayout`, `TrackCatalog`, `OptimizationConfig`, `run_optimization`, `save_results` — names match their current in-repo definitions. Public-API names preserved across `__init__.py` re-exports so no callsite in `tests/`, `main.py`, or `src/problem.py` needs editing.

**Risk of scope creep:** The import-linter phase is the only one that adds tooling rather than moving code. If contracts fail on the first run, treat that as a discovery — do not silently relax the contracts; either fix the violating import or split the phase into a follow-up investigation.
