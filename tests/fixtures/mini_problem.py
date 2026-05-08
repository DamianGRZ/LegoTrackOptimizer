"""Tier-3 mini-optimization GA loop fixture (Phase 17.C.1, PLAN Section 10.1).

A ~30-second full GA loop on a 16-piece-default switch-bearing inventory.
Used by every later phase's T3 regression test to verify "did this change
break the GA loop?" without paying the 10-30 minute per-phase full-run cost.

Output files (``diagnostics.csv``, ``snapshots/``, ``epsilon_archive.json``)
land in the caller's ``output_dir`` per ``run_optimization``'s standard
layout. Read them directly via stdlib ``csv`` / ``json`` / ``Path.glob``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.runner import run_optimization


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = _PROJECT_ROOT / "data" / "track_pieces_v2.yaml"
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


# §10.1 default mini inventory translated to V2 piece IDs (R40_LEFT collapsed
# to R40_CURVE per the V2 catalog migration). Total = 40 pieces, switch-bearing
# so Phase 5+ regression tests still exercise the switch code path on this
# small problem.
DEFAULT_MINI_INVENTORY: dict[str, int] = {
    "R40_CURVE": 16,
    "STRAIGHT_16": 16,
    "R40_SWITCH_LEFT": 4,
    "R40_SWITCH_RIGHT": 4,
}


def mini_optimization_run(
    output_dir: Path,
    *,
    seed: int = 42,
    n_gen: int = 50,
    pop_size: int = 50,
    inventory: Optional[dict[str, int]] = None,
    n_workers: int = 1,
    heuristic_ratio: float = 0.30,
) -> object:
    """Run a tiny GA optimization. Returns the pymoo ``Result``.

    The returned object exposes ``.pop``, ``.opt``, ``.X``, ``.F``, ``.G``,
    ``.exec_time`` etc. Auxiliary artifacts (per-gen CSV, 10-snapshot strip,
    epsilon archive JSON) are written to ``output_dir``.
    """
    config = OptimizationConfig.load(DEFAULT_CONFIG_PATH)
    config.algorithm.seed = seed
    config.algorithm.n_gen = n_gen
    config.algorithm.pop_size = pop_size
    config.algorithm.heuristic_ratio = heuristic_ratio
    config.n_workers = n_workers
    config.inventory = (
        dict(inventory) if inventory is not None else dict(DEFAULT_MINI_INVENTORY)
    )

    catalog = TrackCatalog.load(CATALOG_PATH)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_optimization(config, catalog, output_dir=output_dir, seed=seed)
