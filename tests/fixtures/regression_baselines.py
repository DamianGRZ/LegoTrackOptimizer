"""Regression-baseline helpers (Phase 17.C.5, PLAN Section 10.6).

Provides four small functions for capturing/comparing per-config metrics
across phase boundaries:

- :func:`extract_metrics` — pull the comparison-friendly fields off a
  pymoo ``Result`` (feasibility rate, util range, slowest piece-speed).
- :func:`save_baseline` / :func:`load_baseline` — JSON I/O at
  ``tests/baselines/{config_name}.json``.
- :func:`compare_to_baseline` — relative-delta comparison flagging entries
  inside / outside an acceptance band (default ±5 % per PLAN §10.7 DoD #5).

Hypervolume / topology-diversity / switch-usage metrics are richer and
deferred to ``thesis_metrics.py`` (Phase 17.C.6, PLAN §10.6).

CLI entry point — capture an actual baseline from a full-run:

.. code-block:: powershell

    .venv\\Scripts\\python.exe -m tests.fixtures.regression_baselines default \\
        --n-gen 200 --pop-size 1000 --seed 42

producing ``tests/baselines/default.json``. Repeat per config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def extract_metrics(result) -> dict[str, Any]:
    """Extract regression-relevant metrics from a pymoo ``Result``.

    Returns a flat dict with these keys (None when no feasibles exist):
    ``feasibility_rate``, ``n_feasible``, ``n_pop``, ``mean_util_feasible``,
    ``max_util_feasible``, ``min_min_speed_feasible``.
    """
    F = result.pop.get("F")
    G = result.pop.get("G")
    n_pop = int(F.shape[0])
    feas_mask = np.all(G <= 0, axis=1)
    n_feasible = int(feas_mask.sum())

    if n_feasible == 0:
        return {
            "feasibility_rate": 0.0,
            "n_feasible": 0,
            "n_pop": n_pop,
            "mean_util_feasible": None,
            "max_util_feasible": None,
            "min_min_speed_feasible": None,
        }

    util_feas = -F[feas_mask, 0]
    speed_feas = -F[feas_mask, 1]
    return {
        "feasibility_rate": float(n_feasible) / n_pop,
        "n_feasible": n_feasible,
        "n_pop": n_pop,
        "mean_util_feasible": float(util_feas.mean()),
        "max_util_feasible": float(util_feas.max()),
        "min_min_speed_feasible": float(speed_feas.min()),
    }


def save_baseline(
    metrics: dict[str, Any], baseline_dir: Path, config_name: str,
) -> Path:
    """Write ``metrics`` to ``{baseline_dir}/{config_name}.json``. Returns the path."""
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / f"{config_name}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_baseline(baseline_dir: Path, config_name: str) -> dict[str, Any]:
    """Read ``{baseline_dir}/{config_name}.json``. Raises ``FileNotFoundError`` on miss."""
    return json.loads((Path(baseline_dir) / f"{config_name}.json").read_text(encoding="utf-8"))


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Compare ``current`` to ``baseline``; return a per-key delta dict.

    Each entry has ``baseline``, ``current`` plus either ``rel_delta`` (when
    baseline is a non-zero number) or ``abs_delta`` (zero baseline). The
    ``within_band`` field is ``True`` when within tolerance, ``False`` when
    out, or ``None`` when comparison is undefined (None values, mismatched
    types).
    """
    deltas: dict[str, dict[str, Any]] = {}
    for key, baseline_val in baseline.items():
        current_val = current.get(key)

        if baseline_val is None or current_val is None:
            deltas[key] = {
                "baseline": baseline_val,
                "current": current_val,
                "rel_delta": None,
                "within_band": None,
            }
            continue

        if not isinstance(baseline_val, (int, float)):
            deltas[key] = {
                "baseline": baseline_val,
                "current": current_val,
                "rel_delta": None,
                "within_band": baseline_val == current_val,
            }
            continue

        if baseline_val == 0:
            abs_delta = float(current_val) - 0.0
            deltas[key] = {
                "baseline": baseline_val,
                "current": current_val,
                "abs_delta": abs_delta,
                "within_band": abs(abs_delta) <= tolerance,
            }
            continue

        rel_delta = (float(current_val) - float(baseline_val)) / abs(float(baseline_val))
        deltas[key] = {
            "baseline": baseline_val,
            "current": current_val,
            "rel_delta": rel_delta,
            "within_band": abs(rel_delta) <= tolerance,
        }
    return deltas


def _capture_main() -> None:
    """CLI entry point — capture a baseline JSON from a full optimization run."""
    import argparse

    from src_v2.catalog import TrackCatalog
    from src_v2.config import OptimizationConfig
    from src_v2.runner import run_optimization

    parser = argparse.ArgumentParser(
        description="Capture a regression baseline JSON from a full optimization run.",
    )
    parser.add_argument("config", help="Config name without .yaml (e.g. 'default').")
    parser.add_argument("--n-gen", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-dir", default="tests/baselines")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    catalog = TrackCatalog.load(project_root / "data" / "track_pieces_v2.yaml")
    config = OptimizationConfig.load(project_root / "configs" / f"{args.config}.yaml")
    config.algorithm.n_gen = args.n_gen
    config.algorithm.pop_size = args.pop_size
    config.algorithm.seed = args.seed

    out_dir = project_root / "outputs_v2" / f"{args.config}_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Capturing baseline for '{args.config}' "
        f"(n_gen={args.n_gen}, pop_size={args.pop_size}, seed={args.seed}) ...",
    )
    result = run_optimization(config, catalog, output_dir=out_dir, seed=args.seed)
    metrics = extract_metrics(result)
    metrics["_meta"] = {
        "config": args.config,
        "n_gen": args.n_gen,
        "pop_size": args.pop_size,
        "seed": args.seed,
    }

    path = save_baseline(metrics, project_root / args.baseline_dir, args.config)
    print(f"Saved baseline -> {path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _capture_main()
