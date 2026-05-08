"""Re-render best_layout.png / best_infeasible.png from saved chromosomes.

Usage:
    .venv/Scripts/python.exe tools/rerender_layout.py with_switches
    .venv/Scripts/python.exe tools/rerender_layout.py default with_switches with_crossing

Reads ``outputs_v2/<config>/{chromosomes,fitness,constraints}.csv``,
selects the best feasible (and best infeasible) by argmin(F[0]) inside
the feasibility mask, and re-runs decoder + plot_layout. Useful when
fixing visualization bugs without paying the full ~10 min optimization
cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import decode_chromosome, port_graph_to_layout
from src_v2.problem import PortPairProblem
from src_v2.visualization import plot_layout


def _load_run(config_name: str):
    """Load the saved chromosomes / fitness / constraints for one config."""
    out_dir = REPO_ROOT / "outputs_v2" / config_name
    if not out_dir.exists():
        raise SystemExit(f"missing run directory: {out_dir}")
    X = np.loadtxt(out_dir / "chromosomes.csv", delimiter=",", dtype=int)
    F = np.loadtxt(out_dir / "fitness.csv", delimiter=",", skiprows=1)
    G = np.loadtxt(out_dir / "constraints.csv", delimiter=",", skiprows=1)
    return out_dir, X, F, G


def _problem_for(config_name: str) -> PortPairProblem:
    """Recreate the PortPairProblem matching the saved CSV's chromosome shape."""
    catalog = TrackCatalog.load(REPO_ROOT / "data" / "track_pieces_v2.yaml")
    config = OptimizationConfig.load(REPO_ROOT / "configs" / f"{config_name}.yaml")
    return PortPairProblem(catalog, config)


def rerender_one(config_name: str) -> None:
    out_dir, X, F, G = _load_run(config_name)
    problem = _problem_for(config_name)
    catalog = problem.catalog

    feasible_mask = np.all(G <= 0, axis=1)
    finite = ~np.isinf(F).any(axis=1)
    keep = feasible_mask & finite

    print(f"\n=== {config_name} ===")
    print(f"  rows={len(X)}  feasible={keep.sum()}")

    if keep.any():
        feasible_indices = np.where(keep)[0]
        best_idx = feasible_indices[np.argmin(F[feasible_indices, 0])]
        graph = decode_chromosome(
            X[best_idx], problem.dims, catalog, problem.decoder_config,
        )
        layout = port_graph_to_layout(graph, catalog)
        title = (
            f"Best Feasible ({graph.n_slots} pcs, "
            f"{-F[best_idx, 0]:.1%} util, "
            f"{-F[best_idx, 1]:.2f} m/s, "
            f"{graph.n_components} comp, {graph.n_cycles} cycle)"
        )
        save_path = out_dir / "best_layout.png"
        plot_layout(layout, catalog, problem.config.boundary, title,
                    save_path=save_path)
        print(f"  rendered best_layout.png -> {save_path}")

    infeasible_finite = (~feasible_mask) & finite
    if infeasible_finite.any():
        infeas_idx = np.where(infeasible_finite)[0]
        best_inf = infeas_idx[np.argmin(F[infeas_idx, 0])]
        graph_inf = decode_chromosome(
            X[best_inf], problem.dims, catalog, problem.decoder_config,
        )
        layout_inf = port_graph_to_layout(graph_inf, catalog)
        cv = float(np.sum(np.maximum(0, G[best_inf])))
        title = (
            f"Best Infeasible ({graph_inf.n_slots} pcs, "
            f"{-F[best_inf, 0]:.1%} util, "
            f"{-F[best_inf, 1]:.2f} m/s, "
            f"{graph_inf.n_components} comp, {graph_inf.n_cycles} cycle, "
            f"CV={cv:.2f})"
        )
        save_path = out_dir / "best_infeasible.png"
        plot_layout(layout_inf, catalog, problem.config.boundary, title,
                    save_path=save_path)
        print(f"  rendered best_infeasible.png -> {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="config names (e.g. default with_switches)")
    args = parser.parse_args()
    for name in args.configs:
        rerender_one(name)


if __name__ == "__main__":
    main()
