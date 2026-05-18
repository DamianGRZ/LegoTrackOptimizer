"""Run V1 optimization on all shared configs at matched V2 parameters."""

import logging
import time
from pathlib import Path

import numpy as np

from src.algorithm import run_optimization, save_results
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import decode_chromosome


CONFIGS = ["default", "with_switches", "with_crossing"]
POP_SIZE = 500
N_GEN = 200
HEURISTIC_RATIO = 0.30


def run_one(config_name: str) -> dict:
    config = OptimizationConfig.load(Path(f"configs/{config_name}.yaml"))
    config.algorithm.pop_size = POP_SIZE
    config.algorithm.n_gen = N_GEN
    config.algorithm.heuristic_ratio = HEURISTIC_RATIO

    print(f"\n{'='*70}\nV1 {config_name}: pop={POP_SIZE}, gen={N_GEN}\n{'='*70}")
    catalog = TrackCatalog.load("data/track_pieces_v2.yaml")
    output_dir = Path(f"outputs/{config_name}")

    t0 = time.time()
    res = run_optimization(config, catalog, verbose=True, output_dir=output_dir)
    elapsed = time.time() - t0
    save_results(res, output_dir, catalog, config)

    F = res.pop.get("F")
    G = res.pop.get("G")
    feasible = np.all(G <= 0, axis=1) if G is not None else np.zeros(len(F), dtype=bool)
    n_feasible = int(feasible.sum())

    summary: dict = {
        "config": config_name,
        "elapsed_sec": elapsed,
        "n_feasible": n_feasible,
        "pop_size": POP_SIZE,
    }

    if n_feasible > 0:
        feas_idx = np.where(feasible)[0]
        best_idx = feas_idx[np.argmin(F[feas_idx, 0])]
        from src.encoding import compute_dimensions
        dims = compute_dimensions(config, catalog)
        layout = decode_chromosome(
            res.pop.get("X")[best_idx], catalog, config.inventory, dims=dims,
        )
        summary.update({
            "best_util": float(-F[best_idx, 0]),
            "best_min_speed": float(-F[best_idx, 1]),
            "n_pieces": layout.n_pieces,
            "n_switch_pairs": layout.n_switch_pairs,
        })

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    summaries = []
    total_t0 = time.time()
    for cfg in CONFIGS:
        summaries.append(run_one(cfg))
    total_elapsed = time.time() - total_t0

    print(f"\n\n{'='*70}\nV1 SUMMARY (total {total_elapsed:.1f}s)\n{'='*70}")
    for s in summaries:
        print(f"\n{s['config']}:")
        print(f"  elapsed:        {s['elapsed_sec']:.1f}s")
        print(f"  feasible:       {s['n_feasible']}/{s['pop_size']}")
        if "best_util" in s:
            print(f"  best util:      {s['best_util']:.1%} ({s['n_pieces']} pieces)")
            print(f"  best min_speed: {s['best_min_speed']:.2f} m/s")
            print(f"  switch pairs:   {s['n_switch_pairs']}")
        else:
            print("  no feasible solutions")


if __name__ == "__main__":
    main()
