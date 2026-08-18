"""Batch-run the shared configs at matched parameters (one outputs/<config> dir each)."""

import logging
import time
from pathlib import Path

import numpy as np

from src.algorithm import run_optimization, save_results
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.decoder import DecoderConfig, decode_chromosome
from src.encoding import compute_dimensions


CONFIGS = ["default", "with_switches", "with_crossing"]
# Batch override: every config runs at these matched parameters so results are
# comparable across configs, intentionally ignoring each config's own pop/gen.
POP_SIZE = 500
N_GEN = 200
HEURISTIC_RATIO = 0.30


def run_one(config_name: str) -> dict:
    config = OptimizationConfig.load(Path(f"configs/{config_name}.yaml"))
    config.algorithm.pop_size = POP_SIZE
    config.algorithm.n_gen = N_GEN
    config.algorithm.heuristic_ratio = HEURISTIC_RATIO

    print(f"\n{'='*70}\n{config_name}: pop={POP_SIZE}, gen={N_GEN}\n{'='*70}")
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
        dims = compute_dimensions(config, catalog)
        layout = decode_chromosome(
            res.pop.get("X")[best_idx], catalog, config.inventory,
            dims=dims, config=DecoderConfig.from_optimization_config(config),
        )
        summary.update({
            # F[0] is negated utilization; F[1] is already the minimized
            # traversal time in seconds, so only the former flips sign.
            "best_util": float(-F[best_idx, 0]),
            "best_traversal_time": float(F[best_idx, 1]),
            "n_pieces": layout.n_physical_pieces,
            "n_switch_pairs": layout.n_switch_pairs,
        })

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    summaries = []
    total_t0 = time.time()
    for cfg in CONFIGS:
        # One config's crash must not kill the rest of the batch; the run's
        # own error.log (written by run_optimization) has the details.
        try:
            summaries.append(run_one(cfg))
        except Exception as exc:
            logging.exception(f"{cfg} failed")
            summaries.append({"config": cfg, "error": str(exc)})
    total_elapsed = time.time() - total_t0

    print(f"\n\n{'='*70}\nSUMMARY (total {total_elapsed:.1f}s)\n{'='*70}")
    for s in summaries:
        print(f"\n{s['config']}:")
        if "error" in s:
            print(f"  FAILED: {s['error']}")
            continue
        print(f"  elapsed:        {s['elapsed_sec']:.1f}s")
        print(f"  feasible:       {s['n_feasible']}/{s['pop_size']}")
        if "best_util" in s:
            print(f"  best util:      {s['best_util']:.1%} ({s['n_pieces']} pieces)")
            print(f"  best time:      {s['best_traversal_time']:.2f} s")
            print(f"  switch pairs:   {s['n_switch_pairs']}")
        else:
            print("  no feasible solutions")


if __name__ == "__main__":
    main()
