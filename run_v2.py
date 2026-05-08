"""Run V2 optimization on a single config.

Usage:
    python run_v2.py <config_name> [--pop 500] [--gen 200] [--heuristic 0.30]

Examples:
    python run_v2.py default
    python run_v2.py with_switches --gen 500
    python run_v2.py with_crossing --pop 1000 --gen 300
"""

import argparse
import logging
import time
from pathlib import Path

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.runner import run_optimization, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 port-pair optimizer (single config).")
    parser.add_argument("config", help="Config name (without .yaml), e.g. 'default'")
    parser.add_argument("--pop", type=int, default=500, help="Population size")
    parser.add_argument("--gen", type=int, default=200, help="Number of generations")
    parser.add_argument("--heuristic", type=float, default=0.30,
                        help="Heuristic seed ratio in [0, 1]")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")

    catalog = TrackCatalog.load(Path("data/track_pieces_v2.yaml"))
    config = OptimizationConfig.load(Path(f"configs/{args.config}.yaml"))
    config.algorithm.pop_size = args.pop
    config.algorithm.n_gen = args.gen
    config.algorithm.heuristic_ratio = args.heuristic

<<<<<<< Updated upstream
    out_dir = Path(f"outputs_v2/{args.config}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = run_optimization(config, catalog, verbose=True, output_dir=out_dir)
    elapsed = time.time() - t0

=======
    t0 = time.time()
    res = run_optimization(config, catalog, verbose=True)
    elapsed = time.time() - t0

    out_dir = Path(f"outputs_v2/{args.config}")
>>>>>>> Stashed changes
    save_results(res, out_dir, catalog, config)
    print(f"\nDone in {elapsed:.1f}s -> {out_dir}/")


if __name__ == "__main__":
    main()
