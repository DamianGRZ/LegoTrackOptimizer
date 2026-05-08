"""Run V2 optimization on all shared configs and collect summary statistics."""

import logging
import time
from pathlib import Path

import numpy as np

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.decoder import decode_chromosome
from src_v2.encoding import iter_active_slots
from src_v2.problem import PortPairProblem
from src_v2.runner import run_optimization, save_results


CONFIGS = ["default", "with_switches", "with_crossing"]
POP_SIZE = 500
N_GEN = 200
HEURISTIC_RATIO = 0.30


def run_one(config_name: str, catalog: TrackCatalog) -> dict:
    config = OptimizationConfig.load(Path(f"configs/{config_name}.yaml"))
    config.algorithm.pop_size = POP_SIZE
    config.algorithm.n_gen = N_GEN
    config.algorithm.heuristic_ratio = HEURISTIC_RATIO

<<<<<<< Updated upstream
    out_dir = Path(f"outputs_v2/{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nV2 {config_name}: pop={POP_SIZE}, gen={N_GEN}\n{'='*70}")
    t0 = time.time()
    res = run_optimization(config, catalog, verbose=True, output_dir=out_dir)
    elapsed = time.time() - t0

=======
    print(f"\n{'='*70}\nV2 {config_name}: pop={POP_SIZE}, gen={N_GEN}\n{'='*70}")
    t0 = time.time()
    res = run_optimization(config, catalog, verbose=True)
    elapsed = time.time() - t0

    out_dir = Path(f"outputs_v2/{config_name}")
>>>>>>> Stashed changes
    save_results(res, out_dir, catalog, config)

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
        problem = PortPairProblem(catalog, config)
        graph = decode_chromosome(
            res.pop.get("X")[best_idx], problem.dims, catalog, problem.decoder_config,
        )
        counts: dict = {}
        for _, idx in iter_active_slots(res.pop.get("X")[best_idx], problem.dims):
            pid = catalog.index_to_id.get(idx, f"?{idx}")
            counts[pid] = counts.get(pid, 0) + 1
        summary.update({
            "best_util": float(-F[best_idx, 0]),
            "best_min_speed": float(-F[best_idx, 1]),
            "n_pieces": graph.n_slots,
            "n_cycles": graph.n_cycles,
            "n_components": graph.n_components,
            "closure_pos": graph.max_closure_position,
            "closure_angle_deg": graph.max_closure_angle_deg,
            "piece_counts": counts,
        })

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    catalog = TrackCatalog.load(Path("data/track_pieces_v2.yaml"))

    summaries = []
    total_t0 = time.time()
    for cfg in CONFIGS:
        summaries.append(run_one(cfg, catalog))
    total_elapsed = time.time() - total_t0

    print(f"\n\n{'='*70}\nV2 SUMMARY (total {total_elapsed:.1f}s)\n{'='*70}")
    for s in summaries:
        print(f"\n{s['config']}:")
        print(f"  elapsed:        {s['elapsed_sec']:.1f}s")
        print(f"  feasible:       {s['n_feasible']}/{s['pop_size']}")
        if "best_util" in s:
            print(f"  best util:      {s['best_util']:.1%} ({s['n_pieces']} pieces)")
            print(f"  best min_speed: {s['best_min_speed']:.2f} m/s")
            print(f"  cycles:         {s['n_cycles']}")
            print(f"  components:     {s['n_components']}")
            print(f"  closure pos:    {s['closure_pos']:.3f} studs")
            print(f"  closure angle:  {s['closure_angle_deg']:.3f} deg")
            print(f"  pieces used:")
            for pid, c in sorted(s['piece_counts'].items()):
                print(f"    {pid:30} = {c}")
        else:
            print("  no feasible solutions")


if __name__ == "__main__":
    main()
