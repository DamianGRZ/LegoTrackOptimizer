"""Ablation driver: the same problem solved with and without each custom component.

The baseline arm is stock pymoo -- the problem definition (encoding, decoder,
objectives, constraints) is kept, everything added on top of pymoo is switched
off. Each further arm re-enables one component so its contribution can be read
off against that baseline.

Runs are sequential and foreground; every arm uses the config's own pop_size and
n_gen, so budgets are never shortened to make the campaign cheaper.

``--crowding`` is a second axis crossed with the arms: the survival operator's
crowding metric is an algorithm parameter rather than a component toggle, so a
cell is identified by (config, arm, crowding metric, seed).
"""

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import get_args

import numpy as np

from src.algorithm import run_optimization, save_results
from src.catalog import TrackCatalog
from src.config import AlgorithmConfig, OptimizationConfig
from src.run_info import append_run_summary, write_run_info_header

CONFIG_DIR = Path("configs")
OUTPUT_ROOT = Path("outputs/ablation")
CATALOG_PATH = "data/track_pieces_v2.yaml"

# Component sets per arm. The baseline holds every custom component off; the
# single-component arms answer "what does this one add on its own?", and `full`
# is the production system.
ARMS: dict[str, dict[str, bool]] = {
    "baseline": dict(heuristic_sampling=False, custom_operators=False, repair=False,
                     adaptive_epsilon=False, elite_injection=False, constr_survival=False),
    "seeding": dict(heuristic_sampling=True, custom_operators=False, repair=False,
                    adaptive_epsilon=False, elite_injection=False, constr_survival=False),
    "operators": dict(heuristic_sampling=False, custom_operators=True, repair=False,
                      adaptive_epsilon=False, elite_injection=False, constr_survival=False),
    "repair": dict(heuristic_sampling=False, custom_operators=False, repair=True,
                   adaptive_epsilon=False, elite_injection=False, constr_survival=False),
    "full": dict(heuristic_sampling=True, custom_operators=True, repair=True,
                 adaptive_epsilon=True, elite_injection=True, constr_survival=True),
    # Leave-one-out: the gap to `full` prices a component inside the working
    # system, where the single-component arms measure it in isolation instead.
    # The two readings can disagree in sign, so both are needed.
    "full_minus_operators": dict(heuristic_sampling=True, custom_operators=False, repair=True,
                                 adaptive_epsilon=True, elite_injection=True,
                                 constr_survival=True),
    "full_minus_seeding": dict(heuristic_sampling=False, custom_operators=True, repair=True,
                               adaptive_epsilon=True, elite_injection=True,
                               constr_survival=True),
    "full_minus_repair": dict(heuristic_sampling=True, custom_operators=True, repair=False,
                              adaptive_epsilon=True, elite_injection=True,
                              constr_survival=True),
    # Prices the epsilon schedule together with the x1000 hard-constraint CV
    # weighting: both are set inside LegoAdaptiveEpsilon and cannot be
    # separated by this flag.
    "full_minus_epsilon": dict(heuristic_sampling=True, custom_operators=True, repair=True,
                               adaptive_epsilon=False, elite_injection=True,
                               constr_survival=True),
    "full_minus_elites": dict(heuristic_sampling=True, custom_operators=True, repair=True,
                              adaptive_epsilon=True, elite_injection=False,
                              constr_survival=True),
}


CROWDING_FUNCS = get_args(AlgorithmConfig.model_fields["crowding_func"].annotation)


def config_names() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def arm_dir_name(arm: str, crowding_func: str) -> str:
    """Cell directory for an (arm, crowding metric) pair.

    ``cd`` keeps the bare arm name so archives written before the metric became
    a campaign axis stay resumable and comparable.
    """
    return arm if crowding_func == "cd" else f"{arm}_{crowding_func}"


def load_arm_config(config_name: str, arm: str, seed: int,
                    crowding_func: str) -> OptimizationConfig:
    """Config for one cell, loaded from its original path so relative paths resolve.

    ``termination.period`` is forced off: an improvement-based early stop would
    hand different arms different budgets and make the comparison meaningless.
    It can only lengthen a run, never shorten one.
    """
    config = OptimizationConfig.load(CONFIG_DIR / f"{config_name}.yaml")
    for flag, value in ARMS[arm].items():
        setattr(config.algorithm.components, flag, value)
    config.algorithm.crowding_func = crowding_func
    config.algorithm.seed = seed
    config.algorithm.termination.period = 0
    return config


def already_done(run_dir: Path) -> bool:
    """A finished run has a summary appended; a header alone means it died."""
    info = run_dir / "run_info.md"
    return info.exists() and "## Run Summary" in info.read_text(encoding="utf-8")


@contextmanager
def _cell_log(run_dir: Path):
    """Tee log records into ``<run_dir>/run.log`` for the duration of one cell.

    Opened for writing, so a cell retried after an interruption replaces the
    abandoned attempt's log instead of appending a second run to it. Attached to
    the root logger rather than ``src``, so this cell's failure record lands here
    too — ``run_cell`` reports a crash through the root logger.
    """
    handler = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def run_cell(config_name: str, arm: str, seed: int, crowding_func: str,
             catalog: TrackCatalog) -> dict:
    arm_dir = arm_dir_name(arm, crowding_func)
    run_dir = OUTPUT_ROOT / config_name / arm_dir / f"s{seed}"
    record = {"config": config_name, "arm": arm_dir, "base_arm": arm,
              "crowding_func": crowding_func, "seed": seed, "run_dir": str(run_dir)}
    if already_done(run_dir):
        return {**record, "status": "skipped"}

    config = load_arm_config(config_name, arm, seed, crowding_func)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Read the flags back off the loaded model: pydantic silently ignores an
    # unknown key, so the intended patch is not evidence of what actually ran.
    (run_dir / "arm.json").write_text(
        json.dumps({**record, "components": config.algorithm.components.model_dump(),
                    "resolved_crowding_func": config.algorithm.crowding_func,
                    "pop_size": config.algorithm.pop_size,
                    "n_gen": config.algorithm.n_gen}, indent=2),
        encoding="utf-8",
    )

    write_run_info_header(run_dir, CONFIG_DIR / f"{config_name}.yaml", config)
    started = time.perf_counter()
    with _cell_log(run_dir):
        try:
            res = run_optimization(config, catalog, verbose=False, output_dir=run_dir)
            save_results(res, run_dir, catalog, config)
            append_run_summary(run_dir, res, catalog, config)
        except Exception as exc:
            logging.exception(f"{config_name}/{arm_dir}/s{seed} failed")
            (run_dir / "error.md").write_text(f"# Run failed\n\n```\n{exc}\n```\n",
                                              encoding="utf-8")
            return {**record, "status": "crashed",
                    "elapsed_s": time.perf_counter() - started}

    # save_results tolerates a result carrying no population, so the feasibility
    # census degrades the same way: the artifacts are written either way, and an
    # unreadable census must not take the rest of the campaign down with it.
    pop = getattr(res, "pop", None)
    G = pop.get("G") if pop is not None else None
    n_feas = int(np.all(G <= 0, axis=1).sum()) if G is not None else 0
    return {**record, "status": "ok", "n_feasible": n_feas,
            "elapsed_s": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="baseline",
                        help="comma-separated arm names (default: baseline only)")
    parser.add_argument("--configs", default=None,
                        help="comma-separated config stems (default: every config)")
    parser.add_argument("--seeds", default="1", help="comma-separated seeds")
    parser.add_argument("--crowding", default="cd",
                        help=f"comma-separated crowding metrics, crossed with the arms "
                             f"(default: cd; available: {', '.join(CROWDING_FUNCS)})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    arms = [a.strip() for a in args.arms.split(",")]
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}; available: {sorted(ARMS)}")
    crowdings = [c.strip() for c in args.crowding.split(",")]
    unknown = set(crowdings) - set(CROWDING_FUNCS)
    if unknown:
        parser.error(f"unknown crowding metrics: {sorted(unknown)}; "
                     f"available: {sorted(CROWDING_FUNCS)}")
    configs = ([c.strip() for c in args.configs.split(",")] if args.configs
               else config_names())
    seeds = [int(s) for s in args.seeds.split(",")]

    catalog = TrackCatalog.load(CATALOG_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Crowding metric innermost: both members of a pair run back to back, so an
    # interrupted campaign still leaves whole arms paired rather than one metric
    # complete and the other empty.
    for arm in arms:
        for config_name in configs:
            for seed in seeds:
                for crowding_func in crowdings:
                    logging.info(f"--- {config_name} / {arm_dir_name(arm, crowding_func)}"
                                 f" / seed {seed} ---")
                    row = run_cell(config_name, arm, seed, crowding_func, catalog)
                    rows.append(row)
                    logging.info(f"    {row['status']}"
                                 + (f", feasible {row['n_feasible']}, "
                                    f"{row['elapsed_s']:.0f}s"
                                    if row["status"] == "ok" else ""))

    manifest = OUTPUT_ROOT / "manifest.json"
    existing = (json.loads(manifest.read_text(encoding="utf-8"))
                if manifest.exists() else [])
    manifest.write_text(json.dumps(existing + rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if r["status"] == "ok"]
    solved = [r for r in ok if r["n_feasible"] > 0]
    print(f"\n{len(ok)} runs, {len(solved)} produced at least one feasible layout")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['config']:30s} {r['arm']:26s} {r['status']}")
            continue
        print(f"  {r['config']:30s} {r['arm']:26s} feasible {r['n_feasible']:5d}"
              f"  {r['elapsed_s']:6.0f}s")


if __name__ == "__main__":
    main()
