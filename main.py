"""Main entry point for LEGO Track Optimizer.

NSGA-II multi-objective optimization with partitioned chromosome encoding.
Maximizes piece utilization and train speed with template-based passing sidings.
"""

import argparse
import logging
import re
from pathlib import Path

from src.algorithm import run_optimization, save_results
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.run_info import append_run_summary, write_run_info_header


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if verbose:
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)


def next_output_dir(config_path: str, base: str = "outputs") -> Path:
    """Auto-named output dir that never overwrites a previous run.

    Returns ``<base>/verify_<config>_<N>`` where ``<config>`` is the config
    file stem and ``N`` is one past the highest existing run for that config.
    A bare ``verify_<config>`` (no number) counts as 1.
    """
    prefix = f"verify_{Path(config_path).stem}"
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?$")
    base_dir = Path(base)
    highest = 0
    if base_dir.is_dir():
        for entry in base_dir.iterdir():
            match = pattern.match(entry.name) if entry.is_dir() else None
            if match:
                highest = max(highest, int(match.group(1)) if match.group(1) else 1)
    return base_dir / f"{prefix}_{highest + 1}"


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LEGO Track Optimizer - NSGA-II multi-objective layout optimization"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results. If omitted, auto-named "
                             "outputs/verify_<config>_<N>, with N one past the highest "
                             "existing run for that config (never overwrites).")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test (20 generations, pop_size=20)")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading configuration from {args.config}")
    config = OptimizationConfig.load(args.config)

    if args.quick_test:
        config.algorithm.n_gen = 20
        config.algorithm.pop_size = 20
        logger.info("Quick test mode: 20 generations, pop_size=20")

    logger.info("Loading track catalog from data/track_pieces_v2.yaml")
    catalog = TrackCatalog.load("data/track_pieces_v2.yaml")

    output_dir = Path(args.output) if args.output else next_output_dir(args.config)
    logger.info(f"Saving results to {output_dir}")
    write_run_info_header(output_dir, args.config, config, quick_test=args.quick_test)
    res = run_optimization(config, catalog, verbose=args.verbose, output_dir=output_dir)
    save_results(res, output_dir, catalog, config)
    append_run_summary(output_dir, res, catalog, config)

    logger.info("Done!")


if __name__ == "__main__":
    main()
