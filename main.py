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
    """Configure logging level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if verbose:
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LEGO Track Optimizer - NSGA-II multi-objective layout optimization"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for results")
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

    logger.info("Loading track catalog from data/track_pieces.yaml")
    catalog = TrackCatalog.load("data/track_pieces.yaml")

    output_dir = Path(args.output)
    res = run_optimization(config, catalog, verbose=args.verbose)
    save_results(res, output_dir, catalog, config)

    logger.info("Done!")


if __name__ == "__main__":
    main()
