"""LEGO Track Optimizer — port-pair encoding (v2).

Reused from src/ verbatim:
    train/, types.py, visualization/

Reused as transitive dependency:
    catalog/, config.py, geometry.py, lego_track_models.py

Pending port-pair adaptation:
    sampling.py — emits V1 partitioned chromosomes; pattern logic preserved
                  but operator output format must be rewritten.

To be created:
    encoding.py    — port-pair gene layout + dimensions
    decoder.py     — port-graph FK propagation + path enumeration
    operators.py   — port-pair-aware Sampling/Crossover/Mutation
    repair.py      — port-graph validity (1 port -> 1 pair, connectedness)
    problem.py     — pymoo ElementwiseProblem
    algorithm/     — NSGA-II runner
"""

# Force the non-interactive Agg matplotlib backend BEFORE any submodule
# imports pyplot. Tk-based backends crash under multiprocessing.Pool on
# Windows ("Tcl_AsyncDelete: async handler deleted by the wrong thread")
# because SnapshotCallback renders PNGs in the main process while worker
# pipes are still open. Agg is pure file-output, no GUI thread needed.
import matplotlib as _matplotlib

_matplotlib.use("Agg")

from .config import OptimizationConfig
from .catalog import TrackCatalog

__all__ = [
    "OptimizationConfig",
    "TrackCatalog",
]
