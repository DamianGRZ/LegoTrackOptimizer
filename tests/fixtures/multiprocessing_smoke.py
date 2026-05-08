"""2-worker StarmapParallelization smoke fixture (Phase 17.C.4, PLAN Section 10.6).

Wraps :func:`tests.fixtures.mini_problem.mini_optimization_run` with
``n_workers=2`` so callers get one-line access to a multiprocessing-flavored
mini-opt run.

What this validates (Rule 1's positive half):

- Standard pymoo ``out`` keys (``F``, ``G``, ``dF``, ``dG``, ``pheno``,
  ``feasible``) round-trip correctly from worker → main process.
- The ``Problem``, ``Operators``, ``Repair``, ``Sampling`` objects all pickle
  cleanly (else ``Pool(2)`` setup would crash with ``PicklingError``).

What this does NOT validate (deliberately, see PLAN §6 Risk 11 + Rule 24):

- Custom ``out`` keys (``out["graph_hash"]``, ``out["topology_sig"]``, etc.)
  are dropped silently by pymoo's ``_eval_elementwise`` — testing this needs
  a negative-path fixture that lives with Phase 8's archive admission code,
  not here.
- ``out["pheno"]`` round-trip with a Phase-1 ``PortGraph`` payload — Phase 1
  test 1.6 (Coupling B) is the formal check; not preempting here.
"""
from __future__ import annotations

from pathlib import Path

from .mini_problem import mini_optimization_run


def multiprocessing_smoke_run(output_dir: Path, **kwargs) -> object:
    """Run :func:`mini_optimization_run` with ``n_workers=2``.

    Returns the same pymoo ``Result`` shape as ``mini_optimization_run``;
    output files (``diagnostics.csv``, ``snapshots/``, ``epsilon_archive.json``)
    land in ``output_dir`` per the standard layout.
    """
    return mini_optimization_run(output_dir, n_workers=2, **kwargs)
