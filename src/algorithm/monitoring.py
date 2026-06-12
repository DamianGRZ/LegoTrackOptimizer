"""Convergence instrumentation for NSGA-II: HV, IGD, feasibility rate.

HV reference point (+0.10, -0.55) is 10% beyond empirical nadir (0, -0.60) per
Ishibuchi et al. 2018 / Auger et al. 2009. The callback always filters to
feasible-only before computing HV/IGD — the +inf infeasibility sentinel never
reaches the indicators.

See docs/superpowers/plans/2026-04-20-batch-2-implementation-research.md §Problem Q5.
"""

from __future__ import annotations

import math
import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


_S_XY = 0.5
_S_THETA = math.pi / 180  # 1 degree in rad — used only for de-normalizing mean_closure_*


class ConvergenceMonitorCallback(Callback):
    """Per-generation: HV, IGD, feasibility rate, mean closure residuals."""

    def __init__(
        self,
        ref_point: tuple[float, float] = (0.10, -0.55),
        pareto_ref: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self._pareto_ref = pareto_ref
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        for k in (
            "n_gen", "n_eval",
            "hv", "igd",
            "n_feas", "feas_rate",
            "mean_closure_x", "mean_closure_y", "mean_closure_theta",
        ):
            self.data[k] = []
        self._best_F: np.ndarray | None = None

    @property
    def best_front(self) -> np.ndarray:
        """Run-cumulative non-dominated front of every feasible F seen.

        Selection discards mid-run trade-off solutions; this archive is the
        only record of them (the terminal population is a converged
        monoculture). Consumed by the Pareto plot and saved artifacts.
        """
        if self._best_F is None:
            return np.empty((0, 2))
        return self._best_F

    def notify(self, algorithm) -> None:
        pop = algorithm.pop
        F = pop.get("F")
        CV = pop.get("CV")
        G = pop.get("G")

        if F is None or CV is None:
            return

        feas_mask = CV.ravel() <= 0.0
        F_feas = F[feas_mask]
        n_feas = int(feas_mask.sum())
        pop_size = max(1, len(pop))

        hv_val = float(self.hv.do(F_feas)) if n_feas > 0 else 0.0

        igd_val = float("nan")
        if n_feas > 0:
            self._best_F = self._update_best_front(F_feas)
            if self._pareto_ref is not None and self.igd is not None:
                igd_val = float(self.igd.do(F_feas))
            elif self._best_F is not None and len(self._best_F) > 0:
                igd_val = float(IGD(self._best_F).do(F_feas))

        # De-normalize closure residuals from G for human-readable logging.
        # Shim-aware: post-Stage-B, G[0..2] are V2 per-axis closure; pre-Stage-B,
        # G[0..2] include the current magnitude closure + angle + boundary.
        # The block runs on any 3+-column G; meanings change with stage.
        mean_cx = mean_cy = mean_ct = float("nan")
        if G is not None and G.shape[1] >= 3 and n_feas > 0:
            G_feas = G[feas_mask]
            mean_cx = float(np.mean((G_feas[:, 0] + 1.0) * _S_XY))
            mean_cy = float(np.mean((G_feas[:, 1] + 1.0) * _S_XY))
            mean_ct = float(np.mean((G_feas[:, 2] + 1.0) * _S_THETA))

        self.data["n_gen"].append(algorithm.n_gen)
        self.data["n_eval"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(hv_val)
        self.data["igd"].append(igd_val)
        self.data["n_feas"].append(n_feas)
        self.data["feas_rate"].append(n_feas / pop_size)
        self.data["mean_closure_x"].append(mean_cx)
        self.data["mean_closure_y"].append(mean_cy)
        self.data["mean_closure_theta"].append(mean_ct)

    def _update_best_front(self, F_new: np.ndarray) -> np.ndarray:
        if self._best_F is None or len(self._best_F) == 0:
            combined = F_new
        else:
            combined = np.vstack([self._best_F, F_new])
        # Deduplicate BEFORE sorting: identical F rows are mutually
        # non-dominating, so without this the archive accumulates every
        # generation's repeated points and the NDS below grows quadratically
        # over the run (the runaway per-generation slowdown).
        combined = np.unique(combined, axis=0)
        idx = NonDominatedSorting().do(combined, only_non_dominated_front=True)
        return combined[idx]
