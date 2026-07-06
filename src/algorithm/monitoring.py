from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


_S_XY = 0.5
_S_THETA = math.pi / 180  # 1 degree in rad — used only for de-normalizing mean_closure_*

# Column order of convergence.csv (and of the per-generation row values).
_CSV_COLUMNS = (
    "n_gen", "n_eval",
    "hv", "igd",
    "n_feas", "feas_rate",
    "best_f0", "best_f1",
    "n_unique_F", "n_unique_F_feas",
    "cv_eps",
    "mean_closure_x", "mean_closure_y", "mean_closure_theta",
    "gen_seconds",
)


class ConvergenceMonitorCallback(Callback):
    """Per-generation: HV, IGD, feasibility rate, mean closure residuals.

    With ``output_dir`` set, every generation is also appended to
    ``<output_dir>/convergence.csv`` as it happens — the incremental append
    doubles as crash forensics (a dead run still leaves its full trajectory).

    Caveats for consumers:
    - ``igd`` without an external ``pareto_ref`` is measured against the
      rolling best-known front, so values are NOT comparable across
      generations (the reference itself moves).
    - ``cv_eps`` is read from ``epsilon_source.last_cv_eps`` when a source is
      attached (see ``LegoAdaptiveEpsilon``); NaN otherwise.
    """

    def __init__(
        self,
        ref_point: tuple[float, float] = (0.10, -0.55),
        pareto_ref: np.ndarray | None = None,
        output_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self._pareto_ref = pareto_ref
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        for k in _CSV_COLUMNS:
            self.data[k] = []
        self._best_F: np.ndarray | None = None
        # Set by run_optimization to the LegoAdaptiveEpsilon instance so the
        # live epsilon lands next to the feasibility trajectory it explains.
        self.epsilon_source = None
        self._csv_path: Path | None = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = output_dir / "convergence.csv"
            # Construction time == run start: a fresh run must not append to
            # a previous run's trajectory.
            self._csv_path.unlink(missing_ok=True)
        self._last_notify_time = time.perf_counter()

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

        best_f0 = float(np.min(F_feas[:, 0])) if n_feas > 0 else float("nan")
        best_f1 = float(np.min(F_feas[:, 1])) if n_feas > 0 else float("nan")
        n_unique_F = int(len(np.unique(F, axis=0)))
        n_unique_F_feas = int(len(np.unique(F_feas, axis=0))) if n_feas > 0 else 0

        cv_eps = float("nan")
        if self.epsilon_source is not None:
            cv_eps = float(getattr(self.epsilon_source, "last_cv_eps", float("nan")))

        # De-normalize closure residuals from G for human-readable logging.
        # G[0..2] are the per-axis closure residuals (dx, dy, dtheta).
        mean_cx = mean_cy = mean_ct = float("nan")
        if G is not None and G.shape[1] >= 3 and n_feas > 0:
            G_feas = G[feas_mask]
            mean_cx = float(np.mean((G_feas[:, 0] + 1.0) * _S_XY))
            mean_cy = float(np.mean((G_feas[:, 1] + 1.0) * _S_XY))
            mean_ct = float(np.mean((G_feas[:, 2] + 1.0) * _S_THETA))

        now = time.perf_counter()
        gen_seconds = now - self._last_notify_time
        self._last_notify_time = now

        row = {
            "n_gen": int(algorithm.n_gen),
            "n_eval": int(algorithm.evaluator.n_eval),
            "hv": hv_val,
            "igd": igd_val,
            "n_feas": n_feas,
            "feas_rate": n_feas / pop_size,
            "best_f0": best_f0,
            "best_f1": best_f1,
            "n_unique_F": n_unique_F,
            "n_unique_F_feas": n_unique_F_feas,
            "cv_eps": cv_eps,
            "mean_closure_x": mean_cx,
            "mean_closure_y": mean_cy,
            "mean_closure_theta": mean_ct,
            "gen_seconds": gen_seconds,
        }
        for key in _CSV_COLUMNS:
            self.data[key].append(row[key])
        self._append_csv_row(row)

    def _append_csv_row(self, row: dict) -> None:
        if self._csv_path is None:
            return
        write_header = not self._csv_path.exists()
        cells = []
        for key in _CSV_COLUMNS:
            v = row[key]
            cells.append(str(v) if isinstance(v, int) else f"{v:.6g}")
        with self._csv_path.open("a", encoding="utf-8") as fh:
            if write_header:
                fh.write(",".join(_CSV_COLUMNS) + "\n")
            fh.write(",".join(cells) + "\n")

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
