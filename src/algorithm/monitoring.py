from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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

_CLOSURE_COLUMNS = ("mean_closure_x", "mean_closure_y", "mean_closure_theta")

_NAN = float("nan")


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
    - ``mean_closure_x/y`` are in studs and ``mean_closure_theta`` in degrees,
      de-normalized from G[0..2] with the problem's configured tolerances.
      Without ``closure_tolerance``/``angle_tolerance`` they are NaN — the G
      scale is config-dependent, so no default can be correct.
    """

    def __init__(
        self,
        ref_point: tuple[float, float] = (0.10, -0.55),
        pareto_ref: np.ndarray | None = None,
        output_dir: Path | None = None,
        closure_tolerance: float | None = None,
        angle_tolerance: float | None = None,
    ) -> None:
        super().__init__()
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self.igd = IGD(pareto_ref) if pareto_ref is not None else None
        self.closure_tolerance = closure_tolerance
        self.angle_tolerance = angle_tolerance
        self.data.update((k, []) for k in _CSV_COLUMNS)
        self._best_F: np.ndarray | None = None
        # Set by run_optimization to the LegoAdaptiveEpsilon instance so the
        # live epsilon lands next to the feasibility trajectory it explains.
        self.epsilon_source = None
        self._csv_path = self._init_csv(output_dir)
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
        F, CV, G = pop.get("F"), pop.get("CV"), pop.get("G")
        if F is None or CV is None:
            return

        feas_mask = CV.ravel() <= 0.0
        F_feas = F[feas_mask]
        n_feas = int(feas_mask.sum())
        if n_feas > 0:
            self._best_F = self._update_best_front(F_feas)

        row: dict[str, int | float] = {
            "n_gen": int(algorithm.n_gen),
            "n_eval": int(algorithm.evaluator.n_eval),
            "hv": float(self.hv.do(F_feas)) if n_feas else 0.0,
            "igd": self._igd(F_feas) if n_feas else _NAN,
            "n_feas": n_feas,
            "feas_rate": n_feas / max(1, len(pop)),
            "best_f0": float(F_feas[:, 0].min()) if n_feas else _NAN,
            "best_f1": float(F_feas[:, 1].min()) if n_feas else _NAN,
            "n_unique_F": len(np.unique(F, axis=0)),
            "n_unique_F_feas": len(np.unique(F_feas, axis=0)) if n_feas else 0,
            "cv_eps": self._cv_eps(),
            **self._mean_closure(G, feas_mask, n_feas),
            "gen_seconds": self._tick(),
        }
        for key in _CSV_COLUMNS:
            self.data[key].append(row[key])
        self._append_csv_row(row)

    # ------------------------------------------------------------------
    # Per-metric helpers

    def _igd(self, F_feas: np.ndarray) -> float:
        """IGD vs the external reference if given, else vs the rolling front."""
        indicator = self.igd if self.igd is not None else IGD(self._best_F)
        return float(indicator.do(F_feas))

    def _cv_eps(self) -> float:
        if self.epsilon_source is None:
            return _NAN
        return float(getattr(self.epsilon_source, "last_cv_eps", _NAN))

    def _mean_closure(
        self, G: np.ndarray | None, feas_mask: np.ndarray, n_feas: int,
    ) -> dict[str, float]:
        """Mean feasible closure residuals, de-normalized to studs/degrees.

        G[0..2] are ``|dx|/closure_tol - 1``, ``|dy|/closure_tol - 1`` and
        ``|dtheta_deg|/angle_tol - 1`` (see ``TrackOptimizationProblem``).
        """
        if (G is None or G.shape[1] < 3 or n_feas == 0
                or self.closure_tolerance is None or self.angle_tolerance is None):
            return dict.fromkeys(_CLOSURE_COLUMNS, _NAN)
        means = (G[feas_mask, :3] + 1.0).mean(axis=0)
        scales = (self.closure_tolerance, self.closure_tolerance, self.angle_tolerance)
        return {k: float(m * s) for k, m, s in zip(_CLOSURE_COLUMNS, means, scales)}

    def _tick(self) -> float:
        """Seconds since the previous notify (or since construction)."""
        now = time.perf_counter()
        seconds = now - self._last_notify_time
        self._last_notify_time = now
        return seconds

    def _update_best_front(self, F_new: np.ndarray) -> np.ndarray:
        if self._best_F is None or len(self._best_F) == 0:
            combined = F_new
        else:
            combined = np.vstack([self._best_F, F_new])
        # Deduplicate BEFORE sorting: identical F rows are mutually
        # non-dominating, so the archive would otherwise accumulate every
        # generation's repeated points and the NDS cost would grow
        # quadratically over the run.
        combined = np.unique(combined, axis=0)
        idx = NonDominatedSorting().do(combined, only_non_dominated_front=True)
        return combined[idx]

    # ------------------------------------------------------------------
    # CSV persistence

    @staticmethod
    def _init_csv(output_dir: Path | None) -> Path | None:
        if output_dir is None:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "convergence.csv"
        # Construction time == run start: a fresh run must not append to a
        # previous run's trajectory.
        path.unlink(missing_ok=True)
        return path

    def _append_csv_row(self, row: dict[str, int | float]) -> None:
        if self._csv_path is None:
            return
        cells = (str(v) if isinstance(v, int) else f"{v:.6g}"
                 for v in (row[k] for k in _CSV_COLUMNS))
        with self._csv_path.open("a", encoding="utf-8") as fh:
            if fh.tell() == 0:
                fh.write(",".join(_CSV_COLUMNS) + "\n")
            fh.write(",".join(cells) + "\n")