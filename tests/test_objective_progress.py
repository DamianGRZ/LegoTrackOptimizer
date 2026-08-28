"""convergence.csv -> objective-progress chart: column contract, NaN policy, rendering."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.algorithm.monitoring import _CSV_COLUMNS
from src.visualization import load_score_progress, plot_score_progress


def _write_convergence(path: Path, rows: list[dict[str, float]]) -> Path:
    """Write convergence.csv the way the monitor does: bare comma-joined
    header from ``_CSV_COLUMNS``, one row per generation, absent metrics nan.
    Renaming a column the loader depends on must break these tests."""
    csv_path = path / "convergence.csv"
    lines = [",".join(_CSV_COLUMNS)]
    lines += [",".join(f"{row.get(col, float('nan')):.6g}" for col in _CSV_COLUMNS)
              for row in rows]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


class TestLoadScoreProgress:
    def test_reads_generations_and_negated_scores(self, tmp_path):
        csv = _write_convergence(tmp_path, [
            {"n_gen": 1, "best_f0": -0.25},
            {"n_gen": 2, "best_f0": -0.40},
        ])
        generations, scores = load_score_progress(csv)
        np.testing.assert_allclose(generations, [1.0, 2.0])
        np.testing.assert_allclose(scores, [0.25, 0.40])

    def test_generations_without_feasible_are_dropped(self, tmp_path):
        csv = _write_convergence(tmp_path, [
            {"n_gen": 1, "best_f0": float("nan")},
            {"n_gen": 2, "best_f0": -0.10},
        ])
        generations, scores = load_score_progress(csv)
        np.testing.assert_allclose(generations, [2.0])
        np.testing.assert_allclose(scores, [0.10])

    def test_single_row_csv_still_yields_arrays(self, tmp_path):
        csv = _write_convergence(tmp_path, [{"n_gen": 1, "best_f0": -0.5}])
        generations, scores = load_score_progress(csv)
        assert generations.shape == (1,)
        assert scores.shape == (1,)


class TestPlotScoreProgress:
    def test_saves_png_with_ceiling_and_planned_budget(self, tmp_path):
        out = tmp_path / "objective_progress.png"
        fig = plot_score_progress([1, 2, 3], [0.1, 0.2, 0.3], max_score=1.2,
                                  save_path=out, n_gen_planned=10)
        plt.close(fig)
        assert out.stat().st_size > 0

    def test_no_feasible_data_still_renders(self, tmp_path):
        out = tmp_path / "empty.png"
        fig = plot_score_progress([], [], save_path=out)
        plt.close(fig)
        assert out.stat().st_size > 0
