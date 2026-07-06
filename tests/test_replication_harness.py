"""Replication-harness helpers: seed parsing, result reading, summary."""

import numpy as np
import pytest

from run_replications import (
    best_feasible_util,
    parse_seeds,
    resolve_config,
    summarize,
)


class TestSeedSpec:
    def test_range(self):
        assert parse_seeds("1..5") == [1, 2, 3, 4, 5]

    def test_list(self):
        assert parse_seeds("1,2,5") == [1, 2, 5]

    def test_single(self):
        assert parse_seeds("42") == [42]


class TestResolveConfig:
    def test_name_maps_to_configs_dir(self):
        assert resolve_config("all_pieces").as_posix() == "configs/all_pieces.yaml"

    def test_path_used_verbatim(self):
        assert resolve_config("configs/compact.yaml").as_posix() == "configs/compact.yaml"


class TestBestFeasibleUtil:
    def _write_run(self, path, fitness_rows, constraint_rows):
        (path / "fitness.csv").write_text(
            "neg_utilization,neg_slowest_route_speed\n" + fitness_rows,
            encoding="utf-8",
        )
        (path / "constraints.csv").write_text(
            "closure,boundary\n" + constraint_rows, encoding="utf-8",
        )

    def test_picks_best_feasible_only(self, tmp_path):
        self._write_run(tmp_path,
                        "-0.9,-1.0\n-0.6,-1.0\n",
                        "0.5,0.0\n-1.0,-1.0\n")  # -0.9 row is infeasible
        assert best_feasible_util(tmp_path) == pytest.approx(0.6)

    def test_nan_without_feasible_rows(self, tmp_path):
        self._write_run(tmp_path, "-0.9,-1.0\n", "0.5,0.0\n")
        assert np.isnan(best_feasible_util(tmp_path))

    def test_nan_without_files(self, tmp_path):
        assert np.isnan(best_feasible_util(tmp_path))


class TestSummarize:
    def test_median_and_iqr(self):
        line = summarize([0.5, 0.6, 0.7, float("nan")])
        assert "median 60.0%" in line
        assert "n=3" in line

    def test_flags_zero_variance(self):
        assert "WARNING" in summarize([0.6, 0.6, 0.6])

    def test_no_successful_runs(self):
        assert summarize([float("nan")]) == "no successful runs"