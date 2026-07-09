"""Run-artifact plumbing: CSV headers, crash salvage, run_info termination."""

import logging
from types import SimpleNamespace

from src.algorithm.runner import _salvage_failed_run
from src.encoding import chromosome_csv_header, compute_dimensions
from src.run_info import _termination_lines


class TestChromosomeCsvHeader:
    def test_names_every_gene(self, default_config, catalog):
        dims = compute_dimensions(default_config, catalog)
        names = chromosome_csv_header(dims).split(",")
        assert len(names) == dims.n_var
        assert names[0] == "main_0"
        assert names[-2:] == ["start_x", "start_y"]
        assert len(set(names)) == len(names)

    def test_covers_descriptor_blocks(self, switches_config, catalog):
        dims = compute_dimensions(switches_config, catalog)
        names = chromosome_csv_header(dims).split(",")
        assert len(names) == dims.n_var
        assert dims.max_junctions > 0
        assert "junc0_active" in names


class TestSalvageFailedRun:
    def _crash(self, algorithm, monitor, output_dir):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            return _salvage_failed_run(algorithm, monitor, output_dir,
                                       logging.getLogger(__name__))

    def test_writes_error_log_and_partial_result(self, tmp_path):
        monitor = SimpleNamespace(data={"n_gen": [1, 2, 3]})
        algorithm = SimpleNamespace(pop=[object(), object()])
        res = self._crash(algorithm, monitor, tmp_path)
        assert res.crashed
        assert res.pop is algorithm.pop
        text = (tmp_path / "error.log").read_text(encoding="utf-8")
        assert "boom" in text
        assert "generation 3" in text

    def test_returns_none_without_population(self, tmp_path):
        monitor = SimpleNamespace(data={"n_gen": []})
        algorithm = SimpleNamespace(pop=None)
        assert self._crash(algorithm, monitor, tmp_path) is None
        assert (tmp_path / "error.log").exists()


class TestTerminationLines:
    def test_early_stop(self, switches_config):
        res = SimpleNamespace(algorithm=SimpleNamespace(n_gen=259))
        [line] = _termination_lines(res, switches_config)
        assert "259/500" in line
        assert "early-stop" in line

    def test_max_generations(self, default_config):
        planned = default_config.algorithm.n_gen
        res = SimpleNamespace(algorithm=SimpleNamespace(n_gen=planned))
        [line] = _termination_lines(res, default_config)
        assert "max-generations" in line

    def test_crashed(self, default_config):
        res = SimpleNamespace(algorithm=SimpleNamespace(n_gen=42), crashed=True)
        [line] = _termination_lines(res, default_config)
        assert "crashed" in line

    def test_missing_algorithm_reports_planned_only(self, default_config):
        [line] = _termination_lines(SimpleNamespace(), default_config)
        assert "planned" in line
