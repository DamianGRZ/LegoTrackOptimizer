"""Per-generation snapshots: cadence, per-category foldering, file naming."""

from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from src.algorithm.runner import (
    CATEGORIES,
    SnapshotCallback,
    _compute_snapshot_targets,
)
from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import compute_dimensions, create_chromosome_from_pieces
from src.problem import TrackOptimizationProblem


@pytest.fixture(scope="module")
def cat():
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture(scope="module")
def cfg():
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture(scope="module")
def dims(cfg, cat):
    return compute_dimensions(cfg, cat)


@pytest.fixture(scope="module")
def population(cfg, cat, dims):
    """One evaluated individual per category, from the validated heuristic seeds."""
    from src.sampling import (
        _gen_figure_eight_cross,
        _gen_figure_eight_dbl_crossover,
        _gen_oval_with_siding,
        _gen_racetrack,
    )

    inv = {cat._id_to_index[k]: v for k, v in cfg.inventory.items()
           if k in cat._id_to_index}
    rows = []
    for generate in (_gen_racetrack, _gen_oval_with_siding,
                     _gen_figure_eight_cross, _gen_figure_eight_dbl_crossover):
        variants = generate(inv, dims)
        assert variants, f"expected a seed from {generate.__name__}"
        pieces, flips, junctions, crossings, dcs = variants[0]
        rows.append(create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, junctions=junctions,
            cross_junctions=crossings, double_crossovers=dcs,
        ))
    pop = Population.new("X", np.array(rows))
    Evaluator().eval(TrackOptimizationProblem(cat, cfg), pop)
    return pop


class TestSnapshotCadence:
    def test_every_generation_is_a_target(self):
        assert _compute_snapshot_targets(5) == [1, 2, 3, 4, 5]

    def test_degenerate_budget_still_yields_one_target(self):
        assert _compute_snapshot_targets(0) == [1]


class TestSnapshotFiles:
    def _run_one(self, tmp_path, cat, cfg, dims, population, n_gen, gen):
        callback = SnapshotCallback(
            _compute_snapshot_targets(n_gen), tmp_path, cat, cfg, dims,
        )
        callback.notify(SimpleNamespace(n_gen=gen, pop=population))
        return tmp_path / "snapshots"

    def test_each_category_gets_its_own_folder(
        self, tmp_path, cat, cfg, dims, population,
    ):
        snap_dir = self._run_one(tmp_path, cat, cfg, dims, population, 5, 1)
        for category in CATEGORIES:
            produced = list((snap_dir / category).glob("gen*.png"))
            assert produced, f"{category} rendered no snapshot"
            # Status rides in the filename, not in a separate folder.
            assert all(p.stem.endswith(("_feasible", "_infeasible"))
                       for p in produced), produced

    def test_generation_is_zero_padded_to_the_run_width(
        self, tmp_path, cat, cfg, dims, population,
    ):
        """Names must sort in run order: gen009 before gen010 before gen100."""
        snap_dir = self._run_one(tmp_path, cat, cfg, dims, population, 300, 9)
        rendered = list(snap_dir.rglob("gen*.png"))
        assert rendered
        assert all(p.name.startswith("gen009_") for p in rendered), rendered

    def test_second_visit_to_a_generation_is_ignored(
        self, tmp_path, cat, cfg, dims, population,
    ):
        callback = SnapshotCallback(
            _compute_snapshot_targets(5), tmp_path, cat, cfg, dims,
        )
        callback.notify(SimpleNamespace(n_gen=1, pop=population))
        callback.notify(SimpleNamespace(n_gen=1, pop=population))
        assert len(callback.snapshots) == 1


class TestPlainCategory:
    def test_plain_excludes_every_special_element(self, population):
        from src.algorithm.runner import CATEGORY_KEYS, category_masks

        masks = category_masks(population)
        assert masks is not None
        special = np.logical_or.reduce([masks[c] for c in CATEGORY_KEYS])
        # "plain" is exactly the complement — no layout is in both.
        assert not np.any(masks["plain"] & special)
        assert np.all(masks["plain"] | special)

    def test_missing_census_keys_report_none(self):
        from src.algorithm.runner import category_masks

        bare = Population.new("X", np.zeros((2, 4)))
        assert category_masks(bare) is None
