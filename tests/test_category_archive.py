"""Category elite archive: out-key capture, archive semantics, injection.

Spec: docs/superpowers/specs/2026-06-11-category-elite-archive-design.md
"""

import numpy as np
import pytest
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from src.catalog import TrackCatalog
from src.config import OptimizationConfig
from src.encoding import compute_dimensions, create_chromosome_from_pieces, create_empty_chromosome
from src.problem import TrackOptimizationProblem


@pytest.fixture(scope="module")
def cat():
    return TrackCatalog.load("data/track_pieces_v2.yaml")


@pytest.fixture(scope="module")
def cfg():
    return OptimizationConfig.load("configs/all_pieces.yaml")


@pytest.fixture(scope="module")
def problem(cat, cfg):
    return TrackOptimizationProblem(cat, cfg)


@pytest.fixture(scope="module")
def dims(cfg, cat):
    return compute_dimensions(cfg, cat)


@pytest.fixture(scope="module")
def inv(cfg, cat):
    return {cat._id_to_index[k]: v for k, v in cfg.inventory.items()
            if k in cat._id_to_index}


def _seed_chromosomes(inv, dims):
    """One chromosome per category from the validated heuristic seeds:
    plain racetrack, oval+siding (switch), figure-8-cross, DC figure-8."""
    from src.sampling import (
        _gen_figure_eight_cross,
        _gen_figure_eight_dbl_crossover,
        _gen_oval_with_siding,
        _gen_racetrack,
    )

    out = {}
    for name, gen in (
        ("plain", _gen_racetrack),
        ("switch", _gen_oval_with_siding),
        ("cross", _gen_figure_eight_cross),
        ("dc", _gen_figure_eight_dbl_crossover),
    ):
        variants = gen(inv, dims)
        assert variants, f"expected {name} seed for all_pieces config"
        pieces, flips, junctions, cross_junctions, dcs = variants[0]
        out[name] = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips, junctions=junctions,
            cross_junctions=cross_junctions, double_crossovers=dcs,
        )
    return out


def _fab_pop(rows):
    """Fabricated evaluated Population: rows of dict(F0, feas, sw, cross, dc)."""
    n = len(rows)
    pop = Population.new("X", np.arange(n * 4, dtype=float).reshape(n, 4))
    pop.set("F", np.array([[r["F0"], -1.0] for r in rows]))
    pop.set("G", np.array([[-1.0 if r["feas"] else 1.0 + i * 0.1]
                           for i, r in enumerate(rows)]))
    for key, short in (("n_sw_pairs", "sw"), ("n_cross_comm", "cross"),
                       ("n_dc_comm", "dc")):
        pop.set(key, np.array([float(r.get(short, 0)) for r in rows]))
    return pop


def _algo(pop):
    from types import SimpleNamespace
    return SimpleNamespace(pop=pop)


class TestCategoryEliteArchive:

    def test_captures_best_feasible_and_infeasible_separately(self):
        from src.algorithm.runner import CategoryEliteArchive
        arch = CategoryEliteArchive(inject=False)
        pop = _fab_pop([
            {"F0": -0.30, "feas": True, "cross": 1},
            {"F0": -0.20, "feas": True, "cross": 1},   # worse feasible cross
            {"F0": -0.90, "feas": False, "cross": 1},  # infeasible: report-only
            {"F0": -0.60, "feas": True},               # plain — no category
        ])
        arch.notify(_algo(pop))
        assert arch.feasible["cross"]["util"] == pytest.approx(0.30)
        assert arch.infeasible["cross"]["util"] == pytest.approx(0.90)
        assert "switch" not in arch.feasible and "dc" not in arch.feasible

    def test_injects_archived_elite_when_category_extinct(self):
        from src.algorithm.runner import CategoryEliteArchive
        arch = CategoryEliteArchive()
        gen1 = _fab_pop([
            {"F0": -0.30, "feas": True, "cross": 1},
            {"F0": -0.60, "feas": True},
        ])
        arch.notify(_algo(gen1))
        # Next generation: category extinct; worst = the infeasible slot 2.
        gen2 = _fab_pop([
            {"F0": -0.62, "feas": True},
            {"F0": -0.61, "feas": True},
            {"F0": -0.10, "feas": False},
        ])
        arch.notify(_algo(gen2))
        cross_counts = np.asarray(gen2.get("n_cross_comm"), dtype=float)
        assert cross_counts.max() > 0, "elite must be re-injected"
        assert cross_counts[2] > 0, "worst (infeasible) slot must be replaced"

    def test_no_injection_when_better_member_present(self):
        from src.algorithm.runner import CategoryEliteArchive
        arch = CategoryEliteArchive()
        gen1 = _fab_pop([{"F0": -0.30, "feas": True, "cross": 1}])
        arch.notify(_algo(gen1))
        gen2 = _fab_pop([
            {"F0": -0.35, "feas": True, "cross": 1},  # better than archive
            {"F0": -0.60, "feas": True},
        ])
        x_before = gen2.get("X").copy()
        arch.notify(_algo(gen2))
        np.testing.assert_array_equal(gen2.get("X"), x_before)
        # And the archive itself upgraded to the better member.
        assert arch.feasible["cross"]["util"] == pytest.approx(0.35)

    def test_two_categories_inject_into_distinct_slots(self):
        from src.algorithm.runner import CategoryEliteArchive
        arch = CategoryEliteArchive()
        gen1 = _fab_pop([
            {"F0": -0.30, "feas": True, "cross": 1},
            {"F0": -0.25, "feas": True, "dc": 1},
        ])
        arch.notify(_algo(gen1))
        gen2 = _fab_pop([
            {"F0": -0.62, "feas": True},
            {"F0": -0.10, "feas": False},
            {"F0": -0.15, "feas": False},
        ])
        arch.notify(_algo(gen2))
        cross_counts = np.asarray(gen2.get("n_cross_comm"), dtype=float)
        dc_counts = np.asarray(gen2.get("n_dc_comm"), dtype=float)
        assert cross_counts.max() > 0 and dc_counts.max() > 0
        # Injected into two different slots.
        assert int(np.argmax(cross_counts)) != int(np.argmax(dc_counts))


class TestCategoryReport:

    def test_save_results_writes_category_artifacts(
        self, problem, cfg, cat, dims, inv, tmp_path,
    ):
        from types import SimpleNamespace
        from src.algorithm.runner import CategoryEliteArchive, save_results

        seeds = _seed_chromosomes(inv, dims)
        X = np.array([seeds["plain"], seeds["switch"], seeds["cross"], seeds["dc"]])
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)

        arch = CategoryEliteArchive(inject=False)
        arch.notify(SimpleNamespace(pop=pop))
        assert "cross" in arch.feasible, "seed sanity: cross elite captured"

        res = SimpleNamespace(pop=pop, category_elites=arch)
        save_results(res, tmp_path, cat, cfg)

        report = (tmp_path / "category_report.md").read_text(encoding="utf-8")
        for category in ("switch", "cross", "dc"):
            assert f"## {category}" in report
            assert (tmp_path / f"best_with_{category}.png").exists()
        assert "utilization" in report


class TestDecoderDropLog:
    """Skipped descriptors must leave a human-readable trace on the layout."""

    def _decode(self, x, cfg, cat, dims):
        from src.decoder import decode_chromosome
        return decode_chromosome(x, cat, cfg.inventory, dims=dims)

    def test_dc_descriptor_on_curve_slots_logs_drop(self, cfg, cat, dims, inv):
        from src.sampling import _gen_racetrack
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        # Racetrack starts with a 4-R40 corner: slots 0 and 1 are curves, so a
        # DC descriptor naming them must be dropped (valid both-cross routes).
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips,
            double_crossovers=[(1, 0, 2, 1, 3)],
        )
        layout = self._decode(x, cfg, cat, dims)
        assert layout.n_dbl_crossovers == 0
        assert any("DC[" in entry for entry in layout.drop_log), layout.drop_log

    def test_cross_descriptor_on_parallel_straights_logs_drop(self, cfg, cat, dims, inv):
        from src.sampling import _gen_racetrack
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        # Slots 4 and 6 are straights on the same run — parallel, never
        # perpendicular-coincident.
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips,
            cross_junctions=[(1, 4, 6)],
        )
        layout = self._decode(x, cfg, cat, dims)
        assert layout.n_cross_junctions == 0
        assert any("CROSS[" in entry for entry in layout.drop_log), layout.drop_log

    def test_failed_junction_logs_drop(self, cfg, cat, dims, inv):
        from src.sampling import _gen_racetrack
        pieces, flips, *_ = _gen_racetrack(inv, dims)[0]
        # A siding anchored inside the first corner cannot validate.
        x = create_chromosome_from_pieces(
            dims, pieces, main_loop_flips=flips,
            junctions=[(1, 0, 0, 1)],
        )
        layout = self._decode(x, cfg, cat, dims)
        assert layout.n_switch_pairs == 0
        assert any("junction[" in entry for entry in layout.drop_log), layout.drop_log

    def test_valid_seeds_leave_empty_drop_log(self, cfg, cat, dims, inv):
        for name, x in _seed_chromosomes(inv, dims).items():
            layout = self._decode(x, cfg, cat, dims)
            assert layout.drop_log == [], (name, layout.drop_log)


class TestCategoryOutKeys:
    """_evaluate must report committed element counts as custom out-keys."""

    def test_keys_present_and_correct_per_seed(self, problem, inv, dims):
        seeds = _seed_chromosomes(inv, dims)
        X = np.array([seeds["plain"], seeds["switch"], seeds["cross"], seeds["dc"]])
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)

        n_sw = np.asarray(pop.get("n_sw_pairs"), dtype=int)
        n_cross = np.asarray(pop.get("n_cross_comm"), dtype=int)
        n_dc = np.asarray(pop.get("n_dc_comm"), dtype=int)

        # plain racetrack: nothing special
        assert (n_sw[0], n_cross[0], n_dc[0]) == (0, 0, 0)
        # oval with one siding: exactly one committed switch pair
        assert n_sw[1] == 1 and n_cross[1] == 0 and n_dc[1] == 0
        # figure-8-cross: exactly one committed CROSS_90
        assert n_cross[2] == 1 and n_sw[2] == 0 and n_dc[2] == 0
        # DC figure-8: exactly one committed DOUBLE_CROSSOVER
        assert n_dc[3] == 1 and n_sw[3] == 0 and n_cross[3] == 0

    def test_empty_layout_reports_zeros(self, problem, dims):
        pop = Population.new("X", np.array([create_empty_chromosome(dims)]))
        Evaluator().eval(problem, pop)
        assert int(np.asarray(pop.get("n_sw_pairs"))[0]) == 0
        assert int(np.asarray(pop.get("n_cross_comm"))[0]) == 0
        assert int(np.asarray(pop.get("n_dc_comm"))[0]) == 0


class TestEmergentCrossCounting:
    """An emergent (Step-4 self-intersection repair) CROSS_90 must count as
    cross-bearing. Physical pieces = CROSS_90 slots minus one per
    CrossJunction record: a descriptor commit marks BOTH its slots, an
    emergent conversion marks one slot and carries no record."""

    def test_n_cross_pieces_counts_both_origins(self):
        from src.types import CrossJunction, MultiPathLayout
        emergent = MultiPathLayout(main_loop_pieces=[0, 3, 0, 0])
        assert emergent.n_cross_pieces == 1
        committed = MultiPathLayout(
            main_loop_pieces=[0, 3, 0, 3],
            cross_junctions=[CrossJunction(
                slot=0, positions=(1, 3), origin=(0.0, 0.0, 0.0),
            )],
        )
        assert committed.n_cross_pieces == 1
        plain = MultiPathLayout(main_loop_pieces=[0, 0, 2, 2])
        assert plain.n_cross_pieces == 0

    def test_descriptorless_figure_eight_reports_emergent_cross(
        self, problem, inv, dims,
    ):
        """The figure-8 WITHOUT its descriptor: Step-4 repair converts the
        perpendicular self-crossing into a CROSS_90, so the n_cross_comm
        out-key (and with it the category archive) must see the genome as
        cross-bearing."""
        from src.sampling import _gen_figure_eight_cross
        pieces, flips, *_ = _gen_figure_eight_cross(inv, dims)[-1]  # 34-pc base
        x = create_chromosome_from_pieces(dims, pieces, main_loop_flips=flips)
        pop = Population.new("X", np.array([x]))
        Evaluator().eval(problem, pop)
        assert int(np.asarray(pop.get("n_cross_comm"))[0]) == 1


class TestArchiveUtilIndConsistency:
    """After an earlier category injects mid-notify, later categories must
    see the CURRENT population: every archive entry's util must equal the
    archived individual's own -F[0] (no stale-snapshot bookkeeping)."""

    def test_injected_elite_not_recorded_with_stale_util(self):
        from src.algorithm.runner import CategoryEliteArchive
        arch = CategoryEliteArchive()
        # gen1: one feasible individual carrying BOTH cross and dc.
        gen1 = _fab_pop([
            {"F0": -0.30, "feas": True, "cross": 1, "dc": 1},
            {"F0": -0.60, "feas": True},
        ])
        arch.notify(_algo(gen1))
        # gen2: both categories extinct. The cross category injects its
        # elite into the worst (infeasible) slot BEFORE dc is processed —
        # dc bookkeeping must describe the injected individual, not the
        # replaced occupant.
        gen2 = _fab_pop([
            {"F0": -0.62, "feas": True},
            {"F0": -0.10, "feas": False},
        ])
        arch.notify(_algo(gen2))
        for store in (arch.feasible, arch.infeasible):
            for category, entry in store.items():
                assert entry["util"] == pytest.approx(
                    -float(entry["ind"].F[0])
                ), (category, entry["util"], float(entry["ind"].F[0]))
