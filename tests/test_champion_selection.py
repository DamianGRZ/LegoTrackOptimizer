"""``config.champion_selection`` wiring: the rule is a real control, not a label."""

from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.population import Population

from src.algorithm.runner import FeasibleEliteCallback
from src.config import OptimizationConfig
from src.normalization import champion_ranking

# A front whose F0 extreme pays badly on F1: the first-objective rule takes
# row 0 anyway, the balanced rule walks past it to row 2.
F_DISAGREE = np.array([[-0.60, 15.0], [-0.05, 2.5], [-0.35, 8.0]])


def _pop(F):
    n = len(F)
    pop = Population.new("X", np.arange(n * 3, dtype=float).reshape(n, 3))
    pop.set("F", np.asarray(F, dtype=float))
    pop.set("G", np.full((n, 1), -1.0))
    return pop


class TestConfigKnob:
    def test_default_is_first_objective(self):
        cfg = OptimizationConfig.load("configs/all_pieces.yaml")
        assert cfg.champion_selection == "first_objective"

    def test_balanced_champion_configs_state_the_rule(self):
        expected_n_obj = {
            "all_pieces_balanced_champion": 2,
            "all_pieces_three_objectives_balanced_champion": 3,
            "all_pieces_three_objectives_constant_speed_balanced_champion": 3,
        }
        for name, n_obj in expected_n_obj.items():
            cfg = OptimizationConfig.load(f"configs/{name}.yaml")
            assert cfg.champion_selection == "balanced", name
            assert len(cfg.objectives) == n_obj, name

    def test_balanced_is_accepted_and_unknown_rejected(self):
        data = OptimizationConfig.load("configs/all_pieces.yaml").model_dump()
        cfg = OptimizationConfig.model_validate({**data, "champion_selection": "balanced"})
        assert cfg.champion_selection == "balanced"
        with pytest.raises(ValueError):
            OptimizationConfig.model_validate({**data, "champion_selection": "best"})


class TestRuleReachesTheEliteCallback:
    """A wiring test: first that the two rules are distinguishable on this
    front, then that the callback preserves whichever one it was built with."""

    def test_elites_differ_between_the_two_rules(self):
        elites = {}
        for name in ("first_objective", "balanced"):
            cb = FeasibleEliteCallback(champion_ranking(name))
            cb.notify(SimpleNamespace(pop=_pop(F_DISAGREE)))
            elites[name] = float(cb._elite.F[0])
        assert elites["first_objective"] == pytest.approx(-0.60)
        assert elites["balanced"] == pytest.approx(-0.35)
