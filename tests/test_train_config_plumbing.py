"""The train physics a run uses must be the one its config names."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.config import OptimizationConfig
from src.decoder import decode_chromosome
from src.encoding import PieceIndex, create_chromosome_from_pieces
from src.problem import SPEED_SAFETY_MARGIN, TrackOptimizationProblem, _expected_traversal_time

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))
MEASURED_CONSIST = "trains/measured_consist.yaml"

# Four 4-curve corners joined by straight runs — the shape `_gen_racetrack`
# (src/sampling.py:185) seeds. The straights are what makes it usable here: the measured
# consist and the code defaults share mu_design, g, gauge and CoG height, so an all-curve
# loop runs at the same lateral-slide cap under either and cannot tell them apart. Only
# on straights do v_motor_max and max_accel differ.
CORNER = [PieceIndex.R40_CURVE] * 4
RACETRACK = (CORNER + [PieceIndex.STRAIGHT_16] * 4) * 4


def _evaluate_racetrack(catalog, config):
    """Score the racetrack loop; return its problem, decoded layout and out-dict."""
    problem = TrackOptimizationProblem(catalog, config)
    x = create_chromosome_from_pieces(problem.dims, RACETRACK)
    out = {}
    problem._evaluate(x, out)
    layout = decode_chromosome(
        x, catalog, config.inventory, dims=problem.dims, config=problem.decoder_config,
    )
    return problem, layout, out


def _traversal_time(layout, catalog, problem, train_config):
    """F[1] recomputed from an explicitly supplied train config."""
    return _expected_traversal_time(
        layout, catalog, train_config,
        safety_margin=SPEED_SAFETY_MARGIN,
        closure_pos_tol=problem.closure_tolerance,
        closure_angle_tol=problem.angle_tolerance,
    )


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_every_shipped_config_names_the_measured_consist(path):
    assert OptimizationConfig.load(path).train_config_path == MEASURED_CONSIST


def test_shipped_config_set_is_not_empty():
    """Stops the parametrized check above from passing vacuously on a broken glob."""
    assert len(CONFIG_PATHS) >= 22


def test_a_config_must_name_its_train_physics():
    """No config can exist without saying which locomotive it runs."""
    with pytest.raises(ValidationError, match="train_config_path"):
        OptimizationConfig(inventory={"STRAIGHT_16": 8})


def test_train_path_resolves_against_the_config_file_not_cwd(default_config):
    resolved = (default_config._base_dir / default_config.train_config_path).resolve()
    assert resolved == (CONFIG_DIR / MEASURED_CONSIST).resolve()
    assert resolved.is_file()


def test_loaded_config_returns_the_measured_consist(default_config, measured_train_config):
    assert default_config.load_train_config() == measured_train_config


def test_measured_consist_is_not_the_assumed_baseline(measured_train_config, train_config):
    """Equality alone only proves the file parsed; these pin the fields that differ,
    so one field silently reverting to the assumed baseline stays visible."""
    assert measured_train_config != train_config
    assert measured_train_config.v_motor_max == pytest.approx(1.26)
    assert measured_train_config.max_accel == pytest.approx(0.68)
    assert measured_train_config.mass_trailing == pytest.approx(0.327)


def test_problem_holds_the_configs_train_physics(catalog, default_config, measured_train_config):
    problem = TrackOptimizationProblem(catalog, default_config)
    assert problem._train_config == measured_train_config


def test_f1_is_the_traversal_time_of_the_measured_consist(
    catalog, default_config, measured_train_config,
):
    problem, layout, out = _evaluate_racetrack(catalog, default_config)
    assert np.isfinite(out["F"][1])
    expected = _traversal_time(layout, catalog, problem, measured_train_config)
    assert out["F"][1] == pytest.approx(expected, abs=1e-9)


def test_f1_would_move_under_the_assumed_baseline(catalog, default_config, train_config):
    """The existing F[1] tests feed ``problem._train_config`` into their own oracle, so
    both sides shift together and a physics swap is invisible. Here the oracle is
    independent: the two presets must produce measurably different times."""
    problem, layout, out = _evaluate_racetrack(catalog, default_config)
    baseline_time = _traversal_time(layout, catalog, problem, train_config)
    assert abs(out["F"][1] - baseline_time) > 1e-3


def _constant_variant(speed: float = 0.8) -> OptimizationConfig:
    """default.yaml switched to the constant-speed F[1] model."""
    config = OptimizationConfig.load(CONFIG_DIR / "default.yaml")
    config.f1_speed_model = "constant"
    config.f1_constant_speed = speed
    return config


def test_f1_speed_model_defaults_to_physics(catalog, default_config):
    assert default_config.f1_speed_model == "physics"
    assert TrackOptimizationProblem(catalog, default_config).f1_constant_speed is None


def test_constant_speed_config_states_the_flat_model():
    config = OptimizationConfig.load(CONFIG_DIR / "all_pieces_constant_speed.yaml")
    assert config.f1_speed_model == "constant"
    assert config.f1_constant_speed == pytest.approx(0.8)


def test_three_objectives_constant_speed_config_states_both(catalog):
    """The 3-objective constant-speed experiment: both knobs must survive the
    trip into the problem, not just parse."""
    config = OptimizationConfig.load(
        CONFIG_DIR / "all_pieces_three_objectives_constant_speed.yaml")
    assert config.objectives == ["weighted_piece_score", "traversal_time", "route_length"]
    assert config.f1_speed_model == "constant"
    problem = TrackOptimizationProblem(catalog, config)
    assert problem.n_obj == 3
    assert problem.f1_constant_speed == pytest.approx(0.8)
    assert "constant 0.8 m/s" in problem.objective_labels[1]


def test_constant_f1_differs_from_physics(catalog, default_config):
    """The two speed models must be distinguishable on a layout with straights:
    physics runs them at the motor cap and curves at the slide cap, neither 0.8."""
    _, _, out_physics = _evaluate_racetrack(catalog, default_config)
    _, _, out_constant = _evaluate_racetrack(catalog, _constant_variant())
    assert abs(out_constant["F"][1] - out_physics["F"][1]) > 1e-3


def test_constant_f1_is_length_over_the_configured_speed(catalog):
    """Single-route closed racetrack: F[1] must equal an oracle built straight
    from catalog arc lengths, with no profiler anywhere near it."""
    _, _, out = _evaluate_racetrack(catalog, _constant_variant())
    indices = np.asarray(RACETRACK, dtype=np.int32)
    arc_studs = catalog.get_route_arc_lengths(indices, np.zeros_like(indices)).sum()
    expected = float(arc_studs) * catalog.stud_mm / 1000.0 / 0.8
    assert out["F"][1] == pytest.approx(expected, abs=1e-9)


def test_constant_speed_value_is_wired(catalog):
    """Halving the configured speed must exactly double F[1]."""
    _, _, out_full = _evaluate_racetrack(catalog, _constant_variant(0.8))
    _, _, out_half = _evaluate_racetrack(catalog, _constant_variant(0.4))
    assert out_half["F"][1] == pytest.approx(2.0 * out_full["F"][1])
