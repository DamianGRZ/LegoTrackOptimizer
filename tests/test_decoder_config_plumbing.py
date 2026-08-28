"""Post-run decodes must use the same DecoderConfig as evaluation."""

import numpy as np

from src.decoder import DecoderConfig, decode_chromosome
from src.problem import TrackOptimizationProblem
from src.sampling import IntegerSampling


def test_from_optimization_config_matches_evaluation(compact_config, catalog):
    problem = TrackOptimizationProblem(catalog, compact_config)
    assert DecoderConfig.from_optimization_config(compact_config) == problem.decoder_config


def test_joint_tolerances_come_from_the_config(compact_config):
    # A siding admitted with a gap scores feasible and is rewarded for track it
    # does not have, so the decoder must gate on the config's own tolerance
    # rather than a looser built-in default.
    cfg = DecoderConfig.from_optimization_config(compact_config)
    assert cfg.siding_position_tolerance == compact_config.closure_tolerance
    assert cfg.siding_angle_tolerance == compact_config.angle_tolerance


def test_default_decoder_config_is_not_the_configured_one(compact_config):
    # compact's boundary is asymmetric ([-100, 60]); the dataclass default
    # (±100) would auto-center 20 studs away from where the boundary
    # constraint judged the layout.
    default_cfg = DecoderConfig()
    eval_cfg = DecoderConfig.from_optimization_config(compact_config)
    assert default_cfg != eval_cfg
    assert eval_cfg.boundary_max_x == compact_config.boundary.max_x


def test_postrun_decode_reproduces_evaluation_geometry(compact_config, catalog):
    problem = TrackOptimizationProblem(catalog, compact_config)
    sampler = IntegerSampling(catalog, compact_config, heuristic_ratio=1.0, seed=42)
    X = sampler._do(problem, 8)
    helper_cfg = DecoderConfig.from_optimization_config(compact_config)

    for x in X:
        x = np.asarray(x, dtype=np.int16)
        evaluated = decode_chromosome(x, catalog, compact_config.inventory,
                                      dims=problem.dims, config=problem.decoder_config)
        rendered = decode_chromosome(x, catalog, compact_config.inventory,
                                     dims=problem.dims, config=helper_cfg)
        assert np.allclose(evaluated.states, rendered.states)
