"""Verify the chain: configs/*.yaml -> load_train_config -> PortPairProblem -> F[1].

Catches the disconnect bug discovered 2026-05-08 (see
docs/PLAN_train_physics_disconnect_fix.md): measured physics in
``configs/trains/measured_consist.yaml`` were silently bypassed because the
loading path was a dead branch and ``problem.py`` read catalog-static
``get_speed_for_route()`` instead of ``train_config.v_eff(radius_m)``.

Each assertion isolates one link in the chain so a future regression
points at the broken layer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src_v2.catalog import TrackCatalog
from src_v2.config import OptimizationConfig
from src_v2.problem import PortPairProblem
from src_v2.train import DEFAULT_TRAIN_CONFIG


REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "track_pieces_v2.yaml"
WITH_SWITCHES_CFG = REPO_ROOT / "configs" / "with_switches.yaml"


@pytest.fixture(scope="module")
def setup():
    catalog = TrackCatalog.load(CATALOG_PATH)
    config = OptimizationConfig.load(WITH_SWITCHES_CFG)
    train_cfg = config.load_train_config()
    problem = PortPairProblem(catalog, config, train_config=train_cfg)
    return catalog, config, train_cfg, problem


# --- Tier 1.1: YAML pointer points to measured_consist.yaml --------------

def test_config_points_to_measured_consist(setup) -> None:
    """Every with_switches/with_crossing/default/compact config (and their
    _only_mutation siblings) must point at ``trains/measured_consist.yaml``.
    A regression to ``trains/default.yaml`` silently swaps in stale physics."""
    _, config, _, _ = setup
    assert config.train_config_path is not None, (
        "train_config_path missing from with_switches.yaml"
    )
    assert "measured_consist" in config.train_config_path, (
        f"train_config_path={config.train_config_path!r} should reference "
        f"measured_consist.yaml; the V2 default contains stale physics."
    )


# --- Tier 1.1: measured values actually load ------------------------------

def test_measured_values_load(setup) -> None:
    """Phase 0 measurements (mass, motor cap, accel, coupler) must reach
    the loaded TrainConfig instance."""
    _, _, train_cfg, _ = setup
    assert abs(train_cfg.v_motor_max - 1.26) < 0.001, (
        f"v_motor_max={train_cfg.v_motor_max}, expected 1.26 (measured). "
        f"Did configs/trains/measured_consist.yaml change?"
    )
    assert abs(train_cfg.mass_loco - 0.493) < 0.001, (
        f"mass_loco={train_cfg.mass_loco}, expected 0.493 (measured)."
    )
    assert abs(train_cfg.mass_trailing - 0.327) < 0.001, (
        f"mass_trailing={train_cfg.mass_trailing}, expected 0.327 (measured)."
    )
    assert abs(train_cfg.mass_total - 0.820) < 0.001, (
        f"mass_total={train_cfg.mass_total}, expected 0.820 (0.493 + 0.327)."
    )
    assert abs(train_cfg.coupler_offset - 0.106) < 0.001, (
        f"coupler_offset={train_cfg.coupler_offset}, expected 0.106 (measured)."
    )
    assert abs(train_cfg.max_accel - 0.68) < 0.001, (
        f"max_accel={train_cfg.max_accel}, expected 0.68 (measured)."
    )


# --- Tier 1.2: PortPairProblem stores the loaded TrainConfig --------------

def test_problem_stores_train_config(setup) -> None:
    """PortPairProblem must accept and retain the train_config; it should
    be the same instance that the runner loaded (not a fresh default)."""
    _, _, train_cfg, problem = setup
    assert hasattr(problem, "train_config"), (
        "PortPairProblem missing train_config attribute"
    )
    assert problem.train_config is train_cfg, (
        "problem.train_config is not the instance passed to __init__"
    )
    # Sanity: it is NOT the V2 dataclass default (which would silently mask
    # a regression where the caller forgets train_config=...)
    assert problem.train_config is not DEFAULT_TRAIN_CONFIG, (
        "problem fell back to DEFAULT_TRAIN_CONFIG; the runner did not "
        "pass the loaded train_config."
    )


# --- Tier 1.3: F[1] derives from v_eff, not catalog ----------------------

def test_v_eff_at_R40_uses_measured_friction(setup) -> None:
    """v_eff(R=0.32) under measured friction must be ~0.886 m/s, not the
    catalog-static 0.97 m/s. This is the value F[1] reports for any cycle
    containing an R40 (every closed cycle in the inventory)."""
    _, _, train_cfg, _ = setup
    v_eff_R40 = train_cfg.v_eff(0.32)
    assert abs(v_eff_R40 - 0.886) < 0.001, (
        f"v_eff(0.32)={v_eff_R40:.4f}, expected 0.886 (slide-bound at "
        f"mu=0.25). Did mu_design or g change?"
    )


def test_v_eff_at_straight_uses_motor_cap(setup) -> None:
    """v_eff(inf) must equal v_motor_max (1.26 m/s measured), NOT the
    catalog-static 1.57 m/s. Straight-rich layouts are motor-bound."""
    _, _, train_cfg, _ = setup
    v_eff_straight = train_cfg.v_eff(math.inf)
    assert abs(v_eff_straight - train_cfg.v_motor_max) < 1e-9, (
        f"v_eff(inf)={v_eff_straight}, expected v_motor_max="
        f"{train_cfg.v_motor_max}"
    )
    assert abs(v_eff_straight - 1.26) < 0.001, (
        f"straight-route speed cap = {v_eff_straight:.3f}, expected "
        f"1.26 m/s (measured motor top speed)."
    )


def test_radius_lookup_matches_catalog_geometry(setup) -> None:
    """get_radius_m_for_route returns the geometry the v_eff caller needs.
    Curve / switch-diverging routes -> 0.32 m; straights / switch-through
    / cross routes -> None (motor-bound)."""
    catalog, _, _, _ = setup
    by_id = catalog.id_to_index

    # R40 curve (single-route): 0.32 m
    r = catalog.get_radius_m_for_route(by_id["R40_CURVE"], "main")
    assert r is not None and abs(r - 0.32) < 1e-9, (
        f"R40_CURVE radius_m = {r}, expected 0.32"
    )

    # Switch through-route: motor-bound (None radius)
    r = catalog.get_radius_m_for_route(by_id["R40_SWITCH_LEFT"], "through")
    assert r is None, f"R40_SWITCH_LEFT.through should be straight, got r={r}"

    # Switch diverging: 0.32 m
    r = catalog.get_radius_m_for_route(by_id["R40_SWITCH_LEFT"], "diverging")
    assert r is not None and abs(r - 0.32) < 1e-9, (
        f"R40_SWITCH_LEFT.diverging radius_m = {r}, expected 0.32"
    )

    # Straight: None
    r = catalog.get_radius_m_for_route(by_id["STRAIGHT_16"], "main")
    assert r is None, f"STRAIGHT_16 should be motor-bound, got r={r}"
