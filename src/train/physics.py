"""Portable LEGO train physics module.

Self-contained: imports only stdlib, numpy, yaml. Lift this file into
another project unchanged. All quantities in SI units (m, m/s, deg).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


@dataclass(frozen=True)
class TrainConfig:
    """Immutable lateral-stability physical parameters for a LEGO train.

    The friction coefficient is expressed as a (nominal, design) pair:
    mu_nominal is the central estimate for diagnostics; mu_design is the
    pessimistic value that every speed cap formula actually reads. This
    keeps the friction-uncertainty band explicit in the data model while
    making the optimiser design for the worst plausible friction.
    """

    # --- Friction (nominal = reference, design = the value used by formulas) ---
    mu_nominal: float = 0.30         # central wheel-rail friction estimate (diagnostics only)
    mu_design: float = 0.25          # pessimistic friction used by v_slide / v_nadal / v_eff_array
    # --- Environment ---
    g: float = 9.81                  # gravitational acceleration (m/s^2)
    # --- Motor ---
    v_motor_max: float = 1.10        # Powered Up drive-train top speed (m/s)
    # --- Bogie / wheel geometry ---
    gauge_b: float = 0.0375          # inner rail-to-rail gauge (m)
    cog_height_h: float = 0.030      # CoG above rail head (m)
    flange_angle_deg: float = 50.0   # effective flange contact angle (deg)
    # --- Speed-profile dynamics ---
    max_accel: float = 3.92          # maximum acceleration (m/s^2)
    brake_decel: float = 2.45        # braking deceleration (m/s^2)
    # --- Rolling resistance ---
    mu_roll: float = 0.05            # rolling-friction coefficient (literature default; tunable)

    # --- Consist mass and coupler geometry ---
    mass_loco: float = 0.370         # locomotive mass (kg)
    mass_trailing: float = 0.0       # total trailing vehicle mass (kg), 0.0 = bare loco
    coupler_offset: float = 0.100    # vehicle length for coupler angle calc (m)

    @property
    def mass_total(self) -> float:
        """Total consist mass: locomotive + trailing vehicles."""
        return self.mass_loco + self.mass_trailing

    # ---------------- Derailment-mode scalar formulas ----------------

    def v_slide(self, r_m: float) -> float:
        """Lateral sliding cap: sqrt(mu_design * g * R). Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        return math.sqrt(self.mu_design * self.g * r_m)

    def v_tip(self, r_m: float) -> float:
        """Tip-over cap: sqrt(g * R * (b/2) / h). Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        return math.sqrt(self.g * r_m * (self.gauge_b / 2.0) / self.cog_height_h)

    def v_nadal(self, r_m: float) -> float:
        """Nadal wheel-climb cap via L/V criterion at mu_design. Inf radius -> inf."""
        if math.isinf(r_m):
            return math.inf
        tan_d = math.tan(math.radians(self.flange_angle_deg))
        lv_crit = (tan_d - self.mu_design) / (1.0 + self.mu_design * tan_d)
        if lv_crit <= 0:
            return math.inf
        return math.sqrt(self.g * r_m * lv_crit)

    def v_max(self, r_m: float) -> float:
        """Derailment-mode minimum: min(slide, tip, nadal)."""
        return min(self.v_slide(r_m), self.v_tip(r_m), self.v_nadal(r_m))

    def v_eff(self, r_m: float) -> float:
        """Effective speed cap: min(v_max, v_motor_max)."""
        return min(self.v_max(r_m), self.v_motor_max)

    # ---------------- YAML I/O ----------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load from YAML; missing fields keep defaults; empty file allowed."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        """Dump to a plain dict for export or comparison."""
        return asdict(self)


def v_eff_array(config: TrainConfig, radii_m: np.ndarray) -> np.ndarray:
    """Vectorized effective speed cap over an array of radii (in metres).

    Uses config.mu_design (pessimistic friction). Handles inf radii: np.sqrt(inf)
    = inf, and np.minimum with v_motor_max collapses straights to the motor cap.
    """
    r = np.asarray(radii_m, dtype=np.float64)
    v_slide = np.sqrt(config.mu_design * config.g * r)
    v_tip = np.sqrt(config.g * r * (config.gauge_b / 2.0) / config.cog_height_h)
    tan_d = math.tan(math.radians(config.flange_angle_deg))
    lv_crit = (tan_d - config.mu_design) / (1.0 + config.mu_design * tan_d)
    v_nadal = np.sqrt(config.g * r * lv_crit) if lv_crit > 0 else np.full_like(r, np.inf)
    v_max = np.minimum.reduce([v_slide, v_tip, v_nadal])
    return np.minimum(v_max, config.v_motor_max)


def available_accel(
    config: TrainConfig,
    v: float,
    radius_m: float,
    is_braking: bool = False,
) -> float:
    """Available longitudinal accel/brake at speed v on a curve of radius_m.

    Implements the friction ellipse (Kapania & Gerdes 2015, TUM/FTM pattern):
    lateral and longitudinal friction limits may differ, giving an elliptical
    combined-grip boundary rather than a circular one.

    Lateral limit: mu_design * g (all-wheel sliding friction).
    Longitudinal limit: max_accel (driven-wheel traction) or brake_decel.

    During acceleration, a one-iteration coupler correction adds the lateral
    destabilising force from trailing vehicles to the lateral demand.
    During braking, coupler compression is stabilising; omitting it is
    conservative, so no correction is applied.

    Args:
        config: Train physics configuration.
        v: Current speed (m/s).
        radius_m: Curve radius (m). math.inf for straights.
        is_braking: If True, use brake_decel as longitudinal cap.

    Returns:
        Available longitudinal acceleration/deceleration (m/s^2), >= 0.
    """
    a_lat_max = config.mu_design * config.g
    cap = config.brake_decel if is_braking else config.max_accel

    # No lateral demand on straights
    if math.isinf(radius_m) or radius_m <= 0:
        return cap

    a_lat = v * v / radius_m

    # Friction ellipse: a_long = cap * sqrt(1 - (a_lat / a_lat_max)^2)
    ratio = a_lat / a_lat_max
    if ratio >= 1.0:
        return 0.0
    a_long = cap * math.sqrt(1.0 - ratio * ratio)

    # Coupler correction: acceleration only, one iteration
    if not is_braking and config.mass_trailing > 0:
        phi = config.coupler_offset / (2.0 * radius_m)
        a_coupler_lat = (config.mass_trailing / config.mass_loco) * a_long * math.sin(phi)
        a_lat_total = a_lat + a_coupler_lat
        ratio_total = a_lat_total / a_lat_max
        if ratio_total >= 1.0:
            return 0.0
        a_long = cap * math.sqrt(1.0 - ratio_total * ratio_total)

    return a_long


DEFAULT_TRAIN_CONFIG = TrainConfig()
