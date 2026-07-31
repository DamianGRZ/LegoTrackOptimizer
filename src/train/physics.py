"""Portable LEGO train physics module.

Self-contained: imports only stdlib, numpy, yaml. Lift this file into
another project unchanged. All quantities in SI units (m, m/s, deg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

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

    No default below is a measurement; each is tagged assumed or standard.
    Real values come from the file a config names in train_config_path, which
    for every shipped config is configs/trains/measured_consist.yaml.
    """

    # --- Friction (nominal = reference, design = the value used by formulas) ---
    mu_nominal: float = 0.30         # assumed; central estimate, no formula reads it
    mu_design: float = 0.25          # assumed; pessimistic value every cap formula reads
    # --- Environment ---
    g: float = 9.81                  # standard; gravitational acceleration (m/s^2)
    # --- Motor ---
    v_motor_max: float = 1.10        # assumed; drive-train top speed (m/s)
    # --- Bogie / wheel geometry ---
    gauge_b: float = 0.0375          # standard; LEGO L-gauge inner rail-to-rail (m)
    cog_height_h: float = 0.030      # assumed; CoG above rail head (m), tilt test pending
    flange_angle_deg: float = 50.0   # assumed; effective flange contact angle (deg)
    # --- Speed-profile dynamics ---
    max_accel: float = 3.92          # assumed; maximum acceleration (m/s^2)
    brake_decel: float = 2.45        # assumed; braking deceleration (m/s^2)
    # --- Rolling resistance ---
    mu_roll: float = 0.05            # assumed; rolling-friction coefficient (literature)

    # --- Consist mass and coupler geometry ---
    mass_loco: float = 0.370         # assumed; locomotive mass (kg)
    mass_trailing: float = 0.0       # assumed; trailing vehicle mass (kg), 0.0 = bare loco
    coupler_offset: float = 0.100    # assumed; vehicle length for coupler angle (m)

    @property
    def mass_total(self) -> float:
        """Total consist mass: locomotive + trailing vehicles."""
        return self.mass_loco + self.mass_trailing

    # ---------------- Derailment-mode scalar caps ----------------
    # Scalar wrappers over the vectorized derailment_caps(); the formulas live
    # in exactly one place. np.sqrt(inf) == inf preserves the inf-radius result.

    def _derailment_caps(self, r_m: float) -> tuple[float, float, float]:
        slide, tip, nadal = derailment_caps(self, np.array([r_m], dtype=np.float64))
        return float(slide[0]), float(tip[0]), float(nadal[0])

    def v_slide(self, r_m: float) -> float:
        """Lateral sliding cap: sqrt(mu_design * g * R). Inf radius -> inf."""
        return self._derailment_caps(r_m)[0]

    def v_tip(self, r_m: float) -> float:
        """Tip-over cap: sqrt(g * R * (b/2) / h). Inf radius -> inf."""
        return self._derailment_caps(r_m)[1]

    def v_nadal(self, r_m: float) -> float:
        """Nadal wheel-climb cap via L/V criterion at mu_design. Inf radius -> inf."""
        return self._derailment_caps(r_m)[2]

    def v_max(self, r_m: float) -> float:
        """Derailment-mode minimum: min(slide, tip, nadal)."""
        return min(self._derailment_caps(r_m))

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


def derailment_caps(
    config: TrainConfig, radii_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-radius (v_slide, v_tip, v_nadal) derailment-mode speed caps.

    Single source of truth for the three derailment formulas; v_eff_array and
    the evaluation stability domain both build on it. Uses config.mu_design
    (pessimistic friction). Inf radius -> inf cap (np.sqrt(inf) == inf).

    Formulas and the LEGO-scale parameters come from
    docs/Lateral stability model for LEGO train layout optimization.md.
    """
    r = np.asarray(radii_m, dtype=np.float64)

    # Poślizg boczny: koła zjeżdżają w bok, gdy siła odśrodkowa przekroczy
    # tarcie o szynę.
    #   v = sqrt(mu * g * R)
    v_slide = np.sqrt(config.mu_design * config.g * r)

    # Wywrotka: pociąg przewraca się na zewnątrz łuku, gdy siła odśrodkowa
    # działająca na wysokości środka ciężkości (h) przeważy nad ciężarem
    # opartym o zewnętrzną szynę, czyli o połowę rozstawu (b/2) od środka.
    #   v = sqrt(g * R * (b/2) / h)
    v_tip = np.sqrt(config.g * r * (config.gauge_b / 2.0) / config.cog_height_h)

    # Wspinanie obrzeża na główkę szyny — kryterium Nadala (1908). Obrzeże
    # zaczyna się wspinać, gdy stosunek siły bocznej do pionowej przekroczy
    # próg zależny od kąta obrzeża (d) i tarcia:
    #   L/V = (tan(d) - mu) / (1 + mu * tan(d)),  potem v = sqrt(L/V * g * R)
    # Próg <= 0 oznacza, że przy tym kącie i tarciu obrzeże nie wjedzie nigdy.
    tan_d = math.tan(math.radians(config.flange_angle_deg))
    lv_crit = (tan_d - config.mu_design) / (1.0 + config.mu_design * tan_d)
    v_nadal = np.sqrt(config.g * r * lv_crit) if lv_crit > 0 else np.full_like(r, np.inf)

    return v_slide, v_tip, v_nadal


def v_eff_array(config: TrainConfig, radii_m: np.ndarray) -> np.ndarray:
    """Vectorized effective speed cap over an array of radii (in metres).

    Effective cap = min(v_slide, v_tip, v_nadal, v_motor_max). Straights
    (inf radius) collapse to the motor cap.
    """
    v_slide, v_tip, v_nadal = derailment_caps(config, radii_m)
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
