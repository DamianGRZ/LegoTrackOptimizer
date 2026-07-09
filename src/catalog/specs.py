"""Pydantic domain models for the port-centric track catalog schema."""

from __future__ import annotations

import logging
import math
from typing import Literal, Mapping

from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_FROZEN = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PortDef(BaseModel):
    """SE(2) pose of a port relative to the piece-local origin (port A)."""

    model_config = _FROZEN

    dx: float = Field(description="forward offset in studs")
    dy: float = Field(description="left offset in studs (y = left)")
    dtheta: float = Field(description="heading delta in radians, CCW positive")


ATOMIC_ANGLE_RAD = math.pi / 32  # 5.625° = π/32, the track geometry's atomic angle
LATTICE_TOLERANCE = 1e-6         # tolerance for the π/32 angle-lattice check


class TrackPieceSpec(BaseModel):
    """A single track piece: kind, manufacturer, ports, and named routes."""

    model_config = _FROZEN

    piece_id: str = Field(min_length=1)
    kind: Literal["straight", "curve", "switch", "wye", "crossing"]
    manufacturer: Literal["lego", "4dbrix", "fxbricks", "bricktracks", "trixbrix"]
    part_numbers: tuple[str, ...] = ()

    # kind-conditional geometry
    length_studs: float | None = None
    radius_studs: float | None = None
    sector_angle_rad: float | None = None
    hand: Literal["left", "right"] | None = None
    body_length_studs: float | None = None
    diverging_radius_studs: float | None = None

    # topology
    ports: Mapping[str, PortDef]
    routes: Mapping[str, tuple[str, ...]]

    @field_validator("ports")
    @classmethod
    def _port_A_is_origin(cls, v: Mapping[str, PortDef]) -> Mapping[str, PortDef]:
        a = v.get("A")
        if a is None or (a.dx, a.dy, a.dtheta) != (0.0, 0.0, 0.0):
            raise ValueError(
                "Port 'A' must exist and be at (0, 0, 0); "
                "the piece-local frame is defined by port A."
            )
        return v

    @model_validator(mode="after")
    def _routes_reference_real_ports(self):
        for route_name, port_seq in self.routes.items():
            missing = [p for p in port_seq if p not in self.ports]
            if missing:
                raise ValueError(
                    f"Route '{route_name}' references undefined port(s) {missing}; "
                    f"known ports are {sorted(self.ports)}."
                )
        return self

    @model_validator(mode="after")
    def _kind_geometry_complete(self):
        required: dict[str, list[str]] = {
            "straight": ["length_studs"],
            "curve": ["radius_studs", "sector_angle_rad"],
            "switch": ["body_length_studs", "diverging_radius_studs"],
            "wye": [],
            "crossing": [],
        }
        missing = [f for f in required.get(self.kind, [])
                   if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"kind='{self.kind}' requires field(s) {missing}; "
                f"piece_id='{self.piece_id}' is missing them."
            )
        return self

    @property
    def on_angle_lattice(self) -> bool:
        """True iff every port.dtheta is an integer multiple of π/32 within tolerance."""
        for port in self.ports.values():
            ratio = port.dtheta / ATOMIC_ANGLE_RAD
            nearest = round(ratio)
            if abs(nearest * ATOMIC_ANGLE_RAD - port.dtheta) > LATTICE_TOLERANCE:
                return False
        return True


class CatalogMeta(BaseModel):
    model_config = _FROZEN

    schema_version: str = Field(min_length=1, description="MAJOR.MINOR.PATCH")
    unit: Literal["stud"] = "stud"
    stud_mm: float = 8.0
    angle_unit: Literal["rad"] = "rad"
    atomic_angle_rad: float = ATOMIC_ANGLE_RAD


class TrackCatalogSpec(BaseModel):
    """Top-level catalog: meta block + tuple of pieces."""

    model_config = _FROZEN

    meta: CatalogMeta
    pieces: tuple[TrackPieceSpec, ...]

    @model_validator(mode="after")
    def _piece_ids_unique(self):
        ids = [p.piece_id for p in self.pieces]
        dups = sorted({x for x in ids if ids.count(x) > 1})
        if dups:
            raise ValueError(f"Duplicate piece_id(s): {dups}. "
                             f"Each TrackPieceSpec.piece_id must be unique.")
        return self

    @property
    def by_id(self) -> Mapping[str, TrackPieceSpec]:
        return {p.piece_id: p for p in self.pieces}

    def by_kind(self, kind: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.kind == kind)

    def by_manufacturer(self, m: str) -> tuple[TrackPieceSpec, ...]:
        return tuple(p for p in self.pieces if p.manufacturer == m)

    @property
    def n_types(self) -> int:
        return len(self.pieces)

    @property
    def piece_ids(self) -> tuple[str, ...]:
        """Canonical stable ordering; chromosome uses this index mapping."""
        return tuple(p.piece_id for p in self.pieces)


log = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSION = Version("1.0.0")


class SchemaVersionError(RuntimeError):
    """Raised when catalog schema MAJOR version is incompatible with code."""


def check_schema_version(file_version_str: str, path: str = "<catalog>") -> None:
    """Reject MAJOR mismatch, warn when file MINOR is newer, otherwise accept silently."""
    file_ver = Version(file_version_str)
    code_ver = SUPPORTED_SCHEMA_VERSION

    if file_ver.major != code_ver.major:
        raise SchemaVersionError(
            f"{path}: schema MAJOR version mismatch — "
            f"file is {file_ver}, code supports {code_ver}. "
            f"Regenerate the catalog for schema v{code_ver.major} "
            f"(see data/track_pieces_v2.yaml)."
        )
    if file_ver.minor > code_ver.minor:
        log.warning(
            "%s: schema MINOR version %s is newer than supported %s — "
            "unknown additive fields will be rejected by extra='forbid'.",
            path, file_ver, code_ver,
        )
