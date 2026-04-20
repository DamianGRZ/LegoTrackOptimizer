"""Pydantic v2 domain models for the track catalog (V2 schema)."""

from __future__ import annotations

import math
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_FROZEN = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PortDef(BaseModel):
    """SE(2) pose of a port relative to the piece-local origin (port A)."""

    model_config = _FROZEN

    dx: float = Field(description="forward offset in studs")
    dy: float = Field(description="left offset in studs (y = left)")
    dtheta: float = Field(description="heading delta in radians, CCW positive")
