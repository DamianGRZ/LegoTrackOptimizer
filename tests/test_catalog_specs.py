"""Tests for Pydantic v2 catalog specs (V2 schema)."""

import math
import pytest
from pydantic import ValidationError

from src.catalog.specs import PortDef


class TestPortDef:
    def test_valid_construction(self):
        """PortDef accepts three floats: dx, dy, dtheta."""
        port = PortDef(dx=0.0, dy=0.0, dtheta=0.0)
        assert port.dx == 0.0
        assert port.dy == 0.0
        assert port.dtheta == 0.0

    def test_frozen(self):
        """PortDef is immutable."""
        port = PortDef(dx=1.0, dy=2.0, dtheta=math.pi)
        with pytest.raises(ValidationError):
            port.dx = 99.0

    def test_extra_field_rejected(self):
        """Unknown fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError) as exc:
            PortDef(dx=0.0, dy=0.0, dtheta=0.0, color="red")
        assert "extra_forbidden" in str(exc.value) or "Extra inputs" in str(exc.value)

    def test_missing_field_rejected(self):
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            PortDef(dx=0.0, dy=0.0)  # missing dtheta
