"""Track catalog package — V1 (legacy) and V2 (port-centric) surfaces."""

# V1 (legacy) — kept for backward compatibility during migration
from .catalog import TrackCatalog
from .pieces import FKDeltas, Port, TrackPiece

# V2 — new port-centric domain model + loader
from .specs import (
    PortDef,
    TrackPieceSpec,
    CatalogMeta,
    TrackCatalogSpec,
    SchemaVersionError,
    check_schema_version,
    ATOMIC_ANGLE_RAD,
    LATTICE_TOLERANCE,
    SUPPORTED_SCHEMA_VERSION,
)
from .loader import load_catalog_spec, CatalogLoadError


__all__ = [
    # V1
    "TrackCatalog", "FKDeltas", "Port", "TrackPiece",
    # V2
    "PortDef", "TrackPieceSpec", "CatalogMeta", "TrackCatalogSpec",
    "SchemaVersionError", "check_schema_version",
    "load_catalog_spec", "CatalogLoadError",
    "ATOMIC_ANGLE_RAD", "LATTICE_TOLERANCE", "SUPPORTED_SCHEMA_VERSION",
]
