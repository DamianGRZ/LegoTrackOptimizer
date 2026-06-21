"""Track catalog: validate the piece-catalog YAML and expose it to the optimizer.

The catalog lives in a port-centric YAML (data/track_pieces_v2.yaml). The specs +
loader modules validate it with Pydantic; TrackCatalog then builds the vectorized
FK / speed / topology tables the optimizer consumes at runtime.
"""

# Runtime catalog + piece dataclasses consumed by the optimizer.
from .catalog import TrackCatalog
from .pieces import FKDeltas, Port, TrackPiece

# Port-centric YAML schema: Pydantic models + loader/validator.
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
    # Runtime catalog
    "TrackCatalog", "FKDeltas", "Port", "TrackPiece",
    # Schema + loader
    "PortDef", "TrackPieceSpec", "CatalogMeta", "TrackCatalogSpec",
    "SchemaVersionError", "check_schema_version",
    "load_catalog_spec", "CatalogLoadError",
    "ATOMIC_ANGLE_RAD", "LATTICE_TOLERANCE", "SUPPORTED_SCHEMA_VERSION",
]
