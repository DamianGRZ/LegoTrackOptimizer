"""BlueBrick/NCP export functionality for LEGO Track Optimizer.

This package provides tools to export optimized track layouts to NCP format
(4DBrix nControl), which can be imported by BlueBrick for visualization.

Main classes:
    BlueBrickExporter: Main exporter for converting layouts to NCP files.

Data classes:
    PartMapping: Mapping definition from our piece IDs to NCP part names.
    BlueBrickPose: Position and orientation in BlueBrick coordinate system.
    NCPSegment: Single track segment for NCP export.

Functions:
    get_ncp_mapping: Get NCP mapping for a piece ID.
    transform_to_bluebrick: Transform coordinates to BlueBrick system.
    generate_ncp_xml: Generate NCP XML content.

Example usage:
    >>> from src.export import BlueBrickExporter
    >>> from src.data import TrackCatalog
    >>> from src.geometry import build_layout
    >>>
    >>> catalog = TrackCatalog.load("data/track_pieces.yaml")
    >>> layout = build_layout(chromosome, catalog)
    >>>
    >>> exporter = BlueBrickExporter(catalog)
    >>> exporter.export_layout(layout, "output.ncp")
"""

from .bluebrick import BlueBrickExporter
from .coordinate_transform import (
    BlueBrickPose,
    transform_angle_to_bluebrick,
    transform_point_to_bluebrick,
    transform_to_bluebrick,
)
from .ncp_writer import NCPSegment, generate_ncp_xml, write_ncp_file
from .part_mapping import (
    PIECE_TO_NCP,
    PartMapping,
    get_all_mappings,
    get_ncp_mapping,
    has_ncp_mapping,
)

__all__ = [
    # Main exporter
    "BlueBrickExporter",
    # Part mapping
    "PIECE_TO_NCP",
    "PartMapping",
    "get_ncp_mapping",
    "has_ncp_mapping",
    "get_all_mappings",
    # Coordinate transform
    "BlueBrickPose",
    "transform_to_bluebrick",
    "transform_point_to_bluebrick",
    "transform_angle_to_bluebrick",
    # NCP writer
    "NCPSegment",
    "generate_ncp_xml",
    "write_ncp_file",
]
