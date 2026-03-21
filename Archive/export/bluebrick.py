"""BlueBrick/NCP export functionality for LEGO Track Optimizer.

This module provides the main exporter class for converting optimized track layouts
to NCP (4DBrix nControl) format, which can be imported by BlueBrick for visualization
and further editing.

Usage:
    from src.export import BlueBrickExporter
    from src.data import TrackCatalog
    from src.geometry import build_layout

    catalog = TrackCatalog.load("data/track_pieces.yaml")
    layout = build_layout(chromosome, catalog)

    exporter = BlueBrickExporter(catalog)
    exporter.export_layout(layout, "output.ncp")
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from src.data import TrackCatalog
from src.geometry import Layout

from .coordinate_transform import transform_to_bluebrick
from .ncp_writer import NCPSegment, generate_ncp_xml
from .part_mapping import get_ncp_mapping

logger = logging.getLogger(__name__)


class BlueBrickExporter:
    """Export track layouts to BlueBrick-compatible NCP format.

    The exporter converts Layout objects (from the optimizer) to NCP XML files
    that can be imported into BlueBrick or 4DBrix nControl.

    Attributes:
        catalog: Track catalog for piece information lookup.
    """

    def __init__(self, catalog: TrackCatalog):
        """Initialize exporter.

        Args:
            catalog: Track catalog containing piece definitions.
        """
        self.catalog = catalog

    def export_layout(
        self,
        layout: Layout,
        output_path: Union[str, Path],
        layout_name: Optional[str] = None,
    ) -> Path:
        """Export layout to NCP file.

        Args:
            layout: Layout object to export.
            output_path: Output file path (.ncp extension added if missing).
            layout_name: Optional name for the layout (shown in BlueBrick).
                Defaults to "Layout_N_pieces" format.

        Returns:
            Path to the created NCP file.

        Raises:
            ValueError: If layout has no valid pieces.
        """
        segments = self._layout_to_segments(layout)

        if not segments:
            raise ValueError("Layout has no exportable pieces")

        # Generate layout name if not provided
        name = layout_name or f"Layout_{layout.n_pieces}_pieces"

        # Generate XML content
        xml_content = generate_ncp_xml(segments, name)

        # Ensure .ncp extension
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".ncp":
            output_path = output_path.with_suffix(".ncp")

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_content, encoding="utf-8")

        logger.info(f"Exported {len(segments)} pieces to {output_path}")
        return output_path

    def _layout_to_segments(self, layout: Layout) -> List[NCPSegment]:
        """Convert layout to NCP segments.

        Transforms each piece in the layout to an NCPSegment with proper
        BlueBrick coordinates and part names.

        Args:
            layout: Layout to convert.

        Returns:
            List of NCPSegment objects ready for XML export.
        """
        segments = []
        skipped = 0

        for i, piece_idx in enumerate(layout.indices):
            # Get piece definition from catalog
            piece = self.catalog[int(piece_idx)]
            if piece is None:
                logger.warning(f"Unknown piece index: {piece_idx}")
                skipped += 1
                continue

            # Get NCP mapping
            mapping = get_ncp_mapping(piece.id)
            if mapping is None:
                logger.warning(f"No NCP mapping for piece: {piece.id}")
                skipped += 1
                continue

            # Get FK delta for this piece
            fk = (piece.fk.dx, piece.fk.dy, piece.fk.dtheta)

            # Get entry state (position before this piece)
            entry_state = (
                float(layout.states[i, 0]),
                float(layout.states[i, 1]),
                float(layout.states[i, 2]),
            )

            # Transform to BlueBrick coordinates
            pose = transform_to_bluebrick(entry_state, fk, mapping)

            # Create segment
            segment = NCPSegment(
                part_name=mapping.ncp_part_name,
                x=pose.x,
                y=pose.y,
                angle=pose.angle,
            )
            segments.append(segment)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} pieces due to missing mappings")

        return segments

    def export_to_string(self, layout: Layout, layout_name: Optional[str] = None) -> str:
        """Export layout to NCP XML string (without writing to file).

        Useful for testing or when the XML content is needed directly.

        Args:
            layout: Layout object to export.
            layout_name: Optional name for the layout.

        Returns:
            NCP XML content as string.
        """
        segments = self._layout_to_segments(layout)
        name = layout_name or f"Layout_{layout.n_pieces}_pieces"
        return generate_ncp_xml(segments, name)

    def get_segment_count(self, layout: Layout) -> int:
        """Get the number of exportable segments in a layout.

        Args:
            layout: Layout to analyze.

        Returns:
            Number of pieces that have valid NCP mappings.
        """
        return len(self._layout_to_segments(layout))

    def validate_layout(self, layout: Layout) -> tuple[bool, List[str]]:
        """Validate that all pieces in layout can be exported.

        Args:
            layout: Layout to validate.

        Returns:
            Tuple of (is_valid, list of warning messages).
        """
        warnings = []

        for i, piece_idx in enumerate(layout.indices):
            piece = self.catalog[int(piece_idx)]
            if piece is None:
                warnings.append(f"Piece {i}: Unknown index {piece_idx}")
                continue

            mapping = get_ncp_mapping(piece.id)
            if mapping is None:
                warnings.append(f"Piece {i}: No NCP mapping for {piece.id}")

        is_valid = len(warnings) == 0
        return is_valid, warnings
