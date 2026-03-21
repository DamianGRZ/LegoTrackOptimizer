"""Part mapping from LEGO Track Optimizer IDs to NCP (4DBrix nControl) part names.

This module defines the mapping between our internal piece IDs and the corresponding
4DBrix NCP part names used by BlueBrick for import/export.

Note: NCP part names may vary by library version. These mappings are based on the
4DBrix track library available in BlueBrick's Part Tracker.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PartMapping:
    """Mapping from our piece to NCP part.

    Attributes:
        ncp_part_name: Part identifier as recognized by nControl/BlueBrick.
        orientation_diff: Degrees to add to our angle to match NCP orientation.
            Accounts for orientation differences between coordinate systems.
        center_offset_x: X offset in studs from FK center to part visual center.
        center_offset_y: Y offset in studs from FK center to part visual center.
    """

    ncp_part_name: str
    orientation_diff: float = 0.0
    center_offset_x: float = 0.0
    center_offset_y: float = 0.0


# Main mapping dictionary: Our piece IDs -> NCP part definitions
# Part names based on 4DBrix track library in BlueBrick Part Tracker
PIECE_TO_NCP: Dict[str, PartMapping] = {
    # Straights
    "STRAIGHT_16": PartMapping("TS_STRAIGHT_1"),
    "STRAIGHT_24": PartMapping("TS_STRAIGHT_1_5"),
    # Curves (R40 = standard LEGO radius)
    # Left and right curves use same part, orientation handled by angle
    "R40_LEFT": PartMapping("TS_CURVE_R40_8"),
    "R40_RIGHT": PartMapping("TS_CURVE_R40_8"),
    # Crossings
    "CROSS_90": PartMapping("TS_CROSSING_90"),
    # Switches - IN variants (diverging from main line)
    "R40_SWITCH_LEFT_IN": PartMapping("TS_LEFTSPLITINSIDE"),
    "R40_SWITCH_RIGHT_IN": PartMapping("TS_RIGHTSPLITINSIDE"),
    # Switches - OUT variants (merging to main line)
    "R40_SWITCH_LEFT_OUT": PartMapping("TS_LEFTSPLITOUTSIDE"),
    "R40_SWITCH_RIGHT_OUT": PartMapping("TS_RIGHTSPLITOUTSIDE"),
    # Double crossover
    "DOUBLE_CROSSOVER": PartMapping("TS_DOUBLECROSSOVER"),
}


def get_ncp_mapping(piece_id: str) -> Optional[PartMapping]:
    """Get NCP mapping for a piece ID.

    Args:
        piece_id: Our internal piece ID (e.g., "STRAIGHT_16", "R40_LEFT").

    Returns:
        PartMapping if found, None otherwise.
    """
    return PIECE_TO_NCP.get(piece_id)


def has_ncp_mapping(piece_id: str) -> bool:
    """Check if a piece ID has an NCP mapping.

    Args:
        piece_id: Our internal piece ID.

    Returns:
        True if mapping exists, False otherwise.
    """
    return piece_id in PIECE_TO_NCP


def get_all_mappings() -> Dict[str, PartMapping]:
    """Get all piece-to-NCP mappings.

    Returns:
        Copy of the complete mapping dictionary.
    """
    return PIECE_TO_NCP.copy()
