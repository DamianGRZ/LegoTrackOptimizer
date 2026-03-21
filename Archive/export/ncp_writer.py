"""NCP (4DBrix nControl) XML file writer.

The NCP format is used by 4DBrix nControl and can be imported by BlueBrick.
It uses a simple XML structure with segment elements containing type, position,
and angle information.

Example NCP structure:
```xml
<?xml version="1.0" ?>
<layout name="Track Layout" version="1.0">
  <segment id="0">
    <type value="TS_STRAIGHT_1"/>
    <position x="8.0000" y="0.0000"/>
    <angle value="0.0000"/>
  </segment>
  ...
</layout>
```
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from xml.dom import minidom


@dataclass
class NCPSegment:
    """Single track segment for NCP export.

    Attributes:
        part_name: NCP part type identifier (e.g., "TS_STRAIGHT_1").
        x: X position in studs.
        y: Y position in studs.
        angle: Rotation angle in degrees.
        origin: Connection index used as origin (default 0).
    """

    part_name: str
    x: float
    y: float
    angle: float
    origin: int = 0


def generate_ncp_xml(
    segments: List[NCPSegment],
    layout_name: str = "Track Layout",
    version: str = "1.0",
    pretty_print: bool = True,
) -> str:
    """Generate NCP file content as XML string.

    Args:
        segments: List of track segments to include.
        layout_name: Name for the layout (appears in BlueBrick).
        version: NCP format version.
        pretty_print: If True, format XML with indentation.

    Returns:
        NCP XML content as formatted string.
    """
    # Create root element
    root = ET.Element("layout")
    root.set("name", layout_name)
    root.set("version", version)

    # Add each segment
    for i, seg in enumerate(segments):
        segment_el = ET.SubElement(root, "segment")
        segment_el.set("id", str(i))

        # Part type
        type_el = ET.SubElement(segment_el, "type")
        type_el.set("value", seg.part_name)

        # Position
        pos_el = ET.SubElement(segment_el, "position")
        pos_el.set("x", f"{seg.x:.4f}")
        pos_el.set("y", f"{seg.y:.4f}")

        # Angle
        angle_el = ET.SubElement(segment_el, "angle")
        angle_el.set("value", f"{seg.angle:.4f}")

        # Origin connection index (optional, default 0)
        if seg.origin != 0:
            origin_el = ET.SubElement(segment_el, "origin")
            origin_el.set("value", str(seg.origin))

    # Convert to string
    xml_str = ET.tostring(root, encoding="unicode")

    if pretty_print:
        # Use minidom for pretty printing
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

    return f'<?xml version="1.0" ?>\n{xml_str}'


def write_ncp_file(
    segments: List[NCPSegment],
    file_path: str,
    layout_name: Optional[str] = None,
) -> None:
    """Write NCP file to disk.

    Args:
        segments: List of track segments to include.
        file_path: Output file path (should end with .ncp).
        layout_name: Optional layout name, defaults to filename.
    """
    if layout_name is None:
        # Use filename as layout name
        from pathlib import Path

        layout_name = Path(file_path).stem

    xml_content = generate_ncp_xml(segments, layout_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
