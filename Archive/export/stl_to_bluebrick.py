"""Convert STL files to BlueBrick part library files (GIF + XML).

Renders STL files from top-down view and generates BlueBrick-compatible
GIF images and XML definition files.

BlueBrick format:
- GIF: 1 pixel = 1mm (8 pixels per stud)
- XML: Connection points, metadata
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageDraw
from stl import mesh

# Scale: 1 stud = 8mm = 8 pixels in BlueBrick
PIXELS_PER_MM = 1
PIXELS_PER_STUD = 8


@dataclass
class ConnectionPoint:
    """Connection point for BlueBrick XML."""
    x_studs: float  # studs from center
    y_studs: float  # studs from center (Y down positive in BlueBrick)
    angle: float  # degrees


def load_stl(stl_path: Path) -> mesh.Mesh:
    """Load an STL file."""
    return mesh.Mesh.from_file(str(stl_path))


def get_stl_bounds(stl_mesh: mesh.Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounding box of STL mesh.

    Returns:
        Tuple of (min_coords, max_coords) as [x, y, z] arrays in mm.
    """
    # Get all vertices
    vertices = stl_mesh.vectors.reshape(-1, 3)
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    return min_coords, max_coords


def render_stl_topdown(
    stl_mesh: mesh.Mesh,
    output_path: Path,
    scale: float = PIXELS_PER_MM,
    margin: int = 4,
    fill_color: Tuple[int, int, int] = (80, 80, 80),
    edge_color: Tuple[int, int, int] = (40, 40, 40),
    background: Tuple[int, int, int, int] = (255, 255, 255, 0),
) -> Tuple[Path, float, float]:
    """Render STL from top-down view to GIF.

    Args:
        stl_mesh: Loaded STL mesh.
        output_path: Output GIF path.
        scale: Pixels per mm.
        margin: Margin in pixels.
        fill_color: Fill color for triangles.
        edge_color: Edge color for triangles.
        background: Background color (RGBA).

    Returns:
        Tuple of (output_path, width_mm, height_mm).
    """
    # Get bounds
    min_coords, max_coords = get_stl_bounds(stl_mesh)

    # Dimensions in mm (XY plane for top-down view)
    width_mm = max_coords[0] - min_coords[0]
    height_mm = max_coords[1] - min_coords[1]

    # Image size
    img_width = int(width_mm * scale) + margin * 2
    img_height = int(height_mm * scale) + margin * 2

    # Create image
    img = Image.new('RGBA', (img_width, img_height), background)
    draw = ImageDraw.Draw(img)

    # Project triangles to 2D (top-down = XY plane)
    for triangle in stl_mesh.vectors:
        # Convert 3D points to 2D image coordinates
        points_2d = []
        for vertex in triangle:
            x = (vertex[0] - min_coords[0]) * scale + margin
            # Flip Y for image coordinates (Y increases downward)
            y = (max_coords[1] - vertex[1]) * scale + margin
            points_2d.append((x, y))

        # Draw filled triangle
        draw.polygon(points_2d, fill=fill_color, outline=edge_color)

    # Convert to palette mode for GIF with transparency
    # First, create a version with white background for areas we want transparent
    img_p = img.convert('P', palette=Image.ADAPTIVE, colors=255)

    # Save as GIF
    output_path = Path(output_path)
    img.save(output_path.with_suffix('.png'), 'PNG')  # Also save PNG for quality

    # For GIF, we need to handle transparency differently
    # Convert to RGB first, then to P
    img_rgb = Image.new('RGB', img.size, (255, 255, 255))
    img_rgb.paste(img, mask=img.split()[3])  # Use alpha as mask
    img_gif = img_rgb.convert('P', palette=Image.ADAPTIVE)
    img_gif.save(output_path, 'GIF', transparency=img_gif.getpixel((0, 0)))

    return output_path, width_mm, height_mm


def create_part_xml(
    part_id: str,
    part_name: str,
    connections: List[ConnectionPoint],
    output_path: Path,
) -> Path:
    """Create BlueBrick XML definition file.

    Args:
        part_id: Part identifier.
        part_name: Human-readable name.
        connections: List of connection points.
        output_path: Output XML path.

    Returns:
        Path to created XML file.
    """
    root = ET.Element("part")

    # Metadata
    ET.SubElement(root, "Author").text = "LEGO Track Optimizer"

    desc = ET.SubElement(root, "Description")
    desc_en = ET.SubElement(desc, "en")
    desc_en.text = part_name

    # Sorting key (for library ordering)
    ET.SubElement(root, "SortingKey").text = part_id

    # Snap margin
    snap = ET.SubElement(root, "SnapMargin")
    ET.SubElement(snap, "left").text = "0"
    ET.SubElement(snap, "right").text = "0"
    ET.SubElement(snap, "top").text = "0"
    ET.SubElement(snap, "bottom").text = "0"

    # Connection points
    if connections:
        conn_list = ET.SubElement(root, "ConnexionList")
        for i, conn in enumerate(connections):
            conn_el = ET.SubElement(conn_list, "connexion")

            # Type 1 = rail connection
            ET.SubElement(conn_el, "type").text = "1"

            pos = ET.SubElement(conn_el, "position")
            ET.SubElement(pos, "x").text = f"{conn.x_studs:.4f}"
            ET.SubElement(pos, "y").text = f"{conn.y_studs:.4f}"

            ET.SubElement(conn_el, "angle").text = f"{conn.angle:.1f}"
            ET.SubElement(conn_el, "angleToPrev").text = "0"
            ET.SubElement(conn_el, "angleToNext").text = "0"
            ET.SubElement(conn_el, "nextConnexionPreference").text = str((i + 1) % len(connections))
            ET.SubElement(conn_el, "electricPlug").text = "0"

    # Pretty print
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Save
    output_path = Path(output_path)
    output_path.write_text(pretty_xml, encoding='utf-8')

    return output_path


def convert_stl_to_bluebrick(
    stl_path: Path,
    output_dir: Path,
    part_id: Optional[str] = None,
    part_name: Optional[str] = None,
    connections: Optional[List[ConnectionPoint]] = None,
) -> Tuple[Path, Path]:
    """Convert STL file to BlueBrick GIF + XML.

    Args:
        stl_path: Path to STL file.
        output_dir: Output directory.
        part_id: Part ID (defaults to filename stem).
        part_name: Part name (defaults to part_id).
        connections: Connection points (optional).

    Returns:
        Tuple of (gif_path, xml_path).
    """
    stl_path = Path(stl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default part ID from filename
    if part_id is None:
        part_id = stl_path.stem.replace('-', '_').replace(' ', '_')

    if part_name is None:
        part_name = part_id

    # Load STL
    print(f"Loading: {stl_path}")
    stl_mesh = load_stl(stl_path)

    # Get bounds for info
    min_coords, max_coords = get_stl_bounds(stl_mesh)
    print(f"  Bounds: X={min_coords[0]:.1f} to {max_coords[0]:.1f} mm")
    print(f"          Y={min_coords[1]:.1f} to {max_coords[1]:.1f} mm")
    print(f"          Z={min_coords[2]:.1f} to {max_coords[2]:.1f} mm")

    # Render to GIF
    gif_path = output_dir / f"{part_id}.gif"
    gif_path, width_mm, height_mm = render_stl_topdown(stl_mesh, gif_path)
    print(f"  Created GIF: {gif_path} ({width_mm:.1f} x {height_mm:.1f} mm)")

    # Create XML
    xml_path = output_dir / f"{part_id}.xml"
    if connections is None:
        # Default connections for a switch (based on our track_pieces.yaml)
        # Switch is 16 studs long, connections at entry and two exits
        connections = [
            ConnectionPoint(x_studs=-8, y_studs=0, angle=180),  # Entry
            ConnectionPoint(x_studs=8, y_studs=0, angle=0),     # Straight exit
            ConnectionPoint(x_studs=7.65, y_studs=-3, angle=-22.5),  # Diverge exit (right)
        ]

    xml_path = create_part_xml(part_id, part_name, connections, xml_path)
    print(f"  Created XML: {xml_path}")

    return gif_path, xml_path


# Track piece connection definitions based on track_pieces.yaml
TRACK_CONNECTIONS = {
    "2_04_018": [  # R40_SWITCH_RIGHT_IN
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
        ConnectionPoint(x_studs=7.65, y_studs=3.04, angle=22.5),
    ],
    "2_04_019": [  # R40_SWITCH_RIGHT_OUT
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
        ConnectionPoint(x_studs=0.69, y_studs=3.04, angle=22.5),
    ],
    "2_04_021": [  # R40_SWITCH_LEFT_IN
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
        ConnectionPoint(x_studs=7.65, y_studs=-3.04, angle=-22.5),
    ],
    "2_04_022": [  # R40_SWITCH_LEFT_OUT
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
        ConnectionPoint(x_studs=0.69, y_studs=-3.04, angle=-22.5),
    ],
    "2_04_065": [  # STRAIGHT_16
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
    ],
    "2_04_066": [  # STRAIGHT_24
        ConnectionPoint(x_studs=-12, y_studs=0, angle=180),
        ConnectionPoint(x_studs=12, y_studs=0, angle=0),
    ],
    "2_04_069": [  # R40 curve
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=7.65, y_studs=-3.04, angle=-22.5),
    ],
    "2_04_002": [  # CROSS_90
        ConnectionPoint(x_studs=8, y_studs=0, angle=0),
        ConnectionPoint(x_studs=-8, y_studs=0, angle=180),
        ConnectionPoint(x_studs=0, y_studs=-8, angle=270),
        ConnectionPoint(x_studs=0, y_studs=8, angle=90),
    ],
}


def get_connections_for_part(part_number: str) -> Optional[List[ConnectionPoint]]:
    """Get connection points for a part number."""
    # Normalize part number
    normalized = part_number.replace('-', '_').replace('.', '_').split('_std')[0]
    # Try to find matching connections
    for key, conns in TRACK_CONNECTIONS.items():
        if key in normalized or normalized in key:
            return conns
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stl_to_bluebrick.py <stl_file> [output_dir]")
        print("\nExample:")
        print("  python stl_to_bluebrick.py data/2-04-018-std_v31.stl bluebrick_parts")
        sys.exit(1)

    stl_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bluebrick_parts")

    # Get connections based on part number
    connections = get_connections_for_part(stl_file.stem)

    gif_path, xml_path = convert_stl_to_bluebrick(
        stl_file,
        output_dir,
        connections=connections,
    )

    print(f"\nDone! Files created in {output_dir}")
    print(f"  - {gif_path.name}")
    print(f"  - {xml_path.name}")
