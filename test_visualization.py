"""Quick test to verify visualization features."""
import numpy as np
from src.config import OptimizationConfig
from src.catalog import TrackCatalog
from src.geometry import build_layout
from src.visualization import plot_layout

# Load catalog and config
catalog = TrackCatalog.load('data/track_pieces_v2.yaml')
config = OptimizationConfig.load('configs/compact.yaml')

# Create simple test layout: 16 R40_LEFT pieces (circle)
pieces = np.array([2] * 16, dtype=np.int32)  # R40_LEFT index is 2
layout = build_layout(pieces, catalog)

print(f"Layout has {layout.n_pieces} pieces")
print(f"Closure error: {layout.closure_error:.2f} studs")
print(f"Angle error: {layout.angle_error:.2f}°")
print(f"Is closed: {layout.is_closed()}")

# Generate visualization with save
fig = plot_layout(layout, catalog, config.boundary, title="Test Circle Layout", save_path="test_layout.png")
print("\n✓ Test visualization saved to test_layout.png")
print("\nFeatures to verify in the PNG:")
print("  - Green SQUARE at start position")
print("  - Red CIRCLE at end position")
print("  - Black dots at connection points (between pieces)")
print("  - All pieces same color (R40_LEFT = one piece type)")
