# Logging and Output Guide

## Overview

The LEGO Track Optimizer provides comprehensive logging and multiple output formats to help you understand optimization progress and results.

## Logging Configuration

### Log Destinations

Logs are written to **two locations**:
1. **Console (stdout)**: Real-time progress during optimization
2. **track_optimizer.log**: Persistent log file in the working directory

### Log Format

```
YYYY-MM-DD HH:MM:SS - LEVEL - Message
```

Example:
```
2025-02-25 14:32:15 - INFO - Loaded 28 track piece types
```

## CLI Output

### Startup Phase

When you run the optimizer, you'll see:

```
================================================================================
LEGO TRACK LAYOUT OPTIMIZER
================================================================================
Configuration file: configs/default.yaml
Track catalog: data/track_pieces.yaml
Output directory: outputs

Loading track catalog...
✓ Loaded 28 track piece types

Loading optimization config...
✓ Total inventory: 138 pieces

Inventory breakdown:
--------------------------------------------------------------------------------
  STRAIGHT_16                x  16
  R40_LEFT                   x  20
  R40_RIGHT                  x  20
  ...

Boundary constraints:
  X range: [-150.0, 150.0] studs
  Y range: [-150.0, 150.0] studs
  Area: 300.0 × 300.0 = 90000.0 studs²

Problem configuration:
  Decision variables: 138
  Objectives: 3 (speed, utilization, compactness)
  Constraints: 4 (closure, angle, boundary, inventory)
  ...
```

### Optimization Progress

During optimization, you'll see generation-level progress:

```
Generation   10 | Feasible:   42/1000 | Best: speed=1.245 m/s, util=78.3%, area=12450.5 studs²
Generation   20 | Feasible:   89/1000 | Best: speed=1.312 m/s, util=82.1%, area=11230.2 studs²
Generation   30 | Feasible:  134/1000 | Best: speed=1.389 m/s, util=85.6%, area=10890.7 studs²
...
```

Progress is logged every **10 generations** by default.

### Results Phase

After optimization completes, you'll see detailed chromosome information:

```
SOLUTION #1
Objectives:
  Average Speed: 1.423 m/s
  Utilization: 87.7%
  Bounding Box Area: 9845.3 studs²
Geometry:
  Pieces in Layout: 121
  Closure Error: 0.123 studs
  Angle Error: 1.45°

================================================================================
CHROMOSOME DETAILS
================================================================================
Total Pieces Used: 121
Unique Piece Types: 8
Empty Slots: 17

Piece Breakdown:
--------------------------------------------------------------------------------
Piece ID                     Count Type            Details
--------------------------------------------------------------------------------
  R40_LEFT                      18 curve           R=40, θ=22.5°
  R40_RIGHT                     18 curve           R=40, θ=22.5°
  STRAIGHT_16                   15 straight        L=16.0 studs
  STRAIGHT_32                    6 straight        L=32.0 studs
  ...

Raw Sequence (piece indices):
--------------------------------------------------------------------------------
  6, 6, 6, 6, 2, 2, 7, 7, 7, 7, 2, 2, 6, 6, 6, 6, 2, 2, 7, 7, 7, 7, 2, 2,
  4, 4, 6, 6, 6, 6, 7, 7, 7, 7, 4, 4, 2, 2, 2, ...
================================================================================
```

## Output Files

All results are saved to the output directory (default: `outputs/`).

### Visual Outputs (PNG)

| File | Description |
|------|-------------|
| `layout_1.png` | Best solution by speed |
| `layout_2.png` | Second-best solution |
| `layout_N.png` | Nth-best solution (up to 5) |
| `pareto_front.png` | 3D Pareto front visualization |
| `summary.png` | Grid of top 4 layouts |

### Chromosome Files (TXT)

Human-readable chromosome details for each solution:

**`chromosome_1.txt`**:
```
================================================================================
CHROMOSOME DETAILS
================================================================================
Total Pieces Used: 121
Unique Piece Types: 8
Empty Slots: 17

Piece Breakdown:
--------------------------------------------------------------------------------
  R40_LEFT                      18
  R40_RIGHT                     18
  STRAIGHT_16                   15
  ...

Raw Sequence (piece indices):
--------------------------------------------------------------------------------
  6, 6, 6, 6, 2, 2, 7, 7, 7, 7, ...
================================================================================

Objectives:
  Average Speed: 1.423 m/s
  Utilization: 87.7%
  Bounding Box Area: 9845.3 studs²
```

### Chromosome Files (JSON)

Machine-readable structured data for each solution:

**`chromosome_1.json`**:
```json
{
  "chromosome": {
    "raw_array": [6, 6, 6, 6, 2, 2, 7, 7, -1, -1, ...],
    "valid_indices": [6, 6, 6, 6, 2, 2, 7, 7, ...],
    "total_pieces": 121,
    "unique_types": 8,
    "empty_slots": 17
  },
  "piece_breakdown": {
    "R40_LEFT": 18,
    "R40_RIGHT": 18,
    "STRAIGHT_16": 15,
    ...
  },
  "objectives": {
    "average_speed_ms": 1.423,
    "utilization": 0.877,
    "bounding_box_area_studs2": 9845.3
  },
  "geometry": {
    "n_pieces": 121,
    "closure_error_studs": 0.123,
    "angle_error_degrees": 1.45,
    "total_angle_degrees": 358.55,
    "bounding_box": {
      "min_x": -48.2,
      "min_y": -65.3,
      "max_x": 52.1,
      "max_y": 89.8
    }
  }
}
```

### Summary Files

| File | Description |
|------|-------------|
| `summary.txt` | Text report with top solutions overview |
| `results.npz` | NumPy archive with raw optimization data (X, F, G arrays) |

**`summary.txt`**:
```
================================================================================
LEGO TRACK LAYOUT OPTIMIZATION - RESULTS SUMMARY
================================================================================

Total solutions evaluated: 850342
Feasible solutions found: 542/1000 (54.2%)
Top 5 solutions saved

Top Solutions Overview:
--------------------------------------------------------------------------------

Solution #1:
  Average Speed: 1.423 m/s
  Utilization: 87.7%
  Bounding Box Area: 9845.3 studs²
  Pieces Used: 121
  Closure Error: 0.123 studs
  Angle Error: 1.45°
...
```

## Using the JSON Output

The JSON chromosome files are designed for:

1. **Reproducing Layouts**: Load chromosome and pass to optimizer
2. **Analysis Scripts**: Parse objectives and geometry metrics
3. **Integration**: Connect to other tools/pipelines
4. **Database Storage**: Direct import into document stores

Example Python usage:

```python
import json
import numpy as np
from src.data import TrackCatalog
from src.geometry import build_layout

# Load chromosome
with open('outputs/chromosome_1.json', 'r') as f:
    data = json.load(f)

# Reconstruct layout
catalog = TrackCatalog.load('data/track_pieces.yaml')
chromosome = np.array(data['chromosome']['raw_array'], dtype=np.int32)
layout = build_layout(chromosome, catalog)

# Access metrics
print(f"Speed: {data['objectives']['average_speed_ms']} m/s")
print(f"Pieces: {data['piece_breakdown']}")
```

## Logging Levels

The optimizer uses standard Python logging levels:

- **INFO**: Normal operation (default)
- **WARNING**: Potential issues (e.g., no feasible solutions)
- **ERROR**: Serious problems

To increase verbosity, edit `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO
    ...
)
```

## Log File Management

The log file `track_optimizer.log` is **appended** on each run. To start fresh:

```bash
rm track_optimizer.log
python main.py
```

Or rotate logs manually:

```bash
mv track_optimizer.log track_optimizer_$(date +%Y%m%d_%H%M%S).log
python main.py
```

## Performance Monitoring

The logs capture key performance metrics:

- **Generation time**: Time per generation (from pymoo verbose output)
- **Evaluation count**: Total function evaluations
- **Feasibility rate**: Percentage of solutions meeting constraints
- **Convergence**: Best objective values over time

For detailed performance profiling, use:

```bash
python -m cProfile -o profile.stats main.py
python -m pstats profile.stats
```

## Troubleshooting

### No feasible solutions

Look for constraint violations in the log:
```
WARNING - No feasible solutions found! Check constraints.
```

Common causes:
- Inventory too limited
- Boundary too tight
- Closure tolerance too strict

### Slow convergence

Check generation progress:
```
Generation  100 | Feasible:   12/1000 | Best: speed=0.845 m/s, ...
Generation  200 | Feasible:   14/1000 | Best: speed=0.852 m/s, ...  # Barely improving
```

Solutions:
- Increase mutation probability
- Decrease population size for faster iterations
- Adjust termination criteria

### Memory issues

Large populations (1000+) with large inventories (200+ pieces) can use significant RAM. Monitor with:

```bash
# Linux/Mac
top -p $(pgrep -f main.py)

# Windows
tasklist | findstr python
```

If memory is constrained, reduce `pop_size` in config files.
