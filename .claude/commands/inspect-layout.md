---
description: Visually inspect a run's layout/snapshot PNGs for geometry correctness — switch branch direction, siding spacing, crossings, path closure — and decide visualization bug vs real geometry bug. Use when asked to analyze snapshots, check a layout image, or verify visual correctness.
---

# /inspect-layout - Visual Geometry Inspection

Read the actual layout images for a run and check them against the project's
geometry invariants. This is the **visual** counterpart to `/diag` (which parses
CSVs) — here you must Read the PNGs and reason about what is drawn.

## Arguments

| Argument | Description |
|----------|-------------|
| (none) | Inspect `outputs/best_layout.png`. |
| `<run-dir>` | Inspect that run dir (e.g. `outputs/verify_with_switches`). |
| `--snapshots` | Also walk `<run-dir>/snapshots/*.png` over time, not just the final layout. |
| `--infeasible` | Include `best_infeasible.png` / `snapshot_NN_infeasible.png`. |

## Examples

```
/inspect-layout                                  # outputs/best_layout.png
/inspect-layout outputs/verify_with_switches     # final best layout
/inspect-layout outputs/verify_with_crossing --snapshots
```

## Execution

### 1. Read the images

Use the Read tool on the real PNGs — never infer geometry from logs or CSVs.

- Final layout: `<run-dir>/best_layout.png` (and `best_infeasible.png` with `--infeasible`).
- With `--snapshots`: `<run-dir>/snapshots/snapshot_NN_feasible.png` (and
  `..._infeasible.png` with `--infeasible`), in numeric order, to see how geometry
  evolved. Sample a handful (first, middle, last) unless asked for all.

### 2. Check against geometry invariants

| Invariant | What to look for |
|-----------|------------------|
| **Closure** | Main loop returns to start; every traversal path labeled CLOSED, not OPEN. |
| **Switch pairing** | Passing sidings use one LEFT + one RIGHT switch (opposite-handed), OUT installed reversed — not two same-handed switches. |
| **Branch direction** | Each diverging branch exits **one** side of the main loop and returns; never both directions, never an arm drifting off-track. |
| **Siding spacing** | Parallel siding sits ~8 studs from the main loop; switch port-C lateral offset ~5 studs (the catalog's 6.2 is wrong). |
| **Crossings** | Where two segments cross, a CROSS_90 piece is placed at the crossing — not an unsubstituted overlap. |
| **Boundary** | Whole layout stays inside the boundary box. |

### 3. Decide: visualization bug vs real geometry bug

The user repeatedly needs this distinction. Cross-check the suspicious visual
against the data:

- If `/diag` reports the path CLOSED and constraints feasible but it *looks* wrong
  → likely a **renderer** issue (`src/visualization/track_renderer.py`).
- If the FK chain / constraints also show the defect → **real geometry** issue
  (decoder, templates, or operators). For root-causing, use
  `superpowers:systematic-debugging`.

State which one you believe it is and the evidence for it.

### 4. Report

Per image (or per snapshot range), list invariants that PASS and any that FAIL,
naming the specific file. End with a one-line verdict: clean / visualization bug
in `<module>` / real geometry bug in `<module>`. Don't claim "looks good" without
having actually Read the image.

## Notes

- A missing 5-stud gap can be purely a rendering scale artifact — verify against
  the FK data before calling it a geometry defect.
- The final feasible solution must be closed and connected with no dangling
  ports; dangling ports are acceptable only in early infeasible snapshots.
