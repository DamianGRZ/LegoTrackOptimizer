# Progression Snapshots — Best Feasible and Infeasible Designs Across Generations

**Date:** 2026-04-22
**Status:** Approved — ready for implementation plan

## Motivation

The current optimizer only persists images for the final population: `best_layout.png` (best feasible) and `best_infeasible.png` (best infeasible in final pop). When a user wants to understand *how* the GA converged — which designs looked promising at gen 10 vs gen 50 vs gen 100, or how infeasible sidings evolved toward feasibility — the final snapshot hides all of it.

This spec adds a lightweight progression view: at ten evenly spaced generations, save the best feasible and the best infeasible individual to disk as standalone PNGs.

## Scope

### In scope
- A new callback `SnapshotCallback` that records the best feasible and best infeasible individual at ten target generations.
- A render step in `save_results` that produces per-snapshot PNGs under `outputs/snapshots/`.
- A test that executes a short run and asserts the expected files appear.

### Out of scope
- Grid/mosaic summary image combining all snapshots.
- CSV dump of snapshot chromosomes, fitness, or constraint vectors.
- Config flag to enable/disable the feature. It is always on.
- Feasibility-progression animation (GIF/video).
- Snapshots at non-uniform generations (e.g., at first feasibility, at Pareto-front shifts).

## Target Generations

Given `n_gen` from the active config:

```
stride = max(1, n_gen // 10)
targets_raw = [stride, 2*stride, 3*stride, ..., 10*stride]
targets = sorted(set(min(g, n_gen) for g in targets_raw))
```

The final entry is clamped so it never exceeds `n_gen`. Duplicates (possible when `n_gen < 10`) are removed. For `n_gen = 100` the targets are exactly `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`.

## Selection Criterion

At each target generation, the callback partitions the population by the existing feasibility rule:

- **Feasible** = rows where `all(G <= 0)`.
- **Infeasible** = rows where `any(G > 0)`.

Within each partition, the "best" individual is the one with the **lowest `F[0]`** (i.e., highest utilization — the first objective is `-utilization`). This matches the convention already used in `save_results` for picking `best_layout.png` and `best_infeasible.png`.

If a partition is empty at that generation, no file is written for that (snapshot, category) pair. The missing file is not an error.

## Storage Mechanism

`SnapshotCallback` maintains `self.snapshots: list[dict]`, one entry per taken snapshot with these fields:

| Field | Type | Notes |
|---|---|---|
| `snapshot_idx` | int | 1-based index, 1..10 |
| `gen` | int | Actual generation number at snapshot |
| `feasible` | dict or None | `{X, F, G, cv}` copy of best feasible, or None if partition empty |
| `infeasible` | dict or None | Same shape, for best infeasible |

`X`, `F`, `G` are `np.ndarray` copies (`.copy()`, not deep copy of the Individual object) to avoid pymoo mutating them downstream. `cv` is the scalar constraint violation sum, cached so we don't recompute in the renderer.

## Rendering

`save_results` calls a new helper `render_snapshots(callback, output_dir, catalog, config, dims)` that:

1. Creates `output_dir / "snapshots/"`.
2. Iterates over `callback.snapshots`.
3. For each recorded `feasible`/`infeasible` entry, decodes the chromosome via `decode_chromosome`, dispatches to `plot_layout` or `plot_multi_path_layout` by `layout.n_switch_pairs`, and writes to:

```
outputs/snapshots/snapshot_{idx:02d}_feasible.png
outputs/snapshots/snapshot_{idx:02d}_infeasible.png
```

Filename contains **only** the snapshot index and category — no generation number, no fitness. The generation is surfaced in the image title:

```
Snapshot 3/10 · Gen 30 · feasible · 42 pcs, util=87.0%, speed=0.92 m/s
Snapshot 7/10 · Gen 70 · infeasible · 38 pcs, util=79.2%, speed=0.88 m/s, CV=4.31
```

`CV` is only shown in the infeasible title.

## Integration Points

### `src/algorithm/runner.py`

1. New class `SnapshotCallback(Callback)` alongside `ProgressCallback` and `FeasibleEliteCallback`. Constructor takes the precomputed `targets: list[int]` (1-indexed generation numbers). On each `notify`:
   - If `algorithm.n_gen` matches a target not yet taken, partition by feasibility, pick best in each, store a snapshot entry with `snapshot_idx = targets.index(gen) + 1`.
2. Construct it in `run_optimization` via a module-level helper `_compute_snapshot_targets(n_gen: int) -> list[int]`.
3. Append to the existing `CallbackChain` (both `verbose` and non-verbose branches).
4. Attach it to the result: `res.snapshots = snapshot_callback.snapshots` so `save_results` can read it.
5. Extend `save_results` signature to read `res.snapshots` and dispatch to `render_snapshots`. No signature change for external callers.

### `src/visualization/`

No new public function required. `plot_layout` and `plot_multi_path_layout` are already called with an explicit `save_path`; `render_snapshots` reuses them unchanged.

## Error Handling

- If `render_snapshots` fails on one snapshot (e.g., decoder raises on a mutated chromosome), log a warning with the snapshot index and continue with the rest. One bad snapshot should not block the others.
- If `outputs/snapshots/` already exists from a previous run, overwrite per-file (matplotlib's `savefig` does this by default). No pre-clean step.
- If the callback was never attached (shouldn't happen, but defensive), `res.snapshots` is absent; `save_results` skips the render call with `getattr(res, "snapshots", None)`.

## Testing

New test `tests/test_snapshots.py`:

1. Build a minimal `OptimizationConfig` with `n_gen=20`, `pop_size=16`, default inventory.
2. Run `run_optimization` + `save_results` into a `tmp_path`.
3. Assert `tmp_path / "snapshots"` exists.
4. Assert that for each `snapshot_idx` in 1..10, at least one of `snapshot_{idx:02d}_feasible.png` / `snapshot_{idx:02d}_infeasible.png` was produced (given a 20-gen run, every generation yields at least one category).
5. Assert at least one `*_feasible.png` and one `*_infeasible.png` exist overall (otherwise the feature is not exercised).

Test runtime target: < 60 s on a developer machine.

Existing `tests/test_visualization.py` remains untouched.

## Design Decisions and Rationale

**Why a callback, not `save_history=True`?** Pymoo's history mode stores the entire algorithm state at every generation — population arrays, duplicates of operator state, etc. For a 100-generation run with pop_size 100 this is hundreds of MB. The callback stores only 20 individuals max (10 snapshots × 2 categories), each a small numpy copy. Constant-memory cost regardless of `n_gen`.

**Why no config flag?** The output is cheap (≤ 20 PNGs, each a few hundred KB) and always useful for diagnosis. A flag would be one more thing to remember when something goes wrong.

**Why `F[0]` (highest utilization) as "best"?** Consistency with the existing `save_results` logic. If later the project wants to show snapshots ranked by Pareto rank or by speed, that's a separate design.

**Why 1-based snapshot indices in filenames?** Humans read `snapshot_01` more naturally than `snapshot_00`. The zero-based internal index stays zero-based in code.

**Why no generation in filename?** Per user preference — the filename stays stable across runs with the same `n_gen`, and the generation number is right there in the title for context.
