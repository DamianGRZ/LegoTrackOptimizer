---
description: Run a named verify optimization in the background, watch it to completion, then report and hand off to /diag. Use when asked to run/verify a config, launch an optimizer run, or babysit a running optimization.
---

# /verify-run - Background Verify Run + Monitor

Launch one config as a named `verify_<config>` run **in the background**, let it
finish, then summarize and hand off to `/diag`. This replaces hand-typing the
full `main.py` command and manually polling "is it done yet".

## Arguments

| Argument | Description |
|----------|-------------|
| `<config>` | Config name (see mapping) or full path. **Required.** |
| `--fg` | Run in foreground and block (only for `--quick`-style short runs). |
| `--quick` or `-q` | Quick test mode (passes `--quick-test`). |
| `--name <suffix>` | Override the run name; default is `verify_<config>`. |

## Examples

```
/verify-run with_switches              # -> outputs/verify_with_switches, background
/verify-run with_crossing              # -> outputs/verify_with_crossing
/verify-run with_double_crossover -q   # quick smoke test
/verify-run with_switches --name sw_b  # -> outputs/sw_b
```

## Config name mapping

Names map to `configs/<name>.yaml`. Available: `default`, `compact`,
`with_switches`, `with_crossing`, `with_switches_and_crossing`,
`with_double_crossover`, `with_double_crossover_narrow`,
`with_double_crossover_small`. A value containing `/` or `.yaml` is used as-is.

## Execution

### 1. Resolve output dir

Output goes to `outputs/verify_<config>` (or `outputs/<--name>`). **Keep all
artifacts under the single `outputs/` tree** — never create `outputs_v1/` or any
parallel top-level output dir (that convention is retired; see commit
"consolidate run artifacts under single outputs/ tree").

If the dir already exists with results, ask before overwriting, or append a
numeric suffix (`verify_with_switches_2`) — match whatever the user intends.

### 2. Build the command

```bash
.venv/Scripts/python.exe main.py \
    --config configs/<config>.yaml \
    --output outputs/verify_<config> \
    [--quick-test] \
    --verbose
```

Always pass `--verbose` so the log shows per-generation progress.

### 3. Launch

- Default: launch with the Bash tool using `run_in_background: true`. The harness
  re-invokes you when the process exits — **do not poll in a sleep loop**. If you
  must surface mid-run progress, tail the captured output once, then wait for the
  completion notification.
- `--fg` / `--quick`: run in the foreground and block on the result.

### 4. On completion

Confirm the run actually finished (process exited 0; `outputs/verify_<config>/`
contains `fitness.csv`, `constraints.csv`, `best_layout.png`, `run_info.md`).
If it crashed, surface the traceback — do not report success.

### 5. Report + hand off

Print a 3-line status (config, generations reached, feasible-or-not), then run
`/diag outputs/verify_<config>` for the full fitness/constraint/layout breakdown.
Per project rules, verify feasibility via `/diag` before declaring success —
a clean exit code is not evidence of a good layout.

If the layout has switches/crossings/double-crossovers, suggest
`/inspect-layout outputs/verify_<config>` to visually check branch geometry.

## Notes

- Background launch + harness notification is the intended path; reserve
  `ScheduleWakeup` only as a long fallback if a run hangs with no notification.
- `snapshots/` fills during the run — useful for `/inspect-layout` even before the
  final layout is written.
