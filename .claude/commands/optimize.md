---
description: Run LEGO Track Optimizer with flexible configuration options (quick test, different configs, multi-segment mode)
---

# /optimize - Run LEGO Track Optimization

Run the LEGO Track Optimizer with flexible configuration options.

## Arguments

The user may provide arguments after `/optimize`. Parse these arguments:

| Argument | Description |
|----------|-------------|
| `--quick` or `-q` | Quick test mode (20 gen, pop_size=20) |
| `--config <name>` or `-c <name>` | Config name or path (default, compact, with_switches, with_crossing) |
| `--multi-segment` or `-m` | Use multi-segment encoding (Phase 1 experimental) |
| `--output <dir>` or `-o <dir>` | Output directory (default: outputs) |
| `--verbose` or `-v` | Enable verbose logging |

## Examples

```
/optimize                           # Default config, full run
/optimize --quick                   # Quick test (20 gen)
/optimize -c compact                # Use compact.yaml config
/optimize --config with_switches    # Use with_switches.yaml
/optimize -q -c with_crossing       # Quick test with crossing config
/optimize --multi-segment           # Experimental multi-segment mode
```

## Execution Steps

### 1. Parse Arguments

Map shorthand config names to full paths:
- `default` -> `configs/default.yaml`
- `compact` -> `configs/compact.yaml`
- `with_switches` -> `configs/with_switches.yaml`
- `with_crossing` -> `configs/with_crossing.yaml`

If a full path is provided (contains `/` or `.yaml`), use it directly.

### 2. Build Command

Construct the command based on parsed arguments:

```bash
python main.py \
    --config configs/<config>.yaml \
    [--quick-test] \
    [--multi-segment] \
    [--output <dir>] \
    --verbose
```

Always include `--verbose` to show progress.

### 3. Run Optimization

Execute the command and capture output. The optimization will log:
- Population size and generations
- Progress every 10 generations (feasible count, best utilization, best speed)
- Final Pareto front size or feasible solution count

### 4. Parse Results

After completion, read the output files to extract metrics:

**For legacy mode (bi-objective):**
- Read `outputs/objectives.csv` to get Pareto front size
- Calculate best utilization: `-min(F[:, 0])`
- Calculate best speed: `-min(F[:, 1])`

**For multi-segment mode:**
- Read `outputs/fitness.csv` to get best fitness
- Read `outputs/constraints.csv` to count feasible solutions

### 5. Display Summary

Present results in a clear format:

```
## Optimization Complete

**Mode**: Legacy (NSGA-II) / Multi-segment (GA)
**Config**: <config_name>
**Generations**: <n_gen>
**Population**: <pop_size>

### Results
- **Pareto Front Size**: <N> solutions
- **Best Utilization**: <X>%
- **Best Avg Speed**: <X.XX> m/s
- **Feasible Solutions**: <N>

### Output Files
- Layout: outputs/best_layout.png
- Pareto Front: outputs/pareto_front.png
- Objectives: outputs/objectives.csv
```

### 6. Quick Tips

If the run fails or produces poor results, suggest:
- If no feasible solutions: Try `--quick` first to validate setup
- If low utilization: Check inventory in config
- If optimization is slow: Use `--quick` for testing

## Error Handling

- If config file not found: List available configs in `configs/` directory
- If Python errors: Show the traceback and suggest checking dependencies
- If memory issues: Suggest using `--quick` or reducing `pop_size` in config
- If interrupted: Note that partial results may be in the output directory

## Notes

- Always use `--verbose` flag so progress is visible
- Output files are saved to `outputs/` by default (or custom `--output` dir)
- Quick mode is useful for testing changes before full runs
- Multi-segment mode is experimental - use legacy mode for production
