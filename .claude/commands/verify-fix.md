---
description: Verify a bug fix by running tests and the relevant optimizer config, then reporting literal output. Enforces the no-assertion-without-evidence rule.
---

# /verify-fix - Verify a Bug Fix End-to-End

Run the full verification loop after a code change. This skill exists because past sessions have shown Claude claiming fixes work without actually running the code. Do not skip any step.

## Arguments

| Argument | Description |
|----------|-------------|
| (none) | Full suite + default config |
| `<config>` | Full suite + a specific config (e.g. `with_switches`, `with_crossing`) |
| `<config> <module>` | Narrow tests to `tests/test_<module>.py` + run config |

## Examples

```
/verify-fix                          # Full suite, default config
/verify-fix with_switches            # Full suite, with_switches config
/verify-fix with_switches decoder    # test_decoder.py + with_switches
```

## Execution

### 1. Run the full test suite

```bash
pytest --tb=short -q
```

If a module arg was given, narrow to that file — but still run all other tests first with `pytest --tb=short -q` to catch regressions.

**Paste the literal tail of pytest output (last ~20 lines).** Do not paraphrase. Do not say "tests pass" without showing the summary line.

### 2. Run the optimizer config

```bash
python main.py --config configs/<config>.yaml --verbose
```

Default config if none specified: `default.yaml`. Do **not** use `--quick` — it is not valid for fix verification.

**Paste the last ~30 lines of stdout.** Include the final feasibility / fitness summary.

### 3. Run diagnostics

Invoke `/diag` on the fresh `outputs/` directory. Capture:

- Feasible-solution count (N / M)
- Closure error of best solution (studs)
- Orphan / loose switch count
- Whether `best_layout.png` shows a closed loop and the expected branches

### 4. Compare against the hypothesis

State explicitly:

- **What the fix claimed to do**
- **What the output actually shows**
- **Whether they agree**

If they disagree: do NOT explain the discrepancy away. Investigate the root cause, fix it, and rerun this skill from step 1.

### 5. Final verdict

One of:

- **VERIFIED** — all tests pass, optimizer produces feasible solutions matching the fix's intent, diagnostics confirm no regression on the target metric.
- **NOT VERIFIED** — state which step failed and what the next investigation step is. Do not claim the fix works.

---

## Rules

- **Never skip a step** to save time. The whole point of this skill is enforcement.
- **Never use `--quick`** for verification. Full runs only.
- **Never paraphrase output.** Copy literal lines.
- **Never explain away contradictions.** If the output contradicts the hypothesis, the hypothesis is wrong — investigate.
