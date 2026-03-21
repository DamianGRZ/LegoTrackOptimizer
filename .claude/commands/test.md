---
description: Run pytest inline without spawning an agent. Fast, low-token test execution.
---

# /test - Inline Test Runner

Run pytest and report results directly in the conversation. No agent spawn overhead.

## Arguments

Parse arguments after `/test`:

| Argument | Description |
|----------|-------------|
| (none) | Run full test suite: `pytest --tb=short -q` |
| `<module>` | Run specific module: `pytest tests/test_<module>.py -v` |
| `-x` | Stop on first failure |
| `-v` | Verbose output |
| `-k <expr>` | Filter by test name expression |
| `all` | Run full suite verbose: `pytest -v` |

Module shortcuts: `geometry`, `evaluation`, `data`, `decoder`, `sampling`, `templates`, `problem`, `main`, `visualization`.

## Examples

```
/test                    # Quick full suite
/test geometry           # Just geometry tests
/test decoder -v         # Decoder tests verbose
/test -k "test_closure"  # Filter by name
/test all                # Full suite verbose
```

## Execution

### 1. Build the pytest command

- Default: `pytest --tb=short -q`
- If a module name is given, map it: `pytest tests/test_<module>.py -v`
- Append any extra flags (`-x`, `-v`, `-k`)

### 2. Run it

Execute with Bash tool. Timeout 120 seconds.

### 3. Report results

Output a compact summary:

```
## Test Results

**Command**: `pytest ...`
**Result**: X passed, Y failed, Z skipped in T seconds

### Failures (if any)
- `test_name` (file:line): error message (1 line)
```

Keep it SHORT. No preamble, no recommendations unless failures need explanation.
If all tests pass, just show the one-line summary. Do not list every passing test.