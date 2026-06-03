---
description: Map src/ and hunt unused/dead code via parallel agents, then write findings into CLAUDE.md. Use when asked to audit code for dead/unused parts, build a code map, scan for cleanup candidates, or refresh the code-map section.
---

# /code-map-audit - Dead-Code Audit -> CLAUDE.md

Fan out parallel read-only agents to produce a one-line-per-module code map plus
a list of cleanup candidates, then write the result into CLAUDE.md's
"Code Map & Cleanup Findings" section. **This skill never deletes anything** — it
produces a reviewed inventory; deletion is a separate, explicitly-approved step.

## Arguments

| Argument | Description |
|----------|-------------|
| (none) | Audit `src/`, `tests/`, `configs/`, `data/`, root scripts, top-level docs. |
| `<path>` | Scope the audit to one subtree. |

## Execution

### 1. Fan out (parallel agents)

Dispatch independent `Explore` (or `general-purpose`) agents — one per subtree —
to map modules and flag symbols/files with no in-code importer or loader. Run
them in a single message so they execute concurrently. See
`superpowers:dispatching-parallel-agents`.

Each agent returns: one-line purpose per module, and a candidate list (symbol/file
+ the grep evidence that it has no caller).

### 2. Apply the HARD RULES before trusting any candidate

These caused real errors in the first audit pass — do not skip:

- **Re-grep every candidate immediately before listing it as dead, and quote the
  hits.** An earlier audit "verifying" a symbol is not enough — verify again now.
- **`configs/*.yaml` are CLI inputs** to `main.py --config <path>`, not Python
  imports. "No `.py` reference" is **NOT** evidence a config is dead. Check
  `archive/` and `outputs/` for runs that used it.
- **String references hide consumers**: pymoo callback/operator class names, log
  message strings, and YAML field names can be the only reference. Grep for the
  bare name, not just `import`.
- A Pydantic/YAML field with no reader is a real finding (orphan control) — but
  confirm it isn't consumed indirectly inside a property/method.

### 3. Self-audit pass

After assembling findings, re-read your own candidate list and challenge each
claim ("what of this seems out of place?"). Run the full test suite before and
after any later deletion and confirm the pass/fail counts are identical — the
suite has known pre-existing failures, so use the delta, not the absolute count.

### 4. Write into CLAUDE.md

Update the **"Code Map & Cleanup Findings"** section with:
- **Code Map**: one line per module (purpose + key symbols).
- **Candidates Needing Review**: table of path/symbol + evidence + notes.
  User-authored docs and research dirs → "ask before deleting", never auto-list
  as dead.
- **Verified Deletions log**: only after the user approves and you've confirmed
  the test delta.

Date the snapshot and label it point-in-time / re-verify before any `rm`.

### 5. Stop — do not delete

Present the candidates and wait for explicit approval. "Remove" / "untrack" means
`git rm --cached` or `.gitignore`, never `rm` from disk. Deleting files without
asking is a hard-don't.

## Notes

- Inventory is immutable elsewhere in this project, but here the deliverable is an
  *inventory of code* — keep it honest about uncertainty rather than over-claiming
  deadness.
- This pairs with `claude-md-management:claude-md-improver` if the broader
  CLAUDE.md also needs a quality pass.
