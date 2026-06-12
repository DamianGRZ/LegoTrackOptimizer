---
description: Slice a mixed working tree into logical, individually-green commits. Use when asked to split/cut session work into commits ("potnij na commity"), when one file contains hunks belonging to different concerns, or before pushing a long session's accumulated changes.
---

# /commit-slices - Logical, Individually-Green Commits

Every commit must (a) tell one story and (b) pass the FULL test suite on its
own tree — the history must stay bisect-able.

## Hard rules

- **NEVER `git stash`** — `stash@{0}` in this repo is a foreign GitHub
  Desktop stash; pop/apply would corrupt the tree. Use temp-file backups.
- **Never create/switch branches** without explicit user OK. Commit on the
  current branch.
- **Never commit**: `outputs/`, scratch files (review diffs, temp scripts),
  anything you didn't change deliberately.
- **Never `--no-verify`**, never amend other people's commits.
- **No AI-attribution footers** (no `Co-Authored-By: Claude ...`) — plain
  commit messages only.

## Execution

### 1. Plan the slices

`git status --short` + `git diff --stat`. Group by concern, not by file type.
User-authored edits (e.g. config changes) get their own commit, first.
Order so each intermediate tree is self-consistent — a test asserting
behavior introduced in slice 3 must be committed in slice 3, even if the
rest of its file belongs to slice 2.

### 2. Pure-file slices

`git add <files of this slice>` → commit. Done.

### 3. Mixed files (hunks from several slices)

For each file whose hunks span slices:

```bash
cp <file> "$TEMP/<file>_full"        # backup the complete version
git checkout HEAD -- <file>          # reset to last commit
# re-apply ONLY this slice's hunks with the Edit tool (you made them once —
# redo them precisely), then commit with this slice's other files
cp "$TEMP/<file>_full" <file>        # restore; remaining hunks = later slices
```

Re-apply any test-file hunks the same way (often one small reverse-edit on
the shared test file is cheaper than checkout + forward edits — pick per file).

### 4. Verify EVERY commit's tree

The working tree during slicing contains FUTURE slices — a green suite there
proves nothing about the commit. After all commits exist:

```bash
git worktree add "$TEMP/verify_cN" <sha> --detach
cd "$TEMP/verify_cN" && python -m pytest -q --tb=line   # record literal tail
cd - && git worktree remove --force "$TEMP/verify_cN"
```

HEAD equals the working tree — test it directly. Quote the literal pass/fail
counts per commit; pre-existing failures must match the baseline exactly
(baseline = the suite result on the pre-slicing HEAD; if unknown, measure it
in a detached worktree first).

Message format: `type(scope): summary` with a short body saying WHY — no
footers.

### 5. Report

Table: commit hash → one-line content → literal suite result. Mention
untracked leftovers (scratch files) — do not delete them yourself.

## Red flags — stop and re-read this skill

- "stash will be faster here"
- "the intermediate commits don't need their own green suite"
- "I'll just put everything in one commit, it's all related"
- "the working tree passed, so the commits are fine"
