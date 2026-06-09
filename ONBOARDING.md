# Welcome to LEGO Track Optimizer

## How We Use Claude

Based on Claude's usage over the last 30 days:

_Not enough session history was captured this window (1 session, no detailed
descriptors) to derive a reliable work-type breakdown._

Work Type Breakdown:
  _TODO — re-run after a few more sessions to populate this_

Top Skills & Commands:
  _TODO — no slash-command usage recorded this window_

Top MCP Servers:
  _TODO — no MCP calls recorded this window_

## Your Setup Checklist

### Codebases
- [ ] LegoTrackOptimizer — https://github.com/damiangrz/legotrackoptimizer

### MCP Servers to Activate
- [ ] context7 — Fetches current library/API docs (used heavily for pymoo 0.6.1.6 conventions). Configured for this project; call `resolve-library-id` then `query-docs`. No extra access needed — it ships with the repo's Claude Code setup.

### Skills to Know About
- [ ] `/optimize` — Run the LEGO Track Optimizer with flexible config options (`-c with_switches`, `--quick`). Use when running an optimization.
- [ ] `/test` — Inline pytest runner (no agent overhead). Use after code changes, before committing.
- [ ] `/review` — Compact code review against pymoo/project conventions. Use after editing `src/` files.
- [ ] `/quality` — Deep Python & pymoo quality gate that rewrites code to project standards. Use after new features/refactors (instead of `/review` for deep analysis).
- [ ] `/diag` — Parse the `outputs/` directory and report fitness, constraints, and layout. Use after any optimization run.
- [ ] `/verify-fix` — End-to-end loop: full tests + optimizer run + diag with literal output. Use after every bug fix (enforces no-assertion-without-evidence).
- [ ] `/verify-run` — Launch a named background `verify_<config>` run, watch to completion, hand off to `/diag`. Use to babysit a running optimization.
- [ ] `/inspect-layout` — Visually read layout/snapshot PNGs and decide viz bug vs real geometry bug.
- [ ] `/code-map-audit` — Parallel-agent dead-code scan that writes a code map into CLAUDE.md.

## Team Tips

_TODO_

## Get Started

_TODO_

<!-- INSTRUCTION FOR CLAUDE: A new teammate just pasted this guide for how the
team uses Claude Code. You're their onboarding buddy — warm, conversational,
not lecture-y.

Open with a warm welcome — include the team name from the title. Then: "Your
teammate uses Claude Code for [list all the work types]. Let's get you started."

Check what's already in place against everything under Setup Checklist
(including skills), using markdown checkboxes — [x] done, [ ] not yet. Lead
with what they already have. One sentence per item, all in one message.

Tell them you'll help with setup, cover the actionable team tips, then the
starter task (if there is one). Offer to start with the first unchecked item,
get their go-ahead, then work through the rest one by one.

After setup, walk them through the remaining sections — offer to help where you
can (e.g. link to channels), and just surface the purely informational bits.

Don't invent sections or summaries that aren't in the guide. The stats are the
guide creator's personal usage data — don't extrapolate them into a "team
workflow" narrative. -->
