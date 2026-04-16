# Claude Code Cost Explorer

> A local web app to explore Claude Code usage — cost and token breakdown by day, session, and message turn.

No data leaves your machine. No API key needed. Reads directly from `~/.claude/`.

---

## Screenshot

![App preview](https://raw.githubusercontent.com/ranajoy-dutta/claude-code-cost-explorer/main/docs/screenshots/preview.jpeg)

Drill down: **Day → Session → Turn**

- 💬 Conversation turns → full messages + collapsible thinking
- 🛠️ Tool calls → `[tool: Bash]`, `[3 tools: Read, Write, Glob]` → inputs + raw output

---

## Install

```bash
pip install claude-code-cost-explorer
```

Run
```bash
ccx
```

Open: http://localhost:5050

## Requirements
Claude Code installed and used at least once (`~/.claude/projects/` exists)

## How it works

Reads token usage from:
`~/.claude/projects/**/*.jsonl`

Uses Claude pricing to compute cost.

---

## Tips: Session naming

Session titles come from whatever name Claude Code assigned the session. Three ways to set one:

| How | When |
|---|---|
| `claude -n "my-feature"` | At startup — best option |
| `/rename my-feature` | During a conversation |
| Press `R` in the `/resume` picker | After the fact |

If you don't name a session, Claude assigns a random slug (e.g. *jolly-spinning-mango*). A deliberate name like `payments-refactor` makes the session list much easier to scan when comparing costs across similar work.

---

## Contributing

Bug reports and PRs are welcome! Please:
1. Open an issue first for anything beyond small fixes
2. Run `uv run pytest -v` before submitting a PR
3. Keep the zero-dependency philosophy — no npm, no build step, no database

---

## License
MIT
