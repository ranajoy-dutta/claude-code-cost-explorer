# Claude Code Cost Explorer

> A local web app that reads your [Claude Code](https://claude.ai/code) conversation files in your local machine and shows you API cost and token usage — broken down by day, session, and individual message turn.

No data leaves your machine. No API key needed. Reads directly from `~/.claude/`.

---

## Screenshots

![App preview](https://raw.githubusercontent.com/ranajoy-dutta/claude-code-cost-explorer/main/docs/screenshots/preview.jpeg)

**Navigation flow:** Day view → click a date → Session list → click a session → Turn-by-turn breakdown

Each turn is a hyperlink:
- Conversation turns → full user message + full assistant response (with collapsible thinking blocks)
- Tool call turns → `[tool: Bash]` or `[3 tools: Read, Write, Glob]` → input params + raw output

---

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — install once, handles everything else:
  - **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - **Windows:** `winget install astral-sh.uv`
- **[Claude Code](https://claude.ai/code)** installed and used at least once (so `~/.claude/projects/` exists)

---

## Installation

```bash
git clone https://github.com/ranajoy-dutta/claude-code-cost-explorer.git
cd claude-code-cost-explorer
uv sync
```

That's it — `uv sync` creates the virtual environment and installs all dependencies automatically.

---

## Running

```bash
uv run flask --app app run --port 5050
```

Open [http://localhost:5050](http://localhost:5050).

No venv activation needed — `uv run` handles it automatically.

---

## How costs are calculated

Tokens are read directly from `~/.claude/projects/**/*.jsonl` — Claude Code writes them on every API response. Costs use Claude's published pricing:

| Model | Input | Output | Cache write | Cache read |
|---|---|---|---|---|
| claude-opus-4-x | $15/MTok | $75/MTok | $18.75/MTok | $1.50/MTok |
| claude-sonnet-4-x | $3/MTok | $15/MTok | $3.75/MTok | $0.30/MTok |
| claude-haiku-4-x | $0.80/MTok | $4/MTok | $1/MTok | $0.08/MTok |

If Anthropic changes pricing or you're on a custom plan, update the `PRICING` table in [`cost.py`](cost.py).

---

## Running tests

```bash
uv run pytest -v
```

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

## Notes

- **~79% of turns are tool-call responses**, not direct replies to your messages — this is normal. Each `Read`, `Bash`, `Write`, etc. generates its own API call with its own token usage.
- Cache read tokens are 10× cheaper than input tokens. A session that looks expensive at first glance often has most of its cost in cache reads.
- Subagent sessions (parallel agents) are excluded from their parent session's cost to avoid double-counting.

---

## Contributing

Bug reports and PRs are welcome! Please:
1. Open an issue first for anything beyond small fixes
2. Run `uv run pytest -v` before submitting a PR
3. Keep the zero-dependency philosophy — no npm, no build step, no database

---

## License

MIT — see [LICENSE](LICENSE).
