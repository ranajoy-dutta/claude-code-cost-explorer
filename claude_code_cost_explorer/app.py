"""Claude Code Cost Tracker. Run: ccx"""

from flask import Flask, render_template, request, abort, redirect, url_for
from claude_code_cost_explorer.reader import (
    load_all_sessions as _load_all_sessions,
    build_day_summaries,
    get_sessions_for_date,
    get_session_by_id,
    CLAUDE_DIR,
    _PROJECTS_DIR_OVERRIDE,
    append_custom_session_title,
)

import os
import pathlib
from datetime import date, timedelta
import markdown
from markupsafe import Markup

app = Flask(__name__, template_folder=str(pathlib.Path(__file__).parent / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True
# ---------------------------------------------------------------------------
# Session cache — invalidates automatically when any JSONL file changes on disk
# ---------------------------------------------------------------------------
_session_cache: list | None = None
_session_cache_key: frozenset | None = None


def _jsonl_fingerprint(claude_dir: str = CLAUDE_DIR) -> frozenset:
    """Return a frozenset of (path, mtime_ns) for every JSONL file under projects/."""
    if _PROJECTS_DIR_OVERRIDE:
        projects_dir = os.path.expanduser(_PROJECTS_DIR_OVERRIDE)
    else:
        projects_dir = os.path.join(claude_dir, "projects")
    entries = []
    if os.path.isdir(projects_dir):
        for proj in os.listdir(projects_dir):
            proj_dir = os.path.join(projects_dir, proj)
            if not os.path.isdir(proj_dir):
                continue
            for fname in os.listdir(proj_dir):
                if fname.endswith(".jsonl"):
                    full = os.path.join(proj_dir, fname)
                    try:
                        entries.append((full, os.stat(full).st_mtime_ns))
                    except OSError:
                        pass
    return frozenset(entries)


def load_all_sessions() -> list:
    """Return cached sessions, re-parsing from disk only when files have changed."""
    global _session_cache, _session_cache_key
    key = _jsonl_fingerprint()
    if key != _session_cache_key:
        _session_cache = _load_all_sessions()
        _session_cache_key = key
    return _session_cache


DAY_SORTS = {
    "date": lambda d: d.date,
    "cost": lambda d: d.total_cost,
    "bedrock": lambda d: d.bedrock_cost,
    "api": lambda d: d.api_cost,
    "sessions": lambda d: d.session_count,
    "calls": lambda d: d.message_count,
    "input": lambda d: d.total_input_tokens,
    "output": lambda d: d.total_output_tokens,
}

SESSION_SORTS = {
    "session": lambda s: s.title.casefold(),
    "project": lambda s: s.project_name.casefold(),
    "cost": lambda s: s.total_cost,
    "bedrock": lambda s: s.bedrock_cost,
    "api": lambda s: s.api_cost,
    "calls": lambda s: s.message_count,
    "input": lambda s: s.total_input_tokens,
    "output": lambda s: s.total_output_tokens,
    "time": lambda s: s.first_timestamp,
}


def _normalize_sort(
    sort_by: str,
    sort_order: str,
    allowed: dict,
    default_sort: str,
    default_order: str,
) -> tuple[str, str]:
    if sort_by not in allowed:
        sort_by = default_sort
    if sort_order not in {"asc", "desc"}:
        sort_order = default_order
    return sort_by, sort_order


def _sort_items(items: list, sort_by: str, sort_order: str, sorts: dict) -> list:
    return sorted(items, key=sorts[sort_by], reverse=sort_order == "desc")


def _sort_url(
    endpoint: str, column: str, current_sort: str, current_order: str, **values
):
    args = request.args.to_dict(flat=True)
    for key in values:
        args.pop(key, None)
    args["sort"] = column
    args["order"] = (
        "desc" if current_sort == column and current_order == "asc" else "asc"
    )
    return url_for(endpoint, **values, **args)


def _format_cost(v: float) -> str:
    if v < 0:
        return f"-${abs(v):.4f}"
    if v < 0.001:
        return "<$0.001"
    if v < 1.0:
        return f"${v:.4f}"
    return f"${v:.2f}"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if secs == 0:
        return f"{minutes}m"
    return f"{minutes}m {secs}s"


def _cost_severity(cost: float) -> str:
    if cost >= 15.0:
        return "cost-critical"
    if cost >= 5.0:
        return "cost-high"
    if cost >= 1.0:
        return "cost-med"
    return "cost-low"


def _render_markdown(text: str) -> Markup:
    if not text:
        return Markup("")
    html = markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    return Markup(html)


def _action_label(turn) -> str:
    """Derive a short action label for a turn (used in step pills)."""
    # Check tool_calls first (tool results from the preceding user record)
    if turn.tool_calls:
        names = list(dict.fromkeys(tc.name for tc in turn.tool_calls))
        if len(names) == 1:
            return names[0]
        return f"{names[0]} +{len(names) - 1}"
    # Check assistant_content for tool_use blocks
    tool_names = []
    has_text = False
    has_thinking = False
    for block in turn.assistant_content or []:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                tool_names.append(block.get("name", "tool"))
            elif block.get("type") == "text" and block.get("text", "").strip():
                has_text = True
            elif block.get("type") == "thinking":
                has_thinking = True
    if tool_names:
        unique = list(dict.fromkeys(tool_names))
        if len(unique) == 1:
            return unique[0]
        return f"{unique[0]} +{len(unique) - 1}"
    if has_thinking and has_text:
        return "Thinking + Response"
    if has_thinking:
        return "Thinking"
    if has_text:
        return "Response"
    return "API Call"


def _build_exchanges(turns, compaction_events, away_summary_events=None):
    """Group turns into exchanges and interleave compaction/away_summary markers by timestamp.

    Returns a list of dicts, each one of:
      {"type": "exchange", "user_turn": ..., "intermediate_turns": [...], "final_turn": ...}
      {"type": "compaction", "event": CompactionEvent}
      {"type": "away_summary", "event": AwaySummaryEvent}
    """
    raw_exchanges = []
    current = None
    for turn in turns:
        if turn.user_prompt_full or turn.user_prompt:
            if current is not None:
                raw_exchanges.append(current)
            current = {
                "type": "exchange",
                "user_turn": turn,
                "intermediate_turns": [],
                "final_turn": turn,
            }
        else:
            if current is None:
                current = {
                    "type": "exchange",
                    "user_turn": None,
                    "intermediate_turns": [],
                    "final_turn": turn,
                }
            current["intermediate_turns"].append(turn)
            current["final_turn"] = turn
    if current is not None:
        raw_exchanges.append(current)

    marker_items = []
    for ev in compaction_events or []:
        marker_items.append({"type": "compaction", "event": ev})
    for ev in away_summary_events or []:
        marker_items.append({"type": "away_summary", "event": ev})

    if not marker_items:
        return raw_exchanges

    def _ts(item):
        if item.get("type") == "exchange":
            t = item["final_turn"]
            return t.timestamp if t else ""
        return item["event"].timestamp

    merged = raw_exchanges + marker_items
    merged.sort(key=_ts)
    return merged


def _source_label(source: str) -> str:
    return "Bedrock" if source == "bedrock" else "API"


app.jinja_env.filters["markdown"] = _render_markdown

# Hidden in production; set CCX_SHOW_SOURCE_SPLIT=1 locally to surface the
# Bedrock-vs-API breakdown in the UI. The split is still computed either way.
SHOW_SOURCE_SPLIT = os.environ.get("CCX_SHOW_SOURCE_SPLIT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app.jinja_env.globals.update(
    format_cost=_format_cost,
    format_tokens=_format_tokens,
    format_duration=_format_duration,
    cost_severity=_cost_severity,
    action_label=_action_label,
    sort_url=_sort_url,
    source_label=_source_label,
    show_source_split=SHOW_SOURCE_SPLIT,
)


DEFAULT_LOOKBACK_DAYS = 30


def _default_date_range(today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    return (
        (today - timedelta(days=DEFAULT_LOOKBACK_DAYS - 1)).isoformat(),
        today.isoformat(),
    )


@app.route("/")
def day_view():
    if "from" not in request.args and "to" not in request.args:
        from_date, to_date = _default_date_range()
        args = request.args.to_dict(flat=True)
        args["from"] = from_date
        args["to"] = to_date
        return redirect(url_for("day_view", **args))

    from_date = request.args.get("from", "")
    to_date = request.args.get("to", "")
    sort_by, sort_order = _normalize_sort(
        request.args.get("sort", ""),
        request.args.get("order", ""),
        DAY_SORTS,
        "date",
        "desc",
    )
    sessions = load_all_sessions()
    days = build_day_summaries(sessions, from_date=from_date, to_date=to_date)
    days = _sort_items(days, sort_by, sort_order, DAY_SORTS)
    return render_template(
        "days.html",
        days=days,
        from_date=from_date,
        to_date=to_date,
        sort_by=sort_by,
        sort_order=sort_order,
        total_cost=sum(d.total_cost for d in days),
        total_bedrock_cost=sum(d.bedrock_cost for d in days),
        total_api_cost=sum(d.api_cost for d in days),
    )


@app.route("/day/<date>")
def day_sessions_view(date):
    sort_by, sort_order = _normalize_sort(
        request.args.get("sort", ""),
        request.args.get("order", ""),
        SESSION_SORTS,
        "time",
        "desc",
    )
    sessions = load_all_sessions()
    day_sessions = get_sessions_for_date(sessions, date)
    if not day_sessions:
        abort(404)

    day_costs: dict[str, float] = {}
    day_bedrock: dict[str, float] = {}
    day_api: dict[str, float] = {}
    for s in day_sessions:
        dc = db = da = 0.0
        for t in s.turns:
            if t.timestamp and t.timestamp[:10] == date:
                dc += t.cost_usd
                if t.source == "bedrock":
                    db += t.cost_usd
                else:
                    da += t.cost_usd
        day_costs[s.session_id] = dc
        day_bedrock[s.session_id] = db
        day_api[s.session_id] = da

    day_session_sorts = dict(SESSION_SORTS)
    day_session_sorts["cost"] = lambda s: day_costs.get(s.session_id, 0.0)
    day_session_sorts["bedrock"] = lambda s: day_bedrock.get(s.session_id, 0.0)
    day_session_sorts["api"] = lambda s: day_api.get(s.session_id, 0.0)
    day_sessions = _sort_items(day_sessions, sort_by, sort_order, day_session_sorts)

    return render_template(
        "sessions.html",
        date=date,
        sessions=day_sessions,
        sort_by=sort_by,
        sort_order=sort_order,
        day_costs=day_costs,
        day_bedrock=day_bedrock,
        day_api=day_api,
        total_cost=sum(day_costs.values()),
        total_bedrock_cost=sum(day_bedrock.values()),
        total_api_cost=sum(day_api.values()),
    )


@app.route("/session/<session_id>", methods=["GET", "POST"])
def session_detail_view(session_id):
    sessions = load_all_sessions()
    session = get_session_by_id(sessions, session_id)
    if not session:
        abort(404)
    if request.method == "POST":
        try:
            append_custom_session_title(session, request.form.get("title", ""))
        except ValueError as exc:
            abort(400, description=str(exc))
        except OSError:
            abort(500)
        return redirect(url_for("session_detail_view", session_id=session_id))
    exchanges = _build_exchanges(
        session.turns, session.compaction_events, session.away_summary_events
    )
    highlight_date = request.args.get("from_date", "")
    return render_template(
        "session.html",
        session=session,
        exchanges=exchanges,
        highlight_date=highlight_date,
    )


@app.route("/session/<session_id>/turn/<turn_uuid>")
def turn_detail_view(session_id, turn_uuid):
    sessions = load_all_sessions()
    session = get_session_by_id(sessions, session_id)
    if not session:
        abort(404)
    turn = next((t for t in session.turns if t.uuid == turn_uuid), None)
    if not turn:
        abort(404)
    return render_template("turn.html", session=session, turn=turn)


def main():
    import webbrowser
    import threading
    import os
    import argparse
    from waitress import serve

    parser = argparse.ArgumentParser(description="Claude Code Cost Explorer")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args()

    browser_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{browser_host}:{args.port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"  Claude Code Cost Explorer running at {url}")
    print(f"  Listening on {args.host}:{args.port}")
    print("  Press Ctrl+C to quit.")

    serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
