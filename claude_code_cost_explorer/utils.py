import markdown
from markupsafe import Markup
from flask import request, url_for
from .settings import _load_settings


def _sort_items(items: list, sort_by: str, sort_order: str, sorts: dict) -> list:
    return sorted(items, key=sorts[sort_by], reverse=sort_order == "desc")


def _sort_url(
    endpoint: str, column: str, current_sort: str, current_order: str, **values
):
    if not endpoint.startswith("main."):
        endpoint = f"main.{endpoint}"
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
    t = _load_settings()
    if cost >= t["high"]:
        return "cost-critical"
    if cost >= t["medium"]:
        return "cost-high"
    if cost >= t["low"]:
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


def _build_exchanges(
    turns,
    compaction_events,
    away_summary_events=None,
    unlinked_subagents=None,
    session=None,
):
    """Group turns into exchanges and interleave compaction/away_summary/unlinked_subagent/fork markers by timestamp."""
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
    for c in compaction_events or []:
        marker_items.append(
            {"type": "compaction", "event": c, "timestamp": c.timestamp}
        )
    for a in away_summary_events or []:
        marker_items.append(
            {"type": "away_summary", "event": a, "timestamp": a.timestamp}
        )
    for s in unlinked_subagents or []:
        marker_items.append(
            {"type": "unlinked_subagent", "subagent": s, "timestamp": s.timestamp}
        )
    if (
        session
        and session.fork_parent_id
        and 0 < session.fork_point_turn_index < len(turns)
    ):
        fork_turn = turns[session.fork_point_turn_index]
        marker_items.append(
            {
                "type": "fork_point",
                "timestamp": fork_turn.timestamp,
                "parent_id": session.fork_parent_id,
                "parent_title": session.fork_parent_title,
                "shared_turns": session.fork_point_turn_index,
                "new_turns": session.fork_new_turns_count,
                "inc_cost": session.fork_incremental_cost,
            }
        )
    marker_items.sort(key=lambda x: x["timestamp"])

    final_list = []
    marker_idx = 0
    for ex in raw_exchanges:
        ex_ts = None
        if ex["user_turn"]:
            ex_ts = ex["user_turn"].timestamp
        elif ex["intermediate_turns"]:
            ex_ts = ex["intermediate_turns"][0].timestamp
        elif ex["final_turn"]:
            ex_ts = ex["final_turn"].timestamp

        while marker_idx < len(marker_items) and (
            ex_ts is None or marker_items[marker_idx]["timestamp"] <= ex_ts
        ):
            final_list.append(marker_items[marker_idx])
            marker_idx += 1
        final_list.append(ex)

    while marker_idx < len(marker_items):
        final_list.append(marker_items[marker_idx])
        marker_idx += 1

    return final_list


def register_filters(app):
    import os

    SHOW_SOURCE_SPLIT = os.environ.get("CCX_SHOW_SOURCE_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _source_label(source: str) -> str:
        return "Bedrock" if source == "bedrock" else "API"

    app.jinja_env.filters["markdown"] = _render_markdown
    app.jinja_env.filters["nl2br"] = lambda text: (
        text.replace("\n", "<br>\n") if text else ""
    )
    app.jinja_env.filters["is_error"] = lambda text: (
        text and ("error" in text.lower() or "exception" in text.lower())
    )

    app.jinja_env.globals.update(
        format_cost=_format_cost,
        format_tokens=_format_tokens,
        format_duration=_format_duration,
        sort_url=_sort_url,
        cost_severity=_cost_severity,
        get_cost_thresholds=_load_settings,
        action_label=_action_label,
        source_label=_source_label,
        show_source_split=SHOW_SOURCE_SPLIT,
    )
