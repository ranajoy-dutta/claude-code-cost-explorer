import os
import json
from datetime import date
from flask import Blueprint, render_template, request, abort, redirect, url_for, jsonify

from .reader import (
    load_all_sessions as _load_all_sessions,
    build_day_summaries,
    get_sessions_for_date,
    get_session_by_id,
    CLAUDE_DIR,
    append_custom_session_title,
)
from .utils import _sort_items, _build_exchanges
from .settings import _save_settings, _SETTINGS_PATH, _DEFAULT_THRESHOLDS

bp = Blueprint("main", __name__)

_session_cache: list | None = None
_session_cache_key: frozenset | None = None


def _jsonl_fingerprint(claude_dir: str = CLAUDE_DIR) -> frozenset:
    entries = []
    projects_dir = os.path.join(claude_dir, "projects")
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
    for folder in ("jobs", "tasks"):
        folder_dir = os.path.join(claude_dir, folder)
        if not os.path.isdir(folder_dir):
            continue
        for root, _, files in os.walk(folder_dir):
            for fname in files:
                if fname.endswith(".jsonl"):
                    full = os.path.join(root, fname)
                    try:
                        entries.append((full, os.stat(full).st_mtime_ns))
                    except OSError:
                        pass
    return frozenset(entries)


def load_all_sessions() -> list:
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


def _default_date_range(today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    return (
        today.replace(day=1).isoformat(),
        today.isoformat(),
    )


@bp.route("/")
def day_view():
    if "from" not in request.args and "to" not in request.args:
        from_date, to_date = _default_date_range()
        args = request.args.to_dict(flat=True)
        args["from"] = from_date
        args["to"] = to_date
        return redirect(url_for("main.day_view", **args))

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
        has_unknown_models=any(s.has_unknown_models for s in sessions),
    )


@bp.route("/day/<date>")
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
        for sa in s.unlinked_subagents:
            for sat in sa.turns:
                if sat.cost_usd == 0.0:
                    continue
                if sat.timestamp and sat.timestamp[:10] == date:
                    dc += sat.cost_usd
                    if sat.source == "bedrock":
                        db += sat.cost_usd
                    else:
                        da += sat.cost_usd
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
        has_unknown_models=any(s.has_unknown_models for s in day_sessions),
    )


@bp.route("/session/<session_id>", methods=["GET", "POST"])
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
        return redirect(url_for("main.session_detail_view", session_id=session_id))
    exchanges = _build_exchanges(
        session.turns,
        session.compaction_events,
        session.away_summary_events,
        session.unlinked_subagents,
        session=session,
    )
    highlight_date = request.args.get("from_date", "")
    return render_template(
        "session.html",
        session=session,
        exchanges=exchanges,
        highlight_date=highlight_date,
    )


@bp.route("/settings", methods=["POST"])
def settings_view():
    data = request.get_json(silent=True) or {}
    try:
        low = float(data["low"])
        medium = float(data["medium"])
        high = float(data["high"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "low, medium, high must be numbers"}), 400
    if not (low > 0 and medium > 0 and high > 0):
        return jsonify({"error": "All thresholds must be positive"}), 400
    if not (low < medium < high):
        return jsonify({"error": "Thresholds must satisfy: low < medium < high"}), 400
    _save_settings(low, medium, high)
    return jsonify({"ok": True})


@bp.route("/raw_settings", methods=["GET", "POST"])
def raw_settings_view():
    if request.method == "GET":
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                raw_json = f.read()
            return jsonify({"raw_json": raw_json})
        except Exception:
            return jsonify(
                {
                    "raw_json": '{\n  "cost_thresholds": {\n    "low": 1.0,\n    "medium": 5.0,\n    "high": 15.0\n  }\n}'
                }
            )

    req_data = request.get_json(silent=True) or {}
    raw_str = req_data.get("raw_json", "")
    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # Validate structure
    if not isinstance(data, dict):
        return jsonify({"error": "Settings must be a JSON object"}), 400

    if "cost_thresholds" in data:
        t = data["cost_thresholds"]
        if not isinstance(t, dict):
            return jsonify({"error": "cost_thresholds must be an object"}), 400
        try:
            low = float(t.get("low", 0))
            med = float(t.get("medium", 0))
            high = float(t.get("high", 0))
            if not (low > 0 and med > 0 and high > 0 and low < med < high):
                return jsonify({"error": "Invalid thresholds"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Thresholds must be numeric"}), 400
    else:
        # Automatically inject defaults if removed by user
        data["cost_thresholds"] = dict(_DEFAULT_THRESHOLDS)

    if "bedrock_rates" in data:
        r = data["bedrock_rates"]
        if not isinstance(r, dict):
            return jsonify({"error": "bedrock_rates must be an object"}), 400
        for model, rates in r.items():
            if not isinstance(rates, dict):
                return jsonify({"error": f"Rates for {model} must be an object"}), 400
            for key in ["input", "output", "cache_write", "cache_read"]:
                if key in rates and not isinstance(rates[key], (int, float)):
                    return jsonify(
                        {"error": f"Rate '{key}' for {model} must be a number"}
                    ), 400

    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return jsonify({"ok": True})


@bp.route("/session/<session_id>/turn/<turn_uuid>")
def turn_detail_view(session_id, turn_uuid):
    sessions = load_all_sessions()
    session = get_session_by_id(sessions, session_id)
    if not session:
        abort(404)
    turn = next((t for t in session.turns if t.uuid == turn_uuid), None)
    if not turn:
        abort(404)
    return render_template("turn.html", session=session, turn=turn)
