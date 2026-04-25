"""Claude Code Cost Tracker. Run: flask --app app run --port 5050"""

from flask import Flask, render_template, request, abort
from claude_code_cost_explorer.reader import (
    load_all_sessions,
    build_day_summaries,
    get_sessions_for_date,
    get_session_by_id,
)

import pathlib
import markdown
from markupsafe import Markup

app = Flask(__name__, template_folder=str(pathlib.Path(__file__).parent / "templates"))


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


app.jinja_env.filters["markdown"] = _render_markdown

app.jinja_env.globals.update(
    format_cost=_format_cost,
    format_tokens=_format_tokens,
    format_duration=_format_duration,
    cost_severity=_cost_severity,
)


@app.route("/")
def day_view():
    from_date = request.args.get("from", "")
    to_date = request.args.get("to", "")
    sessions = load_all_sessions()
    days = build_day_summaries(sessions, from_date=from_date, to_date=to_date)
    return render_template(
        "days.html",
        days=days,
        from_date=from_date,
        to_date=to_date,
        total_cost=sum(d.total_cost for d in days),
    )


@app.route("/day/<date>")
def day_sessions_view(date):
    sessions = load_all_sessions()
    day_sessions = get_sessions_for_date(sessions, date)
    if not day_sessions:
        abort(404)
    return render_template(
        "sessions.html",
        date=date,
        sessions=day_sessions,
        total_cost=sum(s.total_cost for s in day_sessions),
    )


@app.route("/session/<session_id>")
def session_detail_view(session_id):
    sessions = load_all_sessions()
    session = get_session_by_id(sessions, session_id)
    if not session:
        abort(404)
    return render_template("session.html", session=session)


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

    parser = argparse.ArgumentParser(description="Claude Code Cost Explorer")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)))
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"  Claude Code Cost Explorer running at {url}")
    print("  Press Ctrl+C to quit.")

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
