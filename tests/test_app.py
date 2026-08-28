import os
from datetime import date
from dataclasses import replace

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture()
def mock_sessions(monkeypatch):
    from claude_code_cost_explorer.reader import parse_session_file
    import claude_code_cost_explorer.reader as reader

    sessions = [
        parse_session_file(os.path.join(FIXTURES, f), "-tmp")
        for f in ["session_simple.jsonl", "session_with_title.jsonl"]
    ]
    sessions = [s for s in sessions if s]
    monkeypatch.setattr(reader, "load_all_sessions", lambda: sessions)
    return sessions


@pytest.fixture()
def client(mock_sessions, monkeypatch):
    import claude_code_cost_explorer.app as flask_app
    import claude_code_cost_explorer.routes as routes_mod

    monkeypatch.setattr(routes_mod, "load_all_sessions", lambda: mock_sessions)
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


class TestDayView:
    def test_200(self, client):
        assert client.get("/?from=2025-10-01").status_code == 200

    def test_default_month_filter_matches_current_month_preset(self):
        from claude_code_cost_explorer.routes import _default_date_range

        assert _default_date_range(date(2025, 11, 30)) == (
            "2025-11-01",
            "2025-11-30",
        )
        assert _default_date_range(date(2026, 8, 20)) == (
            "2026-08-01",
            "2026-08-20",
        )

    def test_redirects_to_default_month_filter(self, client, monkeypatch):
        import claude_code_cost_explorer.routes as routes_mod

        monkeypatch.setattr(
            routes_mod,
            "_default_date_range",
            lambda: ("2025-10-02", "2025-11-01"),
        )

        resp = client.get("/")

        assert resp.status_code == 302
        assert resp.headers["Location"] == "/?from=2025-10-02&to=2025-11-01"

    def test_default_month_filter_preserves_sort_params(self, client, monkeypatch):
        import claude_code_cost_explorer.routes as routes_mod

        monkeypatch.setattr(
            routes_mod,
            "_default_date_range",
            lambda: ("2025-10-02", "2025-11-01"),
        )

        resp = client.get("/?sort=cost&order=asc")

        assert resp.status_code == 302
        assert (
            resp.headers["Location"]
            == "/?sort=cost&order=asc&from=2025-10-02&to=2025-11-01"
        )

    def test_default_month_filter_limits_landing_data(self, client, monkeypatch):
        import claude_code_cost_explorer.routes as routes_mod

        monkeypatch.setattr(
            routes_mod,
            "_default_date_range",
            lambda: ("2025-10-26", "2025-11-25"),
        )

        data = client.get("/", follow_redirects=True).data

        assert b"2025-11-01" in data
        assert b"2025-10-25" not in data

    def test_contains_favicon(self, client):
        assert b'rel="icon"' in client.get("/?from=2025-10-01").data

    def test_contains_date(self, client):
        assert b"2025-10-25" in client.get("/?from=2025-10-01").data

    def test_uses_single_visible_date_range_filter(self, client):
        data = client.get("/?from=2025-10-01&to=2025-11-30").data
        assert b'id="date-range"' in data
        assert b'name="from"' in data
        assert b'name="to"' in data
        assert b'<label for="from">From</label>' not in data
        assert b'<label for="to">To</label>' not in data

    def test_from_filter_excludes_older(self, client):
        assert b"2025-10-25" not in client.get("/?from=2025-11-01").data

    def test_to_filter_excludes_newer(self, client):
        assert b"2025-11-01" not in client.get("/?to=2025-10-31").data

    def test_sort_date_ascending(self, client):
        data = client.get("/?from=2025-10-01&sort=date&order=asc").data
        assert data.find(b"2025-10-25") < data.find(b"2025-11-01")

    def test_sort_links_preserve_filters(self, client):
        data = client.get("/?from=2025-10-01&to=2025-11-30").data
        assert (
            b"/?from=2025-10-01&amp;to=2025-11-30&amp;sort=cost&amp;order=asc" in data
        )


class TestDaySessionsView:
    def test_valid_date_200(self, client):
        assert client.get("/day/2025-10-25").status_code == 200

    def test_nonexistent_date_404(self, client):
        assert client.get("/day/2000-01-01").status_code == 404

    def test_session_title_shown(self, client):
        assert b"test-session" in client.get("/day/2025-10-25").data

    def test_sort_session_title_ascending(self, client, mock_sessions):
        mock_sessions.append(
            replace(
                mock_sessions[0],
                session_id="sess-aaa",
                title="AAA cost check",
                first_timestamp="2025-10-25T09:00:00.000Z",
                last_timestamp="2025-10-25T09:01:00.000Z",
            )
        )

        data = client.get("/day/2025-10-25?sort=session&order=asc").data

        assert data.find(b"AAA cost check") < data.find(b"test-session")

    def test_session_sort_link_toggles_active_column(self, client):
        data = client.get("/day/2025-10-25?sort=time&order=asc").data
        assert b"/day/2025-10-25?sort=time&amp;order=desc" in data


class TestSessionDetailView:
    def test_valid_session_200(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        assert resp.status_code == 200

    def test_nonexistent_session_404(self, client):
        assert client.get("/session/no-such-id").status_code == 404

    def test_model_name_shown(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        assert b"sonnet-4-6" in resp.data

    def test_session_id_chip_copies_full_id(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        html = resp.get_data(as_text=True)

        assert f'data-session-id="{mock_sessions[0].session_id}"' in html
        assert mock_sessions[0].session_id[:8] in html
        assert "copySessionId(this)" in html
        assert "Copied!" in html

    def test_session_title_edit_ui_shown(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        html = resp.get_data(as_text=True)

        assert f'action="/session/{mock_sessions[0].session_id}"' in html
        assert 'name="title"' in html
        assert "openSessionTitleEditor()" in html

    def test_session_title_update_redirects(self, client, mock_sessions, monkeypatch):
        calls = []

        def fake_append_title(session, title):
            calls.append((session.session_id, title))
            return title

        monkeypatch.setattr(
            "claude_code_cost_explorer.routes.append_custom_session_title",
            fake_append_title,
        )
        resp = client.post(
            f"/session/{mock_sessions[0].session_id}",
            data={"title": "Renamed session"},
        )

        assert resp.status_code == 302
        assert resp.headers["Location"] == f"/session/{mock_sessions[0].session_id}"
        assert calls == [(mock_sessions[0].session_id, "Renamed session")]


class TestBuildExchanges:
    def test_no_compaction_events(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="hi",
            ),
            Turn(
                uuid="t2",
                timestamp="2025-10-25T10:01:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="",
            ),
        ]
        result = _build_exchanges(turns, [])
        assert len(result) == 1
        assert result[0]["type"] == "exchange"

    def test_compaction_inserted_between_exchanges(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn, CompactionEvent

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="first",
            ),
            Turn(
                uuid="t2",
                timestamp="2025-10-25T10:03:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="second",
            ),
        ]
        events = [
            CompactionEvent(
                timestamp="2025-10-25T10:02:00.000Z",
                trigger="manual",
                pre_tokens=50000,
                post_tokens=4000,
                duration_ms=30000,
            )
        ]
        result = _build_exchanges(turns, events)
        assert len(result) == 3
        assert result[0]["type"] == "exchange"
        assert result[1]["type"] == "compaction"
        assert result[1]["event"].trigger == "manual"
        assert result[2]["type"] == "exchange"

    def test_compaction_before_all_exchanges(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn, CompactionEvent

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:05:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="only",
            ),
        ]
        events = [
            CompactionEvent(
                timestamp="2025-10-25T10:01:00.000Z",
                trigger="auto",
                pre_tokens=1000,
                post_tokens=100,
                duration_ms=5000,
            )
        ]
        result = _build_exchanges(turns, events)
        assert len(result) == 2
        assert result[0]["type"] == "compaction"
        assert result[1]["type"] == "exchange"

    def test_away_summary_inserted_between_exchanges(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn, AwaySummaryEvent

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="first",
            ),
            Turn(
                uuid="t2",
                timestamp="2025-10-25T10:05:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="second",
            ),
        ]
        away_events = [
            AwaySummaryEvent(
                timestamp="2025-10-25T10:02:00.000Z", content="We were working on X."
            )
        ]
        result = _build_exchanges(turns, [], away_events)
        assert len(result) == 3
        assert result[0]["type"] == "exchange"
        assert result[1]["type"] == "away_summary"
        assert result[1]["event"].content == "We were working on X."
        assert result[2]["type"] == "exchange"

    def test_away_summary_and_compaction_sorted_together(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import (
            Turn,
            CompactionEvent,
            AwaySummaryEvent,
        )

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="msg",
            ),
            Turn(
                uuid="t2",
                timestamp="2025-10-25T10:10:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="msg2",
            ),
        ]
        compaction = [
            CompactionEvent(
                timestamp="2025-10-25T10:06:00.000Z",
                trigger="auto",
                pre_tokens=1000,
                post_tokens=100,
                duration_ms=5000,
            )
        ]
        away = [
            AwaySummaryEvent(timestamp="2025-10-25T10:03:00.000Z", content="recap text")
        ]
        result = _build_exchanges(turns, compaction, away)
        assert len(result) == 4
        types = [r["type"] for r in result]
        assert types == ["exchange", "away_summary", "compaction", "exchange"]

    def test_away_summary_none_treated_as_empty(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="hi",
            ),
        ]
        result = _build_exchanges(turns, [], None)
        assert len(result) == 1
        assert result[0]["type"] == "exchange"

    def test_unlinked_subagent_inserted_by_timestamp(self):
        from claude_code_cost_explorer.utils import _build_exchanges
        from claude_code_cost_explorer.reader import Turn, SubagentData

        turns = [
            Turn(
                uuid="t1",
                timestamp="2025-10-25T10:00:00.000Z",
                model="m",
                usage={},
                cost_usd=0.0,
                user_prompt="hi",
            ),
        ]
        sa = SubagentData(
            agent_id="a1",
            description="desc",
            agent_type="workflow",
            turns=[],
            timestamp="2025-10-25T10:02:00.000Z",
        )
        result = _build_exchanges(turns, [], None, [sa])
        assert len(result) == 2
        assert result[0]["type"] == "exchange"
        assert result[1]["type"] == "unlinked_subagent"
        assert result[1]["subagent"] is sa


class TestSettings:
    def test_load_settings_returns_defaults_when_missing(self, tmp_path, monkeypatch):
        import claude_code_cost_explorer.settings as app_mod

        monkeypatch.setattr(
            app_mod, "_SETTINGS_PATH", str(tmp_path / "ccx_settings.json")
        )
        monkeypatch.setattr(app_mod, "_settings_cache", None)
        monkeypatch.setattr(app_mod, "_settings_cache_mtime", None)
        settings = app_mod._load_settings()
        assert settings == {"low": 1.0, "medium": 5.0, "high": 15.0}

    def test_load_settings_reads_file(self, tmp_path, monkeypatch):
        import claude_code_cost_explorer.settings as app_mod

        settings_file = tmp_path / "ccx_settings.json"
        settings_file.write_text(
            '{"cost_thresholds": {"low": 2.0, "medium": 8.0, "high": 20.0}}'
        )
        monkeypatch.setattr(app_mod, "_SETTINGS_PATH", str(settings_file))
        monkeypatch.setattr(app_mod, "_settings_cache", None)
        monkeypatch.setattr(app_mod, "_settings_cache_mtime", None)
        settings = app_mod._load_settings()
        assert settings == {"low": 2.0, "medium": 8.0, "high": 20.0}

    def test_load_settings_ignores_malformed_json(self, tmp_path, monkeypatch):
        import claude_code_cost_explorer.settings as app_mod

        settings_file = tmp_path / "ccx_settings.json"
        settings_file.write_text("not json")
        monkeypatch.setattr(app_mod, "_SETTINGS_PATH", str(settings_file))
        monkeypatch.setattr(app_mod, "_settings_cache", None)
        monkeypatch.setattr(app_mod, "_settings_cache_mtime", None)
        settings = app_mod._load_settings()
        assert settings == {"low": 1.0, "medium": 5.0, "high": 15.0}

    def test_cost_severity_uses_custom_thresholds(self, monkeypatch):
        import claude_code_cost_explorer.utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "_load_settings",
            lambda: {"low": 2.0, "medium": 10.0, "high": 25.0},
        )
        assert utils_mod._cost_severity(1.5) == "cost-low"
        assert utils_mod._cost_severity(5.0) == "cost-med"
        assert utils_mod._cost_severity(15.0) == "cost-high"
        assert utils_mod._cost_severity(30.0) == "cost-critical"

    def test_post_settings_saves_valid_thresholds(self, client, tmp_path, monkeypatch):
        import json
        import claude_code_cost_explorer.settings as app_mod

        monkeypatch.setattr(
            app_mod, "_SETTINGS_PATH", str(tmp_path / "ccx_settings.json")
        )
        monkeypatch.setattr(app_mod, "_settings_cache", None)
        monkeypatch.setattr(app_mod, "_settings_cache_mtime", None)
        resp = client.post("/settings", json={"low": 2.0, "medium": 8.0, "high": 20.0})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        saved = json.loads((tmp_path / "ccx_settings.json").read_text())
        assert saved["cost_thresholds"] == {"low": 2.0, "medium": 8.0, "high": 20.0}

    def test_post_settings_rejects_wrong_order(self, client, monkeypatch, tmp_path):
        import claude_code_cost_explorer.settings as app_mod

        monkeypatch.setattr(
            app_mod, "_SETTINGS_PATH", str(tmp_path / "ccx_settings.json")
        )
        resp = client.post("/settings", json={"low": 10.0, "medium": 5.0, "high": 20.0})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_post_settings_rejects_non_positive(self, client, monkeypatch, tmp_path):
        import claude_code_cost_explorer.settings as app_mod

        monkeypatch.setattr(
            app_mod, "_SETTINGS_PATH", str(tmp_path / "ccx_settings.json")
        )
        resp = client.post("/settings", json={"low": 0.0, "medium": 5.0, "high": 15.0})
        assert resp.status_code == 400
