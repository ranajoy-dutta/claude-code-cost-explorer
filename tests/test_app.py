import os
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

    monkeypatch.setattr(flask_app, "load_all_sessions", lambda: mock_sessions)
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


class TestDayView:
    def test_200(self, client):
        assert client.get("/?from=2025-10-01").status_code == 200

    def test_redirects_to_default_month_filter(self, client, monkeypatch):
        import claude_code_cost_explorer.app as flask_app

        monkeypatch.setattr(
            flask_app,
            "_default_date_range",
            lambda: ("2025-10-02", "2025-11-01"),
        )

        resp = client.get("/")

        assert resp.status_code == 302
        assert resp.headers["Location"] == "/?from=2025-10-02&to=2025-11-01"

    def test_default_month_filter_preserves_sort_params(self, client, monkeypatch):
        import claude_code_cost_explorer.app as flask_app

        monkeypatch.setattr(
            flask_app,
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
        import claude_code_cost_explorer.app as flask_app

        monkeypatch.setattr(
            flask_app,
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
