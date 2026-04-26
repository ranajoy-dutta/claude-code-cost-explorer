import os
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
def client(mock_sessions):
    import claude_code_cost_explorer.app as flask_app

    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


class TestDayView:
    def test_200(self, client):
        assert client.get("/").status_code == 200

    def test_contains_favicon(self, client):
        assert b'rel="icon"' in client.get("/").data

    def test_contains_date(self, client):
        assert b"2025-10-25" in client.get("/").data

    def test_from_filter_excludes_older(self, client):
        assert b"2025-10-25" not in client.get("/?from=2025-11-01").data

    def test_to_filter_excludes_newer(self, client):
        assert b"2025-11-01" not in client.get("/?to=2025-10-31").data


class TestDaySessionsView:
    def test_valid_date_200(self, client):
        assert client.get("/day/2025-10-25").status_code == 200

    def test_nonexistent_date_404(self, client):
        assert client.get("/day/2000-01-01").status_code == 404

    def test_session_title_shown(self, client):
        assert b"test-session" in client.get("/day/2025-10-25").data


class TestSessionDetailView:
    def test_valid_session_200(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        assert resp.status_code == 200

    def test_nonexistent_session_404(self, client):
        assert client.get("/session/no-such-id").status_code == 404

    def test_model_name_shown(self, client, mock_sessions):
        resp = client.get(f"/session/{mock_sessions[0].session_id}")
        assert b"sonnet-4-6" in resp.data
