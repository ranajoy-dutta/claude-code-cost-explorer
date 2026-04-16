import os
from reader import (
    parse_session_file,
    build_day_summaries,
    get_session_by_id,
    _extract_user_prompt,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestExtractUserPrompt:
    def test_list_content(self):
        assert _extract_user_prompt([{"type": "text", "text": "hello"}]) == "hello"

    def test_string_content(self):
        assert _extract_user_prompt("direct") == "direct"

    def test_skips_interrupted(self):
        assert (
            _extract_user_prompt([{"type": "text", "text": "[Request interrupted]"}])
            == ""
        )

    def test_truncates_long(self):
        assert len(_extract_user_prompt([{"type": "text", "text": "x" * 200}])) == 120

    def test_empty(self):
        assert _extract_user_prompt([]) == ""


class TestParseSessionFile:
    def test_turn_count_deduplication(self):
        """asst-2 shares parentUuid=asst-1 → non-root, not counted."""
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.message_count == 2

    def test_cost_correct(self):
        from cost import calculate_cost

        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        t1 = calculate_cost(
            "claude-sonnet-4-6",
            {
                "input_tokens": 10,
                "output_tokens": 50,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 0,
            },
        )
        t2 = calculate_cost(
            "claude-sonnet-4-6",
            {
                "input_tokens": 5,
                "output_tokens": 30,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 100,
            },
        )
        assert abs(s.total_cost - (t1 + t2)) < 1e-9

    def test_cwd_overrides_project_path(self):
        s = parse_session_file(
            os.path.join(FIXTURES, "session_simple.jsonl"), "/encoded"
        )
        assert s.project_path == "/Users/test/my-project"
        assert s.project_name == "my-project"

    def test_slug_used_as_title(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.title == "test-session"

    def test_custom_title_priority(self):
        s = parse_session_file(
            os.path.join(FIXTURES, "session_with_title.jsonl"), "/tmp"
        )
        assert s.title == "Deploy pipeline fix"

    def test_date_from_timestamp(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.date == "2025-10-25"

    def test_empty_session_returns_none(self):
        assert (
            parse_session_file(os.path.join(FIXTURES, "session_empty.jsonl"), "/tmp")
            is None
        )

    def test_nonexistent_file_returns_none(self):
        assert parse_session_file("/tmp/nonexistent-99.jsonl", "/tmp") is None

    def test_user_prompt_associated_with_turn(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.turns[0].user_prompt == "Hello, please help me write a function"
        assert s.turns[1].user_prompt == "Can you add type hints?"


class TestBuildDaySummaries:
    def _sessions(self):
        p1 = os.path.join(FIXTURES, "session_simple.jsonl")
        p2 = os.path.join(FIXTURES, "session_with_title.jsonl")
        return [
            s
            for s in [parse_session_file(p1, "/tmp"), parse_session_file(p2, "/tmp")]
            if s
        ]

    def test_sorted_descending(self):
        days = build_day_summaries(self._sessions())
        dates = [d.date for d in days]
        assert dates == sorted(dates, reverse=True)

    def test_two_distinct_dates(self):
        days = build_day_summaries(self._sessions())
        dates = {d.date for d in days}
        assert "2025-10-25" in dates and "2025-11-01" in dates

    def test_from_filter(self):
        days = build_day_summaries(self._sessions(), from_date="2025-11-01")
        assert all(d.date >= "2025-11-01" for d in days)

    def test_to_filter(self):
        days = build_day_summaries(self._sessions(), to_date="2025-10-31")
        assert all(d.date <= "2025-10-31" for d in days)

    def test_empty_input(self):
        assert build_day_summaries([]) == []


class TestGetSessionById:
    def test_found(self):
        p = os.path.join(FIXTURES, "session_simple.jsonl")
        s = parse_session_file(p, "/tmp")
        assert get_session_by_id([s], s.session_id) is s

    def test_not_found(self):
        p = os.path.join(FIXTURES, "session_simple.jsonl")
        s = parse_session_file(p, "/tmp")
        assert get_session_by_id([s], "missing") is None
