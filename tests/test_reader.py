import os
import shutil
from claude_code_cost_explorer.reader import (
    append_custom_session_title,
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
        from claude_code_cost_explorer.cost import calculate_cost

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

    def test_latest_custom_title_wins(self, tmp_path):
        source = os.path.join(FIXTURES, "session_with_title.jsonl")
        target = tmp_path / "session_with_title.jsonl"
        shutil.copyfile(source, target)
        with open(target, "a", encoding="utf-8") as f:
            f.write(
                '{"type":"custom-title","sessionId":"session_with_title",'
                '"customTitle":"Latest title"}\n'
            )

        s = parse_session_file(str(target), "/tmp")
        assert s.title == "Latest title"

    def test_source_path_stored(self):
        path = os.path.join(FIXTURES, "session_simple.jsonl")
        s = parse_session_file(path, "/tmp")
        assert s.source_path == path

    def test_append_custom_session_title_preserves_metrics(self, tmp_path):
        source = os.path.join(FIXTURES, "session_simple.jsonl")
        target = tmp_path / "session_simple.jsonl"
        shutil.copyfile(source, target)
        before = parse_session_file(str(target), "/tmp")

        written_title = append_custom_session_title(before, "  Renamed session  ")
        after = parse_session_file(str(target), "/tmp")

        assert written_title == "Renamed session"
        assert after.title == "Renamed session"
        assert after.session_id == before.session_id
        assert after.message_count == before.message_count
        assert after.total_cost == before.total_cost

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


class TestAwaySummaryEvents:
    def test_away_summary_parsed(self, tmp_path):
        from claude_code_cost_explorer.reader import AwaySummaryEvent

        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-10-25T10:00:00.000Z","sessionId":"s1","cwd":"/tmp","slug":"s"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-10-25T10:01:00.000Z","sessionId":"s1"}\n'
            '{"type":"system","subtype":"away_summary","content":"We were fixing a bug.","timestamp":"2025-10-25T10:02:00.000Z","uuid":"as1","sessionId":"s1","isSidechain":false,"isMeta":false,"parentUuid":"a1"}\n'
        )
        s = parse_session_file(str(jsonl), "/tmp")
        assert len(s.away_summary_events) == 1
        ev = s.away_summary_events[0]
        assert isinstance(ev, AwaySummaryEvent)
        assert ev.content == "We were fixing a bug."
        assert ev.timestamp == "2025-10-25T10:02:00.000Z"

    def test_multiple_away_summaries(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-10-25T10:00:00.000Z","sessionId":"s1","cwd":"/tmp","slug":"s"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-10-25T10:01:00.000Z","sessionId":"s1"}\n'
            '{"type":"system","subtype":"away_summary","content":"First recap.","timestamp":"2025-10-25T10:02:00.000Z","uuid":"as1","sessionId":"s1","isSidechain":false,"isMeta":false,"parentUuid":"a1"}\n'
            '{"type":"system","subtype":"away_summary","content":"Second recap.","timestamp":"2025-10-25T10:05:00.000Z","uuid":"as2","sessionId":"s1","isSidechain":false,"isMeta":false,"parentUuid":"a1"}\n'
        )
        s = parse_session_file(str(jsonl), "/tmp")
        assert len(s.away_summary_events) == 2
        assert s.away_summary_events[0].content == "First recap."
        assert s.away_summary_events[1].content == "Second recap."

    def test_empty_away_summary_skipped(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-10-25T10:00:00.000Z","sessionId":"s1","cwd":"/tmp","slug":"s"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-10-25T10:01:00.000Z","sessionId":"s1"}\n'
            '{"type":"system","subtype":"away_summary","content":"","timestamp":"2025-10-25T10:02:00.000Z","uuid":"as1","sessionId":"s1"}\n'
        )
        s = parse_session_file(str(jsonl), "/tmp")
        assert s.away_summary_events == []

    def test_no_away_summaries_on_plain_session(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.away_summary_events == []


class TestAiTitleEvent:
    def test_ai_title_event_stored(self, tmp_path):
        from claude_code_cost_explorer.reader import AiTitleEvent

        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text(
            '{"type":"ai-title","aiTitle":"Investigate deployment pipeline","sessionId":"s1"}\n'
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"fix deploy"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-11-01T09:00:00.000Z","sessionId":"s1","cwd":"/tmp"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-11-01T09:00:05.000Z","sessionId":"s1"}\n'
        )
        s = parse_session_file(str(jsonl), "/tmp")
        assert s.ai_title_event is not None
        assert isinstance(s.ai_title_event, AiTitleEvent)
        assert s.ai_title_event.ai_title == "Investigate deployment pipeline"

    def test_no_ai_title_event_when_absent(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.ai_title_event is None

    def test_custom_title_takes_priority_over_ai_title(self):
        # session_with_title.jsonl has both ai-title and custom-title
        s = parse_session_file(
            os.path.join(FIXTURES, "session_with_title.jsonl"), "/tmp"
        )
        assert s.title == "Deploy pipeline fix"
        # but ai_title_event is still stored
        assert s.ai_title_event is not None
        assert s.ai_title_event.ai_title == "Investigate deployment pipeline"


class TestCompactionEvents:
    def test_compact_boundary_parsed(self):
        from claude_code_cost_explorer.reader import CompactionEvent

        s = parse_session_file(
            os.path.join(FIXTURES, "session_with_compaction.jsonl"), "/tmp"
        )
        assert len(s.compaction_events) == 1
        ev = s.compaction_events[0]
        assert isinstance(ev, CompactionEvent)
        assert ev.trigger == "manual"
        assert ev.pre_tokens == 50000
        assert ev.post_tokens == 4000
        assert ev.duration_ms == 30000
        assert ev.timestamp == "2025-10-25T10:02:00.000Z"

    def test_no_compaction_events_on_plain_session(self):
        s = parse_session_file(os.path.join(FIXTURES, "session_simple.jsonl"), "/tmp")
        assert s.compaction_events == []

    def test_compact_boundary_missing_metadata(self, tmp_path):
        jsonl = tmp_path / "session_no_meta.jsonl"
        jsonl.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-10-25T10:00:00.000Z","sessionId":"s1","cwd":"/tmp","version":"1.0","slug":"s"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-10-25T10:01:00.000Z","sessionId":"s1"}\n'
            '{"type":"system","subtype":"compact_boundary","timestamp":"2025-10-25T10:02:00.000Z","uuid":"c1","sessionId":"s1"}\n'
        )
        from claude_code_cost_explorer.reader import CompactionEvent

        s = parse_session_file(str(jsonl), "/tmp")
        assert len(s.compaction_events) == 1
        ev = s.compaction_events[0]
        assert isinstance(ev, CompactionEvent)
        assert ev.trigger == ""
        assert ev.pre_tokens == 0
        assert ev.post_tokens == 0
        assert ev.duration_ms == 0

    def test_multiple_compaction_events(self, tmp_path):
        jsonl = tmp_path / "session_multi_compact.jsonl"
        jsonl.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"msg1"}]},"uuid":"u1","parentUuid":null,"timestamp":"2025-10-25T10:00:00.000Z","sessionId":"s2","cwd":"/tmp","version":"1.0","slug":"s2"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"r1"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2025-10-25T10:01:00.000Z","sessionId":"s2"}\n'
            '{"type":"system","subtype":"compact_boundary","timestamp":"2025-10-25T10:02:00.000Z","uuid":"c1","compactMetadata":{"trigger":"auto","preTokens":10000,"postTokens":1000,"durationMs":5000},"sessionId":"s2"}\n'
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"msg2"}]},"uuid":"u2","parentUuid":null,"timestamp":"2025-10-25T10:03:00.000Z","sessionId":"s2"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"r2"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a2","parentUuid":"u2","timestamp":"2025-10-25T10:04:00.000Z","sessionId":"s2"}\n'
            '{"type":"system","subtype":"compact_boundary","timestamp":"2025-10-25T10:05:00.000Z","uuid":"c2","compactMetadata":{"trigger":"manual","preTokens":20000,"postTokens":2000,"durationMs":10000},"sessionId":"s2"}\n'
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"msg3"}]},"uuid":"u3","parentUuid":null,"timestamp":"2025-10-25T10:06:00.000Z","sessionId":"s2"}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"r3"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":5,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a3","parentUuid":"u3","timestamp":"2025-10-25T10:07:00.000Z","sessionId":"s2"}\n'
        )
        s = parse_session_file(str(jsonl), "/tmp")
        assert len(s.compaction_events) == 2
        assert s.compaction_events[0].trigger == "auto"
        assert s.compaction_events[1].trigger == "manual"
        assert s.message_count == 3


def test_unlinked_subagent_cost_in_daily_summary():
    from claude_code_cost_explorer.reader import SessionData, SubagentData, SubagentTurn
    from claude_code_cost_explorer.reader import build_day_summaries

    sa_turn = SubagentTurn(
        uuid="sa-t1",
        model="claude-sonnet-4-6",
        tool_uses=[],
        tool_results=[],
        text_blocks=[],
        timestamp="2026-06-02T10:00:00.000Z",
        cost_usd=2.5,
        source="bedrock",
        usage={"input_tokens": 100, "output_tokens": 100},
    )
    sa = SubagentData(
        agent_id="agent-1",
        description="desc",
        agent_type="workflow",
        turns=[sa_turn],
        total_tool_uses=0,
        total_cost=2.5,
        source="bedrock",
        bedrock_cost=2.5,
        api_cost=0.0,
        timestamp="2026-06-02T10:00:00.000Z",
        end_timestamp="2026-06-02T10:00:00.000Z",
    )
    session = SessionData(
        session_id="session-1",
        source_path="/tmp/session-1.jsonl",
        project_path="/tmp",
        project_name="test",
        title="test-session",
        turns=[],
        total_cost=2.5,
        unlinked_subagents=[sa],
    )

    days = build_day_summaries([session])
    assert len(days) == 1
    day = days[0]
    assert day.date == "2026-06-02"
    assert day.total_cost == 2.5
    assert day.bedrock_cost == 2.5
    assert day.api_cost == 0.0
    assert day.total_input_tokens == 100
    assert day.total_output_tokens == 100
    assert len(day.sessions) == 1
    assert day.sessions[0] is session


def test_workflow_subagent_linking(tmp_path):
    from claude_code_cost_explorer.reader import parse_session_file

    # 1. Create session JSONL file
    session_jsonl = tmp_path / "session-1.jsonl"
    session_jsonl.write_text(
        '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"msg1"}]},"uuid":"u1","parentUuid":null,"timestamp":"2026-06-02T10:00:00.000Z","sessionId":"session-1","cwd":"/tmp","version":"1.0","slug":"session-1"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tool-1","name":"run_workflow","input":{}}],"model":"claude-sonnet-4-6","usage":{"input_tokens":500,"output_tokens":100,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a1","parentUuid":"u1","timestamp":"2026-06-02T10:01:00.000Z","sessionId":"session-1"}\n'
        '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool-1","content":[{"type":"text","text":"Transcript dir: /tmp/session-1/subagents/workflows/wf_123"}]}]},"uuid":"u2","parentUuid":"a1","timestamp":"2026-06-02T10:02:00.000Z","sessionId":"session-1"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"workflow launched"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"a2","parentUuid":"u2","timestamp":"2026-06-02T10:03:00.000Z","sessionId":"session-1"}\n'
    )

    # 2. Create subagents directories and files
    subagents_dir = tmp_path / "session-1" / "subagents" / "workflows" / "wf_123"
    subagents_dir.mkdir(parents=True)

    sa_meta = subagents_dir / "agent-wf-sub.meta.json"
    sa_meta.write_text(
        '{"description": "Workflow Subagent test", "agentType": "workflow"}'
    )

    sa_jsonl = subagents_dir / "agent-wf-sub.jsonl"
    sa_jsonl.write_text(
        '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"sub_prompt"}]},"uuid":"sa-u1","parentUuid":null,"timestamp":"2026-06-02T10:01:30.000Z"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"sub_response"}],"model":"claude-sonnet-4-6","usage":{"input_tokens":1000,"output_tokens":200,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"uuid":"sa-a1","parentUuid":"sa-u1","timestamp":"2026-06-02T10:01:45.000Z"}\n'
    )

    # 3. Parse session
    s = parse_session_file(str(session_jsonl), "test-project")

    # 4. Verify results
    assert s is not None
    # Unlinked subagents list should be empty since the workflow subagent was linked
    assert len(s.unlinked_subagents) == 0

    # Check turns
    assert len(s.turns) == 2
    turn1 = s.turns[0]
    turn2 = s.turns[1]

    assert len(turn1.tool_calls) == 0
    assert len(turn2.tool_calls) == 1
    tc = turn2.tool_calls[0]
    assert tc.tool_use_id == "tool-1"

    # Subagents should be associated with tool-1
    assert len(tc.subagents) == 1
    sa_data = tc.subagents[0]
    assert sa_data.agent_id == "wf-sub"
    assert sa_data.description == "Workflow Subagent test"
    assert sa_data.agent_type == "workflow"

    # Subagent cost should be correct
    # Input token: 1000, output token: 200, model: claude-sonnet-4-6 (cost: $3/M input, $15/M output)
    # Cost = (1000 * 3) / 10^6 + (200 * 15) / 10^6 = 0.003 + 0.003 = 0.006
    assert abs(sa_data.total_cost - 0.006) < 1e-6

    # Turn 2 cost should include its own cost + subagent's cost
    # Turn 2 own cost: usage {input_tokens: 100, output_tokens: 50} -> (100*3 + 50*15)/10^6 = 0.0003 + 0.00075 = 0.00105
    # Turn 2 rolled up cost: 0.00105 + 0.006 = 0.00705
    assert abs(turn2.cost_usd - 0.00705) < 1e-6

    # Turn 1 cost should be just its own cost
    # Turn 1 own cost: usage {input_tokens: 500, output_tokens: 100} -> (500*3 + 100*15)/10^6 = 0.003
    assert abs(turn1.cost_usd - 0.003) < 1e-6

    # Total cost of session should include all costs
    # Total session cost = Turn 1 cost + Turn 2 cost = 0.003 + 0.00705 = 0.01005
    assert abs(s.total_cost - 0.01005) < 1e-6
