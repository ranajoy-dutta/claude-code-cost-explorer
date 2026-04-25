"""Reads and parses ~/.claude/projects JSONL session files."""

from __future__ import annotations
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from claude_code_cost_explorer.cost import calculate_cost

CLAUDE_DIR = os.path.expanduser("~/.claude")


@dataclass
class ToolCallInfo:
    tool_use_id: str
    name: str
    input: dict
    result_content: list  # raw content from the tool_result block


@dataclass
class Turn:
    uuid: str
    timestamp: str
    model: str
    usage: dict
    cost_usd: float
    user_prompt: str = ""
    tool_calls: list = field(default_factory=list)  # list[ToolCallInfo]
    user_prompt_full: str = ""  # full untruncated user message text
    assistant_content: list = field(
        default_factory=list
    )  # assistant response content blocks
    duration_seconds: float = 0.0  # time until next turn (latency)
    thinking_chars: int = 0  # total chars in thinking blocks


@dataclass
class SessionData:
    session_id: str
    project_path: str
    project_name: str
    title: str
    turns: list = field(default_factory=list)
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    message_count: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""
    date: str = ""
    duration_seconds: float = 0.0  # total session wall-clock duration


@dataclass
class DaySummary:
    date: str
    total_cost: float = 0.0
    session_count: int = 0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    sessions: list = field(default_factory=list)


def _extract_user_prompt(content) -> str:
    if isinstance(content, str):
        return content[:120]
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text and not text.startswith("[Request"):
                    return text[:120]
    return ""


def parse_session_file(jsonl_path: str, project_hint: str) -> Optional[SessionData]:
    session_id = os.path.basename(jsonl_path).replace(".jsonl", "")
    # Encoding is ambiguous for hyphenated names; cwd field overrides this for real sessions
    fallback_name = (project_hint.lstrip("-").rsplit("-", 1)[-1]) or project_hint
    project_path = ""  # will be set from cwd records
    project_name = fallback_name

    records = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return None

    if not records:
        return None

    assistant_uuids = {
        r["uuid"] for r in records if r.get("type") == "assistant" and r.get("uuid")
    }

    # Title: custom-title > ai-title > slug > fallback
    title = None
    slug = None
    for r in records:
        rtype = r.get("type")
        if rtype == "custom-title" and r.get("customTitle"):
            title = r["customTitle"]
            break
        if rtype == "ai-title" and not title and r.get("aiTitle"):
            title = r["aiTitle"]
        if not slug and r.get("slug"):
            slug = r["slug"]
    if not title:
        title = slug or session_id[:8]

    # Use cwd as authoritative project path
    for r in records:
        if r.get("cwd"):
            project_path = r["cwd"]
            project_name = (
                os.path.basename(os.path.normpath(project_path)) or project_path
            )
            break

    # Step A: build a map of tool_use_id -> {name, input} from all assistant records
    tool_use_map: dict = {}
    for r in records:
        if r.get("type") == "assistant":
            for block in (r.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_map[block["id"]] = {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }

    last_user_prompt = ""
    pending_user_prompt_full = ""
    pending_tool_calls: list = []
    turns = []
    _uuid_to_turn = {}
    _assistant_parent_map = {}
    for r in records:
        rtype = r.get("type")
        if rtype == "user":
            msg = r.get("message", {})
            content = msg.get("content") if msg else r.get("content")
            # Step B: collect tool_result blocks into pending_tool_calls
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_use_id = item.get("tool_use_id", "")
                        tool_info = tool_use_map.get(tool_use_id, {})
                        raw_result = item.get("content")
                        # Normalize: content can be a list, a string, or None
                        if isinstance(raw_result, list):
                            result_content = raw_result
                        elif isinstance(raw_result, str):
                            result_content = [{"type": "text", "text": raw_result}]
                        else:
                            result_content = []
                        name = tool_info.get("name", "")
                        if not name:
                            for rc in result_content:
                                if (
                                    isinstance(rc, dict)
                                    and rc.get("type") == "tool_reference"
                                ):
                                    name = rc.get("tool_name", "tool")
                                    break
                        if not name:
                            name = "tool"
                        pending_tool_calls.append(
                            ToolCallInfo(
                                tool_use_id=tool_use_id,
                                name=name,
                                input=tool_info.get("input", {}),
                                result_content=result_content,
                            )
                        )
            # Only set last_user_prompt if there's actual text content (not just tool results)
            text = _extract_user_prompt(content)
            if text:
                last_user_prompt = text
                pending_tool_calls = []  # clear pending if this is a real human message
                # Capture full (untruncated) user text
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            t = item.get("text", "")
                            if t and not t.startswith("[Request"):
                                pending_user_prompt_full = t
                                break
                elif isinstance(content, str) and not content.startswith("[Request"):
                    pending_user_prompt_full = content
        elif rtype == "assistant":
            parent_uuid = r.get("parentUuid", "")
            model = r.get("message", {}).get("model", "")
            if model == "<synthetic>":
                continue
            usage = r.get("message", {}).get("usage")
            asst_content = r.get("message", {}).get("content") or []

            if parent_uuid in assistant_uuids:
                # Child record: merge content blocks into the parent turn
                # Walk the chain to find the root turn
                root_uuid = parent_uuid
                visited = set()
                while root_uuid in _assistant_parent_map and root_uuid not in visited:
                    visited.add(root_uuid)
                    if _assistant_parent_map[root_uuid] in assistant_uuids:
                        root_uuid = _assistant_parent_map[root_uuid]
                    else:
                        break
                if root_uuid in _uuid_to_turn:
                    parent_turn = _uuid_to_turn[root_uuid]
                    parent_turn.assistant_content.extend(asst_content)
                    # The child record's usage is cumulative for the turn, so take the max
                    if usage:
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        ):
                            parent_turn.usage[key] = max(
                                parent_turn.usage.get(key, 0), usage.get(key, 0)
                            )
                        parent_turn.cost_usd = calculate_cost(
                            parent_turn.model, parent_turn.usage
                        )
                # Track this child's parent so grandchildren can find the root
                _assistant_parent_map[r.get("uuid", "")] = parent_uuid
                continue

            if not usage:
                continue
            # Root assistant record: create a new Turn
            turn = Turn(
                uuid=r.get("uuid", ""),
                timestamp=r.get("timestamp", ""),
                model=model,
                usage=usage,
                cost_usd=calculate_cost(model, usage),
                user_prompt=last_user_prompt,
                user_prompt_full=pending_user_prompt_full,
                tool_calls=pending_tool_calls,
                assistant_content=asst_content,
            )
            turns.append(turn)
            _uuid_to_turn[r.get("uuid", "")] = turn
            _assistant_parent_map[r.get("uuid", "")] = parent_uuid
            last_user_prompt = ""
            pending_user_prompt_full = ""
            pending_tool_calls = []

    if not turns:
        return None

    if not project_path:
        project_path = fallback_name

    # Compute per-turn duration from consecutive timestamps
    for i, turn in enumerate(turns):
        # Count thinking chars
        for block in turn.assistant_content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                turn.thinking_chars += len(block.get("thinking", ""))
        # Compute latency to next turn
        if i + 1 < len(turns) and turn.timestamp and turns[i + 1].timestamp:
            try:
                t0 = datetime.fromisoformat(turn.timestamp.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(
                    turns[i + 1].timestamp.replace("Z", "+00:00")
                )
                turn.duration_seconds = max(0, (t1 - t0).total_seconds())
            except (ValueError, TypeError):
                pass

    # Session-level duration
    session_duration = 0.0
    timestamps = [t.timestamp for t in turns if t.timestamp]
    if len(timestamps) >= 2:
        try:
            t_first = datetime.fromisoformat(min(timestamps).replace("Z", "+00:00"))
            t_last = datetime.fromisoformat(max(timestamps).replace("Z", "+00:00"))
            session_duration = max(0, (t_last - t_first).total_seconds())
        except (ValueError, TypeError):
            pass

    return SessionData(
        session_id=session_id,
        project_path=project_path,
        project_name=project_name,
        title=title,
        turns=turns,
        total_cost=sum(t.cost_usd for t in turns),
        total_input_tokens=sum(t.usage.get("input_tokens", 0) for t in turns),
        total_output_tokens=sum(t.usage.get("output_tokens", 0) for t in turns),
        total_cache_write_tokens=sum(
            t.usage.get("cache_creation_input_tokens", 0) for t in turns
        ),
        total_cache_read_tokens=sum(
            t.usage.get("cache_read_input_tokens", 0) for t in turns
        ),
        message_count=len(turns),
        first_timestamp=min(timestamps) if timestamps else "",
        last_timestamp=max(timestamps) if timestamps else "",
        date=min(timestamps)[:10] if timestamps else "",
        duration_seconds=session_duration,
    )


def load_all_sessions(claude_dir: str = CLAUDE_DIR) -> list[SessionData]:
    """Scan ~/.claude/projects/ — top-level .jsonl files only (subdirectories are not recursed into)."""
    projects_dir = os.path.join(claude_dir, "projects")
    if not os.path.isdir(projects_dir):
        return []
    sessions = []
    for encoded_name in os.listdir(projects_dir):
        proj_dir = os.path.join(projects_dir, encoded_name)
        if not os.path.isdir(proj_dir):
            continue
        for entry in os.listdir(proj_dir):
            full_path = os.path.join(proj_dir, entry)
            if entry.endswith(".jsonl") and os.path.isfile(full_path):
                s = parse_session_file(full_path, encoded_name)
                if s:
                    sessions.append(s)
    return sessions


def build_day_summaries(
    sessions: list[SessionData], from_date="", to_date=""
) -> list[DaySummary]:
    days: dict[str, DaySummary] = {}
    for s in sessions:
        d = s.date
        if not d or (from_date and d < from_date) or (to_date and d > to_date):
            continue
        if d not in days:
            days[d] = DaySummary(date=d)
        day = days[d]
        day.total_cost += s.total_cost
        day.session_count += 1
        day.message_count += s.message_count
        day.total_input_tokens += s.total_input_tokens
        day.total_output_tokens += s.total_output_tokens
        day.sessions.append(s)
    return sorted(days.values(), key=lambda x: x.date, reverse=True)


def get_sessions_for_date(sessions: list[SessionData], date: str) -> list[SessionData]:
    return sorted(
        [s for s in sessions if s.date == date],
        key=lambda s: s.first_timestamp,
        reverse=True,
    )


def get_session_by_id(
    sessions: list[SessionData], session_id: str
) -> Optional[SessionData]:
    return next((s for s in sessions if s.session_id == session_id), None)
