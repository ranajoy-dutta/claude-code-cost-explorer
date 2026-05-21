"""Reads and parses ~/.claude/projects JSONL session files."""

from __future__ import annotations
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from claude_code_cost_explorer.cost import calculate_cost

CLAUDE_DIR = os.path.expanduser(os.environ.get("CLAUDE_DIR", "~/.claude"))
_PROJECTS_DIR_OVERRIDE = os.environ.get("CLAUDE_PROJECTS_DIR")


def _infer_source(message_id: str) -> str:
    """Return 'bedrock' for AWS Bedrock calls, 'api' otherwise.

    Bedrock assigns IDs like msg_bdrk_<hex>; the Anthropic API uses msg_<hex>.
    """
    mid = message_id or ""
    if mid.startswith("msg_bdrk_"):
        return "bedrock"
    return "api"


@dataclass
class SubagentTurn:
    """A single assistant turn inside a subagent session."""

    uuid: str
    model: str
    tool_uses: list  # list of {"name": str, "id": str, "input": dict}
    tool_results: list  # list of {"tool_use_id": str, "name": str, "content": list}
    text_blocks: list  # list of str (text content)
    thinking_chars: int = 0


@dataclass
class SubagentData:
    agent_id: str
    description: str
    agent_type: str
    turns: list  # list[SubagentTurn]
    total_tool_uses: int = 0
    total_cost: float = 0.0
    source: str = "api"  # 'bedrock' or 'api'
    bedrock_cost: float = 0.0
    api_cost: float = 0.0


@dataclass
class ToolCallInfo:
    tool_use_id: str
    name: str
    input: dict
    result_content: list  # raw content from the tool_result block
    subagent: Optional[SubagentData] = None  # populated for Agent tool calls


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
    source: str = "api"  # 'bedrock' or 'api' based on message.id prefix


@dataclass
class SessionData:
    session_id: str
    source_path: str
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
    bedrock_cost: float = 0.0
    api_cost: float = 0.0
    source: str = "api"  # dominant source for this session
    compaction_events: list = field(default_factory=list)  # list[CompactionEvent]
    away_summary_events: list = field(default_factory=list)  # list[AwaySummaryEvent]
    ai_title_event: Optional[AiTitleEvent] = None


@dataclass
class DaySummary:
    date: str
    total_cost: float = 0.0
    session_count: int = 0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    sessions: list = field(default_factory=list)
    bedrock_cost: float = 0.0
    api_cost: float = 0.0


@dataclass
class CompactionEvent:
    timestamp: str
    trigger: str
    pre_tokens: int
    post_tokens: int
    duration_ms: int


@dataclass
class AwaySummaryEvent:
    timestamp: str
    content: str


@dataclass
class AiTitleEvent:
    ai_title: str


def _parse_subagent_jsonl(jsonl_path: str, agent_id: str) -> SubagentData:
    """Parse a subagents/agent-{id}.jsonl into a SubagentData."""
    meta_path = jsonl_path.replace(".jsonl", ".meta.json")
    description = ""
    agent_type = ""
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            description = meta.get("description", "")
            agent_type = meta.get("agentType", "")
    except (OSError, json.JSONDecodeError):
        pass

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
        return SubagentData(
            agent_id=agent_id, description=description, agent_type=agent_type, turns=[]
        )

    # Build tool_use_id -> (name, input) map from assistant records
    tool_use_map: dict = {}
    for r in records:
        if r.get("type") == "assistant":
            for block in (r.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_map[block["id"]] = {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }

    # Build tool_result lookup from user records: tool_use_id -> content list
    tool_result_map: dict = {}
    for r in records:
        if r.get("type") == "user":
            content = (r.get("message") or {}).get("content") or r.get("content") or []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tid = item.get("tool_use_id", "")
                        raw = item.get("content")
                        if isinstance(raw, list):
                            tool_result_map[tid] = raw
                        elif isinstance(raw, str):
                            tool_result_map[tid] = [{"type": "text", "text": raw}]
                        else:
                            tool_result_map[tid] = []

    # Build assistant turns — one per assistant record (no merging: each record in a
    # subagent has its own distinct tool call or text, even when parent-chained)
    seen_uuids: set = set()
    turns: list = []
    for r in records:
        if r.get("type") != "assistant":
            continue
        uuid = r.get("uuid", "")
        if uuid in seen_uuids:
            continue
        seen_uuids.add(uuid)
        content = (r.get("message") or {}).get("content") or []
        tool_uses = []
        text_blocks = []
        thinking_chars = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "tool_use":
                tool_uses.append(
                    {
                        "name": block.get("name", ""),
                        "id": block.get("id", ""),
                        "input": block.get("input", {}),
                        "result": tool_result_map.get(block.get("id", ""), []),
                    }
                )
            elif bt == "text" and block.get("text", "").strip():
                text_blocks.append(block["text"])
            elif bt == "thinking":
                thinking_chars += len(block.get("thinking", ""))
        if not tool_uses and not text_blocks and not thinking_chars:
            continue
        turns.append(
            SubagentTurn(
                uuid=uuid,
                model=(r.get("message") or {}).get("model", ""),
                tool_uses=tool_uses,
                tool_results=[],
                text_blocks=text_blocks,
                thinking_chars=thinking_chars,
            )
        )

    total_tool_uses = sum(len(t.tool_uses) for t in turns)

    # Compute deduplicated cost using the same max-merge approach as parse_session_file:
    # child records share cumulative usage with their parent, so only sum root records
    # after walking their chain and taking the max for each usage field.
    from collections import defaultdict as _defaultdict

    _children_of: dict = _defaultdict(list)
    for r in records:
        if r.get("type") == "assistant":
            p = r.get("parentUuid", "")
            if p in {
                x["uuid"]
                for x in records
                if x.get("type") == "assistant" and x.get("uuid")
            }:
                _children_of[p].append(r)

    _asst_uuid_set = {
        r["uuid"] for r in records if r.get("type") == "assistant" and r.get("uuid")
    }
    _records_by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}

    def _chain_usage(uuid: str) -> dict:
        r = _records_by_uuid.get(uuid)
        if not r:
            return {}
        u = dict((r.get("message") or {}).get("usage") or {})
        for child in _children_of.get(uuid, []):
            child_u = _chain_usage(child["uuid"])
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                u[key] = max(u.get(key, 0), child_u.get(key, 0))
        return u

    bedrock_cost = 0.0
    api_cost = 0.0
    # Dedup roots by message.id (same Bedrock-fragmentation pattern as parent file).
    seen_root_mids: set = set()
    for r in records:
        if r.get("type") != "assistant":
            continue
        if r.get("parentUuid", "") in _asst_uuid_set:
            continue  # skip child records — their cost is captured via the root
        msg = r.get("message") or {}
        msg_id = msg.get("id", "") or ""
        if msg_id and msg_id in seen_root_mids:
            continue
        if msg_id:
            seen_root_mids.add(msg_id)
        model = msg.get("model", "")
        usage = _chain_usage(r.get("uuid", ""))
        cost = calculate_cost(model, usage)
        if _infer_source(msg_id) == "bedrock":
            bedrock_cost += cost
        else:
            api_cost += cost

    total_cost = bedrock_cost + api_cost
    source = "bedrock" if bedrock_cost >= api_cost else "api"

    return SubagentData(
        agent_id=agent_id,
        description=description,
        agent_type=agent_type,
        turns=turns,
        total_tool_uses=total_tool_uses,
        total_cost=total_cost,
        source=source,
        bedrock_cost=bedrock_cost,
        api_cost=api_cost,
    )


def _load_subagents(session_jsonl_path: str) -> dict:
    """Return {agent_id: SubagentData} for all subagents of a session, if any."""
    session_id = os.path.basename(session_jsonl_path).replace(".jsonl", "")
    subagents_dir = os.path.join(
        os.path.dirname(session_jsonl_path), session_id, "subagents"
    )
    if not os.path.isdir(subagents_dir):
        return {}
    result = {}
    for fname in os.listdir(subagents_dir):
        if not fname.startswith("agent-") or not fname.endswith(".jsonl"):
            continue
        agent_id = fname[len("agent-") : -len(".jsonl")]
        full_path = os.path.join(subagents_dir, fname)
        result[agent_id] = _parse_subagent_jsonl(full_path, agent_id)
    return result


_SYSTEM_INJECTED_PREFIXES = (
    "[Request",
    "<task-notification>",
    "<user-prompt-submit-hook>",
    "<system-reminder>",
    "Base directory for this skill:",
)


def _is_system_text(text: str) -> bool:
    return any(text.startswith(p) for p in _SYSTEM_INJECTED_PREFIXES)


def _extract_user_prompt(content) -> str:
    if isinstance(content, str):
        if not _is_system_text(content):
            return content[:120]
        return ""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text and not _is_system_text(text):
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

    # Title: latest custom-title > ai-title > slug > fallback
    title = None
    slug = None
    seen_ai_title: Optional[str] = None
    for r in records:
        rtype = r.get("type")
        if rtype == "custom-title" and r.get("customTitle"):
            title = r["customTitle"]
        elif rtype == "ai-title" and r.get("aiTitle"):
            if not title:
                title = r["aiTitle"]
            seen_ai_title = r["aiTitle"]
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

    # Load subagent sessions stored next to this JSONL file
    subagents: dict = _load_subagents(jsonl_path)

    # Build a map from tool_use_id -> agentId by scanning Agent tool results
    _agent_id_by_tool_use_id: dict = {}
    import re as _re

    for r in records:
        if r.get("type") == "user":
            content = (r.get("message") or {}).get("content") or r.get("content") or []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        raw = item.get("content")
                        texts = []
                        if isinstance(raw, list):
                            texts = [
                                c.get("text", "")
                                for c in raw
                                if isinstance(c, dict) and c.get("type") == "text"
                            ]
                        elif isinstance(raw, str):
                            texts = [raw]
                        for txt in texts:
                            m = _re.search(r"agentId:\s*([a-f0-9]+)", txt)
                            if m:
                                _agent_id_by_tool_use_id[
                                    item.get("tool_use_id", "")
                                ] = m.group(1)

    last_user_prompt = ""
    pending_user_prompt_full = ""
    pending_tool_calls: list = []
    turns = []
    _uuid_to_turn = {}
    _assistant_parent_map = {}
    _mid_to_turn: dict = {}
    compaction_events_list: list = []
    away_summary_events_list: list = []
    for r in records:
        rtype = r.get("type")
        if rtype == "system" and r.get("subtype") == "compact_boundary":
            meta = r.get("compactMetadata") or {}
            compaction_events_list.append(
                CompactionEvent(
                    timestamp=r.get("timestamp", ""),
                    trigger=meta.get("trigger", ""),
                    pre_tokens=meta.get("preTokens", 0),
                    post_tokens=meta.get("postTokens", 0),
                    duration_ms=meta.get("durationMs", 0),
                )
            )
        elif rtype == "system" and r.get("subtype") == "away_summary":
            content = r.get("content", "")
            if content:
                away_summary_events_list.append(
                    AwaySummaryEvent(
                        timestamp=r.get("timestamp", ""),
                        content=content,
                    )
                )
        elif rtype == "user":
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
                        agent_id = _agent_id_by_tool_use_id.get(tool_use_id)
                        subagent = subagents.get(agent_id) if agent_id else None
                        pending_tool_calls.append(
                            ToolCallInfo(
                                tool_use_id=tool_use_id,
                                name=name,
                                input=tool_info.get("input", {}),
                                result_content=result_content,
                                subagent=subagent,
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
                            if t and not _is_system_text(t):
                                pending_user_prompt_full = t
                                break
                elif isinstance(content, str) and not _is_system_text(content):
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
            # Root assistant record: if we've already seen this message.id as a
            # root (Bedrock sometimes emits the same msg as two fragmented roots),
            # max-merge into the existing turn instead of creating a duplicate.
            msg_id = (r.get("message", {}) or {}).get("id", "") or ""
            existing = _mid_to_turn.get(msg_id) if msg_id else None
            if existing is not None:
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    existing.usage[key] = max(
                        existing.usage.get(key, 0), usage.get(key, 0)
                    )
                existing.cost_usd = calculate_cost(existing.model, existing.usage)
                existing.assistant_content.extend(asst_content)
                _uuid_to_turn[r.get("uuid", "")] = existing
                _assistant_parent_map[r.get("uuid", "")] = parent_uuid
                continue
            turn = Turn(
                uuid=r.get("uuid", ""),
                timestamp=r.get("timestamp", ""),
                model=model,
                usage=dict(usage),
                cost_usd=calculate_cost(model, usage),
                user_prompt=last_user_prompt,
                user_prompt_full=pending_user_prompt_full,
                tool_calls=pending_tool_calls,
                assistant_content=asst_content,
                source=_infer_source(msg_id),
            )
            turns.append(turn)
            _uuid_to_turn[r.get("uuid", "")] = turn
            _assistant_parent_map[r.get("uuid", "")] = parent_uuid
            if msg_id:
                _mid_to_turn[msg_id] = turn
            last_user_prompt = ""
            pending_user_prompt_full = ""
            pending_tool_calls = []

    if not turns:
        return None

    if not project_path:
        project_path = fallback_name

    # Roll subagent costs up into the parent turn that invoked them
    linked_agent_ids: set = set()
    for turn in turns:
        for tc in turn.tool_calls:
            if tc.subagent and tc.subagent.total_cost > 0:
                turn.cost_usd += tc.subagent.total_cost
                linked_agent_ids.add(tc.subagent.agent_id)

    # Add costs of subagents that couldn't be linked to a specific turn via agentId regex
    unlinked_subagents = [
        sa for aid, sa in subagents.items() if aid not in linked_agent_ids
    ]
    unlinked_subagent_cost = sum(sa.total_cost for sa in unlinked_subagents)
    unlinked_bedrock_cost = sum(sa.bedrock_cost for sa in unlinked_subagents)
    unlinked_api_cost = sum(sa.api_cost for sa in unlinked_subagents)

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

    # Per-source totals: turn base cost by turn.source, plus each linked subagent's
    # own bedrock_cost / api_cost, plus any unlinked subagent buckets.
    bedrock_cost = 0.0
    api_cost = 0.0
    for turn in turns:
        base_cost = turn.cost_usd
        for tc in turn.tool_calls:
            if (
                tc.subagent
                and tc.subagent.total_cost > 0
                and tc.subagent.agent_id in linked_agent_ids
            ):
                base_cost -= tc.subagent.total_cost
                bedrock_cost += tc.subagent.bedrock_cost
                api_cost += tc.subagent.api_cost
        if turn.source == "bedrock":
            bedrock_cost += base_cost
        else:
            api_cost += base_cost
    bedrock_cost += unlinked_bedrock_cost
    api_cost += unlinked_api_cost

    total_cost = sum(t.cost_usd for t in turns) + unlinked_subagent_cost
    session_source = "bedrock" if bedrock_cost >= api_cost else "api"

    return SessionData(
        session_id=session_id,
        source_path=jsonl_path,
        project_path=project_path,
        project_name=project_name,
        title=title,
        turns=turns,
        total_cost=total_cost,
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
        bedrock_cost=bedrock_cost,
        api_cost=api_cost,
        source=session_source,
        compaction_events=compaction_events_list,
        away_summary_events=away_summary_events_list,
        ai_title_event=AiTitleEvent(ai_title=seen_ai_title) if seen_ai_title else None,
    )


def load_all_sessions(claude_dir: str = CLAUDE_DIR) -> list[SessionData]:
    """Scan ~/.claude/projects/ — top-level .jsonl files only (subdirectories are not recursed into)."""
    if _PROJECTS_DIR_OVERRIDE:
        projects_dir = os.path.expanduser(_PROJECTS_DIR_OVERRIDE)
    else:
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
        dates_seen: set = set()
        for turn in s.turns:
            d = turn.timestamp[:10] if turn.timestamp else ""
            if not d or (from_date and d < from_date) or (to_date and d > to_date):
                continue
            if d not in days:
                days[d] = DaySummary(date=d)
            day = days[d]
            day.total_cost += turn.cost_usd
            if turn.source == "bedrock":
                day.bedrock_cost += turn.cost_usd
            else:
                day.api_cost += turn.cost_usd
            day.message_count += 1
            day.total_input_tokens += turn.usage.get("input_tokens", 0)
            day.total_output_tokens += turn.usage.get("output_tokens", 0)
            if d not in dates_seen:
                dates_seen.add(d)
                day.session_count += 1
                day.sessions.append(s)
    return sorted(days.values(), key=lambda x: x.date, reverse=True)


def get_sessions_for_date(sessions: list[SessionData], date: str) -> list[SessionData]:
    def _first_ts_on_date(s: SessionData) -> str:
        for t in s.turns:
            if t.timestamp and t.timestamp[:10] == date:
                return t.timestamp
        return s.first_timestamp

    matching = [
        s
        for s in sessions
        if any(t.timestamp and t.timestamp[:10] == date for t in s.turns)
    ]
    return sorted(matching, key=_first_ts_on_date, reverse=True)


def get_session_by_id(
    sessions: list[SessionData], session_id: str
) -> Optional[SessionData]:
    return next((s for s in sessions if s.session_id == session_id), None)


def normalize_session_title(title: str) -> str:
    return " ".join(title.strip().split())


def append_custom_session_title(session: SessionData, title: str) -> str:
    clean_title = normalize_session_title(title)
    if not clean_title:
        raise ValueError("Session name cannot be empty.")
    if len(clean_title) > 160:
        raise ValueError("Session name must be 160 characters or fewer.")
    if not session.source_path:
        raise ValueError("Session source file is unknown.")

    expected_name = f"{session.session_id}.jsonl"
    if os.path.basename(session.source_path) != expected_name:
        raise ValueError("Session source file does not match the session id.")

    record = {
        "type": "custom-title",
        "sessionId": session.session_id,
        "customTitle": clean_title,
    }
    payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
    with open(session.source_path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(payload + b"\n")
    return clean_title
