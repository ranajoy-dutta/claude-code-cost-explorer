"""Reads and parses ~/.claude JSONL session files (projects, jobs, and tasks)."""

from __future__ import annotations
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from claude_code_cost_explorer.cost import calculate_cost, is_model_known

CLAUDE_DIR = os.path.expanduser(os.environ.get("CLAUDE_DIR", "~/.claude"))


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
    timestamp: str = ""
    cost_usd: float = 0.0
    source: str = "api"
    usage: dict = field(default_factory=dict)


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
    timestamp: str = ""
    end_timestamp: str = ""
    source_path: str = ""
    has_unknown_models: bool = False


@dataclass
class ToolCallInfo:
    tool_use_id: str
    name: str
    input: dict
    result_content: list  # raw content from the tool_result block
    subagent: Optional[SubagentData] = None  # populated for Agent tool calls
    subagents: list[SubagentData] = field(default_factory=list)


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
    assistant_content: list = field(default_factory=list)  # list of raw content blocks
    source: str = "api"  # 'bedrock' or 'api'
    thinking_chars: int = 0
    duration_seconds: float = 0.0
    is_forked_turn: bool = False
    _reqs: dict = field(
        default_factory=dict, repr=False
    )  # stores unique (msg_id, req_id)
    _adv_usage: dict = field(default_factory=dict, repr=False)
    _adv_cost: float = 0.0

    def update_usage_from_record(
        self, msg_id: str, req_id: str, model: str, usage: dict, source: str
    ) -> None:
        """Update usage and cost properly deduplicating by msg_id and req_id."""
        if not msg_id or model == "<synthetic>":
            return
        total_toks = sum(
            usage.get(k, 0)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        key = (msg_id, req_id)
        if key not in self._reqs or total_toks > self._reqs[key][2]:
            self._reqs[key] = (model, usage, total_toks, source)

        # parse advisors
        for it in usage.get("iterations") or []:
            if isinstance(it, dict) and it.get("kind") == "advisor_message":
                adv_mod = it.get("model", "")
                adv_u = dict(it.get("usage") or {})
                if adv_mod and adv_u:
                    from .cost import calculate_cost

                    self._adv_cost += calculate_cost(adv_mod, adv_u, "api")
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        self._adv_usage[k] = self._adv_usage.get(k, 0) + adv_u.get(k, 0)

        self.recalculate_totals()

    def recalculate_totals(self) -> None:
        from .cost import calculate_cost

        self.usage = dict(self._adv_usage)
        self.cost_usd = self._adv_cost
        for m, u, _, src in self._reqs.values():
            self.cost_usd += calculate_cost(m, u, src)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                self.usage[k] = self.usage.get(k, 0) + u.get(k, 0)


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
    unlinked_subagents: list[SubagentData] = field(default_factory=list)
    has_unknown_models: bool = False
    fork_parent_id: Optional[str] = None
    fork_parent_title: Optional[str] = None
    fork_point_turn_index: int = 0
    fork_shared_cost: float = 0.0
    fork_incremental_cost: float = 0.0
    fork_new_turns_count: int = 0


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

    # Build assistant chain and usage helper structures
    from collections import defaultdict as _defaultdict

    _asst_uuid_set = {
        r["uuid"] for r in records if r.get("type") == "assistant" and r.get("uuid")
    }
    _records_by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}
    _children_of: dict = _defaultdict(list)
    for r in records:
        if r.get("type") == "assistant":
            p = r.get("parentUuid", "")
            if p in _asst_uuid_set:
                _children_of[p].append(r)

    def _tree_usage_and_cost(root_uuid: str) -> tuple[dict, float, str]:
        descendants = [root_uuid]
        queue = [root_uuid]
        while queue:
            curr = queue.pop(0)
            for child in _children_of.get(curr, []):
                queue.append(child["uuid"])
                descendants.append(child["uuid"])

        reqs = {}
        adv_cost = 0.0
        adv_usage = {}
        primary_source = "api"
        for d_uuid in descendants:
            r = _records_by_uuid.get(d_uuid)
            if not r:
                continue
            msg = r.get("message") or {}
            msg_id = msg.get("id") or r.get("uuid")
            req_id = r.get("requestId")

            for it in (msg.get("usage") or {}).get("iterations") or []:
                if isinstance(it, dict) and it.get("kind") == "advisor_message":
                    adv_mod = it.get("model", "")
                    adv_u = dict(it.get("usage") or {})
                    if adv_mod and adv_u:
                        adv_cost += calculate_cost(adv_mod, adv_u, "api")
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        ):
                            adv_usage[k] = adv_usage.get(k, 0) + adv_u.get(k, 0)

            if not msg_id:
                continue
            u = dict(msg.get("usage") or {})
            model = msg.get("model", "")
            if model == "<synthetic>":
                continue
            total_toks = sum(
                u.get(k, 0)
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )

            src = _infer_source(msg_id)
            if d_uuid == root_uuid:
                primary_source = src
            key = (msg_id, req_id)
            if key not in reqs or total_toks > reqs[key][2]:
                reqs[key] = (model, u, total_toks, src)

        total_u = dict(adv_usage)
        total_c = adv_cost
        for m, u, _, src in reqs.values():
            total_c += calculate_cost(m, u, src)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                total_u[k] = total_u.get(k, 0) + u.get(k, 0)

        return total_u, total_c, primary_source

    # Compute deduplicated cost/source per root assistant record
    root_costs = {}
    root_sources = {}
    root_usages = {}
    seen_root_mids: set = set()
    bedrock_cost = 0.0
    api_cost = 0.0
    has_unknown_models = False
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
        if model == "<synthetic>":
            continue
        usage, cost, source = _tree_usage_and_cost(r.get("uuid", ""))
        uuid = r.get("uuid", "")

        root_costs[uuid] = cost
        root_sources[uuid] = source
        root_usages[uuid] = usage

        if source == "bedrock":
            bedrock_cost += cost
        else:
            api_cost += cost

        if not is_model_known(model, source):
            has_unknown_models = True

    total_cost = bedrock_cost + api_cost
    source = "bedrock" if bedrock_cost >= api_cost else "api"

    # Build assistant turns — one per assistant record
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

        is_root = uuid in root_costs
        cost_usd = root_costs[uuid] if is_root else 0.0
        turn_source = root_sources[uuid] if is_root else "api"
        turn_usage = root_usages[uuid] if is_root else {}

        turns.append(
            SubagentTurn(
                uuid=uuid,
                model=(r.get("message") or {}).get("model", ""),
                tool_uses=tool_uses,
                tool_results=[],
                text_blocks=text_blocks,
                thinking_chars=thinking_chars,
                timestamp=r.get("timestamp", ""),
                cost_usd=cost_usd,
                source=turn_source,
                usage=turn_usage,
            )
        )

    total_tool_uses = sum(len(t.tool_uses) for t in turns)

    timestamps = [r["timestamp"] for r in records if r.get("timestamp")]
    start_timestamp = min(timestamps) if timestamps else ""
    end_timestamp = max(timestamps) if timestamps else ""

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
        timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        source_path=jsonl_path,
        has_unknown_models=has_unknown_models,
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
    for root, dirs, files in os.walk(subagents_dir):
        for fname in files:
            if fname.startswith("agent-") and fname.endswith(".jsonl"):
                agent_id = fname[len("agent-") : -len(".jsonl")]
                full_path = os.path.join(root, fname)
                result[agent_id] = _parse_subagent_jsonl(full_path, agent_id)
            elif fname == "journal.jsonl":
                parent_dir = os.path.basename(root)
                if parent_dir.startswith("wf_"):
                    agent_id = parent_dir
                    full_path = os.path.join(root, fname)
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
    _tool_use_id_by_workflow_id: dict = {}
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
                            m_wf = _re.search(r"workflows/(wf_[a-f0-9\-]+)", txt)
                            if m_wf:
                                _tool_use_id_by_workflow_id[m_wf.group(1)] = item.get(
                                    "tool_use_id", ""
                                )

    # Group subagents by tool_use_id for workflows
    _subagents_by_tool_use_id: dict = {}
    for workflow_id, tuid in _tool_use_id_by_workflow_id.items():
        _subagents_by_tool_use_id[tuid] = [
            sa
            for sa in subagents.values()
            if sa.source_path
            and f"workflows/{workflow_id}/" in sa.source_path.replace("\\", "/")
        ]

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
            is_meta = bool(r.get("isMeta"))
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
                        wf_subagents = _subagents_by_tool_use_id.get(tool_use_id, [])
                        pending_tool_calls.append(
                            ToolCallInfo(
                                tool_use_id=tool_use_id,
                                name=name,
                                input=tool_info.get("input", {}),
                                result_content=result_content,
                                subagent=subagent,
                                subagents=wf_subagents,
                            )
                        )
            # isMeta records are skill injections from the harness, not human messages
            if is_meta:
                pass
            # Only set last_user_prompt if there's actual text content (not just tool results)
            elif text := _extract_user_prompt(content):
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
                    msg_id = (r.get("message", {}) or {}).get(
                        "id", ""
                    ) or parent_turn.uuid
                    req_id = r.get("requestId", "")
                    if usage:
                        parent_turn.update_usage_from_record(
                            msg_id, req_id, model, usage, parent_turn.source
                        )
                # Track this child's parent so grandchildren can find the root
                _assistant_parent_map[r.get("uuid", "")] = parent_uuid
                continue

            if not usage:
                continue
            # Root assistant record: if we've already seen this message.id as a
            # root (Bedrock sometimes emits the same msg as two fragmented roots),
            # max-merge into the existing turn instead of creating a duplicate.
            msg_id = (r.get("message", {}) or {}).get("id", "") or r.get("uuid", "")
            req_id = r.get("requestId", "")
            existing = _mid_to_turn.get(msg_id) if msg_id else None
            if existing is not None:
                existing.update_usage_from_record(
                    msg_id, req_id, model, usage, existing.source
                )
                existing.assistant_content.extend(asst_content)
                _uuid_to_turn[r.get("uuid", "")] = existing
                _assistant_parent_map[r.get("uuid", "")] = parent_uuid
                continue
            turn_source = _infer_source(msg_id)
            turn = Turn(
                uuid=r.get("uuid", ""),
                timestamp=r.get("timestamp", ""),
                model=model,
                usage={},
                cost_usd=0.0,
                user_prompt=last_user_prompt,
                user_prompt_full=pending_user_prompt_full,
                tool_calls=pending_tool_calls,
                assistant_content=asst_content,
                source=turn_source,
            )
            turn.update_usage_from_record(msg_id, req_id, model, usage, turn_source)
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

    # Roll subagent costs and tokens up into the parent turn that invoked them
    linked_agent_ids: set = set()
    for turn in turns:
        for tc in turn.tool_calls:
            if tc.subagent:
                if tc.subagent.total_cost > 0:
                    turn.cost_usd += tc.subagent.total_cost
                for sat in tc.subagent.turns:
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        turn.usage[key] = turn.usage.get(key, 0) + sat.usage.get(key, 0)
                linked_agent_ids.add(tc.subagent.agent_id)
            for sa in tc.subagents:
                if sa.total_cost > 0:
                    turn.cost_usd += sa.total_cost
                for sat in sa.turns:
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        turn.usage[key] = turn.usage.get(key, 0) + sat.usage.get(key, 0)
                linked_agent_ids.add(sa.agent_id)

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
            for sa in tc.subagents:
                if sa.total_cost > 0 and sa.agent_id in linked_agent_ids:
                    base_cost -= sa.total_cost
                    bedrock_cost += sa.bedrock_cost
                    api_cost += sa.api_cost
        if turn.source == "bedrock":
            bedrock_cost += base_cost
        else:
            api_cost += base_cost
    bedrock_cost += unlinked_bedrock_cost
    api_cost += unlinked_api_cost

    total_cost = sum(t.cost_usd for t in turns) + unlinked_subagent_cost
    session_source = "bedrock" if bedrock_cost >= api_cost else "api"

    session_has_unknown_models = False
    for t in turns:
        if not is_model_known(t.model, t.source):
            session_has_unknown_models = True
            break
    if not session_has_unknown_models:
        for sa in subagents.values():
            if sa.has_unknown_models:
                session_has_unknown_models = True
                break

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
        unlinked_subagents=unlinked_subagents,
        has_unknown_models=session_has_unknown_models,
    )


def detect_forks(sessions: list[SessionData]) -> list[SessionData]:
    """Detect forked sessions in the same project and compute fork metadata."""
    by_proj: dict[str, list[SessionData]] = {}
    for s in sessions:
        by_proj.setdefault(s.project_path, []).append(s)

    for s_list in by_proj.values():
        s_list_sorted = sorted(s_list, key=lambda s: s.first_timestamp)
        for i, s2 in enumerate(s_list_sorted):
            if not s2.turns:
                continue
            t2_mids = [
                list(t._reqs.keys())[0][0] if t._reqs else t.uuid for t in s2.turns
            ]
            best_match = None
            best_common = 0
            for s1 in s_list_sorted[:i]:
                if s1.session_id == s2.session_id or not s1.turns:
                    continue
                t1_mids = [
                    list(t._reqs.keys())[0][0] if t._reqs else t.uuid for t in s1.turns
                ]
                common = 0
                for k in range(min(len(t1_mids), len(t2_mids))):
                    if t1_mids[k] == t2_mids[k]:
                        common += 1
                    else:
                        break
                if common >= 2 and common > best_common:
                    best_common = common
                    best_match = s1

            if best_match:
                s2.fork_parent_id = best_match.session_id
                s2.fork_parent_title = best_match.title
                s2.fork_point_turn_index = best_common
                s2.fork_shared_cost = sum(t.cost_usd for t in s2.turns[:best_common])
                s2.fork_incremental_cost = sum(
                    t.cost_usd for t in s2.turns[best_common:]
                )
                s2.fork_new_turns_count = len(s2.turns) - best_common
                for idx, t in enumerate(s2.turns):
                    t.is_forked_turn = idx < best_common

    return sessions


def load_all_sessions(claude_dir: str = CLAUDE_DIR) -> list[SessionData]:
    """Scan ~/.claude/projects/, as well as ~/.claude/jobs/ and ~/.claude/tasks/."""
    sessions = []

    projects_dir = os.path.join(claude_dir, "projects")
    base_dir = claude_dir

    if os.path.isdir(projects_dir):
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

    for folder in ("jobs", "tasks"):
        folder_dir = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_dir):
            continue
        for root, _, files in os.walk(folder_dir):
            for entry in files:
                if entry.endswith(".jsonl"):
                    full_path = os.path.join(root, entry)
                    s = parse_session_file(full_path, f"claude-{folder}")
                    if s:
                        sessions.append(s)

    return detect_forks(sessions)


def build_day_summaries(
    sessions: list[SessionData], from_date="", to_date=""
) -> list[DaySummary]:
    days: dict[str, DaySummary] = {}
    seen_reqs_by_day: dict[str, set] = {}
    seen_subagent_turns: dict[str, set] = {}

    for s in sessions:
        dates_seen: set = set()
        for turn in s.turns:
            d = turn.timestamp[:10] if turn.timestamp else ""
            if not d or (from_date and d < from_date) or (to_date and d > to_date):
                continue
            if d not in days:
                days[d] = DaySummary(date=d)
                seen_reqs_by_day[d] = set()
                seen_subagent_turns[d] = set()
            day = days[d]

            if d not in dates_seen:
                dates_seen.add(d)
                day.session_count += 1
                day.sessions.append(s)

            # Deduplicate turn requests across sessions
            if turn._reqs:
                for req_key, (m, u, _, src) in turn._reqs.items():
                    if req_key in seen_reqs_by_day[d]:
                        continue
                    seen_reqs_by_day[d].add(req_key)
                    req_cost = calculate_cost(m, u, src)
                    day.total_cost += req_cost
                    if src == "bedrock":
                        day.bedrock_cost += req_cost
                    else:
                        day.api_cost += req_cost
                    day.message_count += 1
                    day.total_input_tokens += u.get("input_tokens", 0)
                    day.total_output_tokens += u.get("output_tokens", 0)
                if turn._adv_cost > 0:
                    adv_key = (f"adv_{turn.uuid}", "")
                    if adv_key not in seen_reqs_by_day[d]:
                        seen_reqs_by_day[d].add(adv_key)
                        day.total_cost += turn._adv_cost
                        day.api_cost += turn._adv_cost
                        for k, v in turn._adv_usage.items():
                            if k == "input_tokens":
                                day.total_input_tokens += v
                            elif k == "output_tokens":
                                day.total_output_tokens += v
            else:
                turn_key = (turn.uuid, "")
                if turn_key not in seen_reqs_by_day[d]:
                    seen_reqs_by_day[d].add(turn_key)
                    day.total_cost += turn.cost_usd
                    if turn.source == "bedrock":
                        day.bedrock_cost += turn.cost_usd
                    else:
                        day.api_cost += turn.cost_usd
                    day.message_count += 1
                    day.total_input_tokens += turn.usage.get("input_tokens", 0)
                    day.total_output_tokens += turn.usage.get("output_tokens", 0)

            # Deduplicate subagents linked to tool calls across sessions
            for tc in turn.tool_calls:
                subagents = []
                if tc.subagent:
                    subagents.append(tc.subagent)
                if tc.subagents:
                    subagents.extend(tc.subagents)
                for sa in subagents:
                    for sat in sa.turns:
                        sat_key = (sa.agent_id, sat.uuid)
                        if sat_key in seen_subagent_turns[d]:
                            continue
                        seen_subagent_turns[d].add(sat_key)
                        day.total_cost += sat.cost_usd
                        if sat.source == "bedrock":
                            day.bedrock_cost += sat.cost_usd
                        else:
                            day.api_cost += sat.cost_usd
                        day.total_input_tokens += sat.usage.get("input_tokens", 0)
                        day.total_output_tokens += sat.usage.get("output_tokens", 0)

        for sa in s.unlinked_subagents:
            for sat in sa.turns:
                if sat.cost_usd == 0.0:
                    continue
                d = sat.timestamp[:10] if sat.timestamp else ""
                if not d or (from_date and d < from_date) or (to_date and d > to_date):
                    continue
                if d not in days:
                    days[d] = DaySummary(date=d)
                    seen_reqs_by_day[d] = set()
                    seen_subagent_turns[d] = set()
                sat_key = (sa.agent_id, sat.uuid)
                if sat_key in seen_subagent_turns[d]:
                    continue
                seen_subagent_turns[d].add(sat_key)
                day = days[d]
                day.total_cost += sat.cost_usd
                if sat.source == "bedrock":
                    day.bedrock_cost += sat.cost_usd
                else:
                    day.api_cost += sat.cost_usd
                day.total_input_tokens += sat.usage.get("input_tokens", 0)
                day.total_output_tokens += sat.usage.get("output_tokens", 0)
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
        for sa in s.unlinked_subagents:
            for sat in sa.turns:
                if sat.timestamp and sat.timestamp[:10] == date:
                    return sat.timestamp
        return s.first_timestamp

    matching = []
    for s in sessions:
        has_activity = any(t.timestamp and t.timestamp[:10] == date for t in s.turns)
        if not has_activity:
            for sa in s.unlinked_subagents:
                if any(
                    sat.timestamp and sat.timestamp[:10] == date for sat in sa.turns
                ):
                    has_activity = True
                    break
        if has_activity:
            matching.append(s)

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
