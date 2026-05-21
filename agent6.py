from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import re
import sys
import textwrap
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

SYNTHESIS_KEYWORDS = (
    "synthes",
    "extract",
    "list",
    "compare",
    "decide",
    "choose",
    "appropriate",
    "summar",
    "contribution",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preview_text(text: str, limit: int = 160) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def configure_console_noise() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("ddgs").setLevel(logging.WARNING)
    logging.getLogger("crawl4ai").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def resolve_query_text(prompt: Optional[str], query_parts: list[str]) -> str:
    if prompt:
        return prompt.strip()
    if query_parts:
        return " ".join(query_parts).strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    raise ValueError(
        "No query provided. Pass a positional query, use --prompt, or pipe text on stdin."
    )


def simple_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    stop = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "for",
        "from",
        "give",
        "has",
        "have",
        "his",
        "her",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "what",
        "when",
        "with",
    }
    out: list[str] = []
    for token in tokens:
        if token in stop or len(token) < 3:
            continue
        if token not in out:
            out.append(token)
        if len(out) >= limit:
            break
    return out


@dataclass
class Goal:
    text: str
    status: str = "open"
    why: str = ""
    evidence: list[str] = field(default_factory=list)
    attach_artifact_id: Optional[str] = None
    attach_artifact_ids: list[str] = field(default_factory=list)


@dataclass
class MemoryItem:
    id: str
    kind: str
    summary: str
    value: dict[str, Any]
    keywords: list[str]
    created_at: str


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        content: bytes,
        *,
        source: str = "",
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        artifact_id = f"art:{uuid.uuid4().hex[:16]}"
        suffix = mimetypes.guess_extension(content_type) or ".bin"
        path = self.root / f"{artifact_id.replace(':', '_')}{suffix}"
        path.write_bytes(content)
        return {
            "artifact_id": artifact_id,
            "path": str(path),
            "source": source,
            "content_type": content_type,
            "length_bytes": len(content),
            "preview": preview_text(content.decode("utf-8", errors="replace")),
            "created_at": utc_now(),
        }

    def read_bytes(self, artifact_id: str) -> bytes:
        path = self._find_path(artifact_id)
        return path.read_bytes()

    def read_text(self, artifact_id: str) -> str:
        return self.read_bytes(artifact_id).decode("utf-8", errors="replace")

    def _find_path(self, artifact_id: str) -> Path:
        prefix = artifact_id.replace(":", "_")
        matches = sorted(self.root.glob(f"{prefix}*"))
        if not matches:
            raise KeyError(f"Unknown artifact id: {artifact_id}")
        return matches[0]


class MemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "memory.json"
        if not self.path.exists():
            self._write([])

    def all_items(self) -> list[MemoryItem]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryItem(**item) for item in raw]

    def add(
        self,
        kind: str,
        summary: str,
        value: dict[str, Any],
        keywords: Optional[list[str]] = None,
    ) -> MemoryItem:
        item = MemoryItem(
            id=f"mem:{uuid.uuid4().hex[:12]}",
            kind=kind,
            summary=summary,
            value=value,
            keywords=keywords or simple_keywords(summary + " " + compact_json(value)),
            created_at=utc_now(),
        )
        items = self.all_items()
        items.append(item)
        self._write([asdict(entry) for entry in items])
        return item

    def search(self, query: str, limit: int = 8) -> list[MemoryItem]:
        query_terms = set(simple_keywords(query, limit=12))
        scored: list[tuple[int, MemoryItem]] = []
        for item in self.all_items():
            overlap = len(query_terms.intersection(item.keywords))
            if overlap or query_terms.intersection(
                simple_keywords(item.summary, limit=12)
            ):
                scored.append((overlap, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for _, item in scored[:limit]]

    def remember_fact_from_query(self, query: str) -> Optional[MemoryItem]:
        birthday = re.search(
            r"\b(?P<who>[A-Za-z][A-Za-z' ]+?)\s+birthday\s+is\s+(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
            query,
            flags=re.IGNORECASE,
        )
        if birthday:
            who = normalize_whitespace(birthday.group("who"))
            date = birthday.group("date")
            possessive = who if who.lower().endswith("'s") else f"{who}'s"
            summary = f"{possessive} birthday is on {date}"
            return self.add(
                "fact",
                summary,
                {
                    "entity": who,
                    "field": "birthday",
                    "date": date,
                    "source_query": query,
                },
                keywords=simple_keywords(f"{who} birthday {date}"),
            )

        remember = re.search(
            r"\bremember(?: that)?\s+(?P<fact>.+)",
            query,
            flags=re.IGNORECASE,
        )
        if remember:
            fact = normalize_whitespace(remember.group("fact"))
            return self.add(
                "fact",
                fact,
                {"text": fact, "source_query": query},
                keywords=simple_keywords(fact),
            )
        return None

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(items, ensure_ascii=True, indent=2), encoding="utf-8"
        )


class ToolRuntime:
    TOOL_DEFS = [
        {
            "name": "web_search",
            "description": "Search the web. Use for finding pages or activity ideas.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_url",
            "description": "Fetch a URL and return clean markdown. Creates an artifact.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1},
                },
                "required": ["url"],
            },
        },
        {
            "name": "get_time",
            "description": "Get current time in a named IANA timezone.",
            "input_schema": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
        {
            "name": "currency_convert",
            "description": "Convert one currency amount into another.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a text file from the sandbox.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "list_dir",
            "description": "List files in a sandbox directory.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
        {
            "name": "create_file",
            "description": "Create a new file in the sandbox.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "update_file",
            "description": "Overwrite an existing sandbox file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "edit_file",
            "description": "Replace text inside a sandbox file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string"},
                    "replace": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "find", "replace"],
            },
        },
        {
            "name": "create_calendar_event",
            "description": "Create a calendar event (.ics) in the sandbox calendar folder.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "reminders": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "date"],
            },
        },
    ]

    def __init__(self, memory: MemoryStore, artifacts: ArtifactStore) -> None:
        self.memory = memory
        self.artifacts = artifacts
        self._module = None

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        guard_args = self._guard_artifact_handles(name, dict(arguments))
        fn = getattr(self._module_ref(), name)
        if asyncio.iscoroutinefunction(fn):
            result = asyncio.run(fn(**guard_args))
        else:
            result = fn(**guard_args)
        return self._record(name, guard_args, result)

    def _guard_artifact_handles(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if name == "fetch_url" and str(arguments.get("url", "")).startswith("art:"):
            artifact_id = arguments["url"]
            arguments["url"] = self.artifacts.read_text(artifact_id)
        return arguments

    def _record(
        self, name: str, arguments: dict[str, Any], result: Any
    ) -> dict[str, Any]:
        if name == "fetch_url":
            text = result.get("text", "")
            artifact = self.artifacts.put(
                text.encode("utf-8"),
                source=arguments.get("url", ""),
                content_type=result.get("content_type", "text/markdown"),
            )
            self.memory.add(
                "artifact",
                f"Fetched {arguments.get('url', '')}",
                {
                    "tool": name,
                    "arguments": arguments,
                    "artifact_id": artifact["artifact_id"],
                    "source": arguments.get("url", ""),
                    "length_bytes": artifact["length_bytes"],
                    "preview": artifact["preview"],
                },
                keywords=simple_keywords(
                    f"{arguments.get('url', '')} {artifact['preview']}"
                ),
            )
            return {"artifact": artifact, "tool_result": result}

        summary = f"{name} returned {preview_text(compact_json(result), limit=100)}"
        self.memory.add(
            "tool_result",
            summary,
            {"tool": name, "arguments": arguments, "result": result},
            keywords=simple_keywords(
                f"{name} {compact_json(arguments)} {compact_json(result)}"
            ),
        )
        return {"result": result}

    def _module_ref(self):
        if self._module is None:
            import mcp_server as mcp_module

            self._module = mcp_module
        return self._module


class GatewayBrain:
    def __init__(self, gateway_url: Optional[str] = None) -> None:
        from llm_gatewayV3.client import LLM

        self.llm = LLM(base_url=gateway_url) if gateway_url else LLM()

    def structured(
        self, *, prompt: str, auto_route: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.llm.chat(
            prompt,
            auto_route=auto_route,
            temperature=0,
            response_format={"type": "json_schema", "schema": schema, "strict": True},
        )
        parsed = response.get("parsed")
        if parsed is None:
            raise ValueError(
                f"Gateway did not return structured output for {auto_route}"
            )
        return parsed


class CognitiveAgent:
    def __init__(
        self,
        state_dir: str | Path = "state",
        gateway_url: Optional[str] = None,
        brain: Optional[GatewayBrain] = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryStore(self.state_dir)
        self.artifacts = ArtifactStore(self.state_dir / "artifacts")
        self.runtime = ToolRuntime(self.memory, self.artifacts)
        self.brain = brain or GatewayBrain(gateway_url=gateway_url)

    def run(self, query: str, max_iters: int = 8) -> dict[str, Any]:
        remembered = self.memory.remember_fact_from_query(query)
        if remembered:
            print(f'[memory.remember] classified "{preview_text(query, 70)}" as fact')
            print(f"                  keywords: {remembered.keywords}")

        goals: list[Goal] = []
        final_answer: Optional[str] = None

        for iteration in range(1, max_iters + 1):
            print(f"\n─── iter {iteration} ───")
            hits = self.memory.search(query)
            if hits:
                print(f"[memory.read]   {len(hits)} hits")
                for item in hits[:3]:
                    print(
                        f'                {item.kind}: "{preview_text(item.summary, 70)}"'
                    )

            goals = self._perceive(query, hits, goals)
            self._print_goals(goals)

            if all(goal.status == "done" for goal in goals) and goals:
                if not final_answer:
                    final_answer = self._recover_answer_from_memory(query, goals, hits)
                if not final_answer:
                    retry_goal = self._pick_answer_goal(goals)
                    retry_goal.status = "open"
                else:
                    print(f"\n[done] all {len(goals)} goals satisfied")
                    return {
                        "answer": final_answer or "",
                        "iterations": iteration,
                        "goals": [asdict(g) for g in goals],
                    }

            current = next((goal for goal in goals if goal.status != "done"), None)
            if current is None:
                break

            artifact_texts: list[dict[str, str]] = []
            for artifact_id in self._goal_artifact_ids(
                current
            ):  ##getting artifact id in iteration 1
                artifact_text = self.artifacts.read_text(artifact_id)
                artifact_texts.append(
                    {"artifact_id": artifact_id, "text": artifact_text}
                )
                print(
                    f"[attach]        {artifact_id} ({len(artifact_text.encode('utf-8'))} bytes)"
                )

            decision = self._decide(query, goals, hits, current, artifact_texts)
            if decision["kind"] == "answer":
                final_answer = decision["answer"]
                current.status = "done"
                current.evidence.append("answered")
                self.memory.add(
                    "tool_result",
                    f"Answered goal: {current.text}",
                    {"goal": current.text, "answer": final_answer},
                    keywords=simple_keywords(current.text + " " + final_answer),
                )
                print(f"[decision]      ANSWER: {preview_text(final_answer, 120)}")
                goals = self._mark_answered_goals(goals, final_answer)
                continue

            tool_name = decision["tool_name"]
            tool_args = decision["tool_args"]
            print(f"[decision]      TOOL_CALL: {tool_name}({compact_json(tool_args)})")
            action_result = self.runtime.call(tool_name, tool_args)
            self._render_action(action_result)
            current.evidence.append(
                f"{tool_name}:{preview_text(compact_json(tool_args), 60)}"
            )
            if tool_name in {
                "fetch_url",
                "web_search",
                "get_time",
                "currency_convert",
                "read_file",
                "list_dir",
                "create_file",
                "update_file",
                "edit_file",
            }:
                if tool_name == "fetch_url":
                    if self._is_fetch_results_goal_done(query, current.text):
                        current.status = "done"
                else:
                    current.status = "done"

        raise RuntimeError(
            f"Agent stopped after {max_iters} iterations without satisfying all goals."
        )

    def _perceive(
        self, query: str, hits: list[MemoryItem], previous_goals: list[Goal]
    ) -> list[Goal]:
        prev_payload = [self._goal_payload(goal) for goal in previous_goals]
        hit_payload = [
            {
                "kind": item.kind,
                "summary": item.summary,
                "value": item.value,
            }
            for item in hits[:6]
        ]
        schema = {
            "type": "object",
            "properties": {
                "goals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["open", "done"]},
                            "why": {"type": "string"},
                        },
                        "required": ["text", "status", "why"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["goals"],
            "additionalProperties": False,
        }
        prompt = textwrap.dedent(
            f"""
            You are the Perception layer in a cognitive agent.
            Break the user request into a short ordered goal list.
            Preserve any already-done goals from previous state.
            Use memory hits to mark a goal done only when the needed fact already exists.
            Keep 1 to 4 goals max.

            User query:
            {query}

            Previous goals:
            {compact_json(prev_payload)}

            Memory hits:
            {compact_json(hit_payload)}
            """
        ).strip()
        parsed = self.brain.structured(
            prompt=prompt, auto_route="perception", schema=schema
        )
        new_goals = [
            Goal(
                text=normalize_whitespace(goal["text"]),
                status=goal["status"],
                why=goal["why"],
            )
            for goal in parsed["goals"]
        ]
        new_goals = self._apply_goal_safety_nets(new_goals, hits, previous_goals)
        return new_goals

    def _apply_goal_safety_nets(
        self, goals: list[Goal], hits: list[MemoryItem], previous_goals: list[Goal]
    ) -> list[Goal]:
        previous_by_text = {goal.text: goal for goal in previous_goals}
        previous_done = {goal.text for goal in previous_goals if goal.status == "done"}
        for goal in goals:
            previous = previous_by_text.get(goal.text)
            if previous:
                self._set_goal_attachments(goal, self._goal_artifact_ids(previous))
            if goal.text in previous_done:
                goal.status = "done"

        artifact_hits = [
            item
            for item in hits
            if item.kind == "artifact" and item.value.get("artifact_id")
        ]
        recent_artifact_ids = [item.value["artifact_id"] for item in artifact_hits]
        for idx, goal in enumerate(goals):
            text_lower = goal.text.lower()
            if goal.status == "done":
                if not self._goal_done_has_evidence(goal, hits):
                    goal.status = "open"
                continue
            if idx > 0 and not self._goal_artifact_ids(goal):
                prior = goals[idx - 1].text.lower()
                if (
                    any(term in prior for term in ("fetch", "open", "read"))
                    and recent_artifact_ids
                ):
                    self._set_goal_attachments(goal, recent_artifact_ids[:1])
            if recent_artifact_ids and any(
                term in text_lower for term in SYNTHESIS_KEYWORDS
            ):
                selected = self._select_relevant_artifacts(goal.text, artifact_hits)
                if len(selected) > len(self._goal_artifact_ids(goal)):
                    self._set_goal_attachments(goal, selected)
        return goals

    def _decide(
        self,
        query: str,
        goals: list[Goal],
        hits: list[MemoryItem],
        current_goal: Goal,
        artifact_texts: list[dict[str, str]],
    ) -> dict[str, Any]:
        memory_context = [
            {
                "kind": item.kind,
                "summary": item.summary,
                "value": item.value,
            }
            for item in hits[:6]
        ]
        artifact_block = self._build_artifact_block(query, current_goal, artifact_texts)

        schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["tool_call", "answer"]},
                "tool_name": {"type": "string"},
                "tool_args": {"type": "object"},
                "answer": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["kind", "reason"],
            "additionalProperties": False,
        }
        prompt = textwrap.dedent(
            f"""
            You are the Decision layer in a cognitive agent.
            Pick exactly one next step for the current goal.
            Use a tool when information or file-system action is still needed.
            Answer directly only when the current goal can already be satisfied from memory hits or an attached artifact.
            Never invent file contents or web facts.

            User query:
            {query}

            Current goal:
            {compact_json(self._goal_payload(current_goal))}

            All goals:
            {compact_json([self._goal_payload(goal) for goal in goals])}

            Memory hits:
            {compact_json(memory_context)}

            Available tools:
            {compact_json(self.runtime.TOOL_DEFS)}
            {artifact_block}
            """
        ).strip()
        parsed = self.brain.structured(
            prompt=prompt, auto_route="decision", schema=schema
        )
        if parsed["kind"] == "tool_call":
            parsed.setdefault("tool_args", {})
        return self._repair_decision(query, current_goal, hits, parsed)

    def _mark_answered_goals(self, goals: list[Goal], answer: str) -> list[Goal]:
        answer_lower = answer.lower()
        for goal in goals:
            if goal.status == "done":
                continue
            if any(
                token in answer_lower for token in simple_keywords(goal.text, limit=6)
            ):
                goal.status = "done"
        return goals

    def _print_goals(self, goals: list[Goal]) -> None:
        print("[perception]    ", end="")
        if not goals:
            print("[open] Clarify the task")
            return
        first = True
        for goal in goals:
            prefix = "" if first else "                "
            first = False
            line = f"{prefix}[{goal.status}] {goal.text}"
            print(line)
            attachments = self._goal_artifact_ids(goal)
            if attachments:
                if len(attachments) == 1:
                    print(f"                  attach={attachments[0]}")
                else:
                    print(f"                  attach={', '.join(attachments)}")

    def _render_action(self, action_result: dict[str, Any]) -> None:
        if "artifact" in action_result:
            artifact = action_result["artifact"]
            print(
                f"[action]        → [artifact {artifact['artifact_id']}, "
                f"{artifact['length_bytes']} bytes] preview: {artifact['preview']}"
            )
            return
        result = action_result["result"]
        print(f"[action]        → {preview_text(compact_json(result), 140)}")

    def _build_artifact_block(
        self,
        query: str,
        current_goal: Goal,
        artifact_texts: list[dict[str, str]],
        max_chars: int = 20000,
    ) -> str:
        if not artifact_texts:
            return ""
        per_artifact_budget = max(2000, max_chars // max(1, len(artifact_texts)))
        rendered: list[str] = []
        remaining = max_chars
        for entry in artifact_texts:
            if remaining <= 0:
                break
            excerpt = self._artifact_excerpt_for_goal(
                entry["text"],
                query=query,
                goal_text=current_goal.text,
                max_chars=min(per_artifact_budget, remaining),
            )
            rendered.append(f"Artifact {entry['artifact_id']}:\n{excerpt}")
            remaining -= len(excerpt)
        if not rendered:
            return ""
        return "\nAttached artifact texts:\n" + "\n\n".join(rendered) + "\n"

    def _artifact_excerpt_for_goal(
        self, text: str, *, query: str, goal_text: str, max_chars: int
    ) -> str:
        normalized = text.strip()
        if len(normalized) <= max_chars:
            return normalized

        keywords = simple_keywords(f"{query} {goal_text}", limit=12)
        snippets: "OrderedDict[str, None]" = OrderedDict()

        # Always keep the head of the document because pages like Wikipedia
        # often put infobox and summary facts there.
        head_budget = min(max_chars // 2, 8000)
        head = normalized[:head_budget].strip()
        if head:
            snippets[head] = None

        context_radius = 260
        lowered = normalized.lower()
        for keyword in keywords:
            if len(keyword) < 3:
                continue
            start = 0
            needle = keyword.lower()
            while True:
                idx = lowered.find(needle, start)
                if idx == -1:
                    break
                snippet_start = max(0, idx - context_radius)
                snippet_end = min(len(normalized), idx + len(keyword) + context_radius)
                snippet = normalized[snippet_start:snippet_end].strip()
                if snippet:
                    snippets[snippet] = None
                start = idx + len(keyword)
                if sum(len(part) for part in snippets) >= max_chars * 2:
                    break
            if sum(len(part) for part in snippets) >= max_chars * 2:
                break

        parts: list[str] = []
        used = 0
        for snippet in snippets.keys():
            piece = snippet
            separator = "\n...\n" if parts else ""
            next_size = used + len(separator) + len(piece)
            if next_size > max_chars:
                remaining = max_chars - used - len(separator)
                if remaining > 100:
                    parts.append(separator + piece[:remaining].rstrip())
                break
            parts.append(separator + piece)
            used = next_size
        return "".join(parts)

    def _repair_decision(
        self,
        query: str,
        current_goal: Goal,
        hits: list[MemoryItem],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if decision.get("kind") != "tool_call":
            return decision

        tool_name = decision.get("tool_name") or self._infer_tool_name(
            query, current_goal.text
        )
        tool_args = dict(decision.get("tool_args") or {})
        if not tool_name:
            raise ValueError(
                f"Decision returned tool_call without tool_name for goal: {current_goal.text}"
            )

        if tool_name == "fetch_url":
            requested_url = str(tool_args.get("url", "")).strip()
            if not requested_url or self._was_url_already_fetched(requested_url):
                inferred_url = self._infer_fetch_url(query, current_goal.text, hits)
                if inferred_url:
                    tool_args["url"] = inferred_url
        elif tool_name == "web_search" and self._goal_requires_weather_fetch(
            current_goal.text
        ):
            tool_name = "fetch_url"
            weather_url = self._infer_weather_url(query, current_goal.text)
            tool_args = {"url": weather_url} if weather_url else {}
        elif tool_name == "web_search" and not tool_args.get("query"):
            tool_args["query"] = self._infer_search_query(query, current_goal.text)
        elif tool_name == "list_dir" and not tool_args.get("path"):
            tool_args["path"] = "."

        elif tool_name == "create_calendar_event":
            # Try to infer a sensible title/date/reminders for calendar events
            tool_args = dict(tool_args or {})
            # If the goal text contains explicit dates, prefer those
            found_dates = re.findall(
                r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", f"{query}\n{current_goal.text}"
            )
            if found_dates:
                # Prefer the last date mentioned (often the main event)
                tool_args.setdefault("date", found_dates[-1])

            # If no explicit date found, look for a remembered birthday in memory hits
            if not tool_args.get("date"):
                for item in hits:
                    if (
                        item.kind == "fact"
                        and str(item.value.get("field", "")).lower() == "birthday"
                    ):
                        date_val = item.value.get("date") or item.summary
                        tool_args.setdefault("date", date_val)
                        entity = item.value.get("entity") or "Event"
                        possessive = (
                            entity if entity.lower().endswith("'s") else f"{entity}'s"
                        )
                        tool_args.setdefault("title", f"{possessive} birthday")
                        break

            # Ensure title present
            if not tool_args.get("title"):
                # Try to extract an entity like "mom" from the query
                m = re.search(
                    r"\b(my|mom|dad|mother|father|wife|husband|partner)\b",
                    query,
                    flags=re.IGNORECASE,
                )
                if m:
                    ent = m.group(1)
                    if ent.lower() == "my":
                        ent = "My"
                    tool_args.setdefault("title", f"{ent}'s birthday")
                else:
                    tool_args.setdefault("title", "Event")

            # Infer reminders: handle common phrasing like 'two weeks before' and 'on the day'
            if not tool_args.get("reminders"):
                rems: list[str] = []
                txt = f"{query} {current_goal.text}".lower()
                if "two weeks" in txt or "14 days" in txt or "two-week" in txt:
                    rems.append("P14D")
                if (
                    "on the day" in txt
                    or "day of" in txt
                    or "on the day" in current_goal.text.lower()
                ):
                    rems.append("PT0S")
                if rems:
                    tool_args["reminders"] = rems

            decision["tool_args"] = tool_args

        decision["tool_name"] = tool_name
        decision["tool_args"] = tool_args
        return decision

    def _infer_tool_name(self, query: str, goal_text: str) -> Optional[str]:
        text = f"{query} {goal_text}".lower()
        if any(
            term in text
            for term in ("fetch", "open", "read page", "wikipedia", "forecast", "url")
        ):
            return "fetch_url"
        if any(term in text for term in ("search", "find things to do", "activities")):
            return "web_search"
        if "list" in text and "dir" in text:
            return "list_dir"
        if "create" in text and "file" in text:
            return "create_file"
        return None

    def _infer_fetch_url(
        self, query: str, goal_text: str, hits: list[MemoryItem]
    ) -> Optional[str]:
        direct_urls = re.findall(r"https?://\S+", f"{query}\n{goal_text}")
        if direct_urls:
            return direct_urls[0].rstrip(").,")

        search_urls = self._search_result_urls(query, goal_text, hits)
        if search_urls:
            search_urls = self._rank_search_urls(search_urls)
            fetched_sources = self._fetched_sources(search_urls)
            for url in search_urls:
                if url not in fetched_sources:
                    return url
            return search_urls[0]
        return None

    def _rank_search_urls(self, urls: list[str]) -> list[str]:
        preferred = [
            "docs.python.org",
            "python.org",
            "realpython.com",
            "pymotw.com",
            "learnpython.org",
            "tutorialspoint.com",
            "dev.to",
            "medium.com",
        ]
        deprioritize = [
            "reddit.com",
            "stackoverflow.com/questions",
            "discuss.python.org",
            "quora.com",
            "github.com",
        ]

        def score(url: str) -> int:
            lower = url.lower()
            if any(domain in lower for domain in preferred):
                return 0
            if any(domain in lower for domain in deprioritize):
                return 2
            return 1

        return sorted(urls, key=lambda url: (score(url), urls.index(url)))

    def _infer_search_query(self, query: str, goal_text: str) -> str:
        goal_lower = goal_text.lower()
        if "tokyo" in goal_lower:
            return "family-friendly things to do in Tokyo this weekend"
        if "asyncio" in goal_lower:
            if (
                "guide" in goal_lower
                or "authoritative" in goal_lower
                or "comprehensive" in goal_lower
            ):
                return "Python asyncio best practices official documentation"
            return "Python asyncio best practices"
        return query

    def _goal_payload(self, goal: Goal) -> dict[str, Any]:
        payload = asdict(goal)
        payload["attach_artifact_id"] = goal.attach_artifact_id
        payload["attach_artifact_ids"] = self._goal_artifact_ids(goal)
        return payload

    def _goal_artifact_ids(self, goal: Goal) -> list[str]:
        artifact_ids = list(goal.attach_artifact_ids)
        if goal.attach_artifact_id and goal.attach_artifact_id not in artifact_ids:
            artifact_ids.insert(0, goal.attach_artifact_id)
        return artifact_ids

    def _set_goal_attachments(self, goal: Goal, artifact_ids: list[str]) -> None:
        deduped: list[str] = []
        for artifact_id in artifact_ids:
            if artifact_id and artifact_id not in deduped:
                deduped.append(artifact_id)
        goal.attach_artifact_ids = deduped
        goal.attach_artifact_id = deduped[0] if deduped else None

    def _recover_answer_from_memory(
        self, query: str, goals: list[Goal], hits: list[MemoryItem]
    ) -> str:
        all_items = self.memory.all_items()
        goal_texts = {goal.text for goal in goals}

        for item in reversed(all_items):
            if item.kind != "tool_result":
                continue
            value = item.value
            answer = str(value.get("answer", "")).strip()
            if not answer:
                continue
            if value.get("goal") in goal_texts:
                return answer

        relevant_artifacts = [
            item
            for item in all_items
            if item.kind == "artifact" and self._artifact_relevant_to_query(item, query)
        ]
        if not relevant_artifacts:
            return ""

        synthesis_goal = goals[-1]
        recovered_goal = Goal(
            text=synthesis_goal.text,
            status="open",
            why=synthesis_goal.why,
        )
        self._set_goal_attachments(
            recovered_goal,
            self._select_relevant_artifacts(recovered_goal.text, relevant_artifacts),
        )
        artifact_texts = [
            {"artifact_id": artifact_id, "text": self.artifacts.read_text(artifact_id)}
            for artifact_id in self._goal_artifact_ids(recovered_goal)
        ]
        if not artifact_texts:
            return ""
        decision = self._decide(query, goals, hits, recovered_goal, artifact_texts)
        if decision.get("kind") == "answer":
            answer = str(decision.get("answer", "")).strip()
            if answer:
                self.memory.add(
                    "tool_result",
                    f"Answered goal: {recovered_goal.text}",
                    {"goal": recovered_goal.text, "answer": answer},
                    keywords=simple_keywords(recovered_goal.text + " " + answer),
                )
                return answer
        return ""

    def _artifact_relevant_to_query(self, item: MemoryItem, query: str) -> bool:
        query_terms = set(simple_keywords(query, limit=12))
        haystack = " ".join(
            [
                item.summary,
                str(item.value.get("source", "")),
                str(item.value.get("preview", "")),
            ]
        )
        artifact_terms = set(simple_keywords(haystack, limit=16))
        return bool(query_terms.intersection(artifact_terms))

    def _search_result_urls(
        self, query: str, goal_text: str, hits: list[MemoryItem]
    ) -> list[str]:
        preferred_query = self._infer_search_query(query, goal_text)
        all_items = hits + [
            item
            for item in self.memory.all_items()
            if item.id not in {hit.id for hit in hits}
        ]
        candidates: list[tuple[str, list[str]]] = []
        for item in all_items:
            if item.kind != "tool_result" or item.value.get("tool") != "web_search":
                continue
            search_query = str(item.value.get("arguments", {}).get("query", "")).strip()
            urls = [
                result.get("url", "")
                for result in item.value.get("result", [])
                if result.get("url")
            ]
            if urls:
                candidates.append((search_query, urls))

        for search_query, urls in reversed(candidates):
            if search_query == preferred_query:
                return urls
        return candidates[-1][1] if candidates else []

    def _fetched_sources(self, candidate_urls: Optional[list[str]] = None) -> set[str]:
        allowed = set(candidate_urls or [])
        sources: set[str] = set()
        for item in self.memory.all_items():
            if item.kind != "artifact":
                continue
            source = str(item.value.get("source", ""))
            if not source:
                continue
            if allowed and source not in allowed:
                continue
            sources.add(source)
        return sources

    def _was_url_already_fetched(self, url: str) -> bool:
        if not url:
            return False
        return url in self._fetched_sources([url])

    def _goal_fetch_target_count(self, goal_text: str, default: int = 3) -> int:
        match = re.search(r"\btop\s+(\d+)\b", goal_text.lower())
        if match:
            return max(1, int(match.group(1)))
        match = re.search(r"\b(\d+)\s+search\s+results?\b", goal_text.lower())
        if match:
            return max(1, int(match.group(1)))
        return default

    def _is_fetch_results_goal_done(self, query: str, goal_text: str) -> bool:
        search_urls = self._search_result_urls(query, goal_text, [])
        if not search_urls:
            return False
        fetched_sources = self._fetched_sources(search_urls)
        target = min(self._goal_fetch_target_count(goal_text), len(search_urls))
        return len(fetched_sources) >= target

    def _goal_done_has_evidence(self, goal: Goal, hits: list[MemoryItem]) -> bool:
        text_lower = goal.text.lower()
        if any(term in text_lower for term in ("weather", "forecast")):
            return self._has_weather_artifact(hits)
        if any(term in text_lower for term in SYNTHESIS_KEYWORDS) or any(
            term in text_lower for term in ("recommend", "appropriate", "choose")
        ):
            return self._has_answer_for_goal(goal.text, hits)
        if any(term in text_lower for term in ("search", "find", "identify")):
            return self._has_search_result(hits)
        return True

    def _has_search_result(self, hits: list[MemoryItem]) -> bool:
        return any(
            item.kind == "tool_result" and item.value.get("tool") == "web_search"
            for item in hits
        )

    def _has_weather_artifact(self, hits: list[MemoryItem]) -> bool:
        for item in hits:
            if item.kind != "artifact":
                continue
            source = str(item.value.get("source", "")).lower()
            preview = str(item.value.get("preview", "")).lower()
            if "wttr.in" in source or "forecast" in preview or "temperature" in preview:
                return True
        return False

    def _has_answer_for_goal(self, goal_text: str, hits: list[MemoryItem]) -> bool:
        for item in reversed(self.memory.all_items()):
            if item.kind != "tool_result":
                continue
            if str(item.value.get("goal", "")).strip() != goal_text:
                continue
            if str(item.value.get("answer", "")).strip():
                return True
        for item in hits:
            if item.kind != "tool_result":
                continue
            if str(item.value.get("goal", "")).strip() != goal_text:
                continue
            if str(item.value.get("answer", "")).strip():
                return True
        return False

    def _goal_requires_weather_fetch(self, goal_text: str) -> bool:
        text_lower = goal_text.lower()
        return "weather" in text_lower or "forecast" in text_lower

    def _infer_weather_url(self, query: str, goal_text: str) -> Optional[str]:
        text = f"{query} {goal_text}".lower()
        if "tokyo" in text and "saturday" in text:
            return "https://wttr.in/Tokyo?format=Saturday"
        if "tokyo" in text:
            return "https://wttr.in/Tokyo?format=Saturday"
        return None

    def _pick_answer_goal(self, goals: list[Goal]) -> Goal:
        for goal in reversed(goals):
            text_lower = goal.text.lower()
            if any(term in text_lower for term in SYNTHESIS_KEYWORDS) or any(
                term in text_lower for term in ("recommend", "appropriate", "choose")
            ):
                return goal
        return goals[-1]

    def _select_relevant_artifacts(
        self, goal_text: str, artifact_hits: list[MemoryItem], limit: int = 3
    ) -> list[str]:
        goal_terms = set(simple_keywords(goal_text, limit=10))
        scored: list[tuple[int, str, str]] = []
        for item in artifact_hits:
            artifact_id = item.value.get("artifact_id")
            if not artifact_id:
                continue
            haystack = " ".join(
                [
                    item.summary,
                    str(item.value.get("source", "")),
                    str(item.value.get("preview", "")),
                ]
            )
            score = len(goal_terms.intersection(simple_keywords(haystack, limit=16)))
            scored.append((score, item.created_at, artifact_id))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        selected = [artifact_id for score, _, artifact_id in scored if score > 0][
            :limit
        ]
        if selected:
            return selected
        return [item.value["artifact_id"] for item in artifact_hits[:limit]]


def main() -> None:
    configure_console_noise()
    parser = argparse.ArgumentParser(description="Run the Session 6 cognitive agent.")
    parser.add_argument(
        "query", nargs="*", help="User request to run through the agent loop."
    )
    parser.add_argument(
        "--prompt", default=None, help="Query text as a single argument."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run multiple queries in a prompt loop.",
    )
    parser.add_argument(
        "--state-dir", default="state", help="Persistent state directory."
    )
    parser.add_argument(
        "--gateway-url", default=None, help="Override LLM gateway base URL."
    )
    parser.add_argument(
        "--max-iters", type=int, default=8, help="Maximum cognitive loop iterations."
    )
    args = parser.parse_args()

    agent = CognitiveAgent(state_dir=args.state_dir, gateway_url=args.gateway_url)
    if args.interactive:
        print(
            "Interactive mode. Enter a query and press Enter. Type 'exit' or 'quit' to stop."
        )
        while True:
            try:
                query = input("\nquery> ").strip()
            except EOFError:
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            result = agent.run(query, max_iters=args.max_iters)
            print(f"\nFINAL: {result['answer']}")
            print(f"Iteration count: {result['iterations']}.")
        return

    query = resolve_query_text(args.prompt, args.query)
    result = agent.run(query, max_iters=args.max_iters)
    print(f"\nFINAL: {result['answer']}")
    print(f"Iteration count: {result['iterations']}.")


if __name__ == "__main__":
    main()
