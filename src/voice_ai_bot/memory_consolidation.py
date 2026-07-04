from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI

from .config import Config
from .memory import MemoryStore

LOGGER = logging.getLogger(__name__)


CONSOLIDATION_INSTRUCTIONS = (
    "You are the long-term memory consolidation worker for SipQuest, an AI vending assistant for a mystery "
    "drink machine. "
    "You are smarter and slower than the realtime voice model. Your job is to read raw conversation notes "
    "and propose durable memory operations. Return only a single JSON object. "
    "Keep MEMORY.md compact. Add or update only stable facts, preferences, explicit user instructions, "
    "project decisions, and corrections that will likely matter in future conversations. "
    "Do not store secrets, credentials, API keys, passwords, private tokens, or one-off chit-chat. "
    "Do not store facts the assistant merely said unless the user confirmed or requested them. "
    "Forget or update memory only when the user explicitly asks or clearly corrects existing memory. "
    "Use this schema exactly: "
    '{"summary":"short explanation","operations":[{"action":"add|update|forget|ignore",'
    '"kind":"preference|fact|instruction|project|note","entry_id":"optional existing id",'
    '"query":"optional fallback for forget","text":"durable memory text","reason":"brief reason"}]}. '
    "Use action=ignore when there is nothing durable to save."
)


class MemoryConsolidator:
    def __init__(self, config: Config, store: MemoryStore, client: Any | None = None):
        self.config = config
        self.store = store
        self.client = client or OpenAI(api_key=config.openai_api_key, timeout=config.task_timeout_seconds)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def request(self, reason: str = "") -> None:
        if not self.config.memory_consolidation_enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(reason,),
                name="memory-consolidation",
                daemon=True,
            )
            self._thread.start()

    def flush(self, timeout: float | None = None) -> None:
        if not self.config.memory_consolidation_enabled:
            return
        self.request("flush")
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        thread.join(
            self.config.memory_consolidation_shutdown_timeout_seconds if timeout is None else timeout
        )
        if thread.is_alive():
            LOGGER.warning("memory consolidation still running after shutdown timeout")

    def run_once(self) -> dict[str, Any]:
        if not self.config.memory_consolidation_enabled:
            return {"ok": True, "processed": 0, "disabled": True}
        notes = self.store.pending_consolidation_notes(
            max_notes=self.config.memory_consolidation_max_notes,
            max_chars=self.config.memory_consolidation_max_chars,
        )
        if not notes:
            return {"ok": True, "processed": 0}

        note_ids = [str(note.get("id") or "") for note in notes if note.get("id")]
        try:
            payload = self._request_operations(notes)
            operations = _normalize_operations(payload.get("operations"))
            if not operations:
                operations = [{"action": "ignore", "reason": "no durable memory changes"}]
            operation_results = self.store.apply_consolidation_operations(operations)
            self.store.mark_consolidation_processed(
                note_ids,
                summary=str(payload.get("summary") or ""),
                operation_results=operation_results,
            )
            LOGGER.info(
                "memory consolidation processed %d notes with %d operations",
                len(note_ids),
                len(operations),
            )
            return {
                "ok": True,
                "processed": len(note_ids),
                "operations": operations,
                "operation_results": operation_results,
            }
        except BaseException as exc:
            self.store.mark_consolidation_failed(note_ids, str(exc))
            LOGGER.exception("memory consolidation failed")
            return {"ok": False, "processed": 0, "error": str(exc)}

    def _run_loop(self, reason: str) -> None:
        if self.config.memory_consolidation_debounce_seconds > 0 and reason != "flush":
            time.sleep(self.config.memory_consolidation_debounce_seconds)
        while True:
            result = self.run_once()
            if not result.get("ok"):
                return
            if not self.store.pending_consolidation_notes(
                max_notes=1,
                max_chars=max(1, self.config.memory_consolidation_max_chars),
            ):
                return

    def _request_operations(self, notes: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self._prompt(notes)
        response = self.client.responses.create(
            model=self.config.memory_consolidation_model,
            instructions=CONSOLIDATION_INSTRUCTIONS,
            input=prompt,
            reasoning={"effort": self.config.memory_consolidation_reasoning_effort},
            store=False,
            truncation="auto",
        )
        text = _response_output_text(response).strip()
        if not text:
            raise RuntimeError("memory consolidation returned empty response")
        return _parse_json_object(text)

    def _prompt(self, notes: list[dict[str, Any]]) -> str:
        active_entries = self.store.list_entries(include_forgotten=False).get("entries", [])
        note_blocks = []
        for note in notes:
            note_blocks.append(
                "\n".join(
                    [
                        f"NOTE ID: {note.get('id')}",
                        f"TYPE: {note.get('type')}",
                        f"CREATED: {note.get('created_at')}",
                        f"SOURCE: {note.get('path')}",
                        "TEXT:",
                        str(note.get("text") or ""),
                    ]
                )
            )
        return (
            f"{_local_context(self.config)}\n\n"
            "Existing durable memory entries JSON:\n"
            f"{json.dumps(active_entries, ensure_ascii=False, indent=2)}\n\n"
            "New raw notes to consolidate:\n"
            + "\n\n---\n\n".join(note_blocks)
        )


def _normalize_operations(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    operations: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        if action not in {"add", "update", "forget", "ignore", "noop", "none"}:
            continue
        operation = {
            "action": action,
            "kind": str(item.get("kind") or "note"),
            "entry_id": str(item.get("entry_id") or ""),
            "query": str(item.get("query") or ""),
            "text": str(item.get("text") or ""),
            "reason": str(item.get("reason") or ""),
        }
        operations.append(operation)
    return operations


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("memory consolidation response was not a JSON object")
    return payload


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return str(output_text)
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(str(text))
    return "".join(parts)


def _local_context(config: Config) -> str:
    try:
        now = datetime.now(ZoneInfo(config.user_timezone))
    except Exception:
        now = datetime.now().astimezone()
    return (
        f"Current local time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()}. "
        f"User location: {config.user_city}, {config.user_region}, {config.user_country}. "
        f"User timezone: {config.user_timezone}."
    )
