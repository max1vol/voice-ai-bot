from __future__ import annotations

import base64
import json
import logging
import queue
import socket
import threading
import time
import wave
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import certifi
import websocket
from openai import OpenAI

from .audio_io import PcmOutputStream, PcmPlayer, RawPcmRecorder
from .config import Config, REALTIME_SYSTEM_PROMPT
from .conversation import Message
from .memory import MemoryStore
from .scheduled_tasks import ScheduledTaskStore

LOGGER = logging.getLogger(__name__)


CLOSE_TOOL_NAME = "close_realtime_session"
WEB_SEARCH_TOOL_NAME = "web_search"
START_TASK_TOOL_NAME = "start_background_task"
LIST_TASKS_TOOL_NAME = "list_background_tasks"
GET_TASK_TOOL_NAME = "get_background_task"
CANCEL_TASK_TOOL_NAME = "cancel_background_task"
ADD_SCHEDULED_TASK_TOOL_NAME = "add_scheduled_task"
LIST_SCHEDULED_TASKS_TOOL_NAME = "list_scheduled_tasks"
DELETE_SCHEDULED_TASK_TOOL_NAME = "delete_scheduled_task"
MEMORY_SEARCH_TOOL_NAME = "memory_search"
MEMORY_LIST_TOOL_NAME = "memory_list"
MEMORY_ADD_TOOL_NAME = "memory_add"
MEMORY_UPDATE_TOOL_NAME = "memory_update"
MEMORY_FORGET_TOOL_NAME = "memory_forget"
MEMORY_GET_SOURCE_TOOL_NAME = "memory_get_source"
ASYNC_TASK_TOOL_NAMES = {
    START_TASK_TOOL_NAME,
    LIST_TASKS_TOOL_NAME,
    GET_TASK_TOOL_NAME,
    CANCEL_TASK_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    ADD_SCHEDULED_TASK_TOOL_NAME,
    LIST_SCHEDULED_TASKS_TOOL_NAME,
    DELETE_SCHEDULED_TASK_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_LIST_TOOL_NAME,
    MEMORY_ADD_TOOL_NAME,
    MEMORY_UPDATE_TOOL_NAME,
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_GET_SOURCE_TOOL_NAME,
}


@dataclass(frozen=True)
class RealtimeTurnResult:
    user_text: str
    assistant_text: str
    requested_close: bool


@dataclass
class BackgroundTask:
    id: str
    query: str
    title: str = ""
    wakeup_on_complete: bool = False
    wakeup_reported: bool = False
    status: str = "queued"
    progress: str = "queued"
    response_id: str = ""
    result: str = ""
    reasoning_summary: str = ""
    error: str = ""
    events: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


def local_context(config: Config) -> str:
    try:
        now = datetime.now(ZoneInfo(config.user_timezone))
    except Exception:
        now = datetime.now().astimezone()
    utc_offset = now.strftime("%z")
    formatted_offset = f"{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "unknown UTC offset"
    return (
        f"Current local time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()} (UTC{formatted_offset}). "
        f"User location: {config.user_city}, {config.user_region}, {config.user_country}. "
        f"User timezone: {config.user_timezone}."
    )


def realtime_session_instructions(config: Config, memory_context: str = "") -> str:
    parts = [
        f"{REALTIME_SYSTEM_PROMPT} "
        f"The user is in {config.user_city}, {config.user_region}, {config.user_country}. "
        "For each response, the application will provide current local time and location context. "
        "Use the user's current local time, date, timezone, and location for words like today, tomorrow, "
        "local, nearby, morning, evening, and current. "
        "Use start_background_task when the user asks about current, recent, local, weather, news, opening-hours, "
        "price, schedule, or otherwise time-sensitive facts. Prefer the background task tools for web, code, "
        "calculation, or multi-step research so the voice session stays interruptible. "
        "Background GPT-5.5 tasks receive the user's current local time, timezone, and location, and can use "
        "hosted web search plus code execution. "
        "When start_background_task returns a running task, briefly tell the user it started instead of polling "
        "repeatedly in the same response. Set wakeup_on_complete true when the user expects the final answer "
        "to be spoken automatically after the task finishes; set it false for tasks the user only wants to check later. "
        "Use list_background_tasks or get_background_task when the user asks what is running, asks for progress, "
        "or asks for results. "
        "The user can ask you to create, list, or remove scheduled reminders, alarms, and other timed tasks. "
        "Use the scheduled-task tools for that. Spoken scheduled tasks must not start during quiet hours "
        f"{config.schedule_quiet_start}-{config.schedule_quiet_end} local time; if the requested time falls there, "
        "explain that the device will stay quiet then and ask for another time. "
        "Use memory_search or memory_list before answering questions about what you remember, prior conversations, "
        "user preferences, personal facts, or saved decisions. If the user asks you to remember something, call "
        "memory_add. If the user says memory is wrong, call memory_update. If the user asks to forget/delete/remove "
        "memory, call memory_forget. Confirm memory changes briefly. Do not store secrets or credentials. "
        "If interrupted by a new button press, stop the previous reply and treat the new speech as the latest user turn."
    ]
    if memory_context.strip():
        parts.append("\n\nPersistent workspace context:\n" + memory_context.strip())
    return " ".join(parts)


def realtime_turn_instructions(config: Config, memory_context: str = "") -> str:
    parts = [
        f"{local_context(config)} "
        "Answer the latest user turn. Keep spoken replies concise. "
        "Use background and scheduled-task tools when they fit the user's request."
    ]
    if memory_context.strip():
        parts.append(memory_context.strip())
    return " ".join(parts)


def realtime_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": CLOSE_TOOL_NAME,
            "description": (
                "Close the current realtime session when the user asks to stop, disconnect, sleep, "
                "end the conversation for now, or otherwise stop spending on this live connection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason the session should be closed.",
                    }
                },
                "required": ["reason"],
            },
        },
        {
            "type": "function",
            "name": START_TASK_TOOL_NAME,
            "description": (
                "Start an asynchronous GPT-5.5 task for current web research, code generation, code execution, "
                "calculation, or multi-step analysis. The task runs in the app while the realtime voice session "
                "continues; use list_background_tasks or get_background_task later to report progress or results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task request, including location, date, or constraints when relevant.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional short human-readable title for the task.",
                    },
                    "wakeup_on_complete": {
                        "type": "boolean",
                        "description": (
                            "Set true when the device should speak the final result automatically after the task "
                            "finishes. Set false for silent tasks the user will ask about later."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": LIST_TASKS_TOOL_NAME,
            "description": "List background GPT-5.5 tasks, including running status and latest progress summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_completed": {
                        "type": "boolean",
                        "description": "Include completed, failed, and cancelled tasks as well as running tasks.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": GET_TASK_TOOL_NAME,
            "description": "Get progress, reasoning-summary text, and final result for a background task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task id returned by start_background_task. If omitted, returns the newest task.",
                    },
                    "include_result": {
                        "type": "boolean",
                        "description": "Include the task result text. Use true when reporting a completed task.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": CANCEL_TASK_TOOL_NAME,
            "description": "Cancel a running background GPT-5.5 task when the user no longer wants it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task id returned by start_background_task.",
                    }
                },
                "required": ["task_id"],
            },
        },
        {
            "type": "function",
            "name": ADD_SCHEDULED_TASK_TOOL_NAME,
            "description": (
                "Create a reminder, alarm, or other timed task. Use this when the user wants the device to do "
                "something at a specific future time. Spoken tasks will be rejected during quiet hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short label for the scheduled task, such as 'Wake up' or 'Medication reminder'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What the assistant should say or do when the time arrives. For a spoken reminder, "
                            "write the message to speak."
                        ),
                    },
                    "run_at": {
                        "type": "string",
                        "description": (
                            "ISO local datetime for when it should run, e.g. 2026-06-02T07:30:00. "
                            "Use the current local time from the turn instructions to resolve relative requests."
                        ),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["speak", "background_task"],
                        "description": "Use speak for alarms/reminders; use background_task for silent research/work.",
                    },
                    "repeat": {
                        "type": "string",
                        "enum": ["once", "daily"],
                        "description": "Use daily for repeating alarms or reminders; otherwise use once.",
                    },
                },
                "required": ["title", "prompt", "run_at"],
            },
        },
        {
            "type": "function",
            "name": LIST_SCHEDULED_TASKS_TOOL_NAME,
            "description": "List reminders, alarms, and other timed tasks so the user can review or discuss them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Include completed and skipped items as well as active items.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": DELETE_SCHEDULED_TASK_TOOL_NAME,
            "description": "Remove a reminder, alarm, or other timed task by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Scheduled task id returned by add_scheduled_task or list_scheduled_tasks.",
                    }
                },
                "required": ["task_id"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_SEARCH_TOOL_NAME,
            "description": (
                "Search Max Code's persisted memory before answering questions about previous conversations, "
                "stored user facts, preferences, decisions, or what the assistant remembers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the user's words.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of memory snippets to return.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_LIST_TOOL_NAME,
            "description": "List durable long-term memory entries so the user can review them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_forgotten": {
                        "type": "boolean",
                        "description": "Include entries that were explicitly forgotten.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": MEMORY_ADD_TOOL_NAME,
            "description": (
                "Add a durable memory entry when the user explicitly asks to remember something, or when they "
                "clearly approve saving a stable preference or fact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The durable fact, preference, or instruction to remember.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Short category such as preference, fact, instruction, project, or note.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_UPDATE_TOOL_NAME,
            "description": "Correct an existing durable memory entry when the user says it is wrong or outdated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "Memory entry id returned by memory_list or memory_search.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Replacement memory text.",
                    },
                },
                "required": ["entry_id", "text"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_FORGET_TOOL_NAME,
            "description": (
                "Forget a durable memory entry when the user asks to delete, remove, or forget it. Use entry_id "
                "when available; otherwise use query to match active entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "Memory entry id to forget.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Fallback text to match against active memory entries.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief user-facing reason, if any.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": MEMORY_GET_SOURCE_TOOL_NAME,
            "description": "Read a bounded excerpt from a memory source path returned by memory_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Memory path returned by memory_search, such as MEMORY.md or memory/YYYY-MM-DD.md.",
                    },
                    "from_line": {
                        "type": "integer",
                        "description": "1-based line number to start reading.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to read.",
                    },
                },
                "required": ["path"],
            },
        },
    ]


class BackgroundTaskManager:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key, timeout=config.task_timeout_seconds)
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.RLock()

    def start(self, query: str, title: str = "", wakeup_on_complete: bool = False) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        clean_title = title.strip()
        task = BackgroundTask(
            id=task_id,
            query=query,
            title=clean_title,
            wakeup_on_complete=wakeup_on_complete,
        )
        if clean_title:
            task.progress = f"queued: {clean_title}"
        with self._lock:
            self._prune_locked()
            if len(self._tasks) >= self.config.max_background_tasks:
                return {"ok": False, "error": "too many retained background tasks; cancel or wait for older tasks"}
            self._tasks[task_id] = task
        thread = threading.Thread(target=self._run_task, args=(task_id,), name=f"task-{task_id}", daemon=True)
        thread.start()
        LOGGER.info("started background task %s wakeup=%s: %s", task_id, wakeup_on_complete, query)
        return {"ok": True, "task": self._snapshot(task, include_result=False)}

    def list(self, include_completed: bool = False) -> dict[str, Any]:
        with self._lock:
            tasks = list(self._tasks.values())
            if not include_completed:
                tasks = [task for task in tasks if task.status in {"queued", "running", "cancelling"}]
            tasks.sort(key=lambda task: task.created_at, reverse=True)
            return {
                "ok": True,
                "tasks": [self._snapshot(task, include_result=False) for task in tasks],
                "running_count": sum(
                    1 for task in self._tasks.values() if task.status in {"queued", "running", "cancelling"}
                ),
            }

    def get(self, task_id: str = "", include_result: bool = True) -> dict[str, Any]:
        with self._lock:
            task = self._get_locked(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task id: {task_id or '<latest>'}"}
            return {"ok": True, "task": self._snapshot(task, include_result=include_result)}

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task id: {task_id}"}
            if task.status not in {"queued", "running", "cancelling"}:
                return {"ok": True, "task": self._snapshot(task, include_result=False)}
            task.cancel_event.set()
            self._update_locked(task, status="cancelling", progress="cancellation requested")
            LOGGER.info("cancellation requested for background task %s", task_id)
            return {"ok": True, "task": self._snapshot(task, include_result=False)}

    def has_running(self) -> bool:
        with self._lock:
            return any(task.status in {"queued", "running", "cancelling"} for task in self._tasks.values())

    def pending_wakeups(self, limit: int = 1) -> list[dict[str, Any]]:
        with self._lock:
            tasks = [
                task
                for task in self._tasks.values()
                if task.wakeup_on_complete
                and not task.wakeup_reported
                and task.status in {"completed", "failed"}
            ]
            tasks.sort(key=lambda task: task.completed_at or task.updated_at)
            return [self._snapshot(task, include_result=True) for task in tasks[:limit]]

    def mark_wakeup_reported(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.wakeup_reported = True
                task.updated_at = time.time()

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            self._update_locked(task, status="running", progress="starting GPT-5.5 task")

        stream = None
        try:
            stream = self.client.responses.create(
                model=self.config.task_model,
                instructions=background_task_instructions(self.config),
                input=background_task_prompt(self.config, task.query),
                reasoning={
                    "effort": self.config.task_reasoning_effort,
                    "summary": self.config.task_reasoning_summary,
                },
                tools=self._task_tools(),
                include=self._task_include(),
                stream=True,
                store=False,
                truncation="auto",
                parallel_tool_calls=True,
            )
            for event in stream:
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None:
                        return
                    if task.cancel_event.is_set():
                        close = getattr(stream, "close", None)
                        if close is not None:
                            close()
                        self._finish_locked(task, "cancelled", "cancelled")
                        LOGGER.info("background task %s cancelled", task_id)
                        return
                self._handle_stream_event(task_id, event)

            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None and task.status in {"queued", "running", "cancelling"}:
                    self._finish_locked(task, "completed", "completed")
                    LOGGER.info("background task %s completed", task_id)
        except BaseException as exc:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.error = str(exc)
                    self._finish_locked(task, "failed", "failed")
            LOGGER.exception("background task %s failed", task_id)

    def _task_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "web_search",
                "search_context_size": self.config.web_search_context_size,
                "user_location": {
                    "type": "approximate",
                    "country": self.config.user_country,
                    "city": self.config.user_city,
                    "region": self.config.user_region,
                    "timezone": self.config.user_timezone,
                },
            }
        ]
        if self.config.task_code_execution:
            tools.append(
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"},
                }
            )
        return tools

    def _task_include(self) -> list[str]:
        include = ["web_search_call.action.sources"]
        if self.config.task_code_execution:
            include.append("code_interpreter_call.outputs")
        return include

    def _handle_stream_event(self, task_id: str, event: Any) -> None:
        event_type = getattr(event, "type", "")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in {"queued", "running", "cancelling"}:
                return

            if event_type == "response.created":
                response = getattr(event, "response", None)
                task.response_id = getattr(response, "id", "") or task.response_id
                self._update_locked(task, progress="response created")
            elif event_type == "response.in_progress":
                self._update_locked(task, progress="model working")
            elif event_type == "response.web_search_call.in_progress":
                self._update_locked(task, progress="web search starting")
                self._append_event_locked(task, "web search starting")
            elif event_type == "response.web_search_call.searching":
                self._update_locked(task, progress="searching the web")
                self._append_event_locked(task, "searching the web")
            elif event_type == "response.web_search_call.completed":
                self._update_locked(task, progress="web search completed")
                self._append_event_locked(task, "web search completed")
            elif event_type == "response.code_interpreter_call.in_progress":
                self._update_locked(task, progress="code interpreter starting")
                self._append_event_locked(task, "code interpreter starting")
            elif event_type == "response.code_interpreter_call.interpreting":
                self._update_locked(task, progress="running code")
                self._append_event_locked(task, "running code")
            elif event_type == "response.code_interpreter_call.completed":
                self._update_locked(task, progress="code execution completed")
                self._append_event_locked(task, "code execution completed")
            elif event_type == "response.code_interpreter_call_code.delta":
                code_delta = getattr(event, "delta", "")
                if code_delta:
                    self._update_locked(task, progress=f"writing code: {_truncate(code_delta.strip(), 160)}")
            elif event_type == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    task.reasoning_summary = _truncate(
                        task.reasoning_summary + delta,
                        self.config.task_summary_chars,
                        keep_tail=True,
                    )
                    self._update_locked(task, progress="reasoning")
            elif event_type == "response.reasoning_summary_text.done":
                text = getattr(event, "text", "")
                if text:
                    task.reasoning_summary = _truncate(text, self.config.task_summary_chars, keep_tail=True)
                    self._update_locked(task, progress="reasoning summary updated")
            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    task.result = _truncate(task.result + delta, self.config.task_result_chars, keep_tail=True)
                    self._update_locked(task, progress="writing answer")
            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                final_text = getattr(response, "output_text", "") or extract_responses_output_text(response)
                if final_text and not task.result.strip():
                    task.result = _truncate(final_text, self.config.task_result_chars, keep_tail=True)
                task.response_id = getattr(response, "id", "") or task.response_id
                self._finish_locked(task, "completed", "completed")
                LOGGER.info("background task %s completed", task_id)
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                error = getattr(event, "error", None)
                task.error = str(error or event_type)
                self._finish_locked(task, "failed", "failed")

    def _get_locked(self, task_id: str) -> BackgroundTask | None:
        if task_id:
            return self._tasks.get(task_id)
        if not self._tasks:
            return None
        return max(self._tasks.values(), key=lambda task: task.created_at)

    def _prune_locked(self) -> None:
        if len(self._tasks) < self.config.max_background_tasks:
            return
        done = [
            task
            for task in self._tasks.values()
            if task.status not in {"queued", "running", "cancelling"}
        ]
        done.sort(key=lambda task: task.updated_at)
        while done and len(self._tasks) >= self.config.max_background_tasks:
            task = done.pop(0)
            self._tasks.pop(task.id, None)

    def _snapshot(self, task: BackgroundTask, include_result: bool) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "query": task.query,
            "wakeup_on_complete": task.wakeup_on_complete,
            "wakeup_reported": task.wakeup_reported,
            "status": task.status,
            "progress": task.progress,
            "created_at": _iso_time(task.created_at),
            "updated_at": _iso_time(task.updated_at),
            "completed_at": _iso_time(task.completed_at) if task.completed_at else None,
            "response_id": task.response_id,
            "reasoning_summary": _truncate(task.reasoning_summary, self.config.task_summary_chars, keep_tail=True),
            "result": task.result if include_result else _truncate(task.result, 1200),
            "error": task.error,
            "events": task.events[-12:],
        }

    def _update_locked(self, task: BackgroundTask, status: str | None = None, progress: str | None = None) -> None:
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = progress
        task.updated_at = time.time()

    def _finish_locked(self, task: BackgroundTask, status: str, progress: str) -> None:
        self._update_locked(task, status=status, progress=progress)
        task.completed_at = time.time()

    def _append_event_locked(self, task: BackgroundTask, text: str) -> None:
        task.events.append(f"{_iso_time(time.time())}: {text}")
        if len(task.events) > 50:
            del task.events[:-50]


def background_task_instructions(config: Config) -> str:
    return (
        "You are a background research and code-execution worker for a realtime voice assistant. "
        "Use web_search for current, local, or source-dependent facts. Use code_interpreter for calculations, "
        "small programs, data processing, writing code, or checking code. Interpret relative dates like today, "
        "tomorrow, this morning, and tonight using the user's local timezone from the context. For web results, "
        "prefer sources relevant to the user's location and current local date when the query is local or time-sensitive. "
        "Produce a concise final answer suitable for the voice assistant to summarize aloud, with source names or URLs "
        "when web facts matter. "
        f"{local_context(config)}"
    )


def background_task_prompt(config: Config, query: str) -> str:
    return (
        "This task was started from a live push-to-talk Realtime conversation. "
        "Work independently and stream reasoning summaries so the voice model can report progress. "
        "Use hosted web search for current facts and code_interpreter for calculations, code generation, code execution, "
        "or verification. Treat the following local context as authoritative for 'today', 'now', local weather, events, "
        f"and schedules. {local_context(config)} "
        f"Task: {query}"
    )


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _truncate(text: str, max_chars: int, keep_tail: bool = False) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[truncated]\n"
    if keep_tail:
        return marker + text[-max(0, max_chars - len(marker)) :]
    return text[: max(0, max_chars - len(marker))] + marker


class RealtimeVoiceClient:
    def __init__(self, config: Config):
        self.config = config
        self.player = PcmPlayer(config.audio_playback_device)

    def wait_for_connectivity(self) -> None:
        host = self.config.openai_connectivity_host
        deadline = time.monotonic() + self.config.openai_connectivity_wait_seconds
        last_error: OSError | None = None
        while True:
            try:
                with socket.create_connection((host, 443), timeout=5):
                    return
            except OSError as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"network is not ready for {host}:443") from last_error
                LOGGER.warning("waiting for network/DNS before OpenAI realtime call: %s", exc)
                time.sleep(min(5.0, remaining))

    def respond_to_audio(self, history: Iterable[Message], audio_path: Path) -> RealtimeTurnResult:
        deadline = time.monotonic() + self.config.realtime_max_session_seconds
        ws = self._connect()
        try:
            self._send_session_update(ws)
            self._wait_for_session_updated(ws, deadline)
            self._seed_history(ws, history)
            self._send_audio_turn(ws, audio_path)
            return self._read_response(ws, deadline)
        finally:
            ws.close()

    def _connect(self) -> websocket.WebSocket:
        url = f"wss://api.openai.com/v1/realtime?model={self.config.realtime_model}"
        headers = [
            f"Authorization: Bearer {self.config.openai_api_key}",
            f"OpenAI-Safety-Identifier: {self.config.realtime_safety_identifier}",
        ]
        LOGGER.info("opening realtime websocket with %s", self.config.realtime_model)
        return websocket.create_connection(
            url,
            header=headers,
            sslopt={"ca_certs": certifi.where()},
            timeout=self.config.realtime_response_timeout_seconds,
        )

    def _send_session_update(self, ws: websocket.WebSocket, memory_context: str = "") -> None:
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.config.realtime_model,
            "output_modalities": ["audio"],
            "instructions": realtime_session_instructions(self.config, memory_context),
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.realtime_input_rate,
                    },
                    "turn_detection": None,
                    "transcription": {
                        "model": self.config.realtime_input_transcription_model,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "voice": self.config.realtime_voice,
                },
            },
            "reasoning": {
                "effort": self.config.realtime_reasoning_effort,
            },
            "tools": realtime_tools(),
            "tool_choice": "auto",
        }
        self._send(ws, {"type": "session.update", "session": session})

    def _wait_for_session_updated(self, ws: websocket.WebSocket, deadline: float) -> None:
        while True:
            event = self._recv(ws, deadline)
            event_type = event.get("type")
            if event_type == "session.updated":
                LOGGER.info("realtime session updated")
                return
            if event_type == "session.created":
                continue
            self._raise_for_error(event)

    def _seed_history(self, ws: websocket.WebSocket, history: Iterable[Message]) -> None:
        messages = list(history)[-self.config.realtime_history_messages :]
        if not messages:
            return
        LOGGER.info("seeding realtime session with %d saved messages", len(messages))
        for message in messages:
            self._send(ws, {"type": "conversation.item.create", "item": conversation_item_for_message(message)})

    def _send_audio_turn(self, ws: websocket.WebSocket, audio_path: Path) -> None:
        self._send(ws, {"type": "input_audio_buffer.clear"})
        total_bytes = 0
        for chunk in iter_wav_pcm16_chunks(audio_path, self.config.realtime_input_rate):
            total_bytes += len(chunk)
            self._send(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )
        if total_bytes == 0:
            raise RuntimeError(f"recording has no PCM audio: {audio_path}")
        LOGGER.info("sent %d bytes of PCM audio to realtime session", total_bytes)
        self._send(ws, {"type": "input_audio_buffer.commit"})
        self._send(
            ws,
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": realtime_turn_instructions(self.config),
                },
            },
        )

    def _read_response(self, ws: websocket.WebSocket, deadline: float) -> RealtimeTurnResult:
        input_transcript_parts: list[str] = []
        input_transcript_final = ""
        output_transcript_parts: list[str] = []
        output_transcript_final = ""
        response_done_text = ""
        requested_close = False

        with self.player.open_stream(rate=24000, channels=1) as audio:
            while True:
                event = self._recv(ws, deadline)
                event_type = event.get("type")
                self._raise_for_error(event)

                if event_type == "conversation.item.input_audio_transcription.delta":
                    input_transcript_parts.append(event.get("delta", ""))
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    input_transcript_final = event.get("transcript", "").strip()
                    if input_transcript_final:
                        LOGGER.info("realtime input transcript: %s", input_transcript_final)
                elif event_type == "response.output_audio.delta":
                    audio.write(base64.b64decode(event.get("delta", "")))
                elif event_type == "response.output_audio_transcript.delta":
                    output_transcript_parts.append(event.get("delta", ""))
                elif event_type == "response.output_audio_transcript.done":
                    output_transcript_final = event.get("transcript", "").strip()
                    if output_transcript_final:
                        LOGGER.info("realtime assistant transcript: %s", output_transcript_final)
                elif event_type == "response.done":
                    response = event.get("response", {})
                    requested_close = response_requested_close(response)
                    response_done_text = extract_response_text(response)
                    break

        user_text = input_transcript_final or "".join(input_transcript_parts).strip()
        assistant_text = output_transcript_final or "".join(output_transcript_parts).strip() or response_done_text
        if requested_close:
            LOGGER.info("model requested realtime session close")
        if not user_text:
            LOGGER.warning("realtime session completed without an input transcript")
        if not assistant_text and not requested_close:
            raise RuntimeError("realtime model returned no assistant transcript")
        return RealtimeTurnResult(user_text=user_text, assistant_text=assistant_text, requested_close=requested_close)

    def _send(self, ws: websocket.WebSocket, event: dict[str, Any]) -> None:
        ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("realtime session exceeded local maximum duration")
            ws.settimeout(min(5.0, remaining))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise RuntimeError(f"unexpected realtime event: {event!r}")
            return event

    def _raise_for_error(self, event: dict[str, Any]) -> None:
        if event.get("type") != "error":
            return
        error = event.get("error", event)
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"realtime API error: {message}")


class RealtimeConversationSession:
    def __init__(
        self,
        config: Config,
        scheduled_tasks: ScheduledTaskStore | None = None,
        memory: MemoryStore | None = None,
    ):
        self.config = config
        self.player = PcmPlayer(config.audio_playback_device)
        self.tasks = BackgroundTaskManager(config)
        self.scheduled_tasks = scheduled_tasks or ScheduledTaskStore(config)
        self.memory = memory or MemoryStore(config)
        self.memory.ensure_workspace()
        self._ws: websocket.WebSocket | None = None
        self._receiver_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._errors: queue.Queue[BaseException] = queue.Queue()
        self._completed_turns: queue.Queue[RealtimeTurnResult] = queue.Queue()

        self._session_started_at = 0.0
        self._last_activity_at = 0.0
        self._closing = False
        self._seeded = False

        self._recorder: RawPcmRecorder | None = None
        self._capture_queue: queue.Queue[bytes | None] | None = None
        self._capture_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None
        self._recording_started_at = 0.0
        self._capture_bytes = 0
        self._sent_audio_bytes = 0

        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._tool_active = False
        self._tool_generation = 0
        self._turn_generation = 0
        self._pending_input_generation = 0
        self._active_response_id = ""
        self._playback: PcmOutputStream | None = None
        self._playback_closing = False
        self._output_item_id = ""
        self._output_audio_bytes = 0
        self._input_transcript_parts: list[str] = []
        self._input_transcript_final = ""
        self._output_transcript_parts: list[str] = []
        self._output_transcript_final = ""

    def begin_turn(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._recorder is not None:
                raise RuntimeError("realtime recording is already active")
            self._last_activity_at = time.monotonic()
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._recorder = RawPcmRecorder(self.config)
            self._capture_queue = queue.Queue()
            self._capture_bytes = 0
            self._sent_audio_bytes = 0
            self._recording_started_at = time.monotonic()
            recorder = self._recorder
            capture_queue = self._capture_queue

        LOGGER.info("starting realtime turn capture")
        recorder.start()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(recorder, capture_queue),
            name="realtime-capture",
            daemon=True,
        )
        self._capture_thread.start()

        try:
            self.ensure_open(history)
            self._send({"type": "input_audio_buffer.clear"})
            self._sender_thread = threading.Thread(
                target=self._send_audio_loop,
                args=(capture_queue,),
                name="realtime-audio-sender",
                daemon=True,
            )
            self._sender_thread.start()
            LOGGER.info("realtime audio sender started")
        except BaseException:
            self._stop_capture_threads()
            raise

    def stop_recording(self) -> float:
        return self._stop_capture_threads()

    def commit_recording(self) -> None:
        self._raise_background_error()
        with self._state_lock:
            sent_audio_bytes = self._sent_audio_bytes
        if sent_audio_bytes == 0:
            self.clear_pending_input()
            raise RuntimeError("no audio was captured for realtime turn")
        self._send({"type": "input_audio_buffer.commit"})
        with self._state_lock:
            self._turn_generation += 1
            self._pending_input_generation = self._turn_generation
            self._response_pending = True
            self._response_create_pending = False
            self._waiting_for_input_transcript = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        LOGGER.info("committed realtime turn: %d PCM bytes; waiting for transcript", sent_audio_bytes)

    def end_turn(self, commit: bool) -> float:
        duration = self._stop_capture_threads()
        if not commit:
            self.clear_pending_input()
            return duration
        self.commit_recording()
        LOGGER.info("realtime recording duration: %.3fs", duration)
        return duration

    def clear_pending_input(self) -> None:
        if self.is_open:
            self._send({"type": "input_audio_buffer.clear"})

    def check_health(self) -> None:
        self._raise_background_error()

    def ensure_open(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._ws is not None and self._receiver_thread is not None and self._receiver_thread.is_alive():
                return
        ws = RealtimeVoiceClient(self.config)._connect()
        deadline = time.monotonic() + self.config.realtime_response_timeout_seconds
        try:
            RealtimeVoiceClient(self.config)._send_session_update(ws, self.memory.bootstrap_context())
            RealtimeVoiceClient(self.config)._wait_for_session_updated(ws, deadline)
            RealtimeVoiceClient(self.config)._seed_history(ws, history)
        except BaseException:
            ws.close()
            raise

        with self._state_lock:
            self._ws = ws
            self._closing = False
            self._seeded = True
            self._session_started_at = time.monotonic()
            self._last_activity_at = self._session_started_at
        self._receiver_thread = threading.Thread(target=self._receive_loop, name="realtime-receiver", daemon=True)
        self._receiver_thread.start()
        LOGGER.info("persistent realtime session open")

    def close_if_idle(self) -> None:
        with self._state_lock:
            if self._ws is None or self._is_busy_locked():
                return
            idle_for = time.monotonic() - self._last_activity_at
            if idle_for < self.config.realtime_idle_timeout_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs idle", idle_for)
        self.close()

    def close_if_too_old(self) -> None:
        with self._state_lock:
            if self._ws is None:
                return
            age = time.monotonic() - self._session_started_at
            if age < self.config.realtime_max_session_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs hard max age", age)
        self.close()

    def cool_down_if_silent(self) -> None:
        cooldown = self.config.realtime_silent_cooldown_seconds
        if cooldown <= 0:
            return
        should_start_response = False
        fallback_query = ""
        generation = 0
        with self._state_lock:
            if self._ws is None or self._recorder is not None:
                return
            if not (self._response_pending or self._response_active or self._tool_active):
                return
            if self._playback is not None or self._output_transcript_parts or self._output_transcript_final:
                return
            wait_started_at = self._response_wait_started_at or self._last_activity_at
            silent_for = time.monotonic() - wait_started_at
            if self._waiting_for_input_transcript and silent_for >= min(2.0, cooldown):
                LOGGER.info("starting realtime response before final transcript after %.1fs", silent_for)
                self._waiting_for_input_transcript = False
                generation = self._pending_input_generation
                fallback_query = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
                should_start_response = True
            elif self._waiting_for_input_transcript:
                return
            if not should_start_response and silent_for < cooldown:
                return
            if not should_start_response:
                LOGGER.info("cooling down silent realtime response after %.1fs", silent_for)
                self._interrupt_response_locked()
                self._reset_turn_text_locked()
        if should_start_response:
            self._start_response_for_current_turn(fallback_query, generation)

    def close(self) -> None:
        receiver_thread: threading.Thread | None
        with self._state_lock:
            self._closing = True
            ws = self._ws
            self._ws = None
            receiver_thread = self._receiver_thread
            self._receiver_thread = None
            self._response_pending = False
            self._response_create_pending = False
            self._response_active = False
            self._response_wait_started_at = 0.0
            self._waiting_for_input_transcript = False
            self._tool_active = False
            self._tool_generation += 1
            self._abort_playback_locked()
            self._reset_turn_text_locked()
        self._stop_capture_threads()
        if ws is not None:
            ws.close()
        if receiver_thread is not None and receiver_thread is not threading.current_thread():
            receiver_thread.join(timeout=2)
        with self._state_lock:
            self._closing = False
            self._seeded = False
        LOGGER.info("persistent realtime session closed")

    @property
    def is_open(self) -> bool:
        with self._state_lock:
            return self._ws is not None

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recorder is not None

    @property
    def is_responding(self) -> bool:
        with self._state_lock:
            return self._response_pending or self._response_active or self._tool_active or self._playback is not None

    @property
    def is_voice_busy(self) -> bool:
        with self._state_lock:
            return self._is_busy_locked()

    def pop_completed_turns(self) -> list[RealtimeTurnResult]:
        turns: list[RealtimeTurnResult] = []
        while True:
            try:
                turns.append(self._completed_turns.get_nowait())
            except queue.Empty:
                return turns

    def _capture_loop(self, recorder: RawPcmRecorder, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = recorder.read(4096)
                if not chunk:
                    return
                with self._state_lock:
                    self._capture_bytes += len(chunk)
                capture_queue.put(chunk)
        except BaseException as exc:
            self._errors.put(exc)
        finally:
            capture_queue.put(None)

    def _send_audio_loop(self, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = capture_queue.get()
                if chunk is None:
                    return
                self._send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                with self._state_lock:
                    self._sent_audio_bytes += len(chunk)
                    self._last_activity_at = time.monotonic()
        except BaseException as exc:
            self._errors.put(exc)

    def _stop_capture_threads(self) -> float:
        with self._state_lock:
            recorder = self._recorder
            self._recorder = None
            capture_thread = self._capture_thread
            sender_thread = self._sender_thread
            self._capture_thread = None
            self._sender_thread = None
        duration = recorder.stop() if recorder is not None else 0.0
        if capture_thread is not None:
            capture_thread.join(timeout=2)
        if sender_thread is not None:
            sender_thread.join(timeout=5)
        self._raise_background_error()
        return duration

    def _receive_loop(self) -> None:
        try:
            while True:
                with self._state_lock:
                    ws = self._ws
                    closing = self._closing
                    deadline = self._session_started_at + self.config.realtime_max_session_seconds
                if ws is None or closing:
                    return
                try:
                    event = self._recv(ws, deadline)
                except websocket.WebSocketConnectionClosedException:
                    with self._state_lock:
                        if self._closing or self._ws is None:
                            return
                    raise
                should_close = self._handle_event(event)
                if should_close:
                    self.close()
                    return
        except BaseException as exc:
            with self._state_lock:
                closing = self._closing
            if not closing:
                self._errors.put(exc)

    def _handle_event(self, event: dict[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == "error":
            if event_is_ignorable_control_error(event):
                LOGGER.warning("ignoring realtime control error: %s", event)
                return False
            RealtimeVoiceClient(self.config)._raise_for_error(event)

        audio_chunk: bytes | None = None
        with self._state_lock:
            self._last_activity_at = time.monotonic()

            if event_type == "conversation.item.input_audio_transcription.delta":
                self._input_transcript_parts.append(event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._input_transcript_final = event.get("transcript", "").strip()
                if self._input_transcript_final:
                    LOGGER.info("realtime input transcript: %s", self._input_transcript_final)
                should_start_response = self._waiting_for_input_transcript
                self._waiting_for_input_transcript = False
                generation = self._pending_input_generation
                transcript = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
                if should_start_response:
                    threading.Thread(
                        target=self._start_response_for_current_turn,
                        args=(transcript, generation),
                        name="realtime-active-memory",
                        daemon=True,
                    ).start()
            elif event_type == "response.created":
                response = event.get("response", {})
                if self._recorder is not None or not self._response_create_pending:
                    LOGGER.info("ignoring stale realtime response.created while not waiting for output")
                    return False
                self._response_create_pending = False
                self._response_pending = False
                self._response_active = True
                self._active_response_id = response.get("id", "")
                if not self._response_wait_started_at:
                    self._response_wait_started_at = time.monotonic()
            elif event_type in {"response.output_item.created", "response.output_item.added"}:
                if not self._is_current_response_event_locked(event):
                    return False
                item = event.get("item", {})
                if item.get("type") == "message":
                    self._output_item_id = item.get("id", "")
                    self._output_audio_bytes = 0
            elif event_type == "response.output_audio.delta":
                if not self._is_current_response_event_locked(event):
                    return False
                self._response_wait_started_at = 0.0
                audio_chunk = base64.b64decode(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.delta":
                if not self._is_current_response_event_locked(event):
                    return False
                self._response_wait_started_at = 0.0
                self._output_transcript_parts.append(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.done":
                if not self._is_current_response_event_locked(event):
                    return False
                self._output_transcript_final = event.get("transcript", "").strip()
                if self._output_transcript_final:
                    LOGGER.info("realtime assistant transcript: %s", self._output_transcript_final)
            elif event_type == "response.done":
                if not self._is_current_response_event_locked(event):
                    return False
                return self._complete_response_locked(event.get("response", {}))
        if audio_chunk:
            self._write_output_audio(audio_chunk)
        return False

    def _is_current_response_event_locked(self, event: dict[str, Any]) -> bool:
        if self._recorder is not None:
            LOGGER.info("ignoring realtime response event while recording: %s", event.get("type"))
            return False
        if not self._response_active:
            LOGGER.info("ignoring realtime response event without an active response: %s", event.get("type"))
            return False
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        response_id = str(event.get("response_id") or response.get("id") or "").strip()
        if response_id and self._active_response_id and response_id != self._active_response_id:
            LOGGER.info("ignoring stale realtime response event for %s", response_id)
            return False
        if response_id and not self._active_response_id and event.get("type") != "response.created":
            LOGGER.info("ignoring realtime response event before active response id is known: %s", response_id)
            return False
        return True

    def _complete_response_locked(self, response: dict[str, Any]) -> bool:
        status = response.get("status", "")
        usage = response.get("usage")
        if usage:
            LOGGER.info("realtime response usage: %s", usage)
        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._active_response_id = ""
        self._close_playback_locked(check=True)

        if status == "cancelled":
            LOGGER.info("realtime response cancelled")
            self._reset_turn_text_locked()
            return False

        tool_calls = response_function_calls(response)
        async_tool_calls = [call for call in tool_calls if call.get("name") in ASYNC_TASK_TOOL_NAMES]
        if async_tool_calls:
            self._tool_active = True
            self._response_wait_started_at = time.monotonic()
            self._tool_generation += 1
            generation = self._tool_generation
            threading.Thread(
                target=self._run_realtime_tool_calls,
                args=(generation, async_tool_calls),
                name="realtime-tool-calls",
                daemon=True,
            ).start()
            return False

        requested_close = response_requested_close(response)
        user_text = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
        assistant_text = (
            self._output_transcript_final
            or "".join(self._output_transcript_parts).strip()
            or extract_response_text(response)
        )
        if user_text or assistant_text or requested_close:
            self._completed_turns.put(
                RealtimeTurnResult(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    requested_close=requested_close,
                )
            )
        self._reset_turn_text_locked()
        if requested_close:
            LOGGER.info("model requested realtime session close")
        return requested_close

    def _run_realtime_tool_calls(self, generation: int, calls: list[dict[str, Any]]) -> None:
        try:
            for call in calls:
                output = self._execute_realtime_tool_call(call)
                with self._state_lock:
                    if generation != self._tool_generation or self._ws is None or self._closing:
                        LOGGER.info("discarding stale realtime tool result")
                        return
                self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call.get("call_id", ""),
                            "output": json.dumps(output, ensure_ascii=False),
                        },
                    }
                )
            with self._state_lock:
                if generation != self._tool_generation or self._ws is None or self._closing:
                    return
                self._tool_active = False
                self._response_pending = True
                self._response_create_pending = True
                self._response_wait_started_at = time.monotonic()
            self._send(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "instructions": (
                            f"{realtime_turn_instructions(self.config)} "
                            "If a background task is still running, tell the user briefly and do not wait."
                        ),
                    },
                }
            )
        except BaseException as exc:
            self._errors.put(exc)

    def _start_response_for_current_turn(self, query: str = "", generation: int = 0) -> None:
        try:
            with self._state_lock:
                if (
                    self._ws is None
                    or self._closing
                    or self._recorder is not None
                    or generation != self._pending_input_generation
                ):
                    return
            memory_context = self.memory.active_context(query) if query.strip() else ""
            with self._state_lock:
                if (
                    self._ws is None
                    or self._closing
                    or self._recorder is not None
                    or generation != self._pending_input_generation
                ):
                    LOGGER.info("discarding stale realtime response start")
                    return
                self._response_pending = True
                self._response_create_pending = True
                self._response_wait_started_at = time.monotonic()
                self._last_activity_at = time.monotonic()
                self._send(
                    {
                        "type": "response.create",
                        "response": {
                            "output_modalities": ["audio"],
                            "instructions": realtime_turn_instructions(self.config, memory_context),
                        },
                    }
                )
            LOGGER.info(
                "started realtime response with active memory chars=%d",
                len(memory_context),
            )
        except BaseException as exc:
            self._errors.put(exc)

    def _execute_realtime_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name", "")
        arguments = parse_tool_arguments(call)
        LOGGER.info("running realtime tool %s with args keys=%s", name, sorted(arguments.keys()))
        if name in {START_TASK_TOOL_NAME, WEB_SEARCH_TOOL_NAME}:
            query = string_argument(arguments, "query")
            title = string_argument(arguments, "title")
            wakeup_on_complete = parse_bool_argument(arguments.get("wakeup_on_complete"), default=False)
            return self.tasks.start(query, title=title, wakeup_on_complete=wakeup_on_complete)
        if name == LIST_TASKS_TOOL_NAME:
            return self.tasks.list(
                include_completed=parse_bool_argument(arguments.get("include_completed"), default=False)
            )
        if name == GET_TASK_TOOL_NAME:
            return self.tasks.get(
                task_id=string_argument(arguments, "task_id"),
                include_result=parse_bool_argument(arguments.get("include_result"), default=True),
            )
        if name == CANCEL_TASK_TOOL_NAME:
            return self.tasks.cancel(string_argument(arguments, "task_id"))
        if name == ADD_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled_tasks.add(
                title=string_argument(arguments, "title"),
                prompt=string_argument(arguments, "prompt"),
                run_at=string_argument(arguments, "run_at"),
                action=string_argument(arguments, "action", "speak"),
                repeat=string_argument(arguments, "repeat", "once"),
            )
        if name == LIST_SCHEDULED_TASKS_TOOL_NAME:
            return self.scheduled_tasks.list(
                include_inactive=parse_bool_argument(arguments.get("include_inactive"), default=False)
            )
        if name == DELETE_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled_tasks.delete(string_argument(arguments, "task_id"))
        if name == MEMORY_SEARCH_TOOL_NAME:
            return self.memory.search(
                string_argument(arguments, "query"),
                max_results=int_argument(arguments, "max_results", 5),
            )
        if name == MEMORY_LIST_TOOL_NAME:
            return self.memory.list_entries(
                include_forgotten=parse_bool_argument(arguments.get("include_forgotten"), default=False)
            )
        if name == MEMORY_ADD_TOOL_NAME:
            return self.memory.add_entry(
                text=string_argument(arguments, "text"),
                kind=string_argument(arguments, "kind", "note"),
                source="user",
            )
        if name == MEMORY_UPDATE_TOOL_NAME:
            return self.memory.update_entry(
                entry_id=string_argument(arguments, "entry_id"),
                text=string_argument(arguments, "text"),
            )
        if name == MEMORY_FORGET_TOOL_NAME:
            return self.memory.forget_entry(
                entry_id=string_argument(arguments, "entry_id"),
                query=string_argument(arguments, "query"),
                reason=string_argument(arguments, "reason"),
            )
        if name == MEMORY_GET_SOURCE_TOOL_NAME:
            return self.memory.get_source(
                path=string_argument(arguments, "path"),
                from_line=int_argument(arguments, "from_line", 1),
                lines=int_argument(arguments, "lines", 12),
            )
        return {"ok": False, "error": f"unknown realtime tool: {name}"}

    def trigger_scheduled_speech(self, history: Iterable[Message], title: str, prompt: str) -> bool:
        self._raise_background_error()
        title = title.strip() or "Scheduled task"
        prompt = prompt.strip()
        if not prompt:
            LOGGER.warning("not starting scheduled speech with empty prompt")
            return False
        with self._state_lock:
            if self._is_busy_locked():
                return False
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._input_transcript_final = f"[scheduled task: {title}] {prompt}"

        self.ensure_open(history)
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "A scheduled item is due now. "
                                f"Title: {title}. Message/request: {prompt}"
                            ),
                        }
                    ],
                },
            }
        )
        with self._state_lock:
            self._response_pending = True
            self._response_create_pending = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": (
                        f"{local_context(self.config)} "
                        "This scheduled item is due now. Speak naturally and concisely. "
                        "Do not use implementation terminology."
                    ),
                },
            }
        )
        LOGGER.info("started scheduled speech: %s", title)
        return True

    def pending_background_wakeups(self, limit: int = 1) -> list[dict[str, Any]]:
        return self.tasks.pending_wakeups(limit=limit)

    def mark_background_wakeup_reported(self, task_id: str) -> None:
        self.tasks.mark_wakeup_reported(task_id)

    def trigger_background_task_wakeup(self, history: Iterable[Message], task: dict[str, Any]) -> bool:
        self._raise_background_error()
        task_id = str(task.get("id") or "").strip()
        title = str(task.get("title") or task.get("progress") or task_id or "Background task").strip()
        if not task_id:
            return False
        with self._state_lock:
            if self._is_busy_locked():
                return False
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._input_transcript_final = f"[background task completed: {title}]"

        self.ensure_open(history)
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "A background task has finished and was marked to wake the user. "
                                "Summarize the result aloud now. Task JSON: "
                                f"{json.dumps(task, ensure_ascii=False)}"
                            ),
                        }
                    ],
                },
            }
        )
        with self._state_lock:
            self._response_pending = True
            self._response_create_pending = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": (
                        f"{local_context(self.config)} "
                        "A background task finished. Briefly tell the user the result or failure. "
                        "Do not mention internal tool names unless useful."
                    ),
                },
            }
        )
        LOGGER.info("started background task wakeup for %s", task_id)
        return True

    def _write_output_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._state_lock:
            if self._recorder is not None:
                self._output_audio_bytes += len(chunk)
                return
            if self._playback is None:
                self._playback = self.player.open_stream(rate=24000, channels=1)
                self._playback.__enter__()
                self._playback_closing = False
            if self._playback_closing:
                return
            playback = self._playback
        playback.write(chunk)
        with self._state_lock:
            if self._playback is playback and not self._playback_closing:
                self._output_audio_bytes += len(chunk)

    def _interrupt_response_locked(self) -> None:
        ws_is_open = self._ws is not None
        self._abort_playback_locked()
        if ws_is_open and (self._response_pending or self._response_active):
            self._send(
                {
                    "event_id": f"cancel_{time.monotonic_ns()}",
                    "type": "response.cancel",
                }
            )
        if ws_is_open and self._output_item_id and self._output_audio_bytes:
            audio_end_ms = int(self._output_audio_bytes / 2 / 24000 * 1000)
            self._send(
                {
                    "event_id": f"truncate_{time.monotonic_ns()}",
                    "type": "conversation.item.truncate",
                    "item_id": self._output_item_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, audio_end_ms),
                }
            )
        elif self._output_item_id and self._output_audio_bytes:
            LOGGER.info("skipping audio truncate because realtime session is already closed")
        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._active_response_id = ""
        self._tool_active = False
        self._tool_generation += 1
        self._turn_generation += 1
        self._pending_input_generation = self._turn_generation
        self._output_item_id = ""
        self._output_audio_bytes = 0

    def _abort_playback_locked(self) -> None:
        playback = self._playback
        if playback is None:
            return
        self._playback = None
        self._playback_closing = False
        playback.abort()
        self._close_playback_in_background(playback, check=False)

    def _close_playback_in_background(self, playback: PcmOutputStream, check: bool) -> None:
        def close_playback() -> None:
            try:
                playback.close(check=check)
            except BaseException:
                LOGGER.exception("PCM playback close failed")
            finally:
                with self._state_lock:
                    if self._playback is playback:
                        self._playback = None
                        self._playback_closing = False

        threading.Thread(target=close_playback, name="pcm-playback-close", daemon=True).start()

    def _close_playback_locked(self, check: bool) -> None:
        playback = self._playback
        if playback is None or self._playback_closing:
            return
        self._playback_closing = True
        self._close_playback_in_background(playback, check=check)

    def _reset_turn_text_locked(self) -> None:
        self._input_transcript_parts = []
        self._input_transcript_final = ""
        self._output_transcript_parts = []
        self._output_transcript_final = ""
        self._output_item_id = ""
        self._output_audio_bytes = 0

    def _is_busy_locked(self) -> bool:
        return (
            self._recorder is not None
            or self._response_pending
            or self._response_active
            or self._tool_active
            or self._playback is not None
        )

    def _send(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            ws = self._ws
        if ws is None:
            raise RuntimeError("realtime session is not open")
        with self._send_lock:
            ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        return RealtimeVoiceClient(self.config)._recv(ws, deadline)

    def _raise_background_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("realtime background worker failed") from error


def iter_wav_pcm16_chunks(path: Path, expected_rate: int, chunk_frames: int = 4096) -> Iterator[bytes]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        if channels != 1:
            raise RuntimeError(f"realtime audio must be mono; got {channels} channels in {path}")
        if sample_width != 2:
            raise RuntimeError(f"realtime audio must be 16-bit PCM; got sample width {sample_width} in {path}")
        if rate != expected_rate:
            raise RuntimeError(f"realtime audio must be {expected_rate} Hz; got {rate} Hz in {path}")
        while True:
            data = wav.readframes(chunk_frames)
            if not data:
                return
            yield data


def conversation_item_for_message(message: Message) -> dict[str, Any]:
    content_type = "input_text" if message.role == "user" else "output_text"
    return {
        "type": "message",
        "role": message.role,
        "content": [
            {
                "type": content_type,
                "text": message.content,
            }
        ],
    }


def response_requested_close(response: dict[str, Any]) -> bool:
    return bool(response_function_calls(response, CLOSE_TOOL_NAME))


def response_function_calls(response: dict[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output", [])
        if item.get("type") == "function_call" and (name is None or item.get("name") == name)
    ]


def parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def string_argument(arguments: dict[str, Any], name: str, default: str = "") -> str:
    value = arguments.get(name, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def int_argument(arguments: dict[str, Any], name: str, default: int) -> int:
    value = arguments.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def parse_bool_argument(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def extract_responses_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def event_is_ignorable_control_error(event: dict[str, Any]) -> bool:
    error = event.get("error", {})
    if not isinstance(error, dict):
        return False
    event_id = str(error.get("event_id") or event.get("event_id") or "")
    return event_id.startswith(("cancel_", "truncate_"))
