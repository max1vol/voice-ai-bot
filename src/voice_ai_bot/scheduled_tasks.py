from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .config import Config


@dataclass(frozen=True)
class ScheduledTask:
    id: str
    title: str
    prompt: str
    run_at: str
    action: str
    repeat: str
    status: str
    created_at: str
    updated_at: str
    last_started_at: str = ""
    skipped_reason: str = ""


class ScheduledTaskStore:
    def __init__(self, config: Config):
        self.path = config.scheduled_tasks_file
        self.timezone_name = config.user_timezone
        self.quiet_start = _parse_hhmm(config.schedule_quiet_start)
        self.quiet_end = _parse_hhmm(config.schedule_quiet_end)
        self._lock = threading.RLock()

    def add(
        self,
        *,
        title: str,
        prompt: str,
        run_at: str,
        action: str = "speak",
        repeat: str = "once",
    ) -> dict[str, Any]:
        title = title.strip() or "Scheduled task"
        prompt = prompt.strip()
        if not prompt:
            return {"ok": False, "error": "prompt is required"}
        action = (action or "speak").strip()
        if action not in {"speak", "background_task"}:
            return {"ok": False, "error": "action must be 'speak' or 'background_task'"}
        repeat = (repeat or "once").strip()
        if repeat not in {"once", "daily"}:
            return {"ok": False, "error": "repeat must be 'once' or 'daily'"}

        try:
            run_dt = parse_local_datetime(run_at, self.timezone_name)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        if action == "speak" and self.is_quiet_time(run_dt):
            return {
                "ok": False,
                "error": (
                    "spoken scheduled tasks cannot start during quiet hours "
                    f"{self.quiet_start.strftime('%H:%M')}-{self.quiet_end.strftime('%H:%M')} "
                    f"{self.timezone_name}"
                ),
            }

        with self._lock:
            now = _now_utc()
            task = ScheduledTask(
                id=f"sched_{uuid.uuid4().hex[:10]}",
                title=title,
                prompt=prompt,
                run_at=run_dt.isoformat(),
                action=action,
                repeat=repeat,
                status="active",
                created_at=now,
                updated_at=now,
            )
            tasks = self._load_tasks()
            tasks.append(task)
            self._save_tasks(tasks)
        return {"ok": True, "task": task_snapshot(task)}

    def list(self, include_inactive: bool = False) -> dict[str, Any]:
        with self._lock:
            tasks = self._load_tasks()
            if not include_inactive:
                tasks = [task for task in tasks if task.status == "active"]
            tasks.sort(key=lambda task: task.run_at)
            return {"ok": True, "tasks": [task_snapshot(task) for task in tasks]}

    def delete(self, task_id: str) -> dict[str, Any]:
        task_id = task_id.strip()
        if not task_id:
            return {"ok": False, "error": "task_id is required"}
        with self._lock:
            tasks = self._load_tasks()
            kept = [task for task in tasks if task.id != task_id]
            if len(kept) == len(tasks):
                return {"ok": False, "error": f"unknown scheduled task id: {task_id}"}
            self._save_tasks(kept)
        return {"ok": True, "deleted_task_id": task_id}

    def due(self, now: datetime | None = None, limit: int = 1) -> list[ScheduledTask]:
        now = _coerce_local(now or datetime.now(ZoneInfo(self.timezone_name)), self.timezone_name)
        with self._lock:
            tasks = self._load_tasks()
            due_tasks: list[ScheduledTask] = []
            updated: list[ScheduledTask] = []
            changed = False

            for task in tasks:
                if task.status != "active":
                    updated.append(task)
                    continue
                run_at = parse_local_datetime(task.run_at, self.timezone_name)
                if run_at > now:
                    updated.append(task)
                    continue
                if task.action == "speak" and self.is_quiet_time(now):
                    updated.append(self._skip_or_advance(task, now, "quiet_hours"))
                    changed = True
                    continue
                if len(due_tasks) < limit:
                    due_tasks.append(task)
                updated.append(task)

            if changed:
                self._save_tasks(updated)
            return due_tasks

    def mark_started(self, task_id: str, now: datetime | None = None) -> None:
        now = _coerce_local(now or datetime.now(ZoneInfo(self.timezone_name)), self.timezone_name)
        with self._lock:
            tasks = self._load_tasks()
            updated: list[ScheduledTask] = []
            for task in tasks:
                if task.id != task_id:
                    updated.append(task)
                    continue
                if task.repeat == "daily":
                    next_run = parse_local_datetime(task.run_at, self.timezone_name)
                    while next_run <= now:
                        next_run += timedelta(days=1)
                    updated.append(
                        _replace_task(
                            task,
                            run_at=next_run.isoformat(),
                            updated_at=_now_utc(),
                            last_started_at=now.isoformat(),
                            skipped_reason="",
                        )
                    )
                else:
                    updated.append(
                        _replace_task(
                            task,
                            status="completed",
                            updated_at=_now_utc(),
                            last_started_at=now.isoformat(),
                            skipped_reason="",
                        )
                    )
            self._save_tasks(updated)

    def is_quiet_time(self, dt: datetime) -> bool:
        local_dt = _coerce_local(dt, self.timezone_name)
        local_time = local_dt.time().replace(second=0, microsecond=0)
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= local_time < self.quiet_end
        return local_time >= self.quiet_start or local_time < self.quiet_end

    def quiet_window_text(self) -> str:
        return f"{self.quiet_start.strftime('%H:%M')}-{self.quiet_end.strftime('%H:%M')} {self.timezone_name}"

    def _skip_or_advance(self, task: ScheduledTask, now: datetime, reason: str) -> ScheduledTask:
        if task.repeat == "daily":
            next_run = parse_local_datetime(task.run_at, self.timezone_name)
            while next_run <= now:
                next_run += timedelta(days=1)
            return _replace_task(
                task,
                run_at=next_run.isoformat(),
                updated_at=_now_utc(),
                last_started_at=now.isoformat(),
                skipped_reason=reason,
            )
        return _replace_task(
            task,
            status="skipped",
            updated_at=_now_utc(),
            last_started_at=now.isoformat(),
            skipped_reason=reason,
        )

    def _load_tasks(self) -> list[ScheduledTask]:
        return self._load_tasks_from_payload(self._load_payload())

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "tasks": []}
        with self.path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {"version": 1, "tasks": []}

    def _load_tasks_from_payload(self, payload: dict[str, Any]) -> list[ScheduledTask]:
        tasks: list[ScheduledTask] = []
        for item in payload.get("tasks", []):
            if not isinstance(item, dict):
                continue
            try:
                tasks.append(
                    ScheduledTask(
                        id=str(item["id"]),
                        title=str(item.get("title") or "Scheduled task"),
                        prompt=str(item.get("prompt") or ""),
                        run_at=str(item["run_at"]),
                        action=str(item.get("action") or "speak"),
                        repeat=str(item.get("repeat") or "once"),
                        status=str(item.get("status") or "active"),
                        created_at=str(item.get("created_at") or _now_utc()),
                        updated_at=str(item.get("updated_at") or _now_utc()),
                        last_started_at=str(item.get("last_started_at") or ""),
                        skipped_reason=str(item.get("skipped_reason") or ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return tasks

    def _save_tasks(self, tasks: list[ScheduledTask]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _now_utc(),
            "timezone": self.timezone_name,
            "quiet_hours": {
                "start": self.quiet_start.strftime("%H:%M"),
                "end": self.quiet_end.strftime("%H:%M"),
            },
            "tasks": [asdict(task) for task in tasks],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, self.path)


def parse_local_datetime(value: str, timezone_name: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("run_at is required")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("run_at must be an ISO datetime, for example 2026-06-02T07:30:00") from exc
    return _coerce_local(parsed, timezone_name)


def task_snapshot(task: ScheduledTask) -> dict[str, str]:
    return asdict(task)


def _coerce_local(dt: datetime, timezone_name: str) -> datetime:
    zone = ZoneInfo(timezone_name)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=zone)
    return dt.astimezone(zone)


def _parse_hhmm(value: str) -> time:
    try:
        hour_text, minute_text = value.strip().split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))
    except Exception as exc:
        raise ValueError(f"invalid HH:MM time: {value}") from exc


def _replace_task(task: ScheduledTask, **changes: str) -> ScheduledTask:
    data = asdict(task)
    data.update(changes)
    return ScheduledTask(**data)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
