from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import signal
import socket
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .camera import CameraCapture, CameraError, CameraSnapshot
from .config import Config
from .daemon import configure_logging
from .memory import MemoryStore
from .scheduled_tasks import ScheduledTaskStore

LOGGER = logging.getLogger(__name__)


class DebugWebApp:
    def __init__(
        self,
        config: Config,
        camera: CameraCapture | None = None,
    ):
        self.config = config
        self.camera = camera if camera is not None else CameraCapture(config)
        self._live_preview_active = False
        self.started_at = time.time()
        self._capture_lock = threading.Lock()
        self._last_frame: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "voice-ai-bot-debug",
            "host": socket.gethostname(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "camera": {
                "enabled": self.config.camera_enabled,
                "device": self.config.camera_device,
                "frame_size": self.config.camera_frame_size,
                "jpeg_quality": self.config.camera_jpeg_quality,
                "image_dir": str(self.config.camera_images_dir),
                "settle_seconds": self.config.camera_snapshot_settle_seconds,
                "continuous_active": False,
                "live_preview_active": self._live_preview_active,
                "mode": "single-shot",
                "warmup_remaining_seconds": 0.0,
            },
            "last_frame": self._last_frame,
        }

    def memory_snapshot(self) -> dict[str, Any]:
        store = MemoryStore(self.config)
        try:
            entries_payload = store.list_entries(include_forgotten=True)
            entries = entries_payload.get("entries", []) if entries_payload.get("ok") else []
        except Exception as exc:
            LOGGER.exception("failed to read memory entries")
            return {"ok": False, "error": str(exc)}

        root = self.config.memory_dir
        files = []
        for name in ("IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md", "AGENTS.md"):
            files.append(file_snapshot(root / name, root, max_chars=20000))

        daily_notes = []
        daily_dir = root / "memory"
        if daily_dir.exists():
            for path in sorted(daily_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)[:5]:
                daily_notes.append(file_snapshot(path, root, max_chars=12000))

        consolidation = json_file_snapshot(root / "memory" / ".consolidation.json")
        tombstones = json_file_snapshot(root / "memory" / ".tombstones.json")
        pending = consolidation.get("data", {}).get("pending", []) if consolidation.get("ok") else []
        runs = consolidation.get("data", {}).get("runs", []) if consolidation.get("ok") else []
        tombstone_items = tombstones.get("data", {}).get("tombstones", []) if tombstones.get("ok") else []

        return {
            "ok": True,
            "root": str(root),
            "entries": entries,
            "entry_count": len(entries),
            "active_entry_count": sum(1 for entry in entries if entry.get("status") == "active"),
            "files": files,
            "daily_notes": daily_notes,
            "consolidation": {
                "ok": consolidation.get("ok", False),
                "path": consolidation.get("path", ""),
                "pending_count": len(pending) if isinstance(pending, list) else 0,
                "recent_pending": pending[-8:] if isinstance(pending, list) else [],
                "recent_runs": runs[-5:] if isinstance(runs, list) else [],
                "error": consolidation.get("error", ""),
            },
            "tombstone_count": len(tombstone_items) if isinstance(tombstone_items, list) else 0,
        }

    def task_snapshot(self) -> dict[str, Any]:
        background_tasks = read_background_tasks(self.config.memory_dir / "tasks")
        try:
            scheduled_payload = ScheduledTaskStore(self.config).list(include_inactive=True)
            scheduled_tasks = scheduled_payload.get("tasks", []) if scheduled_payload.get("ok") else []
            scheduled_error = "" if scheduled_payload.get("ok") else str(scheduled_payload.get("error", ""))
        except Exception as exc:
            LOGGER.exception("failed to read scheduled tasks")
            scheduled_tasks = []
            scheduled_error = str(exc)

        active_statuses = {"queued", "running", "cancelling"}
        background_counts = count_by_status(background_tasks)
        scheduled_counts = count_by_status(scheduled_tasks)
        return {
            "ok": True,
            "background": {
                "root": str(self.config.memory_dir / "tasks"),
                "tasks": background_tasks,
                "counts": background_counts,
                "active_count": sum(1 for task in background_tasks if task.get("status") in active_statuses),
            },
            "scheduled": {
                "path": str(self.config.scheduled_tasks_file),
                "tasks": scheduled_tasks,
                "counts": scheduled_counts,
                "active_count": sum(1 for task in scheduled_tasks if task.get("status") == "active"),
                "error": scheduled_error,
            },
        }

    def start_camera(self) -> dict[str, Any]:
        self._live_preview_active = True
        return self.status()

    def stop_camera(self) -> dict[str, Any]:
        self._live_preview_active = False
        return self.status()

    def capture_frame(self, mode: str = "bot") -> tuple[bytes, dict[str, Any]]:
        start = time.monotonic()
        with self._capture_lock:
            snapshot = self.camera.capture(
                settle_seconds=self.config.camera_snapshot_settle_seconds,
                shutter_callback=noop_shutter if self.config.camera_shutter_sound_enabled else None,
            )
            source = "bot-snapshot-live" if mode in {"live", "continuous"} else "bot-snapshot"
        payload = snapshot_payload(snapshot)
        remove_temporary_snapshot(snapshot)
        metadata = {
            "captured_at": time.time(),
            "duration_ms": round((time.monotonic() - start) * 1000, 1),
            "size_bytes": len(payload),
            "mime_type": snapshot.mime_type,
            "source": source,
        }
        self._last_frame = metadata
        return payload, metadata


def noop_shutter() -> None:
    pass


def file_snapshot(path: Path, root: Path, max_chars: int = 12000) -> dict[str, Any]:
    rel_path = relative_path(path, root)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {
            "ok": False,
            "path": rel_path,
            "exists": False,
            "size_bytes": 0,
            "updated_at": "",
            "text": "",
            "truncated": False,
            "error": "not found",
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {
            "ok": False,
            "path": rel_path,
            "exists": True,
            "size_bytes": stat.st_size,
            "updated_at": iso_from_timestamp(stat.st_mtime),
            "text": "",
            "truncated": False,
            "error": str(exc),
        }
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "\n[truncated]"
    return {
        "ok": True,
        "path": rel_path,
        "exists": True,
        "size_bytes": stat.st_size,
        "updated_at": iso_from_timestamp(stat.st_mtime),
        "text": text,
        "truncated": truncated,
        "error": "",
    }


def json_file_snapshot(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"ok": False, "path": str(path), "data": {}, "error": "not found"}
    except Exception as exc:
        return {"ok": False, "path": str(path), "data": {}, "error": str(exc)}
    return {"ok": True, "path": str(path), "data": data if isinstance(data, dict) else {}, "error": ""}


def read_background_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if not tasks_dir.exists():
        return tasks
    for path in sorted(tasks_dir.glob("task_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = payload.get("task", payload) if isinstance(payload, dict) else {}
            if not isinstance(data, dict):
                continue
            tasks.append(normalize_background_task(data, path))
        except Exception as exc:
            tasks.append(
                {
                    "id": path.stem,
                    "title": path.stem,
                    "status": "unreadable",
                    "progress": "failed to read task file",
                    "updated_at": "",
                    "created_at": "",
                    "completed_at": None,
                    "request_text": "",
                    "result": "",
                    "error": str(exc),
                    "status_updates": [],
                    "events": [],
                    "steering_messages": [],
                    "path": str(path),
                }
            )
    tasks.sort(key=lambda task: str(task.get("updated_at") or task.get("created_at") or ""), reverse=True)
    return tasks


def normalize_background_task(data: dict[str, Any], path: Path) -> dict[str, Any]:
    status_updates = [item for item in data.get("status_updates", []) if isinstance(item, dict)]
    steering = [item for item in data.get("steering_messages", []) if isinstance(item, dict)]
    events = [str(item) for item in data.get("events", []) if isinstance(item, str)]
    return {
        "id": str(data.get("id") or path.stem),
        "title": str(data.get("title") or ""),
        "request_text": str(data.get("request_text") or ""),
        "source": str(data.get("source") or ""),
        "status": str(data.get("status") or "unknown"),
        "progress": str(data.get("progress") or ""),
        "created_at": coerce_task_time(data.get("created_at")),
        "updated_at": coerce_task_time(data.get("updated_at")),
        "completed_at": coerce_task_time(data.get("completed_at")) if data.get("completed_at") else None,
        "response_id": str(data.get("response_id") or ""),
        "result": truncate_text(str(data.get("result") or ""), 4000),
        "reasoning_summary": truncate_text(str(data.get("reasoning_summary") or ""), 2000),
        "error": str(data.get("error") or ""),
        "status_updates": status_updates[-12:],
        "events": events[-20:],
        "steering_messages": steering[-12:],
        "wakeup_on_complete": bool(data.get("wakeup_on_complete")),
        "wakeup_on_progress": bool(data.get("wakeup_on_progress")),
        "wakeup_reported": bool(data.get("wakeup_reported")),
        "history_count": len(data.get("history", [])) if isinstance(data.get("history"), list) else 0,
        "path": str(path),
    }


def count_by_status(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def coerce_task_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return iso_from_timestamp(float(value))
    return str(value)


def iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def snapshot_payload(snapshot: CameraSnapshot) -> bytes:
    prefix = f"data:{snapshot.mime_type};base64,"
    if snapshot.data_url.startswith(prefix):
        try:
            return base64.b64decode(snapshot.data_url[len(prefix) :], validate=True)
        except (binascii.Error, ValueError):
            LOGGER.warning("failed to decode camera data URL; falling back to %s", snapshot.path)
    return snapshot.path.read_bytes()


def remove_temporary_snapshot(snapshot: CameraSnapshot) -> None:
    try:
        snapshot.path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("failed to remove temporary debug snapshot %s", snapshot.path, exc_info=True)


def create_handler(app: DebugWebApp) -> type[BaseHTTPRequestHandler]:
    class DebugWebHandler(BaseHTTPRequestHandler):
        server_version = "VoiceAIBotDebug/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/":
                self._send_html(INDEX_HTML)
                return
            if path == "/api/status":
                self._send_json(app.status())
                return
            if path == "/api/memory":
                self._send_json(app.memory_snapshot())
                return
            if path == "/api/tasks":
                self._send_json(app.task_snapshot())
                return
            if path == "/api/camera.jpg":
                self._send_camera_frame(parse_qs(parsed.query))
                return
            if path == "/health":
                self._send_json({"ok": True})
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/api/camera/start":
                self._run_camera_action(app.start_camera)
                return
            if path == "/api/camera/stop":
                self._run_camera_action(app.stop_camera)
                return
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _send_camera_frame(self, _query: dict[str, list[str]]) -> None:
            mode = (_query.get("mode") or ["bot"])[0].strip().lower()
            try:
                payload, metadata = app.capture_frame(mode=mode)
            except CameraError as exc:
                LOGGER.warning("camera debug frame unavailable: %s", exc)
                self._send_json(
                    {"ok": False, "error": str(exc), "status": app.status()},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            except Exception as exc:
                LOGGER.exception("camera debug frame failed")
                self._send_json(
                    {"ok": False, "error": f"camera capture failed: {exc}", "status": app.status()},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Capture-Duration-Ms", str(metadata["duration_ms"]))
            self.end_headers()
            self.wfile.write(payload)

        def _run_camera_action(self, action: Callable[[], dict[str, Any]]) -> None:
            try:
                self._send_json(action())
            except CameraError as exc:
                LOGGER.warning("camera debug action unavailable: %s", exc)
                self._send_json(
                    {"ok": False, "error": str(exc), "status": app.status()},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            except Exception as exc:
                LOGGER.exception("camera debug action failed")
                self._send_json(
                    {"ok": False, "error": f"camera action failed: {exc}", "status": app.status()},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, body: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(encoded)

    return DebugWebHandler


def run_server(config: Config) -> ThreadingHTTPServer:
    app = DebugWebApp(config)
    server = ThreadingHTTPServer((config.debug_web_host, config.debug_web_port), create_handler(app))
    server.daemon_threads = True
    LOGGER.info(
        "voice-ai-bot debug web listening on http://%s:%s",
        config.debug_web_host,
        config.debug_web_port,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Voice AI Bot debug web UI.")
    parser.add_argument("--host", default=None, help="Host/interface to bind. Defaults to DEBUG_WEB_HOST.")
    parser.add_argument("--port", type=int, default=None, help="TCP port to bind. Defaults to DEBUG_WEB_PORT.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["debug_web_host"] = args.host
    if args.port is not None:
        updates["debug_web_port"] = args.port
    return replace(config, **updates) if updates else config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = config_from_args(args)
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    configure_logging(config.log_level)
    server: ThreadingHTTPServer | None = None

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("received signal %s", signum)
        if server is not None:
            threading.Thread(target=server.shutdown, name="debug-web-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    app = DebugWebApp(config)
    server = ThreadingHTTPServer((config.debug_web_host, config.debug_web_port), create_handler(app))
    server.daemon_threads = True
    LOGGER.info(
        "voice-ai-bot debug web listening on http://%s:%s",
        config.debug_web_host,
        config.debug_web_port,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Voice AI Bot Debug</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f4f1ea;
      --ink: #191815;
      --muted: #6b655a;
      --line: #c9c0b2;
      --panel: #fffaf0;
      --black: #080807;
      --amber: #c77b1b;
      --green: #2f7d4a;
      --red: #b5342d;
      --blue: #275f8f;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(90deg, rgba(25, 24, 21, 0.035) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(180deg, rgba(25, 24, 21, 0.035) 1px, transparent 1px) 0 0 / 28px 28px,
        var(--paper);
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      letter-spacing: 0;
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }

    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      border-bottom: 2px solid var(--ink);
      padding-bottom: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.4rem, 3vw, 2.7rem);
      line-height: 1;
      text-transform: uppercase;
    }

    .eyebrow {
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 8px;
      text-transform: uppercase;
    }

    .pill {
      border: 2px solid var(--ink);
      padding: 8px 10px;
      background: var(--panel);
      min-width: 138px;
      text-align: center;
      font-weight: 700;
      text-transform: uppercase;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      align-items: start;
    }

    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 18px;
    }

    .tab {
      min-height: 38px;
      box-shadow: none;
      background: #e8dfd1;
    }

    .tab.active {
      background: var(--ink);
      color: var(--paper);
    }

    .view { display: none; }
    .view.active { display: block; }

    .debug-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }

    .debug-grid.single { grid-template-columns: 1fr; }

    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .toolbar h2 {
      margin: 0;
      font-size: 1rem;
      text-transform: uppercase;
    }

    .small-button {
      min-height: 34px;
      padding: 0 10px;
      box-shadow: 2px 2px 0 var(--ink);
      font-size: 0.8rem;
    }

    .stack {
      display: grid;
      gap: 10px;
    }

    .item {
      border-top: 1px solid var(--line);
      padding: 10px 12px;
    }

    .item:first-child { border-top: 0; }

    .item-title {
      display: flex;
      gap: 8px;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 6px;
      font-weight: 800;
      text-transform: uppercase;
      overflow-wrap: anywhere;
    }

    .meta {
      color: var(--muted);
      font-size: 0.76rem;
      overflow-wrap: anywhere;
    }

    .chip {
      border: 1px solid var(--ink);
      padding: 2px 6px;
      background: #fff3c9;
      color: var(--ink);
      font-size: 0.72rem;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .chip.running, .chip.active, .chip.queued { background: #d8e8d7; }
    .chip.failed, .chip.error { background: #ead8d5; color: var(--red); }
    .chip.completed { background: #d8e2ea; color: var(--blue); }

    .text-block {
      margin: 8px 0 0;
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 0.82rem;
      line-height: 1.35;
      max-height: 280px;
      overflow: auto;
    }

    .summary-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px;
    }

    .camera-stage {
      border: 2px solid var(--ink);
      background: var(--black);
      min-height: 320px;
      aspect-ratio: 16 / 9;
      position: relative;
      overflow: hidden;
    }

    .camera-stage img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      opacity: 0;
      transition: opacity 160ms ease-out;
    }

    .camera-stage img.ready { opacity: 1; }

    .empty-state {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #f4f1ea;
      text-transform: uppercase;
      font-size: clamp(1rem, 2.2vw, 1.7rem);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 14px 0 0;
    }

    button {
      appearance: none;
      border: 2px solid var(--ink);
      background: var(--panel);
      color: var(--ink);
      min-height: 44px;
      padding: 0 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      text-transform: uppercase;
      box-shadow: 3px 3px 0 var(--ink);
    }

    button:hover { transform: translate(1px, 1px); box-shadow: 2px 2px 0 var(--ink); }
    button:active { transform: translate(3px, 3px); box-shadow: none; }
    button:disabled { color: var(--muted); border-color: var(--line); box-shadow: none; cursor: not-allowed; }
    button.primary { background: var(--amber); }
    button.danger { background: #ead8d5; color: var(--red); }

    .side {
      display: grid;
      gap: 12px;
    }

    .panel {
      border: 2px solid var(--ink);
      background: var(--panel);
    }

    .panel h2 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 2px solid var(--ink);
      font-size: 0.95rem;
      text-transform: uppercase;
      background: #e8dfd1;
    }

    dl {
      margin: 0;
      display: grid;
      grid-template-columns: 116px minmax(0, 1fr);
    }

    dt, dd {
      margin: 0;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
      overflow-wrap: anywhere;
      font-size: 0.82rem;
    }

    dt { color: var(--muted); text-transform: uppercase; }
    dd { color: var(--ink); }
    dl dt:last-of-type, dl dd:last-of-type { border-bottom: 0; }

    .status-ok { color: var(--green); }
    .status-error { color: var(--red); }
    .status-work { color: var(--blue); }

    @media (max-width: 860px) {
      header, main { grid-template-columns: 1fr; }
      .debug-grid { grid-template-columns: 1fr; }
      .pill { width: max-content; }
      .shell { width: min(100vw - 20px, 720px); padding: 14px 0; }
      dl { grid-template-columns: 102px minmax(0, 1fr); }
      button { flex: 1 1 130px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <div class="eyebrow">Pi runtime console</div>
        <h1>Voice AI Bot Debug</h1>
      </div>
      <div class="pill" id="topStatus">Offline</div>
    </header>
    <nav class="tabs" aria-label="Debug views">
      <button class="tab active" type="button" data-view="cameraView">Camera</button>
      <button class="tab" type="button" data-view="memoryView">Memory</button>
      <button class="tab" type="button" data-view="tasksView">Tasks</button>
    </nav>
    <section class="view active" id="cameraView">
      <main>
        <section>
          <div class="camera-stage" aria-label="Camera frame">
            <div class="empty-state" id="emptyState">Camera stopped</div>
            <img id="cameraImage" alt="Pi camera frame">
          </div>
          <div class="controls">
            <button class="primary" id="startButton" type="button">Start Live</button>
            <button id="refreshButton" type="button">Capture</button>
            <button class="danger" id="stopButton" type="button" disabled>Stop</button>
          </div>
        </section>
        <aside class="side">
          <section class="panel">
            <h2>Status</h2>
            <dl>
              <dt>Camera</dt><dd id="cameraStatus">Unknown</dd>
              <dt>Last Frame</dt><dd id="lastFrame">None</dd>
              <dt>Latency</dt><dd id="latency">-</dd>
              <dt>Host</dt><dd id="host">-</dd>
            </dl>
          </section>
          <section class="panel">
            <h2>Camera</h2>
            <dl>
              <dt>Device</dt><dd id="device">-</dd>
              <dt>Frame</dt><dd id="frameSize">-</dd>
              <dt>Quality</dt><dd id="quality">-</dd>
              <dt>Settle</dt><dd id="settle">-</dd>
            </dl>
          </section>
        </aside>
      </main>
    </section>
    <section class="view" id="memoryView">
      <div class="toolbar">
        <h2>Memory</h2>
        <button class="small-button" id="memoryRefreshButton" type="button">Refresh</button>
      </div>
      <div class="debug-grid">
        <section class="panel">
          <h2>Entries</h2>
          <div class="summary-row" id="memorySummary"></div>
          <div class="stack" id="memoryEntries"></div>
        </section>
        <section class="panel">
          <h2>Files</h2>
          <div class="stack" id="memoryFiles"></div>
        </section>
        <section class="panel">
          <h2>Daily Notes</h2>
          <div class="stack" id="dailyNotes"></div>
        </section>
        <section class="panel">
          <h2>Consolidation</h2>
          <div class="stack" id="consolidation"></div>
        </section>
      </div>
    </section>
    <section class="view" id="tasksView">
      <div class="toolbar">
        <h2>Tasks</h2>
        <button class="small-button" id="tasksRefreshButton" type="button">Refresh</button>
      </div>
      <div class="debug-grid">
        <section class="panel">
          <h2>Background</h2>
          <div class="summary-row" id="backgroundSummary"></div>
          <div class="stack" id="backgroundTasks"></div>
        </section>
        <section class="panel">
          <h2>Scheduled</h2>
          <div class="summary-row" id="scheduledSummary"></div>
          <div class="stack" id="scheduledTasks"></div>
        </section>
      </div>
    </section>
  </div>
  <script>
    const refreshMs = 500;
    const topStatus = document.getElementById("topStatus");
    const cameraStatus = document.getElementById("cameraStatus");
    const lastFrame = document.getElementById("lastFrame");
    const latency = document.getElementById("latency");
    const host = document.getElementById("host");
    const device = document.getElementById("device");
    const frameSize = document.getElementById("frameSize");
    const quality = document.getElementById("quality");
    const settle = document.getElementById("settle");
    const image = document.getElementById("cameraImage");
    const emptyState = document.getElementById("emptyState");
    const startButton = document.getElementById("startButton");
    const refreshButton = document.getElementById("refreshButton");
    const stopButton = document.getElementById("stopButton");
    const memoryRefreshButton = document.getElementById("memoryRefreshButton");
    const tasksRefreshButton = document.getElementById("tasksRefreshButton");
    const memorySummary = document.getElementById("memorySummary");
    const memoryEntries = document.getElementById("memoryEntries");
    const memoryFiles = document.getElementById("memoryFiles");
    const dailyNotes = document.getElementById("dailyNotes");
    const consolidation = document.getElementById("consolidation");
    const backgroundSummary = document.getElementById("backgroundSummary");
    const backgroundTasks = document.getElementById("backgroundTasks");
    const scheduledSummary = document.getElementById("scheduledSummary");
    const scheduledTasks = document.getElementById("scheduledTasks");
    let running = false;
    let timer = null;
    let objectUrl = null;
    let memoryLoaded = false;
    let tasksLoaded = false;

    function setStatus(text, cls) {
      topStatus.textContent = text;
      topStatus.className = "pill " + cls;
      cameraStatus.textContent = text;
      cameraStatus.className = cls;
    }

    function renderStatus(payload) {
      host.textContent = payload.host || "-";
      if (payload.camera) {
        device.textContent = payload.camera.device;
        frameSize.textContent = payload.camera.frame_size;
        quality.textContent = payload.camera.jpeg_quality;
        settle.textContent = payload.camera.settle_seconds + "s";
      }
      if (payload.last_frame) {
        const date = new Date(payload.last_frame.captured_at * 1000);
        lastFrame.textContent = date.toLocaleTimeString();
        latency.textContent = payload.last_frame.duration_ms + "ms";
      }
      if (!running) setStatus(payload.camera && payload.camera.enabled ? "Ready" : "Disabled", "status-ok");
    }

    function activateView(id) {
      document.querySelectorAll(".tab").forEach((button) => {
        button.classList.toggle("active", button.dataset.view === id);
      });
      document.querySelectorAll(".view").forEach((view) => {
        view.classList.toggle("active", view.id === id);
      });
      if (id === "memoryView" && !memoryLoaded) loadMemory();
      if (id === "tasksView" && !tasksLoaded) loadTasks();
    }

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => activateView(button.dataset.view));
    });

    function chip(text, cls = "") {
      const node = document.createElement("span");
      node.className = "chip " + cls;
      node.textContent = text;
      return node;
    }

    function item(title, metaParts = [], status = "") {
      const node = document.createElement("div");
      node.className = "item";
      const titleRow = document.createElement("div");
      titleRow.className = "item-title";
      const titleNode = document.createElement("span");
      titleNode.textContent = title || "Untitled";
      titleRow.appendChild(titleNode);
      if (status) titleRow.appendChild(chip(status, status));
      node.appendChild(titleRow);
      if (metaParts.length) {
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = metaParts.filter(Boolean).join(" | ");
        node.appendChild(meta);
      }
      return node;
    }

    function textBlock(text) {
      const block = document.createElement("div");
      block.className = "text-block";
      block.textContent = text || "";
      return block;
    }

    function setLoading(target) {
      target.replaceChildren(item("Loading"));
    }

    function setEmpty(target, label) {
      target.replaceChildren(item(label));
    }

    function renderChips(target, values) {
      target.replaceChildren(...values.map(([text, cls]) => chip(text, cls || "")));
    }

    async function fetchJson(url) {
      const response = await fetch(url, { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) throw new Error(payload.error || "request " + response.status);
      return payload;
    }

    async function loadMemory() {
      memoryLoaded = true;
      setLoading(memoryEntries);
      setLoading(memoryFiles);
      setLoading(dailyNotes);
      setLoading(consolidation);
      try {
        const payload = await fetchJson("/api/memory");
        renderChips(memorySummary, [
          ["active " + payload.active_entry_count, "active"],
          ["entries " + payload.entry_count, ""],
          ["tombstones " + payload.tombstone_count, ""],
        ]);
        renderMemoryEntries(payload.entries || []);
        renderMemoryFiles(payload.files || [], memoryFiles);
        renderMemoryFiles(payload.daily_notes || [], dailyNotes);
        renderConsolidation(payload.consolidation || {});
      } catch (error) {
        setEmpty(memoryEntries, error.message);
        memorySummary.replaceChildren(chip("error", "error"));
      }
    }

    function renderMemoryEntries(entries) {
      if (!entries.length) return setEmpty(memoryEntries, "No memory entries");
      const nodes = entries.slice().reverse().map((entry) => {
        const node = item(entry.id, [entry.kind, entry.source, entry.updated_at], entry.status);
        node.appendChild(textBlock(entry.text));
        return node;
      });
      memoryEntries.replaceChildren(...nodes);
    }

    function renderMemoryFiles(files, target) {
      if (!files.length) return setEmpty(target, "No files");
      const nodes = files.map((file) => {
        const title = file.path + (file.truncated ? " [truncated]" : "");
        const node = item(title, [file.exists ? file.size_bytes + " bytes" : "missing", file.updated_at], file.ok ? "ok" : "error");
        if (file.text) node.appendChild(textBlock(file.text));
        if (file.error) node.appendChild(textBlock(file.error));
        return node;
      });
      target.replaceChildren(...nodes);
    }

    function renderConsolidation(data) {
      const nodes = [
        item("Pending notes", [String(data.pending_count || 0)], data.ok ? "ok" : "error"),
      ];
      (data.recent_pending || []).slice().reverse().forEach((pending) => {
        const node = item(pending.id || "pending", [pending.type, pending.created_at, "attempts " + (pending.attempts || 0)], "");
        node.appendChild(textBlock(pending.text || ""));
        nodes.push(node);
      });
      (data.recent_runs || []).slice().reverse().forEach((run) => {
        const node = item("Run", [run.completed_at, "ops " + (run.operation_count || 0)], "completed");
        node.appendChild(textBlock(run.summary || ""));
        nodes.push(node);
      });
      if (data.error) nodes.push(item(data.error, [], "error"));
      consolidation.replaceChildren(...nodes);
    }

    async function loadTasks() {
      tasksLoaded = true;
      setLoading(backgroundTasks);
      setLoading(scheduledTasks);
      try {
        const payload = await fetchJson("/api/tasks");
        renderTaskSummary(backgroundSummary, payload.background || {});
        renderTaskSummary(scheduledSummary, payload.scheduled || {});
        renderBackgroundTasks((payload.background || {}).tasks || []);
        renderScheduledTasks((payload.scheduled || {}).tasks || []);
      } catch (error) {
        setEmpty(backgroundTasks, error.message);
        backgroundSummary.replaceChildren(chip("error", "error"));
      }
    }

    function renderTaskSummary(target, payload) {
      const counts = payload.counts || {};
      const chips = [["active " + (payload.active_count || 0), payload.active_count ? "active" : ""]];
      Object.keys(counts).sort().forEach((status) => chips.push([status + " " + counts[status], status]));
      renderChips(target, chips);
    }

    function renderBackgroundTasks(tasks) {
      if (!tasks.length) return setEmpty(backgroundTasks, "No background tasks");
      const nodes = tasks.map((task) => {
        const node = item(task.title || task.id, [task.id, task.updated_at || task.created_at, task.source], task.status);
        if (task.progress) node.appendChild(textBlock(task.progress));
        if (task.request_text) node.appendChild(textBlock(task.request_text));
        (task.status_updates || []).slice(-4).forEach((update) => {
          node.appendChild(textBlock((update.created_at || "") + "\\n" + (update.text || "")));
        });
        if (task.result) node.appendChild(textBlock(task.result));
        if (task.error) node.appendChild(textBlock(task.error));
        return node;
      });
      backgroundTasks.replaceChildren(...nodes);
    }

    function renderScheduledTasks(tasks) {
      if (!tasks.length) return setEmpty(scheduledTasks, "No scheduled tasks");
      const nodes = tasks.map((task) => {
        const node = item(task.title || task.id, [task.id, task.run_at, task.action, task.repeat], task.status);
        if (task.prompt) node.appendChild(textBlock(task.prompt));
        if (task.last_started_at || task.skipped_reason) {
          node.appendChild(textBlock(["last_started_at: " + (task.last_started_at || ""), "skipped_reason: " + (task.skipped_reason || "")].join("\\n")));
        }
        return node;
      });
      scheduledTasks.replaceChildren(...nodes);
    }

    async function getStatus() {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error("status " + response.status);
      renderStatus(await response.json());
    }

    async function postJson(url) {
      const response = await fetch(url, { method: "POST", cache: "no-store" });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || "request " + response.status);
      return body;
    }

    async function loadFrame(mode = "bot") {
      setStatus(mode === "live" ? "Live capture" : "Bot capture", "status-work");
      const query = mode === "live" ? "mode=live&" : "";
      const response = await fetch("/api/camera.jpg?" + query + "ts=" + Date.now(), { cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: "camera error " + response.status }));
        throw new Error(body.error || "camera error " + response.status);
      }
      const duration = response.headers.get("X-Capture-Duration-Ms");
      const blob = await response.blob();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      image.classList.remove("ready");
      image.onload = () => image.classList.add("ready");
      image.src = objectUrl;
      emptyState.style.display = "none";
      latency.textContent = duration ? duration + "ms" : "-";
      lastFrame.textContent = new Date().toLocaleTimeString();
      setStatus(running ? "Live" : "Ready", "status-ok");
    }

    async function loop() {
      if (!running) return;
      try {
        await loadFrame("live");
      } catch (error) {
        setStatus("Error", "status-error");
        lastFrame.textContent = error.message;
      }
      if (running) timer = setTimeout(loop, refreshMs);
    }

    startButton.addEventListener("click", async () => {
      try {
        await postJson("/api/camera/start");
        running = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        loop();
      } catch (error) {
        setStatus("Error", "status-error");
        lastFrame.textContent = error.message;
      }
    });

    stopButton.addEventListener("click", async () => {
      running = false;
      clearTimeout(timer);
      startButton.disabled = false;
      stopButton.disabled = true;
      try {
        await postJson("/api/camera/stop");
      } catch (error) {
        lastFrame.textContent = error.message;
      }
      setStatus("Stopped", "status-work");
    });

    refreshButton.addEventListener("click", async () => {
      try {
        await loadFrame("bot");
      } catch (error) {
        setStatus("Error", "status-error");
        lastFrame.textContent = error.message;
      }
    });

    memoryRefreshButton.addEventListener("click", loadMemory);
    tasksRefreshButton.addEventListener("click", loadTasks);

    getStatus().catch((error) => {
      setStatus("Error", "status-error");
      lastFrame.textContent = error.message;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
