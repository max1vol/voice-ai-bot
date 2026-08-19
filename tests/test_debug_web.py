from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from voice_ai_bot.camera import CameraError, CameraSnapshot
from voice_ai_bot.debug_web import DebugWebApp, create_handler


JPEG = b"\xff\xd8debug-frame\xff\xd9"


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        camera_enabled=True,
        camera_device="/dev/video0",
        camera_frame_size="1280x720",
        camera_jpeg_quality=85,
        camera_images_dir=tmp_path,
        camera_snapshot_settle_seconds=3.0,
        camera_shutter_sound_enabled=True,
        memory_dir=tmp_path / "agent",
        memory_bootstrap_chars=12000,
        memory_active_context_chars=1800,
        user_city="Cambridge",
        user_region="Cambridgeshire",
        user_country="GB",
        user_timezone="Europe/London",
        scheduled_tasks_file=tmp_path / "scheduled_tasks.json",
        schedule_quiet_start="21:00",
        schedule_quiet_end="07:30",
    )


class FakeCamera:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.settle_calls: list[float | None] = []
        self.shutter_calls = 0
        self.last_path: Path | None = None

    def capture(self, settle_seconds=None, shutter_callback=None):
        self.settle_calls.append(settle_seconds)
        if shutter_callback is not None:
            shutter_callback()
            self.shutter_calls += 1
        path = self.tmp_path / f"frame-{len(self.settle_calls)}.jpg"
        path.write_bytes(JPEG)
        self.last_path = path
        return CameraSnapshot(
            path=path,
            mime_type="image/jpeg",
            size_bytes=len(JPEG),
            data_url="data:image/jpeg;base64,/9hkZWJ1Zy1mcmFtZf/Z",
        )


class FailingCamera:
    def capture(self, settle_seconds=None, shutter_callback=None):
        raise CameraError("camera unplugged")


def _serve(app: DebugWebApp):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def test_debug_status_reports_camera_config(tmp_path):
    app = DebugWebApp(_config(tmp_path), camera=FakeCamera(tmp_path))

    status = app.status()

    assert status["ok"] is True
    assert status["service"] == "voice-ai-bot-debug"
    assert status["camera"]["device"] == "/dev/video0"
    assert status["camera"]["settle_seconds"] == 3.0


def test_camera_endpoint_uses_bot_snapshot_path_by_default(tmp_path):
    camera = FakeCamera(tmp_path)
    server, thread, base_url = _serve(DebugWebApp(_config(tmp_path), camera=camera))
    try:
        with urlopen(f"{base_url}/api/camera.jpg", timeout=2) as response:
            payload = response.read()
            content_type = response.headers["Content-Type"]
            duration = response.headers["X-Capture-Duration-Ms"]

        assert payload == JPEG
        assert content_type == "image/jpeg"
        assert float(duration) >= 0
        assert camera.settle_calls == [3.0]
        assert camera.shutter_calls == 1
        assert camera.last_path is not None
        assert not camera.last_path.exists()

        with urlopen(f"{base_url}/api/status", timeout=2) as response:
            status = json.loads(response.read())
        assert status["last_frame"]["size_bytes"] == len(JPEG)
        assert status["last_frame"]["source"] == "bot-snapshot"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_live_camera_endpoint_uses_bot_snapshot_path_without_holding_camera(tmp_path):
    camera = FakeCamera(tmp_path)
    app = DebugWebApp(_config(tmp_path), camera=camera)
    server, thread, base_url = _serve(app)
    try:
        with urlopen(Request(f"{base_url}/api/camera/start", method="POST"), timeout=2) as response:
            assert response.status == 200

        with urlopen(f"{base_url}/api/camera.jpg?mode=live", timeout=5) as response:
            payload = response.read()

        assert payload == JPEG
        assert camera.settle_calls == [3.0]
        assert camera.shutter_calls == 1
        assert camera.last_path is not None
        assert not camera.last_path.exists()

        with urlopen(f"{base_url}/api/status", timeout=2) as response:
            status = json.loads(response.read())
        assert status["camera"]["continuous_active"] is False
        assert status["camera"]["live_preview_active"] is True
        assert status["last_frame"]["source"] == "bot-snapshot-live"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_camera_endpoint_returns_json_error_when_capture_fails(tmp_path):
    server, thread, base_url = _serve(DebugWebApp(_config(tmp_path), camera=FailingCamera()))
    try:
        try:
            urlopen(f"{base_url}/api/camera.jpg", timeout=2)
        except HTTPError as exc:
            assert exc.code == 503
            body = json.loads(exc.read())
        else:
            raise AssertionError("expected HTTP 503")

        assert body["ok"] is False
        assert body["error"] == "camera unplugged"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_memory_endpoint_reports_entries_files_and_consolidation(tmp_path):
    config = _config(tmp_path)
    memory_root = config.memory_dir
    (memory_root / "memory").mkdir(parents=True)
    (memory_root / "MEMORY.md").write_text(
        "# Long-Term Memory\n\n"
        "<!-- voice-ai-bot:memory-section:start -->\n"
        '<!-- voice-ai-bot:memory id="mem_1" kind="note" source="user" status="active" '
        'created_at="2026-07-01T10:00:00Z" updated_at="2026-07-01T10:00:00Z" text_hash="abc" -->\n'
        "- Likes concise debug views.\n"
        "<!-- voice-ai-bot:memory-section:end -->\n",
        encoding="utf-8",
    )
    (memory_root / "memory" / "2026-07-01.md").write_text("# 2026-07-01\n\nDaily note\n", encoding="utf-8")
    (memory_root / "memory" / ".consolidation.json").write_text(
        json.dumps({"version": 1, "pending": [{"id": "note_1", "text": "Pending"}], "runs": []}),
        encoding="utf-8",
    )
    app = DebugWebApp(config, camera=FakeCamera(tmp_path))
    server, thread, base_url = _serve(app)
    try:
        with urlopen(f"{base_url}/api/memory", timeout=2) as response:
            payload = json.loads(response.read())

        assert payload["ok"] is True
        assert payload["entry_count"] == 1
        assert payload["active_entry_count"] == 1
        assert payload["entries"][0]["text"] == "Likes concise debug views."
        assert payload["daily_notes"][0]["path"] == "memory/2026-07-01.md"
        assert payload["consolidation"]["pending_count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_tasks_endpoint_reports_background_and_scheduled_tasks(tmp_path):
    config = _config(tmp_path)
    tasks_dir = config.memory_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "task_abc.json").write_text(
        json.dumps(
            {
                "version": 1,
                "task": {
                    "id": "task_abc",
                    "title": "Research",
                    "request_text": "Look something up",
                    "status": "completed",
                    "progress": "completed",
                    "result": "Done",
                    "created_at": 1782930000.0,
                    "updated_at": 1782930005.0,
                    "status_updates": [{"text": "Working", "created_at": "2026-07-01T10:00:00Z"}],
                },
            }
        ),
        encoding="utf-8",
    )
    config.scheduled_tasks_file.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": [
                    {
                        "id": "sched_1",
                        "title": "Reminder",
                        "prompt": "Stand up",
                        "run_at": "2026-07-02T09:00:00+01:00",
                        "action": "speak",
                        "repeat": "once",
                        "status": "active",
                        "created_at": "2026-07-01T10:00:00+00:00",
                        "updated_at": "2026-07-01T10:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = DebugWebApp(config, camera=FakeCamera(tmp_path))
    server, thread, base_url = _serve(app)
    try:
        with urlopen(f"{base_url}/api/tasks", timeout=2) as response:
            payload = json.loads(response.read())

        assert payload["ok"] is True
        assert payload["background"]["counts"] == {"completed": 1}
        assert payload["background"]["tasks"][0]["id"] == "task_abc"
        assert payload["background"]["tasks"][0]["result"] == "Done"
        assert payload["scheduled"]["active_count"] == 1
        assert payload["scheduled"]["tasks"][0]["id"] == "sched_1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
