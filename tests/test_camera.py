from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from voice_ai_bot.camera import CameraCapture


def test_camera_capture_uses_v4l2_ctl_for_usb_webcam(monkeypatch, tmp_path):
    config = SimpleNamespace(
        camera_enabled=True,
        camera_images_dir=tmp_path,
        camera_capture_command="",
        camera_device="/dev/video0",
        camera_frame_size="1280x720",
        camera_jpeg_quality=85,
        camera_capture_timeout_seconds=6.0,
        camera_max_image_bytes=1_500_000,
        camera_snapshot_settle_seconds=3.0,
    )
    capture = CameraCapture(config)

    monkeypatch.setattr(CameraCapture, "_capture_with_pyav", lambda self, output, settle, shutter_callback: False)
    monkeypatch.setattr(
        "voice_ai_bot.camera.shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "v4l2-ctl" else None,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_arg = next(part for part in command if part.startswith("--stream-to="))
        Path(output_arg.split("=", 1)[1]).write_bytes(b"\xff\xd8jpeg\xff\xd9")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("voice_ai_bot.camera.subprocess.run", fake_run)

    snapshot = capture.capture()

    command = calls[0][0]
    assert command[:2] == ["v4l2-ctl", "--device"]
    assert "/dev/video0" in command
    assert "--set-fmt-video=width=1280,height=720,pixelformat=MJPG" in command
    assert "--stream-mmap" in command
    assert "--stream-skip=90" in command
    assert "--stream-count=1" in command
    assert snapshot.mime_type == "image/jpeg"
    assert snapshot.data_url.startswith("data:image/jpeg;base64,")
