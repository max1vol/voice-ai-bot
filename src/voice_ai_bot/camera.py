from __future__ import annotations

import base64
import fcntl
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSnapshot:
    path: Path
    mime_type: str
    size_bytes: int
    data_url: str


class CameraDeviceLock:
    def __init__(self, config: Config, timeout_seconds: float | None = None):
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._handle = None

    def __enter__(self) -> "CameraDeviceLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()

    def acquire(self) -> None:
        lock_path = Path(getattr(self.config, "camera_lock_file", "/tmp/voice-ai-bot-camera.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = lock_path.open("a+")
        timeout = self.timeout_seconds
        if timeout is None:
            timeout = float(getattr(self.config, "camera_capture_timeout_seconds", 6.0))
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()}\n")
                self._handle.flush()
                return
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    self.release()
                    raise CameraError("camera is busy") from exc
                time.sleep(0.05)

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class CameraCapture:
    def __init__(self, config: Config):
        self.config = config

    def capture(
        self,
        settle_seconds: float | None = None,
        shutter_callback: Callable[[], None] | None = None,
    ) -> CameraSnapshot:
        if not self.config.camera_enabled:
            raise CameraError("camera is disabled")

        images_dir = self.config.camera_images_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        output = images_dir / f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}-{time.monotonic_ns()}.jpg"
        errors: list[str] = []
        settle = self._settle_seconds(settle_seconds)

        with CameraDeviceLock(self.config):
            if not self.config.camera_capture_command.strip():
                try:
                    if self._capture_with_pyav(output, settle, shutter_callback):
                        return self._snapshot_from_file(output)
                except CameraError as exc:
                    errors.append(str(exc))
                except BaseException as exc:
                    errors.append(f"pyav capture failed: {exc}")

            shutter_played = False
            for command in self._candidate_commands(output, settle):
                try:
                    LOGGER.info("capturing camera snapshot with %s", command[0])
                    if shutter_callback is not None and not shutter_played:
                        shutter_callback()
                        shutter_played = True
                    subprocess.run(
                        command,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=self.config.camera_capture_timeout_seconds,
                    )
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    errors.append(self._format_command_error(command, exc))
                    continue
                if output.exists() and output.stat().st_size > 0:
                    break
                errors.append(f"{command[0]} did not create a snapshot")
            else:
                detail = "; ".join(errors[-3:]) if errors else "no supported capture command found"
                raise CameraError(detail)

        return self._snapshot_from_file(output)

    def _snapshot_from_file(self, output: Path) -> CameraSnapshot:
        size = output.stat().st_size
        if size > self.config.camera_max_image_bytes:
            raise CameraError(
                f"camera snapshot is too large: {size} bytes > {self.config.camera_max_image_bytes} bytes"
            )

        payload = base64.b64encode(output.read_bytes()).decode("ascii")
        return CameraSnapshot(
            path=output,
            mime_type="image/jpeg",
            size_bytes=size,
            data_url=f"data:image/jpeg;base64,{payload}",
        )

    def _capture_with_pyav(
        self,
        output: Path,
        settle_seconds: float,
        shutter_callback: Callable[[], None] | None,
    ) -> bool:
        device = Path(self.config.camera_device)
        if not device.exists():
            return False
        try:
            import av
        except ImportError:
            return False

        start = time.monotonic()
        settle_until = start + settle_seconds
        timeout_at = start + max(self.config.camera_capture_timeout_seconds, settle_seconds + 2.0)
        shutter_played = False
        container = None
        try:
            LOGGER.info("capturing camera snapshot with pyav from %s", self.config.camera_device)
            container = av.open(
                self.config.camera_device,
                format="v4l2",
                options={
                    "video_size": self.config.camera_frame_size,
                    "input_format": "mjpeg",
                },
            )
            for packet in container.demux(video=0):
                if time.monotonic() > timeout_at:
                    raise CameraError("pyav camera capture timed out")
                jpeg = jpeg_bytes_from_mjpeg_packet(bytes(packet))
                if not jpeg:
                    continue
                if time.monotonic() < settle_until:
                    continue
                if shutter_callback is not None and not shutter_played:
                    shutter_callback()
                    shutter_played = True
                    continue
                output.write_bytes(jpeg)
                return True
        finally:
            if container is not None:
                container.close()
        return False

    def _candidate_commands(self, output: Path, settle_seconds: float) -> list[list[str]]:
        commands: list[list[str]] = []
        custom = self.config.camera_capture_command.strip()
        if custom:
            if "{output}" in custom:
                commands.append(shlex.split(custom.format(output=str(output))))
            else:
                commands.append([*shlex.split(custom), str(output)])

        width, height = self._frame_size()
        if shutil.which("fswebcam"):
            command = [
                "fswebcam",
                "-q",
                "-d",
                self.config.camera_device,
                "-r",
                self.config.camera_frame_size,
                "--jpeg",
                str(self.config.camera_jpeg_quality),
                "--no-banner",
            ]
            skip_frames = self._settle_frames(settle_seconds)
            if skip_frames:
                command.extend(["--skip", str(skip_frames)])
            command.append(str(output))
            commands.append(command)
        if shutil.which("ffmpeg"):
            commands.append(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "video4linux2",
                    "-video_size",
                    self.config.camera_frame_size,
                    "-i",
                    self.config.camera_device,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(output),
                ]
            )
        if shutil.which("v4l2-ctl"):
            command = [
                "v4l2-ctl",
                "--device",
                self.config.camera_device,
                f"--set-fmt-video=width={width},height={height},pixelformat=MJPG",
                "--stream-mmap",
            ]
            skip_frames = self._settle_frames(settle_seconds)
            if skip_frames:
                command.append(f"--stream-skip={skip_frames}")
            command.extend(
                [
                    "--stream-count=1",
                    f"--stream-to={output}",
                ]
            )
            commands.append(command)
        for binary in ("rpicam-still", "libcamera-still"):
            if shutil.which(binary):
                timeout_ms = max(1000, int(settle_seconds * 1000))
                commands.append(
                    [
                        binary,
                        "-n",
                        "--width",
                        str(width),
                        "--height",
                        str(height),
                        "--encoding",
                        "jpg",
                        "-q",
                        str(self.config.camera_jpeg_quality),
                        "--timeout",
                        str(timeout_ms),
                        "-o",
                        str(output),
                    ]
                )
        return commands

    def _settle_seconds(self, override: float | None) -> float:
        if override is None:
            override = getattr(self.config, "camera_snapshot_settle_seconds", 0.0)
        try:
            return max(0.0, float(override))
        except (TypeError, ValueError):
            return 0.0

    def _settle_frames(self, settle_seconds: float) -> int:
        if settle_seconds <= 0:
            return 0
        return max(1, int(round(settle_seconds * 30)))

    def _frame_size(self) -> tuple[int, int]:
        raw = self.config.camera_frame_size.lower().replace(" ", "")
        if "x" not in raw:
            return 1280, 720
        left, right = raw.split("x", 1)
        try:
            width = max(160, int(left))
            height = max(120, int(right))
        except ValueError:
            return 1280, 720
        return width, height

    def _format_command_error(
        self,
        command: list[str],
        exc: OSError | subprocess.CalledProcessError | subprocess.TimeoutExpired,
    ) -> str:
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
            return f"{command[0]} exited {exc.returncode}: {stderr}"
        if isinstance(exc, subprocess.TimeoutExpired):
            return f"{command[0]} timed out"
        return f"{command[0]} failed: {exc}"


class ContinuousCameraCapture:
    def __init__(self, config: Config):
        self.config = config
        self._container = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._latest_at = 0.0
        self._error: BaseException | None = None
        self._device_lock: CameraDeviceLock | None = None

    def start(self) -> None:
        if not self.config.camera_enabled:
            raise CameraError("camera is disabled")
        if self._thread is not None:
            raise CameraError("continuous camera is already active")
        device = Path(self.config.camera_device)
        if not device.exists():
            raise CameraError(f"camera device not found: {self.config.camera_device}")
        try:
            import av
        except ImportError as exc:
            raise CameraError("continuous camera capture requires PyAV") from exc

        self.config.camera_images_dir.mkdir(parents=True, exist_ok=True)
        device_lock = CameraDeviceLock(self.config)
        device_lock.acquire()
        try:
            self._container = av.open(
                self.config.camera_device,
                format="v4l2",
                options={
                    "video_size": self.config.camera_frame_size,
                    "input_format": "mjpeg",
                },
            )
        except BaseException:
            device_lock.release()
            raise
        self._device_lock = device_lock
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="continuous-camera-reader", daemon=True)
        self._thread.start()
        LOGGER.info("continuous camera stream opened on %s", self.config.camera_device)

    def snapshot(self, note: str = "", timeout: float | None = None) -> CameraSnapshot:
        deadline = time.monotonic() + (
            self.config.camera_capture_timeout_seconds if timeout is None else max(0.1, timeout)
        )
        with self._condition:
            while self._latest_jpeg is None and self._error is None:
                if self._stop.is_set():
                    raise CameraError("continuous camera stopped before producing a frame")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("continuous camera did not produce a frame")
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise CameraError(f"continuous camera failed: {self._error}")
            assert self._latest_jpeg is not None
            payload = self._latest_jpeg

        if len(payload) > self.config.camera_max_image_bytes:
            raise CameraError(
                f"camera snapshot is too large: {len(payload)} bytes > {self.config.camera_max_image_bytes} bytes"
            )
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = self.config.camera_images_dir / f"continuous-{stamp}-{time.monotonic_ns()}.jpg"
        output.write_bytes(payload)
        encoded = base64.b64encode(payload).decode("ascii")
        return CameraSnapshot(
            path=output,
            mime_type="image/jpeg",
            size_bytes=len(payload),
            data_url=f"data:image/jpeg;base64,{encoded}",
        )

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        container = self._container
        if container is not None:
            try:
                container.close()
            except BaseException:
                LOGGER.exception("failed to close continuous camera container")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        device_lock = self._device_lock
        if device_lock is not None:
            device_lock.release()
        self._device_lock = None
        self._container = None
        self._thread = None
        LOGGER.info("continuous camera stream closed")

    def _read_loop(self) -> None:
        try:
            assert self._container is not None
            for packet in self._container.demux(video=0):
                if self._stop.is_set():
                    return
                jpeg = jpeg_bytes_from_mjpeg_packet(bytes(packet))
                if not jpeg:
                    continue
                with self._condition:
                    self._latest_jpeg = jpeg
                    self._latest_at = time.monotonic()
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                LOGGER.exception("continuous camera reader failed")
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()


def jpeg_bytes_from_mjpeg_packet(data: bytes) -> bytes:
    start = data.find(b"\xff\xd8")
    end = data.rfind(b"\xff\xd9")
    if start < 0 or end <= start:
        return b""
    return data[start : end + 2]
