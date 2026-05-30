from __future__ import annotations

import logging
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

LOGGER = logging.getLogger(__name__)


class Recorder:
    def __init__(self, config: Config):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.path: Path | None = None
        self.started_at = 0.0

    def start(self) -> Path:
        if self.process is not None:
            raise RuntimeError("recording is already active")
        self.config.recordings_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.config.recordings_dir / f"recording-{stamp}.wav"
        command = [
            "arecord",
            "-q",
            "-D",
            self.config.audio_capture_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.config.record_rate),
            "-c",
            str(self.config.record_channels),
            "-t",
            "wav",
            str(self.path),
        ]
        LOGGER.info("starting recorder: %s", " ".join(command))
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.started_at = time.monotonic()
        return self.path

    def stop(self) -> tuple[Path, float]:
        if self.process is None or self.path is None:
            raise RuntimeError("recording is not active")
        duration = time.monotonic() - self.started_at
        process = self.process
        path = self.path
        self.process = None
        self.path = None
        process.send_signal(signal.SIGINT)
        try:
            _, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate(timeout=1)
        if process.returncode not in {0, 1, -signal.SIGINT}:
            LOGGER.warning("arecord exited with %s: %s", process.returncode, stderr.decode(errors="replace").strip())
        return path, duration

    def is_usable(self, path: Path, duration: float) -> bool:
        if duration < self.config.min_record_seconds:
            return False
        return path.exists() and path.stat().st_size > 44


class PcmPlayer:
    def __init__(self, playback_device: str):
        self.playback_device = playback_device

    def play_pcm_stream(self, chunks) -> None:
        command = [
            "aplay",
            "-q",
            "-D",
            self.playback_device,
            "-f",
            "S16_LE",
            "-r",
            "24000",
            "-c",
            "1",
            "-t",
            "raw",
            "-",
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for chunk in chunks:
                if chunk:
                    process.stdin.write(chunk)
                    process.stdin.flush()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            rc = process.wait(timeout=30)
            if rc != 0:
                raise RuntimeError(f"aplay exited with {rc}")
