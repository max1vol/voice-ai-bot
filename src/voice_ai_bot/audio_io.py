from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
import time
from array import array
from collections.abc import Callable
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
        self.last_stderr = ""

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
        if process.poll() is None:
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
        self.last_stderr = stderr.decode(errors="replace").strip()
        size = path.stat().st_size if path.exists() else 0
        LOGGER.info("recording stopped: %.3fs, %d bytes, arecord rc=%s", duration, size, process.returncode)
        if self.last_stderr:
            LOGGER.warning("arecord stderr: %s", self.last_stderr)
        if process.returncode not in {0, 1, -signal.SIGINT}:
            LOGGER.warning("arecord exited with %s", process.returncode)
        return path, duration

    def is_usable(self, path: Path, duration: float) -> bool:
        size = path.stat().st_size if path.exists() else 0
        if duration < self.config.min_record_seconds:
            LOGGER.info("recording ignored: duration %.3fs below minimum %.3fs", duration, self.config.min_record_seconds)
            return False
        if size <= 44:
            LOGGER.warning("recording ignored: empty WAV file %s (%d bytes)", path, size)
            return False
        return True


class PcmPlayer:
    def __init__(self, playback_device: str, volume_level: int = 10):
        self.playback_device = playback_device
        self._volume_lock = threading.Lock()
        self._volume_level = clamp_volume_level(volume_level)

    def play_pcm_stream(self, chunks) -> None:
        with self.open_stream() as stream:
            for chunk in chunks:
                stream.write(chunk)

    def open_stream(self, rate: int = 24000, channels: int = 1) -> "PcmOutputStream":
        return PcmOutputStream(self.playback_device, rate, channels, self.volume_level)

    def set_volume_level(self, level: int) -> int:
        clean_level = clamp_volume_level(level)
        with self._volume_lock:
            self._volume_level = clean_level
        LOGGER.info("software voice volume set to %d/10", clean_level)
        return clean_level

    def volume_level(self) -> int:
        with self._volume_lock:
            return self._volume_level


class PcmOutputStream:
    def __init__(
        self,
        playback_device: str,
        rate: int,
        channels: int,
        volume_getter: Callable[[], int] | None = None,
    ):
        self.playback_device = playback_device
        self.rate = rate
        self.channels = channels
        self.volume_getter = volume_getter
        self.process: subprocess.Popen[bytes] | None = None
        self.bytes_written = 0
        self._lock = threading.Lock()
        self._aborted = False

    def __enter__(self) -> "PcmOutputStream":
        command = [
            "aplay",
            "-q",
            "-D",
            self.playback_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
            "-",
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            if self._aborted:
                return
            if self.process is None or self.process.stdin is None:
                raise RuntimeError("PCM output stream is not open")
            stdin = self.process.stdin
        try:
            if self.volume_getter is not None:
                chunk = scale_pcm16(chunk, self.volume_getter())
            stdin.write(chunk)
            stdin.flush()
        except (BrokenPipeError, OSError):
            with self._lock:
                if self._aborted:
                    return
            raise
        with self._lock:
            if self._aborted:
                return
            self.bytes_written += len(chunk)

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            process = self.process
            stdin = process.stdin if process is not None else None
        if process is None:
            return
        if stdin is not None:
            try:
                stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if process.poll() is None:
            process.terminate()

    def close(self, check: bool = True) -> None:
        with self._lock:
            process = self.process
            self.process = None
            if process is None:
                return
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        try:
            stderr = process.stderr.read() if process.stderr is not None else b""
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            stderr = process.stderr.read() if process.stderr is not None else b""
            process.wait(timeout=5)
        rc = process.returncode
        if check and rc != 0 and not self._aborted:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"aplay exited with {rc}: {detail}")

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(check=exc_type is None)


def clamp_volume_level(level: int) -> int:
    try:
        clean_level = int(level)
    except (TypeError, ValueError):
        clean_level = 10
    return max(1, min(10, clean_level))


def scale_pcm16(chunk: bytes, volume_level: int) -> bytes:
    clean_level = clamp_volume_level(volume_level)
    if clean_level >= 10 or not chunk:
        return chunk
    even_length = len(chunk) - (len(chunk) % 2)
    if even_length <= 0:
        return chunk
    samples = array("h")
    samples.frombytes(chunk[:even_length])
    if sys.byteorder != "little":
        samples.byteswap()
    scale = clean_level / 10.0
    for index, sample in enumerate(samples):
        samples[index] = int(sample * scale)
    if sys.byteorder != "little":
        samples.byteswap()
    scaled = samples.tobytes()
    if even_length != len(chunk):
        scaled += chunk[even_length:]
    return scaled


class RawPcmRecorder:
    def __init__(self, config: Config):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.started_at = 0.0
        self.last_stderr = ""

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("raw recording is already active")
        command = [
            "arecord",
            "-q",
            "-D",
            self.config.audio_capture_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.config.realtime_input_rate),
            "-c",
            "1",
            "-t",
            "raw",
            "-",
        ]
        LOGGER.info("starting raw recorder: %s", " ".join(command))
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.started_at = time.monotonic()

    def read(self, size: int = 4096) -> bytes:
        if self.process is None or self.process.stdout is None:
            return b""
        return self.process.stdout.read(size)

    def stop(self) -> float:
        if self.process is None:
            return 0.0
        duration = time.monotonic() - self.started_at
        process = self.process
        self.process = None
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        stderr = process.stderr.read() if process.stderr is not None else b""
        self.last_stderr = stderr.decode(errors="replace").strip()
        LOGGER.info("raw recording stopped: %.3fs, arecord rc=%s", duration, process.returncode)
        if self.last_stderr:
            LOGGER.warning("arecord stderr: %s", self.last_stderr)
        if process.returncode not in {0, 1, -signal.SIGINT}:
            LOGGER.warning("arecord exited with %s", process.returncode)
        return duration
