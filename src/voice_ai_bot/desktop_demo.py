from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from typing import Callable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .audio_io import PcmOutputStream, clamp_volume_level, scale_pcm16
from .config import Config
from .conversation import ConversationStore
from .memory import MemoryStore
from .memory_consolidation import MemoryConsolidator
from .music import MusicPlayer
from .realtime_voice import RealtimeConversationSession
from .scheduled_tasks import ScheduledTask, ScheduledTaskStore
from .settings import RuntimeSettings

LOGGER = logging.getLogger(__name__)


class MacRawPcmRecorder:
    def __init__(self, config: Config):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.started_at = 0.0
        self.last_stderr = ""

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("raw recording is already active")
        audio_device = _mac_audio_input_device(self.config.audio_capture_device)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{audio_device}",
            "-vn",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.config.realtime_input_rate),
            "-ac",
            "1",
            "-",
        ]
        LOGGER.info("starting macOS raw recorder: %s", " ".join(command))
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
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        stderr = process.stderr.read() if process.stderr is not None else b""
        self.last_stderr = stderr.decode(errors="replace").strip()
        LOGGER.info("macOS raw recording stopped: %.3fs, ffmpeg rc=%s", duration, process.returncode)
        if self.last_stderr:
            LOGGER.warning("ffmpeg recorder stderr: %s", self.last_stderr)
        return duration


class MacPcmPlayer:
    def __init__(self, playback_device: str = "", volume_level: int = 10, buffered_playback: bool = False):
        self.playback_device = playback_device
        self.buffered_playback = buffered_playback
        self._volume_level = clamp_volume_level(volume_level)
        self._volume_lock = threading.Lock()

    def play_pcm_stream(self, chunks) -> None:
        with self.open_stream() as stream:
            for chunk in chunks:
                stream.write(chunk)

    def open_stream(self, rate: int = 24000, channels: int = 1) -> PcmOutputStream:
        if self.buffered_playback:
            return MacBufferedPcmOutputStream(rate, channels, self.volume_level)
        return MacPcmOutputStream(rate, channels, self.volume_level)

    def set_volume_level(self, level: int) -> int:
        clean = clamp_volume_level(level)
        with self._volume_lock:
            self._volume_level = clean
        LOGGER.info("software demo volume set to %d/10", clean)
        return clean

    def volume_level(self) -> int:
        with self._volume_lock:
            return self._volume_level


class MacPcmOutputStream(PcmOutputStream):
    def __init__(self, rate: int, channels: int, volume_getter: Callable[[], int]):
        self.playback_device = "ffplay"
        self.rate = rate
        self.channels = channels
        self.volume_getter = volume_getter
        self.process: subprocess.Popen[bytes] | None = None
        self.bytes_written = 0
        self._lock = threading.Lock()
        self._aborted = False

    def __enter__(self) -> "MacPcmOutputStream":
        command = [
            "ffplay",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nodisp",
            "-autoexit",
            "-f",
            "s16le",
            "-ar",
            str(self.rate),
            "-ac",
            str(self.channels),
            "-i",
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
            chunk = scale_pcm16(chunk, self.volume_getter())
            stdin.write(chunk)
            stdin.flush()
        except (BrokenPipeError, OSError):
            with self._lock:
                self._aborted = True
            LOGGER.warning("macOS PCM playback stream closed")
            return
        with self._lock:
            if not self._aborted:
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
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        stderr = process.stderr.read() if process.stderr is not None else b""
        if check and process.returncode != 0 and not self._aborted:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffplay exited with {process.returncode}: {detail}")


class MacBufferedPcmOutputStream(PcmOutputStream):
    def __init__(self, rate: int, channels: int, volume_getter: Callable[[], int]):
        self.playback_device = "afplay"
        self.rate = rate
        self.channels = channels
        self.volume_getter = volume_getter
        self.path: Path | None = None
        self.writer: wave.Wave_write | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.bytes_written = 0
        self._lock = threading.Lock()
        self._aborted = False

    def __enter__(self) -> "MacBufferedPcmOutputStream":
        handle = tempfile.NamedTemporaryFile(prefix="voice-demo-", suffix=".wav", delete=False)
        path = Path(handle.name)
        handle.close()
        writer = wave.open(str(path), "wb")
        writer.setnchannels(self.channels)
        writer.setsampwidth(2)
        writer.setframerate(self.rate)
        with self._lock:
            self.path = path
            self.writer = writer
        return self

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        chunk = scale_pcm16(chunk, self.volume_getter())
        with self._lock:
            if self._aborted:
                return
            if self.writer is None:
                raise RuntimeError("PCM output stream is not open")
            self.writer.writeframes(chunk)
            self.bytes_written += len(chunk)

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            writer = self.writer
            self.writer = None
            process = self.process
            path = self.path
        if writer is not None:
            writer.close()
        if process is not None and process.poll() is None:
            process.terminate()
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self, check: bool = True) -> None:
        with self._lock:
            writer = self.writer
            self.writer = None
            path = self.path
            aborted = self._aborted
            bytes_written = self.bytes_written
        if writer is not None:
            writer.close()
        if path is None:
            return
        if aborted or bytes_written <= 0:
            path.unlink(missing_ok=True)
            return

        process = subprocess.Popen(["afplay", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self._lock:
            if self._aborted:
                process.terminate()
            else:
                self.process = process

        duration = bytes_written / max(1, self.rate * max(1, self.channels) * 2)
        timeout = max(30.0, duration + 15.0)
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=5)
        finally:
            with self._lock:
                if self.process is process:
                    self.process = None
                aborted = self._aborted
            path.unlink(missing_ok=True)

        if check and process.returncode != 0 and not aborted:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"afplay exited with {process.returncode}: {detail}")


class DesktopDemoApp:
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.conversation = ConversationStore(config.conversation_file)
        self.memory = MemoryStore(config)
        self.memory.ensure_workspace()
        self.memory_consolidator = MemoryConsolidator(config, self.memory)
        self.scheduled_tasks = ScheduledTaskStore(config)
        self.settings = RuntimeSettings(config.settings_file, config.voice_volume, config.music_volume)
        self.settings.ensure()
        self.player = MacPcmPlayer(volume_level=self.settings.voice_volume(), buffered_playback=True)
        self.music = MusicPlayer(config, MacPcmPlayer(volume_level=self.settings.music_volume()), settings=self.settings)
        self.session = RealtimeConversationSession(
            config,
            self.scheduled_tasks,
            self.memory,
            music=self.music,
            settings=self.settings,
            recorder_factory=lambda: MacRawPcmRecorder(config),
        )
        self.session.player = self.player

        self._space_down = False
        self._recording_ready = False
        self._start_in_progress = False
        self._space_release_after_id: str | None = None
        self._closing = False
        self._action_lock = threading.Lock()
        self._status_hold_until = 0.0

        self.status_var = tk.StringVar(value="Ready. Hold Space to talk.")
        self.detail_var = tk.StringVar(value=f"Mic device {os.getenv('VOICE_DEMO_AUDIO_INPUT_DEVICE', '1')}.")
        self._build_ui()
        self.root.bind_all("<KeyPress-space>", self._on_space_press, add=True)
        self.root.bind_all("<KeyRelease-space>", self._on_space_release, add=True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._tick)

    def _build_ui(self) -> None:
        self.root.title("SipQuest Laptop Demo")
        self.root.geometry("760x520")
        self.root.configure(bg="#101418")

        title = tk.Label(
            self.root,
            text="SipQuest Laptop Demo",
            font=("Helvetica", 24, "bold"),
            fg="#f4f7fb",
            bg="#101418",
        )
        title.pack(pady=(22, 6))

        instruction = tk.Label(
            self.root,
            text="Hold Space to record. Release Space to send. Press Space during a reply to interrupt.",
            font=("Helvetica", 15),
            fg="#cbd5e1",
            bg="#101418",
        )
        instruction.pack(pady=(0, 18))

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 18, "bold"),
            fg="#87f7b5",
            bg="#101418",
        )
        status.pack(pady=(0, 4))

        detail = tk.Label(
            self.root,
            textvariable=self.detail_var,
            font=("Helvetica", 12),
            fg="#94a3b8",
            bg="#101418",
        )
        detail.pack(pady=(0, 18))

        self.log = ScrolledText(
            self.root,
            height=16,
            font=("Menlo", 13),
            bg="#0b0f14",
            fg="#e5edf5",
            insertbackground="#e5edf5",
            relief=tk.FLAT,
            padx=12,
            pady=12,
            wrap=tk.WORD,
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))
        self.log.configure(state=tk.DISABLED)
        self._append_log("Ready. Click this window, then hold Space to talk.\n")

    def _on_space_press(self, _event) -> None:
        if self._space_release_after_id is not None:
            self.root.after_cancel(self._space_release_after_id)
            self._space_release_after_id = None
        if self._space_down:
            return "break"
        self._space_down = True
        threading.Thread(target=self._start_recording, name="desktop-demo-start", daemon=True).start()
        return "break"

    def _on_space_release(self, _event) -> None:
        if self._space_release_after_id is not None:
            self.root.after_cancel(self._space_release_after_id)
        self._space_release_after_id = self.root.after(80, self._finish_space_release)
        return "break"

    def _finish_space_release(self) -> None:
        self._space_release_after_id = None
        self._space_down = False
        threading.Thread(target=self._finish_recording, name="desktop-demo-finish", daemon=True).start()

    def _start_recording(self) -> None:
        with self._action_lock:
            if self._recording_ready or self._start_in_progress:
                return
            self._start_in_progress = True
        self._set_status("Starting recorder...", "Opening microphone and realtime session.")
        try:
            self.session.pause_music_for_voice()
            self.session.begin_turn(self.conversation.load())
            with self._action_lock:
                self._recording_ready = True
                self._start_in_progress = False
                should_finish = not self._space_down
            self._set_status("Recording...", "Release Space to send.")
            if should_finish:
                self._finish_recording()
        except BaseException as exc:
            with self._action_lock:
                self._recording_ready = False
                self._start_in_progress = False
            LOGGER.exception("desktop demo failed to start recording")
            self._show_error(exc)

    def _finish_recording(self) -> None:
        with self._action_lock:
            if self._start_in_progress and not self._recording_ready:
                return
            if not self._recording_ready:
                return
            self._recording_ready = False
        self._set_status("Sending...", "Committing recorded audio.")
        try:
            duration = self.session.stop_recording()
            if duration < self.config.min_record_seconds:
                self.session.clear_pending_input()
                self.session.apply_deferred_music_after_voice()
                self._set_status("Ignored", f"Recording was too short: {duration:.2f}s.")
                return
            self.session.commit_recording()
            self._set_status("Thinking...", "Waiting for SipQuest.")
        except BaseException as exc:
            LOGGER.exception("desktop demo failed to finish recording")
            self.session.close()
            self._show_error(exc)

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            self._drain_completed_turns()
            if not self.session.is_voice_busy:
                self.session.apply_deferred_music_after_voice()
            self.session.check_health()
            self.session.close_if_too_old()
            self.session.close_if_idle()
            self.session.cool_down_if_silent()
            self._run_due_scheduled_tasks()
            self._run_due_background_wakeups()
            self._sync_status()
        except BaseException as exc:
            LOGGER.exception("desktop demo loop failed")
            self.session.close()
            self._show_error(exc)
        finally:
            self.root.after(100, self._tick)

    def _drain_completed_turns(self) -> None:
        for result in self.session.pop_completed_turns():
            if result.user_text:
                self._append_log(f"You: {result.user_text}\n")
            if result.assistant_text:
                self._append_log(f"SipQuest: {result.assistant_text}\n\n")
                self.conversation.append_pair(result.user_text or "[voice input]", result.assistant_text)
                self.memory.append_turn(result.user_text or "[voice input]", result.assistant_text)
                self.memory_consolidator.request("desktop demo turn")
            elif result.requested_close and result.user_text:
                self.conversation.append_pair(result.user_text, "[realtime session closed]")
                self.memory.append_turn(result.user_text, "[realtime session closed]")
                self.memory_consolidator.request("desktop demo close")

    def _run_due_scheduled_tasks(self) -> None:
        if self.session.is_voice_busy:
            return
        for task in self.scheduled_tasks.due(limit=1):
            if self._start_scheduled_task(task):
                return

    def _start_scheduled_task(self, task: ScheduledTask) -> bool:
        if task.action == "speak":
            paused_music = self.session.pause_music_for_voice()
            started = self.session.trigger_scheduled_speech(
                self.conversation.load(),
                title=task.title,
                prompt=task.prompt,
            )
            if not started:
                if paused_music:
                    self.session.apply_deferred_music_after_voice()
                return False
            self.scheduled_tasks.mark_started(task.id)
            self._set_status("Scheduled Speech", task.title)
            return True
        if task.action == "background_task":
            result = self.session.tasks.start(
                task.prompt,
                history=self.conversation.load(),
                title=task.title,
                source="scheduled",
            )
            if not result.get("ok"):
                LOGGER.warning("failed to start scheduled background task %s: %s", task.id, result)
                return False
            self.scheduled_tasks.mark_started(task.id)
            self._set_status("Background Task", task.title)
            return True
        return False

    def _run_due_background_wakeups(self) -> None:
        if self.session.is_voice_busy:
            return
        if self.scheduled_tasks.is_quiet_time(datetime.now(ZoneInfo(self.config.user_timezone))):
            return
        for task in self.session.pending_background_wakeups(limit=1):
            paused_music = self.session.pause_music_for_voice()
            if self.session.trigger_background_task_wakeup(self.conversation.load(), task):
                task_id = str(task.get("id") or "")
                wakeup = task.get("wakeup") if isinstance(task.get("wakeup"), dict) else {}
                self.session.mark_background_wakeup_reported(task_id, str(wakeup.get("message_id") or ""))
                self._set_status("Background Update", task_id)
                return
            if paused_music:
                self.session.apply_deferred_music_after_voice()

    def _sync_status(self) -> None:
        if time.monotonic() < self._status_hold_until:
            return
        if self._recording_ready or self._start_in_progress:
            return
        if self.session.is_responding:
            self._set_status("Speaking...", "Press Space to interrupt.")
        elif self.session.is_voice_busy:
            self._set_status("Working...", "Waiting for current turn.")
        else:
            self._set_status("Ready. Hold Space to talk.", "Release Space to send.")

    def _set_status(self, status: str, detail: str = "", hold_seconds: float = 0.0) -> None:
        self._status_hold_until = time.monotonic() + hold_seconds if hold_seconds > 0 else 0.0
        self.root.after(0, self.status_var.set, status)
        self.root.after(0, self.detail_var.set, detail)

    def _show_error(self, exc: BaseException) -> None:
        detail = _user_facing_error(exc)
        self._append_log(f"Error: {detail}\n")
        self._set_status("Error", detail, hold_seconds=8.0)

    def _append_log(self, text: str) -> None:
        def append() -> None:
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

        self.root.after(0, append)

    def close(self) -> None:
        self._closing = True
        self.root.unbind_all("<KeyPress-space>")
        self.root.unbind_all("<KeyRelease-space>")
        try:
            self.session.music.stop()
            self.session.close()
            self.memory_consolidator.flush()
        finally:
            self.root.destroy()


def configure_demo_environment() -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=True)
    demo_dir = repo_root / ".desktop-demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "recordings").mkdir(exist_ok=True)
    (demo_dir / "images").mkdir(exist_ok=True)

    demo_overrides = {
        "VOICE_BOT_BACKEND": "realtime",
        "REALTIME_TURN_DETECTION": "manual",
        "REALTIME_SILENT_COOLDOWN_SECONDS": "15",
        "AUDIO_CAPTURE_DEVICE": "macos",
        "AUDIO_PLAYBACK_DEVICE": "macos",
        "CONVERSATION_FILE": str(demo_dir / "conversation.json"),
        "SCHEDULED_TASKS_FILE": str(demo_dir / "scheduled_tasks.json"),
        "SETTINGS_FILE": str(demo_dir / "settings.json"),
        "RECORDINGS_DIR": str(demo_dir / "recordings"),
        "MEMORY_DIR": str(demo_dir / "agent"),
        "CAMERA_CAPTURE_ON_BUTTON_PRESS": "false",
        "CAMERA_IMAGES_DIR": str(demo_dir / "images"),
        "MUSIC_DIR": os.getenv("VOICE_DEMO_MUSIC_DIR", str(repo_root / ".local-music")),
    }
    for key, value in demo_overrides.items():
        os.environ[key] = value
    os.environ.setdefault("LOG_LEVEL", "INFO")
    return Config.from_env()


def _mac_audio_input_device(configured_device: str) -> str:
    env_device = os.getenv("VOICE_DEMO_AUDIO_INPUT_DEVICE", "").strip()
    if env_device:
        return env_device
    configured_device = configured_device.strip()
    if configured_device and not configured_device.startswith(("hw:", "plughw:", "macos")):
        return configured_device
    return "1"


def _user_facing_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if "Incorrect API key" in text:
        return "OpenAI rejected OPENAI_API_KEY in .env."
    if "No such file or directory: 'arecord'" in text:
        return "Desktop demo recorder is misconfigured for macOS."
    return text or exc.__class__.__name__


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    config = configure_demo_environment()
    root = tk.Tk()
    app = DesktopDemoApp(root, config)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()


if __name__ == "__main__":
    main()
