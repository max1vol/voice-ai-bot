from __future__ import annotations

import logging
import threading
import time
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .audio_io import PcmOutputStream, PcmPlayer, clamp_volume_level
from .config import Config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int

    @property
    def duration(self) -> str:
        total = max(0, int(round(self.duration_seconds)))
        minutes, seconds = divmod(total, 60)
        return f"{minutes}:{seconds:02d}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "duration_seconds": round(self.duration_seconds, 3),
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": "pcm_s16le_wav",
        }


class MusicLibrary:
    def __init__(self, root: Path):
        self.root = root

    def list_songs(self) -> list[Song]:
        if not self.root.exists():
            return []
        songs = []
        for path in sorted(self.root.glob("*.wav")):
            song = self._read_song(path)
            if song is not None:
                songs.append(song)
        return sorted(songs, key=lambda song: song.title.casefold())

    def match(self, query: str) -> Song | None:
        songs = self.list_songs()
        if not songs:
            return None
        needle = normalize_song_text(query)
        if not needle:
            return None
        for song in songs:
            if song.id == needle:
                return song
        scored = []
        needle_tokens = set(needle.split())
        for song in songs:
            haystack = normalize_song_text(song.title)
            haystack_tokens = set(haystack.split())
            if needle == haystack:
                score = 1.0
            elif needle in haystack or haystack in needle:
                score = 0.92
            elif needle_tokens and needle_tokens.issubset(haystack_tokens):
                score = 0.9
            else:
                score = SequenceMatcher(None, needle, haystack).ratio()
            scored.append((score, song))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] < 0.55:
            return None
        return scored[0][1]

    def _read_song(self, path: Path) -> Song | None:
        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.getnframes()
        except (wave.Error, OSError):
            LOGGER.exception("failed to read music file %s", path)
            return None
        if sample_width != 2:
            LOGGER.warning("skipping %s: expected 16-bit PCM WAV, got sample width %s", path, sample_width)
            return None
        duration_seconds = frames / sample_rate if sample_rate else 0.0
        return Song(
            id=normalize_song_text(path.stem),
            title=title_from_stem(path.stem),
            path=path,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )


class MusicPlayer:
    def __init__(self, config: Config, player: PcmPlayer | None = None):
        self.config = config
        self.library = MusicLibrary(config.music_dir)
        self.player = player or PcmPlayer(config.audio_playback_device, volume_level=config.music_volume)
        self._lock = threading.RLock()
        self._state = "stopped"
        self._current: Song | None = None
        self._position_frames = 0
        self._pause_reason = ""
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._active_stream: PcmOutputStream | None = None
        self._generation = 0
        self._pending_action: dict[str, Any] | None = None

    def list(self) -> dict[str, Any]:
        return {"ok": True, "songs": [song.snapshot() for song in self.library.list_songs()], "status": self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            current = self._current.snapshot() if self._current is not None else None
            position_seconds = (
                self._position_frames / self._current.sample_rate
                if self._current is not None and self._current.sample_rate
                else 0.0
            )
            pending = dict(self._pending_action) if self._pending_action else None
            return {
                "ok": True,
                "state": self._state,
                "current": current,
                "position_seconds": round(position_seconds, 3),
                "position": format_duration(position_seconds),
                "pause_reason": self._pause_reason,
                "volume": self.player.volume_level(),
                "pending_action": pending,
            }

    def request_play(self, query: str) -> dict[str, Any]:
        song = self.library.match(query)
        if song is None:
            return {
                "ok": False,
                "error": f"unknown song: {query or '<empty>'}",
                "songs": [item.snapshot() for item in self.library.list_songs()],
            }
        self.stop(clear_current=True)
        with self._lock:
            self._current = song
            self._position_frames = 0
            self._state = "queued"
            self._pause_reason = ""
            self._pending_action = {"action": "play", "song_id": song.id, "title": song.title}
        LOGGER.info("queued music playback: %s", song.title)
        return {"ok": True, "deferred": True, "song": song.snapshot(), "status": self.status()}

    def request_resume(self) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                return {"ok": False, "error": "no song is selected"}
            if self._state == "playing":
                return {"ok": True, "deferred": False, "status": self.status()}
            self._pending_action = {
                "action": "resume",
                "song_id": self._current.id,
                "title": self._current.title,
            }
            self._state = "queued"
            self._pause_reason = ""
        LOGGER.info("queued music resume")
        return {"ok": True, "deferred": True, "status": self.status()}

    def pause(self, reason: str = "user") -> dict[str, Any]:
        changed = self._pause_locked(reason)
        return {"ok": True, "changed": changed, "status": self.status()}

    def pause_for_voice(self) -> bool:
        return self._pause_locked("voice")

    def stop(self, clear_current: bool = True) -> dict[str, Any]:
        stream = None
        with self._lock:
            self._pending_action = None
            self._generation += 1
            if self._stop_event is not None:
                self._stop_event.set()
            stream = self._active_stream
            self._active_stream = None
            self._state = "stopped"
            self._pause_reason = ""
            self._position_frames = 0
            if clear_current:
                self._current = None
        if stream is not None:
            stream.abort()
        LOGGER.info("music stopped")
        return {"ok": True, "status": self.status()}

    def set_volume(self, level: int) -> dict[str, Any]:
        clean = self.player.set_volume_level(clamp_volume_level(level))
        return {"ok": True, "volume": clean, "scale": clean / 10.0, "status": self.status()}

    def apply_deferred_after_voice(self) -> None:
        action = None
        with self._lock:
            if self._pending_action:
                action = dict(self._pending_action)
                self._pending_action = None
        if action:
            if action.get("action") == "play":
                self._start_current_from(0)
            elif action.get("action") == "resume":
                self._resume_now()
            return
        self.resume_if_paused_for_voice()

    def resume_if_paused_for_voice(self) -> bool:
        with self._lock:
            if self._state != "paused" or self._pause_reason != "voice":
                return False
        self._resume_now()
        return True

    def _pause_locked(self, reason: str) -> bool:
        stream = None
        with self._lock:
            if self._state != "playing":
                return False
            self._state = "paused"
            self._pause_reason = reason
            self._pending_action = None
            self._generation += 1
            if self._stop_event is not None:
                self._stop_event.set()
            stream = self._active_stream
            self._active_stream = None
        if stream is not None:
            stream.abort()
        LOGGER.info("music paused: %s", reason)
        return True

    def _resume_now(self) -> None:
        with self._lock:
            if self._current is None:
                return
            start_frame = self._position_frames
        self._start_current_from(start_frame)

    def _start_current_from(self, start_frame: int) -> None:
        with self._lock:
            song = self._current
            if song is None:
                return
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._state = "playing"
            self._pause_reason = ""
            self._position_frames = max(0, start_frame)
        thread = threading.Thread(
            target=self._play_loop,
            args=(song, self._position_frames, generation, stop_event),
            name=f"music-{song.id}",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        LOGGER.info("music playback started: %s at frame %s", song.title, start_frame)

    def _play_loop(self, song: Song, start_frame: int, generation: int, stop_event: threading.Event) -> None:
        stream = None
        try:
            with wave.open(str(song.path), "rb") as wav:
                wav.setpos(min(max(0, start_frame), wav.getnframes()))
                with self.player.open_stream(rate=wav.getframerate(), channels=wav.getnchannels()) as stream:
                    with self._lock:
                        if self._generation != generation:
                            return
                        self._active_stream = stream
                    while not stop_event.is_set():
                        chunk = wav.readframes(2048)
                        if not chunk:
                            with self._lock:
                                if self._generation == generation:
                                    self._state = "stopped"
                                    self._current = None
                                    self._position_frames = 0
                                    self._pause_reason = ""
                                    self._active_stream = None
                            LOGGER.info("music playback finished: %s", song.title)
                            return
                        stream.write(chunk)
                        with self._lock:
                            if self._generation != generation:
                                return
                            self._position_frames = wav.tell()
        except BaseException:
            with self._lock:
                if self._generation == generation:
                    self._state = "stopped"
                    self._pause_reason = ""
                    self._active_stream = None
            LOGGER.exception("music playback failed: %s", song.title)
        finally:
            with self._lock:
                if self._generation == generation and self._active_stream is stream:
                    self._active_stream = None


def normalize_song_text(text: str) -> str:
    clean = []
    for char in text.casefold().replace("_", " ").replace("-", " "):
        clean.append(char if char.isalnum() else " ")
    return " ".join("".join(clean).split())


def title_from_stem(stem: str) -> str:
    words = normalize_song_text(stem).split()
    return " ".join(word.capitalize() for word in words)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"
