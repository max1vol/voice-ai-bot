from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import av
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from av.error import FFmpegError

from .audio_io import PcmOutputStream, PcmPlayer, clamp_volume_level
from .config import Config
from .settings import RuntimeSettings

LOGGER = logging.getLogger(__name__)
SUPPORTED_MUSIC_SUFFIXES = (".opus", ".ogg", ".wav")
PREFERRED_MUSIC_SUFFIX_ORDER = {".opus": 0, ".ogg": 1, ".wav": 2}
PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_CHANNELS = 1
PLAYBACK_SAMPLE_WIDTH = 2
PLAYBACK_FRAME_BYTES = PLAYBACK_CHANNELS * PLAYBACK_SAMPLE_WIDTH
GENERIC_MUSIC_QUERY_WORDS = {
    "a",
    "an",
    "anything",
    "can",
    "could",
    "dance",
    "fun",
    "music",
    "on",
    "play",
    "playing",
    "please",
    "put",
    "random",
    "some",
    "something",
    "song",
    "songs",
    "start",
    "track",
    "tracks",
    "tune",
    "tunes",
    "upbeat",
    "whatever",
    "you",
}


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int
    audio_format: str

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
            "format": self.audio_format,
        }


class MusicLibrary:
    def __init__(self, root: Path):
        self.root = root

    def list_songs(self) -> list[Song]:
        if not self.root.exists():
            return []
        songs_by_id: dict[str, Song] = {}
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_MUSIC_SUFFIXES:
                continue
            song = self._read_song(path)
            if song is not None:
                existing = songs_by_id.get(song.id)
                if existing is None or _music_suffix_order(path) < _music_suffix_order(existing.path):
                    songs_by_id[song.id] = song
        return sorted(songs_by_id.values(), key=lambda song: song.title.casefold())

    def match(self, query: str) -> Song | None:
        songs = self.list_songs()
        if not songs:
            return None
        needle = normalize_song_text(query)
        if not needle:
            return songs[0]
        if _is_generic_music_query(needle):
            return songs[0]
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
            with av.open(str(path)) as container:
                audio_stream = _first_audio_stream(container)
                if audio_stream is None:
                    LOGGER.warning("skipping %s: no audio stream", path)
                    return None
                duration_seconds = _audio_duration_seconds(container, audio_stream)
                codec_name = audio_stream.codec_context.name or path.suffix.lstrip(".")
        except (FFmpegError, OSError, ValueError):
            LOGGER.exception("failed to read music file %s", path)
            return None
        return Song(
            id=normalize_song_text(path.stem),
            title=title_from_stem(path.stem),
            path=path,
            duration_seconds=duration_seconds,
            sample_rate=PLAYBACK_SAMPLE_RATE,
            channels=PLAYBACK_CHANNELS,
            sample_width=PLAYBACK_SAMPLE_WIDTH,
            audio_format=codec_name,
        )


class MusicPlayer:
    def __init__(self, config: Config, player: PcmPlayer | None = None, settings: RuntimeSettings | None = None):
        self.config = config
        self.settings = settings or RuntimeSettings(config.settings_file, config.voice_volume, config.music_volume)
        self.library = MusicLibrary(config.music_dir)
        music_volume = self.settings.music_volume()
        self.player = player or PcmPlayer(config.audio_playback_device, volume_level=music_volume)
        if player is not None:
            self.player.set_volume_level(music_volume)
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
        volumes = self.settings.set_music_volume(clamp_volume_level(level))
        clean = self.player.set_volume_level(volumes["music_volume"])
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
            start_seconds = start_frame / song.sample_rate if song.sample_rate else 0.0
            with av.open(str(song.path)) as container:
                audio_stream = _first_audio_stream(container)
                if audio_stream is None:
                    raise RuntimeError(f"music file has no audio stream: {song.path}")
                if start_seconds > 0:
                    _seek_audio_stream(container, audio_stream, start_seconds)
                resampler = AudioResampler(format="s16", layout="mono", rate=song.sample_rate)
                played_frames = max(0, start_frame)
                with self.player.open_stream(rate=song.sample_rate, channels=song.channels) as stream:
                    with self._lock:
                        if self._generation != generation:
                            return
                        self._active_stream = stream
                    for chunk in _decoded_pcm_chunks(container, audio_stream, resampler, start_seconds):
                        if stop_event.is_set():
                            return
                        if not chunk:
                            continue
                        stream.write(chunk)
                        played_frames += len(chunk) // PLAYBACK_FRAME_BYTES
                        with self._lock:
                            if self._generation != generation:
                                return
                            self._position_frames = played_frames
            with self._lock:
                if self._generation == generation:
                    self._state = "stopped"
                    self._current = None
                    self._position_frames = 0
                    self._pause_reason = ""
                    self._active_stream = None
            LOGGER.info("music playback finished: %s", song.title)
            return
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


def _is_generic_music_query(text: str) -> bool:
    tokens = set(text.split())
    return bool(tokens) and tokens.issubset(GENERIC_MUSIC_QUERY_WORDS)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _music_suffix_order(path: Path) -> int:
    return PREFERRED_MUSIC_SUFFIX_ORDER.get(path.suffix.casefold(), len(PREFERRED_MUSIC_SUFFIX_ORDER))


def _first_audio_stream(container: av.container.InputContainer):
    for stream in container.streams:
        if stream.type == "audio":
            return stream
    return None


def _audio_duration_seconds(container: av.container.InputContainer, audio_stream) -> float:
    if audio_stream.duration is not None and audio_stream.time_base is not None:
        return max(0.0, float(audio_stream.duration * audio_stream.time_base))
    if container.duration is not None:
        return max(0.0, float(container.duration) / float(av.time_base))
    return 0.0


def _seek_audio_stream(container: av.container.InputContainer, audio_stream, start_seconds: float) -> None:
    if start_seconds <= 0 or audio_stream.time_base is None:
        return
    target = max(0, int(start_seconds / float(audio_stream.time_base)))
    try:
        container.seek(target, backward=True, any_frame=False, stream=audio_stream)
    except (FFmpegError, OSError, ValueError):
        LOGGER.warning("music seek failed for %s seconds", round(start_seconds, 3), exc_info=True)


def _decoded_pcm_chunks(
    container: av.container.InputContainer,
    audio_stream,
    resampler: AudioResampler,
    start_seconds: float,
):
    drop_until_seconds = max(0.0, start_seconds)
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            skip_frames = 0
            frame_start = _frame_start_seconds(frame)
            frame_duration = (frame.samples / frame.sample_rate) if frame.sample_rate else 0.0
            frame_end = frame_start + frame_duration if frame_start is not None else None
            if drop_until_seconds > 0:
                if frame_start is None or frame_end is None:
                    drop_until_seconds = 0.0
                elif frame_end <= drop_until_seconds:
                    continue
                else:
                    if frame_start < drop_until_seconds:
                        skip_frames = max(0, int(round((drop_until_seconds - frame_start) * PLAYBACK_SAMPLE_RATE)))
                    drop_until_seconds = 0.0
            for out_frame in _coerce_audio_frames(resampler.resample(frame)):
                chunk = bytes(out_frame.planes[0])
                if skip_frames > 0:
                    chunk_frames = out_frame.samples
                    trim_now = min(skip_frames, chunk_frames)
                    chunk = _trim_pcm_frames(chunk, trim_now)
                    skip_frames -= trim_now
                if chunk:
                    yield chunk
    for out_frame in _coerce_audio_frames(resampler.resample(None)):
        chunk = bytes(out_frame.planes[0])
        if chunk:
            yield chunk


def _coerce_audio_frames(result) -> list[AudioFrame]:
    if result is None:
        return []
    if isinstance(result, list):
        return [frame for frame in result if frame is not None]
    return [result]


def _frame_start_seconds(frame: AudioFrame) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


def _trim_pcm_frames(chunk: bytes, skip_frames: int) -> bytes:
    if skip_frames <= 0:
        return chunk
    byte_offset = min(len(chunk), skip_frames * PLAYBACK_FRAME_BYTES)
    return chunk[byte_offset:]
