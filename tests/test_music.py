from __future__ import annotations

import threading
import time
import wave

from test_realtime_voice import _config
from voice_ai_bot.music import MusicLibrary, MusicPlayer


class BlockingStream:
    def __init__(self):
        self.write_started = threading.Event()
        self.release = threading.Event()
        self.aborted = False
        self.writes = 0

    def __enter__(self):
        return self

    def write(self, chunk):
        self.writes += 1
        self.write_started.set()
        self.release.wait(timeout=2)

    def abort(self):
        self.aborted = True
        self.release.set()

    def close(self, check=True):
        self.release.set()

    def __exit__(self, exc_type, exc, tb):
        self.close(check=exc_type is None)


class FakePcmPlayer:
    def __init__(self):
        self.streams = []
        self._volume = 8

    def open_stream(self, rate=24000, channels=1):
        stream = BlockingStream()
        self.streams.append((rate, channels, stream))
        return stream

    def set_volume_level(self, level):
        self._volume = level
        return level

    def volume_level(self):
        return self._volume


def test_music_library_lists_duration_metadata(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_wav(music_dir / "baby-shark.wav", seconds=1.5)
    _write_wav(music_dir / "chopin-spring.wav", seconds=65.0)

    songs = MusicLibrary(music_dir).list_songs()

    assert [song.id for song in songs] == ["baby shark", "chopin spring"]
    assert songs[0].snapshot()["duration_seconds"] == 1.5
    assert songs[0].snapshot()["duration"] == "0:02"
    assert songs[1].snapshot()["duration"] == "1:05"


def test_music_library_matches_natural_song_name(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_wav(music_dir / "baby-shark.wav", seconds=1.0)

    song = MusicLibrary(music_dir).match("please play Baby Shark")

    assert song is not None
    assert song.id == "baby shark"


def test_music_player_defers_play_and_pauses_for_voice(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_wav(music_dir / "baby-shark.wav", seconds=1.0)
    config = _config(tmp_path)
    config = config.__class__(**{**config.__dict__, "music_dir": music_dir})
    player = MusicPlayer(config, FakePcmPlayer())

    queued = player.request_play("baby shark")
    assert queued["ok"]
    assert queued["deferred"]
    assert player.status()["state"] == "queued"

    player.apply_deferred_after_voice()
    first_stream = _wait_for_stream(player.player, 0)
    assert first_stream.write_started.wait(timeout=2)
    assert player.status()["state"] == "playing"

    assert player.pause_for_voice()
    assert first_stream.aborted
    assert player.status()["state"] == "paused"
    assert player.status()["pause_reason"] == "voice"

    player.apply_deferred_after_voice()
    second_stream = _wait_for_stream(player.player, 1)
    assert second_stream.write_started.wait(timeout=2)
    assert player.status()["state"] == "playing"

    player.stop()
    assert second_stream.aborted


def _write_wav(path, seconds: float):
    frames = int(24000 * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * frames)


def _wait_for_stream(player: FakePcmPlayer, index: int) -> BlockingStream:
    deadline = time.time() + 2
    while time.time() < deadline:
        if len(player.streams) > index:
            return player.streams[index][2]
        time.sleep(0.01)
    raise AssertionError(f"stream {index} was not opened")
