from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from .audio_io import clamp_volume_level

LOGGER = logging.getLogger(__name__)


class RuntimeSettings:
    def __init__(self, path: Path, default_voice_volume: int, default_music_volume: int):
        self.path = path
        self._defaults = {
            "voice_volume": clamp_volume_level(default_voice_volume),
            "music_volume": clamp_volume_level(default_music_volume),
        }
        self._lock = threading.RLock()

    def ensure(self) -> dict[str, int]:
        with self._lock:
            values, changed = self._read_locked()
            if changed or not self.path.exists():
                self._write_locked(values)
            return dict(values)

    def volumes(self) -> dict[str, int]:
        with self._lock:
            values, _changed = self._read_locked()
            return dict(values)

    def voice_volume(self) -> int:
        return self.volumes()["voice_volume"]

    def music_volume(self) -> int:
        return self.volumes()["music_volume"]

    def set_voice_volume(self, level: int) -> dict[str, int]:
        return self._set_volume("voice_volume", level)

    def set_music_volume(self, level: int) -> dict[str, int]:
        return self._set_volume("music_volume", level)

    def _set_volume(self, key: str, level: int) -> dict[str, int]:
        with self._lock:
            values, _changed = self._read_locked()
            values[key] = clamp_volume_level(level)
            self._write_locked(values)
            return dict(values)

    def _read_locked(self) -> tuple[dict[str, int], bool]:
        values = dict(self._defaults)
        if not self.path.exists():
            return values, True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.exception("failed to read runtime settings from %s; using defaults", self.path)
            return values, True
        if not isinstance(raw, dict):
            LOGGER.warning("runtime settings file %s is not a JSON object; using defaults", self.path)
            return values, True
        changed = False
        for key in ("voice_volume", "music_volume"):
            if key in raw:
                clean = clamp_volume_level(raw[key])
                values[key] = clean
                if raw[key] != clean:
                    changed = True
            else:
                changed = True
        return values, changed

    def _write_locked(self, values: dict[str, Any]) -> None:
        clean = {
            "voice_volume": clamp_volume_level(values.get("voice_volume", self._defaults["voice_volume"])),
            "music_volume": clamp_volume_level(values.get("music_volume", self._defaults["music_volume"])),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)
