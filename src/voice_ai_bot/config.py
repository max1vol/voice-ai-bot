from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are a voice assistant running on a small push-to-talk speaker. "
    "The user's language is either English or Russian and is unlikely to be any other language. "
    "Reply in the same language as the user's latest request unless they ask otherwise. "
    "Keep replies concise, natural, and suitable for being spoken aloud."
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_model: str
    openai_reasoning_effort: str
    transcription_model: str
    tts_model: str
    tts_voice: str
    tts_instructions: str
    button_gpio: int
    led_gpio: int
    button_pull_up: bool
    short_click_seconds: float
    double_click_window_seconds: float
    audio_capture_device: str
    audio_playback_device: str
    record_rate: int
    record_channels: int
    min_record_seconds: float
    conversation_file: Path
    recordings_dir: Path
    tts_chunk_chars: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
            transcription_model=os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-transcribe"),
            tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            tts_voice=os.getenv("TTS_VOICE", "cedar"),
            tts_instructions=os.getenv(
                "TTS_INSTRUCTIONS",
                "Speak naturally. Match the user's language. Keep the response clear and comfortable to listen to.",
            ),
            button_gpio=_int_env("BUTTON_GPIO", 23),
            led_gpio=_int_env("LED_GPIO", 25),
            button_pull_up=_bool_env("BUTTON_PULL_UP", True),
            short_click_seconds=_float_env("SHORT_CLICK_SECONDS", 0.45),
            double_click_window_seconds=_float_env("DOUBLE_CLICK_WINDOW_SECONDS", 0.65),
            audio_capture_device=os.getenv("AUDIO_CAPTURE_DEVICE", "plughw:1,0"),
            audio_playback_device=os.getenv("AUDIO_PLAYBACK_DEVICE", "plughw:1,0"),
            record_rate=_int_env("RECORD_RATE", 16000),
            record_channels=_int_env("RECORD_CHANNELS", 1),
            min_record_seconds=_float_env("MIN_RECORD_SECONDS", 0.25),
            conversation_file=Path(os.getenv("CONVERSATION_FILE", "/var/lib/voice-ai-bot/conversation.json")),
            recordings_dir=Path(os.getenv("RECORDINGS_DIR", "/var/lib/voice-ai-bot/recordings")),
            tts_chunk_chars=_int_env("TTS_CHUNK_CHARS", 240),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
