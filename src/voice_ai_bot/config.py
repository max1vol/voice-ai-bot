from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "Your name is Max Code. You are a voice assistant running on a small push-to-talk speaker. "
    "If asked who you are, say that you are Max Code. Do not introduce yourself as ChatGPT. "
    "The user's language is either English or Russian and is unlikely to be any other language. "
    "Reply in the same language as the user's latest request unless they ask otherwise. "
    "Keep replies concise, natural, and suitable for being spoken aloud."
)

REALTIME_SYSTEM_PROMPT = (
    "Your name is Max Code. You are a push-to-talk English/Russian voice assistant running on a small speaker. "
    "If asked who you are, say that you are Max Code. Do not introduce yourself as ChatGPT. "
    "The user's language is either English or Russian and is unlikely to be any other language. "
    "Be an assistant first: answer questions and follow commands concisely in the user's language. "
    "Translate only when the user asks for translation or clearly starts a translation task. "
    "If the user asks you to stop, close, disconnect, sleep, or end the session, call the close_realtime_session tool."
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
    voice_bot_backend: str
    openai_model: str
    openai_reasoning_effort: str
    openai_connectivity_host: str
    openai_connectivity_wait_seconds: float
    openai_timeout_seconds: float
    transcription_model: str
    tts_model: str
    tts_voice: str
    tts_instructions: str
    realtime_model: str
    realtime_reasoning_effort: str
    realtime_voice: str
    realtime_input_rate: int
    realtime_input_transcription_model: str
    realtime_response_timeout_seconds: float
    realtime_idle_timeout_seconds: float
    realtime_max_session_seconds: float
    realtime_silent_cooldown_seconds: float
    realtime_history_messages: int
    realtime_safety_identifier: str
    user_city: str
    user_region: str
    user_country: str
    user_timezone: str
    web_search_model: str
    web_search_reasoning_effort: str
    web_search_context_size: str
    web_search_timeout_seconds: float
    task_model: str
    task_reasoning_effort: str
    task_reasoning_summary: str
    task_timeout_seconds: float
    task_code_execution: bool
    max_background_tasks: int
    task_result_chars: int
    task_summary_chars: int
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
    scheduled_tasks_file: Path
    schedule_quiet_start: str
    schedule_quiet_end: str
    recordings_dir: Path
    tts_chunk_chars: int
    log_level: str
    memory_dir: Path = Path("/var/lib/voice-ai-bot/agent")
    memory_bootstrap_chars: int = 12000
    memory_active_context_chars: int = 1800
    memory_consolidation_enabled: bool = True
    memory_consolidation_model: str = "gpt-5.5"
    memory_consolidation_reasoning_effort: str = "high"
    memory_consolidation_debounce_seconds: float = 5.0
    memory_consolidation_shutdown_timeout_seconds: float = 30.0
    memory_consolidation_max_notes: int = 12
    memory_consolidation_max_chars: int = 16000
    openweather_api_key: str = ""
    openweather_timeout_seconds: float = 10.0
    weather_cache_seconds: float = 600.0
    voice_volume: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        backend = os.getenv("VOICE_BOT_BACKEND", "responses").strip().lower()
        if backend not in {"responses", "realtime"}:
            raise RuntimeError("VOICE_BOT_BACKEND must be either 'responses' or 'realtime'")
        realtime_input_rate = _int_env("REALTIME_INPUT_RATE", 24000)

        return cls(
            openai_api_key=api_key,
            voice_bot_backend=backend,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "high"),
            openai_connectivity_host=os.getenv("OPENAI_CONNECTIVITY_HOST", "api.openai.com"),
            openai_connectivity_wait_seconds=_float_env("OPENAI_CONNECTIVITY_WAIT_SECONDS", 120.0),
            openai_timeout_seconds=_float_env("OPENAI_TIMEOUT_SECONDS", 120.0),
            transcription_model=os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-transcribe"),
            tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            tts_voice=os.getenv("TTS_VOICE", "cedar"),
            tts_instructions=os.getenv(
                "TTS_INSTRUCTIONS",
                "Speak naturally. Match the user's language. Keep the response clear and comfortable to listen to.",
            ),
            realtime_model=os.getenv("REALTIME_MODEL", "gpt-realtime-2"),
            realtime_reasoning_effort=os.getenv("REALTIME_REASONING_EFFORT", "medium"),
            realtime_voice=os.getenv("REALTIME_VOICE", "marin"),
            realtime_input_rate=realtime_input_rate,
            realtime_input_transcription_model=os.getenv("REALTIME_INPUT_TRANSCRIPTION_MODEL", "gpt-4o-transcribe"),
            realtime_response_timeout_seconds=_float_env("REALTIME_RESPONSE_TIMEOUT_SECONDS", 90.0),
            realtime_idle_timeout_seconds=_float_env("REALTIME_IDLE_TIMEOUT_SECONDS", 45.0),
            realtime_max_session_seconds=_float_env("REALTIME_MAX_SESSION_SECONDS", 120.0),
            realtime_silent_cooldown_seconds=_float_env("REALTIME_SILENT_COOLDOWN_SECONDS", 5.0),
            realtime_history_messages=_int_env("REALTIME_HISTORY_MESSAGES", 16),
            realtime_safety_identifier=os.getenv("REALTIME_SAFETY_IDENTIFIER", "voice-ai-bot-local"),
            user_city=os.getenv("USER_CITY", "Cambridge"),
            user_region=os.getenv("USER_REGION", "Cambridgeshire"),
            user_country=os.getenv("USER_COUNTRY", "GB"),
            user_timezone=os.getenv("USER_TIMEZONE", "Europe/London"),
            web_search_model=os.getenv("WEB_SEARCH_MODEL", "gpt-5.5"),
            web_search_reasoning_effort=os.getenv("WEB_SEARCH_REASONING_EFFORT", "high"),
            web_search_context_size=os.getenv("WEB_SEARCH_CONTEXT_SIZE", "medium"),
            web_search_timeout_seconds=_float_env("WEB_SEARCH_TIMEOUT_SECONDS", 90.0),
            task_model=os.getenv("TASK_MODEL", os.getenv("WEB_SEARCH_MODEL", "gpt-5.5")),
            task_reasoning_effort=os.getenv(
                "TASK_REASONING_EFFORT", os.getenv("WEB_SEARCH_REASONING_EFFORT", "high")
            ),
            task_reasoning_summary=os.getenv("TASK_REASONING_SUMMARY", "auto"),
            task_timeout_seconds=_float_env("TASK_TIMEOUT_SECONDS", 180.0),
            task_code_execution=_bool_env("TASK_CODE_EXECUTION", True),
            max_background_tasks=_int_env("MAX_BACKGROUND_TASKS", 20),
            task_result_chars=_int_env("TASK_RESULT_CHARS", 12000),
            task_summary_chars=_int_env("TASK_SUMMARY_CHARS", 4000),
            button_gpio=_int_env("BUTTON_GPIO", 23),
            led_gpio=_int_env("LED_GPIO", 25),
            button_pull_up=_bool_env("BUTTON_PULL_UP", True),
            short_click_seconds=_float_env("SHORT_CLICK_SECONDS", 0.45),
            double_click_window_seconds=_float_env("DOUBLE_CLICK_WINDOW_SECONDS", 0.65),
            audio_capture_device=os.getenv("AUDIO_CAPTURE_DEVICE", "plughw:1,0"),
            audio_playback_device=os.getenv("AUDIO_PLAYBACK_DEVICE", "plughw:1,0"),
            record_rate=_int_env("RECORD_RATE", realtime_input_rate if backend == "realtime" else 16000),
            record_channels=_int_env("RECORD_CHANNELS", 1),
            min_record_seconds=_float_env("MIN_RECORD_SECONDS", 0.25),
            conversation_file=Path(os.getenv("CONVERSATION_FILE", "/var/lib/voice-ai-bot/conversation.json")),
            scheduled_tasks_file=Path(
                os.getenv("SCHEDULED_TASKS_FILE", "/var/lib/voice-ai-bot/scheduled_tasks.json")
            ),
            schedule_quiet_start=os.getenv("SCHEDULE_QUIET_START", "21:00"),
            schedule_quiet_end=os.getenv("SCHEDULE_QUIET_END", "07:30"),
            recordings_dir=Path(os.getenv("RECORDINGS_DIR", "/var/lib/voice-ai-bot/recordings")),
            tts_chunk_chars=_int_env("TTS_CHUNK_CHARS", 240),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            memory_dir=Path(os.getenv("MEMORY_DIR", "/var/lib/voice-ai-bot/agent")),
            memory_bootstrap_chars=_int_env("MEMORY_BOOTSTRAP_CHARS", 12000),
            memory_active_context_chars=_int_env("MEMORY_ACTIVE_CONTEXT_CHARS", 1800),
            memory_consolidation_enabled=_bool_env("MEMORY_CONSOLIDATION_ENABLED", True),
            memory_consolidation_model=os.getenv(
                "MEMORY_CONSOLIDATION_MODEL",
                os.getenv("TASK_MODEL", "gpt-5.5"),
            ),
            memory_consolidation_reasoning_effort=os.getenv(
                "MEMORY_CONSOLIDATION_REASONING_EFFORT",
                os.getenv("TASK_REASONING_EFFORT", "high"),
            ),
            memory_consolidation_debounce_seconds=_float_env("MEMORY_CONSOLIDATION_DEBOUNCE_SECONDS", 5.0),
            memory_consolidation_shutdown_timeout_seconds=_float_env(
                "MEMORY_CONSOLIDATION_SHUTDOWN_TIMEOUT_SECONDS",
                30.0,
            ),
            memory_consolidation_max_notes=_int_env("MEMORY_CONSOLIDATION_MAX_NOTES", 12),
            memory_consolidation_max_chars=_int_env("MEMORY_CONSOLIDATION_MAX_CHARS", 16000),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
            openweather_timeout_seconds=_float_env("OPENWEATHER_TIMEOUT_SECONDS", 10.0),
            weather_cache_seconds=_float_env("WEATHER_CACHE_SECONDS", 600.0),
            voice_volume=max(1, min(10, _int_env("VOICE_VOLUME", 10))),
        )
