from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import certifi
import pytest
import websocket

from voice_ai_bot.config import Config
from voice_ai_bot.memory import MemoryStore
from voice_ai_bot.realtime_voice import (
    ADD_SCHEDULED_TASK_TOOL_NAME,
    CLOSE_TOOL_NAME,
    DELETE_SCHEDULED_TASK_TOOL_NAME,
    GET_MUSIC_STATUS_TOOL_NAME,
    GET_WEATHER_TOOL_NAME,
    LIST_MUSIC_TOOL_NAME,
    LIST_SCHEDULED_TASKS_TOOL_NAME,
    MEMORY_ADD_TOOL_NAME,
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_GET_SOURCE_TOOL_NAME,
    MEMORY_LIST_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_UPDATE_TOOL_NAME,
    PAUSE_MUSIC_TOOL_NAME,
    PLAY_MUSIC_TOOL_NAME,
    RESUME_MUSIC_TOOL_NAME,
    SET_MUSIC_VOLUME_TOOL_NAME,
    SET_VOICE_VOLUME_TOOL_NAME,
    START_TASK_TOOL_NAME,
    STEER_TASK_TOOL_NAME,
    STOP_MUSIC_TOOL_NAME,
    TASK_STATUS_TOOL_NAME,
    extract_response_text,
    parse_tool_arguments,
    realtime_session_instructions,
    realtime_tools,
    realtime_turn_instructions,
    response_function_calls,
)
from voice_ai_bot.scheduled_tasks import ScheduledTaskStore, parse_local_datetime


def _integration_enabled() -> bool:
    return os.getenv("VOICE_AI_BOT_REALTIME_INTEGRATION", "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _integration_enabled() or not os.getenv("OPENAI_API_KEY"),
    reason="set VOICE_AI_BOT_REALTIME_INTEGRATION=1 and OPENAI_API_KEY to run live realtime integration tests",
)


@dataclass
class RealtimeEvalResult:
    prompt: str
    texts: list[str]
    function_calls: list[dict[str, Any]]
    final_text: str

    @property
    def combined_text(self) -> str:
        return " ".join(text.strip() for text in self.texts if text.strip()).strip()


class EvalToolExecutor:
    def __init__(self, config: Config):
        self.config = config
        self.memory = MemoryStore(config)
        self.memory.ensure_workspace()
        self.scheduled = ScheduledTaskStore(config)
        self.background_tasks: dict[str, dict[str, Any]] = {}
        self.voice_volume = config.voice_volume
        self.music_volume = config.music_volume
        self.music_state = "stopped"
        self.music_title = ""
        self.call_log: list[dict[str, Any]] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append({"name": name, "arguments": dict(arguments)})
        if name == CLOSE_TOOL_NAME:
            return {"ok": True, "closed": True}
        if name == ADD_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled.add(
                title=str(arguments.get("title") or ""),
                prompt=str(arguments.get("prompt") or ""),
                run_at=str(arguments.get("run_at") or ""),
                action=str(arguments.get("action") or "speak"),
                repeat=str(arguments.get("repeat") or "once"),
            )
        if name == LIST_SCHEDULED_TASKS_TOOL_NAME:
            return self.scheduled.list(include_inactive=bool(arguments.get("include_inactive")))
        if name == DELETE_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled.delete(str(arguments.get("task_id") or ""))
        if name == SET_VOICE_VOLUME_TOOL_NAME:
            self.voice_volume = max(1, min(10, int(arguments.get("level") or self.voice_volume)))
            return {"ok": True, "volume": self.voice_volume, "scale": self.voice_volume / 10.0}
        if name == GET_WEATHER_TOOL_NAME:
            return self._weather(arguments)
        if name == START_TASK_TOOL_NAME:
            task_id = f"task_eval_{uuid.uuid4().hex[:8]}"
            task = {
                "id": task_id,
                "title": str(arguments.get("title") or "Background task"),
                "status": "running",
                "wakeup_on_complete": bool(arguments.get("wakeup_on_complete")),
                "wakeup_on_progress": bool(arguments.get("wakeup_on_progress")),
            }
            self.background_tasks[task_id] = task
            return {"ok": True, "task": task}
        if name == STEER_TASK_TOOL_NAME:
            task_id = str(arguments.get("task_id") or "")
            if not task_id and self.background_tasks:
                task_id = next(reversed(self.background_tasks))
            task = self.background_tasks.get(task_id)
            if not task:
                return {"ok": False, "error": "unknown task"}
            return {"ok": True, "task": task, "steered": True}
        if name == "list_background_tasks":
            return {"ok": True, "tasks": list(self.background_tasks.values())}
        if name == "get_background_task":
            task_id = str(arguments.get("task_id") or "")
            task = self.background_tasks.get(task_id)
            return {"ok": bool(task), "task": task, "error": "" if task else "unknown task"}
        if name == "cancel_background_task":
            task_id = str(arguments.get("task_id") or "")
            task = self.background_tasks.get(task_id)
            if not task:
                return {"ok": False, "error": "unknown task"}
            task = {**task, "status": "cancelled"}
            self.background_tasks[task_id] = task
            return {"ok": True, "task": task}
        if name == TASK_STATUS_TOOL_NAME:
            return {
                "ok": True,
                "status_update": {
                    "text": str(arguments.get("text") or ""),
                    "speak_now": bool(arguments.get("speak_now")),
                },
            }
        if name == MEMORY_ADD_TOOL_NAME:
            return self.memory.add_entry(
                text=str(arguments.get("text") or ""),
                kind=str(arguments.get("kind") or "note"),
                source="user",
            )
        if name == MEMORY_LIST_TOOL_NAME:
            return self.memory.list_entries(include_forgotten=bool(arguments.get("include_forgotten")))
        if name == MEMORY_SEARCH_TOOL_NAME:
            return self.memory.search(
                str(arguments.get("query") or ""),
                max_results=int(arguments.get("max_results") or 5),
            )
        if name == MEMORY_UPDATE_TOOL_NAME:
            return self.memory.update_entry(
                entry_id=str(arguments.get("entry_id") or ""),
                text=str(arguments.get("text") or ""),
            )
        if name == MEMORY_FORGET_TOOL_NAME:
            return self.memory.forget_entry(
                entry_id=str(arguments.get("entry_id") or ""),
                query=str(arguments.get("query") or ""),
                reason=str(arguments.get("reason") or ""),
            )
        if name == MEMORY_GET_SOURCE_TOOL_NAME:
            return self.memory.get_source(
                path=str(arguments.get("path") or ""),
                from_line=int(arguments.get("from_line") or 1),
                lines=int(arguments.get("lines") or 12),
            )
        if name == LIST_MUSIC_TOOL_NAME:
            return {
                "ok": True,
                "songs": [
                    {"title": "Baby Shark", "duration": "1:51"},
                    {"title": "Chopin Spring", "duration": "2:34"},
                ],
            }
        if name == GET_MUSIC_STATUS_TOOL_NAME:
            return {
                "ok": True,
                "state": self.music_state,
                "title": self.music_title,
                "volume": self.music_volume,
            }
        if name == PLAY_MUSIC_TOOL_NAME:
            query = str(arguments.get("query") or "").strip()
            self.music_state = "playing"
            self.music_title = "Baby Shark" if "baby" in query.lower() else "Chopin Spring"
            return {"ok": True, "deferred": True, "title": self.music_title}
        if name == PAUSE_MUSIC_TOOL_NAME:
            self.music_state = "paused"
            return {"ok": True, "state": self.music_state}
        if name == RESUME_MUSIC_TOOL_NAME:
            self.music_state = "playing"
            return {"ok": True, "deferred": True, "state": self.music_state}
        if name == STOP_MUSIC_TOOL_NAME:
            self.music_state = "stopped"
            return {"ok": True, "state": self.music_state}
        if name == SET_MUSIC_VOLUME_TOOL_NAME:
            self.music_volume = max(1, min(10, int(arguments.get("level") or self.music_volume)))
            return {"ok": True, "volume": self.music_volume}
        return {"ok": False, "error": f"unhandled test tool: {name}"}

    def _weather(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location_query = str(arguments.get("location") or "").strip() or "Cambridge,GB"
        forecast_days = max(0, min(5, int(arguments.get("forecast_days") or 0)))
        base_name = location_query.split(",")[0].strip() or "Cambridge"
        now = datetime.now(timezone.utc)
        daily = []
        for offset in range(forecast_days):
            day = now + timedelta(days=offset + 1)
            daily.append(
                {
                    "date": day.date().isoformat(),
                    "weekday": day.strftime("%A"),
                    "condition": {"main": "Clouds", "description": "broken clouds"},
                    "temperature": {"min": 14 + offset, "max": 20 + offset},
                    "precipitation_probability_percent": 10 + offset * 5,
                }
            )
        return {
            "ok": True,
            "source": "FakeWeather",
            "cached": not bool(arguments.get("no_cache")),
            "cache_age_seconds": 0,
            "location_query": location_query,
            "location": {
                "name": base_name,
                "state": "",
                "country": "GB" if "san francisco" not in base_name.lower() else "US",
                "lat": 52.2053,
                "lon": 0.1218,
            },
            "units": str(arguments.get("units") or "metric"),
            "observed_at": now.isoformat(),
            "condition": {"main": "Clouds", "description": "broken clouds"},
            "temperature": {
                "current": 19 if "san francisco" not in base_name.lower() else 14.5,
                "feels_like": 18 if "san francisco" not in base_name.lower() else 13.0,
                "min": 16,
                "max": 20,
            },
            "humidity_percent": 55,
            "pressure_hpa": 1013,
            "wind": {"speed": 3.4, "degrees": 150, "gust": 5.0},
            "clouds_percent": 60,
            "forecast_days_requested": forecast_days,
            "forecast": {
                "days_requested": forecast_days,
                "days_returned": len(daily),
                "daily": daily,
                "periods": [],
            },
        }


class LiveRealtimeTextSession:
    def __init__(self, config: Config, executor: EvalToolExecutor):
        self.config = config
        self.executor = executor
        self.memory = executor.memory
        self.ws: websocket.WebSocket | None = None

    def __enter__(self) -> "LiveRealtimeTextSession":
        url = f"wss://api.openai.com/v1/realtime?model={self.config.realtime_model}"
        headers = [
            f"Authorization: Bearer {self.config.openai_api_key}",
            f"OpenAI-Safety-Identifier: {self.config.realtime_safety_identifier}",
        ]
        self.ws = websocket.create_connection(
            url,
            header=headers,
            sslopt={"ca_certs": certifi.where()},
            timeout=self.config.realtime_response_timeout_seconds,
        )
        self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.config.realtime_model,
                    "output_modalities": ["text"],
                    "instructions": realtime_session_instructions(self.config, self.memory.bootstrap_context()),
                    "reasoning": {"effort": self.config.realtime_reasoning_effort},
                    "tools": realtime_tools(),
                    "tool_choice": "auto",
                },
            }
        )
        self._wait_for_session_updated()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def run_turn(self, prompt: str) -> RealtimeEvalResult:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            }
        )
        texts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        instructions = realtime_turn_instructions(self.config, self.memory.active_context(prompt))
        while True:
            self._send({"type": "response.create", "response": {"output_modalities": ["text"], "instructions": instructions}})
            response, response_text = self._read_response_done()
            if response_text:
                texts.append(response_text)
            calls = response_function_calls(response)
            if not calls:
                return RealtimeEvalResult(
                    prompt=prompt,
                    texts=texts,
                    function_calls=function_calls,
                    final_text=response_text,
                )
            for call in calls:
                arguments = parse_tool_arguments(call)
                function_calls.append({"name": call.get("name", ""), "arguments": arguments})
                output = self.executor.execute(str(call.get("name") or ""), arguments)
                self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call.get("call_id", ""),
                            "output": json.dumps(output, ensure_ascii=False),
                        },
                    }
                )
            instructions = (
                f"{realtime_turn_instructions(self.config, self.memory.active_context(prompt))}\n"
                "If a background task is still running, tell the user briefly and do not wait."
            )

    def _wait_for_session_updated(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            event = self._recv()
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(event.get("error", event))
        raise RuntimeError("timed out waiting for realtime session.updated")

    def _read_response_done(self) -> tuple[dict[str, Any], str]:
        text_parts: list[str] = []
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            event = self._recv()
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(event.get("error", event))
            if event_type == "response.output_text.delta":
                text_parts.append(str(event.get("delta") or ""))
                continue
            if event_type in {
                "session.created",
                "session.updated",
                "rate_limits.updated",
                "response.created",
                "response.output_item.added",
                "response.output_item.created",
                "response.output_item.done",
                "conversation.item.created",
            }:
                continue
            if event_type == "response.done":
                response = event.get("response", {})
                text = "".join(text_parts).strip() or extract_response_text(response)
                return response, text.strip()
        raise RuntimeError("timed out waiting for realtime response.done")

    def _send(self, event: dict[str, Any]) -> None:
        assert self.ws is not None
        self.ws.send(json.dumps(event))

    def _recv(self) -> dict[str, Any]:
        assert self.ws is not None
        raw = self.ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise RuntimeError(f"unexpected realtime event: {event!r}")
        return event


@pytest.fixture
def integration_config(tmp_path: Path) -> Config:
    return Config(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        voice_bot_backend="realtime",
        openai_model="gpt-5.5",
        openai_reasoning_effort="high",
        openai_connectivity_host="api.openai.com",
        openai_connectivity_wait_seconds=5.0,
        openai_timeout_seconds=30.0,
        transcription_model="gpt-4o-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="cedar",
        tts_instructions="test",
        realtime_model=os.getenv("REALTIME_MODEL", "gpt-realtime-2"),
        realtime_reasoning_effort="medium",
        realtime_voice="marin",
        realtime_input_rate=24000,
        realtime_input_transcription_model="gpt-4o-transcribe",
        realtime_response_timeout_seconds=30.0,
        realtime_idle_timeout_seconds=45.0,
        realtime_max_session_seconds=120.0,
        realtime_silent_cooldown_seconds=5.0,
        realtime_turn_detection="server_vad",
        realtime_vad_threshold=0.5,
        realtime_vad_prefix_padding_ms=300,
        realtime_vad_silence_duration_ms=850,
        realtime_semantic_vad_eagerness="medium",
        realtime_history_messages=8,
        realtime_safety_identifier="voice-ai-bot-realtime-integration-test",
        user_city="Cambridge",
        user_region="Cambridgeshire",
        user_country="GB",
        user_timezone="Europe/London",
        web_search_model="gpt-5.5",
        web_search_reasoning_effort="high",
        web_search_context_size="medium",
        web_search_timeout_seconds=30.0,
        task_model="gpt-5.5",
        task_reasoning_effort="high",
        task_reasoning_summary="auto",
        task_timeout_seconds=60.0,
        task_code_execution=True,
        max_background_tasks=20,
        task_result_chars=12000,
        task_summary_chars=4000,
        button_gpio=23,
        led_gpio=25,
        button_pull_up=True,
        short_click_seconds=0.45,
        double_click_window_seconds=0.65,
        audio_capture_device="plughw:1,0",
        audio_playback_device="plughw:1,0",
        record_rate=24000,
        record_channels=1,
        min_record_seconds=0.25,
        conversation_file=tmp_path / "conversation.json",
        scheduled_tasks_file=tmp_path / "scheduled_tasks.json",
        schedule_quiet_start="21:00",
        schedule_quiet_end="07:30",
        recordings_dir=tmp_path / "recordings",
        tts_chunk_chars=240,
        log_level="INFO",
        memory_dir=tmp_path / "agent",
    )


@pytest.fixture
def live_session(integration_config: Config):
    executor = EvalToolExecutor(integration_config)
    with LiveRealtimeTextSession(integration_config, executor) as session:
        yield session, executor


def _words(text: str) -> int:
    return len(re.findall(r"[\\w']+", text, flags=re.UNICODE))


def _call_names(result: RealtimeEvalResult) -> list[str]:
    return [call["name"] for call in result.function_calls]


def _assert_no_quick_tool_preamble(text: str) -> None:
    lowered = text.lower()
    banned = [
        "let me check",
        "one moment",
        "just a moment",
        "i'll do that",
        "i will do that",
        "i'm adjusting",
        "i am adjusting",
    ]
    assert not any(phrase in lowered for phrase in banned), text


@pytest.mark.realtime_integration
@pytest.mark.parametrize("prompt", ["Who are you?", "Tell me about yourself."])
def test_realtime_identity_uses_sipquest_name(live_session, prompt: str):
    session, _executor = live_session

    result = session.run_turn(prompt)

    lowered = result.combined_text.lower()
    assert "sipquest" in lowered, result
    assert "chatgpt" not in lowered, result
    assert _words(result.combined_text) <= 20, result


@pytest.mark.realtime_integration
def test_realtime_timer_confirmation_is_short_and_direct(live_session, integration_config: Config):
    session, _executor = live_session

    result = session.run_turn("Please set a timer for eight minutes.")

    assert ADD_SCHEDULED_TASK_TOOL_NAME in _call_names(result), result
    timer_call = next(call for call in result.function_calls if call["name"] == ADD_SCHEDULED_TASK_TOOL_NAME)
    run_at = parse_local_datetime(str(timer_call["arguments"]["run_at"]), integration_config.user_timezone)
    now = datetime.now(run_at.tzinfo)
    delta_seconds = (run_at - now).total_seconds()
    assert 6 * 60 <= delta_seconds <= 10 * 60, result
    assert timer_call["arguments"].get("repeat") == "once", result
    assert "timer" in result.combined_text.lower(), result
    assert "set" in result.combined_text.lower(), result
    assert _words(result.combined_text) <= 10, result
    _assert_no_quick_tool_preamble(result.combined_text)


@pytest.mark.realtime_integration
def test_realtime_volume_confirmation_is_short_and_accurate(live_session):
    session, executor = live_session

    result = session.run_turn("Please set the voice volume to four.")

    assert SET_VOICE_VOLUME_TOOL_NAME in _call_names(result), result
    volume_call = next(call for call in result.function_calls if call["name"] == SET_VOICE_VOLUME_TOOL_NAME)
    assert volume_call["arguments"].get("level") == 4, result
    assert executor.voice_volume == 4, result
    assert "4" in result.combined_text or "four" in result.combined_text.lower(), result
    assert _words(result.combined_text) <= 10, result
    _assert_no_quick_tool_preamble(result.combined_text)


@pytest.mark.realtime_integration
def test_realtime_weather_forecast_uses_weather_tool_with_forecast_days(live_session):
    session, _executor = live_session

    result = session.run_turn("What's the weather in San Francisco tomorrow?")

    assert GET_WEATHER_TOOL_NAME in _call_names(result), result
    weather_call = next(call for call in result.function_calls if call["name"] == GET_WEATHER_TOOL_NAME)
    assert "san francisco" in str(weather_call["arguments"].get("location", "")).lower(), result
    assert int(weather_call["arguments"].get("forecast_days") or 0) >= 1, result
    assert "san francisco" in result.combined_text.lower(), result
    _assert_no_quick_tool_preamble(result.combined_text)


@pytest.mark.realtime_integration
def test_realtime_background_task_start_is_brief(live_session):
    session, _executor = live_session

    result = session.run_turn(
        "Please check the latest LeetCode daily challenge difficulty and tell me when you know."
    )

    assert START_TASK_TOOL_NAME in _call_names(result), result
    task_call = next(call for call in result.function_calls if call["name"] == START_TASK_TOOL_NAME)
    assert "query" not in task_call["arguments"], result
    assert "started" in result.combined_text.lower() or "running" in result.combined_text.lower(), result
    assert _words(result.combined_text) <= 18, result


@pytest.mark.realtime_integration
def test_realtime_memory_write_ack_is_brief(live_session):
    session, _executor = live_session

    result = session.run_turn("Remember that I prefer terse answers.")

    assert MEMORY_ADD_TOOL_NAME in _call_names(result), result
    assert "remember" in result.combined_text.lower() or "noted" in result.combined_text.lower(), result
    assert _words(result.combined_text) <= 12, result


@pytest.mark.realtime_integration
def test_realtime_music_listing_mentions_titles_and_durations(live_session):
    session, _executor = live_session

    result = session.run_turn("What songs are available?")

    assert LIST_MUSIC_TOOL_NAME in _call_names(result), result
    lowered = result.combined_text.lower()
    assert "baby shark" in lowered, result
    assert "chopin spring" in lowered, result
    assert "1:51" in result.combined_text or "1 51" in lowered, result
