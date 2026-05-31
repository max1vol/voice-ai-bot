from __future__ import annotations

import base64
import json
import logging
import queue
import socket
import threading
import time
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import certifi
import websocket
from openai import OpenAI

from .audio_io import PcmOutputStream, PcmPlayer, RawPcmRecorder
from .config import Config, REALTIME_SYSTEM_PROMPT
from .conversation import Message

LOGGER = logging.getLogger(__name__)


CLOSE_TOOL_NAME = "close_realtime_session"
WEB_SEARCH_TOOL_NAME = "web_search"


@dataclass(frozen=True)
class RealtimeTurnResult:
    user_text: str
    assistant_text: str
    requested_close: bool


def local_context(config: Config) -> str:
    try:
        now = datetime.now(ZoneInfo(config.user_timezone))
    except Exception:
        now = datetime.now().astimezone()
    utc_offset = now.strftime("%z")
    formatted_offset = f"{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "unknown UTC offset"
    return (
        f"Current local time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()} (UTC{formatted_offset}). "
        f"User location: {config.user_city}, {config.user_region}, {config.user_country}. "
        f"User timezone: {config.user_timezone}."
    )


def realtime_session_instructions(config: Config) -> str:
    return (
        f"{REALTIME_SYSTEM_PROMPT} "
        f"The user is in {config.user_city}, {config.user_region}, {config.user_country}. "
        "For each response, the application will provide current local time and location context. "
        "Use the web_search tool when the user asks about current, recent, local, weather, news, opening-hours, "
        "price, schedule, or otherwise time-sensitive facts. "
        "If interrupted by a new button press, stop the previous reply and treat the new speech as the latest user turn."
    )


def realtime_turn_instructions(config: Config) -> str:
    return (
        f"{local_context(config)} "
        "Answer the latest user turn. Keep spoken replies concise. "
        "If using web_search, ask for exactly the information needed."
    )


def realtime_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": CLOSE_TOOL_NAME,
            "description": (
                "Close the current realtime session when the user asks to stop, disconnect, sleep, "
                "end the conversation for now, or otherwise stop spending on this live connection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason the session should be closed.",
                    }
                },
                "required": ["reason"],
            },
        },
        {
            "type": "function",
            "name": WEB_SEARCH_TOOL_NAME,
            "description": (
                "Search the live web through a GPT-5.5 Responses request. Use for current, recent, local, "
                "weather, news, opening-hours, price, schedule, or other facts that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search question, including location or date when relevant.",
                    }
                },
                "required": ["query"],
            },
        },
    ]


class RealtimeVoiceClient:
    def __init__(self, config: Config):
        self.config = config
        self.player = PcmPlayer(config.audio_playback_device)

    def wait_for_connectivity(self) -> None:
        host = self.config.openai_connectivity_host
        deadline = time.monotonic() + self.config.openai_connectivity_wait_seconds
        last_error: OSError | None = None
        while True:
            try:
                with socket.create_connection((host, 443), timeout=5):
                    return
            except OSError as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"network is not ready for {host}:443") from last_error
                LOGGER.warning("waiting for network/DNS before OpenAI realtime call: %s", exc)
                time.sleep(min(5.0, remaining))

    def respond_to_audio(self, history: Iterable[Message], audio_path: Path) -> RealtimeTurnResult:
        deadline = time.monotonic() + self.config.realtime_max_session_seconds
        ws = self._connect()
        try:
            self._send_session_update(ws)
            self._wait_for_session_updated(ws, deadline)
            self._seed_history(ws, history)
            self._send_audio_turn(ws, audio_path)
            return self._read_response(ws, deadline)
        finally:
            ws.close()

    def _connect(self) -> websocket.WebSocket:
        url = f"wss://api.openai.com/v1/realtime?model={self.config.realtime_model}"
        headers = [
            f"Authorization: Bearer {self.config.openai_api_key}",
            f"OpenAI-Safety-Identifier: {self.config.realtime_safety_identifier}",
        ]
        LOGGER.info("opening realtime websocket with %s", self.config.realtime_model)
        return websocket.create_connection(
            url,
            header=headers,
            sslopt={"ca_certs": certifi.where()},
            timeout=self.config.realtime_response_timeout_seconds,
        )

    def _send_session_update(self, ws: websocket.WebSocket) -> None:
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.config.realtime_model,
            "output_modalities": ["audio"],
            "instructions": realtime_session_instructions(self.config),
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.realtime_input_rate,
                    },
                    "turn_detection": None,
                    "transcription": {
                        "model": self.config.realtime_input_transcription_model,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "voice": self.config.realtime_voice,
                },
            },
            "reasoning": {
                "effort": self.config.realtime_reasoning_effort,
            },
            "tools": realtime_tools(),
            "tool_choice": "auto",
        }
        self._send(ws, {"type": "session.update", "session": session})

    def _wait_for_session_updated(self, ws: websocket.WebSocket, deadline: float) -> None:
        while True:
            event = self._recv(ws, deadline)
            event_type = event.get("type")
            if event_type == "session.updated":
                LOGGER.info("realtime session updated")
                return
            if event_type == "session.created":
                continue
            self._raise_for_error(event)

    def _seed_history(self, ws: websocket.WebSocket, history: Iterable[Message]) -> None:
        messages = list(history)[-self.config.realtime_history_messages :]
        if not messages:
            return
        LOGGER.info("seeding realtime session with %d saved messages", len(messages))
        for message in messages:
            self._send(ws, {"type": "conversation.item.create", "item": conversation_item_for_message(message)})

    def _send_audio_turn(self, ws: websocket.WebSocket, audio_path: Path) -> None:
        self._send(ws, {"type": "input_audio_buffer.clear"})
        total_bytes = 0
        for chunk in iter_wav_pcm16_chunks(audio_path, self.config.realtime_input_rate):
            total_bytes += len(chunk)
            self._send(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )
        if total_bytes == 0:
            raise RuntimeError(f"recording has no PCM audio: {audio_path}")
        LOGGER.info("sent %d bytes of PCM audio to realtime session", total_bytes)
        self._send(ws, {"type": "input_audio_buffer.commit"})
        self._send(
            ws,
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": realtime_turn_instructions(self.config),
                },
            },
        )

    def _read_response(self, ws: websocket.WebSocket, deadline: float) -> RealtimeTurnResult:
        input_transcript_parts: list[str] = []
        input_transcript_final = ""
        output_transcript_parts: list[str] = []
        output_transcript_final = ""
        response_done_text = ""
        requested_close = False

        with self.player.open_stream(rate=24000, channels=1) as audio:
            while True:
                event = self._recv(ws, deadline)
                event_type = event.get("type")
                self._raise_for_error(event)

                if event_type == "conversation.item.input_audio_transcription.delta":
                    input_transcript_parts.append(event.get("delta", ""))
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    input_transcript_final = event.get("transcript", "").strip()
                    if input_transcript_final:
                        LOGGER.info("realtime input transcript: %s", input_transcript_final)
                elif event_type == "response.output_audio.delta":
                    audio.write(base64.b64decode(event.get("delta", "")))
                elif event_type == "response.output_audio_transcript.delta":
                    output_transcript_parts.append(event.get("delta", ""))
                elif event_type == "response.output_audio_transcript.done":
                    output_transcript_final = event.get("transcript", "").strip()
                    if output_transcript_final:
                        LOGGER.info("realtime assistant transcript: %s", output_transcript_final)
                elif event_type == "response.done":
                    response = event.get("response", {})
                    requested_close = response_requested_close(response)
                    response_done_text = extract_response_text(response)
                    break

        user_text = input_transcript_final or "".join(input_transcript_parts).strip()
        assistant_text = output_transcript_final or "".join(output_transcript_parts).strip() or response_done_text
        if requested_close:
            LOGGER.info("model requested realtime session close")
        if not user_text:
            LOGGER.warning("realtime session completed without an input transcript")
        if not assistant_text and not requested_close:
            raise RuntimeError("realtime model returned no assistant transcript")
        return RealtimeTurnResult(user_text=user_text, assistant_text=assistant_text, requested_close=requested_close)

    def _send(self, ws: websocket.WebSocket, event: dict[str, Any]) -> None:
        ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("realtime session exceeded local maximum duration")
            ws.settimeout(min(5.0, remaining))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise RuntimeError(f"unexpected realtime event: {event!r}")
            return event

    def _raise_for_error(self, event: dict[str, Any]) -> None:
        if event.get("type") != "error":
            return
        error = event.get("error", event)
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"realtime API error: {message}")


class RealtimeConversationSession:
    def __init__(self, config: Config):
        self.config = config
        self.player = PcmPlayer(config.audio_playback_device)
        self.search_client = OpenAI(api_key=config.openai_api_key, timeout=config.web_search_timeout_seconds)
        self._ws: websocket.WebSocket | None = None
        self._receiver_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._errors: queue.Queue[BaseException] = queue.Queue()
        self._completed_turns: queue.Queue[RealtimeTurnResult] = queue.Queue()

        self._session_started_at = 0.0
        self._last_activity_at = 0.0
        self._closing = False
        self._seeded = False

        self._recorder: RawPcmRecorder | None = None
        self._capture_queue: queue.Queue[bytes | None] | None = None
        self._capture_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None
        self._recording_started_at = 0.0
        self._capture_bytes = 0
        self._sent_audio_bytes = 0

        self._response_pending = False
        self._response_active = False
        self._tool_active = False
        self._tool_generation = 0
        self._active_response_id = ""
        self._playback: PcmOutputStream | None = None
        self._output_item_id = ""
        self._output_audio_bytes = 0
        self._input_transcript_parts: list[str] = []
        self._input_transcript_final = ""
        self._output_transcript_parts: list[str] = []
        self._output_transcript_final = ""

    def begin_turn(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._recorder is not None:
                raise RuntimeError("realtime recording is already active")
            self._last_activity_at = time.monotonic()
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._recorder = RawPcmRecorder(self.config)
            self._capture_queue = queue.Queue()
            self._capture_bytes = 0
            self._sent_audio_bytes = 0
            self._recording_started_at = time.monotonic()
            recorder = self._recorder
            capture_queue = self._capture_queue

        recorder.start()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(recorder, capture_queue),
            name="realtime-capture",
            daemon=True,
        )
        self._capture_thread.start()

        try:
            self.ensure_open(history)
            self._send({"type": "input_audio_buffer.clear"})
            self._sender_thread = threading.Thread(
                target=self._send_audio_loop,
                args=(capture_queue,),
                name="realtime-audio-sender",
                daemon=True,
            )
            self._sender_thread.start()
        except BaseException:
            self._stop_capture_threads()
            raise

    def stop_recording(self) -> float:
        return self._stop_capture_threads()

    def commit_recording(self) -> None:
        self._raise_background_error()
        with self._state_lock:
            sent_audio_bytes = self._sent_audio_bytes
        if sent_audio_bytes == 0:
            self.clear_pending_input()
            raise RuntimeError("no audio was captured for realtime turn")
        self._send({"type": "input_audio_buffer.commit"})
        self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": realtime_turn_instructions(self.config),
                },
            }
        )
        with self._state_lock:
            self._response_pending = True
            self._last_activity_at = time.monotonic()
        LOGGER.info("committed realtime turn: %d PCM bytes", sent_audio_bytes)

    def end_turn(self, commit: bool) -> float:
        duration = self._stop_capture_threads()
        if not commit:
            self.clear_pending_input()
            return duration
        self.commit_recording()
        LOGGER.info("realtime recording duration: %.3fs", duration)
        return duration

    def clear_pending_input(self) -> None:
        if self.is_open:
            self._send({"type": "input_audio_buffer.clear"})

    def check_health(self) -> None:
        self._raise_background_error()

    def ensure_open(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._ws is not None and self._receiver_thread is not None and self._receiver_thread.is_alive():
                return
        ws = RealtimeVoiceClient(self.config)._connect()
        deadline = time.monotonic() + self.config.realtime_response_timeout_seconds
        try:
            RealtimeVoiceClient(self.config)._send_session_update(ws)
            RealtimeVoiceClient(self.config)._wait_for_session_updated(ws, deadline)
            RealtimeVoiceClient(self.config)._seed_history(ws, history)
        except BaseException:
            ws.close()
            raise

        with self._state_lock:
            self._ws = ws
            self._closing = False
            self._seeded = True
            self._session_started_at = time.monotonic()
            self._last_activity_at = self._session_started_at
        self._receiver_thread = threading.Thread(target=self._receive_loop, name="realtime-receiver", daemon=True)
        self._receiver_thread.start()
        LOGGER.info("persistent realtime session open")

    def close_if_idle(self) -> None:
        with self._state_lock:
            if self._ws is None or self._is_busy_locked():
                return
            idle_for = time.monotonic() - self._last_activity_at
            if idle_for < self.config.realtime_idle_timeout_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs idle", idle_for)
        self.close()

    def close_if_too_old(self) -> None:
        with self._state_lock:
            if self._ws is None:
                return
            age = time.monotonic() - self._session_started_at
            if age < self.config.realtime_max_session_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs hard max age", age)
        self.close()

    def close(self) -> None:
        receiver_thread: threading.Thread | None
        with self._state_lock:
            self._closing = True
            ws = self._ws
            self._ws = None
            receiver_thread = self._receiver_thread
            self._receiver_thread = None
            self._response_pending = False
            self._response_active = False
            self._tool_active = False
            self._tool_generation += 1
            self._abort_playback_locked()
        self._stop_capture_threads()
        if ws is not None:
            ws.close()
        if receiver_thread is not None and receiver_thread is not threading.current_thread():
            receiver_thread.join(timeout=2)
        with self._state_lock:
            self._closing = False
            self._seeded = False
        LOGGER.info("persistent realtime session closed")

    @property
    def is_open(self) -> bool:
        with self._state_lock:
            return self._ws is not None

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recorder is not None

    @property
    def is_responding(self) -> bool:
        with self._state_lock:
            return self._response_pending or self._response_active or self._tool_active or self._playback is not None

    def pop_completed_turns(self) -> list[RealtimeTurnResult]:
        turns: list[RealtimeTurnResult] = []
        while True:
            try:
                turns.append(self._completed_turns.get_nowait())
            except queue.Empty:
                return turns

    def _capture_loop(self, recorder: RawPcmRecorder, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = recorder.read(4096)
                if not chunk:
                    return
                with self._state_lock:
                    self._capture_bytes += len(chunk)
                capture_queue.put(chunk)
        except BaseException as exc:
            self._errors.put(exc)
        finally:
            capture_queue.put(None)

    def _send_audio_loop(self, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = capture_queue.get()
                if chunk is None:
                    return
                self._send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                with self._state_lock:
                    self._sent_audio_bytes += len(chunk)
                    self._last_activity_at = time.monotonic()
        except BaseException as exc:
            self._errors.put(exc)

    def _stop_capture_threads(self) -> float:
        with self._state_lock:
            recorder = self._recorder
            self._recorder = None
            capture_thread = self._capture_thread
            sender_thread = self._sender_thread
            self._capture_thread = None
            self._sender_thread = None
        duration = recorder.stop() if recorder is not None else 0.0
        if capture_thread is not None:
            capture_thread.join(timeout=2)
        if sender_thread is not None:
            sender_thread.join(timeout=5)
        self._raise_background_error()
        return duration

    def _receive_loop(self) -> None:
        try:
            while True:
                with self._state_lock:
                    ws = self._ws
                    closing = self._closing
                    deadline = self._session_started_at + self.config.realtime_max_session_seconds
                if ws is None or closing:
                    return
                try:
                    event = self._recv(ws, deadline)
                except websocket.WebSocketConnectionClosedException:
                    with self._state_lock:
                        if self._closing or self._ws is None:
                            return
                    raise
                should_close = self._handle_event(event)
                if should_close:
                    self.close()
                    return
        except BaseException as exc:
            with self._state_lock:
                closing = self._closing
            if not closing:
                self._errors.put(exc)

    def _handle_event(self, event: dict[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == "error":
            if event_is_ignorable_control_error(event):
                LOGGER.warning("ignoring realtime control error: %s", event)
                return False
            RealtimeVoiceClient(self.config)._raise_for_error(event)

        with self._state_lock:
            self._last_activity_at = time.monotonic()

            if event_type == "conversation.item.input_audio_transcription.delta":
                self._input_transcript_parts.append(event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._input_transcript_final = event.get("transcript", "").strip()
                if self._input_transcript_final:
                    LOGGER.info("realtime input transcript: %s", self._input_transcript_final)
            elif event_type == "response.created":
                response = event.get("response", {})
                self._response_pending = False
                self._response_active = True
                self._active_response_id = response.get("id", "")
            elif event_type in {"response.output_item.created", "response.output_item.added"}:
                item = event.get("item", {})
                if item.get("type") == "message":
                    self._output_item_id = item.get("id", "")
                    self._output_audio_bytes = 0
            elif event_type == "response.output_audio.delta":
                self._write_output_audio_locked(base64.b64decode(event.get("delta", "")))
            elif event_type == "response.output_audio_transcript.delta":
                self._output_transcript_parts.append(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.done":
                self._output_transcript_final = event.get("transcript", "").strip()
                if self._output_transcript_final:
                    LOGGER.info("realtime assistant transcript: %s", self._output_transcript_final)
            elif event_type == "response.done":
                return self._complete_response_locked(event.get("response", {}))
        return False

    def _complete_response_locked(self, response: dict[str, Any]) -> bool:
        status = response.get("status", "")
        self._response_pending = False
        self._response_active = False
        self._active_response_id = ""
        self._close_playback_locked(check=True)

        if status == "cancelled":
            LOGGER.info("realtime response cancelled")
            self._reset_turn_text_locked()
            return False

        web_search_calls = response_function_calls(response, WEB_SEARCH_TOOL_NAME)
        if web_search_calls:
            self._tool_active = True
            self._tool_generation += 1
            generation = self._tool_generation
            threading.Thread(
                target=self._run_web_search_calls,
                args=(generation, web_search_calls),
                name="realtime-web-search",
                daemon=True,
            ).start()
            return False

        requested_close = response_requested_close(response)
        user_text = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
        assistant_text = (
            self._output_transcript_final
            or "".join(self._output_transcript_parts).strip()
            or extract_response_text(response)
        )
        if user_text or assistant_text or requested_close:
            self._completed_turns.put(
                RealtimeTurnResult(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    requested_close=requested_close,
                )
            )
        self._reset_turn_text_locked()
        if requested_close:
            LOGGER.info("model requested realtime session close")
        return requested_close

    def _run_web_search_calls(self, generation: int, calls: list[dict[str, Any]]) -> None:
        try:
            for call in calls:
                output = self._execute_web_search_call(call)
                with self._state_lock:
                    if generation != self._tool_generation or self._ws is None or self._closing:
                        LOGGER.info("discarding stale web_search result")
                        return
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
            with self._state_lock:
                if generation != self._tool_generation or self._ws is None or self._closing:
                    return
                self._tool_active = False
                self._response_pending = True
            self._send(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "instructions": realtime_turn_instructions(self.config),
                    },
                }
            )
        except BaseException as exc:
            self._errors.put(exc)

    def _execute_web_search_call(self, call: dict[str, Any]) -> dict[str, Any]:
        query = parse_tool_arguments(call).get("query", "").strip()
        if not query:
            return {"ok": False, "error": "web_search requires a non-empty query"}
        LOGGER.info("running web_search tool: %s", query)
        prompt = (
            "This request came from a realtime voice chat. "
            f"{local_context(self.config)} "
            "Answer the search query concisely for a spoken response. Include source names or URLs when useful. "
            f"Search query: {query}"
        )
        response = self.search_client.responses.create(
            model=self.config.web_search_model,
            input=prompt,
            reasoning={"effort": self.config.web_search_reasoning_effort},
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": self.config.web_search_context_size,
                    "user_location": {
                        "type": "approximate",
                        "country": self.config.user_country,
                        "city": self.config.user_city,
                        "region": self.config.user_region,
                        "timezone": self.config.user_timezone,
                    },
                }
            ],
            store=False,
        )
        text = getattr(response, "output_text", "") or extract_responses_output_text(response)
        return {
            "ok": True,
            "query": query,
            "result": text.strip(),
            "context": local_context(self.config),
        }

    def _write_output_audio_locked(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._playback is None:
            self._playback = self.player.open_stream(rate=24000, channels=1)
            self._playback.__enter__()
        self._playback.write(chunk)
        self._output_audio_bytes += len(chunk)

    def _interrupt_response_locked(self) -> None:
        if self._response_pending or self._response_active:
            self._send(
                {
                    "event_id": f"cancel_{time.monotonic_ns()}",
                    "type": "response.cancel",
                }
            )
        self._abort_playback_locked()
        if self._output_item_id and self._output_audio_bytes:
            audio_end_ms = int(self._output_audio_bytes / 2 / 24000 * 1000)
            self._send(
                {
                    "event_id": f"truncate_{time.monotonic_ns()}",
                    "type": "conversation.item.truncate",
                    "item_id": self._output_item_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, audio_end_ms),
                }
            )
        self._response_pending = False
        self._response_active = False
        self._active_response_id = ""
        self._tool_active = False
        self._tool_generation += 1

    def _abort_playback_locked(self) -> None:
        if self._playback is None:
            return
        self._playback.abort()
        self._playback.close(check=False)
        self._playback = None

    def _close_playback_locked(self, check: bool) -> None:
        if self._playback is None:
            return
        self._playback.close(check=check)
        self._playback = None

    def _reset_turn_text_locked(self) -> None:
        self._input_transcript_parts = []
        self._input_transcript_final = ""
        self._output_transcript_parts = []
        self._output_transcript_final = ""
        self._output_item_id = ""
        self._output_audio_bytes = 0

    def _is_busy_locked(self) -> bool:
        return (
            self._recorder is not None
            or self._response_pending
            or self._response_active
            or self._tool_active
            or self._playback is not None
        )

    def _send(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            ws = self._ws
        if ws is None:
            raise RuntimeError("realtime session is not open")
        with self._send_lock:
            ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        return RealtimeVoiceClient(self.config)._recv(ws, deadline)

    def _raise_background_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("realtime background worker failed") from error


def iter_wav_pcm16_chunks(path: Path, expected_rate: int, chunk_frames: int = 4096) -> Iterator[bytes]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        if channels != 1:
            raise RuntimeError(f"realtime audio must be mono; got {channels} channels in {path}")
        if sample_width != 2:
            raise RuntimeError(f"realtime audio must be 16-bit PCM; got sample width {sample_width} in {path}")
        if rate != expected_rate:
            raise RuntimeError(f"realtime audio must be {expected_rate} Hz; got {rate} Hz in {path}")
        while True:
            data = wav.readframes(chunk_frames)
            if not data:
                return
            yield data


def conversation_item_for_message(message: Message) -> dict[str, Any]:
    content_type = "input_text" if message.role == "user" else "output_text"
    return {
        "type": "message",
        "role": message.role,
        "content": [
            {
                "type": content_type,
                "text": message.content,
            }
        ],
    }


def response_requested_close(response: dict[str, Any]) -> bool:
    return bool(response_function_calls(response, CLOSE_TOOL_NAME))


def response_function_calls(response: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output", [])
        if item.get("type") == "function_call" and item.get("name") == name
    ]


def parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def extract_responses_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def event_is_ignorable_control_error(event: dict[str, Any]) -> bool:
    error = event.get("error", {})
    if not isinstance(error, dict):
        return False
    event_id = str(error.get("event_id") or event.get("event_id") or "")
    return event_id.startswith(("cancel_", "truncate_"))
