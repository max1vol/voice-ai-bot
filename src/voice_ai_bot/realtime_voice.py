from __future__ import annotations

import base64
import json
import logging
import socket
import time
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi
import websocket

from .audio_io import PcmPlayer
from .config import Config, REALTIME_SYSTEM_PROMPT
from .conversation import Message

LOGGER = logging.getLogger(__name__)


CLOSE_TOOL_NAME = "close_realtime_session"


@dataclass(frozen=True)
class RealtimeTurnResult:
    user_text: str
    assistant_text: str
    requested_close: bool


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
            "instructions": REALTIME_SYSTEM_PROMPT,
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
            "tools": [
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
                }
            ],
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
        self._send(ws, {"type": "response.create", "response": {"output_modalities": ["audio"]}})

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
    for item in response.get("output", []):
        if item.get("type") == "function_call" and item.get("name") == CLOSE_TOOL_NAME:
            return True
    return False


def extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()
