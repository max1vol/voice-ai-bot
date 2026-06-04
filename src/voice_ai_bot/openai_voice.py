from __future__ import annotations

import logging
import queue
import re
import socket
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from openai import OpenAI

from .audio_io import PcmPlayer
from .config import Config, SYSTEM_PROMPT
from .conversation import Message

LOGGER = logging.getLogger(__name__)

SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")


class TextChunker:
    def __init__(self, target_chars: int):
        self.target_chars = target_chars
        self.buffer = ""

    def push(self, text: str) -> list[str]:
        self.buffer += text
        chunks: list[str] = []
        while True:
            chunk = self._pop_first_complete_sentence()
            if chunk:
                chunks.append(chunk)
                continue
            if len(self.buffer) <= self.target_chars:
                break
            chunk = self._pop_at_limit()
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str | None:
        text = self.buffer.strip()
        self.buffer = ""
        return text or None

    def _pop_first_complete_sentence(self) -> str | None:
        match = SENTENCE_END_RE.search(self.buffer)
        if not match:
            return None
        end = match.end()
        chunk = self.buffer[:end].strip()
        self.buffer = self.buffer[end:].lstrip()
        return chunk or None

    def _pop_at_limit(self) -> str | None:
        if len(self.buffer) <= self.target_chars:
            return None
        cut = self.buffer.rfind(" ", 0, self.target_chars)
        if cut < self.target_chars // 2:
            cut = self.target_chars
        chunk = self.buffer[:cut].strip()
        self.buffer = self.buffer[cut:].lstrip()
        return chunk or None


class TtsQueue:
    def __init__(self, client: OpenAI, config: Config, player: PcmPlayer):
        self.client = client
        self.config = config
        self.player = player
        self.items: queue.Queue[str | None] = queue.Queue()
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(target=self._run, name="tts-player", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def put(self, text: str) -> None:
        text = text.strip()
        if text:
            self.items.put(text)

    def finish(self) -> None:
        self.items.put(None)
        self.thread.join()
        if self.errors:
            raise RuntimeError("TTS playback failed") from self.errors[0]

    def _run(self) -> None:
        while True:
            text = self.items.get()
            if text is None:
                return
            try:
                self._speak(text)
            except BaseException as exc:
                LOGGER.exception("TTS playback failed")
                self.errors.append(exc)

    def _speak(self, text: str) -> None:
        LOGGER.info("speaking %d characters", len(text))
        with self.client.audio.speech.with_streaming_response.create(
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            input=text,
            instructions=self.config.tts_instructions,
            response_format="pcm",
        ) as response:
            self.player.play_pcm_stream(response.iter_bytes(chunk_size=4096))


class OpenAIVoiceClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key, timeout=config.openai_timeout_seconds)
        self.player = PcmPlayer(config.audio_playback_device, volume_level=config.voice_volume)

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
                LOGGER.warning("waiting for network/DNS before OpenAI call: %s", exc)
                time.sleep(min(5.0, remaining))

    def transcribe(self, audio_path: Path) -> str:
        LOGGER.info("transcribing %s", audio_path)
        with audio_path.open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=self.config.transcription_model,
                file=audio_file,
                prompt="The speech is likely English or Russian.",
                response_format="json",
            )
        text = getattr(result, "text", "").strip()
        if not text:
            raise RuntimeError("transcription returned no text")
        LOGGER.info("transcript: %s", text)
        return text

    def respond_and_speak(self, history: Iterable[Message], user_text: str) -> str:
        context = [message.to_api() for message in history]
        context.append({"role": "user", "content": user_text})
        chunker = TextChunker(self.config.tts_chunk_chars)
        tts = TtsQueue(self.client, self.config, self.player)
        assistant_text_parts: list[str] = []
        tts.start()

        try:
            stream = self.client.responses.create(
                model=self.config.openai_model,
                instructions=SYSTEM_PROMPT,
                input=context,
                reasoning={"effort": self.config.openai_reasoning_effort},
                stream=True,
                store=False,
                truncation="auto",
            )
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if not delta:
                        continue
                    assistant_text_parts.append(delta)
                    for chunk in chunker.push(delta):
                        tts.put(chunk)
                elif event_type == "response.failed":
                    error = getattr(event, "error", None)
                    raise RuntimeError(f"response failed: {error}")
            rest = chunker.flush()
            if rest:
                tts.put(rest)
        finally:
            tts.finish()

        assistant_text = "".join(assistant_text_parts).strip()
        if not assistant_text:
            raise RuntimeError("model returned no text")
        return assistant_text
