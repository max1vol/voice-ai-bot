from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from voice_ai_bot.config import Config
from voice_ai_bot.memory import MemoryStore
from voice_ai_bot.memory_consolidation import MemoryConsolidator


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def test_consolidator_uses_gpt55_payload_to_curate_memory(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config)
    store.append_turn("Please remember that I like terse answers.", "Got it.")
    client = FakeClient(
        {
            "summary": "Saved explicit preference.",
            "operations": [
                {
                    "action": "add",
                    "kind": "preference",
                    "text": "User likes terse answers.",
                    "reason": "User explicitly asked to remember it.",
                }
            ],
        }
    )
    consolidator = MemoryConsolidator(config, store, client=client)

    result = consolidator.run_once()

    assert result["ok"]
    assert result["processed"] == 1
    entries = store.list_entries()["entries"]
    assert entries[0]["text"] == "User likes terse answers."
    assert entries[0]["source"] == "gpt-5-5"
    assert store.pending_consolidation_notes(max_notes=10, max_chars=4000) == []
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["reasoning"]["effort"] == "high"
    assert "Existing durable memory entries JSON" in call["input"]


def test_consolidator_keeps_pending_notes_after_failure(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config)
    store.append_turn("Remember the blue notebook.", "Okay.")
    client = FakeClient({"not": "used"})

    def broken_create(**_kwargs):
        raise RuntimeError("network down")

    client.responses.create = broken_create
    consolidator = MemoryConsolidator(config, store, client=client)

    result = consolidator.run_once()

    assert not result["ok"]
    pending = store.pending_consolidation_notes(max_notes=10, max_chars=4000)
    assert len(pending) == 1
    assert pending[0]["attempts"] == 1


def _config(tmp_path: Path) -> Config:
    return Config(
        openai_api_key="sk-test",
        voice_bot_backend="realtime",
        openai_model="gpt-5.5",
        openai_reasoning_effort="high",
        openai_connectivity_host="api.openai.com",
        openai_connectivity_wait_seconds=1.0,
        openai_timeout_seconds=1.0,
        transcription_model="gpt-4o-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="cedar",
        tts_instructions="test",
        realtime_model="gpt-realtime-2",
        realtime_reasoning_effort="medium",
        realtime_voice="marin",
        realtime_input_rate=24000,
        realtime_input_transcription_model="gpt-4o-transcribe",
        realtime_response_timeout_seconds=1.0,
        realtime_idle_timeout_seconds=1.0,
        realtime_max_session_seconds=1.0,
        realtime_silent_cooldown_seconds=5.0,
        realtime_history_messages=4,
        realtime_safety_identifier="test",
        user_city="Cambridge",
        user_region="Cambridgeshire",
        user_country="GB",
        user_timezone="Europe/London",
        web_search_model="gpt-5.5",
        web_search_reasoning_effort="high",
        web_search_context_size="medium",
        web_search_timeout_seconds=1.0,
        task_model="gpt-5.5",
        task_reasoning_effort="high",
        task_reasoning_summary="auto",
        task_timeout_seconds=1.0,
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
