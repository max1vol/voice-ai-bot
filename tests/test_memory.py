from __future__ import annotations

from pathlib import Path

from voice_ai_bot.config import Config
from voice_ai_bot.conversation import Message
from voice_ai_bot.memory import MemoryStore


def test_memory_add_search_update_forget(tmp_path):
    store = MemoryStore(_config(tmp_path))

    added = store.add_entry("User prefers concise spoken answers.", kind="preference")

    assert added["ok"]
    entry_id = added["entry"]["id"]
    search = store.search("concise answers")
    assert search["results"][0]["entry_id"] == entry_id

    updated = store.update_entry(entry_id, "User prefers detailed spoken answers.")
    assert updated["ok"]
    assert "detailed" in store.search("detailed answers")["results"][0]["snippet"]

    forgotten = store.forget_entry(entry_id=entry_id, reason="user asked")
    assert forgotten["ok"]
    assert store.search("detailed answers")["results"] == []
    assert store.list_entries()["entries"] == []
    assert store.list_entries(include_forgotten=True)["entries"][0]["status"] == "forgotten"


def test_daily_notes_are_searchable_and_conversation_flushes(tmp_path):
    store = MemoryStore(_config(tmp_path))

    store.append_turn("remember the atlas code", "noted")
    store.flush_conversation(
        [
            Message(role="user", content="old request"),
            Message(role="assistant", content="old answer"),
        ],
        "test clear",
    )

    results = store.search("atlas code")["results"]
    assert results
    assert results[0]["source"] == "daily"
    flush_results = store.search("old request")["results"]
    assert flush_results


def test_active_context_prefers_durable_memory_over_daily_turn_logs(tmp_path):
    store = MemoryStore(_config(tmp_path))

    store.add_entry("If asked who you are, say Max Code.", kind="identity")
    store.append_turn("Who are you?", "I'm ChatGPT.")

    active = store.active_context("who are you")
    daily_search = store.search("who are you")["results"]

    assert "Max Code" in active
    assert "ChatGPT" not in active
    assert any("ChatGPT" in result["snippet"] for result in daily_search)


def test_turn_notes_are_queued_for_consolidation(tmp_path):
    store = MemoryStore(_config(tmp_path))

    store.append_turn("remember that I prefer Russian replies", "I will remember that.")

    pending = store.pending_consolidation_notes(max_notes=10, max_chars=4000)
    assert len(pending) == 1
    assert pending[0]["type"] == "turn"
    assert "Russian replies" in pending[0]["text"]

    store.mark_consolidation_processed(
        [pending[0]["id"]],
        summary="saved preference",
        operation_results=[{"result": {"ok": True}}],
    )
    assert store.pending_consolidation_notes(max_notes=10, max_chars=4000) == []


def test_consolidation_operations_are_applied_as_app_owned_writes(tmp_path):
    store = MemoryStore(_config(tmp_path))

    results = store.apply_consolidation_operations(
        [
            {
                "action": "add",
                "kind": "preference",
                "text": "User prefers direct answers.",
            },
            {
                "action": "ignore",
                "reason": "transient chit-chat",
            },
        ]
    )

    assert results[0]["result"]["ok"]
    assert results[1]["result"]["ignored"]
    entries = store.list_entries()["entries"]
    assert entries[0]["text"] == "User prefers direct answers."
    assert entries[0]["source"] == "gpt-5-5"


def test_bootstrap_context_contains_general_assistant_identity(tmp_path):
    store = MemoryStore(_config(tmp_path))

    context = store.bootstrap_context()

    assert "Max Code" in context
    assert "general-purpose voice assistant" in context
    assert "do not present yourself as ChatGPT" in context
    assert "CB-38" not in context
    assert "Cambridge" in context


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
        realtime_turn_detection="server_vad",
        realtime_vad_threshold=0.5,
        realtime_vad_prefix_padding_ms=300,
        realtime_vad_silence_duration_ms=850,
        realtime_semantic_vad_eagerness="medium",
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
