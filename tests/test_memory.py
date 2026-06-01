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


def test_bootstrap_context_contains_max_code_identity(tmp_path):
    store = MemoryStore(_config(tmp_path))

    context = store.bootstrap_context()

    assert "Max Code" in context
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
