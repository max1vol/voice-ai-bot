from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from voice_ai_bot.config import Config
from voice_ai_bot.scheduled_tasks import ScheduledTaskStore, parse_local_datetime


def test_add_list_delete_scheduled_task(tmp_path):
    store = ScheduledTaskStore(_config(tmp_path))

    created = store.add(
        title="Wake up",
        prompt="It is 7:30, wake up.",
        run_at="2026-06-02T07:30:00",
    )

    assert created["ok"]
    task_id = created["task"]["id"]
    assert created["task"]["status"] == "active"
    assert [task["id"] for task in store.list()["tasks"]] == [task_id]
    assert store.delete(task_id) == {"ok": True, "deleted_task_id": task_id}
    assert store.list()["tasks"] == []


def test_rejects_spoken_scheduled_task_during_quiet_hours(tmp_path):
    store = ScheduledTaskStore(_config(tmp_path))

    before_quiet_end = store.add(title="Too early", prompt="Wake up", run_at="2026-06-02T07:00:00")
    one_minute_before_quiet_end = store.add(title="Still too early", prompt="Wake up", run_at="2026-06-02T07:29:00")
    at_quiet_start = store.add(title="Too late", prompt="Reminder", run_at="2026-06-02T21:00:00")
    at_quiet_end = store.add(title="Allowed", prompt="Wake up", run_at="2026-06-02T07:30:00")

    assert not before_quiet_end["ok"]
    assert "quiet hours" in before_quiet_end["error"]
    assert not one_minute_before_quiet_end["ok"]
    assert "quiet hours" in one_minute_before_quiet_end["error"]
    assert not at_quiet_start["ok"]
    assert at_quiet_end["ok"]


def test_due_once_task_completes_when_marked_started(tmp_path):
    store = ScheduledTaskStore(_config(tmp_path))
    created = store.add(title="Reminder", prompt="Stand up", run_at="2026-06-02T08:00:00")
    task_id = created["task"]["id"]

    due = store.due(_dt("2026-06-02T08:00:01"))

    assert [task.id for task in due] == [task_id]
    store.mark_started(task_id, _dt("2026-06-02T08:00:02"))
    assert store.list()["tasks"] == []
    inactive = store.list(include_inactive=True)["tasks"]
    assert inactive[0]["status"] == "completed"
    assert inactive[0]["last_started_at"].startswith("2026-06-02T08:00:02")


def test_daily_task_advances_to_next_day(tmp_path):
    store = ScheduledTaskStore(_config(tmp_path))
    created = store.add(
        title="Daily wake up",
        prompt="Wake up",
        run_at="2026-06-02T07:30:00",
        repeat="daily",
    )
    task_id = created["task"]["id"]

    store.mark_started(task_id, _dt("2026-06-02T07:30:05"))

    active = store.list()["tasks"]
    assert active[0]["id"] == task_id
    assert active[0]["status"] == "active"
    assert active[0]["run_at"].startswith("2026-06-03T07:30:00")


def test_quiet_due_once_task_is_skipped_if_policy_changes(tmp_path):
    store = ScheduledTaskStore(_config(tmp_path))
    path = tmp_path / "scheduled_tasks.json"
    path.write_text(
        """
{
  "version": 1,
  "tasks": [
    {
      "id": "sched_old",
      "title": "Old alarm",
      "prompt": "Wake up",
      "run_at": "2026-06-02T07:00:00+01:00",
      "action": "speak",
      "repeat": "once",
      "status": "active",
      "created_at": "2026-06-01T12:00:00+00:00",
      "updated_at": "2026-06-01T12:00:00+00:00"
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert store.due(_dt("2026-06-02T07:01:00")) == []
    inactive = store.list(include_inactive=True)["tasks"]
    assert inactive[0]["status"] == "skipped"
    assert inactive[0]["skipped_reason"] == "quiet_hours"


def test_parse_local_datetime_converts_aware_time():
    parsed = parse_local_datetime("2026-06-02T06:30:00+00:00", "Europe/London")

    assert parsed.isoformat().startswith("2026-06-02T07:30:00")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo("Europe/London"))


def _config(tmp_path: Path) -> Config:
    return Config(
        openai_api_key="sk-test",
        voice_bot_backend="realtime",
        openai_model="gpt-5.5",
        openai_reasoning_effort="low",
        openai_connectivity_host="api.openai.com",
        openai_connectivity_wait_seconds=1.0,
        openai_timeout_seconds=1.0,
        transcription_model="gpt-4o-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_voice="cedar",
        tts_instructions="test",
        realtime_model="gpt-realtime-2",
        realtime_reasoning_effort="low",
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
        web_search_reasoning_effort="medium",
        web_search_context_size="medium",
        web_search_timeout_seconds=1.0,
        task_model="gpt-5.5",
        task_reasoning_effort="medium",
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
    )
