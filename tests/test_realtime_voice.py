import json
import time
import wave

from voice_ai_bot.conversation import Message
from voice_ai_bot.config import Config
from voice_ai_bot.realtime_voice import (
    ADD_SCHEDULED_TASK_TOOL_NAME,
    CLOSE_TOOL_NAME,
    DELETE_SCHEDULED_TASK_TOOL_NAME,
    LIST_SCHEDULED_TASKS_TOOL_NAME,
    MEMORY_ADD_TOOL_NAME,
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_GET_SOURCE_TOOL_NAME,
    MEMORY_LIST_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_UPDATE_TOOL_NAME,
    BackgroundTask,
    START_TASK_TOOL_NAME,
    RealtimeConversationSession,
    conversation_item_for_message,
    event_is_ignorable_control_error,
    extract_response_text,
    iter_wav_pcm16_chunks,
    parse_tool_arguments,
    realtime_session_instructions,
    realtime_tools,
    response_function_calls,
    response_requested_close,
)


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, raw):
        self.sent.append(json.loads(raw))


def test_iter_wav_pcm16_chunks_reads_expected_audio(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x02" * 10)

    chunks = list(iter_wav_pcm16_chunks(path, expected_rate=24000, chunk_frames=4))

    assert b"".join(chunks) == b"\x01\x02" * 10
    assert len(chunks) == 3


def test_conversation_item_for_saved_messages():
    assert conversation_item_for_message(Message(role="user", content="hi")) == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
    assert conversation_item_for_message(Message(role="assistant", content="hello")) == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
    }


def test_response_requested_close_detects_close_tool():
    response = {
        "output": [
            {
                "type": "function_call",
                "name": CLOSE_TOOL_NAME,
                "arguments": '{"reason":"user asked to stop"}',
            }
        ]
    }

    assert response_requested_close(response)
    assert response_function_calls(response, CLOSE_TOOL_NAME) == response["output"]
    assert response_function_calls(response) == response["output"]
    assert parse_tool_arguments(response["output"][0]) == {"reason": "user asked to stop"}


def test_extract_response_text_reads_audio_transcript():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "audio", "transcript": "Hello"},
                    {"type": "output_text", "text": " there"},
                ],
            }
        ]
    }

    assert extract_response_text(response) == "Hello there"


def test_control_errors_can_be_ignored_by_event_id():
    assert event_is_ignorable_control_error(
        {
            "type": "error",
            "error": {
                "event_id": "cancel_123",
                "message": "No active response.",
            },
        }
    )
    assert event_is_ignorable_control_error(
        {
            "type": "error",
            "error": {
                "event_id": "truncate_123",
                "message": "Item already done.",
            },
        }
    )
    assert not event_is_ignorable_control_error({"type": "error", "error": {"event_id": "other"}})


def test_realtime_tools_use_async_background_task_interface():
    tools = realtime_tools()
    tool_names = {tool["name"] for tool in tools}
    start_task_tool = next(tool for tool in tools if tool["name"] == START_TASK_TOOL_NAME)

    assert START_TASK_TOOL_NAME in tool_names
    assert "wakeup_on_complete" in start_task_tool["parameters"]["properties"]
    assert "list_background_tasks" in tool_names
    assert "get_background_task" in tool_names
    assert "cancel_background_task" in tool_names
    assert ADD_SCHEDULED_TASK_TOOL_NAME in tool_names
    assert LIST_SCHEDULED_TASKS_TOOL_NAME in tool_names
    assert DELETE_SCHEDULED_TASK_TOOL_NAME in tool_names
    assert MEMORY_SEARCH_TOOL_NAME in tool_names
    assert MEMORY_LIST_TOOL_NAME in tool_names
    assert MEMORY_ADD_TOOL_NAME in tool_names
    assert MEMORY_UPDATE_TOOL_NAME in tool_names
    assert MEMORY_FORGET_TOOL_NAME in tool_names
    assert MEMORY_GET_SOURCE_TOOL_NAME in tool_names
    assert "web_search" not in tool_names


def test_realtime_session_instructions_pin_max_code_identity(tmp_path):
    instructions = realtime_session_instructions(_config(tmp_path))

    assert "say that you are Max Code" in instructions
    assert "Do not introduce yourself as ChatGPT" in instructions


def test_interrupt_skips_truncate_when_session_is_closed(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    with session._state_lock:
        session._output_item_id = "item_123"
        session._output_audio_bytes = 4800
        session._interrupt_response_locked()

    assert session._output_item_id == ""
    assert session._output_audio_bytes == 0


def test_silent_cooldown_cancels_pending_response(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    ws = FakeWebSocket()
    with session._state_lock:
        session._ws = ws
        session._response_pending = True
        session._response_wait_started_at = time.monotonic() - 10

    session.cool_down_if_silent()

    assert not session._response_pending
    assert ws.sent[0]["type"] == "response.cancel"


def test_response_created_is_ignored_until_response_create_was_sent(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    with session._state_lock:
        session._response_pending = True
        session._waiting_for_input_transcript = True

    session._handle_event({"type": "response.created", "response": {"id": "old_response"}})

    assert not session._response_active
    assert session._response_pending
    assert session._active_response_id == ""

    with session._state_lock:
        session._waiting_for_input_transcript = False
        session._response_create_pending = True

    session._handle_event({"type": "response.created", "response": {"id": "new_response"}})

    assert session._response_active
    assert not session._response_pending
    assert session._active_response_id == "new_response"


def test_stale_response_events_do_not_pollute_barge_in_turn(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    with session._state_lock:
        session._recorder = object()
        session._response_active = True
        session._active_response_id = "old_response"

    session._handle_event(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "old_response",
            "delta": "old speech",
        }
    )

    assert session._output_transcript_parts == []

    with session._state_lock:
        session._recorder = None
        session._response_active = False
        session._response_pending = True
        session._active_response_id = ""

    session._handle_event(
        {
            "type": "response.done",
            "response": {
                "id": "old_response",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "stale"}]}],
            },
        }
    )

    assert session.pop_completed_turns() == []
    assert session._response_pending


def test_running_background_task_does_not_count_as_live_response(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    task = BackgroundTask(id="task_123", query="check weather", status="running")
    session.tasks._tasks[task.id] = task

    assert session.tasks.has_running()
    assert not session.is_responding
    assert not session.is_voice_busy


def test_background_task_wakeup_queue_marks_reported(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    task = BackgroundTask(
        id="task_123",
        query="find something",
        title="Find something",
        wakeup_on_complete=True,
        status="completed",
        result="done",
        completed_at=time.time(),
    )
    session.tasks._tasks[task.id] = task

    pending = session.pending_background_wakeups()

    assert pending[0]["id"] == "task_123"
    assert pending[0]["result"] == "done"
    session.mark_background_wakeup_reported("task_123")
    assert session.pending_background_wakeups() == []


def test_start_task_tool_passes_wakeup_mode(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    calls = []

    def fake_start(query, title="", wakeup_on_complete=True):
        calls.append((query, title, wakeup_on_complete))
        return {"ok": True}

    session.tasks.start = fake_start
    output = session._execute_realtime_tool_call(
        {
            "name": START_TASK_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "query": "research this",
                    "title": "Research",
                    "wakeup_on_complete": False,
                }
            ),
        }
    )

    assert output == {"ok": True}
    assert calls == [("research this", "Research", False)]


def test_memory_tools_update_store(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))

    added = session._execute_realtime_tool_call(
        {
            "name": MEMORY_ADD_TOOL_NAME,
            "arguments": json.dumps({"text": "User likes terse responses.", "kind": "preference"}),
        }
    )
    entry_id = added["entry"]["id"]

    assert added["ok"]
    assert session._execute_realtime_tool_call(
        {"name": MEMORY_SEARCH_TOOL_NAME, "arguments": json.dumps({"query": "terse"})}
    )["results"][0]["entry_id"] == entry_id
    assert session._execute_realtime_tool_call(
        {
            "name": MEMORY_UPDATE_TOOL_NAME,
            "arguments": json.dumps({"entry_id": entry_id, "text": "User likes detailed responses."}),
        }
    )["ok"]
    assert session._execute_realtime_tool_call(
        {"name": MEMORY_FORGET_TOOL_NAME, "arguments": json.dumps({"entry_id": entry_id})}
    )["ok"]
    assert session._execute_realtime_tool_call(
        {"name": MEMORY_LIST_TOOL_NAME, "arguments": "{}"}
    )["entries"] == []


def test_memory_tool_arguments_are_tolerant(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    session.memory.append_turn("alpha beta", "gamma")

    output = session._execute_realtime_tool_call(
        {
            "name": MEMORY_SEARCH_TOOL_NAME,
            "arguments": json.dumps({"query": 123, "max_results": "not-an-int"}),
        }
    )

    assert output["ok"]


def _config(tmp_path):
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
