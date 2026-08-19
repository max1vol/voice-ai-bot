import json
import threading
import time
import wave
from dataclasses import replace

import pytest

from voice_ai_bot.audio_io import RawPcmRecorder
from voice_ai_bot.conversation import Message
from voice_ai_bot.config import Config
from voice_ai_bot.realtime_voice import (
    ADD_SCHEDULED_TASK_TOOL_NAME,
    CLOSE_TOOL_NAME,
    DELETE_SCHEDULED_TASK_TOOL_NAME,
    GET_WEATHER_TOOL_NAME,
    GET_MUSIC_VOLUME_TOOL_NAME,
    GET_MUSIC_STATUS_TOOL_NAME,
    GET_VOICE_VOLUME_TOOL_NAME,
    LIST_MUSIC_TOOL_NAME,
    LIST_SCHEDULED_TASKS_TOOL_NAME,
    MEMORY_ADD_TOOL_NAME,
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_GET_SOURCE_TOOL_NAME,
    MEMORY_LIST_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_UPDATE_TOOL_NAME,
    BackgroundTask,
    PAUSE_MUSIC_TOOL_NAME,
    PLAY_MUSIC_TOOL_NAME,
    RESUME_MUSIC_TOOL_NAME,
    START_TASK_TOOL_NAME,
    START_CONTINUOUS_CAMERA_TOOL_NAME,
    STEER_TASK_TOOL_NAME,
    SET_MUSIC_VOLUME_TOOL_NAME,
    SET_VOICE_VOLUME_TOOL_NAME,
    STOP_MUSIC_TOOL_NAME,
    STOP_CONTINUOUS_CAMERA_TOOL_NAME,
    TAKE_PICTURE_TOOL_NAME,
    TASK_STATUS_TOOL_NAME,
    RealtimeConversationSession,
    BackgroundTaskManager,
    background_task_prompt,
    conversation_item_for_camera_snapshot,
    conversation_item_for_message,
    event_is_ignorable_control_error,
    extract_response_text,
    iter_wav_pcm16_chunks,
    parse_tool_arguments,
    realtime_session_instructions,
    realtime_turn_detection,
    realtime_turn_instructions,
    realtime_tools,
    response_function_calls,
    response_requested_close,
)
from voice_ai_bot.camera import CameraSnapshot


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, raw):
        self.sent.append(json.loads(raw))


class Event:
    def __init__(self, type, **kwargs):
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeResponsesClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.streams:
            raise AssertionError("unexpected Responses API call")
        return self.streams.pop(0)


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


def test_conversation_item_for_camera_snapshot_uses_auto_detail(tmp_path):
    snapshot = CameraSnapshot(
        path=tmp_path / "snapshot.jpg",
        mime_type="image/jpeg",
        size_bytes=123,
        data_url="data:image/jpeg;base64,abc",
    )

    item = conversation_item_for_camera_snapshot(snapshot, detail="auto", note="button press")

    assert item["role"] == "user"
    assert item["content"][0] == {"type": "input_text", "text": "[Camera snapshot: button press]"}
    assert item["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,abc",
        "detail": "auto",
    }


def test_realtime_turn_detection_defaults_to_server_vad(tmp_path):
    config = _config(tmp_path)

    assert realtime_turn_detection(config) == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 850,
        "create_response": False,
        "interrupt_response": False,
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
    assert STEER_TASK_TOOL_NAME in tool_names
    assert "query" not in start_task_tool["parameters"]["properties"]
    assert "wakeup_on_complete" in start_task_tool["parameters"]["properties"]
    assert "wakeup_on_progress" in start_task_tool["parameters"]["properties"]
    assert "list_background_tasks" in tool_names
    assert "get_background_task" in tool_names
    assert "cancel_background_task" in tool_names
    assert ADD_SCHEDULED_TASK_TOOL_NAME in tool_names
    assert LIST_SCHEDULED_TASKS_TOOL_NAME in tool_names
    assert DELETE_SCHEDULED_TASK_TOOL_NAME in tool_names
    assert GET_WEATHER_TOOL_NAME in tool_names
    assert SET_VOICE_VOLUME_TOOL_NAME in tool_names
    assert GET_VOICE_VOLUME_TOOL_NAME in tool_names
    assert LIST_MUSIC_TOOL_NAME in tool_names
    assert GET_MUSIC_STATUS_TOOL_NAME in tool_names
    assert GET_MUSIC_VOLUME_TOOL_NAME in tool_names
    assert PLAY_MUSIC_TOOL_NAME in tool_names
    assert PAUSE_MUSIC_TOOL_NAME in tool_names
    assert RESUME_MUSIC_TOOL_NAME in tool_names
    assert STOP_MUSIC_TOOL_NAME in tool_names
    assert SET_MUSIC_VOLUME_TOOL_NAME in tool_names
    assert TAKE_PICTURE_TOOL_NAME in tool_names
    assert START_CONTINUOUS_CAMERA_TOOL_NAME in tool_names
    assert STOP_CONTINUOUS_CAMERA_TOOL_NAME in tool_names
    assert MEMORY_SEARCH_TOOL_NAME in tool_names
    assert MEMORY_LIST_TOOL_NAME in tool_names
    assert MEMORY_ADD_TOOL_NAME in tool_names
    assert MEMORY_UPDATE_TOOL_NAME in tool_names
    assert MEMORY_FORGET_TOOL_NAME in tool_names
    assert MEMORY_GET_SOURCE_TOOL_NAME in tool_names
    assert "web_search" not in tool_names


def test_realtime_session_instructions_pin_general_assistant_identity(tmp_path):
    instructions = realtime_session_instructions(_config(tmp_path))

    assert "answer exactly: \"I'm Max Code, your general-purpose voice assistant.\"" in instructions
    assert "Do not present yourself as ChatGPT" in instructions
    assert "Do not assume a shop, vending, drink, or sales scenario" in instructions
    assert "mystery drink vending assistant" not in instructions
    assert "CB-38" not in instructions
    assert "Do not preamble before quick local tool calls" in instructions
    assert "For quick device-control confirmations" in instructions
    assert "button presses do not automatically take pictures" in instructions
    assert "start_continuous_camera" in instructions
    assert "Do not store runtime device settings such as voice volume, music volume" in instructions


def test_memory_add_tool_description_excludes_runtime_device_settings():
    memory_add = next(tool for tool in realtime_tools() if tool["name"] == MEMORY_ADD_TOOL_NAME)

    assert "Do not use this for runtime device settings" in memory_add["description"]
    assert "voice volume or music volume" in memory_add["parameters"]["properties"]["text"]["description"]


def test_realtime_turn_instructions_enforce_direct_short_confirmations(tmp_path):
    instructions = realtime_turn_instructions(_config(tmp_path))

    assert "You are Max Code, a general-purpose voice assistant" in instructions
    assert "Do not present yourself as ChatGPT" in instructions
    assert "Do not steer the conversation into shopkeeping, vending" in instructions
    assert "CB-38" not in instructions
    assert "If a fast local tool result is available, answer directly with no preamble" in instructions
    assert "usually under 10 words" in instructions


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


def test_response_events_are_ignored_while_recording(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    with session._state_lock:
        session._recorder = object()
        session._response_active = True
        session._active_response_id = "old_response"

    session._handle_event(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "old_response",
            "delta": "stale output",
        }
    )

    assert session._output_transcript_parts == []


def test_output_audio_is_not_played_while_recording(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))

    class FailingPlayer:
        def open_stream(self, **kwargs):
            raise AssertionError("playback should not open while recording")

    session.player = FailingPlayer()
    with session._state_lock:
        session._recorder = object()

    session._write_output_audio(b"1234")

    assert session._output_audio_bytes == 4


def test_server_vad_committed_audio_starts_app_created_response(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    ws = FakeWebSocket()
    with session._state_lock:
        session._ws = ws

    session._handle_event({"type": "input_audio_buffer.committed", "item_id": "item_1"})
    assert session._response_pending
    assert session._waiting_for_input_transcript

    session._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "please look at the board",
        }
    )

    deadline = time.time() + 1
    while time.time() < deadline and not any(event["type"] == "response.create" for event in ws.sent):
        time.sleep(0.01)

    response_events = [event for event in ws.sent if event["type"] == "response.create"]
    assert response_events
    assert response_events[0]["response"]["output_modalities"] == ["audio"]


def test_speech_started_during_assistant_response_is_treated_as_echo(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    ws = FakeWebSocket()
    with session._state_lock:
        session._ws = ws
        session._response_active = True
        session._active_response_id = "response_1"
        session._input_transcript_final = "does it work?"

    session._handle_event({"type": "input_audio_buffer.speech_started"})

    assert not any(event["type"] == "response.cancel" for event in ws.sent)
    assert session._response_active

    session._handle_event({"type": "input_audio_buffer.committed"})
    session._handle_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "speaker echo",
        }
    )

    assert session._input_transcript_final == "does it work?"

    session._handle_event(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "response_1",
            "delta": "Yes.",
        }
    )
    session._handle_event({"type": "response.done", "response": {"id": "response_1", "status": "completed"}})

    [turn] = session.pop_completed_turns()
    assert turn.user_text == "does it work?"
    assert turn.assistant_text == "Yes."


def test_running_background_task_does_not_count_as_live_response(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    task = BackgroundTask(id="task_123", request_text="check weather", status="running")
    session.tasks._tasks[task.id] = task

    assert session.tasks.has_running()
    assert not session.is_responding
    assert not session.is_voice_busy


def test_active_recording_counts_as_voice_busy(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    with session._state_lock:
        session._recorder = object()

    assert not session.is_responding
    assert session.is_voice_busy


def test_realtime_session_defaults_to_pi_raw_recorder(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))

    assert isinstance(session._recorder_factory(), RawPcmRecorder)


def test_begin_turn_cleans_up_when_recorder_start_fails(tmp_path):
    class FailingRecorder:
        def __init__(self):
            self.stopped = False

        def start(self):
            raise FileNotFoundError("missing recorder")

        def stop(self):
            self.stopped = True
            return 0.0

        def read(self, size=4096):
            return b""

    recorder = FailingRecorder()
    session = RealtimeConversationSession(_config(tmp_path), recorder_factory=lambda: recorder)

    with pytest.raises(FileNotFoundError):
        session.begin_turn([])

    assert recorder.stopped
    assert not session.is_recording
    assert not session.is_voice_busy
    assert session._capture_queue is None


def test_background_task_wakeup_queue_marks_reported(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    task = BackgroundTask(
        id="task_123",
        request_text="find something",
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


def test_start_task_tool_uses_raw_current_transcript_not_tool_query(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    calls = []

    with session._state_lock:
        session._input_transcript_final = "Please research today's train strikes near Cambridge."
        session._current_history = [Message(role="user", content="Earlier context")]

    def fake_start(request_text, history=(), title="", wakeup_on_complete=True, wakeup_on_progress=False, source="realtime"):
        calls.append((request_text, list(history), title, wakeup_on_complete, wakeup_on_progress, source))
        return {"ok": True}

    session.tasks.start = fake_start
    output = session._execute_realtime_tool_call(
        {
            "name": START_TASK_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "query": "model-written text that must be ignored",
                    "title": "Research",
                    "wakeup_on_progress": True,
                    "wakeup_on_complete": False,
                }
            ),
        }
    )

    assert output == {"ok": True}
    assert calls == [
        (
            "Please research today's train strikes near Cambridge.",
            [Message(role="user", content="Earlier context")],
            "Research",
            False,
            True,
            "realtime",
        )
    ]


def test_steer_task_tool_uses_raw_current_transcript(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    calls = []
    with session._state_lock:
        session._input_transcript_final = "By the way, for that task, compare tomorrow too."

    def fake_steer(message, task_id="", wakeup_on_complete=None):
        calls.append((message, task_id, wakeup_on_complete))
        return {"ok": True}

    session.tasks.steer = fake_steer
    output = session._execute_realtime_tool_call(
        {
            "name": STEER_TASK_TOOL_NAME,
            "arguments": json.dumps({"task_id": "task_abc", "wakeup_on_complete": True}),
        }
    )

    assert output == {"ok": True}
    assert calls == [("By the way, for that task, compare tomorrow too.", "task_abc", True)]


def test_background_task_prompt_contains_raw_transcript_and_steering(tmp_path):
    config = _config(tmp_path)
    task = BackgroundTask(
        id="task_123",
        request_text="Can you check today's weather and whether I need an umbrella?",
        history=[Message(role="user", content="I am in Cambridge today.")],
        memory_context="User usually walks to work.",
        steering_messages=[
            {"created_at": "2026-06-04T08:00:00+00:00", "text": "Also check tomorrow morning."}
        ],
    )

    prompt = background_task_prompt(config, task)

    assert "raw user transcript" in prompt
    assert "Can you check today's weather" in prompt
    assert "Also check tomorrow morning" in prompt
    assert "I am in Cambridge today" in prompt
    assert "User usually walks to work" in prompt


def test_background_task_status_tool_then_final_answer(tmp_path):
    config = _config(tmp_path)
    manager = BackgroundTaskManager(config)
    stream1 = [
        Event("response.created", response=Obj(id="resp_1")),
        Event(
            "response.completed",
            response=Obj(
                id="resp_1",
                output=[
                    Obj(
                        type="function_call",
                        name=TASK_STATUS_TOOL_NAME,
                        call_id="call_1",
                        arguments=json.dumps({"text": "I am checking the latest sources.", "speak_now": True}),
                    )
                ],
                output_text="",
            ),
        ),
    ]
    stream2 = [
        Event("response.created", response=Obj(id="resp_2")),
        Event("response.output_item.added", item=Obj(type="message", phase="final_answer")),
        Event("response.output_text.delta", delta="Take an umbrella."),
        Event("response.completed", response=Obj(id="resp_2", output=[], output_text="Take an umbrella.")),
    ]
    fake_responses = FakeResponsesClient([stream1, stream2])
    manager.client = Obj(responses=fake_responses)

    result = manager.start("Do I need an umbrella today?", history=[Message(role="user", content="Cambridge")])
    task_id = result["task"]["id"]
    deadline = time.time() + 2
    while time.time() < deadline:
        task = manager.get(task_id)["task"]
        if task["status"] == "completed":
            break
        time.sleep(0.01)

    task = manager.get(task_id)["task"]
    assert task["status"] == "completed"
    assert task["result"] == "Take an umbrella."
    assert task["status_updates"][0]["text"] == "I am checking the latest sources."
    assert fake_responses.calls[1]["previous_response_id"] == "resp_1"
    assert fake_responses.calls[1]["input"][0]["type"] == "function_call_output"
    wakeup = manager.pending_wakeups()[0]
    assert wakeup["wakeup"]["type"] == "status_update"


def test_background_task_steering_restarts_with_latest_message(tmp_path):
    config = _config(tmp_path)
    manager = BackgroundTaskManager(config)
    first_stream_ready = threading.Event()
    allow_first_stream_to_finish = threading.Event()

    class BlockingStream:
        def __iter__(self):
            first_stream_ready.set()
            allow_first_stream_to_finish.wait(timeout=2)
            yield Event("response.output_text.delta", delta="stale")

        def close(self):
            allow_first_stream_to_finish.set()

    final_stream = [
        Event("response.created", response=Obj(id="resp_final")),
        Event("response.output_item.added", item=Obj(type="message", phase="final_answer")),
        Event("response.output_text.delta", delta="Updated answer."),
        Event("response.completed", response=Obj(id="resp_final", output=[], output_text="Updated answer.")),
    ]
    fake_responses = FakeResponsesClient([BlockingStream(), final_stream])
    manager.client = Obj(responses=fake_responses)
    result = manager.start("Check weather today.")
    task_id = result["task"]["id"]
    assert first_stream_ready.wait(timeout=2)

    steer = manager.steer("Actually include tomorrow.", task_id=task_id)
    assert steer["ok"]
    assert allow_first_stream_to_finish.wait(timeout=2)

    deadline = time.time() + 2
    while time.time() < deadline:
        task = manager.get(task_id)["task"]
        if task["status"] == "completed":
            break
        time.sleep(0.01)

    task = manager.get(task_id)["task"]
    assert task["status"] == "completed"
    assert task["result"] == "Updated answer."
    assert task["steering_messages"][0]["text"] == "Actually include tomorrow."
    assert "Actually include tomorrow." in fake_responses.calls[-1]["input"]


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


def test_weather_tool_passes_cache_and_location_arguments(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    calls = []

    class FakeWeather:
        def get_weather(self, location="", units="metric", no_cache=False, forecast_days=0):
            calls.append((location, units, no_cache, forecast_days))
            return {"ok": True, "location_query": location, "cached": not no_cache, "forecast_days_requested": forecast_days}

    session.weather = FakeWeather()

    output = session._execute_realtime_tool_call(
        {
            "name": GET_WEATHER_TOOL_NAME,
            "arguments": json.dumps(
                {"location": "London,GB", "units": "metric", "no_cache": True, "forecast_days": 4}
            ),
        }
    )

    assert output["ok"]
    assert output["forecast_days_requested"] == 4
    assert calls == [("London,GB", "metric", True, 4)]


def test_background_task_tools_include_weather(tmp_path):
    manager = BackgroundTaskManager(_config(tmp_path))

    tool_names = {tool["name"] for tool in manager._task_tools() if tool.get("type") == "function"}

    assert TASK_STATUS_TOOL_NAME in tool_names
    assert GET_WEATHER_TOOL_NAME in tool_names


def test_background_task_strict_function_schemas_are_closed(tmp_path):
    manager = BackgroundTaskManager(_config(tmp_path))

    for tool in manager._task_tools():
        if tool.get("type") != "function" or not tool.get("strict"):
            continue
        parameters = tool["parameters"]
        property_names = set(parameters.get("properties", {}))

        assert parameters["type"] == "object"
        assert parameters.get("additionalProperties") is False
        assert set(parameters.get("required", [])) == property_names


def test_background_task_weather_tool_executes_local_service(tmp_path):
    manager = BackgroundTaskManager(_config(tmp_path))
    calls = []

    class FakeWeather:
        def get_weather(self, location="", units="metric", no_cache=False, forecast_days=0):
            calls.append((location, units, no_cache, forecast_days))
            return {"ok": True, "location_query": location, "forecast_days_requested": forecast_days}

    manager.weather = FakeWeather()

    outputs = manager._function_call_outputs(
        "task_123",
        Obj(
            output=[
                Obj(
                    type="function_call",
                    name=GET_WEATHER_TOOL_NAME,
                    call_id="call_weather",
                    arguments=json.dumps(
                        {"location": "Edinburgh,GB", "units": "metric", "forecast_days": 3, "no_cache": True}
                    ),
                )
            ]
        ),
    )

    assert calls == [("Edinburgh,GB", "metric", True, 3)]
    assert json.loads(outputs[0]["output"]) == {
        "ok": True,
        "location_query": "Edinburgh,GB",
        "forecast_days_requested": 3,
    }


def test_set_voice_volume_tool_updates_player(tmp_path):
    config = _config(tmp_path)
    session = RealtimeConversationSession(config)

    output = session._execute_realtime_tool_call(
        {"name": SET_VOICE_VOLUME_TOOL_NAME, "arguments": json.dumps({"level": 4})}
    )

    assert output == {"ok": True, "volume": 4, "scale": 0.4}
    assert session.player.volume_level() == 4
    assert json.loads(config.settings_file.read_text(encoding="utf-8")) == {"music_volume": 4, "voice_volume": 4}


def test_get_volume_tools_report_current_levels(tmp_path):
    config = _config(tmp_path)
    session = RealtimeConversationSession(config)
    session._execute_realtime_tool_call({"name": SET_VOICE_VOLUME_TOOL_NAME, "arguments": json.dumps({"level": 7})})

    class FakeMusic:
        def status(self):
            return {"ok": True, "state": "stopped", "volume": 3}

    session.music = FakeMusic()

    voice = session._execute_realtime_tool_call({"name": GET_VOICE_VOLUME_TOOL_NAME, "arguments": "{}"})
    music = session._execute_realtime_tool_call({"name": GET_MUSIC_VOLUME_TOOL_NAME, "arguments": "{}"})

    assert voice == {"ok": True, "volume": 7, "scale": 0.7}
    assert music == {"ok": True, "volume": 3, "scale": 0.3}


def test_music_tools_delegate_to_music_player(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    calls = []

    class FakeMusic:
        def list(self):
            calls.append(("list",))
            return {"ok": True, "songs": [{"title": "Baby Shark", "duration": "1:51"}]}

        def status(self):
            calls.append(("status",))
            return {"ok": True, "state": "stopped", "volume": 4}

        def request_play(self, query):
            calls.append(("play", query))
            return {"ok": True, "deferred": True}

        def pause(self, reason="user"):
            calls.append(("pause", reason))
            return {"ok": True}

        def request_resume(self):
            calls.append(("resume",))
            return {"ok": True, "deferred": True}

        def stop(self):
            calls.append(("stop",))
            return {"ok": True}

        def set_volume(self, level):
            calls.append(("volume", level))
            return {"ok": True, "volume": level}

    session.music = FakeMusic()

    assert session._execute_realtime_tool_call({"name": LIST_MUSIC_TOOL_NAME, "arguments": "{}"})["songs"][0]["duration"] == "1:51"
    assert session._execute_realtime_tool_call({"name": GET_MUSIC_STATUS_TOOL_NAME, "arguments": "{}"})["state"] == "stopped"
    assert session._execute_realtime_tool_call({"name": PLAY_MUSIC_TOOL_NAME, "arguments": json.dumps({"query": "baby shark"})})["deferred"]
    assert session._execute_realtime_tool_call({"name": PAUSE_MUSIC_TOOL_NAME, "arguments": "{}"})["ok"]
    assert session._execute_realtime_tool_call({"name": RESUME_MUSIC_TOOL_NAME, "arguments": "{}"})["deferred"]
    assert session._execute_realtime_tool_call({"name": STOP_MUSIC_TOOL_NAME, "arguments": "{}"})["ok"]
    assert session._execute_realtime_tool_call({"name": SET_MUSIC_VOLUME_TOOL_NAME, "arguments": json.dumps({"level": 6})})["volume"] == 6
    assert calls == [
        ("list",),
        ("status",),
        ("play", "baby shark"),
        ("pause", "user"),
        ("resume",),
        ("stop",),
        ("volume", 6),
    ]


def test_take_picture_tool_attaches_image_item(tmp_path):
    session = RealtimeConversationSession(_config(tmp_path))
    ws = FakeWebSocket()
    with session._state_lock:
        session._ws = ws
    session.ensure_open = lambda history: None
    shutter_calls = []
    session._play_camera_shutter = lambda: shutter_calls.append("shutter")

    class FakeCamera:
        def capture(self, settle_seconds=None, shutter_callback=None):
            assert settle_seconds == 3.0
            assert shutter_callback is not None
            shutter_callback()
            return CameraSnapshot(
                path=tmp_path / "snapshot.jpg",
                mime_type="image/jpeg",
                size_bytes=3,
                data_url="data:image/jpeg;base64,abc",
            )

    session.camera = FakeCamera()

    output = session._execute_realtime_tool_call(
        {"name": TAKE_PICTURE_TOOL_NAME, "arguments": json.dumps({"reason": "check board"})}
    )

    assert output == {
        "ok": True,
        "path": str(tmp_path / "snapshot.jpg"),
        "bytes": 3,
        "detail": "auto",
    }
    assert ws.sent[0]["type"] == "conversation.item.create"
    assert ws.sent[0]["item"]["content"][1]["type"] == "input_image"
    assert shutter_calls == ["shutter"]


def test_continuous_camera_tool_starts_updates_and_stops(monkeypatch, tmp_path):
    config = replace(_config(tmp_path), camera_snapshot_settle_seconds=0.0)
    session = RealtimeConversationSession(config)
    ws = FakeWebSocket()
    with session._state_lock:
        session._ws = ws
    session.ensure_open = lambda history: None
    started = []
    closed = []

    class FakeContinuousCamera:
        def __init__(self, _config):
            self.config = _config

        def start(self):
            started.append("start")

        def snapshot(self, note="", timeout=None):
            return CameraSnapshot(
                path=tmp_path / "continuous.jpg",
                mime_type="image/jpeg",
                size_bytes=3,
                data_url="data:image/jpeg;base64,abc",
            )

        def close(self):
            closed.append("close")

    monkeypatch.setattr("voice_ai_bot.realtime_voice.ContinuousCameraCapture", FakeContinuousCamera)

    output = session._execute_realtime_tool_call(
        {
            "name": START_CONTINUOUS_CAMERA_TOOL_NAME,
            "arguments": json.dumps({"reason": "tic tac toe", "interval_seconds": 0.2}),
        }
    )

    assert output["ok"]
    assert output["active"]
    assert output["interval_seconds"] == 1.0
    deadline = time.monotonic() + 1.0
    while not ws.sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ws.sent[0]["item"]["content"][0]["text"] == "[Camera snapshot: continuous camera: tic tac toe]"

    updated = session._execute_realtime_tool_call(
        {
            "name": START_CONTINUOUS_CAMERA_TOOL_NAME,
            "arguments": json.dumps({"reason": "faster board", "interval_seconds": 9}),
        }
    )
    assert updated["updated"]
    assert updated["interval_seconds"] == 5.0

    stopped = session._execute_realtime_tool_call(
        {"name": STOP_CONTINUOUS_CAMERA_TOOL_NAME, "arguments": json.dumps({"reason": "done"})}
    )

    assert stopped == {"ok": True, "active": False, "stopped": True}
    assert started == ["start"]
    assert closed


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
        settings_file=tmp_path / "settings.json",
        memory_dir=tmp_path / "agent",
    )
