import wave

from voice_ai_bot.conversation import Message
from voice_ai_bot.realtime_voice import (
    CLOSE_TOOL_NAME,
    conversation_item_for_message,
    event_is_ignorable_control_error,
    extract_response_text,
    iter_wav_pcm16_chunks,
    parse_tool_arguments,
    response_function_calls,
    response_requested_close,
)


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
