from __future__ import annotations

from array import array
from types import SimpleNamespace

from voice_ai_bot.audio_io import PcmOutputStream, scale_pcm16


def test_scale_pcm16_scales_signed_samples():
    samples = array("h", [10000, -10000, 123, -123])
    scaled = array("h")
    scaled.frombytes(scale_pcm16(samples.tobytes(), 5))

    assert list(scaled) == [5000, -5000, 61, -61]


def test_scale_pcm16_leaves_full_volume_unchanged():
    raw = array("h", [1, -2, 300]).tobytes()

    assert scale_pcm16(raw, 10) == raw


def test_pcm_output_stream_contains_broken_pipe():
    class FailingStdin:
        closed = False

        def write(self, chunk):
            raise BrokenPipeError("closed")

        def flush(self):
            raise AssertionError("flush should not run after write fails")

    stream = PcmOutputStream("plughw:test", rate=24000, channels=1)
    stream.process = SimpleNamespace(stdin=FailingStdin(), poll=lambda: 1)

    stream.write(b"\x00\x00")

    assert stream.bytes_written == 0
    assert stream._aborted
