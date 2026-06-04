from __future__ import annotations

from array import array

from voice_ai_bot.audio_io import scale_pcm16


def test_scale_pcm16_scales_signed_samples():
    samples = array("h", [10000, -10000, 123, -123])
    scaled = array("h")
    scaled.frombytes(scale_pcm16(samples.tobytes(), 5))

    assert list(scaled) == [5000, -5000, 61, -61]


def test_scale_pcm16_leaves_full_volume_unchanged():
    raw = array("h", [1, -2, 300]).tobytes()

    assert scale_pcm16(raw, 10) == raw
