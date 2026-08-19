from voice_ai_bot.desktop_demo import MacBufferedPcmOutputStream, MacPcmOutputStream, MacPcmPlayer, _user_facing_error


def test_mac_voice_player_can_use_buffered_afplay_stream():
    player = MacPcmPlayer(volume_level=5, buffered_playback=True)

    stream = player.open_stream()

    assert isinstance(stream, MacBufferedPcmOutputStream)


def test_mac_player_defaults_to_streaming_ffplay_for_music():
    player = MacPcmPlayer(volume_level=4)

    stream = player.open_stream()

    assert isinstance(stream, MacPcmOutputStream)


def test_desktop_demo_sanitizes_openai_key_error():
    error = RuntimeError("realtime API error: Incorrect API key provided: sk-test")

    assert _user_facing_error(error) == "OpenAI rejected OPENAI_API_KEY in .env."
