import json

from voice_ai_bot.settings import RuntimeSettings


def test_runtime_settings_seeds_defaults_and_persists_volume_changes(tmp_path):
    path = tmp_path / "settings.json"
    settings = RuntimeSettings(path, default_voice_volume=5, default_music_volume=4)

    assert settings.ensure() == {"voice_volume": 5, "music_volume": 4}
    assert json.loads(path.read_text(encoding="utf-8")) == {"music_volume": 4, "voice_volume": 5}

    assert settings.set_voice_volume(9) == {"voice_volume": 9, "music_volume": 4}
    assert RuntimeSettings(path, default_voice_volume=5, default_music_volume=4).voice_volume() == 9

    assert settings.set_music_volume(2) == {"voice_volume": 9, "music_volume": 2}
    assert RuntimeSettings(path, default_voice_volume=5, default_music_volume=4).music_volume() == 2


def test_runtime_settings_clamps_and_repairs_existing_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"voice_volume": 99}', encoding="utf-8")
    settings = RuntimeSettings(path, default_voice_volume=5, default_music_volume=4)

    assert settings.ensure() == {"voice_volume": 10, "music_volume": 4}
    assert json.loads(path.read_text(encoding="utf-8")) == {"music_volume": 4, "voice_volume": 10}
