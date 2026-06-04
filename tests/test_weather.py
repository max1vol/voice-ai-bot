from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from voice_ai_bot.weather import OpenWeatherService

from test_realtime_voice import _config


def test_openweather_current_weather_is_cached(tmp_path):
    calls: list[str] = []

    def fake_fetch(url: str, timeout: float):
        calls.append(url)
        parsed = urlparse(url)
        if parsed.path.endswith("/geo/1.0/direct"):
            query = parse_qs(parsed.query)
            assert query["q"] == ["Cambridge,GB"]
            return [{"name": "Cambridge", "country": "GB", "lat": 52.2, "lon": 0.12}]
        return {
            "name": "Cambridge",
            "dt": 1780510000,
            "weather": [{"main": "Clouds", "description": "broken clouds"}],
            "main": {"temp": 18.2, "feels_like": 17.8, "humidity": 70, "pressure": 1012},
            "wind": {"speed": 4.2, "deg": 230},
            "clouds": {"all": 75},
            "sys": {"country": "GB"},
        }

    config = _config(tmp_path)
    config = config.__class__(**{**config.__dict__, "openweather_api_key": "ow-test"})
    service = OpenWeatherService(config, fetch_json=fake_fetch)

    first = service.get_current_weather()
    second = service.get_current_weather()
    refreshed = service.get_current_weather(no_cache=True)

    assert first["ok"]
    assert first["cached"] is False
    assert first["location"]["name"] == "Cambridge"
    assert first["temperature"]["current"] == 18.2
    assert second["cached"] is True
    assert refreshed["cached"] is False
    assert len(calls) == 4


def test_openweather_requires_api_key(tmp_path):
    service = OpenWeatherService(_config(tmp_path))

    result = service.get_current_weather()

    assert not result["ok"]
    assert "OPENWEATHER_API_KEY" in result["error"]
