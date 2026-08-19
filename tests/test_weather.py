from __future__ import annotations

from datetime import datetime, timezone
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


def test_openweather_errors_redact_api_key(tmp_path):
    def fake_fetch(url: str, timeout: float):
        raise RuntimeError(f"failed URL {url}")

    config = _config(tmp_path)
    config = config.__class__(**{**config.__dict__, "openweather_api_key": "ow-secret"})
    service = OpenWeatherService(config, fetch_json=fake_fetch)

    result = service.get_current_weather(location="Cambridge,GB", no_cache=True)

    assert not result["ok"]
    assert "ow-secret" not in result["error"]
    assert "appid=<redacted>" in result["error"]


def test_openweather_forecast_is_cached_and_supports_named_location(tmp_path):
    calls: list[str] = []

    start = int(datetime(2026, 6, 7, 6, tzinfo=timezone.utc).timestamp())

    def fake_fetch(url: str, timeout: float):
        calls.append(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if parsed.path.endswith("/geo/1.0/direct"):
            assert query["q"] == ["Oxford,GB"]
            return [{"name": "Oxford", "country": "GB", "lat": 51.75, "lon": -1.26}]
        if parsed.path.endswith("/data/2.5/weather"):
            return {
                "name": "Oxford",
                "dt": start,
                "weather": [{"main": "Clouds", "description": "overcast clouds"}],
                "main": {"temp": 19.1, "feels_like": 18.7, "humidity": 66, "pressure": 1015},
                "wind": {"speed": 3.2, "deg": 220},
                "clouds": {"all": 92},
                "sys": {"country": "GB"},
            }
        return {
            "city": {"name": "Oxford", "country": "GB", "timezone": 3600},
            "list": [
                {
                    "dt": start,
                    "weather": [{"main": "Clouds", "description": "overcast clouds"}],
                    "main": {"temp": 19.1, "feels_like": 18.7, "temp_min": 18.4, "temp_max": 19.3, "humidity": 66},
                    "wind": {"speed": 3.2, "deg": 220},
                    "clouds": {"all": 92},
                    "pop": 0.15,
                },
                {
                    "dt": start + 6 * 3600,
                    "weather": [{"main": "Rain", "description": "light rain"}],
                    "main": {"temp": 17.0, "feels_like": 16.4, "temp_min": 16.8, "temp_max": 17.1, "humidity": 80},
                    "wind": {"speed": 4.8, "deg": 240, "gust": 7.1},
                    "clouds": {"all": 100},
                    "rain": {"3h": 1.6},
                    "pop": 0.72,
                },
                {
                    "dt": start + 24 * 3600,
                    "weather": [{"main": "Clear", "description": "clear sky"}],
                    "main": {"temp": 20.5, "feels_like": 20.0, "temp_min": 18.9, "temp_max": 21.2, "humidity": 57},
                    "wind": {"speed": 2.5, "deg": 180},
                    "clouds": {"all": 4},
                    "pop": 0.0,
                },
                {
                    "dt": start + 30 * 3600,
                    "weather": [{"main": "Clouds", "description": "scattered clouds"}],
                    "main": {"temp": 21.0, "feels_like": 20.6, "temp_min": 20.1, "temp_max": 21.4, "humidity": 52},
                    "wind": {"speed": 3.9, "deg": 200},
                    "clouds": {"all": 36},
                    "pop": 0.1,
                },
            ],
        }

    config = _config(tmp_path)
    config = config.__class__(**{**config.__dict__, "openweather_api_key": "ow-test"})
    service = OpenWeatherService(config, fetch_json=fake_fetch)

    first = service.get_weather(location="Oxford,GB", forecast_days=2)
    second = service.get_weather(location="Oxford,GB", forecast_days=2)

    assert first["ok"]
    assert first["cached"] is False
    assert first["location"]["name"] == "Oxford"
    assert first["forecast"]["days_requested"] == 2
    assert first["forecast"]["days_returned"] == 2
    assert len(first["forecast"]["daily"]) == 2
    assert first["forecast"]["daily"][0]["condition"]["main"] == "Rain"
    assert first["forecast"]["daily"][0]["precipitation_probability_percent"] == 72
    assert first["forecast"]["daily"][1]["weekday"] == "Monday"
    assert len(first["forecast"]["periods"]) == 4
    assert second["cached"] is True
    assert len(calls) == 3
