from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from .config import Config

LOGGER = logging.getLogger(__name__)

JsonFetcher = Callable[[str, float], Any]


@dataclass
class WeatherCacheEntry:
    created_at: float
    payload: dict[str, Any]


class OpenWeatherService:
    def __init__(self, config: Config, fetch_json: JsonFetcher | None = None):
        self.config = config
        self._fetch_json = fetch_json or fetch_json_url
        self._cache: dict[str, WeatherCacheEntry] = {}
        self._lock = threading.RLock()

    def get_current_weather(self, location: str = "", units: str = "metric", no_cache: bool = False) -> dict[str, Any]:
        if not self.config.openweather_api_key:
            return {"ok": False, "error": "OPENWEATHER_API_KEY is not configured"}

        clean_location = location.strip() or self._default_location()
        clean_units = units.strip().lower() or "metric"
        if clean_units not in {"metric", "imperial", "standard"}:
            clean_units = "metric"

        cache_key = f"{clean_location.lower()}|{clean_units}"
        if not no_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        try:
            geo = self._geocode(clean_location)
            weather = self._weather(geo["lat"], geo["lon"], clean_units)
        except Exception as exc:
            LOGGER.exception("OpenWeather request failed")
            return {"ok": False, "error": str(exc), "location": clean_location}

        payload = format_weather_payload(
            location_query=clean_location,
            units=clean_units,
            geocode=geo,
            weather=weather,
            cached=False,
            cache_age_seconds=0,
        )
        with self._lock:
            self._cache[cache_key] = WeatherCacheEntry(created_at=time.time(), payload=payload)
        return payload

    def _get_cached(self, cache_key: str) -> dict[str, Any] | None:
        ttl = max(0.0, self.config.weather_cache_seconds)
        if ttl <= 0:
            return None
        now = time.time()
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return None
            age = now - entry.created_at
            if age > ttl:
                self._cache.pop(cache_key, None)
                return None
            payload = dict(entry.payload)
        payload["cached"] = True
        payload["cache_age_seconds"] = round(age, 1)
        return payload

    def _default_location(self) -> str:
        if self.config.user_country:
            return f"{self.config.user_city},{self.config.user_country}"
        return self.config.user_city

    def _geocode(self, location: str) -> dict[str, Any]:
        params = {
            "q": location,
            "limit": "1",
            "appid": self.config.openweather_api_key,
        }
        url = "https://api.openweathermap.org/geo/1.0/direct?" + urlencode(params)
        results = self._fetch_json(url, self.config.openweather_timeout_seconds)
        if not isinstance(results, list) or not results:
            raise RuntimeError(f"OpenWeather could not geocode location: {location}")
        first = results[0]
        if not isinstance(first, dict) or "lat" not in first or "lon" not in first:
            raise RuntimeError(f"OpenWeather returned an invalid geocoding result for: {location}")
        return first

    def _weather(self, lat: float, lon: float, units: str) -> dict[str, Any]:
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "appid": self.config.openweather_api_key,
            "units": units,
        }
        url = "https://api.openweathermap.org/data/2.5/weather?" + urlencode(params)
        result = self._fetch_json(url, self.config.openweather_timeout_seconds)
        if not isinstance(result, dict):
            raise RuntimeError("OpenWeather returned an invalid weather response")
        return result


def fetch_json_url(url: str, timeout: float) -> Any:
    request = Request(url, headers={"User-Agent": "voice-ai-bot/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=timeout, context=context) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def format_weather_payload(
    location_query: str,
    units: str,
    geocode: dict[str, Any],
    weather: dict[str, Any],
    cached: bool,
    cache_age_seconds: float,
) -> dict[str, Any]:
    main = weather.get("main") if isinstance(weather.get("main"), dict) else {}
    wind = weather.get("wind") if isinstance(weather.get("wind"), dict) else {}
    clouds = weather.get("clouds") if isinstance(weather.get("clouds"), dict) else {}
    weather_items = weather.get("weather") if isinstance(weather.get("weather"), list) else []
    condition = weather_items[0] if weather_items and isinstance(weather_items[0], dict) else {}
    observed_at = weather.get("dt")
    observed_iso = (
        datetime.fromtimestamp(float(observed_at), tz=timezone.utc).isoformat()
        if isinstance(observed_at, (int, float))
        else None
    )
    sys_info = weather.get("sys") if isinstance(weather.get("sys"), dict) else {}
    return {
        "ok": True,
        "source": "OpenWeather",
        "cached": cached,
        "cache_age_seconds": cache_age_seconds,
        "location_query": location_query,
        "location": {
            "name": geocode.get("name") or weather.get("name") or location_query,
            "state": geocode.get("state", ""),
            "country": geocode.get("country") or sys_info.get("country", ""),
            "lat": geocode.get("lat"),
            "lon": geocode.get("lon"),
        },
        "units": units,
        "observed_at": observed_iso,
        "condition": {
            "main": condition.get("main", ""),
            "description": condition.get("description", ""),
        },
        "temperature": {
            "current": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "min": main.get("temp_min"),
            "max": main.get("temp_max"),
        },
        "humidity_percent": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind": {
            "speed": wind.get("speed"),
            "degrees": wind.get("deg"),
            "gust": wind.get("gust"),
        },
        "clouds_percent": clouds.get("all"),
    }
