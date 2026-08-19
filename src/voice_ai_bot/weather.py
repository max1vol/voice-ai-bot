from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        return self.get_weather(location=location, units=units, no_cache=no_cache, forecast_days=0)

    def get_weather(
        self,
        location: str = "",
        units: str = "metric",
        no_cache: bool = False,
        forecast_days: int = 0,
    ) -> dict[str, Any]:
        if not self.config.openweather_api_key:
            return {"ok": False, "error": "OPENWEATHER_API_KEY is not configured"}

        clean_location = location.strip() or self._default_location()
        clean_units = units.strip().lower() or "metric"
        if clean_units not in {"metric", "imperial", "standard"}:
            clean_units = "metric"
        clean_forecast_days = max(0, min(5, int(forecast_days)))

        cache_key = f"{clean_location.lower()}|{clean_units}|{clean_forecast_days}"
        if not no_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        try:
            geo = self._geocode(clean_location)
            weather = self._weather(geo["lat"], geo["lon"], clean_units)
            forecast = self._forecast(geo["lat"], geo["lon"], clean_units) if clean_forecast_days else None
        except Exception as exc:
            error = safe_weather_error(exc, self.config.openweather_api_key)
            LOGGER.warning("OpenWeather request failed: %s", error)
            return {"ok": False, "error": error, "location": clean_location}

        payload = format_weather_payload(
            location_query=clean_location,
            units=clean_units,
            geocode=geo,
            weather=weather,
            forecast=forecast,
            forecast_days_requested=clean_forecast_days,
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

    def _forecast(self, lat: float, lon: float, units: str) -> dict[str, Any]:
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "appid": self.config.openweather_api_key,
            "units": units,
        }
        url = "https://api.openweathermap.org/data/2.5/forecast?" + urlencode(params)
        result = self._fetch_json(url, self.config.openweather_timeout_seconds)
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise RuntimeError("OpenWeather returned an invalid forecast response")
        return result


def fetch_json_url(url: str, timeout: float) -> Any:
    request = Request(url, headers={"User-Agent": "voice-ai-bot/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=timeout, context=context) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def safe_weather_error(exc: Exception, api_key: str = "") -> str:
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "<redacted>")
    text = re.sub(r"appid=[^&\\s>'\"]+", "appid=<redacted>", text)
    return f"{exc.__class__.__name__}: {text}"


def format_weather_payload(
    location_query: str,
    units: str,
    geocode: dict[str, Any],
    weather: dict[str, Any],
    forecast: dict[str, Any] | None,
    forecast_days_requested: int,
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
    payload = {
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
        "forecast_days_requested": forecast_days_requested,
    }
    if forecast is not None and forecast_days_requested > 0:
        payload["forecast"] = format_forecast_payload(forecast, forecast_days_requested)
    else:
        payload["forecast"] = {"days_requested": 0, "days_returned": 0, "daily": [], "periods": []}
    return payload


def format_forecast_payload(forecast: dict[str, Any], days_requested: int) -> dict[str, Any]:
    city = forecast.get("city") if isinstance(forecast.get("city"), dict) else {}
    timezone_offset = city.get("timezone")
    if not isinstance(timezone_offset, (int, float)):
        timezone_offset = 0
    forecast_timezone = timezone(timedelta(seconds=int(timezone_offset)))

    grouped: dict[str, list[dict[str, Any]]] = {}
    periods: list[dict[str, Any]] = []
    for item in forecast.get("list", []):
        if not isinstance(item, dict):
            continue
        period = format_forecast_period(item, forecast_timezone)
        if period is None:
            continue
        periods.append(period)
        grouped.setdefault(period["date"], []).append(period)

    ordered_dates = sorted(grouped.keys())[:days_requested]
    filtered_periods = [period for period in periods if period["date"] in ordered_dates]
    daily = [summarize_forecast_day(date_key, grouped[date_key]) for date_key in ordered_dates]
    return {
        "days_requested": days_requested,
        "days_returned": len(daily),
        "timezone_offset_seconds": int(timezone_offset),
        "daily": daily,
        "periods": filtered_periods,
    }


def format_forecast_period(item: dict[str, Any], forecast_timezone: timezone) -> dict[str, Any] | None:
    timestamp = item.get("dt")
    if not isinstance(timestamp, (int, float)):
        return None
    local_dt = datetime.fromtimestamp(float(timestamp), tz=forecast_timezone)
    main = item.get("main") if isinstance(item.get("main"), dict) else {}
    wind = item.get("wind") if isinstance(item.get("wind"), dict) else {}
    clouds = item.get("clouds") if isinstance(item.get("clouds"), dict) else {}
    weather_items = item.get("weather") if isinstance(item.get("weather"), list) else []
    condition = weather_items[0] if weather_items and isinstance(weather_items[0], dict) else {}
    rain = item.get("rain") if isinstance(item.get("rain"), dict) else {}
    snow = item.get("snow") if isinstance(item.get("snow"), dict) else {}
    pop = item.get("pop")
    pop_percent = round(float(pop) * 100) if isinstance(pop, (int, float)) else None
    return {
        "local_time": local_dt.isoformat(),
        "date": local_dt.date().isoformat(),
        "time": local_dt.strftime("%H:%M"),
        "weekday": local_dt.strftime("%A"),
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
        "precipitation_probability_percent": pop_percent,
        "rain_mm": rain.get("3h", 0.0),
        "snow_mm": snow.get("3h", 0.0),
    }


def summarize_forecast_day(date_key: str, periods: list[dict[str, Any]]) -> dict[str, Any]:
    representative = pick_representative_period(periods)
    temp_mins = [
        value
        for period in periods
        for value in [period["temperature"].get("min"), period["temperature"].get("current")]
        if isinstance(value, (int, float))
    ]
    temp_maxes = [
        value
        for period in periods
        for value in [period["temperature"].get("max"), period["temperature"].get("current")]
        if isinstance(value, (int, float))
    ]
    humidity_values = [
        value for period in periods for value in [period.get("humidity_percent")] if isinstance(value, (int, float))
    ]
    cloud_values = [
        value for period in periods for value in [period.get("clouds_percent")] if isinstance(value, (int, float))
    ]
    wind_speeds = [
        value
        for period in periods
        for value in [period["wind"].get("speed")]
        if isinstance(value, (int, float))
    ]
    wind_gusts = [
        value
        for period in periods
        for value in [period["wind"].get("gust")]
        if isinstance(value, (int, float))
    ]
    pops = [
        value
        for period in periods
        for value in [period.get("precipitation_probability_percent")]
        if isinstance(value, (int, float))
    ]
    rain_total = sum(
        float(value)
        for period in periods
        for value in [period.get("rain_mm")]
        if isinstance(value, (int, float))
    )
    snow_total = sum(
        float(value)
        for period in periods
        for value in [period.get("snow_mm")]
        if isinstance(value, (int, float))
    )
    return {
        "date": date_key,
        "weekday": representative.get("weekday", ""),
        "condition": representative.get("condition", {"main": "", "description": ""}),
        "representative_time": representative.get("time"),
        "temperature": {
            "min": min(temp_mins) if temp_mins else None,
            "max": max(temp_maxes) if temp_maxes else None,
        },
        "humidity_percent": {
            "min": min(humidity_values) if humidity_values else None,
            "max": max(humidity_values) if humidity_values else None,
        },
        "wind": {
            "max_speed": max(wind_speeds) if wind_speeds else None,
            "max_gust": max(wind_gusts) if wind_gusts else None,
        },
        "clouds_percent": round(sum(cloud_values) / len(cloud_values), 1) if cloud_values else None,
        "precipitation_probability_percent": max(pops) if pops else None,
        "rain_mm": round(rain_total, 1),
        "snow_mm": round(snow_total, 1),
        "period_count": len(periods),
    }


def pick_representative_period(periods: list[dict[str, Any]]) -> dict[str, Any]:
    def score(period: dict[str, Any]) -> tuple[float, float, int]:
        pop = period.get("precipitation_probability_percent")
        rain = period.get("rain_mm")
        time_text = str(period.get("time") or "12:00")
        try:
            hour = int(time_text.split(":", 1)[0])
        except ValueError:
            hour = 12
        return (
            float(pop) if isinstance(pop, (int, float)) else 0.0,
            float(rain) if isinstance(rain, (int, float)) else 0.0,
            -abs(hour - 12),
        )

    return max(periods, key=score) if periods else {}
