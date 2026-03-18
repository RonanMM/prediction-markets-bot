"""
fetch_weather.py — Fetches forecast data from Open-Meteo for all cities.

Returns:
  • Daily max/min temperature forecast (16 days)
  • Hourly temperature (for finer alignment with market snapshots)
All timestamps are UTC-normalized internally; local time also stored.
"""

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from config import (
    CITIES,
    OPEN_METEO_BASE,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)


# ── HTTP helper ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> dict | None:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 429:
                wait = RETRY_BACKOFF ** (attempt + 2)
                logger.warning("Open-Meteo rate-limit. Waiting %.1fs …", wait)
                time.sleep(wait)
            else:
                logger.error("HTTP %s from Open-Meteo: %s", status, exc)
                return None
        except requests.exceptions.RequestException as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Request error (%s). Retry in %.1fs …", exc, wait)
            time.sleep(wait)
    logger.error("All %d attempts failed for Open-Meteo", RETRY_ATTEMPTS)
    return None


# ── Timezone helpers ─────────────────────────────────────────────────────────

def _local_date_str_to_utc(date_str: str, tz: ZoneInfo) -> str:
    """'2025-03-20' in local tz → UTC ISO string (midnight local)."""
    local_dt = datetime.fromisoformat(date_str).replace(tzinfo=tz)
    return local_dt.astimezone(timezone.utc).isoformat()


def _local_datetime_str_to_utc(dt_str: str, tz: ZoneInfo) -> str:
    """'2025-03-20T14:00' in local tz → UTC ISO string."""
    local_dt = datetime.fromisoformat(dt_str).replace(tzinfo=tz)
    return local_dt.astimezone(timezone.utc).isoformat()


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_forecast(city: str) -> dict:
    """
    Fetch Open-Meteo forecast for *city*.

    Returns a dict with keys:
      - city
      - fetched_at_utc
      - daily:  list of daily records
      - hourly: list of hourly records
    """
    cfg = CITIES.get(city)
    if cfg is None:
        logger.error("Unknown city: %s", city)
        return {}

    lat = cfg["lat"]
    lon = cfg["lon"]
    tz  = cfg["timezone"]

    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "hourly":        "temperature_2m,precipitation_probability",
        "forecast_days": 16,
        "timezone":      "auto",   # Open-Meteo returns times in local tz when "auto"
    }

    logger.info("Fetching Open-Meteo forecast for %s (%.4f, %.4f)", city, lat, lon)
    raw = _get(OPEN_METEO_BASE, params)
    if not raw:
        return {}

    fetched_at = datetime.now(timezone.utc).isoformat()

    # ── Daily records ──────────────────────────────────────────────────────
    daily_time      = raw.get("daily", {}).get("time", [])
    daily_tmax      = raw.get("daily", {}).get("temperature_2m_max", [])
    daily_tmin      = raw.get("daily", {}).get("temperature_2m_min", [])
    daily_precip    = raw.get("daily", {}).get("precipitation_sum", [])

    daily_records = []
    for i, date_str in enumerate(daily_time):
        utc_iso = _local_date_str_to_utc(date_str, tz)
        daily_records.append({
            "city":            city,
            "fetched_at_utc":  fetched_at,
            "date_local":      date_str,
            "date_utc":        utc_iso,
            "temp_max_c":      daily_tmax[i]   if i < len(daily_tmax)   else None,
            "temp_min_c":      daily_tmin[i]   if i < len(daily_tmin)   else None,
            "precip_sum_mm":   daily_precip[i] if i < len(daily_precip) else None,
        })

    # ── Hourly records ─────────────────────────────────────────────────────
    hourly_time  = raw.get("hourly", {}).get("time", [])
    hourly_temp  = raw.get("hourly", {}).get("temperature_2m", [])
    hourly_precip_prob = raw.get("hourly", {}).get("precipitation_probability", [])

    hourly_records = []
    for i, dt_str in enumerate(hourly_time):
        utc_iso = _local_datetime_str_to_utc(dt_str, tz)
        hourly_records.append({
            "city":                city,
            "fetched_at_utc":      fetched_at,
            "datetime_local":      dt_str,
            "datetime_utc":        utc_iso,
            "temp_c":              hourly_temp[i]       if i < len(hourly_temp)       else None,
            "precip_prob_pct":     hourly_precip_prob[i] if i < len(hourly_precip_prob) else None,
        })

    return {
        "city":           city,
        "fetched_at_utc": fetched_at,
        "daily":          daily_records,
        "hourly":         hourly_records,
    }


def fetch_all_cities() -> dict[str, dict]:
    """Fetch forecasts for every city in config. Returns {city: forecast_dict}."""
    results = {}
    for city in CITIES:
        forecast = fetch_forecast(city)
        if forecast:
            results[city] = forecast
        else:
            logger.warning("No forecast data retrieved for %s.", city)
        time.sleep(0.5)   # polite delay between cities
    return results
