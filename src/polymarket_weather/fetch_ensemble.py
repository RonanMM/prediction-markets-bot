"""
fetch_ensemble.py — Open-Meteo ensemble forecast fetcher.

Fetches the ICON-Seamless ensemble (≈40 members globally) and computes
per-day temperature statistics: mean, std, percentiles.

These replace the hardcoded NWP sigma lookup table in the analysis, giving
us actual model-spread uncertainty rather than a fixed horizon lookup.
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import requests

from config import (
    CITIES,
    ENSEMBLE_MODEL,
    OPEN_METEO_ENSEMBLE_BASE,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)

ENSEMBLE_FORECAST_DAYS = 16


# ── HTTP helper ───────────────────────────────────────────────────────────────

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
                logger.warning("Ensemble rate-limit. Waiting %.1fs …", wait)
                time.sleep(wait)
            else:
                logger.error("HTTP %s from ensemble API: %s", status, exc)
                return None
        except requests.exceptions.RequestException as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Request error (%s). Retry in %.1fs …", exc, wait)
            time.sleep(wait)
    logger.error("All %d attempts failed for ensemble API", RETRY_ATTEMPTS)
    return None


# ── Timezone helper ───────────────────────────────────────────────────────────

def _local_date_to_utc(date_str: str, tz) -> str:
    from zoneinfo import ZoneInfo
    local_dt = datetime.fromisoformat(date_str).replace(tzinfo=tz)
    return local_dt.astimezone(timezone.utc).isoformat()


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ensemble(city: str, past_days: int = 0) -> dict:
    """
    Fetch ICON-Seamless ensemble forecast for *city*.

    Returns dict with:
      city, fetched_at_utc,
      daily: list[dict] — one record per forecast day with ensemble stats:
        ens_mean, ens_std, ens_p10, ens_p25, ens_median, ens_p75, ens_p90,
        ens_spread (p90-p10), n_members, and the Tmin counterparts (ens_min_mean,
        ens_min_std).

    *past_days* (0-92) also pulls that many recent past dates — used by the schema-repair
    backfill to re-populate the ens_min_* columns the append-freeze bug dropped on disk.

    Falls back to empty dict on any API error.
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
        "daily":         "temperature_2m_max,temperature_2m_min",
        "models":        ENSEMBLE_MODEL,
        "forecast_days": ENSEMBLE_FORECAST_DAYS,
        "timezone":      "auto",
    }
    if past_days:
        params["past_days"] = min(int(past_days), 92)

    logger.info("Fetching ensemble for %s (model=%s)", city, ENSEMBLE_MODEL)
    raw = _get(OPEN_METEO_ENSEMBLE_BASE, params)
    if not raw:
        return {}

    fetched_at = datetime.now(timezone.utc).isoformat()

    daily_raw = raw.get("daily", {})
    dates = daily_raw.get("time", [])
    if not dates:
        logger.warning("No daily data in ensemble response for %s", city)
        return {}

    # Collect member column names.
    # Single model: "temperature_2m_max_member01"
    # Multi-model: "temperature_2m_max_gfs_seamless_member01"
    member_keys = sorted(
        k for k in daily_raw if k.startswith("temperature_2m_max") and "_member" in k
    )
    min_member_keys = sorted(
        k for k in daily_raw if k.startswith("temperature_2m_min") and "_member" in k
    )
    if not member_keys:
        logger.warning("No ensemble member columns found for %s (got keys: %s)",
                       city, list(daily_raw.keys())[:10])
        return {}

    logger.info("  %s: %d ensemble members, %d days", city, len(member_keys), len(dates))

    records = []
    for i, date_str in enumerate(dates):
        vals = []
        for mk in member_keys:
            col = daily_raw[mk]
            v   = col[i] if i < len(col) else None
            if v is not None:
                vals.append(float(v))

        if len(vals) < 3:
            continue   # not enough members to compute meaningful statistics

        arr = np.array(vals)
        utc_iso = _local_date_to_utc(date_str, tz)

        rec = {
            "city":           city,
            "fetched_at_utc": fetched_at,
            "date_local":     date_str,
            "date_utc":       utc_iso,
            "ens_mean":       round(float(np.mean(arr)), 3),
            "ens_std":        round(float(np.std(arr, ddof=1)), 3),
            "ens_p10":        round(float(np.percentile(arr, 10)), 3),
            "ens_p25":        round(float(np.percentile(arr, 25)), 3),
            "ens_median":     round(float(np.median(arr)), 3),
            "ens_p75":        round(float(np.percentile(arr, 75)), 3),
            "ens_p90":        round(float(np.percentile(arr, 90)), 3),
            "ens_spread":     round(float(np.percentile(arr, 90) - np.percentile(arr, 10)), 3),
            "n_members":      len(vals),
        }

        # Daily-MIN member statistics (Tmin markets).
        min_vals = []
        for mk in min_member_keys:
            col = daily_raw[mk]
            v   = col[i] if i < len(col) else None
            if v is not None:
                min_vals.append(float(v))
        if len(min_vals) >= 3:
            marr = np.array(min_vals)
            rec["ens_min_mean"] = round(float(np.mean(marr)), 3)
            rec["ens_min_std"]  = round(float(np.std(marr, ddof=1)), 3)
        records.append(rec)

    return {
        "city":           city,
        "fetched_at_utc": fetched_at,
        "daily":          records,
    }



