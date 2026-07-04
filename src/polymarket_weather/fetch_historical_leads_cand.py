"""fetch_historical_leads_cand.py — candidate-model archived forecasts at leads 1..4.

Same shape as fetch_historical_leads_mm.py but for the EXPANSION candidates evaluated by
the blend sweep (KMA is not archived; UKMO is not archived — both tested empty):

    aifs  — ECMWF AIFS 0.25° (AI model; archived from mid-2024)
    gem   — Canadian GEM global
    mf    — Météo-France seamless (ARPEGE/AROME)
    cma   — CMA GRAPES global
    bom   — BOM ACCESS-G

Columns use the same naming as the mm file (fcst_tmax_lead{n}_{short}) so the trainer can
merge all leads files generically. A model only enters the SERVING blend after it improves
the per-city temporal holdout (see train_calibrator.py MM_MODELS_BY_CITY).

Run from src/polymarket_weather/:   python fetch_historical_leads_cand.py
"""

import logging
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from resolution_anchors import RESOLUTION_ANCHORS

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START_DATE = date(2022, 1, 1)
LEADS = range(1, 5)
MODELS = {
    "ecmwf_aifs025": "aifs",
    "gem_seamless": "gem",
    "meteofrance_seamless": "mf",
    "cma_grapes_global": "cma",
    "bom_access_global": "bom",
}
CHUNK_DAYS = 90
REQUEST_TIMEOUT = 180
OUT_DIR = "data/weather"


def _fetch_chunk(lat: float, lon: float, start: date, end: date) -> pd.DataFrame | None:
    hourly_vars = ",".join(f"temperature_2m_previous_day{n}" for n in LEADS)
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": hourly_vars,
        "models": ",".join(MODELS),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": "auto",
    }
    for attempt in range(4):
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            logger.warning(f"chunk {start}..{end} attempt {attempt}: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    cols = []
    for n in LEADS:
        for api_model, short in MODELS.items():
            key = f"temperature_2m_previous_day{n}_{api_model}"
            col = f"fcst_tmax_lead{n}_{short}"
            if key in hourly:
                df[col] = hourly[key]
                cols.append(col)
    if not cols:
        return None
    df["date_local"] = df["time"].dt.strftime("%Y-%m-%d")
    agg = df.groupby("date_local")[cols].max()
    counts = df.groupby("date_local")["time"].count()
    agg = agg[counts >= 20]
    return agg.reset_index()


def fetch_historical_leads_cand():
    os.makedirs(OUT_DIR, exist_ok=True)
    end_all = datetime.now().date() - timedelta(days=1)

    for city, anchor in RESOLUTION_ANCHORS.items():
        slug = city.replace(" ", "_").lower()
        lat, lon = anchor["forecast_lat"], anchor["forecast_lon"]
        chunks = []
        start = START_DATE
        while start <= end_all:
            end = min(start + timedelta(days=CHUNK_DAYS - 1), end_all)
            chunk = _fetch_chunk(lat, lon, start, end)
            if chunk is not None:
                chunks.append(chunk)
            start = end + timedelta(days=1)
            time.sleep(1.0)

        if not chunks:
            logger.error(f"{city}: nothing fetched — keeping any existing CSV")
            continue

        df = pd.concat(chunks, ignore_index=True).drop_duplicates("date_local")
        df = df.sort_values("date_local")
        out = os.path.join(OUT_DIR, f"{slug}_historical_leads_cand.csv")
        df.to_csv(out, index=False)
        counts = {s: int(df.get(f"fcst_tmax_lead1_{s}", pd.Series(dtype=float)).notna().sum())
                  for s in MODELS.values()}
        logger.info(f"{city}: saved {out} — {len(df)} dates, lead1 coverage {counts}")


if __name__ == "__main__":
    fetch_historical_leads_cand()
