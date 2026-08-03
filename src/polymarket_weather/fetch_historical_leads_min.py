"""fetch_historical_leads_min.py — archived per-lead forecasts of the daily MINIMUM.

The engine excluded "lowest temperature" markets (~20% of all markets) because every
predictor was a Tmax distribution. This fetches the training data for a proper Tmin
model: the same Previous Runs hourly series as the Tmax leads fetchers, reduced to the
LOCAL-DAY MINIMUM per lead.

Two files per city (mirroring the Tmax layout so train_calibrator can reuse its logic):
  {slug}_historical_leads_min.csv     — best_match, leads 1..7 (fcst_tmin_lead{n})
  {slug}_historical_leads_min_mm.csv  — blend models, leads 1..4
                                        (fcst_tmin_lead{n}_{ecmwf,gfs,icon,gem,mf,jma,aifs})

Run from src/polymarket_weather/:   python fetch_historical_leads_min.py
"""

import logging
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from resolution_anchors import modelled_anchors

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START_DATE = date(2022, 1, 1)
BM_LEADS = range(1, 8)
MM_LEADS = range(1, 5)
MM_MODELS = {
    "ecmwf_ifs025": "ecmwf",
    "gfs_seamless": "gfs",
    "icon_seamless": "icon",
    "gem_seamless": "gem",
    "meteofrance_seamless": "mf",
    "jma_seamless": "jma",
    "ecmwf_aifs025": "aifs",
}
CHUNK_DAYS = 90
REQUEST_TIMEOUT = 180
OUT_DIR = "data/weather"

# Incremental refetch window — dates older than this many days before an existing CSV's
# last date are treated as final (their archived leads long since published) and served
# from the CSV rather than re-fetched, mirroring fetch_historical_leads.py. With no CSV
# it is a full 2022→now refetch. Keeps a cached retrain converging in minutes.
OVERLAP_DAYS = 21


def _load_existing(out_path):
    """Existing CSV (already stored with the final fcst_tmin_* columns) and its last
    date, or (None, None) when absent/unreadable → full refetch."""
    if not os.path.exists(out_path):
        return None, None
    try:
        df = pd.read_csv(out_path)
    except Exception as e:  # a corrupt/partial cache must never block a fresh fetch
        logger.warning(f"could not read {out_path} ({e}); refetching from scratch")
        return None, None
    if "date_local" not in df.columns or df.empty:
        return None, None
    return df, pd.to_datetime(df["date_local"]).max().date()


def _get(params: dict) -> dict | None:
    for attempt in range(4):
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"previous-runs: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def _daily_min(hourly: dict, leads, col_of) -> pd.DataFrame | None:
    times = hourly.get("time", [])
    if not times:
        return None
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    cols = []
    for n in leads:
        for key, col in col_of(n).items():
            if key in hourly:
                df[col] = hourly[key]
                cols.append(col)
    if not cols:
        return None
    df["date_local"] = df["time"].dt.strftime("%Y-%m-%d")
    agg = df.groupby("date_local")[cols].min()
    counts = df.groupby("date_local")["time"].count()
    agg = agg[counts >= 20]
    return agg.reset_index()


def _fetch_series(lat, lon, leads, models: dict | None, out_path):
    """Chunked previous-runs fetch reduced to local-day minima, INCREMENTAL: when an
    existing CSV is present, only dates within OVERLAP_DAYS of its last date onward are
    re-fetched; fresh chunks supersede cached rows for the overlap, and a cached row
    SURVIVES if its refetch chunk failed (existing first + keep='last'), so a partial
    network failure never silently drops good data. With no CSV it is a full refetch."""
    end_all = datetime.now().date() - timedelta(days=1)
    existing, max_date = _load_existing(out_path)
    start = START_DATE if max_date is None else max(
        START_DATE, max_date - timedelta(days=OVERLAP_DAYS))
    if max_date is not None:
        logger.info(f"  incremental — cached through {max_date}, refetching from {start}")
    chunks = []
    hourly_vars = ",".join(f"temperature_2m_previous_day{n}" for n in leads)
    while start <= end_all:
        end = min(start + timedelta(days=CHUNK_DAYS - 1), end_all)
        params = {
            "latitude": lat, "longitude": lon, "hourly": hourly_vars,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto",
        }
        if models:
            params["models"] = ",".join(models)
            col_of = lambda n: {f"temperature_2m_previous_day{n}_{api}": f"fcst_tmin_lead{n}_{short}"
                                for api, short in models.items()}
        else:
            col_of = lambda n: {f"temperature_2m_previous_day{n}": f"fcst_tmin_lead{n}"}
        data = _get(params)
        if data:
            chunk = _daily_min(data.get("hourly", {}), leads, col_of)
            if chunk is not None:
                chunks.append(chunk)
        start = end + timedelta(days=1)
        time.sleep(1.0)
    frames = ([existing] if existing is not None else []) + chunks
    if not frames:
        return None
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates("date_local", keep="last").sort_values("date_local"))


def fetch_historical_leads_min():
    os.makedirs(OUT_DIR, exist_ok=True)
    for city, anchor in modelled_anchors().items():
        slug = city.replace(" ", "_").lower()
        lat, lon = anchor["forecast_lat"], anchor["forecast_lon"]

        out_bm = os.path.join(OUT_DIR, f"{slug}_historical_leads_min.csv")
        bm = _fetch_series(lat, lon, BM_LEADS, None, out_bm)
        if bm is not None:
            bm.to_csv(out_bm, index=False)
            logger.info(f"{city}: saved {out_bm} — {len(bm)} dates")
        else:
            logger.error(f"{city}: best_match min fetch failed")

        out_mm = os.path.join(OUT_DIR, f"{slug}_historical_leads_min_mm.csv")
        mm = _fetch_series(lat, lon, MM_LEADS, MM_MODELS, out_mm)
        if mm is not None:
            mm.to_csv(out_mm, index=False)
            logger.info(f"{city}: saved {out_mm} — {len(mm)} dates")
        else:
            logger.error(f"{city}: multi-model min fetch failed")


if __name__ == "__main__":
    fetch_historical_leads_min()
