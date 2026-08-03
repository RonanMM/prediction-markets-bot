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

from resolution_anchors import modelled_anchors

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START_DATE = date(2022, 1, 1)
LEADS = range(1, 5)
MODELS = {
    "ecmwf_aifs025": "aifs",
    "gem_seamless": "gem",
    "meteofrance_seamless": "mf",
    "cma_grapes_global": "cma",
    "bom_access_global": "bom",
    # jma is consumed only by Seoul's blend (MM_MODELS_BY_CITY), but nothing else in the repo
    # produced {slug}_historical_leads_jma.csv, so Seoul's mm_mean could never build. Fetched
    # here for all cities via the same machinery; non-Seoul blends simply don't select it.
    # Model id confirmed from fetch_weather.MULTIMODEL_MODELS (live serving already uses it) and
    # verified to return full previous-runs archive coverage for Seoul (168/168 non-null @ lead1/4).
    "jma_seamless": "jma",
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


# Incremental refetch window — see fetch_historical_leads.py. Columns here are identity
# (_fetch_chunk already emits the stored fcst_tmax_lead{n}_{s} names, no post-rename), so the
# existing CSV concatenates with fresh chunks directly. With no CSV this is a full refetch.
OVERLAP_DAYS = 21


def _load_existing(out_path):
    """Existing CSV + its max date, or (None, None) → full refetch."""
    if not os.path.exists(out_path):
        return None, None
    try:
        df = pd.read_csv(out_path)
    except Exception as e:
        logger.warning(f"could not read {out_path} ({e}); refetching from scratch")
        return None, None
    if "date_local" not in df.columns or df.empty:
        return None, None
    return df, pd.to_datetime(df["date_local"]).max().date()


def fetch_historical_leads_cand():
    os.makedirs(OUT_DIR, exist_ok=True)
    end_all = datetime.now().date() - timedelta(days=1)

    for city, anchor in modelled_anchors().items():
        slug = city.replace(" ", "_").lower()
        lat, lon = anchor["forecast_lat"], anchor["forecast_lon"]
        out = os.path.join(OUT_DIR, f"{slug}_historical_leads_cand.csv")

        existing, max_date = _load_existing(out)
        start = START_DATE if max_date is None else max(
            START_DATE, max_date - timedelta(days=OVERLAP_DAYS))
        if max_date is not None:
            logger.info(f"{city}: incremental — cached through {max_date}, refetching from {start}")

        chunks = []
        while start <= end_all:
            end = min(start + timedelta(days=CHUNK_DAYS - 1), end_all)
            chunk = _fetch_chunk(lat, lon, start, end)
            if chunk is not None:
                chunks.append(chunk)
            start = end + timedelta(days=1)
            time.sleep(1.0)

        # keep='last': fresh refetch wins on overlap dates, cached row survives where a chunk
        # failed — never silently drop good data. See fetch_historical_leads.py.
        frames = ([existing] if existing is not None else []) + chunks
        if not frames:
            logger.error(f"{city}: nothing fetched and no cache — skipping")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates("date_local", keep="last").sort_values("date_local")
        df.to_csv(out, index=False)
        counts = {s: int(df.get(f"fcst_tmax_lead1_{s}", pd.Series(dtype=float)).notna().sum())
                  for s in MODELS.values()}
        logger.info(f"{city}: saved {out} — {len(df)} dates, lead1 coverage {counts}")


if __name__ == "__main__":
    fetch_historical_leads_cand()
