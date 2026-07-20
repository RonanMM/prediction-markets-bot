"""fetch_historical_leads.py — archived real forecasts at explicit lead times.

Pulls hourly `temperature_2m_previous_dayN` (N=1..7) from the Open-Meteo Previous Runs
API at each city's FORECAST anchor (see resolution_anchors.py) back to 2022, and reduces
them to local-day maxima. Output: data/weather/{slug}_historical_leads.csv with one row
per date and one column per lead (fcst_tmax_lead1..lead7, °C).

Why: the EMOS calibrator was previously trained on the ERA5 *reanalysis* archive —
same-day analysis values, not forecasts. Serving that calibration on 1–3-day-ahead
forecasts understates both bias and (badly) spread. Training on real archived forecasts
at the leads the engine actually bets removes that train/serve skew, per lead.

The Previous Runs API serves the same `best_match` model chain as the live forecast
endpoint the engine uses (fetch_weather.py), so train and serve see the same model.

Run from src/polymarket_weather/:   python fetch_historical_leads.py
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
LEADS = range(1, 8)
CHUNK_DAYS = 180
REQUEST_TIMEOUT = 120
OUT_DIR = "data/weather"


def _fetch_chunk(lat: float, lon: float, start: date, end: date) -> pd.DataFrame | None:
    hourly_vars = ",".join(f"temperature_2m_previous_day{n}" for n in LEADS)
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": hourly_vars,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": "auto",
    }
    for attempt in range(4):
        try:
            resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
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
    for n in LEADS:
        df[f"lead{n}"] = hourly.get(f"temperature_2m_previous_day{n}")
    df["date_local"] = df["time"].dt.strftime("%Y-%m-%d")
    # Local-day max per lead — same reduction the daily API applies to hourly temps.
    agg = df.groupby("date_local")[[f"lead{n}" for n in LEADS]].max()
    # A local day needs a full-ish hourly record for a valid max.
    counts = df.groupby("date_local")["time"].count()
    agg = agg[counts >= 20]
    return agg.reset_index()


# Incremental refetch window. When an existing CSV is present (CI cache or a local
# run), dates older than this many days before its last date are treated as final —
# their archived leads long since published — and served from the CSV rather than
# re-fetched. Generous on purpose: date D's last-arriving lead is lead-1, issued D-1,
# and the previous-runs archive publishes within ~1-2 days, so every lead for D is
# final by ~D+2; 21 days is well past that. With no CSV, behaviour is a full refetch,
# identical to before. See _load_existing_raw / the merge below.
OVERLAP_DAYS = 21


def _load_existing_raw(out_path, rename_map):
    """Existing CSV with its stored (renamed) columns mapped BACK to the raw per-chunk
    names, so it concatenates cleanly with freshly-fetched chunks. Returns
    (df_raw, max_date), or (None, None) when absent/unreadable → full refetch."""
    if not os.path.exists(out_path):
        return None, None
    try:
        df = pd.read_csv(out_path)
    except Exception as e:  # corrupt/partial cache must never block a fresh fetch
        logger.warning(f"could not read {out_path} ({e}); refetching from scratch")
        return None, None
    if "date_local" not in df.columns or df.empty:
        return None, None
    df = df.rename(columns={v: k for k, v in rename_map.items()})
    max_date = pd.to_datetime(df["date_local"]).max().date()
    return df, max_date


def fetch_historical_leads():
    os.makedirs(OUT_DIR, exist_ok=True)
    end_all = datetime.now().date() - timedelta(days=1)
    rename_map = {f"lead{n}": f"fcst_tmax_lead{n}" for n in LEADS}

    for city, anchor in RESOLUTION_ANCHORS.items():
        slug = city.replace(" ", "_").lower()
        lat, lon = anchor["forecast_lat"], anchor["forecast_lon"]
        out = os.path.join(OUT_DIR, f"{slug}_historical_leads.csv")

        existing, max_date = _load_existing_raw(out, rename_map)
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
            else:
                logger.warning(f"{city}: no data for {start}..{end}")
            start = end + timedelta(days=1)
            time.sleep(0.5)

        # Existing rows first, fresh chunks after; keep='last' so a successful refetch
        # supersedes the cached row for overlap dates, while a cached row SURVIVES when
        # its refetch chunk failed — a partial network failure must never silently drop
        # good data (the pre-incremental code kept the whole CSV on total failure; this
        # preserves that guarantee per-date).
        frames = ([existing] if existing is not None else []) + chunks
        if not frames:
            logger.error(f"{city}: nothing fetched and no cache — skipping")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates("date_local", keep="last").sort_values("date_local")
        df = df.rename(columns=rename_map)
        n_valid = df["fcst_tmax_lead1"].notna().sum()
        df.to_csv(out, index=False)
        logger.info(f"{city}: saved {out} — {len(df)} dates, {n_valid} with lead1")


if __name__ == "__main__":
    fetch_historical_leads()
