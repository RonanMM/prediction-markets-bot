"""fetch_historical_leads_mm.py — multi-model archived forecasts at leads 1..4.

Same as fetch_historical_leads.py but pulls the three model chains behind the live
ensemble (ECMWF IFS, GFS seamless, ICON seamless) so a multi-model mean can be trained —
the training-time proxy for the live 122-member ensemble mean, which the live-data audit
showed is the strongest mean estimator at every lead ≥ 1.

Output: data/weather/{slug}_historical_leads_mm.csv with columns
fcst_tmax_lead{n}_{model} for n in 1..4, model in {ecmwf, gfs, icon}.

Run from src/polymarket_weather/:   python fetch_historical_leads_mm.py
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
MODELS = {"ecmwf_ifs025": "ecmwf", "gfs_seamless": "gfs", "icon_seamless": "icon"}
CHUNK_DAYS = 120
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
            col = f"lead{n}_{short}"
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


# Incremental refetch window — see fetch_historical_leads.py for the full rationale.
# When an existing CSV is present, dates older than OVERLAP_DAYS before its last date
# are served from it rather than re-fetched; with no CSV this is a full refetch,
# identical to before.
OVERLAP_DAYS = 21


def _load_existing_raw(out_path, rename_map):
    """Existing CSV, its stored (renamed) columns mapped BACK to raw per-chunk names so
    it concatenates with fresh chunks. (df_raw, max_date) or (None, None) → full refetch."""
    if not os.path.exists(out_path):
        return None, None
    try:
        df = pd.read_csv(out_path)
    except Exception as e:
        logger.warning(f"could not read {out_path} ({e}); refetching from scratch")
        return None, None
    if "date_local" not in df.columns or df.empty:
        return None, None
    df = df.rename(columns={v: k for k, v in rename_map.items()})
    max_date = pd.to_datetime(df["date_local"]).max().date()
    return df, max_date


def fetch_historical_leads_mm():
    os.makedirs(OUT_DIR, exist_ok=True)
    end_all = datetime.now().date() - timedelta(days=1)
    rename_map = {f"lead{n}_{s}": f"fcst_tmax_lead{n}_{s}"
                  for n in LEADS for s in MODELS.values()}

    for city, anchor in modelled_anchors().items():
        slug = city.replace(" ", "_").lower()
        lat, lon = anchor["forecast_lat"], anchor["forecast_lon"]
        out = os.path.join(OUT_DIR, f"{slug}_historical_leads_mm.csv")

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
            time.sleep(1.0)

        # keep='last': fresh refetch wins on overlap dates, cached row survives where a
        # chunk failed — never silently drop good data. See fetch_historical_leads.py.
        frames = ([existing] if existing is not None else []) + chunks
        if not frames:
            logger.error(f"{city}: nothing fetched and no cache — skipping")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates("date_local", keep="last").sort_values("date_local")
        df = df.rename(columns=rename_map)
        df.to_csv(out, index=False)
        n_full = df[[f"fcst_tmax_lead1_{s}" for s in MODELS.values()]].notna().all(axis=1).sum()
        logger.info(f"{city}: saved {out} — {len(df)} dates, {n_full} with all 3 models at lead1")


if __name__ == "__main__":
    fetch_historical_leads_mm()
