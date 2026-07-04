"""fetch_station_obs.py — hourly station observations for intraday conditioning.

Same-day temperature markets are priced with the day's observations in hand: by early
afternoon the running maximum already caps the possible outcomes. A pure-forecast model
never sees that, which is exactly where the eval showed the market's biggest advantage
(lead-0/1 bets). This fetcher pulls the routine hourly METAR temperatures for each
resolution station (IEM `asos.py`, report_type=3, station-local time) so the predictor
can condition on the running daily max at any snapshot fetch time.

Output: data/weather/{slug}_obs_hourly.csv  (valid_local, temp_c, source)

Hong Kong is skipped: its truth anchor is the HKO Observatory, which has no METAR feed —
conditioning on the airport (VHHH) would mislead near bin thresholds.

Full backfill (2022→now):    python fetch_station_obs.py
Recent top-up (last 3 days): python fetch_station_obs.py --recent
"""

import argparse
import io
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from resolution_anchors import RESOLUTION_ANCHORS
from config import CITIES

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
START_YEAR = 2022
REQUEST_TIMEOUT = 120
OUT_DIR = "data/weather"

# slug -> (IEM station id, IANA tz). HKO deliberately absent (see module docstring).
OBS_STATIONS = {
    "new_york_city": ("LGA", "America/New_York"),
    "chicago": ("ORD", "America/Chicago"),
    "london": ("EGLC", "Europe/London"),
    "seoul": ("RKSI", "Asia/Seoul"),
}


def _fetch_range(station: str, tz: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    params = {
        "station": station, "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": tz, "format": "onlycomma", "missing": "M", "report_type": 3,
    }
    for attempt in range(4):
        try:
            resp = requests.get(IEM_ASOS_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), na_values=["M"])
            if not {"valid", "tmpf"}.issubset(df.columns):
                return None
            df = df.dropna(subset=["tmpf"])
            df["temp_c"] = (df["tmpf"] - 32.0) * 5.0 / 9.0
            df = df.rename(columns={"valid": "valid_local"})
            return df[["valid_local", "temp_c"]]
        except requests.exceptions.RequestException as e:
            logger.warning(f"{station} {start:%Y-%m-%d}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def fetch_station_obs(recent_only: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.now()

    for slug, (station, tz) in OBS_STATIONS.items():
        out = os.path.join(OUT_DIR, f"{slug}_obs_hourly.csv")

        if recent_only:
            ranges = [(now - timedelta(days=3), now + timedelta(days=1))]
        else:
            ranges = [(datetime(y, 1, 1), datetime(y + 1, 1, 1))
                      for y in range(START_YEAR, now.year + 1)]

        chunks = []
        for start, end in ranges:
            df = _fetch_range(station, tz, start, min(end, now + timedelta(days=1)))
            if df is not None and not df.empty:
                chunks.append(df)
            time.sleep(0.5)
        if not chunks:
            logger.error(f"{slug}: nothing fetched — keeping existing CSV")
            continue

        df = pd.concat(chunks, ignore_index=True)
        df["source"] = f"iem_metar:{station}"

        if recent_only and os.path.exists(out):
            old = pd.read_csv(out)
            df = pd.concat([old, df], ignore_index=True)

        df = (df.drop_duplicates("valid_local", keep="last")
                .sort_values("valid_local").reset_index(drop=True))

        if not recent_only and len(df) < 20000:
            logger.error(f"{slug}: only {len(df)} rows — keeping existing CSV")
            continue
        df.to_csv(out, index=False)
        logger.info(f"{slug}: saved {out} — {len(df)} rows, latest {df['valid_local'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true", help="top-up: last 3 days only")
    args = ap.parse_args()
    fetch_station_obs(recent_only=args.recent)
