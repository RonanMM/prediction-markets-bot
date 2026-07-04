"""fetch_nbm.py — NWS National Blend of Models (NBM) station guidance for KLGA/KORD.

NBM is the NWS's station-calibrated blend — the closest public analogue to what
sophisticated market participants price US temperature markets from. IEM archives the
NBS bulletins (4 runtimes/day: 01/07/13/19Z) back to ~2020 with per-ftime hourly `tmp`
and 12-h `txn` max/min guidance.

Output: data/weather/{slug}_nbm.csv — one row per (runtime, target local day):
    runtime_utc, avail_utc, date_local, lead,
    nbm_tmax_txn_c (official 12-h max guidance — evening-ftime txn),
    nbm_tmax_tmp_c (max of 3-hourly tmp over the local day; fallback parse)

`avail_utc` = runtime + 1 h (conservative publication delay) — the AS-OF join key, so
backtests only ever see runs that were actually available at each snapshot's fetch time.

Full backfill (2022→now):    python fetch_nbm.py
Recent top-up (last 4 days): python fetch_nbm.py --recent
"""

import argparse
import io
import logging
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
START = datetime(2022, 1, 1)
REQUEST_TIMEOUT = 120
OUT_DIR = "data/weather"
AVAIL_DELAY_H = 1     # NBM bulletins publish ~30-60 min after runtime

NBM_STATIONS = {
    "new_york_city": ("KLGA", ZoneInfo("America/New_York")),
    "chicago": ("KORD", ZoneInfo("America/Chicago")),
}


def _f_to_c(f):
    return (float(f) - 32.0) * 5.0 / 9.0


def _fetch_window(station: str, sts: datetime, ets: datetime) -> pd.DataFrame | None:
    params = {
        "station": station, "model": "NBS",
        "sts": sts.strftime("%Y-%m-%dT%H:%MZ"), "ets": ets.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "csv",
    }
    for attempt in range(4):
        try:
            resp = requests.get(IEM_MOS_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), na_values=["M", ""])
            if not {"runtime", "ftime", "tmp"}.issubset(df.columns):
                return None
            return df
        except requests.exceptions.RequestException as e:
            logger.warning(f"{station} {sts:%Y-%m}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def _parse_runs(df: pd.DataFrame, tz: ZoneInfo) -> pd.DataFrame:
    """Reduce raw bulletin rows to one row per (runtime, target local day)."""
    df = df.copy()
    df["runtime"] = pd.to_datetime(df["runtime"], utc=True)
    df["ftime"] = pd.to_datetime(df["ftime"], utc=True)
    df["ftime_local"] = df["ftime"].dt.tz_convert(tz)
    df["date_local"] = df["ftime_local"].dt.strftime("%Y-%m-%d")
    df["hour_local"] = df["ftime_local"].dt.hour

    out = []
    for runtime, run in df.groupby("runtime"):
        run_date_local = runtime.tz_convert(tz).strftime("%Y-%m-%d")
        # Official 12-h max guidance: txn on an evening ftime belongs to that local day.
        txn = run[(run["txn"].notna()) & (run["hour_local"] >= 16)]
        txn_by_day = dict(zip(txn["date_local"], txn["txn"]))
        # Fallback parse: max of 3-hourly tmp across the local day, if the daytime
        # (06-23 local) is reasonably sampled.
        day_ok = run[(run["hour_local"] >= 6) & (run["hour_local"] <= 23)].dropna(subset=["tmp"])
        tmp_by_day = day_ok.groupby("date_local")["tmp"].agg(["max", "count"])
        for date_local in sorted(set(txn_by_day) | set(tmp_by_day.index)):
            lead = (pd.Timestamp(date_local) - pd.Timestamp(run_date_local)).days
            if lead < 0 or lead > 3:
                continue
            tmp_max = tmp_by_day["max"].get(date_local)
            tmp_n = tmp_by_day["count"].get(date_local, 0)
            out.append({
                "runtime_utc": runtime.isoformat(),
                "avail_utc": (runtime + pd.Timedelta(hours=AVAIL_DELAY_H)).isoformat(),
                "date_local": date_local,
                "lead": lead,
                "nbm_tmax_txn_c": _f_to_c(txn_by_day[date_local]) if date_local in txn_by_day else None,
                "nbm_tmax_tmp_c": _f_to_c(tmp_max) if tmp_max is not None and tmp_n >= 4 else None,
            })
    return pd.DataFrame(out)


def fetch_nbm(recent_only: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.utcnow()

    for slug, (station, tz) in NBM_STATIONS.items():
        out_path = os.path.join(OUT_DIR, f"{slug}_nbm.csv")

        if recent_only:
            windows = [(now - timedelta(days=4), now)]
        else:
            windows = []
            cur = START
            while cur < now:
                nxt = (cur.replace(day=1) + timedelta(days=40)).replace(day=1)
                windows.append((cur, min(nxt, now)))
                cur = nxt

        chunks = []
        for sts, ets in windows:
            raw = _fetch_window(station, sts, ets)
            if raw is not None and not raw.empty:
                chunks.append(_parse_runs(raw, tz))
            time.sleep(0.3)
        if not chunks:
            logger.error(f"{slug}: nothing fetched — keeping existing CSV")
            continue

        df = pd.concat(chunks, ignore_index=True)
        if recent_only and os.path.exists(out_path):
            old = pd.read_csv(out_path)
            df = pd.concat([old, df], ignore_index=True)
        df = (df.drop_duplicates(["runtime_utc", "date_local"], keep="last")
                .sort_values(["runtime_utc", "date_local"]).reset_index(drop=True))

        if not recent_only and len(df) < 5000:
            logger.error(f"{slug}: only {len(df)} rows — keeping existing CSV")
            continue
        df.to_csv(out_path, index=False)
        logger.info(f"{slug}: saved {out_path} — {len(df)} rows, "
                    f"latest runtime {df['runtime_utc'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true", help="top-up: last 4 days only")
    args = ap.parse_args()
    fetch_nbm(recent_only=args.recent)
