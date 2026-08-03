"""fetch_historical_truth.py — station truth from RESOLUTION-FAITHFUL sources.

Markets resolve off Wunderground airport history pages (METAR/ASOS-derived) for
KORD/KLGA/EGLC/RKSI and off the HKO climate page for Hong Kong — see
`resolution_anchors.py`. This fetcher reads the closest authoritative feed for each:

  * KLGA / KORD  — NWS CLI daily climate reports via the IEM `cli.py` JSON service
                   (the official °F high/low the market's whole-°F bins settle on;
                   near-real-time, coverage back to 2015).
  * EGLC / RKSI  — IEM METAR daily summaries (`daily.py`, GB__ASOS / KR__ASOS).
                   METARs report whole °C, matching the markets' whole-°C unit.
  * HKO          — HKO official open-data API (CLMMAXT/CLMMINT, 0.1 °C — the exact
                   series the resolution page publishes; lags ~1 month).

Replaces the previous Meteostat-based fetcher. Meteostat was audited on 2026-07-03
and found corrupted for recent weeks (KLGA 2026-06-05/06 off by ~9 °C vs the NWS CLI
report, KORD June days off by ~4 °C — still wrong on refetch a month later) and its
Hong Kong station 45007 disagrees with HKO by >1.5 °C on ~66 days/yr. Historical US
data (2024) matched CLI to ±0.3 °C, so only the source is swapped; the CSV schema is
unchanged apart from a new `source` audit column.

A city's CSV is only overwritten when the new fetch passes basic sanity checks
(row count, plausible values), so a source outage can never wipe good truth.

Run from src/polymarket_weather/:   python fetch_historical_truth.py
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_YEAR = 2015
MIN_EXPECTED_ROWS = 3000
REQUEST_TIMEOUT = 60
OUT_DIR = "data/weather"

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
HKO_URL = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"

# city slug -> (adapter, adapter args). Stations follow resolution_anchors.py.
SOURCES = {
    "new_york_city": ("cli", {"station": "KLGA"}),
    "chicago": ("cli", {"station": "KORD"}),
    "london": ("iem_daily", {"network": "GB__ASOS", "station": "EGLC"}),
    "seoul": ("iem_daily", {"network": "KR__ASOS", "station": "RKSI"}),
    "hong_kong": ("hko", {"station": "HKO"}),
    # ── capture tier (spec 2026-08-03) — KALSHI's ruler, the NWS Climatological Report.
    # Polymarket's ruler for these same cities is Wunderground, reconstructed from the hourly
    # METARs fetched by fetch_station_obs. BOTH are archived; NEITHER is converted at write
    # time — the CLI<->WU transfer function is a later spec.
    "los_angeles":   ("cli", {"station": "KLAX"}),
    "austin":        ("cli", {"station": "KAUS"}),
    "atlanta":       ("cli", {"station": "KATL"}),
    "houston":       ("cli", {"station": "KHOU"}),
    "miami":         ("cli", {"station": "KMIA"}),
    "seattle":       ("cli", {"station": "KSEA"}),
    "san_francisco": ("cli", {"station": "KSFO"}),
}


def _f_to_c(f):
    return (float(f) - 32.0) * 5.0 / 9.0


def _fetch_cli(station: str) -> pd.DataFrame:
    """NWS CLI daily reports via IEM, one request per year. Values are whole °F."""
    rows = []
    for year in range(START_YEAR, datetime.now().year + 1):
        resp = requests.get(IEM_CLI_URL, params={"station": station, "year": year},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            hi, lo = r.get("high"), r.get("low")
            rows.append({
                "date_local": r["valid"],
                "temp_max_c": _f_to_c(hi) if isinstance(hi, (int, float)) else None,
                "temp_min_c": _f_to_c(lo) if isinstance(lo, (int, float)) else None,
            })
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["source"] = f"nws_cli:{station}"
    return df


def _fetch_iem_daily(network: str, station: str) -> pd.DataFrame:
    """IEM METAR daily summaries, full range in one request. Values are °F
    converted from the station's whole-°C METAR reports."""
    now = datetime.now()
    params = {
        "network": network, "stations": station,
        "var": "max_temp_f,min_temp_f", "format": "csv",
        "year1": START_YEAR, "month1": 1, "day1": 1,
        "year2": now.year, "month2": now.month, "day2": now.day,
    }
    resp = requests.get(IEM_DAILY_URL, params=params, timeout=REQUEST_TIMEOUT * 3)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=["M", "None", ""])
    df = df.rename(columns={"day": "date_local"})
    df["temp_max_c"] = df["max_temp_f"].map(lambda v: _f_to_c(v) if pd.notna(v) else None)
    df["temp_min_c"] = df["min_temp_f"].map(lambda v: _f_to_c(v) if pd.notna(v) else None)
    df["source"] = f"iem_metar:{station}"
    return df[["date_local", "temp_max_c", "temp_min_c", "source"]]


def _fetch_hko(station: str) -> pd.DataFrame:
    """HKO open-data daily extremes (0.1 °C), one request per year per element."""
    frames = {}
    for dtype, col in (("CLMMAXT", "temp_max_c"), ("CLMMINT", "temp_min_c")):
        rows = []
        for year in range(START_YEAR, datetime.now().year + 1):
            resp = requests.get(HKO_URL, params={
                "dataType": dtype, "year": year, "rformat": "json", "station": station,
            }, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            for y, m, d, v, completeness in resp.json().get("data", []):
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue   # '***' = not yet published / incomplete
                rows.append({
                    "date_local": f"{int(y):04d}-{int(m):02d}-{int(d):02d}",
                    col: val,
                })
            time.sleep(0.2)
        frames[col] = pd.DataFrame(rows)
    df = pd.merge(frames["temp_max_c"], frames["temp_min_c"], on="date_local", how="outer")
    df["source"] = f"hko:{station}"
    return df


def _sanitize_truth(df: "pd.DataFrame", min_expected_rows: int):
    """Clean a fetched truth frame and gate it before it may overwrite the existing CSV.

    F2: keep a day if EITHER temp_max_c OR temp_min_c is present — dropping on temp_max_c
    alone silently lost valid Tmin truth (worst for Hong Kong, whose max/min come from two
    separately-fetched, outer-merged series). Range-check both series INDEPENDENTLY on
    their non-null values (the old max-only gate let a corrupt min through, and would now
    spuriously reject legitimately max-less days because NaN.between(...) is False).

    Returns the cleaned frame, or None if it fails the sanity gate (short/implausible),
    in which case the caller keeps the existing CSV.
    """
    df = df.dropna(subset=["temp_max_c", "temp_min_c"], how="all").copy()
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date_local").drop_duplicates("date_local", keep="last")
    df = df[["date_local", "temp_max_c", "temp_min_c", "source"]]
    if len(df) < min_expected_rows:
        return None
    max_vals = df["temp_max_c"].dropna()
    min_vals = df["temp_min_c"].dropna()
    if not max_vals.between(-60, 60).all() or not min_vals.between(-60, 60).all():
        return None
    return df


def fetch_historical_truth():
    os.makedirs(OUT_DIR, exist_ok=True)

    for slug, (kind, kw) in SOURCES.items():
        logger.info(f"Fetching truth for {slug} via {kind} {kw}")
        try:
            if kind == "cli":
                df = _fetch_cli(**kw)
            elif kind == "iem_daily":
                df = _fetch_iem_daily(**kw)
            else:
                df = _fetch_hko(**kw)
        except Exception as e:
            logger.error(f"{slug}: fetch failed ({e}) — keeping the existing CSV")
            continue

        # Clean + sanity-gate (F2: keep max-OR-min days; validate both series). Never
        # overwrite good truth with a short or implausible fetch.
        df = _sanitize_truth(df, MIN_EXPECTED_ROWS)
        if df is None:
            logger.error(f"{slug}: fetch failed sanity (too short or implausible) — keeping existing CSV")
            continue

        out = os.path.join(OUT_DIR, f"{slug}_historical_actuals.csv")
        df.to_csv(out, index=False)
        latest = df["date_local"].max()
        lag = (datetime.now() - datetime.strptime(latest, "%Y-%m-%d")).days
        logger.info(f"Saved {out}: {len(df)} rows, latest obs {latest} ({lag}d behind today)")


if __name__ == "__main__":
    fetch_historical_truth()
