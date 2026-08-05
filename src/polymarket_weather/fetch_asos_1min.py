"""fetch_asos_1min.py — 1-minute ASOS temperatures for the US resolution stations.

Routine METARs are hourly, so the running daily max we condition same-day markets on is sampled
24 times a day. The ASOS 1-minute archive samples it 1,440 times. Measured at KLGA over 6 days,
the day's true max sits **+1.20 °C above the hourly max**, and at 14:00 local — while same-day
markets are still trading — the running max we would condition on is **+0.74 °C too low**.

⚠️ WHAT THIS IS AND IS NOT, per venue. Getting this backwards is how W0 happened.
  * Kalshi settles US weather on the NWS CLI, which is computed from these 1-minute sensors. The
    running 1-minute max is therefore a genuine FLOOR on that settlement value, known live rather
    than after the CLI's publication lag.
  * Polymarket settles on wunderground = the hourly-METAR max. The 1-minute max is always >= the
    hourly max, so it is NOT a floor there — it is a leading INDICATOR that a later METAR is
    likely to catch a higher reading.
This module only fetches. It asserts nothing about which target a consumer is predicting.

US ASOS only — the 1-minute archive is a US product, so London (EGLC) and Seoul (RKSI) have no
equivalent and keep the hourly path.

    python fetch_asos_1min.py              # incremental top-up
    python fetch_asos_1min.py --since 2024-01-01
"""
from __future__ import annotations

import argparse
import io
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
OUT_DIR = Path(__file__).resolve().parent / "data" / "weather"

# slug -> (IEM 3-letter id, IANA tz). US ASOS only; mirrors fetch_station_obs.OBS_STATIONS
# minus the non-US stations, which have no 1-minute archive.
ASOS_1MIN = {
    "new_york_city": ("LGA", "America/New_York"),
    "chicago":       ("ORD", "America/Chicago"),
    "los_angeles":   ("LAX", "America/Los_Angeles"),
    "austin":        ("AUS", "America/Chicago"),
    "atlanta":       ("ATL", "America/New_York"),
    "houston":       ("HOU", "America/Chicago"),
    "miami":         ("MIA", "America/New_York"),
    "seattle":       ("SEA", "America/Los_Angeles"),
    "san_francisco": ("SFO", "America/Los_Angeles"),
}

DEFAULT_SINCE = "2023-01-01"
CHUNK_DAYS = 30
RETRIES = 4


def _fetch_chunk(station: str, start: date, end: date) -> pd.DataFrame | None:
    """One date range, or None if every retry failed (the caller must then NOT write)."""
    q = {"station": station, "vars": "tmpf", "sts": f"{start:%Y-%m-%d}T00:00Z",
         "ets": f"{end:%Y-%m-%d}T00:00Z", "sample": "1min", "what": "download",
         "tz": "UTC", "format": "onlycomma"}
    url = f"{ENDPOINT}?{urllib.parse.urlencode(q)}"
    for attempt in range(RETRIES):
        try:
            raw = urllib.request.urlopen(url, timeout=180).read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(raw))
            if df.empty:
                return df
            cols = {c.lower().strip(): c for c in df.columns}
            tcol = next((cols[c] for c in cols if c.startswith("valid")), None)
            fcol = cols.get("tmpf")
            if tcol is None or fcol is None:
                return None
            out = pd.DataFrame({
                "valid_utc": pd.to_datetime(df[tcol], errors="coerce", utc=True),
                "temp_c": (pd.to_numeric(df[fcol], errors="coerce") - 32.0) * 5.0 / 9.0,
            }).dropna()
            return out
        except Exception as exc:                       # noqa: BLE001
            logger.warning("  %s %s..%s attempt %d/%d failed: %s",
                           station, start, end, attempt + 1, RETRIES, str(exc)[:90])
            time.sleep(2 * (attempt + 1))
    return None


def fetch_station(slug: str, since: date, today: date | None = None) -> bool:
    """Top-up one station. Returns True when the file was written."""
    station, tz = ASOS_1MIN[slug]
    path = OUT_DIR / f"{slug}_obs_1min.csv"
    today = today or datetime.utcnow().date()

    existing = None
    if path.exists():
        try:
            existing = pd.read_csv(path)
            existing["valid_utc"] = pd.to_datetime(existing["valid_utc"], utc=True,
                                                   errors="coerce")
            existing = existing.dropna(subset=["valid_utc"])
            last = existing["valid_utc"].max()
            since = max(since, (last - pd.Timedelta(days=2)).date())
        except Exception:
            existing = None

    frames, failed = [], False
    cur = since
    while cur < today:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), today + timedelta(days=1))
        got = _fetch_chunk(station, cur, nxt)
        if got is None:
            failed = True
            break
        if not got.empty:
            frames.append(got)
        cur = nxt

    # GUARD 1 — a partial download must never overwrite a complete file. This is the
    # obs-truncation incident (2026-07-30) written down as code: one failed chunk there was
    # silently skipped, the survivors were concatenated, and a complete file was replaced by a
    # shorter one that still looked plausible.
    if failed:
        logger.error("  %s: a chunk failed after %d retries — keeping the existing file",
                     slug, RETRIES)
        return False
    if not frames:
        logger.info("  %s: nothing new", slug)
        return False

    fresh = pd.concat(frames, ignore_index=True)
    if existing is not None and len(existing):
        fresh = pd.concat([existing, fresh], ignore_index=True)
    fresh = (fresh.dropna(subset=["valid_utc", "temp_c"])
                  .drop_duplicates(subset=["valid_utc"])
                  .sort_values("valid_utc").reset_index(drop=True))
    fresh["valid_local"] = fresh["valid_utc"].dt.tz_convert(tz).dt.tz_localize(None)

    # GUARD 2 — the written file must be a strict SUPERSET of what was on disk.
    #
    # Note what this deliberately is NOT: a row-count or latest-timestamp comparison. Because the
    # fetch is incremental, the new rows are unioned with the existing file above, so
    # len(fresh) >= len(existing) holds by construction and a count test could never fire. A guard
    # that cannot trigger is worse than no guard, because it reads as protection that isn't there.
    #
    # The real risk after a union is that old observations are silently LOST — a timestamp-parsing
    # regression, a dedupe on the wrong key, a future refactor dropping the concat. So check the
    # thing that can actually go wrong: every timestamp previously stored must still be present.
    if existing is not None and len(existing):
        lost = len(set(existing["valid_utc"]) - set(fresh["valid_utc"]))
        if lost:
            logger.error("  %s: the merged file is MISSING %d observations that were on disk "
                         "— refusing to write", slug, lost)
            return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fresh[["valid_utc", "valid_local", "temp_c"]].to_csv(path, index=False)
    logger.info("  %s: %d rows through %s", slug, len(fresh),
                fresh["valid_local"].max().strftime("%Y-%m-%d %H:%M"))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="1-minute ASOS temperatures (US stations)")
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--cities", nargs="*", default=None)
    a = ap.parse_args()
    since = datetime.strptime(a.since, "%Y-%m-%d").date()
    slugs = a.cities or list(ASOS_1MIN)
    for slug in slugs:
        if slug not in ASOS_1MIN:
            logger.warning("  %s has no 1-minute ASOS archive (non-US) — skipped", slug)
            continue
        logger.info("%s (%s)", slug, ASOS_1MIN[slug][0])
        fetch_station(slug, since)


if __name__ == "__main__":
    main()
