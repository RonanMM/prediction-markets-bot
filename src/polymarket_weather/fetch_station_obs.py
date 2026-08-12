"""fetch_station_obs.py — hourly station observations for intraday conditioning.

Same-day temperature markets are priced with the day's observations in hand: by early
afternoon the running maximum already caps the possible outcomes. A pure-forecast model
never sees that, which is exactly where the eval showed the market's biggest advantage
(lead-0/1 bets). This fetcher pulls the hourly station temperatures for each resolution
station (IEM `asos.py`, station-local time) so the predictor can condition on the running
daily max at any snapshot fetch time. The same file is `wu_truth`'s input, so it is also
the GRADING ruler for nine cities.

⚠️ RULER #13 (fixed 2026-08-12) — routine METARs alone are not wunderground's day.
This fetcher requested `report_type=3` (routine METAR only) for its whole life. But
wunderground's daily history table — what Polymarket settles on — lists EVERY observation,
including SPECIs (specials, filed off-schedule when conditions change). Its daily high/low
are the extremes over that full set. Routine-only therefore reconstructs a max that is too
LOW and a min that is too HIGH, and the error is **strictly one-sided**: removing
observations can only shrink an extreme, never widen it.

Measured over 2026-03-01→08-12, nine US stations, 1,476 station-days: specials raise the
daily max on **2.5%** of days and lower the daily min on **6.1%**, changing the whole-°F
max on 35 station-days. Rare, but it is exactly the Atlanta 2026-08-10/11 pair the
settlement audit flagged (ourgrade=0 settled=1 on the hotter bin, and its mirror on the
cooler one). On the four settled market-rows in the whole history where the two candidate
rulers disagree, **+specials agrees with the actual settlement 4/4 and routine-only 0/4**
(e.g. ATL 2026-08-10 peaked 94°F in a 15:39 SPECI; the routine :52 obs top out at 92°F).

Consequences worth knowing: the bias hits **Tmin markets more often than Tmax** (6.1% vs
2.5% of days), and it contaminated `venue_basis.py` directly — that experiment measures
CLI−WU, so a WU reconstruction biased low inflates the very quantity it exists to measure.

Output: data/weather/{slug}_obs_hourly.csv  (valid_local, temp_c, source)
`source` is `iem_metar+speci:{station}`. A file still stamped plain `iem_metar:` predates
ruler #13 and is refetched in full on the next run — see `fetch_station_obs`.

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
    # ── capture tier (spec 2026-08-03) — Polymarket's ruler, Wunderground, reconstructed from
    # hourly METARs (wu_truth). Kalshi's ruler for these same stations is the NWS CLI, archived
    # separately in fetch_historical_truth.SOURCES. IEM uses the 3-letter form for US stations,
    # matching the existing LGA / ORD entries.
    "los_angeles":   ("LAX", "America/Los_Angeles"),
    "austin":        ("AUS", "America/Chicago"),
    "atlanta":       ("ATL", "America/New_York"),
    "houston":       ("HOU", "America/Chicago"),
    "miami":         ("MIA", "America/New_York"),
    "seattle":       ("SEA", "America/Los_Angeles"),
    "san_francisco": ("SFO", "America/Los_Angeles"),
}


def _fetch_range(station: str, tz: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    params = {
        "station": station, "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        # report_type 3 = routine METAR, 4 = SPECI (special). BOTH are required: wunderground's
        # daily table lists every observation, and its high/low are the extremes over that full
        # set — a peak reached between the :52 routines is captured only by a SPECI. Requesting
        # routine-only made our reconstructed max too LOW and our min too HIGH, and the error is
        # strictly one-sided (dropping observations can never widen an extreme). See ruler #13
        # in the module docstring.
        "tz": tz, "format": "onlycomma", "missing": "M", "report_type": [3, 4],
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


def _needs_speci_backfill(out: str) -> bool:
    """True if `out` exists but was written before ruler #13 (routine METARs only).

    Detected from the `source` stamp rather than a version file, so it is self-describing:
    the data itself says which ruler produced it. An unreadable or source-less file is
    treated as needing the backfill — the safe direction, since a full refetch only costs
    time whereas a mixed-ruler file silently corrupts grading.
    """
    if not os.path.exists(out):
        return False                       # nothing there: the normal path already fetches full
    try:
        src = pd.read_csv(out, usecols=["source"], nrows=200)["source"]
    except Exception:
        return True
    return not src.astype(str).str.contains("+speci", regex=False).all()


def fetch_station_obs(recent_only: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.now()
    top_up = recent_only

    for slug, (station, tz) in OBS_STATIONS.items():
        out = os.path.join(OUT_DIR, f"{slug}_obs_hourly.csv")

        # A pre-ruler-#13 file holds routine-only observations. Topping it up with a
        # specials-inclusive window would leave ONE file graded by TWO rulers, split at an
        # arbitrary date — worse than either rule applied consistently, and invisible
        # afterwards. Force the one-time full backfill instead; it is self-healing and
        # happens exactly once per station.
        recent_only = top_up
        if top_up and _needs_speci_backfill(out):
            logger.info(f"{slug}: existing CSV predates ruler #13 (routine-only) — "
                        f"forcing a full backfill so one file is not graded by two rulers")
            recent_only = False

        if recent_only:
            ranges = [(now - timedelta(days=3), now + timedelta(days=1))]
        else:
            ranges = [(datetime(y, 1, 1), datetime(y + 1, 1, 1))
                      for y in range(START_YEAR, now.year + 1)]

        chunks = []
        failed_ranges = []
        for start, end in ranges:
            df = _fetch_range(station, tz, start, min(end, now + timedelta(days=1)))
            if df is not None and not df.empty:
                chunks.append(df)
            else:
                failed_ranges.append(start.year)
            time.sleep(0.5)
        if not chunks:
            logger.error(f"{slug}: nothing fetched — keeping existing CSV")
            continue

        # GUARD 1 — a partial year-set must never overwrite a complete file.
        #
        # 2026-07-30 incident: LGA's 2026 chunk failed all four retries while 2022-2025 succeeded.
        # The failure was silently skipped, the 35,032 surviving rows cleared the old absolute
        # floor of 20,000, and the file was overwritten — dropping every 2026 observation. wu_truth
        # then returned None for all 2026 dates and NYC grading fell back to the NWS CLI (the
        # pre-W0 ruler), taking the settlement audit 97.0% -> 94.7% and the published pooled
        # model-market gap from +0.0178 (CI entirely above zero) to +0.0122 (CI spanning zero).
        # The project's headline verdict flipped on a dropped HTTP request, with a green run.
        if not recent_only and failed_ranges:
            logger.error(
                f"{slug}: year chunk(s) {sorted(failed_ranges)} FAILED after retries — keeping "
                f"the existing CSV. A partial fetch must never overwrite a complete file: the "
                f"missing year would silently disappear from wu_truth and grading would fall "
                f"back to the CLI ruler.")
            continue

        df = pd.concat(chunks, ignore_index=True)
        df["source"] = f"iem_metar+speci:{station}"

        if recent_only and os.path.exists(out):
            old = pd.read_csv(out)
            df = pd.concat([old, df], ignore_index=True)

        df = (df.drop_duplicates("valid_local", keep="last")
                .sort_values("valid_local").reset_index(drop=True))

        if not recent_only and len(df) < 20000:
            logger.error(f"{slug}: only {len(df)} rows — keeping existing CSV")
            continue

        # GUARD 2 — never regress against what is already on disk.
        #
        # Guard 1 catches a chunk that errored. This catches everything else: an upstream range
        # that starts returning fewer rows, a silently-emptied response that still parses, a
        # station rename. Refetched data should only ever grow — obs_hourly is append-only in
        # spirit even though it is rewritten wholesale — so fewer rows or an older latest
        # observation means the NEW copy is worse and must be rejected.
        if os.path.exists(out):
            try:
                prev = pd.read_csv(out)
            except Exception as exc:                      # unreadable existing file: nothing to
                logger.warning(f"{slug}: existing CSV unreadable ({exc}) — writing fresh")
                prev = None
            if prev is not None and len(prev):
                new_latest = str(df["valid_local"].max())
                old_latest = str(prev["valid_local"].max())
                if len(df) < len(prev) or new_latest < old_latest:
                    logger.error(
                        f"{slug}: REGRESSION — refetched {len(df)} rows (latest {new_latest}) "
                        f"vs existing {len(prev)} rows (latest {old_latest}). Keeping the "
                        f"existing CSV. Refetched observations must never shrink; a smaller "
                        f"file means the upstream fetch was incomplete.")
                    continue

        df.to_csv(out, index=False)
        logger.info(f"{slug}: saved {out} — {len(df)} rows, latest {df['valid_local'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true", help="top-up: last 3 days only")
    args = ap.parse_args()
    fetch_station_obs(recent_only=args.recent)
