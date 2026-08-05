"""wu_truth.py — settlement-faithful daily extremes for the US cities (megaplan W0).

The markets resolve on **wunderground.com** station pages (`resolution_anchors.resolution_url`),
NOT on the NWS CLI our historical-actuals feed reads. WU's daily max/min for an airport is the
extreme over the ROUTINE HOURLY METAR observations in the station-local calendar day — which can
differ from the CLI by a whole degree, because the CLI uses continuous 1-minute sensor data (it
sees spikes/dips between METARs) and a local-STANDARD-time day window.

The 2026-07-12 settlement audit (audit_settlements.py) found 4/60 markets graded OPPOSITE to how
they actually settled, all at whole-degree boundaries. Reconstructing the WU reading from our
stored hourly METARs (data/weather/{slug}_obs_hourly.csv, IEM report_type=3, station-local time)
matched the real settlement in ALL THREE US cases:
    NYC 2026-07-03  Tmin: CLI 79°F, hourly-METAR/WU 80°F  → '80-81°F' settled YES
    NYC 2026-06-27  Tmin: CLI 69°F, hourly-METAR/WU 70°F  → '68-69°F' settled NO
    Chicago 2026-05-28 Tmax: CLI 72°F, hourly-METAR/WU 71°F → '72-73°F' settled NO
so for NYC/Chicago this module is the primary truth and the CLI feed is the fallback/sanity rail.

Seoul (RKSI) does NOT reconstruct from hourly METARs (its one audit miss graded 14°C from both
our feeds while the market settled NO — WU most likely ingests SYNOP tenths for RKSI), and
London had no audit misses; both stay on the existing IEM-daily truth. Hong Kong resolves on the
HKO itself (resolution_url), so its existing HKO feed is already settlement-faithful.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

_OBS_DIR = Path(__file__).resolve().parent / "data" / "weather"

# Cities whose WU page is METAR-driven AND whose reconstruction is settlement-validated.
# Keys are city names normalized by _slug_for (lowercase, spaces/underscores stripped).
#
# ⚠️ ADMISSION IS EARNED, NEVER ASSUMED. A city joins this map only after its reconstruction is
# shown to grade REAL settlements at least as well as the CLI feed it replaces. Adding a city
# because "Polymarket says it settles on wunderground" is not enough — that is the claim being
# tested, and a station whose METARs miss its true extreme would grade WORSE on the WU rule.
#
# The 7 capture-tier cities were admitted 2026-08-05 against 412 settled breadth-book markets over
# 11 target dates (the book reads each resolved market's own outcome). Every city improved and none
# regressed:
#     Atlanta 49/55→55/55 · Austin 46/49→49/49 · Houston 38/40→40/40 · Los Angeles 36/43→43/43
#     Miami 101/110→110/110 · San Francisco 58/66→66/66 · Seattle 42/49→49/49
#     OVERALL 370/412 (89.8%) → 412/412 (100.0%)
# 89.8% was BELOW the project's 95% settlement-audit floor, i.e. these cities were being graded
# with a ruler that would have failed the guard had the guard been watching them (it was not —
# audit_settlements covered only the 5 modelled cities; it now covers all 12).
_WU_RECON_SLUGS = {"nyc": "new_york_city", "newyorkcity": "new_york_city",
                   "newyork": "new_york_city", "chicago": "chicago",
                   # capture tier — validated 2026-08-05, see above
                   "losangeles": "los_angeles", "austin": "austin", "atlanta": "atlanta",
                   "houston": "houston", "miami": "miami", "seattle": "seattle",
                   "sanfrancisco": "san_francisco"}

# A day qualifies only with reasonably complete METAR coverage: a missing stretch can hide the
# extreme (the min usually lives 03-06 local, the max 13-17). CLI fallback covers gappy days.
_MIN_OBS_PER_DAY = 18
_MAX_GAP_HOURS = 3.0
# Glitch guard: a lone bad METAR can fabricate an extreme; if the reconstruction sits more than
# this far from the CLI value the caller passes in, the caller should prefer the CLI.
SANITY_MAX_DIVERGENCE_C = 2.5


def _slug_for(city) -> str | None:
    key = str(city).strip().lower().replace("_", "").replace(" ", "")
    return _WU_RECON_SLUGS.get(key)


# 16, not 8: with the capture tier there are 11 reconstruction slugs, and a cache smaller than the
# working set thrashes — every city switch would re-read and re-parse a ~40k-row CSV.
@lru_cache(maxsize=16)
def _load_obs(slug: str):
    path = _OBS_DIR / f"{slug}_obs_hourly.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"valid_local", "temp_c"}.issubset(df.columns):
        return None
    df["valid_local"] = pd.to_datetime(df["valid_local"], errors="coerce")
    df = df.dropna(subset=["valid_local", "temp_c"])
    df["date_local"] = df["valid_local"].dt.date.astype(str)
    return df


def reconstruct(slug: str, date_str, kind: str):
    """WU-style daily extreme in °C from a slug's hourly METARs, WITHOUT the admission check.

    Separated from `wu_daily_extreme` because measuring a station's reconstruction is exactly how
    one decides whether to admit it to `_WU_RECON_SLUGS` — routing that measurement through the
    allowlist would make admission unreachable for any new city. Use this for ANALYSIS only;
    anything that grades a market must call `wu_daily_extreme`, which refuses cities whose
    reconstruction has not been validated against real settlements.
    """
    obs = _load_obs(slug)
    if obs is None:
        return None
    day = obs[obs["date_local"] == str(date_str)].sort_values("valid_local")
    if len(day) < _MIN_OBS_PER_DAY:
        return None
    gaps = day["valid_local"].diff().dt.total_seconds().div(3600.0)
    if gaps.max() > _MAX_GAP_HOURS:
        return None
    return float(day["temp_c"].max() if kind == "max" else day["temp_c"].min())


def wu_daily_extreme(city, date_str, kind: str):
    """WU-style daily extreme in °C for a validated city, or None.

    kind: 'max' or 'min'. Returns None when the city isn't reconstruction-validated
    (Seoul/London/HK) or the day's METAR coverage is too gappy to trust — callers then
    fall back to the historical-actuals feed.
    """
    slug = _slug_for(city)
    if slug is None:
        return None
    obs = _load_obs(slug)
    if obs is None:
        return None
    day = obs[obs["date_local"] == str(date_str)].sort_values("valid_local")
    if len(day) < _MIN_OBS_PER_DAY:
        return None
    gaps = day["valid_local"].diff().dt.total_seconds().div(3600.0)
    if gaps.max() > _MAX_GAP_HOURS:
        return None
    return float(day["temp_c"].max() if kind == "max" else day["temp_c"].min())
