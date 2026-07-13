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
_WU_RECON_SLUGS = {"nyc": "new_york_city", "newyorkcity": "new_york_city",
                   "newyork": "new_york_city", "chicago": "chicago"}

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


@lru_cache(maxsize=8)
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
