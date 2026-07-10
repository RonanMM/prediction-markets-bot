"""
resolution_anchors.py — The Absolute Ground Truth for Market Resolution

IMPORTANT:
Polymarket DOES NOT use coordinates to resolve markets. It resolves by reading
a number off a specific webpage for a named station.

This file separates the distinct anchors for each city:
1. The Resolution Anchor: The exact URL and unit the Polymarket UMA oracle uses.
2. The Forecast Anchor: The latitude/longitude we feed to our numerical weather
   prediction (NWP) models. This is the grid point that best PREDICTS the resolution
   station — usually the station's own location, but NOT always: where a station's own
   grid cell is unreliable (e.g. a coastal/reclaimed-land cell like Incheon), a nearby
   land cell with higher validated predictive skill is used instead (`forecast_lat/lon`).
3. The Truth Anchor: the source `fetch_historical_truth.py` reads the historical
   *actuals* from. Since 2026-07 this is a resolution-faithful feed per city (NWS CLI
   for KLGA/KORD, IEM METAR daily for EGLC/RKSI, the HKO open-data API for HKO) — the
   same sensor/series the market's resolution page publishes. The `meteostat_id` kept
   below is legacy/reference only: a 2026-07-03 audit found Meteostat corrupted for
   recent weeks (KLGA June daily maxes off by up to ~9 °C vs the NWS CLI report) and
   its Hong Kong station 45007 disagreeing with HKO >1.5 °C on ~66 days/yr.

DO NOT modify this file unless Polymarket explicitly changes their market rules.
"""

import re


def slug(city: str) -> str:
    """Canonical data-file slug for a city name or alias: lowercase, with each run of
    non-alphanumeric characters collapsed to a single '_'. THE single slug definition — every
    fetcher / trainer / loader / grader must use it so a punctuated city name (e.g.
    'Washington, D.C.') can never split its data across two different spellings (F7)."""
    return re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")


RESOLUTION_ANCHORS = {
    "London": {
        "resolution_url": "https://www.wunderground.com/history/daily/gb/london/EGLC",
        "resolution_unit": "whole °C",
        "forecast_lat": 51.5053,
        "forecast_lon": 0.0553,
        "station_code": "EGLC",
        # Truth = IEM METAR daily for EGLC itself (whole °C, matching the market unit);
        # resolves the old concern that Meteostat EGLC0 was labelled "Abbey Wood" ~5 km away.
        "meteostat_id": "EGLC0",   # legacy/reference only
    },
    "Seoul": {
        "resolution_url": "https://www.wunderground.com/history/daily/kr/incheon/RKSI",
        "resolution_unit": "whole °C",
        # FORECAST point is SKILL-OPTIMIZED, not the station location. The Incheon airport
        # ERA5 cell is coastal/reclaimed-land and predicts the RKSI station poorly
        # (5-fold CV RMSE 1.96, ~-2°C cold bias). This inland "Bucheon corridor" land cell
        # predicts RKSI best (CV RMSE 1.19, ~0 bias; best on a 2024-26 forward holdout too),
        # validated over 2015-2026 against Meteostat 47113 truth.
        # DO NOT "correct" this back to the airport (37.4691/126.4510) — that DEGRADES skill.
        "forecast_lat": 37.5035,
        "forecast_lon": 126.7660,
        "station_code": "RKSI",
        "meteostat_id": "47113",   # TRUTH = Incheon Intl Airport (RKSI), confirmed via meteostat.net
    },
    "Chicago": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/il/chicago/KORD",
        "resolution_unit": "whole °F",
        "forecast_lat": 41.9786,
        "forecast_lon": -87.9048,
        "station_code": "KORD",
        "meteostat_id": "72530",   # confirmed = Chicago O'Hare (KORD) via meteostat.net
    },
    "New York City": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
        "resolution_unit": "whole °F",
        "forecast_lat": 40.7772,
        "forecast_lon": -73.8726,
        "station_code": "KLGA",
        "meteostat_id": "72503",   # confirmed = New York LaGuardia (KLGA) via meteostat.net
        "aliases": ["NYC"]
    },
    "Hong Kong": {
        "resolution_url": "https://www.weather.gov.hk/en/cis/climat.htm",
        "resolution_unit": "0.1 °C",
        "forecast_lat": 22.3019,
        "forecast_lon": 114.1743,
        "station_code": "HKO",
        # Truth = the HKO open-data API (CLMMAXT/CLMMINT, 0.1 °C) — the exact series the
        # resolution page publishes. The old ⚠️ was justified: Meteostat 45007 disagreed
        # with HKO by >1.5 °C on ~66 days/yr (2024 audit), so it must NOT grade HK bets.
        "meteostat_id": "45007",   # legacy/reference only — do not grade from this
        "aliases": ["HongKong"]
    },
}
