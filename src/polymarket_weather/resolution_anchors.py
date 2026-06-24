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
3. The Truth Anchor: the Meteostat station id (`meteostat_id`) used to fetch the
   historical *actuals* that label/calibrate our ML models. This must point at the
   same physical sensor the market resolves from.

DO NOT modify this file unless Polymarket explicitly changes their market rules.
"""

RESOLUTION_ANCHORS = {
    "London": {
        "resolution_url": "https://www.wunderground.com/history/daily/gb/london/EGLC",
        "resolution_unit": "whole °C",
        "forecast_lat": 51.5053,
        "forecast_lon": 0.0553,
        "station_code": "EGLC",
        # ⚠️ VERIFY: Meteostat labels EGLC0 "London / Abbey Wood" (~5 km from EGLC) though it carries ICAO EGLC.
        "meteostat_id": "EGLC0",
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
        # ⚠️ VERIFY: Meteostat 45007 is labelled "Hong Kong Int'l Airport / VHHH", but its
        # coords (22.33, 114.18) sit near the Observatory (HKO), not the airport (113.91).
        # Market resolves off HKO — confirm this station's data tracks HKO before trusting it.
        "meteostat_id": "45007",
        "aliases": ["HongKong"]
    },
}
