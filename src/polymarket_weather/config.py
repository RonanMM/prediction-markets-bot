"""
config.py — Central configuration for Polymarket Weather Tracker.
All cities, API endpoints, and runtime settings live here.
"""

from zoneinfo import ZoneInfo

# ── Cities ──────────────────────────────────────────────────────────────────
# IMPORTANT: coordinates must match the exact Wunderground/observatory station
# that Polymarket uses for resolution — NOT the city centre.
CITIES = {
    "Seoul": {
        "timezone": ZoneInfo("Asia/Seoul"),
        # Incheon International Airport (RKSI) — Polymarket resolution station
        "lat": 37.4602,
        "lon": 126.4407,
        "search_terms": ["Seoul", "seoul"],
    },
    "London": {
        "timezone": ZoneInfo("Europe/London"),
        # London City Airport (EGLC) — Polymarket resolution station
        "lat": 51.5033,
        "lon": 0.0554,
        "search_terms": ["London", "london"],
    },
    "Chicago": {
        "timezone": ZoneInfo("America/Chicago"),
        # O'Hare International Airport (KORD) — Polymarket resolution station
        "lat": 41.9742,
        "lon": -87.9073,
        "search_terms": ["Chicago", "chicago"],
    },
    "New York City": {
        "timezone": ZoneInfo("America/New_York"),
        # LaGuardia Airport (KLGA) — Polymarket resolution station
        "lat": 40.7769,
        "lon": -73.8740,
        "search_terms": ["New York", "NYC", "new york"],
    },
    "Hong Kong": {
        "timezone": ZoneInfo("Asia/Hong_Kong"),
        # Hong Kong Observatory — Polymarket resolution station
        "lat": 22.3020,
        "lon": 114.1740,
        "search_terms": ["Hong Kong", "hong kong"],
    },
}

# ── Polymarket API ───────────────────────────────────────────────────────────
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE  = "https://clob.polymarket.com"

# Keywords used to match weather/temperature markets
MARKET_KEYWORDS = [
    "highest temperature",
    "temperature in",
    "max temperature",
    "high temperature",
]

# ── Open-Meteo API ───────────────────────────────────────────────────────────
OPEN_METEO_BASE          = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODEL           = "icon_seamless"   # 40 members, global, free tier
OPEN_METEO_PARAMS = {
    "daily": "temperature_2m_max,temperature_2m_min",
    "hourly": "temperature_2m",
    "forecast_days": 16,
    "timezone": "auto",
}

# ── Storage ──────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
POLYMARKET_DIR    = "data/polymarket"
WEATHER_DIR       = "data/weather"
PLOTS_DIR         = "plots"
LOGS_DIR          = "logs"

# ── HTTP ─────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 20        # seconds
RETRY_ATTEMPTS    = 3
RETRY_BACKOFF     = 2.0       # exponential base (seconds)

# ── Visualization ────────────────────────────────────────────────────────────
PLOT_DPI          = 150
PLOT_STYLE        = "dark_background"
COLORS = {
    "forecast":  "#00B4D8",
    "implied":   "#FF6B6B",
    "volume":    "#FFD166",
    "probs": [
        "#06D6A0", "#118AB2", "#FFD166",
        "#EF476F", "#9B5DE5", "#F15BB5",
        "#00BBF9", "#FEE440",
    ],
}
