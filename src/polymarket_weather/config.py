"""
config.py — Central configuration for Polymarket Weather Tracker.
All cities, API endpoints, and runtime settings live here.
"""

from zoneinfo import ZoneInfo

from resolution_anchors import RESOLUTION_ANCHORS

# ── Cities ──────────────────────────────────────────────────────────────────
# Forecast coordinates and the resolution station code come from
# resolution_anchors.py — the single source of truth. Only the fields that are
# NOT part of the resolution anchor live here: the local timezone and the
# Polymarket market search terms.
_CITY_META = {
    "Seoul":         {"timezone": ZoneInfo("Asia/Seoul"),       "search_terms": ["Seoul", "seoul"]},
    "London":        {"timezone": ZoneInfo("Europe/London"),    "search_terms": ["London", "london"]},
    "Chicago":       {"timezone": ZoneInfo("America/Chicago"),  "search_terms": ["Chicago", "chicago"]},
    "New York City": {"timezone": ZoneInfo("America/New_York"), "search_terms": ["New York", "NYC", "new york"]},
    "Hong Kong":     {"timezone": ZoneInfo("Asia/Hong_Kong"),   "search_terms": ["Hong Kong", "hong kong"]},
}

CITIES = {
    city: {
        "timezone":     meta["timezone"],
        "station_id":   RESOLUTION_ANCHORS[city]["station_code"],   # == anchor station_code (ICAO/obs)
        "lat":          RESOLUTION_ANCHORS[city]["forecast_lat"],
        "lon":          RESOLUTION_ANCHORS[city]["forecast_lon"],
        "search_terms": meta["search_terms"],
    }
    for city, meta in _CITY_META.items()
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
# ~119 members (ICON 40 + GFS 31 + ECMWF 50). NOTE: ecmwf_ifs04 was retired upstream and
# silently returned zero members (audit 2026-07-04 found only 69 = ICON+GFS); ifs025 is
# the live ECMWF ensemble id.
ENSEMBLE_MODEL           = "icon_seamless,gfs_seamless,ecmwf_ifs025"
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

CITY_NAMES = {
    "new_york_city": "NYC",
    "new_york":      "NYC",
    "london":        "London",
    "chicago":       "Chicago",
    "hong_kong":     "HongKong",
    "seoul":         "Seoul",
}
# Based on ECMWF verification statistics for 2m Tmax
MIN_EDGE              = 0.06      # 6 pp raw probability
MIN_LIQUIDITY         = 1000      # USDC
MIN_BINS_FOR_PMF      = 3         # need >=3 exact bins to reconstruct distribution
# α5 coherence bonus only applies to markets this liquid. Incoherent bins (sum != 1) are only a
# tradeable mispricing when there is real two-sided liquidity; otherwise incoherence just means the
# market is thin and won't fill — so we do NOT reward it. Set above MIN_LIQUIDITY on purpose.
COHERENCE_MIN_LIQ     = 2000      # USDC
KELLY_FRACTION        = 0.50      # fractional Kelly multiplier
MAX_KELLY_PER_BET     = 0.08      # absolute cap per single market
MAX_KELLY_PER_GROUP   = 0.20      # cap across correlated (city, date) group
MAX_TOTAL_KELLY       = 0.40      # hard cap on total portfolio exposure per run

# Polymarket fee: ~2% of the trade amount (taker fee on winning side)
FEE_RATE              = 0.02

# Backtest execution realism: half of the bid-ask spread paid on entry (in probability units).
# Live fills cross the spread and eat slippage; the backtest otherwise fills at the last
# snapshot mid-price, which flatters ROI. Applied as an extra cost to `their_prob` on entry so
# measured edge must survive realistic execution, not just an idealized mid.
HALF_SPREAD           = 0.01   # 1 cent per share on entry (tune once real order-book data exists)

# Shrink-to-market weight for the model's probability: our_prob = w·model + (1-w)·market.
# The market currently out-predicts the model on Brier, so deviating fully from the price loses.
# w=1.0 → pure model (no shrink, current behaviour); w<1.0 → only deviate on strong signal, which
# lowers Brier and sizes conservatively. Fit the Brier-minimizing w with evaluate_oos.py (it prints
# a recommendation), then set it here. Default 1.0 keeps behaviour unchanged until it is validated.
SHRINK_WEIGHT         = 1.0

# Minimum raw market price on either side of the bet.
# Prices below this are near-settled/expired markets — skip them.
MIN_MARKET_PRICE      = 0.02   # 2% — below this the market has effectively resolved
