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
    # capture tier — see resolution_anchors.py
    "Los Angeles":   {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["Los Angeles", "los angeles"]},
    "Austin":        {"timezone": ZoneInfo("America/Chicago"),     "search_terms": ["Austin", "austin"]},
    "Atlanta":       {"timezone": ZoneInfo("America/New_York"),    "search_terms": ["Atlanta", "atlanta"]},
    "Houston":       {"timezone": ZoneInfo("America/Chicago"),     "search_terms": ["Houston", "houston"]},
    "Miami":         {"timezone": ZoneInfo("America/New_York"),    "search_terms": ["Miami", "miami"]},
    "Seattle":       {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["Seattle", "seattle"]},
    "San Francisco": {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["San Francisco", "san francisco"]},
}


def _city_view(tiers: tuple[str, ...]) -> dict:
    """Cities whose anchor tier is in *tiers*, in _CITY_META order."""
    return {
        city: {
            "timezone":     meta["timezone"],
            "station_id":   RESOLUTION_ANCHORS[city]["station_code"],
            "lat":          RESOLUTION_ANCHORS[city]["forecast_lat"],
            "lon":          RESOLUTION_ANCHORS[city]["forecast_lon"],
            "search_terms": meta["search_terms"],
        }
        for city, meta in _CITY_META.items()
        if RESOLUTION_ANCHORS[city].get("tier", "modelled") in tiers
    }


# ⚠️ CITIES MEANS MODELLED CITIES and must keep meaning exactly that. It is consumed by twelve
# modules, several of which iterate it to fetch forecasts (fetch_weather, fetch_ensemble) or to
# train (train_calibrator does `for city in CITIES.keys()`). Adding capture-only cities here
# would pull forecasts for cities we do not model and attempt EMOS training on cities with no
# archives — silently, on a green run. Use ALL_CITIES for discovery and capture instead.
#
# This distinction is about per-city WORK (fetching archives, training, backtesting), not about
# who is allowed to LOOK a city up. `grading.py` and `historical_backtester.py` legitimately build
# lookup dicts from the full `resolution_anchors.RESOLUTION_ANCHORS` registry — including the
# seven capture cities — because grading needs their units/slugs/resolution URLs too. Per-city
# WORK, by contrast, must iterate `resolution_anchors.modelled_anchors()` (or this `CITIES`),
# never the raw `RESOLUTION_ANCHORS.items()` — see the four `fetch_historical_leads*.py` archive
# fetchers, which take no `--cities` flag and are run unconditionally by `retrain.yml`.
CITIES         = _city_view(("modelled",))
CAPTURE_CITIES = _city_view(("capture",))
ALL_CITIES     = _city_view(("modelled", "capture"))

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
# 119 members MAX, and the count FALLS WITH LEAD — do not quote a single fixed number.
# Measured over the 43,902 stored ensemble rows: 119 out to ~day 6, 80 from day 8, 30 at the
# longest leads (the three models have different horizons). Only 41% of rows carry the full 119;
# at the leads the engine actually bets (1-7 d) the mean is ~114. The five observed values are
# exactly {30, 39, 69, 80, 119}. (An earlier version of this comment decomposed it as
# "ICON 40 + GFS 31 + ECMWF 50", which sums to 121 — that per-model split was never verified,
# whereas the totals above are measured from the data.)
# NOTE: ecmwf_ifs04 was retired upstream and silently returned zero members (audit 2026-07-04
# found only 69 = ICON+GFS); ifs025 is the live ECMWF ensemble id.
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
KALSHI_DIR        = "data/kalshi"
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
# α5 coherence bonus only applies to markets this liquid. Incoherent bins (sum != 1) are only a
# tradeable mispricing when there is real two-sided liquidity; otherwise incoherence just means the
# market is thin and won't fill — so we do NOT reward it. Set above MIN_LIQUIDITY on purpose.
COHERENCE_MIN_LIQ     = 2000      # USDC
KELLY_FRACTION        = 0.50      # fractional Kelly multiplier
MAX_KELLY_PER_BET     = 0.08      # absolute cap per single market
MAX_KELLY_PER_GROUP   = 0.20      # cap across correlated (city, date) group
MAX_TOTAL_KELLY       = 0.40      # hard cap on total portfolio exposure per run

# Polymarket fee: ~2% of the trade amount (taker fee on winning side)
# ⚠️ LEGACY BACKTEST ASSUMPTION — kept because evaluate_oos/backtest_common still settle with it
# and it is conservative (harsher than reality) for the model book. The REAL schedule is below.
FEE_RATE              = 0.02

# ── E1 VERIFIED 2026-07-13 (Polymarket maker-taker schedule, effective 2026-07-01) ───────────
# Weather-category taker fee, charged in USDC at trade time:
#     fee = shares · WEATHER_TAKER_RATE · p · (1 − p)
# → max 1.25¢/share at p=0.50, ~1.05¢ at 0.70/0.30, ~0.24¢ at 0.95. MAKERS PAY NOTHING and the
# Maker Rebates Program pays 25% of collected weather taker fees back to makers daily — so
# maker-first execution (megaplan E2) is decisively favored.
# Sources: help.polymarket.com "Trading fees" (article 13364478), docs.polymarket.us/fees.
# shoulder_book.py settles with this real formula; migrating the model-book backtesters off
# FEE_RATE is queued (their 2% assumption overstates costs, especially near-extreme prices).
WEATHER_TAKER_RATE    = 0.05
MAKER_FEE             = 0.0
MAKER_REBATE_SHARE    = 0.25


def taker_fee_per_share(price: float) -> float:
    """Verified Polymarket weather taker fee per share (USDC) at `price`."""
    p = min(max(float(price), 0.0), 1.0)
    return WEATHER_TAKER_RATE * p * (1.0 - p)

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

# W2 (adverse-selection sigma inflation, sigma·(1+γ·|model−market|)) was TESTED 2026-07-11 on the
# frozen graded set (n=220): Brier improves (0.161→0.154 at γ=4) but ROI does NOT (−20.3% →
# −21/−23% at every γ) — widening sigma cannot fix a wrong CENTER; the surviving flags are the
# same bad book. NOT adopted. Do not re-try without new reason; see docs/EDGE_MEGAPLAN.md §4 W2.

# Minimum raw market price on either side of the bet.
# Prices below this are near-settled/expired markets — skip them.
MIN_MARKET_PRICE      = 0.02   # 2% — below this the market has effectively resolved

# ── E3: per-bucket selective aggression (docs/EDGE_MEGAPLAN.md §5) ───────────────────────────
# A bucket is "City|lead-band" (see bucket_key below). Only buckets listed here are eligible for
# EXECUTION (paper first, then live); the tracker still records EVERY flagged opportunity so the
# evaluation keeps its full sample.
# REVISED 2026-07-12 after the W0 settlement-truth fix: the original NYC|same-day nomination
# (Brier 0.084 vs market 0.114 on 07-11 labels) was substantially a GRADING ARTIFACT — under
# settlement-faithful labels it grades 0.175 vs 0.120 and is OUT. Current nominations on the
# corrected frozen set (both marginal — treat as hypotheses, not edges):
#   Seoul|1d    model 0.123 vs market 0.126 (n=22)
#   Chicago|1d  model 0.106 vs market 0.124 (n=14)
# PAPER ONLY until the forward gate passes per bucket: ≥40 bets with target_date AFTER
# E3_NOMINATION_DATE graded, AND model Brier ≤ market Brier on those forward bets.
# evaluate_oos.py prints per-bucket progress against this gate.
#
# ── 2026-07-28: ALL BUCKETS UNDER TEST, NO HAND-PICKING ────────────────────────
# The two nominations above were chosen by eyeballing in-sample numbers, which is the exact
# procedure a null world defeats. Simulation (model recentred to EXACTLY market-equal in every
# bucket, real sizes and city-day correlations preserved, 5,000 runs): the best-looking bucket
# still shows a gap of −0.046 median / −0.18 at the 5th pct — while our real best (Chicago|2d+)
# is only −0.016. In other words the observed ranking is weaker than what pure noise produces.
# Worse, the ranking systematically crowns the SMALLEST bucket: HongKong|2d+ (n=5) "wins" 34% of
# null runs, London|1d (n=48) only 4% — small samples make extreme numbers, not good forecasts.
#
# So: every city × horizon is nominated and tested on identical prospective terms, and the
# threshold is corrected for testing many at once (evaluate_oos._e3_gate_z, Bonferroni). Buckets
# nominated 2026-07-12 KEEP that clock — their forward sample was earned before anyone looked.
# Everything added today starts today, because its history has already been inspected and a gate
# must never be graded on the data that suggested it.
E3_NOMINATIONS = {
    # bucket                 forward clock starts (UTC date)
    "Seoul|1d":             "2026-07-12",   # original nomination — clock preserved
    "Chicago|1d":           "2026-07-12",   # original nomination — clock preserved
    "Seoul|same-day":       "2026-07-28",
    "Seoul|2d+":            "2026-07-28",
    "London|same-day":      "2026-07-28",
    "London|1d":            "2026-07-28",
    "London|2d+":           "2026-07-28",
    "Chicago|same-day":     "2026-07-28",
    "Chicago|2d+":          "2026-07-28",
    "NYC|same-day":         "2026-07-28",
    "NYC|1d":               "2026-07-28",
    "NYC|2d+":              "2026-07-28",
    "HongKong|same-day":    "2026-07-28",
    "HongKong|1d":          "2026-07-28",
    "HongKong|2d+":         "2026-07-28",
}

# Execution-eligible = a bucket whose FORWARD gate has actually PASSED. Empty is the honest
# state: no bucket has cleared its gate, so nothing is eligible for real size. (Before
# 2026-07-28 this held the hand-picked nominations, which made `live_eligible` read as
# "somebody liked this bucket" rather than "this bucket earned it".)
LIVE_BUCKETS: set[str] = set()

E3_NOMINATION_DATE    = "2026-07-12"   # legacy default for buckets absent from E3_NOMINATIONS
E3_FORWARD_MIN_BETS   = 40


def bucket_key(city: str, days_ahead: float) -> str:
    """Canonical E3 bucket for an opportunity: 'City|same-day', 'City|1d', or 'City|2d+'.

    `city` is the display name (config.CITY_NAMES value, as stored in the eval tracker).
    Bands match the decomposition that nominated the buckets: same-day ≤0.5 d, 1d ≤1.5 d.
    """
    band = "same-day" if days_ahead <= 0.5 else ("1d" if days_ahead <= 1.5 else "2d+")
    return f"{city}|{band}"
