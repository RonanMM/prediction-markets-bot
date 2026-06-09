"""
polymarket_weather_v3.py
========================
Polymarket Weather Inefficiency Analyzer — production-grade.

Alpha signals implemented (all usable with your existing data):
  α1  Forecast momentum       — EMA of Δforecast over recent snapshots
  α2  Min/max spread proxy    — diurnal range as convective uncertainty signal
  α3  Student-t tail model    — heavier tails than Gaussian (nu calibrated per horizon)
  α4  Constrained PMF         — full probability-mass-conserving reconstruction
                                 using exact + gte + lte bins jointly
  α5  Internal consistency    — detect markets whose bins don't sum to ~1.0;
                                 trade the missing-mass bins
  α6  Volume recency          — 24h/total volume ratio flags informed-trader activity
  α7  Forecast convergence    — cross-snapshot variance; high variance → market stale
  α8  Market update lag       — time since last significant market reprice
  α9  Correlated bet grouping — per-(city,date) exposure cap; Kelly across group

Usage:
  python polymarket_weather_v3.py --data_dir ./data [options]
  python polymarket_weather_v3.py --data_dir ./data --min_edge 0.06 --bankroll 2000
"""

from __future__ import annotations
import argparse, json, re, warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import erf, sqrt, log, pi
from pathlib import Path
from typing import Optional
from predictors import MLCalibratorPredictor, EnsemblePredictor
from predictors.nwp_fallback import spread_sigma_boost

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t as student_t
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# NWP-calibrated sigma (°C) and Student-t degrees-of-freedom by days-ahead
# Based on ECMWF verification statistics for 2m Tmax
NWP_PARAMS: dict[int, tuple[float, float]] = {
    # days: (sigma_base, t_nu)
    0: (0.7,  12.0),   # near-observation: almost Gaussian
    1: (1.2,  8.0),
    2: (1.6,  6.5),
    3: (2.0,  5.5),
    4: (2.4,  5.0),
    5: (2.8,  4.5),
}
NWP_BEYOND = (3.2, 4.0)          # sigma, nu for >5 days ahead

# Momentum signal: EMA span (in number of forecast snapshots)
MOMENTUM_EMA_SPAN     = 3
# Minimum momentum magnitude to boost/penalise edge (°C/snapshot)
MOMENTUM_THRESHOLD    = 0.15
# Diurnal spread sigma multiplier  (spread/baseline - 1) * this
SPREAD_SIGMA_WEIGHT   = 0.25

# Edge / sizing parameters
MIN_EDGE              = 0.06      # 6 pp raw probability
MIN_LIQUIDITY         = 400       # USDC
MIN_BINS_FOR_PMF      = 3         # need ≥3 exact bins to reconstruct distribution
KELLY_FRACTION        = 0.25      # fractional Kelly multiplier
MAX_KELLY_PER_BET     = 0.12      # absolute cap per single market
MAX_KELLY_PER_GROUP   = 0.20      # cap across correlated (city, date) group
MAX_TOTAL_KELLY       = 0.40      # hard cap on total portfolio exposure per run

# Polymarket fee: ~2% of the trade amount (taker fee on winning side)
FEE_RATE              = 0.02

# Minimum raw market price on either side of the bet.
# Prices below this are near-settled/expired markets — skip them.
MIN_MARKET_PRICE      = 0.02   # 2% — below this the market has effectively resolved

# Volume-recency threshold: ratio vol_24h/vol_total
INFORMED_RECENCY      = 0.65

# Market staleness: if market hasn't moved >1pp in last N hours → flag stale
STALE_HOURS           = 6
STALE_MOVE_THRESHOLD  = 0.01

# Consistency: if sum(exact bins) deviates from ~1.0 by more than this → flag
PMF_SUM_DEVIATION_MAX = 0.30

CITY_NAMES = {
    "new_york_city": "NYC",
    "new_york":      "NYC",
    "london":        "London",
    "chicago":       "Chicago",
    "hong_kong":     "HongKong",
    "seoul":         "Seoul",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def discover_cities(data_dir: Path) -> list[str]:
    data_dir = data_dir / "polymarket"
    return sorted(f.stem.replace("_snapshots", "")
                  for f in data_dir.glob("*_snapshots.csv"))


def _parse_yes(val) -> float:
    try:
        d = json.loads(str(val).replace("'", '"'))
        return float(d.get("Yes", d.get("yes", np.nan)))
    except Exception:
        return np.nan


def load_snapshots(data_dir: Path, city: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "polymarket" / f"{city}_snapshots.csv")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["end_date_iso"]   = pd.to_datetime(df["end_date_iso"],   utc=True, errors="coerce")
    df["yes_prob"]       = df["outcome_probs_json"].apply(_parse_yes)
    df["city"]           = city
    # deduplicate: keep freshest row per (condition_id, fetched_bucket)
    df["fetch_bucket"]   = df["fetched_at_utc"].dt.floor("10min")
    df = (df.sort_values("fetched_at_utc")
            .groupby(["condition_id", "fetch_bucket"], as_index=False)
            .last())
    return df


def load_daily(data_dir: Path, city: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "weather" / f"{city}_daily.csv")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    return df


def load_hourly(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    p = data_dir / "weather" / f"{city}_hourly.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["datetime_utc"]   = pd.to_datetime(df["datetime_utc"],   utc=True, errors="coerce")
    return df


def load_ensemble(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    """Load ensemble CSV if available. Returns None if not yet fetched."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    p = data_dir / "weather" / f"{slug}_ensemble.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    for col in ["ens_mean", "ens_std", "ens_p10", "ens_p25",
                "ens_median", "ens_p75", "ens_p90", "ens_spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION PARSER
# ══════════════════════════════════════════════════════════════════════════════

# ── Celsius patterns ────────────────────────────────────────────────────────
_RE_GTE_C   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:higher|above|more)",   re.I)
_RE_LTE_C   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:lower|below|less)",    re.I)
_RE_EXACT_C = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+on\b",                         re.I)
# "at least X°C", "exceed X°C", "reach X°C", "X°C or above/higher"
_RE_GTE_C_2 = re.compile(
    r"(?:at\s+least|exceed[s]?|reach(?:es)?)\s+(-?\d+(?:\.\d+)?)\s*°?\s*C", re.I)
_RE_GTE_C_3 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:above|higher|more)",        re.I)
_RE_LTE_C_2 = re.compile(
    r"(?:at\s+most|no\s+more\s+than)\s+(-?\d+(?:\.\d+)?)\s*°?\s*C",   re.I)
_RE_LTE_C_3 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:below|lower|less)",         re.I)

# ── Fahrenheit patterns ──────────────────────────────────────────────────────
_RE_GTE_F   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:higher|above|more)",   re.I)
_RE_LTE_F   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:lower|below|less)",    re.I)
_RE_EXACT_F = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+on\b",                         re.I)
_RE_RANGE_F = re.compile(r"between\s+(\d+)-(\d+)\s*°?\s*F",            re.I)
_RE_GTE_F_2 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:above|higher|more)",        re.I)
_RE_LTE_F_2 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:below|lower|less)",         re.I)

# 0.5°F converted to °C — used as bin half-width for Fahrenheit markets
_HALF_WIDTH_F_IN_C = 0.5 * 5.0 / 9.0   # ≈ 0.2778°C
_HALF_WIDTH_C      = 0.5                # standard Celsius bin


def _f2c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def parse_question(q: str) -> Optional[dict]:
    """
    Returns: {
        condition:  'exact'|'gte'|'lte'|'range',
        temp_c:     float,          # target / midpoint in °C
        half_width: float,          # bin half-width in °C (0.5 for °C, 0.278 for °F)
        temp_lo:    float (range only),
        temp_hi:    float (range only),
    }
    Tries progressively broader patterns so the most specific match wins.
    """
    # ── Celsius gte ───────────────────────────────────────────────────────────
    for pat in (_RE_GTE_C, _RE_GTE_C_2, _RE_GTE_C_3):
        m = pat.search(q)
        if m:
            return {"condition": "gte", "temp_c": float(m.group(1)),
                    "half_width": _HALF_WIDTH_C}
    # ── Celsius lte ───────────────────────────────────────────────────────────
    for pat in (_RE_LTE_C, _RE_LTE_C_2, _RE_LTE_C_3):
        m = pat.search(q)
        if m:
            return {"condition": "lte", "temp_c": float(m.group(1)),
                    "half_width": _HALF_WIDTH_C}
    # ── Celsius exact ─────────────────────────────────────────────────────────
    m = _RE_EXACT_C.search(q)
    if m:
        return {"condition": "exact", "temp_c": float(m.group(1)),
                "half_width": _HALF_WIDTH_C}
    # ── Fahrenheit range ──────────────────────────────────────────────────────
    m = _RE_RANGE_F.search(q)
    if m:
        lo, hi = _f2c(float(m.group(1))), _f2c(float(m.group(2)))
        return {"condition": "range", "temp_c": (lo + hi) / 2,
                "temp_lo": lo, "temp_hi": hi, "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit gte ────────────────────────────────────────────────────────
    for pat in (_RE_GTE_F, _RE_GTE_F_2):
        m = pat.search(q)
        if m:
            return {"condition": "gte", "temp_c": _f2c(float(m.group(1))),
                    "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit lte ────────────────────────────────────────────────────────
    for pat in (_RE_LTE_F, _RE_LTE_F_2):
        m = pat.search(q)
        if m:
            return {"condition": "lte", "temp_c": _f2c(float(m.group(1))),
                    "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit exact ──────────────────────────────────────────────────────
    m = _RE_EXACT_F.search(q)
    if m:
        return {"condition": "exact", "temp_c": _f2c(float(m.group(1))),
                "half_width": _HALF_WIDTH_F_IN_C}
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ENGINE — Student-t with adaptive sigma
# ══════════════════════════════════════════════════════════════════════════════


def _cdf(x: float, mu: float, sigma: float, nu: float) -> float:
    """Student-t CDF; falls back to Gaussian when nu>30."""
    import scipy.stats
    if nu > 30:
        return float(scipy.stats.norm.cdf(x, loc=mu, scale=sigma))
    return float(student_t.cdf((x - mu) / sigma, df=nu))


def _bin_prob(temp_c: float, mu: float, sigma: float, nu: float,
              half_width: float = 0.5) -> float:
    """P(temp_c - half_width < actual <= temp_c + half_width).
    Use half_width=0.278 for Fahrenheit markets (0.5°F in °C)."""
    import scipy.stats
    upper_bound = temp_c + half_width
    lower_bound = temp_c - half_width
    if nu > 30:
        prob = scipy.stats.norm.cdf(upper_bound, loc=mu, scale=sigma) - scipy.stats.norm.cdf(lower_bound, loc=mu, scale=sigma)
        return float(prob)
    return _cdf(upper_bound, mu, sigma, nu) - _cdf(lower_bound, mu, sigma, nu)


def _condition_prob(parsed: dict, mu: float, sigma: float, nu: float) -> float:
    c  = parsed["condition"]
    t  = parsed["temp_c"]
    hw = parsed.get("half_width", 0.5)
    if c == "exact":  return _bin_prob(t, mu, sigma, nu, hw)
    if c == "gte":    return 1.0 - _cdf(t - hw, mu, sigma, nu)
    if c == "lte":    return _cdf(t + hw, mu, sigma, nu)
    if c == "range":  return (_cdf(parsed["temp_hi"], mu, sigma, nu) -
                              _cdf(parsed["temp_lo"], mu, sigma, nu))
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 1 — FORECAST MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum(daily_df: pd.DataFrame, target_date, fetch_time) -> dict:
    """
    Returns momentum metrics for forecast of target_date as seen up to fetch_time.
    Keys: ema_momentum, total_drift, last_delta, n_snaps, forecast_variance
    """
    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)

    sub = daily_df[
        (daily_df["date_local"].dt.normalize() == td) &
        (daily_df["fetched_at_utc"] <= fetch_time)
    ].sort_values("fetched_at_utc")

    if len(sub) < 2:
        return {"ema_momentum": 0.0, "total_drift": 0.0,
                "last_delta": 0.0, "n_snaps": len(sub), "forecast_variance": 0.0}

    temps = sub["temp_max_c"].values.astype(float)
    deltas = np.diff(temps)
    ema = pd.Series(deltas).ewm(span=MOMENTUM_EMA_SPAN, adjust=False).mean().iloc[-1]

    return {
        "ema_momentum":      float(ema),
        "total_drift":       float(temps[-1] - temps[0]),
        "last_delta":        float(deltas[-1]),
        "n_snaps":           len(temps),
        "forecast_variance": float(np.var(temps)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 2 — DIURNAL SPREAD → SIGMA BOOST
# ══════════════════════════════════════════════════════════════════════════════

def spread_sigma_boost(daily_df: pd.DataFrame, target_date, fetch_time,
                        baseline_spread: float = 8.0) -> float:
    """
    Returns additive sigma boost based on forecast diurnal spread (max-min).
    Wider spread = more convective uncertainty = heavier tails needed.
    """
    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)

    sub = daily_df[
        (daily_df["date_local"].dt.normalize() == td) &
        (daily_df["fetched_at_utc"] <= fetch_time) &
        daily_df["temp_min_c"].notna()
    ].sort_values("fetched_at_utc")

    if sub.empty:
        return 0.0

    last = sub.iloc[-1]
    if pd.isna(last.get("temp_min_c", np.nan)):
        return 0.0

    spread = float(last["temp_max_c"]) - float(last["temp_min_c"])
    boost  = SPREAD_SIGMA_WEIGHT * max(0.0, (spread / baseline_spread) - 1.0)
    return round(boost, 3)


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 6 — VOLUME RECENCY
# ══════════════════════════════════════════════════════════════════════════════

def volume_recency_signal(row: pd.Series) -> float:
    """0.0 – 1.0. >0.65 = informed traders active."""
    vol_total = float(row.get("volume_usdc", 0) or 0)
    vol_24h   = float(row.get("volume_24h_usdc", 0) or 0)
    if vol_total < 1:
        return 0.0
    return min(vol_24h / vol_total, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 7 — FORECAST CONVERGENCE VARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def forecast_convergence(daily_df: pd.DataFrame, target_date, fetch_time) -> float:
    """
    Variance of forecast snapshots for this target date.
    High variance = model disagreement = market likely stale or about to reprice.
    """
    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)

    sub = daily_df[
        (daily_df["date_local"].dt.normalize() == td) &
        (daily_df["fetched_at_utc"] <= fetch_time)
    ]["temp_max_c"].dropna()

    return float(sub.var()) if len(sub) >= 2 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 8 — MARKET STALENESS
# ══════════════════════════════════════════════════════════════════════════════

def market_staleness(snap_history: pd.DataFrame, condition_id: str,
                     fetch_time, days_ahead: float = 3.0) -> dict:
    """
    Detect whether a market has been repriced recently.
    Staleness threshold scales with horizon: close-to-expiry markets should
    reprice more often, so we require shorter inactivity to flag as stale.

    Threshold: max(2h, days_ahead * 1.5h) up to STALE_HOURS cap.
    """
    hist = snap_history[snap_history["condition_id"] == condition_id].copy()
    hist = hist[hist["fetched_at_utc"] <= fetch_time].sort_values("fetched_at_utc")

    if len(hist) < 2:
        return {"hours_since_move": 0.0, "is_stale": False,
                "last_move": 0.0, "market_momentum": 0.0}

    probs     = hist["yes_prob"].values.astype(float)
    # Keep timestamps as UTC-aware pd.Timestamp to avoid tz-naive subtraction
    ts_series = hist["fetched_at_utc"].reset_index(drop=True)

    # Find last significant move
    moves  = np.abs(np.diff(probs))
    sig_ix = np.where(moves >= STALE_MOVE_THRESHOLD)[0]
    if len(sig_ix) == 0:
        last_move_time = ts_series.iloc[0]
        last_move_size = 0.0
    else:
        last_move_time = ts_series.iloc[sig_ix[-1] + 1]
        last_move_size = float(np.diff(probs)[sig_ix[-1]])

    ft = pd.Timestamp(fetch_time)
    if ft.tzinfo is None:
        ft = ft.tz_localize("UTC")
    lmt = pd.Timestamp(last_move_time)
    if lmt.tzinfo is None:
        lmt = lmt.tz_localize("UTC")

    dt_h = (ft - lmt).total_seconds() / 3600

    # Horizon-relative staleness threshold
    stale_threshold = min(STALE_HOURS, max(2.0, days_ahead * 1.5))
    is_stale = dt_h > stale_threshold

    # Market momentum: recent directional change
    mkt_mom = float(probs[-1] - probs[max(-4, -len(probs))])

    return {
        "hours_since_move": round(dt_h, 2),
        "is_stale":         is_stale,
        "last_move":        round(last_move_size, 4),
        "market_momentum":  round(mkt_mom, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 4 + 5 — CONSTRAINED PMF RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketBin:
    condition_id: str
    question: str
    condition: str         # exact / gte / lte / range
    temp_c: float
    half_width: float      # bin half-width in °C (0.5 for °C, 0.278 for °F)
    yes_prob: float        # raw Polymarket
    liquidity: float
    volume_24h: float
    volume_total: float


def reconstruct_pmf(bins: list[MarketBin],
                    mu: float, sigma: float, nu: float,
                    temps_range: tuple = (-5, 40)) -> tuple[dict, dict, float]:
    """
    Build market PMF and forecast PMF over a shared temperature support,
    respecting gte/lte bins as constraints.

    Returns:
        market_pmf  : {temp_c -> normalised probability}
        forecast_pmf: {temp_c -> normalised probability}
        consistency : how well bins sum to 1.0 (1.0 = perfect)
    """
    exact = [b for b in bins if b.condition == "exact"]
    gte   = [b for b in bins if b.condition == "gte"]
    lte   = [b for b in bins if b.condition == "lte"]

    # ── Step 1: exact bins → raw market PMF backbone ──────────────────────
    if not exact:
        return {}, {}, 0.0

    raw_mkt  = {b.temp_c: b.yes_prob for b in exact}
    raw_sum  = sum(raw_mkt.values())
    if raw_sum < 0.05:
        return {}, {}, 0.0

    # ── Step 2: measure internal consistency ──────────────────────────────
    # gte bin P(≥T) should ≈ sum of exact bins at temps ≥ T
    consistency_err = 0.0
    n_constraints   = 0
    for b in gte:
        implied = sum(v for k, v in raw_mkt.items() if k >= b.temp_c)
        consistency_err += abs(implied - b.yes_prob)
        n_constraints   += 1
    for b in lte:
        implied = sum(v for k, v in raw_mkt.items() if k <= b.temp_c)
        consistency_err += abs(implied - b.yes_prob)
        n_constraints   += 1

    consistency = 1.0 - (consistency_err / max(n_constraints, 1))

    # ── Step 3: estimate missing bin mass ─────────────────────────────────
    # All observed exact temps
    obs_temps = sorted(raw_mkt.keys())

    # Expected total from boundary bins
    max_gte_temp = max((b.temp_c for b in gte), default=None)
    min_lte_temp = min((b.temp_c for b in lte), default=None)

    # Upper residual: P(> max_exact_temp) from gte bin or forecast tail
    if max_gte_temp is not None:
        upper_residual = max(0.0, gte[0].yes_prob -
                             sum(v for k, v in raw_mkt.items() if k >= max_gte_temp))
    else:
        # Estimate from forecast: probability beyond highest exact bin
        upper_residual = max(0.0, 1.0 - _cdf(max(obs_temps) + 0.5, mu, sigma, nu))

    # Lower residual: P(< min_exact_temp) from lte bin or forecast tail
    if min_lte_temp is not None:
        lower_residual = max(0.0, lte[0].yes_prob -
                             sum(v for k, v in raw_mkt.items() if k <= min_lte_temp))
    else:
        lower_residual = max(0.0, _cdf(min(obs_temps) - 0.5, mu, sigma, nu))

    # Total available probability for exact bins + estimated residuals
    total_prob = raw_sum + upper_residual + lower_residual
    total_prob = max(total_prob, raw_sum)  # can't be less than what we see

    # ── Step 4: normalise market PMF ──────────────────────────────────────
    market_pmf = {k: v / total_prob for k, v in raw_mkt.items()}

    # ── Step 5: forecast PMF over same support ────────────────────────────
    fc_raw = {t: _bin_prob(t, mu, sigma, nu) for t in obs_temps}
    fc_sum = sum(fc_raw.values()) + upper_residual + lower_residual
    fc_sum = max(fc_sum, sum(fc_raw.values()))
    forecast_pmf = {t: v / fc_sum for t, v in fc_raw.items()}

    return market_pmf, forecast_pmf, max(0.0, min(1.0, consistency))


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE EDGE + SIGNAL DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Opportunity:
    # Identity
    city:           str
    city_raw:       str
    condition_id:   str
    question:       str
    target_date:    str
    fetched_at:     pd.Timestamp
    days_ahead:     float

    # Forecast
    forecast_mu:    float
    forecast_sigma: float
    forecast_nu:    float
    sigma_boost:    float   # from diurnal spread (α2)
    sigma_source:   str     # "ensemble" | "nwp_table"

    # Probabilities
    forecast_prob:  float   # our estimate
    market_prob:    float   # normalised market PMF value
    market_prob_raw:float   # raw Polymarket yes_prob

    # Edge
    edge:           float   # forecast - market (signed)
    abs_edge:       float
    bet_side:       str     # "Yes" | "No"
    our_prob:       float   # prob of our side
    their_prob:     float   # what market is offering

    # Alpha signals
    ema_momentum:   float   # α1
    total_drift:    float   # α1
    last_delta:     float   # α1
    forecast_var:   float   # α7
    volume_recency: float   # α6
    market_mkt_mom: float   # α8
    hours_since_move:float  # α8
    is_stale:       bool    # α8
    pmf_consistency:float   # α4/5
    pmf_sum_dev:    float   # α5: abs(raw_sum - 1.0)

    # Market meta
    liquidity:      float
    volume_24h:     float
    n_exact_bins:   int
    market_mode_c:  float
    mode_shift_c:   float   # forecast_mu - market_mode
    bin_temp_c:     float   # actual temperature the question asks about

    # Composite score (computed post-init)
    alpha_score:    float = 0.0
    kelly:          float = 0.0
    ev_per_dollar:  float = 0.0
    group_key:      str   = ""


def _score_opportunity(opp: Opportunity) -> float:
    """
    Composite alpha score combining all signals.
    Higher = better opportunity.
    """
    # Base: edge magnitude
    score = opp.abs_edge

    # α1 Momentum boost: if forecast moved in direction of our bet
    if opp.bet_side == "Yes" and opp.ema_momentum > MOMENTUM_THRESHOLD:
        score *= 1.0 + min(opp.ema_momentum / 0.5, 0.4)
    elif opp.bet_side == "No" and opp.ema_momentum < -MOMENTUM_THRESHOLD:
        score *= 1.0 + min(abs(opp.ema_momentum) / 0.5, 0.4)

    # α5 Consistency bonus: incoherent market = better edge
    if opp.pmf_sum_dev > 0.15:
        score *= 1.0 + min(opp.pmf_sum_dev * 0.5, 0.3)

    # α6 Volume recency — high recent volume means informed traders may have
    # already repriced toward fair value, so our edge may have closed.
    if opp.volume_recency >= INFORMED_RECENCY:
        score *= 0.90   # slight discount: edge may be partially arbitraged

    # α7 High forecast variance → uncertain forecast → penalise slightly
    score *= max(0.6, 1.0 - opp.forecast_var * 0.1)

    # α8 Staleness penalty: stale markets might gap badly on fill
    if opp.is_stale:
        score *= 0.85

    # Liquidity-weighted: larger market = easier fill
    liq_factor = min(log(max(opp.liquidity, 10) / 100 + 1) / log(100), 1.0)
    score *= (0.5 + 0.5 * liq_factor)

    # Horizon decay: edges closer to expiry are more reliable
    score /= (opp.days_ahead + 0.5)

    return round(score, 6)


def _kelly_size(opp: Opportunity, kelly_fraction: float = KELLY_FRACTION, fee: float = FEE_RATE) -> float:
    """
    Fractional Kelly with fee adjustment.

    On Polymarket you pay `their_prob` per share and receive $1 if correct.
    With a fee on the net profit, effective payout per share = (1 - fee).
    So net odds b = ((1 - their_prob) / their_prob) * (1 - fee).
    """
    p = opp.their_prob
    if p <= 1e-4 or p >= (1.0 - 1e-4):
        return 0.0
    b = ((1.0 - p) / p) * (1.0 - fee)
    if b <= 0:
        return 0.0
    q   = opp.our_prob
    raw = kelly_fraction * (b * q - (1.0 - q)) / b
    return float(np.clip(raw, 0.0, MAX_KELLY_PER_BET))


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA 9 — CORRELATED BET GROUPING
# ══════════════════════════════════════════════════════════════════════════════

def apply_group_kelly_cap(opps: list[Opportunity],
                           bankroll: float) -> list[Opportunity]:
    """
    For each (city, target_date) group, scale down kelly fractions so total
    group exposure ≤ MAX_KELLY_PER_GROUP * bankroll.
    Also flags bets that are near-redundant (betting Yes on 14°C and 15°C and ≥14°C).
    """
    from collections import defaultdict

    groups: dict[str, list[Opportunity]] = defaultdict(list)
    for o in opps:
        groups[o.group_key].append(o)

    result = []
    for gkey, group in groups.items():
        # Sort by alpha_score descending
        group.sort(key=lambda x: x.alpha_score, reverse=True)

        # Remove near-duplicate bets: same side, adjacent temp bins within 1°C.
        # Key on actual bin temperature (rounded to nearest degree), not probability.
        seen_temps = set()
        filtered   = []
        for o in group:
            key = (o.bet_side, round(o.bin_temp_c))
            if key in seen_temps:
                continue
            seen_temps.add(key)
            filtered.append(o)

        # Scale kelly to group cap
        total_raw_kelly = sum(o.kelly for o in filtered)
        if total_raw_kelly > MAX_KELLY_PER_GROUP:
            scale = MAX_KELLY_PER_GROUP / total_raw_kelly
            for o in filtered:
                o.kelly = round(o.kelly * scale, 4)

        result.extend(filtered)

    return result


def apply_portfolio_cap(opps: list[Opportunity]) -> list[Opportunity]:
    """
    Hard cap on total portfolio Kelly across all groups.
    After group caps are applied, total exposure can still far exceed bankroll
    (e.g. 5 cities × 7 dates × 20% = 700%). This enforces MAX_TOTAL_KELLY.
    """
    total = sum(o.kelly for o in opps)
    if total > MAX_TOTAL_KELLY:
        scale = MAX_TOTAL_KELLY / total
        for o in opps:
            o.kelly = round(o.kelly * scale, 4)
    return opps


def fetch_live_prices(condition_ids: list[str]) -> dict[str, float]:
    """
    Fetch current Yes-side prices for a list of condition_ids from the Gamma API.
    Returns {condition_id: current_yes_prob}.
    Uses GET /markets/{condition_id} per market (Gamma API doesn't support bulk lookup).
    """
    import requests as _req
    import time as _time

    if not condition_ids:
        return {}

    prices: dict[str, float] = {}

    for cid in condition_ids:
        try:
            resp = _req.get(
                "https://gamma-api.polymarket.com/markets",
                params={"condition_ids": cid},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue
            mkt = data[0] if isinstance(data, list) else data
            raw = mkt.get("outcomePrices")
            if not raw:
                continue
            probs = json.loads(raw) if isinstance(raw, str) else raw
            yes_p = float(probs[0])
            prices[cid] = yes_p
            _time.sleep(0.1)   # polite: ~10 req/s
        except Exception as exc:
            print(f"  [WARN] live price fetch failed for {cid[:16]}…: {exc}")

    return prices


def check_orderbook_vwap(condition_id: str, bet_side: str, target_size_usdc: float) -> float:
    """
    Gets the token_id for the bet_side, pulls the CLOB L2 orderbook,
    and calculates the effective average price (VWAP) for the target size.
    Returns the VWAP price, or 1.0 if insufficient liquidity.
    """
    import requests as _req
    import time as _time
    import json as _json
    try:
        _time.sleep(0.2)
        
        # 1. Get Token ID from Gamma
        gamma_req = _req.get("https://gamma-api.polymarket.com/markets", params={"condition_ids": condition_id})
        gamma_req.raise_for_status()
        gamma_resp = gamma_req.json()
        
        if not gamma_resp: 
            print(f"  [DEBUG] Gamma API returned empty for condition {condition_id[:8]}...")
            return 1.0
        
        market = gamma_resp[0] if isinstance(gamma_resp, list) else gamma_resp
        
        raw_outcomes = market.get("outcomes", "[]")
        outcomes = _json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        
        raw_tokens = market.get("clobTokenIds", "[]")
        clob_tokens = _json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        
        token_id = None
        for i, outcome in enumerate(outcomes):
            if outcome == bet_side and i < len(clob_tokens):
                token_id = clob_tokens[i]
                break
                
        if not token_id: 
            print(f"  [DEBUG] No token_id found for side '{bet_side}' in condition {condition_id[:8]}...")
            return 1.0
        
        _time.sleep(0.2)
        
        # 2. Get Order Book from CLOB
        clob_req = _req.get("https://clob.polymarket.com/book", params={"token_id": token_id})
        clob_req.raise_for_status()
        clob_resp = clob_req.json()
        
        asks = clob_resp.get("asks", [])
        if not asks:
            print(f"  [DEBUG] CLOB returned 0 asks (empty orderbook) for token {token_id[:8]}...")
            return 1.0
            
        asks.sort(key=lambda x: float(x["price"]))
        
        filled_usdc = 0.0
        total_shares = 0.0
        
        for ask in asks:
            price = float(ask["price"])
            size_shares = float(ask["size"])
            cost_usdc = size_shares * price
            
            if filled_usdc + cost_usdc >= target_size_usdc:
                needed_usdc = target_size_usdc - filled_usdc
                total_shares += needed_usdc / price
                filled_usdc += needed_usdc
                break
            else:
                total_shares += size_shares
                filled_usdc += cost_usdc
        
        if filled_usdc < target_size_usdc or total_shares == 0:
            print(f"  [DEBUG] Not enough liquidity to fill ${target_size_usdc:.2f}. Only found ${filled_usdc:.2f}")
            return 1.0  
            
        return round(target_size_usdc / total_shares, 4)
    except Exception as e:
        print(f"  [WARN] Orderbook check failed for {condition_id[:8]}...: {e}")
        return 1.0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_city(data_dir: Path, city: str,
                 min_edge: float = MIN_EDGE,
                 min_liq:  float = MIN_LIQUIDITY,
                 use_ml: bool = True,
                 kelly_fraction: float = KELLY_FRACTION,
                 sigma_threshold: float = 1.2,
                 conflict_gating: bool = True) -> list[Opportunity]:

    try:
        snap_df  = load_snapshots(data_dir, city)
        daily_df = load_daily(data_dir, city)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return []

    # Load ensemble — used to replace the hardcoded sigma lookup table.
    # Falls back gracefully if not yet fetched (run main.py to populate).
    ens_df = load_ensemble(data_dir, city)
    if ens_df is not None and not ens_df.empty:
        print(f"  Ensemble loaded: {len(ens_df)} rows ({ens_df['fetched_at_utc'].max()})")
    else:
        print(f"  Ensemble not available — falling back to NWP_PARAMS table")

    results: list[Opportunity] = []
    cheapest_entries = {}

    # Group by (target_date, fetch_bucket) to process each snapshot window chronologically
    snap_df["end_date_norm"] = snap_df["end_date_iso"].dt.normalize()
    snap_df = snap_df.sort_values("fetched_at_utc")
    groups = snap_df.groupby(["end_date_norm", "fetch_bucket"], sort=False)
    print(f"  {city}: {len(groups)} (date × snapshot) groups, "
          f"{snap_df['condition_id'].nunique()} unique markets", flush=True)

    ml_predictor = MLCalibratorPredictor(use_ml=use_ml, sigma_threshold=sigma_threshold)
    ens_predictor = EnsemblePredictor()

    for (target_date_ts, fetch_bucket_ts), group in groups:
        # Representative fetch time
        fetch_time = group["fetched_at_utc"].max()

        # Days ahead
        if pd.isna(target_date_ts):
            continue
        days_fwd = max(0.0,
                       (target_date_ts.to_pydatetime() -
                        fetch_time.to_pydatetime()).total_seconds() / 86400)

        s_boost = spread_sigma_boost(daily_df, target_date_ts, fetch_time)

        # ── Predict distribution parameters ────────────────────────────────
        try:
            dist_ml = ml_predictor.predict_distribution(
                city=city,
                target_date=target_date_ts,
                fetch_time=fetch_time,
                days_ahead=days_fwd,
                daily_df=daily_df,
                ens_df=ens_df
            )
            mu_ml = dist_ml.mu
            sigma_ml = dist_ml.sigma
            nu_ml = dist_ml.nu
            sigma_source = dist_ml.source
        except Exception:
            continue

        try:
            dist_ens = ens_predictor.predict_distribution(
                city=city,
                target_date=target_date_ts,
                fetch_time=fetch_time,
                days_ahead=days_fwd,
                daily_df=daily_df,
                ens_df=ens_df
            )
            mu_ens = dist_ens.mu
            sigma_ens = dist_ens.sigma
            nu_ens = dist_ens.nu
        except Exception:
            mu_ens, sigma_ens, nu_ens = mu_ml, sigma_ml, nu_ml

        # Keep default mu, sigma, nu for PMF reconstruction and backward compatibility
        mu = mu_ml
        sigma = sigma_ml
        nu = nu_ml

        # ── Alpha signals ──────────────────────────────────────────────────
        mom    = compute_momentum(daily_df, target_date_ts, fetch_time)   # α1
        fc_var = forecast_convergence(daily_df, target_date_ts, fetch_time)  # α7

        # ── Parse all bins in this group ───────────────────────────────────
        market_bins: list[MarketBin] = []
        for _, row in group.iterrows():
            parsed = parse_question(str(row.get("question", "")))
            if parsed is None:
                continue
            yp = row["yes_prob"]
            if np.isnan(yp):
                continue
            market_bins.append(MarketBin(
                condition_id = str(row["condition_id"]),
                question     = str(row.get("question", "")),
                condition    = parsed["condition"],
                temp_c       = parsed["temp_c"],
                half_width   = parsed.get("half_width", 0.5),
                yes_prob     = float(yp),
                liquidity    = float(row.get("liquidity_usdc", 0) or 0),
                volume_24h   = float(row.get("volume_24h_usdc", 0) or 0),
                volume_total = float(row.get("volume_usdc", 0) or 0),
            ))

        if not market_bins:
            continue

        # ── Reconstruct PMF (α4/5) ─────────────────────────────────────────
        market_pmf, forecast_pmf, consistency = reconstruct_pmf(
            market_bins, mu, sigma, nu)

        exact_bins = [b for b in market_bins if b.condition == "exact"]
        raw_sum    = sum(b.yes_prob for b in exact_bins)
        pmf_dev    = abs(raw_sum - 1.0)

        # Mode of market
        market_mode_c = (max(exact_bins, key=lambda b: b.yes_prob).temp_c
                         if exact_bins else mu)

        # ── Per-bin edge computation ───────────────────────────────────────
        # Use RAW probabilities:
        #   f_prob_raw = our model's P(temp = X)  from Student-t distribution
        #   m_prob_raw = b.yes_prob               the actual Polymarket price
        # Do NOT use normalized PMF values for edge/Kelly — those are for the
        for b in exact_bins:
            if b.liquidity < min_liq:
                continue

            # Compute raw probabilities from both models
            f_prob_ml = _bin_prob(b.temp_c, mu_ml, sigma_ml, nu_ml, b.half_width)
            f_prob_ens = _bin_prob(b.temp_c, mu_ens, sigma_ens, nu_ens, b.half_width)

            m_prob_raw = b.yes_prob   # actual market price you pay

            # Skip near-settled markets (price approaching 0 or 1)
            if m_prob_raw < MIN_MARKET_PRICE or m_prob_raw > (1.0 - MIN_MARKET_PRICE):
                continue

            # Compute edges and sides for both models
            edge_ml = f_prob_ml - m_prob_raw
            edge_ens = f_prob_ens - m_prob_raw
            
            bet_side_ml = "Yes" if edge_ml > 0 else "No"
            bet_side_ens = "Yes" if edge_ens > 0 else "No"

            # Conflict Gating: skip if ML and Ensemble bet sides disagree
            if conflict_gating and bet_side_ml != bet_side_ens:
                continue

            bet_side = bet_side_ml
            
            # Sizing: use the maximum of the two probabilities for the bet side (ML vs Ensemble)
            our_prob_ml = f_prob_ml if bet_side == "Yes" else (1.0 - f_prob_ml)
            our_prob_ens = f_prob_ens if bet_side == "Yes" else (1.0 - f_prob_ens)
            our_prob = max(our_prob_ml, our_prob_ens)
            
            # Reconstruct f_prob_raw (our effective probability) and edge for Kelly sizing
            f_prob_raw = our_prob if bet_side == "Yes" else (1.0 - our_prob)
            their_prob = m_prob_raw if bet_side == "Yes" else (1.0 - m_prob_raw)
            edge = our_prob - their_prob

            # Minimum edge check on the combined maximum probability
            if abs(edge) < min_edge:
                continue

            # Lock-In: preserve the early cheap/high-leverage bets in the backtester
            prev_price = cheapest_entries.get(b.condition_id)
            if prev_price is not None:
                # If we entered cheap (<30% price) and the price went up by >10% (more than 10 cents), skip
                if prev_price < 0.30 and their_prob > (prev_price + 0.10):
                    continue
                # General slip: if the price moved against us by >15% (15 cents), skip
                if their_prob > (prev_price + 0.15):
                    continue
            
            # Update cheapest entry price
            if prev_price is None or their_prob < prev_price:
                cheapest_entries[b.condition_id] = their_prob

            # PMF-normalized values kept for reporting/consistency signal
            m_prob = market_pmf.get(b.temp_c, m_prob_raw)
            f_prob = forecast_pmf.get(b.temp_c, f_prob_raw)

            # α6, α8
            vol_rec = volume_recency_signal(
                group[group["condition_id"] == b.condition_id].iloc[0]
                if len(group[group["condition_id"] == b.condition_id]) else pd.Series())
            stale   = market_staleness(snap_df, b.condition_id, fetch_time, days_fwd)

            opp = Opportunity(
                city           = CITY_NAMES.get(city, city),
                city_raw       = city,
                condition_id   = b.condition_id,
                question       = b.question[:80],
                target_date    = str(target_date_ts.date()),
                fetched_at     = fetch_time,
                days_ahead     = round(days_fwd, 2),
                forecast_mu    = round(mu, 2),
                forecast_sigma = round(sigma, 3),
                forecast_nu    = round(nu, 1),
                sigma_boost    = round(s_boost, 3),
                sigma_source   = sigma_source,
                forecast_prob  = round(f_prob_raw, 4),
                market_prob    = round(m_prob, 4),
                market_prob_raw= round(b.yes_prob, 4),
                edge           = round(edge, 4),
                abs_edge       = round(abs(edge), 4),
                bet_side       = bet_side,
                our_prob       = round(our_prob, 4),
                their_prob     = round(their_prob, 4),
                ema_momentum   = round(mom["ema_momentum"], 4),
                total_drift    = round(mom["total_drift"], 3),
                last_delta     = round(mom["last_delta"], 3),
                forecast_var   = round(fc_var, 4),
                volume_recency = round(vol_rec, 3),
                market_mkt_mom = stale["market_momentum"],
                hours_since_move=stale["hours_since_move"],
                is_stale       = stale["is_stale"],
                pmf_consistency= round(consistency, 4),
                pmf_sum_dev    = round(pmf_dev, 4),
                liquidity      = b.liquidity,
                volume_24h     = b.volume_24h,
                n_exact_bins   = len(exact_bins),
                market_mode_c  = market_mode_c,
                mode_shift_c   = round(mu - market_mode_c, 2),
                bin_temp_c     = b.temp_c,
                group_key      = f"{city}|{target_date_ts.date()}",
            )
            opp.alpha_score = _score_opportunity(opp)
            opp.kelly       = _kelly_size(opp, kelly_fraction=kelly_fraction)
            opp.ev_per_dollar = round(our_prob / their_prob - 1.0, 4)
            results.append(opp)

        # ── Boundary bins (gte / lte) — direct CDF comparison ─────────────
        for b in market_bins:
            if b.condition not in ("gte", "lte"):
                continue
            if b.liquidity < min_liq:
                continue

            parsed  = {"condition": b.condition, "temp_c": b.temp_c, "half_width": b.half_width}
            f_prob_ml = _condition_prob(parsed, mu_ml, sigma_ml, nu_ml)
            f_prob_ens = _condition_prob(parsed, mu_ens, sigma_ens, nu_ens)

            m_prob_raw = b.yes_prob

            # Skip near-settled markets
            if m_prob_raw < MIN_MARKET_PRICE or m_prob_raw > (1.0 - MIN_MARKET_PRICE):
                continue

            # Compute edges and sides
            edge_ml = f_prob_ml - m_prob_raw
            edge_ens = f_prob_ens - m_prob_raw

            bet_side_ml = "Yes" if edge_ml > 0 else "No"
            bet_side_ens = "Yes" if edge_ens > 0 else "No"

            # Conflict Gating: skip if ML and Ensemble bet sides disagree
            if conflict_gating and bet_side_ml != bet_side_ens:
                continue

            bet_side = bet_side_ml

            # Sizing: use the maximum of the two probabilities
            our_prob_ml = f_prob_ml if bet_side == "Yes" else (1.0 - f_prob_ml)
            our_prob_ens = f_prob_ens if bet_side == "Yes" else (1.0 - f_prob_ens)
            our_prob = max(our_prob_ml, our_prob_ens)

            f_prob_raw = our_prob if bet_side == "Yes" else (1.0 - our_prob)
            their_prob = m_prob_raw if bet_side == "Yes" else (1.0 - m_prob_raw)
            edge = our_prob - their_prob

            # Minimum edge check
            if abs(edge) < min_edge:
                continue

            vol_rec = volume_recency_signal(
                group[group["condition_id"] == b.condition_id].iloc[0]
                if len(group[group["condition_id"] == b.condition_id]) else pd.Series())
            stale   = market_staleness(snap_df, b.condition_id, fetch_time, days_fwd)

            opp = Opportunity(
                city           = CITY_NAMES.get(city, city),
                city_raw       = city,
                condition_id   = b.condition_id,
                question       = b.question[:80],
                target_date    = str(target_date_ts.date()),
                fetched_at     = fetch_time,
                days_ahead     = round(days_fwd, 2),
                forecast_mu    = round(mu, 2),
                forecast_sigma = round(sigma, 3),
                forecast_nu    = round(nu, 1),
                sigma_boost    = round(s_boost, 3),
                sigma_source   = sigma_source,
                forecast_prob  = round(f_prob_raw, 4),
                market_prob    = round(m_prob_raw, 4),
                market_prob_raw= round(m_prob_raw, 4),
                edge           = round(edge, 4),
                abs_edge       = round(abs(edge), 4),
                bet_side       = bet_side,
                our_prob       = round(our_prob, 4),
                their_prob     = round(their_prob, 4),
                ema_momentum   = round(mom["ema_momentum"], 4),
                total_drift    = round(mom["total_drift"], 3),
                last_delta     = round(mom["last_delta"], 3),
                forecast_var   = round(fc_var, 4),
                volume_recency = round(vol_rec, 3),
                market_mkt_mom = stale["market_momentum"],
                hours_since_move=stale["hours_since_move"],
                is_stale       = stale["is_stale"],
                pmf_consistency= round(consistency, 4),
                pmf_sum_dev    = round(pmf_dev, 4),
                liquidity      = b.liquidity,
                volume_24h     = b.volume_24h,
                n_exact_bins   = len(exact_bins),
                market_mode_c  = market_mode_c,
                mode_shift_c   = round(mu - market_mode_c, 2),
                bin_temp_c     = b.temp_c,
                group_key      = f"{city}|{target_date_ts.date()}",
            )
            opp.alpha_score = _score_opportunity(opp)
            opp.kelly       = _kelly_size(opp, kelly_fraction=kelly_fraction)
            opp.ev_per_dollar = round(our_prob / their_prob - 1.0, 4)
            results.append(opp)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_DARK   = "#0d1117"
_AX     = "#161b22"
_BORDER = "#30363d"
_MUTED  = "#8b949e"
_GREEN  = "#3fb950"
_RED    = "#f85149"
_BLUE   = "#58a6ff"
_YELLOW = "#e3b341"
_TAB    = plt.cm.tab10.colors


def _styled_ax(ax):
    ax.set_facecolor(_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    return ax


def _dark_fig(nrows=1, ncols=1, figsize=(14, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(_DARK)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        _styled_ax(ax)
    return fig, axes


def plot_pmf_comparison(opps_df: pd.DataFrame, all_bins_df: pd.DataFrame,
                         output_dir: Path, top_n=6):
    """Side-by-side forecast vs market PMF for highest-dislocation days."""
    if opps_df.empty:
        return

    # pick top days by max abs_edge
    top_keys = (opps_df.groupby(["city", "target_date", "fetched_at"])
                ["abs_edge"].max().nlargest(top_n).index)

    ncols = min(3, len(top_keys))
    nrows = (len(top_keys) + ncols - 1) // ncols
    fig, axes = _dark_fig(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes_flat  = np.array(axes).flat

    for idx, (city, tdate, ftime) in enumerate(top_keys):
        ax  = next(axes_flat)
        sub = all_bins_df[(all_bins_df["city"] == city) &
                          (all_bins_df["target_date"] == tdate) &
                          (all_bins_df["fetched_at"] == ftime)].sort_values("bin_temp_c")
        if sub.empty:
            ax.set_visible(False); continue

        x  = np.arange(len(sub))
        w  = 0.38
        ax.bar(x - w/2, sub["market_prob"],   w, color=_BLUE,   alpha=0.85, label="Market")
        ax.bar(x + w/2, sub["forecast_prob"], w, color=_YELLOW, alpha=0.85, label="Forecast")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0f}°" for t in sub["bin_temp_c"]], fontsize=7)
        mu   = sub["forecast_mu"].iloc[0]
        days = sub["days_ahead"].iloc[0]
        mom  = sub["ema_momentum"].iloc[0] if "ema_momentum" in sub.columns else 0
        ax.set_title(
            f"{city} · {tdate}\nμ={mu:.1f}°C  +{days:.1f}d  mom={mom:+.2f}",
            color="white", fontsize=8, pad=3)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d",
                  edgecolor=_BORDER, loc="upper right")

    for ax in axes_flat:
        ax.set_visible(False)

    plt.suptitle("Forecast PMF vs Market PMF — Top Dislocations",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "pmf_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_alpha_dashboard(opps_df: pd.DataFrame, output_dir: Path):
    """3×3 grid of all alpha signal diagnostics."""
    if opps_df.empty:
        return

    fig, axes = _dark_fig(3, 3, figsize=(18, 13))
    c_yes = _GREEN; c_no = _RED

    def colors(col): return [c_yes if s=="Yes" else c_no for s in opps_df[col]]

    # 1. Edge distribution
    ax = axes[0, 0]
    for i, (city, g) in enumerate(opps_df.groupby("city")):
        ax.hist(g["edge"], bins=20, alpha=0.65, color=_TAB[i % 10], label=city)
    ax.axvline(0, color=_MUTED, lw=1, ls="--")
    ax.axvline( MIN_EDGE, color=_YELLOW, lw=1.5, ls=":", label=f"±{MIN_EDGE:.0%}")
    ax.axvline(-MIN_EDGE, color=_YELLOW, lw=1.5, ls=":")
    ax.set_title("Edge Distribution", color="white", fontweight="bold")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 2. α1 Momentum vs edge
    ax = axes[0, 1]
    sc = ax.scatter(opps_df["ema_momentum"], opps_df["edge"],
                    c=opps_df["abs_edge"], cmap="plasma", alpha=0.7,
                    s=40, edgecolors="none")
    ax.axvline(0, color=_MUTED, lw=1, ls="--")
    ax.axhline(0, color=_MUTED, lw=1, ls="--")
    plt.colorbar(sc, ax=ax, label="|edge|").ax.tick_params(colors=_MUTED)
    ax.set_title("α1 Momentum vs Edge", color="white", fontweight="bold")
    ax.set_xlabel("EMA momentum (°C/snap)")
    ax.set_ylabel("edge")

    # 3. α2 Sigma boost distribution
    ax = axes[0, 2]
    ax.hist(opps_df["sigma_boost"], bins=20, color=_BLUE, alpha=0.8)
    ax.set_title("α2 Sigma Boost (diurnal spread)", color="white", fontweight="bold")
    ax.set_xlabel("sigma boost (°C)")

    # 4. α3 forecast_sigma used
    ax = axes[1, 0]
    ax.scatter(opps_df["days_ahead"], opps_df["forecast_sigma"],
               c=[c_yes if s=="Yes" else c_no for s in opps_df["bet_side"]],
               alpha=0.6, s=40, edgecolors="none")
    ax.set_title("α3 Adaptive Sigma vs Horizon", color="white", fontweight="bold")
    ax.set_xlabel("days ahead")
    ax.set_ylabel("sigma (°C)")

    # 5. α5 PMF sum deviation
    ax = axes[1, 1]
    ax.scatter(opps_df["pmf_sum_dev"], opps_df["abs_edge"],
               c=opps_df["pmf_consistency"], cmap="RdYlGn",
               alpha=0.7, s=50, edgecolors="none", vmin=0, vmax=1)
    ax.axvline(PMF_SUM_DEVIATION_MAX, color=_YELLOW, lw=1.5, ls=":",
               label=f"max dev {PMF_SUM_DEVIATION_MAX}")
    ax.set_title("α5 Market Consistency vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("PMF sum deviation")
    ax.set_ylabel("|edge|")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 6. α6 Volume recency
    ax = axes[1, 2]
    ax.scatter(opps_df["volume_recency"], opps_df["abs_edge"],
               c=[c_yes if s=="Yes" else c_no for s in opps_df["bet_side"]],
               alpha=0.7, s=40, edgecolors="none")
    ax.axvline(INFORMED_RECENCY, color=_YELLOW, lw=1.5, ls=":",
               label=f"informed >{INFORMED_RECENCY}")
    ax.set_title("α6 Volume Recency vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("vol_24h / vol_total")
    ax.set_ylabel("|edge|")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 7. α7 Forecast variance vs edge
    ax = axes[2, 0]
    ax.scatter(opps_df["forecast_var"], opps_df["abs_edge"],
               c=opps_df["days_ahead"], cmap="viridis",
               alpha=0.7, s=40, edgecolors="none")
    ax.set_title("α7 Forecast Variance vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("forecast temp variance (°C²)")
    ax.set_ylabel("|edge|")

    # 8. α8 Hours since market move vs stale
    ax = axes[2, 1]
    stale    = opps_df[opps_df["is_stale"]]
    notstale = opps_df[~opps_df["is_stale"]]
    ax.scatter(notstale["hours_since_move"], notstale["abs_edge"],
               c=_BLUE, alpha=0.6, s=40, edgecolors="none", label="Fresh")
    ax.scatter(stale["hours_since_move"], stale["abs_edge"],
               c=_RED, alpha=0.8, s=60, edgecolors="none", label="Stale")
    ax.axvline(STALE_HOURS, color=_YELLOW, lw=1.5, ls=":", label=f"stale>{STALE_HOURS}h")
    ax.set_title("α8 Market Staleness vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("hours since last significant move")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 9. Alpha score vs EV
    ax = axes[2, 2]
    sc = ax.scatter(opps_df["alpha_score"], opps_df["ev_per_dollar"] * 100,
                    c=opps_df["kelly"], cmap="hot",
                    alpha=0.7, s=50, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="kelly frac").ax.tick_params(colors=_MUTED)
    ax.axhline(0, color=_MUTED, lw=1, ls="--")
    ax.set_title("Composite Score vs EV/dollar", color="white", fontweight="bold")
    ax.set_xlabel("alpha_score")
    ax.set_ylabel("EV per $100 bet")

    plt.suptitle("Alpha Signal Dashboard", color="white",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "alpha_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_forecast_drift_all(data_dir: Path, cities: list[str], output_dir: Path):
    """One subplot per city showing forecast evolution by target date."""
    n = len(cities)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = _dark_fig(nrows, ncols, figsize=(7*ncols, 4*nrows))
    axes_flat  = np.array(axes).flat

    for city in cities:
        ax = next(axes_flat)
        try:
            daily = load_daily(data_dir, city)
        except FileNotFoundError:
            ax.set_visible(False); continue

        dates   = sorted(daily["date_local"].unique())[-7:]
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(dates)))
        for i, d in enumerate(dates):
            sub = daily[daily["date_local"] == d].sort_values("fetched_at_utc")
            if sub.empty: continue
            label = str(d.date()) if hasattr(d, "date") else str(d)[:10]
            ax.plot(sub["fetched_at_utc"], sub["temp_max_c"],
                    marker="o", ms=3, lw=1.6, color=palette[i], label=label)

        ax.set_title(f"{CITY_NAMES.get(city, city)} — Forecast Drift",
                     color="white", fontsize=9, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=6, labelcolor="white", facecolor="#21262d",
                  edgecolor=_BORDER, loc="upper left", ncol=2)
        ax.set_ylabel("°C max", fontsize=8)

    for ax in axes_flat:
        ax.set_visible(False)

    plt.suptitle("Forecast TempMax Evolution", color="white",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "forecast_drift_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_momentum_heatmap(opps_df: pd.DataFrame, output_dir: Path):
    """Heatmap of EMA momentum by (city, target_date)."""
    if opps_df.empty or "ema_momentum" not in opps_df.columns:
        return

    pivot = (opps_df.groupby(["city", "target_date"])["ema_momentum"]
             .mean().unstack("target_date").fillna(0))
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.2), max(3, len(pivot)*0.8)))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_AX)

    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", color=_MUTED, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=_MUTED, fontsize=8)
    plt.colorbar(im, ax=ax, label="EMA momentum (°C/snap)").ax.tick_params(colors=_MUTED)
    ax.set_title("α1 Forecast Momentum by City × Date",
                 color="white", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "momentum_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# BETTING BOT
# ══════════════════════════════════════════════════════════════════════════════

class WeatherBettingBot:
    """
    Weather prediction market betting bot.

    Modes:
      dry_run=True   — print orders, don't execute (default)
      dry_run=False  — execute via py_clob_client (requires POLYMARKET_PRIVATE_KEY)

    live_mode=True   — recalculate days_ahead from NOW and re-verify prices via
                       the Gamma API before placing any order. Always use this
                       for real-money execution.

    Execution:
      pip install py-clob-client
      export POLYMARKET_PRIVATE_KEY="0x..."

      from py_clob_client.client import ClobClient
      from py_clob_client.clob_types import OrderArgs, BUY

      client = ClobClient(
          host="https://clob.polymarket.com",
          key=os.environ["POLYMARKET_PRIVATE_KEY"],
          chain_id=137,   # Polygon
      )
      # YES bet: buy YES token at price=their_prob, size=usdc/price shares
      client.create_market_order(OrderArgs(
          token_id=YES_TOKEN_ID,   # from clob_token_ids_json field
          price=their_prob,
          size=size_usdc / their_prob,
          side=BUY,
      ))
    """

    def __init__(self, bankroll: float = 1000.0,
                 dry_run: bool = True, live_mode: bool = False,
                 kelly_fraction: float = KELLY_FRACTION):
        self.bankroll  = bankroll
        self.dry_run   = dry_run
        self.live_mode = live_mode
        self.kelly_fraction = kelly_fraction
        self.log: list[dict] = []

    def run(self, opps: list[Opportunity], min_edge: float = MIN_EDGE,
            min_days: float = 0.0):
        if not opps:
            print("\n  No tradeable opportunities.")
            return

        now = datetime.now(timezone.utc)

        # ── Step 1: filter to future markets using CURRENT time ───────────
        # days_ahead stored in each Opportunity was computed from the snapshot
        # fetch_time, which may be hours or days old. Recalculate from NOW.
        def _days_from_now(o: Opportunity) -> float:
            target = pd.Timestamp(o.target_date, tz="UTC")
            return (target - pd.Timestamp(now)).total_seconds() / 86400

        candidate = [
            o for o in opps
            if _days_from_now(o) >= min_days and o.kelly > 0
        ]
        if min_days > 0:
            n_skipped = sum(1 for o in opps if 0 < _days_from_now(o) < min_days)
            if n_skipped:
                print(f"\n  [min_days={min_days}] Skipped {n_skipped} markets "
                      f"resolving within {min_days:.0f}d")

        if not candidate:
            print("\n  No future opportunities with positive Kelly.")
            return

        # ── Step 2 (live_mode): fetch current market prices and re-verify ─
        if self.live_mode:
            print(f"\n  [LIVE] Fetching current prices for {len(candidate)} markets …")
            cids        = list({o.condition_id for o in candidate})
            live_prices = fetch_live_prices(cids)
            verified    = []
            for o in candidate:
                live_p = live_prices.get(o.condition_id)
                if live_p is None:
                    print(f"  [SKIP] {o.condition_id[:16]}… — no live price")
                    continue
                if live_p < MIN_MARKET_PRICE or live_p > (1.0 - MIN_MARKET_PRICE):
                    print(f"  [SKIP] {o.question[:50]} — near-settled ({live_p:.2%})")
                    continue
                # Recompute edge with live price
                live_edge = o.forecast_prob - live_p
                if o.bet_side == "No":
                    live_edge = (1.0 - o.forecast_prob) - (1.0 - live_p)
                if abs(live_edge) < min_edge:
                    print(f"  [SKIP] {o.question[:50]} — edge closed "
                          f"(was {o.abs_edge:.1%}, now {abs(live_edge):.1%})")
                    continue
                # Update the opportunity with the fresh price and edge
                o.market_prob_raw = round(live_p, 4)
                o.their_prob      = round(live_p if o.bet_side == "Yes"
                                          else 1.0 - live_p, 4)
                o.edge            = round(live_edge, 4)
                o.abs_edge        = round(abs(live_edge), 4)
                o.kelly           = _kelly_size(o, kelly_fraction=self.kelly_fraction)
                o.ev_per_dollar   = round(o.our_prob / o.their_prob - 1.0, 4)
                verified.append(o)
            candidate = verified
            print(f"  [LIVE] {len(candidate)} markets still have edge after price check\n")

        # ── Step 3: group cap (α9) ────────────────────────────────────────
        candidate = apply_group_kelly_cap(candidate, self.bankroll)

        # ── Step 4: total portfolio cap ───────────────────────────────────
        candidate = apply_portfolio_cap(candidate)

        # ── Step 5: sort and deduplicate — one bet per condition_id ───────
        candidate.sort(key=lambda o: o.alpha_score, reverse=True)
        seen:  set[str]          = set()
        final: list[Opportunity] = []
        for o in candidate:
            if o.condition_id not in seen:
                final.append(o)
                seen.add(o.condition_id)

        # ── Step 6: print and execute ────────────────────────────────────
        print(f"\n{'─'*78}")
        mode = ("DRY RUN" if self.dry_run else "⚠  LIVE")
        live_tag = " + live-price-verified" if self.live_mode else ""
        print(f"  WeatherBettingBot [{mode}{live_tag}]  bankroll=${self.bankroll:,.0f}")
        print(f"  Fee model: {FEE_RATE:.0%}  |  Max exposure: {MAX_TOTAL_KELLY:.0%} bankroll")
        print(f"{'─'*78}")
        print(f"  {'Side':4s}  {'Size$':>7}  {'Edge':>6}  {'EV$':>7}  "
              f"{'Liq$':>8}  {'Mom':>6}  {'Sig':8s}  Question")
        print(f"  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*7}  "
              f"{'─'*8}  {'─'*6}  {'─'*8}  {'─'*70}")

        total_size = 0.0
        total_ev   = 0.0
        for o in final:
            size = round(self.bankroll * o.kelly, 2)
            if size < 1.0:
                continue

            if self.live_mode:
                # For a "Yes" bet, we are buying Yes tokens, so we check the ask side of the Yes token book.
                # For a "No" bet, we are buying No tokens. The price of a No token is (1 - yes_price).
                # So, buying a No token at 0.70 is equivalent to selling a Yes token at 0.30.
                # We check the VWAP for our side and adjust the effective market probability.
                vwap_price = check_orderbook_vwap(o.condition_id, o.bet_side, size)
                
                # The price from the book is for the token we are buying (e.g. price of "No" token)
                # We need to convert it back to the equivalent "Yes" probability for our edge calculation.
                effective_market_yes_prob = vwap_price if o.bet_side == "Yes" else 1.0 - vwap_price
                real_edge = o.forecast_prob - effective_market_yes_prob

                if (o.bet_side == "Yes" and real_edge < min_edge) or (o.bet_side == "No" and real_edge > -min_edge):
                    print(f"  [SKIP] {o.question[:45]}… — Slippage killed edge (VWAP: {vwap_price:.2f})")
                    continue
                
                # Update the opportunity with the true, slippage-adjusted price
                o.their_prob = vwap_price
                o.abs_edge   = round(abs(real_edge), 4)
                o.ev_per_dollar = round(o.our_prob / o.their_prob - 1.0, 4)

            ev    = round(o.ev_per_dollar * size, 2)
            days  = round(_days_from_now(o), 1)
            order = {
                "condition_id":  o.condition_id,
                "question":      o.question,
                "bet_side":      o.bet_side,
                "size_usdc":     size,
                "market_prob":   o.their_prob,
                "forecast_prob": o.our_prob,
                "edge":          o.abs_edge,
                "ev_dollar":     ev,
                "alpha_score":   o.alpha_score,
                "days_to_expiry":days,
                "sigma_source":  o.sigma_source,
                "is_stale":      o.is_stale,
                "live_verified": self.live_mode,
                "timestamp_utc": now.isoformat(),
            }
            self._execute(order)
            total_size += size
            total_ev   += ev
            print(f"  {o.bet_side:4s}  ${size:>6.2f}  "
                  f"{o.abs_edge:>5.1%}  ${ev:>6.2f}  "
                  f"${o.liquidity:>7,.0f}  "
                  f"{o.ema_momentum:>+6.2f}  "
                  f"{o.sigma_source[:8]:8s}  "
                  f"{o.question[:70]}…  (+{days:.1f}d)")

        print(f"{'─'*78}")
        print(f"  Bets     : {len(self.log)}")
        print(f"  Exposure : ${total_size:,.2f}  ({total_size/self.bankroll:.1%} of bankroll)")
        print(f"  Exp. EV  : ${total_ev:+,.2f}  ({total_ev/max(total_size,1)*100:.1f}% ROI)")

    def _execute(self, order: dict):
        self.log.append(order)
        if not self.dry_run:
            raise NotImplementedError("Wire in py_clob_client. See class docstring.")


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def opps_to_df(opps: list[Opportunity]) -> pd.DataFrame:
    if not opps:
        return pd.DataFrame()
    rows = []
    for o in opps:
        rows.append({k: v for k, v in o.__dict__.items()})
    return pd.DataFrame(rows)


def print_report(opps_df: pd.DataFrame):
    if opps_df.empty:
        print("\n  No opportunities above threshold.")
        return

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n{'═'*90}")
    print(f"  OPPORTUNITIES  —  {len(opps_df)} total  "
          f"(edge≥{MIN_EDGE:.0%}, liq≥{MIN_LIQUIDITY})")
    print(f"{'═'*90}")

    top_cols = ["city", "target_date", "days_ahead", "forecast_mu",
                "bin_temp_c" if "bin_temp_c" in opps_df.columns else "forecast_mu",
                "forecast_prob", "market_prob", "edge", "bet_side",
                "kelly", "ev_per_dollar", "alpha_score",
                "ema_momentum", "is_stale", "liquidity"]
    top_cols = [c for c in top_cols if c in opps_df.columns]
    print(opps_df.sort_values("alpha_score", ascending=False)
                 .head(30)[top_cols].to_string(index=False))

    print(f"\n  ── By City ──")
    for city, g in opps_df.groupby("city"):
        print(f"  {city:10s}  n={len(g):4d}  "
              f"avg|edge|={g['abs_edge'].mean():.1%}  "
              f"max|edge|={g['abs_edge'].max():.1%}  "
              f"stale={g['is_stale'].sum():3d}  "
              f"tot_EV≈${(g['ev_per_dollar']*g['kelly']*1000).sum():.0f}")

    print(f"\n  ── Momentum-aligned bets (α1) ──")
    mom_yes = opps_df[(opps_df["bet_side"]=="Yes") &
                      (opps_df["ema_momentum"] > MOMENTUM_THRESHOLD)]
    mom_no  = opps_df[(opps_df["bet_side"]=="No") &
                      (opps_df["ema_momentum"] < -MOMENTUM_THRESHOLD)]
    aligned = pd.concat([mom_yes, mom_no])
    if not aligned.empty:
        print(aligned[["city","target_date","question","edge",
                        "bet_side","ema_momentum","alpha_score"]]
              .sort_values("alpha_score", ascending=False).head(10).to_string(index=False))
    else:
        print("  None found")

    print(f"\n  ── Stale markets with large edge (α8) ──")
    stale = opps_df[opps_df["is_stale"] & (opps_df["abs_edge"] > 0.10)]
    if not stale.empty:
        print(stale[["city","target_date","question","edge",
                     "hours_since_move","alpha_score"]].head(8).to_string(index=False))
    else:
        print("  None found")

    print(f"\n  ── Incoherent markets (α5: PMF sum deviation) ──")
    incoherent = opps_df[opps_df["pmf_sum_dev"] > 0.15].copy()
    if not incoherent.empty:
        print(incoherent[["city","target_date","pmf_sum_dev",
                          "edge","bet_side"]].drop_duplicates(
                              ["city","target_date"]).head(8).to_string(index=False))
    else:
        print("  None found")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Weather Analyzer v4 — with live-mode betting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--cities",     nargs="+", default=None)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--min_edge",   type=float, default=MIN_EDGE)
    parser.add_argument("--min_liq",    type=float, default=MIN_LIQUIDITY)
    parser.add_argument("--min_days",   type=float, default=0.0,
                        help="Skip markets resolving in fewer than N days from now "
                             "(e.g. --min_days 1 excludes today's markets)")
    parser.add_argument("--bankroll",   type=float, default=1000.0)
    parser.add_argument("--no_plots",   action="store_true")
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: recalculate days_ahead from NOW and re-verify market "
             "prices via the Gamma API before recommending any bet. Always use "
             "this for real-money execution.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually place orders (requires POLYMARKET_PRIVATE_KEY env var). "
             "Implies --live. Use with extreme caution.",
    )
    parser.add_argument("--disable_ml", action="store_true", help="Disable ML models and fall back to ensemble/student-t")
    parser.add_argument("--kelly_fraction", type=float, default=0.25, help="Kelly fraction multiplier for sizing bets")
    parser.add_argument("--sigma_threshold", type=float, default=1.2, help="Ensemble sigma threshold above which we bypass ML calibrator")
    parser.add_argument("--disable_conflict_gating", action="store_true", help="Disable conflict gating check between ML and Ensemble predictions")
    args = parser.parse_args()

    dry_run   = not args.execute
    live_mode = args.live or args.execute

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = args.cities or discover_cities(data_dir)
    if not cities:
        print(f"No *_snapshots.csv files found in {data_dir}")
        return

    print(f"\n{'═'*72}")
    print(f"  Polymarket Weather Analyzer v4")
    print(f"  Cities   : {cities}")
    print(f"  Min edge : {args.min_edge:.0%}  Min liq: ${args.min_liq:.0f}  "
          f"Min days: {args.min_days:.0f}  Bankroll: ${args.bankroll:,.0f}")
    print(f"  Fee      : {FEE_RATE:.0%}  Max portfolio: {MAX_TOTAL_KELLY:.0%}")
    print(f"  Mode     : {'LIVE (price-verified)' if live_mode else 'BACKTEST'} / "
          f"{'EXECUTE' if not dry_run else 'DRY RUN'}")
    print(f"{'═'*72}\n")

    all_opps: list[Opportunity] = []
    for city in cities:
        print(f"── {city.upper()} ──")
        opps = analyse_city(data_dir, city,
                            min_edge=args.min_edge,
                            min_liq=args.min_liq,
                            use_ml=not args.disable_ml,
                            kelly_fraction=args.kelly_fraction,
                            sigma_threshold=args.sigma_threshold,
                            conflict_gating=not args.disable_conflict_gating)
        if opps:
            print(f"  → {len(opps)} opportunities found")
        all_opps.extend(opps)

    opps_df = opps_to_df(all_opps)

    if not args.no_plots:
        if not opps_df.empty:
            all_bins_records = []
            for o in all_opps:
                all_bins_records.append({
                    "city":         o.city,
                    "target_date":  o.target_date,
                    "fetched_at":   o.fetched_at,
                    "bin_temp_c":   o.bin_temp_c,   # actual question temperature
                    "market_prob":  o.market_prob,
                    "forecast_prob":o.forecast_prob,
                    "forecast_mu":  o.forecast_mu,
                    "days_ahead":   o.days_ahead,
                    "ema_momentum": o.ema_momentum,
                })
            all_bins_df = pd.DataFrame(all_bins_records)
            plot_pmf_comparison(opps_df, all_bins_df, output_dir)
            plot_alpha_dashboard(opps_df, output_dir)
            plot_momentum_heatmap(opps_df, output_dir)

        plot_forecast_drift_all(data_dir, cities, output_dir)

    # Only show future markets in the terminal report to avoid confusion
    today_str = datetime.now(timezone.utc).date().isoformat()
    print_report(opps_df[opps_df["target_date"] >= today_str] if not opps_df.empty else opps_df)

    if not opps_df.empty:
        # Standard save for backward compatibility
        opps_df.to_csv(output_dir / "opportunities_v4.csv", index=False)
        
        # Save to evaluation CSVs for the 10-day test (overwrites each time with full history)
        eval_filename = "opportunities_evaluation_ensemble.csv" if args.disable_ml else "opportunities_evaluation_ml.csv"
        eval_path = output_dir / eval_filename
        opps_df.to_csv(eval_path, index=False)
        
        print(f"\n  Saved: {output_dir}/opportunities_v4.csv")
        print(f"  Saved evaluation tracker: {eval_path}")

    bot = WeatherBettingBot(
        bankroll=args.bankroll, dry_run=dry_run, live_mode=live_mode,
        kelly_fraction=args.kelly_fraction
    )
    bot.run(all_opps, min_edge=args.min_edge, min_days=args.min_days)

    print(f"\nDone → {output_dir.resolve()}")


if __name__ == "__main__":
    main()