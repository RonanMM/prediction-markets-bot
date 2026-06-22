import pandas as pd
import numpy as np
from math import log
from models import Opportunity

# Momentum signal: EMA span (in number of forecast snapshots)
MOMENTUM_EMA_SPAN     = 3
# Minimum momentum magnitude to boost/penalise edge (°C/snapshot)
MOMENTUM_THRESHOLD    = 0.15

# Volume-recency threshold: ratio vol_24h/vol_total
INFORMED_RECENCY      = 0.8

# Market staleness: if market hasn't moved >1pp in last N hours → flag stale
STALE_HOURS           = 4
STALE_MOVE_THRESHOLD  = 0.02

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


def volume_recency_signal(row: pd.Series) -> float:
    """0.0 – 1.0. >0.65 = informed traders active."""
    vol_total = float(row.get("volume_usdc", 0) or 0)
    vol_24h   = float(row.get("volume_24h_usdc", 0) or 0)
    if vol_total < 1:
        return 0.0
    return min(vol_24h / vol_total, 1.0)


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
        "last_move":        round(last_move_size, 3),
        "market_momentum":  round(mkt_mom, 3),
    }

def score_opportunity(opp: Opportunity) -> float:
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
