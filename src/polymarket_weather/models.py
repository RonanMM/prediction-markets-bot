import pandas as pd
from dataclasses import dataclass

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
    temp_lo: float = None   # range markets only ('between X-Y°F'): low endpoint °C
    temp_hi: float = None   # range markets only ('between X-Y°F'): high endpoint °C

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
    forecast_floor: float = None   # censoring point of the predictive dist (°C):
                                   # same-day running observed max; None otherwise
    forecast_ceiling: float = None # ceiling censoring for Tmin markets (°C):
                                   # same-day running observed min; None otherwise
    temp_lo_c:      float = None   # range markets ('between X-Y°F'): °C endpoints, so the
    temp_hi_c:      float = None   # eval CSV can grade them without re-parsing the question
    model_prob_side:float = None   # our-side PRE-shrink model probability; lets live re-verify
    shrink_weight:  float = 1.0    # re-shrink toward the FRESH price (E3) instead of the stale one
    bucket:         str   = ""     # E3 selective-aggression bucket, e.g. "NYC|same-day"
    live_eligible:  bool  = False  # bucket ∈ config.LIVE_BUCKETS — execution eligibility ONLY;
                                   # the tracker records every flag regardless (eval needs them)
