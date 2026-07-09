r"""
EnsemblePredictor: Dynamic Student-t Weather Ensemble Model
===========================================================

This model constructs a continuous probability distribution for temperatures
by analyzing the spread across multiple independent weather forecast runs (ensemble members).

Methodology & Core Mechanics:
-----------------------------
1. **Dynamic Uncertainty (Sigma Calculation)**:
   - Traditional models assume a fixed forecast error for a given day (e.g., "forecasts at day 5 
     have a standard deviation of 2.8°C").
   - Instead, we query 40 independent runs of the German DWD ICON-Seamless ensemble model. 
     If the runs are closely clustered, forecast uncertainty is low ($\sigma$ is small). If the 
     runs diverge, uncertainty is high ($\sigma$ is large).
   - Standard deviation of the ensemble is calculated and combined with a convective diurnal 
     spread boost (`spread_sigma_boost`) to yield the dynamic uncertainty $\sigma$.

2. **Heavy-Tail Modeling (Student-t Distribution)**:
   - Real-world weather errors have heavier tails than a standard normal (Gaussian) distribution 
     (i.e., extreme cold or hot snaps are more frequent than a bell curve suggests).
   - We fit a Student-t distribution characterized by degrees of freedom ($\nu$). 
   - To estimate $\nu$, we compare the empirical spread (p90 - p10 percentiles) against the 
     standard deviation. We solve:
     
     (p90 - p10) / std = 2 * t.ppf(0.9, df=nu)
     
     Solving for $\nu$ calibrates the distribution's tail thickness dynamically. A small $\nu$ 
     (e.g., 4.0) means heavy tails, while a large $\nu$ (e.g., 30.0) approaches a standard Gaussian.

3. **Fallback to NWP Table**:
   - If the ensemble data is unavailable, the model falls back to the `NWPFallbackPredictor`, 
     which uses a fixed lookup table (`NWP_PARAMS`) representing typical numerical weather 
     prediction error spreads for each lead day.
"""

from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import t as student_t
from .base import BasePredictor, TemperatureDistribution
from .nwp_fallback import NWPFallbackPredictor, spread_sigma_boost

def fit_nu_from_ensemble(ens_params: dict) -> float:
    """
    Estimate Student-t degrees-of-freedom (nu) from ensemble p10/p90.
    For a Student-t(nu): (p90 - p10) / sigma = 2 * t.ppf(0.9, df=nu).
    We solve for the nu whose theoretical ratio best matches the empirical one.
    Falls back to 8.0 (moderately heavy tails) when data is insufficient.
    """
    spread = ens_params.get("ens_spread", np.nan)   # p90 - p10
    std    = ens_params.get("ens_std",    np.nan)
    if np.isnan(spread) or np.isnan(std) or std < 1e-6:
        return 8.0

    empirical_ratio = spread / std   # should be ~2.56 for Gaussian
    # Search candidate nu values from heavy-tailed to Gaussian
    for nu in (4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0):
        expected = 2.0 * float(student_t.ppf(0.9, df=nu)) * np.sqrt((nu - 2.0) / nu)
        if expected >= empirical_ratio:
            return nu
    return 30.0   # effectively Gaussian

def _num(v) -> float:
    """Coerce a possibly-missing cell to float, NaN when absent/blank."""
    return np.nan if pd.isna(v) else float(v)


def get_ensemble_params(ens_df: Optional[pd.DataFrame], target_date, fetch_time) -> Optional[dict]:
    """
    Look up the ensemble snapshot for target_date as of fetch_time.

    D1: STRICT as-of — only rows fetched at or before fetch_time are eligible; if none
    exist, return None (never fall back to a later run, which would leak look-ahead data
    into a backtest, the exact artifact the pre-committed gate exists to prevent).

    C6/E4: the Tmax and Tmin member stats are validated INDEPENDENTLY. A bad/missing Tmax
    std no longer discards a valid Tmin (so Tmin serving keeps its flow sigma and proxy),
    and a NaN mean no longer propagates downstream; None is returned only when BOTH the
    max and min stats are unusable. Invalid fields are set to None so callers can tell.
    """
    if ens_df is None or ens_df.empty:
        return None

    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)

    date_rows = ens_df[ens_df["date_local"].dt.normalize() == td]
    if date_rows.empty:
        return None

    sub = date_rows[date_rows["fetched_at_utc"] <= fetch_time]
    if sub.empty:
        return None

    row = sub.sort_values("fetched_at_utc").iloc[-1]

    def _usable(mean, std) -> bool:
        return not np.isnan(mean) and not np.isnan(std) and std > 0

    mean,     std     = _num(row.get("ens_mean")),     _num(row.get("ens_std"))
    min_mean, min_std = _num(row.get("ens_min_mean")), _num(row.get("ens_min_std"))
    max_ok = _usable(mean, std)
    min_ok = _usable(min_mean, min_std)
    if not max_ok and not min_ok:
        return None

    n_members = row.get("n_members", 0)
    return {
        "ens_mean":     mean if max_ok else None,
        "ens_std":      std if max_ok else None,
        "ens_p10":      _num(row.get("ens_p10")),
        "ens_p90":      _num(row.get("ens_p90")),
        "ens_spread":   _num(row.get("ens_spread")),
        "n_members":    0 if pd.isna(n_members) else int(n_members),
        "ens_min_mean": min_mean if min_ok else None,
        "ens_min_std":  min_std if min_ok else None,
    }

class EnsemblePredictor(BasePredictor):
    def __init__(self):
        self._fallback_predictor = NWPFallbackPredictor()

    def predict_distribution(
        self,
        city: str,
        target_date: pd.Timestamp,
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None,
        mm_df: pd.DataFrame = None,   # unused; part of the shared predictor interface
        obs_df: pd.DataFrame = None,  # unused; part of the shared predictor interface
        nbm_df: pd.DataFrame = None,  # unused; part of the shared predictor interface
    ) -> TemperatureDistribution:
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        # EnsemblePredictor builds a Tmax distribution; fall back to NWP if the Tmax mean/std
        # are unusable. get_ensemble_params may now return a dict with ONLY the Tmin stats
        # valid (C6) or a None mean (E4) — those must not flow into mu_ens/sigma_ens.
        if (ens_params is None
                or ens_params.get("ens_mean") is None
                or ens_params.get("ens_std") is None):
            return self._fallback_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )
            
        s_boost = spread_sigma_boost(daily_df, target_date, fetch_time)
        
        mu_ens = ens_params["ens_mean"]
        sigma_ens = max(0.5, ens_params["ens_std"]) + s_boost
        nu_ens = fit_nu_from_ensemble(ens_params)
        
        return TemperatureDistribution(
            mu=mu_ens,
            sigma=sigma_ens,
            nu=nu_ens,
            source="ensemble"
        )
