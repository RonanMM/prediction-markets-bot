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
        expected = 2.0 * float(student_t.ppf(0.9, df=nu))
        if expected >= empirical_ratio:
            return nu
    return 30.0   # effectively Gaussian

def get_ensemble_params(ens_df: Optional[pd.DataFrame], target_date, fetch_time) -> Optional[dict]:
    """
    Look up the best ensemble snapshot for target_date.
    Prefers the latest row fetched at or before fetch_time.
    If none exist, falls back to the latest available row for that date.
    """
    if ens_df is None or ens_df.empty:
        return None

    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)

    date_rows = ens_df[ens_df["date_local"].dt.normalize() == td]
    if date_rows.empty:
        return None

    # Prefer rows fetched at or before the snapshot's fetch_time
    sub = date_rows[date_rows["fetched_at_utc"] <= fetch_time]
    if sub.empty:
        sub = date_rows

    row = sub.sort_values("fetched_at_utc").iloc[-1]
    std = float(row.get("ens_std", np.nan))
    if np.isnan(std) or std <= 0:
        return None

    return {
        "ens_mean":   float(row.get("ens_mean",   np.nan)),
        "ens_std":    std,
        "ens_p10":    float(row.get("ens_p10",    np.nan)),
        "ens_p90":    float(row.get("ens_p90",    np.nan)),
        "ens_spread": float(row.get("ens_spread", np.nan)),
        "n_members":  int(row.get("n_members", 0)),
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
        ens_df: pd.DataFrame = None
    ) -> TemperatureDistribution:
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        if ens_params is None:
            # Fall back to NWP if ensemble is not available
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
