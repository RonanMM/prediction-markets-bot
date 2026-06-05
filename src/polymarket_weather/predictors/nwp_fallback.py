import pandas as pd
import numpy as np
from .base import BasePredictor, TemperatureDistribution

# NWP-calibrated sigma (°C) and Student-t degrees-of-freedom by days-ahead
# Based on ECMWF verification statistics for 2m Tmax
NWP_PARAMS: dict[int, tuple[float, float]] = {
    0: (0.7,  12.0),   # near-observation: almost Gaussian
    1: (1.2,  8.0),
    2: (1.6,  6.5),
    3: (2.0,  5.5),
    4: (2.4,  5.0),
    5: (2.8,  4.5),
}
NWP_BEYOND = (3.2, 4.0)          # sigma, nu for >5 days ahead
SPREAD_SIGMA_WEIGHT = 0.25

def spread_sigma_boost(
    daily_df: pd.DataFrame, 
    target_date: pd.Timestamp, 
    fetch_time: pd.Timestamp,
    baseline_spread: float = 8.0
) -> float:
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

def get_nwp_params(days_ahead: float) -> tuple[float, float]:
    d = max(0, int(round(days_ahead)))
    return NWP_PARAMS.get(d, NWP_BEYOND)

class NWPFallbackPredictor(BasePredictor):
    def predict_distribution(
        self, 
        city: str, 
        target_date: pd.Timestamp, 
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None
    ) -> TemperatureDistribution:
        # Get target date row in daily_df for the deterministic temperature forecast
        td = pd.Timestamp(target_date).normalize()
        if td.tzinfo is not None:
            td = td.tz_localize(None)
            
        sub = daily_df[
            (daily_df["date_local"].dt.normalize() == td) &
            (daily_df["fetched_at_utc"] <= fetch_time)
        ].sort_values("fetched_at_utc")
        
        if sub.empty:
            raise ValueError(f"No daily weather forecast found for target date {target_date} up to fetch time {fetch_time}")
            
        fc_row = sub.iloc[-1]
        mu_det = float(fc_row["temp_max_c"])
        
        # Calculate uncertainty parameters
        s_boost = spread_sigma_boost(daily_df, target_date, fetch_time)
        sigma_base, nu_base = get_nwp_params(days_ahead)
        
        return TemperatureDistribution(
            mu=mu_det,
            sigma=sigma_base + s_boost,
            nu=nu_base,
            source="nwp_table"
        )
