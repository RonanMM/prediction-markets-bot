from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass
from typing import Callable

@dataclass
class TemperatureDistribution:
    mu: float        # Forecast mean temperature (°C)
    sigma: float     # Forecast uncertainty / spread (standard deviation in °C)
    nu: float        # Student-t degrees of freedom (tails thickness)
    source: str      # Source name (e.g., 'emos_v2', 'ensemble', 'nwp_table')
    floor: float = None     # Censoring (°C): T = max(floor, Z). Same-day Tmax bets:
                            # the running observed daily max — Tmax cannot end below it.
    ceiling: float = None   # Censoring (°C): T = min(ceiling, Z). Same-day Tmin bets:
                            # the running observed daily min — Tmin cannot end above it.
    cdf: "Callable[[float], float] | None" = None   # optional empirical CDF of the uncensored Z;
                                                     # when set, pmf uses it instead of the Student-t

class BasePredictor(ABC):
    @abstractmethod
    def predict_distribution(
        self,
        city: str,
        target_date: pd.Timestamp,
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None,
        mm_df: pd.DataFrame = None,
        obs_df: pd.DataFrame = None,
        nbm_df: pd.DataFrame = None,
    ) -> TemperatureDistribution:
        """
        Computes forecast distribution parameters for a given city and target date.

        mm_df (optional): live deterministic multi-model forecasts
        ({slug}_daily_mm.csv) — the exact serving input for calibrations trained
        on the multi-model previous-runs mean.
        """
        pass
