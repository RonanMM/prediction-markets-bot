from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass

@dataclass
class TemperatureDistribution:
    mu: float        # Forecast mean temperature (°C)
    sigma: float     # Forecast uncertainty / spread (standard deviation in °C)
    nu: float        # Student-t degrees of freedom (tails thickness)
    source: str      # Source name (e.g., 'ml_portfolio', 'ensemble', 'nwp_table')

class BasePredictor(ABC):
    @abstractmethod
    def predict_distribution(
        self, 
        city: str, 
        target_date: pd.Timestamp, 
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None
    ) -> TemperatureDistribution:
        """
        Computes forecast distribution parameters for a given city and target date.
        """
        pass
