import joblib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .base import BasePredictor, TemperatureDistribution
from .ensemble import EnsemblePredictor, get_ensemble_params, fit_nu_from_ensemble
from .nwp_fallback import spread_sigma_boost

class MLCalibratorPredictor(BasePredictor):
    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self._ensemble_predictor = EnsemblePredictor()

    def predict_distribution(
        self, 
        city: str, 
        target_date: pd.Timestamp, 
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None
    ) -> TemperatureDistribution:
        # If ML is disabled, fall back to pure Ensemble model
        if not self.use_ml:
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )

        # Try to resolve ensemble params first
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        if ens_params is None:
            # If no ensemble data, we cannot run ML calibrator,
            # so fall back to the ensemble predictor (which will fall back to NWP)
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )

        raw_mu = ens_params["ens_mean"]
        raw_sigma = max(0.5, ens_params["ens_std"])
        raw_nu = fit_nu_from_ensemble(ens_params)
        s_boost = spread_sigma_boost(daily_df, target_date, fetch_time)

        city_slug = city.replace(' ', '_').lower()
        
        # Resolve model paths
        current_dir = Path(__file__).parent.resolve()
        # Look in workspace root models/ first, then package models/
        paths_to_try = [
            current_dir / ".." / ".." / "models",
            current_dir / ".." / "models"
        ]
        
        model = None
        sigma_cal = None
        for base_path in paths_to_try:
            model_path = base_path / f"{city_slug}_calibrator.joblib"
            sigma_path = base_path / f"{city_slug}_sigma.json"
            if model_path.exists() and sigma_path.exists():
                try:
                    model = joblib.load(model_path)
                    with open(sigma_path, "r") as f:
                        sigma_data = json.load(f)
                    sigma_cal = sigma_data["sigma"]
                    break
                except Exception:
                    pass

        if model is not None:
            try:
                day_of_year = target_date.dayofyear
                calibrated_mu = model.predict([[raw_mu, day_of_year]])[0]
                
                return TemperatureDistribution(
                    mu=float(calibrated_mu),
                    sigma=raw_sigma + s_boost,  # Keep dynamic flow-dependent uncertainty for ML
                    nu=raw_nu,
                    source="ml_portfolio"
                )
            except Exception:
                pass

        # Fall back to pure ensemble if ML loading or prediction fails
        return self._ensemble_predictor.predict_distribution(
            city, target_date, fetch_time, days_ahead, daily_df, ens_df
        )
