"""
MLCalibratorPredictor: Random Forest Forecast Calibration Model
===============================================================

This model corrects systematic biases in numerical weather predictions (NWP)
using machine learning.

Methodology & Core Mechanics:
-----------------------------
1. **Bias Correction (Mean Calibration)**:
   - Raw weather models (like ECMWF or GFS) often suffer from localized errors due to elevation, 
     urban heat island effects, or grid resolution.
   - We train a city-specific `RandomForestRegressor` on historical grid forecasts versus 
     actual airport station measurements (ground truth via Meteostat).
   - The regressor predicts the calibrated temperature mean ($\mu_{ML}$) by correcting the 
     raw forecast value based on seasonal trends.

2. **Features Used**:
   - `grid_temp_max_c`: The raw deterministic/ensemble mean temperature forecast.
   - `day_of_year`: Captured as a cyclical feature to track seasonal offsets (e.g., the model 
     might tend to overpredict in summer but underpredict in winter).

3. **Uncertainty Modeling**:
   - Instead of predicting a fixed error spread, this model inherits the dynamic, 
     flow-dependent forecast spread ($\sigma_{ens}$) and tail-thickness ($\nu$) from the 
     weather ensemble. This ensures we don't bet overconfidently when the atmosphere is unstable.

4. **Robust Fallbacks**:
   - If the ML model file (`.joblib`) is missing or fails to load, the predictor seamlessly 
     falls back to the pure `EnsemblePredictor`.
"""

import joblib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .base import BasePredictor, TemperatureDistribution
from .ensemble import EnsemblePredictor, get_ensemble_params, fit_nu_from_ensemble
from .nwp_fallback import spread_sigma_boost

class MLCalibratorPredictor(BasePredictor):
    def __init__(self, use_ml: bool = True, sigma_threshold: float = 1.2):
        self.use_ml = use_ml
        self.sigma_threshold = sigma_threshold
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
        
        # Sigma Filter: if ensemble spread is too high (unstable/dynamic weather),
        # trust physics-based ensemble and skip ML bias correction
        if raw_sigma > self.sigma_threshold:
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )
            
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
