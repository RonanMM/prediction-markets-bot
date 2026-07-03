r"""
EMOSPredictor — Nonhomogeneous Regression (EMOS) ensemble post-processing
=========================================================================

EMOS is the standard statistical method for turning a raw weather forecast into a *calibrated*
predictive distribution: **EMOS / Nonhomogeneous Gaussian Regression (NGR)**. It replaced an
overfit RandomForest calibrator (since removed) — the rationale is below.

Why this instead of a RandomForest
----------------------------------
The RF had two features (`grid_temp_max_c`, `day_of_year`), `min_samples_split=5` (it memorized →
12 MB models), no temporal cross-validation, and a train/serve skew (trained on the archive grid
temperature, but served the *ensemble mean* at inference). It also threw away its own residual
variance, so its predictive distribution inherited only the ensemble's (often too-tight) spread —
i.e. it was overconfident, which is exactly what loses on Brier.

EMOS fixes all of that with a handful of parameters (so it cannot overfit):

    mu_cal    = a + b * mu_raw + c_sin * sin(2π·doy/365) + c_cos * cos(2π·doy/365)
    sigma_cal = max(s1 * ens_std, s0_floor) + diurnal_boost

* **Mean**: a per-city linear bias correction with two seasonal harmonics — a genuine, non-overfit
  replacement for the RF's mean nudge. Crucially it is trained on the SAME variable it is served
  at inference (the deterministic forecast), removing the train/serve skew.
* **Spread**: trusts the flow-dependent ensemble spread `ens_std`, but never lets sigma fall below
  a floor `s0_floor` learned from realized forecast errors. That floor is the fix for the
  overconfidence the reliability diagram showed (tight bins, realized frequency far from predicted).

Note on `s1`: ideally the spread-scaling `s1` is fit from historical (ens_std, error) pairs, but the
Open-Meteo *archive* only exposes a deterministic grid temp, not a per-day ensemble spread. So `s1`
defaults to 1.0 and is left for the WS0 evaluation harness to tune on real graded outcomes. `a`, `b`,
`c_sin`, `c_cos`, and `s0_floor` ARE fit from the archive by `train_calibrator.py`.

Fallback: if the per-city params file is missing, or there is no deterministic forecast for the
date, it defers to the pure `EnsemblePredictor` (which itself falls back to the NWP table).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BasePredictor, TemperatureDistribution
from .ensemble import EnsemblePredictor, get_ensemble_params, fit_nu_from_ensemble
from .nwp_fallback import spread_sigma_boost, get_nwp_params

# Canonical params location: src/polymarket_weather/models/{slug}_emos.json — same dir the RF
# used, package-relative so it is found regardless of the working directory.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=None)
def _load_params(city_slug: str):
    """Per-city EMOS params dict, or None if not yet trained."""
    path = _MODELS_DIR / f"{city_slug}_emos.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _seasonal(day_of_year: int):
    ang = 2.0 * np.pi * day_of_year / 365.0
    return float(np.sin(ang)), float(np.cos(ang))


def emos_mean(params: dict, mu_raw: float, day_of_year: int) -> float:
    """Calibrated mean: a + b·mu_raw + c_sin·sin(doy) + c_cos·cos(doy)."""
    s, c = _seasonal(day_of_year)
    return (float(params["a"]) + float(params["b"]) * mu_raw
            + float(params.get("c_sin", 0.0)) * s
            + float(params.get("c_cos", 0.0)) * c)


def emos_sigma(params: dict, ens_std: float, s_boost: float) -> float:
    """Calibrated spread: max(s1·ens_std, s0_floor) + diurnal boost. Never overconfident."""
    s1 = float(params.get("s1", 1.0))
    s0_floor = float(params.get("s0_floor", 0.0))
    return max(s1 * max(0.0, ens_std), s0_floor) + s_boost


def _latest_deterministic_mu(daily_df: pd.DataFrame, target_date, fetch_time):
    """Most recent deterministic forecast temp_max_c for target_date at/-before fetch_time."""
    if daily_df is None or daily_df.empty:
        return None
    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)
    sub = daily_df[
        (daily_df["date_local"].dt.normalize() == td) &
        (daily_df["fetched_at_utc"] <= fetch_time)
    ].sort_values("fetched_at_utc")
    if sub.empty:
        return None
    val = sub.iloc[-1]["temp_max_c"]
    return None if pd.isna(val) else float(val)


class EMOSPredictor(BasePredictor):
    def __init__(self):
        self._ensemble_predictor = EnsemblePredictor()

    def predict_distribution(
        self,
        city: str,
        target_date: pd.Timestamp,
        fetch_time: pd.Timestamp,
        days_ahead: float,
        daily_df: pd.DataFrame,
        ens_df: pd.DataFrame = None,
    ) -> TemperatureDistribution:
        city_slug = city.replace(" ", "_").lower()
        params = _load_params(city_slug)

        # No trained params → defer to the pure ensemble (which falls back to NWP).
        if params is None:
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )

        # Self-gating: EMOS only earns its place where the calibrated mean actually beat the raw
        # forecast on the honest temporal holdout (train_calibrator.py records both RMSEs). Where
        # it did NOT (holdout_rmse_calibrated >= holdout_rmse_raw — e.g. London, NYC), applying the
        # calibrated mean would inject bias, so defer to the pure ensemble for that city. Older
        # param files without these keys keep the prior behaviour (no gating).
        rmse_cal = params.get("holdout_rmse_calibrated")
        rmse_raw = params.get("holdout_rmse_raw")
        if rmse_cal is not None and rmse_raw is not None and float(rmse_cal) >= float(rmse_raw):
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )

        # Mean input = the DETERMINISTIC forecast (same variable EMOS was trained on).
        mu_raw = _latest_deterministic_mu(daily_df, target_date, fetch_time)
        if mu_raw is None:
            return self._ensemble_predictor.predict_distribution(
                city, target_date, fetch_time, days_ahead, daily_df, ens_df
            )

        # Spread input = flow-dependent ensemble std when available, else the NWP lead-time table.
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        if ens_params is not None:
            ens_std = max(0.5, ens_params["ens_std"])
            nu = float(params.get("nu")) if params.get("nu") else fit_nu_from_ensemble(ens_params)
        else:
            ens_std, nu_tab = get_nwp_params(days_ahead)
            nu = float(params.get("nu")) if params.get("nu") else nu_tab

        s_boost = spread_sigma_boost(daily_df, target_date, fetch_time)
        day_of_year = pd.Timestamp(target_date).dayofyear

        return TemperatureDistribution(
            mu=emos_mean(params, mu_raw, day_of_year),
            sigma=emos_sigma(params, ens_std, s_boost),
            nu=nu,
            source="emos",
        )
