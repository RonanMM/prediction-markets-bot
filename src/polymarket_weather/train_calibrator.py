"""train_calibrator.py — fit the per-city EMOS (Nonhomogeneous Regression) calibrators.

Replaces the old RandomForest trainer. EMOS is the standard method for post-processing a raw
weather forecast into a *calibrated* predictive distribution, and — unlike the RF — it has a
handful of parameters (cannot overfit), uses a temporal holdout (honest error estimate), and is
trained on the SAME variable it is served at inference (the deterministic forecast), removing the
RF's train/serve skew.

Model (see predictors/emos.py):
    mu_cal    = a + b·grid_temp + c_sin·sin(2π·doy/365) + c_cos·cos(2π·doy/365)
    sigma_cal = max(s1·ens_std, s0_floor) + diurnal_boost   (s1=1.0, s0_floor fit here)

Fitted from the archive (grid forecast vs station truth): a, b, c_sin, c_cos, s0_floor, nu.
`s1` (spread scaling) stays 1.0 — the archive has no per-day ensemble spread to fit it; the WS0
harness tunes it on real graded outcomes.

Run from src/polymarket_weather/:   python train_calibrator.py
"""
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CITIES

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Fraction of the (realized) mean absolute-ish error kept as the sigma floor. Guards against the
# overconfidence the RF showed on low-spread days without inflating confident forecasts.
FLOOR_FRAC = 0.5
HOLDOUT_FRAC = 0.20   # most-recent 20% of dates held out for an honest error estimate
MIN_ROWS = 500        # need a reasonable history to fit a stable calibration


def _design(mu_raw: np.ndarray, doy: np.ndarray) -> np.ndarray:
    ang = 2.0 * np.pi * doy / 365.0
    return np.column_stack([np.ones_like(mu_raw), mu_raw, np.sin(ang), np.cos(ang)])


def _fit_nu(residuals: np.ndarray) -> float:
    """Estimate Student-t dof from residual excess kurtosis (excess = 6/(nu-4) for nu>4)."""
    try:
        from scipy.stats import kurtosis
        ex = float(kurtosis(residuals, fisher=True, bias=False))
    except Exception:
        return 7.0
    if ex <= 0.05:
        return 30.0            # ~Gaussian
    return float(np.clip(4.0 + 6.0 / ex, 4.0, 30.0))


def train_calibrator():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for city in CITIES.keys():
        city_slug = city.replace(" ", "_").lower()
        actuals_path = f"data/weather/{city_slug}_historical_actuals.csv"
        features_path = f"data/weather/{city_slug}_historical_features.csv"
        try:
            df_a = pd.read_csv(actuals_path)
            df_f = pd.read_csv(features_path)
        except Exception as e:
            logger.error(f"Missing archive data for {city}: {e} — run "
                         "fetch_historical_features.py and fetch_historical_truth.py first")
            continue

        for d in (df_a, df_f):
            d["date_local"] = pd.to_datetime(d["date_local"]).dt.strftime("%Y-%m-%d")
        df = pd.merge(df_f, df_a, on="date_local", how="inner").dropna(
            subset=["grid_temp_max_c", "temp_max_c"])
        df = df.sort_values("date_local").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            logger.warning(f"{city}: only {len(df)} merged rows (<{MIN_ROWS}) — skipping")
            continue

        doy = pd.to_datetime(df["date_local"]).dt.dayofyear.to_numpy()
        mu_raw = df["grid_temp_max_c"].to_numpy(dtype=float)
        y = df["temp_max_c"].to_numpy(dtype=float)

        # Temporal split: fit on the earlier dates, judge on the most-recent HOLDOUT_FRAC.
        cut = int(len(df) * (1.0 - HOLDOUT_FRAC))
        X = _design(mu_raw, doy)
        beta, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
        a, b, c_sin, c_cos = (float(v) for v in beta)

        # Honest holdout residuals (calibrated) and the raw (uncalibrated) baseline.
        pred_hold = X[cut:] @ beta
        resid = y[cut:] - pred_hold
        resid_sigma = float(np.std(resid))
        rmse_cal = float(np.sqrt(np.mean(resid ** 2)))
        rmse_raw = float(np.sqrt(np.mean((y[cut:] - mu_raw[cut:]) ** 2)))
        s0_floor = FLOOR_FRAC * resid_sigma
        nu = _fit_nu(resid)

        params = {
            "a": a, "b": b, "c_sin": c_sin, "c_cos": c_cos,
            "s0_floor": s0_floor, "s1": 1.0, "nu": nu,
            "resid_sigma": resid_sigma, "n_train": cut, "n_holdout": len(df) - cut,
            "holdout_rmse_calibrated": rmse_cal, "holdout_rmse_raw": rmse_raw,
        }
        out = os.path.join(MODELS_DIR, f"{city_slug}_emos.json")
        with open(out, "w") as f:
            json.dump(params, f, indent=2)
        logger.info(
            f"{city}: holdout RMSE raw {rmse_raw:.3f} → calibrated {rmse_cal:.3f} "
            f"(Δ{rmse_raw - rmse_cal:+.3f}), s0_floor {s0_floor:.3f}, nu {nu:.1f} → {out}")


if __name__ == "__main__":
    train_calibrator()
