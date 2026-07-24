"""qrf_features.py — assemble QRF feature vectors, strictly as-of the snapshot (no look-ahead).
Model-forecast columns are per the city's MM_MODELS_BY_CITY set; the intraday running-max is the
same-day information edge (obs truncated at fetch_time)."""
from __future__ import annotations
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo

# ecmwf/gfs/icon are always present; extra city models fill 0 + a presence flag handled by the RF.
BASE_MODELS = ["ecmwf", "gfs", "icon", "aifs", "gem", "mf", "jma"]
FEATURE_COLS = ([f"m_{m}" for m in BASE_MODELS]
                + ["mm_mean", "mm_std", "ens_mean", "ens_std", "ens_p10", "ens_p50", "ens_p90",
                   "lead", "doy_sin", "doy_cos", "running_max", "is_same_day"])


def intraday_running_max(obs_df, target_date, fetch_time, tz):
    if obs_df is None or obs_df.empty:
        return float("nan")
    v = pd.to_datetime(obs_df["valid_local"])
    if v.dt.tz is None:
        v = v.dt.tz_localize(ZoneInfo(tz))
    day = pd.Timestamp(target_date).date()
    ft = pd.Timestamp(fetch_time)
    if ft.tz is None:
        ft = ft.tz_localize(ZoneInfo(tz))
    m = (v.dt.date == day) & (v < ft)
    return float(obs_df.loc[m.values, "temp_c"].max()) if m.any() else float("nan")


def build_row(model_forecasts, ens, running_max, is_same_day, lead, doy):
    row = {f"m_{m}": float(model_forecasts.get(m, np.nan)) for m in BASE_MODELS}
    present = [model_forecasts[m] for m in BASE_MODELS if m in model_forecasts and model_forecasts[m] == model_forecasts[m]]
    row["mm_mean"] = float(np.mean(present)) if present else np.nan
    row["mm_std"] = float(np.std(present)) if len(present) > 1 else 0.0
    row["ens_mean"] = float(ens.get("mean", np.nan)); row["ens_std"] = float(ens.get("std", np.nan))
    row["ens_p10"] = float(ens.get("p10", np.nan)); row["ens_p50"] = float(ens.get("p50", np.nan))
    row["ens_p90"] = float(ens.get("p90", np.nan))
    row["lead"] = float(lead)
    row["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    row["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
    row["running_max"] = float(running_max) if running_max == running_max else -999.0   # NaN sentinel
    row["is_same_day"] = float(is_same_day)
    return row
