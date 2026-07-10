r"""
EMOSPredictor v2 — per-lead Nonhomogeneous Regression on real archived forecasts
================================================================================

v1 trained a single EMOS on the ERA5 *reanalysis* archive. That had two fatal skews:
the mean correction was learned from a different model family than the live forecast,
and — much worse — the sigma floor reflected same-day analysis error (~0.5–0.8 °C)
while the engine bets at 1–3-day leads where real error is 1.3–2.9 °C. The result was
2–3× overconfident tails: the eval's [0, 0.1) probability bin realized at ~15%.

v2 (see train_calibrator.py) is trained on archived real forecasts at explicit lead
times (Open-Meteo Previous Runs API, 2022→now) against resolution-faithful station
truth. Per city and per lead it provides:

  * a mean correction  mu = a + b·f + c_sin·sin + c_cos·cos  — self-gated PER LEAD:
    applied only where it beat the raw input on the temporal holdout;
  * an honest dispersion sigma(lead) — the holdout residual std at that lead. Serving
    sigma is max(ens_std + diurnal_boost, sigma_lead): the flow-dependent ensemble
    spread only takes over when it signals MORE uncertainty than the climatological
    error at that lead. The sigma floor is never gated away — it is the main fix;
  * a Student-t nu(lead) fit from holdout residual kurtosis.

The serving mean input follows params["input"]:
  * "mm_mean"    — the live multi-model ensemble mean (ens_df), whose training proxy
    was the ECMWF/GFS/ICON previous-run mean;
  * "best_match" — the latest deterministic forecast (daily_df).
If the preferred input is unavailable at prediction time, the other input is used WITH
ITS OWN fit (never mixing one input's value with the other's coefficients).

Fallback: no params, or no usable mean input → the pure EnsemblePredictor
(which itself falls back to the NWP table).
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BasePredictor, TemperatureDistribution
from .ensemble import EnsemblePredictor, get_ensemble_params
from .nwp_fallback import spread_sigma_boost, get_nwp_params

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_MIN_LEAD, _MAX_LEAD = 1, 7

logger = logging.getLogger(__name__)

_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    """Log *msg* at WARNING once per distinct *key* (per process) — surfaces silent
    serving degradation (B4) without spamming every prediction."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg)


def _apply_censor(dist, city_slug, target_date, fetch_time, obs_df, kind):
    """Apply same-day censoring (floor for max / ceiling for min) to ANY distribution, so a
    fallback distribution (raw EnsemblePredictor / NWP) honors the running observed extreme
    too. The EMOS path already floors/ceilings; the fallback path silently did not, so it
    could price bins the observed running max/min has already made impossible (C3)."""
    if dist is None:
        return dist
    state = _intraday_state(city_slug, target_date, fetch_time, obs_df, kind=kind)
    if state is not None:
        _, run_val = state
        if kind == "max":
            dist.floor = run_val
        else:
            dist.ceiling = run_val
    return dist


@lru_cache(maxsize=None)
def _load_params(city_slug: str, kind: str = "max"):
    """Per-city EMOS v2 params dict for the daily MAX or MIN, or None if not trained."""
    suffix = "_min" if kind == "min" else ""
    path = _MODELS_DIR / f"{city_slug}_emos{suffix}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            params = json.load(f)
    except Exception:
        return None
    return params if params.get("version") == 2 and params.get("leads") else None


@lru_cache(maxsize=None)
def _load_intraday(city_slug: str, kind: str = "max"):
    """Per-city intraday params (train_intraday.py), or None."""
    suffix = "_min" if kind == "min" else ""
    path = _MODELS_DIR / f"{city_slug}_intraday{suffix}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            params = json.load(f)
    except Exception:
        return None
    return params if params.get("hours") else None


@lru_cache(maxsize=None)
def _city_tz(city_slug: str):
    from config import CITIES
    for name, cfg in CITIES.items():
        if name.replace(" ", "_").lower() == city_slug:
            return cfg["timezone"]
    return None


def _intraday_state(city_slug: str, target_date, fetch_time, obs_df, kind: str = "max"):
    """(local_hour, running max/min °C) when fetch_time falls ON the target's
    station-local day and observations for that day exist up to fetch_time; else None."""
    if obs_df is None or obs_df.empty:
        return None
    tz = _city_tz(city_slug)
    if tz is None:
        return None
    local = pd.Timestamp(fetch_time).tz_convert(tz)
    target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    if local.strftime("%Y-%m-%d") != target_str:
        return None
    day = obs_df[
        (obs_df["date_local"] == target_str) &
        (obs_df["valid_local"] <= local.tz_localize(None))
    ]
    if day.empty:
        return None
    # Censoring uses the running value through fetch_time (a physical fact).
    floor_run_val = float(day["temp_c"].max() if kind == "max" else day["temp_c"].min())
    # C4: the per-hour fit for hour h was trained on the running value through the END of hour h,
    # so serving must apply it to the LAST COMPLETED hour (< the current partial hour, whose
    # running value is biased low for Tmax / high for Tmin). fit_hour=None until an hour completes.
    completed = day[day["valid_local"].dt.hour <= local.hour - 1]
    if completed.empty:
        fit_hour, fit_run_val = None, None
    else:
        fit_hour = int(local.hour - 1)
        fit_run_val = float(completed["temp_c"].max() if kind == "max"
                            else completed["temp_c"].min())
    return fit_hour, fit_run_val, floor_run_val


def _lead_entry(params: dict, days_ahead: float):
    """The per-lead fit dict for the nearest trained lead."""
    lead = int(np.clip(round(days_ahead), _MIN_LEAD, _MAX_LEAD))
    leads = params["leads"]
    if str(lead) in leads:
        return leads[str(lead)]
    trained = sorted(int(k) for k in leads)
    nearest = min(trained, key=lambda n: abs(n - lead))
    return leads[str(nearest)]


def emos_mean(fit: dict, mu_raw: float, day_of_year: int) -> float:
    """Calibrated mean — or the raw input where the holdout gated the correction off."""
    if not fit.get("use_calibrated_mean", False):
        return mu_raw
    ang = 2.0 * np.pi * day_of_year / 365.0
    return (float(fit["a"]) + float(fit["b"]) * mu_raw
            + float(fit["c_sin"]) * np.sin(ang) + float(fit["c_cos"]) * np.cos(ang))


def _latest_deterministic_mu(daily_df: pd.DataFrame, target_date, fetch_time,
                             col: str = "temp_max_c"):
    """Most recent deterministic forecast (max or min) for target_date at/-before fetch_time."""
    if daily_df is None or daily_df.empty or col not in daily_df.columns:
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
    val = sub.iloc[-1][col]
    return None if pd.isna(val) else float(val)


def _latest_nbm(nbm_df: pd.DataFrame, target_date, fetch_time):
    """AS-OF NBM Tmax for target_date: the latest run whose avail_utc <= fetch_time.
    Prefers the official 12-h max guidance (txn); falls back to the 3-hourly-tmp max."""
    if nbm_df is None or nbm_df.empty:
        return None
    target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    sub = nbm_df[
        (nbm_df["date_local"] == target_str) &
        (nbm_df["avail_utc"] <= fetch_time)
    ].sort_values("avail_utc")
    if sub.empty:
        return None
    row = sub.iloc[-1]
    for col in ("nbm_tmax_txn_c", "nbm_tmax_tmp_c"):
        val = row.get(col)
        if val is not None and not pd.isna(val):
            return float(val)
    return None


def _latest_mm_mean(mm_df: pd.DataFrame, target_date, fetch_time, mm_models,
                    prefix: str = "tmax"):
    """Mean of the trained model set's latest live forecasts for target_date, or None.

    This is the EXACT serving counterpart of the training input (the same models'
    previous-run mean), so it is preferred over the ensemble-mean proxy. Requires the
    full trained model set — a partial mean is a different estimator.
    """
    if mm_df is None or mm_df.empty or not mm_models:
        return None
    cols = [f"{prefix}_{m}" for m in mm_models]
    if any(c not in mm_df.columns for c in cols):
        return None
    td = pd.Timestamp(target_date).normalize()
    if td.tzinfo is not None:
        td = td.tz_localize(None)
    sub = mm_df[
        (mm_df["date_local"].dt.normalize() == td) &
        (mm_df["fetched_at_utc"] <= fetch_time)
    ].sort_values("fetched_at_utc")
    if sub.empty:
        return None
    row = sub.iloc[-1][cols]
    if row.isna().any():
        return None
    return float(row.mean())


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
        mm_df: pd.DataFrame = None,
        obs_df: pd.DataFrame = None,
        nbm_df: pd.DataFrame = None,
        kind: str = "max",
    ) -> TemperatureDistribution:
        """Predictive distribution of the daily MAX (kind='max') or MIN (kind='min').

        For kind='min' with no trained Tmin params, returns None — the engine must then
        skip 'lowest temperature' bins rather than price them off a Tmax distribution.
        """
        city_slug = city.replace(" ", "_").lower()
        params = _load_params(city_slug, kind)
        if params is None:
            if kind == "min":
                return None
            return _apply_censor(
                self._ensemble_predictor.predict_distribution(
                    city, target_date, fetch_time, days_ahead, daily_df, ens_df),
                city_slug, target_date, fetch_time, obs_df, kind)

        entry = _lead_entry(params, days_ahead)
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        det_col = "temp_max_c" if kind == "max" else "temp_min_c"
        det_mu = _latest_deterministic_mu(daily_df, target_date, fetch_time, col=det_col)

        # Input values. The exact multi-model mean (same models as training) is served
        # from daily_mm; when it is unavailable (pre-daily_mm snapshots, model outage)
        # the live ensemble mean is used WITH ITS OWN "mm_proxy" fit — the proxy input
        # is blunter than the full blend, so it must carry the proxy's wider sigma.
        mm_prefix = "tmax" if kind == "max" else "tmin"
        mm_mu = _latest_mm_mean(mm_df, target_date, fetch_time,
                                params.get("mm_models"), prefix=mm_prefix)
        ens_mean_key = "ens_mean" if kind == "max" else "ens_min_mean"
        ens_std_key = "ens_std" if kind == "max" else "ens_min_std"
        ens_mu = None if ens_params is None else ens_params.get(ens_mean_key)
        if ens_mu is not None and np.isnan(ens_mu):
            ens_mu = None

        # Mean input in preference order, each paired with ITS OWN fit. The trained
        # per-city preference (params["input"]) leads; the rest follow as fallbacks.
        value_by_kind = {
            "nbm": _latest_nbm(nbm_df, target_date, fetch_time) if kind == "max" else None,
            "mm_mean": mm_mu,
            "mm_proxy": ens_mu,
            "best_match": det_mu,
        }
        # Fallback ranking follows measured holdout quality: the exact blend, then the
        # ensemble-mean proxy, then NBM (tested WORSE than both at KLGA/KORD — kept as a
        # self-gating candidate), then the single deterministic forecast.
        fallback_order = ("mm_mean", "mm_proxy", "nbm", "best_match")
        order = (params["input"],) + tuple(k for k in fallback_order if k != params["input"])
        candidates = []
        for input_kind in order:
            value = value_by_kind.get(input_kind)
            fit = entry.get(input_kind)
            if value is not None and fit is not None:
                candidates.append((input_kind, value, fit))
        if not candidates:
            if kind == "min":
                _warn_once(
                    f"{city}:min:no_input",
                    f"EMOS {city} (min): no usable Tmin mean input "
                    f"(mm_mean/mm_proxy/best_match all unavailable) — Tmin bins skipped. "
                    f"Check tmin_*/ens_min_* columns on disk (append-freeze?).",
                )
                return None
            _warn_once(
                f"{city}:max:ensemble_fallback",
                f"EMOS {city} (max): no usable calibrated input — falling back to the raw "
                f"EnsemblePredictor. Check daily_mm/ensemble schema.",
            )
            return _apply_censor(
                self._ensemble_predictor.predict_distribution(
                    city, target_date, fetch_time, days_ahead, daily_df, ens_df),
                city_slug, target_date, fetch_time, obs_df, kind)
        selected_kind, mu_raw, fit = candidates[0]
        if selected_kind != params["input"]:
            _warn_once(
                f"{city}:{kind}:degraded:{selected_kind}",
                f"EMOS {city} ({kind}): preferred input '{params['input']}' unavailable at "
                f"serve time — degraded to '{selected_kind}' (blunter). Check the "
                f"daily_mm/ensemble columns for this city (append-freeze?).",
            )

        day_of_year = pd.Timestamp(target_date).dayofyear
        mu = emos_mean(fit, mu_raw, day_of_year)

        # Dispersion: never below the honest per-lead residual sigma; the flow-dependent
        # ensemble spread takes over only when it signals more. The convective diurnal
        # boost applies to the MAX only (it widens afternoon peaks, not overnight lows).
        sigma_lead = float(fit["sigma"])
        ens_std_val = None if ens_params is None else ens_params.get(ens_std_key)
        if kind == "max":
            s_boost = spread_sigma_boost(daily_df, target_date, fetch_time)
            if ens_std_val is not None and not np.isnan(ens_std_val):
                flow_sigma = max(0.0, float(ens_std_val)) + s_boost
            else:
                nwp_sigma, _ = get_nwp_params(days_ahead)
                flow_sigma = nwp_sigma + s_boost
        else:
            flow_sigma = (float(ens_std_val)
                          if ens_std_val is not None and not np.isnan(ens_std_val) else 0.0)
        sigma = max(flow_sigma, sigma_lead)
        nu = float(fit.get("nu", 8.0))
        source = f"emos_v2_{kind}" if kind == "min" else "emos_v2"
        floor = None
        ceiling = None

        # Intraday conditioning on the target's own station-local day:
        #   max — Tmax cannot end below the running observed max (floor);
        #   min — Tmin cannot end above the running observed min (ceiling; the overnight
        #         low is usually locked in by mid-morning, a large information edge).
        # From the self-gated hours, the per-hour regression on (forecast, running value)
        # replaces mu/sigma — by mid-afternoon its holdout sigma is 2-3× tighter than any
        # pure forecast.
        state = _intraday_state(city_slug, target_date, fetch_time, obs_df, kind=kind)
        if state is not None:
            fit_hour, fit_run_val, floor_run_val = state
            if kind == "max":
                floor = floor_run_val
            else:
                ceiling = floor_run_val
            intr = _load_intraday(city_slug, kind)
            fit_h = None if (intr is None or fit_hour is None) else intr["hours"].get(str(fit_hour))
            if fit_h is not None:
                # C5: feed the DAY-AHEAD (lead-1) deterministic forecast the fit was trained on —
                # the newest run issued >=1 day before the target's UTC midnight — not the latest
                # (~lead-0) det_mu. Fall back to det_mu / mu_raw when that run is unavailable.
                lead1_cutoff = pd.Timestamp(target_date).normalize()
                lead1_cutoff = (lead1_cutoff.tz_localize("UTC") if lead1_cutoff.tzinfo is None
                                else lead1_cutoff) - pd.Timedelta(days=1)
                fcst_in = _latest_deterministic_mu(daily_df, target_date, lead1_cutoff, col=det_col)
                if fcst_in is None:
                    fcst_in = det_mu if det_mu is not None else mu_raw
                mu = (float(fit_h["a"]) + float(fit_h["b"]) * fcst_in
                      + float(fit_h["c"]) * fit_run_val)
                sigma = float(fit_h["sigma"])
                nu = float(intr.get("nu", nu))
                source += "_intraday" if not source.endswith("intraday") else ""

        return TemperatureDistribution(
            mu=mu,
            sigma=sigma,
            nu=nu,
            source=source,
            floor=floor,
            ceiling=ceiling,
        )
