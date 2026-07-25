r"""
QRFPredictor — serves a moment-matched Student-t distribution from a per-city
Quantile Regression Forest (predictors/qrf_core.QuantileForest), self-gated against
the raw ensemble (models/{slug}_qrf_meta.json["beats_ensemble"]).

v1 scope is TMAX ONLY (see docs/superpowers/specs/2026-07-24-qrf-forecaster-design.md
§1/§6 — Tmin is an explicit fast-follow, not built yet: there is no {slug}_qrf_min
artifact). `kind="min"` therefore always returns None so the engine falls back to
EMOS/ensemble for Tmin bins, exactly like an un-gated/missing-artifact city.

Self-gate: if the artifact or its meta sidecar is missing, or meta["beats_ensemble"]
is False, `predict_distribution` returns None — the engine's fallback chain (EMOS ->
raw ensemble) then handles the bin. This is the guarantee that QRF is never served
worse than the ensemble it is trained to beat.

Serving reuses EMOSPredictor's exact leakage-safe as-of extraction helpers rather than
re-deriving them (see predictors/emos.py):
  * `_latest_mm_mean` — called once per model with a SINGLETON model list, which
    collapses its "mean of the trained set" to that one model's own as-of value. This
    gets the per-model multi-model forecasts with the identical row-selection/NaN
    handling EMOS uses for its blended mm_mean, without inventing a second extraction.
  * `_latest_deterministic_mu` — the single deterministic ("best_match") forecast,
    used as EMOS-style fallback filler when no live per-model columns are available yet
    (pre-daily_mm snapshot / model outage), instead of silently serving an all-NaN row.
  * `get_ensemble_params` — the live ensemble mean/std/p10/p90 as-of fetch_time.
  * `_city_tz` — the same city -> IANA timezone lookup EMOS uses for its own
    same-day/intraday logic.

The quantile ladder [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95] (matching
train_qrf.py's `_Q` and qrf_core.moment_match's expected levels) is predicted by the
loaded QuantileForest and moment-matched to (mu, sigma, nu). `floor` is set to the
running observed max ONLY for same-day Tmax bets (qrf_features.intraday_running_max,
truncated strictly before fetch_time — no look-ahead); otherwise floor is None.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base import BasePredictor, TemperatureDistribution
from .qrf_core import moment_match, empirical_cdf_from_quantiles, Q_FINE
from .emos import _latest_mm_mean, _latest_deterministic_mu, _city_tz
from .ensemble import get_ensemble_params
from qrf_features import BASE_MODELS, FEATURE_COLS, build_row, intraday_running_max

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_Q_LEVELS = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]

# _Q_LEVELS's exact values all live in Q_FINE (0.01-step grid), so a single
# predict_quantiles(Q_FINE) call can serve both: the 7-level summary is read back
# out by nearest-index lookup instead of firing a second (expensive) leaf-weighting
# pass over the same feature row.
_Q_FINE_ARR = np.asarray(Q_FINE, dtype=float)
_Q7_IDX = [int(np.argmin(np.abs(_Q_FINE_ARR - lvl))) for lvl in _Q_LEVELS]


def _load_artifact(city_slug: str):
    """(QuantileForest, meta dict) for city_slug, or (None, None) when the joblib
    artifact / meta sidecar is missing, unreadable, or the city hasn't beaten the
    ensemble on its holdout. Reads fresh each call (no lru_cache): the meta sidecar
    is the self-gate switch, so a retrain flipping it must be picked up immediately
    rather than serving a cached stale verdict."""
    model_path = _MODELS_DIR / f"{city_slug}_qrf.joblib"
    meta_path = _MODELS_DIR / f"{city_slug}_qrf_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None, None
    if not meta.get("beats_ensemble", False):
        return None, None
    try:
        model = joblib.load(model_path)
    except Exception:
        return None, None
    return model, meta


def _model_forecasts(mm_df, daily_df, target_date, fetch_time, kind: str) -> dict:
    """Per-model latest live forecasts (BASE_MODELS keys), as-of fetch_time.

    Reuses EMOSPredictor's `_latest_mm_mean` per SINGLE model (a one-model list
    collapses its "mean of the trained set" to that model's own as-of value) so this
    never re-derives the leakage-safe row-selection/NaN-handling logic. Falls back to
    spreading the single deterministic forecast (`_latest_deterministic_mu`, the same
    'best_match' EMOS degrades to) across the always-present ecmwf/gfs/icon trio when
    no live per-model columns exist yet — mirrors EMOS's own mm_mean -> best_match
    degrade chain instead of silently serving an all-NaN feature row.
    """
    prefix = "tmax" if kind == "max" else "tmin"
    out = {}
    for m in BASE_MODELS:
        val = _latest_mm_mean(mm_df, target_date, fetch_time, [m], prefix=prefix)
        if val is not None:
            out[m] = val
    if not out:
        det_col = "temp_max_c" if kind == "max" else "temp_min_c"
        det_mu = _latest_deterministic_mu(daily_df, target_date, fetch_time, col=det_col)
        if det_mu is not None:
            out = {"ecmwf": det_mu, "gfs": det_mu, "icon": det_mu}
    return out


def _ens_dict(ens_params) -> dict:
    """Map `get_ensemble_params`'s keys onto the {mean,std,p10,p50,p90} shape
    `qrf_features.build_row` expects. `ens_mean`/`ens_std` can be an explicit None
    (not just absent) when the max stats are unusable — coerced to NaN so
    `build_row`'s `float(...)` never sees a bare None. There is no stored p50; the
    mean is used as its proxy (one of several RF input features, not load-bearing)."""
    if ens_params is None:
        return {}
    mean = ens_params.get("ens_mean")
    std = ens_params.get("ens_std")
    mean = float("nan") if mean is None else mean
    std = float("nan") if std is None else std
    return {
        "mean": mean,
        "std": std,
        "p10": ens_params.get("ens_p10", float("nan")),
        "p90": ens_params.get("ens_p90", float("nan")),
        "p50": mean,
    }


class QRFPredictor(BasePredictor):
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
        """Predictive distribution of the daily MAX, moment-matched from the QRF's
        quantile ladder. Returns None (engine falls back to EMOS/ensemble) when:
          * kind != "max" (Tmin is out of v1 scope — no artifact exists for it), or
          * the city's {slug}_qrf.joblib / _qrf_meta.json is missing, or
          * the city's meta["beats_ensemble"] is False (self-gate), or
          * days_ahead exceeds meta["max_lead"] (train/serve safety bound — see below).

        `max_lead` guard: train_qrf.py's per-lead archives (fetch_historical_leads_mm.py /
        _cand.py) only ever cover leads 1-4, so the `beats_ensemble` self-gate is validated
        ONLY on leads 1-4 even though this predictor has no inherent bound on `days_ahead`.
        Without this check, real "2d+" bucket traffic (leads 5-7) would silently inherit a
        guarantee ("never served worse than the ensemble") that was never tested at those
        leads. `max_lead` is written by train_qrf() from the actual assembled training data;
        when it's absent from the meta (older/test artifacts written before this fix), no
        restriction is applied so existing fixtures/behavior are unaffected.
        """
        if kind != "max":
            return None

        city_slug = city.replace(" ", "_").lower()
        model, meta = _load_artifact(city_slug)
        if model is None:
            return None
        max_lead = meta.get("max_lead")
        if max_lead is not None and days_ahead > max_lead:
            return None

        model_forecasts = _model_forecasts(mm_df, daily_df, target_date, fetch_time, kind)
        ens_params = get_ensemble_params(ens_df, target_date, fetch_time)
        ens = _ens_dict(ens_params)

        # Same-day determination mirrors EMOS's own inline check (_intraday_state):
        # station-local calendar date of fetch_time == target_date. Computed
        # independently of obs_df availability, because is_same_day is a genuine
        # calendar fact the RF should see even when obs are missing (the running-max
        # sentinel -999.0, from qrf_features.build_row, covers that case) — unlike
        # EMOS's _intraday_state, which conflates "same day" with "obs data present"
        # and would wrongly report not-same-day when obs are simply absent.
        tz = _city_tz(city_slug)
        is_same_day = False
        running_max = float("nan")
        if tz is not None:
            local = pd.Timestamp(fetch_time).tz_convert(tz)
            target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
            is_same_day = local.strftime("%Y-%m-%d") == target_str
            if is_same_day:
                # qrf_features.intraday_running_max wraps its tz arg in ZoneInfo(tz),
                # so it needs the IANA key string — config.CITIES stores a ZoneInfo
                # instance (used directly above for tz_convert), hence str(tz) here.
                running_max = intraday_running_max(obs_df, target_date, fetch_time, str(tz))

        doy = pd.Timestamp(target_date).dayofyear
        row = build_row(
            model_forecasts,
            ens,
            running_max=running_max,   # always a float (NaN when unknown), never None
            is_same_day=1 if is_same_day else 0,
            lead=days_ahead,
            doy=doy,
        )
        X = np.array([[row[c] for c in FEATURE_COLS]], dtype=float)
        # Single predict_quantiles call over the fine 99-level grid (Q_FINE) -- one
        # leaf-weighting pass per row regardless of how many levels are requested, so
        # deriving the 7-level summary from it (nearest Q_FINE index per _Q_LEVELS
        # entry) avoids a second, redundant leaf-weighting call for the same row.
        q_fine = model.predict_quantiles(X, Q_FINE)[0]
        q7 = q_fine[_Q7_IDX]
        mu, sigma, nu = moment_match(_Q_LEVELS, q7)
        cdf = empirical_cdf_from_quantiles(Q_FINE, q_fine)

        # Intraday floor: same-day Tmax cannot end below what has already been
        # recorded. running_max == running_max is the NaN guard (NaN != NaN).
        floor = running_max if (is_same_day and running_max == running_max) else None

        return TemperatureDistribution(mu=mu, sigma=sigma, nu=nu, source="qrf", floor=floor, cdf=cdf)
