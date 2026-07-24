"""train_qrf.py — per-city Quantile-Regression-Forest training + temporal holdout self-gate.

Mirrors train_calibrator.py's data loading: per city it reads the archived per-lead
multi-model forecasts ({slug}_historical_leads_mm.csv + the _cand / _jma companions,
all Open-Meteo Previous-Runs, already as-of the issue time) and joins them to the
settlement-faithful station truth ({slug}_settlement_actuals.csv via
settlement_truth.load_training_truth). Every (date, lead) row becomes one QRF feature
vector via qrf_features.build_row, y = observed daily Tmax. Rows are emitted in
date order so fit_city's temporal holdout (train early / test late) is honest.

The QRF is served only where it beats the raw ensemble on the city's holdout
(meta['beats_ensemble']). The gate is on a CRPS scale (°C): fit_city compares the QRF's
holdout CRPS proxy to the ensemble baseline's holdout CRPS proxy — the SAME
`crps_gaussian_proxy` formula on the SAME held-out rows, using the ensemble's own
(mu, sigma). The ensemble baseline here is the multi-model deterministic mean
(build_row's `mm_mean`, the exact information the forest also sees) with a Gaussian
spread set to the honest TRAIN-portion residual std — so both learner and baseline set
their parameters on train only and are scored on the identical holdout.

⚠️ LEADS below is an ASPIRATIONAL range (1-7), not what the archives deliver. The per-model
historical fetchers (`fetch_historical_leads_mm.py`, `_cand.py`) cap at `range(1, 5)` and the
jma companion file only has `jma1..jma4` — so leads 5-7 never find a `fcst_tmax_lead{n}_{m}`
column in `_assemble_xy` and are silently skipped (`forecasts` stays `{}`). In practice the QRF
is trained and self-gated on leads **1-4 only**. `train_qrf()` computes the true max lead
present in the assembled data (`X["lead"].max()`) and writes it as `max_lead` in the meta
sidecar; `predictors/qrf.py` refuses to serve (`predict_distribution` returns None) for
`days_ahead > max_lead`, so the untested 5-7 extrapolation regime falls back to EMOS/ensemble
instead of silently inheriting a `beats_ensemble` guarantee that was never validated there.

Leakage/alignment guarantees (mirroring train_calibrator + spec §4):
  * Features come only from previous-runs archives (already as-of the forecast issue
    time) — no target-day information.
  * The temporal holdout splits by row index over date-ordered rows; all leads of a
    given date are contiguous, so a date lands entirely in train OR entirely in holdout
    (bar the single boundary date) — no target-day leaks across the split.
  * The live ensemble spread (ens_mean/std/p10/p50/p90) has no per-lead historical
    archive, so those feature columns are left NaN in training (sklearn's RF handles the
    NaN and simply never splits on a constant-NaN feature). This is deliberate: fabricating
    ens_* from the model spread would be a DIFFERENT distribution than the 122-member
    live ensemble serves, i.e. a train/serve misalignment. The forest learns from the
    per-model forecasts + mm_mean/mm_std + lead + day-of-year, which ARE aligned.
  * Same as train_calibrator, station obs are NOT consumed here: intraday same-day
    conditioning (running max / is_same_day) is a live-only, lead-0 feature and is a
    separate concern; archived lead>=1 rows carry the -999 running-max sentinel and
    is_same_day=0.

Run from src/polymarket_weather/:   python train_qrf.py
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from predictors.qrf_core import QuantileForest, moment_match
from qrf_features import BASE_MODELS, FEATURE_COLS, build_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_Q = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]

LEADS = range(1, 8)     # aspirational upper bound the loader scans; the archives only ever
                        # populate leads 1-4 (see module docstring) — the true trained/served
                        # bound is computed per-city as `max_lead` in train_qrf(), not this constant.
_TRAIN_FRAC = 0.75          # temporal holdout: earliest 75% train, latest 25% test
_MIN_ROWS = 120             # per city; below this a QRF/holdout split is too small to gate


def crps_gaussian_proxy(y, mu, sigma):
    """Cheap continuous ranked probability score (°C scale) under a Gaussian predictive.

    The closed-form CRPS of N(mu, sigma) against observation y. Used identically for the
    QRF (moment-matched mu/sigma) and the ensemble baseline so the self-gate compares
    like with like — a °C-scale score, NOT a 0-1 Brier.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-3)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def _holdout_cut(n: int) -> int:
    """First holdout row index for a length-n date-ordered sample (train = [:cut])."""
    return int(n * _TRAIN_FRAC)


def fit_city(slug, X, y, ens_holdout_crps):
    """Fit the city's QRF on the temporal-train split, score its holdout CRPS proxy, and
    self-gate against `ens_holdout_crps` (the ensemble baseline's holdout CRPS proxy — the
    caller MUST compute it with `crps_gaussian_proxy` on the SAME held-out rows). Persist the
    forest (refit on ALL rows for serving) + a meta sidecar with the gate verdict.

    `ens_holdout_crps` is a CRPS-scale (°C) value, matching `holdout_crps` in the returned
    meta — never a 0-1 market Brier (that unit mismatch would make `beats_ensemble` always
    False and silently kill the gate).
    """
    X = X[FEATURE_COLS].to_numpy(float)
    y = np.asarray(y, float)
    n = len(y)
    cut = _holdout_cut(n)                                # temporal holdout (rows date-ordered)
    qf = QuantileForest().fit(X[:cut], y[:cut])
    q = qf.predict_quantiles(X[cut:], _Q)
    mm = np.array([moment_match(_Q, q[i]) for i in range(len(q))])   # (mu,sigma,nu) per row
    holdout = crps_gaussian_proxy(y[cut:], mm[:, 0], mm[:, 1])
    beats = holdout <= ens_holdout_crps
    qf_full = QuantileForest().fit(X, y)                # refit on all data for serving
    joblib.dump(qf_full, _MODELS_DIR / f"{slug}_qrf.joblib")
    meta = {
        "beats_ensemble": bool(beats),
        "holdout_crps": float(holdout),
        "ens_holdout_crps": float(ens_holdout_crps),
        "n": int(n),
    }
    (_MODELS_DIR / f"{slug}_qrf_meta.json").write_text(json.dumps(meta))
    return meta


def _load_city_frame(slug: str) -> pd.DataFrame | None:
    """Wide per-date training frame for a city, mirroring train_calibrator._train_city_target's
    Tmax join: {slug}_historical_leads_mm.csv merged (on date_local) with the _jma and _cand
    companion archives, then inner-joined to settlement truth's temp_max_c. Returns None when the
    core archive or truth is missing (fresh clone) — the caller skips the city with a warning.

    Columns out: date_local, temp_max_c, and fcst_tmax_lead{n}_{m} for whichever
    (lead n, model m) the archives cover.
    """
    from settlement_truth import load_training_truth

    mm_path = f"data/weather/{slug}_historical_leads_mm.csv"
    if not os.path.exists(mm_path):
        return None
    try:
        mm = pd.read_csv(mm_path)
        truth = load_training_truth(slug)
    except Exception as e:  # pragma: no cover - defensive I/O guard
        logger.error(f"{slug}: missing/unreadable archive data ({e})")
        return None

    mm["date_local"] = pd.to_datetime(mm["date_local"]).dt.strftime("%Y-%m-%d")

    # jma (Seoul) lives in its own file as jma{n}; rename to the fcst_tmax_lead{n}_jma schema.
    jma_path = f"data/weather/{slug}_historical_leads_jma.csv"
    if os.path.exists(jma_path):
        jma = pd.read_csv(jma_path)
        jma["date_local"] = pd.to_datetime(jma["date_local"]).dt.strftime("%Y-%m-%d")
        jma = jma.rename(columns={f"jma{n}": f"fcst_tmax_lead{n}_jma" for n in LEADS})
        mm = mm.merge(jma, on="date_local", how="left")

    # candidate blend models (aifs/gem/mf/…) already use the fcst_tmax_lead{n}_{m} schema.
    cand_path = f"data/weather/{slug}_historical_leads_cand.csv"
    if os.path.exists(cand_path):
        cand = pd.read_csv(cand_path)
        cand["date_local"] = pd.to_datetime(cand["date_local"]).dt.strftime("%Y-%m-%d")
        mm = mm.merge(cand, on="date_local", how="left")

    truth = truth.copy()
    truth["date_local"] = pd.to_datetime(truth["date_local"]).dt.strftime("%Y-%m-%d")
    df = mm.merge(truth[["date_local", "temp_max_c"]].dropna(), on="date_local", how="inner")
    return df


def _assemble_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Melt the wide per-date frame into date-ordered (X, y) rows, one per (date, lead).

    For each date (sorted ascending) and each lead 1..7, gather the present per-model
    forecasts (BASE_MODELS) and build a feature row via qrf_features.build_row with the
    live-only fields absent (ens={} → NaN; running_max=NaN sentinel; is_same_day=0). Rows
    with no model forecast at that lead are skipped. y = observed Tmax for the date.
    Contiguous per-date blocks keep the temporal holdout leak-free.
    """
    rows: list[dict] = []
    ys: list[float] = []
    work = df.sort_values("date_local")
    for _, r in work.iterrows():
        y = r.get("temp_max_c")
        if pd.isna(y):
            continue
        doy = pd.Timestamp(r["date_local"]).dayofyear
        for n in LEADS:
            forecasts = {}
            for m in BASE_MODELS:
                col = f"fcst_tmax_lead{n}_{m}"
                if col in work.columns:
                    v = r[col]
                    if pd.notna(v):
                        forecasts[m] = float(v)
            if not forecasts:                       # no model forecast at this lead
                continue
            rows.append(build_row(
                forecasts, {},                      # ens absent in the historical archive → NaN
                running_max=float("nan"),           # lead>=1 rows are pre-day: no running max
                is_same_day=0,
                lead=n,
                doy=doy,
            ))
            ys.append(float(y))
    X = pd.DataFrame(rows, columns=FEATURE_COLS) if rows else pd.DataFrame(columns=FEATURE_COLS)
    return X, np.asarray(ys, float)


def _ensemble_holdout_crps(X: pd.DataFrame, y: np.ndarray) -> float:
    """Ensemble-baseline holdout CRPS proxy on the SAME temporal holdout fit_city uses.

    Baseline: mu = the multi-model deterministic mean (build_row's `mm_mean` column, always
    finite for an emitted row), sigma = std of the TRAIN-portion residuals (fit on train only,
    exactly like the QRF). Scored with `crps_gaussian_proxy` on the holdout rows — identical
    formula, identical rows — so `fit_city`'s `beats_ensemble = qrf_holdout <= this` is a fair,
    unit-consistent (°C CRPS) comparison.
    """
    mu = X["mm_mean"].to_numpy(float)
    n = len(y)
    cut = _holdout_cut(n)
    resid_train = y[:cut] - mu[:cut]
    resid_train = resid_train[np.isfinite(resid_train)]
    sigma = float(np.std(resid_train)) if resid_train.size else float(np.std(y[:cut]))
    sigma = max(sigma, 0.1)
    return crps_gaussian_proxy(y[cut:], mu[cut:], sigma)


def train_qrf(cities=None):
    """Per city: load the archived per-lead multi-model forecasts + settlement truth, assemble
    date-ordered (X, y), compute the ensemble baseline's holdout CRPS proxy, then fit_city with
    the temporal-holdout self-gate. Writes models/{slug}_qrf.joblib + {slug}_qrf_meta.json.

    After fit_city persists its meta, this also computes `max_lead` — the actual maximum
    `lead` value present in the assembled training rows (NOT the aspirational LEADS constant;
    see module docstring) — and adds it to the on-disk meta sidecar. That's the train/serve
    safety bound predictors/qrf.py enforces: it must never be trusted to extrapolate the
    `beats_ensemble` gate to leads the archive never actually trained/validated on.
    """
    if cities is None:
        from config import CITIES
        cities = list(CITIES.keys())
    os.makedirs(_MODELS_DIR, exist_ok=True)
    results = {}
    for city in cities:
        slug = city.replace(" ", "_").lower()
        df = _load_city_frame(slug)
        if df is None or df.empty:
            logger.warning(f"{city}: no training archive/truth on disk — skipping QRF.")
            continue
        X, y = _assemble_xy(df)
        if len(y) < _MIN_ROWS:
            logger.warning(f"{city}: only {len(y)} QRF rows (< {_MIN_ROWS}) — skipping.")
            continue
        ens_crps = _ensemble_holdout_crps(X, y)
        meta = fit_city(slug, X, y, ens_crps)

        # Record the true trained/served lead bound (computed, not hardcoded): the max
        # `lead` feature value actually present in the assembled rows. Loaded back from the
        # just-written meta sidecar and rewritten, rather than threaded through fit_city's
        # signature, to keep Task 4's fit_city test untouched.
        max_lead = int(X["lead"].max()) if len(X) else 0
        meta_path = _MODELS_DIR / f"{slug}_qrf_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["max_lead"] = max_lead
        meta_path.write_text(json.dumps(meta))

        results[slug] = meta
        logger.info(
            f"{city}: QRF n={meta['n']} holdout_crps={meta['holdout_crps']:.3f} "
            f"vs ensemble {ens_crps:.3f} -> beats_ensemble={meta['beats_ensemble']} "
            f"max_lead={max_lead}")
    return results


if __name__ == "__main__":
    train_qrf()
