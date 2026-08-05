"""train_calibrator.py — fit the per-city, PER-LEAD EMOS calibrators (v2), Tmax AND Tmin.

v1 trained a single EMOS on the ERA5 *reanalysis* archive: same-day analysis values,
not forecasts. That understated live forecast error 2–3× at the 1–3-day leads the
engine actually bets (measured: archived-forecast RMSE 1.3–2.9 °C vs v1 sigma floors of
0.5–0.8 °C), producing the overconfident tails that lost on Brier.

v2 trains on REAL archived forecasts at explicit lead times (fetch_historical_leads*.py,
Open-Meteo Previous Runs API, 2022→now) against the resolution-faithful station truth
(fetch_historical_truth.py). Per city, per target (daily MAX → {slug}_emos.json, daily
MIN → {slug}_emos_min.json) and per lead 1..7 it fits:

    mu_cal(ℓ)  = a + b·fcst(ℓ) + c_sin·sin(2π·doy/365) + c_cos·cos(2π·doy/365)
    sigma(ℓ)   = std of the temporal-holdout residuals  (honest dispersion at that lead)

Candidate mean inputs (each with its own fit; serving picks by params["input"] with the
rest as fallbacks): `best_match` (deterministic), `mm_mean` (per-city multi-model blend,
see MM_MODELS_BY_CITY), `mm_proxy` (3-model mean matching the live ensemble families —
the honest fallback when exact multi-model inputs are missing), and for US-city Tmax
`nbm` (NWS National Blend station guidance — tested WORSE than the blend, kept as a
self-gating candidate). Input selection is scored on PER-LEAD COMMON WINDOWS (paired
dates), since the archives start at different times.

Self-gating is PER LEAD and applies to the MEAN only: where the calibrated mean does not
beat the raw input on the holdout, serving uses the raw input — but the per-lead sigma
always applies (the spread fix must never be gated away; it is the main correction).

Run from src/polymarket_weather/:   python train_calibrator.py
"""
import json
from datetime import datetime, timezone
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CITIES

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

LEADS = range(1, 8)
# Per-city multi-model sets, from the 2026-07-03 blend-expansion sweep (greedy forward
# selection on paired common-coverage temporal holdouts, >2% RMSE margin; AIFS judged on
# its own shorter archive window, adopted only where it improved EVERY lead 1-3):
#   GEM  helps everywhere (adopted except NYC where it missed the margin);
#   MF   (Météo-France) helps NYC & London; CMA/BOM rejected (hurt or tiny window);
#   AIFS (ECMWF AI model, archive from mid-2024) adopted Seoul/NYC/London — Seoul gains
#        5-8% at ALL leads, exactly the measured lead-2-3 deficit. Hurts HK; mixed Chicago.
#   JMA  Seoul only (lead1 1.41→1.25); neutral HK.
# Serving reads the set from the params JSON (mm_models) and averages the same models'
# live forecasts ({slug}_daily_mm.csv) — keep fetch_weather.MULTIMODEL_MODELS a superset.
# NOTE: rows must have the FULL set, so AIFS cities train on its shorter window (~360d);
# the paired-holdout tests above already price that in. Re-sweep as archives grow.
# The same sets are reused for the Tmin calibrations (sensible prior; re-sweep later).
MM_MODELS_DEFAULT = ("ecmwf", "gfs", "icon", "gem")
MM_MODELS_BY_CITY = {
    "seoul": ("ecmwf", "gfs", "icon", "jma", "gem", "aifs"),
    "london": ("ecmwf", "gfs", "icon", "gem", "mf", "aifs"),
    "new_york_city": ("ecmwf", "gfs", "icon", "mf", "aifs"),
}
# Proxy fit: when serving lacks the exact multi-model inputs (pre-daily_mm snapshots, a
# model outage), the predictor falls back to the live 122-member ensemble mean — whose
# model families are ECMWF/GFS/ICON. A calibration fit on THAT mean keeps the fallback's
# sigma honest; applying the sharper full-blend sigma to the blunter proxy input would be
# silently overconfident.
MM_PROXY_MODELS = ("ecmwf", "gfs", "icon")
HOLDOUT_FRAC = 0.20   # most-recent 20% of dates held out for honest error estimates
MIN_ROWS = 400        # per lead; ~4.5y of archive gives ~1300+
MIN_ROWS_MM = 250     # mm-mean fits may sit on the shorter AIFS window

# Target specs: everything that differs between the Tmax and Tmin calibrations.
TARGET_SPECS = {
    "max": {
        "target_col": "temp_max_c",
        "bm_file": "data/weather/{slug}_historical_leads.csv",
        "bm_col": "fcst_tmax_lead{n}",
        "mm_file": "data/weather/{slug}_historical_leads_mm.csv",
        "model_col": "fcst_tmax_lead{n}_{m}",
        "merge_jma_cand": True,   # jma/cand live in separate files for the max target
        "use_nbm": True,
        "out_file": "{slug}_emos.json",
    },
    "min": {
        "target_col": "temp_min_c",
        "bm_file": "data/weather/{slug}_historical_leads_min.csv",
        "bm_col": "fcst_tmin_lead{n}",
        "mm_file": "data/weather/{slug}_historical_leads_min_mm.csv",
        "model_col": "fcst_tmin_lead{n}_{m}",
        "merge_jma_cand": False,  # the min mm file already carries all 7 models
        "use_nbm": False,
        "out_file": "{slug}_emos_min.json",
    },
}


def _design(f: np.ndarray, doy: np.ndarray) -> np.ndarray:
    ang = 2.0 * np.pi * doy / 365.0
    return np.column_stack([np.ones_like(f), f, np.sin(ang), np.cos(ang)])


def _fit_nu(residuals: np.ndarray) -> float:
    """Student-t dof from residual excess kurtosis (excess = 6/(nu-4) for nu>4).

    NOTE (b, tested & REJECTED 2026-07-15): a conditional-MLE dof (best nu given the honest
    empirical std) was tried in place of this moment match to fatten the tails behind the
    [0,0.1) probability bucket (predicted 0.04, realized 0.19). It did NOT help — that bucket
    is ADVERSE SELECTION, not thin tails: on those bins the market prices 0.17 and they realize
    0.19 (the market is right; the model flags them only because it disagreed by ≥6pp). The MLE
    dof left Brier flat and cost the calibrator its paired-CRPS win over the ensemble, so it was
    reverted. Widening dispersion cannot fix a wrong center on an adversely-selected sample.
    """
    try:
        from scipy.stats import kurtosis
        ex = float(kurtosis(residuals, fisher=True, bias=False))
    except Exception:
        return 7.0
    if ex <= 0.05:
        return 30.0
    return float(np.clip(4.0 + 6.0 / ex, 4.0, 30.0))


def _fit_lead(df: pd.DataFrame, fcst_col: str, target_col: str,
              min_rows: int = MIN_ROWS) -> dict | None:
    """Fit one (city, lead, input) calibration with a temporal holdout."""
    sub = df.dropna(subset=[fcst_col, target_col]).sort_values("date_local")
    if len(sub) < min_rows:
        return None
    f = sub[fcst_col].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    doy = pd.to_datetime(sub["date_local"]).dt.dayofyear.to_numpy()

    cut = int(len(sub) * (1.0 - HOLDOUT_FRAC))
    X = _design(f, doy)
    beta, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
    a, b, c_sin, c_cos = (float(v) for v in beta)

    resid_cal = y[cut:] - X[cut:] @ beta
    resid_raw = y[cut:] - f[cut:]
    rmse_cal = float(np.sqrt(np.mean(resid_cal ** 2)))
    rmse_raw = float(np.sqrt(np.mean(resid_raw ** 2)))
    use_cal = rmse_cal < rmse_raw
    resid = resid_cal if use_cal else resid_raw
    return {
        "a": a, "b": b, "c_sin": c_sin, "c_cos": c_cos,
        "use_calibrated_mean": bool(use_cal),
        "sigma": float(np.std(resid)),
        "nu": _fit_nu(resid),
        "n_train": int(cut), "n_holdout": int(len(sub) - cut),
        "holdout_rmse_calibrated": rmse_cal, "holdout_rmse_raw": rmse_raw,
    }


def _train_city_target(city: str, slug: str, spec: dict) -> None:
    target = spec["target_col"]
    try:
        from settlement_truth import load_training_truth
        leads_df = pd.read_csv(spec["bm_file"].format(slug=slug))
        truth = load_training_truth(slug)   # settlement-faithful target (W0.2); CLI fallback
    except Exception as e:
        logger.error(f"{city} [{target}]: missing archive data ({e})")
        return

    for d in (leads_df, truth):
        d["date_local"] = pd.to_datetime(d["date_local"]).dt.strftime("%Y-%m-%d")
    df = leads_df.merge(truth[["date_local", target]].dropna(), on="date_local", how="inner")

    mm_models = MM_MODELS_BY_CITY.get(slug, MM_MODELS_DEFAULT)
    mm_path = spec["mm_file"].format(slug=slug)
    if os.path.exists(mm_path):
        mm = pd.read_csv(mm_path)
        mm["date_local"] = pd.to_datetime(mm["date_local"]).dt.strftime("%Y-%m-%d")
        if spec["merge_jma_cand"]:
            jma_path = f"data/weather/{slug}_historical_leads_jma.csv"
            if "jma" in mm_models and os.path.exists(jma_path):
                jma = pd.read_csv(jma_path)
                jma["date_local"] = pd.to_datetime(jma["date_local"]).dt.strftime("%Y-%m-%d")
                jma = jma.rename(columns={f"jma{n}": f"fcst_tmax_lead{n}_jma" for n in LEADS})
                mm = mm.merge(jma, on="date_local", how="left")
            cand_path = f"data/weather/{slug}_historical_leads_cand.csv"
            if os.path.exists(cand_path):
                cand = pd.read_csv(cand_path)
                cand["date_local"] = pd.to_datetime(cand["date_local"]).dt.strftime("%Y-%m-%d")
                mm = mm.merge(cand, on="date_local", how="left")
        # NBM station guidance (US cities, Tmax only): per (date, lead) take the LATEST
        # archived run, official txn max guidance preferred — mirrors the serving as-of join.
        nbm_path = f"data/weather/{slug}_nbm.csv"
        if spec["use_nbm"] and os.path.exists(nbm_path):
            nbm = pd.read_csv(nbm_path)
            nbm["nbm_tmax"] = nbm["nbm_tmax_txn_c"].fillna(nbm["nbm_tmax_tmp_c"])
            nbm = (nbm.dropna(subset=["nbm_tmax"])
                      .sort_values("runtime_utc")
                      .groupby(["date_local", "lead"], as_index=False).last())
            wide = nbm.pivot(index="date_local", columns="lead", values="nbm_tmax")
            wide = wide.rename(columns={n: f"fcst_tmax_lead{n}_nbm" for n in wide.columns})
            mm = mm.merge(wide.reset_index(), on="date_local", how="left")
        for n in LEADS:
            cols = [spec["model_col"].format(n=n, m=m) for m in mm_models
                    if spec["model_col"].format(n=n, m=m) in mm.columns]
            # Require the FULL model set — a partial mean is a different estimator
            # than what serving computes.
            if len(cols) == len(mm_models):
                vals = mm[cols]
                mm[f"mm_mean_lead{n}"] = vals.mean(axis=1).where(vals.notna().all(axis=1))
            pcols = [spec["model_col"].format(n=n, m=m) for m in MM_PROXY_MODELS
                     if spec["model_col"].format(n=n, m=m) in mm.columns]
            if len(pcols) == len(MM_PROXY_MODELS):
                pvals = mm[pcols]
                mm[f"mm_proxy_lead{n}"] = pvals.mean(axis=1).where(pvals.notna().all(axis=1))
        df = df.merge(mm, on="date_local", how="left")

    per_lead = {}
    for n in LEADS:
        entry = {}
        fit_bm = _fit_lead(df, spec["bm_col"].format(n=n), target)
        if fit_bm:
            entry["best_match"] = fit_bm
        if f"mm_mean_lead{n}" in df.columns:
            fit_mm = _fit_lead(df, f"mm_mean_lead{n}", target, min_rows=MIN_ROWS_MM)
            if fit_mm:
                entry["mm_mean"] = fit_mm
        if f"mm_proxy_lead{n}" in df.columns:
            fit_px = _fit_lead(df, f"mm_proxy_lead{n}", target, min_rows=MIN_ROWS_MM)
            if fit_px:
                entry["mm_proxy"] = fit_px
        if f"fcst_tmax_lead{n}_nbm" in df.columns and spec["use_nbm"]:
            fit_nbm = _fit_lead(df, f"fcst_tmax_lead{n}_nbm", target, min_rows=MIN_ROWS_MM)
            if fit_nbm:
                entry["nbm"] = fit_nbm
        if entry:
            per_lead[str(n)] = entry

    if not per_lead:
        logger.warning(f"{city} [{target}]: no leads fit — skipping")
        return

    # Serving-input selection on per-lead common windows (paired dates) — the inputs'
    # archives start at different times, so full-window scores would mix seasons.
    input_cols = {"best_match": spec["bm_col"],
                  "mm_mean": "mm_mean_lead{n}"}
    if spec["use_nbm"]:
        input_cols["nbm"] = "fcst_tmax_lead{n}_nbm"
    input_cols = {k: t for k, t in input_cols.items()
                  if all(t.format(n=n) in df.columns for n in (1, 2, 3))}

    scores = {k: [] for k in input_cols}
    for n in (1, 2, 3):
        cols = [t.format(n=n) for t in input_cols.values()]
        common = df.dropna(subset=cols + [target])
        for k, t in input_cols.items():
            fit = _fit_lead(common, t.format(n=n), target, min_rows=200)
            scores[k].append(np.inf if fit is None else
                             min(fit["holdout_rmse_calibrated"], fit["holdout_rmse_raw"]))
    scores = {k: float(np.mean(v)) for k, v in scores.items()}
    serving_input = min(scores, key=scores.get)

    params = {
        "version": 2,
        # Stamped so staleness is READABLE from the artifact. The models sat frozen at the
        # 2026-07-25 build for 11 days and nothing reported it, because a retrain that dies
        # and a retrain that runs and changes nothing both produce no commit.
        "trained_at": datetime.now(timezone.utc).isoformat(),

        "target": target,
        "input": serving_input,
        "input_scores": {k: (None if np.isinf(v) else v) for k, v in scores.items()},
        "mm_models": list(mm_models),
        "leads": per_lead,
    }
    out = os.path.join(MODELS_DIR, spec["out_file"].format(slug=slug))
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2)

    l1 = per_lead.get("1", {}).get(serving_input) or next(iter(per_lead["1"].values()))
    l3 = per_lead.get("3", {}).get(serving_input, {})
    logger.info(
        f"{city} [{target}]: input={serving_input} (scores {scores}) "
        f"lead1 sigma {l1['sigma']:.2f} rmse raw {l1['holdout_rmse_raw']:.2f}"
        f"→cal {l1['holdout_rmse_calibrated']:.2f} | "
        f"lead3 sigma {l3.get('sigma', float('nan')):.2f} → {out}")


def train_calibrator():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for city in CITIES.keys():
        slug = city.replace(" ", "_").lower()
        for spec in TARGET_SPECS.values():
            _train_city_target(city, slug, spec)


if __name__ == "__main__":
    train_calibrator()
