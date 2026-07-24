"""train_qrf.py — per-city Quantile-Regression-Forest training + temporal holdout self-gate.
Mirrors train_calibrator.py's data loading (see discovery step). QRF is served only where it beats
the raw ensemble on the city's holdout (meta['beats_ensemble'])."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib

from predictors.qrf_core import QuantileForest, moment_match
from qrf_features import FEATURE_COLS

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_Q = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]


def _crps_gaussian_proxy(y, mu, sigma):     # cheap continuous score for the gate
    z = (y - mu) / np.maximum(sigma, 1e-3)
    from scipy.stats import norm
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def fit_city(slug, X, y, ens_holdout_brier):
    X = X[FEATURE_COLS].to_numpy(float); y = np.asarray(y, float)
    n = len(y); cut = int(n * 0.75)                     # temporal holdout (rows are date-ordered)
    qf = QuantileForest().fit(X[:cut], y[:cut])
    q = qf.predict_quantiles(X[cut:], _Q)
    mm = np.array([moment_match(_Q, q[i]) for i in range(len(q))])   # (mu,sigma,nu) per row
    holdout = _crps_gaussian_proxy(y[cut:], mm[:, 0], mm[:, 1])
    beats = holdout <= ens_holdout_brier
    qf_full = QuantileForest().fit(X, y)                # refit on all data for serving
    joblib.dump(qf_full, _MODELS_DIR / f"{slug}_qrf.joblib")
    meta = {"beats_ensemble": bool(beats), "holdout_brier": holdout, "n": int(n)}
    (_MODELS_DIR / f"{slug}_qrf_meta.json").write_text(json.dumps(meta))
    return meta


def train_qrf(cities=None):
    # mirror train_calibrator.py: for each city, load leads_mm(+cand,jma)+truth+obs, build X/y,
    # compute the ensemble's holdout CRPS proxy the same way, then fit_city(...). (See discovery.)
    raise NotImplementedError("wire the train_calibrator loader here (discovery step)")


if __name__ == "__main__":
    train_qrf()
