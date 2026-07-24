# Design — Serve the empirical QRF distribution (fix the moment-match loss)

**Date:** 2026-07-24
**Status:** Approved design; ready for implementation plan
**Component:** `predictors/base.py`, `pmf.py`, `predictors/qrf.py`, `train_qrf.py`
**Author:** Claude (Opus 4.8) with Ronan

---

## 0. Why (the diagnostic, pinned)

The QRF forecaster's first live retrain **failed M1** — it did not beat the raw ensemble on
settled-market Brier:

```
QRF Brier 0.1713  >  ENSEMBLE 0.1685   (same 104 markets)  → FAIL
PAIRED (QRF) CRPS  1.2491 vs ENSEMBLE 1.2264 (n=72 → ensemble better)
```

A read-only diagnostic (same holdout, one common sample-CRPS estimator) found the root cause:
**moment-matching the QRF quantiles into a single Student-t is lossy in all five cities**, and the
*raw* QRF distribution beats the (proxy) ensemble everywhere:

| city | RAW quantiles | COLLAPSE (served) | ens proxy |
|---|--:|--:|--:|
| Seoul | 0.675 | 0.732 | 0.947 |
| NYC | 0.942 | 0.974 | 1.134 |
| Hong Kong | 0.532 | 0.553 | 1.165 |
| London | 0.586 | 0.650 | 0.657 |
| Chicago | 0.887 | 0.913 | 1.007 |

The Student-t collapse (chosen originally for "zero engine changes") throws away 3–11% of the
forecast quality — the exact tail/shape flexibility that was the point of going nonparametric. It
also corrupts the self-gate: `fit_city` scores the *collapsed* distribution, which is why London
and Chicago were wrongly marked `beats_ensemble=False`.

**Honest bound:** the diagnostic is CRPS vs the *proxy* ensemble. The real M1 bar is market-Brier
vs the *real* ensemble, where collapsed-QRF lost by ~1.7%. The measured collapse loss (~3–11%
CRPS) is in the ballpark to flip M1 but does **not** prove it — re-running M1 on the empirical
serving is the test, not this design.

## 1. The fix — an empirical CDF threaded through the existing PMF layer

Everything downstream of a forecast (`pmf._bin_prob`, `_condition_prob`, `reconstruct_pmf`, CRPS)
consumes only a **CDF** `F(x)` (`pmf._cdf`). The Student-t is one way to produce it. QRF will
supply its own empirical CDF; the same tested bin-math and censoring run unchanged.

### 1.1 One optional field on the distribution (`predictors/base.py`)
Add `cdf: Callable[[float], float] | None = None` to `TemperatureDistribution`. When set, it is the
predictive CDF of the *uncensored* variable Z (floor/ceiling censoring is applied on top by
`_cdf`, exactly as for the Student-t). `mu/sigma/nu` remain as **summary statistics** (logging,
α-signals) but no longer drive the bins when `cdf` is present.

### 1.2 One branch in `pmf._cdf` (`pmf.py`)
`_cdf(x, mu, sigma, nu, floor=None, ceiling=None, cdf=None)`:
- keep the identical `floor`/`ceiling` short-circuits (return 0.0 below floor, 1.0 at/above
  ceiling) — unchanged;
- if `cdf is not None`: return `min(1.0, max(0.0, cdf(x)))` instead of the `student_t.cdf(...)`
  line;
- else: the existing Student-t line, byte-for-byte.

`_bin_prob`, `_condition_prob`, `reconstruct_pmf` gain a pass-through `cdf=None` kwarg and forward
it to `_cdf`. **EMOS/ensemble callers pass nothing → identical behavior** (regression-tested).

### 1.3 The empirical CDF (semi-parametric: empirical body, Gaussian tails) (`predictors/qrf.py`)
`build_empirical_cdf(qf, x_row) -> Callable`:
- predict a **fine quantile grid** `Q_FINE = [0.01, 0.02, …, 0.99]` (99 knots) from the fitted
  forest for the feature row. `predict_quantiles` builds the leaf-weighted empirical distribution
  once per row, so 99 levels costs ≈ the same as 7 (the cost is the leaf-weighting, not the knot
  count).
- **body** (`q[0.01] ≤ x ≤ q[0.99]`): monotone linear interpolation of `(quantile_values →
  quantile_levels)`. QRF quantiles are already sorted (the weighted-quantile construction), so the
  interpolant is monotone.
- **tails**: fit a Gaussian to the two outer knots on each side (left: solve μ,σ so the Gaussian
  CDF hits `(q0.01, 0.01)` and `(q0.05, 0.05)`; right: `(q0.95, 0.95)`, `(q0.99, 0.99)`), and use
  that Gaussian CDF beyond the outer knot — continuous at the knot, finite and monotone in the far
  tail. This is the standard body-empirical/parametric-tail construction and keeps far-out bins
  sane where a forest cannot extrapolate.
- **degenerate guard**: if the knots are (near-)constant (zero spread), fall back to a tiny-σ step
  at the median so `_cdf` stays well-defined.

`QRFPredictor.predict_distribution` (gated ON) sets `cdf = build_empirical_cdf(...)`, keeps
`mu/sigma/nu` from `moment_match` as summary, sets `floor` for same-day Tmax as today, and returns
`TemperatureDistribution(mu, sigma, nu, "qrf", floor=floor, cdf=cdf)`. Gated-OFF / missing artifact
/ `days_ahead > max_lead` still returns `None` (unchanged self-gate & lead bound).

### 1.4 Re-gate `fit_city` on the empirical (`train_qrf.py`)
`fit_city`'s holdout score currently uses `crps_gaussian_proxy` on the *collapsed* `(mu, sigma)`.
Change it to score the **empirical-CDF CRPS** on the holdout rows (a sample-based CRPS over the 99
predicted quantiles per row), so `beats_ensemble` judges what is actually served. The ensemble
baseline stays `crps_gaussian_proxy` on the mm_mean proxy (`_ensemble_holdout_crps`, unchanged) —
still an honest CRPS-scale comparison; the *real* ensemble check remains `evaluate_oos`'s M1 gate.
This is expected to flip London/Chicago (their raw beats the proxy) — a live-data outcome, not a
tested assertion.

## 2. Files touched
- `predictors/base.py` — `+ cdf` field on `TemperatureDistribution`.
- `pmf.py` — `+ cdf` kwarg on `_cdf`/`_bin_prob`/`_condition_prob`/`reconstruct_pmf` (pass-through;
  one branch in `_cdf`).
- `predictors/qrf.py` — `build_empirical_cdf`; set `cdf` in `predict_distribution`.
- `predictors/qrf_core.py` — a small sample-CRPS helper if not already present (reuse if it is).
- `train_qrf.py` — `fit_city` scores the empirical-CDF CRPS.
- `moment_match` is **retained** (summary stats), not deleted.

## 3. Testing
- **Regression (the load-bearing safety test):** `_cdf`/`reconstruct_pmf` with `cdf=None` produce
  byte-identical results to before for a Student-t distribution — EMOS/ensemble are untouched.
- **`_cdf` with a `cdf_fn`:** returns the fn's value clamped to [0,1] in-range; still returns 0
  below `floor` and 1 at/above `ceiling` (censoring wraps the empirical CDF).
- **`build_empirical_cdf`:** monotone non-decreasing; in [0,1]; continuous at the tail knots;
  finite far into both tails; the degenerate (constant-quantile) case is well-defined.
- **Bin PMF via empirical:** a `TemperatureDistribution` with `cdf` set yields bin probabilities
  that sum to ≈1 over a full bin range and respect the same-day floor (bins below the running max
  ≈0).
- **`QRFPredictor`:** gated-ON returns a distribution whose `cdf` is set and whose bins price off
  the empirical CDF; gated-OFF / `days_ahead > max_lead` still returns `None`.
- **`fit_city` empirical gate:** the holdout score is computed via the empirical-CDF CRPS path (not
  `crps_gaussian_proxy` on the collapse); on a fixture where the raw distribution is sharper than
  the collapse, the empirical score is ≤ the collapsed score.

Run: `pytest -o addopts="" tests/ -v` from repo root.

## 4. Success criteria
- With `cdf=None`, every existing predictor and every existing test behaves identically (additive).
- QRF serves an empirical CDF; bins price off it; floor/leak/gate invariants preserved.
- A re-triggered retrain prints an M1 gate for the **empirical** QRF vs the ensemble; London/Chicago
  gate on their raw quality. Whether M1 flips is the empirical question the change exists to answer
  — reported honestly either way. Nothing trades real money regardless (QRF stays paper /
  self-gated / forward-gated).

## 5. Out of scope
- Tmin; the `min_lead` lower-bound guard (tracked separately); GBM/NN learners; M2 intraday
  training (a data build, gated on M1 actually passing here); any real-money change.
