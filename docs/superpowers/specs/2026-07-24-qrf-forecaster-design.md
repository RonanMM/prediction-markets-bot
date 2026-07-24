# Design — QRF probabilistic forecaster (beat the ensemble, aim at the market)

**Date:** 2026-07-24
**Status:** Approved design; ready for implementation plan
**Component:** new `predictors/qrf.py`, `qrf_features.py`, `train_qrf.py` (+ eval/dashboard hooks)
**Author:** Claude (Opus 4.8) with Ronan

---

## 0. The honest frame (pinned — read first)

The calibrated EMOS model currently **loses to the raw ensemble it is built on** (settlement-truth
Brier 0.150 vs 0.142 on the common set). Diagnosis (2026-07-24): both forecasters are
**overconfident / underdispersed**, and EMOS makes it *worse* by sharpening further — it hurts most
at leads 0–1 (where the bets are) and only helps at lead 2. Reliability: the model says "27%" on
bins that happen 12% of the time, and "86%" on bins that happen 25% of the time. This is a
**dispersion/calibration** failure, not a mean/bias one — the same reason the tails fail (Seoul
27 °C priced 8%, happened).

Two milestones, with honest confidence:
- **M1 — beat the ensemble.** High confidence. A nonparametric learner that is *self-gated* against
  the ensemble on holdout mathematically cannot be worse than it where active.
- **M2 — beat the market (0.117).** Uncertain, and we will not pretend otherwise. Calibration alone
  cannot beat the market — the ensemble members already encode the NWP the market's bots price.
  Beating the market needs **information the market underweights**; our one real candidate is the
  **intraday running max/min** on same-day markets (the locked-in state the crowd is slow to
  reprice). The eval measures M2; **nothing trades real money until a pre-registered forward gate
  passes.** If M2 fails, M1 is still a real improvement and the answer on market efficiency is
  itself valuable.

Literature grounding: underdispersion is the universal disease of raw ensembles
([Mlakar 2024](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4809)); Quantile Regression
Forests beat parametric EMOS with no distributional assumption
([Taillardat 2016](https://journals.ametsoc.org/mwr/article/144/6/2375)); NN distributional
regression is stronger but data-hungry ([Rasp & Lerch 2018](https://arxiv.org/abs/1805.09091)) —
hence QRF first for our 5-city / limited-history regime.

## 1. Scope (v1)

- **Tmax only, all 5 cities, leads pooled** (lead is a feature, not separate models — more data).
- **QRF is the only learner in v1**; GBM and NN are *challengers* added later through the same
  harness (§3.2), never all-three-picked-on-one-small-holdout (that is a selection-bias trap).
- Runs **alongside** EMOS as a new predictor; **never replaces** it until it wins the holdout AND a
  forward gate. Tmin mirrors this as a fast follow-up, out of scope for v1.

## 2. Architecture

```
existing fetchers ─┐
  multi-model members (per MM_MODELS_BY_CITY)  ┐
  ensemble mean/std/quantiles                  ├─▶ qrf_features.build_features() ─▶ X  (+ y = settlement truth)
  intraday running max (station obs, same-day) │        as-of the snapshot time (NO look-ahead)
  lead, day-of-year, diurnal range             ┘
                                                        │
                              train_qrf.py (per city) ──┤ temporal holdout, self-gate vs ensemble
                                                        ▼
                                 models/{slug}_qrf.joblib  (fitted forest + training X,y)
                                                        ▼
                     predictors/qrf.QRFPredictor.predict_distribution(...) ─▶ TemperatureDistribution
                        quantiles → monotone CDF → tail-extrapolate → PMF over bins → intraday floor
                                                        ▼
                        evaluate_oos:  QRF vs EMOS vs ensemble vs MARKET  (Brier/CRPS/reliability)
                        M1 gate: QRF ≤ ensemble · M2: vs market · forward-gated before real money
```

Five independently testable components (§3).

## 3. Components

### 3.1 Feature builder — `qrf_features.py`
`build_features(rows) -> (X: DataFrame, meta)`. One feature vector per (city, target_date, lead,
snapshot). Columns:
- **per-model deterministic Tmax** for the target day at this lead, for the city's model set
  (`train_calibrator.MM_MODELS_BY_CITY`): e.g. ecmwf, gfs, icon, aifs, gem, mf, jma;
- **mm_mean, mm_std** (mean & spread of that model set);
- **ensemble mean, ensemble std, ensemble p10/p50/p90** (from the 122-member ensemble);
- **lead** (days_ahead); **day-of-year** as sin/cos; **diurnal range** (α2 proxy);
- **intraday**: running observed max as-of the snapshot (same-day only; a sentinel + an
  `is_same_day` flag pre-day) and snapshot hour-of-day.

**Leakage is the cardinal rule (§4).** Every feature is *as-of the snapshot time*. Training reuses
the archived per-lead multi-model forecasts (`fetch_historical_leads_mm`, previous-runs, already
as-of) joined to settlement truth; the intraday running-max comes from station obs truncated at the
snapshot timestamp, never end-of-day.

### 3.2 Learner interface + QRF — `predictors/qrf.py`
A minimal interface so GBM/NN drop in later:
```python
class Learner(Protocol):
    def fit(self, X, y) -> None: ...
    def predict_quantiles(self, X, q: list[float]) -> np.ndarray: ...   # shape (n, len(q))
```
**QRF impl uses sklearn `RandomForestRegressor` — no new dependency.** Fit a standard RF; at predict,
use leaf memberships (`rf.apply`) to gather the empirical distribution of training `y` in the shared
leaves and read weighted quantiles (Meinshausen 2006 QRF). `min_samples_leaf ≈ 30` is the main
sharpness/overfit lever (large leaf → wider, smoother quantiles). Artifact
`models/{slug}_qrf.joblib` = the fitted forest + training `(X, y)` for the quantile lookup.

### 3.3 Trainer — `train_qrf.py`
Mirrors `train_calibrator.py`: per city, load the archived per-lead multi-model forecasts + intraday
obs, `build_features`, `fit` the QRF, evaluate on a **temporal holdout** (train early / test late),
and **self-gate**: record whether QRF beats the raw ensemble on that city's holdout. Persist the
artifact + a small JSON sidecar (`{slug}_qrf_meta.json`) with the holdout Brier/CRPS and the
`beats_ensemble` flag. Runs inside `retrain.yml` (manual, like EMOS).

### 3.4 Server / PMF — `predictors/qrf.QRFPredictor(BasePredictor)`
`predict_distribution(...) -> TemperatureDistribution` (identical contract to `EMOSPredictor`):
predict a quantile ladder (5,10,25,50,75,90,95) → interpolate a **monotone** predictive CDF →
**tail-extrapolate** beyond the outer quantiles with a real tail model (fit the outer quantiles to a
Gaussian/Student-t tail; never a flat cutoff — the tails are where we fail) → `P(bin)=CDF(hi)−CDF(lo)`
over the market's bins → **floor at the running observed max** for same-day (reuse
`TemperatureDistribution.floor` / existing censoring). If a city's `beats_ensemble` flag is false (or
no artifact), `QRFPredictor` returns `None` and the engine falls back to EMOS/ensemble — the
self-gate that guarantees M1 city-by-city.

### 3.5 Eval & dashboard hooks
Add a `--predictor qrf` path so `polymarket_weather_analysis.py` regenerates a QRF eval tracker
(`opportunities_evaluation_qrf.csv`), and extend `evaluate_oos.py` to print **QRF vs EMOS vs ensemble
vs market** Brier/CRPS + reliability + the **M1 gate line** (QRF holdout Brier ≤ ensemble, per city
and aggregate). The dashboard scoreboard gains QRF as a fourth forecaster once trained (additive,
behind the existing completeness guard).

## 4. Leakage discipline (the make-or-break correctness property)
A single look-ahead feature manufactures a fake edge — the exact "127% ROI" mirage this project
exists to avoid. Non-negotiable:
- Training features come only from **previous-runs archives** (already as-of the forecast issue
  time) and **station obs truncated at the snapshot timestamp**.
- The intraday running-max is computed strictly `< snapshot_time`.
- The temporal holdout splits by date; no target-day information leaks into features.
- A dedicated test (§5) asserts a synthetic post-snapshot observation does NOT change the features.

## 5. Testing
- **Leakage (paramount):** feed a feature row, then a copy with a fabricated *post-snapshot* obs;
  `build_features` output is identical. Holdout split leaks no target-day data.
- **QRF correctness:** predicted quantiles are monotone; on synthetic data drawn from a known
  conditional distribution, recovered quantiles track the truth; holdout CRPS ≤ raw-ensemble CRPS on
  a seeded sample.
- **PMF reconstruction:** bin probabilities sum to ≈1; CDF monotone; tail-extrapolation finite and
  sane beyond the outer quantiles.
- **Intraday floor:** same-day bins below the running observed max receive ≈0 probability.
- **Self-gate:** a city whose QRF loses the holdout → `QRFPredictor` returns `None` → engine falls
  back (no worse than ensemble).
- **Eval integration:** the QRF tracker regenerates and `evaluate_oos` prints the M1 gate line.

Run: `pytest -o addopts="" tests/ -v` from repo root.

## 6. Out of scope (v1)
Tmin (fast follow-up), GBM/NN learners (later challengers via the same interface), any real-money
trading (forward-gated separately), replacing EMOS (QRF only supersedes it per-city after it wins
holdout + forward gate).

## 7. Success criteria
- `train_qrf.py` trains per-city QRF artifacts with holdout Brier/CRPS + `beats_ensemble` flags.
- `evaluate_oos` reports QRF vs EMOS vs ensemble vs market, with an explicit **M1 gate** result.
- Where QRF's holdout beats the ensemble, the served QRF distribution is used; elsewhere it
  self-gates to fallback — so the forecaster is **never worse than the ensemble** in aggregate.
- The leakage test passes (no look-ahead). All existing tests still pass.
- No real-money change: QRF bets are paper until a pre-registered forward gate passes.
