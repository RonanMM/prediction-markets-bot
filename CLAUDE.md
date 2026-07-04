# CLAUDE.md

This file provides guidance to coding assistants (like Claude/Antigravity) working on the Raincheck weather prediction tracker repository.

## Project Overview

Raincheck is a Polymarket weather prediction market tracker that compares market-implied temperatures against meteorological forecasts to identify pricing inefficiencies. 

The active project code lives in `src/polymarket_weather/`, not in `src/raincheck/` (which is a near-empty PyScaffold skeleton).

---

## Commands

Most commands run from the `src/polymarket_weather/` directory. **Exceptions:** the two
optimizer scripts (`optimizer.py`, `optimizer_full.py`) live at the **repo root** and must be
run from there — they do `sys.path.insert(0, "src/polymarket_weather")`, a path relative to the
cwd — and the unit tests also run from the repo root (see below).

```bash
# Install dependencies
pip install -r requirements.txt

# Full Pipeline: fetch Polymarket + weather, print summary, generate plots
python main.py

# Skip fetches: re-generate plots from stored CSVs only
python main.py --plots-only

# Print market vs forecast summary without fetching or plotting
python main.py --summary-only

# Target specific cities
python main.py --cities London Seoul "New York City"

# Run the Inefficiency Analyzer (Live Dry Run / Real-Time Verification)
python polymarket_weather_analysis.py --data_dir ./data --live

# Run the Inefficiency Analyzer (Backtest Mode)
python polymarket_weather_analysis.py --data_dir ./data

# Run the Parameter Sweep Optimizer (reads pre-calculated CSV, very fast)
# NOTE: run from the REPO ROOT, not src/polymarket_weather/ (see above)
python optimizer.py

# Run the Full-Pipeline Grid Search Optimizer (runs engine back-to-front, slow)
# NOTE: run from the REPO ROOT, not src/polymarket_weather/ (see above)
python optimizer_full.py
```

To run unit tests (from the repository root):
```bash
# Run pytest overriding inifile addopts coverage checks
pytest -o addopts="" tests/ -v
```

---

## Architecture & Data Flow

### Data Flow Diagram
```
Gamma API (Polymarket) → fetch_polymarket.py ──┐
                                               ├──→ data/polymarket/*.csv
CLOB API (Price History) → fetch_polymarket.py ┘
                                               
Open-Meteo Forecast API → fetch_weather.py ────┐
                                               ├──→ data/weather/*.csv
Open-Meteo Ensemble API → fetch_ensemble.py ───┘
                                                     ↓
                                      processing.py (deduplication & normalisation)
                                                     ↓
              [EMOSPredictor / EnsemblePredictor / NWPFallbackPredictor]
                                                     ↓
                              engine.py (analyses city markets & computes Kelly sizing)
                                                     ↓
               polymarket_weather_analysis.py (live analysis & terminal reports)
                                                     ↓
                                     visualization.py (PNG output plots)
```

### Key Design Decisions
- **Append-only CSVs**: Data is never overwritten. `processing.py` deduplicates on read using composite keys (e.g. `condition_id + fetched_at_utc` for market snapshots).
- **Dual timezone storage**: Records store both `_local` and `_utc` timestamps; plots use UTC.
- **Stateless fetchers**: `fetch_polymarket.py` and `fetch_weather.py` have no side effects beyond returning raw dicts; `processing.py` handles persistence.
- **Bypassed time expirations in backtesting**: The engine allows evaluating expired historical bets for simulations by checking date logic instead of calling `datetime.now()`.
- **City Name Normalization**: The bot utilizes a mapping dictionary `CITY_NAMES` in `config.py` to reconcile differences between Polymarket city tags (`"NYC"`) and weather station labels (`"new_york_city"`), preventing grading mismatches.

### Resolution Anchors — the single most important model (do not break)
**A market resolves on a NAMED STATION's daily reading, not a coordinate.**
`src/polymarket_weather/resolution_anchors.py` is the **single source of truth** and deliberately
separates THREE anchors per city. `config.CITIES` and every aux script DERIVE their coordinates
from it — there are no hardcoded coords anywhere.
- **Resolution anchor** — `resolution_url` + `resolution_unit` (what the UMA oracle reads).
- **Truth anchor** — the resolution-faithful observation feed `fetch_historical_truth.py` reads
  (NWS CLI via IEM for KLGA/KORD, IEM METAR daily for EGLC/RKSI, the HKO open-data API for HKO).
  Grading (`grading.py`) reads the resulting `{slug}_historical_actuals.csv`. **Meteostat is
  legacy/reference only** — a 2026-07-03 audit found it corrupted for recent weeks (KLGA June
  daily maxes off by up to ~9 °C vs the NWS CLI report) and its HK station 45007 off HKO by
  >1.5 °C on ~66 days/yr. Truth lag is now ~1 day (HKO ~1 month).
- **Forecast anchor** — `forecast_lat`/`forecast_lon` (where Open-Meteo is pointed). Usually the
  station's location, but **not always**.

⚠️ **Seoul is the key exception. Do NOT "correct" its forecast coords back to the airport.** The
Incheon-airport ERA5 grid cell is coastal/sea-damped and predicts the station poorly, so the
forecast anchor is a skill-optimized inland **"Bucheon corridor" point (37.5035, 126.766)**, while
truth/resolution stay **Incheon RKSI** (truth now read from IEM METAR daily for RKSI). Validated:
CV RMSE ≈1.2 (Bucheon) vs ≈1.96 (airport cell). A loud comment in the file says the same.

---

## Module Responsibilities

| File | Responsibility |
|---|---|
| `resolution_anchors.py` | **Single source of truth** for per-city resolution / truth / forecast anchors (see above). |
| `config.py` | `CITIES` (coords DERIVED from `resolution_anchors.py`), endpoints, fees, and optimal sizing limits. |
| `grading.py` | Grades markets against the resolution-**station** observation (station truth), not a forecast grid cell. |
| `data_status.py` | Track-record progress report: collected / resolved / station-gradable counts vs the pre-committed gate. |
| `fetch_polymarket.py` | Gamma API (market search/details) + CLOB API (price history). Parses temperature bins from market questions. |
| `fetch_weather.py` | Open-Meteo 16-day forecast (daily + hourly) **plus `fetch_forecast_multimodel`** — per-model deterministic Tmax (ECMWF/GFS/ICON/JMA → `{slug}_daily_mm.csv`), the exact serving input for the calibrated multi-model mean. |
| `fetch_ensemble.py` | Fetches Open-Meteo ensemble forecasts (ICON+GFS+ECMWF, 122 members) to compute dynamic uncertainty metrics. |
| `fetch_historical_truth.py` | **Resolution-faithful station truth** (NWS CLI / IEM METAR / HKO API, 2015→now, ~1-day lag). Replaced Meteostat (corrupted; see Resolution Anchors). |
| `fetch_historical_leads.py` | Archived real forecasts at leads 1–7 (Open-Meteo Previous Runs, `best_match`, 2022→now) — EMOS v2 training data. |
| `fetch_historical_leads_mm.py` | Same, per model chain (ECMWF/GFS/ICON) for the multi-model mean input. Seoul additionally uses a JMA file (`{slug}_historical_leads_jma.csv`). |
| `fetch_historical_leads_cand.py` | Blend-expansion candidates (AIFS/GEM/MF/CMA/BOM) per lead — feeds the per-city model-set selection in `train_calibrator.py` (see `MM_MODELS_BY_CITY`). |
| `fetch_nbm.py` | NBM (NWS National Blend) station guidance for KLGA/KORD via the IEM NBS archive — runtime-stamped (`avail_utc`) so backtests as-of join only runs that actually existed. **Tested and NOT selected**: our multi-model blend beats raw NBM by 0.4–0.75 °C at leads 1–3; NBM stays a self-gating input candidate (scores recorded in the params JSON). `--recent` top-up runs inside main.py. |
| `fetch_historical_leads_min.py` | Per-lead archived forecasts of the daily **MIN** (best_match leads 1–7 + the 7 blend models leads 1–4) — training data for the Tmin model. |
| `train_intraday.py` | Per-city, per-local-hour fits of `Tmax \| (fcst, running max)` AND `Tmin \| (fcst, running min)` → `models/{slug}_intraday[_min].json`, self-gated per hour. |
| `fetch_station_obs.py` | Hourly METAR observations per resolution station (IEM, 2022→now; `--recent` top-up runs inside main.py). Feeds intraday conditioning. No HKO feed → Hong Kong un-conditioned. |
| `processing.py` | CSV append, normalisation, and deduplication logic. |
| `engine.py` | Runs the core analysis loop, applying conflict gating, group caps, and portfolio exposure limits via `WeatherBettingBot`. |
| `signals.py` | Implementation of alpha signals (momentum, convergence, staleness, volume ratios). |
| `models.py` | Dataclasses for `MarketBin` and `Opportunity`. |
| `pmf.py` | Parsing rules for questions (exact, gte, lte) and reconstruction of the probability mass function (PMF). |
| `reports.py` | Terminal reports, summary generators, and plotting wrappers. |
| `visualization.py` | Generates diagnostic PNG plots (forecast drift, efficiency signals, price momentum). |
| `predictors/emos.py` | **Default calibrator (v2, per-lead).** EMOS / Nonhomogeneous Regression trained on real archived forecasts per lead time (params in `models/{slug}_emos.json`, fit by `train_calibrator.py`). Mean self-gates per lead; the per-lead sigma floor is never gated (it is the overconfidence fix). |

---

## Predictor Models

Predictors live under `predictors/`. The engine's **default calibrator is `EMOSPredictor` v2**.
(An earlier RandomForest calibrator was verified net-negative and removed; EMOS **v1** — trained on
the ERA5 *reanalysis* archive — was replaced 2026-07 after an audit showed it understated live
forecast error 2–3× at betting leads, producing the overconfident tails that lost on Brier.)
1. **`EMOSPredictor` v2** *(default)*: per-city, **per-lead (1–7)** Nonhomogeneous Regression
   trained on REAL archived forecasts (Open-Meteo Previous Runs, 2022→now) against
   resolution-faithful station truth — so train and serve share both the model family and the
   lead time. Per lead: `mu = a + b·input + seasonal` (self-gated per lead by temporal holdout),
   `sigma = max(ens_std + diurnal_boost, sigma_lead)` where `sigma_lead` is the honest holdout
   residual std at that lead (never gated — it is the main fix), and Student-t `nu` from residual
   kurtosis. The mean input per city is chosen by holdout RMSE between `mm_mean` (multi-model
   deterministic mean — served exactly from `{slug}_daily_mm.csv`) and `best_match`
   (deterministic forecast). Per-city model sets come from the 2026-07-03 blend-expansion sweep
   (`train_calibrator.MM_MODELS_BY_CITY`): GEM broadly adopted, Météo-France for NYC/London,
   ECMWF-AIFS for Seoul/NYC/London (Seoul gains 5–8% at every lead), JMA Seoul-only; CMA/BOM
   rejected. When the exact multi-model inputs are unavailable (pre-daily_mm snapshots, model
   outage) serving falls back to the live ensemble mean WITH ITS OWN `mm_proxy` fit — never the
   full-blend sigma on the blunter proxy input. Live AIFS is `ecmwf_aifs025_single` (the
   previous-runs archive calls it `ecmwf_aifs025`). Input selection between `mm_mean`,
   `best_match` and `nbm` (US cities) is scored on PER-LEAD COMMON WINDOWS (paired dates), since
   the archives start at different times. NBM was tested and lost to the blend at both US
   stations (e.g. KORD lead-1 1.94 vs 1.19) — a useful negative: the US-city deficit is not
   mean-forecast quality. When the v2 distribution is active the engine does NOT average it
   with the raw ensemble (that would re-thin the tails); conflict gating remains.
2. **`EnsemblePredictor`**: Predicts probabilities using the ICON+GFS+ECMWF ensemble spread
   (standard deviation and Student-t tails). Also the fallback for EMOS, and the
   `--disable-calibrator` baseline.
3. **`NWPFallbackPredictor`**: Used when ensemble data is missing; falls back to static standard
   deviation tables defined in `config.py`.

### Intraday conditioning (same-day bets)
For a bet placed ON the target's station-local day, the predictor conditions on the running
observed daily max M (hourly METARs via `fetch_station_obs.py`): the predictive distribution is
**floored** at M (`TemperatureDistribution.floor` — Tmax cannot end below what was already
recorded; `pmf._cdf/_bin_prob/_condition_prob` and the CRPS all honor the censoring), and for
self-gated local hours a per-hour regression `Tmax = a_h + b_h·fcst + c_h·M_h` replaces mu/sigma
(`train_intraday.py`; by 14:00 local it roughly halves sigma, by 17:00 it's ~0.4 °C with c≈0.95).
This mostly matters LIVE (morning/midday, before the market converges); sparse historical
snapshots rarely exercise it, so its effect barely shows in the backtest tracker.

### Market-type coverage — Tmin ("lowest temperature") markets
~20% of markets settle on the daily MIN. They were briefly excluded (pricing them off the Tmax
distribution produced garbage: model 0.2% vs market 38%); since 2026-07-04 they are priced by a
dedicated **Tmin model** mirroring the Tmax stack: per-lead Tmin EMOS
(`fetch_historical_leads_min.py` → `models/{slug}_emos_min.json`, same per-city blends), live min
inputs (`temperature_2m_min` in the ensemble + daily_mm fetches), and intraday conditioning with a
**ceiling** at the running observed min (`models/{slug}_intraday_min.json`; the overnight low is
largely locked by mid-morning — London sigma@12 ≈ 0.7 °C). The engine routes `lowest` bins to the
min distribution (own PMF group, no raw-ensemble averaging or conflict gating — there is no
ensemble Tmin baseline); if a city has no trained Tmin params the predictor returns None and those
bins are skipped, never mispriced. `grading.py` grades them from `temp_min_c` as before.

Status (2026-07-03, 71 gradable markets — preview, gate not met): v2+intraday beats the raw
ensemble on Brier (0.140 vs 0.157) and temperature CRPS (1.24 vs 1.41); London (largest-n city,
most same-day bets) now edges the market (0.146 vs 0.149, n=29). Overall the market still beats
the model on model-flagged bets (0.111 vs 0.140) — driven mainly by Seoul at leads 2–3 — so
**edge remains unproven**. The fast truth feed (~1-day lag) is filling the gate quickly.

### Recent engine corrections (edge honesty)
- **No more max-selection.** `engine.py` combined ML+ensemble via `our_prob = max(...)`, which
  cherry-picked the more optimistic model and manufactured edge. It now **averages** the two.
- **Lock-in artifact removed.** The `cheapest_entries` block that preserved the cheapest historical
  entry price (an optimistic backtest artifact) is gone.
- **Honest backtest costs.** `evaluate_oos.py` and `historical_backtester.py` now cross a half-spread
  on entry (`config.HALF_SPREAD`) and pay the fee on the winning payout — so measured edge must
  survive realistic execution. Expect apparent ROI to drop; that is the point.
- **`evaluate_oos.py` is the arbiter.** It now prints per-city Brier, temperature-level CRPS
  (MODEL vs ENSEMBLE), an explicit EDGE CHECK PASS/FAIL (model Brier < market AND ≤ ensemble), and a
  shrink-to-market sweep recommending the Brier-minimizing `w`.
- **Shrink-to-market (`config.SHRINK_WEIGHT`, default 1.0 = no-op).** `our_prob = w·model +
  (1-w)·market`. Because the market currently out-predicts the model, betting the pure model loses;
  set `w<1` (from the `evaluate_oos.py` sweep) to deviate from the price only on strong signal. This
  scales edge and Kelly by `w`, so it also sizes conservatively. Override per-run with `--shrink_weight`.
- **α5 coherence bonus is liquidity-guarded (`config.COHERENCE_MIN_LIQ`).** Incoherent bins (sum≠1)
  only earn the α5 score bonus when the market is liquid enough to actually fill both sides;
  otherwise incoherence is just thinness, so it is no longer rewarded.

### One-command validation
`./scripts/raincheck_validate.sh` runs the whole honest loop in a networked env: fetch station truth
(NWS CLI / IEM METAR / HKO) + per-lead archived forecasts (Previous Runs, single- and multi-model) →
`train_calibrator.py` (per-lead EMOS v2) → regenerate both eval trackers (calibrated +
`--disable-calibrator` ensemble) → `evaluate_oos.py` (EDGE CHECK + shrink-weight recommendation) →
`data_status.py` (gate).

---

## Inefficiency Analysis & Alpha Signals (α1–α9)

The engine implements 9 alpha signals to evaluate trading opportunities:
* **α1 (Momentum)**: EMA of forecast movements across recent snapshots.
* **α2 (Spread Proxy)**: Diurnal temperature range used as a convective uncertainty metric.
* **α3 (Student-t tail)**: Custom nu-calibrated Student-t distribution for heavier tails.
* **α4 (Constrained PMF)**: Reconstructs probability-mass-conserving distribution over all market bins.
* **α5 (PMF Consistency)**: Detects markets where the sum of bin probabilities deviates from 1.0.
* **α6 (Volume Recency)**: 24h volume ratio compared to total volume to detect informed traders.
* **α7 (Forecast Convergence)**: Cross-snapshot variance; high variance flags unstable weather.
* **α8 (Market Update Lag)**: Penalizes markets that have not updated in 4+ hours.
* **α9 (Correlated Bet Grouping)**: Caps group exposure to 20% and portfolio exposure to 40%.

### Calibrated Execution Parameters
Grid searches (coordinate ascent + out-of-sample split) endorse the parameter set below as
near-optimal — both `optimizer.py` and `optimizer_full.py` keep the current config rather than
moving off it. **Do not re-tune; the bottleneck is data, not parameters.**

> ⚠️ **Performance honesty.** Earlier "**127.5% ROI**" / "~70% win-rate" figures were *grid-graded
> artifacts*: the backtest graded the "actual" temperature from the **same Open-Meteo grid the
> model forecasts from**, so prediction and outcome shared the grid's error → inflated ROI, often
> further boosted by in-sample filtering. Grading the identical bet set against **station truth**
> (`grading.py`) roughly **halved** measured ROI (e.g. 48% → 22% on one 64–76 bet sample). The
> honest reading: **in-sample station-truth ROI ≈ 20–35%, and out-of-sample edge is unproven**
> (held-out sample still tiny). Params are validated; *edge is not* — see the gate above.
- `MIN_EDGE = 0.06` (6% minimum advantage over market price)
- `MIN_LIQUIDITY = 1000` (USDC)
- `KELLY_FRACTION = 0.50` (half-Kelly bet sizing multiplier)
- `MAX_KELLY_PER_BET = 0.08` (8% maximum bet size on any single market)
- `STALE_HOURS = 4` (penalizes markets with no activity in 4 hours)
- `STALE_MOVE_THRESHOLD = 0.02` (requires a 2% price move to reset the staleness clock)
- `INFORMED_RECENCY = 0.80` (only flags volume as informed if 80%+ occurred in the last 24h)

---

## Out-of-Sample Track Record & Pre-Committed Sample-Size Gate

**Do not draw any performance conclusion (ROI, win rate) until the gate below is met.** This
threshold is committed in advance specifically so it cannot be moved post-hoc to fit a result.

**The gate (both conditions required):**
- **≥ 150 resolved markets graded against STATION TRUTH**, and
- **≥ 100 out-of-sample bets** at the production params.

These constants live in `src/polymarket_weather/data_status.py` (`GATE_RESOLVED_MARKETS`,
`GATE_OOS_BETS`); keep the two in sync if either changes.

**Why:** prior ROI numbers (e.g. the "127.5% ROI" above) were graded from the same Open-Meteo
grid the model forecasts from, so prediction and outcome shared the grid's error → inflated ROI.
Grading the same bets against station truth roughly halved measured ROI, and the honest
out-of-sample sample is currently too small (a handful of held-out bets) to confirm any edge.
Until the gate is met, every ROI figure is noise. (Full re-framing is Handoff Step 2.)

**Check progress** (run from `src/polymarket_weather/`):
```bash
python data_status.py   # prints collected / resolved / station-gradable counts vs the gate
```
Truth publishing lag is no longer the bottleneck: the resolution-faithful feeds (NWS CLI / IEM
METAR) publish within ~1 day (HKO ~1 month), so a market becomes gradable almost as soon as it
resolves. Refresh truth (`fetch_historical_truth.py`) before regenerating the eval tracker.

---

## Unit Testing

Unit tests reside in [tests/test_polymarket_weather.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/tests/test_polymarket_weather.py). Ensure any changes to the question parsers, config mappings, or Kelly calculations pass the test suite:
```bash
pytest -o addopts="" tests/ -v
```
