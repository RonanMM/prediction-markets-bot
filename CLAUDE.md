# CLAUDE.md

This file provides guidance to coding assistants (like Claude/Antigravity) working on the Raincheck weather prediction tracker repository.

## Project Overview

Raincheck is a Polymarket weather prediction market tracker that compares market-implied temperatures against meteorological forecasts to identify pricing inefficiencies. 

The active project code lives in `src/polymarket_weather/`, not in `src/raincheck/` (which is a near-empty PyScaffold skeleton).

---

## Commands

All commands should be executed from the `src/polymarket_weather/` directory:

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
python optimizer.py

# Run the Full-Pipeline Grid Search Optimizer (runs engine back-to-front, slow)
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
              [MLCalibratorPredictor / EnsemblePredictor / NWPFallbackPredictor]
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
- **Truth anchor** — `meteostat_id` (the station whose observations grade/label bets, via `grading.py`).
- **Forecast anchor** — `forecast_lat`/`forecast_lon` (where Open-Meteo is pointed). Usually the
  station's location, but **not always**.

⚠️ **Seoul is the key exception. Do NOT "correct" its forecast coords back to the airport.** The
Incheon-airport ERA5 grid cell is coastal/sea-damped and predicts the station poorly, so the
forecast anchor is a skill-optimized inland **"Bucheon corridor" point (37.5035, 126.766)**, while
truth/resolution stay **Incheon RKSI / Meteostat 47113**. Validated: CV RMSE ≈1.2 (Bucheon) vs ≈1.96
(airport cell). A loud comment in the file says the same.

---

## Module Responsibilities

| File | Responsibility |
|---|---|
| `resolution_anchors.py` | **Single source of truth** for per-city resolution / truth / forecast anchors (see above). |
| `config.py` | `CITIES` (coords DERIVED from `resolution_anchors.py`), endpoints, fees, and optimal sizing limits. |
| `grading.py` | Grades markets against the resolution-**station** observation (station truth), not a forecast grid cell. |
| `data_status.py` | Track-record progress report: collected / resolved / station-gradable counts vs the pre-committed gate. |
| `fetch_polymarket.py` | Gamma API (market search/details) + CLOB API (price history). Parses temperature bins from market questions. |
| `fetch_weather.py` | Open-Meteo 16-day forecast. Returns daily and hourly forecasts. |
| `fetch_ensemble.py` | Fetches Open-Meteo ensemble forecasts (ECMWF/ICON spreads) to compute dynamic uncertainty metrics. |
| `processing.py` | CSV append, normalisation, and deduplication logic. |
| `engine.py` | Runs the core analysis loop, applying conflict gating, group caps, and portfolio exposure limits via `WeatherBettingBot`. |
| `signals.py` | Implementation of alpha signals (momentum, convergence, staleness, volume ratios). |
| `models.py` | Dataclasses for `MarketBin` and `Opportunity`. |
| `pmf.py` | Parsing rules for questions (exact, gte, lte) and reconstruction of the probability mass function (PMF). |
| `reports.py` | Terminal reports, summary generators, and plotting wrappers. |
| `visualization.py` | Generates diagnostic PNG plots (forecast drift, efficiency signals, price momentum). |
| `ml_calibrator.py` | Random Forest Regressor to calibrate raw numerical weather forecasts. |

---

## Predictor Models

The system supports three predictor models under `predictors/`:
1. **`MLCalibratorPredictor`**: City-specific Random Forest models trained on historical forecast errors. Uses `day_of_year` and `raw_mu` to correct forecasts. If the ensemble standard deviation (`sigma`) is too high (`> 1.2`), it bypasses itself and falls back to the physics-based ensemble.
2. **`EnsemblePredictor`**: Predicts probabilities using the ECMWF/ICON ensemble spread (standard deviation and Student-t tails).
3. **`NWPFallbackPredictor`**: Used when ensemble data is missing; falls back to static standard deviation tables defined in `config.py`.

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
The binding constraint is usually **Meteostat's publishing lag** (~2–3 weeks): a market only
becomes gradable once its resolution-station observation is published, so refresh truth
(`fetch_historical_truth.py`) before regenerating the eval tracker.

---

## Unit Testing

Unit tests reside in [tests/test_polymarket_weather.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/tests/test_polymarket_weather.py). Ensure any changes to the question parsers, config mappings, or Kelly calculations pass the test suite:
```bash
pytest -o addopts="" tests/ -v
```
