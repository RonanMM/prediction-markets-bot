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

# The honest-evaluation loop (all from src/polymarket_weather/)
python evaluate_oos.py         # arbiter: model vs market vs ensemble + per-bucket forward gates
python data_status.py          # pre-committed sample-size gate progress
python audit_settlements.py    # grading vs ACTUAL settlements (keep >=95%)
python shoulder_book.py --report   # structure paper book vs its pre-registered gates
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
- **Truth anchor** — the SETTLEMENT-faithful reading. ⚠️ The markets resolve on
  **wunderground.com** pages (`resolution_url`), whose daily extremes are the hourly-METAR
  max/min over the local calendar day — NOT the NWS CLI (1-minute sensors, LST day), which can
  differ by 1°F exactly at bin boundaries (4/60 settlements were once graded backwards; see
  `docs/EDGE_MEGAPLAN.md` §10a). For NYC/Chicago, grading truth is therefore the WU-style
  reconstruction (`wu_truth.py`, from `{slug}_obs_hourly.csv`) with the CLI feed
  (`fetch_historical_truth.py` → `{slug}_historical_actuals.csv`: NWS CLI via IEM for
  KLGA/KORD, IEM METAR daily for EGLC/RKSI, HKO API for HKO) as fallback + glitch guard.
  `audit_settlements.py` checks grades against actual settlements (keep ≥95%). **Meteostat is
  legacy/reference only** (2026-07-03 audit: corrupted). Truth lag ~1 day — **except Hong Kong,
  see below.**

  ⚠️ **HKO publishes in MONTHLY BATCHES, not on a rolling lag** (measured 2026-07-28; the old
  "HKO ~1 month" wording read as rolling and cost an investigation). `CLMMAXT`/`CLMMINT` return
  a whole month at once, roughly **3 weeks after that month ends**: on 2026-07-28 the API served
  exactly 181 rows — Jan 1 to Jun 30 — and the June batch had landed 2026-07-23. So Hong Kong is
  ungradable for the *entire* current month and then jumps in one step. Measured cost at the
  blackout's peak: **84 of 102 HK tracker markets ungradable (18 graded vs 70–116 elsewhere)**;
  the July batch (~2026-08-21) should take HK to ~102 in a single jump. Every other city grades
  July fine, so a lone frozen `hong_kong_historical_actuals.csv` is EXPECTED, not a dead feed —
  check the API's row count before diagnosing.

  🚫 **Do NOT substitute a faster HKO endpoint** (standing decision, Ronan, 2026-07-28). Grading
  must use the source Polymarket actually resolves on, for every city, even when a quicker feed
  exists. Waiting three weeks is cheaper than a fourth broken ruler — see W0, where the wrong
  source graded 4/60 settlements backwards.

  ⚠️ **`resolution_unit` is the MARKET'S BIN GRID, not the source's precision (ruler #7, fixed
  2026-07-31).** HKO publishes tenths (31.6 °C); the questions ask "be 31°C". The field read
  `"0.1 °C"` — the *publishing precision* — so `grading.native_round` put the observation on a
  tenths grid and compared it to a whole-degree bin. `27.8 == 27` is never true, so **every
  Hong Kong bin graded NO: 0/179 YES in the eval tracker while the other four cities ran
  12–23%.** Any shoulder SELL on an HK bin therefore looked like a guaranteed win, and HK was
  **24% of the 5-city structure book** — it would have inflated Leg 1 the moment the July HKO
  batch landed (~2026-08-21), right as that gate matured.

  The rule was **inferred, not assumed**: across all 171 HK markets holding both HKO truth and a
  real Polymarket settlement (2026-03-17 → 2026-06-30), **floor agrees 171/171 (100%)**,
  round-to-nearest 93.6%, the old tenths rule 90.6% — and of the 42 rows that discriminate floor
  from round, every one favours floor (truth 27.8 settled YES for bin **27**, not 28). Polymarket
  truncates: bin "31" means [31.0, 32.0). The unit is now `"whole °C (floor)"`. The settlement
  audit rose 129/133 (97.0%) → **131/133 (98.5%)** on the change.

  🔍 **Why the audit could not catch it, and the general lesson.** `audit_settlements.py` passed
  the whole time: HK agreement is dominated by trivially-correct NOs, so a rule that says NO to
  *everything* still scores ~97%. **An aggregate agreement rate cannot detect a systematically
  one-sided error.** The cheap detector is a per-city P(YES) table — 0.000 against 0.124–0.228
  is visible instantly. Add that comparison whenever a new city or unit is introduced.
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
| `grading.py` | Grades markets against **settlement-faithful** truth: WU-style reconstruction for NYC/Chicago (`wu_truth.py`), station feeds elsewhere. |
| `wu_truth.py` | **Settlement truth (W0).** Markets resolve on wunderground.com, not the NWS CLI; reconstructs WU's daily extremes (hourly-METAR max/min over the local calendar day) for NYC/Chicago. Validated: fixed 3/4 settlement-audit misses without breaking the other 56. |
| `audit_settlements.py` | Permanent guard: compares `resolves_yes` to ACTUAL settlements (a resolved market's final pinned price). Run after any truth/grading change; exits 1 below 95% agreement. |
| `shoulder_book.py` | Model-free **two-leg structure paper book** (megaplan §10b/§10d): Leg 1 sells pre-day 5–35¢ shoulder bins; Leg 2 buys 65–85¢ YES-favorites >12 h before day end. Auto-records each collector cycle (hook in main.py), settles with the verified 0.05·p·(1−p) taker fee, prints pre-registered forward gates. No real orders until a gate passes. |
| `shoulder_book_breadth.py` | The same structure legs across EVERY active weather city (~49), graded on Polymarket's own settlement. Carries **four** pre-registered forward gates — 1b moderate [10,25)¢, 1c deep [5,10)¢, and 1d/1e city-selection (megaplan §12c). ⚠️ **`CITYSEL_A`/`CITYSEL_B` are FROZEN LITERALS — never recompute them from the book.** A selection rebuilt at report time re-fits on every 2-hourly dashboard build, so the gate can never fail: it scores its own training set and prints a green number. Multiplicity is derived (`family_z()` from `BREADTH_GATE_FAMILY`), never hard-coded — add a gate, bump the family. |
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

Status (2026-07-13, gate MET at 240 gradable markets, settlement-faithful labels): **no
forecasting edge** — market Brier **0.128** < ensemble 0.160 < model 0.166 (backtest ROI −8.7%,
but small-sample ROI swings several points/day; read the Brier), so the model book stays OFF. Under settlement labels the calibrator
currently loses the paired check to the raw ensemble (it was trained on the old CLI-style
truth) — **retraining EMOS/Tmin/intraday against settlement truth is the top queued model
task** (`docs/EDGE_MEGAPLAN.md` W0.2). The live positive-EV candidates are the model-free
structure legs in `shoulder_book.py` (see EDGE_MEGAPLAN §10b/§10d), in forward paper trials.

> ⚠️ **The cloud-served "model" is a DEGRADED version of the one described above — so read
> "the calibrator loses to the ensemble" as measuring the pipeline, not the design (verified from
> the 2026-07-20 retrain log `26515a5`).** Three gaps: (1) `retrain.yml` fetched only
> `fetch_historical_leads[_mm].py`, never `_cand`/`_jma`, so `train_calibrator` could never build
> `mm_mean` (it needs the FULL per-city model set, `MM_MODELS_BY_CITY`) and selected `best_match`
> for every city — the multi-model blend never trained. **Fixed:** `retrain.yml` now fetches
> `fetch_historical_leads_cand.py`, and cand also fetches `jma_seamless` (Seoul's blend needs jma;
> the model id is confirmed from `fetch_weather.MULTIMODEL_MODELS` and verified to have full
> previous-runs archive coverage for Seoul). So training can now build `mm_mean` for **all five
> cities**, and serving `daily_mm.csv` already carries all six blend models incl. jma
> (well-covered on recent rows) — so the plumbing is complete end-to-end and a future retrain
> will train+serve `mm_mean` where it wins the holdout. (2) **Serve-time fallback — investigated
> 2026-07-21, NOT a contamination.** The "no usable calibrated input — raw EnsemblePredictor"
> warning fires once-per-city (`_warn_once`) for early-March boundary snapshots that get *skipped*;
> the committed calibrated tracker has **zero** `ensemble`-`sigma_source` rows (all 493 are
> `emos_v2*`), so MODEL≈ENSEMBLE is NOT tautological — the model genuinely lost as the calibrated
> best_match model. **Root-caused 2026-07-23 (why best_match, quantified):** the blend (`mm_mean`)
> serves only **~16%** of eval rows. `_latest_mm_mean` requires the FULL trained model set non-NaN
> in the live `daily_mm` feed, but (a) `daily_mm` only starts **2026-07-03**, so 55% of the eval
> (Mar–Jun) has no multimodel data at serve time, and (b) early-July rows hit collection-ramp-up
> NaNs in `aifs`/`gem`/`mf` — any ONE missing model degrades the whole blend (the all-or-nothing
> rule is a deliberate anti-skew guard). **Self-healing** (`aifs`/`gem` now 0% NaN in current
> fetches); a future retrain, once `daily_mm` has weeks of complete data, blend-serves far more.
> **Deliberately NOT fixed:** serving the backtest from the historical archive would make it
> measure a cleaner data path than live delivers (a backtest/live mismatch), and per the bet
> meta-analysis even a fully-served blend loses to the market (0.145 vs 0.121) — this is
> measurement-fairness, not edge. (3) Tmin archives (`_min`) **now fetched in cloud** (`retrain.yml`, 2026-07-22):
> `fetch_historical_leads_min.py` (made incremental to match the Tmax fetchers) writes both min
> files and a retrain now rebuilds the Tmin stack (all 5 cities select `mm_mean`, e.g. Seoul min
> RMSE 1.10 blend vs 1.73 best_match) instead of freezing at the last hand-commit (2026-07-04).
> **Bottom line:** the edge verdict is unchanged and sound. The model loses on its overconfident
> TAILS (`[0,0.1)` predicts 3.6%, realizes 15.5%) — a distributional problem the mean-input blend
> won't fix. Making the model competitive needs a tails/dispersion change, not more data plumbing.

### Recent engine corrections (edge honesty)
- **No more max-selection.** `engine.py` combined ML+ensemble via `our_prob = max(...)`, which
  cherry-picked the more optimistic model and manufactured edge. It now **averages** the two.
- **Lock-in artifact removed.** The `cheapest_entries` block that preserved the cheapest historical
  entry price (an optimistic backtest artifact) is gone.
- **Honest backtest costs.** `evaluate_oos.py` and `historical_backtester.py` now cross a half-spread
  on entry (`config.HALF_SPREAD`) and pay the fee on the winning payout — so measured edge must
  survive realistic execution. Expect apparent ROI to drop; that is the point.
  ⚠️ **Two fee models exist; know which one you are reading.** The model-book backtesters above
  still settle with the LEGACY `config.FEE_RATE = 0.02` (2% of the winning payout). The REAL
  verified schedule (E1, 2026-07-13) is `config.taker_fee_per_share` = `0.05·p·(1−p)` — max
  1.25¢/share at p=0.50, ~0.24¢ at p=0.95 — and **makers pay NOTHING**, plus a Maker Rebates
  Program returning 25% of collected weather taker fees (`MAKER_FEE`, `MAKER_REBATE_SHARE`).
  `shoulder_book.py` / `shoulder_book_breadth.py` already use the real formula; migrating the
  model-book backtesters off `FEE_RATE` is still queued. The 2% assumption is *conservative*
  (harsher than reality, most so at extreme prices), so it depresses reported ROI — but it does
  NOT touch Brier, which is the arbiter, so the edge verdict is unaffected either way. Live
  schedule re-verified 2026-08-01: `{rate 0.05, takerOnly: true, rebateRate: 0.25}`.
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
- **E3 per-bucket selective aggression (2026-07-11, `docs/EDGE_MEGAPLAN.md`).** Every opportunity
  carries `bucket` ("City|same-day" / "|1d" / "|2d+", `config.bucket_key`) and `live_eligible`
  (bucket ∈ `config.LIVE_BUCKETS`); the tracker still records ALL flags so the eval keeps its full
  sample. `evaluate_oos.py` prints the per-bucket Brier table and each nominated bucket's FORWARD
  gate (≥40 bets graded after `E3_NOMINATION_DATE` with model ≤ market Brier) — no real size until
  a bucket passes. W2 sigma-inflation-on-disagreement was tested the same day and REJECTED (Brier
  improves, ROI does not — the failure is the center, not the spread); see the note above
  `config.SHRINK_WEIGHT`.
- **W0 settlement truth (2026-07-12) — the fourth broken ruler.** Markets resolve on
  wunderground.com (see `resolution_anchors.resolution_url`), whose extremes can differ from the
  NWS CLI by 1°F at bin boundaries; 4/60 settlements were graded backwards. `wu_truth.py` is now
  primary truth for NYC/Chicago (audit 59/60 via `audit_settlements.py`). This KILLED the original
  NYC|same-day nomination (its edge was the wrong labels); current nominations Seoul|1d +
  Chicago|1d, forward clock restarted 2026-07-12. Queued: retrain EMOS/Tmin/intraday against
  settlement-faithful targets (they were trained on CLI truth; under corrected labels the
  calibrator loses the paired ensemble check 0.171 vs 0.159).

### One-command validation
`./scripts/raincheck_validate.sh` runs the whole honest loop in a networked env: fetch station truth
(NWS CLI / IEM METAR / HKO) + per-lead archived forecasts (Previous Runs, single- and multi-model) →
`train_calibrator.py` (per-lead EMOS v2) → regenerate both eval trackers (calibrated +
`--disable-calibrator` ensemble) → `evaluate_oos.py` (EDGE CHECK + shrink-weight recommendation) →
`data_status.py` (gate).

---

## Cloud Automation (GitHub Actions) — everything runs in CI, not on a laptop

Four workflows in `.github/workflows/`. All commit as `raincheck-collector`
(`actions@users.noreply.github.com`); a commit authored by a human is a manual intervention.

| workflow | trigger | runs | commits |
|---|---|---|---|
| `collect.yml` | hourly (`13 * * * *`) | `main.py --collect-only` | `data/polymarket`, `data/weather`, `shoulder_paper.csv` |
| `truth-eval.yml` | daily 09:00 UTC | truth + obs refresh → regenerate tracker → **audit** → gate | `output/` trackers |
| `dashboard.yml` | every 2h (`40 */2 * * *`) | truth + obs refresh → `build_dashboard.py` | **separate public repo** (below) |
| `retrain.yml` | weekly (`20 4 * * 0`) + `workflow_dispatch` | full fetch → rebuild targets → `train_calibrator.py` → both trackers → arbiter | `models/`, `output/` |

⚠️ **Those crons are deliberately ~2× the cadence we want.** GitHub schedules are best-effort:
measured 2026-07-27, the old every-2h collect cron delivered **3h49 gaps** (roughly every other
slot, each 1-2h late) and dashboard runs fired **~3h after** their slot with some dropped. Runs
are idempotent (append-only + dedupe on read) and the repo is public (free minutes), so
over-scheduling is the fix. Judge health by the gap between *data* timestamps, never by the cron.

`retrain.yml` runs **weekly (Sun 04:20 UTC)** as of 2026-07-28, plus manual dispatch. Weekly is
cheaper than monthly, not dearer: the per-fetch archive cache is evicted after 7 days unused, so
a monthly schedule would run COLD every time (full 2022→now refetch, ~3-4h) while weekly keeps it
warm and pulls only the new week. **Retrain is NOT the measurement tool** — `truth-eval` (daily)
regenerates the trackers and grades; retrain does so only as a side effect. ⚠️ Model artifacts now
change under the E3 forward gates, so a gate's forward sample can span model versions — fine while
everything is paper, revisit before any bucket goes live. ⚠️ It is also timeout-constrained: run #1 (2026-07-17) was killed at
`timeout-minutes: 60`, so W0.2 did not land on that attempt. The heavy steps are
`fetch_historical_leads*.py` (2022→now × leads 1–7 × models × 5 cities, Open-Meteo rate limits).
Raise the timeout rather than assuming the run "just took a while".

The dashboard publishes to **`RonanMM/prediction-markets-bot-dashboard`** (`PAGES_REPO`), a
separate public repo, via the `DASHBOARD_TOKEN` secret. Without that secret the build *succeeds*
and publishes nothing but a `::warning::` — a broken publish looks identical to a healthy one in
the run list. Verify by checking the timestamp in the published `data.json`, not the run status.

### The two-tier data policy — and the trap inside it
`.gitignore` splits data by one question: *can this be refetched?*
- **Perishable → committed.** Market snapshots, price history, forecast/ensemble/multi-model
  rows. A price not recorded at 14:00 is gone forever; nobody re-serves it.
- **Refetchable → gitignored.** `*_historical_*.csv`, `*_obs_hourly.csv`, `*_nbm.csv`. IEM / NWS
  / HKO / Open-Meteo serve full history on demand. **EXCEPTION (do not re-ignore):**
  `*_historical_actuals.csv` (the daily settlement truth, ~200 KB/city) IS committed as a
  resilience fallback — IEM 503s intermittently and a fresh cloud runner has no other copy, so an
  outage was dropping whole cities from grading (2026-07-23: a dashboard build published with only
  3/5 cities, corrupting every Brier/ROI/bucket number). `fetch_historical_truth` "keeps the
  existing CSV" on a failed refetch, so the committed copy keeps the build complete; `truth-eval`
  re-commits the refreshed file daily. `build_dashboard._missing_cities` is a hard guard on top:
  it refuses to publish (exits non-zero → last-good copy kept) if any `CITY_ORDER` city is
  ungradable.

⚠️ **The trap (fixed 2026-07-20, do not regress).** `wu_truth.py` reads
`{slug}_obs_hourly.csv` and returns `None` **silently** when it is missing, so
`grading.fetch_actual_weather` falls back to the NWS CLI feed — the pre-W0 ruler. That file is
gitignored and was written only by `main.py` on the *collect* runner, so `truth-eval`,
`dashboard` and `retrain` — each a fresh runner — had **never once graded with it present**.
Measured cost: of 174 rows where both sources exist they disagreed on 99 (mean 0.56 °C, WU
lower — hourly METARs miss peaks the CLI's 1-minute sensors catch) and **18/573 rows (3.1%,
15 market-days) flipped verdict**, clustered in `NYC|same-day` (8) and `Chicago|1d` (3 of n=20,
a live nomination). **Any workflow that grades must run `fetch_station_obs.py` first** — full,
not `--recent`, since a 3-day top-up onto a nonexistent file leaves the same gap.

⚠️ **The SAME trap, worse form — a TRUNCATED obs file (2026-07-30 incident, fixed, do not
regress).** The trap above is about a *missing* file. A **partially-written** one is far more
dangerous: it has the right name, a plausible row count, and produces a green run.

What happened: `fetch_station_obs.py` fetches one chunk per year (2022…now). LGA's **2026** chunk
failed all four retries; the loop **silently skipped it** and concatenated the rest. The surviving
35,032 rows cleared the only guard — an *absolute* floor of `len(df) < 20000` — so the complete
40,071-row file was **overwritten, losing every 2026 observation**. `wu_truth` then returned `None`
for all 2026 dates and NYC grading fell back to the CLI ruler.

Measured cost, both directions, same sample (n=421 / 178 city-days):

| | settlement audit | pooled model−market gap |
|---|---|---|
| healthy | 129/133 = **97.0%** ✅ | **+0.0178** CI [+0.0040, +0.0317] |
| truncated | 126/133 = **94.7%** ❌ | **+0.0122** CI [−0.0012, **+0.0257**] |

The interval went from entirely-above-zero to **spanning zero** — i.e. the project's headline
verdict flipped from "the model is measurably worse than the market" to "indistinguishable" **on a
dropped HTTP request**, and was published to the public dashboard. Nothing errored.

Three fixes, all in place:
1. **`fetch_station_obs` GUARD 1** — if any year chunk fails after retries, keep the existing CSV.
   A partial fetch must never overwrite a complete file.
2. **`fetch_station_obs` GUARD 2** — refuse to write when the refetch has fewer rows *or* an older
   latest observation than what is on disk. Refetched obs must only ever grow. This catches what
   guard 1 misses (an upstream range quietly returning less, a station rename).
3. **`dashboard.yml` now runs `audit_settlements.py` before building.** `truth-eval` has run the
   audit since W0; the dashboard never did, and *that asymmetry is why this reached the public
   page*. A degraded ruler now blocks the publish and the last good copy stays up.

The absolute `< 20000` floor is kept as a third, weakest guard — it is what failed here, because
35,032 > 20,000. **An absolute floor cannot detect a regression; only a comparison against the
previous state can.**

⚠️ **RULER #13 — routine METARs are not wunderground's day (found + fixed 2026-08-12).** The two
incidents above are about the obs file being *missing* or *truncated*. This one is about the file
being **complete and wrong**: `fetch_station_obs.py` requested `report_type=3` (routine METAR) for
its entire life, but WU's daily table — what Polymarket settles on — lists **every** observation
including **SPECIs**, filed off-schedule when conditions change, and takes its high/low over that
full set. Routine-only therefore reconstructs a max that is too LOW and a min too HIGH, and the
error is **strictly one-sided**: dropping observations can never widen an extreme.

Caught from the settlement audit's *disagreement list*, not its rate: 4 of 6 disagreements were
Atlanta, paired (adjacent bins wrong in opposite directions on the same day) and all ~2 °F low —
the signature of a ruler, not of noise. ATL 2026-08-10 peaked **94 °F in a 15:39 SPECI**; the :52
routines top out at 92 °F, so we graded the 92-93 bin YES against a 94-95 settlement.

Measured 2026-03-01→08-12 over nine US stations, 1,476 station-days: specials raise the daily max
on **2.5%** of days and lower the daily min on **6.1%** (Tmin markets are hit MORE often, and are
~20% of the book), changing the whole-°F max on 35 station-days. On the four settled market-rows
in the entire history where the two candidate rulers grade differently, **+specials agrees with
the actual settlement 4/4 and routine-only 0/4** — the same discriminating inference that settled
the HK floor rule, just with a much smaller n, so treat it as strong-mechanism/weak-sample.

Two things this changes beyond grading:
- **`venue_basis.py` (§13b) was measuring partly its own bug.** It compares the CLI against OUR WU
  reconstruction, so a one-sidedly-low WU inflates the "CLI reads higher" basis that IS its
  estimand. Its headline 53.3%/32.4% figures are quarantined until a specials-inclusive backfill
  regenerates them; the forward gate was only 13/30 city-days, so nothing mature was lost.
- **Direction of the correction.** Truth biased low makes hot bins look like losers, so the market
  — which priced the true outcome — was being scored against labels tilted against it. Fixing this
  makes the market look *better*, consistent with every prior measurement change.

The fix is self-healing: obs rows are now stamped `iem_metar+speci:{station}`, and a `--recent`
top-up against a file still stamped plain `iem_metar:` forces a one-time full backfill rather than
leaving **one file graded by two rulers** split at the deploy date. Three mutation-tested guards
cover it. **Corollary: the settlement audit's aggregate rate did not move (99.6%) — 4 bad rows in
1,409 is invisible in the percentage. Read the disagreement LIST, not just the rate; a ruler bug
shows up as clustering by city and by direction long before it shows up in the total.**

Corollary worth remembering: the eval trackers carry **no grade column**. Grading is applied at
*read time* by `evaluate_oos.py` / `data_status.py` / `build_dashboard.py`. So a truth-source bug
corrupts *reported numbers*, never stored data — which is why these fixes need no backfill. It also
means a bad run leaves **no trace**: the numbers were simply wrong while it was live, and correct
again afterwards. Judge truth health from `audit_settlements.py`, never from a green workflow run.

⚠️ **Training targets go stale silently.** `train_calibrator.py` reads
`{slug}_settlement_actuals.csv` via `settlement_truth.load_training_truth`. That file **is
committed** but is regenerated by no workflow and no `main.py` path — `settlement_truth.py` is a
manual entrypoint. It had frozen at 2026-07-12/13 (Hong Kong 2026-05-31), and because training
merges against it with `how="inner"`, every newer settled market was dropped from training with
no warning. `retrain.yml` now rebuilds it in-runner; it is not committed, and the in-runner copy
is what trains.

### Working from a fresh clone
A new machine has the committed (perishable) data but **none** of the refetchable archives, so
`data_status.py` reports `0 gradable` and `evaluate_oos.py` refuses to run. That is expected, not
data loss. Before any local evaluation:
```bash
cd src/polymarket_weather
python fetch_historical_truth.py    # station truth, all 5 cities (~1 min)
python fetch_station_obs.py         # hourly METARs — REQUIRED for settlement-faithful grading
```

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
The values themselves live in `config.py` (`MIN_EDGE`, `MIN_LIQUIDITY`, `KELLY_FRACTION`,
`MAX_KELLY_PER_BET`, `STALE_HOURS`, `STALE_MOVE_THRESHOLD`, `INFORMED_RECENCY`) — read them
there rather than from a copy that can drift.

---

## Out-of-Sample Track Record & Pre-Committed Sample-Size Gate

**Do not draw any performance conclusion (ROI, win rate) until the gate below is met.** This
threshold is committed in advance specifically so it cannot be moved post-hoc to fit a result.

**The gate (both conditions required):**
- **≥ 150 resolved markets graded against STATION TRUTH**, and
- **≥ 100 out-of-sample bets** at the production params.

These constants live in `src/polymarket_weather/data_status.py` (`GATE_RESOLVED_MARKETS`,
`GATE_OOS_BETS`); keep the two in sync if either changes.

**The gate is MET (2026-07-13: 240 markets / 354 bets) and the verdict is in:** the model does
NOT out-predict the market (Brier 0.166 vs 0.128; ROI −8.7% and day-to-day noisy at production
params — read the Brier, not the ROI) — see
`STATUS.md` and `docs/EDGE_MEGAPLAN.md`. The gate machinery stays in force for everything new:
per-bucket forward gates (`config.LIVE_BUCKETS`, `E3_NOMINATION_DATE`) and the structure book's
pre-registered gates (`shoulder_book.py`) — nothing trades real money until its own gate passes.

**Why the gate discipline exists:** prior ROI numbers (e.g. the "127.5% ROI" above) were graded
from the same Open-Meteo grid the model forecasts from → inflated ROI. Station-truth grading
halved it; settlement-truth grading (W0) moved it again. Every measurement change so far has
made the model look worse and the market look better — assume the same until proven otherwise.

**Check progress** (run from `src/polymarket_weather/`):
```bash
python data_status.py        # collected / resolved / gradable counts vs the gate
python audit_settlements.py  # grading vs actual settlements (must stay ≥95%)
```
Truth publishing lag is no longer the bottleneck **for the four METAR/CLI cities**: those feeds
publish within ~1 day, so a market becomes gradable almost as soon as it resolves. **Hong Kong is
the exception** — HKO's monthly batch (see Resolution Anchors) leaves the whole current month
ungradable, so HK is permanently the smallest per-city sample and its dashboard row is showing
last month's answer. Refresh truth
(`fetch_historical_truth.py`) before regenerating the eval tracker; `wu_truth.py` needs only the
obs top-up that runs inside `main.py`.

---

## Unit Testing

Unit tests reside in [tests/test_polymarket_weather.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/tests/test_polymarket_weather.py). Ensure any changes to the question parsers, config mappings, or Kelly calculations pass the test suite:
```bash
pytest -o addopts="" tests/ -v
```
