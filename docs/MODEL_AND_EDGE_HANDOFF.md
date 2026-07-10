# Raincheck — Model & Edge-Seeking Handoff

A self-contained brief on **how the predictor works** and **everything we've tried to gain an
edge**, for picking the project up in a fresh conversation. Companion docs: `STATUS.md`
(plain-English state), `CLAUDE.md` (technical reference), `docs/BUGFIX_MEGAPLAN.md` +
`docs/BUGFIX_EXECUTION_REPORT.md` (the recent bug-fix pass).

---

## 0. TL;DR

- **Goal:** bet Polymarket "what will the temperature be?" markets by forecasting the resolving
  weather station better than the market price does.
- **Current verdict (gate met, honest eval):** **no edge over the market.** Brier — market
  **0.128** < model **0.163** < ensemble **0.166**. The model is a better forecaster than its own
  ensemble baseline, but the *market* is a better forecaster than the model, so betting the model
  loses (**−20% ROI over 199 graded bets**; the shrink sweep says trust the model 0%).
- **The one place edge might still hide (untested):** same-day *intraday* bets, where the model
  can condition on the temperature already observed that day. The backtest barely exercises this.

---

## 1. What the system does (pipeline)

```
Polymarket Gamma/CLOB API ── markets, prices ──┐
Open-Meteo (forecast / ensemble / multi-model) ─┼─→ append-only CSVs ─→ predictor ─→ engine ─→ bets
NWS CLI / IEM METAR / HKO ── station truth ─────┘                                   (+ eval)
```

- **Markets** (`fetch_polymarket.py`): temperature markets per city, parsed into bins. Question
  formats: `exact` ("be 18°C on X"), `gte` ("18°C or higher"), `lte` ("or lower"), and
  `range` ("between 62-63°F" — the dominant US format).
- **Weather** (`fetch_weather.py`, `fetch_ensemble.py`): Open-Meteo deterministic forecast, a
  122-member ICON+GFS+ECMWF ensemble, and per-model deterministic runs for the blend.
- **Truth** (`fetch_historical_truth.py`): the *resolution-faithful* station reading the market
  actually pays out on — NWS CLI (KLGA/KORD), IEM METAR daily (EGLC London, RKSI Seoul), HKO API
  (Hong Kong). **Not** a forecast grid cell, and **not** Meteostat (found corrupted, retired).
- **Cities:** New York (KLGA), Chicago (KORD), London (EGLC), Seoul (RKSI), Hong Kong (HKO).
- **Collection:** append-only CSVs, deduped on read; local macOS launchd agents
  (`com.raincheck.collect` every 2h, `com.raincheck.truth-eval` daily) + a GitHub Actions mirror.

**Resolution anchors** (`resolution_anchors.py`, the single source of truth) separate three things
per city: the **resolution** station (what the oracle reads), the **truth** feed (what grading
reads), and the **forecast** point (where Open-Meteo is aimed). Usually the same location — the key
exception is **Seoul**, whose forecast point is a skill-optimized inland "Bucheon corridor"
(37.5035, 126.766), not the airport (the airport's ERA5 cell is sea-damped; CV RMSE ≈1.2 vs ≈1.96).

---

## 2. The prediction model (current stack, in detail)

The predictor turns a forecast into a **probability distribution over the day's temperature**,
then reads off bin probabilities. Predictors live in `predictors/`; the default is **EMOS v2**.

### 2a. EMOS v2 — the calibrator (`predictors/emos.py`, default)
Per-city, **per-lead (1–7 days)** Nonhomogeneous Regression trained on **real archived forecasts**
(Open-Meteo Previous Runs, 2022→now) against station truth — so train and serve share both the
model family and the lead time. For a bet at lead ℓ:

```
mu    = a_ℓ + b_ℓ · input + seasonal(day_of_year)      # self-gated per lead by temporal holdout
sigma = max( flow_sigma , sigma_ℓ )                    # sigma_ℓ = honest holdout residual std (NEVER gated)
        where flow_sigma = ensemble_std + diurnal_boost
nu    = from residual kurtosis (Student-t tails)
```

- `sigma_ℓ` (the per-lead residual std) is the **overconfidence fix** — v1 was trained on *reanalysis*
  and understated live forecast error 2–3× at betting leads, producing overconfident tails that lost
  on Brier. It is never gated.
- The predictive distribution is **Student-t** (`pmf.py`), with `sigma` treated as a **standard
  deviation** and converted to the t-scale via `_t_scale(σ,ν)=σ·√((ν-2)/ν)` (a recent fix — it used
  to pass σ straight in as the scale, over-dispersing by ~40%).

### 2b. The mean `input` — multi-model blend
Per city we choose the mean input by holdout RMSE between:
- **`mm_mean`** — the deterministic **multi-model mean** served exactly from `{slug}_daily_mm.csv`
  (same models as training). Models per city come from a blend-expansion sweep:
  base ECMWF/GFS/ICON, **GEM** broadly adopted, **Météo-France** (NYC/London), **ECMWF-AIFS**
  (Seoul/NYC/London — Seoul gains 5–8% at every lead), **JMA** (Seoul only). **CMA/BOM rejected.**
- **`best_match`** — a single deterministic forecast.
- Fallbacks when the exact blend is unavailable: **`mm_proxy`** (live ensemble mean with its *own*
  wider-sigma fit) → NBM (US, self-gating candidate) → best_match.

### 2c. Ensemble predictor (`predictors/ensemble.py`)
122-member ICON+GFS+ECMWF ensemble → dynamic σ from the spread (+ convective diurnal boost) and
Student-t ν from the (p90−p10)/std ratio. It's both the **EMOS fallback** and the
**`--disable-calibrator` baseline** the model is measured against. `NWPFallbackPredictor` (static σ
tables) is the last resort.

### 2d. Intraday conditioning — same-day bets (the most promising, least-tested lever)
For a bet placed **on the target's station-local day**, the model can see the temperature already
observed (hourly METARs, `fetch_station_obs.py`):
- The predictive distribution is **floored** at the running observed max `M` (Tmax can't finish
  below what's already been recorded); censoring threads through the whole PMF/CRPS.
- For self-gated local hours, a per-hour regression `Tmax = a_h + b_h·fcst + c_h·M_h` **replaces**
  mu/sigma. By ~14:00 local it roughly halves σ; by ~17:00 σ≈0.4°C with c≈0.95.
- Recently fixed for train/serve consistency (apply the *last completed* hour's fit; feed the
  lead-1 forecast the fit was trained on). **Effect is live-only — the sparse historical backtest
  barely exercises it, so its value is genuinely unknown.**

### 2e. Tmin markets ("lowest temperature", ~20% of markets)
A dedicated **Tmin EMOS** mirroring the Tmax stack (per-lead, same per-city blends, live min
inputs), with a **ceiling** at the running observed min (the overnight low is usually locked in by
mid-morning; London σ@12 ≈ 0.7°C). If a city has no trained Tmin params the predictor returns
None and those bins are skipped — never mispriced off the Tmax distribution.

---

## 3. From distribution → bets (the engine, `engine.py`)

1. **Parse** each market question → bins (`pmf.parse_question`); reconstruct a PMF over the bins.
2. **Price** each bin from the predictive distribution: `P(bin)` via the Student-t CDF, honoring
   floor/ceiling censoring. Range bins integrate `[lo−hw, hi+hw)` to match whole-°F rounding.
3. **Edge** = `our_prob − market_price`, computed for both the calibrated model and the ensemble.
4. **Conflict gating:** skip a bet if the model and ensemble disagree on the side.
5. **Combine:** average model+ensemble probabilities (never `max` — that manufactured edge);
   EMOS v2 stands alone (averaging it with the raw ensemble would re-thin the tails).
6. **Shrink to market:** `our_prob = w·model + (1−w)·market` (`SHRINK_WEIGHT`, default 1.0). Because
   the market currently out-predicts the model, the honest sweep recommends **w≈0**.
7. **Size:** fractional Kelly with fee-adjusted odds, capped per-bet / per-group / per-portfolio.
8. **Honest costs:** cross `HALF_SPREAD` (1¢) on entry, pay `FEE_RATE` (2%) on the winning payout.

### Alpha signals (α1–α9), used for scoring/gating, not raw pricing
α1 momentum (EMA of forecast drift) · α2 diurnal-spread proxy · α3 Student-t tails · α4 constrained
PMF · α5 PMF-consistency (liquidity-guarded) · α6 volume recency · α7 forecast convergence · α8
market-update staleness · α9 correlated-bet grouping (group ≤20%, portfolio ≤40%).

### Calibrated execution params (grid-searched, "don't re-tune; the bottleneck is data")
`MIN_EDGE=0.06 · MIN_LIQUIDITY=1000 · KELLY_FRACTION=0.50 · MAX_KELLY_PER_BET=0.08 ·
MAX_KELLY_PER_GROUP=0.20 · MAX_TOTAL_KELLY=0.40 · FEE_RATE=0.02 · HALF_SPREAD=0.01 · SHRINK_WEIGHT=1.0`

---

## 4. Everything we've tried to gain edge

**Adopted (in the model today):**
1. **Per-lead EMOS v2 on real archived forecasts** — replaced a reanalysis-trained v1 that was 2–3×
   overconfident at betting leads.
2. **Honest holdout residual σ floor per lead** — the overconfidence fix; never gated.
3. **Multi-model blend, per-city model selection** — GEM (broad), Météo-France (NYC/London),
   ECMWF-AIFS (Seoul/NYC/London), JMA (Seoul). Chosen by holdout RMSE.
4. **Student-t heavy tails** with ν from ensemble/residual kurtosis.
5. **Dynamic (flow-dependent) σ** from the live ensemble spread + convective diurnal boost.
6. **Intraday same-day conditioning** — floor at running max, per-hour regression on observed M.
7. **Dedicated Tmin model** for "lowest temperature" markets, with a ceiling at the running min.
8. **Seoul Bucheon forecast anchor** — a skill-optimized inland point (CV RMSE 1.2 vs 1.96).
9. **Resolution-faithful station truth** (NWS CLI / IEM METAR / HKO) — replaced corrupted Meteostat.
10. **Alpha signals** (momentum, convergence, staleness, volume recency, coherence) for scoring.
11. **Conflict gating + averaging** model & ensemble; group/portfolio Kelly caps.
12. **Shrink-to-market** — deviate from the price only in proportion to signal strength.

**Tested and REJECTED (valuable negatives — don't re-try without new reason):**
- **RandomForest bias-correction calibrator** — verified net-negative, removed.
- **EMOS v1 (reanalysis-trained)** — understated live error 2–3×; replaced by v2.
- **NBM (NWS National Blend) station guidance for US cities** — our blend beats raw NBM by
  0.4–0.75°C at leads 1–3 (e.g. KORD lead-1 1.19 vs 1.94). Kept only as a self-gating candidate.
  *Note:* the US-city deficit is **not** mean-forecast quality — so NBM didn't help.
- **CMA / BOM blend models** — rejected by the sweep.
- **Grid self-grading** — the source of the retired "127.5% ROI" fantasy (graded bets against the
  same grid it forecast from). Replaced by station-truth grading (roughly halved measured ROI).
- **`max(model, ensemble)` selection** — cherry-picked the optimistic model, manufactured edge.
  Replaced by averaging.

**Recently fixed bugs that were *hiding the truth* (not edge techniques, but they changed every
number):** voided range markets (~83% of US-city markets were never priced/graded), a backtest
look-ahead leak, over-dispersed σ (std-as-t-scale), grading↔pricing mismatch, dishonest optimizer
costs. See `docs/BUGFIX_EXECUTION_REPORT.md`.

---

## 5. The honest evaluation (how we decide if there's edge)

**Arbiter:** `evaluate_oos.py` — per-city **Brier** (model vs market vs ensemble), temperature
**CRPS** (paired MODEL-vs-ENSEMBLE on identical support), an explicit **EDGE CHECK**, and a
shrink-weight sweep. Backtests cross the half-spread and pay the fee.

**Pre-committed gate** (`data_status.py`, so it can't be moved post-hoc): **≥150 station-graded
markets AND ≥100 OOS bets.** Now **MET** (211 markets / 302 bets) — chiefly because fixing the
voided range markets pulled ~140 markets into the graded set.

**Result (gate met):**
| Predictor | Brier | Notes |
|---|---|---|
| Market | **0.128** | best |
| Model | 0.163 | beats ensemble, loses to market |
| Ensemble | 0.166 | baseline |

Paired CRPS: model **1.28** vs ensemble **1.33** (model better). ROI at production params:
**−20.2%** over 199 bets (52.8% win). Shrink sweep: **w=0** (pure market). Calibration table shows
the model is **overconfident at the extremes** (predicts ~0.04 where reality is ~0.19).

**Interpretation:** the model is a genuinely better *forecaster than the raw ensemble*, and the eval
is now trustworthy — but it is **not** a better predictor than the market, so there is **no
demonstrated edge**. To gain edge, improve the **model's accuracy**, not the bet sizing.

---

## 6. Where to look next for edge (promising / untested)

1. **Same-day intraday** (the biggest untested lever) — the model *can* see the day's observed temp
   (floor + per-hour regression), and the market prices this in, but the backtest can't test it
   (too few same-day snapshots). Live paper-trading same-day bets is the only way to know. The
   C4/C5 train/serve fixes just landed; their live value is unmeasured.
2. **Better US-city mean forecast** — NYC/Chicago are the worst cities and the deficit isn't NBM-
   fixable, so the gap is something else (micro-siting? the KLGA/KORD grid cell? a better blend or a
   bias term per synoptic regime?). This is where the model most needs to improve.
3. **Regime-conditional calibration** — the model is overconfident at the extremes; a heavier-tailed
   or regime-aware σ (e.g. wider on high-spread / frontal days) might close some of the Brier gap.
4. **Market microstructure** — we currently only forecast temperature. Edge might live in *timing*
   (stale prices, informed-volume signals α6/α8) rather than forecast accuracy — largely unexplored.

**Deferred code** (scoped in `docs/BUGFIX_EXECUTION_REPORT.md` §7, none affect accuracy): F8 engine
`_evaluate_bin` extraction, F6 leads-fetcher merge, F7 slug-migration remainder, F11/F14 efficiency.

---

## 7. Code map & how to run

Active code: `src/polymarket_weather/`. Run in-package commands from there; the optimizers and
tests from the repo root.

| File | Role |
|---|---|
| `resolution_anchors.py` | single source of truth for per-city resolution/truth/forecast anchors + `slug()` |
| `predictors/emos.py` | EMOS v2 calibrator (default), intraday conditioning, Tmin |
| `predictors/ensemble.py` | ensemble Student-t predictor (fallback + baseline) |
| `pmf.py` | question parsing, Student-t CDF/`_t_scale`, bin/condition probabilities, PMF reconstruction |
| `engine.py` | pricing, edge, gating, Kelly + caps, the betting loop |
| `backtest_common.py` | shared honest settlement / Kelly / caps (used by optimizer/simulator/backtester) |
| `train_calibrator.py` / `train_intraday.py` | fit EMOS v2 / intraday params → `models/*.json` |
| `evaluate_oos.py` | the arbiter (Brier, CRPS, EDGE CHECK, shrink sweep) |
| `data_status.py` | the pre-committed gate |

```bash
# regenerate everything + read the verdict (needs network; pause the launchd collectors first):
python backfill_schema.py                 # from src/polymarket_weather/ — repair CSV schema
./scripts/raincheck_validate.sh            # truth → leads → train → regenerate trackers → eval → gate
python data_status.py ; python evaluate_oos.py

pytest -o addopts="" tests/ -v             # from repo root (53 tests)
```

**Discipline:** no ROI/win-rate claim until the gate is met (it now is) AND model Brier drops below
market Brier (it has not). Pre-fix and post-fix metrics are non-comparable. All recent work is on
branch `megaplan-execution` (not pushed).
