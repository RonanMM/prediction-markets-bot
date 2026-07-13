# Bug-Fix Execution Report (2026-07)

What was done, why, and what it produced. Companion to `STATUS.md` (the plain-English verdict)
and `docs/EDGE_MEGAPLAN.md` (the edge strategy built on top of this pass). The original plan
document (`docs/BUGFIX_MEGAPLAN.md`) was removed 2026-07-13 once fully executed — see git history. All code landed on branch `megaplan-execution`, one
commit per phase, suite green after each (**50 unit tests pass**).

---

## 1. How this started

A whole-repo `/code-review` surfaced ~25 verified bugs. They were triaged into a 9-phase,
dependency-ordered remediation plan (BUGFIX_MEGAPLAN, since removed — git history) with an adversarial review that
added 18 corrections. This report covers executing that plan and running the honest evaluation.

The central discipline: several bugs were **hiding the real answer**, so nothing downstream was
trustworthy until the data, calibration, market coverage, economics, and engine were all fixed and
the evaluation was regenerated **once** from clean inputs.

---

## 2. What was implemented (committed, phases 0–6)

| Phase | Fixes | One-line effect |
|---|---|---|
| **0 Quick wins/safety** | F1, F3, F9, F10, B3 | restored TLS on the price feed; deleted the dead PyScaffold skeleton so `pytest` runs clean; killed dead code; fixed stale Meteostat comments; hardened the dedup key |
| **1 Append-freeze** | B1, B2, B4 | `_append_csv` now **unions** columns (a fetcher's new columns can't be silently dropped again); `ensure_schema` + `backfill_schema.py` repair the already-corrupted CSVs; serving warns instead of silently degrading |
| **2 Truth feed** | F2 | truth fetch keeps days with a valid MIN even when MAX is missing, and range-checks both series independently |
| **3 Calibration** | C1, C2, C3, C6, E4, D1 | **C1** the big one: `sigma` is a standard deviation but was used as the Student-t *scale*, over-dispersing every predictive distribution by ~40% — fixed via `_t_scale`; censoring applied to the ensemble gate + fallbacks; independent max/min ensemble validity + NaN guards; **D1** removed a backtest **look-ahead leak** |
| **4 Range bins** | A1–A5 | **"between X-Y°F" markets (~83% of US-city markets) are now priced AND graded** through one shared `resolves_yes_temp` condition spine; parse audit: 0 of 1372 committed questions fail to parse |
| **5 Engine/live** | E1, E3, E4, E5, E6, F4 | same-day bets no longer dropped; live re-verify re-shrinks toward the fresh price; NaN hygiene; loud predictor-failure logging; fixed the inert adjacent-bin filter; momentum reads the right series for Tmin markets |
| **6 Arbiter honesty** | D2, D3, D6, D7 | fixed the CRPS **chimera** (a Tmax row's floor spliced onto a Tmin row); added a **paired** MODEL-vs-ENSEMBLE CRPS over identical support; new `backtest_common.py` (honest settlement / Kelly / caps) with the optimizer rewired onto it |

Every phase added regression tests so each bug class can't silently return.

---

## 3. What was deferred (and why)

Not skipped — each is blocked on something that can't be done in a pure code pass, and the plan's
own review said so:

- **C4 / C5 (intraday hour-keying + lead-1 feature)** — change the served mean on same-day bets;
  the plan requires an offline holdout RMSE re-validation (and a lead-1 availability measurement)
  before shipping, which needs the training data / a run. Guardrail, not laziness.
- **E2 (target date from the question, not endDateIso)** — needs a date parser with year-wrap
  handling plus a per-city audit proving it's a no-op outside the ~32 Hong Kong rows; lowest-N bug.
- **D8 / D9 / D10 / D11** — mechanical rewires of `simulate_strategies.py`,
  `historical_backtester.py`, `optimizer_full.py`, and the tests helper onto `backtest_common`.
- **Phase 8** — behavior-preserving refactors (shared HTTP client, one Open-Meteo chunker, one slug
  helper, `_evaluate_bin` extraction); the plan sequences them after the regeneration.

---

## 4. The execution run (2026-07, this machine)

Steps taken to regenerate the evaluation from clean inputs:

1. **Paused both local collectors** — `com.raincheck.collect` (every 2h) and
   `com.raincheck.truth-eval` (daily 07:00) launchd agents — so they couldn't append mid-run.
   (These are LOCAL launchd agents, not the GitHub Action of the same name.) Reloaded afterward.
2. **Backed up** `data/` and `output/` to `/tmp/raincheck-backup`.
3. **`backfill_schema.py`** — widened the frozen `daily_mm`/`ensemble` headers and re-fetched the
   last 7 days. Verified on disk: `london_daily_mm.csv` now carries all 7 models × (tmax/tmin)
   including gem/mf/aifs (111+ populated rows), and the ensemble files carry `ens_min_mean/std`.
4. **`scripts/raincheck_validate.sh`** — the full honest loop: station truth → per-lead archived
   forecasts (single + multi-model) → retrain per-lead EMOS v2 → regenerate BOTH eval trackers
   (calibrated + pure-ensemble) → `evaluate_oos.py` → `data_status.py`. Exit 0.

**Honesty limit carried forward:** backtest rows over the append-freeze-corrupted window
(~Jul 3–8) still serve the proxy input, because the as-of join (`fetched_at <= fetch_time`)
correctly refuses to use runs re-fetched *today*. The per-snapshot forecast history for that window
is permanently lost; only forward data gets the full blend. So the model's *current* calibration is
marginally better than this backtest shows — not enough to change the verdict.

---

## 5. The result — gate MET, no edge over the market

The range-bin fix pulled ~140 previously-voided markets into the graded set, so the pre-committed
gate is finally satisfied:

```
gradable markets   211 / 150   [MET]
gradable bets      302 / 100   [MET]
```

`evaluate_oos.py` (Brier, lower = better):

| Predictor | Brier |
|---|---|
| Market price | **0.128** |
| Model (rebuilt + fixed) | 0.163 |
| Ensemble baseline | 0.166 |

- **Model beats the ensemble** — Brier 0.163 < 0.166, and the paired CRPS is 1.28 vs 1.33 (model
  better). The model stack is a genuinely better *forecaster* than the raw ensemble.
- **Model does NOT beat the market** — 0.163 > 0.128. The shrink-weight sweep recommends **w = 0**
  (pure market beats pure model), and betting the model at production sizing returns **−20.2% ROI
  over 199 bets** (52.8% win rate). The calibration table shows the model is overconfident at the
  extremes (predicts ~0.04 where reality is ~0.19).

**Verdict:** with the gate met and the evaluation trustworthy, there is **no demonstrated edge over
Polymarket**. The fixes didn't create edge — they removed the bugs that were hiding the truth, and
the truth is that the market is the better predictor. Improving this means improving the *model's
accuracy* (the deferred intraday / National-Blend work), not the bet sizing. Don't bet live.

---

## 6. What was committed

- **Code** (phases 0–6): 8 commits on `megaplan-execution`.
- **Docs**: this report + updated `STATUS.md` verdict (+ the since-removed plan's status table).
- **Regenerated data**: repaired/backfilled `data/weather/*` (daily_mm, ensemble, truth, leads),
  retrained `models/*_emos.json`, and the regenerated eval trackers in `output/`
  (`opportunities_evaluation_{calibrated,ensemble}.csv`, `opportunities_v4.csv`) + plots.

## 7. Follow-up session — deferred code landed

A second pass cleared most of §3's deferrals (one commit each, suite green throughout, now
**53 unit tests**):

- **C4 / C5** — intraday train/serve consistency (apply each per-hour fit to the last *completed*
  hour; feed the day-ahead lead-1 forecast the fit was trained on). Serving-side, no retrain.
- **E2** — target date from the question's named date (with year-wrap inference), audited to be a
  no-op for every city except the 32 Hong Kong rows.
- **D8 / D9 / D11** — `simulate_strategies`, `historical_backtester` (group/portfolio caps), and
  `optimizer_full` rewired onto `backtest_common`. No forked settlement or floor/ceiling-blind
  pricing remains anywhere (grep-clean). **D10** did not exist (no forked settlement in tests).
- **Phase 8 (partial)** — F7 canonical `resolution_anchors.slug()` (adopted in the read/write
  path), F5 shared `http_util.get_json` (the three fetchers), F13 vectorized CRPS grid
  (bit-identical), F12 lazy matplotlib import for collect-only runs.

**Still open** (pure refactors — zero behaviour change, deliberately left for a focused session
because they need heavy characterization fixtures to change safely):
- **F8** — extract the triplicated per-bin eval block in `engine.analyse_city` into one
  `_evaluate_bin` helper; requires a `test_analyse_city_unchanged` characterization test.
- **F6** — collapse the 4 diverged Open-Meteo previous-runs leads fetchers into one parameterized
  chunker (they differ in backoff/timeout/chunk-size/column naming).
- **F7 remainder** — migrate the other ~9 ad-hoc slug call sites onto `slug()` (identical output
  today).
- **F11 / F14** — efficiency: reduce the Gamma full-list pagination fan-out; avoid per-bin
  re-filtering of snapshot history in `market_staleness` (depends on F8).

The network follow-ups remain one-command re-runs (`backfill_schema.py`, `raincheck_validate.sh`).
The committed evaluation reflects the phase 0–6 fixes; C4/C5 affect live serving (not the sparse
intraday backtest) and E2 only moves the 32 Hong Kong rows, so the verdict is unchanged. No
ROI/win-rate claim beyond "no edge" until model Brier drops below market Brier on the graded set.
