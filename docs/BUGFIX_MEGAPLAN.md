# Raincheck Bug-Fix Mega-Plan

Remediation plan for the ~37 fixes (across the ~25 verified bugs) found in the whole-repo
`/code-review` run. Each bug was confirmed against the actual code and, where relevant, the
on-disk CSVs. The plan is a **9-phase, dependency-ordered sequence** (one PR per phase) plus a
data-backfill plan, a validation/regression-test harness, and the adversarial-review corrections
folded in (marked **⚑**).

## The one thing to internalize first

Several of these bugs independently invalidate every current per-city Brier/CRPS number and the
gate counts. So the plan is not "fix 37 things in any order" — it is: **repair the data (B) and
truth (F2), correct the calibration (C), restore market coverage (A), unify the economics (D), fix
the engine/live path (E) — then do ONE full regeneration (Phase 7)** and read the honest arbiter.
Per `CLAUDE.md`, **no ROI/win-rate claim until the gate (150 station-graded markets AND 100 OOS
bets) is met**; every regeneration invalidates prior numbers and they are non-comparable.

Hard ordering constraints (baked into the phases below):
- Cluster **B** (append-freeze + backfill) precedes any eval/optimizer re-run.
- Cluster **A** engine range-pricing (A2) and grading range branch (A5) **ship together** — one
  without the other corrupts the gate.
- The shared-helper extractions (D6, and the C/E/F spines) land before their consumers.
- **C1** (sigma→scale) changes calibration, so it precedes all re-validation.

---

## The backbone — 9 phases

### Phase 0 — Quick wins & safety (no money/eval-path behavior)
**Fixes:** F1, F3, F9, F10, B3
- **F1** remove `verify=False` + the `urllib3.disable_warnings` on the real-money Polymarket feed.
- **F3** delete the PyScaffold `src/raincheck/` skeleton + `tests/test_skeleton.py`, and strip
  `--cov raincheck` from `setup.cfg` so `pytest tests/ -v` runs clean **without** `-o addopts=""`.
- **F9** delete dead `fetch_historical_features.py` and `processing.load_ensemble`.
- **F10** fix stale Meteostat comments (`grading.py:5`, `data_status.py:68`).
- **B3** harden the dedup key (`_row_key`/`_existing_key_set` sharing one column list +
  stringification) so a missing dedup column can't silently append duplicates. Landing it here lets
  B1 consume the shared helpers.
- **Exit:** `pytest tests/ -v` green (no override after F3); `import raincheck` → ModuleNotFound;
  no `output/*.csv` change.

### Phase 1 — Append-freeze root fix + schema repair (Cluster B core)
**Fixes:** B1, B2, B4
- **B1** `_append_csv` reindexes new rows to the existing header → **union** old+new columns
  (existing-order first, then first-seen new), NA-backfill, atomic `tmp`+`os.replace` widening.
- **B2** one-shot `scripts/backfill_schema.py`: `ensure_schema()` widens each narrow CSV, then
  `fetch_forecast_multimodel(city, past_days=7)` + `fetch_ensemble(city, past_days=7)` re-fetch and
  append (new `fetched_at_utc`, so dedup never collides). See Data Backfill §.
- **B4** serving warns loudly (`_warn_once`) instead of silently degrading when the chosen
  `mm_mean`/Tmin inputs are missing; `load_daily_mm` coerces `tmin_` numeric.
- **⚑ Safety:** pause the GitHub Actions `collect` schedule (→ `workflow_dispatch`-only) during the
  migration+commit so the atomic rewrite can't race a 2-hourly append. Re-enable after commit.
- **Exit:** every `{slug}_daily_mm.csv` header ⊇ `tmax_gem/mf/aifs` + `tmin_*`; every
  `{slug}_ensemble.csv` ⊇ `ens_min_mean/ens_min_std`; `git diff` additive-only (row counts
  non-decreasing); collector re-enabled and next run appends cheaply.

### Phase 2 — Truth-feed integrity + backfill (F2)
**Fixes:** F2
- `dropna(subset=['temp_max_c','temp_min_c'], how='all')` (keep max-OR-min rows); range-check both
  series independently; keep-existing-on-bad-fetch invariant. Re-run `fetch_historical_truth.py`
  to rebuild `*_historical_actuals.csv`.
- **Exit:** per-city row counts non-decreasing; a known max-less HK day now has `temp_min_c`;
  `grading._truth('min')` returns min for those days.

### Phase 3 — Calibration correctness (Cluster C, serving boundary)
**Fixes:** C1, C2, C3, C4, C5, C6 **(+ ⚑ E4's `get_ensemble_params` guard pulled forward)**
- **C1** add `pmf._t_scale(sigma, nu) = sigma*sqrt((nu-2)/nu)` used **only** inside `_cdf`, so
  `sigma` is unambiguously a std everywhere and the Student-t is served at the correct scale; make
  the `nu>30` Gaussian branch continuous. **⚑** On `nu<=2`, warn/clamp (don't silently serve std as
  scale). Do **not** touch trainers or `ensemble.py:59` (already assume the conversion).
- **C2** pass `floor`/`ceiling` into the ensemble-side `f_prob_ens` in conflict gating (engine
  `:387`/`:460`) so both sides are censored.
- **C3** apply censoring at a shared layer so EMOS→Ensemble fallbacks are censored too.
- **C4** serve the intraday per-hour fit for the **last completed hour** (or retrain keyed on
  as-of-minute); **C5** feed the **same lead** the fit was trained on (lead-1), not `det_mu`.
- **C6** validate max and min ensemble stats **independently**.
- **⚑ C6+E4 merge:** both edit `get_ensemble_params`; merge into one predicate —
  *return None iff (max unusable) AND (min unusable)*; add the EnsemblePredictor NaN-max→NWP guard
  here. Leave only E4's engine-gate finite guards for Phase 5.
- **⚑ Hard gate (not prose):** record `evaluate_oos` recommended-`w` and per-city CRPS on the
  identical (pre-D1) tracker **before and after C1**; if `|Δw| > 0.2`, STOP and investigate a
  masked mean bias before proceeding. **⚑** Also snapshot the ensemble tracker after C1-only so the
  final EDGE CHECK movement is attributable to C1 vs D1.
- **⚑ C4+C5 re-validation:** the change is **live-only** (backtest rarely exercises intraday). On
  the intraday holdout, assert new-path RMSE ≤ old per city/hour; measure lead-1 availability
  before shipping C5 (if low, feed `mu_raw`).
- **Exit:** calibration tests green; CRPS continuity `<1e-3` across `nu≈30`; no ROI claim.

### Phase 4 — Range-bin coverage (Cluster A, atomic)
**Fixes:** A1, A3, A5, A2, A4
- **A1** carry `temp_lo`/`temp_hi` through `MarketBin`; `temp_lo_c`/`temp_hi_c` through
  `Opportunity` (so endpoints persist to the eval CSV).
- **A3** fix `_condition_prob` range to `_cdf(hi+hw) − _cdf(lo−hw)` (honor ±0.5°F rounding); the
  current `_cdf(hi)−_cdf(lo)` understates P by ~2×.
- **A5** delete `grading.resolves_yes`'s substring scan; delegate to a new
  `pmf.resolves_yes_temp(parsed, actual_native, unit, native_round)` — the single condition spine
  used by BOTH pricing and grading. Fixes the range grading landmine AND the
  `no more than`/`exceed`/`reach` direction bugs at once.
- **A2** widen the boundary loop to price `range` via `_condition_prob` (generalize, don't add a
  third loop). **⚑** Wire the **min-bin** ("lowest between X-Y°F") path too — the Tmin loop's
  `else: continue` and its `_condition_prob` call currently omit `temp_lo`/`temp_hi`.
- **A4** extend `reconstruct_pmf` range backbone (α5 coherence). **⚑** Step-5 must **replace**, not
  augment, the `_bin_prob`-on-midpoint mass for range bins (else overlapping ±0.5°C windows
  double-count).
- **⚑ A5 parse-coverage audit before landing:** run `parse_question` over every distinct question
  in the committed snapshots; assert **zero** return `None` (else A5's `exact==` fallback silently
  flips already-graded non-range rows). Keep the old scan for any that don't parse.
- **⚑ Fahrenheit threshold gap:** add `°F` lte/gte patterns (`no more than/at most …°F`,
  `at least/exceed/reach …°F`) to `parse_question` — today only Celsius `_RE_LTE_C_2` exists, so a
  `°F` threshold market is a latent grading landmine.
- **⚑ Ship A2+A5 atomically.** **Exit:** Chicago/NYC trackers show range rows; a hand-checked
  `between 62-63°F` prices `≈ _cdf(63.5°F+)−_cdf(61.5°F−)` and grades YES at 63°F.
- **Open:** confirm UMA rounding mode (banker's vs half-up) on a resolved range market.

### Phase 5 — Engine execution / live path (Cluster E)
**Fixes:** E4, E5, E1, E2, E3, E6 **(+ ⚑ F4 moved here)**
- **E1** delete the shadowing `_days_from_now` closure (use the clamped module-level fn) so same-day
  markets aren't dropped after 00:00 UTC.
- **E2** derive `target_date` from the question's named date (fallback `endDateIso`); **⚑** audit
  that this is a no-op outside the ~32 HK rows (prove `question_date == end_date_norm` for every
  other parseable row before flipping the join key).
- **E3** store `model_prob_raw` + shrink weight on the Opportunity; re-shrink toward the **fresh**
  live price (no-op at `SHRINK_WEIGHT=1.0`). **⚑** Populate `model_prob_raw` in the shared
  boundary-loop `_create_opportunity` (now serves gte/lte/**range** post-A2) and the min path.
- **E4** validate `ens_mean` finite; guard NaN edge/kelly before emitting an Opportunity.
- **E5** log+count predictor failures instead of `except Exception: continue`.
- **E6** fix the inert adjacent-bin dedupe (`<=` + spacing; derive unit from the bin, not
  substring `'f'`).
- **⚑ F4** parameterize α1 momentum (and convergence) by market **kind** so Tmin markets score off
  `temp_min_c`, not `temp_max_c`. Moved here (from Phase 8) so its Tmin-ranking change is captured
  in the single Phase-7 regeneration.
- **Exit:** engine/live tests green (network monkeypatched); same-day survives at `min_days=0.0`,
  excluded at `1.0`. No live run until this phase + Phase 3 are green.

### Phase 6 — Backtest/eval arbiter honesty (Cluster D)
**Fixes:** D6, D1, D2, D3, D7, D8, D9, D10, D11
- **D6** new `backtest_common.py`: `settle_bet()` (crosses `HALF_SPREAD`, pays `FEE_RATE`),
  `single_kelly()` (byte-for-byte `engine._kelly_size` incl. guards), `apply_caps()` (group→
  portfolio, matching `evaluate_oos._roi_at_production`), and a thin re-export of
  `pmf._cdf/_bin_prob/_condition_prob`. **⚑** Give `apply_caps` an optional adjacent-bin dedup
  matching the **E6-fixed** engine logic (default-on for tools meant to mirror live sizing) so the
  arbiters don't size a book the live engine would never place.
- **D1** delete the ensemble as-of **look-ahead** fallback (`ensemble.py:82-84`); return None when
  no row `<= fetch_time`.
- **D2** fix the CRPS **chimera**: select the real last **row** per `(city,target_date,kind)`.
  **⚑** Don't infer `kind` from the `sigma_source` substring `'min'` — add an explicit `kind`
  column at write time (`reports.opps_to_df`); a Tmin fallback source lacking `'min'` would
  re-chimera.
- **D3** intersect MODEL-vs-ENSEMBLE CRPS on identical `(city,date,kind)` supports (mirror the
  Brier logic).
- **D7/D8/D9/D10/D11** repoint `optimizer.py`, `simulate_strategies.py`,
  `historical_backtester.py`, the tests helper, and `optimizer_full.py` onto `backtest_common`;
  delete all forked `_cdf/_bin_prob` and every `(payout-size)*0.98` settlement.
- **Exit:** parity tests green; `grep` finds zero `*0.98` settlements and zero forked pmf copies.

### Phase 7 — Full regeneration + gate + honest re-validation
**Fixes:** A6 (regenerate range-void trackers, persist endpoints), F2 re-run (currency)
- `./scripts/raincheck_validate.sh` end-to-end: truth + per-lead archived forecasts →
  `train_calibrator.py` (per-lead EMOS v2; Tmin params; blend reflects gem/mf/aifs) + intraday/Tmin
  re-train → regenerate **both** trackers (calibrated shows range rows; ensemble rebuilt without
  look-ahead) → `evaluate_oos.py` (per-city Brier, **paired** CRPS, EDGE CHECK, shrink sweep) →
  `data_status.py` (gate counts).
- **⚑** Break the new gate count down by `(city, condition=='range', graded_non_null)` to confirm
  the increase is **gradable** range rows (resolved + truth present), not merely priced.
- **⚑** Update the `CLAUDE.md` market-type-coverage note; flag all prior numbers invalid;
  **no** pre-fix/post-fix ROI comparison.

### Phase 8 — De-risking refactors + optional efficiency (Cluster F remainder)
**Fixes:** F5, F6, F7, F8, F11, F12, F13, F14 (F4 moved to Phase 5)
- **F5** one `http_util.get_json` (after F1). **F6** one Open-Meteo previous-runs chunker (collapse
  the 4 diverged `fetch_historical_leads*` copies). **F7** canonical `resolution_anchors.slug`
  (depends F6). **F8** extract the triplicated bin-eval block into `engine._evaluate_bin` — **⚑**
  must run **after** A2/C2/E4/E6 so the characterization test (`test_analyse_city_unchanged`)
  captures the finalized branches, not stale ones.
- **F11–F14** deferrable efficiency (pagination fan-out, matplotlib import in collect-only,
  vectorize CRPS grid, per-bin staleness re-filter). **⚑ F14** must keep `market_staleness`'s
  `condition_id` param (add `prefiltered=None`, don't drop it).
- **Exit:** refactor + characterization tests green; leads fetchers reproduce byte-compatible CSV
  schemas.

---

## Full fix roster (37)

**A — market coverage / range bins**
- A1 (S) carry range endpoints through MarketBin/Opportunity
- A2 (M) price range bins in the engine [A1,A3]
- A3 (S) `_condition_prob` range ±0.5°F rounding
- A4 (M) `reconstruct_pmf` range backbone [A1,A3]
- A5 (M) unify grading on `parse_question`, add range branch, delete substring scan [A1]
- A6 (S) persist endpoints + regenerate range-void trackers [A1,A2,A5]

**B — data pipeline integrity**
- B1 (S) `_append_csv` union columns (fix column-freeze)
- B2 (M) backfill/rebuild truncated daily_mm + ensemble CSVs [B1]
- B3 (S) dedup key robust to missing column
- B4 (S) serving warns instead of silent proxy/None degrade [B1]

**C — calibration correctness**
- C1 (S) Student-t sigma is a std, not the scale (over-dispersion + nu>30 discontinuity)
- C2 (S) censor the ensemble side of conflict gating
- C3 (S) censor EMOS→Ensemble fallbacks
- C4 (S) intraday: serve last-completed-hour fit
- C5 (M) intraday: feed lead-1 forecast feature [C4]
- C6 (M) decouple max/min ensemble validity

**D — backtest/eval honesty**
- D6 (M) shared `backtest_common` (settlement, Kelly+caps, pmf)
- D1 (S) kill ensemble look-ahead as-of fallback
- D2 (S) fix CRPS Tmin/Tmax chimera
- D3 (S) paired MODEL-vs-ENSEMBLE CRPS supports [D2]
- D7 (M) optimizer onto shared helpers [D6]
- D8 (M) simulate_strategies onto shared helpers [D6]
- D9 (S) historical_backtester group/portfolio caps [D6]
- D10 (S) tests helper onto shared settlement [D6]
- D11 (S) optimizer_full onto shared settlement [D6]

**E — engine / live path**
- E1 (S) same-day filter clamp (unshadow `_days_from_now`)
- E2 (M) target_date from question date, not endDateIso
- E3 (M) live re-shrink toward fresh price
- E4 (S) NaN ens_mean/edge/kelly hygiene
- E5 (S) loud predictor-failure logging [E4]
- E6 (M) fix inert adjacent-bin dedupe + °F sniff

**F — security / robustness / cleanup / infra**
- F1 (S) restore TLS verification
- F2 (M) keep min-only truth days + backfill actuals
- F4 (S) α1 momentum by market kind *(scheduled in Phase 5)*
- F5 (S) unify HTTP retry helpers [F1]
- F6 (L) unify Open-Meteo previous-runs chunkers
- F7 (M) canonical slug helper [F6]
- F8 (M) extract `engine._evaluate_bin` [F4]
- F3 (S) delete skeleton + `--cov` addopts
- F9 (S) delete dead modules/functions
- F10 (S) fix stale Meteostat comments
- F11 (M, defer) reduce Gamma pagination fan-out
- F12 (S, defer) skip matplotlib import in collect-only
- F13 (S, defer) vectorize CRPS grid
- F14 (M, defer) avoid per-bin staleness re-filter [F8]

---

## Data backfill — three distinct on-disk corruptions

1. **daily_mm / ensemble schema-freeze (B):** widen headers via `ensure_schema` (atomic
   `tmp`+`os.replace`), then `past_days=7` re-fetch (cap 92) appended with today's `fetched_at_utc`
   (additive, dedup-safe). **⚑ Honesty limit:** the per-snapshot *sequence* of gem/mf/aifs/tmin
   revisions over the corrupted ~2026-07-03→07-08 window is permanently lost; historical rows keep
   NA for the new columns. **⚑** And because the eval as-of join uses `fetched_at <= fetch_time`,
   those corrupted-window snapshots stay **proxy-served even after backfill** (re-fetched runs are
   stamped today) — only forward/near-future data gets the full blend. Do not claim blend coverage
   over that window.
2. **historical_actuals truth (F2):** re-fetch 2015→now with max-OR-min rows; verify row counts
   non-decreasing before commit; clear grading's `lru_cache`. Open: confirm upstream still serves
   the oldest previously-dropped min-only days.
3. **eval trackers:** derived full-rewrite outputs, invalid for **five** independent reasons
   (range-void, frozen/proxy inputs, over-dispersed sigma, look-ahead+chimera, wrong target_date).
   **One** regeneration in Phase 7 after A/B/C/D/E land satisfies all five.

**Re-train note:** committed `*_emos.json` already encode `input='mm_mean'` with the full model set,
so training read the (uncorrupted) `historical_leads_mm` archive, not the frozen live file — serving
should light up once the schema is repaired; re-fit anyway in Phase 7 and confirm `london_emos.json`
`mm_models` is unchanged.

---

## Validation harness + regression tests

Run `pytest tests/ -v` from the repo root after every phase; run the full honest loop
(`raincheck_validate.sh` → `evaluate_oos.py` EDGE CHECK → `data_status.py`) after phases that touch
pricing/sizing/grading/eval. New regression tests so each bug class can't silently return, e.g.:
- **append-freeze:** `test_append_csv_unions_new_columns`, `…_preserves_old_columns`, `…_column_order_deterministic`
- **dedup:** `test_dedup_uses_full_key_even_when_col_missing`
- **sigma-scale:** `test_student_t_scale_is_std`, `test_cdf_gaussian_continuity`
- **look-ahead:** `test_get_ensemble_params_is_as_of_only`
- **CRPS chimera:** `test_mean_crps_no_chimera`, `test_crps_paired_support`
- **range end-to-end:** `test_condition_prob_range_honors_rounding`, `test_condition_prob_range_tiles_to_one`,
  `test_engine_prices_range_bins`, `test_engine_prices_min_range_bin`, `test_resolves_yes_range`,
  `test_all_tracker_questions_parse` (⚑ parse-coverage guard)
- **same-day / date:** `test_days_from_now_clamps_same_day`, `test_engine_uses_question_date_over_enddate`
- **live re-shrink:** `test_live_reshrink_toward_fresh_price`, `test_live_reshrink_range_bin`
- **NaN hygiene / loud failure:** `test_edge_gate_skips_non_finite`, `test_predictor_failure_is_logged_and_counted`
- **economics parity:** `test_settle_bet_matches_arbiter`, `test_simulator_kelly_respects_config_cap`,
  `test_backtester_applies_portfolio_cap`
- **safety/hygiene:** `test_fetch_polymarket_get_verifies_tls`, `test_slug_canonical`, `test_truth_keeps_min_only_rows`

---

## Corrections folded in from the adversarial review (⚑)

1. **A5 is broader than range** — audit `parse_question` coverage over committed questions before
   landing; keep the old scan for any that don't parse (else silent non-range outcome flips).
2. **Min-bin range path** ("lowest between X-Y°F") must be wired in A2/A4 (currently voided).
3. **A3/A4 double-count** — Step-5 must replace, not augment, midpoint `_bin_prob` mass.
4. **C1 hard gate** on `|Δw|` (stop if >0.2); **nu≤2** must warn/clamp, not silently mis-serve.
5. **Fahrenheit lte/gte regex** missing in `parse_question` — add it as part of A5's spine.
6. **B2 corrupted-window** stays proxy-served post-backfill — acknowledge, don't over-claim.
7. **D6 apply_caps + E6 dedup** divergence — add dedup to `apply_caps`.
8. **C1 vs D1 confounded EDGE CHECK** — snapshot the ensemble tracker after C1-only and after D1.
9. **E2 fallback audit** across all cities (prove no-op outside HK).
10. **D2 kind** from an explicit column, not a `sigma_source` substring.
11. **F14** keep `market_staleness(condition_id=…)`; add `prefiltered=`.
12. **F4 → Phase 5** (avoid a second regeneration).
13. **C6+E4** merge into one atomic `get_ensemble_params` edit in Phase 3.
14. **A6 gate count** broken down by gradable range rows, not merely priced.
15. **F8** extraction only after A2/C2/E4/E6 (characterization test guards it).

---

## Risk / rollback / safety

- One PR per phase (bisectable); feature branches off `master`, never commit to `master` directly.
- **Critical:** disable the CI `collect` schedule during the B2 migration; B1 widening is
  crash-safe (`tmp`+`os.replace`).
- git-tag/copy `data/weather/*.csv` and `output/opportunities_*.csv` before Phases 1/2/7; verify
  additive-only diffs before committing any backfill.
- E3 is a no-op at default `SHRINK_WEIGHT=1.0`; never ship A2 without A5; roll back in reverse phase
  order.
- **Gate discipline as a safety rail:** no ROI/win-rate claim until `data_status.py` reports both
  gates met; treat pre/post-fix metrics as non-comparable.
- No `--live` (non-dry-run) until Phase 3 + Phase 5 are green.

## Open decisions for you
- UMA rounding mode (banker's vs half-up) on resolved range markets — confirm before trusting
  boundary-day range bets.
- `apply_caps` dedup: match the fixed engine exactly (recommended) vs leave caps-only.
- Whether to run C5 (lead-1 intraday feature) now given possibly-low lead-1 availability, or feed
  `mu_raw` instead.
