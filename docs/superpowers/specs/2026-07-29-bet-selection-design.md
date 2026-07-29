# Bet selection for positive ROI — design

**Date:** 2026-07-29
**Status:** approved, not yet implemented
**Implementation scope:** Phase A only (§4–§7, §9). Phase C (§8) is specified here so its
terms are pre-registered rather than invented after A fails, but it is a separate build and
must not be pulled into A's plan.
**Goal:** find a subset of the model's opportunities that is genuinely profitable, validated
out-of-sample, without repeating the in-sample mistakes this project has already made.

---

## 1. Why this and not a better model

The forecasting thread is closed. Pooled over 401 gradable markets and 171 city-days, paired per
market and clustered by city-day:

```
model − market Brier gap   +0.0211   95% CI [+0.0068, +0.0353]   t = 2.90
```

The interval sits entirely above zero: the model is measurably *worse* than the market, not
merely unproven. Decomposed:

```
market 0.1158  →  ensemble 0.1360  →  model 0.1417
market→ensemble  +0.0202   (78% of the deficit — raw NWP, upstream of anything we build)
ensemble→model   +0.0057   (22% — our calibration layer, currently negative value)
```

Improving the forecast means attacking the 22% we control, where our contribution is already
harmful. Twelve approaches over four months have not beaten the raw ensemble.

**But ROI is not Brier.** Average accuracy across all markets is a different question from being
right on the specific bets we choose to place. That gap is the only reading of "positive ROI"
that the Brier evidence does not already rule out, so it is what this spec pursues.

## 2. What we are actually up against

Measured 2026-07-29 on the production bet set (`abs_edge ≥ MIN_EDGE`, `kelly > 0`), clustered
bootstrap over city-days.

> **On the two market counts.** §1 quotes 401 markets / 171 city-days — the published pooled
> verdict. The tables below say 407 / 174. Both are correct: grading is applied at *read* time,
> and a local truth refresh on 2026-07-29 (`fetch_historical_truth.py` +
> `fetch_station_obs.py`) made six more markets gradable. The counts move daily. Whatever the
> count is when the code runs, the split date is what stays fixed.

| window | n | city-days | Kelly ROI | 95% CI | flat-stake ROI |
|---|---:|---:|---:|---|---:|
| train (< 2026-07-08) | 201 | 99 | −25.3% | [−36.6%, −16.1%] | −28.6% |
| held-out (≥ 2026-07-08) | 206 | 75 | +3.3% | [−23.7%, +21.9%] | −4.5% |
| all | 407 | 174 | −12.4% | [−26.9%, −3.0%] | −16.4% |

Three facts that shape the whole design:

1. **The current strategy loses significantly.** The full-sample interval is entirely below zero
   on both sizing schemes. This is not "unproven"; it is a measured loss. Win rate 51.4%.
2. **The held-out third's +3.3% is a sizing artifact.** Under equal stakes the same bets lose
   4.5%. Kelly weighting concentrated the sample into a few lucky bets.
3. **ROI cannot be the test statistic.** The held-out ROI interval is ~46 percentage points wide
   (50pp flat). It cannot distinguish +3% from −20%. A selector could double the money and this
   test would shrug.

Point 3 is the reason this design validates on Brier rather than ROI. "Profitable on a subset"
is exactly the claim "our probability beats the price on those markets" — a Brier question, and
Brier has power here where ROI does not.

## 3. Power — the bar, stated before searching

Held-out paired Brier gap, clustered by city-day:

| slice | n | clusters | se | MDE (z=1.96) |
|---|---:|---:|---:|---:|
| all held-out | 206 | 75 | 0.0089 | **0.0174** |
| selector keeps 50% | 103 | 58 | 0.0131 | **0.0256** |
| selector keeps 25% | 51 | 38 | 0.0169 | **0.0331** |

Current held-out gap: **+0.0064**.

A selector keeping half the bets must reach a gap of **−0.026 or better** — a swing of ~0.032.
Keeping a quarter needs −0.033. Marginal improvement will not clear this. Only a substantial,
genuine subset edge will register, and that is the honest expectation going in.

## 4. Protocol

**Split, frozen before any searching.** Chronological at `SPLIT_DATE = 2026-07-08`, partitioned
on city-day so no day straddles the boundary. Train 201 bets / 99 city-days; held-out 206 / 75.
Deterministic, no RNG.

**Discovery on train is unconstrained.** Search it as hard as we like. The multiplicity control
is not a correction factor — it is the held-out set. We pick exactly **one `(selector,
threshold)` pair** — not one family with its threshold still free — and test it **once** at
z=1.96. Every rule tried is logged, including the ones abandoned, so the record is honest even
though the count does not enter the test.

**The test is run at a stated data cutoff**, recorded. Held-out grows as markets grade; markets
settling after the test date are *forward* sample for stage 2, not more held-out. Without this,
"one shot" silently becomes "one shot per week".

**Nothing trades real money on the strength of this.** A held-out pass produces a pre-registered
candidate that must then clear its own forward gate, per existing project discipline.

## 5. Search space

Seven families, all confirmed present with usable counts on train:

| signal | train spread | rationale |
|---|---|---|
| `forecast_prob` floor | median 0.138 | **theory-driven**: the model's [0,0.1) bin predicts 3.6% and realizes 15.5%. Excluding its overconfident tail fixes a known defect rather than dredging. |
| `bet_side` | No 128 / Yes 73 | the market's cheap bins are honestly cheap — 0 of 64 markets priced under 10¢ landed. Betting No into them may be systematically wrong. |
| `forecast_sigma` | median 1.60 | model confidence |
| `liquidity` | median 1905 | the structure book already found thin books lose −0.064/contract as maker |
| `bucket` / `days_ahead` | 15 buckets | lead time |
| `pmf_sum_dev` | median 0.595 | market coherence |
| `volume_recency` | median 0.934 | informed flow |

### Deliberately excluded

- **`abs_edge` / edge magnitude.** Adverse selection measured at z-std 1.41 (`docs/EDGE_MEGAPLAN.md`
  §63: realized z on the flagged set has mean ≈ 0 but std 1.41 — our sigma is honest on average
  days and ~40% too small on the days we flag). The model is most wrong exactly where it disagrees
  most with the price, so the intuitive selector is the measured trap. Any design here must select
  on something other than disagreement size.
- **`is_stale` (22 on train) and intraday conditioning (12 on train).** Too thin to test. Worth
  recording that intraday conditioning — the model's one genuine informational advantage over
  the market — fires on only 12 of 201 training bets. That is itself a finding, and a reason
  the live edge may differ from the backtest.

## 6. Components

New module `src/polymarket_weather/bet_selection.py`, kept out of `evaluate_oos.py` so the
arbiter does not grow an experiment inside it.

**The leakage guard is structural, not disciplinary.** Two CLI entry points, and `--search`
never receives held-out rows:

```bash
python bet_selection.py --search               # train only
python bet_selection.py --validate <sel> <t>   # the one shot
```

| function | role |
|---|---|
| `split_frozen(df)` | chronological city-day split at `SPLIT_DATE`; deterministic |
| `SELECTORS` | frozen registry: 7 families × fixed threshold grid, pre-registered |
| `evaluate_selector(df, mask)` | paired Brier gap + clustered interval, kept-fraction, both ROI figures (reported, never gating) |
| `search_train(train)` | all selector×threshold combinations, ranked, **all logged including losers** |
| `validate_holdout(...)` | the one shot |

**Enforcing "one shot".** A code lock can be commented out, so instead every held-out evaluation
appends to `holdout_log.jsonl` — timestamp, selector, threshold, result, data cutoff.
Append-only and auditable. If that file accumulates twelve entries, any reader knows the p-value
is fiction. An indelible record beats a removable lock.

## 7. Decision rules

The rule that matters most is **when not to spend the held-out shot**:

> If the best train selector's gap is not already better than the projected held-out MDE
> (≈ −0.026 at 50% keep), stop and go to Phase C **without touching held-out**.

A train gap of −0.01 cannot be confirmed by a set that can only resolve −0.026. Testing it would
burn the one clean measurement available to learn nothing.

| outcome | action |
|---|---|
| no train selector reaches the MDE | log; held-out untouched; → Phase C |
| held-out passes (`gap + 1.96·se < 0`) | pre-register; forward gate; still no real money |
| held-out fails | log; → Phase C |

## 8. Phase C — widen the universe

Reached if A returns nothing. The eval tracker is **post-filter**: every row already passed
`MIN_EDGE ≥ 0.06` and `MIN_LIQUIDITY ≥ 1000` (verified — minimum `abs_edge` in the tracker is
0.0602). Selection can therefore only *tighten*, never redirect; markets the engine discards are
invisible to Phase A.

Phase C regenerates the tracker with `MIN_EDGE=0`, `MIN_LIQUIDITY=0` and repeats the protocol at
the same split date. Motivation: those thresholds were endorsed by grid searches that predate
the settlement-truth corrections, so they were optimized against a broken ruler.

Phase C's held-out set is a *different* set from A's — a fresh pre-registered shot, logged as
such, not a second attempt at the same one.

## 9. Testing

- `split_frozen` is deterministic; no city-day straddles the boundary
- no `condition_id` appears in both halves
- selectors are pure functions of the frame
- `evaluate_selector` clusters by city-day (the guard `stats_util` exists for)
- **the search path never touches held-out** — pass a sentinel-poisoned held-out frame and
  assert it is unread
- `validate_holdout` appends exactly one record per invocation

## 10. What would make us stop

Stated in advance, so it cannot be moved afterwards. Phase A fails if no train selector reaches
the MDE, or if the one held-out test does not clear `gap + 1.96·se < 0`. Phase C fails on the
same terms against its own held-out set. If both fail, the honest conclusion is that bet
selection on this model does not produce positive ROI, and the remaining live candidates are the
model-free structure books — not a thirteenth forecasting approach.
