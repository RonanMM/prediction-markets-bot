# Design — Breadth structure book (all Polymarket weather cities)

**Date:** 2026-07-23
**Status:** Approved design; ready for implementation plan
**Component:** new `src/polymarket_weather/shoulder_book_breadth.py` (+ `collect.yml` step)
**Author:** Claude (Opus 4.8) with Ronan

---

## 1. Motivation

Polymarket runs daily temperature books on **~51 cities** (both *highest* and *lowest*,
~11 bins each, ~$10M live liquidity). The bot tracks **5**, holding just **14%** of that
liquidity — the deepest books (Shanghai, Paris, Jeddah, Miami, Qingdao) are untracked.

The one live edge candidate is the **model-free structure book** (`shoulder_book.py`):
favorite-longshot-bias shoulder selling. Its pre-registered moderate-shoulder gate
(Leg 1b, `[10,25)¢`) accrues forward entries on the 5-city stream at ~2 bets/day (Seoul)
down to ~0.2/day (Chicago) — so it resolves in **weeks to quarters**.

The structure book needs **no forecast model and no weather-truth feed to run or grade**:
it trades on live prices (already available for every city) and settles on **Polymarket's
own resolved outcome** (the P&L ground truth for a model-free trade). Extending it to all
cities is therefore cheap and multiplies the forward sample ~10×, resolving the edge
question far faster and — if it holds — across ~10× the inventory.

This is **not** an edge claim. It starts a clean, separately-pre-registered forward
measurement across the full city universe.

## 2. Scope

- **New, standalone module.** No change to `shoulder_book.py`, its CSV, or its
  pre-registered gates — the 5-city stream stays pristine so the two measurements
  cross-validate.
- **Record all bands**, mirroring the 5-city book: Leg 1 shoulder-sell `[5,35)¢`
  (buy NO, pre-day), Leg 2 favorite-buy `[65,85)¢` (>12 h to end). The gate is the
  moderate-shoulder `[10,25)¢` refinement (Leg 1b), computed at report time.
- **All ~49 cities with a live book.** Paper book, no fill constraint → more forward
  data is strictly better. A liquidity filter applies only at go-live (real orders),
  not to validation.
- **No real orders. No edge claim.**

## 3. Architecture

### 3.1 Discovery (live Gamma, no stored snapshots for new cities)

`fetch_weather_events()` pages `GET /events?tag_slug=weather&active=true&closed=false`
(limit 100, offset paging, **`User-Agent` header required** — default urllib UA gets 403),
keeps events whose title matches `^(Highest|Lowest) temperature in (.+?) on (.+?)\??$`,
and yields per **bin-market**:

| field | source |
|---|---|
| `condition_id` | market `conditionId` |
| `market_id` | market `id` (**used for settlement lookup** — `/markets/{id}`) |
| `city`, `kind` (max/min), `date_str` | parsed from event title |
| `question` | market `question` |
| `yes` | `json.loads(outcomePrices)[0]` |
| `liquidity` | `liquidityNum` (fallback `liquidity`) |
| `end` | event/market `endDate` (UTC) |

### 3.2 Recording — `scan_and_record_breadth()`

Mirrors `shoulder_book.scan_and_record` but reads the **live event feed** instead of
per-city snapshot CSVs, and uses **tz-free, endDate-relative** day logic:

- `hours_to_end = (end - now_utc) / 1h`
- **pre-day** ⟺ `hours_to_end > 24` (conservative proxy for "before the local target day
  starts", avoiding a 49-city timezone table)
- **Leg 1 (shoulder, buy NO):** `(cid,"shoulder") not in known` AND pre-day AND
  `BAND_LO <= yes < BAND_HI` (`[0.05,0.35)`). Record `side="No"`,
  `entry_side_price = round(1-yes,4)`, `band = "core" if yes>=CORE_LO else "outer"`.
- **Leg 2 (favorite, buy YES):** `(cid,"favorite") not in known` AND
  `hours_to_end > FAV_MIN_HOURS_TO_END` (12) AND `FAV_LO <= yes < FAV_HI` (`[0.65,0.85)`).
  Record `side="Yes"`, `entry_side_price = round(yes,4)`.

Bands/constants (`BAND_LO/HI`, `CORE_LO`, `FAV_*`) are **imported from `shoulder_book`**
— single source of truth, no duplication.

Entries are **append-only, deduped on `(condition_id, leg)`**, written to
`output/shoulder_paper_breadth.csv`. Columns mirror the 5-city book **plus `market_id`**
and **`settled_outcome`** (see grading). Recording cost per cycle is one row per newly
qualifying market (hundreds/day max), so no snapshot-CSV bloat.

### 3.3 Grading — settlement, frozen once observed

`settlement_outcome(market_id) -> 1 | 0 | None`:
- `GET /markets/{market_id}`; if `closed` is true and `outcomePrices` is pinned
  (`max(prices) >= 1 - _PIN`, `_PIN=0.03`), return `1` if `prices[0] >= 1-_PIN` (YES won)
  else `0`. Otherwise `None` (not yet settled).
- Closed markets remain fetchable by `id` indefinitely (verified against an April-2024
  market), so there is no aging-out risk.

`grade_book()` fills `settled_outcome` **once, then freezes it** in the CSV (never
re-fetched once set — the terminal settlement can't change). `side_won =
(settled_outcome==1) == (side=="Yes")`; `net_edge = shoulder_book._net_edge(side_won,
entry_side_price)` — the **verified taker-fee model reused unchanged**
(`exec = side_price + HALF_SPREAD`, `fee = 0.05·p·(1−p)`).

### 3.4 Pre-registered breadth gate

```python
BREADTH_PREREG_DATE = "2026-07-23"        # forward clock (go-live)
GATE_MOD_BREADTH    = (80, 0.03)          # (min forward graded, min mean net taker $/share)
```

`shoulder_book.moderate_gate_stats` is **generalized to accept a `prereg_date` argument**
(default keeps `MOD_PREREG_DATE` — the existing 5-city behavior is unchanged) and reused
on the breadth graded frame with `BREADTH_PREREG_DATE`. Forward = entries with
`entered_at_utc >= BREADTH_PREREG_DATE`; the 5-city discovery sample cannot leak in because
this is a separate CSV populated only from today.

### 3.5 Report — `report_breadth()`

Prints, mirroring the 5-city report:
- total entries / graded / awaiting-settlement counts and city coverage;
- Leg 1 shoulder full `[5,35)` and core `[20,35)`: n, win-rate, mean net taker;
- Leg 2 favorite `[65,85)`: n, win-rate, mean net taker;
- **Leg 1b moderate `[10,25)` breadth gate**: context (all graded) + FORWARD
  (`entered >= BREADTH_PREREG_DATE`) `n/80` vs `+0.03`, PASS/pending.

### 3.6 Cloud wiring

`collect.yml` (every 2 h): after the existing 5-city collect + `shoulder_book` hook, add
one step running `scan_and_record_breadth()` and committing
`output/shoulder_paper_breadth.csv`. Grading/reporting runs in `truth-eval.yml` and the
dashboard build (read-time, like everything else). No new secrets, no schedule change.

## 4. Data-flow diagram

```
Gamma /events?tag_slug=weather (all cities, live)
        │  parse title + bins
        ▼
scan_and_record_breadth()  ── append-only, dedup (cid,leg) ─▶ output/shoulder_paper_breadth.csv
        ▲                                                              │
   every 2h (collect.yml)                                              │ read-time
                                                                       ▼
                                        grade_book(): /markets/{id} settlement (frozen)
                                                                       ▼
                                        report_breadth(): legs + Leg1b breadth gate
```

## 5. Testing

Add to `tests/test_polymarket_weather.py` (synthetic, no network — the discovery/settlement
HTTP calls are injected/monkeypatched):

1. **Recording band+dedup:** feed synthetic bin rows spanning bands and `hours_to_end`;
   assert shoulder recorded only for pre-day `[5,35)`, favorite only for `>12h` `[65,85)`,
   dedup on `(cid,leg)`.
2. **endDate day logic:** `hours_to_end<=24` row is NOT recorded as shoulder.
3. **Settlement parse:** `settlement_outcome` maps `["1","0"]→1`, `["0","1"]→0`,
   unpinned `["0.5","0.5"]→None`.
4. **Frozen settlement:** once `settled_outcome` is set, `grade_book` does not overwrite it.
5. **Breadth gate:** `moderate_gate_stats(graded, prereg_date=BREADTH_PREREG_DATE)` counts
   only forward in-band entries; PASS iff `n>=80 and mean_net>=0.03`; existing 5-city call
   (no `prereg_date` arg) is unchanged.

Run: `pytest -o addopts="" tests/ -v` from repo root.

## 6. Success criteria

- `report_breadth()` prints the breadth legs + a correctly-computed Leg 1b forward gate
  reading `n=0/80` (or the true small forward count) today — never inheriting the 5-city
  sample.
- A live `scan_and_record_breadth()` records entries across many untracked cities.
- `settlement_outcome` grades a genuinely closed market correctly.
- All existing tests still pass; the 5-city `shoulder_book` report is byte-identical
  (its `moderate_gate_stats` call unchanged).

## 7. Out of scope

- No change to `shoulder_book.py` behavior, CSV, or gates (only an additive `prereg_date`
  kwarg with an unchanged default).
- No forecast model, resolution anchors, or weather-truth feeds for new cities.
- No full market/price-history snapshots committed for new cities.
- No real orders, no liquidity-filtered go-live list, no edge claim — a forward
  measurement only.
