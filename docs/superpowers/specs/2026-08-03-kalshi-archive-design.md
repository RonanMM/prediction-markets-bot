# Kalshi archive + multi-city Polymarket capture — design

**Date:** 2026-08-03
**Status:** approved, ready for implementation planning
**Scope:** data layer only. No edge claim, no model, no trading.

---

## 1. Why

Kalshi lists daily-temperature markets whose books are 5–20× deeper than Polymarket's. That
makes Kalshi's price a genuinely *new* information source — not another read of the same
Open-Meteo runs we already consume — which is the project's own stated bar for revisiting
forecast alpha at all (`docs/EDGE_MEGAPLAN.md`; the model's forecasting thread is closed,
pooled paired Brier gap +0.0183, CI [+0.0045, +0.0321]).

The eventual question is:

> Does Kalshi's price, converted through a measured ruler function, predict the **Polymarket**
> outcome better than Polymarket's own price does?

That question cannot be asked yet, for a reason discovered while designing this: **the
Polymarket half of the pair does not exist.** We discover ~50 cities per collector cycle but
persist full snapshots for only five. This spec builds both halves of the data layer so the
question becomes askable in roughly six weeks. It deliberately does not attempt to answer it.

### What this is NOT

Not cross-venue arbitrage. Kalshi and Polymarket resolve on **different rulers** (below), so a
Kalshi leg does not hedge a Polymarket leg. Treating them as equivalent is basis risk, and it is
the same trap that produced seven broken rulers in this project. Nothing here trades.

---

## 2. Verified facts

Everything in this section was queried live on 2026-08-03 and is reproducible. Nothing is taken
from secondary summary.

### 2.1 The rulers differ, categorically

| venue | source | day boundary | sampling |
|---|---|---|---|
| **Polymarket** | wunderground.com | station-local calendar day | hourly METARs |
| **Kalshi** | NWS Climatological Report (Daily) | local **standard** time day | 1-minute ASOS |

Every Kalshi temperature market checked resolves on the CLI; every Polymarket one on
Wunderground. This is categorical, not per-city.

Measured consequence (from this repo's own archives, `{slug}_obs_hourly.csv` vs
`{slug}_historical_actuals.csv`, 2022-01-01 → 2026-07-28, whole °F):

| station | n days | P(CLI ≥ WU) | mean CLI − WU |
|---|---:|---:|---:|
| KLGA | 1,668 | 99.40 % | +0.66 °F |
| KORD | 1,670 | 99.40 % | +0.61 °F |

`WU_max ≤ CLI_max` almost surely, because hourly METARs are a strict subsample of the 1-minute
record. The relationship is **one-sided**, and with 2 °F bins a +0.6 °F shift moves ~25–30 % of
probability mass across an edge. Measured at KLGA/KORD only; **must be re-measured per station
before any use.**

### 2.2 Our five cities have zero usable overlap

| our city | our station / ruler | Kalshi | usable |
|---|---|---|---|
| NYC | KLGA / WU | Central Park (KNYC) / CLI | no — station *and* ruler differ |
| Chicago | KORD / WU | Midway (KMDW) / CLI | no — station *and* ruler differ |
| London, Seoul, Hong Kong | — | not listed | no |

### 2.3 The overlap set is SEVEN — exhaustively, not by inheritance

Kalshi lists **19 live CLI high-temperature series**; Polymarket lists **47 temperature cities**
with a Wunderground station. Every Kalshi series was resolved to an ICAO (retrying on empty
responses — two separate runs silently dropped Houston and Seattle to transient empties before
the retry was added) and intersected against Polymarket's stations:

| outcome | count | cities |
|---|---|---|
| **same station — usable** | **7** | Atlanta KATL, Austin KAUS, Houston KHOU, Los Angeles KLAX, Miami KMIA, San Francisco KSFO, Seattle KSEA |
| both venues, **different station** | 4 | Chicago (KMDW vs KORD), NYC (KNYC vs KLGA), Denver (KDEN vs KBKF), Dallas (KDFW vs KDAL) |
| Kalshi only, no Polymarket market | 8 | Boston, Washington DC, Las Vegas, Minneapolis, New Orleans, Oklahoma City, Phoenix, San Antonio |

The four different-station cities are **permanently excluded**, not deferred. Central Park vs
LaGuardia and Midway vs O'Hare routinely differ by several °F — park versus waterfront tarmac —
which is basis risk far larger than the ruler gap this project can correct for. Recording them
here so they are not re-proposed.

### 2.3.1 Verified station evidence for the seven

Polymarket station read from each market's `wunderground.com/history/daily/.../{ICAO}` URL;
Kalshi station read from `rules_secondary`. Independently confirmed, both sides, 2026-08-03:

| city | Polymarket (WU) | Kalshi (CLI) | Kalshi series | Kalshi station stated in | CLI product |
|---|---|---|---|---|---|
| Los Angeles | KLAX | KLAX | `KXHIGHLAX` | `rules_primary` — "Los Angeles Airport, CA" | not stated |
| Austin | KAUS | KAUS | `KXHIGHAUS` | `rules_primary` — "Austin Bergstrom" | not stated |
| Miami | KMIA | KMIA | `KXHIGHMIA` | `rules_primary` — "Miami International Airport" | not stated |
| Atlanta | KATL | KATL | `KXHIGHTATL` | `rules_secondary` — "Atlanta, GA" | CLIATL |
| Houston | KHOU | KHOU | `KXHIGHTHOU` | `rules_secondary` — "Houston-Hobby, TX" | CLIHOU |
| Seattle | KSEA | KSEA | `KXHIGHTSEA` | `rules_secondary` — "Seattle-Tacoma, WA" | CLISEA |
| San Francisco | KSFO | KSFO | `KXHIGHTSFO` | `rules_secondary` — "San Francisco Airport" | CLISFO |

Control case, confirming the method detects mismatches: Chicago reads KORD on Polymarket and
KMDW on Kalshi. Correctly excluded.

⚠️ **The station is stated in a DIFFERENT FIELD depending on the series generation.** The older
`KXHIGH*` series name the airport in `rules_primary` and state no CLI product code at all; the
newer `KXHIGHT*` series give only a bare city name in `rules_primary` and put the station and
product code in `rules_secondary`. **Neither field alone identifies the station across the set**,
which is why both are archived verbatim (§4.4, §3). Any future parser that reads only one field
will be silently wrong for half the cities — and "Houston" alone is ambiguous between Bush and
Hobby, which differ materially.

### 2.4 Kalshi API shape

Base host `https://api.elections.kalshi.com`, unauthenticated, no key, no account.

- `GET /trade-api/v2/series/?category=Climate and Weather` — 291 series, 34 high-temperature.
- `GET /trade-api/v2/markets?series_ticker=…&limit=…&cursor=…` — market objects.

Fields available per market, all captured: `ticker`, `event_ticker`, `title`,
`yes_bid_dollars`, `yes_ask_dollars`, `yes_bid_size_fp`, `yes_ask_size_fp`, `no_bid_dollars`,
`no_ask_dollars`, `last_price_dollars`, `previous_price_dollars`, `volume_fp`, `volume_24h_fp`,
`open_interest_fp`, `liquidity_dollars`, `floor_strike`, `strike_type`, `status`, `result`,
`close_time`, `expiration_time`, `rules_primary`, `rules_secondary`.

**Three hazards found while probing, each of which must be handled:**

1. **Series tickers rot.** Legacy tickers still enumerate but serve zero markets:
   `HIGHNY`, `HIGHCHI`, `HIGHAUS`, `HIGHMIA`, `KXHIGHHOU`, `KXHIGHOU`, `KXHOUHIGH` are all
   dead. Houston has four tickers of which only `KXHIGHTHOU` is live. Naming is inconsistent
   (`KXHIGHLAX` vs `KXHIGHTATL` vs `KXHOUHIGH`), and `HIGHNY` → `KXHIGHNY` shows a completed
   migration. A hardcoded ticker list would silently archive nothing.
2. ~~**Invalid JSON.** `rules_secondary` contains raw newline characters inside a JSON string.
   Python's `json.loads` rejects this by default; `strict=False` is required.~~
   **RETRACTED 2026-08-04 — this hazard is not real and never was.** It was recorded here as
   "verified live 2026-08-03" and propagated into the plan, `kalshi_series`'s module docstring
   and a test name. Re-verified against all seven series: every raw body parses with plain
   `json.loads`, no flags, and the decoded `rules_secondary` contains no literal newline —
   Kalshi escapes newlines as `\n`, which is valid JSON. (The likely origin of the
   misdiagnosis: the *decoded* `early_close_condition` and `rules_secondary` strings do contain
   newlines, which is what a correctly-escaped `\n` decodes TO.) `strict=False` is KEPT, because
   it only ever widens what parses and would absorb a genuine control character if one ever
   appeared — but it is a defensive guard against an unobserved condition, not a workaround for
   observed vendor behaviour.
3. **The station is only in `rules_secondary`.** `rules_primary` for four of the seven says
   just "Atlanta" / "Houston" / "Seattle" / "San Francisco". Houston in particular has two
   plausible stations (Bush vs Hobby) that differ materially; only `rules_secondary` names
   "Houston-Hobby, TX".

Every live series returned exactly 200 markets at `limit=200` — that is the page cap, not the
total. Pagination is required, with explicit truncation detection.

---

## 3. Architecture principles

Each principle exists to structurally prevent a specific defect this project has already
shipped. The shared signature of all of them: **a green run, a plausible number, and an error
that flattered us.**

| principle | prevents |
|---|---|
| **Store raw, derive late.** Archive vendor fields verbatim; never store an interpretation. | The Hong Kong ruler bug — `resolution_unit` stored the source's precision rather than the market's bin grid, so every HK bin graded NO (0/179) behind a passing 97 % audit. |
| **Discover, never hardcode.** Series enumerated from the API each cycle. | A renamed ticker archiving zero rows on a green run. `KXHIGHHOU` is *already* dead. |
| **Completeness is recorded data, not inference.** Persist requested-vs-received, pages walked, cursor exhaustion. | The Polymarket discovery ceiling — a 422 at offset 2100 read as "end of list", capturing ~3 % of markets. |
| **Absence is a value, never a sentinel.** | `check_orderbook_vwap` returning `1.0` for "no liquidity", making it indistinguishable from a real price of 1.0. |
| **Append-only. A partial fetch never replaces a complete file.** | The obs truncation — a failed year chunk was skipped and the survivors overwrote a complete file, flipping the project's headline verdict. |
| **Per-entity health, never aggregates alone.** | The settlement audit sitting at 97 % while one city was 100 % wrong in one direction. |
| **Separate namespace.** Kalshi writes only under `data/kalshi/`; no shared mutable state with Polymarket grading. | A new data source corrupting an existing, validated one. |

---

## 4. Components

### 4.0 ONE overlap registry drives BOTH venues

The entire value of this data layer is the **paired** comparison. A Kalshi city we do not also
capture on Polymarket is unusable, and a Polymarket capture city with no Kalshi counterpart is
equally pointless. Symmetry is not a convention to maintain — it is the product.

So the seven cities are declared **once**, and both venue fetchers derive their target list from
that single declaration:

```python
# resolution_anchors.py — one entry per overlap city, both venues' identifiers together
"Los Angeles": {
    "tier": "capture",
    "station_code": "KLAX",             # the SAME station on both venues — that is the point
    "resolution_url": "https://www.wunderground.com/history/daily/us/ca/los-angeles/KLAX",
    "resolution_unit": "whole °F",      # Polymarket's bin grid
    "kalshi_series": "KXHIGHLAX",       # None for modelled cities with no Kalshi counterpart
    ...
}
```

`config.CAPTURE_CITIES` is derived from `tier == "capture"`. The Kalshi fetcher iterates the
same set, keyed by `kalshi_series`. There is no second list to drift.

**Requirement:** a test asserts that the set of capture cities with a `kalshi_series` is exactly
the set captured on Polymarket. If someone adds a Kalshi series without the Polymarket side — or
removes one — the suite fails rather than quietly producing an unpairable dataset. This directly
encodes the design constraint that a one-sided capture is worthless.

### 4.1 City tiering — `resolution_anchors.py`, `config.py`

`CITIES` is consumed by twelve modules, several of which iterate it to fetch forecasts
(`fetch_weather`, `fetch_ensemble`) or to train (`train_calibrator` does
`for city in CITIES.keys()`). Adding capture-only cities to `CITIES` would silently pull
forecasts for cities we do not model and attempt EMOS training on cities with no archives.

Each entry in `RESOLUTION_ANCHORS` gains a `tier` field:

- `"modelled"` — the existing five. Forecast + ensemble + models + grading. Unchanged.
- `"capture"` — the seven new. Market snapshots + station truth only. **No forecasts, no
  ensembles, no models, no training.**

`config.CITIES` filters to `tier == "modelled"`, so it means exactly what it means today and no
existing consumer changes behaviour. A new `config.CAPTURE_CITIES` and `config.ALL_CITIES`
expose the wider sets to the paths that want them.

Entries missing a `tier` default to `"modelled"`, preserving current behaviour for anything not
explicitly migrated.

**Requirement:** a test asserts that every `tier == "capture"` city is absent from
`config.CITIES`, so a capture city leaking into a modelling path fails loudly.

### 4.2 `kalshi_series.py` — discovery and the health manifest

Responsibilities:

- Enumerate temperature series from `/series/?category=Climate and Weather`.
- Parse with `json.loads(text, strict=False)` — a defensive widening, not a required workaround
  (see hazard 2's retraction).
- Write `data/kalshi/series_manifest.csv`, append-only, one row per series per cycle:
  `fetched_at_utc, series_ticker, city, markets_returned, live_markets, truncated`
  (the column was named `title` and always held the CITY; renamed 2026-08-04 to match its content).

  `markets_returned` counts every market the paginated fetch produced. `live_markets` counts
  those with `status` in `{"active", "initialized"}` — i.e. tradeable or about to be. Both are
  recorded because they answer different questions: `markets_returned == 0` means the ticker is
  dead, whereas `live_markets == 0` with a positive `markets_returned` is the normal overnight
  state between listings and must not raise.

The manifest is the anti-rot device. A series that has previously served markets and now serves
zero is the alarm condition. **`live → 0` is an error, not an absence of news**, and is
reported loudly rather than logged at debug.

Filtering must not rely on substring heuristics alone: `KXHIGHTMIN` is *Minneapolis daily
**high** temperature*, not a minimum-temperature series. Selection is by explicit mapping from
the seven target cities to their verified live series ticker (§2.3), with the manifest recording
everything discovered so a new or renamed series for a target city is visible.

### 4.3 `fetch_kalshi.py` — market capture

- `fetch_series_markets(series_ticker) -> (markets, truncated)` — cursor pagination, returning
  the truncation flag explicitly, mirroring `fetch_polymarket._paged_events`. A page cap
  prevents runaway loops; hitting it sets `truncated=True` and warns.
- `summarize_market(market) -> dict` — pure, testable, no network. Extracts the captured fields.
- Absence is `None`, never a sentinel. A market with no bid yields `yes_bid = None`, not `0.0`.

### 4.3.1 Depth and history — capture everything, because it is unrecoverable

An earlier draft of this spec applied YAGNI and captured only what arrives free in the markets
response. **That was wrong, and the reason is specific: Kalshi serves market objects for only
~2 months.** Anything not taken now cannot be taken later, at any price. Snapshots accumulate
forward only; they can never recover the past. So the rule for Kalshi inverts the usual default.

Three streams, verified live 2026-08-03:

**(a) Market snapshots** — hourly, all fields listed in §2.4.

**(b) Order-book depth** — `GET /markets/{ticker}/orderbook`, hourly, per live market.
~7 cities × ~12 live markets ≈ 84 requests per cycle, trivial. Justified directly by the
Polymarket experience two days earlier: a mid price without a book is misleading, and reading
only one side of a two-sided market produced a confidently wrong executability figure
(71 % vs 27 %). Stored as a compact summary in the same shape as `fetch_orderbook.summarize_book`,
so both venues are analysed by one code path.

**(c) Hourly candlesticks — the backfill, and the time-critical piece.**
`GET /series/{s}/markets/{m}/candlesticks?period_interval=60`. Returns per hour: OHLC plus mean
price, **`yes_bid`, `yes_ask`**, `volume_fp`, `open_interest_fp` — i.e. a genuine bid/ask history,
not merely last trade.

`period_interval=1` (one minute) returns **HTTP 400** on multi-day windows, so hourly is the
practical granularity. It also matches our snapshot cadence, so nothing is lost on the paired
comparison, whose resolution is capped by the coarser Polymarket side regardless.

Sizing, measured on a real settled market (`KXHIGHNY-26JUL21-B79.5`, $181 k volume, 39 hourly
candles over its life): ~2 months × 7 cities × ~12 markets/day ≈ 5,000 markets × ~40 candles
≈ **200 k rows**, on the order of tens of MB. Comfortably committable.

Run as a **one-time initial backfill** over everything currently served, then a **daily top-up**
that fetches candles for markets which have settled since the last run — a market's candles stay
available while its market object is served, so each market is captured once, completely, after
its life ends.

⚠️ Query windows must bracket the market's **actual trading life**, not "the last N days". While
drafting this, querying a 7-day trailing window against a market that settled on 21 July returned
zero candles at every interval — indistinguishable from "no candles exist". Derive the window
from `open_time`/`close_time` on the market object.

### 4.4 Bin semantics — derived at read time, never at write

A market carries three representations of one threshold: `floor_strike: 82`,
`strike_type: "greater"`, and `yes_sub_title: "83° or above"`. The off-by-one between them is
precisely the Hong Kong bug's shape.

All three are stored verbatim. A read-time helper derives the bin, and **a test asserts the
derivation against the subtitle string across live fixtures.** If the structured fields and the
human-readable subtitle ever disagree, that is a loud failure rather than a silently wrong bin.

### 4.5 Polymarket capture for the seven cities

These cities are already *discovered* by tag-based discovery; only persistence is missing.
`save_market_snapshots` groups by city and writes `data/polymarket/{slug}_snapshots.csv`.
Making the seven capture-tier cities flow through the existing path requires the tier work in
§4.1 and nothing else structural. Order-book annotation (`fetch_orderbook`) applies to them
identically.

### 4.6 Station truth for the seven

`fetch_historical_truth.SOURCES` gains seven entries using the existing `cli` adapter, which
already takes an ICAO and fetches NWS CLI via IEM:

```python
"los_angeles":   ("cli", {"station": "KLAX"}),
"austin":        ("cli", {"station": "KAUS"}),
"atlanta":       ("cli", {"station": "KATL"}),
"houston":       ("cli", {"station": "KHOU"}),
"miami":         ("cli", {"station": "KMIA"}),
"seattle":       ("cli", {"station": "KSEA"}),
"san_francisco": ("cli", {"station": "KSFO"}),
```

This is the **Kalshi-side** ruler (CLI). The Polymarket-side ruler for these cities is
Wunderground, reconstructed from hourly METARs by `wu_truth.py`, which requires
`fetch_station_obs` coverage for the same seven stations. **Both rulers are archived
separately. Neither is converted at write time.** The conversion function is a later spec.

---

## 5. Storage

All under `data/kalshi/`, append-only CSV, deduped on read — matching the project's existing
convention.

| file | key | contents |
|---|---|---|
| `data/kalshi/{city_slug}_markets.csv` | `ticker + fetched_at_utc` | per-market snapshot rows (§4.3) |
| `data/kalshi/{city_slug}_books.csv` | `ticker + fetched_at_utc` | order-book summary (§4.3.1b) |
| `data/kalshi/{city_slug}_candles.csv` | `ticker + end_period_ts` | hourly OHLC/bid/ask history (§4.3.1c) |
| `data/kalshi/series_manifest.csv` | `series_ticker + fetched_at_utc` | discovery health per cycle |

`city_slug` is the same lowercase-underscore form the Polymarket side already uses, so the two
venues' files sit side by side and join on an obvious key: `los_angeles`, `austin`, `atlanta`,
`houston`, `miami`, `seattle`, `san_francisco`.

Per the two-tier data policy in `CLAUDE.md`: Kalshi market snapshots are **perishable** — a
price at 14:00 is gone forever, and Kalshi serves market objects only ~2 months back — so they
are **committed**. The series manifest is small and committed with them.

`processing._append_csv` already unions columns and NA-backfills on schema widening, so new
vendor fields persist rather than being silently dropped.

---

## 6. Error handling

| failure | behaviour |
|---|---|
| Series discovery request fails | Keep the existing manifest; skip the cycle; warn. Never write an empty manifest. |
| A target city's series serves zero markets but previously served some | **Loud error.** This is the ticker-rot signature. |
| Pagination hits the page cap | `truncated=True` recorded in the manifest and warned. Never treated as completion. |
| An unescaped control character in a JSON string (never observed — see hazard 2's retraction) | Parsed with `strict=False`. A genuine parse failure keeps existing data and warns. |
| A market lacks bid/ask | Fields are `None`. No sentinel. |
| Kalshi wholly unreachable | Polymarket collection is unaffected. Kalshi is additive and must never block the irreplaceable Polymarket snapshot. |
| Candle fetch returns zero for a market | Recorded with its requested window (§7.10), retried next run. Never written as "market had no trading". |
| Order-book fetch fails for one market | That market is un-annotated for the cycle; the rest proceed. Snapshots are never withheld because depth was unavailable. |
| A capture-tier city reaches a modelling path | Test failure (§4.1), not a runtime surprise. |

---

## 7. Testing

Every guard must be **mutation-tested** — the discipline that caught four real defects in the
week before this spec. A test that passes when its guard is deleted is not a test.

Required cases, each with its mutation:

1. **Truncation detection** — fails if truncation is treated as completion.
2. **Ticker rot** — fails if a previously-live series serving zero markets is treated as
   "no data today".
3. **Bin derivation** — fails if the derived bin is off by one against `yes_sub_title`.
4. **Tier isolation** — fails if a capture-tier city appears in `config.CITIES`.
5. **Sentinel-free absence** — fails if a missing bid is reported as `0.0` rather than `None`.
6. **Malformed JSON** — the real `rules_secondary` newline case parses.
7. **Venue symmetry (§4.0)** — fails if any city carries a `kalshi_series` without Polymarket
   capture, or is captured on Polymarket as an overlap city without a `kalshi_series`. An
   unpairable city is a bug, not a partial success.
8. **Retry on empty, not just on error** — fails if a series returning an empty market list is
   accepted as "no markets" without retrying. Two throwaway scripts written while drafting this
   spec silently dropped Houston and then Seattle to transient empty responses, each time
   producing a confident wrong overlap count (6 instead of 7). An empty response and a genuine
   absence are indistinguishable at the call site; only retry separates them.
9. **Candle window derived from the market's life** — fails if the request window is a trailing
   "last N days" rather than bracketing `open_time`/`close_time`. A trailing window against a
   market that settled outside it returns zero candles at every interval, which reads as "this
   market has no history" and would silently produce an empty backfill.
10. **Backfill completeness is recorded** — each market's candle fetch records the window
    requested and the candle count returned, so a market archived with zero candles is
    distinguishable from one never attempted. A backfill that quietly covers half the window is
    the obs-truncation failure in a new costume.

Network is faked in all unit tests via an injected session, following the pattern established
in `fetch_orderbook`'s tests.

---

## 8. Out of scope

Deferred to their own specs, each requiring weeks of accumulated data:

- **Ruler conversion function** — per-station CLI↔WU transfer, measured not assumed.
- **The paired test** — does transferred Kalshi price beat Polymarket's own price on Brier.
- **Any trading.** A UK-resident operator's access to either venue is unresolved and is a
  precondition for any execution work, not for data capture.

---

## 9. Open risks

1. **The ruler conversion is measured at two stations, not seven.** KLGA and KORD gave
   +0.66/+0.61 °F. The other seven must be measured before the transfer function is trusted;
   IEM serves both feeds for all of them.
2. **Bin edges float daily.** Station mapping is fixed, but the 2 °F bin edges move with the
   forecast, so alignment must be recomputed per day rather than assumed.
3. **The seven cities are new to us.** We have no history, no models and no prior grading for
   them. That is acceptable for capture, but no result from these cities should be pooled with
   the original five without an explicit decision.
4. **Kalshi's ~2-month market-object window** means backfill is bounded. Forward capture from
   the moment this ships is the only way to accumulate depth.
