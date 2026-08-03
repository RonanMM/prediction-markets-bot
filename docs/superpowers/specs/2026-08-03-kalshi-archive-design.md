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

### 2.3 Seven cities ARE same-station — verified from both APIs

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
2. **Invalid JSON.** `rules_secondary` contains raw newline characters inside a JSON string.
   Python's `json.loads` rejects this by default; `strict=False` is required.
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
- Parse with `json.loads(text, strict=False)`.
- Write `data/kalshi/series_manifest.csv`, append-only, one row per series per cycle:
  `fetched_at_utc, series_ticker, title, markets_returned, live_markets, truncated`.

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

**No per-market orderbook or candlestick calls.** Top-of-book arrives free in the markets
response; full depth ladders are one extra request per market per cycle for a question that
cannot yet be asked. YAGNI — easy to add later, and the module boundary makes it a local change.

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
| `data/kalshi/{city_slug}_markets.csv` | `ticker + fetched_at_utc` | per-market snapshot rows |
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
| Invalid JSON (raw newline) | Parsed with `strict=False`. A genuine parse failure keeps existing data and warns. |
| A market lacks bid/ask | Fields are `None`. No sentinel. |
| Kalshi wholly unreachable | Polymarket collection is unaffected. Kalshi is additive and must never block the irreplaceable Polymarket snapshot. |
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

Network is faked in all unit tests via an injected session, following the pattern established
in `fetch_orderbook`'s tests.

---

## 8. Out of scope

Deferred to their own specs, each requiring weeks of accumulated data:

- **Ruler conversion function** — per-station CLI↔WU transfer, measured not assumed.
- **The paired test** — does transferred Kalshi price beat Polymarket's own price on Brier.
- **Order-book depth and candlesticks** from Kalshi.
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
