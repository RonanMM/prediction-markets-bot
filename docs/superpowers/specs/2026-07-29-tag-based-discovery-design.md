# Tag-based market discovery — design

**Date:** 2026-07-29
**Status:** approved, not yet implemented
**Goal:** stop losing ~97% of the weather markets Polymarket lists, by discovering them through the
`weather` tag instead of a volume-ranked scan of every active market.

---

## 1. The defect

Discovery calls `search_markets_by_query(f"{kw} {term}")`, which pages `GET /markets` ordered by
volume descending and filters client-side on whether the query appears in the question text.

Two independent faults compound.

### 1a. A hard volume ceiling, silently hit

`GET /markets` returns **422 Unprocessable Entity at offset 2100** (measured 2026-07-29). Only the
top ~2100 active markets by volume are reachable, ever. Weather markets are low-volume, so nearly
all of them sit below the line.

`_get` delegates to `http_util.get_json`, which returns `None` on any failure. The pager reads
`None` as "no more data" and stops. **A truncation is therefore indistinguishable from reaching the
last page** — the same silent-failure shape catalogued throughout this project. A 3% capture rate
looked healthy for months.

A market becomes visible only once its volume rises enough to cross the ceiling, which happens near
expiry. That directly explains:

- **1104 of 2483 collected markets (44%) have exactly one snapshot**; median 2; 88% have fewer
  than five.
- α8 (`signals.py:97`) returns `hours_since_move = 0.0, is_stale = False` when `len(hist) < 2` —
  the *maximum-freshness* reading. Absence of evidence is encoded as positive evidence of activity,
  for 44% of markets.
- α1 (momentum) and α7 (forecast convergence) need cross-snapshot history and mostly cannot compute.
- The `2d+` buckets are starved (`NYC|2d+` n=10, `Chicago|2d+` n=9): markets are least traded
  exactly when they are furthest from resolution, so we never see them early.
- The shoulder book's maker-fill detector needs *later* price observations to judge whether a
  resting order would have filled.

### 1b. Three of four keywords never match

`MARKET_KEYWORDS = ["highest temperature", "temperature in", "max temperature", "high temperature"]`
is joined to the city as `f"{kw} {term}"` and tested as a **literal substring**. Real questions read
`"Will the highest temperature in London be 22°C on July 29?"` — so `"highest temperature London"`
never matches, because of the intervening `"in"`. Only `"temperature in"` works. The other three page
all 2100 markets and return nothing.

### Measured, 2026-07-29

```
current  "temperature in London"       ->  2 markets
current  "highest temperature London"  ->  0 markets   (dead keyword)
weather tag                            -> 66 markets
```

Tag-wide: **216 open events / 1994 markets / 264 daily city-temperature markets** for our five
cities, in one pass.

**We are capturing roughly 3% of what is available.** This starves every alpha signal, every
sample-size gate, and the structure books — the only strategy in this repo with a live shot at
positive ROI (see `docs/EDGE_MEGAPLAN.md` §12: the model itself has none).

## 2. Approach

Replace discovery with tag enumeration; keep the query scan as a repaired fallback.

Rejected alternatives: fixing only the keyword bug (leaves the 2100 ceiling, which is the dominant
fault); adding historical backfill (Gamma serves current state, past prices come from the CLOB
endpoint already in use — a separate question from stopping the ongoing loss, and it would make this
change harder to verify).

**Scope decision (Ronan, 2026-07-29):** all weather-tagged markets for the existing five cities.
Not every weather-tagged city — most have no truth feed, so their markets cannot be graded and are
useless for anything model-related.

## 3. Architecture

Today discovery runs 4 keywords × 11 search_terms (Seoul 2, London 2, Chicago 2, NYC 3, Hong Kong 2)
= **44 full paginations per collect cycle**, each grinding through 2100 markets, three-quarters of
them matching nothing. The new design pages the tag **once** and partitions across all five cities.

```
_paged_events("weather")      one pagination, ~216 events
        ↓
flatten event["markets"]      ~1994 markets
        ↓
match_city(question)          CITIES[*]["search_terms"] + temperature regex
        ↓
{city: [market, ...]}         ~264 across 5 cities
        ↓
fetch_city_markets(city)      UNCHANGED downstream — same snapshot dicts,
                              same processing.py append + dedupe
```

The return shape is identical, so `processing.py`, the engine and every consumer are untouched.
Dedupe is already on `condition_id + fetched_at_utc`; more markets simply means more rows. Blast
radius is one module.

| function | responsibility |
|---|---|
| `_paged_events(tag_slug)` | page `/events`; **fail loudly on truncation** |
| `discover_by_tag(cities)` | flatten, match to city, return `{city: [market]}` |
| `match_city(question)` | the single place deciding "is this London?" — `New York` vs `NYC` live here |
| `fetch_city_markets(city)` | try tag; fall back to `search_markets_by_query` |

## 4. Failure handling

`_get` returns `None` for every failure, so the pager cannot ask *why* a page was empty. It infers:

```
page full (100)   → keep going
page short (<100) → legitimate end, stop quietly
None after full   → TRUNCATED at offset N — warn loudly, return partial
None on first     → endpoint unavailable → fall back to search
```

| condition | behaviour |
|---|---|
| tag endpoint unreachable | warn; fall back to `search_markets_by_query` for all cities |
| tag returns 0 for a city that had markets last cycle | warn loudly — a taxonomy change, not a quiet day |
| event missing `markets` key | skip, count, report the count |
| pagination truncated | warn with the offset reached; never silently treat as complete |

**The fallback stays, and is repaired.** Depending on a third-party tag with no backup is fragile —
if Polymarket retags these markets we would silently collect nothing. But the fallback is currently
3/4 broken by §1b, so this change must also repair it. **Requirement, stated unambiguously:** a
market matches when the keyword AND the city term each appear somewhere in the question,
independently and case-insensitively — NOT when the concatenation `f"{kw} {term}"` appears as one
substring. `"Will the highest temperature in London be 22°C on July 29?"` must match keyword
`"highest temperature"` with term `"London"`. A fallback that does not work is not a fallback.

## 5. Testing

- **`match_city` boundaries** — `"New York"` / `"NYC"` / `"new york"` map to one city;
  `"New Yorker"` must NOT match. Hong Kong with and without the space.
- **Tmin markets survive.** ~20% of markets settle on the daily minimum
  (`"Will the lowest temperature in London be 15°C or below…"`). A filter written around "highest"
  would silently drop them — and Tmin is already a market type this repo excluded once before.
- **The repaired fallback actually matches.** `"highest temperature London"` must find
  `"Will the highest temperature in London be 22°C on July 29?"`. Pins §1b so it cannot return.
- **Truncation is not silent.** Simulate `None` after a full page; assert a warning is emitted and
  the result is flagged partial. This is the specific defect that hid a 3% capture rate, so it gets
  a dedicated test.
- **Fallback fires** when the tag yields nothing.

## 6. Out of scope

No historical backfill. No per-cycle cap. No retry tuning. No change to `processing.py`, the engine,
or anything downstream of the returned snapshot dicts.

## 7. What success looks like

Per-cycle discovery for the five cities rises to ~264 markets, and snapshots per market rises from
a median of 2. The current per-cycle figure is **~10, extrapolated** — only London was measured
directly (2 markets), so the implementation should record the real before/after rather than trust
that estimate. The check is direct: run one collect cycle and compare
`groupby("condition_id").size()` against today's distribution (44% at exactly one snapshot).

Note that snapshot CSVs are committed perishable data, so the repository will grow substantially
faster. That is the accepted cost of the scope decision in §2.
