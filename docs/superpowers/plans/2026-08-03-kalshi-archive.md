# Kalshi Archive + Multi-City Polymarket Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive Kalshi's daily-temperature markets alongside Polymarket for the seven cities where both venues resolve on the same weather station, so a paired comparison becomes possible in ~6 weeks.

**Architecture:** A `tier` field on each resolution anchor splits cities into `modelled` (the existing five — forecasts, ensembles, training) and `capture` (the seven new — market snapshots and station truth only). `config.CITIES` keeps meaning "modelled", so none of its twelve consumers change behaviour; a new `ALL_CITIES` view feeds discovery and capture. Kalshi gets two new modules in its own namespace under `data/kalshi/`, and cannot affect Polymarket grading.

**Tech Stack:** Python 3.11, pandas, requests. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-08-03-kalshi-archive-design.md`. These bind every task.

- **Data layer only.** No model, no edge claim, no trading. Nothing in this plan may bet or size.
- **Kalshi base host:** `https://api.elections.kalshi.com`, unauthenticated, no key, no account.
- **Kalshi JSON MUST be parsed with `json.loads(text, strict=False)`.** `rules_secondary` contains raw newline characters inside a JSON string, which Python's parser rejects by default.
- **Store raw, derive late.** Archive vendor fields verbatim, including BOTH `rules_primary` and `rules_secondary`. Never store an interpretation.
- **Absence is `None`, never a sentinel.** A missing bid is `None`, not `0.0` and not `1.0`.
- **Append-only.** A partial fetch must never replace a complete file.
- **Retry on EMPTY responses, not only on errors.** An empty response and a genuine absence are indistinguishable at the call site.
- **Candle windows must bracket `open_time`/`close_time`,** never a trailing "last N days".
- **The seven cities are declared once and drive BOTH venues.** A city with a `kalshi_series` but no Polymarket capture (or the reverse) is a bug.
- **Nothing may break the existing five-city path.** The collector runs hourly in GitHub Actions and commits to master.
- **Use `resolution_anchors.slug()`** for every data-file slug. It is the single slug definition.
- Tests run from the repo root: `pytest -o addopts="" tests/ -v`. Baseline: **193 passing**.
- **Every guard must be mutation-tested.** Delete the guard, watch the test fail, restore. A test that passes without its guard is not a test.

### The seven cities — verified 2026-08-03, use verbatim, do NOT re-derive

| city | station (both venues) | Kalshi series | timezone |
|---|---|---|---|
| Los Angeles | KLAX | `KXHIGHLAX` | America/Los_Angeles |
| Austin | KAUS | `KXHIGHAUS` | America/Chicago |
| Atlanta | KATL | `KXHIGHTATL` | America/New_York |
| Houston | KHOU | `KXHIGHTHOU` | America/Chicago |
| Miami | KMIA | `KXHIGHMIA` | America/New_York |
| Seattle | KSEA | `KXHIGHTSEA` | America/Los_Angeles |
| San Francisco | KSFO | `KXHIGHTSFO` | America/Los_Angeles |

Polymarket resolution URLs (Wunderground, whole °F):
`https://www.wunderground.com/history/daily/us/{state}/{city}/{ICAO}` — `ca/los-angeles/KLAX`,
`tx/austin/KAUS`, `ga/atlanta/KATL`, `tx/houston/KHOU`, `fl/miami/KMIA`, `wa/seattle/KSEA`,
`ca/san-francisco/KSFO`.

---

## File Structure

| file | responsibility |
|---|---|
| `src/polymarket_weather/resolution_anchors.py` | MODIFY — add `tier` + `kalshi_series` to the 5 existing entries; add the 7 capture entries |
| `src/polymarket_weather/config.py` | MODIFY — `CITIES` filters to modelled; add `CAPTURE_CITIES`, `ALL_CITIES` |
| `src/polymarket_weather/fetch_polymarket.py` | MODIFY — `match_city`, `_matching_cities`, `discover_by_tag` iterate `ALL_CITIES` |
| `src/polymarket_weather/kalshi_series.py` | CREATE — series discovery + append-only health manifest |
| `src/polymarket_weather/fetch_kalshi.py` | CREATE — market capture, order books, candle backfill |
| `src/polymarket_weather/fetch_historical_truth.py` | MODIFY — 7 new `cli` sources |
| `src/polymarket_weather/fetch_station_obs.py` | MODIFY — 7 new METAR stations (Polymarket's WU-side ruler) |
| `src/polymarket_weather/main.py` | MODIFY — iterate `ALL_CITIES`; add `step_fetch_kalshi` |
| `tests/test_polymarket_weather.py` | MODIFY — the ten mandatory guards |

---

## Task 1: City tiering

**Files:**
- Modify: `src/polymarket_weather/resolution_anchors.py`
- Modify: `src/polymarket_weather/config.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `RESOLUTION_ANCHORS[city]["tier"]` → `"modelled"` | `"capture"`
  - `RESOLUTION_ANCHORS[city]["kalshi_series"]` → `str | None`
  - `config.CITIES` → modelled cities only (unchanged shape: `timezone`, `station_id`, `lat`, `lon`, `search_terms`)
  - `config.CAPTURE_CITIES` → capture cities only, same shape
  - `config.ALL_CITIES` → both, modelled first

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_capture_tier_cities_never_enter_config_cities():
    """CITIES is consumed by twelve modules, several of which iterate it to fetch forecasts
    or to TRAIN (train_calibrator does `for city in CITIES.keys()`). A capture-only city
    reaching those paths would pull forecasts we do not model and attempt EMOS training on
    cities with no archives — silently, on a green run."""
    import config
    from resolution_anchors import RESOLUTION_ANCHORS

    capture = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert capture, "expected capture-tier cities to exist"
    assert capture & set(config.CITIES) == set(), (
        f"capture-tier cities leaked into config.CITIES: {capture & set(config.CITIES)}")
    assert capture <= set(config.ALL_CITIES), "ALL_CITIES must contain every capture city"
    assert set(config.CITIES) <= set(config.ALL_CITIES)
    # The original five must still be modelled — this plan must not change their behaviour.
    for city in ("London", "Seoul", "Chicago", "New York City", "Hong Kong"):
        assert RESOLUTION_ANCHORS[city].get("tier", "modelled") == "modelled"
        assert city in config.CITIES


def test_venue_symmetry_kalshi_and_polymarket_cover_the_same_cities():
    """The entire value of this data layer is the PAIRED comparison. A Kalshi city we do not
    also capture on Polymarket is unusable, and vice versa. Symmetry is the product, not a
    convention to maintain."""
    import config
    from resolution_anchors import RESOLUTION_ANCHORS

    with_kalshi = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("kalshi_series")}
    capture = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert with_kalshi == capture, (
        f"unpairable cities — kalshi-only {with_kalshi - capture}, "
        f"polymarket-only {capture - with_kalshi}")
    assert len(with_kalshi) == 7, f"expected the 7 verified overlap cities, got {len(with_kalshi)}"
    # Every capture city must also be Polymarket-capturable: it needs search terms and a
    # Wunderground resolution URL naming the SAME station Kalshi reads.
    for city in capture:
        a = RESOLUTION_ANCHORS[city]
        assert a["station_code"] in a["resolution_url"], (
            f"{city}: resolution_url must name station {a['station_code']}")
        assert config.ALL_CITIES[city]["search_terms"], f"{city} has no search terms"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -o addopts="" tests/ -k "capture_tier or venue_symmetry" -v`
Expected: FAIL — `KeyError: 'ALL_CITIES'` / `assert capture` fails because no city has `tier`.

- [ ] **Step 3: Add `tier` and `kalshi_series` to the five existing anchors**

In `src/polymarket_weather/resolution_anchors.py`, add these two keys to each of the five
existing entries (`London`, `Seoul`, `Chicago`, `New York City`, `Hong Kong`):

```python
        "tier": "modelled",
        "kalshi_series": None,   # Kalshi lists NYC on Central Park and Chicago on Midway —
                                 # different stations from ours. See spec §2.2.
```

- [ ] **Step 4: Add the seven capture anchors**

Append inside `RESOLUTION_ANCHORS` in `src/polymarket_weather/resolution_anchors.py`:

```python
    # ── CAPTURE TIER (spec 2026-08-03) ───────────────────────────────────────────────────
    # Seven cities where Polymarket and Kalshi resolve on the SAME station and differ only in
    # the RULER (Wunderground hourly-METAR max vs NWS CLI 1-minute max). Verified live against
    # both APIs 2026-08-03. We capture prices and truth for them; we do NOT forecast or model
    # them — see config.CITIES, which deliberately excludes this tier.
    "Los Angeles": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/ca/los-angeles/KLAX",
        "resolution_unit": "whole °F",
        "forecast_lat": 33.9425, "forecast_lon": -118.4081,
        "station_code": "KLAX",
        "tier": "capture",
        "kalshi_series": "KXHIGHLAX",
    },
    "Austin": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/tx/austin/KAUS",
        "resolution_unit": "whole °F",
        "forecast_lat": 30.1975, "forecast_lon": -97.6664,
        "station_code": "KAUS",
        "tier": "capture",
        "kalshi_series": "KXHIGHAUS",
    },
    "Atlanta": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/ga/atlanta/KATL",
        "resolution_unit": "whole °F",
        "forecast_lat": 33.6407, "forecast_lon": -84.4277,
        "station_code": "KATL",
        "tier": "capture",
        "kalshi_series": "KXHIGHTATL",
    },
    "Houston": {
        # Kalshi reads Houston-HOBBY (KHOU), not Bush (KIAH) — stated only in rules_secondary.
        "resolution_url": "https://www.wunderground.com/history/daily/us/tx/houston/KHOU",
        "resolution_unit": "whole °F",
        "forecast_lat": 29.6454, "forecast_lon": -95.2789,
        "station_code": "KHOU",
        "tier": "capture",
        "kalshi_series": "KXHIGHTHOU",
    },
    "Miami": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/fl/miami/KMIA",
        "resolution_unit": "whole °F",
        "forecast_lat": 25.7932, "forecast_lon": -80.2906,
        "station_code": "KMIA",
        "tier": "capture",
        "kalshi_series": "KXHIGHMIA",
    },
    "Seattle": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/wa/seattle/KSEA",
        "resolution_unit": "whole °F",
        "forecast_lat": 47.4444, "forecast_lon": -122.3139,
        "station_code": "KSEA",
        "tier": "capture",
        "kalshi_series": "KXHIGHTSEA",
    },
    "San Francisco": {
        "resolution_url": "https://www.wunderground.com/history/daily/us/ca/san-francisco/KSFO",
        "resolution_unit": "whole °F",
        "forecast_lat": 37.6188, "forecast_lon": -122.3750,
        "station_code": "KSFO",
        "tier": "capture",
        "kalshi_series": "KXHIGHTSFO",
    },
```

- [ ] **Step 5: Split the city views in `config.py`**

In `src/polymarket_weather/config.py`, add the seven to `_CITY_META` (they need timezone and
search terms), then replace the single `CITIES` comprehension with three views:

```python
_CITY_META = {
    "Seoul":         {"timezone": ZoneInfo("Asia/Seoul"),       "search_terms": ["Seoul", "seoul"]},
    "London":        {"timezone": ZoneInfo("Europe/London"),    "search_terms": ["London", "london"]},
    "Chicago":       {"timezone": ZoneInfo("America/Chicago"),  "search_terms": ["Chicago", "chicago"]},
    "New York City": {"timezone": ZoneInfo("America/New_York"), "search_terms": ["New York", "NYC", "new york"]},
    "Hong Kong":     {"timezone": ZoneInfo("Asia/Hong_Kong"),   "search_terms": ["Hong Kong", "hong kong"]},
    # capture tier — see resolution_anchors.py
    "Los Angeles":   {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["Los Angeles", "los angeles"]},
    "Austin":        {"timezone": ZoneInfo("America/Chicago"),     "search_terms": ["Austin", "austin"]},
    "Atlanta":       {"timezone": ZoneInfo("America/New_York"),    "search_terms": ["Atlanta", "atlanta"]},
    "Houston":       {"timezone": ZoneInfo("America/Chicago"),     "search_terms": ["Houston", "houston"]},
    "Miami":         {"timezone": ZoneInfo("America/New_York"),    "search_terms": ["Miami", "miami"]},
    "Seattle":       {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["Seattle", "seattle"]},
    "San Francisco": {"timezone": ZoneInfo("America/Los_Angeles"), "search_terms": ["San Francisco", "san francisco"]},
}


def _city_view(tiers: tuple[str, ...]) -> dict:
    """Cities whose anchor tier is in *tiers*, in _CITY_META order."""
    return {
        city: {
            "timezone":     meta["timezone"],
            "station_id":   RESOLUTION_ANCHORS[city]["station_code"],
            "lat":          RESOLUTION_ANCHORS[city]["forecast_lat"],
            "lon":          RESOLUTION_ANCHORS[city]["forecast_lon"],
            "search_terms": meta["search_terms"],
        }
        for city, meta in _CITY_META.items()
        if RESOLUTION_ANCHORS[city].get("tier", "modelled") in tiers
    }


# ⚠️ CITIES MEANS MODELLED CITIES and must keep meaning exactly that. It is consumed by twelve
# modules, several of which iterate it to fetch forecasts (fetch_weather, fetch_ensemble) or to
# train (train_calibrator does `for city in CITIES.keys()`). Adding capture-only cities here
# would pull forecasts for cities we do not model and attempt EMOS training on cities with no
# archives — silently, on a green run. Use ALL_CITIES for discovery and capture instead.
CITIES         = _city_view(("modelled",))
CAPTURE_CITIES = _city_view(("capture",))
ALL_CITIES     = _city_view(("modelled", "capture"))
```

Also add the Kalshi data directory next to the existing ones at `config.py:61-63`
(`DATA_DIR`, `POLYMARKET_DIR`, `WEATHER_DIR`) — `processing.py` reads these, it has no
`_DATA_DIR` of its own:

```python
KALSHI_DIR        = "data/kalshi"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest -o addopts="" tests/ -k "capture_tier or venue_symmetry" -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Run the FULL suite — nothing may break the five-city path**

Run: `pytest -o addopts="" tests/ -q`
Expected: 195 passed (193 baseline + 2 new). If any pre-existing test fails, `CITIES` has
changed meaning — fix that before continuing.

- [ ] **Step 8: Mutation-test the tier guard**

Temporarily change `CITIES = _city_view(("modelled",))` to `_city_view(("modelled", "capture"))`.
Run: `pytest -o addopts="" tests/ -k capture_tier -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 9: Commit**

```bash
git add src/polymarket_weather/resolution_anchors.py src/polymarket_weather/config.py tests/test_polymarket_weather.py
git commit -m "config: tier cities into modelled vs capture; add the 7 same-station overlap cities

CITIES keeps meaning 'modelled' so none of its twelve consumers change behaviour. ALL_CITIES is
the new wider view for discovery and capture. Seven capture cities added with both venues'
identifiers in one place, so the two sides cannot drift.

Guards: capture cities can never appear in CITIES; kalshi_series and capture tier must cover
exactly the same set (an unpairable city is a bug, not a partial success)."
```

---

## Task 2: Polymarket capture for the seven cities

**Files:**
- Modify: `src/polymarket_weather/fetch_polymarket.py` (`match_city`, `_matching_cities`, `discover_by_tag`)
- Modify: `src/polymarket_weather/main.py` (`step_fetch_polymarket` caller)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `config.ALL_CITIES` from Task 1.
- Produces: `fetch_weather_markets("Los Angeles")` returns snapshots; `data/polymarket/los_angeles_snapshots.csv` is written by the existing `processing.save_market_snapshots`.

- [ ] **Step 1: Write the failing test**

```python
def test_discovery_matches_capture_tier_cities():
    """The seven capture cities are already DISCOVERED by tag — only persistence was missing.
    match_city and discover_by_tag iterate the city registry, so they must see ALL_CITIES, not
    just the modelled five, or the capture cities are found and then dropped on the floor."""
    from fetch_polymarket import match_city

    assert match_city("Highest temperature in Los Angeles on August 4?") == "Los Angeles"
    assert match_city("Highest temperature in Houston on August 4?") == "Houston"
    assert match_city("Highest temperature in San Francisco on August 4?") == "San Francisco"
    # Word-boundary matching must still hold for the new cities.
    assert match_city("Austin Powers trivia market") == "Austin"      # bare word does match
    assert match_city("Highest temperature in Austintown on August 4?") is None
    # The original five are unaffected.
    assert match_city("Highest temperature in London on August 4?") == "London"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -o addopts="" tests/ -k discovery_matches_capture -v`
Expected: FAIL — `match_city(... Los Angeles ...)` returns `None`.

- [ ] **Step 3: Switch the three discovery call sites to `ALL_CITIES`**

In `src/polymarket_weather/fetch_polymarket.py`, change the import and the three loops.

Import line — add `ALL_CITIES` alongside the existing `CITIES` import.

In `match_city`, replace `for city, cfg in CITIES.items():` with:

```python
    # ALL_CITIES, not CITIES: capture-tier cities must be discovered and filed too. CITIES is
    # deliberately modelled-only (see config.py) and using it here would find the seven overlap
    # cities and then silently drop them.
    for city, cfg in ALL_CITIES.items():
```

In `_matching_cities`, replace `for city, cfg in CITIES.items():` with `for city, cfg in ALL_CITIES.items():`.

In `discover_by_tag`, replace any `CITIES` reference used to enumerate target cities with `ALL_CITIES`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -o addopts="" tests/ -k discovery_matches_capture -v`
Expected: PASS

- [ ] **Step 5: Point the collector at every city**

In `src/polymarket_weather/main.py`, import `ALL_CITIES` (line 28 currently imports `CITIES`)
and change exactly two places:

- **line ~293** — the `--cities` argparse default: `default=list(CITIES.keys())` becomes
  `default=list(ALL_CITIES.keys())`
- **lines ~320-323** — the validation filter: both `c not in CITIES` and `c in CITIES` become
  `ALL_CITIES`

⚠️ **Do NOT touch the `if city not in CITIES: continue` guards inside `step_fetch_weather`
(line ~152) and `step_fetch_ensemble` (line ~211).** Those are load-bearing: they are what keeps
capture-tier cities out of the forecast and ensemble paths once the city list widens. Leaving
them referencing `CITIES` is the whole tiering mechanism working as designed — a capture city
flows into Polymarket snapshot capture and is skipped by every modelling step, automatically.

- [ ] **Step 5b: Prove the forecast steps skip capture cities**

```python
def test_forecast_steps_skip_capture_tier_cities(monkeypatch, caplog):
    """Widening the collector's city list must not widen the FORECAST paths. main's
    step_fetch_weather / step_fetch_ensemble guard on `city not in CITIES`, and CITIES is
    modelled-only — so a capture city is skipped without an API call."""
    import main
    called = []
    monkeypatch.setattr(main, "fetch_forecast", lambda c: called.append(c) or None)
    main.step_fetch_weather(["Los Angeles", "London"])
    assert called == ["London"], f"capture city reached the forecast fetcher: {called}"
```

- [ ] **Step 6: Verify end to end against the live API**

Run from `src/polymarket_weather/`:

```bash
python -c "
from fetch_polymarket import fetch_weather_markets
s = fetch_weather_markets('Los Angeles')
print('Los Angeles snapshots:', len(s))
assert s, 'expected live LA markets'
print(s[0]['question'][:70])
"
```
Expected: a non-zero count and a Los Angeles temperature question.

- [ ] **Step 7: Run the full suite**

Run: `pytest -o addopts="" tests/ -q`
Expected: 196 passed.

- [ ] **Step 8: Commit**

```bash
git add src/polymarket_weather/fetch_polymarket.py src/polymarket_weather/main.py tests/test_polymarket_weather.py
git commit -m "discovery: file capture-tier cities too (ALL_CITIES, not CITIES)

The seven overlap cities were already discovered by tag discovery and then dropped, because
match_city/_matching_cities/discover_by_tag iterate the modelled-only registry. Snapshots for
them now persist through the existing processing.save_market_snapshots path."
```

---

## Task 3: Kalshi series discovery and health manifest

**Files:**
- Create: `src/polymarket_weather/kalshi_series.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `resolution_anchors.RESOLUTION_ANCHORS` (for `kalshi_series`), `resolution_anchors.slug`.
- Produces:
  - `KALSHI_BASE: str` — `"https://api.elections.kalshi.com/trade-api/v2"`
  - `kalshi_get(path: str, params: dict, session=None, retries: int = 4, nonempty_key: str | None = None) -> tuple[dict | None, bool]`
  - `target_series() -> dict[str, str]` — `{city: series_ticker}` for capture cities
  - `manifest_row(series_ticker, title, markets_returned, live_markets, truncated, fetched_at_utc) -> dict`
  - `MANIFEST_COLS: list[str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_kalshi_get_retries_on_EMPTY_not_only_on_error():
    """An empty response and a genuine absence are indistinguishable at the call site.

    Two throwaway scripts written while drafting this spec silently dropped Houston, then
    Seattle, to transient empty results — each time producing a confident wrong overlap count
    of 6 instead of 7. Only retrying separates the two cases.
    """
    from kalshi_series import kalshi_get

    class FlakySession:
        def __init__(self): self.calls = 0
        def get(self, url, params=None, timeout=None):
            self.calls += 1
            body = '{"markets": []}' if self.calls < 3 else '{"markets": [{"ticker": "T1"}]}'
            return _Resp(body)

    s = FlakySession()
    payload, ok = kalshi_get("/markets", {}, session=s, nonempty_key="markets")
    assert ok is True
    assert payload["markets"] == [{"ticker": "T1"}], "must retry past the transient empties"
    assert s.calls == 3

    # A genuinely empty series is accepted after the retries are exhausted — ok stays True,
    # because the request SUCCEEDED. Only transport failure yields ok=False.
    class EmptySession:
        def get(self, url, params=None, timeout=None): return _Resp('{"markets": []}')
    payload, ok = kalshi_get("/markets", {}, session=EmptySession(), retries=2,
                             nonempty_key="markets")
    assert ok is True and payload["markets"] == []


def test_kalshi_get_parses_the_real_raw_newline_in_rules_secondary():
    """Kalshi emits literal newlines inside JSON strings, which is invalid JSON. Python's
    parser rejects it by default; strict=False is required."""
    from kalshi_series import kalshi_get

    class NewlineSession:
        def get(self, url, params=None, timeout=None):
            return _Resp('{"markets": [{"rules_secondary": "line one\nline two"}]}')

    payload, ok = kalshi_get("/markets", {}, session=NewlineSession())
    assert ok is True
    assert "line one" in payload["markets"][0]["rules_secondary"]


def test_kalshi_get_reports_transport_failure_as_not_ok():
    """Absence is a value. A failed request must never look like 'no data'."""
    from kalshi_series import kalshi_get

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    payload, ok = kalshi_get("/markets", {}, session=DeadSession(), retries=2)
    assert ok is False and payload is None


def test_target_series_covers_exactly_the_capture_cities():
    from kalshi_series import target_series
    from resolution_anchors import RESOLUTION_ANCHORS

    ts = target_series()
    assert set(ts) == {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert ts["Houston"] == "KXHIGHTHOU", "Houston's other three tickers are DEAD"
    assert ts["Los Angeles"] == "KXHIGHLAX"
```

Add this shared fake-response helper once, near the other Kalshi tests:

```python
class _Resp:
    """Minimal requests.Response stand-in: .text carries the raw body."""
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").exceptions.HTTPError(str(self.status_code))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -o addopts="" tests/ -k "kalshi_get or target_series" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kalshi_series'`.

- [ ] **Step 3: Create `kalshi_series.py`**

```python
"""kalshi_series.py — Kalshi series discovery and the archive health manifest.

Kalshi's daily-temperature markets resolve on the NWS Climatological Report; Polymarket's on
wunderground.com. Those are DIFFERENT RULERS (CLI >= WU on 99.40% of 1,668 station-days at KLGA,
mean +0.66 F), so a Kalshi leg never hedges a Polymarket leg. Kalshi is an INFORMATION source
here — nothing in this module trades.

WHY DISCOVERY IS DYNAMIC. Kalshi series tickers rot. Verified 2026-08-03: HIGHNY, HIGHCHI,
HIGHAUS, HIGHMIA, KXHIGHHOU, KXHIGHOU and KXHOUHIGH all still ENUMERATE but serve zero markets;
Houston has four tickers of which only KXHIGHTHOU is live, and HIGHNY -> KXHIGHNY shows a
completed migration. A hardcoded list would archive nothing behind a green run — which is the
failure mode this whole archive is designed against.

So every cycle records what it found in `data/kalshi/series_manifest.csv`, and a series that
previously served markets and now serves none is an ERROR, not an absence of news.
"""

import json
import logging
import time

import requests

from resolution_anchors import RESOLUTION_ANCHORS

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_TIMEOUT = 30
DEFAULT_RETRIES = 4

MANIFEST_COLS = ["fetched_at_utc", "series_ticker", "title",
                 "markets_returned", "live_markets", "truncated"]

# A market that is tradeable or about to be. `markets_returned == 0` means the ticker is dead;
# `live_markets == 0` with markets_returned > 0 is the normal overnight state and must not raise.
LIVE_STATUSES = {"active", "initialized"}


def kalshi_get(path: str, params: dict, session=None, retries: int = DEFAULT_RETRIES,
               nonempty_key: str | None = None):
    """GET and parse a Kalshi endpoint. Returns (payload, ok).

    ok=False  -> every attempt failed at the transport/parse level. The caller MUST NOT read
                 this as "no data"; it means we do not know.
    ok=True   -> a payload was obtained. When `nonempty_key` is given, an empty list at that key
                 is retried like a failure before being accepted, because an empty response and
                 a genuine absence are indistinguishable at the call site (this silently dropped
                 Houston and then Seattle during design).

    Parsing uses strict=False: `rules_secondary` contains raw newlines inside JSON strings,
    which is invalid JSON that Python's parser rejects by default.
    """
    sess = session or requests
    last = None
    for attempt in range(max(1, retries)):
        try:
            resp = sess.get(f"{KALSHI_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = json.loads(resp.text, strict=False)
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning("kalshi %s attempt %d/%d failed: %s", path, attempt + 1, retries, exc)
            time.sleep(1.5 * (attempt + 1))
            continue
        last = payload
        if nonempty_key is None or payload.get(nonempty_key):
            return payload, True
        logger.info("kalshi %s returned an EMPTY %s (attempt %d/%d) — retrying before "
                    "accepting it as genuine absence", path, nonempty_key, attempt + 1, retries)
        time.sleep(1.5 * (attempt + 1))
    if last is not None:
        return last, True          # genuinely empty after exhausting retries
    return None, False             # never got a parseable response


def target_series() -> dict:
    """{city: kalshi_series_ticker} for every capture-tier city.

    Derived from resolution_anchors so the Kalshi and Polymarket target sets cannot drift —
    the paired comparison is the entire product (see the venue-symmetry test).
    """
    return {c: a["kalshi_series"] for c, a in RESOLUTION_ANCHORS.items()
            if a.get("tier") == "capture" and a.get("kalshi_series")}


def manifest_row(series_ticker: str, title: str, markets_returned: int, live_markets: int,
                 truncated: bool, fetched_at_utc: str) -> dict:
    """One append-only health record. See MANIFEST_COLS."""
    return {"fetched_at_utc": fetched_at_utc, "series_ticker": series_ticker, "title": title,
            "markets_returned": int(markets_returned), "live_markets": int(live_markets),
            "truncated": bool(truncated)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -o addopts="" tests/ -k "kalshi_get or target_series" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Mutation-test the retry-on-empty guard**

In `kalshi_get`, temporarily change `if nonempty_key is None or payload.get(nonempty_key):` to
`if True:`.
Run: `pytest -o addopts="" tests/ -k kalshi_get_retries -v`
Expected: FAIL (`s.calls == 1`). Restore, confirm PASS.

- [ ] **Step 6: Mutation-test the strict=False guard**

Temporarily change `json.loads(resp.text, strict=False)` to `json.loads(resp.text)`.
Run: `pytest -o addopts="" tests/ -k raw_newline -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 7: Commit**

```bash
git add src/polymarket_weather/kalshi_series.py tests/test_polymarket_weather.py
git commit -m "kalshi: series registry + retry-on-empty HTTP helper

Kalshi tickers rot — HIGHNY/HIGHCHI/HIGHAUS/HIGHMIA/KXHIGHHOU/KXHIGHOU/KXHOUHIGH all enumerate
but serve zero markets, and Houston has four tickers of which only KXHIGHTHOU is live. Target
series derive from resolution_anchors so the two venues' sets cannot drift.

kalshi_get retries on EMPTY responses, not only on errors, and distinguishes 'we do not know'
(ok=False) from 'genuinely empty'. Parses with strict=False for the raw newlines Kalshi emits
inside rules_secondary."
```

---

## Task 4: Kalshi market capture

**Files:**
- Create: `src/polymarket_weather/fetch_kalshi.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `kalshi_series.kalshi_get`, `kalshi_series.LIVE_STATUSES`, `kalshi_series.MANIFEST_COLS`.
- Produces:
  - `fetch_series_markets(series_ticker: str, session=None, page_size: int = 200, max_pages: int = 50) -> tuple[list[dict], bool]`
  - `summarize_market(market: dict) -> dict`
  - `derive_bin(market: dict) -> dict | None`
  - `MARKET_COLS: list[str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_fetch_series_markets_reports_truncation_explicitly():
    """The Polymarket discovery bug was a hard API ceiling read as 'that is the end of the
    list', which captured ~3% of markets for months. Pagination must return truncation as a
    VALUE, never leave the caller to infer it."""
    from fetch_kalshi import fetch_series_markets

    class CappedSession:
        """Always returns a full page with a cursor — an infinite list."""
        def get(self, url, params=None, timeout=None):
            page = [{"ticker": f"T{i}", "status": "active"} for i in range(3)]
            return _Resp(__import__("json").dumps({"markets": page, "cursor": "more"}))

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=CappedSession(),
                                              page_size=3, max_pages=4)
    assert truncated is True, "hitting the page cap MUST report truncation"
    assert len(markets) == 12

    class ShortSession:
        def get(self, url, params=None, timeout=None):
            return _Resp(__import__("json").dumps({"markets": [{"ticker": "T1"}], "cursor": ""}))

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=ShortSession(), page_size=3)
    assert truncated is False, "a short page is a legitimate end of list"
    assert len(markets) == 1


def test_summarize_market_absence_is_none_never_a_sentinel():
    """data_loader.check_orderbook_vwap returns 1.0 when it cannot fill, which makes 'no
    liquidity' indistinguishable from 'priced at 1.0'. A price of 0 or 1 is a tradeable claim;
    absence is not."""
    from fetch_kalshi import summarize_market

    s = summarize_market({"ticker": "T1", "status": "active"})
    assert s["yes_bid"] is None and s["yes_ask"] is None
    assert s["volume"] is None
    assert s["ticker"] == "T1"

    s2 = summarize_market({"ticker": "T2", "yes_bid_dollars": "0.0000",
                           "yes_ask_dollars": "0.0700", "volume_fp": "0.00"})
    assert s2["yes_bid"] == 0.0, "a real zero bid is 0.0, NOT None"
    assert s2["yes_ask"] == 0.07
    assert s2["volume"] == 0.0


def test_summarize_market_keeps_both_rules_fields_verbatim():
    """The station is stated in a DIFFERENT FIELD per series generation: older KXHIGH* name the
    airport in rules_primary with no product code, newer KXHIGHT* give a bare city there and put
    the station in rules_secondary. Neither field alone identifies the station, and 'Houston' is
    ambiguous between Bush and Hobby."""
    from fetch_kalshi import summarize_market

    s = summarize_market({
        "ticker": "KXHIGHTHOU-26AUG04-T94",
        "rules_primary": "...recorded at Houston for Aug 4, 2026...",
        "rules_secondary": 'Data for CLIHOU ... location "Houston-Hobby, TX" ...',
    })
    assert "Houston-Hobby, TX" in s["rules_secondary"]
    assert "recorded at Houston" in s["rules_primary"]


def test_derive_bin_agrees_with_the_human_readable_subtitle():
    """floor_strike + strike_type + yes_sub_title are three representations of ONE threshold.
    The off-by-one between them is exactly the Hong Kong ruler bug's shape: floor_strike 82 with
    strike_type 'greater' means YES from 83, and the subtitle says '83° or above'."""
    from fetch_kalshi import derive_bin

    # Real market, captured live 2026-08-03.
    got = derive_bin({"floor_strike": 82, "strike_type": "greater",
                      "yes_sub_title": "83° or above"})
    assert got["op"] == "greater"
    assert got["yes_from_f"] == 83
    assert got["yes_to_f"] is None
    assert got["subtitle_bound"] == 83
    assert got["agrees_with_subtitle"] is True

    got_less = derive_bin({"floor_strike": 97, "strike_type": "less",
                           "yes_sub_title": "96° or below"})
    assert got_less["op"] == "less"
    assert got_less["yes_to_f"] == 96
    assert got_less["agrees_with_subtitle"] is True

    # A disagreement must be VISIBLE, not silently resolved in favour of either side.
    bad = derive_bin({"floor_strike": 82, "strike_type": "greater",
                      "yes_sub_title": "99° or above"})
    assert bad["agrees_with_subtitle"] is False

    # An unknown strike_type must not be guessed.
    assert derive_bin({"floor_strike": 82, "strike_type": "between",
                       "yes_sub_title": "x"}) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -o addopts="" tests/ -k "fetch_series_markets or summarize_market or derive_bin" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_kalshi'`.

- [ ] **Step 3: Create `fetch_kalshi.py` with market capture**

```python
"""fetch_kalshi.py — Kalshi daily-temperature market capture.

Three streams, all archived because Kalshi serves market objects for only ~2 MONTHS. Anything
not taken now cannot be taken later at any price, and snapshots only ever accumulate forward.
YAGNI is the right default when the data will still be there tomorrow; here it will not be.

  (a) market snapshots      — this module, hourly
  (b) order-book depth      — fetch_orderbooks, hourly
  (c) hourly candlesticks   — fetch_candles, the backfill

Nothing here trades or sizes. Kalshi resolves on the NWS CLI and Polymarket on wunderground.com,
so a Kalshi leg never hedges a Polymarket leg (spec 2026-08-03 sect 2.1).
"""

import logging

from kalshi_series import kalshi_get, LIVE_STATUSES

logger = logging.getLogger(__name__)

MARKET_COLS = [
    "fetched_at_utc", "city", "series_ticker", "ticker", "event_ticker", "title",
    "status", "result", "floor_strike", "strike_type", "yes_sub_title",
    "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "no_bid", "no_ask",
    "last_price", "previous_price", "volume", "volume_24h", "open_interest", "liquidity",
    "open_time", "close_time", "expiration_time", "rules_primary", "rules_secondary",
]


def _num(market: dict, key: str):
    """float(market[key]) or None. Absence is None; a real 0.0 stays 0.0."""
    v = market.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_series_markets(series_ticker: str, session=None, page_size: int = 200,
                         max_pages: int = 50):
    """All markets for one series. Returns (markets, truncated).

    `truncated` is a VALUE, not something the caller infers. The Polymarket discovery bug was a
    hard API ceiling read as "that is the end of the list", which captured ~3% of markets for
    months behind a green run.

        cursor present + page cap not reached -> keep walking
        no cursor, or an empty page           -> legitimate end of list
        max_pages reached                     -> TRUNCATED
        transport failure mid-walk            -> TRUNCATED (we have a partial list)
    """
    out, cursor, pages = [], None, 0
    while pages < max_pages:
        params = {"series_ticker": series_ticker, "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        payload, ok = kalshi_get("/markets", params, session=session,
                                 nonempty_key="markets" if pages == 0 else None)
        if not ok:
            logger.error("%s: transport failure on page %d — returning %d markets as TRUNCATED",
                         series_ticker, pages + 1, len(out))
            return out, True
        batch = payload.get("markets") or []
        out.extend(batch)
        cursor = payload.get("cursor") or None
        pages += 1
        if not cursor or not batch:
            return out, False
    logger.warning("%s: hit the %d-page cap — TRUNCATED at %d markets",
                   series_ticker, max_pages, len(out))
    return out, True


def derive_bin(market: dict):
    """Resolve floor_strike + strike_type into an explicit YES range, cross-checked against the
    human-readable subtitle. Returns None for an unrecognised strike_type — never a guess.

    Three representations of one threshold, and the off-by-one between them is exactly the shape
    of the Hong Kong ruler bug (a whole-degree bin compared against a tenths-rounded reading, so
    every market graded NO behind a passing audit). `agrees_with_subtitle` makes a disagreement
    visible instead of silently picking a side.
    """
    st = market.get("strike_type")
    if st not in ("greater", "less"):
        return None
    try:
        strike = float(market["floor_strike"])
    except (KeyError, TypeError, ValueError):
        return None

    if st == "greater":
        yes_from, yes_to = strike + 1, None
        bound = yes_from
    else:
        yes_from, yes_to = None, strike - 1
        bound = yes_to

    sub = str(market.get("yes_sub_title") or "")
    digits = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in sub).split()
    sub_bound = float(digits[0]) if digits else None

    return {"op": st, "threshold_f": strike, "yes_from_f": yes_from, "yes_to_f": yes_to,
            "subtitle_bound": sub_bound,
            "agrees_with_subtitle": sub_bound is not None and sub_bound == bound}


def summarize_market(market: dict) -> dict:
    """One archive row from a Kalshi market object. Vendor fields kept verbatim.

    BOTH rules fields are stored: the station is named in `rules_primary` for the older KXHIGH*
    series and only in `rules_secondary` for the newer KXHIGHT* ones, so neither alone identifies
    it — and "Houston" is ambiguous between Bush and Hobby.
    """
    return {
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "title": market.get("title"),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "floor_strike": _num(market, "floor_strike"),
        "strike_type": market.get("strike_type"),
        "yes_sub_title": market.get("yes_sub_title"),
        "yes_bid": _num(market, "yes_bid_dollars"),
        "yes_ask": _num(market, "yes_ask_dollars"),
        "yes_bid_size": _num(market, "yes_bid_size_fp"),
        "yes_ask_size": _num(market, "yes_ask_size_fp"),
        "no_bid": _num(market, "no_bid_dollars"),
        "no_ask": _num(market, "no_ask_dollars"),
        "last_price": _num(market, "last_price_dollars"),
        "previous_price": _num(market, "previous_price_dollars"),
        "volume": _num(market, "volume_fp"),
        "volume_24h": _num(market, "volume_24h_fp"),
        "open_interest": _num(market, "open_interest_fp"),
        "liquidity": _num(market, "liquidity_dollars"),
        "open_time": market.get("open_time"),
        "close_time": market.get("close_time"),
        "expiration_time": market.get("expiration_time"),
        "rules_primary": market.get("rules_primary"),
        "rules_secondary": market.get("rules_secondary"),
    }


def count_live(markets: list) -> int:
    """Markets that are tradeable or about to be. See kalshi_series.LIVE_STATUSES."""
    return sum(1 for m in markets if m.get("status") in LIVE_STATUSES)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -o addopts="" tests/ -k "fetch_series_markets or summarize_market or derive_bin" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Mutation-test the truncation guard**

In `fetch_series_markets`, temporarily change the final `return out, True` to `return out, False`.
Run: `pytest -o addopts="" tests/ -k fetch_series_markets_reports_truncation -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 6: Mutation-test the sentinel guard**

In `_num`, temporarily change `return None` (the absence branch) to `return 0.0`.
Run: `pytest -o addopts="" tests/ -k absence_is_none -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 7: Mutation-test the bin derivation**

In `derive_bin`, temporarily change `yes_from, yes_to = strike + 1, None` to `strike, None`.
Run: `pytest -o addopts="" tests/ -k derive_bin -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 8: Verify against the live API**

```bash
cd src/polymarket_weather && python -c "
from fetch_kalshi import fetch_series_markets, summarize_market, derive_bin, count_live
m, trunc = fetch_series_markets('KXHIGHLAX')
print(f'markets={len(m)} truncated={trunc} live={count_live(m)}')
assert m and not trunc
s = summarize_market(m[0]); b = derive_bin(m[0])
print(s['ticker'], s['yes_bid'], s['yes_ask'], b)
assert b is None or b['agrees_with_subtitle'], 'bin derivation disagrees with the subtitle'
"
```
Expected: a non-zero market count, `truncated=False`, and no assertion failure.

- [ ] **Step 9: Commit**

```bash
git add src/polymarket_weather/fetch_kalshi.py tests/test_polymarket_weather.py
git commit -m "kalshi: paginated market capture with explicit truncation

Truncation is returned as a value, never inferred — the Polymarket discovery bug was a hard API
ceiling read as end-of-list, which captured ~3% of markets for months. Absence is None, never a
sentinel. Both rules fields stored verbatim, because the station is named in rules_primary for
older series and only in rules_secondary for newer ones.

derive_bin cross-checks floor_strike + strike_type against the human-readable subtitle and
reports disagreement rather than silently picking a side."
```

---

## Task 5: Kalshi order-book capture

**Files:**
- Modify: `src/polymarket_weather/fetch_kalshi.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `kalshi_series.kalshi_get`; `fetch_orderbook.summarize_book` (existing, for the shared shape).
- Produces: `fetch_orderbooks(tickers: list[str], session=None) -> dict[str, dict]`, `BOOK_COLS: list[str]`

- [ ] **Step 1: Write the failing test**

```python
def test_kalshi_orderbooks_use_the_shared_summary_shape_and_omit_failures():
    """Both venues' books must be analysable by ONE code path, and a book that did not return
    must be OMITTED rather than faked — reading only one side of a two-sided Polymarket market
    produced a confidently wrong executability figure (71% vs 27% on the same markets)."""
    import json as _json
    from fetch_kalshi import fetch_orderbooks

    class BookSession:
        def get(self, url, params=None, timeout=None):
            if "T_DEAD" in url:
                return _Resp('{"orderbook_fp": {}}')
            return _Resp(_json.dumps({"orderbook_fp": {
                "yes_dollars": [{"price": "0.05", "size": "100"},
                                {"price": "0.07", "size": "9000"}],
                "no_dollars":  [{"price": "0.90", "size": "500"}],
            }}))

    out = fetch_orderbooks(["T_LIVE", "T_DEAD"], session=BookSession())
    assert "T_LIVE" in out
    live = out["T_LIVE"]
    assert live["yes_best_ask"] == 0.05, "best ask is the LOWEST ask"
    assert live["no_best_ask"] == 0.90
    assert live["yes_ask_depth_usdc"] > 0
    # An empty book yields None fields, never 0.0 masquerading as a price.
    assert out["T_DEAD"]["yes_best_ask"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -o addopts="" tests/ -k kalshi_orderbooks -v`
Expected: FAIL — `cannot import name 'fetch_orderbooks'`.

- [ ] **Step 3: Add order-book capture to `fetch_kalshi.py`**

```python
from fetch_orderbook import summarize_book

BOOK_COLS = ["fetched_at_utc", "city", "ticker",
             "yes_best_bid", "yes_best_ask", "yes_ask_depth_usdc", "yes_vwap_buy_100",
             "no_best_bid", "no_best_ask", "no_ask_depth_usdc", "no_vwap_buy_100"]


def fetch_orderbooks(tickers: list, session=None) -> dict:
    """{ticker: summary} for every book that returned. One request per ticker.

    Reuses fetch_orderbook.summarize_book so BOTH venues are analysed by one code path — the
    Polymarket work established that a mid without a book is misleading, and that reading one
    side of a two-sided market gives a confidently wrong answer.

    Kalshi nests its ladders under `orderbook_fp` as `yes_dollars` / `no_dollars`, each a list
    of {price, size} — the same shape summarize_book already consumes as `bids`/`asks`.

    Best-effort: a ticker whose book does not return is OMITTED, so a missing book reads as NaN
    downstream (the honest value) rather than as an empty book.
    """
    out = {}
    for t in tickers:
        payload, ok = kalshi_get(f"/markets/{t}/orderbook", {}, session=session)
        if not ok:
            logger.warning("kalshi orderbook %s: no response — omitted, not faked", t)
            continue
        ob = (payload or {}).get("orderbook_fp") or {}
        # A Kalshi ladder lists resting ASKS on each side; there is no separate bid array, so
        # `bids` is left empty and best_bid is legitimately None.
        yes = summarize_book({"bids": [], "asks": ob.get("yes_dollars") or []})
        no = summarize_book({"bids": [], "asks": ob.get("no_dollars") or []})
        out[t] = {
            "yes_best_bid": yes["best_bid"], "yes_best_ask": yes["best_ask"],
            "yes_ask_depth_usdc": yes["ask_depth_usdc"], "yes_vwap_buy_100": yes["vwap_buy_100"],
            "no_best_bid": no["best_bid"], "no_best_ask": no["best_ask"],
            "no_ask_depth_usdc": no["ask_depth_usdc"], "no_vwap_buy_100": no["vwap_buy_100"],
        }
    logger.info("kalshi order books: %d/%d returned", len(out), len(tickers))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -o addopts="" tests/ -k kalshi_orderbooks -v`
Expected: PASS

- [ ] **Step 5: Mutation-test the omit-on-failure guard**

Temporarily change `continue` (in the `if not ok:` branch) to
`out[t] = {k: 0.0 for k in ("yes_best_bid", "yes_best_ask")}`.
Run: `pytest -o addopts="" tests/ -k kalshi_orderbooks -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 6: Commit**

```bash
git add src/polymarket_weather/fetch_kalshi.py tests/test_polymarket_weather.py
git commit -m "kalshi: order-book depth via the shared summarize_book path

Both venues' books are now summarised by one code path, so a mid is never analysed without the
book behind it. A book that does not return is omitted rather than faked — absence reads as NaN
downstream, which is the honest value."
```

---

## Task 6: Kalshi candlestick backfill

**Files:**
- Modify: `src/polymarket_weather/fetch_kalshi.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `kalshi_series.kalshi_get`.
- Produces: `fetch_candles(series_ticker: str, market: dict, session=None) -> tuple[list[dict], dict]`, `CANDLE_COLS: list[str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_candle_window_brackets_the_markets_life_not_a_trailing_window():
    """A trailing 'last N days' window against a market that settled outside it returns zero
    candles at every interval — indistinguishable from 'this market never traded'. That mistake
    was made live while drafting the spec, against KXHIGHNY-26JUL21-B79.5 ($181k volume)."""
    from fetch_kalshi import fetch_candles

    captured = {}

    class CandleSession:
        def get(self, url, params=None, timeout=None):
            captured.update(params or {})
            return _Resp('{"candlesticks": [{"end_period_ts": 1784628000, '
                         '"price": {"close_dollars": "0.35"}, "volume_fp": "633.96", '
                         '"open_interest_fp": "7230.16", "yes_bid": {}, "yes_ask": {}}]}')

    market = {"ticker": "KXHIGHNY-26JUL21-B79.5",
              "open_time": "2026-07-19T14:00:00Z", "close_time": "2026-07-22T04:00:00Z"}
    candles, meta = fetch_candles("KXHIGHNY", market, session=CandleSession())

    assert len(candles) == 1
    assert captured["period_interval"] == 60, "1-minute returns HTTP 400 on multi-day windows"
    # The window must come from the market's own life, with a margin, not from 'now'.
    import datetime as _dt
    open_ts = int(_dt.datetime.fromisoformat("2026-07-19T14:00:00+00:00").timestamp())
    close_ts = int(_dt.datetime.fromisoformat("2026-07-22T04:00:00+00:00").timestamp())
    assert captured["start_ts"] <= open_ts, "window must start at or before open_time"
    assert captured["end_ts"] >= close_ts, "window must end at or after close_time"


def test_candle_backfill_records_completeness():
    """A market archived with zero candles must be distinguishable from one never attempted.
    A backfill quietly covering half its window is the obs-truncation failure in a new costume."""
    from fetch_kalshi import fetch_candles

    class EmptySession:
        def get(self, url, params=None, timeout=None):
            return _Resp('{"candlesticks": []}')

    market = {"ticker": "T1", "open_time": "2026-07-19T14:00:00Z",
              "close_time": "2026-07-22T04:00:00Z"}
    candles, meta = fetch_candles("KXHIGHNY", market, session=EmptySession())
    assert candles == []
    assert meta["candles"] == 0
    assert meta["ok"] is True, "the request SUCCEEDED and returned nothing — that is a fact"
    assert meta["start_ts"] and meta["end_ts"], "the requested window must be recorded"

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    candles, meta = fetch_candles("KXHIGHNY", market, session=DeadSession())
    assert meta["ok"] is False, "a failed fetch must NOT look like 'this market had no trading'"


def test_fetch_candles_refuses_a_market_with_no_life_window():
    """Without open_time/close_time there is no honest window to request."""
    from fetch_kalshi import fetch_candles
    candles, meta = fetch_candles("KXHIGHNY", {"ticker": "T1"}, session=None)
    assert candles == [] and meta["ok"] is False and meta["reason"] == "no_window"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -o addopts="" tests/ -k "candle" -v`
Expected: FAIL — `cannot import name 'fetch_candles'`.

- [ ] **Step 3: Add candle backfill to `fetch_kalshi.py`**

```python
import datetime as _dt

CANDLE_PERIOD_MINUTES = 60      # period_interval=1 returns HTTP 400 on multi-day windows
CANDLE_MARGIN_SECONDS = 3600    # a little slack either side of the market's life

CANDLE_COLS = ["city", "series_ticker", "ticker", "end_period_ts",
               "open_dollars", "high_dollars", "low_dollars", "close_dollars", "mean_dollars",
               "yes_bid_close", "yes_ask_close", "volume", "open_interest"]


def _ts(value):
    """Unix seconds from a Kalshi ISO timestamp, or None."""
    if not value:
        return None
    try:
        return int(_dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return None


def fetch_candles(series_ticker: str, market: dict, session=None):
    """Hourly candles covering ONE market's whole life. Returns (candles, meta).

    Candles are the only route to Kalshi's ~2-month backfill — snapshots accumulate forward only
    and can never recover the past.

    ⚠️ The window is derived from the market's own open_time/close_time, never from a trailing
    "last N days". A trailing window against a market that settled outside it returns zero
    candles at every interval, which reads as "this market never traded". That mistake was made
    live while drafting the spec against a market with $181k of volume.

    `meta` records the window requested and the candle count, so a market archived with zero
    candles is distinguishable from one never attempted:
        {"ticker", "start_ts", "end_ts", "candles", "ok", "reason"}
    """
    start = _ts(market.get("open_time"))
    end = _ts(market.get("close_time"))
    ticker = market.get("ticker")
    if start is None or end is None:
        return [], {"ticker": ticker, "start_ts": None, "end_ts": None, "candles": 0,
                    "ok": False, "reason": "no_window"}

    start -= CANDLE_MARGIN_SECONDS
    end += CANDLE_MARGIN_SECONDS
    payload, ok = kalshi_get(
        f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        {"start_ts": start, "end_ts": end, "period_interval": CANDLE_PERIOD_MINUTES},
        session=session,
    )
    if not ok:
        return [], {"ticker": ticker, "start_ts": start, "end_ts": end, "candles": 0,
                    "ok": False, "reason": "fetch_failed"}
    candles = (payload or {}).get("candlesticks") or []
    return candles, {"ticker": ticker, "start_ts": start, "end_ts": end,
                     "candles": len(candles), "ok": True, "reason": ""}


def summarize_candle(candle: dict, city: str, series_ticker: str, ticker: str) -> dict:
    """One archive row from a Kalshi candlestick. Absence is None."""
    price = candle.get("price") or {}
    bid = candle.get("yes_bid") or {}
    ask = candle.get("yes_ask") or {}
    return {
        "city": city, "series_ticker": series_ticker, "ticker": ticker,
        "end_period_ts": candle.get("end_period_ts"),
        "open_dollars": _num(price, "open_dollars"),
        "high_dollars": _num(price, "high_dollars"),
        "low_dollars": _num(price, "low_dollars"),
        "close_dollars": _num(price, "close_dollars"),
        "mean_dollars": _num(price, "mean_dollars"),
        "yes_bid_close": _num(bid, "close_dollars"),
        "yes_ask_close": _num(ask, "close_dollars"),
        "volume": _num(candle, "volume_fp"),
        "open_interest": _num(candle, "open_interest_fp"),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -o addopts="" tests/ -k "candle" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Mutation-test the window guard**

Temporarily replace the window derivation with a trailing one:
```python
    end = int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    start = end - 7 * 86400
```
Run: `pytest -o addopts="" tests/ -k candle_window_brackets -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 6: Mutation-test the completeness guard**

In the `if not ok:` branch, temporarily change `"ok": False` to `"ok": True`.
Run: `pytest -o addopts="" tests/ -k candle_backfill_records_completeness -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 7: Verify against the live API on a real settled market**

```bash
cd src/polymarket_weather && python -c "
from fetch_kalshi import fetch_series_markets, fetch_candles, summarize_candle
m, _ = fetch_series_markets('KXHIGHNY')
traded = sorted([x for x in m if float(x.get('volume_fp') or 0) > 0],
                key=lambda x: float(x['volume_fp']), reverse=True)
mk = traded[0]
c, meta = fetch_candles('KXHIGHNY', mk)
print(mk['ticker'], 'candles=', meta['candles'], 'ok=', meta['ok'])
assert meta['ok'] and meta['candles'] > 0, 'expected candles for a traded market'
print(summarize_candle(c[len(c)//2], 'NYC', 'KXHIGHNY', mk['ticker']))
"
```
Expected: a non-zero candle count and a populated summary row.

- [ ] **Step 8: Commit**

```bash
git add src/polymarket_weather/fetch_kalshi.py tests/test_polymarket_weather.py
git commit -m "kalshi: hourly candlestick backfill with recorded completeness

Candles are the only route to Kalshi's ~2-month backfill; snapshots accumulate forward only.
Windows bracket the market's own open_time/close_time rather than a trailing 'last N days' — a
trailing window against a market that settled outside it returns zero candles at every interval,
which reads as 'never traded'. meta records the requested window and the count, so zero candles
is distinguishable from never attempted."
```

---

## Task 7: Station truth for the seven

**Files:**
- Modify: `src/polymarket_weather/fetch_historical_truth.py` (`SOURCES`)
- Modify: `src/polymarket_weather/fetch_station_obs.py` (`OBS_STATIONS`)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `RESOLUTION_ANCHORS` tiers from Task 1; the existing `_fetch_cli(station)` adapter.
- Produces: `data/weather/{slug}_historical_actuals.csv` and `{slug}_obs_hourly.csv` for the seven.

- [ ] **Step 1: Write the failing test**

```python
def test_both_rulers_are_configured_for_every_capture_city():
    """Each overlap city needs BOTH rulers archived, because the venues differ:
    Kalshi resolves on the NWS CLI, Polymarket on Wunderground (reconstructed from hourly
    METARs by wu_truth). Neither is converted at write time — the transfer function is a later
    spec. A city with only one ruler is half-useless."""
    from resolution_anchors import RESOLUTION_ANCHORS, slug
    from fetch_historical_truth import SOURCES
    from fetch_station_obs import OBS_STATIONS

    capture = {c: a for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    for city, anchor in capture.items():
        s = slug(city)
        assert s in SOURCES, f"{city}: no CLI truth source (Kalshi's ruler)"
        kind, kw = SOURCES[s]
        assert kind == "cli" and kw["station"] == anchor["station_code"]
        assert s in OBS_STATIONS, f"{city}: no METAR obs (Polymarket's WU ruler)"
        assert OBS_STATIONS[s][0] == anchor["station_code"].lstrip("K")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -o addopts="" tests/ -k both_rulers -v`
Expected: FAIL — `'los_angeles' not in SOURCES`.

- [ ] **Step 3: Add the seven CLI truth sources**

In `src/polymarket_weather/fetch_historical_truth.py`, extend `SOURCES`:

```python
    # ── capture tier (spec 2026-08-03) — KALSHI's ruler, the NWS Climatological Report.
    # Polymarket's ruler for these same cities is Wunderground, reconstructed from the hourly
    # METARs fetched by fetch_station_obs. BOTH are archived; NEITHER is converted at write
    # time — the CLI<->WU transfer function is a later spec.
    "los_angeles":   ("cli", {"station": "KLAX"}),
    "austin":        ("cli", {"station": "KAUS"}),
    "atlanta":       ("cli", {"station": "KATL"}),
    "houston":       ("cli", {"station": "KHOU"}),
    "miami":         ("cli", {"station": "KMIA"}),
    "seattle":       ("cli", {"station": "KSEA"}),
    "san_francisco": ("cli", {"station": "KSFO"}),
```

- [ ] **Step 4: Add the seven METAR obs stations**

In `src/polymarket_weather/fetch_station_obs.py`, extend `OBS_STATIONS` (IEM uses the 3-letter
form for US stations, matching the existing `LGA` / `ORD` entries):

```python
    "los_angeles":   ("LAX", "America/Los_Angeles"),
    "austin":        ("AUS", "America/Chicago"),
    "atlanta":       ("ATL", "America/New_York"),
    "houston":       ("HOU", "America/Chicago"),
    "miami":         ("MIA", "America/New_York"),
    "seattle":       ("SEA", "America/Los_Angeles"),
    "san_francisco": ("SFO", "America/Los_Angeles"),
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest -o addopts="" tests/ -k both_rulers -v`
Expected: PASS

- [ ] **Step 6: Verify both feeds return data for one new city**

```bash
cd src/polymarket_weather && python -c "
from fetch_historical_truth import _fetch_cli
df = _fetch_cli('KLAX')
print('KLAX CLI rows:', len(df), 'latest:', df['date_local'].max())
assert len(df) > 1000, 'expected years of CLI history'
"
```
Expected: thousands of rows.

- [ ] **Step 7: Run the full suite**

Run: `pytest -o addopts="" tests/ -q`
Expected: all passing.

- [ ] **Step 8: Commit**

```bash
git add src/polymarket_weather/fetch_historical_truth.py src/polymarket_weather/fetch_station_obs.py tests/test_polymarket_weather.py
git commit -m "truth: both rulers for the seven overlap cities

NWS CLI (Kalshi's ruler) via the existing cli adapter, and hourly METARs (Polymarket's
Wunderground ruler, via wu_truth) for the same seven stations. Both archived, neither converted
at write time — the CLI<->WU transfer function is a later spec. A test asserts every capture
city has both."
```

---

## Task 8: Wire Kalshi into the collector

**Files:**
- Modify: `src/polymarket_weather/main.py`
- Modify: `src/polymarket_weather/processing.py` (two save helpers)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: everything from Tasks 3–6.
- Produces: `step_fetch_kalshi()` in `main.py`; `data/kalshi/*.csv` written each collector cycle.

- [ ] **Step 1: Write the failing test**

```python
def test_ticker_rot_is_an_error_not_an_absence():
    """A series that previously served markets and now serves none is the rot signature —
    KXHIGHHOU is already dead. It must be loud, not logged at debug."""
    import pandas as pd
    from main import _kalshi_rot_alarms

    prev = pd.DataFrame([
        {"series_ticker": "KXHIGHLAX", "markets_returned": 200},
        {"series_ticker": "KXHIGHTHOU", "markets_returned": 200},
    ])
    now = [{"series_ticker": "KXHIGHLAX", "markets_returned": 200},
           {"series_ticker": "KXHIGHTHOU", "markets_returned": 0}]
    alarms = _kalshi_rot_alarms(now, prev)
    assert alarms == ["KXHIGHTHOU"]

    # A series that has NEVER served markets is not rot — nothing was lost.
    prev2 = pd.DataFrame([{"series_ticker": "KXNEW", "markets_returned": 0}])
    assert _kalshi_rot_alarms([{"series_ticker": "KXNEW", "markets_returned": 0}], prev2) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -o addopts="" tests/ -k ticker_rot -v`
Expected: FAIL — `cannot import name '_kalshi_rot_alarms'`.

- [ ] **Step 3: Add the Kalshi save helpers to `processing.py`**

```python
# processing.py already imports POLYMARKET_DIR / WEATHER_DIR from config; add KALSHI_DIR
# to that import. There is no _DATA_DIR in this module.
_KALSHI_DIR = Path(KALSHI_DIR)


def save_kalshi_rows(kind: str, city_slug: str, rows: list[dict], dedup_cols: list[str]) -> int:
    """Append Kalshi rows to data/kalshi/{city_slug}_{kind}.csv. Append-only + dedupe on read,
    matching the Polymarket convention. `_append_csv` already unions columns on schema widening,
    so new vendor fields persist rather than being silently dropped."""
    if not rows:
        return 0
    _KALSHI_DIR.mkdir(parents=True, exist_ok=True)
    return _append_csv(_KALSHI_DIR / f"{city_slug}_{kind}.csv", rows, dedup_cols=dedup_cols)


def save_kalshi_manifest(rows: list[dict]) -> int:
    """Append series-health rows to data/kalshi/series_manifest.csv."""
    if not rows:
        return 0
    _KALSHI_DIR.mkdir(parents=True, exist_ok=True)
    return _append_csv(_KALSHI_DIR / "series_manifest.csv", rows,
                       dedup_cols=["series_ticker", "fetched_at_utc"])
```

Update the existing import at `processing.py:22` from
`from config import POLYMARKET_DIR, WEATHER_DIR` to
`from config import KALSHI_DIR, POLYMARKET_DIR, WEATHER_DIR`.

- [ ] **Step 4: Add the collector step to `main.py`**

```python
def _kalshi_rot_alarms(rows: list[dict], previous) -> list[str]:
    """Series that HAVE served markets before and serve none now — the ticker-rot signature.

    Kalshi renames series (HIGHNY -> KXHIGHNY) and leaves the old ticker enumerable but empty.
    A series that never served markets is not rot; nothing was lost.
    """
    if previous is None or not len(previous):
        return []
    ever = set(previous.loc[previous["markets_returned"].astype(float) > 0, "series_ticker"])
    return [r["series_ticker"] for r in rows
            if r["series_ticker"] in ever and int(r["markets_returned"]) == 0]


def step_fetch_kalshi() -> None:
    """Archive Kalshi markets, order books and candles for the capture-tier cities.

    Additive and best-effort: a Kalshi outage must never block the irreplaceable Polymarket
    snapshot, so every failure here is logged and swallowed.
    """
    import pandas as pd
    from kalshi_series import target_series, manifest_row
    from fetch_kalshi import (fetch_series_markets, summarize_market, count_live,
                              fetch_orderbooks, fetch_candles, summarize_candle)
    from processing import save_kalshi_rows, save_kalshi_manifest, kalshi_manifest_path
    from resolution_anchors import slug

    logger.info("═══ Step 1b: Kalshi archive ═══")
    now = datetime.now(timezone.utc).isoformat()
    try:
        previous = pd.read_csv(kalshi_manifest_path()) if kalshi_manifest_path().exists() else None
    except Exception:
        previous = None

    manifest = []
    for city, series in target_series().items():
        cslug = slug(city)
        markets, truncated = fetch_series_markets(series)
        manifest.append(manifest_row(series, city, len(markets), count_live(markets),
                                     truncated, now))
        if not markets:
            continue

        rows = [{**summarize_market(m), "fetched_at_utc": now, "city": city,
                 "series_ticker": series} for m in markets]
        save_kalshi_rows("markets", cslug, rows, ["ticker", "fetched_at_utc"])

        live = [m["ticker"] for m in markets if m.get("status") in ("active", "initialized")]
        books = fetch_orderbooks(live)
        if books:
            save_kalshi_rows("books", cslug,
                             [{"fetched_at_utc": now, "city": city, "ticker": t, **b}
                              for t, b in books.items()],
                             ["ticker", "fetched_at_utc"])

        # Candle backfill: settled markets only, captured once, completely, after their life.
        for m in markets:
            if m.get("status") not in ("finalized", "settled", "closed"):
                continue
            candles, meta = fetch_candles(series, m)
            if not meta["ok"]:
                logger.warning("kalshi candles %s: %s", meta["ticker"], meta["reason"])
                continue
            if candles:
                save_kalshi_rows("candles", cslug,
                                 [summarize_candle(c, city, series, m["ticker"]) for c in candles],
                                 ["ticker", "end_period_ts"])

    save_kalshi_manifest(manifest)
    for rotted in _kalshi_rot_alarms(manifest, previous):
        logger.error("KALSHI TICKER ROT: %s served markets before and serves NONE now. Kalshi "
                     "renames series and leaves the old ticker enumerable but empty; a "
                     "hardcoded list would archive nothing behind a green run.", rotted)
```

Add `kalshi_manifest_path()` to `processing.py`:

```python
def kalshi_manifest_path():
    """Path to the Kalshi series-health manifest."""
    return _KALSHI_DIR / "series_manifest.csv"
```

Then call `step_fetch_kalshi()` from the collect path in `main()`, immediately after
`step_fetch_polymarket(...)`, wrapped so it can never break collection:

```python
    try:
        step_fetch_kalshi()
    except Exception as exc:            # noqa: BLE001 — Kalshi must never block collection
        logger.warning("Kalshi archive failed (%s) — Polymarket collection unaffected.", exc)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest -o addopts="" tests/ -k ticker_rot -v`
Expected: PASS

- [ ] **Step 6: Mutation-test the rot alarm**

In `_kalshi_rot_alarms`, temporarily change `return []` (the no-previous branch) to
`return [r["series_ticker"] for r in rows]`.
Run: `pytest -o addopts="" tests/ -k ticker_rot -v`
Expected: FAIL. Restore, confirm PASS.

- [ ] **Step 7: Run one real collector cycle and inspect the output**

```bash
cd src/polymarket_weather && python -c "
import logging; logging.basicConfig(level=logging.INFO)
from main import step_fetch_kalshi
step_fetch_kalshi()
"
ls -la src/polymarket_weather/data/kalshi/
```
Expected: `series_manifest.csv` plus `{city}_markets.csv` / `_books.csv` for the seven cities.
No `KALSHI TICKER ROT` errors on a first run (there is no previous manifest to compare against).

- [ ] **Step 8: Run the full suite**

Run: `pytest -o addopts="" tests/ -q`
Expected: all passing. Confirm the exit code directly — `echo $?` — **not** through a pipe;
piping pytest through `tail` discards its exit status and has shipped failing tests before.

- [ ] **Step 9: Commit**

```bash
git add src/polymarket_weather/main.py src/polymarket_weather/processing.py src/polymarket_weather/data/kalshi tests/test_polymarket_weather.py
git commit -m "collect: archive Kalshi markets, books and candles each cycle

Additive and best-effort — a Kalshi outage can never block the irreplaceable Polymarket
snapshot. Every cycle writes a series-health manifest, and a series that served markets before
and serves none now raises a loud TICKER ROT error rather than being logged as an absence."
```

---

## Self-Review

**Spec coverage.** §2.3.1 seven cities → Task 1. §3 principles → distributed: store-raw (Task 4
Step 3), discover-never-hardcode (Task 3, Task 8), completeness-as-data (Tasks 4, 6),
absence-never-sentinel (Task 4), append-only (Task 8 via `_append_csv`), per-entity health
(Task 8 manifest), separate namespace (`data/kalshi/`, Task 8). §4.0 symmetry → Task 1 Step 1.
§4.1 tiering → Task 1. §4.2 manifest → Tasks 3, 8. §4.3 market capture → Task 4. §4.3.1(b)
books → Task 5. §4.3.1(c) candles → Task 6. §4.4 bin semantics → Task 4. §4.5 Polymarket
capture → Task 2. §4.6 truth → Task 7. §5 storage → Task 8. §6 error handling → Tasks 3, 5, 6, 8.

**All ten §7 guards have a task step:** 1 truncation (T4 S5), 2 ticker rot (T8 S6), 3 bin
derivation (T4 S7), 4 tier isolation (T1 S8), 5 sentinel-free absence (T4 S6), 6 malformed JSON
(T3 S6), 7 venue symmetry (T1 S1 — asserted; no mutation needed, the assertion IS the guard),
8 retry-on-empty (T3 S5), 9 candle window (T6 S5), 10 backfill completeness (T6 S6).

**Placeholder scan:** none — every step carries runnable code or an exact command.

**Type consistency:** `kalshi_get` returns `(payload, ok)` and is consumed with that shape in
Tasks 4, 5, 6. `fetch_series_markets` returns `(markets, truncated)` and is consumed that way in
Task 8. `fetch_candles` returns `(candles, meta)` with `meta` keys `ticker/start_ts/end_ts/
candles/ok/reason`, consumed in Task 8. `summarize_book` returns unprefixed keys
(`best_bid`, `best_ask`, `ask_depth_usdc`, `vwap_buy_100`) — matching `fetch_orderbook.py` as it
exists today — and Task 5 prefixes them. `slug()` from `resolution_anchors` is used for every
file path.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-03-kalshi-archive.md`.
