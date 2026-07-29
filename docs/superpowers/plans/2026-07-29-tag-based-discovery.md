# Tag-Based Market Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover weather markets through Polymarket's `weather` tag instead of a volume-ranked scan that can only reach the top ~2100 active markets, recovering roughly 30× more markets per collect cycle.

**Architecture:** All changes live in `src/polymarket_weather/fetch_polymarket.py`. `fetch_weather_markets(city)` keeps its exact signature and return shape, so `processing.py`, the engine and every consumer are untouched. Internally it now reads from a once-per-process tag enumeration instead of running 44 paginations, and falls back to the (repaired) query scan if the tag yields nothing.

**Tech Stack:** Python 3.11, `requests` via the repo's `http_util.get_json`, pytest.

## Global Constraints

- **`fetch_weather_markets(city) -> list[dict]` signature and return shape must not change.** Verified 2026-07-29: event-nested market dicts carry every field `extract_market_snapshot` reads (`conditionId`, `question`, `active`, `closed`, `endDateIso`, `startDateIso`, `volume`, `volume24hr`, `liquidity`, `clobTokenIds`, `outcomePrices`, `marketMakerAddress`), so nothing downstream needs touching.
- **Page the tag ONCE per process**, not once per city. A module-level cache provides this; `_reset_tag_cache()` exists so tests are not order-dependent.
- **Truncation must never look like completion.** `_paged_events` returns an explicit `truncated` flag AND logs a warning. The swallowed 422 at offset 2100 is what hid a 3% capture rate for months.
- **Tmin markets must survive discovery.** ~20% of markets settle on the daily minimum (`"Will the lowest temperature in London be 15°C or below…"`). A filter written around "highest" silently drops them.
- **The fallback must be repaired, not just retained.** Keyword and city term must each appear somewhere in the question, independently and case-insensitively — NOT as the concatenated substring `f"{kw} {term}"`.
- No historical backfill. No per-cycle cap. No changes to `processing.py`, the engine, or anything downstream.
- Modules run from `src/polymarket_weather/`; tests run from the repo root with `pytest -o addopts="" tests/ -v`. `tests/conftest.py` already puts `src/polymarket_weather` on `sys.path`.
- Tests are appended to the existing `tests/test_polymarket_weather.py` (this repo keeps one test file).

---

### Task 1: Question classification

**Files:**
- Modify: `src/polymarket_weather/fetch_polymarket.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `config.CITIES` (each city has a `search_terms` list).
- Produces: `is_temperature_question(question: str) -> bool`, `match_city(question: str) -> str | None` (returns the `CITIES` key, e.g. `"New York City"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_match_city_handles_every_configured_alias():
    """Discovery is only as good as this function — a city it cannot name is a city we
    silently stop collecting."""
    import fetch_polymarket as fp
    assert fp.match_city("Will the highest temperature in London be 22°C on July 29?") == "London"
    assert fp.match_city("Will the highest temperature in Seoul be 30°C on July 29?") == "Seoul"
    assert fp.match_city("Will the highest temperature in Chicago be 30°C on July 29?") == "Chicago"
    assert fp.match_city("Will the highest temperature in Hong Kong be 33°C on July 29?") == "Hong Kong"
    # New York City has three aliases and all must land on the same key
    for q in ["Highest temperature in New York on July 29?",
              "Highest temperature in NYC on July 29?",
              "highest temperature in new york on July 29?"]:
        assert fp.match_city(q) == "New York City", q


def test_match_city_does_not_match_substrings_of_other_words():
    """'New Yorker' contains 'New York'. Matching it would file an unrelated market under NYC
    and corrupt that city's series."""
    import fetch_polymarket as fp
    assert fp.match_city("Will the New Yorker publish a temperature piece?") is None
    assert fp.match_city("Will the highest temperature in Paris be 30°C?") is None


def test_is_temperature_question_keeps_tmin_markets():
    """~20% of markets settle on the daily MINIMUM. A filter written around 'highest' drops
    them silently — and Tmin is already a market type this repo excluded once before."""
    import fetch_polymarket as fp
    assert fp.is_temperature_question("Will the lowest temperature in London be 15°C or below on July 29?")
    assert fp.is_temperature_question("Will the highest temperature in London be 22°C on July 29?")
    assert not fp.is_temperature_question("Will it rain in London on July 29?")
    assert not fp.is_temperature_question("Will 2026 be the hottest year on record?")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "match_city or is_temperature_question" -v`
Expected: FAIL with `AttributeError: module 'fetch_polymarket' has no attribute 'match_city'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/fetch_polymarket.py` (put `import re` with the existing imports):

```python
import re

# "temperature" plus a superlative. Deliberately matches BOTH highest and lowest: about 20% of
# markets settle on the daily minimum, and a pattern written around "highest" drops them without
# a trace — Tmin is already a market type this repo excluded once by accident.
_TEMP_RE = re.compile(r"\b(highest|lowest|high|low|max|min|maximum|minimum)\b[^?]*\btemperature\b"
                      r"|\btemperature\b[^?]*\b(highest|lowest|high|low|max|min)\b", re.I)


def is_temperature_question(question: str) -> bool:
    """Is this a daily high/low temperature market?"""
    return bool(_TEMP_RE.search(question or ""))


def match_city(question: str) -> str | None:
    """Which configured city this question is about, or None.

    The single place that decides city membership, so "New York" / "NYC" / "new york" collapsing
    to one key is asserted in one test rather than assumed in five call sites. Matching is on WORD
    BOUNDARIES: "New Yorker" contains "New York", and filing that under NYC would quietly corrupt
    the city's series.
    """
    q = question or ""
    for city, cfg in CITIES.items():
        for term in cfg.get("search_terms", [city]):
            if re.search(rf"\b{re.escape(term)}\b", q, re.I):
                return city
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "match_city or is_temperature_question" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/fetch_polymarket.py tests/test_polymarket_weather.py
git commit -m "discovery: question classification (city + temperature), word-boundary safe"
```

---

### Task 2: Event pagination with loud truncation

**Files:**
- Modify: `src/polymarket_weather/fetch_polymarket.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `_get(url, params)` (returns the decoded body, or `None` on any failure).
- Produces: `_paged_events(tag_slug: str, page_size: int = 100) -> tuple[list[dict], bool]` returning `(events, truncated)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_paged_events_stops_cleanly_on_a_short_page(monkeypatch):
    """A short page is a legitimate end of list — no truncation flag, no warning."""
    import fetch_polymarket as fp
    pages = [[{"id": i} for i in range(100)], [{"id": 100}]]
    calls = []

    def fake_get(url, params=None):
        calls.append(params["offset"])
        return pages.pop(0) if pages else []

    monkeypatch.setattr(fp, "_get", fake_get)
    events, truncated = fp._paged_events("weather", page_size=100)
    assert len(events) == 101
    assert truncated is False
    assert calls == [0, 100]


def test_paged_events_flags_truncation_when_a_full_page_is_followed_by_failure():
    """THE bug this whole change exists for. GET /markets 422s at offset 2100 and _get returns
    None, so the pager read a hard truncation as 'last page'. A 3% capture rate looked healthy
    for months. None after a FULL page must be reported as truncation, never as completion."""
    import fetch_polymarket as fp

    class _Stub:
        def __init__(self):
            self.n = 0

        def __call__(self, url, params=None):
            self.n += 1
            return [{"id": i} for i in range(100)] if self.n == 1 else None

    import unittest.mock as m
    with m.patch.object(fp, "_get", _Stub()):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert len(events) == 100
    assert truncated is True, "a full page followed by an error is a TRUNCATION, not the end"


def test_paged_events_reports_no_truncation_when_the_first_page_fails():
    """An endpoint that is down from the first call is an outage, not a truncation — the caller
    falls back rather than trusting a partial list."""
    import fetch_polymarket as fp
    import unittest.mock as m
    with m.patch.object(fp, "_get", lambda url, params=None: None):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert events == []
    assert truncated is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "paged_events" -v`
Expected: FAIL with `AttributeError: module 'fetch_polymarket' has no attribute '_paged_events'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/fetch_polymarket.py`:

```python
def _paged_events(tag_slug: str, page_size: int = 100) -> tuple[list[dict], bool]:
    """Page GET /events for one tag. Returns (events, truncated).

    `_get` collapses every failure to None, so the pager cannot ask WHY a page was empty and has
    to infer:

        full page (== page_size)   -> keep going
        short page (< page_size)   -> legitimate end of list
        None after a full page     -> TRUNCATED: warn loudly, return what we have
        None on the first page     -> endpoint down: caller falls back

    The distinction is the entire point of this change. GET /markets returns 422 at offset 2100;
    the old pager read that None as "last page" and stopped, so a hard ceiling on how much of
    Polymarket we could even see was indistinguishable from having seen all of it.
    """
    url = f"{GAMMA_API_BASE}/events"
    events: list[dict] = []
    offset = 0
    while True:
        page = _get(url, {"tag_slug": tag_slug, "limit": page_size,
                          "offset": offset, "closed": "false"})
        if page is None:
            if events:
                logger.warning(
                    "TRUNCATED: /events?tag_slug=%s failed at offset %d after %d events. "
                    "Returning a PARTIAL list — do not treat this as the full set.",
                    tag_slug, offset, len(events))
                return events, True
            logger.warning("/events?tag_slug=%s unavailable at offset 0", tag_slug)
            return [], False
        if not isinstance(page, list):
            page = page.get("data", []) or []
        if not page:
            return events, False
        events.extend(page)
        if len(page) < page_size:
            return events, False
        offset += page_size
        time.sleep(0.2)   # be polite to the API
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "paged_events" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/fetch_polymarket.py tests/test_polymarket_weather.py
git commit -m "discovery: paginate /events with explicit truncation detection"
```

---

### Task 3: Partition tagged markets by city

**Files:**
- Modify: `src/polymarket_weather/fetch_polymarket.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `_paged_events` (Task 2), `match_city` / `is_temperature_question` (Task 1).
- Produces: `discover_by_tag(tag_slug: str = "weather") -> dict[str, list[dict]]` mapping a `CITIES` key to raw Gamma market dicts; `_reset_tag_cache() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
_TAG_EVENTS = [
    {"title": "London temps", "markets": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C on July 29?"},
        {"conditionId": "b", "question": "Will the lowest temperature in London be 15°C or below on July 29?"},
    ]},
    {"title": "NYC temps", "markets": [
        {"conditionId": "c", "question": "Will the highest temperature in NYC be 30°C on July 29?"},
    ]},
    {"title": "climate", "markets": [
        {"conditionId": "d", "question": "Will 2026 be the hottest year on record?"},
    ]},
    {"title": "malformed — no markets key"},
    {"title": "dupe", "markets": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C on July 29?"},
    ]},
]


def test_discover_by_tag_partitions_by_city_and_keeps_tmin(monkeypatch):
    """Both London rows must survive: one Tmax, one Tmin. Dropping Tmin here would repeat an
    exclusion this repo has already made once by accident."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "_paged_events", lambda tag, page_size=100: (_TAG_EVENTS, False))
    out = fp.discover_by_tag("weather")
    assert sorted(out) == ["London", "New York City"]
    assert len(out["London"]) == 2, "the Tmin market was dropped"
    assert len(out["New York City"]) == 1


def test_discover_by_tag_ignores_non_temperature_and_malformed_events(monkeypatch):
    """The weather tag also carries climate markets ('hottest year on record') and events with
    no markets key at all. Neither may reach a city bucket or raise."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "_paged_events", lambda tag, page_size=100: (_TAG_EVENTS, False))
    out = fp.discover_by_tag("weather")
    qs = [m["question"] for ms in out.values() for m in ms]
    assert not any("hottest year" in q for q in qs)


def test_discover_by_tag_dedupes_and_pages_only_once(monkeypatch):
    """Condition 'a' appears in two events. And the tag must be paged ONCE per process, not once
    per city — otherwise five cities means five full paginations, which is the cost this change
    exists to remove."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    calls = []

    def fake_paged(tag, page_size=100):
        calls.append(tag)
        return _TAG_EVENTS, False

    monkeypatch.setattr(fp, "_paged_events", fake_paged)
    fp.discover_by_tag("weather")
    fp.discover_by_tag("weather")
    fp.discover_by_tag("weather")
    assert len(calls) == 1, "tag was paged more than once per process"
    assert len(fp.discover_by_tag("weather")["London"]) == 2, "duplicate conditionId not deduped"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "discover_by_tag" -v`
Expected: FAIL with `AttributeError: module 'fetch_polymarket' has no attribute '_reset_tag_cache'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/fetch_polymarket.py`:

```python
# One pagination per process. fetch_weather_markets is called once per city, and the old code
# paid 44 full paginations per collect cycle for that; the tag is global, so it is fetched once
# and partitioned. Collect runs as a fresh process each cycle, so process lifetime IS the
# correct cache lifetime — no TTL needed.
_TAG_CACHE: dict[str, dict[str, list[dict]]] = {}


def _reset_tag_cache() -> None:
    """Clear the per-process tag cache. Exists so tests are not order-dependent."""
    _TAG_CACHE.clear()


def discover_by_tag(tag_slug: str = "weather") -> dict[str, list[dict]]:
    """{city: [raw Gamma market dicts]} for every configured city, from one tag enumeration.

    Returns raw market dicts, NOT snapshots: event-nested markets carry every field
    extract_market_snapshot reads (verified 2026-07-29), so the caller's existing conversion is
    unchanged and nothing downstream of it needs to know discovery changed.
    """
    if tag_slug in _TAG_CACHE:
        return _TAG_CACHE[tag_slug]

    events, truncated = _paged_events(tag_slug)
    if truncated:
        logger.warning("tag '%s' enumeration was truncated — city counts below are a FLOOR, "
                       "not a complete picture", tag_slug)

    by_city: dict[str, list[dict]] = {}
    seen: set[str] = set()
    skipped_events = 0
    for ev in events:
        markets = ev.get("markets")
        if not markets:
            skipped_events += 1
            continue
        for m in markets:
            q = m.get("question") or ""
            if not is_temperature_question(q):
                continue
            city = match_city(q)
            if city is None:
                continue
            cid = m.get("conditionId") or ""
            if not cid or cid in seen:
                continue
            seen.add(cid)
            by_city.setdefault(city, []).append(m)

    if skipped_events:
        logger.info("tag '%s': %d event(s) carried no markets key", tag_slug, skipped_events)
    logger.info("tag '%s': %d markets across %d cities (%s)", tag_slug, len(seen), len(by_city),
                ", ".join(f"{c}={len(v)}" for c, v in sorted(by_city.items())))

    _TAG_CACHE[tag_slug] = by_city
    return by_city
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "discover_by_tag" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/fetch_polymarket.py tests/test_polymarket_weather.py
git commit -m "discovery: partition tagged markets by city, cached once per process"
```

---

### Task 4: Wire it in and repair the fallback

**Files:**
- Modify: `src/polymarket_weather/fetch_polymarket.py` (`search_markets_by_query`, `fetch_weather_markets`)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `discover_by_tag` (Task 3), `extract_market_snapshot(market, city) -> dict` (existing, unchanged).
- Produces: `search_markets_by_query(query: str, limit: int = 100) -> list[dict]` with independent-term matching; `fetch_weather_markets(city: str) -> list[dict]` unchanged in signature and return shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_search_matches_keyword_and_term_independently(monkeypatch):
    """The fallback was 3/4 dead. Queries were built as f"{kw} {term}" and matched as a literal
    substring, but real questions read "highest temperature IN London" — so
    "highest temperature London" never matched anything, and three of four keywords merely paged
    2100 markets for nothing. A fallback that does not work is not a fallback."""
    import fetch_polymarket as fp
    page = [{"question": "Will the highest temperature in London be 22°C on July 29?"},
            {"question": "Will the highest temperature in Paris be 30°C on July 29?"}]
    calls = []

    def fake_get(url, params=None):
        calls.append(params["offset"])
        return page if params["offset"] == 0 else []

    monkeypatch.setattr(fp, "_get", fake_get)
    got = fp.search_markets_by_query("highest temperature London")
    assert len(got) == 1, "keyword and city must match independently, not as one substring"
    assert "London" in got[0]["question"]


def test_fetch_weather_markets_uses_the_tag(monkeypatch):
    """Return shape is unchanged: a list of snapshot dicts from extract_market_snapshot."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {"London": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C?",
         "active": True, "closed": False, "endDateIso": "2026-07-29",
         "startDateIso": "2026-07-27", "volume": 1.0, "volume24hr": 1.0, "liquidity": 1.0,
         "clobTokenIds": '["123"]', "outcomePrices": '["0.5", "0.5"]'}]})
    out = fp.fetch_weather_markets("London")
    assert len(out) == 1
    assert out[0]["city"] == "London"
    assert out[0]["condition_id"] == "a"
    assert out[0]["clob_token_ids"] == ["123"]


def test_fetch_weather_markets_falls_back_when_the_tag_yields_nothing(monkeypatch):
    """If Polymarket retags these markets, tag discovery returns nothing for a city that had
    plenty. That must fall back to the query scan and warn — not silently collect zero."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {})
    used = []
    monkeypatch.setattr(fp, "search_markets_by_query",
                        lambda q, limit=100: used.append(q) or [])
    fp.fetch_weather_markets("London")
    assert used, "fallback did not fire when the tag returned nothing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "matches_keyword_and_term or fetch_weather_markets" -v`
Expected: FAIL — `search_markets_by_query` returns 0 matches (substring bug), and `fetch_weather_markets` still calls the query scan.

- [ ] **Step 3: Write the implementation**

In `src/polymarket_weather/fetch_polymarket.py`, replace the matching line inside `search_markets_by_query`'s page loop:

```python
        # BEFORE:  if q_lower in (m.get("question") or "").lower():
        # Queries arrive as "highest temperature London", but questions read "highest temperature
        # IN London" — as one substring that never matches. Require each whitespace-separated
        # token instead, so the fallback actually finds things.
        for m in page:
            question = (m.get("question") or "").lower()
            if all(tok in question for tok in q_lower.split()):
                matched.append(m)
```

Then replace the body of `fetch_weather_markets` above the snapshot conversion:

```python
def fetch_weather_markets(city: str) -> list[dict[str, Any]]:
    """
    Main function: find all temperature/weather markets for *city*
    and return a list of snapshot dicts (one per matching market).

    Discovery is tag-based (one /events enumeration per process, partitioned across cities). The
    old query scan remains as a fallback, because depending on a third-party tag with no backup
    would mean silently collecting nothing if Polymarket ever retags these markets. It is a
    DEGRADED path: GET /markets 422s at offset 2100, so it can only ever see the top ~2100 active
    markets by volume, and weather markets sit below that line until near expiry.
    """
    found: dict[str, dict] = {}

    for m in discover_by_tag().get(city, []):
        cid = m.get("conditionId", "")
        if cid:
            found[cid] = m

    if not found:
        logger.warning("Tag discovery returned no markets for %s — falling back to the query "
                       "scan. If this persists, the Polymarket 'weather' tag may have changed.",
                       city)
        city_cfg = CITIES.get(city, {})
        search_terms = city_cfg.get("search_terms", [city])
        for kw in MARKET_KEYWORDS:
            for term in search_terms:
                for m in search_markets_by_query(f"{kw} {term}"):
                    cid = m.get("conditionId", "")
                    if cid and cid not in found:
                        found[cid] = m
                time.sleep(0.3)

    if not found:
        logger.warning("No Polymarket temperature markets found for %s.", city)
        return []

    logger.info("Found %d unique markets for %s.", len(found), city)

    snapshots = []
    for market in found.values():
        snap = extract_market_snapshot(market, city)
        snapshots.append(snap)

    return snapshots
```

- [ ] **Step 4: Run the targeted tests, then the full suite**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "matches_keyword_and_term or fetch_weather_markets" -v`
Expected: PASS (3 passed)

Run: `pytest -o addopts="" tests/ -q`
Expected: PASS — every pre-existing test plus the 12 added by this plan.

- [ ] **Step 5: Verify against the live API and record the real before/after**

The spec's "~10 markets per cycle today" is extrapolated from measuring London only. Record the actual figure:

```bash
cd src/polymarket_weather
python -c "
import fetch_polymarket as fp
d = fp.discover_by_tag()
for c, ms in sorted(d.items()):
    print(f'{c:16} {len(ms)}')
print('total', sum(len(v) for v in d.values()))
"
```
Expected: five cities, roughly 264 markets total (London ~66, Seoul ~66, Hong Kong ~66, New York City ~44, Chicago ~22 as measured 2026-07-29). If a city is missing or near zero, stop — that is a `match_city` failure, not a quiet day.

- [ ] **Step 6: Commit**

```bash
git add src/polymarket_weather/fetch_polymarket.py tests/test_polymarket_weather.py
git commit -m "discovery: tag-based market discovery with repaired query-scan fallback"
```

---

## After the plan

The first collect cycle after this lands should show a large jump in markets per cycle and, over the following days, a fall in the share of markets with exactly one snapshot (44% today, median 2). Check with:

```bash
python -c "
import pandas as pd, glob
c = pd.concat([pd.read_csv(f, low_memory=False).groupby('condition_id').size()
               for f in glob.glob('data/polymarket/*_snapshots.csv')])
print('markets:', len(c), '| exactly one snapshot: %.0f%%' % (100*(c==1).mean()),
      '| median', int(c.median()))
"
```

Note that snapshot CSVs are committed perishable data, so the repository will start growing substantially faster — the accepted cost of the scope decision in the spec.
