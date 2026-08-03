"""
fetch_polymarket.py — Fetches weather-temperature markets from Polymarket.

Uses the Gamma API (read-only, no auth) to:
  • Search for markets matching temperature keywords per city
  • Extract outcomes, probabilities, volume, timestamps
  • Augment with CLOB price history for time-series tracking
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

from config import (
    CLOB_API_BASE,
    GAMMA_API_BASE,
    MARKET_KEYWORDS,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    CITIES,
    ALL_CITIES,
)

logger = logging.getLogger(__name__)


# ── Question classification ──────────────────────────────────────────────────

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
    # ALL_CITIES, not CITIES: capture-tier cities must be discovered and filed too. CITIES is
    # deliberately modelled-only (see config.py) and using it here would find the seven overlap
    # cities and then silently drop them.
    for city, cfg in ALL_CITIES.items():
        for term in cfg.get("search_terms", [city]):
            if re.search(rf"\b{re.escape(term)}\b", q, re.I):
                return city
    return None


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with exponential-backoff retry.  Returns None on failure."""
    from http_util import get_json          # F5: shared retry client
    return get_json(url, params, "Polymarket")


# ── Pagination ───────────────────────────────────────────────────────────────

_MAX_EVENT_PAGES = 100   # 10k events at page_size=100; measured reality is ~216. Generous
                         # headroom, but BOUNDED: an API that ignores `offset` and re-serves
                         # page 1 forever would otherwise hang the hourly collector.


def _paged_events(tag_slug: str, page_size: int = 100) -> tuple[list[dict], bool]:
    """Page GET /events for one tag. Returns (events, truncated).

    `_get` collapses every failure to None, so the pager cannot ask WHY a page was empty and has
    to infer:

        full page (== page_size)   -> keep going
        short page (< page_size)   -> legitimate end of list
        None after a full page     -> TRUNCATED: warn loudly, return what we have
        None on the first page     -> endpoint down: caller falls back
        _MAX_EVENT_PAGES reached   -> TRUNCATED: cap enforced, return what we have

    The distinction is the entire point of this change. GET /markets returns 422 at offset 2100;
    the old pager read that None as "last page" and stopped, so a hard ceiling on how much of
    Polymarket we could even see was indistinguishable from having seen all of it.
    """
    url = f"{GAMMA_API_BASE}/events"
    events: list[dict] = []
    offset = 0
    page_count = 0
    while True:
        if page_count >= _MAX_EVENT_PAGES:
            logger.warning(
                "TRUNCATED: /events?tag_slug=%s hit page cap at %d pages after %d events. "
                "Returning a PARTIAL list — do not treat this as the full set.",
                tag_slug, _MAX_EVENT_PAGES, len(events))
            return events, True
        page = _get(url, {"tag_slug": tag_slug, "limit": page_size,
                          "offset": offset, "closed": "false", "active": "true"})
        if page is None:
            if events:
                logger.warning(
                    "TRUNCATED: /events?tag_slug=%s failed at offset %d after %d events. "
                    "Returning a PARTIAL list — do not treat this as the full set.",
                    tag_slug, offset, len(events))
                return events, True
            logger.warning("/events?tag_slug=%s unavailable at offset 0", tag_slug)
            return [], False
        if not isinstance(page, (list, dict)):
            # A body that is neither list nor dict (e.g. a bare string like "rate limited")
            # must degrade to a clean stop, not an AttributeError out of .get() — see
            # search_markets_by_query for the twin fix and why this matters for the collector.
            logger.warning(
                "/events?tag_slug=%s returned a non-list/non-dict body at offset %d "
                "(%s) — treating as an empty page.", tag_slug, offset, type(page).__name__)
            page = []
        elif not isinstance(page, list):
            page = page.get("data", []) or []
        if not page:
            return events, False
        events.extend(page)
        page_count += 1
        if len(page) < page_size:
            return events, False
        offset += page_size
        time.sleep(0.2)   # be polite to the API


# ── Gamma API ────────────────────────────────────────────────────────────────

def search_markets_by_query(query: str, limit: int = 100) -> list[dict]:
    url = f"{GAMMA_API_BASE}/markets"
    matched = []
    offset = 0
    page_size = 100

    while True:
        params = {
            "limit":     page_size,
            "offset":    offset,
            "order":     "volume",
            "ascending": "false",
            "active":    "true",
            "closed":    "false",
        }
        data = _get(url, params)
        if not data:
            break

        if isinstance(data, list):
            page: list[dict] = data
        elif isinstance(data, dict):
            page = data.get("data", [])
        else:
            # A truthy body that is neither list nor dict (e.g. a bare string like
            # "rate limited") must degrade to a clean stop, not an AttributeError that
            # propagates out of fetch_weather_markets -> step_fetch_polymarket -> main.py,
            # which has no try/except and would abort the whole collect cycle before
            # step_fetch_weather/step_fetch_ensemble even run.
            logger.warning(
                "GET %s returned a non-list/non-dict body (%s) at offset %d — "
                "treating as an empty page.", url, type(data).__name__, offset)
            page = []
        if not page:
            break

        # Queries arrive as "highest temperature London", but questions read "highest temperature
        # IN London" — as one substring that never matches. Require each whitespace-separated
        # token instead, so the fallback actually finds things. WORD-BOUNDARY matching (not bare
        # substring) mirrors match_city: a substring match would let "temperature in New York"
        # match "...temperature in New Yorker Magazine's HQ city..." and "highest temperature
        # London" match "...in Londonderry...".
        for m in page:
            question = m.get("question") or ""
            if all(re.search(rf"\b{re.escape(tok)}\b", question, re.I) for tok in query.split()):
                matched.append(m)

        if len(page) < page_size:
            break   # last page

        offset += page_size
        if offset > 2000:   # safety cap — ~20 pages
            break
        time.sleep(0.2)

    return matched

# ── CLOB price history ───────────────────────────────────────────────────────

def _epoch_to_utc_iso(ts) -> str:
    """ISO-8601 UTC timestamp from a CLOB history epoch.

    The CLOB API returns `t` in epoch SECONDS. The original code assumed milliseconds and
    divided by 1000, which collapsed every 2026 timestamp onto 1970-01-21 in the stored
    price-history CSVs (repaired by backfill_schema.repair_price_history_timestamps).
    Values > 1e12 can only be milliseconds, so those are still handled, defensively.
    """
    ts = float(ts)
    if ts > 1e12:  # milliseconds
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def fetch_price_history(token_id: str, interval: str = "1d") -> list[dict]:
    """
    Fetch CLOB price history for a token.
    interval choices: 1h, 6h, 1d, 1w, 1m
    Returns list of {t: unix_ts_ms, p: price} dicts.
    """
    url  = f"{CLOB_API_BASE}/prices-history"
    params = {"market": token_id, "interval": interval, "fidelity": 60}
    data = _get(url, params)
    if not data:
        return []
    return data.get("history", [])


# ── Core extraction ──────────────────────────────────────────────────────────

def _parse_outcome_prices(market: dict) -> dict[str, float]:
    """
    Gamma encodes outcomePrices as a JSON-string array, e.g. '["0.60","0.40"]'
    and outcomes as a JSON-string array, e.g. '["Yes","No"]'.
    Returns {outcome_label: probability}.
    """
    raw_outcomes = market.get("outcomes", "[]")
    raw_prices   = market.get("outcomePrices", "[]")

    try:
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        prices   = json.loads(raw_prices)   if isinstance(raw_prices,   str) else raw_prices
    except (json.JSONDecodeError, TypeError):
        return {}

    return {
        str(o): float(p)
        for o, p in zip(outcomes, prices)
        if p is not None
    }


def extract_market_snapshot(market: dict, city: str) -> dict[str, Any]:
    """Turn a raw Gamma market object into a clean snapshot dict."""
    outcome_probs = _parse_outcome_prices(market)

    raw_token_ids = market.get("clobTokenIds", []) or []
    if isinstance(raw_token_ids, str):
        try:
            raw_token_ids = json.loads(raw_token_ids)
        except (json.JSONDecodeError, TypeError):
            raw_token_ids = []
    # Keep every non-empty token id. Polymarket CLOB token ids are *decimal* strings
    # (e.g. "71047..."), NOT 0x-prefixed hex — an earlier `startswith("0x")` filter
    # silently dropped all of them, so no price-history/order-book series was ever stored.
    clob_token_ids: list[str] = [
        str(t) for t in raw_token_ids
        if t is not None and str(t).strip() != ""
    ]

    # Volumes
    volume      = float(market.get("volume",      0) or 0)
    volume_24h  = float(market.get("volume24hr",  0) or 0)
    liquidity   = float(market.get("liquidity",   0) or 0)

    return {
        "fetched_at_utc":  datetime.now(timezone.utc).isoformat(),
        "city":            city,
        "condition_id":    market.get("conditionId", ""),
        "question":        market.get("question", ""),
        "active":          market.get("active", False),
        "closed":          market.get("closed", False),
        "end_date_iso":    market.get("endDateIso", ""),
        "start_date_iso":  market.get("startDateIso", ""),
        "volume_usdc":     volume,
        "volume_24h_usdc": volume_24h,
        "liquidity_usdc":  liquidity,
        "outcome_probs":   outcome_probs,      # {label: prob}
        "clob_token_ids":  clob_token_ids,
        "market_slug":     market.get("marketMakerAddress", ""),
    }


# ── Public entry point ───────────────────────────────────────────────────────

def fetch_weather_markets(city: str) -> list[dict[str, Any]]:
    """
    Main function: find all temperature/weather markets for *city*
    and return a list of snapshot dicts (one per matching market).

    Discovery is tag-based (one /events enumeration per process, partitioned across cities). The
    old query scan remains as a fallback, because depending on a third-party tag with no backup
    would mean silently collecting nothing if Polymarket ever retags these markets. It is a
    DEGRADED path: GET /markets 422s at offset 2100, so it can only ever see the top ~2100 active
    markets by volume, and weather markets sit below that line until near expiry. The fallback
    also re-applies is_temperature_question/match_city to every hit: search_markets_by_query only
    guarantees every query token appears somewhere in the question, so a question that happens to
    contain every token without being ABOUT this city's temperature (e.g. "Will Bitcoin's high be
    reached in Chicago while the London Stock Exchange sets a temperature record?") would
    otherwise be snapshotted under the wrong city — a mislabeled row appended permanently to the
    committed CSVs, since data/polymarket/ is never overwritten.
    """
    found: dict[str, dict] = {}

    try:
        tagged = discover_by_tag().get(city, [])
    except Exception as exc:                      # noqa: BLE001 — deliberate: a malformed
        # /events body must degrade discovery for THIS city only, not kill the whole collect
        # run for the other four cities sharing this process.
        logger.warning("Tag discovery raised for %s (%s: %s) — falling back to the query "
                       "scan. A malformed /events body must degrade this ONE city, not kill "
                       "the whole collect run for every city.", city, type(exc).__name__, exc)
        tagged = []

    for m in tagged:
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
                    q = m.get("question") or ""
                    if not is_temperature_question(q) or match_city(q) != city:
                        continue
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


def fetch_price_history_for_market(snap: dict, interval: str = "1d") -> list[dict]:
    """
    Given a snapshot dict, fetch CLOB price history for every token.
    Returns list of {token_id, outcome, t_utc, price} dicts.
    """
    records = []
    probs   = snap.get("outcome_probs", {})
    outcomes = list(probs.keys())

    for i, token_id in enumerate(snap.get("clob_token_ids", [])):
        outcome_label = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
        history = fetch_price_history(token_id, interval)
        for point in history:
            ts_raw = point.get("t")
            price = point.get("p")
            if ts_raw is None or price is None:
                continue
            records.append({
                "condition_id":   snap["condition_id"],
                "city":           snap["city"],
                "question":       snap["question"],
                "token_id":       token_id,
                "outcome":        outcome_label,
                "timestamp_utc":  _epoch_to_utc_iso(ts_raw),
                "price":          float(price),
            })
        time.sleep(0.2)

    return records


# One pagination per process. fetch_weather_markets is called once per city, and the old code
# paid 44 full paginations per collect cycle for that; the tag is global, so it is fetched once
# and partitioned. Collect runs as a fresh process each cycle, so process lifetime IS the
# correct cache lifetime — no TTL needed.
_TAG_CACHE: dict[str, dict[str, list[dict]]] = {}


def _reset_tag_cache() -> None:
    """Clear the per-process tag cache. Exists so tests are not order-dependent."""
    _TAG_CACHE.clear()


def _matching_cities(question: str) -> list[str]:
    """Every configured city whose search terms appear in *question*, not just the first.

    `match_city` deliberately keeps returning the FIRST match in ALL_CITIES order (its contract
    is unchanged here — reordering ALL_CITIES would silently change that answer, and 0/2747 real
    questions have hit this so far, so there is no data yet to justify picking a real
    disambiguation rule). This helper exists only so `discover_by_tag` can DETECT the ambiguous
    case and skip it loudly instead of silently filing it under whichever city sorts first.
    """
    q = question or ""
    matches = []
    for city, cfg in ALL_CITIES.items():
        for term in cfg.get("search_terms", [city]):
            if re.search(rf"\b{re.escape(term)}\b", q, re.I):
                matches.append(city)
                break
    return matches


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
            candidates = _matching_cities(q)
            if len(candidates) > 1:
                # An ambiguous market (names >=2 configured cities) is worth less than a
                # mislabeled one: filing it under match_city's first-match winner would be a
                # SILENT, permanent corruption of that city's committed CSV series. Skipping it
                # loudly converts a latent silent-corruption bug into a visible, cheap event —
                # three lines now, versus discovering it later in a data audit. Measured 0/2747
                # real questions today, so this is not expected to fire in practice.
                logger.warning(
                    "tag '%s': ambiguous market matches multiple cities %s, skipping: %r",
                    tag_slug, candidates, q)
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

    # Only cache a non-empty, COMPLETE partition. Caching an empty result would freeze one
    # transient /events blip in for the rest of the process: every later city in the same
    # collect cycle would pay the expensive fallback instead of just retrying the tag lookup.
    # Caching a TRUNCATED partition is the same failure in a quieter shape: city 1 gets the
    # FLOOR warning above, but cities 2-5 would silently read a partial partition from cache —
    # `found` is non-empty for them, so fetch_weather_markets never falls back and nothing signals
    # the loss.
    if by_city and not truncated:
        _TAG_CACHE[tag_slug] = by_city
    return by_city
