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
    for city, cfg in CITIES.items():
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


# ── Gamma API ────────────────────────────────────────────────────────────────

def search_markets_by_query(query: str, limit: int = 100) -> list[dict]:
    url = f"{GAMMA_API_BASE}/markets"
    q_lower = query.lower()
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

        page: list[dict] = data if isinstance(data, list) else data.get("data", [])
        if not page:
            break

        for m in page:
            if q_lower in (m.get("question") or "").lower():
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
    """
    city_cfg  = CITIES.get(city, {})
    search_terms = city_cfg.get("search_terms", [city])

    found: dict[str, dict] = {}   # condition_id → market

    for kw in MARKET_KEYWORDS:
        for term in search_terms:
            query = f"{kw} {term}"
            logger.info("Searching Polymarket for: '%s'", query)
            markets = search_markets_by_query(query)
            for m in markets:
                cid = m.get("conditionId", "")
                if cid and cid not in found:
                    found[cid] = m
            time.sleep(0.3)   # be polite to the API

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
