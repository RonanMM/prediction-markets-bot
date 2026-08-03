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
