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

from fetch_orderbook import summarize_book
from kalshi_series import kalshi_get, LIVE_STATUSES

logger = logging.getLogger(__name__)

MARKET_COLS = [
    "fetched_at_utc", "city", "series_ticker", "ticker", "event_ticker", "title",
    "status", "result", "floor_strike", "cap_strike", "strike_type", "yes_sub_title",
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
    """Resolve floor_strike/cap_strike + strike_type into an explicit YES range, cross-checked
    against the human-readable subtitle. Returns None for an unrecognised strike_type, or a
    recognised one missing the field it needs — never a guess.

    The three live Kalshi strike types are NOT symmetric (verified against live KXHIGHLAX
    2026-08-03, 200 markets: 34 greater / 34 less / 132 between — `between` is the dominant
    type and floor_strike alone, the original design, silently dropped 83% of rows):
        greater: floor_strike is EXCLUSIVE -> yes_from = floor + 1, yes_to = None
        less:    cap_strike   is EXCLUSIVE -> yes_from = None,      yes_to = cap - 1
        between: floor_strike AND cap_strike are BOTH INCLUSIVE -> yes_from = floor, yes_to = cap

    The off-by-one on the exclusive types is exactly the shape of the Hong Kong ruler bug (a
    whole-degree bin compared against a tenths-rounded reading, so every market graded NO behind
    a passing audit). `agrees_with_subtitle` makes a disagreement visible instead of silently
    picking a side; for `between` BOTH subtitle numbers are checked against BOTH bounds.
    """
    st = market.get("strike_type")
    if st not in ("greater", "less", "between"):
        return None

    def _strike(key):
        try:
            return float(market[key])
        except (KeyError, TypeError, ValueError):
            return None

    floor, cap = _strike("floor_strike"), _strike("cap_strike")

    if st == "greater":
        if floor is None:
            return None
        yes_from, yes_to = floor + 1, None
    elif st == "less":
        if cap is None:
            return None
        yes_from, yes_to = None, cap - 1
    else:  # between
        if floor is None or cap is None:
            return None
        yes_from, yes_to = floor, cap

    sub = str(market.get("yes_sub_title") or "")
    digits = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in sub).split()
    nums = [float(d) for d in digits]
    sub_bound = nums[0] if nums else None
    sub_bound2 = nums[1] if len(nums) >= 2 else None

    if st == "greater":
        agrees = sub_bound is not None and sub_bound == yes_from
    elif st == "less":
        agrees = sub_bound is not None and sub_bound == yes_to
    else:  # between: both numbers must match both bounds
        agrees = sub_bound is not None and sub_bound2 is not None \
            and sub_bound == yes_from and sub_bound2 == yes_to

    return {"op": st, "floor_strike_f": floor, "cap_strike_f": cap,
            "yes_from_f": yes_from, "yes_to_f": yes_to,
            "subtitle_bound": sub_bound, "subtitle_bound2": sub_bound2,
            "agrees_with_subtitle": agrees}


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
        "cap_strike": _num(market, "cap_strike"),
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


BOOK_COLS = ["fetched_at_utc", "city", "ticker",
             "yes_best_bid", "yes_best_ask", "yes_ask_depth_usdc", "yes_vwap_buy_100",
             "no_best_bid", "no_best_ask", "no_ask_depth_usdc", "no_vwap_buy_100"]


def _ladder(levels, invert: bool = False) -> list:
    """Kalshi's `[price, size]` string pairs -> the `{"price", "size"}` dicts summarize_book wants.

    Kalshi's `/orderbook` endpoint publishes only BID ladders: `yes_dollars` and `no_dollars`
    are BOTH resting bids, and there is no ask ladder anywhere in the payload — a bid of `p` for
    NO *is* an offer to sell YES at `1 - p` (Kalshi nets the two representations onto one book).
    Verified live 2026-08-03 against KXHIGHLAX-26AUG03-B80.5: max(yes_dollars price) == 0.04 ==
    that market's own yes_bid_dollars, and 1 - max(no_dollars price) == 0.05 == yes_ask_dollars.

    `invert=True` performs exactly that conversion, reconstructing one token's ASKS from the
    OTHER token's bids: price -> 1 - price, size unchanged.

    A level that fails to parse is skipped, not raised — but if `levels` is non-empty and EVERY
    level fails, that is a payload SHAPE CHANGE, not a legitimately empty book, and must be
    visible rather than silently read as "no orders" (this is exactly how the original
    dict-shaped assumption would have failed: silently, plausibly, and wrong).
    """
    out = []
    bad = 0
    for lvl in levels or []:
        try:
            price, size = float(lvl[0]), float(lvl[1])
        except (IndexError, KeyError, TypeError, ValueError):
            bad += 1
            continue
        out.append({"price": round(1 - price, 4) if invert else price, "size": size})
    if levels and bad == len(levels):
        logger.error("kalshi ladder: all %d levels failed to parse — this is a SHAPE CHANGE, "
                     "not an empty book; investigate before trusting any book from this run",
                     len(levels))
    return out


def fetch_orderbooks(tickers: list, session=None) -> dict:
    """{ticker: summary} for every book that returned. One request per ticker.

    Reuses fetch_orderbook.summarize_book so BOTH venues are analysed by one code path — the
    Polymarket work established that a mid without a book is misleading, and that reading one
    side of a two-sided market gives a confidently wrong answer.

    Kalshi nests its ladders under `orderbook_fp` as `yes_dollars` / `no_dollars`, each a list of
    `[price, size]` pairs — BOTH are bid ladders (see `_ladder`), so each token's asks are
    reconstructed from the OTHER token's bids before handing them to summarize_book, which then
    sees a normal-shaped two-sided book for each side.

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
        yes_bids_raw = ob.get("yes_dollars") or []
        no_bids_raw = ob.get("no_dollars") or []
        yes = summarize_book({"bids": _ladder(yes_bids_raw),
                              "asks": _ladder(no_bids_raw, invert=True)})
        no = summarize_book({"bids": _ladder(no_bids_raw),
                             "asks": _ladder(yes_bids_raw, invert=True)})
        out[t] = {
            "yes_best_bid": yes["best_bid"], "yes_best_ask": yes["best_ask"],
            "yes_ask_depth_usdc": yes["ask_depth_usdc"], "yes_vwap_buy_100": yes["vwap_buy_100"],
            "no_best_bid": no["best_bid"], "no_best_ask": no["best_ask"],
            "no_ask_depth_usdc": no["ask_depth_usdc"], "no_vwap_buy_100": no["vwap_buy_100"],
        }
    logger.info("kalshi order books: %d/%d returned", len(out), len(tickers))
    return out
