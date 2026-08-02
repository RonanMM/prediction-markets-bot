"""fetch_orderbook.py — top-of-book and depth capture for weather markets.

WHY THIS EXISTS. Every structure result in this project is measured at the snapshot MID and
costed with `config.HALF_SPREAD`, a hard-coded 1¢ guess whose own comment says "tune once real
order-book data exists". That data never existed, so three separate findings have died on the
same unanswerable question:

  * the deep-shoulder band nets +3.2¢/share — but the empirical pass showed it dies at 3¢ of
    slippage, and nobody could say whether a 6.9¢ mid is sellable at 3.9¢;
  * weather books carry a +2.2¢ median overround — takeable only if Σ(best bid) > 1, which needs
    best bids, not mids;
  * the maker legs report higher net than the taker legs, but a maker only earns that if a
    resting order at the mid is actually inside the spread.

A mid is not a price. The very first book sampled while writing this module (a live London Tmin
bin) had **zero bids** and asks starting at 99.9¢ — a quoted "price" you could not sell into at
any size. This module records what is actually there.

WHAT IS STORED. A compact per-market summary, not the raw L2 (full books across ~264 markets
hourly would add ~19 MB/day to a git-backed store). Best bid/ask, resting USDC, and the
size-weighted price to fill a realistic clip — for BOTH tokens.

⚠️ BOTH SIDES, because YES and NO are separate books and the difference is not cosmetic. Leg 1
of the structure book SELLS YES, which is executed by BUYING NO. Measured live on 66 markets
2026-08-02: reading only the YES book says 71% cannot absorb a $100 sell, which looks
catastrophic; reading the NO book — the one that trade actually touches — says 27%. Same markets,
same instant. A one-sided capture would have produced a confident, wrong conclusion.

The round-trip cost falls straight out as `yes_best_ask + no_best_ask` (1.0 = free). Measured
median 1.0100, mean 1.0160 — so `config.HALF_SPREAD = 0.01` one-way is roughly 2x conservative
at the median. That is the number the constant's own comment has been waiting for.

NULLS ARE REAL ANSWERS. An empty side yields None, and a VWAP that cannot be filled yields None.
It never falls back to a sentinel like 1.0 — `data_loader.check_orderbook_vwap` does exactly that
and "no liquidity" becomes indistinguishable from "priced at 1.0", which is the silent-failure
shape this project has been bitten by seven times.

Usage:
    from fetch_orderbook import fetch_book_summaries
    summaries = fetch_book_summaries({cid: yes_token_id, ...})   # {condition_id: {...}}
"""

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

CLOB_BOOKS_URL = "https://clob.polymarket.com/books"
REQUEST_TIMEOUT = 30
BATCH_SIZE = 100          # tokens per POST; the endpoint accepts a list
MAX_RETRIES = 3

# Clip sizes (USDC) for the effective-price columns. 100 is a realistic small taker order;
# 500 is near the top of what these books support and is where slippage bites.
CLIP_SMALL_USDC = 100.0
CLIP_LARGE_USDC = 500.0

BOOK_COLS = [
    "book_ts",
    "yes_best_bid", "yes_best_ask", "yes_ask_depth_usdc", "yes_vwap_buy_100",
    "no_best_bid", "no_best_ask", "no_ask_depth_usdc", "no_vwap_buy_100",
]


def _levels(raw, reverse: bool):
    """[(price, size)] sorted best-first. `reverse=True` for bids (highest first)."""
    out = []
    for lvl in raw or []:
        try:
            p, s = float(lvl["price"]), float(lvl["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if p > 0 and s > 0:
            out.append((p, s))
    out.sort(key=lambda x: x[0], reverse=reverse)
    return out


def _vwap(levels, clip_usdc: float):
    """Size-weighted price to trade `clip_usdc` against `levels` (already best-first).

    Returns None when the book cannot fill the clip — the caller must not treat a partial fill
    as a price. Walking a book and returning the price of whatever you could get would understate
    slippage exactly when the book is thinnest, which is the case we care about.
    """
    if not levels or clip_usdc <= 0:
        return None
    spent = shares = 0.0
    for price, size in levels:
        avail = price * size
        if spent + avail >= clip_usdc:
            shares += (clip_usdc - spent) / price
            return round(clip_usdc / shares, 6) if shares else None
        spent += avail
        shares += size
    return None                       # book exhausted before the clip filled


def summarize_book(book: dict) -> dict:
    """Compact summary of ONE CLOB L2 book, side-agnostic. Pure — the caller owns the network.

    Keys are unprefixed (`best_bid`, `best_ask`, …); `fetch_book_summaries` prefixes them `yes_`
    or `no_`. Every field is independently nullable: a book may have asks and no bids (common
    here), or enough depth for a small clip but not a large one.
    """
    bids = _levels((book or {}).get("bids"), reverse=True)
    asks = _levels((book or {}).get("asks"), reverse=False)
    return {
        "best_bid": bids[0][0] if bids else None,
        "best_ask": asks[0][0] if asks else None,
        "ask_depth_usdc": round(sum(p * s for p, s in asks), 2) if asks else 0.0,
        # Buying this token = lifting its asks. That is the only direction a taker needs, and
        # the SELL of the other token is exactly this trade on the opposite book.
        "vwap_buy_100": _vwap(asks, CLIP_SMALL_USDC),
    }


def _post_books(token_ids: list[str], session=None) -> list[dict]:
    """One batched POST. Returns [] on failure — a missing book must never look like an empty one."""
    sess = session or requests
    payload = [{"token_id": str(t)} for t in token_ids]
    for attempt in range(MAX_RETRIES):
        try:
            resp = sess.post(CLOB_BOOKS_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"books batch failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")
            time.sleep(2 * (attempt + 1))
    return []


def _summaries_for(token_by_cid: dict, session=None) -> dict:
    """{condition_id: unprefixed summary} for one token side."""
    cid_by_token = {str(t): c for c, t in token_by_cid.items() if t}
    tokens = list(cid_by_token)
    out = {}
    for i in range(0, len(tokens), BATCH_SIZE):
        for book in _post_books(tokens[i:i + BATCH_SIZE], session=session):
            cid = cid_by_token.get(str(book.get("asset_id")))
            if cid:
                out[cid] = (summarize_book(book), book.get("timestamp"))
    return out


def fetch_book_summaries(tokens_by_cid: dict, session=None) -> dict:
    """{condition_id: {book_ts, yes_*, no_*}} for every market whose books came back.

    `tokens_by_cid` maps condition_id -> (yes_token_id, no_token_id); a bare token id is accepted
    and treated as YES-only. Best-effort by design: a market missing from the result is simply not
    annotated, so the collector is never blocked by a CLOB outage, and absent columns read as NaN
    downstream — the honest value, since we do not know that book's state. A market is included if
    EITHER side returned, with the missing side's fields left None.
    """
    if not tokens_by_cid:
        return {}
    yes_t, no_t = {}, {}
    for cid, tok in tokens_by_cid.items():
        if isinstance(tok, (tuple, list)):
            if len(tok) > 0 and tok[0]:
                yes_t[cid] = tok[0]
            if len(tok) > 1 and tok[1]:
                no_t[cid] = tok[1]
        elif tok:
            yes_t[cid] = tok

    y = _summaries_for(yes_t, session=session)
    n = _summaries_for(no_t, session=session)
    out = {}
    for cid in set(y) | set(n):
        ys, yts = y.get(cid, ({}, None))
        ns, nts = n.get(cid, ({}, None))
        row = {"book_ts": yts or nts}
        for k in ("best_bid", "best_ask", "ask_depth_usdc", "vwap_buy_100"):
            row[f"yes_{k}"] = ys.get(k)
            row[f"no_{k}"] = ns.get(k)
        out[cid] = row
    logger.info(f"order books: {len(out)}/{len(tokens_by_cid)} markets summarized "
                f"({len(y)} yes, {len(n)} no)")
    return out


def tokens_of(clob_token_ids_json):
    """(yes_token, no_token) from a snapshot's `clob_token_ids_json`.

    Polymarket convention: index 0 is YES, index 1 is NO. Returns (None, None) on anything
    unparseable — historical rows carry '[]' and some carry non-JSON.
    """
    try:
        ids = (json.loads(clob_token_ids_json)
               if isinstance(clob_token_ids_json, str) else clob_token_ids_json)
    except (json.JSONDecodeError, TypeError):
        return (None, None)
    if not isinstance(ids, (list, tuple)):
        return (None, None)
    return (str(ids[0]) if len(ids) > 0 else None,
            str(ids[1]) if len(ids) > 1 else None)


def yes_token_of(clob_token_ids_json) -> str | None:
    """The YES token id alone. Thin wrapper over `tokens_of` for callers that only buy YES."""
    return tokens_of(clob_token_ids_json)[0]
