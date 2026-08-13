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

import datetime as _dt
import json
import logging

from fetch_orderbook import summarize_book
from kalshi_series import kalshi_get, LIVE_STATUSES

logger = logging.getLogger(__name__)

def _num(market: dict, key: str):
    """float(market[key]) or None. Absence is None; a real 0.0 stays 0.0."""
    v = market.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_field(market: dict, key: str):
    """A nested vendor structure (list/dict) as a compact JSON string, or None.

    CSV cannot hold a list, and `str(obj)` would write a Python repr that only `ast.literal_eval`
    can read back. JSON round-trips.
    """
    v = market.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return None


# ── The market field table — capture EVERYTHING ───────────────────────────────
# One ordered table drives `summarize_market`, `MARKET_COLS` and `CAPTURED_VENDOR_KEYS`, so the
# three can never drift apart. `(archive column, vendor key, reader)`.
#
# WHY EVERY FIELD. Kalshi serves market objects for ~2 MONTHS; a field not captured now is
# unrecoverable at any later date and at any price. The original table kept 26 of the 45 keys a
# real market object carries and dropped, among others, **`expiration_value`** — the CLI daily
# high THE RESOLVING VENUE ITSELF PUBLISHED (present on 188/188 finalized markets, verified live
# 2026-08-03/04; e.g. '79.00' on KXHIGHLAX-26AUG02-T85). In a project that has shipped seven
# wrong-ruler defects, an independent settlement reading from the venue is the highest-value
# field in the payload — it cross-checks our own IEM/CLI truth feed. Curation is a LATER
# decision; capture is a now-or-never one.
#
# Readers:
#   "num"      -> _num: numeric strings ("0.0400", "255516.05") -> float. Absence stays None.
#   "raw"      -> stored verbatim exactly as the vendor sent it.
#   "json"     -> nested list/dict -> compact JSON string.
#   "blank_is_absent" -> "" means UNSETTLED, not an empty value (144/5,256 archived `result`
#                        rows are ""). ONLY `result` behaves this way; see the docstring test.
_MARKET_FIELDS = (
    # ── the original 26, in their original archive order (header stability) ──
    ("ticker",                   "ticker",                   "raw"),
    ("event_ticker",             "event_ticker",             "raw"),
    ("title",                    "title",                    "raw"),
    ("status",                   "status",                   "raw"),
    ("result",                   "result",                   "blank_is_absent"),
    ("floor_strike",             "floor_strike",             "num"),
    ("cap_strike",               "cap_strike",               "num"),
    ("strike_type",              "strike_type",              "raw"),
    ("yes_sub_title",            "yes_sub_title",            "raw"),
    ("yes_bid",                  "yes_bid_dollars",          "num"),
    ("yes_ask",                  "yes_ask_dollars",          "num"),
    ("yes_bid_size",             "yes_bid_size_fp",          "num"),
    ("yes_ask_size",             "yes_ask_size_fp",          "num"),
    ("no_bid",                   "no_bid_dollars",           "num"),
    ("no_ask",                   "no_ask_dollars",           "num"),
    ("last_price",               "last_price_dollars",       "num"),
    ("previous_price",           "previous_price_dollars",   "num"),
    ("volume",                   "volume_fp",                "num"),
    ("volume_24h",               "volume_24h_fp",            "num"),
    ("open_interest",            "open_interest_fp",         "num"),
    ("liquidity",                "liquidity_dollars",        "num"),
    ("open_time",                "open_time",                "raw"),
    ("close_time",               "close_time",               "raw"),
    ("expiration_time",          "expiration_time",          "raw"),
    ("rules_primary",            "rules_primary",            "raw"),
    ("rules_secondary",          "rules_secondary",          "raw"),
    # ── the 19 that were being dropped ──
    # The settlement reading Kalshi itself published, kept BOTH ways ON PURPOSE: the raw string
    # is the audit copy (a coercion failure must never silently blank the one field that
    # cross-checks our truth feed), `_f` is the usable number.
    ("expiration_value",         "expiration_value",         "raw"),
    ("expiration_value_f",       "expiration_value",         "num"),
    ("settlement_ts",            "settlement_ts",            "raw"),
    ("settlement_value",         "settlement_value_dollars", "num"),
    ("previous_yes_bid",         "previous_yes_bid_dollars", "num"),
    ("previous_yes_ask",         "previous_yes_ask_dollars", "num"),
    ("no_sub_title",             "no_sub_title",             "raw"),
    ("market_type",              "market_type",              "raw"),
    ("created_time",             "created_time",             "raw"),
    ("updated_time",             "updated_time",             "raw"),
    ("expected_expiration_time", "expected_expiration_time", "raw"),
    ("latest_expiration_time",   "latest_expiration_time",   "raw"),
    ("occurrence_datetime",      "occurrence_datetime",      "raw"),
    ("notional_value",           "notional_value_dollars",   "num"),
    ("price_ranges",             "price_ranges",             "json"),
    ("price_level_structure",    "price_level_structure",    "raw"),
    ("can_close_early",          "can_close_early",          "raw"),
    ("early_close_condition",    "early_close_condition",    "raw"),
    ("settlement_timer_seconds", "settlement_timer_seconds", "num"),
    ("exchange_index",           "exchange_index",           "num"),
)

# Every vendor key this module knows about.
#
# ⚠️ WHAT THE TEST AROUND THIS SET CAN AND CANNOT DO.
# `test_summarize_market_captures_every_key_a_real_market_object_carries` diffs this set against
# a FROZEN LITERAL SNAPSHOT of the live key union (captured 2026-08-04). Both sides are static,
# so it catches a CODE regression — a field deleted from `_MARKET_FIELDS`, or a table entry
# naming a key Kalshi does not send — and it fails the build for those. It CANNOT see a field
# Kalshi ADDS after that snapshot was taken, because no test here may touch the network. A
# static test cannot detect a change in a remote system; only something that reads a live
# payload can.
#
# That gap is closed at the only layer that sees live payloads: `summarize_market` compares each
# real market object against this set at runtime and logs any unknown key at WARNING (see
# `_warn_unknown_vendor_keys`). So a vendor addition surfaces in the COLLECTOR LOG, not in CI.
CAPTURED_VENDOR_KEYS = frozenset(key for _, key, _ in _MARKET_FIELDS)

# Unknown vendor keys already reported, so the warning fires once per key per process rather
# than once per market (438 markets x 7 cities would bury it). Same pattern as pmf._WARNED_NU.
_WARNED_UNKNOWN_KEYS: set = set()


def _warn_unknown_vendor_keys(market: dict) -> None:
    """Log any vendor key `_MARKET_FIELDS` does not name. Once per key per process.

    THE LIVE HALF of the completeness guard. The static test cannot see a field Kalshi adds, so
    without this a new vendor field would be silently dropped on every cycle for as long as it
    took someone to re-probe the API by hand — and the ~2-month serving window means "silently
    dropped" is "permanently lost". The collector runs hourly against real payloads, so it is the
    one place the addition is actually observable.
    """
    unknown = set(market) - CAPTURED_VENDOR_KEYS - _WARNED_UNKNOWN_KEYS
    if not unknown:
        return
    _WARNED_UNKNOWN_KEYS.update(unknown)
    logger.warning(
        "kalshi %s: NEW VENDOR FIELD(S) NOT BEING ARCHIVED: %s. Kalshi serves market objects "
        "for only ~2 months, so every cycle this goes unfixed is data lost permanently — add "
        "them to fetch_kalshi._MARKET_FIELDS and refresh the test's frozen key union.",
        market.get("ticker") or "<no ticker>", sorted(unknown))

# What the writer actually produces: summarize_market's keys, then the three context columns
# main.py attaches. Order matches the committed archive header (I9) — the constant describes the
# real file, it does not merely resemble it.
MARKET_COLS = [col for col, _, _ in _MARKET_FIELDS] + [
    "fetched_at_utc", "city", "series_ticker",
]

# ── Static per-ticker text, stored ONCE in a dimension table ───────────────────────────
#
# These four fields are constant for the life of a ticker but were re-serialised on EVERY
# hourly snapshot — and each ticker is snapshotted ~103 times. Measured on the real
# san_francisco_markets.csv (50,748 rows, 100.3 MB): they accounted for **71.3 MB, 74% of all
# field bytes**, `rules_secondary` alone being 44.5 MB across exactly ONE distinct value.
#
# That growth pushed the file past GitHub's HARD 100 MB per-file limit on 2026-08-12, and the
# pre-receive hook then rejected every collect push: TEN consecutive failed runs, ~16 hours of
# perishable market snapshots lost, with six more cities at 80-100 MB right behind it. The size
# warnings had been printed on every push for days; nothing read them, because the runs were
# green until the moment the limit was crossed.
#
# NOTHING IS DROPPED — this only stops writing the same text 103 times. That distinction matters
# here: `_MARKET_FIELDS` above documents why every vendor field is captured (Kalshi serves market
# objects for only ~2 months, so an uncaptured field is unrecoverable at any price). Normalising
# preserves that guarantee; curating would break it.
META_COLS = ["rules_primary", "rules_secondary", "early_close_condition", "price_ranges"]


def split_market_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split snapshot rows into (fact rows, dimension rows) — see META_COLS.

    The dimension row carries `ticker` plus the static text, and is deduped on ticker AND
    content (see main.py). Keying on content as well as ticker is deliberate: if Kalshi ever
    edits a market's rules mid-life, a ticker-only key would silently keep whichever version
    landed first and discard the amendment, which is exactly the kind of quiet loss this
    archive exists to prevent. Keyed on content, an edit simply appends a second row and both
    survive.
    """
    facts, meta = [], []
    for r in rows:
        facts.append({k: v for k, v in r.items() if k not in META_COLS})
        if not r.get("ticker"):
            continue
        vals = {c: r.get(c) for c in META_COLS if c in r and r.get(c) not in (None, "")}
        # An all-empty meta row is skipped rather than archived. The vendor omits these static
        # fields on ~0.9% of snapshots; writing that row would append a spurious ticker-with-no-
        # text entry that the content-keyed dedupe treats as a distinct variant, so the
        # dimension table would carry two rows per ticker for no information.
        if vals:
            meta.append({"ticker": r["ticker"], **vals})
    return facts, meta


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
    # A token can be a bare "-" (e.g. a subtitle using a dash as a separator: "80 - 82"), which
    # float() raises on. A malformed subtitle must degrade to "no bound to cross-check", never
    # take down the capture of a market whose strikes parsed fine.
    nums = []
    for d in digits:
        try:
            nums.append(float(d))
        except ValueError:
            continue
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
    """One archive row from a Kalshi market object — EVERY vendor field, none dropped.

    Driven by `_MARKET_FIELDS` (see `CAPTURED_VENDOR_KEYS` for why capture is total). Adding a
    column is a one-line table edit.

    Completeness is enforced in TWO places, because neither alone is sufficient:
      • `test_summarize_market_captures_every_key_a_real_market_object_carries` compares the
        table against a frozen snapshot of the live key union — it fails the build when a field
        is dropped from the CODE, but is blind to a field Kalshi ADDS (a static test cannot
        observe a remote change).
      • `_warn_unknown_vendor_keys`, called below on every real payload, logs at WARNING any key
        the table does not name — that is the half which can see a vendor addition, and it runs
        hourly in the collector.

    BOTH rules fields are stored: the station is named in `rules_primary` for the older KXHIGH*
    series and only in `rules_secondary` for the newer KXHIGHT* ones, so neither alone identifies
    it — and "Houston" is ambiguous between Bush and Hobby.
    """
    _warn_unknown_vendor_keys(market)
    readers = {"num": _num, "json": _json_field,
               "raw": lambda m, k: m.get(k),
               "blank_is_absent": lambda m, k: m.get(k) or None}
    return {col: readers[kind](market, key) for col, key, kind in _MARKET_FIELDS}


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
        # A RENAMED ladder key is invisible to _ladder's shape guard: `ob.get("yes_dollars") or
        # []` yields [] on a rename, which _ladder reads as a legitimately empty book, so every
        # book archives all-None while the run log still cheerfully says "12/12 returned". Only
        # a check at THIS level — the payload is present but names neither ladder we know — can
        # tell a renamed key from a genuinely two-sided-empty book.
        if ob and not ({"yes_dollars", "no_dollars"} & set(ob)):
            logger.error("kalshi orderbook %s: `orderbook_fp` is present but carries NEITHER "
                         "`yes_dollars` nor `no_dollars` (keys: %s) — this is a KEY RENAME, not "
                         "an empty book. Every book this run will archive as all-None while the "
                         "count below still reports success; fix the key names before trusting "
                         "any of it.", t, sorted(ob))
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


CANDLE_PERIOD_MINUTES = 60      # period_interval=1 returns HTTP 400 once a multi-day window
                                 # exceeds Kalshi's 5000-candlestick cap; 60 (hourly) stays well
                                 # under it for any market's whole life. Verified live 2026-08-03:
                                 # accepted period_interval values are {1, 60, 1440} only.
CANDLE_MARGIN_SECONDS = 3600    # a little slack either side of the market's life

CANDLE_COLS = ["city", "series_ticker", "ticker", "end_period_ts",
               "open_dollars", "high_dollars", "low_dollars", "close_dollars", "mean_dollars",
               "previous_dollars",
               "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
               "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
               "volume", "open_interest"]


CANDLE_LOG_COLS = ["ticker", "start_ts", "end_ts", "candles", "ok", "reason",
                   "city", "series", "fetched_at_utc"]


def candle_log_row(meta: dict, city: str, series: str, fetched_at_utc: str) -> dict:
    """One append-only candle-completeness record. See CANDLE_LOG_COLS.

    `fetch_candles` has always COMPUTED completeness; until now the caller read `meta["ok"]` and
    threw the rest away. Persisting it is what makes the binding "incompleteness is a recorded
    VALUE, not an inference" constraint true rather than aspirational: with this log a market
    archived with ZERO candles is distinguishable from one never attempted, and a transport
    failure is distinguishable from "this market never traded". Without it all three look
    identical — an absent row in the candles file.
    """
    return {"ticker": meta.get("ticker"), "start_ts": meta.get("start_ts"),
            "end_ts": meta.get("end_ts"), "candles": int(meta.get("candles") or 0),
            "ok": bool(meta.get("ok")), "reason": meta.get("reason") or "",
            "city": city, "series": series, "fetched_at_utc": fetched_at_utc}


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

    Candles are the ONLY route to Kalshi's ~2-month backfill — snapshots accumulate forward only
    and can never recover the past.

    ⚠️ The window is derived from the market's own open_time/close_time, never from a trailing
    "last N days". A trailing window against a market that settled outside it returns zero
    candles at every interval, which reads as "this market never traded". That mistake was made
    live while drafting the spec against KXHIGHNY-26JUL21-B79.5, a market with $181k of volume.

    `meta` records the window requested and the candle count, so a market archived with zero
    candles is distinguishable from one never attempted:
        {"ticker", "start_ts", "end_ts", "candles", "ok", "reason"}
    ok=False (transport failure) must never look like "this market had no trading" — the caller
    reads `ok` before `candles`, exactly like kalshi_get itself.
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
    """One archive row from a Kalshi candlestick — the FULL OHLC of all three sub-books.

    `price`/`yes_bid`/`yes_ask` are DICTS keyed by `*_dollars` (verified live 2026-08-03/04
    against a settled $181k-volume market) — never bare scalars. All three carry a full
    open/high/low/close; `price` additionally carries `mean_dollars` and a carry-forward
    `previous_dollars`. Keeping only `close_dollars` from the two quote books discarded the
    intra-hour bid/ask range — the only record of how wide the spread actually got — and Kalshi
    serves this history for ~2 months, so it is not refetchable later.

    ⚠️ `yes_bid_close == 0.0` MUST BE READ AS "NO BID", not as a price of zero. That is Kalshi's
    real no-bid encoding: across all 116,995 committed candle rows `yes_bid_close` is non-null in
    every single one, and 37,856 of them are exactly 0.0. Storing the vendor sentinel raw is the
    correct capture-time choice (never destroy what the venue said); DERIVING the "no bid"
    meaning is the reader's job, and this comment is the other half of that contract.

    A missing KEY is still absence: `_num` reads it as None, never 0.0 or a KeyError.
    """
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
        "previous_dollars": _num(price, "previous_dollars"),
        "yes_bid_open": _num(bid, "open_dollars"),
        "yes_bid_high": _num(bid, "high_dollars"),
        "yes_bid_low": _num(bid, "low_dollars"),
        "yes_bid_close": _num(bid, "close_dollars"),
        "yes_ask_open": _num(ask, "open_dollars"),
        "yes_ask_high": _num(ask, "high_dollars"),
        "yes_ask_low": _num(ask, "low_dollars"),
        "yes_ask_close": _num(ask, "close_dollars"),
        "volume": _num(candle, "volume_fp"),
        "open_interest": _num(candle, "open_interest_fp"),
    }
