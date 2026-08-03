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
