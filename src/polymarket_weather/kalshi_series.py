"""kalshi_series.py — Kalshi series discovery and the archive health manifest.

Kalshi's daily-temperature markets resolve on the NWS Climatological Report; Polymarket's on
wunderground.com. Those are DIFFERENT RULERS (CLI >= WU on 99.40% of 1,668 station-days at KLGA,
mean +0.66 F), so a Kalshi leg never hedges a Polymarket leg. Kalshi is an INFORMATION source
here — nothing in this module trades.

TARGETS ARE STATIC; ROT DETECTION IS DYNAMIC. `target_series()` reads a HARDCODED per-city
ticker out of `resolution_anchors` — nothing in this module calls Kalshi's `/series/` endpoint,
and no discovery happens here. That is deliberate: the capture set must equal the Polymarket
capture set exactly (see the venue-symmetry test), and a discovered set could not guarantee that.

Static targets are exposed to TICKER ROT, which is real. Verified 2026-08-03: HIGHNY, HIGHCHI,
HIGHAUS, HIGHMIA, KXHIGHHOU, KXHIGHOU and KXHOUHIGH all still ENUMERATE but serve zero markets;
Houston has four tickers of which only KXHIGHTHOU is live, and HIGHNY -> KXHIGHNY shows a
completed migration. So every cycle records what it found in `data/kalshi/series_manifest.csv`,
and a series that previously served markets and now serves none is an ERROR, not an absence of
news.

⚠️ THE LIMIT OF THAT ALARM: it detects that a series went dead, but it CANNOT surface the
replacement name — that requires enumerating `/series/`, which nothing here does. A rot alarm is
therefore a prompt for a human to find the new ticker and edit `resolution_anchors`, not a
self-healing mechanism. Until that edit lands, the city is not being captured.
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

MANIFEST_COLS = ["fetched_at_utc", "series_ticker", "city",
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

    Parsing uses strict=False as a DEFENSIVE guard against an UNOBSERVED condition, not as a
    workaround for observed behaviour. An earlier claim here — that `rules_secondary` contains
    raw newlines inside JSON strings and so is invalid JSON — is FALSE and was carried as
    "verified live". Re-verified 2026-08-04: all seven series' raw bodies parse with plain
    `json.loads`, no flags; Kalshi escapes newlines as `\\n`, which is valid JSON, and the only
    literal newline in a response body is the trailing one AFTER the JSON document. The flag is
    kept because it only ever WIDENS what parses — it costs nothing and would absorb a genuine
    unescaped control character if Kalshi ever emitted one — but nothing here should be read as
    evidence that Kalshi sends malformed JSON.
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


def manifest_row(series_ticker: str, city: str, markets_returned: int, live_markets: int,
                 truncated: bool, fetched_at_utc: str) -> dict:
    """One append-only health record. See MANIFEST_COLS.

    The second parameter was called `title` and every caller passed the CITY into it, so the
    committed manifest's `title` column held "Los Angeles" — a column whose name disagreed with
    its content. The city is what we actually want (it keys the manifest to the archive files),
    so the PARAMETER and the COLUMN are both renamed to `city` rather than the caller being
    changed to pass a real series title. The existing archive's `title` header was renamed in
    place at the same time: the content was already the city, so the rename is lossless and
    leaves no permanently-NaN column behind.
    """
    return {"fetched_at_utc": fetched_at_utc, "series_ticker": series_ticker, "city": city,
            "markets_returned": int(markets_returned), "live_markets": int(live_markets),
            "truncated": bool(truncated)}
