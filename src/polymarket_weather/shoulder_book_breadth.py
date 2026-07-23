"""shoulder_book_breadth.py — the model-free structure paper book (shoulder sell + favorite
buy), extended to EVERY active Polymarket weather city.

Discovery is live from Gamma (weather tag). Grading is against Polymarket's OWN settlement
(the resolved market's terminal pinned outcome via /markets/{id}) — for a model-free
structure trade the market's settlement IS the P&L ground truth, so no weather-truth feed
or forecast model is needed. Entries are append-only + deduped on (condition_id, leg) in
output/shoulder_paper_breadth.csv. Bands and the taker-fee model are imported unchanged
from shoulder_book; the moderate [10,25)¢ gate is separately PRE-REGISTERED
(BREADTH_PREREG_DATE) so the 5-city stream is never contaminated. No real orders, no edge
claim — a clean forward measurement across the full city universe.

CLI:  python shoulder_book_breadth.py --record   # scan live markets, record entries
      python shoulder_book_breadth.py             # grade + report
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from http_util import get_json
from pmf import parse_question_date
from shoulder_book import (BAND_LO, BAND_HI, CORE_LO, FAV_LO, FAV_HI, FAV_CORE_HI,
                           FAV_MIN_HOURS_TO_END, _net_edge, moderate_gate_stats)

GAMMA = "https://gamma-api.polymarket.com"
BREADTH_PREREG_DATE = "2026-07-23"          # forward clock (go-live)
GATE_MOD_BREADTH    = (80, 0.03)            # (min forward graded, min mean net taker $/share)
PREDAY_HOURS        = 24                     # tz-free "pre-day": > this many hours to market end
_PIN                = 0.03                   # terminal price within this of 0/1 counts as settled

_OUT = Path(__file__).resolve().parent.parent.parent / "output" / "shoulder_paper_breadth.csv"
_BCOLS = ["entered_at_utc", "city", "condition_id", "market_id", "question", "target_date",
          "entry_yes_price", "liquidity", "leg", "side", "entry_side_price", "band",
          "settled_outcome"]

_TITLE = re.compile(r"^(Highest|Lowest) temperature in (.+?) on (.+?)\??$")


def parse_event_title(title: str):
    """('max'|'min', city, date_str) from a temperature-event title, or None."""
    m = _TITLE.match((title or "").strip())
    if not m:
        return None
    return ("max" if m.group(1) == "Highest" else "min"), m.group(2).strip(), m.group(3).strip()


def _yes_of(mk: dict):
    try:
        op = json.loads(mk.get("outcomePrices", "[]"))
        return float(op[0]) if op else None
    except Exception:
        return None


def bins_from_event(event: dict) -> list[dict]:
    """Flatten a temperature event into per-bin dicts; skip bins with unparseable price."""
    p = parse_event_title(event.get("title", ""))
    if not p:
        return []
    kind, city, date_str = p
    end = pd.to_datetime(event.get("endDate"), utc=True, errors="coerce")
    out = []
    for mk in event.get("markets", []):
        yes = _yes_of(mk)
        if yes is None:
            continue
        try:
            liq = float(mk.get("liquidityNum") or mk.get("liquidity") or 0.0)
        except Exception:
            liq = 0.0
        out.append(dict(condition_id=mk.get("conditionId"), market_id=str(mk.get("id")),
                        city=city, kind=kind, date_str=date_str,
                        question=mk.get("question") or event.get("title", ""),
                        yes=yes, liquidity=liq, end=end))
    return out


def fetch_weather_events(fetch=get_json) -> list[dict]:
    """All active weather-tag temperature events (paged, injectable fetcher)."""
    out, off = [], 0
    while True:
        page = fetch(f"{GAMMA}/events",
                     {"tag_slug": "weather", "active": "true", "closed": "false",
                      "limit": 100, "offset": off}, "Gamma")
        page = page if isinstance(page, list) else (page or {}).get("data", [])
        if not page:
            break
        out += [e for e in page if parse_event_title(e.get("title", ""))]
        if len(page) < 100:
            break
        off += 100
        if off > 3000:          # safety cap
            break
    return out


def fetch_weather_bins(fetch=get_json) -> list[dict]:
    """Flatten every active temperature event into per-bin dicts."""
    bins = []
    for ev in fetch_weather_events(fetch=fetch):
        bins.extend(bins_from_event(ev))
    return bins
