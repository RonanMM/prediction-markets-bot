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


def _load_book(path=_OUT) -> pd.DataFrame:
    if Path(path).exists():
        df = pd.read_csv(path)
        for c in _BCOLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    return pd.DataFrame(columns=_BCOLS)


def scan_and_record_breadth(bins=None, now_utc=None, out_path=_OUT, fetch=get_json) -> int:
    """Record shoulder-sell / favorite-buy paper entries across all cities, deduped on
    (condition_id, leg). Pre-day is tz-free: hours-to-end > PREDAY_HOURS. Returns count added.
    No liquidity gate (paper book — a liquidity filter belongs only at go-live)."""
    if bins is None:
        bins = fetch_weather_bins(fetch=fetch)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    book = _load_book(out_path)
    known = set(zip(book["condition_id"], book["leg"]))
    added = []
    for r in bins:
        end = r["end"]
        if pd.isna(end):
            continue
        yes = float(r["yes"])
        hours_to_end = (pd.Timestamp(end) - pd.Timestamp(now_utc)).total_seconds() / 3600.0
        tgt = parse_question_date(r["question"], pd.Timestamp(end).tz_localize(None)) \
            or pd.Timestamp(end).date()
        base = {"entered_at_utc": now_utc.isoformat(), "city": r["city"],
                "condition_id": r["condition_id"], "market_id": r["market_id"],
                "question": r["question"], "target_date": str(tgt),
                "entry_yes_price": round(yes, 4), "liquidity": round(float(r["liquidity"]), 2),
                "settled_outcome": ""}
        cid = r["condition_id"]
        if (cid, "shoulder") not in known and hours_to_end > PREDAY_HOURS \
                and BAND_LO <= yes < BAND_HI:
            added.append({**base, "leg": "shoulder", "side": "No",
                          "entry_side_price": round(1.0 - yes, 4),
                          "band": "core" if yes >= CORE_LO else "outer"})
            known.add((cid, "shoulder"))
        if (cid, "favorite") not in known and hours_to_end > FAV_MIN_HOURS_TO_END \
                and FAV_LO <= yes < FAV_HI:
            added.append({**base, "leg": "favorite", "side": "Yes",
                          "entry_side_price": round(yes, 4),
                          "band": "fav_core" if yes < FAV_CORE_HI else "fav_outer"})
            known.add((cid, "favorite"))
    if added:
        book = pd.concat([book, pd.DataFrame(added)], ignore_index=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        book.reindex(columns=_BCOLS).to_csv(out_path, index=False)
    return len(added)


def _market_dict(resp):
    if isinstance(resp, list):
        return resp[0] if resp else None
    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        return resp["data"][0] if resp["data"] else None
    return resp if isinstance(resp, dict) else None


def settlement_outcome(market_id, fetch=get_json):
    """Resolved outcome of a market from Polymarket's own settlement: 1 (YES won), 0 (NO won),
    or None if not yet settled. Reads the terminal pinned outcomePrices via /markets/{id}."""
    if not market_id or str(market_id) in ("", "nan", "None"):
        return None
    resp = fetch(f"{GAMMA}/markets/{market_id}", None, "Gamma")
    mk = _market_dict(resp)
    if not mk or not mk.get("closed"):
        return None
    try:
        prices = [float(x) for x in json.loads(mk.get("outcomePrices", "[]"))]
    except Exception:
        return None
    if not prices or max(prices) < 1.0 - _PIN:
        return None
    return 1 if prices[0] >= 1.0 - _PIN else 0


def _is_unset(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) \
        or str(v).strip() in ("", "nan", "None")


def grade_book(book=None, out_path=_OUT, fetch=get_json) -> pd.DataFrame:
    """Fill+freeze settled_outcome for any ungraded entry (looked up once, never re-fetched),
    persist, and return the graded frame with side_won + net_edge (verified taker-fee model)."""
    if book is None:
        book = _load_book(out_path)
    if book.empty:
        return book
    changed = False
    for i, r in book.iterrows():
        if _is_unset(r.get("settled_outcome")):
            o = settlement_outcome(r.get("market_id"), fetch=fetch)
            if o is not None:
                book.at[i, "settled_outcome"] = o
                changed = True
    if changed:
        book.reindex(columns=_BCOLS).to_csv(out_path, index=False)
    graded = book[book["settled_outcome"].map(lambda v: not _is_unset(v))].copy()
    if graded.empty:
        return graded
    graded["settled_outcome"] = graded["settled_outcome"].astype(float).astype(int)
    graded["side_won"] = (graded["settled_outcome"] == 1) == (graded["side"] == "Yes")
    graded["net_edge"] = _net_edge(graded["side_won"], graded["entry_side_price"].astype(float))
    return graded


def _leg_line(graded, mask, label):
    sub = graded[mask]
    if sub.empty:
        print(f"  {label}: 0 graded")
        return
    print(f"  {label}: n={len(sub)}  win={sub['side_won'].mean():.1%}  "
          f"net taker={sub['net_edge'].mean():+.4f}/share")


def report_breadth(out_path=_OUT, fetch=get_json) -> None:
    """Grade (freeze new settlements) and print the breadth legs + the pre-registered
    Leg 1b moderate-shoulder forward gate."""
    book = _load_book(out_path)
    if book.empty:
        print("BREADTH structure book: no paper entries yet.")
        return
    graded = grade_book(book=book, out_path=out_path, fetch=fetch)
    ncities = book["city"].nunique()
    print(f"BREADTH STRUCTURE PAPER BOOK: {len(book)} entries across {ncities} cities "
          f"({len(graded)} graded, {len(book) - len(graded)} awaiting settlement)")
    if graded.empty:
        return
    yes = graded["entry_yes_price"].astype(float)
    sh = graded["leg"] == "shoulder"
    _leg_line(graded, sh, "Leg1 shoulder [5,35)")
    _leg_line(graded, sh & (yes >= CORE_LO), "Leg1 core   [20,35)")
    _leg_line(graded, graded["leg"] == "favorite", "Leg2 favorite [65,85)")
    st = moderate_gate_stats(graded, prereg_date=BREADTH_PREREG_DATE)
    if st:
        c, f = st["context"], st["forward"]
        need_n, need_e = GATE_MOD_BREADTH
        mark = "PASS" if f.get("gate_pass") else "pending"
        print(f"  Leg1b moderate [10,25) — pre-registered {BREADTH_PREREG_DATE}")
        print(f"    context (all graded):  n={c['n']}  wr {c['wr']:.1%}  taker {c['taker']:+.4f}")
        print(f"    FORWARD gate:          n={f['n']}/{need_n}  taker {f['taker']:+.4f} "
              f"v +{need_e:.3f}  [{mark}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Breadth structure paper book (all weather cities)")
    ap.add_argument("--record", action="store_true", help="scan live markets and record entries")
    a = ap.parse_args()
    if a.record:
        print(f"recorded {scan_and_record_breadth()} new breadth entries")
    else:
        report_breadth()
