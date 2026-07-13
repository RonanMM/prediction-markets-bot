"""shoulder_book.py — model-free market-STRUCTURE paper book (edge megaplan §10b/§10e).

Two legs, both pure price-structure (no forecast model involved), both discovered by calibrating
the market against SETTLEMENT-faithful truth over every bin (not just model-flagged ones):

LEG 1 — SHOULDER SELL (§10b, found 2026-07-12, n=885 pre-day bins):
  Bins priced 5–35¢ the day before are overpriced; the 20–35¢ core sells at ≈ +8¢/share
  (price 0.271 vs realized 0.190, n=147). Selling the YES = buying its NO. Entry: pre-day only.

LEG 2 — FAVORITE BUY (§10e, found 2026-07-13 testing the "easy wins are ignored" hypothesis,
  n=1,646 market-band rows): the 65–75¢ FAVORITE side realizes ≈ 0.81 pre-day (n=109) and
  ≈ 0.82 day-early (n=34) — underpriced by 7–12¢ raw, ≈ +7.7¢ net at real taker fees. High
  favorites (85–97¢) are FAIRLY priced (the naive version of the hypothesis is dead); the edge
  is specifically the modest-favorite band, the mirror image of Leg 1's shoulders. NO-side
  favorites at 65–75¢ are economically Leg 1's core (buying NO at 0.65–0.80), so Leg 2 records
  only YES-side favorites (yes ∈ [0.65, 0.85)) to avoid double-counting. Entry: while >12 h
  remain before the target local day ends (pre-day + early-day, the bands that tested positive;
  mid-day tested negative and is excluded).

Settlement uses the VERIFIED 2026-07-01 Polymarket schedule (config, E1 2026-07-13):
taker fee = 0.05·p·(1−p) per share (weather), plus HALF_SPREAD crossed on entry — i.e.
taker-conservative; maker fills would pay no fee and earn rebates.

PRE-REGISTERED FORWARD GATES (real orders only after a gate passes):
  Leg 1 full band  [5,35)¢ : ≥150 graded entries AND mean net edge ≥ +2¢/share
  Leg 1 core       [20,35)¢: ≥80  graded entries AND mean net edge ≥ +3¢/share
  Leg 2 core   yes∈[65,75)¢: ≥80  graded entries AND mean net edge ≥ +3¢/share
  (Leg 2 outer [75,85)¢ is recorded and reported but carries no gate.)

Usage (from src/polymarket_weather/; scan runs automatically each collector cycle via main.py):
    python shoulder_book.py            # record new entries, then print the report
    python shoulder_book.py --report   # report only
Entries are append-only in output/shoulder_paper.csv, deduped on (condition_id, leg); grading
goes through grading.resolves_yes — the settlement-faithful truth channel — automatically.
"""
import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config
from grading import resolves_yes
from pmf import parse_question, parse_question_date

_BASE = Path(__file__).resolve().parent
_SNAP_DIR = _BASE / "data" / "polymarket"
_OUT = _BASE / "output" / "shoulder_paper.csv"

_SLUGS = {"chicago": "Chicago", "new_york_city": "NYC", "london": "London",
          "seoul": "Seoul", "hong_kong": "HongKong"}
_TZ = {"Chicago": "America/Chicago", "NYC": "America/New_York", "London": "Europe/London",
       "Seoul": "Asia/Seoul", "HongKong": "Asia/Hong_Kong"}

# Leg 1 (shoulder sell) — pre-registered 2026-07-12
BAND_LO, BAND_HI = 0.05, 0.35     # sell YES priced in [lo, hi), pre-day only
CORE_LO = 0.20
GATE_FULL = (150, 0.02)           # (min graded entries, min mean net edge $/share)
GATE_CORE = (80, 0.03)
# Leg 2 (favorite buy) — pre-registered 2026-07-13
FAV_LO, FAV_CORE_HI, FAV_HI = 0.65, 0.75, 0.85   # buy YES priced in [FAV_LO, FAV_HI)
FAV_MIN_HOURS_TO_END = 12.0                       # only while >12h to local day end
GATE_FAV_CORE = (80, 0.03)

_SNAP_WINDOW_MIN = 60             # a "run" = rows within this window of the newest row

_COLS = ["entered_at_utc", "city", "condition_id", "question", "target_date",
         "leg", "side", "entry_side_price", "entry_yes_price", "band", "liquidity"]


def _yes_prob(j):
    if not isinstance(j, str):
        return None
    try:
        return json.loads(j).get("Yes")
    except Exception:
        try:
            return ast.literal_eval(j).get("Yes")
        except Exception:
            return None


def _load_book() -> pd.DataFrame:
    if not _OUT.exists():
        return pd.DataFrame(columns=_COLS)
    book = pd.read_csv(_OUT)
    # migrate rows written before Leg 2 existed (they were all shoulder NO-buys)
    if "leg" not in book.columns:
        book["leg"] = "shoulder"
        book["side"] = "No"
        book["entry_side_price"] = 1.0 - book["entry_yes_price"]
    return book.reindex(columns=_COLS)


def _net_edge(side_won: pd.Series, side_price: pd.Series) -> pd.Series:
    """Per-share paper P&L: win pays $1; entry crosses the half-spread and pays the REAL
    weather taker fee 0.05·p·(1−p) at the execution price (taker-conservative)."""
    exec_price = side_price + config.HALF_SPREAD
    fee = exec_price.map(config.taker_fee_per_share)
    return side_won.astype(float) - (exec_price + fee)


def scan_and_record() -> int:
    """Record paper entries for both legs from the latest snapshot run per city."""
    book = _load_book()
    known = set(zip(book["condition_id"], book["leg"]))
    added = []
    now_utc = datetime.now(timezone.utc)
    for slug, city in _SLUGS.items():
        path = _SNAP_DIR / f"{slug}_snapshots.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s["t"] = pd.to_datetime(s["fetched_at_utc"], utc=True, format="mixed")
        latest = s["t"].max()
        run = s[s["t"] >= latest - pd.Timedelta(minutes=_SNAP_WINDOW_MIN)].copy()
        run["yes"] = run["outcome_probs_json"].map(_yes_prob)
        run["end"] = pd.to_datetime(run["end_date_iso"], errors="coerce",
                                    utc=True).dt.tz_localize(None)
        run = run.dropna(subset=["yes", "end"])
        run = run.sort_values("t").groupby("condition_id").last().reset_index()
        for _, r in run.iterrows():
            if float(r.get("liquidity_usdc") or 0.0) < config.MIN_LIQUIDITY:
                continue
            pq = parse_question(r["question"])
            if not pq:
                continue
            tgt = parse_question_date(r["question"], r["end"]) or r["end"].date()
            tz = ZoneInfo(_TZ[city])
            day_start = pd.Timestamp(tgt).tz_localize(tz).tz_convert("UTC")
            day_end = (pd.Timestamp(tgt) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
            hours_to_end = (day_end - r["t"]).total_seconds() / 3600.0
            base = {"entered_at_utc": now_utc.isoformat(), "city": city,
                    "condition_id": r["condition_id"], "question": r["question"],
                    "target_date": str(tgt), "entry_yes_price": round(float(r["yes"]), 4),
                    "liquidity": round(float(r.get("liquidity_usdc") or 0.0), 2)}
            # Leg 1 — shoulder sell (buy NO), pre-day only
            if (r["condition_id"], "shoulder") not in known \
                    and r["t"] < day_start and BAND_LO <= r["yes"] < BAND_HI:
                added.append({**base, "leg": "shoulder", "side": "No",
                              "entry_side_price": round(1.0 - float(r["yes"]), 4),
                              "band": "core" if r["yes"] >= CORE_LO else "outer"})
                known.add((r["condition_id"], "shoulder"))
            # Leg 2 — YES-favorite buy, while >12h to day end
            if (r["condition_id"], "favorite") not in known \
                    and hours_to_end > FAV_MIN_HOURS_TO_END and FAV_LO <= r["yes"] < FAV_HI:
                added.append({**base, "leg": "favorite", "side": "Yes",
                              "entry_side_price": round(float(r["yes"]), 4),
                              "band": "fav_core" if r["yes"] < FAV_CORE_HI else "fav_outer"})
                known.add((r["condition_id"], "favorite"))
    if added:
        book = pd.concat([book, pd.DataFrame(added)], ignore_index=True)
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        book.reindex(columns=_COLS).to_csv(_OUT, index=False)
    return len(added)


def report() -> None:
    book = _load_book()
    if book.empty:
        print("structure book: no paper entries yet — run after a collector snapshot lands.")
        return
    rows = []
    for _, r in book.iterrows():
        pq = parse_question(r["question"])
        if not pq:
            continue
        y = resolves_yes(r["city"], r["target_date"], r["question"], pq["temp_c"])
        rows.append({**r, "outcome": None if y is None else int(y)})
    df = pd.DataFrame(rows)
    graded = df.dropna(subset=["outcome"]).copy()
    print(f"STRUCTURE PAPER BOOK: {len(df)} entries "
          f"({len(graded)} graded, {len(df) - len(graded)} awaiting truth)")
    if graded.empty:
        return
    graded["side_won"] = (graded["outcome"] == 1) == (graded["side"] == "Yes")
    graded["net_edge"] = _net_edge(graded["side_won"], graded["entry_side_price"].astype(float))
    views = [
        ("Leg1 shoulder full [5,35)¢", graded[graded["leg"] == "shoulder"], GATE_FULL),
        ("Leg1 shoulder core [20,35)¢",
         graded[(graded["leg"] == "shoulder") & (graded["band"] == "core")], GATE_CORE),
        ("Leg2 favorite core [65,75)¢",
         graded[(graded["leg"] == "favorite") & (graded["band"] == "fav_core")], GATE_FAV_CORE),
        ("Leg2 favorite outer [75,85)¢",
         graded[(graded["leg"] == "favorite") & (graded["band"] == "fav_outer")], None),
    ]
    for name, sub, gate in views:
        n = len(sub)
        if n == 0:
            print(f"  {name}: 0 graded")
            continue
        e = sub["net_edge"].mean()
        wr = sub["side_won"].mean()
        if gate is None:
            print(f"  {name}: n={n}  win rate {wr:.1%}  mean net edge {e:+.3f} $/share  (tracked, no gate)")
            continue
        need_n, need_e = gate
        status = ("✅ GATE PASSED" if (n >= need_n and e >= need_e)
                  else f"pending ({n}/{need_n} @ {e:+.3f} vs {need_e:+.3f})")
        print(f"  {name}: n={n}  win rate {wr:.1%}  mean net edge {e:+.3f} $/share  → {status}")
    print("  (pre-registered gates; taker-costed with the verified 0.05·p·(1−p) weather fee; "
          "no real orders until a gate passes — megaplan §10b/§10e)")


def main():
    ap = argparse.ArgumentParser(description="Market-structure paper book (megaplan §10b/§10e).")
    ap.add_argument("--report", action="store_true", help="report only; record no new entries")
    args = ap.parse_args()
    if not args.report:
        n = scan_and_record()
        print(f"recorded {n} new paper entries")
    report()


if __name__ == "__main__":
    main()
