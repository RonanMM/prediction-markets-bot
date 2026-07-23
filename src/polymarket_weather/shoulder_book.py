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
  Leg 1b moderate yes∈[10,25)¢: ≥80 graded FORWARD entries AND mean net taker ≥ +3¢/share
    (§10f, pre-reg 2026-07-23) — a report-time refinement of Leg 1. Discovered in-sample
    (n=109): the full band nets ~0 because it lumps the fat-tail-UNDER-priced deep band
    [5,10)¢ with the OVER-priced moderate band. Sells only the moderate band; gate counts
    only entries recorded on/after the pre-reg date, so it is never graded on the discovery
    data. Maker reported, not gated.

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

# Leg 1 MAKER gate — pre-registered 2026-07-15 (a NEW instrument; the taker gates above are
# UNCHANGED — not moved to fit a result). The maker backtest (`shoulder_book.py --backtest`) showed
# the conservative maker fill is itself adversely selected: a resting SELL-YES fills only when the
# price ticks back UP, i.e. on bins more likely to resolve YES — the losing side for a seller. That
# flips the FULL band [5,35)¢ NEGATIVE, but the CORE [20,35)¢ overpricing is large enough to survive
# it (in-sample reconstruction +7.6¢/share at ~31% fill). So the maker book is gated on the CORE
# ONLY; the full band carries no maker gate. Threshold shape mirrors the taker core gate (chosen by
# analogy, NOT tuned to the observed number). Leg 2 (favorite) has NO pre-day maker entries — it is
# a same-day phenomenon, deferred to a future same-day entry window (megaplan §10e follow-up).
GATE_CORE_MAKER = (80, 0.03)   # (min FILLED core entries, min mean maker net $/share)

# Leg 1b (moderate shoulder) — pre-registered 2026-07-23. A REPORT-TIME refinement of Leg 1,
# not a new recorded leg. Sell only the OVER-priced moderate band; the deep band [0.05,0.10) is
# fat-tail-UNDER-priced (excluded) and the core [0.25,0.35) is fair (excluded). Discovered
# in-sample (n=109, where the full band nets ~0 by lumping the two). The gate counts FORWARD
# entries only (entered_at_utc >= MOD_PREREG_DATE), so the hypothesis is never graded on the data
# that generated it. Taker-gated; maker reported, not gated. Threshold shape mirrors GATE_CORE
# (chosen by analogy, NOT tuned to the observed ~+3¢).
MOD_LO, MOD_HI  = 0.10, 0.25
MOD_PREREG_DATE = "2026-07-23"     # forward clock (UTC): gate counts entries entered_at_utc >= this
GATE_MOD        = (80, 0.03)       # (min graded FORWARD entries, min mean net taker $/share)

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


def _maker_net(side_won, side_price):
    """Per-share MAKER P&L for a FILLED resting order: no spread crossed (you set the price) and
    no taker fee (Polymarket makers pay zero on weather). The 25% maker rebate is left OUT — it is
    real upside, funded by the taker on the other side, but attributing it per-fill needs taker
    volume, so booking it to 0 keeps the number conservative. Only filled entries reach here."""
    return float(side_won) - float(side_price)


def maker_filled(side: str, entry_yes: float, later_yes) -> bool:
    """Conservative maker fill from 2-hourly snapshots: a resting order at the entry yes-price
    fills ONLY if a LATER snapshot trades THROUGH it from the taker side —
      SELL YES (Leg 1, side='No'):  a later yes-price ≥ entry  (a buyer lifts your ask)
      BUY  YES (Leg 2, side='Yes'): a later yes-price ≤ entry  (a seller hits your bid)
    Snapshots are sparse, so this UNDERstates fills (same-price touches between snapshots are
    missed) — deliberately, so every booked fill is one a later trade demonstrably supports. It
    also excludes the runaway winners (a shoulder that only ever falls, a favorite that only ever
    rises), biasing booked maker P&L DOWNWARD. Unfilled orders earn nothing."""
    later = pd.Series(list(later_yes), dtype=float)
    later = later[later.notna()]
    if later.empty:
        return False
    if side == "No":                          # short YES: filled only if price ticks back up
        return bool((later >= entry_yes - 1e-9).any())
    return bool((later <= entry_yes + 1e-9).any())   # long YES: filled only if price ticks down


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


def _price_paths(cids: set) -> dict:
    """condition_id -> forward yes-price path (t, yes), for conservative maker-fill checks."""
    snaps = _all_snapshots()
    if snaps.empty:
        return {}
    snaps = snaps[snaps["condition_id"].isin(cids)]
    return {cid: g.sort_values("t")[["t", "yes"]].reset_index(drop=True)
            for cid, g in snaps.groupby("condition_id")}


def moderate_gate_stats(graded: pd.DataFrame) -> dict:
    """Leg 1b — the moderate-shoulder [MOD_LO, MOD_HI) refinement of Leg 1, computed at REPORT
    TIME from existing shoulder entries (no new recording). Returns 'context' (all graded
    in-band, incl. the in-sample discovery sample) and 'forward' (entries entered on/after
    MOD_PREREG_DATE only — the pre-registered, forward-only gate). Taker-gated; maker reported,
    not gated. Returns {} if graded is empty or missing a required column.

    graded needs: entry_yes_price, entered_at_utc, side_won, entry_side_price (optional: maker_filled).
    """
    need = {"entry_yes_price", "entered_at_utc", "side_won", "entry_side_price"}
    if graded.empty or not need.issubset(graded.columns):
        return {}
    yes = graded["entry_yes_price"].astype(float)
    inband = graded[(yes >= MOD_LO) & (yes < MOD_HI)].copy()
    inband["_taker"] = _net_edge(inband["side_won"], inband["entry_side_price"].astype(float))
    entered = pd.to_datetime(inband["entered_at_utc"], utc=True, errors="coerce")
    prereg = pd.Timestamp(MOD_PREREG_DATE, tz="UTC")

    def _agg(sub: pd.DataFrame) -> dict:
        d = {"n": int(len(sub)),
             "wr": float(sub["side_won"].mean()) if len(sub) else 0.0,
             "taker": float(sub["_taker"].mean()) if len(sub) else 0.0,
             "maker_n": 0, "maker": 0.0}
        if "maker_filled" in sub.columns and len(sub):
            f = sub[sub["maker_filled"].astype(bool)]
            d["maker_n"] = int(len(f))
            if len(f):
                d["maker"] = float((f["side_won"].astype(float)
                                    - f["entry_side_price"].astype(float)).mean())
        return d

    need_n, need_e = GATE_MOD
    forward = _agg(inband[entered >= prereg])
    forward["gate_pass"] = bool(forward["n"] >= need_n and forward["taker"] >= need_e)
    return {"context": _agg(inband), "forward": forward}


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
    # Conservative maker fill per entry (needs the forward price path after the entry snapshot).
    paths = _price_paths(set(graded["condition_id"]))

    def _fill(r) -> bool:
        path = paths.get(r["condition_id"])
        if path is None:
            return False
        after = path[path["t"] > pd.to_datetime(r["entered_at_utc"], utc=True)]["yes"]
        return maker_filled(r["side"], float(r["entry_yes_price"]), after)

    graded["maker_filled"] = graded.apply(_fill, axis=1)

    def _line(name, sub, taker_gate, maker_gate) -> None:
        n = len(sub)
        if n == 0:
            print(f"  {name}: 0 graded")
            return
        te, wr = sub["net_edge"].mean(), sub["side_won"].mean()
        tstr = f"taker {te:+.3f}"
        if taker_gate:
            need_n, need_e = taker_gate
            tstr += ("  ✅TAKER-GATE" if (n >= need_n and te >= need_e)
                     else f" ({n}/{need_n}@{te:+.3f}v{need_e:+.3f})")
        filled = sub[sub["maker_filled"]]
        if len(filled):
            me = (filled["side_won"].astype(float) - filled["entry_side_price"].astype(float)).mean()
            mstr = f"maker {len(filled)}/{n} filled {me:+.3f}"
            if maker_gate:
                need_n, need_e = maker_gate
                mstr += ("  ✅MAKER-GATE" if (len(filled) >= need_n and me >= need_e)
                         else f" ({len(filled)}/{need_n}@{me:+.3f}v{need_e:+.3f})")
        else:
            mstr = "maker 0 filled"
        print(f"  {name}: n={n} wr {wr:.0%} | {tstr} | {mstr}")

    sh = graded[graded["leg"] == "shoulder"]
    fav = graded[graded["leg"] == "favorite"]
    # Full band: taker gate only — the maker fill is adversely selected here (see GATE_CORE_MAKER).
    _line("Leg1 shoulder full [5,35)¢", sh, GATE_FULL, None)
    _line("Leg1 shoulder core [20,35)¢", sh[sh["band"] == "core"], GATE_CORE, GATE_CORE_MAKER)
    # Leg 1b — moderate shoulder [0.10,0.25): report-time refinement, FORWARD-only gate.
    mod = moderate_gate_stats(sh)
    if mod:
        c, f = mod["context"], mod["forward"]
        need_n, need_e = GATE_MOD
        gate = ("✅MOD-GATE" if f["gate_pass"]
                else f"{f['n']}/{need_n}@{f['taker']:+.3f}v{need_e:+.3f}")
        print(f"  Leg1b moderate shoulder [10,25)¢ — pre-reg {MOD_PREREG_DATE}")
        print(f"    context (incl. in-sample): n={c['n']} wr {c['wr']:.0%} "
              f"taker {c['taker']:+.3f} | maker {c['maker_n']}/{c['n']} {c['maker']:+.3f}")
        print(f"    FORWARD gate (entered≥pre-reg): n={f['n']} taker {f['taker']:+.3f} "
              f"[{gate}] | maker {f['maker_n']}/{f['n']} {f['maker']:+.3f}")
    _line("Leg2 favorite core [65,75)¢", fav[fav["band"] == "fav_core"], GATE_FAV_CORE, None)
    _line("Leg2 favorite outer [75,85)¢", fav[fav["band"] == "fav_outer"], None, None)
    print("  (pre-registered gates; taker crosses spread + 0.05·p·(1−p) fee; maker = filled-only, "
          "no fee/spread, rebate excluded; conservative fill; no real orders until a gate passes)")


def _all_snapshots() -> pd.DataFrame:
    """Every snapshot across cities with parsed yes/end, for the historical maker backtest."""
    frames = []
    for slug, city in _SLUGS.items():
        p = _SNAP_DIR / f"{slug}_snapshots.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        s["t"] = pd.to_datetime(s["fetched_at_utc"], utc=True, format="mixed")
        s["yes"] = s["outcome_probs_json"].map(_yes_prob)
        s["end"] = pd.to_datetime(s["end_date_iso"], errors="coerce", utc=True).dt.tz_localize(None)
        s["city"] = city
        s = s.dropna(subset=["yes", "end"])
        frames.append(s[["city", "condition_id", "question", "t", "yes", "end", "liquidity_usdc"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backtest_maker() -> None:
    """Reconstruct both legs over the FULL snapshot history, apply the conservative maker fill,
    grade vs settlement truth, and compare MAKER (filled only, no fee/spread) vs TAKER economics.

    Entry = the FIRST in-band snapshot in the leg's eligibility window (shoulder: pre-day;
    favorite: >12h to local day end), liquidity ≥ MIN_LIQUIDITY. This is the historical analogue
    of the live paper book — it exists to size the maker edge NOW, before the forward gate."""
    snaps = _all_snapshots()
    if snaps.empty:
        print("maker backtest: no snapshots."); return
    rows = []
    for cid, g in snaps.groupby("condition_id"):
        g = g.sort_values("t").reset_index(drop=True)
        city, q = g["city"].iloc[0], g["question"].iloc[0]
        if not parse_question(q):
            continue
        pq = parse_question(q)
        tgt = parse_question_date(q, g["end"].iloc[0]) or g["end"].iloc[0].date()
        tz = ZoneInfo(_TZ[city])
        day_start = pd.Timestamp(tgt).tz_localize(tz).tz_convert("UTC")
        day_end = (pd.Timestamp(tgt) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
        y = resolves_yes(city, str(tgt), q, pq["temp_c"])
        if y is None:
            continue
        liq_ok = g["liquidity_usdc"].fillna(0.0) >= config.MIN_LIQUIDITY
        # Leg 1 — shoulder SELL YES (side No): first pre-day in-band snapshot
        pre = g[liq_ok & (g["t"] < day_start) & (g["yes"] >= BAND_LO) & (g["yes"] < BAND_HI)]
        if not pre.empty:
            e = pre.iloc[0]; p = float(e["yes"])
            filled = maker_filled("No", p, g[g["t"] > e["t"]]["yes"])
            rows.append({"leg": "shoulder", "band": "core" if p >= CORE_LO else "outer",
                         "side": "No", "side_price": 1.0 - p, "side_won": (y == 0),
                         "filled": filled})
        # Leg 2 — favorite BUY YES (side Yes): first >12h-to-end in-band snapshot
        h = (day_end - g["t"]).dt.total_seconds() / 3600.0
        fav = g[liq_ok & (h > FAV_MIN_HOURS_TO_END) & (g["yes"] >= FAV_LO) & (g["yes"] < FAV_HI)]
        if not fav.empty:
            e = fav.iloc[0]; p = float(e["yes"])
            filled = maker_filled("Yes", p, g[g["t"] > e["t"]]["yes"])
            rows.append({"leg": "favorite", "band": "fav_core" if p < FAV_CORE_HI else "fav_outer",
                         "side": "Yes", "side_price": p, "side_won": (y == 1),
                         "filled": filled})
    df = pd.DataFrame(rows)
    if df.empty:
        print("maker backtest: no gradable reconstructed entries."); return
    df["taker_net"] = _net_edge(df["side_won"], df["side_price"])
    print(f"\nMAKER-vs-TAKER BACKTEST (reconstructed over full history, graded vs settlement truth)")
    print(f"  conservative fill: a resting order fills only if a later snapshot trades through it\n")
    print(f"  {'leg / band':<26} {'n':>4} {'fill%':>6} {'taker net':>10} {'maker net(filled)':>18}")
    views = [("Leg1 shoulder [5,35)c", df["leg"] == "shoulder"),
             ("  core [20,35)c", (df["leg"] == "shoulder") & (df["band"] == "core")),
             ("Leg2 favorite [65,85)c", df["leg"] == "favorite"),
             ("  core [65,75)c", (df["leg"] == "favorite") & (df["band"] == "fav_core"))]
    for name, mask in views:
        sub = df[mask]
        if sub.empty:
            continue
        fills = sub[sub["filled"]]
        fill_rate = len(fills) / len(sub)
        taker = sub["taker_net"].mean()
        maker = (fills["side_won"].astype(float) - fills["side_price"]).mean() if len(fills) else float("nan")
        mk = f"{maker:+.3f}" if len(fills) else "  n/a"
        print(f"  {name:<26} {len(sub):>4} {fill_rate:>5.0%} {taker:>+10.3f} {mk:>18} (n={len(fills)})")
    print("  maker net EXCLUDES the 25% rebate (conservative upside); taker net crosses spread + 0.05·p·(1−p) fee")


def main():
    ap = argparse.ArgumentParser(description="Market-structure paper book (megaplan §10b/§10e).")
    ap.add_argument("--report", action="store_true", help="report only; record no new entries")
    ap.add_argument("--backtest", action="store_true",
                    help="historical maker-vs-taker backtest over the full snapshot history")
    args = ap.parse_args()
    if args.backtest:
        backtest_maker()
        return
    if not args.report:
        n = scan_and_record()
        print(f"recorded {n} new paper entries")
    report()


if __name__ == "__main__":
    main()
