"""fetch_crossvenue_minute.py — MINUTE-resolution prices for bins quoted on BOTH venues.

Kalshi and Polymarket list the same 2 °F temperature bins for the same US stations, and settle
them on DIFFERENT rulers (§13b). If one venue repriced before the other, that would be a
model-free edge — no forecast involved. It cannot currently be tested: the routine collector
samples Polymarket price history at `fidelity: 60` and Kalshi candles at `period_interval=60`, so
both series carry one point per hour and "who moved first inside the hour" has no observations at
all. An hourly-resolution attempt (2026-08-05) returned a contemporaneous correlation of −0.169 on
24 bins, which is not a dynamics result — two venues pricing the same bin cannot genuinely
anti-correlate — it is the signature of comparing stale hourly stamps.

So this module captures minute data, and ONLY where the test needs it:

  * the seven cities listed on both venues
  * bins that exist on BOTH venues for the same target date (matched on the Kalshi market's
    `floor_strike`, never on the ticker suffix — between-tickers encode the MIDPOINT, e.g.
    `-B99.5` for the 99-100 bin, which silently matches nothing against Polymarket's floor of 99)
  * target dates within NEAR_DAYS of today

Scope is the point, not an optimisation: minute data is ~60x the rows, and price history is
perishable, therefore committed. Capturing every bin of every city would add gigabytes a year to
the repo to answer a question that only lives in near-dated matched bins.

⚠️ Kalshi caps a candlestick response at 5000 rows and accepts period_interval ∈ {1, 60, 1440}
only (verified live 2026-08-03, see fetch_kalshi.CANDLE_PERIOD_MINUTES). At period_interval=1 a
window longer than ~3.5 days silently exceeds that cap, so minute windows are chunked and each
chunk's completeness is recorded rather than inferred.

    python fetch_crossvenue_minute.py
"""
from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from fetch_kalshi import kalshi_get, load_markets, markets_available
from processing import load_partitioned, partitioned_available
from fetch_polymarket import fetch_price_history
from processing import _append_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "crossvenue"

CITIES = ["austin", "atlanta", "houston", "miami", "seattle", "san_francisco", "los_angeles"]
NEAR_DAYS = 2                  # target dates within +/- this many days of today
MINUTE_WINDOW_HOURS = 48       # 2880 minute-candles per request, safely under Kalshi's 5000 cap
PM_FIDELITY = 1                # minutes

# Retention. This data MUST be committed — a fresh CI runner has no local state, so gitignoring it
# would mean it never accumulates across runs, which is the entire point.
#
# ⚠️ MEASURED 2026-08-06, correcting the estimate this comment originally carried. Actual cost is
# 187 bytes/row, not the ~106 assumed, and growth is ~79,000 rows/day:
#     RETAIN_DAYS = 21  ->  1.66M rows  ->  311 MB      (the original setting; ~2.4x the
#                                                        "~130 MB" first written here)
#     RETAIN_DAYS = 16  ->  1.27M rows  ->  237 MB
# Against a repo already carrying 460 MB of data and 593 MB of history, 311 MB is material, so the
# window is now 16: the lead-lag test needs 14 distinct dates, and 2 days of slack is enough to
# absorb a late collector run without buying a third of a gigabyte of headroom nobody uses.
#
# ⚠️ AND RETENTION BOUNDS THE WORKING TREE, NOT THE HISTORY. Every hourly commit stores a new blob
# per changed file; appends delta-compress well, but `prune` REWRITES whole files, so each prune
# cycle costs a full copy that deltas poorly. Pruning caps what a checkout weighs; it does not cap
# what the repo weighs. That is the reason this experiment needs an end date rather than a
# retention window — see docs/EDGE_MEGAPLAN.md §13f.
RETAIN_DAYS = 16

_PM_BIN = re.compile(r"between (\d+)-(\d+)°F on (\w+ \d+)", re.I)

KAL_COLS = ["city", "ticker", "floor_strike", "target_date", "end_period_ts",
            "open_dollars", "high_dollars", "low_dollars", "close_dollars",
            "yes_bid_close", "yes_ask_close", "volume", "fetched_at_utc"]
PM_COLS = ["city", "condition_id", "token_id", "floor_strike", "target_date",
           "timestamp_utc", "price", "fetched_at_utc"]


def _near_dates(today: datetime | None = None) -> set[str]:
    t = (today or datetime.now(timezone.utc)).date()
    return {str(t + timedelta(days=d)) for d in range(-NEAR_DAYS, NEAR_DAYS + 1)}


def matched_bins(slug: str, today=None) -> pd.DataFrame:
    """Bins quoted on both venues for a near-dated target: (target_date, floor_strike, ...)."""
    kf, sf = DATA / "kalshi" / f"{slug}_markets.csv", DATA / "polymarket" / f"{slug}_snapshots.csv"
    # markets_available, NOT kf.exists() — see venue_basis; the legacy file no longer exists.
    if not (markets_available(kf) and partitioned_available(sf)):
        return pd.DataFrame()
    near = _near_dates(today)

    # load_markets, not read_csv: strike_type/close_time/open_time/series_ticker/floor_strike
    # all live in the dimension table now, and a bare read would leave `k.get("strike_type")`
    # returning None -> an all-False filter -> a silently EMPTY match set.
    k = load_markets(kf)
    k = k[k.get("strike_type") == "between"].drop_duplicates("ticker")
    if k.empty:
        return pd.DataFrame()
    # Target day from close_time, not the ticker: the ticker's date is reliable but its strike is
    # the bin MIDPOINT, and mixing the two invites the -B99.5-vs-99 mismatch this module exists to
    # avoid. Kalshi weather markets close at 06:00 UTC the following day.
    k["target_date"] = (pd.to_datetime(k["close_time"], utc=True, errors="coerce")
                        - pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")
    k = k[k["target_date"].isin(near)]
    k = k.dropna(subset=["floor_strike"])[["ticker", "series_ticker", "floor_strike",
                                           "target_date", "open_time", "close_time"]]

    s = load_partitioned(sf)
    ex = s["question"].astype(str).str.extract(_PM_BIN)
    s["floor_strike"] = pd.to_numeric(ex[0], errors="coerce")
    s["target_date"] = pd.to_datetime(ex[2] + " 2026", format="%B %d %Y",
                                      errors="coerce").dt.strftime("%Y-%m-%d")
    s = s.dropna(subset=["floor_strike", "target_date"])
    s = s[s["target_date"].isin(near)].sort_values("fetched_at_utc").drop_duplicates(
        "condition_id", keep="last")
    if s.empty:
        return pd.DataFrame()
    return k.merge(s[["condition_id", "clob_token_ids_json", "floor_strike", "target_date"]],
                   on=["floor_strike", "target_date"])


def _yes_token(raw) -> str | None:
    """First CLOB token id = the YES outcome, matching how the snapshot stores them."""
    import ast
    import json
    for parse in (json.loads, ast.literal_eval):
        try:
            ids = parse(raw)
            if isinstance(ids, (list, tuple)) and ids:
                return str(ids[0])
        except Exception:
            continue
    return None


def fetch_kalshi_minute(row, session=None) -> list[dict]:
    """Minute candles for one market, chunked so no request can exceed Kalshi's 5000-row cap."""
    start = pd.to_datetime(row["open_time"], utc=True, errors="coerce")
    end = pd.to_datetime(row["close_time"], utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return []
    now = pd.Timestamp.now(tz="UTC")
    end = min(end, now)
    start = max(start, now - pd.Timedelta(days=NEAR_DAYS + 1))
    out, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(hours=MINUTE_WINDOW_HOURS), end)
        payload, ok = kalshi_get(
            f"/series/{row['series_ticker']}/markets/{row['ticker']}/candlesticks",
            {"start_ts": int(cur.timestamp()), "end_ts": int(nxt.timestamp()),
             "period_interval": 1},
            session=session)
        if not ok:
            logger.warning("  kalshi minute chunk failed: %s %s..%s", row["ticker"], cur, nxt)
            cur = nxt
            continue
        cs = (payload or {}).get("candlesticks") or []
        # Completeness is a recorded fact, not an inference: a chunk at the cap is truncated.
        if len(cs) >= 5000:
            logger.warning("  ::warning:: %s chunk hit the 5000-candle cap — window too wide",
                           row["ticker"])
        out.extend(cs)
        cur = nxt
    return out


def prune(path: Path, today=None) -> int:
    """Drop rows whose target_date is older than RETAIN_DAYS. Returns rows removed.

    Deliberately keyed on target_date, not on fetched_at: a row's usefulness expires with the
    market it describes, not with when we happened to download it.
    """
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return 0
    if "target_date" not in df.columns or df.empty:
        return 0
    cutoff = ((today or datetime.now(timezone.utc)).date() - timedelta(days=RETAIN_DAYS))
    keep = pd.to_datetime(df["target_date"], errors="coerce").dt.date >= cutoff
    dropped = int((~keep).sum())
    if dropped:
        df[keep].to_csv(path, index=False)
    return dropped


def run(cities=None, today=None, session=None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    totals = {"bins": 0, "kal_rows": 0, "pm_rows": 0, "pruned": 0}
    for slug in (cities or CITIES):
        m = matched_bins(slug, today=today)
        if m.empty:
            logger.info("%s: no matched near-dated bins", slug)
            continue
        totals["bins"] += len(m)
        krows, prows = [], []
        for _, r in m.iterrows():
            for c in fetch_kalshi_minute(r, session=session):
                krows.append({"city": slug, "ticker": r["ticker"],
                              "floor_strike": r["floor_strike"], "target_date": r["target_date"],
                              "end_period_ts": c.get("end_period_ts"),
                              "open_dollars": (c.get("price") or {}).get("open"),
                              "high_dollars": (c.get("price") or {}).get("high"),
                              "low_dollars": (c.get("price") or {}).get("low"),
                              "close_dollars": (c.get("price") or {}).get("close"),
                              "yes_bid_close": (c.get("yes_bid") or {}).get("close"),
                              "yes_ask_close": (c.get("yes_ask") or {}).get("close"),
                              "volume": c.get("volume"), "fetched_at_utc": stamp})
            tok = _yes_token(r.get("clob_token_ids_json"))
            if tok:
                for pt in fetch_price_history(tok, interval="1d", fidelity=PM_FIDELITY):
                    ts = pt.get("t")
                    prows.append({"city": slug, "condition_id": r["condition_id"],
                                  "token_id": tok, "floor_strike": r["floor_strike"],
                                  "target_date": r["target_date"],
                                  "timestamp_utc": datetime.fromtimestamp(
                                      float(ts) / (1000.0 if float(ts) > 1e11 else 1.0),
                                      tz=timezone.utc).isoformat() if ts else None,
                                  "price": pt.get("p"), "fetched_at_utc": stamp})
        if krows:
            _append_csv(OUT / f"{slug}_kal_minute.csv",
                        pd.DataFrame(krows).reindex(columns=KAL_COLS).to_dict("records"),
                        ["ticker", "end_period_ts"])
            totals["kal_rows"] += len(krows)
        if prows:
            _append_csv(OUT / f"{slug}_pm_minute.csv",
                        pd.DataFrame(prows).reindex(columns=PM_COLS).to_dict("records"),
                        ["token_id", "timestamp_utc"])
            totals["pm_rows"] += len(prows)
        dropped = sum(prune(OUT / f"{slug}_{v}_minute.csv", today=today) for v in ("kal", "pm"))
        totals["pruned"] += dropped
        logger.info("%s: %d matched bins · kalshi %d rows · polymarket %d rows%s",
                    slug, len(m), len(krows), len(prows),
                    f" · pruned {dropped} beyond {RETAIN_DAYS}d" if dropped else "")
    logger.info("TOTAL: %d matched bins · %d kalshi · %d polymarket minute rows",
                totals["bins"], totals["kal_rows"], totals["pm_rows"])
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="minute prices for cross-venue matched bins")
    ap.add_argument("--cities", nargs="*", default=None)
    run(cities=ap.parse_args().cities)
