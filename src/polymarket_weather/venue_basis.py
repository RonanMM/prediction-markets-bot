"""venue_basis.py — the two venues settle the same bin on DIFFERENT rulers (megaplan §13b).

Kalshi's US weather markets resolve on the **NWS Climatological Report (Daily)** — 1-minute
sensors over the LST day. Polymarket resolves the same stations on **wunderground**, whose daily
extremes are the hourly-METAR max over the local calendar day. Measured over 3,335 station-days
where both feeds exist, the CLI reads HIGHER on 53.3% of days and LOWER on 1.0%, and the two land
in a different 2 °F bin on 32.4% of days.

So "the same bin" is a materially different bet on the two venues, one-directionally. This module
measures (a) that basis per station and (b) whether the venues' PRICES reflect it. It takes no
forecast view at all — the hypothesis is about the ruler, not the weather.

⚠️ **The pre-2026-08-12 basis numbers above are INFLATED and must be re-measured.** This module
compares the CLI against OUR wu_truth reconstruction, so any error in that reconstruction is
indistinguishable from a real venue difference — the one measurement in the project where a
ruler bug lands directly on the estimand rather than on the noise. Ruler #13 (see
`fetch_station_obs.py`) is exactly such an error: the obs feed requested routine METARs only,
omitting SPECIs, which biased our WU max LOW and its min HIGH — one-sidedly, in the same
direction as the "CLI reads higher" finding this module reports. Some unknown share of the
53.3%/32.4% above is therefore our own bug, not the venues disagreeing.

The direction gate below is forward-only from BASIS_PREREG_DATE and was at 13/30 city-days when
this was found, so nothing measured under the old ruler has matured — but the headline basis
figures in this docstring stay quarantined until a specials-inclusive obs backfill has run and
the table is regenerated. Do not quote them meanwhile.

Nothing here trades. The gate is pre-registered below and forward-only.

CLI:  python venue_basis.py            # basis table + cross-venue price gap + gate status
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_kalshi import load_markets, markets_available
from processing import load_partitioned, partitioned_available

import wu_truth
from resolution_anchors import RESOLUTION_ANCHORS, slug as _slug

BASIS_PREREG_DATE = "2026-08-05"     # forward clock (UTC) — written before the data existed
GATE_MIN_CLUSTERS = 30               # city-days, matching the rest of the project's gates
GATE_MIN_DATES = 30                  # distinct target dates — breadth must not substitute for calendar

# A quote is only a quote if someone is on both sides of it. The first pass of this analysis
# reported "49% crossed books" purely from Kalshi rows with yes_bid=yes_ask=0.000 — an EMPTY book
# read as a price of zero, which manufactures spread against a real bid on the other venue.
MIN_BID, MAX_ASK, MAX_SPREAD = 0.005, 0.995, 0.25
MAX_QUOTE_LAG_H = 2.0                # comparing a fresh quote to a stale one invents disagreement

_DATA = Path(__file__).resolve().parent / "data"
# Kalshi abbreviates the month ("Aug 5, 2026"); Polymarket spells it out ("August 6"). Parsing
# both with one format silently produced NaT for every Kalshi row and an empty join.
_K_DATE = re.compile(r"on (\w+ \d{1,2}, \d{4})")
_PM_BIN = re.compile(r"between (\d+)-(\d+)°F on (\w+ \d+)", re.I)

C2F = lambda c: c * 9.0 / 5.0 + 32.0


def capture_cities() -> list[str]:
    """Data-file SLUGS for the cities captured on both venues (the tier added 2026-08-03).

    Must be slugged, not passed through raw. RESOLUTION_ANCHORS is keyed by display name
    ("Los Angeles"), and every path here is `{slug}_*.csv`. Single-word cities happened to work on
    a case-insensitive macOS filesystem — "Atlanta" opened atlanta_*.csv — so the bug hid, while
    the two-word cities failed silently and simply vanished from the table. On the Linux CI runner
    NONE of them would have resolved and the whole report would have read "no data".
    """
    return sorted(_slug(k) for k, v in RESOLUTION_ANCHORS.items() if v.get("tier") == "capture")


def station_basis(slug: str) -> dict | None:
    """CLI-minus-WU daily-max basis in °F for one station, or None if a feed is missing.

    ⚠️ Per station, never pooled. The NYC/Chicago basis is not evidence about KAUS: it depends on
    that station's sensor package and on how often its true max falls between METAR hours.
    """
    act = _DATA / "weather" / f"{slug}_historical_actuals.csv"
    if not act.exists():
        return None
    cli = pd.read_csv(act).dropna(subset=["temp_max_c"])
    rows = []
    for _, r in cli.iterrows():
        # `reconstruct`, not `wu_daily_extreme`: the latter refuses any city not yet admitted to
        # wu_truth's allowlist, and this measurement is how a city earns admission.
        wu = wu_truth.reconstruct(slug, r["date_local"], "max")
        if wu is None:
            continue
        rows.append((C2F(float(r["temp_max_c"])), C2F(float(wu))))
    if len(rows) < 30:
        return None
    d = pd.DataFrame(rows, columns=["cli", "wu"])
    d["diff"] = d["cli"] - d["wu"]
    binof = lambda f: np.floor(np.floor(f + 1e-9) / 2) * 2      # Kalshi's 2 °F grid, even floors
    flip = binof(d["cli"]) != binof(d["wu"])
    return {"slug": slug, "n": len(d), "mean": float(d["diff"].mean()),
            "median": float(d["diff"].median()), "sd": float(d["diff"].std()),
            "cli_higher": float((d["diff"] > 0.05).mean()),
            "wu_higher": float((d["diff"] < -0.05).mean()),
            "bin_flip": float(flip.mean()),
            "flip_cli_up": float((binof(d.loc[flip, "cli"]) > binof(d.loc[flip, "wu"])).mean())
            if flip.any() else float("nan")}


def _two_sided(bid, ask) -> pd.Series:
    b, a = pd.to_numeric(bid, errors="coerce"), pd.to_numeric(ask, errors="coerce")
    return (b.notna() & a.notna() & (b > MIN_BID) & (a < MAX_ASK) & (a > b)
            & ((a - b) < MAX_SPREAD))


def matched_bins(slug: str) -> pd.DataFrame:
    """One row per (city, target date, bin) quoted two-sided on BOTH venues within MAX_QUOTE_LAG_H."""
    kf, pf = _DATA / "kalshi" / f"{slug}_markets.csv", _DATA / "polymarket" / f"{slug}_snapshots.csv"
    # markets_available, NOT kf.exists(): the fact table is daily partitions now and the
    # legacy file is gone, so an exists() guard here returns an empty frame forever — the
    # silent failure this module's own findings were once quarantined for.
    if not (markets_available(kf) and partitioned_available(pf)):
        return pd.DataFrame()
    # load_markets, not read_csv — `title` and `strike_type` live in the dimension table
    # and a bare read would make both filters below match nothing, silently.
    k = load_markets(kf)
    k = k[(k.get("strike_type") == "between") & (k.get("status") == "active")].copy()
    if k.empty:
        return pd.DataFrame()
    kd = k["title"].astype(str).str.extract(_K_DATE)[0]
    k["date"] = pd.to_datetime(kd, format="%b %d, %Y", errors="coerce").fillna(
        pd.to_datetime(kd, format="%B %d, %Y", errors="coerce")).dt.strftime("%Y-%m-%d")
    k["ts"] = pd.to_datetime(k["fetched_at_utc"], utc=True, errors="coerce")
    k = k.sort_values("ts").groupby("ticker").last().reset_index()
    k = k[_two_sided(k["yes_bid"], k["yes_ask"])]
    if k.empty:
        return pd.DataFrame()
    k["kal"] = (k["yes_bid"] + k["yes_ask"]) / 2

    p = load_partitioned(pf)
    p["ts"] = pd.to_datetime(p["fetched_at_utc"], utc=True, errors="coerce")
    p = p.sort_values("ts").groupby("condition_id").last().reset_index()
    ex = p["question"].astype(str).str.extract(_PM_BIN)
    p["lo"] = pd.to_numeric(ex[0], errors="coerce")
    p["date"] = pd.to_datetime(ex[2] + " 2026", format="%B %d %Y",
                               errors="coerce").dt.strftime("%Y-%m-%d")
    p = p.dropna(subset=["lo", "date"])
    p = p[_two_sided(p["yes_best_bid"], p["yes_best_ask"])]
    if p.empty:
        return pd.DataFrame()
    p["pm"] = (p["yes_best_bid"] + p["yes_best_ask"]) / 2

    m = p.merge(k, left_on=["date", "lo"], right_on=["date", "floor_strike"], suffixes=("_p", "_k"))
    if m.empty:
        return pd.DataFrame()
    m["lag_h"] = (m["ts_p"] - m["ts_k"]).dt.total_seconds().abs() / 3600.0
    m = m[m["lag_h"] <= MAX_QUOTE_LAG_H]
    if m.empty:
        return pd.DataFrame()
    out = m[["date", "lo", "pm", "kal", "lag_h"]].copy()
    out["city"] = slug
    out["gap"] = out["pm"] - out["kal"]
    out["cluster"] = out["city"] + "|" + out["date"]
    return out


def directional_test(bins: pd.DataFrame) -> dict:
    """Does the price gap move the way the ruler shift predicts?

    CLI (Kalshi) reads higher, so Kalshi's settlement distribution sits ABOVE Polymarket's. For a
    bin above the day's centre Kalshi should be richer (gap negative); below it, Polymarket should
    be. The prediction is therefore a NEGATIVE slope of gap on bin-position — one number, signed in
    advance, which is what makes it a test rather than a description.
    """
    if bins.empty:
        return {}
    parts = []
    for _, g in bins.groupby("cluster"):
        w = g["pm"].clip(lower=0)
        if len(g) < 4 or w.sum() <= 0:
            continue
        parts.append(g.assign(rel=g["lo"] - float((g["lo"] * w).sum() / w.sum())))
    if not parts:
        return {}
    o = pd.concat(parts)
    import stats_util
    # Per city-day slope of gap on relative bin position, then a clustered mean of those slopes.
    slopes, keys = [], []
    for cl, g in o.groupby("cluster"):
        if g["rel"].std() > 0:
            slopes.append(float(np.polyfit(g["rel"], g["gap"], 1)[0]))
            keys.append(cl)
    if not slopes:
        return {}
    iv = stats_util.interval(pd.Series(slopes), pd.Series(keys))
    lo, hi = iv["mean"] - 1.96 * iv["se"], iv["mean"] + 1.96 * iv["se"]
    dates = o["date"].nunique()
    return {"slope": float(iv["mean"]), "lo": float(lo), "hi": float(hi),
            "n_bins": int(len(o)), "n_clusters": int(len(slopes)), "n_dates": int(dates),
            "pass": bool(hi < 0 and len(slopes) >= GATE_MIN_CLUSTERS and dates >= GATE_MIN_DATES)}


def report() -> None:
    print(f"VENUE BASIS — Kalshi settles on the NWS CLI, Polymarket on wunderground "
          f"(pre-registered {BASIS_PREREG_DATE})\n")
    print("  STATION BASIS  (CLI − WU, °F; per station, never pooled)")
    print(f"    {'station':<16}{'days':>6}{'mean':>8}{'median':>8}{'CLI hi':>8}{'WU hi':>7}"
          f"{'diff bin':>10}")
    any_basis = False
    for slug in sorted({*capture_cities(), "new_york_city", "chicago"}):
        b = station_basis(slug)
        if not b:
            print(f"    {slug:<16}{'—':>6}   no hourly METARs yet (run fetch_station_obs.py)")
            continue
        any_basis = True
        print(f"    {slug:<16}{b['n']:>6}{b['mean']:>+8.2f}{b['median']:>+8.2f}"
              f"{b['cli_higher']:>7.0%}{b['wu_higher']:>7.0%}{b['bin_flip']:>10.1%}")
    if any_basis:
        print("    'diff bin' = share of days the two rulers land in a different 2 °F bin — i.e.")
        print("    the share on which the identical bin is a different bet on the two venues.")

    frames = [matched_bins(c) for c in capture_cities()]
    bins = pd.concat([f for f in frames if not f.empty]) if any(not f.empty for f in frames) \
        else pd.DataFrame()
    print(f"\n  CROSS-VENUE QUOTES  (two-sided on both venues, within {MAX_QUOTE_LAG_H:.0f}h)")
    if bins.empty:
        print("    none yet — capture began 2026-08-03. Nothing to conclude.")
        return
    print(f"    {len(bins)} bins · {bins['city'].nunique()} cities · {bins['date'].nunique()} dates"
          f" · {bins['cluster'].nunique()} city-days")
    print(f"    |Polymarket − Kalshi| median {bins['gap'].abs().median():.3f}   "
          f">5¢ on {(bins['gap'].abs() > 0.05).mean():.0%}   signed mean {bins['gap'].mean():+.4f}")
    t = directional_test(bins)
    if not t:
        print("    directional test: not enough bins per city-day yet.")
        return
    mark = "PASS" if t["pass"] else "pending"
    print(f"    DIRECTIONAL slope {t['slope']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]  "
          f"({t['n_clusters']}/{GATE_MIN_CLUSTERS} city-days, {t['n_dates']}/{GATE_MIN_DATES} "
          f"dates)  [{mark}]")
    print("    (prediction, signed in advance: NEGATIVE — Kalshi richer above the day's centre.)")
    if t["n_clusters"] < GATE_MIN_CLUSTERS or t["n_dates"] < GATE_MIN_DATES:
        print("    Point estimates here are not evidence; the gate is the interval AND the calendar.")


if __name__ == "__main__":
    report()
