"""A2 — favorite-longshot bias. Model-free: price in, outcome out.

Estimator: per-bin residual r = y - p (y = graded outcome, p = market price at a fixed rule).
Bucketed by p; cluster-robust mean by CITY-DAY via stats_util.
Trading net (sell YES at p): (p - y) - HALF_SPREAD - verified taker fee(p), per share.
"""
import sys

import numpy as np
import pandas as pd

import rc_lib as R
import config

sys.path.insert(0, R.SRC)
import grading  # noqa: E402
from pmf import parse_question, resolves_yes_temp  # noqa: E402

H = config.HALF_SPREAD
TZ = {c: config.CITIES[c]["timezone"] for c in config.CITIES}


def taker(p):
    return config.WEATHER_TAKER_RATE * p * (1 - p)


# ---------------------------------------------------------------- universe + both gradings
uni = pd.read_pickle("universe.pkl")
uni = uni[uni["cond"].notna() & uni["target_date"].notna()].copy()


def regrade_wholeC(r):
    a = grading.fetch_actual_weather(r["city"], r["target_date"], r["question"])
    if a is None:
        return None
    p = parse_question(str(r["question"]))
    if p is None:
        return None
    unit = "whole °C" if r["city"] == "Hong Kong" else r["unit"]
    return int(resolves_yes_temp(p, grading.native_round(a, unit), unit, grading.native_round))


uni["outcome_fix"] = [regrade_wholeC(r) for _, r in uni.iterrows()]
uni["day_end"] = [pd.Timestamp(d, tz=TZ[c]) + pd.Timedelta(days=1)
                  for c, d in zip(uni["city"], uni["target_date"])]
uni["day_end"] = pd.to_datetime(uni["day_end"], utc=True)
META = uni.set_index("condition_id")

# ---------------------------------------------------------------- price rules
sn = pd.read_pickle("snapshots.pkl")
sn = sn[sn["condition_id"].isin(META.index) & sn["yes"].notna()].copy()
sn["day_end"] = sn["condition_id"].map(META["day_end"])
pre = sn[sn["fetched_at"] < sn["day_end"]]
last_snap = pre.sort_values("fetched_at").groupby("condition_id").last()

ph = R.load_price_history()
ph = ph[ph["condition_id"].isin(META.index)].copy()
ph["day_end"] = ph["condition_id"].map(META["day_end"])
ph["h_to_end"] = (ph["day_end"] - ph["ts"]).dt.total_seconds() / 3600.0
ph = ph[ph["h_to_end"] > 0]


def clob_at(hours):
    d = ph[ph["h_to_end"] >= hours].sort_values("h_to_end")
    return d.groupby("condition_id").first()


RULES = {
    "last snapshot pre-resolution": last_snap[["yes", "liquidity_usdc", "volume_usdc"]]
        .rename(columns={"yes": "p"}),
    "CLOB, last price >6h out": clob_at(6)[["price"]].rename(columns={"price": "p"}),
    "CLOB, last price >24h out": clob_at(24)[["price"]].rename(columns={"price": "p"}),
    "CLOB, last price >48h out": clob_at(48)[["price"]].rename(columns={"price": "p"}),
}

EDGES = [0, .01, .02, .05, .10, .20, .35, .50, .65, .80, .90, 1.0]


def flb_table(df, ycol, label):
    df = df.dropna(subset=["p", ycol]).copy()
    df["y"] = df[ycol].astype(float)
    df["resid"] = df["y"] - df["p"]
    df["sell_net"] = (df["p"] - df["y"]) - H - taker(df["p"])
    df["buy_net"] = (df["y"] - df["p"]) - H - taker(df["p"])
    df["band"] = pd.cut(df["p"], EDGES, include_lowest=True)
    print(f"\n  --- {label}  (n={len(df)} bins, "
          f"{df.groupby(['city','target_date']).ngroups} city-days) ---")
    print(f"    {'price band':>14} {'n':>5} {'cd':>4} {'mean p':>7} {'realized':>9} "
          f"{'p-q':>8} {'95% CI (clustered)':>22} {'sell net':>9} {'sell CI':>22}")
    for band, g in df.groupby("band", observed=True):
        ck = R.stats_util.cluster_key(g)
        r = R.ci(-g["resid"], ck)           # p - q
        s = R.ci(g["sell_net"], ck)
        print(f"    {str(band):>14} {len(g):>5} {r['n_clusters']:>4} {g['p'].mean():>7.4f} "
              f"{g['y'].mean():>9.4f} {r['mean']:>+8.4f} {R.fmt_ci(r):>22} "
              f"{s['mean']:>+9.4f} {R.fmt_ci(s):>22}")
    # pooled
    ck = R.stats_util.cluster_key(df)
    r = R.ci(-df["resid"], ck)
    print(f"    {'POOLED':>14} {len(df):>5} {r['n_clusters']:>4} {df['p'].mean():>7.4f} "
          f"{df['y'].mean():>9.4f} {r['mean']:>+8.4f} {R.fmt_ci(r):>22}")
    return df


def build(rule_df):
    d = rule_df.join(META[["city", "target_date", "kind", "cond", "outcome", "outcome_fix",
                           "question", "unit"]], how="inner")
    return d.reset_index()


print("=" * 130)
print("A2  FAVORITE-LONGSHOT BIAS")
print("    'realized' = graded YES frequency; p-q>0 means the market OVERPRICES that band")
print("    sell net   = per-share EV of selling YES (buying NO), net of "
      f"{H:.2f} half-spread + verified weather taker fee")

for rule, rd in RULES.items():
    d = build(rd)
    # A: repo grading, Hong Kong EXCLUDED (its resolution_unit makes exact bins ~never YES; see A6)
    flb_table(d[d["city"] != "Hong Kong"], "outcome", f"{rule}  |  repo grading, HK excluded")

print("\n" + "=" * 130)
print("A2b  ROBUSTNESS — same tables with Hong Kong included under the CORRECTED whole-°C rule")
for rule, rd in RULES.items():
    d = build(rd)
    flb_table(d, "outcome_fix", f"{rule}  |  all 5 cities, HK regraded whole-°C")
