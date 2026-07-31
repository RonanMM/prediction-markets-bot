"""A7c — the execution question for the breadth deep band, plus overlap/replication splits.

If the recorded `entry_yes_price` is a WIDE-BOOK MIDPOINT rather than a hittable bid, the whole
deep-band edge is the spread. Three probes:
  (a) how far the price moves right after entry (min/max_yes_after) — a proxy for book width;
  (b) liquidity of the deep-band bins;
  (c) for the 5 cities where we hold the CLOB feed, does the Gamma mid agree with traded price?
Plus: does the deep band replicate on the 44 cities NOT in the original 5, and on the 5 alone?
"""
import sys

import numpy as np
import pandas as pd

import rc_lib as R

sys.path.insert(0, R.SRC)
import shoulder_book as SB  # noqa: E402
import stats_util  # noqa: E402
import config  # noqa: E402

d = pd.read_csv(f"{R.SRC}/output/shoulder_paper_breadth.csv")
s = d[(d["leg"] == "shoulder") & d["settled_outcome"].notna()].copy()
s["target_date"] = s["target_date"].astype(str)
s["yes"] = s["entry_yes_price"].astype(float)
s["entry"] = s["entry_side_price"].astype(float)
s["won"] = (s["settled_outcome"].astype(float) == 0).astype(int)
s["net"] = SB._net_edge(s["won"], s["entry"])
deep = s[(s["yes"] >= 0.05) & (s["yes"] < 0.10)].copy()

print("=" * 110)
print("A7c.1  BOOK WIDTH PROXY — where does the price go right after we 'sell' at the mid?")
for lab, g in [("deep [5,10)", deep), ("core [20,35)", s[(s["yes"] >= .20) & (s["yes"] < .35)])]:
    print(f"\n    {lab}  n={len(g)}  entry_yes mean {g['yes'].mean():.4f}")
    print(f"      min_yes_after : {g['min_yes_after'].describe(percentiles=[.1,.5,.9]).round(4).to_dict()}")
    print(f"      max_yes_after : {g['max_yes_after'].describe(percentiles=[.1,.5,.9]).round(4).to_dict()}")
    print(f"      median (max_after - min_after) range = "
          f"{(g['max_yes_after']-g['min_yes_after']).median():.4f}")
    print(f"      frac where min_yes_after <= 0.02 (price collapses): "
          f"{(g['min_yes_after']<=0.02).mean():.3f}")

print("\n" + "=" * 110)
print("A7c.2  LIQUIDITY of the deep band vs the rest")
s["band"] = pd.cut(s["yes"], [0.05, 0.10, 0.20, 0.35], right=False)
print(s.groupby("band", observed=True).agg(
    n=("liquidity", "size"), med_liq=("liquidity", "median"),
    q10_liq=("liquidity", lambda x: x.quantile(.1)),
    frac_below_1000=("liquidity", lambda x: (x < 1000).mean()),
    net=("net", "mean")).round(3).to_string())
print(f"    config.MIN_LIQUIDITY = {config.MIN_LIQUIDITY}")

print("\n" + "=" * 110)
print("A7c.3  BREAK-EVEN EXECUTION — how much worse a fill can the deep band absorb?")
wr = deep["won"].mean()
print(f"    deep band NO win rate {wr:.4f}; recorded NO entry mean {deep['entry'].mean():.4f}")
for h in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
    ep = deep["entry"] + h
    net = deep["won"].astype(float) - (ep + ep.map(config.taker_fee_per_share))
    t = stats_util.interval(net, stats_util.cluster_key(deep))
    flag = "  <- dead" if t["ci_lo"] <= 0 else ""
    print(f"      slippage {h:.2f} on entry: net {t['mean']:>+8.4f} "
          f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]{flag}")
print("    (equivalently: the recorded 6.9c mid must be sellable at >= 2.9c for the edge to exist)")

print("\n" + "=" * 110)
print("A7c.4  REPLICATION SPLITS")
FIVE = {"Hong Kong", "London", "Seoul", "NYC", "New York City", "Chicago"}
for lab, sel in [("the original 5 cities", deep[deep["city"].isin(FIVE)]),
                 ("the 44 NEW cities", deep[~deep["city"].isin(FIVE)])]:
    t = stats_util.interval(sel["net"], stats_util.cluster_key(sel))
    print(f"    deep band, {lab:<24} n={len(sel):>4} cd={t['n_clusters']:>3} "
          f"realized {1-sel['won'].mean():.4f} net {t['mean']:>+8.4f} "
          f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]")
for lab, sel in [("the original 5 cities", s[(s["yes"] >= .20) & (s["yes"] < .35) & s["city"].isin(FIVE)]),
                 ("the 44 NEW cities", s[(s["yes"] >= .20) & (s["yes"] < .35) & ~s["city"].isin(FIVE)])]:
    t = stats_util.interval(sel["net"], stats_util.cluster_key(sel))
    print(f"    core band, {lab:<24} n={len(sel):>4} cd={t['n_clusters']:>3} "
          f"realized {1-sel['won'].mean():.4f} net {t['mean']:>+8.4f} "
          f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]")

print("\n  half-split by city (alphabetical halves) — a crude internal replication:")
cities = sorted(deep["city"].unique())
h1, h2 = set(cities[::2]), set(cities[1::2])
for lab, sel in [("half A", deep[deep["city"].isin(h1)]), ("half B", deep[deep["city"].isin(h2)])]:
    t = stats_util.interval(sel["net"], stats_util.cluster_key(sel))
    print(f"    deep {lab}: n={len(sel):>4} net {t['mean']:>+8.4f} [{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]")

print("\n" + "=" * 110)
print("A7c.5  IS THE DEEP BAND JUST 'THE MARKET IS OVER-ROUND'?  bins per city-day")
print("    A shoulder sell profits from the vig whenever the book sums to >1. Deep bins are the")
print("    cheapest members of an over-round book, so an over-round of v spread over n bins gives")
print("    each bin roughly v/n of overpricing -- in RELATIVE terms that is enormous for a 7c bin.")
allsh = s.copy()
allsh["k"] = allsh.groupby(["city", "target_date"])["condition_id"].transform("size")
print(f"    mean recorded bins per city-day: {allsh['k'].mean():.2f}")
print(f"    deep band: priced {deep['yes'].mean():.4f}, realized {1-deep['won'].mean():.4f}, "
      f"absolute overpricing {deep['yes'].mean()-(1-deep['won'].mean()):+.4f}")
core = s[(s["yes"] >= .20) & (s["yes"] < .35)]
print(f"    core band: priced {core['yes'].mean():.4f}, realized {1-core['won'].mean():.4f}, "
      f"absolute overpricing {core['yes'].mean()-(1-core['won'].mean()):+.4f}")
mod = s[(s["yes"] >= .10) & (s["yes"] < .20)]
print(f"    mod  band: priced {mod['yes'].mean():.4f}, realized {1-mod['won'].mean():.4f}, "
      f"absolute overpricing {mod['yes'].mean()-(1-mod['won'].mean()):+.4f}")
tot = s.groupby(["city", "target_date"]).agg(sump=("yes", "sum"), sumy=("won", lambda x: (1-x).sum()),
                                             k=("yes", "size"))
print(f"\n    sum of RECORDED shoulder-bin prices per city-day: mean {tot['sump'].mean():.4f} "
      f"vs realized YES count {tot['sumy'].mean():.4f} over {tot['k'].mean():.2f} bins")
print(f"    -> recorded 5-35c bins alone are overpriced by {tot['sump'].mean()-tot['sumy'].mean():+.4f} "
      f"per city-day in aggregate")
