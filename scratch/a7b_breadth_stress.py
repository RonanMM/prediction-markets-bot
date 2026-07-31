"""A7b — stress the breadth deep-band claim.

The city-day cluster assumes 50 cities on the SAME calendar day are 50 independent draws.
The breadth book spans only SIX target dates. If the effect rides a common daily factor, the
city-day interval is far too narrow. Test by re-clustering on DATE, on CITY, and by looking at
the between-date spread directly. Also: does the 5-city book (a different, earlier sample)
agree on the SIGN?
"""
import sys

import numpy as np
import pandas as pd

import rc_lib as R

sys.path.insert(0, R.SRC)
import shoulder_book as SB  # noqa: E402
import stats_util  # noqa: E402
import grading  # noqa: E402
from pmf import parse_question  # noqa: E402

d = pd.read_csv(f"{R.SRC}/output/shoulder_paper_breadth.csv")
s = d[(d["leg"] == "shoulder") & d["settled_outcome"].notna()].copy()
s["target_date"] = s["target_date"].astype(str)
s["yes"] = s["entry_yes_price"].astype(float)
s["entry"] = s["entry_side_price"].astype(float)
s["won"] = (s["settled_outcome"].astype(float) == 0).astype(int)
s["net"] = SB._net_edge(s["won"], s["entry"])
BANDS = {"deep [5,10)": (0.05, 0.10), "mod [10,20)": (0.10, 0.20),
         "core [20,35)": (0.20, 0.35), "FULL [5,35)": (0.05, 0.35)}

print("=" * 110)
print("A7b.1  HOW MANY INDEPENDENT DAYS?")
print(f"    distinct target dates: {s['target_date'].nunique()}  -> {sorted(s['target_date'].unique())}")
print(f"    distinct cities:       {s['city'].nunique()}")
print(f"    'city-days' counted by stats_util.cluster_key: {stats_util.cluster_key(s).nunique()}")
print("    => the 249-294 'independent clusters' are 50 cities x 6 days, not 294 days.")

print("\n" + "=" * 110)
print("A7b.2  THE SAME EFFECT UNDER THREE CLUSTERINGS")
print(f"    {'band':<14} {'n':>5} {'mean':>9} {'CI city-day':>24} {'CI city':>24} {'CI DATE':>24}")
for lab, (lo, hi) in BANDS.items():
    g = s[(s["yes"] >= lo) & (s["yes"] < hi)]
    a = stats_util.interval(g["net"], stats_util.cluster_key(g))
    b = stats_util.interval(g["net"], g["city"])
    c = stats_util.interval(g["net"], g["target_date"])
    print(f"    {lab:<14} {len(g):>5} {a['mean']:>+9.4f} "
          f"[{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}]{'':>5}"
          f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]  (g={b['n_clusters']})"
          f"  [{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] (g={c['n_clusters']})")
print("    (DATE clustering has g=6 < stats_util.MIN_CLUSTERS=30 — by the repo's OWN rule that")
print("     interval is 'not meaningful'. That is the finding, not a nuisance.)")

print("\n" + "=" * 110)
print("A7b.3  BETWEEN-DATE SPREAD — is the deep-band edge a common daily factor?")
for lab, (lo, hi) in BANDS.items():
    g = s[(s["yes"] >= lo) & (s["yes"] < hi)]
    t = g.groupby("target_date").agg(n=("net", "size"), mean=("net", "mean"),
                                     realized=("won", lambda x: 1 - x.mean()))
    print(f"\n    {lab}")
    print(t.round(4).to_string())
    print(f"      between-date std of the daily mean: {t['mean'].std():.4f}   "
          f"SE of the 6-day mean: {t['mean'].std()/np.sqrt(len(t)):.4f}   "
          f"vs city-day SE {stats_util.interval(g['net'], stats_util.cluster_key(g))['se']:.4f}")

print("\n" + "=" * 110)
print("A7b.4  SIGN CHECK against the 5-city book (an EARLIER, DISJOINT sample, weather-truth graded)")
b5 = SB._load_book()
rows = []
for _, r in b5[b5["leg"] == "shoulder"].iterrows():
    pq = parse_question(str(r["question"]))
    if not pq:
        continue
    full = {"Chicago": "Chicago", "NYC": "New York City", "London": "London",
            "Seoul": "Seoul", "HongKong": "Hong Kong"}[r["city"]]
    y = grading.resolves_yes(full, r["target_date"], r["question"], pq["temp_c"])
    if y is None:
        continue
    rows.append(dict(city=r["city"], target_date=str(r["target_date"]),
                     yes=float(r["entry_yes_price"]), entry=float(r["entry_side_price"]),
                     won=int(y == 0)))
f5 = pd.DataFrame(rows)
f5["net"] = SB._net_edge(f5["won"], f5["entry"])
print(f"    5-city book graded shoulder entries: {len(f5)}, "
      f"dates {f5['target_date'].min()}..{f5['target_date'].max()}, "
      f"{f5['target_date'].nunique()} distinct target dates")
print(f"    {'band':<14} {'n':>5} {'cd':>4} {'realized':>9} {'taker net':>10} {'CI city-day':>24}")
for lab, (lo, hi) in BANDS.items():
    g = f5[(f5["yes"] >= lo) & (f5["yes"] < hi)]
    if not len(g):
        print(f"    {lab:<14} n=0")
        continue
    t = stats_util.interval(g["net"], stats_util.cluster_key(g))
    print(f"    {lab:<14} {len(g):>5} {t['n_clusters']:>4} {1-g['won'].mean():>9.4f} "
          f"{t['mean']:>+10.4f} [{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]")
print("    -> compare SIGNS with A7.1. shoulder_book's own docstring (§10f, 2026-07-23, n=109)")
print("       claims the DEEP band is fat-tail UNDER-priced (selling it LOSES) and the moderate")
print("       band over-priced. The breadth claim says the deep band is the WINNER.")

print("\n" + "=" * 110)
print("A7b.5  what actually drives the deep band: bin-count / market shape")
deep = s[(s["yes"] >= 0.05) & (s["yes"] < 0.10)].copy()
deep["nbins_that_day"] = deep.groupby(["city", "target_date"])["condition_id"].transform("size")
print(deep.groupby(pd.cut(deep["nbins_that_day"], [0, 1, 2, 3, 5, 20]), observed=True)
      .agg(n=("net", "size"), mean=("net", "mean"), realized=("won", lambda x: 1 - x.mean())).round(4).to_string())
print("\n    entries recorded per (city,date) in the whole shoulder book:")
print(s.groupby(["city", "target_date"]).size().describe().round(2).to_string())
