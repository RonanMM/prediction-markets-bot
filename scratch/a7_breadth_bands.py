"""A7 — falsification test of the claimed breadth-book band decomposition.

CLAIM (from a parallel agent):
    deep shoulder [5,10)c  net +0.0317  CI [+0.0192,+0.0442]  n=555 / 254 city-days
    core shoulder [20,35)c significantly NEGATIVE at -0.0394
    -> "the two cancel, which is why the full-band gate reads ~zero"

Everything below uses the repo's OWN machinery: shoulder_book._net_edge (half-spread crossed on
entry + verified 0.05*p*(1-p) weather taker fee) and stats_util.interval / cluster_key
(clustered by city-day). Grading is the breadth book's own `settled_outcome` = Polymarket's
terminal settlement, so no weather-truth path is involved.
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
print("=" * 110)
print(f"A7  breadth paper book: {len(d)} entries, {d['city'].nunique()} cities, "
      f"legs {dict(d['leg'].value_counts())}")
print(f"    graded (settled_outcome not null): {d['settled_outcome'].notna().sum()}")

s = d[(d["leg"] == "shoulder") & d["settled_outcome"].notna()].copy()
s["target_date"] = s["target_date"].astype(str)
s["yes"] = s["entry_yes_price"].astype(float)
s["entry"] = s["entry_side_price"].astype(float)
# shoulder leg is always a SELL of YES -> the NO side wins when the market settles NO
s["won"] = (s["settled_outcome"].astype(float) == 0).astype(int)
s["net"] = SB._net_edge(s["won"], s["entry"])
s["net_maker"] = s["won"].astype(float) - s["entry"]
print(f"    graded SHOULDER entries: {len(s)}  "
      f"city-days: {stats_util.cluster_key(s).nunique()}  "
      f"date span {s['target_date'].min()}..{s['target_date'].max()}")
print(f"    sanity: side=={set(s['side'])}, entry ~= 1-yes: "
      f"max|entry-(1-yes)| = {(s['entry']-(1-s['yes'])).abs().max():.6f}")


def band(sub, label, extra=""):
    if not len(sub):
        print(f"    {label:<26} n=0")
        return None
    ck = stats_util.cluster_key(sub)
    t = stats_util.interval(sub["net"], ck)
    m = stats_util.interval(sub["net_maker"], ck)
    star = "  <-- CI excludes 0" if (t["ci_lo"] > 0 or t["ci_hi"] < 0) else ""
    print(f"    {label:<26} n={len(sub):>5} cd={t['n_clusters']:>4} "
          f"mean_yes {sub['yes'].mean():.4f} realized_YES {1-sub['won'].mean():.4f} "
          f"taker {t['mean']:>+8.4f} [{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}] "
          f"maker {m['mean']:>+7.4f}{star}{extra}")
    return t


print("\n" + "=" * 110)
print("A7.1  THE CLAIMED DECOMPOSITION, recomputed (SELL YES / buy NO; net per share)")
BANDS = [("deep    [5,10)c", 0.05, 0.10), ("moderate[10,20)c", 0.10, 0.20),
         ("core    [20,35)c", 0.20, 0.35), ("FULL    [5,35)c", 0.05, 0.35),
         ("pre-reg mod [10,25)c", 0.10, 0.25)]
res = {}
for lab, lo, hi in BANDS:
    res[lab] = band(s[(s["yes"] >= lo) & (s["yes"] < hi)], lab)

print("\n  finer slicing of the deep end (is the effect a boundary artifact?):")
for lo, hi in [(0.05, 0.06), (0.06, 0.07), (0.07, 0.08), (0.08, 0.09), (0.09, 0.10),
               (0.10, 0.125), (0.125, 0.15), (0.15, 0.20)]:
    band(s[(s["yes"] >= lo) & (s["yes"] < hi)], f"[{lo:.3f},{hi:.3f})")

print("\n  ... and the upper end:")
for lo, hi in [(0.20, 0.25), (0.25, 0.30), (0.30, 0.35)]:
    band(s[(s["yes"] >= lo) & (s["yes"] < hi)], f"[{lo:.2f},{hi:.2f})")

print("\n" + "=" * 110)
print("A7.2  cost sensitivity of the DEEP band (entry price ~0.90-0.95 on the NO side)")
deep = s[(s["yes"] >= 0.05) & (s["yes"] < 0.10)]
print(f"    deep band: n={len(deep)}, mean NO entry price {deep['entry'].mean():.4f}, "
      f"NO win rate {deep['won'].mean():.4f}")
print(f"    {'half-spread':>12} {'fee model':>22} {'mean net':>10} {'clustered 95% CI':>24}")
for h in [0.0, 0.005, 0.01, 0.02]:
    for feelab, feefn in [("verified 0.05p(1-p)", lambda p: config.WEATHER_TAKER_RATE * p * (1 - p)),
                          ("legacy 2% of payout", lambda p: 0.02 * 1.0)]:
        ep = deep["entry"] + h
        net = deep["won"].astype(float) - (ep + ep.map(feefn) if feelab.startswith("verified")
                                           else ep + deep["won"].astype(float) * 0.02)
        t = stats_util.interval(net, stats_util.cluster_key(deep))
        print(f"    {h:>12.3f} {feelab:>22} {t['mean']:>+10.4f} "
              f"[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]")

print("\n" + "=" * 110)
print("A7.3  IS THE SPLIT POST-HOC?  (the same bands, split at the pre-registration date)")
s["entered"] = s["entered_at_utc"].astype(str).str[:10]
print(f"    BREADTH_PREREG_DATE = 2026-07-23 ; entries run "
      f"{s['entered'].min()} .. {s['entered'].max()}")
for lab, lo, hi in BANDS:
    sub = s[(s["yes"] >= lo) & (s["yes"] < hi)]
    band(sub[sub["entered"] < "2026-07-23"], f"{lab} pre-prereg")
    band(sub[sub["entered"] >= "2026-07-23"], f"{lab} FORWARD")

print("\n" + "=" * 110)
print("A7.4  robustness of the deep band: leave-one-city-out and by-city")
for c, g in sorted(deep.groupby("city"), key=lambda kv: -len(kv[1]))[:12]:
    band(g, f"city={c}")
print()
lo_stats = []
for c in deep["city"].unique():
    sub = deep[deep["city"] != c]
    t = stats_util.interval(sub["net"], stats_util.cluster_key(sub))
    lo_stats.append((c, t["mean"], t["ci_lo"], t["ci_hi"], len(sub)))
lo = pd.DataFrame(lo_stats, columns=["dropped", "mean", "lo", "hi", "n"]).sort_values("mean")
print("    leave-one-city-out on the deep band (most influential first):")
print(lo.head(5).round(4).to_string(index=False))
print(lo.tail(3).round(4).to_string(index=False))
print(f"    range of LOO means: {lo['mean'].min():+.4f} .. {lo['mean'].max():+.4f}; "
      f"all LOO CIs exclude 0: {bool((lo['lo'] > 0).all())}")

print("\n  by month of target date:")
deep2 = deep.copy(); deep2["m"] = deep2["target_date"].str[:7]
for m, g in deep2.groupby("m"):
    band(g, f"month {m}")

print("\n  weather markets are not all temperature — leg composition check:")
print(f"    questions containing 'temperature': "
      f"{s['question'].str.contains('temperature', case=False).mean():.3f}")
