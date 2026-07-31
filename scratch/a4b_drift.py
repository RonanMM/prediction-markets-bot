"""A4b — the over-round decay, measured cleanly: ONE observation per bin per lead band.

This is an INDEPENDENT replication attempt of the breadth 'sell the cheap bins early' claim, on
a different sample (the 5 original cities, CLOB traded prices, weather-truth graded).
"""
import sys

import numpy as np
import pandas as pd

import rc_lib as R

sys.path.insert(0, R.SRC)
import config  # noqa: E402
import stats_util  # noqa: E402

H = config.HALF_SPREAD
uni = pd.read_pickle("universe.pkl")
uni = uni[uni["cond"].notna() & uni["target_date"].notna()]
META = uni.set_index("condition_id")
tzs = {c: config.CITIES[c]["timezone"] for c in config.CITIES}

ph = R.load_price_history()
ph["city"] = ph["condition_id"].map(META["city"])
ph["target_date"] = ph["condition_id"].map(META["target_date"])
ph["y"] = ph["condition_id"].map(META["outcome"])
ph = ph.dropna(subset=["city", "target_date"])
ends = {(c, d): pd.Timestamp(d, tz=tzs[c]) + pd.Timedelta(days=1)
        for c, d in set(zip(ph["city"], ph["target_date"]))}
ph["end_utc"] = pd.to_datetime([ends[(c, d)] for c, d in zip(ph["city"], ph["target_date"])], utc=True)
ph["h"] = (ph["end_utc"] - ph["ts"]).dt.total_seconds() / 3600
ph = ph[ph["h"] > 0]

BANDS = [("60h (48-72)", 48, 72), ("36h (24-48)", 24, 48), ("18h (12-24)", 12, 24),
         ("9h (6-12)", 6, 12), ("3h (0-6)", 0, 6)]

print("=" * 115)
print("A4b  ONE OBSERVATION PER BIN PER LEAD BAND — is a bin's price above its realized frequency?")
print("     'sell net' = (p - y) - half_spread - 0.05*p*(1-p), per share, SELLING the YES.")
print("     Hong Kong EXCLUDED (its resolution_unit grading is broken — see A6).")

g0 = ph[ph["city"] != "Hong Kong"].dropna(subset=["y"]).copy()


def table(df, title, pmin=0.0, pmax=1.0):
    print(f"\n  {title}")
    print(f"    {'lead':>12} {'bins':>6} {'cd':>4} {'mean p':>8} {'realized':>9} {'p - q':>9} "
          f"{'clustered 95% CI':>22} {'sell net':>9} {'net CI':>22}")
    for lab, lo, hi in BANDS:
        sub = df[(df["h"] >= lo) & (df["h"] < hi) & (df["price"] >= pmin) & (df["price"] < pmax)]
        if not len(sub):
            continue
        # one obs per bin: the observation nearest the middle of the band
        mid = (lo + hi) / 2
        sub = sub.assign(dist=(sub["h"] - mid).abs()).sort_values("dist") \
                 .groupby("condition_id", as_index=False).first()
        if len(sub) < 20:
            continue
        y = sub["y"].astype(float)
        pq = sub["price"] - y
        net = pq - H - config.WEATHER_TAKER_RATE * sub["price"] * (1 - sub["price"])
        ck = stats_util.cluster_key(sub)
        a = stats_util.interval(pq, ck)
        b = stats_util.interval(net, ck)
        print(f"    {lab:>12} {len(sub):>6} {a['n_clusters']:>4} {sub['price'].mean():>8.4f} "
              f"{y.mean():>9.4f} {a['mean']:>+9.4f} [{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}]"
              f" {b['mean']:>+9.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]")


table(g0, "ALL bins")
table(g0, "DEEP bins only, price in [0.05, 0.10)  <- the breadth claim's band", 0.05, 0.10)
table(g0, "MODERATE bins, price in [0.10, 0.20)", 0.10, 0.20)
table(g0, "CORE bins, price in [0.20, 0.35)", 0.20, 0.35)
table(g0, "CHEAP TAILS, price < 0.05", 0.0, 0.05)

print("\n" + "=" * 115)
print("A4b.2  the same DEEP-band test pooled over all leads >24h (max power on this sample)")
sub = g0[(g0["h"] > 24) & (g0["price"] >= 0.05) & (g0["price"] < 0.10)]
sub = sub.sort_values("h").groupby("condition_id", as_index=False).first()
y = sub["y"].astype(float)
net = (sub["price"] - y) - H - config.WEATHER_TAKER_RATE * sub["price"] * (1 - sub["price"])
ck = stats_util.cluster_key(sub)
a = stats_util.interval(sub["price"] - y, ck)
b = stats_util.interval(net, ck)
print(f"    n={len(sub)} bins over {a['n_clusters']} city-days; priced {sub['price'].mean():.4f}, "
      f"realized {y.mean():.4f}")
print(f"    gross p-q {a['mean']:+.4f} [{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}]")
print(f"    taker net {b['mean']:+.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]   "
      f"MDE (1.96*se) = {1.96*b['se']:.4f}")
print(f"    -> breadth book claims +0.0297 [+0.0167,+0.0427] for this band. "
      f"Is that inside this interval? {b['ci_lo'] <= 0.0297 <= b['ci_hi']}")

print("\n" + "=" * 115)
print("A4b.3  DOES THE PRICE PATH ITSELF PREDICT?  price change over the last 24h -> next move")
p = ph.sort_values(["condition_id", "ts"]).copy()
p["p24"] = p.groupby("condition_id")["price"].shift(24)
p["mom"] = p["price"] - p["p24"]
pp = p.dropna(subset=["mom", "y"])
pp = pp[(pp["city"] != "Hong Kong") & (pp["h"] > 6)]
pp = pp.sort_values("h").groupby("condition_id", as_index=False).first()
pp["q"] = pd.qcut(pp["mom"], 5, duplicates="drop")
print(f"    n={len(pp)} bins over {stats_util.cluster_key(pp).nunique()} city-days")
print(f"    {'24h momentum quintile':>24} {'n':>5} {'mean p':>8} {'realized':>9} {'p - q':>9} {'CI':>22}")
for qq, g in pp.groupby("q", observed=True):
    a = stats_util.interval(g["price"] - g["y"].astype(float), stats_util.cluster_key(g))
    print(f"    {str(qq):>24} {len(g):>5} {g['price'].mean():>8.4f} "
          f"{g['y'].astype(float).mean():>9.4f} {a['mean']:>+9.4f} "
          f"[{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}]")
