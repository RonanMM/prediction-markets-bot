"""A1b — how big is the over-round, when does it occur, and what execution cost kills it."""
import numpy as np
import pandas as pd

import rc_lib as R
import config

uni = pd.read_pickle("universe.pkl").set_index("condition_id")
b = pd.read_pickle("baskets_clob.pkl")
ph = R.load_price_history()

TZ = {c: config.CITIES[c]["timezone"] for c in config.CITIES}

# hours from basket timestamp to the END of the target local day (resolution moment)
end_utc = {}
for (city, td), _ in b.groupby(["city", "target_date"]):
    tz = TZ[city]
    e = pd.Timestamp(td, tz=tz) + pd.Timedelta(days=1)
    end_utc[(city, td)] = e.tz_convert("UTC")
b["end_utc"] = [end_utc[(c, d)] for c, d in zip(b["city"], b["target_date"])]
b["h_to_end"] = (b["end_utc"] - b["t"]).dt.total_seconds() / 3600.0

# listing age: hours since the FIRST clob observation of the youngest bin in the basket
first_seen = ph.groupby("condition_id")["ts"].min()
uni2 = uni.reset_index()[["condition_id", "city", "target_date", "kind"]]
uni2["first_seen"] = uni2["condition_id"].map(first_seen)
grp_first = uni2.dropna(subset=["first_seen"]).groupby(["city", "target_date", "kind"])["first_seen"].max()
b["grp_first"] = [grp_first.get((c, d, k), pd.NaT) for c, d, k in zip(b["city"], b["target_date"], b["kind"])]
b["age_h"] = (b["t"] - b["grp_first"]).dt.total_seconds() / 3600.0

print("=" * 100)
print("A1b.1  over-round vs time-to-resolution (CLOB hourly, baskets with >=6 bins)")
big = b[b["n_bins"] >= 6].copy()
big["band"] = pd.cut(big["h_to_end"], [-1, 6, 12, 24, 36, 48, 72, 1e9],
                     labels=["<6h", "6-12h", "12-24h", "24-36h", "36-48h", "48-72h", ">72h"])
print(big.groupby("band", observed=True).agg(
    n=("sum_p", "size"), citydays=("target_date", "nunique"), med_bins=("n_bins", "median"),
    med_sum=("sum_p", "median"), mean_sum=("sum_p", "mean"),
    frac_gt1=("sum_p", lambda s: (s > 1).mean())).round(4).to_string())

print("\n  same, restricted to EXHAUSTIVE baskets (true full books):")
e = b[b["exhaustive"]].copy()
e["band"] = pd.cut(e["h_to_end"], [-1, 6, 12, 24, 36, 48, 72, 1e9],
                   labels=["<6h", "6-12h", "12-24h", "24-36h", "36-48h", "48-72h", ">72h"])
print(e.groupby("band", observed=True).agg(
    n=("sum_p", "size"), citydays=("target_date", "nunique"),
    med_sum=("sum_p", "median"), mean_sum=("sum_p", "mean"),
    frac_gt1=("sum_p", lambda s: (s > 1).mean())).round(4).to_string())

print("\n  by hours since the book was first quoted (age):")
big["ageband"] = pd.cut(big["age_h"], [-1e9, 0.5, 2, 6, 12, 24, 1e9],
                        labels=["<0.5h (fresh)", "0.5-2h", "2-6h", "6-12h", "12-24h", ">24h"])
print(big.groupby("ageband", observed=True).agg(
    n=("sum_p", "size"), med_sum=("sum_p", "median"), mean_sum=("sum_p", "mean"),
    frac_gt1=("sum_p", lambda s: (s > 1).mean())).round(4).to_string())

print("\n" + "=" * 100)
print("A1b.2  execution-cost sensitivity of the sell-everything dutch book")
print("  (net = (sum_p - 1) - n_bins*h - verified taker fee; guaranteed profit if > 0)")
fee = b.apply(lambda r: 0.0, axis=1)  # recomputed below per basket from the stored net at h
print(f"    {'half-spread h':>14} {'profitable baskets':>19} {'city-days':>10} {'median net':>11} {'max net':>9}")
for h in [0.000, 0.002, 0.005, 0.010, 0.020]:
    net = b["over_net"] + b["n_bins"] * (config.HALF_SPREAD - h)
    ok = b[net > 0]
    print(f"    {h:>14.3f} {len(ok):>19} {ok.groupby(['city','target_date']).ngroups:>10} "
          f"{(net[net>0].median() if len(ok) else float('nan')):>11.4f} {net.max():>9.4f}")

print("\n  ... excluding books younger than 2h (the fresh-listing wide-book regime):")
m = b["age_h"] >= 2
for h in [0.000, 0.002, 0.005, 0.010]:
    net = (b["over_net"] + b["n_bins"] * (config.HALF_SPREAD - h))[m]
    ok = b[m][net > 0]
    print(f"    {h:>14.3f} {len(ok):>19} {ok.groupby(['city','target_date']).ngroups:>10} "
          f"{(net[net>0].median() if len(ok) else float('nan')):>11.4f} {net.max():>9.4f}")

print("\n" + "=" * 100)
print("A1b.3  is the over-round tradeable?  liquidity of the bins in over-round snapshot baskets")
sn = pd.read_pickle("snapshots.pkl")
sn = sn[sn["cond"].notna() & sn["yes"].notna()].copy()
sn["cycle"] = sn["fetched_at"].dt.floor("5min")
bs = pd.read_pickle("baskets_snap.pkl")
key = bs.set_index(["city", "target_date", "kind", "t"])["sum_p"]
sn["sum_p"] = [key.get((c, d, k, t), np.nan) for c, d, k, t in
               zip(sn["city"], sn["target_date"], sn["kind"], sn["cycle"])]
sn2 = sn.dropna(subset=["sum_p"])
sn2 = sn2.assign(band=pd.cut(sn2["sum_p"], [0, .5, .9, 1.0, 1.02, 1.05, 10]))
print(sn2.groupby("band", observed=True).agg(
    n=("liquidity_usdc", "size"), med_liq=("liquidity_usdc", "median"),
    med_vol=("volume_usdc", "median"), med_vol24=("volume_24h_usdc", "median")).round(1).to_string())
