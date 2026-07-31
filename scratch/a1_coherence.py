"""A1 — PMF coherence / dutch book. Basket-level completeness, both snapshot and CLOB feeds."""
import numpy as np
import pandas as pd

import rc_lib as R
import config

H = config.HALF_SPREAD


def taker(p):
    return config.WEATHER_TAKER_RATE * p * (1 - p)


uni = pd.read_pickle("universe.pkl")
uni = uni[uni["cond"].notna()]
META = uni.set_index("condition_id")[["city", "target_date", "kind", "cond", "lo_n", "hi_n",
                                      "unit", "outcome", "question", "temp_c"]]


def basket_frame(df, price_col, time_col, extra=()):
    """df: rows of (condition_id, price, time). Returns one row per (city,date,kind,time)."""
    d = df.join(META, on="condition_id", how="inner")
    d = d[d["target_date"].notna()]
    recs = []
    for (city, td, kind, t), g in d.groupby(["city", "target_date", "kind", time_col]):
        g = g.drop_duplicates("condition_id")
        p = g[price_col].to_numpy(float)
        n = len(p)
        if n < 2:
            continue
        s = p.sum()
        # exhaustive?  one lte, one gte, interior tiles the whole gap
        lte = g[g["cond"] == "lte"]; gte = g[g["cond"] == "gte"]
        inner = g[g["cond"].isin(["range", "exact"])]
        exhaustive = False
        if len(lte) == 1 and len(gte) == 1 and len(inner) > 0:
            lo = float(lte["hi_n"].iloc[0]) + 1
            hi = float(gte["lo_n"].iloc[0]) - 1
            cov = set()
            for _, r in inner.iterrows():
                cov |= set(np.round(np.arange(r["lo_n"], r["hi_n"] + .5, 1.0), 1))
            need = set(np.round(np.arange(lo, hi + .5, 1.0), 1))
            exhaustive = (need == cov)
        rec = dict(city=city, target_date=td, kind=kind, t=t, n_bins=n, sum_p=s,
                   exhaustive=exhaustive,
                   over_net=(s - 1) - n * H - taker(p).sum(),
                   over_net_maker=(s - 1),
                   under_net=(1 - s) - n * H - taker(p).sum(),
                   under_net_maker=(1 - s),
                   n_yes=g["outcome"].sum(skipna=True) if g["outcome"].notna().all() else np.nan)
        for c in extra:
            rec[c] = g[c].min()
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- CLOB hourly (July, dense)
ph = R.load_price_history()
ph = ph.rename(columns={"price": "p"})
ph["hour"] = ph["ts"].dt.floor("h")
bh = basket_frame(ph[["condition_id", "p", "hour"]], "p", "hour")
bh.to_pickle("baskets_clob.pkl")

# ---------------------------------------------------------------- snapshot cycles (Mar-Jul)
sn = pd.read_pickle("snapshots.pkl")
sn = sn[sn["cond"].notna() & sn["yes"].notna()].copy()
sn["cycle"] = sn["fetched_at"].dt.floor("5min")
bs = basket_frame(sn[["condition_id", "yes", "cycle", "liquidity_usdc", "volume_usdc"]],
                  "yes", "cycle", extra=("liquidity_usdc",))
bs.to_pickle("baskets_snap.pkl")


def report(b, label):
    print("=" * 100)
    print(f"{label}: {len(b)} baskets (>=2 bins), "
          f"{b.groupby(['city','target_date']).ngroups} city-days, "
          f"{b['exhaustive'].sum()} exhaustive")
    print("\n  sum(YES) over observed bins:")
    print("   ", b["sum_p"].describe(percentiles=[.05, .5, .9, .95, .99]).round(4).to_dict())
    print("\n  by n_bins:")
    print(b.groupby(pd.cut(b["n_bins"], [1, 2, 3, 4, 6, 8, 10, 30]), observed=True)
          .agg(n=("sum_p", "size"), med_sum=("sum_p", "median"),
               frac_gt1=("sum_p", lambda s: (s > 1).mean()),
               n_exh=("exhaustive", "sum")).round(4).to_string())

    print("\n  -- OVER-ROUND (buy NO on every observed bin; needs only mutual exclusivity) --")
    ov = b[b["over_net"] > 0]
    print(f"    taker, net of {H:.2f}/leg half-spread + verified fee: {len(ov)}/{len(b)} baskets profitable")
    if len(ov):
        print(f"      city-days {ov.groupby(['city','target_date']).ngroups}, "
              f"median net {ov['over_net'].median():+.4f}, max {ov['over_net'].max():+.4f}")
    ovm = b[b["over_net_maker"] > 0]
    print(f"    maker (zero fee, no crossing): {len(ovm)}/{len(b)} with sum>1, "
          f"median excess {ovm['over_net_maker'].median() if len(ovm) else float('nan'):+.4f}, "
          f"max {b['over_net_maker'].max():+.4f}")
    # what half-spread would be needed
    hb = b.copy()
    hb["breakeven_h"] = (hb["sum_p"] - 1 - hb["n_bins"] * 0) / hb["n_bins"]
    print(f"    break-even half-spread per leg (ignoring fees) at the 99th pct basket: "
          f"{hb['breakeven_h'].quantile(.99):.4f}  max {hb['breakeven_h'].max():.4f}")

    print("\n  -- UNDER-ROUND (buy YES on every bin; needs EXHAUSTIVE set) --")
    e = b[b["exhaustive"]]
    if len(e):
        print(f"    exhaustive baskets: {len(e)} over "
              f"{e.groupby(['city','target_date','kind']).ngroups} city-day-kinds")
        print("      sum_p:", e["sum_p"].describe(percentiles=[.05, .5, .95]).round(4).to_dict())
        print(f"      taker buy-all-YES net>0: {(e['under_net']>0).sum()}/{len(e)}   "
              f"median net {e['under_net'].median():+.4f}")
        print(f"      taker sell-all-NO  net>0: {(e['over_net']>0).sum()}/{len(e)}")
        if (e["under_net"] > 0).any():
            print(e[e["under_net"] > 0].nlargest(8, "under_net")[
                ["city", "target_date", "kind", "t", "n_bins", "sum_p", "under_net"]].to_string(index=False))
    else:
        print("    none")
    print()


report(bh, "A1a  CLOB hourly price history (2026-07-01 .. 07-30)")
report(bs, "A1b  Snapshot collector cycles (2026-03-17 .. 07-30)")
