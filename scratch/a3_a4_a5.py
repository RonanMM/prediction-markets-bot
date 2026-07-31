"""A3 Tmax/Tmin joint coherence | A4 price dynamics | A5 thin-market cohort."""
import sys

import numpy as np
import pandas as pd

import rc_lib as R

sys.path.insert(0, R.SRC)
import config  # noqa: E402
import stats_util  # noqa: E402

H = config.HALF_SPREAD
uni = pd.read_pickle("universe.pkl")
uni = uni[uni["cond"].notna() & uni["target_date"].notna()].copy()
META = uni.set_index("condition_id")

# =====================================================================================  A3
print("=" * 110)
print("A3  Tmax / Tmin JOINT COHERENCE")
print("    Hard constraint: Tmin <= Tmax always. So for any a < b, the events")
print("      A = {Tmax <= a}   and   B = {Tmin >= b}")
print("    are MUTUALLY EXCLUSIVE -> p(A) + p(B) <= 1. Any pair pricing above 1 is a dutch book")
print("    (buy NO on both: at most one can pay, so payout >= 1 against an outlay of 2-p(A)-p(B)).")

ph = R.load_price_history()
ph["hour"] = ph["ts"].dt.floor("h")
px = ph[["condition_id", "hour", "price"]].join(
    META[["city", "target_date", "kind", "cond", "lo_n", "hi_n"]], on="condition_id")
px = px.dropna(subset=["city", "target_date"])


def cumulative(g, kind, direction):
    """P(T <= a) for every a (direction='le') or P(T >= b) (direction='ge'), from bin prices."""
    sub = g[g["kind"] == kind]
    out = {}
    for _, r in sub.iterrows():
        if direction == "le" and r["cond"] == "lte":
            out[r["hi_n"]] = r["price"]
        if direction == "ge" and r["cond"] == "gte":
            out[r["lo_n"]] = r["price"]
    # augment from the tiled bins (cumulative sums)
    inner = sub[sub["cond"].isin(["range", "exact"])].sort_values("lo_n")
    if len(inner):
        lte = sub[sub["cond"] == "lte"]
        base = float(lte["price"].iloc[0]) if len(lte) == 1 else None
        if base is not None and direction == "le":
            c = base
            for _, r in inner.iterrows():
                c += r["price"]
                out[r["hi_n"]] = c
        gte = sub[sub["cond"] == "gte"]
        base = float(gte["price"].iloc[0]) if len(gte) == 1 else None
        if base is not None and direction == "ge":
            c = base
            for _, r in inner.sort_values("lo_n", ascending=False).iterrows():
                c += r["price"]
                out[r["lo_n"]] = c
    return out


viol = []
pairs_tested = 0
for (city, td, hour), g in px.groupby(["city", "target_date", "hour"]):
    if g["kind"].nunique() < 2:
        continue
    le = cumulative(g, "max", "le")     # P(Tmax <= a)
    ge = cumulative(g, "min", "ge")     # P(Tmin >= b)
    for a, pa in le.items():
        for bb, pb in ge.items():
            if bb <= a:
                continue                # need b > a for exclusivity
            pairs_tested += 1
            tot = pa + pb
            if tot > 1.0:
                net = (tot - 1) - 2 * H - config.taker_fee_per_share(pa) - config.taker_fee_per_share(pb)
                viol.append(dict(city=city, target_date=td, hour=hour, a=a, b=bb,
                                 p_max_le=pa, p_min_ge=pb, sum=tot, net=net))
v = pd.DataFrame(viol)
print(f"\n    (city, date, hour) states with BOTH a Tmax and a Tmin book: "
      f"{px.groupby(['city','target_date','hour']).kind.nunique().gt(1).sum()}")
print(f"    exclusive (a,b) pairs tested: {pairs_tested}")
print(f"    pairs pricing above 1.0 (gross violation): {len(v)}")
if len(v):
    print(f"    ... over {v.groupby(['city','target_date']).ngroups} city-days")
    print(f"    max gross excess: {(v['sum']-1).max():+.4f}; "
          f"pairs with net>0 after 2 legs of half-spread + fee: {(v['net']>0).sum()}")
    if (v["net"] > 0).any():
        w = v[v["net"] > 0]
        print(f"      -> {len(w)} profitable pairs over "
              f"{w.groupby(['city','target_date']).ngroups} city-days, median net {w['net'].median():+.4f}")
        print(w.nlargest(8, "net").to_string(index=False))
else:
    print("    -> NO violations at all: the market never prices Tmin above Tmax.")

# soft check: implied means
print("\n    Soft check — implied E[Tmax] vs E[Tmin] from normalized bin prices:")
soft = []
for (city, td, hour), g in px.groupby(["city", "target_date", "hour"]):
    ex = {}
    for kind in ("max", "min"):
        sub = g[(g["kind"] == kind) & g["cond"].isin(["range", "exact"])]
        if len(sub) < 3 or sub["price"].sum() < 0.5:
            continue
        mid = (sub["lo_n"] + sub["hi_n"]) / 2
        ex[kind] = float((mid * sub["price"]).sum() / sub["price"].sum())
    if len(ex) == 2:
        soft.append(dict(city=city, target_date=td, hour=hour, emax=ex["max"], emin=ex["min"]))
sf = pd.DataFrame(soft)
if len(sf):
    sf["gap"] = sf["emax"] - sf["emin"]
    print(f"      n={len(sf)} states over {sf.groupby(['city','target_date']).ngroups} city-days; "
          f"implied diurnal range E[Tmax]-E[Tmin]: median {sf['gap'].median():.2f}, "
          f"min {sf['gap'].min():.2f}, frac <= 0: {(sf['gap']<=0).mean():.4f}")

# =====================================================================================  A4
print("\n" + "=" * 110)
print("A4  PRICE DYNAMICS (CLOB hourly, 2026-07)")
p = ph.sort_values(["condition_id", "ts"]).copy()
p["d1"] = p.groupby("condition_id")["price"].diff()
p["d_next"] = p.groupby("condition_id")["d1"].shift(-1)
p["day_end"] = p["condition_id"].map(META["target_date"])
p = p.dropna(subset=["day_end"])
tzs = {c: config.CITIES[c]["timezone"] for c in config.CITIES}
p["city2"] = p["condition_id"].map(META["city"])
ends = {(c, d): pd.Timestamp(d, tz=tzs[c]) + pd.Timedelta(days=1)
        for c, d in set(zip(p["city2"], p["day_end"]))}
p["end_utc"] = pd.to_datetime([ends[(c, d)] for c, d in zip(p["city2"], p["day_end"])], utc=True)
p["h_to_end"] = (p["end_utc"] - p["ts"]).dt.total_seconds() / 3600
p["target_date"] = p["day_end"]
p["city"] = p["city2"]
q = p.dropna(subset=["d1", "d_next"])
q = q[(q["h_to_end"] > 0)]

print(f"    hourly bin-price changes with a next-step change: n={len(q)} over "
      f"{q.groupby(['city','target_date']).ngroups} city-days")
print("\n    AUTOCORRELATION of consecutive hourly changes (momentum>0 / mean-reversion<0):")
for lab, sub in [("all", q), ("price 0.05-0.35", q[(q["price"] >= .05) & (q["price"] <= .35)]),
                 ("price >0.35", q[q["price"] > .35]),
                 (">24h to end", q[q["h_to_end"] > 24]), ("<=24h to end", q[q["h_to_end"] <= 24])]:
    if len(sub) < 50:
        continue
    r = np.corrcoef(sub["d1"], sub["d_next"])[0, 1]
    # cluster-robust interval on the product d1*d_next (sign test of serial dependence)
    st = stats_util.interval(sub["d1"] * sub["d_next"], stats_util.cluster_key(sub))
    print(f"      {lab:<18} n={len(sub):>6} corr {r:>+7.4f}   E[d1*d_next] {st['mean']:>+.6f} "
          f"[{st['ci_lo']:+.6f},{st['ci_hi']:+.6f}]  cd={st['n_clusters']}")

print("\n    Is the price a martingale?  E[final outcome - price] by hours-to-end")
uni2 = R.add_outcomes(uni.reset_index(drop=True)) if "outcome" not in uni.columns else uni
oc = META["outcome"]
p["y"] = p["condition_id"].map(oc)
pg = p.dropna(subset=["y"])
pg = pg[pg["city"] != "Hong Kong"]
pg["band"] = pd.cut(pg["h_to_end"], [0, 6, 12, 24, 48, 72, 1e9],
                    labels=["<6h", "6-12h", "12-24h", "24-48h", "48-72h", ">72h"])
print(f"      {'lead':>8} {'n':>7} {'cd':>4} {'mean p':>8} {'realized':>9} {'p - q':>9} {'clustered 95% CI':>24}")
for bnd, g in pg.groupby("band", observed=True):
    st = stats_util.interval(g["price"] - g["y"].astype(float), stats_util.cluster_key(g))
    print(f"      {str(bnd):>8} {len(g):>7} {st['n_clusters']:>4} {g['price'].mean():>8.4f} "
          f"{g['y'].astype(float).mean():>9.4f} {st['mean']:>+9.4f} "
          f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]")

# =====================================================================================  A5
print("\n" + "=" * 110)
print("A5  THE THIN-MARKET COHORT (discovery fix 2026-07-30)")
sn = pd.read_pickle("snapshots.pkl")
first = sn.groupby("condition_id")["fetched_at"].min()
sn["first_seen"] = sn["condition_id"].map(first)
sn["cohort"] = np.where(sn["first_seen"] >= pd.Timestamp("2026-07-30", tz="UTC"),
                        "NEW (post-fix)", "OLD (pre-fix)")
u = sn.sort_values("fetched_at").groupby("condition_id").last()
print(f"    markets first seen before 2026-07-30: {(u['cohort']=='OLD (pre-fix)').sum()}")
print(f"    markets first seen on/after         : {(u['cohort']=='NEW (post-fix)').sum()}")
print("\n    per-collector-cycle market counts, last 10 days:")
cyc = sn.assign(day=sn["fetched_at"].dt.date).groupby(["day"]).agg(
    cycles=("fetched_at", lambda x: x.dt.floor("5min").nunique()),
    rows=("condition_id", "size"), markets=("condition_id", "nunique"))
cyc["per_cycle"] = (cyc["rows"] / cyc["cycles"]).round(1)
print(cyc.tail(10).to_string())

print("\n    VOLUME / LIQUIDITY distribution by cohort (last snapshot per market):")
print(u.groupby("cohort").agg(
    n=("volume_usdc", "size"),
    vol_p10=("volume_usdc", lambda x: x.quantile(.1)), vol_med=("volume_usdc", "median"),
    vol_p90=("volume_usdc", lambda x: x.quantile(.9)),
    liq_med=("liquidity_usdc", "median"),
    frac_liq_lt_1000=("liquidity_usdc", lambda x: (x < 1000).mean())).round(1).to_string())

# fair comparison: only markets whose FIRST snapshot happened on 2026-07-30 itself
d30 = sn[sn["fetched_at"].dt.date == pd.Timestamp("2026-07-30").date()]
u30 = d30.sort_values("fetched_at").groupby("condition_id").last()
print("\n    Same-cycle comparison — markets VISIBLE on 2026-07-30, split by whether we had")
print("    ever seen them before that day (removes any time trend):")
print(u30.groupby("cohort").agg(
    n=("volume_usdc", "size"), vol_med=("volume_usdc", "median"),
    vol_p25=("volume_usdc", lambda x: x.quantile(.25)),
    vol_p75=("volume_usdc", lambda x: x.quantile(.75)),
    liq_med=("liquidity_usdc", "median"), vol24_med=("volume_24h_usdc", "median")).round(1).to_string())

print("\n    graded outcomes available in the NEW cohort (can we say anything yet?):")
nu = u[u["cohort"] == "NEW (post-fix)"]
print(f"      new-cohort markets with a graded outcome: "
      f"{META.loc[META.index.isin(nu.index), 'outcome'].notna().sum()} / {len(nu)}")
print(f"      their target dates: {sorted(set(META.loc[META.index.isin(nu.index), 'target_date'].dropna()))[:12]}")
