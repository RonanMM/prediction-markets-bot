"""A0 — structural sanity: do the bin sets partition, and does grading put exactly one YES in each?"""
import numpy as np
import pandas as pd

import rc_lib as R

u = pd.read_pickle("universe.pkl")
u = u[u["cond"].notna()].copy()
u["step"] = np.where(u["unit"].str.contains("°F"), 1.0, 1.0)   # native grid step (whole °F / whole °C)

# --- partition test: with lte T_lo and gte T_hi and interior bins covering (T_lo, T_hi) -----
rows = []
for (city, td, kind), g in u.groupby(["city", "target_date", "kind"]):
    lte = g[g["cond"] == "lte"]
    gte = g[g["cond"] == "gte"]
    inner = g[g["cond"].isin(["range", "exact"])].sort_values("lo_n")
    covered = set()
    for _, r in inner.iterrows():
        for t in np.arange(r["lo_n"], r["hi_n"] + 0.5, 1.0):
            covered.add(round(t, 1))
    gap = np.nan
    complete = False
    if len(lte) == 1 and len(gte) == 1 and len(inner) > 0:
        lo = float(lte["hi_n"].iloc[0]) + 1
        hi = float(gte["lo_n"].iloc[0]) - 1
        need = set(round(t, 1) for t in np.arange(lo, hi + 0.5, 1.0))
        gap = len(need - covered)
        complete = (gap == 0) and (min(covered) >= lo) and (max(covered) <= hi)
    ny = g["outcome"].sum(skipna=True) if g["outcome"].notna().any() else np.nan
    rows.append(dict(city=city, target_date=td, kind=kind, n_bins=len(g),
                     has_lte=len(lte), has_gte=len(gte), n_inner=len(inner),
                     gap=gap, complete=complete,
                     n_graded=int(g["outcome"].notna().sum()),
                     n_yes=ny, all_graded=bool(g["outcome"].notna().all())))
d = pd.DataFrame(rows)
d.to_pickle("citydays.pkl")

print("=" * 100)
print("A0.1  bin-set completeness per (city, target_date, kind)")
print(f"total city-day-kinds: {len(d)}")
print(d.groupby("city").agg(n=("complete", "size"), complete=("complete", "sum"),
                            med_bins=("n_bins", "median"), has_both_ends=("has_lte", lambda s: (s > 0).sum())).to_string())

print("\n  completeness by month (fetched universe):")
d["month"] = d["target_date"].str[:7]
print(d.groupby(["month"]).agg(n=("complete", "size"), complete=("complete", "sum"),
                               med_bins=("n_bins", "median")).to_string())

print("\n" + "=" * 100)
print("A0.2  EXACTLY-ONE-YES test (a true partition must have exactly one winning bin)")
full = d[d["complete"] & d["all_graded"]]
print(f"complete AND fully graded city-day-kinds: {len(full)}")
print(full.groupby("city").apply(
    lambda g: pd.Series({"n": len(g), "mean_yes": g["n_yes"].mean(),
                         "frac_exactly_1": (g["n_yes"] == 1).mean()}), include_groups=False).round(3).to_string())

print("\n  same test on ALL fully-graded city-day-kinds with >=4 bins (not just 'complete'):")
sub = d[(d["all_graded"]) & (d["n_bins"] >= 4)]
print(sub.groupby("city").apply(
    lambda g: pd.Series({"n": len(g), "mean_bins": g["n_bins"].mean(), "mean_yes": g["n_yes"].mean(),
                         "frac_exactly_1": (g["n_yes"] == 1).mean()}), include_groups=False).round(3).to_string())
