"""stats_util.py — the inference shared by every pre-registered gate in this repo.

One estimator, one place. Both gate families ask the same question — "is this effect
distinguishable from zero?" — and both were originally answered with a bare point estimate:

  * the structure book's taker/maker gates (`shoulder_book`), which on 2026-07-27 reported
    "gate MET" at n=150 / +0.0234 with a clustered 95% CI of [-0.023, +0.070];
  * the E3 per-bucket forward gates (`evaluate_oos`), which passed on `model_brier <=
    market_brier` over as few as 5 graded bets.

THE UNIT OF INDEPENDENCE IS A CITY-DAY, NOT A BET. Every temperature bin for one city on one
day settles on a single weather outcome, so those bets are one observation, not eleven. Note
the correction runs in BOTH directions and is about correctness, not conservatism: same-day
bins are mutually exclusive (one bin winning forces its neighbours to lose), so clustering
TIGHTENS these intervals — measured 0.0236 clustered vs 0.0282 iid on the full shoulder band.
It would widen them for positively-correlated bets. Either way the iid number is the wrong one.
"""
from __future__ import annotations

import pandas as pd

MIN_CLUSTERS = 30      # below this a cluster-robust interval is not meaningful
Z = 1.96               # two-sided 95% (equivalently a one-sided 97.5% test against zero)
CLUSTER_COLS = ("city", "target_date")


def cluster_key(sub: pd.DataFrame) -> pd.Series:
    """One city-day = one weather outcome, however many bins were traded on it.

    Falls back to one-cluster-per-row when the frame carries no city/target_date, which is
    exactly the iid assumption — never a weaker one."""
    if all(c in sub.columns for c in CLUSTER_COLS):
        return sub["city"].astype(str) + "|" + sub["target_date"].astype(str)
    return pd.Series([str(i) for i in range(len(sub))], index=sub.index)


def clustered_mean_se(values, clusters) -> tuple[float, float, int]:
    """Cluster-robust SE of the mean → (mean, se, n_clusters). se is inf below 2 clusters."""
    v = pd.Series(list(values), dtype=float).reset_index(drop=True)
    c = pd.Series(list(clusters)).reset_index(drop=True)
    n = len(v)
    if n == 0:
        return 0.0, float("inf"), 0
    m = float(v.mean())
    sums = (v - m).groupby(c).sum()
    g = int(len(sums))
    if g < 2:
        return m, float("inf"), g
    var = float((sums ** 2).sum()) / (n ** 2) * (g / (g - 1))
    return m, var ** 0.5, g


def interval(values, clusters) -> dict:
    """mean + clustered 95% interval, as a dict the gate reporters can print directly."""
    mean, se, g = clustered_mean_se(values, clusters)
    lo, hi = ((mean - Z * se, mean + Z * se) if se != float("inf")
              else (float("-inf"), float("inf")))
    return {"n": len(pd.Series(list(values))), "n_clusters": g, "mean": mean, "se": se,
            "ci_lo": lo, "ci_hi": hi}
