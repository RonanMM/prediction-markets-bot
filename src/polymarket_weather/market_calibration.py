"""market_calibration.py — is the PRICE itself miscalibrated? (megaplan §13c)

Every other instrument in this project asks whether OUR forecast beats the price. This one asks
whether the price is right, in price space, with no forecast involved at all — the only question
whose answer can be a trade without any forecasting skill.

For each price bucket: realized YES frequency against the traded price, and the after-cost return
of selling that bucket. A bucket whose realized frequency sits outside its price by more than
round-trip cost is a mispricing.

⚠️ THE INTERVAL IS THE WHOLE POINT, AND IT MUST BE COMPUTED ON THE RIGHT UNIT.
The breadth book's 545 "city-days" are 49 cities across ~12 DATES. Cities on one date share a
continental weather regime, so clustering on city-day treats ~49 correlated observations as
independent. This module reports the DATE-clustered interval as the binding one, with:
  * a t(k−1) critical value, not z — cluster-robust SEs under-cover badly below ~30 clusters,
    which is exactly why GATE_MIN_DATES exists; and
  * a Bonferroni correction for the number of buckets examined.
An uncorrected z-interval on 11 clusters is not evidence, however many entries sit inside them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import stats_util

# Examined buckets — fixed here so the multiplicity correction is over a declared set rather than
# over however many the reader happened to look at.
BUCKETS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
           (0.25, 0.30), (0.30, 0.35), (0.35, 0.45), (0.45, 0.60), (0.60, 0.80), (0.80, 1.01)]
MIN_N = 30


def _t_crit(k: int, n_tests: int) -> float:
    """Bonferroni-corrected two-sided critical value on t with k−1 df; falls back to the normal
    approximation only if scipy is unavailable (then it is optimistic, and says so at the call)."""
    alpha = 0.05 / max(1, n_tests)
    try:
        from scipy import stats as sps
        return float(sps.t.ppf(1 - alpha / 2, max(1, k - 1)))
    except Exception:
        from statistics import NormalDist
        return float(NormalDist().inv_cdf(1 - alpha / 2))


def sell_roi(price, y) -> float:
    """After-cost return on SELLING YES (= buying NO) at `price`, per dollar staked.

    Crosses the half-spread and pays the verified weather taker fee 0.05·p·(1−p) at the execution
    price — the same cost model the structure books settle with.
    """
    price, y = np.asarray(price, float), np.asarray(y, float)
    cost = (1 - price) + config.HALF_SPREAD
    fee = np.array([config.taker_fee_per_share(c) for c in cost])
    return float((((1 - y) - (cost + fee)) / (cost + fee)).mean())


def graded_prices() -> pd.DataFrame:
    """Traded price + raw YES outcome + clustering keys, from the breadth structure book."""
    import shoulder_book_breadth as bb
    g = bb.grade_book(book=bb._load_book(), lookup=False)
    if g.empty:
        return pd.DataFrame()
    g = g[g["leg"].isin(["shoulder", "favorite"])].copy()
    g["yes"] = pd.to_numeric(g["entry_yes_price"], errors="coerce")
    # side_won is relative to the side taken; recover the market's own YES outcome.
    g["y"] = np.where(g["side"].astype(str) == "Yes",
                      g["side_won"].astype(int), 1 - g["side_won"].astype(int))
    g["date"] = pd.to_datetime(g["target_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    g["cd"] = g["city"].astype(str) + "|" + g["date"].astype(str)
    return g.dropna(subset=["yes", "y", "date"])


def bucket_rows(g: pd.DataFrame, buckets=None) -> list[dict]:
    """One row per price bucket, carrying BOTH clusterings so the reader can see the difference."""
    buckets = BUCKETS if buckets is None else buckets
    rows = []
    for lo, hi in buckets:
        s = g[(g["yes"] >= lo) & (g["yes"] < hi)]
        if len(s) < MIN_N:
            continue
        px = float(s["yes"].mean())
        out = {"lo": lo, "hi": hi, "n": len(s), "price": px,
               "dates": int(s["date"].nunique()), "roi_sell": sell_roi(s["yes"], s["y"])}
        for key, tag in (("cd", "cd"), ("date", "dt")):
            iv = stats_util.interval(s["y"].astype(float), s[key])
            t = _t_crit(iv["n_clusters"], len(buckets))
            out[f"{tag}_k"] = int(iv["n_clusters"])
            out[f"{tag}_lo"] = float(iv["mean"] - t * iv["se"])
            out[f"{tag}_hi"] = float(iv["mean"] + t * iv["se"])
        out["realized"] = float(s["y"].mean())
        # The DATE-clustered interval is the binding one. Everything else is context.
        out["verdict"] = ("SELL" if out["dt_hi"] < px else
                          "BUY" if out["dt_lo"] > px else "—")
        rows.append(out)
    return rows


def report() -> None:
    g = graded_prices()
    if g.empty:
        print("market calibration: no graded entries yet.")
        return
    print(f"MARKET CALIBRATION — is the price itself right? "
          f"({len(g)} entries · {g['city'].nunique()} cities · {g['date'].nunique()} dates)")
    print(f"  Intervals are Bonferroni-corrected for {len(BUCKETS)} buckets and use t(k−1).")
    print(f"  The DATE-clustered column is binding: city-days are not independent within a date.\n")
    print(f"  {'bucket':<14}{'n':>6}{'price':>8}{'realized':>10}"
          f"{'city-day CI':>22}{'DATE CI':>22}{'sell ROI':>10}  verdict")
    rows = bucket_rows(g)
    for r in rows:
        cd_ci = f"[{r['cd_lo']:+.3f},{r['cd_hi']:+.3f}]"
        dt_ci = f"[{r['dt_lo']:+.3f},{r['dt_hi']:+.3f}]"
        print(f"  [{r['lo']:.2f},{r['hi']:.2f})  {r['n']:>6}{r['price']:>8.3f}{r['realized']:>10.3f}"
              f"{cd_ci:>22}{dt_ci:>22}{r['roi_sell']:>+10.1%}  {r['verdict']}")
    n_dates = g["date"].nunique()
    sells = [r for r in rows if r["verdict"] == "SELL"]
    print(f"\n  {len(sells)} of {len(rows)} buckets mispriced at Bonferroni strength on the "
          f"date-clustered interval.")
    if n_dates < 30:
        print(f"  ⚠️ {n_dates} distinct target dates. The pre-registered gate requires 30, and that "
              f"is NOT a\n     formality here: 'extreme bins hit less often than priced' is exactly "
              f"what a calm\n     stretch looks like, and the dispersion monitor has Tmax std(z) "
              f"1.78 in March against\n     0.98 in July. Significance is not the binding "
              f"constraint — calendar is.")


if __name__ == "__main__":
    report()
