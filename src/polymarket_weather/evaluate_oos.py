"""evaluate_oos.py — honest model evaluation against station truth (Handoff Step 3).

Answers "does the MODEL beat the market?" with the metrics that matter on a small sample —
**calibration and Brier/log-loss**, not just ROI — and benchmarks the ML calibrator against two
baselines on the same graded markets:
  * MARKET   — the Polymarket price itself (`market_prob`). If the model can't out-predict the
               price, there is no edge.
  * ENSEMBLE — the pure ECMWF/ICON ensemble predictor (`opportunities_evaluation_ensemble.csv`).
               If ML doesn't beat ensemble, the ML layer isn't adding value.

It grades each market's LAST snapshot against the resolution station (`grading.resolves_yes`,
native-unit rounding). Markets without published station truth are dropped.

⚠️ Per the pre-committed gate (CLAUDE.md / data_status.py) this prints metrics as a **PREVIEW**
and refuses a go/no-go verdict until the gate is met — small-sample ROI is noise.

Run from src/polymarket_weather/:   python evaluate_oos.py
"""
from math import log
from pathlib import Path

import pandas as pd

from grading import resolves_yes
from data_status import GATE_RESOLVED_MARKETS, GATE_OOS_BETS
import config

_OUT = Path(__file__).resolve().parent / "output"
_EPS = 1e-6


def _graded_markets(csv_path):
    """Last snapshot per market, graded against station truth.

    Returns a DataFrame with one row per gradable condition_id plus an `outcome` column
    (1 if the market resolved YES, else 0).
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df.sort_values("fetched_at").groupby("condition_id").last().reset_index()
    outcomes = []
    for _, r in df.iterrows():
        ry = resolves_yes(r["city"], r["target_date"], r["question"], r["bin_temp_c"])
        outcomes.append(None if ry is None else int(ry))
    df["outcome"] = outcomes
    return df.dropna(subset=["outcome"]).reset_index(drop=True)


def _brier(p, y):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


def _logloss(p, y):
    s = 0.0
    for pi, yi in zip(p, y):
        pi = min(1 - _EPS, max(_EPS, pi))
        s += -(yi * log(pi) + (1 - yi) * log(1 - pi))
    return s / len(p)


def _calibration(p, y, bins=10):
    """Reliability table: (range, n, mean predicted, realized frequency) per probability bin."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, pi in enumerate(p) if (pi >= lo and (pi < hi or (b == bins - 1 and pi <= hi)))]
        if not idx:
            continue
        pred = sum(p[i] for i in idx) / len(idx)
        real = sum(y[i] for i in idx) / len(idx)
        rows.append((lo, hi, len(idx), pred, real))
    return rows


def _roi_at_production(df):
    """ROI / win-rate at production params, applying per-group and portfolio Kelly caps.

    Each eval row already passed MIN_EDGE and has a per-bet kelly (capped at MAX_KELLY_PER_BET);
    here we apply the (city, date) group cap and the total portfolio cap from config, then settle
    each $1,000-bankroll bet against the graded outcome (2% fee).
    """
    g = df.copy()
    g["group_key"] = g["city"].astype(str) + "|" + g["target_date"].astype(str)
    g["k"] = g["kelly"].astype(float)
    # group cap
    for key, grp in g.groupby("group_key"):
        tot = grp["k"].sum()
        if tot > config.MAX_KELLY_PER_GROUP:
            g.loc[grp.index, "k"] = grp["k"] * (config.MAX_KELLY_PER_GROUP / tot)
    # portfolio cap
    tot = g["k"].sum()
    if tot > config.MAX_TOTAL_KELLY:
        g["k"] = g["k"] * (config.MAX_TOTAL_KELLY / tot)

    profit = staked = 0.0
    wins = n = 0
    for _, r in g.iterrows():
        size = 1000.0 * r["k"]
        if size < 1.0:
            continue
        their = r["their_prob"]
        if not (0 < their < 1):
            continue
        won = (r["bet_side"] == "Yes" and r["outcome"] == 1) or \
              (r["bet_side"] == "No" and r["outcome"] == 0)
        profit += (size / their - size) * 0.98 if won else -size
        staked += size
        wins += int(won)
        n += 1
    return {"bets": n, "wins": wins, "roi": (profit / staked if staked else 0.0),
            "profit": profit, "staked": staked}


def main():
    ml = _graded_markets(_OUT / "opportunities_evaluation_ml.csv")
    if ml is None or ml.empty:
        print("No gradable ML markets yet — run the pipeline and refresh station truth first.")
        return

    y = ml["outcome"].tolist()
    p_model = ml["forecast_prob"].tolist()   # model P(YES)
    p_mkt = ml["market_prob"].tolist()       # market P(YES)

    print("\n==================== OOS MODEL EVALUATION (station truth) ====================")
    print(f"Gradable markets: {len(ml)}   (date span {ml['target_date'].min()}..{ml['target_date'].max()})")
    print("\n  Probabilistic accuracy (lower Brier / log-loss = better):")
    print(f"    {'predictor':<10} {'Brier':>8} {'log-loss':>10}")
    print(f"    {'MODEL':<10} {_brier(p_model, y):>8.4f} {_logloss(p_model, y):>10.4f}")
    print(f"    {'MARKET':<10} {_brier(p_mkt, y):>8.4f} {_logloss(p_mkt, y):>10.4f}")

    # Ensemble baseline on the intersection of gradable markets.
    ens = _graded_markets(_OUT / "opportunities_evaluation_ensemble.csv")
    if ens is not None and not ens.empty:
        common = set(ml["condition_id"]) & set(ens["condition_id"])
        if common:
            e = ens[ens["condition_id"].isin(common)].set_index("condition_id")
            m = ml[ml["condition_id"].isin(common)].set_index("condition_id")
            ye = e["outcome"].tolist()
            print(f"    {'ENSEMBLE':<10} {_brier(e['forecast_prob'].tolist(), ye):>8.4f} "
                  f"{_logloss(e['forecast_prob'].tolist(), ye):>10.4f}   "
                  f"(vs MODEL {_brier(m['forecast_prob'].tolist(), m['outcome'].tolist()):.4f} "
                  f"on the same {len(common)} markets)")

    print("\n  Model calibration (predicted P(YES) vs realized frequency):")
    print(f"    {'bin':<12} {'n':>4} {'pred':>7} {'realized':>9}")
    for lo, hi, n_, pred, real in _calibration(p_model, y):
        print(f"    [{lo:.1f},{hi:.1f})    {n_:>4} {pred:>7.3f} {real:>9.3f}")

    roi = _roi_at_production(ml)
    wr = roi["wins"] / roi["bets"] if roi["bets"] else 0.0
    print(f"\n  ROI at production params: {roi['roi']:.1%}  on {roi['bets']} bets  "
          f"(win rate {wr:.1%}, staked ${roi['staked']:.0f})")

    # ---- Gate guard: metrics are a PREVIEW until the pre-committed sample gate is met ----------
    gate_met = len(ml) >= GATE_RESOLVED_MARKETS and roi["bets"] >= GATE_OOS_BETS
    print()
    if gate_met:
        print("  ✅ GATE MET — these metrics support a genuine go/no-go decision.")
        print("     Read calibration + Brier-vs-MARKET first; ROI second. If MODEL Brier is not")
        print("     below MARKET Brier, there is no demonstrated edge regardless of ROI.")
    else:
        print(f"  ⚠️  PREVIEW ONLY — NOT A VERDICT. Gate not met "
              f"(markets {len(ml)}/{GATE_RESOLVED_MARKETS}, bets {roi['bets']}/{GATE_OOS_BETS}).")
        print("     Do not act on these numbers; the sample is too small. Keep accumulating "
              "(see data_status.py).")
    print("=============================================================================")


if __name__ == "__main__":
    main()
