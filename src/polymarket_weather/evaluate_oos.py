"""evaluate_oos.py — honest model evaluation against station truth (Handoff Step 3).

Answers "does the MODEL beat the market?" with the metrics that matter on a small sample —
**calibration and Brier/log-loss**, not just ROI — and benchmarks the calibrator (EMOS) against two
baselines on the same graded markets:
  * MARKET   — the Polymarket price itself (`market_prob`). If the model can't out-predict the
               price, there is no edge.
  * ENSEMBLE — the pure ECMWF/ICON ensemble predictor (`opportunities_evaluation_ensemble.csv`).
               If the calibrator doesn't beat ensemble, the calibration layer isn't adding value.

It grades each market's LAST snapshot against the resolution station (`grading.resolves_yes`,
native-unit rounding). Markets without published station truth are dropped.

⚠️ Per the pre-committed gate (CLAUDE.md / data_status.py) this prints metrics as a **PREVIEW**
and refuses a go/no-go verdict until the gate is met — small-sample ROI is noise.

Run from src/polymarket_weather/:   python evaluate_oos.py
"""
from math import log
from pathlib import Path

import pandas as pd

import numpy as np

from grading import resolves_yes, fetch_actual_weather
from data_status import GATE_RESOLVED_MARKETS, GATE_OOS_BETS
from pmf import _cdf
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


def _crps_student_t(mu, sigma, nu, y):
    """CRPS of a Student-t predictive distribution vs the observed temperature y (°C).

    CRPS = ∫ (F(x) - 1{x>=y})^2 dx, evaluated numerically over a grid wide enough to cover the
    tails. Lower is better. Unlike per-market Brier, this scores the whole *temperature*
    distribution, so it distinguishes a well-calibrated forecast from an overconfident one.
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (mu, sigma, nu, y)):
        return None
    hw = max(12.0, 6.0 * float(sigma))
    xs = np.linspace(float(mu) - hw, float(mu) + hw, 241)
    F = np.array([_cdf(x, float(mu), float(sigma), float(nu)) for x in xs])
    step = (xs >= float(y)).astype(float)
    integrand = (F - step) ** 2
    # Trapezoid rule, written out to stay compatible across NumPy 1.x/2.x (np.trapz removed in 2.0).
    return float(np.sum((integrand[:-1] + integrand[1:]) / 2.0 * np.diff(xs)))


def _mean_crps(csv_path):
    """Mean temperature-level CRPS for one predictor's eval tracker, or None if ungradable.

    One predictive distribution per (city, target_date) (last snapshot), scored against the
    resolution-station observation. Uses the daily MIN for 'lowest' markets (via fetch_actual_weather).
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    need = {"forecast_mu", "forecast_sigma", "forecast_nu", "city", "target_date"}
    if not need.issubset(df.columns):
        return None
    df = df.sort_values("fetched_at").groupby(["city", "target_date"]).last().reset_index()
    vals = []
    for _, r in df.iterrows():
        y = fetch_actual_weather(r["city"], r["target_date"], r.get("question", ""))
        if y is None:
            continue
        c = _crps_student_t(r["forecast_mu"], r["forecast_sigma"], r["forecast_nu"], y)
        if c is not None:
            vals.append(c)
    return (sum(vals) / len(vals), len(vals)) if vals else None


def _best_shrink_weight(ml):
    """Sweep the shrink-to-market weight and return (best_w, best_brier, brier_w1, brier_w0).

    Valid only when the tracker was generated with no shrink (config.SHRINK_WEIGHT=1.0, the default),
    so `forecast_prob` is the pure model P(YES). Blends toward the RAW market price on the same YES
    scale the engine trades on. w=1 is pure model, w=0 is pure market.
    """
    if "market_prob_raw" not in ml.columns:
        return None
    y = ml["outcome"].tolist()
    fp = ml["forecast_prob"].tolist()
    mp = ml["market_prob_raw"].tolist()
    best = None
    b_w1 = b_w0 = None
    for i in range(21):
        w = i / 20.0
        p = [w * f + (1.0 - w) * m for f, m in zip(fp, mp)]
        b = _brier(p, y)
        if w == 1.0:
            b_w1 = b
        if w == 0.0:
            b_w0 = b
        if best is None or b < best[1]:
            best = (w, b)
    return best[0], best[1], b_w1, b_w0


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
        # Honest execution cost: cross half the spread on entry, so we buy fewer shares.
        their_eff = min(1.0 - _EPS, their + config.HALF_SPREAD)
        shares = size / their_eff
        won = (r["bet_side"] == "Yes" and r["outcome"] == 1) or \
              (r["bet_side"] == "No" and r["outcome"] == 0)
        # Winning payout is $1/share minus the taker fee; a loss forfeits the stake.
        profit += (shares * (1.0 - config.FEE_RATE) - size) if won else -size
        staked += size
        wins += int(won)
        n += 1
    return {"bets": n, "wins": wins, "roi": (profit / staked if staked else 0.0),
            "profit": profit, "staked": staked}


def main():
    ml = _graded_markets(_OUT / "opportunities_evaluation_calibrated.csv")
    if ml is None or ml.empty:
        print("No gradable ML markets yet — run the pipeline and refresh station truth first.")
        return

    y = ml["outcome"].tolist()
    p_model = ml["forecast_prob"].tolist()   # model P(YES)
    # MARKET benchmark = the RAW tradeable price (market_prob_raw), NOT the normalized market_prob.
    # Normalizing the bins to sum to 1 flatters the market's Brier; the EDGE CHECK must grade the
    # model against the price we would actually pay — the same series the shrink sweep already uses.
    # Fall back to the normalized column only if an older tracker lacks the raw price.
    mkt_col = "market_prob_raw" if "market_prob_raw" in ml.columns else "market_prob"
    p_mkt = ml[mkt_col].tolist()             # market P(YES) — raw tradeable price

    print("\n==================== OOS MODEL EVALUATION (station truth) ====================")
    print(f"Gradable markets: {len(ml)}   (date span {ml['target_date'].min()}..{ml['target_date'].max()})")
    brier_model, brier_mkt = _brier(p_model, y), _brier(p_mkt, y)
    print("\n  Probabilistic accuracy (lower Brier / log-loss = better; MARKET = raw tradeable price):")
    print(f"    {'predictor':<10} {'Brier':>8} {'log-loss':>10}")
    print(f"    {'MODEL':<10} {brier_model:>8.4f} {_logloss(p_model, y):>10.4f}")
    print(f"    {'MARKET':<10} {brier_mkt:>8.4f} {_logloss(p_mkt, y):>10.4f}")

    # Ensemble baseline on the intersection of gradable markets.
    brier_ens = None
    ens = _graded_markets(_OUT / "opportunities_evaluation_ensemble.csv")
    if ens is not None and not ens.empty:
        common = set(ml["condition_id"]) & set(ens["condition_id"])
        if common:
            e = ens[ens["condition_id"].isin(common)].set_index("condition_id")
            m = ml[ml["condition_id"].isin(common)].set_index("condition_id")
            ye = e["outcome"].tolist()
            brier_ens = _brier(e["forecast_prob"].tolist(), ye)
            print(f"    {'ENSEMBLE':<10} {brier_ens:>8.4f} "
                  f"{_logloss(e['forecast_prob'].tolist(), ye):>10.4f}   "
                  f"(vs MODEL {_brier(m['forecast_prob'].tolist(), m['outcome'].tolist()):.4f} "
                  f"on the same {len(common)} markets)")

    # Per-city Brier (MODEL vs MARKET) — a pooled number can hide a city that's dragging us down.
    print("\n  Per-city Brier (MODEL vs MARKET, lower = better):")
    print(f"    {'city':<14} {'n':>4} {'model':>8} {'market':>8}")
    for city, grp in ml.groupby("city"):
        yc = grp["outcome"].tolist()
        print(f"    {str(city):<14} {len(grp):>4} "
              f"{_brier(grp['forecast_prob'].tolist(), yc):>8.4f} "
              f"{_brier(grp[mkt_col].tolist(), yc):>8.4f}")

    # Distribution-level CRPS (MODEL vs ENSEMBLE) — scores the whole temperature forecast, so it
    # rewards correct *calibration*, not just per-bin direction. Lower = better.
    crps_m = _mean_crps(_OUT / "opportunities_evaluation_calibrated.csv")
    crps_e = _mean_crps(_OUT / "opportunities_evaluation_ensemble.csv")
    if crps_m or crps_e:
        print("\n  Temperature CRPS (lower = better):")
        if crps_m:
            print(f"    {'MODEL':<10} {crps_m[0]:>8.4f}   (over {crps_m[1]} city-dates)")
        if crps_e:
            print(f"    {'ENSEMBLE':<10} {crps_e[0]:>8.4f}   (over {crps_e[1]} city-dates)")

    # Shrink-to-market recommendation: the w that would have minimized Brier on this graded set.
    sw = _best_shrink_weight(ml)
    if sw is not None:
        best_w, best_b, b_w1, b_w0 = sw
        print("\n  Shrink-to-market sweep (our_prob = w·model + (1-w)·market):")
        print(f"    Brier-minimizing w = {best_w:.2f}  (Brier {best_b:.4f})   "
              f"[w=1 pure model {b_w1:.4f}, w=0 pure market {b_w0:.4f}]")
        if best_w < 1.0:
            print(f"    → the model is best trusted only partially; set config.SHRINK_WEIGHT≈{best_w:.2f} "
                  "once the sample gate is met.")

    print("\n  Model calibration (predicted P(YES) vs realized frequency):")
    print(f"    {'bin':<12} {'n':>4} {'pred':>7} {'realized':>9}")
    for lo, hi, n_, pred, real in _calibration(p_model, y):
        print(f"    [{lo:.1f},{hi:.1f})    {n_:>4} {pred:>7.3f} {real:>9.3f}")

    roi = _roi_at_production(ml)
    wr = roi["wins"] / roi["bets"] if roi["bets"] else 0.0
    print(f"\n  ROI at production params: {roi['roi']:.1%}  on {roi['bets']} bets  "
          f"(win rate {wr:.1%}, staked ${roi['staked']:.0f})")

    # ---- Single arbiter: does the model out-predict the market AND at least match the ensemble? --
    beats_market = brier_model < brier_mkt
    beats_ens = (brier_ens is None) or (brier_model <= brier_ens + 1e-9)
    print("\n  EDGE CHECK (the only thing that matters):")
    print(f"    model Brier {brier_model:.4f}  {'<' if beats_market else '>='}  "
          f"market Brier {brier_mkt:.4f}   → "
          f"{'PASS' if beats_market else 'FAIL — no forecasting edge over the market'}")
    if brier_ens is not None:
        print(f"    model Brier {brier_model:.4f}  {'<=' if beats_ens else '>'}  "
              f"ensemble Brier {brier_ens:.4f}   → "
              f"{'PASS' if beats_ens else 'FAIL — the plain ensemble is better; the calibrator is hurting'}")
    print(f"    OVERALL: {'✅ PASS' if (beats_market and beats_ens) else '❌ FAIL'} "
          "(a PASS is necessary, not sufficient — the sample gate below must also be met)")

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
