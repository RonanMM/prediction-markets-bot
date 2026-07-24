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
from pmf import _cdf, _t_scale
from scipy.stats import t as student_t
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


def m1_gate(qrf_brier, ens_brier):
    """M1 (megaplan): does the QRF forecaster beat the raw ensemble on Brier?

    Pure comparison, no I/O — `main()` guards its call on the QRF tracker existing
    (see the M1 GATE print below), so this is trivially safe to call once both
    Briers are in hand. `<=` mirrors the self-gate already applied at serve time
    (`predictors/qrf.py`'s `beats_ensemble`): ties count as a pass.
    """
    return qrf_brier <= ens_brier


def _logloss(p, y):
    s = 0.0
    for pi, yi in zip(p, y):
        pi = min(1 - _EPS, max(_EPS, pi))
        s += -(yi * log(pi) + (1 - yi) * log(1 - pi))
    return s / len(p)


def _crps_student_t(mu, sigma, nu, y, floor=None, ceiling=None):
    """CRPS of a Student-t predictive distribution vs the observed temperature y (°C).

    CRPS = ∫ (F(x) - 1{x>=y})^2 dx, evaluated numerically over a grid wide enough to cover the
    tails. Lower is better. Unlike per-market Brier, this scores the whole *temperature*
    distribution, so it distinguishes a well-calibrated forecast from an overconfident one.
    With a censoring `floor` f (same-day bets: the running observed max), T = max(f, Z), so
    F(x) = 0 below the floor.
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (mu, sigma, nu, y)):
        return None
    if floor is not None and (isinstance(floor, float) and np.isnan(floor)):
        floor = None
    if ceiling is not None and (isinstance(ceiling, float) and np.isnan(ceiling)):
        ceiling = None
    hw = max(12.0, 6.0 * float(sigma))
    lo = float(mu) - hw
    hi = float(mu) + hw
    if floor is not None:
        lo = min(lo, float(floor) - 1.0)
        hi = max(hi, float(floor) + 1.0)
    if ceiling is not None:
        lo = min(lo, float(ceiling) - 1.0)
        hi = max(hi, float(ceiling) + 1.0)
    xs = np.linspace(lo, hi, 241)
    # F13: one vectorized Student-t CDF over the whole grid (was 241 scalar _cdf calls per row).
    # sigma is a standard deviation → convert to the t-scale via _t_scale, exactly as _cdf does;
    # floor/ceiling are then applied to F below (as before), so the result is identical.
    F = student_t.cdf((xs - float(mu)) / _t_scale(float(sigma), float(nu)), df=float(nu))
    if floor is not None:
        F[xs < float(floor)] = 0.0
    if ceiling is not None:
        F[xs >= float(ceiling)] = 1.0
    step = (xs >= float(y)).astype(float)
    integrand = (F - step) ** 2
    # Trapezoid rule, written out to stay compatible across NumPy 1.x/2.x (np.trapz removed in 2.0).
    return float(np.sum((integrand[:-1] + integrand[1:]) / 2.0 * np.diff(xs)))


def _crps_by_key(csv_path):
    """{(city, target_date, kind): temperature-CRPS} for one predictor's tracker, or {}.

    D2: kind ('min'/'max') is derived from the question ('lowest' → min) — the SAME routing the
    engine uses — so a Tmax and a Tmin market on the same city-date are scored as SEPARATE keys,
    and the LAST ROW per key is taken via .tail(1) (the old groupby.last() returned the last
    non-null value PER COLUMN, splicing a Tmax row's floor onto a Tmin row's mu/sigma → a garbage
    point-mass CRPS).
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    need = {"forecast_mu", "forecast_sigma", "forecast_nu", "city", "target_date"}
    if not need.issubset(df.columns):
        return {}
    q = (df["question"].astype(str).str.lower() if "question" in df.columns
         else pd.Series([""] * len(df), index=df.index))
    df["_kind"] = np.where(q.str.contains("lowest"), "min", "max")
    df = df.sort_values("fetched_at").groupby(
        ["city", "target_date", "_kind"], as_index=False).tail(1)
    out = {}
    for _, r in df.iterrows():
        y = fetch_actual_weather(r["city"], r["target_date"], r.get("question", ""))
        if y is None:
            continue
        c = _crps_student_t(r["forecast_mu"], r["forecast_sigma"], r["forecast_nu"], y,
                            floor=r.get("forecast_floor"),
                            ceiling=r.get("forecast_ceiling"))
        if c is not None:
            out[(r["city"], r["target_date"], r["_kind"])] = c
    return out


def _mean_crps(csv_path):
    """Mean temperature-level CRPS over all gradable (city, target_date, kind) keys, or None."""
    by = _crps_by_key(csv_path)
    return (sum(by.values()) / len(by), len(by)) if by else None


def _dispersion_by_segment(csv_path):
    """Spread-skill / PIT calibration of the served predictive SPREAD, by segment.

    `sigma` is the predictive STANDARD DEVIATION (see `_crps_student_t`), so under an honest
    spread the standardized residual z=(truth-mu)/sigma has std ~1.0 and the PIT (computed via
    the SAME `_t_scale` CDF the engine grades with) is ~Uniform. The spread-skill ratio is
    therefore just std(z): **>1 = OVERCONFIDENT (spread too narrow for the realized errors),
    <1 = spread too wide.** Reported overall (Tmax / Tmin) and by target month for Tmax — the
    latter is the point: it catches under-dispersion drifting back in (e.g. spring volatility)
    that the aggregate hides. Intraday-CENSORED distributions (floor/ceiling set) are excluded —
    they are not a plain Student-t, so z is meaningless there.

    Returns {label: (n, ratio_or_None, pit_lo, pit_hi)} (ratio None when too thin to read).
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    need = {"forecast_mu", "forecast_sigma", "forecast_nu", "city", "target_date"}
    if not need.issubset(df.columns):
        return {}
    q = (df["question"].astype(str).str.lower() if "question" in df.columns
         else pd.Series([""] * len(df), index=df.index))
    df["_kind"] = np.where(q.str.contains("lowest"), "min", "max")
    df = df.sort_values("fetched_at").groupby(
        ["city", "target_date", "_kind"], as_index=False).tail(1)

    recs = []
    for _, r in df.iterrows():
        sigma = r["forecast_sigma"]
        if sigma is None or pd.isna(sigma) or float(sigma) <= 0:
            continue
        if pd.notna(r.get("forecast_floor")) or pd.notna(r.get("forecast_ceiling")):
            continue  # intraday-censored: not a plain Student-t
        y = fetch_actual_weather(r["city"], r["target_date"], r.get("question", ""))
        if y is None:
            continue
        mu, sig, nu = float(r["forecast_mu"]), float(sigma), max(float(r["forecast_nu"]), 2.1)
        recs.append((str(r["target_date"])[:7], r["_kind"],
                     (float(y) - mu) / sig,
                     float(student_t.cdf((float(y) - mu) / _t_scale(sig, nu), df=nu))))
    if not recs:
        return {}
    d = pd.DataFrame(recs, columns=["month", "kind", "z", "pit"])

    def agg(sub):
        if len(sub) < 6:
            return (len(sub), None, None, None)
        return (len(sub), float(sub["z"].std()),
                float((sub["pit"] < 0.1).mean()), float((sub["pit"] > 0.9).mean()))

    out = {"Tmax (all)": agg(d[d["kind"] == "max"]),
           "Tmin (all)": agg(d[d["kind"] == "min"])}
    for m, sub in d[d["kind"] == "max"].groupby("month"):
        out[f"Tmax {m}"] = agg(sub)
    return out


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


def _paired_brier_test(p_a, p_b, y, n_boot=20000):
    """Paired test on the per-market Brier difference (A − B). Positive ⇒ A worse.

    Both predictors are scored on the SAME markets, so market-difficulty variance differences
    out — the paired test is far tighter than comparing two pooled Brier means. Returns
    (mean_diff, se, t, ci_lo, ci_hi, frac_A_better).
    """
    d = np.array([(pa - yi) ** 2 - (pb - yi) ** 2 for pa, pb, yi in zip(p_a, p_b, y)], float)
    n = len(d)
    if n < 2:
        return None
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = mean_d / se if se else float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean_d, se, t, float(lo), float(hi), float((boot < 0).mean())


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
    # Portfolio cap = concurrent exposure on any one settlement date. Grouping by
    # target_date (not summing across the whole passed set) is essential: a global sum
    # scales every bet by MAX_TOTAL_KELLY / (sum over ALL bets), so bet sizes shrink as
    # the backtest lengthens and — via the $1 size floor below — silently drops the
    # smallest bets, differently for a 60-bet window vs a 264-bet one, making windows
    # non-comparable. Per-date keeps sizing window-independent (fixes that inconsistency).
    for dt, grp in g.groupby("target_date"):
        tot = grp["k"].sum()
        if tot > config.MAX_TOTAL_KELLY:
            g.loc[grp.index, "k"] = grp["k"] * (config.MAX_TOTAL_KELLY / tot)

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

    # QRF baseline (M1, docs/superpowers/specs/2026-07-24-qrf-forecaster-design.md) — opt-in:
    # opportunities_evaluation_qrf.csv only exists once Task 7 has trained + served QRF at least
    # once, so this whole block is guarded on the file existing. Until then `main()` must behave
    # EXACTLY as before (evaluate_oos runs daily in the truth-eval workflow and on every dashboard
    # build — a crash or spurious output here would break the live pipeline for every other city).
    brier_qrf = None
    if (_OUT / "opportunities_evaluation_qrf.csv").exists():
        qrf = _graded_markets(_OUT / "opportunities_evaluation_qrf.csv")
        if qrf is not None and not qrf.empty and brier_ens is not None:
            common_q = set(ml["condition_id"]) & set(qrf["condition_id"]) & set(ens["condition_id"])
            if common_q:
                q = qrf[qrf["condition_id"].isin(common_q)].set_index("condition_id")
                e2 = ens[ens["condition_id"].isin(common_q)].set_index("condition_id")
                yq = q["outcome"].tolist()
                brier_qrf = _brier(q["forecast_prob"].tolist(), yq)
                brier_ens_paired = _brier(e2["forecast_prob"].tolist(), e2["outcome"].tolist())
                print(f"    {'QRF':<10} {brier_qrf:>8.4f} "
                      f"{_logloss(q['forecast_prob'].tolist(), yq):>10.4f}   "
                      f"(vs ENSEMBLE {brier_ens_paired:.4f} on the same {len(common_q)} markets)")
                print(f"\n  M1 GATE: QRF <= ensemble   "
                      f"QRF {brier_qrf:.4f}  {'<=' if m1_gate(brier_qrf, brier_ens_paired) else '>'}  "
                      f"ENSEMBLE {brier_ens_paired:.4f}   → "
                      f"{'PASS' if m1_gate(brier_qrf, brier_ens_paired) else 'FAIL'}")

    # Per-city Brier (MODEL vs MARKET) — a pooled number can hide a city that's dragging us down.
    print("\n  Per-city Brier (MODEL vs MARKET, lower = better):")
    print(f"    {'city':<14} {'n':>4} {'model':>8} {'market':>8}")
    for city, grp in ml.groupby("city"):
        yc = grp["outcome"].tolist()
        print(f"    {str(city):<14} {len(grp):>4} "
              f"{_brier(grp['forecast_prob'].tolist(), yc):>8.4f} "
              f"{_brier(grp[mkt_col].tolist(), yc):>8.4f}")

    # E3 per-bucket view (docs/EDGE_MEGAPLAN.md §5): buckets are derived from city+days_ahead so
    # this works on trackers written before the bucket column existed. ON buckets are execution-
    # eligible per config.LIVE_BUCKETS (paper until their FORWARD gate passes); every bucket keeps
    # being tracked either way. Forward gate = bets with target_date AFTER the nomination date, so
    # the in-sample selection that nominated the buckets cannot grade itself.
    ml["_bucket"] = [config.bucket_key(c, d) for c, d in zip(ml["city"], ml["days_ahead"])]
    print("\n  E3 buckets (model vs market Brier; ON = execution-eligible, paper until gate):")
    print(f"    {'bucket':<20} {'n':>4} {'model':>8} {'market':>8}  status")
    for bk, grp in sorted(ml.groupby("_bucket"), key=lambda kv: -len(kv[1])):
        yb = grp["outcome"].tolist()
        bm, bmk = _brier(grp["forecast_prob"].tolist(), yb), _brier(grp[mkt_col].tolist(), yb)
        status = "ON (paper)" if bk in config.LIVE_BUCKETS else "off"
        print(f"    {bk:<20} {len(grp):>4} {bm:>8.4f} {bmk:>8.4f}  {status}")
    on = ml[ml["_bucket"].isin(config.LIVE_BUCKETS)]
    if not on.empty:
        yo = on["outcome"].tolist()
        roi_on = _roi_at_production(on)
        print(f"    ON-book: n={len(on)}  model {_brier(on['forecast_prob'].tolist(), yo):.4f} "
              f"vs market {_brier(on[mkt_col].tolist(), yo):.4f}  "
              f"ROI {roi_on['roi']:+.1%} on {roi_on['bets']} bets"
              "   [in-sample until the forward gate passes]")
        print(f"    Forward gate (target_date > {config.E3_NOMINATION_DATE}, "
              f"need ≥{config.E3_FORWARD_MIN_BETS}/bucket AND model ≤ market Brier):")
        for bk in sorted(config.LIVE_BUCKETS):
            fwd = ml[(ml["_bucket"] == bk) & (ml["target_date"] > config.E3_NOMINATION_DATE)]
            if fwd.empty:
                print(f"      {bk:<20}    0/{config.E3_FORWARD_MIN_BETS}  (no forward bets graded yet)")
                continue
            yf = fwd["outcome"].tolist()
            fm, fk = _brier(fwd["forecast_prob"].tolist(), yf), _brier(fwd[mkt_col].tolist(), yf)
            met = len(fwd) >= config.E3_FORWARD_MIN_BETS and fm <= fk
            print(f"      {bk:<20} {len(fwd):>4}/{config.E3_FORWARD_MIN_BETS}  "
                  f"model {fm:.4f} vs market {fk:.4f}  → "
                  f"{'✅ GATE PASSED — smallest real size OK' if met else 'pending'}")

    # Distribution-level CRPS (MODEL vs ENSEMBLE) — scores the whole temperature forecast, so it
    # rewards correct *calibration*, not just per-bin direction. Lower = better.
    crps_m_by = _crps_by_key(_OUT / "opportunities_evaluation_calibrated.csv")
    crps_e_by = _crps_by_key(_OUT / "opportunities_evaluation_ensemble.csv")
    crps_m = (sum(crps_m_by.values()) / len(crps_m_by), len(crps_m_by)) if crps_m_by else None
    crps_e = (sum(crps_e_by.values()) / len(crps_e_by), len(crps_e_by)) if crps_e_by else None
    if crps_m or crps_e:
        print("\n  Temperature CRPS (lower = better):")
        if crps_m:
            print(f"    {'MODEL':<10} {crps_m[0]:>8.4f}   (over {crps_m[1]} city-date-kinds)")
        if crps_e:
            print(f"    {'ENSEMBLE':<10} {crps_e[0]:>8.4f}   (over {crps_e[1]} city-date-kinds)")
        # D3: PAIRED over the identical (city, target_date, kind) support — the only
        # apples-to-apples 'does the calibrator beat the ensemble' number. The unpaired means
        # above average over different days AND different market types (Tmin rows exist only in
        # the calibrated tracker), so they are not directly comparable.
        common = set(crps_m_by) & set(crps_e_by)
        if common:
            pm = sum(crps_m_by[k] for k in common) / len(common)
            pe = sum(crps_e_by[k] for k in common) / len(common)
            print(f"    {'PAIRED':<10} MODEL {pm:>7.4f}  vs  ENSEMBLE {pe:>7.4f}   "
                  f"(n={len(common)}, → {'MODEL' if pm < pe else 'ENSEMBLE'} better)")

    # QRF CRPS (M1) — same opt-in guard as the Brier block above: `_crps_by_key` already
    # returns {} for a missing file, so this whole section is a silent no-op until Task 7
    # trains + serves QRF and the tracker exists.
    if (_OUT / "opportunities_evaluation_qrf.csv").exists():
        crps_q_by = _crps_by_key(_OUT / "opportunities_evaluation_qrf.csv")
        crps_q = (sum(crps_q_by.values()) / len(crps_q_by), len(crps_q_by)) if crps_q_by else None
        if crps_q:
            print(f"    {'QRF':<10} {crps_q[0]:>8.4f}   (over {crps_q[1]} city-date-kinds)")
            common_qc = set(crps_q_by) & set(crps_e_by)
            if common_qc:
                pq = sum(crps_q_by[k] for k in common_qc) / len(common_qc)
                peq = sum(crps_e_by[k] for k in common_qc) / len(common_qc)
                print(f"    {'PAIRED (QRF)':<10} QRF {pq:>7.4f}  vs  ENSEMBLE {peq:>7.4f}   "
                      f"(n={len(common_qc)}, → {'QRF' if pq < peq else 'ENSEMBLE'} better)")

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

    # Dispersion monitor — is the predictive SPREAD honest? std(z) ~1.0 = calibrated; >1.15 =
    # OVERCONFIDENT (tails too thin), <0.85 = too wide. The by-month Tmax rows are the point:
    # they catch under-dispersion returning (e.g. spring) that the pooled number hides. A static
    # sigma widen is NOT the fix — it over-widens the calm months (see docs/tails-fix note).
    disp = _dispersion_by_segment(_OUT / "opportunities_evaluation_calibrated.csv")
    if disp:
        print("\n  Dispersion monitor (served spread vs realized error; ~1.0 = calibrated,")
        print("  >1.15 OVERCONFIDENT / <0.85 too-wide; intraday-censored excluded):")
        print(f"    {'segment':<16} {'n':>4} {'std(z)':>8} {'PIT<.1':>7} {'PIT>.9':>7}  flag")
        for label, (n_, ratio, plo, phi) in disp.items():
            if ratio is None:
                print(f"    {label:<16} {n_:>4} {'(thin)':>8}")
                continue
            flag = "OVERCONFIDENT" if ratio > 1.15 else ("too-wide" if ratio < 0.85 else "ok")
            print(f"    {label:<16} {n_:>4} {ratio:>8.2f} {plo:>6.0%} {phi:>6.0%}  {flag}")
        print("    Read the by-MONTH Tmax rows, not the pooled one: dispersion drifts seasonally")
        print("    (spring volatile → summer calm). A static sigma widen fixes spring but")
        print("    over-widens summer — validated OOS to hurt; see the tails-fix investigation.")

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
    pt = _paired_brier_test(p_model, p_mkt, y)
    if pt is not None:
        md, se, t, lo, hi, frac = pt
        print(f"    paired Δ(model−market) Brier {md:+.4f}  SE {se:.4f}  t {t:+.2f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(model better) {frac:.1%}")
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
