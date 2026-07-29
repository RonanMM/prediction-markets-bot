"""Bet selection (megaplan Phase A) — is there a SUBSET of the model's opportunities where it
beats the market?

The model loses to the market on average (pooled Brier gap +0.0211, CI entirely above zero), but
average accuracy is a different question from being right on the bets we choose to place. This
module searches for a profitable subset under a protocol that cannot fool itself:

  * the split is FROZEN in code, so 'held-out' is the same set on every run
  * discovery runs on TRAIN only, and the search path never receives held-out rows
  * exactly ONE (selector, threshold) pair is validated, ONCE, at z=1.96
  * every held-out evaluation is appended to an auditable log

Validation is on the paired Brier gap, not ROI. ROI's held-out interval is ~46 percentage points
wide at this sample size and cannot distinguish +3% from -20%; the Brier gap resolves 0.017-0.033.
See docs/superpowers/specs/2026-07-29-bet-selection-design.md.
"""
from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

import config
import stats_util

# Frozen. Deriving this from the data (a 2/3 quantile, say) would move the boundary every time
# new markets grade, silently redefining the held-out set between runs.
SPLIT_DATE = "2026-07-08"

MKT_COL_CANDIDATES = ("market_prob_raw", "market_prob")


def split_frozen(df: pd.DataFrame, split_date: str = SPLIT_DATE):
    """Chronological split at the frozen date. Returns (train, holdout).

    Partitioned on `target_date`, which IS the day component of the city-day cluster key, so no
    city-day can straddle the boundary by construction.
    """
    d = df.copy()
    d["_td"] = pd.to_datetime(d["target_date"], errors="coerce")
    d = d.dropna(subset=["_td"])
    cut = pd.Timestamp(split_date)
    train = d[d["_td"] < cut].drop(columns=["_td"]).reset_index(drop=True)
    holdout = d[d["_td"] >= cut].drop(columns=["_td"]).reset_index(drop=True)
    return train, holdout


def market_col(df: pd.DataFrame) -> str:
    """The RAW tradeable price when present. `market_prob` is normalised so bins sum to 1, which
    flatters the market's Brier — grading against it would understate our own deficit."""
    return MKT_COL_CANDIDATES[0] if MKT_COL_CANDIDATES[0] in df.columns else MKT_COL_CANDIDATES[1]


def flat_roi(sel: pd.DataFrame) -> float:
    """Equal-stake ROI with the same execution costs as evaluate_oos._roi_at_production: cross
    half the spread on entry, pay the taker fee on a winning payout.

    Flat rather than Kelly deliberately. The held-out third's apparent +3.3% Kelly ROI reverses
    to -4.5% at equal stakes — that number was sizing luck concentrated into a few bets, not
    selection skill. Reported only; the gate is the Brier gap.
    """
    if len(sel) == 0:
        return float("nan")
    their = sel["their_prob"].astype(float).clip(1e-6, 1 - 1e-6)
    eff = (their + config.HALF_SPREAD).clip(upper=1 - 1e-6)
    won = (((sel["bet_side"] == "Yes") & (sel["outcome"].astype(int) == 1)) |
           ((sel["bet_side"] == "No") & (sel["outcome"].astype(int) == 0)))
    pnl = np.where(won, (1.0 - config.FEE_RATE) / eff - 1.0, -1.0)
    return float(np.mean(pnl))


# Pre-registered before any searching. Each entry maps a family name to a pure predicate and a
# fixed threshold grid. Adding a family after seeing results is how a search launders noise into
# a finding — if one has to be added, say so in the log and treat the run as exploratory.
SELECTORS: dict = {
    # Theory-driven, not dredged: the model's [0,0.1) confidence bin predicts 3.6% and realizes
    # 15.5%. Excluding its overconfident tail fixes a known, measured defect.
    "forecast_prob_floor": (
        lambda d, t: d["forecast_prob"].astype(float) >= t, [0.10, 0.15, 0.20]),
    # The market's cheap bins are honestly cheap — 0 of 64 markets priced under 10c landed — so
    # betting No into them may be systematically wrong.
    "bet_side": (
        lambda d, t: d["bet_side"].astype(str) == t, ["Yes", "No"]),
    "forecast_sigma_max": (
        lambda d, t: d["forecast_sigma"].astype(float) <= t, [1.2, 1.6, 2.0]),
    # The structure book already found thin books lose -0.064/contract as maker.
    "liquidity_min": (
        lambda d, t: d["liquidity"].astype(float) >= t, [1500, 2500, 4000]),
    "pmf_sum_dev_max": (
        lambda d, t: d["pmf_sum_dev"].astype(float) <= t, [0.3, 0.6, 0.9]),
    "volume_recency_min": (
        lambda d, t: d["volume_recency"].astype(float) >= t, [0.5, 0.8, 0.95]),
    "bucket": (
        lambda d, t: d["bucket"].astype(str) == t,
        ["Chicago|1d", "Chicago|2d+", "Chicago|same-day",
         "HongKong|1d", "HongKong|2d+", "HongKong|same-day",
         "London|1d", "London|2d+", "London|same-day",
         "NYC|1d", "NYC|2d+", "NYC|same-day",
         "Seoul|1d", "Seoul|2d+", "Seoul|same-day"]),
}

# Kept as data, not a comment, so the exclusion is testable and survives refactoring.
EXCLUDED_BY_DESIGN: dict = {
    "abs_edge": ("Adverse selection, z-std 1.41 (EDGE_MEGAPLAN §63): the model is most wrong "
                 "exactly where it disagrees most with the price, so selecting on edge size is "
                 "the measured trap."),
    "is_stale": "Only 22 of 201 training bets — far too thin to resolve anything.",
    "intraday": ("Only 12 of 201 training bets carry intraday conditioning. Worth recording "
                 "that the model's one genuine informational edge over the market fires this "
                 "rarely in the backtest."),
}


def iter_candidates() -> list:
    """Every (family, threshold) pair, in a deterministic order."""
    return [(name, t) for name, (_, thresholds) in SELECTORS.items() for t in thresholds]


def evaluate_selector(df: pd.DataFrame, mask) -> dict | None:
    """Paired model-minus-market Brier gap for the selected rows, clustered by city-day.

    Negative gap = model beats the market on this subset. Returns None when the selection is
    empty, so a threshold that keeps nothing cannot report a spurious result.
    """
    sel = df[mask]
    if len(sel) == 0:
        return None
    mkt = market_col(sel)
    y = sel["outcome"].to_numpy(dtype=float)
    d = (sel["forecast_prob"].to_numpy(dtype=float) - y) ** 2 - \
        (sel[mkt].to_numpy(dtype=float) - y) ** 2
    iv = stats_util.interval(d, stats_util.cluster_key(sel))
    return {
        "n": int(iv["n"]),
        "clusters": int(iv["n_clusters"]),
        "gap": float(iv["mean"]),
        "se": float(iv["se"]),
        "ci_lo": float(iv["ci_lo"]),
        "ci_hi": float(iv["ci_hi"]),
        "mde": float(stats_util.Z * iv["se"]),
        "kept": len(sel) / len(df),
        "roi_flat": flat_roi(sel) if "their_prob" in sel.columns else float("nan"),
    }


def search_train(train: pd.DataFrame) -> list:
    """Score every pre-registered (selector, threshold) pair on TRAIN.

    Deliberately unconstrained — search train as hard as you like, because the multiplicity
    control is the held-out set, not a correction factor applied here. Every candidate is
    returned, including the losers, so the written record shows how wide the net was.

    A family whose column is absent from the frame is skipped rather than raising, so an older
    tracker missing one signal does not block the whole search.
    """
    out = []
    for name, threshold in iter_candidates():
        pred, _ = SELECTORS[name]
        try:
            mask = pred(train, threshold)
        except KeyError:
            continue       # signal not present in this tracker
        r = evaluate_selector(train, mask)
        if r is None:
            continue       # threshold kept nothing
        r["selector"], r["threshold"] = name, threshold
        out.append(r)
    return sorted(out, key=lambda r: r["gap"])


HOLDOUT_LOG = Path(__file__).resolve().parent / "output" / "holdout_log.jsonl"


def interval_clears_zero(gap: float, se: float) -> bool:
    """The pass rule: the ENTIRE 95% interval must lie below zero.

    A negative point estimate whose interval spans zero is not a result. Extracted from
    validate_holdout so the rule can be tested against synthetic (gap, se) pairs — derived
    from data it was only ever exercised at one point, where every loosening of the rule
    happened to give the same answer.
    """
    if se == float("inf") or se != se:      # inf or nan
        return False
    return bool(gap + stats_util.Z * se < 0)


def _json_safe(d: dict) -> dict:
    """NaN/Infinity are not valid JSON. json.dumps emits them bare, so a record from an empty
    selection is readable by Python but rejected by jq and every non-Python reader — and this
    log exists precisely to be audited later, possibly not from Python."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            out[k] = None
        else:
            out[k] = v
    return out


def train_clears_bar(train_result: dict, holdout_gap_se: float) -> bool:
    """Is the train effect big enough that held-out could actually see it?

    Held-out is the one clean measurement available. Spending it on an effect smaller than its
    own minimum detectable size guarantees an uninformative answer while permanently using up
    the test. The bar is the held-out MDE at z=1.96.
    """
    return bool(train_result["gap"] < -(stats_util.Z * float(holdout_gap_se)))


def validate_holdout(train: pd.DataFrame, holdout: pd.DataFrame, selector: str, threshold,
                     data_cutoff: str, log_path=None) -> dict:
    """THE ONE SHOT. Score a single (selector, threshold) pair on held-out and record it forever.

    `data_cutoff` is stamped because held-out grows as markets grade. Markets settling after this
    date are FORWARD sample for stage 2, not more held-out — without the stamp, 'one shot'
    silently becomes 'one shot per week'.

    Passing here is not permission to trade. A pass produces a pre-registered candidate that must
    then clear its own forward gate.
    """
    pred, _ = SELECTORS[selector]
    r = evaluate_selector(holdout, pred(holdout, threshold))
    if r is None:
        r = {"n": 0, "clusters": 0, "gap": float("nan"), "se": float("inf"),
             "ci_lo": float("nan"), "ci_hi": float("nan"), "mde": float("nan"),
             "kept": 0.0, "roi_flat": float("nan")}
    r["selector"], r["threshold"] = selector, threshold
    # The train gap for the SAME rule goes in the record. A held-out result read months later is
    # hard to judge without knowing what it was predicted to be — a rule that showed -0.30 on
    # train and -0.02 on held-out tells a very different story from one that showed -0.03 twice.
    tr = evaluate_selector(train, pred(train, threshold))
    r["train_gap"] = float(tr["gap"]) if tr else float("nan")
    r["train_n"] = int(tr["n"]) if tr else 0
    r["passed"] = interval_clears_zero(r["gap"], r["se"])
    r["data_cutoff"] = data_cutoff
    r["logged_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    path = Path(log_path) if log_path is not None else HOLDOUT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:      # append-only, never truncate
        fh.write(json.dumps(_json_safe(r)) + "\n")
    return r
