"""backtest_common.py — the single source of truth for backtest ECONOMICS (megaplan D6).

Every backtest / optimizer / simulator tool must settle bets, size Kelly, and apply exposure
caps IDENTICALLY to the production engine and the evaluate_oos arbiter — otherwise a sweep
tunes parameters under economics production never uses (the "grid-graded artifact" CLAUDE.md
warns about). Four root scripts previously carried forked copies of this logic, several already
diverged (no half-spread, flat 0.98 settlement, hardcoded 0.12 Kelly cap, uncensored priced
distributions). This module centralizes:

  * settle_bet  — honest per-bet PnL: cross config.HALF_SPREAD on entry, pay config.FEE_RATE on
                  the winning payout (mirrors evaluate_oos._roi_at_production).
  * single_kelly — fractional Kelly with fee-adjusted odds + guards, matching engine._kelly_size.
  * apply_caps  — per-(city,date) group and portfolio Kelly caps (config.MAX_KELLY_PER_GROUP /
                  MAX_TOTAL_KELLY), mirroring evaluate_oos._roi_at_production.

It also re-exports pmf._cdf/_bin_prob/_condition_prob so scripts price with the SAME censored
distribution as the engine, never a floor/ceiling-blind fork.
"""
from collections import defaultdict

import numpy as np

import config
from pmf import _cdf, _bin_prob, _condition_prob   # re-exported: price like the engine

__all__ = ["settle_bet", "single_kelly", "apply_caps", "_cdf", "_bin_prob", "_condition_prob"]

_EPS = 1e-9


def settle_bet(their_prob: float, won: bool, size: float) -> float:
    """Honest realized PnL for a $`size` stake at price `their_prob`. Cross half the spread on
    entry (buy fewer shares at their_prob + HALF_SPREAD); a win pays $1/share minus FEE_RATE,
    a loss forfeits the stake."""
    their_eff = min(1.0 - _EPS, float(their_prob) + config.HALF_SPREAD)
    if their_eff <= 0:
        return 0.0
    shares = size / their_eff
    return (shares * (1.0 - config.FEE_RATE) - size) if won else -size


def single_kelly(our_prob: float, their_prob: float,
                 kelly_fraction: float = config.KELLY_FRACTION,
                 fee: float = config.FEE_RATE) -> float:
    """Fractional Kelly with fee-adjusted odds — identical to engine._kelly_size, including the
    p∈(1e-4, 1-1e-4) and b<=0 guards and the MAX_KELLY_PER_BET clip."""
    p = their_prob
    if not (np.isfinite(p) and np.isfinite(our_prob)):
        return 0.0
    if p <= 1e-4 or p >= (1.0 - 1e-4):
        return 0.0
    b = ((1.0 - p) / p) * (1.0 - fee)
    if b <= 0:
        return 0.0
    q = our_prob
    raw = kelly_fraction * (b * q - (1.0 - q)) / b
    return float(np.clip(raw, 0.0, config.MAX_KELLY_PER_BET))


def apply_caps(kellys, group_keys):
    """Apply the per-group then portfolio Kelly caps, mirroring evaluate_oos._roi_at_production.

    kellys, group_keys: equal-length sequences (per-bet kelly, its (city, date) group key).
    Returns a numpy array of capped kellys — each group scaled down to MAX_KELLY_PER_GROUP,
    then the whole book scaled down to MAX_TOTAL_KELLY.
    """
    k = np.asarray(kellys, dtype=float).copy()
    idx = defaultdict(list)
    for i, g in enumerate(group_keys):
        idx[g].append(i)
    for ids in idx.values():
        tot = k[ids].sum()
        if tot > config.MAX_KELLY_PER_GROUP:
            k[ids] *= config.MAX_KELLY_PER_GROUP / tot
    tot = k.sum()
    if tot > config.MAX_TOTAL_KELLY:
        k *= config.MAX_TOTAL_KELLY / tot
    return k
