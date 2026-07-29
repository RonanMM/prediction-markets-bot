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

import pandas as pd

# Frozen. Deriving this from the data (a 2/3 quantile, say) would move the boundary every time
# new markets grade, silently redefining the held-out set between runs.
SPLIT_DATE = "2026-07-08"


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
