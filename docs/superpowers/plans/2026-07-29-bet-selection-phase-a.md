# Bet Selection (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find whether a pre-registered subset of the model's opportunities beats the market on a held-out set, validated on the paired Brier gap rather than on ROI.

**Architecture:** One new module `src/polymarket_weather/bet_selection.py`. It loads the calibrated eval tracker, applies a frozen chronological split, searches selector rules on the training half only, and offers a single one-shot validation entry point against the held-out half. Leakage prevention is structural: the `--search` code path never receives held-out rows. Every held-out evaluation is appended to `output/holdout_log.jsonl`, which is append-only and auditable.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. Reuses `stats_util` (clustered inference), `evaluate_oos._graded_markets` (read-time grading), and `config` (execution costs).

## Global Constraints

- **Split date is frozen at `2026-07-08`.** Never recompute it from data quantiles at runtime.
- **Clustering is always by city-day** via `stats_util.cluster_key`. Never treat bins as independent.
- **The market benchmark column is `market_prob_raw`** (the tradeable price), falling back to `market_prob` only if absent. Never use the normalised column when raw exists.
- **ROI is reported, never gating.** The gate is the paired Brier gap.
- **Exactly one `(selector, threshold)` pair may be validated** — not one family with its threshold free.
- **Phase C is out of scope.** Do not add pre-filter regeneration to this plan.
- Modules run from `src/polymarket_weather/`; tests run from the repo root with `pytest -o addopts="" tests/ -v`.
- Tests are appended to the existing `tests/test_polymarket_weather.py` (this repo keeps one test file).

---

### Task 1: Frozen chronological split

**Files:**
- Create: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SPLIT_DATE: str`, `split_frozen(df: pd.DataFrame, split_date: str = SPLIT_DATE) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(train, holdout)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def _selection_frame():
    """Minimal frame shaped like the calibrated eval tracker."""
    import pandas as pd
    return pd.DataFrame({
        "condition_id": ["a", "b", "c", "d"],
        "city": ["Seoul", "Seoul", "London", "London"],
        "target_date": ["2026-07-07", "2026-07-08", "2026-07-07", "2026-07-09"],
        "outcome": [0, 1, 0, 1],
        "forecast_prob": [0.2, 0.8, 0.3, 0.7],
        "market_prob_raw": [0.3, 0.7, 0.2, 0.8],
    })


def test_split_frozen_is_chronological_and_deterministic():
    """The split date is frozen in code, not derived from the data. Deriving it (e.g. a 2/3
    quantile) would move the boundary every time new markets grade, so 'held-out' would quietly
    become a different set on each run."""
    import bet_selection as bs
    df = _selection_frame()
    train, holdout = bs.split_frozen(df)
    assert bs.SPLIT_DATE == "2026-07-08"
    assert sorted(train["condition_id"]) == ["a", "c"]      # strictly before the cut
    assert sorted(holdout["condition_id"]) == ["b", "d"]    # on or after
    # deterministic: same input, same output, no RNG
    again = bs.split_frozen(df)
    assert list(again[0]["condition_id"]) == list(train["condition_id"])


def test_split_frozen_never_leaks_or_straddles():
    """A condition_id in both halves would make the held-out test meaningless, and a city-day
    spanning the boundary would put correlated bins (one weather outcome) on both sides."""
    import bet_selection as bs
    train, holdout = bs.split_frozen(_selection_frame())
    assert set(train["condition_id"]).isdisjoint(set(holdout["condition_id"]))
    tr_days = set(train["city"] + "|" + train["target_date"])
    ho_days = set(holdout["city"] + "|" + holdout["target_date"])
    assert tr_days.isdisjoint(ho_days)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "split_frozen" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bet_selection'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/polymarket_weather/bet_selection.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "split_frozen" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: frozen chronological split (Phase A task 1)"
```

---

### Task 2: Selector evaluation on the paired Brier gap

**Files:**
- Modify: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `split_frozen` from Task 1.
- Produces: `MKT_COL_CANDIDATES: tuple[str, str]`, `market_col(df) -> str`, `flat_roi(sel) -> float`, `evaluate_selector(df: pd.DataFrame, mask: pd.Series) -> dict | None`. The returned dict has keys `n, clusters, gap, se, ci_lo, ci_hi, mde, kept, roi_flat`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def _scored_frame():
    """Frame with two city-days per city so clustering has something to collapse."""
    import pandas as pd
    return pd.DataFrame({
        "condition_id": list("abcdef"),
        "city": ["Seoul"] * 3 + ["London"] * 3,
        "target_date": ["2026-07-01", "2026-07-01", "2026-07-02",
                        "2026-07-01", "2026-07-02", "2026-07-02"],
        "outcome": [1, 0, 1, 0, 1, 0],
        # model is PERFECT on the first three, terrible on the last three
        "forecast_prob": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "market_prob_raw": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "market_prob": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # must be IGNORED in favour of raw
        "their_prob": [0.5] * 6,
        "bet_side": ["Yes", "No", "Yes", "Yes", "No", "Yes"],
    })


def test_evaluate_selector_scores_the_paired_gap_on_the_raw_price():
    """The benchmark is the RAW tradeable price. market_prob is normalised so bins sum to 1,
    which flatters the market's Brier and would understate our own gap."""
    import bet_selection as bs
    df = _scored_frame()
    mask = df["city"] == "Seoul"          # the subset where the model is perfect
    r = bs.evaluate_selector(df, mask)
    assert r["n"] == 3
    assert r["clusters"] == 2             # Seoul 07-01 and Seoul 07-02
    assert r["kept"] == pytest.approx(0.5)
    # model Brier 0, market Brier 0.25 -> gap = -0.25 (negative = model better)
    assert r["gap"] == pytest.approx(-0.25, abs=1e-9)


def test_evaluate_selector_clusters_by_city_day():
    """Bins settling on one city-day share a single weather outcome. Counting them as
    independent shrinks the SE and manufactures significance."""
    import bet_selection as bs
    df = _scored_frame()
    r = bs.evaluate_selector(df, df["condition_id"].notna())
    assert r["n"] == 6 and r["clusters"] == 4      # 2 cities x 2 days, not 6
    assert r["mde"] > 0


def test_evaluate_selector_returns_none_on_empty_selection():
    """A threshold that keeps nothing must not raise or report a spurious gap."""
    import bet_selection as bs
    df = _scored_frame()
    assert bs.evaluate_selector(df, df["city"] == "Nowhere") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "evaluate_selector" -v`
Expected: FAIL with `AttributeError: module 'bet_selection' has no attribute 'evaluate_selector'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/bet_selection.py` (imports go at the top of the file):

```python
import numpy as np

import config
import stats_util

MKT_COL_CANDIDATES = ("market_prob_raw", "market_prob")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "evaluate_selector" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: paired Brier gap scoring, clustered by city-day (Phase A task 2)"
```

---

### Task 3: Pre-registered selector registry

**Files:**
- Modify: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: nothing (pure predicates).
- Produces: `SELECTORS: dict[str, tuple[callable, list]]` mapping a family name to `(predicate(df, threshold) -> boolean Series, [thresholds])`; `EXCLUDED_BY_DESIGN: dict[str, str]`; `iter_candidates() -> list[tuple[str, object]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_selector_registry_is_pre_registered_and_pure():
    """Every family is a pure predicate over the frame, so a rule cannot silently depend on
    global state or on which rows were evaluated before it."""
    import bet_selection as bs
    df = _scored_frame()
    df["forecast_sigma"] = [1.0, 1.5, 2.5, 1.0, 1.5, 2.5]
    df["liquidity"] = [1200, 3000, 5000, 1200, 3000, 5000]
    df["pmf_sum_dev"] = [0.1, 0.5, 0.95, 0.1, 0.5, 0.95]
    df["volume_recency"] = [0.4, 0.85, 0.99, 0.4, 0.85, 0.99]
    df["bucket"] = ["Seoul|1d"] * 3 + ["London|1d"] * 3
    for name, (pred, thresholds) in bs.SELECTORS.items():
        assert thresholds, f"{name} has no thresholds"
        for t in thresholds:
            m = pred(df, t)
            assert len(m) == len(df), f"{name}@{t} returned the wrong length"
            assert m.dtype == bool, f"{name}@{t} did not return a boolean mask"


def test_edge_magnitude_is_excluded_by_design():
    """Adverse selection at z-std 1.41 (EDGE_MEGAPLAN §63): the model is most wrong exactly
    where it disagrees most with the price. Selecting on edge size is the measured trap, so it
    must not be reachable through the registry."""
    import bet_selection as bs
    assert "abs_edge" not in bs.SELECTORS
    assert "edge" not in bs.SELECTORS
    assert "abs_edge" in bs.EXCLUDED_BY_DESIGN


def test_candidate_count_is_the_pre_registered_number():
    """The count is logged with every search so the record shows how wide the net was, even
    though multiplicity is controlled by the held-out set rather than by a correction."""
    import bet_selection as bs
    cands = bs.iter_candidates()
    assert len(cands) == sum(len(t) for _, t in bs.SELECTORS.values())
    assert all(isinstance(name, str) for name, _ in cands)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "selector_registry or excluded_by_design or candidate_count" -v`
Expected: FAIL with `AttributeError: module 'bet_selection' has no attribute 'SELECTORS'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/bet_selection.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "selector_registry or excluded_by_design or candidate_count" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: pre-registered selector registry, edge magnitude excluded (Phase A task 3)"
```

---

### Task 4: Train-only search

**Files:**
- Modify: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `evaluate_selector` (Task 2), `SELECTORS` / `iter_candidates` (Task 3).
- Produces: `search_train(train: pd.DataFrame) -> list[dict]`, sorted by `gap` ascending. Each dict is an `evaluate_selector` result plus `selector: str` and `threshold`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def _searchable_frame():
    import pandas as pd
    df = pd.concat([_scored_frame()] * 3, ignore_index=True)
    df["condition_id"] = [f"c{i}" for i in range(len(df))]
    df["target_date"] = ["2026-07-01", "2026-07-02", "2026-07-03"] * 6
    df["forecast_sigma"] = 1.5
    df["liquidity"] = 3000
    df["pmf_sum_dev"] = 0.5
    df["volume_recency"] = 0.9
    df["bucket"] = "Seoul|1d"
    return df


def test_search_train_ranks_every_candidate_and_keeps_the_losers():
    """The record has to show how wide the net was. Reporting only the winner is how a search of
    32 rules gets written up as if one rule had been tried."""
    import bet_selection as bs
    res = bs.search_train(_searchable_frame())
    assert len(res) >= 1
    gaps = [r["gap"] for r in res]
    assert gaps == sorted(gaps), "results must be ranked by gap ascending (most negative first)"
    assert all("selector" in r and "threshold" in r for r in res)
    # empty selections are dropped, not reported as gap=0
    assert all(r["n"] > 0 for r in res)


def test_search_train_is_pure_and_does_not_mutate_its_input():
    """A search that mutates the frame makes results depend on evaluation order."""
    import bet_selection as bs
    df = _searchable_frame()
    before = df.copy(deep=True)
    bs.search_train(df)
    pd_testing = __import__("pandas").testing
    pd_testing.assert_frame_equal(df, before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "search_train" -v`
Expected: FAIL with `AttributeError: module 'bet_selection' has no attribute 'search_train'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/bet_selection.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "search_train" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: train-only search over all candidates (Phase A task 4)"
```

---

### Task 5: One-shot held-out validation with an append-only log

**Files:**
- Modify: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `evaluate_selector` (Task 2), `SELECTORS` (Task 3), `search_train` (Task 4).
- Produces: `HOLDOUT_LOG: pathlib.Path`, `train_clears_bar(train_result: dict, holdout_gap_se: float) -> bool`, `validate_holdout(train, holdout, selector: str, threshold, data_cutoff: str, log_path=None) -> dict`. The returned dict has the `evaluate_selector` keys plus `selector`, `threshold`, `passed: bool`, `data_cutoff`, `logged_at`, `train_gap`, `train_n`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_polymarket_weather.py`:

```python
def test_validate_holdout_refuses_when_train_cannot_clear_the_bar():
    """The decision rule that matters most. A train gap of -0.01 cannot be confirmed by a
    held-out set that can only resolve -0.026 — running it would burn the one clean measurement
    we have to learn nothing. Held-out is spent only when train shows something detectable."""
    import bet_selection as bs
    df = _searchable_frame()
    train, holdout = df.iloc[:9].copy(), df.iloc[9:].copy()
    # a train result far too small to be visible on held-out
    weak = {"gap": -0.001, "n": 9, "clusters": 3, "se": 0.02}
    assert bs.train_clears_bar(weak, holdout_gap_se=0.013) is False
    strong = {"gap": -0.30, "n": 9, "clusters": 3, "se": 0.02}
    assert bs.train_clears_bar(strong, holdout_gap_se=0.013) is True


def test_validate_holdout_appends_exactly_one_record_per_call(tmp_path):
    """A code lock can be commented out; a record cannot be un-written. If this file ends up
    with twelve entries, any reader knows the p-value is fiction."""
    import json
    import bet_selection as bs
    df = _searchable_frame()
    train, holdout = df.iloc[:9].copy(), df.iloc[9:].copy()
    log = tmp_path / "holdout_log.jsonl"
    for _ in range(3):
        bs.validate_holdout(train, holdout, "bucket", "Seoul|1d",
                            data_cutoff="2026-07-29", log_path=log)
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 3, "every held-out evaluation must leave a permanent trace"
    assert {"selector", "threshold", "gap", "passed", "data_cutoff", "logged_at",
            "train_gap", "train_n"} <= set(lines[0])


def test_validate_holdout_passes_only_when_the_interval_clears_zero(tmp_path):
    """gap + 1.96*se < 0. A negative point estimate whose interval spans zero is not a result —
    that is exactly how the full-band structure gate read 'MET' while underpowered."""
    import bet_selection as bs
    df = _searchable_frame()
    train, holdout = df.iloc[:9].copy(), df.iloc[9:].copy()
    r = bs.validate_holdout(train, holdout, "bucket", "Seoul|1d",
                            data_cutoff="2026-07-29", log_path=tmp_path / "l.jsonl")
    assert r["passed"] == bool(r["gap"] + 1.96 * r["se"] < 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "validate_holdout or clears_the_bar" -v`
Expected: FAIL with `AttributeError: module 'bet_selection' has no attribute 'train_clears_bar'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/bet_selection.py` (add `import json`, `import datetime as _dt`, `from pathlib import Path` at the top):

```python
HOLDOUT_LOG = Path(__file__).resolve().parent / "output" / "holdout_log.jsonl"


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
    r["passed"] = bool(r["se"] != float("inf") and r["gap"] + stats_util.Z * r["se"] < 0)
    r["data_cutoff"] = data_cutoff
    r["logged_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    path = Path(log_path) if log_path is not None else HOLDOUT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:      # append-only, never truncate
        fh.write(json.dumps(r) + "\n")
    return r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "validate_holdout or clears_the_bar" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: one-shot held-out validation with append-only log (Phase A task 5)"
```

---

### Task 6: CLI with a structural leakage guard

**Files:**
- Modify: `src/polymarket_weather/bet_selection.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `load_bets(csv_path=None) -> pd.DataFrame`, `cmd_search(csv_path=None) -> list[dict]`, `cmd_validate(selector, threshold, data_cutoff, csv_path=None) -> dict`, and a `main()` guarded by `if __name__ == "__main__":`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_polymarket_weather.py`:

```python
def test_search_path_never_reads_holdout_rows(tmp_path):
    """The leakage guard is structural, not disciplinary. If cmd_search ever passed the full
    frame to search_train, poisoning the held-out rows would change its output. It must not.

    Written as a behavioural test rather than a code-shape assertion, because the failure this
    prevents is a one-word edit (train -> df) that no signature check would catch.
    """
    import bet_selection as bs
    import pandas as pd

    base = _searchable_frame()
    base["target_date"] = ["2026-07-01", "2026-07-02", "2026-07-03"] * 6   # all pre-split
    poisoned = base.copy()
    poisoned["target_date"] = ["2026-07-20"] * len(poisoned)               # all post-split
    poisoned["forecast_prob"] = 0.0                                        # absurdly wrong
    poisoned["outcome"] = 1
    poisoned["condition_id"] = [f"p{i}" for i in range(len(poisoned))]

    clean_csv = tmp_path / "clean.csv"
    mixed_csv = tmp_path / "mixed.csv"
    base.to_csv(clean_csv, index=False)
    pd.concat([base, poisoned], ignore_index=True).to_csv(mixed_csv, index=False)

    a = bs.cmd_search(csv_path=clean_csv)
    b = bs.cmd_search(csv_path=mixed_csv)
    assert [(r["selector"], r["threshold"], round(r["gap"], 9)) for r in a] == \
           [(r["selector"], r["threshold"], round(r["gap"], 9)) for r in b], \
        "search results changed when held-out rows were added — the search read held-out data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "never_reads_holdout" -v`
Expected: FAIL with `AttributeError: module 'bet_selection' has no attribute 'cmd_search'`

- [ ] **Step 3: Write the implementation**

Add to `src/polymarket_weather/bet_selection.py`:

```python
import argparse

import evaluate_oos

DEFAULT_TRACKER = evaluate_oos._OUT / "opportunities_evaluation_calibrated.csv"


def load_bets(csv_path=None) -> pd.DataFrame:
    """Graded markets from the calibrated tracker.

    Grading is applied at READ time, so refresh station truth and hourly obs first
    (`fetch_historical_truth.py`, then `fetch_station_obs.py` FULL) or wu_truth silently falls
    back to the pre-W0 ruler and every number below is graded against the wrong source.

    A frame that ALREADY carries `outcome` is taken as-is. The real trackers carry no grade
    column — grading happens at read time (CLAUDE.md) — so that branch is unreachable for
    production data and exists to let the CLI be tested without standing up a full station-truth
    fixture.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_TRACKER
    if not Path(path).exists():
        raise SystemExit(f"no tracker at {path}")
    raw = pd.read_csv(path)
    df = raw if "outcome" in raw.columns else evaluate_oos._graded_markets(Path(path))
    if df is None or df.empty:
        raise SystemExit(f"no gradable markets in {path} — refresh truth first")
    return df


def cmd_search(csv_path=None) -> list:
    """Discovery. Loads, splits, and passes ONLY train onward.

    `holdout` is deliberately discarded on the next line rather than being kept in scope: this
    function must not be able to see it even by accident.
    """
    train, _holdout = split_frozen(load_bets(csv_path))
    del _holdout
    return search_train(train)


def cmd_validate(selector: str, threshold, data_cutoff: str, csv_path=None) -> dict:
    """The one shot. Refuses unless the same rule already clears the bar on train."""
    train, holdout = split_frozen(load_bets(csv_path))
    pred, _ = SELECTORS[selector]
    tr = evaluate_selector(train, pred(train, threshold))
    if tr is None:
        raise SystemExit(f"{selector}@{threshold} selects nothing on train")
    ho_probe = evaluate_selector(holdout, pred(holdout, threshold))
    if ho_probe is None:
        raise SystemExit(f"{selector}@{threshold} selects nothing on held-out")
    if not train_clears_bar(tr, ho_probe["se"]):
        raise SystemExit(
            f"REFUSED: train gap {tr['gap']:+.4f} is smaller than the held-out MDE "
            f"{stats_util.Z * ho_probe['se']:.4f}. Held-out cannot resolve an effect this size, "
            f"so spending the one shot here would use up the test and learn nothing. "
            f"Go to Phase C (see the spec).")
    return validate_holdout(train, holdout, selector, threshold, data_cutoff)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bet selection (Phase A). See "
                                             "docs/superpowers/specs/2026-07-29-bet-selection-design.md")
    ap.add_argument("--search", action="store_true",
                    help="rank every pre-registered candidate on TRAIN only")
    ap.add_argument("--validate", nargs=2, metavar=("SELECTOR", "THRESHOLD"),
                    help="THE ONE SHOT: score one pair on held-out and log it forever")
    ap.add_argument("--data-cutoff", default=_dt.date.today().isoformat(),
                    help="stamped into the log; held-out grows as markets grade")
    a = ap.parse_args()

    if a.search:
        rows = cmd_search()
        print(f"{'selector':<22}{'thresh':>12}{'n':>5}{'cd':>5}{'gap':>9}{'mde':>8}"
              f"{'kept':>7}{'roiflat':>9}")
        for r in rows:
            print(f"{r['selector']:<22}{str(r['threshold']):>12}{r['n']:>5}{r['clusters']:>5}"
                  f"{r['gap']:>+9.4f}{r['mde']:>8.4f}{r['kept']:>7.2f}{r['roi_flat']:>+9.3f}")
        print(f"\n{len(rows)} candidates scored on TRAIN. Held-out untouched.")
    elif a.validate:
        sel, raw = a.validate
        _, thresholds = SELECTORS[sel]
        # match the pre-registered threshold's type rather than guessing from the string
        threshold = next((t for t in thresholds if str(t) == raw), None)
        if threshold is None:
            raise SystemExit(f"{raw!r} is not a pre-registered threshold for {sel}: {thresholds}")
        r = cmd_validate(sel, threshold, a.data_cutoff)
        print(f"HELD-OUT  {sel}@{threshold}  n={r['n']} clusters={r['clusters']}")
        print(f"  gap {r['gap']:+.4f}  95% CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
        print(f"  -> {'PASS' if r['passed'] else 'FAIL'}   (logged to {HOLDOUT_LOG})")
        print("  A pass is a pre-registered candidate, NOT permission to trade.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and then the whole suite**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py -k "never_reads_holdout" -v`
Expected: PASS

Run: `pytest -o addopts="" tests/ -q`
Expected: PASS, all tests including the 130 that existed before this plan

- [ ] **Step 5: Run the real search and confirm held-out is untouched**

```bash
cd src/polymarket_weather
python fetch_historical_truth.py
python fetch_station_obs.py
python bet_selection.py --search
```
Expected: a ranked table of ~32 candidates, and no `output/holdout_log.jsonl` created.

- [ ] **Step 6: Commit**

```bash
git add src/polymarket_weather/bet_selection.py tests/test_polymarket_weather.py
git commit -m "bet-selection: CLI with structural leakage guard (Phase A task 6)"
```

---

## After the plan

Running `--search` produces the ranked train table. **Do not run `--validate` reflexively.** Per §7 of the spec, if no candidate's train gap beats the projected held-out MDE, the correct move is to stop and go to Phase C with held-out unspent — `cmd_validate` enforces this and will refuse.

Record the search outcome in `docs/EDGE_MEGAPLAN.md` either way. A clean negative is the deliverable if that is what the data says.
