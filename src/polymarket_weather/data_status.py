"""data_status.py — progress report for the out-of-sample track record (Handoff Step 1).

Prints how much GRADABLE evidence we have accumulated toward a pre-committed sample-size
gate, so we stop drawing ROI conclusions off 4-bet samples. It deliberately reports COUNTS
only — no ROI / win-rate — because the gate must be met BEFORE any performance verdict
(see the pre-committed gate in CLAUDE.md and Handoff Steps 1 & 3).

A "bet" is one flagged opportunity row in the eval tracker
(`output/opportunities_evaluation_calibrated.csv`); every row there already passed MIN_EDGE and has
kelly > 0. A bet is GRADABLE once the resolution-station observation exists for its
target_date (via grading.fetch_actual_weather, i.e. station truth — NOT the forecast grid).

Run from src/polymarket_weather/:   python data_status.py
"""
from datetime import date
from pathlib import Path

import pandas as pd

from grading import fetch_actual_weather, _SLUG, _truth

# ---- Pre-committed sample-size gate (mirrors CLAUDE.md; do NOT move post-hoc) ----------
GATE_RESOLVED_MARKETS = 150   # minimum resolved, station-gradable markets before any verdict
GATE_OOS_BETS = 100           # minimum gradable bets before any ROI/calibration conclusion

_EVAL = Path(__file__).resolve().parent / "output" / "opportunities_evaluation_calibrated.csv"
_TRUTH_DIR = Path(__file__).resolve().parent / "data" / "weather"


def _latest_truth_date(slug):
    """Most recent date_local present in a city's station-truth file, or None."""
    m = _truth(slug, "max")
    return max(m) if m else None


def main():
    if not _EVAL.exists():
        print(f"Eval tracker not found at {_EVAL}\n"
              "Run: python polymarket_weather_analysis.py --data_dir ./data")
        return

    df = pd.read_csv(_EVAL)
    today = date.today()
    df["target"] = pd.to_datetime(df["target_date"]).dt.date

    # Gradability per row (station truth available for that city + date).
    df["gradable"] = [
        fetch_actual_weather(c, str(t), q) is not None
        for c, t, q in zip(df["city"], df["target_date"], df["question"])
    ]
    df["resolved"] = df["target"] < today

    total_bets = len(df)
    total_markets = df["condition_id"].nunique()
    resolved_bets = int(df["resolved"].sum())
    resolved_markets = df.loc[df["resolved"], "condition_id"].nunique()
    gradable_bets = int(df["gradable"].sum())
    gradable_markets = df.loc[df["gradable"], "condition_id"].nunique()

    print("\n================== RAINCHECK DATA STATUS (track-record progress) ==================")
    print(f"Eval tracker: {_EVAL.name}")
    print(f"Date span (target dates): {df['target'].min()} .. {df['target'].max()}")
    print()
    print(f"  Flagged bets (rows)          : {total_bets:>5}   over {total_markets} markets")
    print(f"  Resolved (target date passed): {resolved_bets:>5}   over {resolved_markets} markets")
    print(f"  GRADABLE by station truth    : {gradable_bets:>5}   over {gradable_markets} markets")

    # The binding constraint is usually Meteostat's publishing lag, not market resolution.
    ungradable_resolved = int((df["resolved"] & ~df["gradable"]).sum())
    print(f"  Resolved but NOT yet gradable: {ungradable_resolved:>5}   (awaiting station obs)")
    print()
    print("  Station-truth coverage lag (latest observed date per city vs today):")
    for city in sorted(df["city"].unique()):
        slug = _SLUG.get(city)
        latest = _latest_truth_date(slug) if slug else None
        if latest:
            lag = (today - date.fromisoformat(latest)).days
            print(f"    {city:<12} latest obs {latest}  ({lag}d behind today)")
        else:
            print(f"    {city:<12} no station-truth file")

    # ---- Gate check (counts only — no ROI until the gate is met) -------------------------
    m_ok = gradable_markets >= GATE_RESOLVED_MARKETS
    b_ok = gradable_bets >= GATE_OOS_BETS
    print()
    print("  PRE-COMMITTED GATE (no performance verdict until BOTH are met):")
    print(f"    gradable markets {gradable_markets:>5} / {GATE_RESOLVED_MARKETS}  "
          f"[{'MET' if m_ok else 'not met'}]")
    print(f"    gradable bets    {gradable_bets:>5} / {GATE_OOS_BETS}  "
          f"[{'MET' if b_ok else 'not met'}]")
    if m_ok and b_ok:
        print("\n  ✅ GATE MET — proceed to Step 3 (calibration + OOS ROI vs baselines).")
    else:
        need_m = max(0, GATE_RESOLVED_MARKETS - gradable_markets)
        need_b = max(0, GATE_OOS_BETS - gradable_bets)
        print(f"\n  ⛔ GATE NOT MET — need +{need_m} markets, +{need_b} bets. "
              "Do NOT draw ROI/win-rate conclusions yet.")
        print("     Keep accumulating: collect (main.py) → refresh truth "
              "(fetch_historical_truth.py) → regenerate eval (polymarket_weather_analysis.py).")
    print("==================================================================================")


if __name__ == "__main__":
    main()
