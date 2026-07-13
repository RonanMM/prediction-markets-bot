"""audit_settlements.py — validate our grading against ACTUAL market settlements (megaplan W0).

The trick: a resolved market's terminal price IS its settlement. For every market whose LAST
stored snapshot is (a) after the target local day ended and (b) pinned (≤3¢ or ≥97¢), treat the
pinned side as the real-world resolution and compare it to `grading.resolves_yes`. No oracle
feed needed, works across the whole snapshot history.

This is the guard that caught the Wunderground-vs-CLI divergence (4/60 graded opposite to
settlement on 2026-07-12, all whole-degree boundary cases; the three US cases are fixed by
`wu_truth.py`). Run it after ANY truth-source or grading change:

    python audit_settlements.py            # from src/polymarket_weather/

Exit code 0 when agreement ≥ the pre-registered floor (95%), 1 otherwise — usable in CI.
"""
import ast
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from grading import resolves_yes, fetch_actual_weather
from pmf import parse_question, parse_question_date

_DATA = Path(__file__).resolve().parent / "data" / "polymarket"
_SLUGS = {"chicago": "Chicago", "new_york_city": "NYC", "london": "London",
          "seoul": "Seoul", "hong_kong": "HongKong"}
_TZ = {"Chicago": "America/Chicago", "NYC": "America/New_York", "London": "Europe/London",
       "Seoul": "Asia/Seoul", "HongKong": "Asia/Hong_Kong"}
_PIN = 0.03           # last price within this of 0/1 counts as settled
_AGREEMENT_FLOOR = 0.95


def _yes_prob(j):
    if not isinstance(j, str):
        return None
    try:
        return json.loads(j).get("Yes")
    except Exception:
        try:
            return ast.literal_eval(j).get("Yes")
        except Exception:
            return None


def audit():
    """Return (n_agree, n_total, disagreements) over all settled, gradable markets."""
    tot = agree = 0
    dis = []
    for slug, city in _SLUGS.items():
        path = _DATA / f"{slug}_snapshots.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s["t"] = pd.to_datetime(s["fetched_at_utc"], utc=True, format="mixed")
        s["yes"] = s["outcome_probs_json"].map(_yes_prob)
        s["end"] = pd.to_datetime(s["end_date_iso"], errors="coerce", utc=True).dt.tz_localize(None)
        s = s.dropna(subset=["yes", "end"])
        for _cid, g in s.groupby("condition_id"):
            g = g.sort_values("t")
            q = g["question"].iloc[0]
            pq = parse_question(q)
            if not pq:
                continue
            tgt = parse_question_date(q, g["end"].iloc[0]) or g["end"].iloc[0].date()
            day_end = (pd.Timestamp(tgt) + pd.Timedelta(days=1)).tz_localize(
                ZoneInfo(_TZ[city])).tz_convert("UTC")
            last = g.iloc[-1]
            if last["t"] < day_end or not (last["yes"] <= _PIN or last["yes"] >= 1 - _PIN):
                continue
            settled = 1 if last["yes"] >= 1 - _PIN else 0
            ours = resolves_yes(city, str(tgt), q, pq["temp_c"])
            if ours is None:
                continue
            tot += 1
            if int(ours) == settled:
                agree += 1
            else:
                dis.append((city, str(tgt), int(ours), settled,
                            fetch_actual_weather(city, str(tgt), q), q))
    return agree, tot, dis


def main():
    agree, tot, dis = audit()
    rate = agree / tot if tot else 0.0
    print(f"SETTLEMENT AUDIT: {agree}/{tot} graded markets agree with actual settlements "
          f"({rate:.1%}; floor {_AGREEMENT_FLOOR:.0%})")
    for city, tgt, ours, settled, actual, q in sorted(dis):
        print(f"  DISAGREE {city:<9} {tgt}  ourgrade={ours} settled={settled} "
              f"truth={'?' if actual is None else round(actual, 2)}°C | {q[:64]}")
    if tot and rate < _AGREEMENT_FLOOR:
        print("  ❌ below floor — grading truth is NOT settlement-faithful; fix before trusting eval.")
        sys.exit(1)
    print("  ✅ grading is settlement-faithful at the audited floor "
          "(boundary risk remains for Seoul/London tenths — see wu_truth.py docstring).")


if __name__ == "__main__":
    main()
