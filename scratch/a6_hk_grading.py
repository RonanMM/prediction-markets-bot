"""A6 — settlement audit, per city, and the Hong Kong resolution-unit question.

Reuses audit_settlements.py's own definition of "settled" (last snapshot after the local day
ended AND pinned within 3c of 0/1) so the comparison is the repo's, not mine.
"""
import sys

import pandas as pd
from zoneinfo import ZoneInfo

import rc_lib as R

sys.path.insert(0, R.SRC)
import audit_settlements as A  # noqa: E402
import grading  # noqa: E402
from pmf import parse_question, parse_question_date, resolves_yes_temp  # noqa: E402

rows = []
for slug, city in A._SLUGS.items():
    path = A._DATA / f"{slug}_snapshots.csv"
    s = pd.read_csv(path)
    s["t"] = pd.to_datetime(s["fetched_at_utc"], utc=True, format="mixed")
    s["yes"] = s["outcome_probs_json"].map(A._yes_prob)
    s["end"] = pd.to_datetime(s["end_date_iso"], errors="coerce", utc=True).dt.tz_localize(None)
    s = s.dropna(subset=["yes", "end"])
    for cid, g in s.groupby("condition_id"):
        g = g.sort_values("t")
        q = g["question"].iloc[0]
        pq = parse_question(q)
        if not pq:
            continue
        tgt = parse_question_date(q, g["end"].iloc[0]) or g["end"].iloc[0].date()
        day_end = (pd.Timestamp(tgt) + pd.Timedelta(days=1)).tz_localize(
            ZoneInfo(A._TZ[city])).tz_convert("UTC")
        last = g.iloc[-1]
        if last["t"] < day_end or not (last["yes"] <= A._PIN or last["yes"] >= 1 - A._PIN):
            continue
        settled = 1 if last["yes"] >= 1 - A._PIN else 0
        full = {"Chicago": "Chicago", "NYC": "New York City", "London": "London",
                "Seoul": "Seoul", "HongKong": "Hong Kong"}[city]
        ours = grading.resolves_yes(full, str(tgt), q, pq["temp_c"])
        if ours is None:
            continue
        a = grading.fetch_actual_weather(full, str(tgt), q)
        alt = int(resolves_yes_temp(pq, grading.native_round(a, "whole °C"),
                                    "whole °C", grading.native_round))
        rows.append(dict(city=city, cid=cid, target_date=str(tgt), settled=settled,
                         ours=int(ours), alt_wholeC=alt, actual=a, q=q))

d = pd.DataFrame(rows)
print("=" * 100)
print(f"A6  settlement audit — {len(d)} settled+gradable markets")
print("\n  per city, repo grading vs actual settlement:")
t = d.groupby("city").apply(lambda g: pd.Series({
    "n": len(g), "agree": (g["ours"] == g["settled"]).sum(),
    "pct": (g["ours"] == g["settled"]).mean(),
    "settled_YES": g["settled"].sum(), "we_say_YES": g["ours"].sum(),
    "missed_YES": ((g["settled"] == 1) & (g["ours"] == 0)).sum(),
    "false_YES": ((g["settled"] == 0) & (g["ours"] == 1)).sum()}), include_groups=False)
print(t.round(4).to_string())
print(f"\n  OVERALL {(d['ours']==d['settled']).mean():.4f}  (floor {A._AGREEMENT_FLOOR})")

print("\n  Hong Kong ONLY — repo rule (resolution_unit '0.1 °C') vs whole-°C rule:")
hk = d[d["city"] == "HongKong"]
if len(hk):
    print(f"    n={len(hk)}  settled YES={hk['settled'].sum()}")
    print(f"    repo (0.1 °C):  agreement {(hk['ours']==hk['settled']).mean():.4f}  "
          f"YES recall {(hk[hk['settled']==1]['ours']==1).mean() if (hk['settled']==1).any() else float('nan'):.4f}")
    print(f"    whole °C     :  agreement {(hk['alt_wholeC']==hk['settled']).mean():.4f}  "
          f"YES recall {(hk[hk['settled']==1]['alt_wholeC']==1).mean() if (hk['settled']==1).any() else float('nan'):.4f}")

print("\n  Whole-°C alternative applied to EVERY city (sanity: must not hurt the °F cities):")
t2 = d.groupby("city").apply(lambda g: pd.Series({
    "n": len(g), "repo": (g["ours"] == g["settled"]).mean(),
    "wholeC": (g["alt_wholeC"] == g["settled"]).mean()}), include_groups=False)
print(t2.round(4).to_string())
