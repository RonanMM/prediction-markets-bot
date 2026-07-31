"""A6b — how much of the shoulder book's reported edge is the Hong Kong grading bug?

Reuses shoulder_book's own settlement math (_net_edge) and stats_util clustering, and only
changes ONE thing: which grading rule Hong Kong gets.
"""
import sys

import pandas as pd

import rc_lib as R

sys.path.insert(0, R.SRC)
import shoulder_book as SB  # noqa: E402
import grading  # noqa: E402
import stats_util  # noqa: E402
from pmf import parse_question, resolves_yes_temp  # noqa: E402

FULL = {"Chicago": "Chicago", "NYC": "New York City", "London": "London",
        "Seoul": "Seoul", "HongKong": "Hong Kong"}

b = SB._load_book()
rows = []
for _, r in b.iterrows():
    pq = parse_question(str(r["question"]))
    if not pq:
        continue
    city = FULL[r["city"]]
    y_repo = grading.resolves_yes(city, r["target_date"], r["question"], pq["temp_c"])
    if y_repo is None:
        continue
    a = grading.fetch_actual_weather(city, r["target_date"], r["question"])
    unit = "whole °C" if city == "Hong Kong" else grading._UNIT.get(city, "whole °C")
    y_fix = int(resolves_yes_temp(pq, grading.native_round(a, unit), unit, grading.native_round))
    rows.append(dict(city=r["city"], leg=r["leg"], band=r["band"], side=r["side"],
                     target_date=r["target_date"], entry=float(r["entry_side_price"]),
                     yes=float(r["entry_yes_price"]),
                     won_repo=int((y_repo == 1) if r["side"] == "Yes" else (y_repo == 0)),
                     won_fix=int((y_fix == 1) if r["side"] == "Yes" else (y_fix == 0))))
d = pd.DataFrame(rows)
d = d[d["leg"] == "shoulder"]
d["target_date"] = d["target_date"].astype(str)

print("=" * 100)
print(f"A6b  Leg-1 shoulder sell — {len(d)} graded entries; Hong Kong is "
      f"{(d['city']=='HongKong').mean():.0%} of them")


def line(sub, wc, label):
    if not len(sub):
        print(f"    {label:<44} n=0")
        return
    net = SB._net_edge(sub[wc], sub["entry"])
    st = stats_util.interval(net, stats_util.cluster_key(
        sub.rename(columns={"city": "city"}).assign(city=sub["city"])))
    print(f"    {label:<44} n={len(sub):>4} cd={st['n_clusters']:>3} "
          f"winrate {sub[wc].mean():>5.1%}  taker net {st['mean']:>+7.4f}  "
          f"CI [{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]")


for band, sel in [("full [5,35)c", d), ("core [20,35)c", d[d["band"] == "core"]),
                  ("outer [5,20)c", d[d["band"] == "outer"])]:
    print(f"\n  {band}")
    line(sel, "won_repo", "as reported (repo grading, HK included)")
    line(sel, "won_fix", "HK regraded whole-°C (all 5 cities)")
    line(sel[sel["city"] != "HongKong"], "won_repo", "Hong Kong EXCLUDED")
    line(sel[sel["city"] == "HongKong"], "won_repo", "  ...Hong Kong only, repo grading")
    line(sel[sel["city"] == "HongKong"], "won_fix", "  ...Hong Kong only, whole-°C")

print("\n  per-city win rate of the shoulder sell (repo grading):")
print(d.groupby("city").agg(n=("won_repo", "size"), wr_repo=("won_repo", "mean"),
                            wr_fix=("won_fix", "mean")).round(3).to_string())
