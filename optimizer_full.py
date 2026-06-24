import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
import itertools
import sys
sys.path.insert(0, str(Path("src/polymarket_weather").absolute()))

import config
import signals
import predictors.nwp_fallback as nwp_fallback
from engine import analyse_city, WeatherBettingBot
from data_loader import discover_cities

from resolution_anchors import RESOLUTION_ANCHORS

# Forecast coords per resolution station, derived from the single source of truth.
# Keyed by canonical name AND alias (e.g. "NYC", "HongKong") so existing lookups keep working.
RESOLUTION_STATIONS = {}
for _city, _a in RESOLUTION_ANCHORS.items():
    _c = {"lat": _a["forecast_lat"], "lon": _a["forecast_lon"]}
    RESOLUTION_STATIONS[_city] = _c
    for _alias in _a.get("aliases", []):
        RESOLUTION_STATIONS[_alias] = _c

WEATHER_CACHE = {}

from grading import resolves_yes  # resolution-station truth grader, native-unit rounding

def evaluate_bets(bets: list, opportunities_map: dict):
    total_profit = 0.0
    total_staked = 0.0
    wins = 0
    graded = 0  # bets we could actually grade (station truth available) — the honest sample size

    for bet in bets:
        cid = bet["condition_id"]
        opp = opportunities_map[cid]
        city = opp.city
        target_date = opp.target_date
        bin_temp = opp.bin_temp_c
        question = opp.question.lower()
        bet_side = bet["bet_side"]
        their_prob = bet["market_prob"]
        bet_size = bet["size_usdc"]

        resolved_yes = resolves_yes(city, target_date, question, bin_temp)
        if resolved_yes is None:
            continue
        graded += 1

        we_won = (bet_side == "Yes" and resolved_yes) or (bet_side == "No" and not resolved_yes)

        if we_won:
            payout = bet_size / their_prob
            profit = (payout - bet_size) * 0.98
            wins += 1
        else:
            profit = -bet_size

        total_profit += profit
        total_staked += bet_size

    roi = total_profit / total_staked if total_staked > 0 else 0
    return {
        "bets": len(bets),
        "graded": graded,
        "wins": wins,
        "profit": total_profit,
        "staked": total_staked,
        "roi": roi
    }


def _final_bets_for_params(params, cities, data_dir):
    """Run the full engine under `params` and return (final_bets, opps_map).

    Overrides the alpha-signal globals, re-analyses every city, applies the same group/portfolio
    caps and de-dup the in-sample path uses, and returns one bet per market plus the opp map (so
    callers can split bets by target_date for an out-of-sample evaluation).
    """
    import sys, os
    signals.MOMENTUM_EMA_SPAN = params["MOMENTUM_EMA_SPAN"]
    signals.MOMENTUM_THRESHOLD = params["MOMENTUM_THRESHOLD"]
    signals.STALE_HOURS = params["STALE_HOURS"]
    signals.STALE_MOVE_THRESHOLD = params["STALE_MOVE_THRESHOLD"]
    signals.INFORMED_RECENCY = params["INFORMED_RECENCY"]

    all_opps = []
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        for city in cities:
            all_opps.extend(analyse_city(data_dir, city))
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    opps_map = {o.condition_id: o for o in all_opps}
    from engine import apply_group_kelly_cap, apply_portfolio_cap
    candidate = [o for o in all_opps if o.days_ahead >= 0.0 and o.kelly > 0]
    candidate = apply_group_kelly_cap(candidate, 1000.0)
    candidate = apply_portfolio_cap(candidate)
    candidate.sort(key=lambda o: o.alpha_score, reverse=True)

    seen = set()
    final_bets = []
    for o in candidate:
        if o.condition_id not in seen:
            final_bets.append({
                "condition_id": o.condition_id,
                "bet_side": o.bet_side,
                "market_prob": o.their_prob,
                "size_usdc": round(1000.0 * o.kelly, 2),
            })
            seen.add(o.condition_id)
    return [b for b in final_bets if b["size_usdc"] >= 1.0], opps_map


def _split_eval(final_bets, opps_map, cut):
    """Grade train (target_date <= cut) and validation (target_date > cut) bets separately."""
    train = [b for b in final_bets if str(opps_map[b["condition_id"]].target_date) <= cut]
    valid = [b for b in final_bets if str(opps_map[b["condition_id"]].target_date) > cut]
    return evaluate_bets(train, opps_map), evaluate_bets(valid, opps_map)

MIN_VALID_BETS = 30  # do not endorse a tuned winner on fewer held-out (gradable) bets


def run_grid_search(data_dir: Path):
    cities = discover_cities(data_dir)
    print(f"Found cities: {cities}")

    # SEED = current production signal params (read live from signals.py so they stay in sync).
    SEED = {
        "MOMENTUM_EMA_SPAN": signals.MOMENTUM_EMA_SPAN,
        "MOMENTUM_THRESHOLD": signals.MOMENTUM_THRESHOLD,
        "STALE_HOURS": signals.STALE_HOURS,
        "STALE_MOVE_THRESHOLD": signals.STALE_MOVE_THRESHOLD,
        "INFORMED_RECENCY": signals.INFORMED_RECENCY,
    }

    # Grid of alpha signal parameters
    param_grid = {
        "MOMENTUM_EMA_SPAN": [2, 3, 5],
        "MOMENTUM_THRESHOLD": [0.10, 0.15, 0.20],
        "STALE_HOURS": [4, 6, 8],
        "STALE_MOVE_THRESHOLD": [0.005, 0.01, 0.02],
        "INFORMED_RECENCY": [0.5, 0.65, 0.8]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    print(f"Starting grid search over {len(combinations)} combinations.")
    print("This runs the full engine analysis back-to-front for every parameter set.")

    # Warm the cache (clean prints) AND derive the time-ordered train/validate cut from the
    # opportunities' target dates (60th percentile), mirroring optimizer.py's split.
    print("Warming caches...")
    warm_bets, warm_map = _final_bets_for_params(SEED, cities, data_dir)
    dates = sorted({str(o.target_date) for o in warm_map.values()})
    cut = dates[int(len(dates) * 0.6)] if len(dates) > 2 else None
    print(f"Cache warmed. Split: train target_date <= {cut} < validate "
          f"(over {dates[0]}..{dates[-1]}).")

    best_train_roi = -1
    best_params = None
    best_train = None
    best_valid = None

    start_time = time.time()

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        final_bets, opps_map = _final_bets_for_params(params, cities, data_dir)
        if cut is None:
            continue
        train_res, valid_res = _split_eval(final_bets, opps_map, cut)

        # Tune ON TRAIN ONLY (require a non-trivial train sample); judge on held-out validation.
        if train_res['roi'] > best_train_roi and train_res['graded'] > 10:
            best_train_roi = train_res['roi']
            best_params = params
            best_train = train_res
            best_valid = valid_res

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{i+1}/{len(combinations)}] Elapsed: {elapsed:.1f}s. "
                  f"Best train ROI: {best_train_roi:.1%} {best_params}")

    # SEED's out-of-sample performance, the bar a tuned winner must clear.
    seed_bets, seed_map = _final_bets_for_params(SEED, cities, data_dir)
    _, seed_valid = _split_eval(seed_bets, seed_map, cut) if cut else (None, None)

    print("\n" + "=" * 64)
    print("GRID SEARCH COMPLETED (out-of-sample guarded)")
    print("=" * 64)
    if not best_params:
        print("No parameter combination produced >10 gradable train bets.")
        return

    print("Best by TRAIN ROI (in-sample):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  train ROI {best_train['roi']:.1%} ({best_train['graded']} gradable bets)")

    if seed_valid is not None:
        print(f"\n  SEED  (current config) held-out ROI: {seed_valid['roi']:.1%} "
              f"({seed_valid['graded']} gradable bets)")
        print(f"  TUNED (in-sample best) held-out ROI: {best_valid['roi']:.1%} "
              f"({best_valid['graded']} gradable bets)")

    # Guardrail: only endorse a change that beats SEED out-of-sample on a non-tiny sample.
    if cut is None or seed_valid is None:
        verdict = "INSUFFICIENT DATA — no validation split; keep current config."
    elif best_params == SEED:
        verdict = "Best in-sample params EQUAL the current config; nothing to change."
    elif best_valid['graded'] < MIN_VALID_BETS:
        verdict = (f"NOT ROBUST — only {best_valid['graded']} held-out bets "
                   f"(<{MIN_VALID_BETS}); keep current config until more data is available.")
    elif best_valid['roi'] <= seed_valid['roi'] + 1e-9:
        verdict = "NO out-of-sample improvement over current config; keep config."
    else:
        verdict = (f"Tuned beats seed out-of-sample by "
                   f"+{(best_valid['roi'] - seed_valid['roi']) * 100:.1f} pp — candidate, "
                   "but confirm on more data before committing.")
    print(f"\n  VERDICT: {verdict}")
    print("  (Reported only — this script does NOT modify signals.py or config.py.)")
    print("=" * 64)

def main():
    data_dir = Path("src/polymarket_weather/data")
    run_grid_search(data_dir)

if __name__ == "__main__":
    main()
