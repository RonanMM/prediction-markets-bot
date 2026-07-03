"""
historical_backtester.py — Grades your point-in-time weather predictions against actual Polymarket resolutions.
"""
import sys
import pandas as pd
import requests
import time
from pathlib import Path

# ── 1. The Exact Airport Coordinates ─────────────────────────────────────────
from resolution_anchors import RESOLUTION_ANCHORS

# Forecast coords per resolution station, derived from the single source of truth.
# Keyed by canonical name AND alias (e.g. "NYC", "HongKong") so existing lookups keep working.
RESOLUTION_STATIONS = {}
for _city, _a in RESOLUTION_ANCHORS.items():
    _c = {"lat": _a["forecast_lat"], "lon": _a["forecast_lon"]}
    RESOLUTION_STATIONS[_city] = _c
    for _alias in _a.get("aliases", []):
        RESOLUTION_STATIONS[_alias] = _c

from grading import fetch_actual_weather, resolves_yes  # station-truth grader (+ native-unit resolution)
from config import FEE_RATE, HALF_SPREAD

def run_backtest():
    filename = sys.argv[2] if len(sys.argv) > 2 else "output/opportunities_v4.csv"
    opps_file = Path(filename)
    if not opps_file.exists():
        print(f"No {filename} found. Run your analyzer first!")
        return

    df = pd.read_csv(opps_file)
    
    # We only want to backtest bets that were made AT LEAST 12 hours before expiry.
    # Betting 1 hour before expiry is easy because the weather is already happening!
    df = df[df["days_ahead"] >= 0.5].copy()
    
    start_date = sys.argv[1] if len(sys.argv) > 1 else "2000-01-01"
    # ── DATE FILTER ──
    df = df[df["target_date"] >= start_date].copy()

    if df.empty:
        print("No historical bets found with > 0.5 days ahead.")
        return

    # Only grade the final prediction the bot made for each market before the cutoff
    df = df.sort_values("fetched_at").groupby("condition_id").last().reset_index()

    total_bets = 0
    wins = 0
    total_profit = 0.0
    total_staked = 0.0
    weather_cache = {}

    print("\n" + "="*95)
    print(f"{'City':<10} | {'Date':<10} | {'Bet':<4} | {'Target':<6} | {'Actual':<6} | {'Result':<6} | {'Stake':<7} | {'Profit':<8}")
    print("-" * 95)

    for _, row in df.iterrows():
        city = row["city"]
        target_date = row["target_date"]
        bet_side = row["bet_side"]
        their_prob = row["their_prob"]
        bin_temp = row["bin_temp_c"]
        question = str(row["question"]).lower()
        
        # 1. Fetch Truth Data
        actual_temp = fetch_actual_weather(city, target_date, question)
        if actual_temp is None:
            continue

        actual_rounded = round(actual_temp)  # °C, for the display column only

        # 2. Did the market resolve Yes or No? (graded in the market's native unit)
        resolved_yes = resolves_yes(city, target_date, question, bin_temp)

        # 3. Did we win?
        we_won = (bet_side == "Yes" and resolved_yes) or (bet_side == "No" and not resolved_yes)
        
        # 4. Calculate Profit using the model's actual Kelly sizing ($1000 bankroll)
        kelly = float(row.get("kelly", 0.0))
        bet_size = 1000.0 * kelly
        
        if bet_size < 1.0:
            continue

        # Honest execution cost: cross half the spread on entry (fewer shares), and pay the
        # taker fee on the winning payout — not a flat discount on net profit.
        their_eff = min(0.999, their_prob + HALF_SPREAD)
        shares = bet_size / their_eff
        if we_won:
            profit = shares * (1.0 - FEE_RATE) - bet_size
            wins += 1
            result_str = "✅ WIN"
        else:
            profit = -bet_size
            result_str = "❌ LOSS"

        total_bets += 1
        total_profit += profit
        total_staked += bet_size

        print(f"{city[:10]:<10} | {target_date:<10} | {bet_side:<4} | {round(bin_temp)}°C   | {actual_rounded}°C   | {result_str:<6} | ${bet_size:<6.2f} | ${profit:>7.2f}")

    print("="*95)
    print(f"Total Resolved Bets : {total_bets}")
    if total_bets > 0:
        print(f"Win Rate            : {wins/total_bets:.1%}")
        print(f"Total Staked        : ${total_staked:,.2f}")
        print(f"Total P&L           : ${total_profit:+.2f}")
        print(f"ROI                 : {total_profit/total_staked:.1%}")

if __name__ == "__main__":
    run_backtest()