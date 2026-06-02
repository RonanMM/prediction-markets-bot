"""
historical_backtester.py — Grades your point-in-time weather predictions against actual Polymarket resolutions.

HOW THE BACKTESTING ARCHITECTURE WORKS:
1. The Data Vault (data/): Every time you run main.py or the fetch scripts, it saves a snapshot of the live Polymarket odds and Open-Meteo forecasts into the `data/` folder. This folder acts as an append-only historical vault.
2. The Time Machine (polymarket_weather_analysis.py): When you run the analyzer, it iterates through all of those historical snapshots in the `data/` folder. It pretends to be at that exact moment in time, applies your chosen model (e.g. ML Ensemble vs NWP Baseline), makes a betting decision, and saves that massive historical ledger of simulated bets to `output/opportunities_v4.csv`.
3. The Grader (this script): This script simply reads that `opportunities_v4.csv` simulation ledger. It filters out bets made less than 12 hours in advance (too easy), fetches the actual ground-truth recorded temperatures from the archive API for the resolution airports, and grades the simulated bets as WIN/LOSS to calculate your true ROI.

Because of this architecture, you can completely change the bot's math or switch models, re-run the analyzer to overwrite the CSV with a new simulation, and immediately run this script to see if your new model would have been more profitable over the last few months!
"""
import pandas as pd
import requests
import time
from pathlib import Path

# ── 1. The Exact Airport Coordinates ─────────────────────────────────────────
RESOLUTION_STATIONS = {
    "Chicago": {"lat": 41.9742, "lon": -87.9073},       # O'Hare (KORD)
    "NYC": {"lat": 40.7769, "lon": -73.8740},           # LaGuardia (KLGA)
    "London": {"lat": 51.5050, "lon": 0.0553},          # London City Airport (EGLC)
    "HongKong": {"lat": 22.3019, "lon": 114.1741},      # Hong Kong Observatory HQ
    "Seoul": {"lat": 37.4602, "lon": 126.4407},         # Incheon Intl Airport (RKSI)
}

def fetch_actual_weather(city: str, target_date: str) -> float:
    """Fetch the exact historical temperature from the resolution airport."""
    coords = RESOLUTION_STATIONS.get(city)
    if not coords:
        return None
        
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={coords['lat']}&longitude={coords['lon']}"
        f"&start_date={target_date}&end_date={target_date}"
        f"&daily=temperature_2m_max&timezone=UTC"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        # Polymarket rules: take the max temp and round to nearest integer
        return float(resp['daily']['temperature_2m_max'][0])
    except Exception:
        return None

def run_backtest():
    opps_file = Path("output/opportunities_v4.csv")
    if not opps_file.exists():
        print("No opportunities_v4.csv found. Run your analyzer first!")
        return

    df = pd.read_csv(opps_file)
    
    # We only want to backtest bets that were made AT LEAST 12 hours before expiry.
    # Betting 1 hour before expiry is easy because the weather is already happening!
    df = df[df["days_ahead"] >= 0.5].copy()
    
    # ── POST-MARCH FILTER ──
    # Only grade markets from May onwards to see the new model's performance
    # df = df[df["target_date"] >= "2026-05-01"].copy()

    if df.empty:
        print("No historical bets found with > 0.5 days ahead.")
        return

    # ── DEDUPLICATION ──
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
        cache_key = f"{city}_{target_date}"
        if cache_key not in weather_cache:
            weather_cache[cache_key] = fetch_actual_weather(city, target_date)
            time.sleep(0.1) # Be polite to the API
            
        actual_temp = weather_cache[cache_key]
        if actual_temp is None:
            continue
            
        actual_rounded = round(actual_temp)
        
        # 2. Did the market resolve Yes or No?
        if "higher" in question or "above" in question or "more" in question or "at least" in question:
            resolved_yes = actual_rounded >= round(bin_temp)
        elif "lower" in question or "below" in question or "less" in question or "at most" in question:
            resolved_yes = actual_rounded <= round(bin_temp)
        else:
            resolved_yes = actual_rounded == round(bin_temp) # Exact market
            
        # 3. Did we win?
        we_won = (bet_side == "Yes" and resolved_yes) or (bet_side == "No" and not resolved_yes)
        
        # 4. Calculate Profit using the model's actual Kelly sizing ($1000 bankroll)
        kelly = float(row.get("kelly", 0.0))
        bet_size = 1000.0 * kelly
        
        if bet_size < 1.0:
            continue

        if we_won:
            payout = bet_size / their_prob
            profit = (payout - bet_size) * 0.98 # Minus 2% Polymarket fee
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