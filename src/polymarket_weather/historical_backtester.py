"""
historical_backtester.py — Grades your point-in-time weather predictions against actual Polymarket resolutions.
"""
import sys
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