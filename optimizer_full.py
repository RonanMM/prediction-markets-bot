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

RESOLUTION_STATIONS = {
    "Chicago": {"lat": 41.9742, "lon": -87.9073},
    "NYC": {"lat": 40.7769, "lon": -73.8740},
    "London": {"lat": 51.5050, "lon": 0.0553},
    "HongKong": {"lat": 22.3019, "lon": 114.1741},
    "Seoul": {"lat": 37.4602, "lon": 126.4407},
}

WEATHER_CACHE = {}

def fetch_actual_weather(city: str, target_date: str) -> float:
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
        return float(resp['daily']['temperature_2m_max'][0])
    except Exception:
        return None

def evaluate_bets(bets: list, opportunities_map: dict):
    total_profit = 0.0
    total_staked = 0.0
    wins = 0
    
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
        
        cache_key = f"{city}_{target_date}"
        if cache_key not in WEATHER_CACHE:
            WEATHER_CACHE[cache_key] = fetch_actual_weather(city, target_date)
            time.sleep(0.05)
            
        actual_temp = WEATHER_CACHE[cache_key]
        if actual_temp is None:
            continue
            
        actual_rounded = round(actual_temp)
        if "higher" in question or "above" in question or "more" in question or "at least" in question:
            resolved_yes = actual_rounded >= round(bin_temp)
        elif "lower" in question or "below" in question or "less" in question or "at most" in question:
            resolved_yes = actual_rounded <= round(bin_temp)
        else:
            resolved_yes = actual_rounded == round(bin_temp)
            
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
        "wins": wins,
        "profit": total_profit,
        "staked": total_staked,
        "roi": roi
    }

def run_grid_search(data_dir: Path):
    cities = discover_cities(data_dir)
    print(f"Found cities: {cities}")
    
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
    
    # Run once to pre-warm the cache so prints look clean
    print("Warming caches...")
    import sys
    import os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        all_opps = []
        for city in cities:
            all_opps.extend(analyse_city(data_dir, city))
        opps_map = {o.condition_id: o for o in all_opps}
        bot = WeatherBettingBot(bankroll=1000.0, dry_run=True, live_mode=False)
        bot.run(all_opps)
        evaluate_bets(bot.log, opps_map)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    print("Cache warmed.")

    best_roi = -1
    best_params = None
    best_result = None
    
    start_time = time.time()
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        # Override globals
        signals.MOMENTUM_EMA_SPAN = params["MOMENTUM_EMA_SPAN"]
        signals.MOMENTUM_THRESHOLD = params["MOMENTUM_THRESHOLD"]
        signals.STALE_HOURS = params["STALE_HOURS"]
        signals.STALE_MOVE_THRESHOLD = params["STALE_MOVE_THRESHOLD"]
        signals.INFORMED_RECENCY = params["INFORMED_RECENCY"]
        
        all_opps = []
        
        # Suppress prints from analyse_city
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            for city in cities:
                opps = analyse_city(data_dir, city)
                all_opps.extend(opps)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        opps_map = {o.condition_id: o for o in all_opps}
        
        # Manually apply filters and caps using historical days_ahead
        from src.polymarket_weather.engine import apply_group_kelly_cap, apply_portfolio_cap
        
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
                
        # Filter out bets smaller than $1
        final_bets = [b for b in final_bets if b["size_usdc"] >= 1.0]
        
        result = evaluate_bets(final_bets, opps_map)
        
        if result['roi'] > best_roi and result['bets'] > 10:
            best_roi = result['roi']
            best_params = params
            best_result = result
            
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{i+1}/{len(combinations)}] Elapsed: {elapsed:.1f}s. Best ROI so far: {best_roi:.1%} {best_params}")
            
    print("\n" + "="*50)
    print("GRID SEARCH COMPLETED")
    print("="*50)
    if best_params:
        print(f"Optimal Alpha Signal Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"Final Performance:")
        print(f"  ROI: {best_result['roi']:.1%}")
        print(f"  Net Profit: ${best_result['profit']:.2f}")
        print(f"  Total Staked: ${best_result['staked']:.2f}")
        print(f"  Number of Bets: {best_result['bets']}")
        print(f"  Win Rate: {best_result['wins']/best_result['bets']:.1%}")
    else:
        print("No valid parameter combinations found that produced >10 bets.")
    
def main():
    data_dir = Path("src/polymarket_weather/data")
    run_grid_search(data_dir)

if __name__ == "__main__":
    main()
