import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from scipy.stats import t as student_t
import itertools

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

def _cdf(x: float, mu: float, sigma: float, nu: float) -> float:
    from scipy.stats import norm
    if nu > 30:
        return float(norm.cdf(x, loc=mu, scale=sigma))
    return float(student_t.cdf((x - mu) / sigma, df=nu))

def _bin_prob(temp_c: float, mu: float, sigma: float, nu: float, half_width: float = 0.5) -> float:
    upper_bound = temp_c + half_width
    lower_bound = temp_c - half_width
    return _cdf(upper_bound, mu, sigma, nu) - _cdf(lower_bound, mu, sigma, nu)

def apply_group_portfolio_caps(df, max_kelly_per_group, max_total_kelly):
    df['group_key'] = df['city'] + '|' + df['target_date']
    groups = df.groupby('group_key')
    
    scaled_rows = []
    for name, group in groups:
        total_raw = group['kelly'].sum()
        if total_raw > max_kelly_per_group:
            scale = max_kelly_per_group / total_raw
            group = group.copy()
            group['kelly'] = group['kelly'] * scale
        scaled_rows.append(group)
        
    if not scaled_rows:
        return df
    
    df_scaled = pd.concat(scaled_rows)
    
    total = df_scaled['kelly'].sum()
    if total > max_total_kelly:
        scale = max_total_kelly / total
        df_scaled['kelly'] = df_scaled['kelly'] * scale
        
    return df_scaled

def simulate_parameters(df_raw, strategy, params):
    df = df_raw.copy()
    
    if "min_days" in params:
        df = df[df["days_ahead"] >= params["min_days"]].copy()
    else:
        df = df[df["days_ahead"] >= 0.5].copy()
        
    if "MIN_LIQUIDITY" in params:
        df = df[df["liquidity"] >= params["MIN_LIQUIDITY"]].copy()
        
    if df.empty:
        return {"bets": 0, "profit": 0, "staked": 0, "roi": 0}

    df = df.sort_values("fetched_at").groupby("condition_id").last().reset_index()
    
    graded_rows = []
    for _, row in df.iterrows():
        city = row["city"]
        target_date = row["target_date"]
        bin_temp = row["bin_temp_c"]
        question = str(row["question"]).lower()
        
        raw_mu = row["forecast_mu"]
        s_boost = row.get("sigma_boost", 0.0)
        raw_sigma = row["forecast_sigma"] - s_boost
        raw_nu = row["forecast_nu"]
        
        bet_side_ml = row["bet_side"]
        our_prob_ml = row["our_prob"]
        
        f_prob_ml = our_prob_ml if bet_side_ml == "Yes" else (1.0 - our_prob_ml)
        f_prob_ens = _bin_prob(bin_temp, raw_mu, row["forecast_sigma"], raw_nu)
        their_prob_yes = row["their_prob"] if bet_side_ml == "Yes" else (1.0 - row["their_prob"])
        
        edge_ens = f_prob_ens - their_prob_yes
        bet_side_ens = "Yes" if edge_ens > 0 else "No"
        
        skip = False
        if strategy == "conflict_gating":
            if bet_side_ml != bet_side_ens or not params.get("conflict_gating", True):
                skip = True
        elif strategy == "combined_filter":
            threshold = params.get("sigma_threshold", 1.2)
            if raw_sigma > threshold:
                f_prob_ml = f_prob_ens
                bet_side_ml = bet_side_ens
                our_prob_ml = f_prob_ens if bet_side_ml == "Yes" else (1.0 - f_prob_ens)
            else:
                if bet_side_ml != bet_side_ens and params.get("conflict_gating", True):
                    skip = True
        elif strategy == "baseline_ml":
            pass
            
        if skip:
            continue
            
        their_prob = their_prob_yes if bet_side_ml == "Yes" else (1.0 - their_prob_yes)
        edge = our_prob_ml - their_prob
        
        if abs(edge) < params.get("MIN_EDGE", 0.06):
            continue
            
        fee = 0.02
        if their_prob <= 1e-4 or their_prob >= (1.0 - 1e-4):
            continue
            
        b = ((1.0 - their_prob) / their_prob) * (1.0 - fee)
        if b <= 0:
            continue
            
        kf = params.get("KELLY_FRACTION", 0.25)
        kelly = kf * (b * our_prob_ml - (1.0 - our_prob_ml)) / b
        kelly = np.clip(kelly, 0.0, params.get("MAX_KELLY_PER_BET", 0.12))
        
        if kelly <= 0:
            continue
            
        graded_rows.append({
            "city": city,
            "target_date": target_date,
            "bet_side": bet_side_ml,
            "bin_temp_c": bin_temp,
            "question": question,
            "their_prob": their_prob,
            "kelly": kelly
        })
        
    if not graded_rows:
        return {"bets": 0, "profit": 0, "staked": 0, "roi": 0}
        
    df_bets = pd.DataFrame(graded_rows)
    df_bets = apply_group_portfolio_caps(
        df_bets, 
        params.get("MAX_KELLY_PER_GROUP", 0.20),
        params.get("MAX_TOTAL_KELLY", 0.40)
    )
    
    total_profit = 0
    total_staked = 0
    wins = 0
    
    for _, row in df_bets.iterrows():
        city = row["city"]
        target_date = row["target_date"]
        bin_temp = row["bin_temp_c"]
        question = row["question"]
        bet_side = row["bet_side"]
        their_prob = row["their_prob"]
        
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
        
        bet_size = 1000.0 * row["kelly"]
        if bet_size < 1.0:
            continue
            
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
        "bets": len(df_bets),
        "wins": wins,
        "profit": total_profit,
        "staked": total_staked,
        "roi": roi
    }

def main():
    csv_path = Path("src/polymarket_weather/output/opportunities_evaluation_ml.csv")
    if not csv_path.exists():
        print(f"Data not found at {csv_path}")
        return
        
    df_raw = pd.read_csv(csv_path)
    
    print("Warming weather cache...")
    dates_cities = df_raw[['city', 'target_date']].drop_duplicates()
    for _, row in dates_cities.iterrows():
        key = f"{row['city']}_{row['target_date']}"
        if key not in WEATHER_CACHE:
            WEATHER_CACHE[key] = fetch_actual_weather(row['city'], row['target_date'])
            time.sleep(0.05)
            
    tier1 = {
        "MIN_EDGE": [0.04, 0.06, 0.08, 0.10],
        "KELLY_FRACTION": [0.15, 0.25, 0.35, 0.50],
        "MAX_KELLY_PER_BET": [0.08, 0.12, 0.20],
        "sigma_threshold": [1.0, 1.2, 1.5],
        "conflict_gating": [True, False]
    }
    
    print("\n--- Tier 1 Sweep ---")
    best_params = {}
    base_params = {
        "MIN_EDGE": 0.06,
        "KELLY_FRACTION": 0.25,
        "MAX_KELLY_PER_BET": 0.12,
        "MAX_KELLY_PER_GROUP": 0.20,
        "MAX_TOTAL_KELLY": 0.40,
        "sigma_threshold": 1.2,
        "conflict_gating": True,
        "MIN_LIQUIDITY": 400,
        "min_days": 0.0
    }
    
    results = []
    
    for key, values in tier1.items():
        for val in values:
            params = base_params.copy()
            params[key] = val
            res = simulate_parameters(df_raw, "combined_filter", params)
            print(f"Sweep {key}={val}: ROI={res['roi']:.1%} Profit=${res['profit']:.2f} Bets={res['bets']}")
            results.append((res['roi'], params.copy()))
            
    best_params = sorted(results, key=lambda x: x[0], reverse=True)[0][1]
    print(f"\nBest params after Tier 1: {best_params}")

    tier2 = {
        "MAX_KELLY_PER_GROUP": [0.15, 0.20, 0.30],
        "MAX_TOTAL_KELLY": [0.30, 0.40, 0.60],
        "MIN_LIQUIDITY": [200, 400, 1000],
        "min_days": [0.0, 0.5, 1.0]
    }
    
    print("\n--- Tier 2 Sweep ---")
    results2 = []
    for key, values in tier2.items():
        for val in values:
            params = best_params.copy()
            params[key] = val
            res = simulate_parameters(df_raw, "combined_filter", params)
            print(f"Sweep {key}={val}: ROI={res['roi']:.1%} Profit=${res['profit']:.2f} Bets={res['bets']}")
            results2.append((res['roi'], params.copy()))
            
    final_best_params = sorted(results2 + [(best_params.get("roi", 0), best_params)], key=lambda x: x[0], reverse=True)[0][1]
    print(f"\nOptimal parameters: {final_best_params}")
    res_final = simulate_parameters(df_raw, "combined_filter", final_best_params)
    print(f"Final Performance: ROI={res_final['roi']:.1%} Profit=${res_final['profit']:.2f} Bets={res_final['bets']}")

if __name__ == "__main__":
    main()
