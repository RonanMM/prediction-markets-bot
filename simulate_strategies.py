import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from scipy.stats import t as student_t
import sys

# Source root on path so intra-project imports (e.g. resolution_anchors) work when run from repo root.
sys.path.insert(0, str(Path("src/polymarket_weather").absolute()))

from resolution_anchors import RESOLUTION_ANCHORS

# Forecast coords per resolution station, derived from the single source of truth.
# Keyed by canonical name AND alias (e.g. "NYC", "HongKong") so existing lookups keep working.
RESOLUTION_STATIONS = {}
for _city, _a in RESOLUTION_ANCHORS.items():
    _c = {"lat": _a["forecast_lat"], "lon": _a["forecast_lon"]}
    RESOLUTION_STATIONS[_city] = _c
    for _alias in _a.get("aliases", []):
        RESOLUTION_STATIONS[_alias] = _c

from grading import resolves_yes  # resolution-station truth grader, native-unit rounding

def _cdf(x: float, mu: float, sigma: float, nu: float) -> float:
    from scipy.stats import norm
    if nu > 30:
        return float(norm.cdf(x, loc=mu, scale=sigma))
    return float(student_t.cdf((x - mu) / sigma, df=nu))

def _bin_prob(temp_c: float, mu: float, sigma: float, nu: float, half_width: float = 0.5) -> float:
    upper_bound = temp_c + half_width
    lower_bound = temp_c - half_width
    return _cdf(upper_bound, mu, sigma, nu) - _cdf(lower_bound, mu, sigma, nu)

WEATHER_CACHE = {}

def simulate_strategy(ml_path: Path, strategy_name: str, **kwargs):
    if not ml_path.exists():
        return
    df = pd.read_csv(ml_path)
    df = df[df["days_ahead"] >= 0.5].copy()
    if df.empty:
        return
    
    df = df.sort_values("fetched_at").groupby("condition_id").last().reset_index()
    
    global WEATHER_CACHE
    total_bets = 0

    wins = 0
    total_profit = 0.0
    total_staked = 0.0
    
    for _, row in df.iterrows():
        city = row["city"]
        target_date = row["target_date"]
        bin_temp = row["bin_temp_c"]
        their_prob = row["their_prob"]
        question = str(row["question"]).lower()
        
        # 1. Get ensemble values
        raw_mu = row["forecast_mu"]
        s_boost = row.get("sigma_boost", 0.0)
        raw_sigma = row["forecast_sigma"] - s_boost
        raw_nu = row["forecast_nu"]
        
        # 2. Reconstruct ML and Ensemble probabilities for YES
        bet_side_ml = row["bet_side"]
        our_prob_ml = row["our_prob"]
        
        # f_prob_ml is probability of YES predicted by ML
        f_prob_ml = our_prob_ml if bet_side_ml == "Yes" else (1.0 - our_prob_ml)
        
        # f_prob_ens is probability of YES predicted by Ensemble
        f_prob_ens = _bin_prob(bin_temp, raw_mu, row["forecast_sigma"], raw_nu)
        
        # their_prob_yes is the market price of YES
        their_prob_yes = row["their_prob"] if bet_side_ml == "Yes" else (1.0 - row["their_prob"])
        
        # Calculate ensemble edge and side
        edge_ens = f_prob_ens - their_prob_yes
        bet_side_ens = "Yes" if edge_ens > 0 else "No"
        
        # Apply strategy modification
        skip = False
        
        if strategy_name == "conflict_gating":
            if bet_side_ml != bet_side_ens:
                skip = True
                
        elif strategy_name == "sigma_filter":
            threshold = kwargs.get("threshold", 1.8)
            if raw_sigma > threshold:
                f_prob_ml = f_prob_ens
                bet_side_ml = bet_side_ens
                our_prob_ml = f_prob_ens if bet_side_ml == "Yes" else (1.0 - f_prob_ens)
                
        elif strategy_name == "combined_filter":
            threshold = kwargs.get("threshold", 1.2)
            if raw_sigma > threshold:
                f_prob_ml = f_prob_ens
                bet_side_ml = bet_side_ens
                our_prob_ml = f_prob_ens if bet_side_ml == "Yes" else (1.0 - f_prob_ens)
            else:
                if bet_side_ml != bet_side_ens:
                    skip = True
                    
        elif strategy_name == "dynamic_blend":
            tau = kwargs.get("tau", 1.5)
            w = np.exp(-(raw_sigma**2) / (2.0 * tau**2))
            
            f_prob_blended = w * f_prob_ml + (1.0 - w) * f_prob_ens
            edge_blended = f_prob_blended - their_prob_yes
            
            if abs(edge_blended) < 0.06:
                skip = True
            else:
                bet_side_ml = "Yes" if edge_blended > 0 else "No"
                our_prob_ml = f_prob_blended if bet_side_ml == "Yes" else (1.0 - f_prob_blended)
                
        elif strategy_name == "baseline_ml":
            pass
            
        if skip:
            continue
            
        # Recalculate their_prob based on the final active bet side
        their_prob = their_prob_yes if bet_side_ml == "Yes" else (1.0 - their_prob_yes)
            
        # 3. Fetch Truth
        resolved_yes = resolves_yes(city, target_date, question, bin_temp)
        if resolved_yes is None:
            continue

        we_won = (bet_side_ml == "Yes" and resolved_yes) or (bet_side_ml == "No" and not resolved_yes)
        
        # Kelly Sizing
        fee = 0.02
        b = ((1.0 - their_prob) / their_prob) * (1.0 - fee)
        # Fractional Kelly sizing
        kf = kwargs.get("kelly_fraction", 0.25)
        kelly = kf * (b * our_prob_ml - (1.0 - our_prob_ml)) / b
        kelly = np.clip(kelly, 0.0, 0.12) # MAX_KELLY_PER_BET
        
        bet_size = 1000.0 * kelly
        if bet_size < 1.0:
            continue
            
        if we_won:
            payout = bet_size / their_prob
            profit = (payout - bet_size) * 0.98
            wins += 1
        else:
            profit = -bet_size
            
        total_bets += 1
        total_profit += profit
        total_staked += bet_size
        
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    kf_str = f" (kf={kwargs.get('kelly_fraction', 0.25):.2f})"
    name_str = f"{strategy_name}{kf_str}"
    print(f"{name_str:<25}: Bets={total_bets:<3} Wins={wins:<3} WinRate={wins/total_bets if total_bets>0 else 0:.1%} Staked=${total_staked:<7.2f} Profit=${total_profit:<+8.2f} ROI={roi:.1%}")

def main():
    ml_path = Path("src/polymarket_weather/output/opportunities_evaluation_ml.csv")
    print("="*85)
    print("SIMULATING STRATEGIES FOR MAXIMIZING P&L")
    print("="*85)
    
    # 1. Baseline runs
    simulate_strategy(ml_path, "baseline_ml", kelly_fraction=0.25)
    simulate_strategy(ml_path, "conflict_gating", kelly_fraction=0.25)
    
    # 2. Sigma filter sweep
    for th in [1.2, 1.5, 1.8]:
        simulate_strategy(ml_path, "sigma_filter", threshold=th, kelly_fraction=0.25)
        
    # 3. Combined filter sweep
    for th in [1.2, 1.5, 1.8]:
        simulate_strategy(ml_path, "combined_filter", threshold=th, kelly_fraction=0.25)
        
    # 4. Sweep Kelly fraction on our best strategy (Combined Filter th=1.2)
    print("\n-- Combined Filter th=1.2 with different Kelly Fractions --")
    for kf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        simulate_strategy(ml_path, "combined_filter", threshold=1.2, kelly_fraction=kf)

    # 5. Sweep Kelly fraction on Baseline ML (to see how it scales)
    print("\n-- Baseline ML with different Kelly Fractions --")
    for kf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        simulate_strategy(ml_path, "baseline_ml", kelly_fraction=kf)
        
    print("="*85)

if __name__ == "__main__":
    main()

