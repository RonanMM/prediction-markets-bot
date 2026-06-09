import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

RESOLUTION_STATIONS = {
    "Chicago": {"lat": 41.9742, "lon": -87.9073},
    "NYC": {"lat": 40.7769, "lon": -73.8740},
    "London": {"lat": 51.5050, "lon": 0.0553},
    "HongKong": {"lat": 22.3019, "lon": 114.1741},
    "Seoul": {"lat": 37.4602, "lon": 126.4407},
}

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

def grade_opportunities(df_path: Path):
    if not df_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(df_path)
    df = df[df["days_ahead"] >= 0.5].copy()
    if df.empty:
        return pd.DataFrame()
    
    df = df.sort_values("fetched_at").groupby("condition_id").last().reset_index()
    
    graded_rows = []
    weather_cache = {}
    
    for _, row in df.iterrows():
        city = row["city"]
        target_date = row["target_date"]
        bet_side = row["bet_side"]
        their_prob = row["their_prob"]
        bin_temp = row["bin_temp_c"]
        question = str(row["question"]).lower()
        
        cache_key = f"{city}_{target_date}"
        if cache_key not in weather_cache:
            weather_cache[cache_key] = fetch_actual_weather(city, target_date)
            time.sleep(0.05)
            
        actual_temp = weather_cache[cache_key]
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
        
        kelly = float(row.get("kelly", 0.0))
        bet_size = 1000.0 * kelly
        
        if bet_size < 1.0:
            continue
            
        if we_won:
            payout = bet_size / their_prob
            profit = (payout - bet_size) * 0.98
        else:
            profit = -bet_size
            
        graded_rows.append({
            "condition_id": row["condition_id"],
            "city": city,
            "target_date": target_date,
            "bet_side": bet_side,
            "their_prob": their_prob,
            "our_prob": row["our_prob"],
            "edge": row["edge"],
            "kelly": kelly,
            "bet_size": bet_size,
            "we_won": we_won,
            "profit": profit
        })
        
    return pd.DataFrame(graded_rows)

def main():
    ml_path = Path("src/polymarket_weather/output/opportunities_evaluation_ml.csv")
    ens_path = Path("src/polymarket_weather/output/opportunities_evaluation_ensemble.csv")
    
    print("Grading ML predictions...")
    ml_graded = grade_opportunities(ml_path)
    print("Grading Ensemble predictions...")
    ens_graded = grade_opportunities(ens_path)
    
    if ml_graded.empty or ens_graded.empty:
        print("Data missing or empty.")
        return
        
    # Compare overall
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    print(f"ML Model:       Bets={len(ml_graded)}, Wins={ml_graded['we_won'].sum()}, WinRate={ml_graded['we_won'].mean():.1%}, P&L=${ml_graded['profit'].sum():+.2f}, AvgBetSize=${ml_graded['bet_size'].mean():.2f}")
    print(f"Ensemble Model: Bets={len(ens_graded)}, Wins={ens_graded['we_won'].sum()}, WinRate={ens_graded['we_won'].mean():.1%}, P&L=${ens_graded['profit'].sum():+.2f}, AvgBetSize=${ens_graded['bet_size'].mean():.2f}")
    
    # Merge on condition_id to find differences
    merged = pd.merge(
        ml_graded, ens_graded, 
        on="condition_id", 
        suffixes=("_ml", "_ens")
    )
    
    print("\n" + "="*50)
    print("MERGED BET COMPARISON (Bets made by BOTH)")
    print("="*50)
    print(f"Total overlapping bets: {len(merged)}")
    print(f"ML Net Profit on overlap:       ${merged['profit_ml'].sum():+.2f}")
    print(f"Ensemble Net Profit on overlap: ${merged['profit_ens'].sum():+.2f}")
    
    # Find cases where sizes differ significantly
    merged["size_diff"] = merged["bet_size_ens"] - merged["bet_size_ml"]
    big_size_diffs = merged[abs(merged["size_diff"]) > 10].sort_values("size_diff", ascending=False)
    
    print(f"\nTop 10 bets where Ensemble bet LARGER than ML:")
    print(f"{'City':<10} | {'Date':<10} | {'Won':<5} | {'ML Size':<8} | {'Ens Size':<8} | {'ML Prof':<8} | {'Ens Prof':<8} | {'Size Diff':<8}")
    for _, row in big_size_diffs.head(10).iterrows():
        print(f"{row['city_ml'][:10]:<10} | {row['target_date_ml']:<10} | {str(row['we_won_ml']):<5} | ${row['bet_size_ml']:<7.2f} | ${row['bet_size_ens']:<7.2f} | ${row['profit_ml']:<7.2f} | ${row['profit_ens']:<7.2f} | ${row['size_diff']:<7.2f}")
        
    print(f"\nTop 10 bets where ML bet LARGER than Ensemble:")
    print(f"{'City':<10} | {'Date':<10} | {'Won':<5} | {'ML Size':<8} | {'Ens Size':<8} | {'ML Prof':<8} | {'Ens Prof':<8} | {'Size Diff':<8}")
    for _, row in big_size_diffs.tail(10).iloc[::-1].iterrows():
        print(f"{row['city_ml'][:10]:<10} | {row['target_date_ml']:<10} | {str(row['we_won_ml']):<5} | ${row['bet_size_ml']:<7.2f} | ${row['bet_size_ens']:<7.2f} | ${row['profit_ml']:<7.2f} | ${row['profit_ens']:<7.2f} | ${abs(row['size_diff']):<7.2f}")

    # Bets made by Ensemble but NOT ML
    ens_only = ens_graded[~ens_graded["condition_id"].isin(ml_graded["condition_id"])]
    print("\n" + "="*50)
    print("BETS MADE BY ENSEMBLE BUT NOT ML")
    print("="*50)
    print(f"Total Ensemble-only bets: {len(ens_only)}")
    print(f"Total P&L of Ens-only:    ${ens_only['profit'].sum():+.2f}")
    print(f"Win Rate of Ens-only:     {ens_only['we_won'].mean():.1%}")
    print(ens_only[["city", "target_date", "bet_side", "we_won", "bet_size", "profit"]].head(15).to_string(index=False))

    # Bets made by ML but NOT Ensemble
    ml_only = ml_graded[~ml_graded["condition_id"].isin(ens_graded["condition_id"])]
    print("\n" + "="*50)
    print("BETS MADE BY ML BUT NOT ENSEMBLE")
    print("="*50)
    print(f"Total ML-only bets: {len(ml_only)}")
    print(f"Total P&L of ML-only:    ${ml_only['profit'].sum():+.2f}")
    print(f"Win Rate of ML-only:     {ml_only['we_won'].mean():.1%}")
    print(ml_only[["city", "target_date", "bet_side", "we_won", "bet_size", "profit"]].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
