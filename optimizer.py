import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from scipy.stats import t as student_t
import itertools
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

WEATHER_CACHE = {}

from grading import resolves_yes  # resolution-station truth grader, native-unit rounding

# D6/D7: price with the engine's censored distribution and settle with honest costs, instead of
# the forked floor/ceiling-blind _cdf/_bin_prob and the retired (payout-size)*0.98 settlement that
# tuned parameters under economics production never uses (the grid-graded artifact).
from backtest_common import _bin_prob, settle_bet

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
        f_prob_ens = _bin_prob(bin_temp, raw_mu, row["forecast_sigma"], raw_nu,
                               floor=row.get("forecast_floor"), ceiling=row.get("forecast_ceiling"))
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
        
        resolved_yes = resolves_yes(city, target_date, question, bin_temp)
        if resolved_yes is None:
            continue

        we_won = (bet_side == "Yes" and resolved_yes) or (bet_side == "No" and not resolved_yes)
        
        bet_size = 1000.0 * row["kelly"]
        if bet_size < 1.0:
            continue
            
        # Honest settlement (D7): cross config.HALF_SPREAD on entry and pay config.FEE_RATE on
        # the winning payout — shared with evaluate_oos / historical_backtester via settle_bet.
        profit = settle_bet(their_prob, we_won, bet_size)
        if we_won:
            wins += 1
            
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
    csv_path = Path("src/polymarket_weather/output/opportunities_evaluation_calibrated.csv")
    if not csv_path.exists():
        print(f"Data not found at {csv_path}")
        return
        
    df_raw = pd.read_csv(csv_path)
    
    # Search space. Coordinate ascent carries every accepted improvement forward, so it can
    # COMBINE changes — unlike the old one-change-per-tier sweep, which never could.
    GRID = {
        "MIN_EDGE":            [0.04, 0.06, 0.08, 0.10],
        "KELLY_FRACTION":      [0.15, 0.25, 0.35, 0.50],
        "MAX_KELLY_PER_BET":   [0.08, 0.12, 0.20],
        "sigma_threshold":     [1.0, 1.2, 1.5],
        "conflict_gating":     [True, False],
        "MAX_KELLY_PER_GROUP": [0.15, 0.20, 0.30],
        "MAX_TOTAL_KELLY":     [0.30, 0.40, 0.60],
        "MIN_LIQUIDITY":       [200, 400, 1000],
        "min_days":            [0.0, 0.5, 1.0],
    }
    # Seed = current production config — we only move off it if something genuinely beats it.
    SEED = {
        "MIN_EDGE": 0.06, "KELLY_FRACTION": 0.50, "MAX_KELLY_PER_BET": 0.08,
        "MAX_KELLY_PER_GROUP": 0.20, "MAX_TOTAL_KELLY": 0.40, "sigma_threshold": 1.2,
        "conflict_gating": True, "MIN_LIQUIDITY": 1000, "min_days": 0.5,
    }

    def coord_ascent(df, seed):
        best = dict(seed)
        best_roi = simulate_parameters(df, "combined_filter", best)["roi"]
        for _ in range(6):                       # repeat passes until no single change helps
            improved = False
            for key, values in GRID.items():
                for v in values:
                    if v == best.get(key):
                        continue
                    trial = dict(best); trial[key] = v
                    roi = simulate_parameters(df, "combined_filter", trial)["roi"]
                    if roi > best_roi + 1e-9:
                        best, best_roi, improved = trial, roi, True
            if not improved:
                break
        return best, best_roi

    # ---- Out-of-sample, time-ordered: tune on train, JUDGE on held-out validation ----
    df_sorted = df_raw.sort_values("target_date")
    dates = sorted(df_sorted["target_date"].unique())
    cut = dates[int(len(dates) * 0.6)] if len(dates) > 2 else None
    train = df_sorted[df_sorted["target_date"] <= cut] if cut else df_sorted
    valid = df_sorted[df_sorted["target_date"] > cut] if cut else df_sorted.iloc[0:0]

    tuned, train_roi = coord_ascent(train, SEED)
    seed_va  = simulate_parameters(valid, "combined_filter", SEED)
    tuned_va = simulate_parameters(valid, "combined_filter", tuned)

    print("\n================ OPTIMIZER (coordinate ascent, out-of-sample) ================")
    print(f"Eval data: {len(df_raw)} rows, {df_raw['condition_id'].nunique()} markets, "
          f"{dates[0]}..{dates[-1]}   train<= {cut}   validate> {cut}")
    print(f"  SEED  (current config):  train ROI {simulate_parameters(train,'combined_filter',SEED)['roi']:.1%}"
          f"   held-out ROI {seed_va['roi']:.1%} ({seed_va['bets']} bets)")
    print(f"  TUNED (in-sample best):  train ROI {train_roi:.1%}"
          f"   held-out ROI {tuned_va['roi']:.1%} ({tuned_va['bets']} bets)")
    print(f"  Tuned params: {tuned}")

    # Guardrail: only endorse a change that beats seed OUT-OF-SAMPLE on a non-tiny sample.
    MIN_VALID_BETS = 30
    if valid.empty:
        verdict = "INSUFFICIENT DATA — no validation split; keep current config."
    elif tuned_va["bets"] < MIN_VALID_BETS:
        verdict = (f"NOT ROBUST — only {tuned_va['bets']} held-out bets (<{MIN_VALID_BETS}); "
                   "keep current config until more data is available.")
    elif tuned_va["roi"] <= seed_va["roi"] + 1e-9:
        verdict = "NO out-of-sample improvement over current config; keep config."
    else:
        verdict = (f"Tuned beats seed out-of-sample by +{(tuned_va['roi']-seed_va['roi'])*100:.1f} pp — "
                   "candidate, but confirm on more data before committing.")
    print(f"\n  VERDICT: {verdict}")
    print("  (Reported only — this script does NOT modify config.py or CLAUDE.md.)")
    print("==============================================================================")

if __name__ == "__main__":
    main()
