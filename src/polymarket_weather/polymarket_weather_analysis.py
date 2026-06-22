"""
polymarket_weather_v3.py
========================
Polymarket Weather Inefficiency Analyzer — production-grade.

Alpha signals implemented (all usable with your existing data):
  α1  Forecast momentum       — EMA of Δforecast over recent snapshots
  α2  Min/max spread proxy    — diurnal range as convective uncertainty signal
  α3  Student-t tail model    — heavier tails than Gaussian (nu calibrated per horizon)
  α4  Constrained PMF         — full probability-mass-conserving reconstruction
                                 using exact + gte + lte bins jointly
  α5  Internal consistency    — detect markets whose bins don't sum to ~1.0;
                                 trade the missing-mass bins
  α6  Volume recency          — 24h/total volume ratio flags informed-trader activity
  α7  Forecast convergence    — cross-snapshot variance; high variance → market stale
  α8  Market update lag       — time since last significant market reprice
  α9  Correlated bet grouping — per-(city,date) exposure cap; Kelly across group

Usage:
  python polymarket_weather_v3.py --data_dir ./data [options]
  python polymarket_weather_v3.py --data_dir ./data --min_edge 0.06 --bankroll 2000
"""

from __future__ import annotations
import argparse, warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from models import Opportunity
from config import MIN_EDGE, MIN_LIQUIDITY, FEE_RATE, MAX_TOTAL_KELLY, KELLY_FRACTION
from reports import plot_pmf_comparison, plot_alpha_dashboard, plot_forecast_drift_all, plot_momentum_heatmap, print_report, opps_to_df
from engine import analyse_city, WeatherBettingBot
from data_loader import discover_cities

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Weather Analyzer v4 — with live-mode betting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--cities",     nargs="+", default=None)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--min_edge",   type=float, default=MIN_EDGE)
    parser.add_argument("--min_liq",    type=float, default=MIN_LIQUIDITY)
    parser.add_argument("--min_days",   type=float, default=0.0,
                        help="Skip markets resolving in fewer than N days from now "
                             "(e.g. --min_days 1 excludes today's markets)")
    parser.add_argument("--bankroll",   type=float, default=1000.0)
    parser.add_argument("--no_plots",   action="store_true")
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: recalculate days_ahead from NOW and re-verify market "
             "prices via the Gamma API before recommending any bet. Always use "
             "this for real-money execution.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually place orders (requires POLYMARKET_PRIVATE_KEY env var). "
             "Implies --live. Use with extreme caution.",
    )
    parser.add_argument("--disable_ml", action="store_true", help="Disable ML models and fall back to ensemble/student-t")
    parser.add_argument("--kelly_fraction", type=float, default=KELLY_FRACTION, help="Kelly fraction multiplier for sizing bets")
    parser.add_argument("--sigma_threshold", type=float, default=1.2, help="Ensemble sigma threshold above which we bypass ML calibrator")
    parser.add_argument("--disable_conflict_gating", action="store_true", help="Disable conflict gating check between ML and Ensemble predictions")
    args = parser.parse_args()

    dry_run   = not args.execute
    live_mode = args.live or args.execute

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = args.cities or discover_cities(data_dir)
    if not cities:
        print(f"No *_snapshots.csv files found in {data_dir}")
        return

    print(f"\n{'═'*72}")
    print("  Polymarket Weather Analyzer v4")
    print(f"  Cities   : {cities}")
    print(f"  Min edge : {args.min_edge:.0%}  Min liq: ${args.min_liq:.0f}  "
          f"Min days: {args.min_days:.0f}  Bankroll: ${args.bankroll:,.0f}")
    print(f"  Fee      : {FEE_RATE:.0%}  Max portfolio: {MAX_TOTAL_KELLY:.0%}")
    print(f"  Mode     : {'LIVE (price-verified)' if live_mode else 'BACKTEST'} / "
          f"{'EXECUTE' if not dry_run else 'DRY RUN'}")
    print(f"{'═'*72}\n")

    all_opps: list[Opportunity] = []
    for city in cities:
        print(f"── {city.upper()} ──")
        opps = analyse_city(data_dir, city,
                            min_edge=args.min_edge,
                            min_liq=args.min_liq,
                            use_ml=not args.disable_ml,
                            kelly_fraction=args.kelly_fraction,
                            sigma_threshold=args.sigma_threshold,
                            conflict_gating=not args.disable_conflict_gating)
        if opps:
            print(f"  → {len(opps)} opportunities found")
        all_opps.extend(opps)

    opps_df = opps_to_df(all_opps)

    if not args.no_plots:
        if not opps_df.empty:
            all_bins_records = []
            for o in all_opps:
                all_bins_records.append({
                    "city":         o.city,
                    "target_date":  o.target_date,
                    "fetched_at":   o.fetched_at,
                    "bin_temp_c":   o.bin_temp_c,   # actual question temperature
                    "market_prob":  o.market_prob,
                    "forecast_prob":o.forecast_prob,
                    "forecast_mu":  o.forecast_mu,
                    "days_ahead":   o.days_ahead,
                    "ema_momentum": o.ema_momentum,
                })
            all_bins_df = pd.DataFrame(all_bins_records)
            plot_pmf_comparison(opps_df, all_bins_df, output_dir)
            plot_alpha_dashboard(opps_df, output_dir)
            plot_momentum_heatmap(opps_df, output_dir)

        plot_forecast_drift_all(data_dir, cities, output_dir)

    # Only show future markets in the terminal report to avoid confusion
    today_str = datetime.now(timezone.utc).date().isoformat()
    print_report(opps_df[opps_df["target_date"] >= today_str] if not opps_df.empty else opps_df)

    if not opps_df.empty:
        # Standard save for backward compatibility
        opps_df.to_csv(output_dir / "opportunities_v4.csv", index=False)
        
        # Save to evaluation CSVs for the 10-day test (overwrites each time with full history)
        eval_filename = "opportunities_evaluation_ensemble.csv" if args.disable_ml else "opportunities_evaluation_ml.csv"
        eval_path = output_dir / eval_filename
        opps_df.to_csv(eval_path, index=False)
        
        print(f"\n  Saved: {output_dir}/opportunities_v4.csv")
        print(f"  Saved evaluation tracker: {eval_path}")

    bot = WeatherBettingBot(
        bankroll=args.bankroll, dry_run=dry_run, live_mode=live_mode,
        kelly_fraction=args.kelly_fraction
    )
    bot.run(all_opps, min_edge=args.min_edge, min_days=args.min_days)

    print(f"\nDone → {output_dir.resolve()}")


if __name__ == "__main__":
    main()