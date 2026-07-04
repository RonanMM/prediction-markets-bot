"""
main.py — Polymarket Weather Tracker
=====================================
Orchestrates:
  1. Fetch Polymarket temperature markets  (fetch_polymarket.py)
  2. Fetch weather forecasts               (fetch_weather.py)
  3. Normalize + store                     (processing.py)
  4. Generate visualizations               (visualization.py)
  5. Print city summaries

Run daily:
    python main.py

Options:
    python main.py --cities Seoul London     # subset of cities
    python main.py --skip-polymarket         # weather only
    python main.py --skip-weather            # market data only
    python main.py --plots-only              # re-plot from stored data
    python main.py --summary-only            # print summary, no fetching
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import CITIES, LOGS_DIR, PLOTS_DIR
from fetch_polymarket import fetch_weather_markets, fetch_price_history_for_market
from fetch_weather import fetch_forecast, fetch_forecast_multimodel
from fetch_ensemble import fetch_ensemble
from processing import (
    save_market_snapshots,
    save_price_history,
    save_weather_forecast,
    save_ensemble_forecast,
    save_multimodel_forecast,
    compute_city_summary,
)
from visualization import generate_all_plots, plot_efficiency_signal


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"run_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_fetch_polymarket(cities: list[str]) -> None:
    logger.info("═══ Step 1: Fetching Polymarket temperature markets ═══")
    all_snapshots   = []
    all_ph_records  = []

    for city in cities:
        logger.info("── City: %s", city)
        snaps = fetch_weather_markets(city)
        if not snaps:
            logger.warning("No Polymarket markets found for %s.", city)
            continue

        all_snapshots.extend(snaps)

        # Fetch CLOB price history for each market
        for snap in snaps:
            ph = fetch_price_history_for_market(snap, interval="1d")
            all_ph_records.extend(ph)

    if all_snapshots:
        save_market_snapshots(all_snapshots)
        logger.info("Saved %d market snapshots.", len(all_snapshots))
    if all_ph_records:
        save_price_history(all_ph_records)
        logger.info("Saved %d CLOB price-history records.", len(all_ph_records))


def step_fetch_weather(cities: list[str]) -> None:
    logger.info("═══ Step 2: Fetching weather forecasts ═══")
    for city in cities:
        logger.info("── City: %s", city)
        if city not in CITIES:
            logger.warning("Unknown city '%s' — skipping.", city)
            continue
        forecast = fetch_forecast(city)
        if forecast:
            save_weather_forecast(forecast)
            daily_count = len(forecast.get("daily", []))
            logger.info("Saved forecast: %d daily rows for %s.", daily_count, city)
        else:
            logger.error("Failed to fetch forecast for %s.", city)

        mm = fetch_forecast_multimodel(city)
        if mm:
            save_multimodel_forecast(mm)
            logger.info("Saved multi-model forecast: %d daily rows for %s.",
                        len(mm.get("daily", [])), city)
        else:
            logger.warning("No multi-model forecast for %s.", city)

    # Hourly station obs top-up (last 3 days) — same-day bets condition on the
    # running observed max, so the obs file must be fresh at analysis time.
    try:
        from fetch_station_obs import fetch_station_obs
        fetch_station_obs(recent_only=True)
    except Exception as e:
        logger.warning("Station obs top-up failed: %s", e)

    # NBM station guidance top-up (US cities) — runtime-stamped, as-of joined.
    try:
        from fetch_nbm import fetch_nbm
        fetch_nbm(recent_only=True)
    except Exception as e:
        logger.warning("NBM top-up failed: %s", e)


def step_fetch_ensemble(cities: list[str]) -> None:
    logger.info("═══ Step 2b: Fetching ensemble forecasts ═══")
    for city in cities:
        logger.info("── Ensemble: %s", city)
        if city not in CITIES:
            continue
        result = fetch_ensemble(city)
        if result:
            save_ensemble_forecast(result)
            n = len(result.get("daily", []))
            logger.info("Saved ensemble: %d days for %s.", n, city)
        else:
            logger.warning("No ensemble data for %s (API may be unavailable).", city)


def step_generate_plots(cities: list[str], summaries: list[dict]) -> None:
    logger.info("═══ Step 3: Generating visualizations ═══")
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    for city in cities:
        logger.info("── Plotting: %s", city)
        paths = generate_all_plots(city)
        for p in paths:
            logger.info("  ✓ %s", p)

    # Efficiency signal (all cities)
    valid_summaries = [s for s in summaries if "difference_c" in s]
    if valid_summaries:
        p = plot_efficiency_signal(valid_summaries)
        if p:
            logger.info("  ✓ %s", p)


def step_print_summary(cities: list[str]) -> list[dict]:
    logger.info("═══ Summary ═══")
    summaries = []
    sep = "─" * 52

    print(f"\n{'═'*52}")
    print(f"  Polymarket Weather Tracker — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*52}")

    for city in cities:
        s = compute_city_summary(city)
        summaries.append(s)

        print(f"\n  City: {s['city']}")
        print(sep)

        q = s.get("market_question", "")
        if q:
            print(f"  Market : {q[:60]}{'…' if len(q)>60 else ''}")
            vol = s.get("market_volume_usdc")
            if vol is not None:
                print(f"  Volume : ${float(vol):,.0f} USDC")

        impl = s.get("market_implied_temp_c")
        if impl is not None:
            print(f"  Market implied temp : {impl:+.1f}°C")
        else:
            print("  Market implied temp : N/A (no parseable temperature buckets)")

        fc = s.get("forecast_temp_max_c")
        fd = s.get("forecast_date_local", "")
        if fc is not None:
            print(f"  Forecast max temp   : {fc:+.1f}°C  ({fd})")
        else:
            print("  Forecast max temp   : N/A")

        diff = s.get("difference_c")
        if diff is not None:
            arrow = "▲" if diff > 0 else "▼" if diff < 0 else "≈"
            color_tag = "(market ABOVE forecast)" if diff > 0 else "(market BELOW forecast)"
            print(f"  Difference          : {diff:+.1f}°C  {arrow} {color_tag}")

    print(f"\n{'═'*52}\n")
    return summaries


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Polymarket Weather Tracker — daily fetch + visualize"
    )
    p.add_argument(
        "--cities", nargs="+", default=list(CITIES.keys()),
        help="City names to process (default: all configured cities)"
    )
    p.add_argument("--skip-polymarket", action="store_true",
                   help="Skip Polymarket data fetching")
    p.add_argument("--skip-weather", action="store_true",
                   help="Skip weather forecast fetching")
    p.add_argument("--plots-only", action="store_true",
                   help="Re-generate plots from stored data only")
    p.add_argument("--summary-only", action="store_true",
                   help="Print summary only (no fetching, no plots)")
    p.add_argument("--collect-only", action="store_true",
                   help="Fetch + append data only (no summary, no plots) — for "
                        "scheduled collection (launchd/CI) where plot churn is waste")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Debug-level logging")
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    cities = args.cities
    # Validate
    unknown = [c for c in cities if c not in CITIES]
    if unknown:
        logger.warning("Unknown cities (will be skipped): %s", unknown)
        cities = [c for c in cities if c in CITIES]

    if not cities:
        logger.error("No valid cities to process. Exiting.")
        sys.exit(1)

    logger.info("Polymarket Weather Tracker — cities: %s", cities)

    if args.summary_only:
        step_print_summary(cities)
        return

    if args.plots_only:
        summaries = step_print_summary(cities)
        step_generate_plots(cities, summaries)
        return

    # Full pipeline
    if not args.skip_polymarket:
        step_fetch_polymarket(cities)

    if not args.skip_weather:
        step_fetch_weather(cities)
        step_fetch_ensemble(cities)

    if args.collect_only:
        logger.info("Collect-only run complete.")
        return

    summaries = step_print_summary(cities)
    step_generate_plots(cities, summaries)

    logger.info("Run complete. Plots in ./%s/", PLOTS_DIR)


if __name__ == "__main__":
    main()
