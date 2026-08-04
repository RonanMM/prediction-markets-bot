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

from config import CITIES, ALL_CITIES, LOGS_DIR, PLOTS_DIR
from fetch_polymarket import fetch_weather_markets, fetch_price_history_for_market
from fetch_orderbook import fetch_book_summaries
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
# visualization (which imports matplotlib) is imported lazily inside step_generate_plots so that
# --collect-only / --summary-only runs (e.g. the 2-hourly collector) don't pay the plotting-stack
# import for zero plots (F12).


# ── Logging setup ─────────────────────────────────────────────────────────────

_KEEP_RUN_LOGS = 30  # the 2-hourly collector writes one per run; keep ~2.5 days of them


def _setup_logging(verbose: bool = False) -> None:
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"run_{ts}.log"

    # Prune old per-run logs so they can't accumulate again (they reached 123 files / 30 MB
    # before the 2026-07-13 cleanup). Newest _KEEP_RUN_LOGS are kept; failures are ignored —
    # logging must never block a collection run.
    try:
        old = sorted(Path(LOGS_DIR).glob("run_*.log"),
                     key=lambda p: p.stat().st_mtime, reverse=True)[_KEEP_RUN_LOGS:]
        for p in old:
            p.unlink()
    except OSError:
        pass

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

# Terminal (settled) Kalshi market status. An earlier draft of this step guessed
# {"finalized", "settled", "closed"} without checking; that guess was partly fiction. VERIFIED
# LIVE 2026-08-03 across all 7 capture-tier series (3,066 markets total via full pagination):
# Kalshi returns exactly two status values — "active" (84) and "finalized" (2982). "settled" and
# "closed" never appear, and neither does "initialized" (kalshi_series.LIVE_STATUSES allows for
# it, but no currently-enumerable market for these series is in that state). Candle backfill is
# the ONLY route to Kalshi's ~2-month history, so under-matching here silently and permanently
# loses a settled market's whole price history; using the real observed value instead of the
# unverified guess is the point.
_SETTLED_STATUSES = {"finalized"}


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _annotate_order_books(snapshots: list[dict]) -> None:
    """Attach top-of-book + depth to each snapshot, in place. Never raises.

    Order books are the most perishable data this project touches — a book at 14:00 is gone
    forever — and their absence is why `config.HALF_SPREAD` has been a hard-coded guess since it
    was written. Two batched CLOB requests cover a whole cycle (~0.2 s for 66 markets), so this
    is close to free.

    Best-effort ON PURPOSE: a CLOB outage must never cost us the market snapshots themselves,
    which are the irreplaceable part. Markets whose books do not return are simply left
    un-annotated and read as NaN — the honest value, because we do not know that book's state.
    """
    try:
        tokens = {}
        for s in snapshots:
            cid = s.get("condition_id")
            ids = s.get("clob_token_ids") or []
            if cid and ids:
                tokens[cid] = (ids[0] if len(ids) > 0 else None,
                               ids[1] if len(ids) > 1 else None)
        if not tokens:
            logger.warning("Order books: no clob_token_ids on any snapshot — skipping.")
            return
        books = fetch_book_summaries(tokens)
        for s in snapshots:
            s.update(books.get(s.get("condition_id"), {}))
        logger.info("Order books: annotated %d/%d snapshots.", len(books), len(snapshots))
    except Exception as exc:                       # noqa: BLE001 — must not break collection
        logger.warning("Order-book annotation failed (%s) — snapshots saved without it.", exc)


def _kalshi_rot_alarms(rows: list[dict], previous) -> list[str]:
    """Series that HAVE served markets before and serve none now — the ticker-rot signature.

    Kalshi renames series (HIGHNY -> KXHIGHNY) and leaves the old ticker enumerable but empty.
    A series that never served markets is not rot; nothing was lost.

    ⚠️ A TRUNCATED row is excluded, and that exclusion is load-bearing. `fetch_series_markets`
    returns `([], True)` when the transport fails, so a Kalshi OUTAGE produces exactly
    `markets_returned=0, truncated=True` for every series at once. Without this check an outage
    screams TICKER ROT for all seven — the alarm that exists to catch a real, rare, permanent
    data loss would fire routinely for a transient one, and an alarm that cries wolf is an alarm
    that gets ignored on the day it is right. The data already distinguishes the two cases;
    discarding that distinction is the bug. `truncated=True` with zero markets means WE DO NOT
    KNOW, which is never evidence of rot.
    """
    if previous is None or not len(previous):
        return []
    ever = set(previous.loc[previous["markets_returned"].astype(float) > 0, "series_ticker"])
    return [r["series_ticker"] for r in rows
            if r["series_ticker"] in ever
            and int(r["markets_returned"]) == 0
            and not r.get("truncated")]


def _archived_tickers(path) -> set | None:
    """Tickers already present in an existing Kalshi {city}_candles.csv.

    A finalized market's candle history is IMMUTABLE, so once it is archived it never needs
    re-fetching. Re-fetching all ~426 settled markets per city on EVERY hourly cycle is ~3,000
    wasted requests across 7 cities and triggers HTTP 429s — measured 2026-08-03: 0.33s/fetch,
    ~2.4 min/city, ~17 min for one full 7-city cycle if nothing is skipped, which does not fit
    an hourly collector sharing a 45-minute CI timeout with the Polymarket work. This is the
    read side of the skip.

    Absence is a NORMAL first run: no candles file yet -> empty set, so every settled market
    still gets backfilled once (the one-time ~17-minute cost that is actually wanted).

    A file that EXISTS but fails to parse is NOT the same as absence — it is corruption, and
    must never be silently read as "nothing archived": that would look exactly like a fresh
    backfill and either re-fetch everything (wasteful, rate-limited) or — worse, if some rows
    did parse — quietly duplicate what is already there. Returns None so the caller can tell the
    two cases apart and skip the whole candle phase for that city rather than guess.
    """
    import pandas as pd
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["ticker"], dtype=str)
    except Exception:
        return None
    return set(df["ticker"].dropna())


def _markets_needing_candles(markets: list[dict], archived: set) -> list[dict]:
    """Settled markets (see _SETTLED_STATUSES) whose candles are not already archived.

    Pure and separately testable from the network/disk side of the skip (_archived_tickers) —
    together they are the whole "skip already-archived tickers" guard.
    """
    return [m for m in markets
            if m.get("status") in _SETTLED_STATUSES and m.get("ticker") not in archived]


def step_fetch_kalshi() -> None:
    """Archive Kalshi markets, order books and candles for the capture-tier cities.

    Additive and best-effort: a Kalshi outage must never block the irreplaceable Polymarket
    snapshot, so every failure here is logged and swallowed by the caller.
    """
    import pandas as pd
    from kalshi_series import target_series
    from processing import save_kalshi_manifest, kalshi_manifest_path

    logger.info("═══ Step 1b: Kalshi archive ═══")
    now = datetime.now(timezone.utc).isoformat()
    try:
        previous = pd.read_csv(kalshi_manifest_path()) if kalshi_manifest_path().exists() else None
    except Exception:
        previous = None

    manifest = []
    try:
        for city, series in target_series().items():
            # PER-CITY ISOLATION. Without it, an exception on city 4 escapes the loop, main()
            # swallows it, and cities 1-3 end the cycle with their data written but NO health
            # record at all — the manifest write and the rot alarm below both live after the
            # loop. A partial cycle must still be a MEASURED partial cycle; the whole point of
            # the manifest is that it says what happened.
            try:
                _kalshi_one_city(city, series, now, manifest)
            except Exception as exc:                # noqa: BLE001 — one city must not kill six
                logger.error("kalshi %s (%s): FAILED mid-city (%s: %s) — the other cities "
                             "continue and the manifest still records this cycle.",
                             city, series, type(exc).__name__, exc)
    finally:
        # In a `finally` so the health record survives anything the loop can raise, including a
        # KeyboardInterrupt or a bug in the isolation above.
        save_kalshi_manifest(manifest)

    for rotted in _kalshi_rot_alarms(manifest, previous):
        logger.error("KALSHI TICKER ROT: %s served markets before and serves NONE now, and the "
                     "fetch was NOT truncated (so this is not an outage). Kalshi renames series "
                     "and leaves the old ticker enumerable but empty; find the replacement "
                     "ticker and update resolution_anchors — this alarm cannot do it for you.",
                     rotted)


def _kalshi_one_city(city: str, series: str, now: str, manifest: list) -> None:
    """Archive ONE city's markets, books and candles. Raises on failure; the caller isolates it.

    Split out of step_fetch_kalshi so the per-city try/except has a single, obvious unit to wrap
    (a bare `try` around a 40-line loop body invites a later edit to slip outside it). Appends
    this city's health row to `manifest` BEFORE doing any of the heavy work, so a mid-city
    failure still leaves a record of what the fetch returned.
    """
    from kalshi_series import manifest_row, LIVE_STATUSES
    from fetch_kalshi import (fetch_series_markets, summarize_market, count_live,
                              fetch_orderbooks, fetch_candles, summarize_candle, candle_log_row)
    from processing import (save_kalshi_rows, save_kalshi_candle_log, kalshi_candles_path)
    from resolution_anchors import slug

    cslug = slug(city)
    markets, truncated = fetch_series_markets(series)
    manifest.append(manifest_row(series, city, len(markets), count_live(markets),
                                 truncated, now))
    if not markets:
        return

    rows = [{**summarize_market(m), "fetched_at_utc": now, "city": city,
             "series_ticker": series} for m in markets]
    save_kalshi_rows("markets", cslug, rows, ["ticker", "fetched_at_utc"])

    # ONE definition of "live" (kalshi_series.LIVE_STATUSES), shared with fetch_kalshi.count_live.
    # This line used to hardcode ("active", "initialized") one call after count_live consulted the
    # constant, so the manifest's live_markets count and the set of books actually fetched could
    # silently disagree the moment either changed.
    live = [m["ticker"] for m in markets if m.get("status") in LIVE_STATUSES]
    books = fetch_orderbooks(live)
    if books:
        save_kalshi_rows("books", cslug,
                         [{"fetched_at_utc": now, "city": city, "ticker": t, **b}
                          for t, b in books.items()],
                         ["ticker", "fetched_at_utc"])

    # Candle backfill: settled markets only, and only those NOT already archived — a
    # finalized market's candles are immutable (see _archived_tickers for the measured cost
    # of not skipping). A corrupt/unreadable existing archive skips this city's candle phase
    # entirely rather than risk a mass re-fetch or silent duplication.
    archived = _archived_tickers(kalshi_candles_path(cslug))
    if archived is None:
        logger.error("kalshi candles %s: existing archive at %s is unreadable — skipping "
                     "the candle phase for this city rather than re-fetching or "
                     "duplicating; investigate before the next cycle.",
                     cslug, kalshi_candles_path(cslug))
        return

    log = []
    for m in _markets_needing_candles(markets, archived):
        candles, meta = fetch_candles(series, m)
        # Completeness is COMPUTED by fetch_candles and, until now, thrown away — the caller read
        # meta["ok"] and dropped the window, the count and the reason. Persisting every attempt
        # is what makes a zero-candle market distinguishable from one never attempted.
        log.append(candle_log_row(meta, city, series, now))
        if not meta["ok"]:
            logger.warning("kalshi candles %s: %s", meta["ticker"], meta["reason"])
            continue
        if candles:
            save_kalshi_rows("candles", cslug,
                             [summarize_candle(c, city, series, m["ticker"]) for c in candles],
                             ["ticker", "end_period_ts"])
    save_kalshi_candle_log(log)


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
        _annotate_order_books(all_snapshots)
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

    # Shoulder-premium PAPER book (edge megaplan §10b): record pre-day shoulder bins from the
    # snapshot that just landed. Pure price-structure paper tracking — no orders, no model.
    try:
        from shoulder_book import scan_and_record
        n = scan_and_record()
        if n:
            logger.info("Shoulder paper book: recorded %d new pre-day entries.", n)
    except Exception as e:
        logger.warning("Shoulder book scan failed: %s", e)

    # Breadth structure book (ALL Polymarket weather cities) — model-free, settlement-graded.
    # Standalone of the 5-city book/model; pulls the weather tag live from Gamma.
    try:
        from shoulder_book_breadth import scan_and_record_breadth
        nb = scan_and_record_breadth()
        if nb:
            logger.info("Breadth structure book: recorded %d new entries.", nb)
    except Exception as e:
        logger.warning("Breadth book scan failed: %s", e)


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
    from visualization import generate_all_plots, plot_efficiency_signal   # F12: lazy (matplotlib)
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
        "--cities", nargs="+", default=list(ALL_CITIES.keys()),
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
    unknown = [c for c in cities if c not in ALL_CITIES]
    if unknown:
        logger.warning("Unknown cities (will be skipped): %s", unknown)
        cities = [c for c in cities if c in ALL_CITIES]

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
        try:
            step_fetch_kalshi()
        except Exception as exc:            # noqa: BLE001 — Kalshi must never block collection
            logger.warning("Kalshi archive failed (%s) — Polymarket collection unaffected.", exc)

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
