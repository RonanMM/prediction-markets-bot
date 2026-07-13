"""backfill_schema.py — repair the append-freeze corruption (megaplan B2).

The append-freeze bug (processing._append_csv reindexing new rows to the existing header)
silently dropped, on every append, the columns fetchers started emitting after each CSV
first existed: the gem/mf/aifs blend models, all tmin_*, and ens_min_mean/ens_min_std.
This tool widens each {slug}_daily_mm.csv / {slug}_ensemble.csv so those columns are
present (historical rows NA-backfilled), then re-fetches recent history (past_days) so the
newest runs land with the full schema.

Run from src/polymarket_weather/ AFTER the B1 union-append fix, with the 2-hourly CI
collector PAUSED so the atomic rewrite can't race a scheduled append:

    python backfill_schema.py                 # widen + re-fetch past_days=7
    python backfill_schema.py --widen-only    # offline: widen headers only, no network
    python backfill_schema.py --past-days 30  # widen + re-fetch 30 recent days

HONESTY LIMIT: the per-snapshot *sequence* of forecast revisions for the dropped columns
over the corrupted window is permanently lost. past_days reconstructs only the latest run
per recent date (stamped today) — good for near-future serving, NOT a reconstruction of
per-snapshot history. Backtest rows in the corrupted window stay proxy-served because the
as-of join uses fetched_at <= fetch_time and the re-fetched runs are stamped now.
"""
import argparse
import logging

from config import CITIES
from fetch_weather import MULTIMODEL_MODELS, fetch_forecast_multimodel
from fetch_ensemble import fetch_ensemble
import processing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_schema")

_MM_BASE = ["city", "fetched_at_utc", "date_local", "date_utc"]
# Only the columns the freeze could have dropped need forcing; ensure_schema preserves
# every existing column and appends just the missing ones.
_EXPECTED_ENS_ADDITIONS = ["ens_min_mean", "ens_min_std"]


def _expected_mm_cols() -> list[str]:
    cols = list(_MM_BASE)
    for short in MULTIMODEL_MODELS.values():
        cols.append(f"tmax_{short}")
        cols.append(f"tmin_{short}")
    return cols


def widen_all() -> None:
    mm_cols = _expected_mm_cols()
    for city in CITIES:
        if processing.ensure_schema(processing._weather_daily_mm_path(city), mm_cols):
            logger.info("widened %s", processing._weather_daily_mm_path(city).name)
        if processing.ensure_schema(processing._ensemble_path(city), _EXPECTED_ENS_ADDITIONS):
            logger.info("widened %s", processing._ensemble_path(city).name)


def refetch(past_days: int) -> None:
    for city in CITIES:
        mm = fetch_forecast_multimodel(city, past_days=past_days)
        if mm:
            processing.save_multimodel_forecast(mm)
        ens = fetch_ensemble(city, past_days=past_days)
        if ens:
            processing.save_ensemble_forecast(ens)


def repair_price_history_timestamps() -> None:
    """Repair {slug}_price_history.csv timestamps corrupted by the seconds-as-ms bug.

    fetch_polymarket used to divide the CLOB epoch (already SECONDS) by 1000, so every stored
    timestamp collapsed onto 1970-01-21. The corruption is exactly reversible: the stored epoch
    is real_epoch/1000, so multiply back by 1000. Only rows before 2000-01-01 are touched;
    correctly-stamped rows (written after the fetcher fix) pass through unchanged. Offline op.
    """
    import pandas as pd
    for city in CITIES:
        path = processing._mkt_history_path(city)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "timestamp_utc" not in df.columns or df.empty:
            continue
        ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True, format="ISO8601")
        bad = ts.notna() & (ts < pd.Timestamp("2000-01-01", tz="UTC"))
        if not bad.any():
            continue
        epoch_s = (ts[bad] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        fixed = pd.to_datetime(epoch_s * 1000.0, unit="s", utc=True)
        df.loc[bad, "timestamp_utc"] = fixed.map(lambda t: t.isoformat())
        processing._atomic_write_csv(path, df)
        logger.info("repaired %d/%d timestamps in %s (span now %s → %s)",
                    int(bad.sum()), len(df), path.name,
                    fixed.min().date(), fixed.max().date())


def verify() -> bool:
    import pandas as pd
    ok = True
    need_mm = {"tmax_gem", "tmax_mf", "tmax_aifs", "tmin_ecmwf"}
    for city in CITIES:
        mm_path = processing._weather_daily_mm_path(city)
        if mm_path.exists():
            missing = need_mm - set(pd.read_csv(mm_path, nrows=0).columns)
            if missing:
                ok = False
                logger.error("%s still missing %s", mm_path.name, missing)
        ens_path = processing._ensemble_path(city)
        if ens_path.exists():
            missing = {"ens_min_mean", "ens_min_std"} - set(pd.read_csv(ens_path, nrows=0).columns)
            if missing:
                ok = False
                logger.error("%s still missing %s", ens_path.name, missing)
    logger.info("verify: %s", "OK — full schema present" if ok else "INCOMPLETE (see errors)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair append-freeze schema corruption (megaplan B2).")
    ap.add_argument("--widen-only", action="store_true", help="widen headers offline; no network")
    ap.add_argument("--past-days", type=int, default=7, help="recent days to re-fetch (0-92)")
    args = ap.parse_args()

    logger.info("Repairing price-history timestamps (seconds-as-ms bug) …")
    repair_price_history_timestamps()
    logger.info("Widening daily_mm / ensemble CSV headers …")
    widen_all()
    if args.widen_only:
        verify()
        return
    logger.info("Re-fetching recent history (past_days=%d) …", args.past_days)
    refetch(args.past_days)
    verify()


if __name__ == "__main__":
    main()
