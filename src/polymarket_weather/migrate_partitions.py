"""migrate_partitions.py — split this repo's append-only archives into DAILY files.

WHY. Git stores a whole new blob for every VERSION of a file, and the on-disk cost of that blob
scales with the file's SIZE, not with how little changed. Measured on the real archive
(2026-09-03), one snapshot commit cost **1.3-8.9 MB per city** for ~98 KB of genuinely new rows:

    san_francisco   20 versions: logical 1704 MB -> disk 178.2 MB   (10.5%)
    atlanta         20 versions: logical 1690 MB -> disk 108.5 MB   ( 6.4%)
    los_angeles     20 versions: logical 1702 MB -> disk  25.6 MB   ( 1.5%)

Same data, same shape, an 8x spread — delta-chain depth and pack fragmentation, which we do not
control. At 56 commits/day that was **205-354 MB/day**: the repo reached **12.12 GB, of which
10.36 GB was the preceding 30 days alone**, and on 2026-08-27 GitHub began rate-limiting the
repo's scheduled dispatch to roughly one run per workflow per 4-5 hours — which cost ~75% of the
hourly market snapshots for eight days before anyone noticed.

The same arithmetic applies to every hourly-rewritten archive here, and after the Kalshi split
those were the DOMINANT remaining source of growth (~80 MB/day):

    weather/*_hourly           1,540 MB across 3,127 versions
    polymarket/*_price_history 1,318 MB across 6,114 versions
    polymarket/*_snapshots       741 MB across 6,195 versions

WHAT THIS DOES. Rewrites each archive in `_DATASETS` as one file per UTC day,
`{stem}_{YYYY-MM-DD}.csv`. The rewritten unit drops from tens of MB and growing to ~1 MB, and
**a finished day is never rewritten again** — so history stops growing with the square of the
archive. It also retires GitHub's 100 MB per-file wall permanently: a daily partition cannot
approach it whatever the column set, which is the third time that wall has cost this project
data.

NOTHING IS DROPPED. This is a re-partition, not a curation — the same rows, in more files. Read
these archives back with `processing.load_partitioned()` (or `fetch_kalshi.load_markets()`, which
adds the dimension join). Both glob the partitions AND the legacy file, so a half-migrated archive
still reads whole. Test for presence with `partitioned_available()`, never `path.exists()` — the
legacy name is deleted, and an `exists()` guard then returns empty forever, silently.

SAFETY. Nothing is deleted until the split round-trips: the partitions are re-read and compared
against the source as multisets over every column, and the source is removed only if they match
exactly. A file whose rows cannot all be dated is refused and left untouched — guessing a day
would misdate a snapshot and dropping the row would lose perishable data, and no vendor re-serves
a price from three weeks ago at any price.

    python migrate_partitions.py                          # report only, writes nothing
    python migrate_partitions.py --apply                  # split every dataset
    python migrate_partitions.py --dataset weather --apply
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent / "data"
_DIR = _ROOT / "kalshi"

# Which append-only archives get partitioned, and the column that dates each ROW.
#
# The day always comes from the row's own timestamp. Price history is the one that is NOT a fetch
# time: its rows carry the VENUE's `timestamp_utc`, and `fetch_price_history` pulls interval="1d",
# so a cycle touches ~2 partitions rather than rewriting the whole file.
#
# Deliberately NOT partitioned: `*_daily.csv`, `*_ensemble.csv`, `*_daily_mm.csv` (129 MB of pack
# between them — the churn is not worth the extra file count), and everything gitignored
# (`*_historical_*`, `*_obs_*`, `*_nbm`), which costs the repo nothing because it is refetchable.
_DATASETS = {
    "kalshi":     ("*_markets.csv",       "fetched_at_utc"),
    "weather":    ("*_hourly.csv",        "fetched_at_utc"),
    "polymarket": ("*_snapshots.csv",     "fetched_at_utc"),
    "polymarket:history": ("*_price_history.csv", "timestamp_utc"),
}


def _days(df: pd.DataFrame, day_col: str = "fetched_at_utc") -> pd.Series:
    """UTC calendar day per row, from the row's OWN timestamp.

    Never `datetime.now()`: a cycle that starts at 23:59 and writes at 00:01 would file its rows
    under the wrong day, and a backfill would scatter historical rows into today.
    """
    ts = pd.to_datetime(df[day_col], utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d")


def _sorted_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ordering so the comparison is about CONTENT, not row order."""
    return (df.fillna("").astype(str)
              .sort_values(list(df.columns), kind="mergesort")
              .reset_index(drop=True))


def migrate(path: Path, apply: bool, day_col: str = "fetched_at_utc") -> dict:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    before = path.stat().st_size

    if day_col not in df.columns:
        return {"file": path.name, "FAILED": f"no {day_col} column — cannot partition by day"}

    day = _days(df, day_col)
    undated = int(day.isna().sum())
    if undated:
        # Refuse the whole file. A snapshot with no usable timestamp cannot be filed, and both
        # available guesses (invent a day / drop the row) silently corrupt a perishable archive.
        return {"file": path.name,
                "FAILED": f"{undated} row(s) have no parseable {day_col} — left untouched"}

    groups = list(df.groupby(day, sort=True))
    if not apply:
        return {"file": path.name, "rows": len(df), "days": len(groups),
                "before_mb": before / 1e6, "est_part_mb": before / 1e6 / max(len(groups), 1)}

    # Write every partition to a temp name first: nothing is visible, and nothing is deleted,
    # until the whole split has been verified.
    tmps: list[tuple[Path, Path]] = []
    for d, chunk in groups:
        final = path.with_name(f"{path.name[:-len('.csv')]}_{d}.csv")
        tmp = final.with_suffix(".csv.tmp")
        chunk.drop(columns=[]).to_csv(tmp, index=False)
        tmps.append((tmp, final))

    def _abort(msg: str) -> dict:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)
        return {"file": path.name, "FAILED": msg}

    rt = pd.concat([pd.read_csv(t, dtype=str, low_memory=False) for t, _ in tmps],
                   ignore_index=True)
    if len(rt) != len(df):
        return _abort(f"row count {len(rt)} != {len(df)}")
    if list(rt.columns) != list(df.columns):
        return _abort(f"column set changed: {list(rt.columns)} != {list(df.columns)}")
    if not _sorted_for_compare(rt).equals(_sorted_for_compare(df)):
        return _abort("partitions did not round-trip against the source")

    for tmp, final in tmps:
        tmp.replace(final)
    path.unlink()

    part_bytes = sum(f.stat().st_size for _, f in tmps)
    return {"file": path.name, "rows": len(df), "days": len(tmps),
            "before_mb": before / 1e6, "after_mb": part_bytes / 1e6,
            "largest_mb": max(f.stat().st_size for _, f in tmps) / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the split (default: report only)")
    ap.add_argument("--dataset", default="all", choices=sorted(_DATASETS) + ["all"],
                    help="which append-only archive to partition (default: all)")
    args = ap.parse_args()

    wanted = sorted(_DATASETS) if args.dataset == "all" else [args.dataset]
    # `*_hourly.csv` also matches `{slug}_obs_hourly.csv` — a DIFFERENT dataset (station METARs),
    # gitignored because it is refetchable, and carrying no fetched_at_utc at all. Partitioning it
    # would cost the repo nothing and it would only fail noisily, so it is excluded by name.
    files = [(f, day_col) for name in wanted
             for pattern, day_col in [_DATASETS[name]]
             for f in sorted((_ROOT / name.split(":")[0]).glob(pattern))
             if "_obs_" not in f.name]
    if not files:
        print("no un-partitioned archives left — nothing to do")
        return 0

    failed = []
    for p, day_col in files:
        r = migrate(p, args.apply, day_col=day_col)
        if "FAILED" in r:
            failed.append(r)
        print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()))

    if failed:
        print(f"\n{len(failed)} FILE(S) FAILED VERIFICATION and were left untouched:")
        for r in failed:
            print(f"  {r['file']}: {r['FAILED']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
