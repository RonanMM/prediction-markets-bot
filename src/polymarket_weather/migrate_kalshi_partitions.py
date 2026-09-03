"""migrate_kalshi_partitions.py — one-time split of the Kalshi fact table into DAILY files.

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

WHAT THIS DOES. Rewrites `{slug}_markets.csv` as one file per UTC day,
`{slug}_markets_{YYYY-MM-DD}.csv`. The rewritten unit drops from ~40 MB and growing to ~0.4 MB,
and **a finished day is never rewritten again** — so history stops growing with the square of the
archive. It also retires GitHub's 100 MB per-file wall permanently: a daily partition cannot
approach it whatever the column set, which is the third time that wall has cost this project
data.

NOTHING IS DROPPED. This is a re-partition, not a curation — the same rows, in more files. Read
the archive back with `fetch_kalshi.load_markets()`, which globs the partitions (and the legacy
file, so a half-migrated archive still reads whole).

SAFETY. Nothing is deleted until the split round-trips: the partitions are re-read and compared
against the source as multisets over every column, and the source is removed only if they match
exactly. A file whose rows cannot all be dated is refused and left untouched — guessing a day
would misdate a snapshot and dropping the row would lose perishable data, and no vendor re-serves
a price from three weeks ago at any price.

    python migrate_kalshi_partitions.py            # report only, writes nothing
    python migrate_kalshi_partitions.py --apply    # perform the split
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).resolve().parent / "data" / "kalshi"


def _days(df: pd.DataFrame) -> pd.Series:
    """UTC calendar day per row, from the row's OWN timestamp.

    Never `datetime.now()`: a cycle that starts at 23:59 and writes at 00:01 would file its rows
    under the wrong day, and a backfill would scatter historical rows into today.
    """
    ts = pd.to_datetime(df["fetched_at_utc"], utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d")


def _sorted_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ordering so the comparison is about CONTENT, not row order."""
    return (df.fillna("").astype(str)
              .sort_values(list(df.columns), kind="mergesort")
              .reset_index(drop=True))


def migrate(path: Path, apply: bool) -> dict:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    before = path.stat().st_size

    if "fetched_at_utc" not in df.columns:
        return {"file": path.name, "FAILED": "no fetched_at_utc column — cannot partition by day"}

    day = _days(df)
    undated = int(day.isna().sum())
    if undated:
        # Refuse the whole file. A snapshot with no usable timestamp cannot be filed, and both
        # available guesses (invent a day / drop the row) silently corrupt a perishable archive.
        return {"file": path.name,
                "FAILED": f"{undated} row(s) have no parseable fetched_at_utc — left untouched"}

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
    args = ap.parse_args()

    files = sorted(_DIR.glob("*_markets.csv"))
    if not files:
        print(f"no un-partitioned *_markets.csv under {_DIR} — nothing to do")
        return 0

    failed = []
    for p in files:
        r = migrate(p, args.apply)
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
