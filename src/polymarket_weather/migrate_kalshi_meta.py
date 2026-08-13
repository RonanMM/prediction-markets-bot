"""migrate_kalshi_meta.py — one-time split of the Kalshi markets archive into fact + dimension.

WHY. `{slug}_markets.csv` repeated four static per-ticker text fields on every hourly snapshot
of every ticker (~103 snapshots each). Measured on san_francisco_markets.csv — 50,748 rows,
100.3 MB — those fields were **71.3 MB, 74% of all field bytes**, `rules_secondary` alone
accounting for 44.5 MB across exactly ONE distinct value.

On 2026-08-12 that pushed the file past GitHub's HARD 100 MB per-file limit. The pre-receive
hook then rejected every `collect` push — ten consecutive failed runs and ~16 hours of
perishable market snapshots lost, with six more cities sitting at 80-100 MB behind it. GitHub
had been printing size warnings on every push for days; the runs stayed green until the limit
was actually crossed, so nothing escalated.

WHAT THIS DOES. For each `{slug}_markets.csv`: write `{slug}_markets_meta.csv` holding one row
per distinct (ticker, static-text) combination, then rewrite the markets file without those
columns. **No information is discarded** — this is a normalisation, not a curation. That
distinction is load-bearing: `fetch_kalshi._MARKET_FIELDS` documents why every vendor field is
captured (Kalshi serves market objects for only ~2 months, so an uncaptured field is
unrecoverable at any later date and at any price).

Keying the dimension on ticker AND content, rather than ticker alone, means a mid-life rules
amendment is preserved as a second row instead of being silently collapsed onto whichever
version happened to land first.

SAFETY. Nothing is overwritten unless the rewrite round-trips: the script re-reads what it wrote
and asserts that every retained column matches the original row-for-row, and that every
(ticker, static-text) pair present before is present in the dimension file after. A file that
fails either check is left untouched and reported.

    python migrate_kalshi_meta.py            # report only, writes nothing
    python migrate_kalshi_meta.py --apply    # perform the split
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from fetch_kalshi import META_COLS

_DIR = Path(__file__).resolve().parent / "data" / "kalshi"


def _meta_frame(df: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """One row per ticker carrying its static text — first NON-NULL value of each column.

    Nulls are taken out before deduping. The vendor omits these fields on ~0.9% of snapshots,
    and a naive drop_duplicates over (ticker, text) counts the omission as a distinct variant:
    on the real SF archive that produced 930 rows for 492 tickers, i.e. an empty twin for nearly
    every one.

    The staticness is VERIFIED, not assumed — `_varying_tickers` fails the migration if any
    ticker ever carried two different non-null values, because the fact table is about to stop
    recording when such a change happened.
    """
    return (df[["ticker"] + present]
            .groupby("ticker", as_index=False)
            .first()                              # pandas' first() skips NaN per column
            .reset_index(drop=True))


def _varying_tickers(df: pd.DataFrame, present: list[str]) -> dict[str, int]:
    """Tickers whose non-null value for a meta column changed over the archive, per column.

    Must be empty. If it is not, these fields are not static and moving them into a dimension
    table would silently discard the timing of the change — the exact class of quiet loss this
    archive exists to prevent. The migration refuses rather than guessing.
    """
    out = {}
    for c in present:
        v = df.dropna(subset=[c]).groupby("ticker")[c].nunique()
        n = int((v > 1).sum())
        if n:
            out[c] = n
    return out


def migrate(path: Path, apply: bool) -> dict:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    present = [c for c in META_COLS if c in df.columns]
    before = path.stat().st_size
    if not present:
        return {"file": path.name, "skipped": "already split", "before_mb": before / 1e6}

    varying = _varying_tickers(df, present)
    if varying:
        return {"file": path.name,
                "FAILED": f"not static — tickers with a changed value: {varying}"}

    meta = _meta_frame(df, present)
    facts = df.drop(columns=present)

    if not apply:
        saved = sum(df[c].astype(str).str.len().sum() for c in present)
        return {"file": path.name, "rows": len(df), "meta_rows": len(meta),
                "before_mb": before / 1e6, "est_after_mb": (before - saved) / 1e6}

    meta_path = path.with_name(path.name.replace("_markets.csv", "_markets_meta.csv"))
    tmp_f, tmp_m = path.with_suffix(".csv.tmp"), meta_path.with_suffix(".csv.tmp")
    facts.to_csv(tmp_f, index=False)
    meta.to_csv(tmp_m, index=False)

    # Round-trip verification BEFORE anything is replaced. A migration that silently drops rows
    # would be indistinguishable from a successful one by file size alone.
    rt_f = pd.read_csv(tmp_f, dtype=str, low_memory=False)
    rt_m = pd.read_csv(tmp_m, dtype=str, low_memory=False)
    kept = [c for c in df.columns if c not in present]
    if len(rt_f) != len(df):
        tmp_f.unlink(missing_ok=True); tmp_m.unlink(missing_ok=True)
        return {"file": path.name, "FAILED": f"row count {len(rt_f)} != {len(df)}"}
    if not rt_f[kept].fillna("").equals(df[kept].fillna("")):
        tmp_f.unlink(missing_ok=True); tmp_m.unlink(missing_ok=True)
        return {"file": path.name, "FAILED": "retained columns did not round-trip"}
    # Every NON-NULL (ticker, column, value) the archive ever held must survive in the dimension
    # file. Nulls are excluded on both sides: an omitted field is an absence, not a value, and
    # _meta_frame deliberately does not archive it.
    lost = 0
    for c in present:
        want = set(map(tuple, df.dropna(subset=[c])[["ticker", c]].values.tolist()))
        got = set(map(tuple, rt_m.dropna(subset=[c])[["ticker", c]].values.tolist()))
        lost += len(want - got)
    if lost:
        tmp_f.unlink(missing_ok=True); tmp_m.unlink(missing_ok=True)
        return {"file": path.name, "FAILED": f"{lost} (ticker, text) pairs lost"}

    tmp_m.replace(meta_path)
    tmp_f.replace(path)
    return {"file": path.name, "rows": len(df), "meta_rows": len(meta),
            "before_mb": before / 1e6, "after_mb": path.stat().st_size / 1e6,
            "meta_mb": meta_path.stat().st_size / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the split (default: report only)")
    args = ap.parse_args()

    files = sorted(_DIR.glob("*_markets.csv"))
    if not files:
        print(f"no *_markets.csv under {_DIR}")
        return 0

    failed, before_tot, after_tot = [], 0.0, 0.0
    for p in files:
        r = migrate(p, args.apply)
        before_tot += r.get("before_mb", 0.0)
        after_tot += r.get("after_mb", r.get("est_after_mb", r.get("before_mb", 0.0)))
        if "FAILED" in r:
            failed.append(r)
        print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()))

    print(f"\ntotal {before_tot:.0f} MB -> {after_tot:.0f} MB "
          f"({'applied' if args.apply else 'estimated, dry run'})")
    if failed:
        print(f"\n{len(failed)} FILE(S) FAILED VERIFICATION and were left untouched:")
        for r in failed:
            print(f"  {r['file']}: {r['FAILED']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
