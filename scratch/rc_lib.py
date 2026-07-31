"""Shared loaders for the empirical opportunity scan. Read-only on data/."""
from __future__ import annotations

import glob
import json
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

SRC = "/Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config  # noqa: E402
import grading  # noqa: E402
import stats_util  # noqa: E402
from pmf import parse_question, parse_question_date  # noqa: E402
from resolution_anchors import RESOLUTION_ANCHORS  # noqa: E402

CITY_SLUG = {c: c.replace(" ", "_").lower() for c in RESOLUTION_ANCHORS}
SLUG_CITY = {v: k for k, v in CITY_SLUG.items()}
UNIT = {c: a.get("resolution_unit", "whole °C") for c, a in RESOLUTION_ANCHORS.items()}


def _native(temp_c, unit):
    return grading.native_round(temp_c, unit)


def _parse_yes(s):
    if not isinstance(s, str):
        return np.nan
    try:
        return float(json.loads(s).get("Yes", np.nan))
    except Exception:
        try:
            import ast
            return float(ast.literal_eval(s).get("Yes", np.nan))
        except Exception:
            return np.nan


def load_snapshots() -> pd.DataFrame:
    """Every market snapshot, deduped on (condition_id, fetched_at_utc), with parsed metadata."""
    frames = []
    for f in sorted(glob.glob(os.path.join(SRC, "data/polymarket/*_snapshots.csv"))):
        d = pd.read_csv(f)
        d["slug"] = os.path.basename(f).replace("_snapshots.csv", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["condition_id", "fetched_at_utc"])
    df["city"] = df["slug"].map(SLUG_CITY)
    df["fetched_at"] = pd.to_datetime(df["fetched_at_utc"], utc=True, format="mixed")
    df["yes"] = df["outcome_probs_json"].apply(_parse_yes)
    meta = _parse_meta(df[["condition_id", "city", "question", "end_date_iso"]].drop_duplicates("condition_id"))
    df = df.merge(meta, on="condition_id", how="left", suffixes=("", "_m"))
    return df


def _parse_meta(u: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in u.iterrows():
        q = str(r["question"])
        p = parse_question(q)
        td = parse_question_date(q, r.get("end_date_iso"))
        unit = UNIT.get(r["city"], "whole °C")
        if p is None:
            rows.append((r["condition_id"], None, np.nan, np.nan, np.nan, None, unit))
            continue
        lo_c = p.get("temp_lo", p["temp_c"])
        hi_c = p.get("temp_hi", p["temp_c"])
        rows.append((r["condition_id"], p["condition"],
                     _native(lo_c, unit), _native(hi_c, unit), p["temp_c"],
                     "min" if "lowest" in q.lower() else "max", unit))
    m = pd.DataFrame(rows, columns=["condition_id", "cond", "lo_n", "hi_n", "temp_c", "kind", "unit"])
    td = []
    for _, r in u.iterrows():
        d = parse_question_date(str(r["question"]), r.get("end_date_iso"))
        td.append(None if d is None else d.isoformat())
    m["target_date"] = td
    return m


@lru_cache(maxsize=1)
def load_price_history() -> pd.DataFrame:
    """CLOB hourly price history (Yes leg only; No == 1-Yes to 1e-3)."""
    frames = []
    for f in sorted(glob.glob(os.path.join(SRC, "data/polymarket/*_price_history.csv"))):
        d = pd.read_csv(f)
        d["slug"] = os.path.basename(f).replace("_price_history.csv", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["outcome"] == "Yes"].copy()
    df["city"] = df["slug"].map(SLUG_CITY)
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="mixed")
    df = df.drop_duplicates(subset=["condition_id", "ts"])
    return df


def grade(city, target_date, question, temp_c):
    try:
        return grading.resolves_yes(city, target_date, question, temp_c)
    except Exception:
        return None


def add_outcomes(u: pd.DataFrame) -> pd.DataFrame:
    """u must have city, target_date, question, temp_c (one row per condition_id)."""
    out = []
    for _, r in u.iterrows():
        out.append(grade(r["city"], r["target_date"], r["question"], r["temp_c"]))
    u = u.copy()
    u["outcome"] = [None if o is None else int(o) for o in out]
    return u


def ci(values, clusters):
    return stats_util.interval(values, clusters)


def fmt_ci(d, p=4):
    if d["se"] == float("inf"):
        return "—"
    return f"[{d['ci_lo']:+.{p}f}, {d['ci_hi']:+.{p}f}]"
