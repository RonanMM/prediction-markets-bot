"""
processing.py — Data normalization, persistence, and derived metrics.

Responsibilities:
  • Append-only CSV storage (no overwrites)
  • Deduplication by (condition_id, fetched_at_utc) for market snapshots
  • Deduplication by (city, date_local, fetched_at_utc) for weather
  • Implied temperature calculation from outcome probabilities
  • Alignment of market snapshots with forecast rows (nearest fetch)
"""

import logging
import os
import re
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import pandas as pd

from config import POLYMARKET_DIR, WEATHER_DIR

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

def _mkt_snapshot_path(city: str) -> Path:
    return Path(POLYMARKET_DIR) / f"{_slug(city)}_snapshots.csv"

def _mkt_history_path(city: str) -> Path:
    return Path(POLYMARKET_DIR) / f"{_slug(city)}_price_history.csv"

def _weather_daily_path(city: str) -> Path:
    return Path(WEATHER_DIR) / f"{_slug(city)}_daily.csv"

def _weather_hourly_path(city: str) -> Path:
    return Path(WEATHER_DIR) / f"{_slug(city)}_hourly.csv"

def _ensemble_path(city: str) -> Path:
    return Path(WEATHER_DIR) / f"{_slug(city)}_ensemble.csv"

def _weather_daily_mm_path(city: str) -> Path:
    return Path(WEATHER_DIR) / f"{_slug(city)}_daily_mm.csv"

def _slug(city: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")


# ── Generic CSV append-with-dedup ────────────────────────────────────────────

def _append_csv(path: Path, records: list[dict], dedup_cols: list[str]) -> int:
    """
    Append *records* to *path*.csv, skipping rows that already exist
    according to *dedup_cols*.  Returns number of new rows written.
    """
    if not records:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        # Build a set of existing key tuples for fast lookup. Use ONE fixed key-column
        # list on both sides (and stringify both) so a column missing on one side can no
        # longer produce length-mismatched tuples that silently never match → duplicates.
        key_cols = [c for c in dedup_cols if c in existing.columns]
        existing_keys = set(
            zip(*[existing[c].astype(str).tolist() for c in key_cols])
        )
        def _is_new(row) -> bool:
            key = tuple(str(row[c]) if c in row else "" for c in key_cols)
            return key not in existing_keys

        new_df = new_df[new_df.apply(_is_new, axis=1)]
        if new_df.empty:
            logger.debug("No new rows to append to %s.", path.name)
            return 0

        # Align columns to prevent misalignment corruption
        new_df = new_df.reindex(columns=existing.columns)
        new_df.to_csv(path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(path, index=False)

    logger.info("Wrote %d new rows → %s", len(new_df), path.name)
    return len(new_df)


# ── Market snapshot persistence ───────────────────────────────────────────────

def save_market_snapshots(snapshots: list[dict]) -> None:
    """Save market snapshot records grouped by city."""
    by_city: dict[str, list[dict]] = {}
    for snap in snapshots:
        flat = {k: v for k, v in snap.items() if k != "outcome_probs"}
        flat["outcome_probs_json"] = json.dumps(snap.get("outcome_probs", {}))
        flat["clob_token_ids_json"] = json.dumps(snap.get("clob_token_ids", []))
        flat.pop("clob_token_ids", None)
        by_city.setdefault(snap["city"], []).append(flat)

    for city, rows in by_city.items():
        _append_csv(
            _mkt_snapshot_path(city),
            rows,
            dedup_cols=["condition_id", "fetched_at_utc"],
        )


def save_price_history(records: list[dict]) -> None:
    """Save CLOB price-history records grouped by city."""
    by_city: dict[str, list[dict]] = {}
    for r in records:
        by_city.setdefault(r["city"], []).append(r)

    for city, rows in by_city.items():
        _append_csv(
            _mkt_history_path(city),
            rows,
            dedup_cols=["token_id", "timestamp_utc"],
        )


# ── Weather persistence ───────────────────────────────────────────────────────

def save_weather_forecast(forecast: dict) -> None:
    """Persist daily + hourly forecast records for one city."""
    city = forecast.get("city", "unknown")

    _append_csv(
        _weather_daily_path(city),
        forecast.get("daily", []),
        dedup_cols=["city", "date_local", "fetched_at_utc"],
    )
    _append_csv(
        _weather_hourly_path(city),
        forecast.get("hourly", []),
        dedup_cols=["city", "datetime_local", "fetched_at_utc"],
    )


def save_multimodel_forecast(forecast: dict) -> None:
    """Persist per-model deterministic daily Tmax records for one city
    (the calibrated predictor's multi-model mean input)."""
    city = forecast.get("city", "unknown")
    _append_csv(
        _weather_daily_mm_path(city),
        forecast.get("daily", []),
        dedup_cols=["city", "date_local", "fetched_at_utc"],
    )


# ── Ensemble persistence ──────────────────────────────────────────────────────

def save_ensemble_forecast(forecast: dict) -> None:
    """Persist ensemble daily statistics for one city."""
    city = forecast.get("city", "unknown")
    _append_csv(
        _ensemble_path(city),
        forecast.get("daily", []),
        dedup_cols=["city", "date_local", "fetched_at_utc"],
    )


# ── Implied temperature ───────────────────────────────────────────────────────

_RANGE_PATTERN = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*[-–—to]+\s*([+-]?\d+(?:\.\d+)?)"
)
_SINGLE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*°?[CF]?")


def _parse_temp_range(label: str) -> tuple[float, float] | None:
    """
    Try to extract a temperature midpoint from an outcome label.
    Handles: '20-22°C', '68–72°F', '>86', '<10', '22.5', etc.
    Returns (low, high) or None.
    """
    # Convert °F labels to °C if needed
    is_fahrenheit = "°F" in label or label.strip().endswith("F")

    m = _RANGE_PATTERN.search(label)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if is_fahrenheit:
            lo = (lo - 32) * 5 / 9
            hi = (hi - 32) * 5 / 9
        return lo, hi

    # "above X" / "> X"
    above = re.search(r"(?:above|>|≥|>=)\s*([+-]?\d+(?:\.\d+)?)", label, re.I)
    if above:
        v = float(above.group(1))
        if is_fahrenheit:
            v = (v - 32) * 5 / 9
        return v, v + 2   # assume narrow upper open bucket

    # "below X" / "< X"
    below = re.search(r"(?:below|<|≤|<=)\s*([+-]?\d+(?:\.\d+)?)", label, re.I)
    if below:
        v = float(below.group(1))
        if is_fahrenheit:
            v = (v - 32) * 5 / 9
        return v - 2, v

    return None


def compute_implied_temperature(outcome_probs: dict[str, float], question: str = "") -> float | None:
    """
    For bucket markets: weighted midpoint from {outcome_label: probability}.
    For binary Yes/No markets: extract temperature from question text,
    return it weighted by P(Yes).
    """
    # ── Binary Yes/No market ───────────────────────────────────────────────
    keys = {k.strip().lower() for k in outcome_probs}
    if keys <= {"yes", "no"}:
        p_yes = outcome_probs.get("Yes", outcome_probs.get("yes", 0.0))
        temp = _extract_temp_from_question(question)
        if temp is not None:
            return None
        return None

    # ── Bucket market ─────────────────────────────────────────────────────
    total_weight = 0.0
    weighted_sum = 0.0
    for label, prob in outcome_probs.items():
        rng = _parse_temp_range(str(label))
        if rng is None:
            continue
        midpoint = (rng[0] + rng[1]) / 2
        weighted_sum += midpoint * prob
        total_weight += prob

    if total_weight < 1e-6:
        return None
    return weighted_sum / total_weight


def _extract_temp_from_question(question: str) -> float | None:
    """
    Pull a temperature value out of a binary market question.
    Handles:
      '...be 18°C...'           → 18.0
      '...be between 66-67°F...'→ midpoint in °C
      '...be 65°F...'           → converted to °C
    """
    q = question

    # Range pattern: "between 66-67°F" or "between 18-20°C"
    m = re.search(
        r'between\s+([+-]?\d+(?:\.\d+)?)\s*[-–]\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*([CF])',
        q, re.IGNORECASE
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        unit = m.group(3).upper()
        mid = (lo + hi) / 2
        return (mid - 32) * 5 / 9 if unit == "F" else mid

    # Single value: "be 18°C" or "be 65°F"
    m = re.search(
        r'be\s+([+-]?\d+(?:\.\d+)?)\s*°?\s*([CF])\b',
        q, re.IGNORECASE
    )
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        return (val - 32) * 5 / 9 if unit == "F" else val

    return None

# ── Load helpers ──────────────────────────────────────────────────────────────

def load_market_snapshots(city: str) -> pd.DataFrame:
    path = _mkt_snapshot_path(city)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True, errors="coerce")

    # Re-hydrate outcome_probs from JSON string
    def _safe_parse(s):
        if not s or s != s:   # None / NaN guard
            return {}
        s = str(s).strip()
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(s)   # fallback for legacy rows
            except Exception:
                return {}

    if "outcome_probs_json" in df.columns:
        df["outcome_probs"] = df["outcome_probs_json"].apply(_safe_parse)

    # Compute implied temperature per row
    # With this:
    df["implied_temp_c"] = df.apply(
        lambda row: compute_implied_temperature(
            row["outcome_probs"] if isinstance(row["outcome_probs"], dict) else {},
            question=str(row.get("question", ""))
        ),
        axis=1
    )

    return df


def load_weather_daily(city: str) -> pd.DataFrame:
    path = _weather_daily_path(city)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date_utc"]       = pd.to_datetime(df["date_utc"],       utc=True, errors="coerce")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True, errors="coerce")
    df["temp_max_c"]     = pd.to_numeric(df["temp_max_c"],     errors="coerce")
    df["temp_min_c"]     = pd.to_numeric(df["temp_min_c"],     errors="coerce")
    return df




def _extract_temp_from_question_str(q: str) -> float | None:
    # Range: "between 62-63°F" or "between 18-20°C"
    m = re.search(r'between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*°?\s*([CF])', q, re.I)
    if m:
        mid = (float(m.group(1)) + float(m.group(2))) / 2
        return (mid - 32) * 5/9 if m.group(3).upper() == 'F' else mid
    # Single or "X or below/above": "33°F or below", "be 18°C"
    m = re.search(r'be\s+(\d+(?:\.\d+)?)\s*°?\s*([CF])', q, re.I)
    if m:
        val = float(m.group(1))
        return (val - 32) * 5/9 if m.group(2).upper() == 'F' else val
    return None

# ── Summary statistics ────────────────────────────────────────────────────────

def compute_city_summary(city: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"city": city}

    snaps = load_market_snapshots(city)
    weather = load_weather_daily(city)

    if not snaps.empty:
        open_snaps = snaps[snaps["closed"] == False].copy()
        open_snaps = open_snaps.drop_duplicates("condition_id")

        # Extract target temp from question
        open_snaps["target_temp"] = open_snaps["question"].apply(_extract_temp_from_question_str)
        open_snaps["p_yes"] = open_snaps["outcome_probs"].apply(
            lambda p: p.get("Yes", p.get("yes", 0.0)) if isinstance(p, dict) else 0.0
        )

        # Aggregate per end_date: weighted implied temp
        valid = open_snaps.dropna(subset=["target_temp"])
        if not valid.empty:
            by_date = valid.groupby("end_date_iso").apply(
                lambda g: (g["target_temp"] * g["p_yes"]).sum() / g["p_yes"].sum()
                if g["p_yes"].sum() > 0.05 else None
            ).dropna()

            if not by_date.empty:
                # Pick the nearest future date with a valid implied temp
                today = datetime.now(timezone.utc).date().isoformat()
                future = by_date[by_date.index >= today]
                if not future.empty:
                    best_date = future.index[0]
                    summary["market_implied_temp_c"] = round(future.iloc[0], 1)
                    summary["market_target_date"] = best_date
                    # Best market question for that date
                    date_snaps = valid[valid["end_date_iso"] == best_date]
                    summary["market_question"] = date_snaps.iloc[0]["question"]
                    summary["market_volume_usdc"] = date_snaps["volume_usdc"].sum()

    # Weather for the market's target date (not just today)
    if not weather.empty:
        target_date = summary.get("market_target_date", 
                                   datetime.now(timezone.utc).date().isoformat())
        wx = weather[weather["date_local"] == target_date]
        if wx.empty:
            wx = weather.sort_values("date_local")
            wx = wx[wx["date_local"] >= target_date]
        if not wx.empty:
            latest = wx.sort_values("fetched_at_utc").iloc[-1]
            summary["forecast_temp_max_c"] = round(float(latest["temp_max_c"]), 1)
            summary["forecast_date_local"] = latest["date_local"]

    if "market_implied_temp_c" in summary and "forecast_temp_max_c" in summary:
        summary["difference_c"] = round(
            summary["market_implied_temp_c"] - summary["forecast_temp_max_c"], 1
        )

    return summary
