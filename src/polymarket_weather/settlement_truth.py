"""settlement_truth.py — settlement-faithful daily actuals for TRAINING (megaplan W0.2).

The EMOS/intraday models were trained against the CLI feed (`{slug}_historical_actuals.csv`).
But the markets settle on the WUNDERGROUND daily extremes (hourly-METAR max/min over the local
calendar day) for NYC/Chicago, which differ from the CLI by ≥1°F on ~50% of training days with a
systematic ~0.6°F bias (CLI's 1-minute sensors catch spikes the METARs miss). Training the mean
correction against the CLI target therefore biases it away from what the market actually pays out
on, at exactly the whole-°F bin boundaries the US markets resolve at.

This writes `{slug}_settlement_actuals.csv` (date_local, temp_max_c, temp_min_c) using the SAME
blend `grading.fetch_actual_weather` uses — WU reconstruction for NYC/Chicago (with the ≥18-obs /
≤3h-gap coverage guard and the ±2.5°C sanity rail against the CLI), the CLI/station feed
everywhere else — so training and grading share a single ruler. Non-US cities get a CLI copy
(their station feed is already settlement-faithful), so retraining is a no-op for them and any
change is isolated to NYC/Chicago.

Run from src/polymarket_weather/:   python settlement_truth.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import wu_truth
from wu_truth import _WU_RECON_SLUGS, _MIN_OBS_PER_DAY, _MAX_GAP_HOURS

_W = Path(__file__).resolve().parent / "data" / "weather"
_SLUGS = ["new_york_city", "chicago", "london", "seoul", "hong_kong"]


def _wu_daily_table(slug: str) -> pd.DataFrame | None:
    """Vectorized WU daily extremes for a reconstruction-validated slug, or None.

    Mirrors wu_truth.wu_daily_extreme exactly (same coverage guards) but computes every day in
    one pass. Returns date_local -> (wu_max, wu_min) only for days that pass the guard.
    """
    if slug not in _WU_RECON_SLUGS.values():
        return None
    p = _W / f"{slug}_obs_hourly.csv"
    if not p.exists():
        return None
    obs = pd.read_csv(p)
    if not {"valid_local", "temp_c"}.issubset(obs.columns):
        return None
    obs["valid_local"] = pd.to_datetime(obs["valid_local"], errors="coerce")
    obs["temp_c"] = pd.to_numeric(obs["temp_c"], errors="coerce")
    obs = obs.dropna(subset=["valid_local", "temp_c"]).sort_values("valid_local")
    obs["date_local"] = obs["valid_local"].dt.strftime("%Y-%m-%d")
    g = obs.groupby("date_local")
    size = g.size()
    max_gap = g["valid_local"].apply(
        lambda s: s.diff().dt.total_seconds().div(3600.0).max())
    good = (size >= _MIN_OBS_PER_DAY) & (max_gap <= _MAX_GAP_HOURS)
    out = pd.DataFrame({"wu_max": g["temp_c"].max(), "wu_min": g["temp_c"].min()})
    return out[good.reindex(out.index, fill_value=False)]


def build_settlement_actuals(slug: str) -> pd.DataFrame:
    """CLI actuals with WU extremes blended in for NYC/Chicago (±2.5°C sanity rail)."""
    cli = pd.read_csv(_W / f"{slug}_historical_actuals.csv")
    cli["date_local"] = pd.to_datetime(cli["date_local"]).dt.strftime("%Y-%m-%d")
    out = cli[["date_local", "temp_max_c", "temp_min_c"]].copy()

    wu = _wu_daily_table(slug)
    if wu is not None:
        out = out.set_index("date_local")
        for col, wu_col in [("temp_max_c", "wu_max"), ("temp_min_c", "wu_min")]:
            j = out.join(wu[wu_col], how="left")
            cli_v, wu_v = j[col], j[wu_col]
            # prefer WU where it exists and stays within the sanity divergence of the CLI
            use_wu = wu_v.notna() & (
                cli_v.isna() | ((wu_v - cli_v).abs() <= wu_truth.SANITY_MAX_DIVERGENCE_C))
            out[col] = np.where(use_wu, wu_v, cli_v)
        out = out.reset_index()
    return out


def build_all() -> None:
    for slug in _SLUGS:
        try:
            df = build_settlement_actuals(slug)
        except FileNotFoundError:
            print(f"  {slug}: no historical_actuals — skipped")
            continue
        path = _W / f"{slug}_settlement_actuals.csv"
        df.to_csv(path, index=False)
        wu = _wu_daily_table(slug)
        n_wu = 0 if wu is None else len(wu)
        print(f"  {slug:<15} {len(df):>5} days  ({n_wu} WU-reconstructed) → {path.name}")


def load_training_truth(slug: str) -> pd.DataFrame:
    """Settlement actuals for training if built, else the CLI feed (backward compatible)."""
    p = _W / f"{slug}_settlement_actuals.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.read_csv(_W / f"{slug}_historical_actuals.csv")


if __name__ == "__main__":
    build_all()
