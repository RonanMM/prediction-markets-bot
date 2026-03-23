"""
Polymarket Weather Inefficiency Analyzer — v2
==============================================
Key insight from v1 failure:
  Polymarket weather markets form a CATEGORICAL DISTRIBUTION across temp bins
  for a given (city, date). The market-implied σ is ~0.4-0.6°C (overconfident),
  while NWP verification statistics say true forecast uncertainty is σ≈1.5-2.5°C.

  The edge comes from:
    - Market underprices "nearby" bins relative to the modal forecast
    - Market overprices the exact modal bin
    - Forecast shifts (24h update) not yet reflected in market

Approach:
  1. Reconstruct market PMF per (city, target_date, snapshot_time) from ALL bins
  2. Build forecast PMF using NWP-calibrated Gaussian over daily temp_max_c
  3. Compute KL divergence + per-bin edge
  4. Rank opportunities by: edge × liquidity / kelly_risk

Usage:
  python polymarket_weather_v2.py --data_dir ./data [--cities london chicago ...]
"""

import os
import re
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from math import erf, sqrt, log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# NWP verification-based sigma by days-ahead horizon
# Source: ECMWF/GFS skill scores for 2m temp max
NWP_SIGMA_BY_DAY = {
    0: 0.8,   # same-day  (obs almost certain)
    1: 1.2,   # 1-day
    2: 1.6,   # 2-day
    3: 2.0,   # 3-day
    4: 2.4,   # 4-day
    5: 2.8,   # 5-day
}
NWP_SIGMA_DEFAULT = 3.2  # beyond 5 days

MIN_EDGE_FRACTION = 0.07   # 7pp in normalized PMF
MIN_LIQUIDITY     = 400    # USDC
MIN_BINS_FOR_DAY  = 3      # need ≥3 bins to reconstruct distribution
KELLY_FRACTION    = 0.25
MAX_KELLY_BET     = 0.15   # never bet >15% of bankroll on one market

CITY_ALIASES = {
    "new_york_city": "NYC",
    "new_york":      "NYC",
    "london":        "London",
    "chicago":       "Chicago",
    "hong_kong":     "HK",
    "seoul":         "Seoul",
}

# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def discover_cities(data_dir: Path) -> list:
    data_dir = data_dir / Path("polymarket") 
    return [f.stem.replace("_snapshots", "")
            for f in sorted(data_dir.glob("*_snapshots.csv"))]


def load_snapshots(data_dir: Path, city: str) -> pd.DataFrame:
    path = data_dir / "polymarket" / f"{city}_snapshots.csv"
    df = pd.read_csv(path)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["end_date_iso"]   = pd.to_datetime(df["end_date_iso"],   utc=True, errors="coerce")
    df["city"] = city

    def _yes(val):
        try:
            d = json.loads(str(val).replace("'", '"'))
            return float(d.get("Yes", d.get("yes", np.nan)))
        except:
            return np.nan

    df["yes_prob"] = df["outcome_probs_json"].apply(_yes)
    return df


def load_daily(data_dir: Path, city: str) -> pd.DataFrame:
    path = data_dir / "weather" / f"{city}_daily.csv"
    df = pd.read_csv(path)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    return df


# ─── QUESTION PARSER ───────────────────────────────────────────────────────────

RE_EXACT_C  = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+on\b", re.I)
RE_GTE_C    = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+higher", re.I)
RE_LTE_C    = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:lower|below)", re.I)
RE_EXACT_F  = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+on\b", re.I)
RE_RANGE_F  = re.compile(r"between\s+(\d+)-(\d+)\s*°?\s*F", re.I)
RE_GTE_F    = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+higher", re.I)
RE_LTE_F    = re.compile(r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:lower|below)", re.I)


def f2c(f): return (f - 32) * 5 / 9


def parse_question(q: str) -> Optional[dict]:
    """Returns dict with condition, temp_c (target), lo/hi for range, and bin_type."""
    m = RE_GTE_C.search(q)
    if m: return dict(condition="gte", temp_c=float(m.group(1)))

    m = RE_LTE_C.search(q)
    if m: return dict(condition="lte", temp_c=float(m.group(1)))

    m = RE_EXACT_C.search(q)
    if m: return dict(condition="exact", temp_c=float(m.group(1)))

    m = RE_RANGE_F.search(q)
    if m: return dict(condition="range",
                      temp_c=(f2c(float(m.group(1))) + f2c(float(m.group(2)))) / 2,
                      temp_lo=f2c(float(m.group(1))), temp_hi=f2c(float(m.group(2))))

    m = RE_GTE_F.search(q)
    if m: return dict(condition="gte", temp_c=f2c(float(m.group(1))))

    m = RE_LTE_F.search(q)
    if m: return dict(condition="lte", temp_c=f2c(float(m.group(1))))

    m = RE_EXACT_F.search(q)
    if m: return dict(condition="exact", temp_c=f2c(float(m.group(1))))

    return None


# ─── PROBABILITY MATH ──────────────────────────────────────────────────────────

def Φ(x, mu, sigma):
    """Standard normal CDF."""
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))


def forecast_bin_prob(parsed: dict, mu: float, sigma: float) -> float:
    """P(condition | mu, sigma) — raw, unnormalized."""
    c = parsed["condition"]
    t = parsed["temp_c"]
    if c == "exact":
        return Φ(t + 0.5, mu, sigma) - Φ(t - 0.5, mu, sigma)
    elif c == "gte":
        return 1 - Φ(t, mu, sigma)
    elif c == "lte":
        return Φ(t, mu, sigma)
    elif c == "range":
        return Φ(parsed["temp_hi"], mu, sigma) - Φ(parsed["temp_lo"], mu, sigma)
    return np.nan


def get_nwp_sigma(days_ahead: float) -> float:
    day = int(min(max(days_ahead, 0), 5))
    return NWP_SIGMA_BY_DAY.get(day, NWP_SIGMA_DEFAULT)


def kl_divergence(p: dict, q: dict) -> float:
    """KL(p || q) — p=true, q=approx. Keys must match."""
    eps = 1e-8
    return sum(p[k] * log(p[k] / max(q.get(k, eps), eps)) for k in p if p[k] > 0)


# ─── CORE: RECONSTRUCT DISTRIBUTIONS ──────────────────────────────────────────

def build_day_snapshot(snap_group: pd.DataFrame, daily_df: pd.DataFrame,
                        city: str) -> list:
    """
    For a group of snapshot rows sharing (city, target_date, ~fetch_time),
    reconstruct the market PMF and forecast PMF, then compute per-bin edges.
    """
    results = []

    target_date = snap_group["end_date_iso"].iloc[0]
    fetch_time  = snap_group["fetched_at_utc"].iloc[0]

    # ── Parse all bins ──
    bins = []
    for _, row in snap_group.iterrows():
        parsed = parse_question(str(row.get("question", "")))
        if parsed is None:
            continue
        yes_prob  = row["yes_prob"]
        liquidity = float(row.get("liquidity_usdc", 0) or 0)
        if np.isnan(yes_prob):
            continue
        bins.append({
            "parsed":       parsed,
            "yes_prob":     yes_prob,
            "liquidity":    liquidity,
            "question":     row.get("question", ""),
            "condition_id": row.get("condition_id", ""),
            "volume_24h":   float(row.get("volume_24h_usdc", 0) or 0),
        })

    if len(bins) < MIN_BINS_FOR_DAY:
        return results

    # ── Get best forecast for this target_date ──
    td_norm = pd.Timestamp(target_date).normalize()
    if td_norm.tzinfo is not None:
        td_norm = td_norm.tz_localize(None)

    daily_local = daily_df.copy()
    daily_local["date_local"] = pd.to_datetime(daily_local["date_local"]).dt.normalize()
    mask = (daily_local["fetched_at_utc"] <= fetch_time) & \
           (daily_local["date_local"] == td_norm)
    candidates = daily_local[mask]
    if candidates.empty:
        return results

    forecast_row = candidates.sort_values("fetched_at_utc").iloc[-1]
    mu_forecast  = float(forecast_row["temp_max_c"])
    days_fwd     = (target_date - fetch_time).total_seconds() / 86400
    sigma        = get_nwp_sigma(days_fwd)

    # ── Build market PMF ──
    # For exclusive bins (exact), market PMF is direct.
    # For boundary bins (gte, lte), treat them as residuals.
    # Normalize all exact bins, then assign boundary bins the remainder.
    exact_bins    = [b for b in bins if b["parsed"]["condition"] == "exact"]
    boundary_bins = [b for b in bins if b["parsed"]["condition"] != "exact"]

    # Heuristic: use exact bins for the normalized PMF backbone
    if not exact_bins:
        return results

    # Raw market probabilities (already scaled 0-1 individually, not summing to 1)
    market_raw  = {b["parsed"]["temp_c"]: b["yes_prob"] for b in exact_bins}
    total_mkt   = sum(market_raw.values())
    if total_mkt < 0.05:  # degenerate
        return results

    market_pmf  = {k: v / total_mkt for k, v in market_raw.items()}
    temps_exact = sorted(market_raw.keys())

    # ── Build forecast PMF over same bins ──
    forecast_raw = {t: forecast_bin_prob({"condition":"exact","temp_c":t}, mu_forecast, sigma)
                    for t in temps_exact}
    total_fcst   = sum(forecast_raw.values())
    if total_fcst < 1e-6:
        return results
    forecast_pmf = {k: v / total_fcst for k, v in forecast_raw.items()}

    # ── Per-bin edge ──
    kl = kl_divergence(forecast_pmf, market_pmf)

    for b in exact_bins:
        t    = b["parsed"]["temp_c"]
        m_p  = market_pmf.get(t, 0)
        f_p  = forecast_pmf.get(t, 0)
        edge = f_p - m_p

        results.append({
            "city":          CITY_ALIASES.get(city, city),
            "city_raw":      city,
            "condition_id":  b["condition_id"],
            "question":      b["question"][:80],
            "target_date":   target_date.date() if hasattr(target_date, "date") else str(target_date)[:10],
            "fetched_at":    fetch_time,
            "days_ahead":    round(days_fwd, 2),
            "bin_temp_c":    t,
            "forecast_mu":   round(mu_forecast, 1),
            "forecast_sigma":round(sigma, 2),
            "forecast_prob": round(f_p, 4),
            "market_prob":   round(m_p, 4),   # normalized
            "market_prob_raw": round(b["yes_prob"], 4),  # raw Polymarket
            "edge":          round(edge, 4),
            "abs_edge":      round(abs(edge), 4),
            "bet_side":      "Yes" if edge > 0 else "No",
            "our_prob":      round(f_p if edge > 0 else 1 - f_p, 4),
            "their_prob":    round(m_p if edge > 0 else 1 - m_p, 4),
            "liquidity":     b["liquidity"],
            "volume_24h":    b["volume_24h"],
            "day_kl_div":    round(kl, 4),
            "n_bins":        len(exact_bins),
            "market_mode_c": max(market_raw, key=market_raw.get),
            "forecast_mode_c": mu_forecast,
            "mode_shift_c":  round(mu_forecast - max(market_raw, key=market_raw.get), 2),
        })

    # ── Also handle boundary bins (gte / lte) ──
    for b in boundary_bins:
        parsed   = b["parsed"]
        f_raw_p  = forecast_bin_prob(parsed, mu_forecast, sigma)
        m_raw_p  = b["yes_prob"]  # raw Polymarket probability

        # For boundary bins, edge is simpler: compare forecast CDF to market
        edge = f_raw_p - m_raw_p
        if abs(edge) >= MIN_EDGE_FRACTION and b["liquidity"] >= MIN_LIQUIDITY:
            results.append({
                "city":           CITY_ALIASES.get(city, city),
                "city_raw":       city,
                "condition_id":   b["condition_id"],
                "question":       b["question"][:80],
                "target_date":    target_date.date() if hasattr(target_date,"date") else str(target_date)[:10],
                "fetched_at":     fetch_time,
                "days_ahead":     round(days_fwd, 2),
                "bin_temp_c":     parsed["temp_c"],
                "forecast_mu":    round(mu_forecast, 1),
                "forecast_sigma": round(sigma, 2),
                "forecast_prob":  round(f_raw_p, 4),
                "market_prob":    round(m_raw_p, 4),
                "market_prob_raw":round(m_raw_p, 4),
                "edge":           round(edge, 4),
                "abs_edge":       round(abs(edge), 4),
                "bet_side":       "Yes" if edge > 0 else "No",
                "our_prob":       round(f_raw_p if edge > 0 else 1-f_raw_p, 4),
                "their_prob":     round(m_raw_p if edge > 0 else 1-m_raw_p, 4),
                "liquidity":      b["liquidity"],
                "volume_24h":     b["volume_24h"],
                "day_kl_div":     round(kl, 4),
                "n_bins":         len(exact_bins),
                "market_mode_c":  max(market_raw, key=market_raw.get) if market_raw else np.nan,
                "forecast_mode_c":mu_forecast,
                "mode_shift_c":   round(mu_forecast - (max(market_raw, key=market_raw.get) if market_raw else mu_forecast), 2),
            })

    return results


def analyze_city(data_dir: Path, city: str) -> pd.DataFrame:
    try:
        snap  = load_snapshots(data_dir, city)
        daily = load_daily(data_dir, city)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return pd.DataFrame()

    all_results = []

    # Group by (target_date, fetched_at rounded to nearest 10min)
    snap["fetch_bucket"] = snap["fetched_at_utc"].dt.floor("10min")
    snap["end_date_norm"] = snap["end_date_iso"].dt.normalize()

    groups = snap.groupby(["end_date_norm", "fetch_bucket"])
    total_groups = len(groups)
    print(f"  Processing {total_groups} (date × snapshot) groups…", end=" ", flush=True)

    for (target_date, _), group in groups:
        rows = build_day_snapshot(group, daily, city)
        all_results.extend(rows)

    print(f"→ {len(all_results)} bin records")
    return pd.DataFrame(all_results)


# ─── OPPORTUNITY FILTER ────────────────────────────────────────────────────────

def filter_opportunities(df: pd.DataFrame,
                          min_edge=MIN_EDGE_FRACTION,
                          min_liq=MIN_LIQUIDITY) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["abs_edge"] >= min_edge) & (df["liquidity"] >= min_liq)
    opps = df[mask].copy()

    # Kelly sizing
    def kelly(row):
        b = (1.0 / row["their_prob"] - 1.0) if row["their_prob"] > 0 else 0
        if b <= 0:
            return 0.0
        k = KELLY_FRACTION * (b * row["our_prob"] - (1 - row["our_prob"])) / b
        return float(np.clip(k, 0, MAX_KELLY_BET))

    opps["kelly"]     = opps.apply(kelly, axis=1)
    opps["ev_per_$"]  = opps["our_prob"] / opps["their_prob"] - 1.0  # % return
    opps["score"]     = opps["abs_edge"] * np.log1p(opps["liquidity"]) * (1 / (opps["days_ahead"] + 0.5))

    return opps.sort_values("score", ascending=False)


# ─── VISUALIZATIONS ────────────────────────────────────────────────────────────

DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
DARK_BDR = "#30363d"
CLR_MUTED = "#8b949e"
CLR_YES  = "#3fb950"
CLR_NO   = "#f85149"
CLR_ACC  = "#58a6ff"
CLR_WARN = "#e3b341"


def _dark_fig(nrows=1, ncols=1, figsize=(14, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK_BDR)
        ax.tick_params(colors=CLR_MUTED)
        ax.xaxis.label.set_color(CLR_MUTED)
        ax.yaxis.label.set_color(CLR_MUTED)
    return fig, axes


def plot_distribution_comparison(all_df: pd.DataFrame, output_dir: Path,
                                  top_n_days=6):
    """
    For the top-N most dislocated (city, target_date, snapshot),
    plot side-by-side forecast vs market PMF.
    """
    if all_df.empty:
        return

    # Pick most dislocated distributions
    day_kl = (all_df.groupby(["city_raw", "target_date", "fetched_at"])
              ["day_kl_div"].first()
              .nlargest(top_n_days))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.patch.set_facecolor(DARK_BG)

    for idx, ((city_raw, tdate, ftime), kl_val) in enumerate(day_kl.items()):
        if idx >= 6:
            break
        ax = axes.flat[idx]
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK_BDR)

        sub = all_df[(all_df["city_raw"] == city_raw) &
                     (all_df["target_date"] == tdate) &
                     (all_df["fetched_at"] == ftime)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sub = sub.sort_values("bin_temp_c")
        temps     = sub["bin_temp_c"].values
        mkt_probs = sub["market_prob"].values
        fct_probs = sub["forecast_prob"].values

        x = np.arange(len(temps))
        w = 0.38
        bars_m = ax.bar(x - w/2, mkt_probs, w, color=CLR_ACC, alpha=0.85, label="Market PMF")
        bars_f = ax.bar(x + w/2, fct_probs, w, color=CLR_WARN, alpha=0.85, label="Forecast PMF")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0f}°" for t in temps], color=CLR_MUTED, fontsize=8)
        ax.tick_params(colors=CLR_MUTED)
        city_disp = CITY_ALIASES.get(city_raw, city_raw)
        mu        = sub["forecast_mu"].iloc[0]
        days_fwd  = sub["days_ahead"].iloc[0]
        ax.set_title(f"{city_disp} · {tdate}\nforecast={mu:.1f}°C  +{days_fwd:.1f}d  KL={kl_val:.3f}",
                     color="white", fontsize=9, pad=4)
        ax.set_ylabel("Normalized prob", color=CLR_MUTED, fontsize=8)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d",
                  edgecolor=DARK_BDR, loc="upper right")

    plt.suptitle("Forecast PMF vs Market PMF — Highest Dislocations",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "distribution_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_edge_landscape(opps: pd.DataFrame, output_dir: Path):
    if opps.empty:
        return

    fig, axes = _dark_fig(2, 3, figsize=(18, 10))

    # 1. Edge distribution by city
    ax = axes[0, 0]
    city_colors = plt.cm.tab10.colors
    for i, (city, grp) in enumerate(opps.groupby("city")):
        ax.hist(grp["edge"], bins=20, alpha=0.7, color=city_colors[i % 10], label=city)
    ax.axvline(0, color=CLR_MUTED, lw=1, linestyle="--")
    ax.axvline(MIN_EDGE_FRACTION, color=CLR_WARN, lw=1.5, linestyle=":", label="threshold")
    ax.axvline(-MIN_EDGE_FRACTION, color=CLR_WARN, lw=1.5, linestyle=":")
    ax.set_title("Edge Distribution by City", color="white", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=DARK_BDR)

    # 2. Edge vs forecast horizon
    ax = axes[0, 1]
    yes_m = opps[opps["bet_side"]=="Yes"]
    no_m  = opps[opps["bet_side"]=="No"]
    ax.scatter(yes_m["days_ahead"], yes_m["edge"], c=CLR_YES, alpha=0.6, s=40,
               edgecolors="none", label="Bet Yes")
    ax.scatter(no_m["days_ahead"],  no_m["edge"].abs(),  c=CLR_NO,  alpha=0.6, s=40,
               edgecolors="none", label="Bet No (|edge|)")
    ax.set_title("Edge vs Horizon (days ahead)", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Edge")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#21262d", edgecolor=DARK_BDR)

    # 3. Mode shift (forecast_mu - market_mode) vs edge
    ax = axes[0, 2]
    sc = ax.scatter(opps["mode_shift_c"], opps["edge"],
                    c=opps["abs_edge"], cmap="plasma", alpha=0.7, s=40, edgecolors="none")
    ax.axhline(0, color=CLR_MUTED, lw=1, linestyle="--")
    ax.axvline(0, color=CLR_MUTED, lw=1, linestyle="--")
    ax.set_title("Forecast–Market Mode Shift vs Edge", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast μ − Market mode (°C)")
    ax.set_ylabel("Edge")
    plt.colorbar(sc, ax=ax, label="abs_edge").ax.tick_params(colors=CLR_MUTED)

    # 4. EV distribution
    ax = axes[1, 0]
    ax.hist(opps["ev_per_$"] * 100, bins=25, color=CLR_ACC, alpha=0.8)
    ax.axvline(0, color=CLR_MUTED, lw=1, linestyle="--")
    ax.set_title("Expected Value per $100 bet", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("EV ($)")

    # 5. Kelly fraction distribution
    ax = axes[1, 1]
    ax.hist(opps["kelly"], bins=20, color=CLR_WARN, alpha=0.8)
    ax.set_title("Kelly Fraction Distribution", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Kelly (25% fractional, 15% capped)")

    # 6. Liquidity vs edge scatter
    ax = axes[1, 2]
    ax.scatter(np.log10(opps["liquidity"] + 1), opps["abs_edge"],
               c=[CLR_YES if s=="Yes" else CLR_NO for s in opps["bet_side"]],
               alpha=0.6, s=40, edgecolors="none")
    ax.set_title("Liquidity vs |Edge|", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("log10(Liquidity USDC)")
    ax.set_ylabel("|Edge|")

    for ax in axes.flat:
        ax.tick_params(colors=CLR_MUTED)
        ax.xaxis.label.set_color(CLR_MUTED)
        ax.yaxis.label.set_color(CLR_MUTED)
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK_BDR)

    plt.suptitle("Polymarket Weather — Opportunity Landscape",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "edge_landscape.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_forecast_drift(data_dir: Path, city: str, output_dir: Path):
    try:
        daily = load_daily(data_dir, city)
    except FileNotFoundError:
        return

    fig, ax = _dark_fig(1, 1, figsize=(14, 5))
    ax = ax if not hasattr(ax, "__iter__") else ax

    dates = sorted(daily["date_local"].unique())[-7:]
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(dates)))

    for i, d in enumerate(dates):
        sub = daily[daily["date_local"] == d].sort_values("fetched_at_utc")
        if sub.empty: continue
        ax.plot(sub["fetched_at_utc"], sub["temp_max_c"],
                marker="o", markersize=4, color=palette[i], lw=1.8,
                label=str(d.date()) if hasattr(d, "date") else str(d)[:10])

    ax.set_title(f"{CITY_ALIASES.get(city, city)} — Forecast TempMax Drift Over Time",
                 color="white", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color=CLR_MUTED)
    ax.legend(title="Target date", fontsize=7, labelcolor="white",
              facecolor="#21262d", edgecolor=DARK_BDR, title_fontsize=7)
    ax.set_ylabel("°C", color=CLR_MUTED)

    plt.tight_layout()
    out = output_dir / f"drift_{city}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {out}")


# ─── BOT SCAFFOLD ──────────────────────────────────────────────────────────────

class WeatherBettingBot:
    """
    Betting bot scaffold. Replace execute_bet() with Polymarket CLOB API.

    Real integration:
      pip install py-clob-client
      from py_clob_client.client import ClobClient
      client = ClobClient("https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
      # For a YES bet:
      client.create_market_order(OrderArgs(
          token_id=YES_TOKEN_ID,
          price=market_prob,
          size=size_usdc / market_prob,
          side=BUY
      ))
    """

    def __init__(self, bankroll=1000.0, min_edge=MIN_EDGE_FRACTION,
                 min_liq=MIN_LIQUIDITY, dry_run=True):
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.min_liq  = min_liq
        self.dry_run  = dry_run
        self.log      = []

    def run(self, opps: pd.DataFrame):
        if opps.empty:
            print("  No opportunities to trade.")
            return

        # Only trade most recent snapshot per condition_id (avoid stale data)
        latest = (opps.sort_values("fetched_at")
                  .groupby("condition_id").last().reset_index())
        latest = latest[latest["days_ahead"] > 0]  # not yet settled

        print(f"\n{'─'*65}")
        mode = "DRY RUN" if self.dry_run else "⚠ LIVE"
        print(f"  WeatherBettingBot [{mode}]  bankroll=${self.bankroll:.0f}")
        print(f"{'─'*65}")

        placed, total_exp, total_ev = 0, 0.0, 0.0
        for _, row in latest.sort_values("score", ascending=False).iterrows():
            k    = float(row.get("kelly", 0))
            size = round(self.bankroll * k, 2)
            if size < 2.0:
                continue

            order = {
                "condition_id": row["condition_id"],
                "question":     row["question"],
                "bet_side":     row["bet_side"],
                "size_usdc":    size,
                "market_prob":  row["their_prob"],
                "forecast_prob":row["our_prob"],
                "edge":         row["abs_edge"],
                "ev_dollar":    round(row["ev_per_$"] * size, 2),
                "days_ahead":   row["days_ahead"],
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }
            self._execute(order)
            placed    += 1
            total_exp += size
            total_ev  += order["ev_dollar"]

        print(f"\n  Bets placed : {placed}")
        print(f"  Total size  : ${total_exp:.2f}")
        print(f"  Expected EV : ${total_ev:.2f}  ({total_ev/max(total_exp,1)*100:.1f}% ROI)")

    def _execute(self, order):
        self.log.append(order)
        if self.dry_run:
            ev_str = f"+${order['ev_dollar']:.2f}"
            print(f"  {'YES' if order['bet_side']=='Yes' else 'NO ':3s} "
                  f"${order['size_usdc']:6.2f}  "
                  f"edge={order['edge']:.1%}  "
                  f"EV≈{ev_str:>8}  "
                  f"{order['question'][:55]}…")
        else:
            raise NotImplementedError(
                "Wire in py_clob_client here. See docstring.")


# ─── REPORTING ─────────────────────────────────────────────────────────────────

def print_report(opps: pd.DataFrame):
    if opps.empty:
        print("\n  No opportunities above threshold.")
        return

    print(f"\n{'═'*80}")
    print(f"  TOP OPPORTUNITIES  ({len(opps)} total)")
    print(f"{'═'*80}")

    cols = ["city", "target_date", "days_ahead", "forecast_mu", "bin_temp_c",
            "forecast_prob", "market_prob", "edge", "bet_side", "kelly",
            "ev_per_$", "liquidity", "n_bins"]
    top = opps.head(25)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.width", 200)
    print(top[cols].to_string(index=False))

    print(f"\n  ── By City ──")
    for city, g in opps.groupby("city"):
        print(f"  {city:8s}  n={len(g):4d}  "
              f"avg|edge|={g['abs_edge'].mean():.1%}  "
              f"max|edge|={g['abs_edge'].max():.1%}  "
              f"total_EV=${(g['ev_per_$']*g['kelly']*1000).sum():.1f}")

    print(f"\n  ── Mode-shift anomalies (forecast moved ≥1°C from market mode) ──")
    shifted = opps[opps["mode_shift_c"].abs() >= 1.0]
    if not shifted.empty:
        print(shifted[["city","target_date","forecast_mu","market_mode_c",
                        "mode_shift_c","edge","bet_side","liquidity"]].head(15).to_string(index=False))
    else:
        print("  None found")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Weather Inefficiency Analyzer v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--cities",     nargs="+", default=None)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--min_edge",   type=float, default=MIN_EDGE_FRACTION)
    parser.add_argument("--min_liq",    type=float, default=MIN_LIQUIDITY)
    parser.add_argument("--bankroll",   type=float, default=1000.0)
    parser.add_argument("--no_plots",   action="store_true")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = args.cities or discover_cities(data_dir)
    if not cities:
        print(f"No *_snapshots.csv found in {data_dir}")
        return

    print(f"\nCities     : {cities}")
    print(f"Min edge   : {args.min_edge:.0%}  |  Min liquidity: ${args.min_liq:.0f}")
    print(f"Bankroll   : ${args.bankroll:.0f}")
    print(f"NWP sigmas : {NWP_SIGMA_BY_DAY}")

    all_raw  = []
    all_opps = []

    for city in cities:
        print(f"\n── {city.upper()} ──")
        df = analyze_city(data_dir, city)
        if df.empty:
            continue
        all_raw.append(df)
        opps = filter_opportunities(df, args.min_edge, args.min_liq)
        if not opps.empty:
            all_opps.append(opps)
            print(f"  → {len(opps)} opportunities")

        if not args.no_plots:
            plot_forecast_drift(data_dir, city, output_dir)

    if all_raw:
        combined_raw  = pd.concat(all_raw, ignore_index=True)
        combined_opps = pd.concat(all_opps, ignore_index=True) if all_opps else pd.DataFrame()

        if not args.no_plots:
            plot_distribution_comparison(combined_raw, output_dir)
            if not combined_opps.empty:
                plot_edge_landscape(combined_opps, output_dir)

        print_report(combined_opps)

        # Save outputs
        combined_raw.to_csv(output_dir / "all_bins.csv", index=False)
        if not combined_opps.empty:
            combined_opps.to_csv(output_dir / "opportunities_v2.csv", index=False)
            print(f"\n  Saved: {output_dir}/opportunities_v2.csv")

        # Run bot
        bot = WeatherBettingBot(
            bankroll=args.bankroll,
            min_edge=args.min_edge,
            min_liq=args.min_liq,
            dry_run=True,
        )
        bot.run(combined_opps if not combined_opps.empty else pd.DataFrame())
    else:
        print("\nNo data loaded.")

    print(f"\nDone → {output_dir.resolve()}")


if __name__ == "__main__":
    main()