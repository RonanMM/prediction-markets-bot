"""
visualization.py — Generates all plots for the Polymarket Weather Tracker.

Plot 1 (per city): Forecast max temp vs market-implied expected temperature.
Plot 2 (per market): Outcome probabilities + volume over time.
Plot 3 (per city): Forecast change vs probability shifts (delta view).
All plots saved to plots/ as PNG.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import COLORS, PLOT_DPI, PLOT_STYLE, PLOTS_DIR
from processing import (
    load_market_snapshots,
    load_price_history,
    load_weather_daily,
)

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE)

_ACCENT = "#E0E0E0"
_GRID   = "#333333"
_BG     = "#1A1A2E"

def _savefig(fig: plt.Figure, name: str) -> Path:
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(PLOTS_DIR) / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved plot → %s", path)
    return path


def _format_xaxis(ax, df: pd.DataFrame, date_col: str = "fetched_at_utc") -> None:
    """Auto-format x-axis date labels based on data span."""
    if df.empty:
        return
    span = (df[date_col].max() - df[date_col].min()).days
    if span <= 2:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    elif span <= 14:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors=_ACCENT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    if title:
        ax.set_title(title, color=_ACCENT, fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=_ACCENT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=_ACCENT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Forecast vs Market Implied Temperature
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_vs_market(city: str) -> Path | None:
    snaps   = load_market_snapshots(city)
    weather = load_weather_daily(city)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=_BG)
    has_data = False

    # ── Forecast line ──────────────────────────────────────────────────────
    if not weather.empty:
        # Latest-fetch forecast (most up-to-date)
        latest_fetch = weather["fetched_at_utc"].max()
        wx = weather[weather["fetched_at_utc"] == latest_fetch].sort_values("date_utc")
        if not wx.empty:
            ax.plot(
                wx["date_utc"],
                wx["temp_max_c"],
                color=COLORS["forecast"],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="Forecast Max Temp (°C)",
                zorder=3,
            )
            ax.fill_between(
                wx["date_utc"],
                wx["temp_min_c"],
                wx["temp_max_c"],
                alpha=0.15,
                color=COLORS["forecast"],
                label="Forecast Min–Max range",
            )
            has_data = True

    # ── Market implied temperature over time ───────────────────────────────
    if not snaps.empty and "implied_temp_c" in snaps.columns:
        valid = snaps.dropna(subset=["implied_temp_c"]).sort_values("fetched_at_utc")
        if not valid.empty:
            ax.scatter(
                valid["fetched_at_utc"],
                valid["implied_temp_c"],
                color=COLORS["implied"],
                s=60,
                zorder=5,
                label="Market Implied Temp (°C)",
            )
            ax.plot(
                valid["fetched_at_utc"],
                valid["implied_temp_c"],
                color=COLORS["implied"],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )
            has_data = True

    if not has_data:
        logger.warning("No data for Plot 1 – %s", city)
        plt.close(fig)
        return None

    _style_ax(ax, title=f"{city} — Forecast vs Market Implied Temperature",
              xlabel="Date (UTC)", ylabel="Temperature (°C)")
    _format_xaxis(ax, weather if not weather.empty else snaps)
    ax.legend(facecolor="#222", edgecolor=_GRID, labelcolor=_ACCENT, fontsize=9)

    return _savefig(fig, f"{city.lower().replace(' ', '_')}_plot1_forecast_vs_market.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Market Dynamics (probabilities + volume over time)
# ─────────────────────────────────────────────────────────────────────────────

def plot_market_dynamics(city: str) -> list[Path]:
    snaps = load_market_snapshots(city)
    if snaps.empty:
        logger.warning("No snapshot data for Plot 2 – %s", city)
        return []

    paths = []

    # One plot per unique market (condition_id)
    for cid, group in snaps.groupby("condition_id"):
        group = group.sort_values("fetched_at_utc")
        question = group["question"].iloc[0]

        # Collect all unique outcomes across snapshots
        all_outcomes: set[str] = set()
        for probs in group["outcome_probs"]:
            if isinstance(probs, dict):
                all_outcomes.update(probs.keys())

        if not all_outcomes:
            continue

        outcomes = sorted(all_outcomes)
        times    = group["fetched_at_utc"].tolist()

        fig, (ax_prob, ax_vol) = plt.subplots(
            2, 1, figsize=(13, 7), facecolor=_BG,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
        )

        # Probability lines
        for i, outcome in enumerate(outcomes):
            probs = [
                (row["outcome_probs"].get(outcome) if isinstance(row["outcome_probs"], dict) else None)
                for _, row in group.iterrows()
            ]
            color = COLORS["probs"][i % len(COLORS["probs"])]
            ax_prob.plot(
                times, probs,
                label=outcome,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        _style_ax(ax_prob, title=f"{city} — Market Dynamics\n{question[:80]}",
                  ylabel="Probability")
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_prob.legend(facecolor="#222", edgecolor=_GRID, labelcolor=_ACCENT,
                       fontsize=8, ncol=3, loc="upper left")
        ax_prob.tick_params(labelbottom=False)

        # Volume bar chart (bottom panel)
        volumes = pd.to_numeric(group["volume_24h_usdc"], errors="coerce").fillna(0)
        ax_vol.bar(times, volumes, color=COLORS["volume"], alpha=0.75, width=0.01)
        _style_ax(ax_vol, xlabel="Date (UTC)", ylabel="24h Volume (USDC)")
        _format_xaxis(ax_vol, group)

        fname = f"{city.lower().replace(' ','_')}_{cid[:12]}_plot2_dynamics.png"
        paths.append(_savefig(fig, fname))

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Forecast change vs Probability shifts (delta view)
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_view(city: str) -> Path | None:
    snaps   = load_market_snapshots(city)
    weather = load_weather_daily(city)

    if snaps.empty or weather.empty:
        logger.warning("Insufficient data for Plot 3 – %s", city)
        return None

    # Daily max forecast history for a single target date (most traded)
    # Use the earliest forecast date that has >1 fetch (to see change over time)
    wx_counts = weather.groupby("date_local")["fetched_at_utc"].nunique()
    if wx_counts.empty:
        return None

    target_date = wx_counts.idxmax()
    wx_target   = (
        weather[weather["date_local"] == target_date]
        .sort_values("fetched_at_utc")
    )

    # Market implied temp over time
    valid_snaps = snaps.dropna(subset=["implied_temp_c"]).sort_values("fetched_at_utc")
    if valid_snaps.empty or wx_target.empty:
        return None

    # Compute deltas relative to first fetch
    wx_delta = wx_target["temp_max_c"] - wx_target["temp_max_c"].iloc[0]

    # Implied temp delta
    impl_delta = valid_snaps["implied_temp_c"] - valid_snaps["implied_temp_c"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=_BG)

    ax.plot(
        wx_target["fetched_at_utc"], wx_delta,
        color=COLORS["forecast"], linewidth=2, marker="s", markersize=5,
        label=f"Forecast change (target: {target_date})"
    )
    ax.plot(
        valid_snaps["fetched_at_utc"], impl_delta,
        color=COLORS["implied"], linewidth=2, linestyle="--", marker="o", markersize=5,
        label="Market implied temp change"
    )
    ax.axhline(0, color=_ACCENT, linewidth=0.8, linestyle=":")
    ax.fill_between(
        wx_target["fetched_at_utc"], wx_delta, alpha=0.1, color=COLORS["forecast"]
    )

    _style_ax(ax,
              title=f"{city} — Δ Forecast vs Δ Market Implied (relative to first fetch)",
              xlabel="Date (UTC)", ylabel="ΔTemperature (°C)")
    _format_xaxis(ax, wx_target)
    ax.legend(facecolor="#222", edgecolor=_GRID, labelcolor=_ACCENT, fontsize=9)

    return _savefig(fig, f"{city.lower().replace(' ','_')}_plot3_deltas.png")


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency signal scatter (bonus)
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency_signal(summaries: list[dict]) -> Path | None:
    """
    Bar chart of (market implied − forecast) for all cities.
    Highlights potential market inefficiencies.
    """
    cities = []
    diffs  = []
    colors = []

    for s in summaries:
        d = s.get("difference_c")
        if d is None:
            continue
        cities.append(s["city"])
        diffs.append(d)
        colors.append(COLORS["implied"] if d > 0 else COLORS["forecast"])

    if not cities:
        return None

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
    bars = ax.bar(cities, diffs, color=colors, edgecolor=_GRID, linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, diffs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.05 if val >= 0 else -0.12),
            f"{val:+.1f}°C",
            ha="center", va="bottom" if val >= 0 else "top",
            color=_ACCENT, fontsize=9,
        )

    ax.axhline(0, color=_ACCENT, linewidth=1, linestyle="--", alpha=0.5)
    _style_ax(ax,
              title="Market Efficiency Signal\n(Implied Temp − Forecast Max Temp)",
              xlabel="City", ylabel="Difference (°C)")
    ax.tick_params(axis="x", colors=_ACCENT, labelsize=10)

    return _savefig(fig, "efficiency_signal_all_cities.png")


# ─────────────────────────────────────────────────────────────────────────────
# Generate all plots for a city
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(city: str) -> list[Path]:
    saved = []

    p1 = plot_forecast_vs_market(city)
    if p1:
        saved.append(p1)

    p2_list = plot_market_dynamics(city)
    saved.extend(p2_list)

    p3 = plot_delta_view(city)
    if p3:
        saved.append(p3)

    return saved
