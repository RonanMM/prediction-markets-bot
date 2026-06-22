import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from models import Opportunity
from config import MIN_EDGE, MIN_LIQUIDITY, CITY_NAMES
from signals import MOMENTUM_THRESHOLD, STALE_HOURS, INFORMED_RECENCY
from data_loader import load_daily

# ── Dark-theme colour palette ────────────────────────────────────────────────
_DARK   = "#0d1117"
_AX     = "#161b22"
_BORDER = "#30363d"
_MUTED  = "#8b949e"
_GREEN  = "#3fb950"
_RED    = "#f85149"
_BLUE   = "#58a6ff"
_YELLOW = "#e3b341"
_TAB    = plt.cm.tab10.colors

# Threshold for flagging internally-inconsistent markets
PMF_SUM_DEVIATION_MAX = 0.15

def _styled_ax(ax):
    ax.set_facecolor(_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    return ax


def _dark_fig(nrows=1, ncols=1, figsize=(14, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(_DARK)
    for ax in (np.array(axes).flat if hasattr(axes, "__iter__") else [axes]):
        _styled_ax(ax)
    return fig, axes


def plot_pmf_comparison(opps_df: pd.DataFrame, all_bins_df: pd.DataFrame,
                         output_dir: Path, top_n=6):
    """Side-by-side forecast vs market PMF for highest-dislocation days."""
    if opps_df.empty:
        return

    # pick top days by max abs_edge
    top_keys = (opps_df.groupby(["city", "target_date", "fetched_at"])
                ["abs_edge"].max().nlargest(top_n).index)

    ncols = min(3, len(top_keys))
    nrows = (len(top_keys) + ncols - 1) // ncols
    fig, axes = _dark_fig(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes_flat  = np.array(axes).flat

    for idx, (city, tdate, ftime) in enumerate(top_keys):
        ax  = next(axes_flat)
        sub = all_bins_df[(all_bins_df["city"] == city) &
                          (all_bins_df["target_date"] == tdate) &
                          (all_bins_df["fetched_at"] == ftime)].sort_values("bin_temp_c")
        if sub.empty:
            ax.set_visible(False); continue

        x  = np.arange(len(sub))
        w  = 0.38
        ax.bar(x - w/2, sub["market_prob"],   w, color=_BLUE,   alpha=0.85, label="Market")
        ax.bar(x + w/2, sub["forecast_prob"], w, color=_YELLOW, alpha=0.85, label="Forecast")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0f}°" for t in sub["bin_temp_c"]], fontsize=7)
        mu   = sub["forecast_mu"].iloc[0]
        days = sub["days_ahead"].iloc[0]
        mom  = sub["ema_momentum"].iloc[0] if "ema_momentum" in sub.columns else 0
        ax.set_title(
            f"{city} · {tdate}\nμ={mu:.1f}°C  +{days:.1f}d  mom={mom:+.2f}",
            color="white", fontsize=8, pad=3)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d",
                  edgecolor=_BORDER, loc="upper right")

    for ax in axes_flat:
        ax.set_visible(False)

    plt.suptitle("Forecast PMF vs Market PMF — Top Dislocations",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "pmf_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_alpha_dashboard(opps_df: pd.DataFrame, output_dir: Path):
    """3×3 grid of all alpha signal diagnostics."""
    if opps_df.empty:
        return

    fig, axes = _dark_fig(3, 3, figsize=(18, 13))
    c_yes = _GREEN; c_no = _RED

    def colors(col): return [c_yes if s=="Yes" else c_no for s in opps_df[col]]

    # 1. Edge distribution
    ax = axes[0, 0]
    for i, (city, g) in enumerate(opps_df.groupby("city")):
        ax.hist(g["edge"], bins=20, alpha=0.65, color=_TAB[i % 10], label=city)
    ax.axvline(0, color=_MUTED, lw=1, ls="--")
    ax.axvline( MIN_EDGE, color=_YELLOW, lw=1.5, ls=":", label=f"±{MIN_EDGE:.0%}")
    ax.axvline(-MIN_EDGE, color=_YELLOW, lw=1.5, ls=":")
    ax.set_title("Edge Distribution", color="white", fontweight="bold")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 2. α1 Momentum vs edge
    ax = axes[0, 1]
    sc = ax.scatter(opps_df["ema_momentum"], opps_df["edge"],
                    c=opps_df["abs_edge"], cmap="plasma", alpha=0.7,
                    s=40, edgecolors="none")
    ax.axvline(0, color=_MUTED, lw=1, ls="--")
    ax.axhline(0, color=_MUTED, lw=1, ls="--")
    plt.colorbar(sc, ax=ax, label="|edge|").ax.tick_params(colors=_MUTED)
    ax.set_title("α1 Momentum vs Edge", color="white", fontweight="bold")
    ax.set_xlabel("EMA momentum (°C/snap)")
    ax.set_ylabel("edge")

    # 3. α2 Sigma boost distribution
    ax = axes[0, 2]
    ax.hist(opps_df["sigma_boost"], bins=20, color=_BLUE, alpha=0.8)
    ax.set_title("α2 Sigma Boost (diurnal spread)", color="white", fontweight="bold")
    ax.set_xlabel("sigma boost (°C)")

    # 4. α3 forecast_sigma used
    ax = axes[1, 0]
    ax.scatter(opps_df["days_ahead"], opps_df["forecast_sigma"],
               c=[c_yes if s=="Yes" else c_no for s in opps_df["bet_side"]],
               alpha=0.6, s=40, edgecolors="none")
    ax.set_title("α3 Adaptive Sigma vs Horizon", color="white", fontweight="bold")
    ax.set_xlabel("days ahead")
    ax.set_ylabel("sigma (°C)")

    # 5. α5 PMF sum deviation
    ax = axes[1, 1]
    ax.scatter(opps_df["pmf_sum_dev"], opps_df["abs_edge"],
               c=opps_df["pmf_consistency"], cmap="RdYlGn",
               alpha=0.7, s=50, edgecolors="none", vmin=0, vmax=1)
    ax.axvline(PMF_SUM_DEVIATION_MAX, color=_YELLOW, lw=1.5, ls=":",
               label=f"max dev {PMF_SUM_DEVIATION_MAX}")
    ax.set_title("α5 Market Consistency vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("PMF sum deviation")
    ax.set_ylabel("|edge|")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 6. α6 Volume recency
    ax = axes[1, 2]
    ax.scatter(opps_df["volume_recency"], opps_df["abs_edge"],
               c=[c_yes if s=="Yes" else c_no for s in opps_df["bet_side"]],
               alpha=0.7, s=40, edgecolors="none")
    ax.axvline(INFORMED_RECENCY, color=_YELLOW, lw=1.5, ls=":",
               label=f"informed >{INFORMED_RECENCY}")
    ax.set_title("α6 Volume Recency vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("vol_24h / vol_total")
    ax.set_ylabel("|edge|")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 7. α7 Forecast variance vs edge
    ax = axes[2, 0]
    ax.scatter(opps_df["forecast_var"], opps_df["abs_edge"],
               c=opps_df["days_ahead"], cmap="viridis",
               alpha=0.7, s=40, edgecolors="none")
    ax.set_title("α7 Forecast Variance vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("forecast temp variance (°C²)")
    ax.set_ylabel("|edge|")

    # 8. α8 Hours since market move vs stale
    ax = axes[2, 1]
    stale    = opps_df[opps_df["is_stale"]]
    notstale = opps_df[~opps_df["is_stale"]]
    ax.scatter(notstale["hours_since_move"], notstale["abs_edge"],
               c=_BLUE, alpha=0.6, s=40, edgecolors="none", label="Fresh")
    ax.scatter(stale["hours_since_move"], stale["abs_edge"],
               c=_RED, alpha=0.8, s=60, edgecolors="none", label="Stale")
    ax.axvline(STALE_HOURS, color=_YELLOW, lw=1.5, ls=":", label=f"stale>{STALE_HOURS}h")
    ax.set_title("α8 Market Staleness vs |Edge|", color="white", fontweight="bold")
    ax.set_xlabel("hours since last significant move")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#21262d", edgecolor=_BORDER)

    # 9. Alpha score vs EV
    ax = axes[2, 2]
    sc = ax.scatter(opps_df["alpha_score"], opps_df["ev_per_dollar"] * 100,
                    c=opps_df["kelly"], cmap="hot",
                    alpha=0.7, s=50, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="kelly frac").ax.tick_params(colors=_MUTED)
    ax.axhline(0, color=_MUTED, lw=1, ls="--")
    ax.set_title("Composite Score vs EV/dollar", color="white", fontweight="bold")
    ax.set_xlabel("alpha_score")
    ax.set_ylabel("EV per $100 bet")

    plt.suptitle("Alpha Signal Dashboard", color="white",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "alpha_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_forecast_drift_all(data_dir: Path, cities: list[str], output_dir: Path):
    """One subplot per city showing forecast evolution by target date."""
    n = len(cities)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = _dark_fig(nrows, ncols, figsize=(7*ncols, 4*nrows))
    axes_flat  = np.array(axes).flat

    for city in cities:
        ax = next(axes_flat)
        try:
            daily = load_daily(data_dir, city)
        except FileNotFoundError:
            ax.set_visible(False); continue

        dates   = sorted(daily["date_local"].unique())[-7:]
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(dates)))
        for i, d in enumerate(dates):
            sub = daily[daily["date_local"] == d].sort_values("fetched_at_utc")
            if sub.empty: continue
            label = str(d.date()) if hasattr(d, "date") else str(d)[:10]
            ax.plot(sub["fetched_at_utc"], sub["temp_max_c"],
                    marker="o", ms=3, lw=1.6, color=palette[i], label=label)

        ax.set_title(f"{CITY_NAMES.get(city, city)} — Forecast Drift",
                     color="white", fontsize=9, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=6, labelcolor="white", facecolor="#21262d",
                  edgecolor=_BORDER, loc="upper left", ncol=2)
        ax.set_ylabel("°C max", fontsize=8)

    for ax in axes_flat:
        ax.set_visible(False)

    plt.suptitle("Forecast TempMax Evolution", color="white",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "forecast_drift_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")


def plot_momentum_heatmap(opps_df: pd.DataFrame, output_dir: Path):
    """Heatmap of EMA momentum by (city, target_date)."""
    if opps_df.empty or "ema_momentum" not in opps_df.columns:
        return

    pivot = (opps_df.groupby(["city", "target_date"])["ema_momentum"]
             .mean().unstack("target_date").fillna(0))
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.2), max(3, len(pivot)*0.8)))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_AX)

    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", color=_MUTED, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=_MUTED, fontsize=8)
    plt.colorbar(im, ax=ax, label="EMA momentum (°C/snap)").ax.tick_params(colors=_MUTED)
    ax.set_title("α1 Forecast Momentum by City × Date",
                 color="white", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "momentum_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    print(f"  Saved: {out}")





def opps_to_df(opps: list[Opportunity]) -> pd.DataFrame:
    if not opps:
        return pd.DataFrame()
    rows = []
    for o in opps:
        rows.append({k: v for k, v in o.__dict__.items()})
    return pd.DataFrame(rows)


def print_report(opps_df: pd.DataFrame):
    if opps_df.empty:
        print("\n  No opportunities above threshold.")
        return

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n{'═'*90}")
    print(f"  OPPORTUNITIES  —  {len(opps_df)} total  "
          f"(edge≥{MIN_EDGE:.0%}, liq≥{MIN_LIQUIDITY})")
    print(f"{'═'*90}")

    top_cols = ["city", "target_date", "days_ahead", "forecast_mu",
                "bin_temp_c" if "bin_temp_c" in opps_df.columns else "forecast_mu",
                "forecast_prob", "market_prob", "edge", "bet_side",
                "kelly", "ev_per_dollar", "alpha_score",
                "ema_momentum", "is_stale", "liquidity"]
    top_cols = [c for c in top_cols if c in opps_df.columns]
    print(opps_df.sort_values("alpha_score", ascending=False)
                 .head(30)[top_cols].to_string(index=False))

    print("\n  ── By City ──")
    for city, g in opps_df.groupby("city"):
        print(f"  {city:10s}  n={len(g):4d}  "
              f"avg|edge|={g['abs_edge'].mean():.1%}  "
              f"max|edge|={g['abs_edge'].max():.1%}  "
              f"stale={g['is_stale'].sum():3d}  "
              f"tot_EV≈${(g['ev_per_dollar']*g['kelly']*1000).sum():.0f}")

    print("\n  ── Momentum-aligned bets (α1) ──")
    mom_yes = opps_df[(opps_df["bet_side"]=="Yes") &
                      (opps_df["ema_momentum"] > MOMENTUM_THRESHOLD)]
    mom_no  = opps_df[(opps_df["bet_side"]=="No") &
                      (opps_df["ema_momentum"] < -MOMENTUM_THRESHOLD)]
    aligned = pd.concat([mom_yes, mom_no])
    if not aligned.empty:
        print(aligned[["city","target_date","question","edge",
                        "bet_side","ema_momentum","alpha_score"]]
              .sort_values("alpha_score", ascending=False).head(10).to_string(index=False))
    else:
        print("  None found")

    print("\n  ── Stale markets with large edge (α8) ──")
    stale = opps_df[opps_df["is_stale"] & (opps_df["abs_edge"] > 0.10)]
    if not stale.empty:
        print(stale[["city","target_date","question","edge",
                     "hours_since_move","alpha_score"]].head(8).to_string(index=False))
    else:
        print("  None found")

    print("\n  ── Incoherent markets (α5: PMF sum deviation) ──")
    incoherent = opps_df[opps_df["pmf_sum_dev"] > 0.15].copy()
    if not incoherent.empty:
        print(incoherent[["city","target_date","pmf_sum_dev",
                          "edge","bet_side"]].drop_duplicates(
                              ["city","target_date"]).head(8).to_string(index=False))
    else:
        print("  None found")

