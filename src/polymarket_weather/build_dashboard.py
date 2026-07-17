#!/usr/bin/env python3
"""build_dashboard.py — render the live status dashboard as a self-contained HTML page.

Two data paths:
  * SCALARS  — parsed from data_status.py / evaluate_oos.py / shoulder_book.py --report stdout.
  * SERIES   — computed in-process by importing evaluate_oos._graded_markets (per-market model /
    market / outcome) + reading snapshot timestamps, then embedded as JSON and drawn as
    interactive SVG charts (no external libraries — CSP-safe for GitHub Pages / any static host).

    python build_dashboard.py                # writes ../../site/index.html
    python build_dashboard.py /path/out.html # custom path

Every value degrades to "—" / an empty chart on a parse or import miss — the page must always
render. Keep in sync with the published artifact template.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent          # src/polymarket_weather
REPO = PKG.parents[1]                           # repo root
OUT_DIR = PKG / "output"
DEFAULT_OUT = REPO / "site" / "index.html"

CITY_META = {
    "Seoul": ("Seoul", "RKSI"), "London": ("London", "EGLC"), "Chicago": ("Chicago", "KORD"),
    "NYC": ("New York", "KLGA"), "HongKong": ("Hong Kong", "HKO"),
}
CITY_ORDER = ["Seoul", "London", "Chicago", "NYC", "HongKong"]
BRIER_SCALE = 0.18


# ────────────────────────────── scalars (stdout parse) ──────────────────────────────
def _run(args: list[str]) -> str:
    try:
        r = subprocess.run([sys.executable, *args], cwd=PKG,
                           capture_output=True, text=True, timeout=600)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"[build_dashboard: {' '.join(args)} failed: {e}]"


def gather() -> dict:
    status = _run(["data_status.py"])
    oos = _run(["evaluate_oos.py"])
    book = _run(["shoulder_book.py", "--report"])
    d: dict = {}

    span = re.search(r"Date span.*?:\s*([\d-]+)\s*\.\.\s*([\d-]+)", status)
    if span:
        d["span_start"], d["span_end"] = span.group(1), span.group(2)
    gm = re.search(r"gradable markets\s+(\d+)\s*/\s*(\d+)\s*\[(\w+)\]", status)
    gb = re.search(r"gradable bets\s+(\d+)\s*/\s*(\d+)\s*\[(\w+)\]", status)
    if gm:
        d["gate_mkts"], d["gate_mkts_thr"], d["gate_status"] = gm.group(1), gm.group(2), gm.group(3)
    if gb:
        d["gate_bets"], d["gate_bets_thr"] = gb.group(1), gb.group(2)

    d["br_market"] = _search(r"MARKET\s+([\d.]+)\s+[\d.]+", oos)
    d["br_model"] = _search(r"MODEL\s+([\d.]+)\s+[\d.]+", oos)
    d["br_ens"] = _search(r"ENSEMBLE\s+([\d.]+)\s+[\d.]+", oos)
    d["n_mkts"] = _search(r"Gradable markets:\s*(\d+)", oos) or d.get("gate_mkts")
    crps = re.search(r"Temperature CRPS.*?MODEL\s+([\d.]+).*?ENSEMBLE\s+([\d.]+)", oos, re.DOTALL)
    if crps:
        d["crps_model"], d["crps_ens"] = crps.group(1), crps.group(2)
    e3 = re.search(r"ON-book:\s*n=(\d+).*?ROI\s+([+\-][\d.]+%)", oos)
    if e3:
        d["e3_n"], d["e3_roi"] = e3.group(1), e3.group(2)

    hd = re.search(r"STRUCTURE PAPER BOOK:\s*(\d+)\s+entries\s+\((\d+)\s+graded,\s*(\d+)\s+awaiting", book)
    if hd:
        d["sb_entries"], d["sb_graded"], d["sb_await"] = hd.group(1), hd.group(2), hd.group(3)
    full = re.search(r"shoulder full \[5,35\)¢:\s*n=(\d+)\s+wr\s+(\d+)%.*?taker\s+([+\-][\d.]+)", book)
    if full:
        d["sb_full_n"], d["sb_wr"], d["sb_full"] = full.group(1), full.group(2), full.group(3)
    core = re.search(r"shoulder core \[20,35\)¢:.*?taker\s+([+\-][\d.]+)", book)
    if core:
        d["sb_core"] = core.group(1)
    return d


def _search(pattern: str, text: str, default=None):
    m = re.search(pattern, text)
    return m.group(1) if m else default


# ────────────────────────────── series (in-process) ──────────────────────────────
def _sample(items: list, n: int = 55) -> list:
    """Evenly downsample a list to <= n items, always keeping the last."""
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)} | {len(items) - 1})
    return [items[i] for i in idx]


def compute_series() -> dict:
    out: dict = {"acc": [], "city": [], "calib": [], "growth": [], "heartbeat": []}
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))
    try:
        import numpy as np
        import pandas as pd
        import evaluate_oos as ev

        cal = ev._graded_markets(OUT_DIR / "opportunities_evaluation_calibrated.csv")
        mkt_col = "market_prob_raw" if "market_prob_raw" in cal.columns else "market_prob"
        cal = cal.copy()
        cal["td"] = pd.to_datetime(cal["target_date"], errors="coerce")
        cal = cal.dropna(subset=["td"])
        cal["b_model"] = (cal["forecast_prob"] - cal["outcome"]) ** 2
        cal["b_mkt"] = (cal[mkt_col] - cal["outcome"]) ** 2

        ens = ev._graded_markets(OUT_DIR / "opportunities_evaluation_ensemble.csv")
        ens = ens.copy()
        ens["td"] = pd.to_datetime(ens["target_date"], errors="coerce")
        ens = ens.dropna(subset=["td"])
        ens["b_ens"] = (ens["forecast_prob"] - ens["outcome"]) ** 2

        # accuracy over time — cumulative Brier on the COMMON set (markets scored by all three),
        # so model / market / ensemble are directly comparable at every date. Skip the noisy head
        # where the running sample is tiny.
        common = set(cal["condition_id"]) & set(ens["condition_id"])
        c = cal[cal["condition_id"].isin(common)].copy()
        eprob = ens.drop_duplicates("condition_id").set_index("condition_id")["forecast_prob"]
        c["b_ens"] = (c["condition_id"].map(eprob) - c["outcome"]) ** 2
        acc = []
        for dt in sorted(c["td"].unique()):
            w = c[c["td"] <= dt]
            if len(w) < 10:
                continue
            acc.append({
                "t": pd.Timestamp(dt).strftime("%b %-d"),
                "model": round(float(w["b_model"].mean()), 4),
                "market": round(float(w["b_mkt"].mean()), 4),
                "ens": round(float(w["b_ens"].mean()), 4),
                "n": int(len(w)),
            })
        out["acc"] = _sample(acc)

        # accuracy by city
        city = []
        for c, g in cal.groupby("city"):
            city.append({"city": str(c), "model": round(float(g["b_model"].mean()), 4),
                         "market": round(float(g["b_mkt"].mean()), 4), "n": int(len(g))})
        city.sort(key=lambda r: r["n"], reverse=True)
        out["city"] = city

        # calibration — model predicted P(YES) vs realized frequency, 10 bins
        p = cal["forecast_prob"].to_numpy(dtype=float)
        y = cal["outcome"].to_numpy(dtype=float)
        edges = np.linspace(0, 1, 11)
        calib = []
        for i in range(10):
            hi = p <= edges[i + 1] if i == 9 else p < edges[i + 1]
            mask = (p >= edges[i]) & hi
            if mask.sum() >= 3:   # drop single-market bins — too noisy to plot
                calib.append({"p": round(float(p[mask].mean()), 3),
                              "f": round(float(y[mask].mean()), 3), "n": int(mask.sum())})
        out["calib"] = calib

        # track-record growth — cumulative gradable markets by date, vs the pre-committed gate
        gd = cal.groupby("td").agg(mkts=("condition_id", "nunique")).sort_index()
        gd["cum_m"] = gd["mkts"].cumsum()
        out["growth"] = _sample([{"t": pd.Timestamp(idx).strftime("%b %-d"), "m": int(r.cum_m)}
                                 for idx, r in gd.iterrows()])
        try:
            import data_status as ds
            out["gate_line"] = int(getattr(ds, "GATE_RESOLVED_MARKETS", 150))
        except Exception:
            out["gate_line"] = 150
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    # collection heartbeat — distinct collection hours per day (last 30d)
    try:
        import pandas as pd
        allts = []
        for f in glob.glob(str(PKG / "data" / "polymarket" / "*_snapshots.csv")):
            try:
                allts.append(pd.read_csv(f, usecols=["fetched_at_utc"])["fetched_at_utc"])
            except Exception:
                continue
        if allts:
            ts = pd.to_datetime(pd.concat(allts), utc=True, errors="coerce").dropna()
            floored = ts.dt.floor("h")
            by_day = pd.DataFrame({"day": floored.dt.date, "hr": floored})
            cyc = by_day.groupby("day")["hr"].nunique().sort_index()
            out["heartbeat"] = [{"d": str(day), "n": int(v)} for day, v in cyc.items()][-30:]
    except Exception as e:
        out["hb_error"] = f"{type(e).__name__}: {e}"

    return out


# ────────────────────────────── formatting helpers ──────────────────────────────
def _fmt_span(d: dict) -> tuple[str, str]:
    try:
        s = datetime.strptime(d["span_start"], "%Y-%m-%d")
        e = datetime.strptime(d["span_end"], "%Y-%m-%d")
        return str((e - s).days), f"{s.strftime('%b %-d')} → {e.strftime('%b %-d')} '{e.strftime('%y')}"
    except Exception:
        return "—", "—"


def _data_through() -> str:
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%cI"], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        iso = r.stdout.strip()
        if iso:
            return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render(d: dict, series: dict) -> str:
    days, span = _fmt_span(d)
    G = lambda k, dflt="—": str(d.get(k, dflt))
    try:
        model_wins = float(d["br_model"]) < float(d["br_market"])
    except Exception:
        model_wins = False
    edge_chip = ('<span class="chip good">AHEAD</span>' if model_wins
                 else '<span class="chip warn">NOT YET</span>')
    if model_wins:
        takeaway = ("Our calibrated forecast is now predicting settlements <b>more accurately than "
                    "the market price</b> — the lines below have crossed.")
    else:
        takeaway = ("The market price still predicts settlements <b>more accurately</b> than our "
                    "forecast — the gap is narrowing but hasn't closed. Shown on purpose: no edge "
                    "is claimed until the model line drops below the market line.")

    repl = {
        "UPDATED": _data_through(),
        "GATE_STATUS": G("gate_status"), "GATE_MKTS": G("gate_mkts"), "GATE_BETS": G("gate_bets"),
        "GATE_MKTS_THR": G("gate_mkts_thr", "150"),
        "DATA_DAYS": days, "SPAN": span,
        "SB_WR": G("sb_wr"), "SB_GRADED": G("sb_graded"), "SB_ENTRIES": G("sb_entries"),
        "BR_MARKET": G("br_market"), "BR_ENS": G("br_ens"), "BR_MODEL": G("br_model"),
        "N_MKTS": G("n_mkts"), "CRPS_MODEL": G("crps_model"), "CRPS_ENS": G("crps_ens"),
        "EDGE_CHIP": edge_chip, "TAKEAWAY": takeaway,
        "SB_FULL": G("sb_full"), "SB_CORE": G("sb_core"), "SB_AWAIT": G("sb_await"),
        "E3_ROI": G("e3_roi"), "E3_N": G("e3_n"),
        "CITIES_HTML": _cities_html(series.get("city", []), d),
        "CHART_DATA": json.dumps(series),
    }
    html = TEMPLATE
    for k, v in repl.items():
        html = html.replace(f"%%{k}%%", str(v))
    return html


def _cities_html(city_series: list, d: dict) -> str:
    # freshness comes from data_status; fall back gracefully if absent
    status = _run(["data_status.py"]) if not hasattr(_cities_html, "_cache") else _cities_html._cache
    _cities_html._cache = status
    lag = {n: int(v) for n, v in re.findall(r"(\w+)\s+latest obs\s+[\d-]+\s+\((\d+)d behind", status)}
    out = []
    for key in CITY_ORDER:
        disp, station = CITY_META[key]
        days = lag.get(key)
        chip = ('<span class="chip">—</span>' if days is None
                else f'<span class="chip {"good" if days <= 14 else "warn"}">{days}d behind</span>')
        out.append(f'<div class="city"><div class="cn">{disp}</div>'
                   f'<div class="cc mono">{station}</div>{chip}</div>')
    return "\n      ".join(out)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(gather(), compute_series()), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


TEMPLATE = r"""<title>Prediction Markets Bot — Live Dashboard</title>
<style>
  :root {
    --bg: #eef1f5; --surface: #ffffff; --surface-2: #f3f6f9; --border: #dde3ea; --border-strong: #c2ccd8;
    --ink: #131922; --ink-2: #55616f; --ink-3: #8592a1;
    --accent: #0b7f9e; --accent-ink: #075a70;
    --good: #1a7f37; --good-bg: rgba(26,127,55,.10);
    --warn: #9a6700; --warn-bg: rgba(154,103,0,.10);
    --model: #0e91b0; --market: #d97416; --ens: #7a8797;
    --grid: #e6ebf1; --shadow: 0 1px 2px rgba(16,24,40,.06), 0 10px 26px rgba(16,24,40,.05);
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Roboto, sans-serif;
    --font-mono: "SF Mono", "JetBrains Mono", "Roboto Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) { :root {
    --bg: #090d13; --surface: #111823; --surface-2: #18202d; --border: #232f40; --border-strong: #33425a;
    --ink: #e7edf4; --ink-2: #9fadbf; --ink-3: #647184;
    --accent: #48cdea; --accent-ink: #7bdcf1;
    --good: #4bc367; --good-bg: rgba(75,195,103,.13);
    --warn: #e2af3f; --warn-bg: rgba(226,175,63,.13);
    --model: #35c4e0; --market: #f0932f; --ens: #8b97a7;
    --grid: #1e2836; --shadow: 0 1px 2px rgba(0,0,0,.4), 0 14px 34px rgba(0,0,0,.4);
  } }
  :root[data-theme="light"] {
    --bg: #eef1f5; --surface: #ffffff; --surface-2: #f3f6f9; --border: #dde3ea; --border-strong: #c2ccd8;
    --ink: #131922; --ink-2: #55616f; --ink-3: #8592a1; --accent: #0b7f9e; --accent-ink: #075a70;
    --good: #1a7f37; --good-bg: rgba(26,127,55,.10); --warn: #9a6700; --warn-bg: rgba(154,103,0,.10);
    --model: #0e91b0; --market: #d97416; --ens: #7a8797; --grid: #e6ebf1;
    --shadow: 0 1px 2px rgba(16,24,40,.06), 0 10px 26px rgba(16,24,40,.05);
  }
  :root[data-theme="dark"] {
    --bg: #090d13; --surface: #111823; --surface-2: #18202d; --border: #232f40; --border-strong: #33425a;
    --ink: #e7edf4; --ink-2: #9fadbf; --ink-3: #647184; --accent: #48cdea; --accent-ink: #7bdcf1;
    --good: #4bc367; --good-bg: rgba(75,195,103,.13); --warn: #e2af3f; --warn-bg: rgba(226,175,63,.13);
    --model: #35c4e0; --market: #f0932f; --ens: #8b97a7; --grid: #1e2836;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 14px 34px rgba(0,0,0,.4);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-sans); -webkit-font-smoothing: antialiased; line-height: 1.5; }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 30px 20px 64px; }
  .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .eyebrow { font-family: var(--font-mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; }
  h1,h2 { text-wrap: balance; }
  .topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; padding-bottom: 18px; margin-bottom: 24px; border-bottom: 1px solid var(--border); }
  .brand { display: flex; align-items: center; gap: 11px; margin-right: auto; }
  .brand-mark { width: 38px; height: 38px; border-radius: 9px; flex: none; background: linear-gradient(145deg, var(--accent), var(--accent-ink)); display: grid; place-items: center; font-size: 20px; box-shadow: var(--shadow); }
  .brand h1 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -.01em; }
  .brand p { margin: 1px 0 0; font-size: 12.5px; color: var(--ink-2); }
  .updated { text-align: right; font-size: 12px; color: var(--ink-3); }
  .updated b { color: var(--ink-2); font-weight: 600; }
  .pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px 5px 9px; border-radius: 100px; font-size: 12.5px; font-weight: 650; border: 1px solid transparent; white-space: nowrap; }
  .pill.good { color: var(--good); background: var(--good-bg); border-color: color-mix(in srgb, var(--good) 30%, transparent); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex: none; }
  .dot.live { animation: pulse 2.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--good) 55%, transparent); } 50% { box-shadow: 0 0 0 5px transparent; } }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 11px; font-weight: 600; letter-spacing: .02em; padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border-strong); color: var(--ink-2); white-space: nowrap; }
  .chip.good { color: var(--good); border-color: color-mix(in srgb, var(--good) 34%, transparent); background: var(--good-bg); }
  .chip.warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 34%, transparent); background: var(--warn-bg); }
  .chip.accent { color: var(--accent-ink); border-color: color-mix(in srgb, var(--accent) 40%, transparent); background: color-mix(in srgb, var(--accent) 12%, transparent); }
  section { margin-top: 26px; }
  .sec-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
  .sec-head h2 { margin: 0; font-size: 15.5px; font-weight: 680; letter-spacing: -.005em; }
  .sec-head .note { font-size: 12.5px; color: var(--ink-3); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; box-shadow: var(--shadow); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 15px 16px; box-shadow: var(--shadow); }
  .kpi .label { font-size: 11.5px; color: var(--ink-2); font-weight: 600; margin-bottom: 8px; }
  .kpi .big { font-family: var(--font-mono); font-size: 25px; font-weight: 680; letter-spacing: -.02em; line-height: 1.05; }
  .kpi .big.g { color: var(--good); }
  .kpi .sub { font-size: 12px; color: var(--ink-3); margin-top: 5px; font-family: var(--font-mono); }
  .takeaway { font-size: 13.5px; color: var(--ink-2); margin: 2px 0 4px; max-width: 74ch; }
  .takeaway b { color: var(--ink); font-weight: 640; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 2px 0 10px; }
  .lg { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--ink-2); font-weight: 550; }
  .lg i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .chartwrap { position: relative; width: 100%; }
  svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  .axis { fill: var(--ink-3); font-family: var(--font-mono); font-size: 11px; }
  .serieslabel { font-family: var(--font-mono); font-size: 11px; font-weight: 650; }
  .tip { position: absolute; pointer-events: none; opacity: 0; transition: opacity .12s; background: var(--surface); border: 1px solid var(--border-strong); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: var(--shadow); z-index: 5; min-width: 116px; }
  .tip .tt { font-family: var(--font-mono); font-size: 11px; color: var(--ink-3); margin-bottom: 5px; }
  .tip .tr { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .tip .tr span { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); }
  .tip .tr i { width: 9px; height: 9px; border-radius: 2px; }
  .tip .tr b { color: var(--ink); font-weight: 680; }
  .strip { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--surface); box-shadow: var(--shadow); }
  .strip > div { padding: 14px 16px; border-right: 1px solid var(--border); }
  .strip > div:last-child { border-right: none; }
  .strip .n { font-family: var(--font-mono); font-size: 20px; font-weight: 680; }
  .strip .n.pos { color: var(--good); }
  .strip .k { font-size: 11.5px; color: var(--ink-2); margin-top: 3px; }
  .strip .k small { color: var(--ink-3); }
  .lead { font-size: 13.5px; color: var(--ink-2); margin: 0 0 12px; max-width: 74ch; }
  .lead b { color: var(--ink); font-weight: 640; }
  .cities { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .city { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 13px; box-shadow: var(--shadow); }
  .city .cn { font-size: 13.5px; font-weight: 660; margin-bottom: 3px; }
  .city .cc { font-family: var(--font-mono); font-size: 11px; color: var(--ink-3); margin-bottom: 9px; }
  footer { margin-top: 38px; padding-top: 18px; border-top: 1px solid var(--border); font-size: 12px; color: var(--ink-3); line-height: 1.7; }
  footer b { color: var(--ink-2); font-weight: 600; }
  footer .row { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 10px; }
  @media (max-width: 820px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .grid2 { grid-template-columns: 1fr; }
    .strip { grid-template-columns: repeat(2, 1fr); }
    .strip > div:nth-child(2) { border-right: none; }
    .strip > div:nth-child(1), .strip > div:nth-child(2) { border-bottom: 1px solid var(--border); }
    .cities { grid-template-columns: repeat(2, 1fr); }
  }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } .chart * { transition: none !important; } }
</style>

<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">📊</div>
      <div><h1>Prediction Markets Bot</h1><p>Autonomous Polymarket weather-market tracker · runs in the cloud · 5 cities</p></div>
    </div>
    <span class="pill good"><span class="dot live"></span>OPERATIONAL</span>
    <div class="updated">data through<br><b class="mono">%%UPDATED%%</b></div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Collector</div><div class="big g">LIVE</div><div class="sub">every 2h · GitHub Actions</div></div>
    <div class="kpi"><div class="label">Track-record gate</div><div class="big g">%%GATE_STATUS%%</div><div class="sub">%%GATE_MKTS%% mkts · %%GATE_BETS%% bets</div></div>
    <div class="kpi"><div class="label">Data collected</div><div class="big">%%DATA_DAYS%%<span style="font-size:14px;color:var(--ink-3)"> days</span></div><div class="sub">%%SPAN%%</div></div>
    <div class="kpi"><div class="label">Paper book win rate</div><div class="big">%%SB_WR%%<span style="font-size:14px;color:var(--ink-3)">%</span></div><div class="sub">%%SB_GRADED%% graded / %%SB_ENTRIES%% open</div></div>
  </div>

  <section>
    <div class="sec-head"><h2>Is the model catching up to the market?</h2>%%EDGE_CHIP%%</div>
    <div class="card">
      <p class="takeaway">%%TAKEAWAY%%</p>
      <div class="legend">
        <span class="lg"><i style="background:var(--market)"></i>Market price</span>
        <span class="lg"><i style="background:var(--model)"></i>Our model</span>
        <span class="lg"><i style="background:var(--ens)"></i>Raw ensemble</span>
        <span class="lg" style="margin-left:auto;color:var(--ink-3)">cumulative Brier — lower = more accurate</span>
      </div>
      <div class="chartwrap"><div id="c_acc"></div></div>
    </div>
  </section>

  <section class="grid2">
    <div class="card">
      <div class="sec-head" style="margin-bottom:4px"><h2>Accuracy by city</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--market)"></i>Market</span><span class="lg"><i style="background:var(--model)"></i>Model</span><span class="lg" style="margin-left:auto;color:var(--ink-3)">Brier — lower = better</span></div>
      <div class="chartwrap"><div id="c_city"></div></div>
    </div>
    <div class="card">
      <div class="sec-head" style="margin-bottom:4px"><h2>Model calibration</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--model)"></i>Model</span><span class="lg" style="margin-left:auto;color:var(--ink-3)">on the dashed line = perfectly calibrated</span></div>
      <div class="chartwrap"><div id="c_calib"></div></div>
    </div>
  </section>

  <section class="grid2">
    <div class="card">
      <div class="sec-head" style="margin-bottom:4px"><h2>Track record</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--accent)"></i>Gradable markets</span><span class="lg" style="margin-left:auto;color:var(--ink-3)">cumulative vs station truth · dashed = %%GATE_MKTS_THR%%-market gate</span></div>
      <div class="chartwrap"><div id="c_growth"></div></div>
    </div>
    <div class="card">
      <div class="sec-head" style="margin-bottom:4px"><h2>Collection heartbeat</h2><span class="note">runs / day, last 30d</span></div>
      <div class="chartwrap"><div id="c_hb"></div></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="eyebrow">The candidate edge · forward paper trial</span></div>
    <div class="sec-head" style="margin-bottom:12px"><h2>Structure paper book</h2><span class="chip accent">PAPER ONLY — NO REAL MONEY</span></div>
    <p class="lead">A <b>model-free</b> strategy that ignores the weather forecast entirely: it sells over-priced long-shot temperature bins and buys clear favorites, then settles against real results. Early signal is strong, but it stays paper until it clears a pre-registered forward test.</p>
    <div class="strip">
      <div><div class="n pos">%%SB_WR%%%</div><div class="k">win rate <small>· %%SB_GRADED%% settled</small></div></div>
      <div><div class="n pos">%%SB_FULL%%</div><div class="k">edge / contract <small>· full band</small></div></div>
      <div><div class="n pos">%%SB_CORE%%</div><div class="k">edge / contract <small>· core band</small></div></div>
      <div><div class="n">%%SB_AWAIT%%</div><div class="k">awaiting <small>· settlement</small></div></div>
    </div>
    <p class="lead" style="margin-top:12px;margin-bottom:0">A second signal — the model's most selective bucket — shows <b>%%E3_ROI%% ROI on %%E3_N%% bets</b>, but that's still in-sample. Nothing goes live until a bucket logs <b>40+ forward bets</b> beating the market. Discipline first.</p>
  </section>

  <section>
    <div class="sec-head"><h2>Cities &amp; data freshness</h2><span class="note">how far behind the settlement feed is per city</span></div>
    <div class="cities">
      %%CITIES_HTML%%
    </div>
  </section>

  <footer>
    <div class="row"><span><b>Pipeline:</b> collect every 2h · truth-eval daily · retrain on demand</span><span><b>Host:</b> GitHub Actions (free tier)</span><span><b>Data store:</b> git</span></div>
    <div><b>Method:</b> every prediction is graded against the actual weather-station reading each market settles on — no grading a forecast against the same grid it came from. No performance is claimed until a pre-committed sample gate is met (now cleared). All trading shown is paper; no real orders are placed.</div>
    <div style="margin-top:10px;color:var(--ink-3)">Auto-generated from the live data · rebuilt every few hours.</div>
  </footer>
</div>

<script id="D" type="application/json">%%CHART_DATA%%</script>
<script>
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";
  var D = {};
  try { D = JSON.parse(document.getElementById("D").textContent || "{}"); } catch (e) { D = {}; }
  var css = function (v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); };
  function el(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function niceTop(v) { // round a max up to a clean gridline
    if (!(v > 0)) return 1;
    var mag = Math.pow(10, Math.floor(Math.log10(v)));
    var n = v / mag;
    var step = n <= 1 ? 1 : n <= 1.5 ? 1.5 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 3 ? 3 : n <= 5 ? 5 : 10;
    return step * mag;
  }
  // ---- shared line/area chart ----
  function lineChart(id, opts) {
    var host = document.getElementById(id);
    if (!host) return;
    var W = Math.max(host.clientWidth || 700, 320), H = W < 520 ? 232 : (opts.h || 290), P = { l: 46, r: 54, t: 14, b: 30 };
    var series = opts.series.filter(function (s) { return s.points.some(function (p) { return p != null; }); });
    if (!series.length) { host.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; return; }
    var n = opts.xLabels.length;
    var yMin = opts.yMin != null ? opts.yMin : 0;
    var yMax = opts.yMax != null ? opts.yMax : niceTop(Math.max.apply(null, series.flatMap(function (s) { return s.points.filter(function (v) { return v != null; }); })) * 1.08);
    var xOf = function (i) { return P.l + (n <= 1 ? 0 : (W - P.l - P.r) * i / (n - 1)); };
    var yOf = function (v) { return P.t + (H - P.t - P.b) * (1 - (v - yMin) / (yMax - yMin)); };
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    // y gridlines + labels
    var ticks = 4;
    for (var g = 0; g <= ticks; g++) {
      var yv = yMin + (yMax - yMin) * g / ticks, yy = yOf(yv);
      svg.appendChild(el("line", { class: "gridline", x1: P.l, y1: yy, x2: W - P.r, y2: yy }));
      svg.appendChild(el("text", { class: "axis", x: P.l - 8, y: yy + 3.5, "text-anchor": "end" }, opts.yFmt ? opts.yFmt(yv) : yv.toFixed(2)));
    }
    // diagonal reference (calibration)
    if (opts.diagonal) {
      svg.appendChild(el("line", { x1: xOf(0), y1: yOf(0), x2: xOf(n - 1), y2: yOf(yMax), stroke: css("--ink-3"), "stroke-width": 1, "stroke-dasharray": "4 4", opacity: .55 }));
    }
    // horizontal reference line (e.g. the sample-size gate)
    if (opts.refLine && opts.refLine.y <= yMax) {
      var ry = yOf(opts.refLine.y);
      svg.appendChild(el("line", { x1: P.l, y1: ry, x2: W - P.r, y2: ry, stroke: css("--warn"), "stroke-width": 1.2, "stroke-dasharray": "5 4", opacity: .85 }));
      svg.appendChild(el("text", { class: "axis", x: P.l + 4, y: ry - 5, fill: css("--warn") }, opts.refLine.label));
    }
    // x labels (sparse)
    var every = Math.ceil(n / 6);
    for (var i = 0; i < n; i++) if (i % every === 0 || i === n - 1) {
      svg.appendChild(el("text", { class: "axis", x: xOf(i), y: H - 10, "text-anchor": i === n - 1 ? "end" : "middle" }, opts.xLabels[i]));
    }
    // series
    series.forEach(function (s) {
      var pts = s.points.map(function (v, i) { return v == null ? null : [xOf(i), yOf(v)]; }).filter(Boolean);
      if (opts.area && s.fill) {
        var dd = "M" + pts.map(function (p) { return p[0] + "," + p[1]; }).join(" L ");
        dd += " L " + pts[pts.length - 1][0] + "," + yOf(yMin) + " L " + pts[0][0] + "," + yOf(yMin) + " Z";
        svg.appendChild(el("path", { d: dd, fill: s.color, "fill-opacity": .12, stroke: "none" }));
      }
      if (opts.dots) {
        pts.forEach(function (p) { svg.appendChild(el("circle", { cx: p[0], cy: p[1], r: 4.5, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5 })); });
      } else {
        svg.appendChild(el("polyline", { points: pts.map(function (p) { return p[0] + "," + p[1]; }).join(" "), fill: "none", stroke: s.color, "stroke-width": 2.2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
        var last = pts[pts.length - 1];
        svg.appendChild(el("circle", { cx: last[0], cy: last[1], r: 3.6, fill: s.color }));
        svg.appendChild(el("text", { class: "serieslabel", x: last[0] + 7, y: last[1] + 3.5, fill: s.color }, opts.endFmt ? opts.endFmt(s) : ""));
      }
    });
    host.appendChild(svg);
    if (opts.dots) return; // calibration: no crosshair
    // hover crosshair + tooltip
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    var cross = el("line", { class: "gridline", y1: P.t, y2: H - P.b, "stroke-dasharray": "3 3", opacity: 0 }); svg.appendChild(cross);
    var focus = series.map(function (s) { var c = el("circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5, opacity: 0 }); svg.appendChild(c); return c; });
    var rect = el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }); svg.appendChild(rect);
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect();
      var xv = (ev.clientX - r.left) / r.width * W;
      var i = Math.round((xv - P.l) / ((W - P.l - P.r) / (n - 1)));
      i = Math.max(0, Math.min(n - 1, i));
      cross.setAttribute("x1", xOf(i)); cross.setAttribute("x2", xOf(i)); cross.setAttribute("opacity", 1);
      var rows = "";
      series.forEach(function (s, si) {
        var v = s.points[i];
        if (v == null) { focus[si].setAttribute("opacity", 0); return; }
        focus[si].setAttribute("cx", xOf(i)); focus[si].setAttribute("cy", yOf(v)); focus[si].setAttribute("opacity", 1);
        rows += '<div class="tr"><span><i style="background:' + s.color + '"></i>' + s.name + '</span><b>' + (opts.tipFmt ? opts.tipFmt(v) : v) + '</b></div>';
      });
      tip.innerHTML = '<div class="tt">' + opts.xLabels[i] + (opts.tipSuffix ? opts.tipSuffix(i) : "") + '</div>' + rows;
      tip.style.opacity = 1;
      var host_r = host.getBoundingClientRect();
      var px = xOf(i) / W * host_r.width;
      tip.style.left = Math.min(Math.max(px - tip.offsetWidth / 2, 0), host_r.width - tip.offsetWidth) + "px";
      tip.style.top = "6px";
    });
    svg.addEventListener("mouseleave", function () { tip.style.opacity = 0; cross.setAttribute("opacity", 0); focus.forEach(function (c) { c.setAttribute("opacity", 0); }); });
  }
  // ---- grouped bar chart ----
  function barChart(id, opts) {
    var host = document.getElementById(id);
    if (!host || !opts.groups.length) { if (host) host.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">No data yet.</div>'; return; }
    var W = Math.max(host.clientWidth || 700, 320), H = W < 520 ? 232 : (opts.h || 290), P = { l: 46, r: 14, t: 12, b: opts.showN === false ? 26 : 42 };
    var keys = opts.series;
    var yMax = niceTop(Math.max.apply(null, opts.groups.flatMap(function (g) { return keys.map(function (k) { return g[k.key]; }); })) * 1.1);
    var yOf = function (v) { return P.t + (H - P.t - P.b) * (1 - v / yMax); };
    var gw = (W - P.l - P.r) / opts.groups.length;
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    var ticks = 4;
    for (var g = 0; g <= ticks; g++) { var yv = yMax * g / ticks, yy = yOf(yv); svg.appendChild(el("line", { class: "gridline", x1: P.l, y1: yy, x2: W - P.r, y2: yy })); svg.appendChild(el("text", { class: "axis", x: P.l - 8, y: yy + 3.5, "text-anchor": "end" }, opts.yFmt ? opts.yFmt(yv) : yv.toFixed(2))); }
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    var showN = opts.showN !== false;
    var tf = opts.tipFmt || function (v) { return v.toFixed(3); };
    var lblEvery = Math.ceil(opts.groups.length / (W < 520 ? 6 : 13));
    opts.groups.forEach(function (grp, gi) {
      var x0 = P.l + gw * gi, bw = Math.min(26, gw / (keys.length + 1)), inner = bw * keys.length + (keys.length - 1) * 5;
      var start = x0 + (gw - inner) / 2;
      keys.forEach(function (k, ki) {
        var v = grp[k.key], x = start + ki * (bw + 5), y = yOf(v);
        var bar = el("rect", { x: x, y: y, width: bw, height: Math.max(0, yOf(0) - y), rx: 3, fill: k.color });
        bar.addEventListener("mousemove", function (ev) {
          tip.innerHTML = '<div class="tt">' + grp.label + (showN ? ' · n=' + grp.n : '') + '</div>' + keys.map(function (kk) { return '<div class="tr"><span><i style="background:' + kk.color + '"></i>' + kk.name + '</span><b>' + tf(grp[kk.key]) + '</b></div>'; }).join("");
          tip.style.opacity = 1; var hr = host.getBoundingClientRect();
          tip.style.left = Math.min(Math.max((x0 + gw / 2) / W * hr.width - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = "6px";
        });
        bar.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
        svg.appendChild(bar);
      });
      if (gi % lblEvery === 0 || gi === opts.groups.length - 1) {
        svg.appendChild(el("text", { class: "axis", x: x0 + gw / 2, y: H - (showN ? 24 : 10), "text-anchor": "middle" }, grp.label));
        if (showN) svg.appendChild(el("text", { class: "axis", x: x0 + gw / 2, y: H - 10, "text-anchor": "middle", opacity: .7 }, "n=" + grp.n));
      }
    });
    host.appendChild(svg);
  }

  // ===== render =====
  function draw() {
  ["c_acc", "c_city", "c_calib", "c_growth", "c_hb"].forEach(function (id) { var h = document.getElementById(id); if (h) h.innerHTML = ""; });
  document.querySelectorAll(".chartwrap .tip").forEach(function (t) { t.remove(); });
  var acc = D.acc || [];
  if (acc.length) lineChart("c_acc", {
    h: 300, xLabels: acc.map(function (r) { return r.t; }),
    yFmt: function (v) { return v.toFixed(2); }, tipFmt: function (v) { return v.toFixed(4); },
    tipSuffix: function (i) { return "  ·  n=" + acc[i].n; },
    endFmt: function (s) { var last = s.points.filter(function (v) { return v != null; }).slice(-1)[0]; return last != null ? last.toFixed(3) : ""; },
    series: [
      { name: "Market", color: css("--market"), points: acc.map(function (r) { return r.market; }) },
      { name: "Model", color: css("--model"), points: acc.map(function (r) { return r.model; }) },
      { name: "Ensemble", color: css("--ens"), points: acc.map(function (r) { return r.ens; }) }
    ]
  });

  var city = D.city || [];
  barChart("c_city", {
    h: 300, yFmt: function (v) { return v.toFixed(2); },
    groups: city.map(function (r) { return { label: r.city, n: r.n, market: r.market, model: r.model }; }),
    series: [{ key: "market", name: "Market", color: css("--market") }, { key: "model", name: "Model", color: css("--model") }]
  });

  var calib = D.calib || [];
  if (calib.length) lineChart("c_calib", {
    h: 300, dots: true, diagonal: true, yMin: 0, yMax: 1,
    xLabels: calib.map(function (r) { return r.p.toFixed(2); }),
    yFmt: function (v) { return v.toFixed(1); },
    series: [{ name: "Model", color: css("--model"), points: calib.map(function (r) { return r.f; }) }]
  });
  else { var h = document.getElementById("c_calib"); if (h) h.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; }

  var gr = D.growth || [];
  if (gr.length) lineChart("c_growth", {
    area: true, xLabels: gr.map(function (r) { return r.t; }),
    yFmt: function (v) { return v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(Math.round(v)); },
    tipFmt: function (v) { return Math.round(v); },
    refLine: D.gate_line ? { y: D.gate_line, label: "gate " + D.gate_line } : null,
    endFmt: function (s) { var last = s.points.slice(-1)[0]; return last != null ? String(last) : ""; },
    series: [{ name: "Markets", color: css("--accent"), points: gr.map(function (r) { return r.m; }), fill: true }]
  });

  var hb = D.heartbeat || [];
  barChart("c_hb", {
    showN: false, tipFmt: function (v) { return String(Math.round(v)); }, yFmt: function (v) { return String(Math.round(v)); },
    groups: hb.map(function (r) { return { label: r.d.slice(5), n: r.n, runs: r.n }; }),
    series: [{ key: "runs", name: "Runs", color: css("--accent") }]
  });
  }
  draw();
  var _rt; window.addEventListener("resize", function () { clearTimeout(_rt); _rt = setTimeout(draw, 180); });
})();
</script>
"""


if __name__ == "__main__":
    main()
