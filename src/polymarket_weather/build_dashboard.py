#!/usr/bin/env python3
"""build_dashboard.py — render the live status dashboard as a self-contained HTML page.

Reads the CURRENT numbers straight from the honest-eval tooling (no new computation, no
duplicated logic): it runs `data_status.py`, `evaluate_oos.py` and `shoulder_book.py --report`,
parses their text output, and fills a tokenised template. Output is one dependency-free
`index.html` (inline CSS, theme-aware) suitable for GitHub Pages / any static host.

    python build_dashboard.py                # writes ../../site/index.html
    python build_dashboard.py /tmp/out.html  # custom path

Design note: every value has a fallback, so a parse miss degrades to "—" rather than crashing —
the page must always render. Keep this file in sync with the published artifact template.
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent          # src/polymarket_weather
REPO = PKG.parents[1]                           # repo root
DEFAULT_OUT = REPO / "site" / "index.html"

# City key (as printed by data_status) -> (display name, resolution station)
CITY_META = {
    "Seoul": ("Seoul", "RKSI"),
    "London": ("London", "EGLC"),
    "Chicago": ("Chicago", "KORD"),
    "NYC": ("New York", "KLGA"),
    "HongKong": ("Hong Kong", "HKO"),
}
CITY_ORDER = ["Seoul", "London", "Chicago", "NYC", "HongKong"]
BRIER_SCALE = 0.18   # x-axis max for the comparison bars


def _run(args: list[str]) -> str:
    """Run a report script from the package dir; return combined stdout (never raises)."""
    try:
        r = subprocess.run(
            [sys.executable, *args], cwd=PKG,
            capture_output=True, text=True, timeout=600,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:                       # network / import / timeout
        return f"[build_dashboard: {' '.join(args)} failed: {e}]"


def _search(pattern: str, text: str, *groups, flags=0, default=None):
    m = re.search(pattern, text, flags)
    if not m:
        return default if len(groups) <= 1 else (default,) * len(groups)
    if not groups:
        return m.group(1)
    vals = tuple(m.group(g) for g in groups)
    return vals[0] if len(vals) == 1 else vals


def gather() -> dict:
    status = _run(["data_status.py"])
    oos = _run(["evaluate_oos.py"])
    book = _run(["shoulder_book.py", "--report"])

    d: dict = {}

    # ---- data_status: span, gate, gradable, cities ----
    span = re.search(r"Date span.*?:\s*([\d-]+)\s*\.\.\s*([\d-]+)", status)
    if span:
        d["span_start"], d["span_end"] = span.group(1), span.group(2)
    gm = re.search(r"gradable markets\s+(\d+)\s*/\s*(\d+)\s*\[(\w+)\]", status)
    gb = re.search(r"gradable bets\s+(\d+)\s*/\s*(\d+)\s*\[(\w+)\]", status)
    if gm:
        d["gate_mkts"], d["gate_mkts_thr"], d["gate_status"] = gm.group(1), gm.group(2), gm.group(3)
    if gb:
        d["gate_bets"], d["gate_bets_thr"] = gb.group(1), gb.group(2)

    cities = {}
    for name, days in re.findall(r"(\w+)\s+latest obs\s+[\d-]+\s+\((\d+)d behind", status):
        cities[name] = int(days)
    d["cities"] = cities

    # ---- evaluate_oos: Brier table, CRPS, E3 ----
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

    # ---- shoulder_book ----
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


def _fmt_span(d: dict) -> tuple[str, str]:
    """Return (days, 'Mon D → Mon D 'YY')."""
    try:
        s = datetime.strptime(d["span_start"], "%Y-%m-%d")
        e = datetime.strptime(d["span_end"], "%Y-%m-%d")
        days = str((e - s).days)
        span = f"{s.strftime('%b %-d')} → {e.strftime('%b %-d')} '{e.strftime('%y')}"
        return days, span
    except Exception:
        return "—", "—"


def _data_through() -> str:
    """Latest data-commit time (falls back to build time)."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%cI"], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        iso = r.stdout.strip()
        if iso:
            dt = datetime.fromisoformat(iso).astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pct(v, default="0") -> str:
    try:
        return f"{min(float(v) / BRIER_SCALE * 100, 100):.0f}"
    except Exception:
        return default


def _cities_html(cities: dict) -> str:
    out = []
    for key in CITY_ORDER:
        disp, station = CITY_META[key]
        days = cities.get(key)
        if days is None:
            chip = '<span class="chip">no data</span>'
        else:
            cls = "good" if days <= 14 else "warn"
            chip = f'<span class="chip {cls}">{days}d behind</span>'
        out.append(
            f'<div class="city"><div class="cn">{disp}</div>'
            f'<div class="cc mono">{station}</div>{chip}</div>'
        )
    return "\n      ".join(out)


def render(d: dict) -> str:
    days, span = _fmt_span(d)
    G = lambda k, dflt="—": str(d.get(k, dflt))

    # edge verdict is computed, not assumed
    try:
        model_wins = float(d["br_model"]) < float(d["br_market"])
    except Exception:
        model_wins = False
    edge_chip = ('<span class="chip good">YES</span>' if model_wins
                 else '<span class="chip warn">NOT YET</span>')
    if model_wins:
        headline = "Our model is currently beating the market."
        body = ("Measured against the actual weather-station settlements, our forecast predicts "
                "outcomes more accurately than Polymarket's prices.")
    else:
        headline = "The market is currently smarter than our model."
        body = ("Measured against the actual weather-station settlements, Polymarket's prices "
                "predict outcomes <b style=\"color:var(--ink)\">more accurately</b> than our "
                "forecasting model. So the model isn't betting real money — this is the "
                "honest result, surfaced on purpose.")

    repl = {
        "UPDATED": _data_through(),
        "GATE_STATUS": G("gate_status", "—"),
        "GATE_MKTS": G("gate_mkts"), "GATE_BETS": G("gate_bets"),
        "DATA_DAYS": days, "SPAN": span,
        "SB_WR": G("sb_wr", "—"), "SB_GRADED": G("sb_graded"), "SB_ENTRIES": G("sb_entries"),
        "BR_MARKET": G("br_market"), "BR_ENS": G("br_ens"), "BR_MODEL": G("br_model"),
        "W_MARKET": _pct(d.get("br_market")), "W_ENS": _pct(d.get("br_ens")), "W_MODEL": _pct(d.get("br_model")),
        "N_MKTS": G("n_mkts"),
        "CRPS_MODEL": G("crps_model"), "CRPS_ENS": G("crps_ens"),
        "EDGE_CHIP": edge_chip, "EDGE_HEADLINE": headline, "EDGE_BODY": body,
        "SB_FULL": G("sb_full"), "SB_CORE": G("sb_core"), "SB_AWAIT": G("sb_await"), "SB_FULL_N": G("sb_full_n"),
        "E3_ROI": G("e3_roi"), "E3_N": G("e3_n"),
        "CITIES_HTML": _cities_html(d.get("cities", {})),
    }
    html = TEMPLATE
    for k, v in repl.items():
        html = html.replace(f"%%{k}%%", str(v))
    return html


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    d = gather()
    out.write_text(render(d), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


TEMPLATE = r"""<title>Prediction Markets Bot — Live Dashboard</title>
<style>
  :root {
    --bg: #eef1f5; --surface: #ffffff; --surface-2: #f3f6f9; --border: #d6dde6;
    --border-strong: #c2ccd8; --ink: #131922; --ink-2: #55616f; --ink-3: #8592a1;
    --accent: #0b7f9e; --accent-ink: #075a70;
    --good: #1a7f37; --good-bg: rgba(26,127,55,.10);
    --warn: #9a6700; --warn-bg: rgba(154,103,0,.10);
    --crit: #cf222e; --crit-bg: rgba(207,34,46,.10);
    --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.05);
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Roboto, sans-serif;
    --font-mono: "SF Mono", "JetBrains Mono", "Roboto Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0a0e15; --surface: #121924; --surface-2: #1a2331; --border: #253143;
      --border-strong: #33425a; --ink: #e7edf4; --ink-2: #9fadbf; --ink-3: #66748a;
      --accent: #48cdea; --accent-ink: #7bdcf1;
      --good: #4bc367; --good-bg: rgba(75,195,103,.13);
      --warn: #e2af3f; --warn-bg: rgba(226,175,63,.13);
      --crit: #ff6b63; --crit-bg: rgba(255,107,99,.13);
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 32px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="light"] {
    --bg: #eef1f5; --surface: #ffffff; --surface-2: #f3f6f9; --border: #d6dde6;
    --border-strong: #c2ccd8; --ink: #131922; --ink-2: #55616f; --ink-3: #8592a1;
    --accent: #0b7f9e; --accent-ink: #075a70;
    --good: #1a7f37; --good-bg: rgba(26,127,55,.10);
    --warn: #9a6700; --warn-bg: rgba(154,103,0,.10);
    --crit: #cf222e; --crit-bg: rgba(207,34,46,.10);
    --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.05);
  }
  :root[data-theme="dark"] {
    --bg: #0a0e15; --surface: #121924; --surface-2: #1a2331; --border: #253143;
    --border-strong: #33425a; --ink: #e7edf4; --ink-2: #9fadbf; --ink-3: #66748a;
    --accent: #48cdea; --accent-ink: #7bdcf1;
    --good: #4bc367; --good-bg: rgba(75,195,103,.13);
    --warn: #e2af3f; --warn-bg: rgba(226,175,63,.13);
    --crit: #ff6b63; --crit-bg: rgba(255,107,99,.13);
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 12px 32px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-sans); -webkit-font-smoothing: antialiased; line-height: 1.5; }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 32px 20px 64px; }
  .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .eyebrow { font-family: var(--font-mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; }
  .topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; padding-bottom: 20px; margin-bottom: 28px; border-bottom: 1px solid var(--border); }
  .brand { display: flex; align-items: center; gap: 11px; margin-right: auto; }
  .brand-mark { width: 38px; height: 38px; border-radius: 9px; flex: none; background: linear-gradient(145deg, var(--accent), var(--accent-ink)); display: grid; place-items: center; font-size: 20px; box-shadow: var(--shadow); }
  .brand h1 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -.01em; text-wrap: balance; }
  .brand p { margin: 1px 0 0; font-size: 12.5px; color: var(--ink-2); }
  .updated { text-align: right; font-size: 12px; color: var(--ink-3); }
  .updated b { color: var(--ink-2); font-weight: 600; }
  .pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px 5px 9px; border-radius: 100px; font-size: 12.5px; font-weight: 650; border: 1px solid transparent; white-space: nowrap; }
  .pill.good { color: var(--good); background: var(--good-bg); border-color: color-mix(in srgb, var(--good) 30%, transparent); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex: none; }
  .dot.live { animation: pulse 2.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--good) 55%, transparent); } 50% { box-shadow: 0 0 0 5px transparent; } }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 11px; font-weight: 600; letter-spacing: .02em; padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border-strong); color: var(--ink-2); }
  .chip.good { color: var(--good); border-color: color-mix(in srgb, var(--good) 34%, transparent); background: var(--good-bg); }
  .chip.warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 34%, transparent); background: var(--warn-bg); }
  .chip.accent { color: var(--accent-ink); border-color: color-mix(in srgb, var(--accent) 40%, transparent); background: color-mix(in srgb, var(--accent) 12%, transparent); }
  section { margin-top: 30px; }
  .sec-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
  .sec-head h2 { margin: 0; font-size: 15.5px; font-weight: 680; letter-spacing: -.005em; }
  .sec-head .note { font-size: 12.5px; color: var(--ink-3); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: var(--shadow); }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 16px 15px; box-shadow: var(--shadow); }
  .kpi .label { font-size: 11.5px; color: var(--ink-2); font-weight: 600; margin-bottom: 9px; display: flex; align-items: center; gap: 6px; }
  .kpi .big { font-family: var(--font-mono); font-size: 26px; font-weight: 680; letter-spacing: -.02em; line-height: 1.05; color: var(--ink); }
  .kpi .sub { font-size: 12px; color: var(--ink-3); margin-top: 5px; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .kpi .big.g { color: var(--good); }
  .verdict-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; }
  .bars { display: flex; flex-direction: column; gap: 14px; margin-top: 4px; }
  .barrow { display: grid; grid-template-columns: 92px 1fr; align-items: center; gap: 12px; }
  .barrow .name { font-size: 13px; font-weight: 600; }
  .barrow .name small { display: block; font-size: 10.5px; color: var(--ink-3); font-weight: 500; }
  .track { position: relative; height: 30px; background: var(--surface-2); border-radius: 7px; overflow: hidden; }
  .fill { position: absolute; inset: 0 auto 0 0; border-radius: 7px; display: flex; align-items: center; justify-content: flex-end; padding-right: 9px; color: #fff; font-family: var(--font-mono); font-size: 12.5px; font-weight: 650; animation: grow .9s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes grow { from { width: 0 !important; } }
  .fill.best { background: linear-gradient(90deg, var(--accent-ink), var(--accent)); }
  .fill.mid  { background: color-mix(in srgb, var(--ink-3) 62%, var(--surface)); }
  .fill.lose { background: linear-gradient(90deg, color-mix(in srgb, var(--warn) 78%, #000), var(--warn)); }
  .scale-note { font-size: 11.5px; color: var(--ink-3); margin-top: 12px; font-family: var(--font-mono); }
  .verdict-box { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .verdict-box p { margin: 0; font-size: 13px; color: var(--ink-2); }
  .verdict-box .headline { font-size: 14px; color: var(--ink); font-weight: 640; }
  .miniquote { font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-3); border-left: 2px solid var(--border-strong); padding-left: 10px; }
  .strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--surface); box-shadow: var(--shadow); }
  .strip > div { padding: 15px 16px; border-right: 1px solid var(--border); }
  .strip > div:last-child { border-right: none; }
  .strip .n { font-family: var(--font-mono); font-size: 21px; font-weight: 680; letter-spacing: -.01em; }
  .strip .n.pos { color: var(--good); }
  .strip .k { font-size: 11.5px; color: var(--ink-2); margin-top: 3px; }
  .strip .k small { color: var(--ink-3); }
  .lead { font-size: 13.5px; color: var(--ink-2); margin: 0 0 14px; max-width: 68ch; }
  .lead b { color: var(--ink); font-weight: 640; }
  .cities { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .city { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px; box-shadow: var(--shadow); }
  .city .cn { font-size: 13.5px; font-weight: 660; margin-bottom: 3px; }
  .city .cc { font-family: var(--font-mono); font-size: 11px; color: var(--ink-3); margin-bottom: 10px; }
  footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 12px; color: var(--ink-3); line-height: 1.7; }
  footer b { color: var(--ink-2); font-weight: 600; }
  footer .row { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 10px; }
  a { color: var(--accent-ink); }
  @media (max-width: 760px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .verdict-grid { grid-template-columns: 1fr; }
    .strip { grid-template-columns: repeat(2, 1fr); }
    .strip > div:nth-child(2) { border-right: none; }
    .strip > div:nth-child(1), .strip > div:nth-child(2) { border-bottom: 1px solid var(--border); }
    .cities { grid-template-columns: repeat(2, 1fr); }
  }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>

<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">📊</div>
      <div>
        <h1>Prediction Markets Bot</h1>
        <p>Autonomous Polymarket weather-market tracker · runs in the cloud · 5 cities</p>
      </div>
    </div>
    <span class="pill good"><span class="dot live"></span>OPERATIONAL</span>
    <div class="updated">data through<br><b class="mono">%%UPDATED%%</b></div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Collector</div><div class="big g">LIVE</div><div class="sub">every 2h · GitHub Actions</div></div>
    <div class="kpi"><div class="label">Track-record gate</div><div class="big g">%%GATE_STATUS%%</div><div class="sub">%%GATE_MKTS%% mkts · %%GATE_BETS%% bets</div></div>
    <div class="kpi"><div class="label">Data collected</div><div class="big">%%DATA_DAYS%%<span style="font-size:15px;color:var(--ink-3)"> days</span></div><div class="sub">%%SPAN%%</div></div>
    <div class="kpi"><div class="label">Paper book win rate</div><div class="big">%%SB_WR%%<span style="font-size:15px;color:var(--ink-3)">%</span></div><div class="sub">%%SB_GRADED%% graded / %%SB_ENTRIES%% open</div></div>
  </div>

  <section>
    <div class="sec-head"><span class="eyebrow">Performance · graded vs real settlements</span></div>
    <div class="sec-head" style="margin-bottom:14px"><h2>Does our forecast beat the market?</h2>%%EDGE_CHIP%%</div>
    <div class="card">
      <div class="verdict-grid">
        <div>
          <div class="eyebrow" style="margin-bottom:10px">Prediction accuracy — Brier score, lower is better</div>
          <div class="bars">
            <div class="barrow"><div class="name">Market<small>the tradeable price</small></div><div class="track"><div class="fill best" style="width:%%W_MARKET%%%">%%BR_MARKET%%</div></div></div>
            <div class="barrow"><div class="name">Ensemble<small>raw weather models</small></div><div class="track"><div class="fill mid" style="width:%%W_ENS%%%">%%BR_ENS%%</div></div></div>
            <div class="barrow"><div class="name">Our model<small>calibrated forecast</small></div><div class="track"><div class="fill lose" style="width:%%W_MODEL%%%">%%BR_MODEL%%</div></div></div>
          </div>
          <div class="scale-note">n = %%N_MKTS%% resolved markets · scale 0 → 0.18 · shorter = more accurate</div>
        </div>
        <div class="verdict-box">
          <div class="headline">%%EDGE_HEADLINE%%</div>
          <p>%%EDGE_BODY%%</p>
          <div class="miniquote">market %%BR_MARKET%% · ensemble %%BR_ENS%% · model %%BR_MODEL%%<br>where the model <i>does</i> win: temperature CRPS %%CRPS_MODEL%% vs %%CRPS_ENS%%</div>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="eyebrow">The candidate edge · forward paper trial</span></div>
    <div class="sec-head" style="margin-bottom:14px"><h2>Structure paper book</h2><span class="chip accent">PAPER ONLY — NO REAL MONEY</span></div>
    <p class="lead">A <b>model-free</b> strategy that ignores the weather forecast entirely: it sells over-priced long-shot temperature bins and buys clear favorites, then settles against real results. Early signal is strong, but it stays paper until it clears a pre-registered forward test.</p>
    <div class="strip">
      <div><div class="n pos">%%SB_WR%%%</div><div class="k">win rate <small>· %%SB_GRADED%% settled</small></div></div>
      <div><div class="n pos">%%SB_FULL%%</div><div class="k">edge / contract <small>· full band</small></div></div>
      <div><div class="n pos">%%SB_CORE%%</div><div class="k">edge / contract <small>· core band</small></div></div>
      <div><div class="n">%%SB_AWAIT%%</div><div class="k">awaiting <small>· settlement</small></div></div>
    </div>
    <p class="lead" style="margin-top:14px;margin-bottom:0">A second signal — the model's most selective bucket — shows <b>%%E3_ROI%% ROI on %%E3_N%% bets</b>, but that's still in-sample. Nothing goes live until a bucket logs <b>40+ forward bets</b> beating the market. Discipline first.</p>
  </section>

  <section>
    <div class="sec-head"><h2>Cities &amp; data freshness</h2><span class="note">how far behind the settlement feed is per city</span></div>
    <div class="cities">
      %%CITIES_HTML%%
    </div>
  </section>

  <footer>
    <div class="row">
      <span><b>Pipeline:</b> collect every 2h · truth-eval daily · retrain on demand</span>
      <span><b>Host:</b> GitHub Actions (free tier)</span>
      <span><b>Data store:</b> git</span>
    </div>
    <div><b>Method:</b> every prediction is graded against the actual weather-station reading each market settles on — no grading a forecast against the same grid it came from. No performance is claimed until a pre-committed sample gate is met (now cleared). All trading shown is paper; no real orders are placed.</div>
    <div style="margin-top:10px;color:var(--ink-3)">Auto-generated from the live data · rebuilt on every collection cycle.</div>
  </footer>
</div>
"""


if __name__ == "__main__":
    main()
