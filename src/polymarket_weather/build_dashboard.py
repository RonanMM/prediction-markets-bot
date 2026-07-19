#!/usr/bin/env python3
"""build_dashboard.py — render the LIVE status dashboard as a client-side page.

Emits TWO files next to each other:
  * index.html  — a static SHELL: all CSS + charts + a fetch/refresh loop. Contains NO baked
    numbers in its markup; every value is filled in at view time from data.json. It also embeds
    a copy of the current payload inline (``<script id="D0">``) so the page renders instantly on
    first paint and still works standalone (local file://, the Claude Artifact preview) where a
    cross-file fetch is blocked.
  * data.json   — the payload: scalars (parsed from data_status / evaluate_oos / shoulder_book),
    pre-rendered HTML fragments, the chart SERIES, and timestamps. Refreshed every build.

At view time the page loads data.json (cache-busted), fills the ``data-bind`` / ``data-bind-html``
slots, redraws charts/tables if the series changed, and ticks a live "refreshed Xm ago" clock +
a UTC clock — so opening the link always shows the freshest published data and an open tab
updates itself. The OPERATIONAL pill is computed from the last collector snapshot age, never
hardcoded.

    python build_dashboard.py                # writes ../../site/index.html + ../../site/data.json
    python build_dashboard.py /path/out.html # custom path (data.json written alongside)

SCALARS are parsed from stdout; SERIES are computed in-process (evaluate_oos._graded_markets for
accuracy/calibration/buckets/recent, shoulder_book's own settlement functions for the paper-book
equity curve — same fee math as the report). Every value degrades to "—" / an empty chart on a
parse or import miss — the page must always render. Charts are hand-rolled SVG (no external
libraries). Series palette (#c98500 market / #1e93c4 model / #bb62b0 ensemble) is validated
CVD-safe on the dark surface.
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


def _bin_label(q: str) -> str:
    """'Will the highest temperature in Chicago be between 90-91°F on July 13?' → '↑90–91°F'."""
    kind = "↓" if "lowest" in q.lower() else "↑"
    m = re.search(r"\bbe\s+(.+?)\s+on\s", q)
    core = m.group(1) if m else q[:26]
    core = core.replace("between ", "").replace(" and ", "–")
    core = re.sub(r"(\d)-(\d)", r"\1–\2", core)
    core = re.sub(r"(.+?)\s+or\s+(higher|above|more)", r"≥\1", core)
    core = re.sub(r"(.+?)\s+or\s+(lower|below|less)", r"≤\1", core)
    return kind + core.strip()


def compute_series() -> dict:
    out: dict = {"acc": [], "roll": [], "city": [], "calib": [], "growth": [],
                 "heartbeat": [], "buckets": [], "recent": [], "equity": []}
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
        out["n_common"] = int(len(c))

        # rolling form — trailing 60-market window on the same common set (recent form vs history)
        cc = c.sort_values("td").reset_index(drop=True)
        roll = []
        for i in range(19, len(cc)):
            w = cc.iloc[max(0, i - 59): i + 1]
            roll.append({
                "t": cc["td"].iloc[i].strftime("%b %-d"),
                "model": round(float(w["b_model"].mean()), 4),
                "market": round(float(w["b_mkt"].mean()), 4),
                "ens": round(float(w["b_ens"].mean()), 4),
                "n": int(len(w)),
            })
        out["roll"] = _sample(roll)

        # accuracy by city
        city = []
        for cty, g in cal.groupby("city"):
            city.append({"city": str(cty), "model": round(float(g["b_model"].mean()), 4),
                         "market": round(float(g["b_mkt"].mean()), 4), "n": int(len(g))})
        city.sort(key=lambda r: r["n"], reverse=True)
        out["city"] = city

        # calibration — model predicted P(YES) vs realized frequency, 10 bins, dot sized by n
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

        # per-bucket edge table + the pre-registered forward gates (megaplan E3)
        try:
            import config as cfg
            nominated = set(getattr(cfg, "LIVE_BUCKETS", set()))
            nom_date = pd.Timestamp(str(getattr(cfg, "E3_NOMINATION_DATE", "2026-07-12")))
        except Exception:
            nominated, nom_date = set(), pd.Timestamp("2026-07-12")
        buckets = []
        if "bucket" in cal.columns:
            for b, g in cal.groupby("bucket"):
                if len(g) < 3:
                    continue
                fwd = g[g["td"] > nom_date]
                row = {"b": str(b), "n": int(len(g)),
                       "model": round(float(g["b_model"].mean()), 4),
                       "market": round(float(g["b_mkt"].mean()), 4),
                       "nom": str(b) in nominated,
                       "fwd_n": int(len(fwd))}
                if len(fwd) >= 3:
                    row["fwd_model"] = round(float(fwd["b_model"].mean()), 4)
                    row["fwd_market"] = round(float(fwd["b_mkt"].mean()), 4)
                buckets.append(row)
            buckets.sort(key=lambda r: (not r["nom"], -r["n"]))
        out["buckets"] = buckets
        out["gate_fwd_n"] = 40
        out["nom_date"] = nom_date.strftime("%b %-d")

        # recent settlements feed — the last 14 graded markets, newest first
        rec = cal.sort_values("td").tail(14)
        out["recent"] = [{
            "d": pd.Timestamp(r.td).strftime("%b %-d"),
            "city": str(r.city),
            "bin": _bin_label(str(r.question)),
            "model": round(float(r.forecast_prob), 3),
            "market": round(float(getattr(r, mkt_col)), 3),
            "out": int(r.outcome),
        } for r in rec.itertuples()][::-1]

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

    # paper-book equity curve — grade each entry with shoulder_book's OWN settlement + fee math
    # (taker-conservative: half-spread crossed on entry, 0.05·p·(1−p) fee), cumulative by day.
    try:
        import pandas as pd
        import shoulder_book as sb
        book = sb._load_book()
        rows = []
        for _, r in book.iterrows():
            pq = sb.parse_question(r["question"])
            if not pq:
                continue
            yres = sb.resolves_yes(r["city"], r["target_date"], r["question"], pq["temp_c"])
            if yres is None:
                continue
            rows.append({"t": str(r["target_date"]), "leg": str(r["leg"]),
                         "won": (int(yres) == 1) == (r["side"] == "Yes"),
                         "sp": float(r["entry_side_price"])})
        if rows:
            df = pd.DataFrame(rows)
            df["net"] = sb._net_edge(df["won"], df["sp"])
            df["sh"] = df["net"].where(df["leg"] == "shoulder", 0.0)
            df["fav"] = df["net"].where(df["leg"] == "favorite", 0.0)
            g = df.groupby("t").agg(net=("net", "sum"), sh=("sh", "sum"),
                                    fav=("fav", "sum"), n=("net", "size")).sort_index()
            cum = g.cumsum()
            out["equity"] = [{"t": pd.Timestamp(t).strftime("%b %-d"),
                              "v": round(float(cum["net"].loc[t]), 3),
                              "sh": round(float(cum["sh"].loc[t]), 3),
                              "fav": round(float(cum["fav"].loc[t]), 3),
                              "n": int(g["n"].loc[t])} for t in g.index]
    except Exception as e:
        out["eq_error"] = f"{type(e).__name__}: {e}"

    # collection heartbeat — distinct collection hours per day (last 30d) + last snapshot time
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
            out["last_collect_iso"] = ts.max().strftime("%Y-%m-%dT%H:%M:%SZ")
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


def _last_commit_dt() -> datetime:
    """UTC datetime of the last data commit (what the numbers are 'through'); now() on failure."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%cI"], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        iso = r.stdout.strip()
        if iso:
            return datetime.fromisoformat(iso).astimezone(timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


# ────────────────────────────── payload (data.json) ──────────────────────────────
def _scoreboard_html(d: dict, n_common) -> str:
    """The hero scoreboard: market / model / ensemble Brier ranked, leader flagged."""
    entries = []
    for key, name, cls in (("br_market", "Market price", "market"),
                           ("br_model", "Our model", "model"),
                           ("br_ens", "Raw ensemble", "ens")):
        try:
            entries.append({"name": name, "cls": cls, "v": float(d[key])})
        except Exception:
            return ""
    ranked = sorted(entries, key=lambda e: e["v"])
    for i, e in enumerate(ranked):
        e["rank"] = i + 1
    lead = ranked[0]["v"]
    cells = []
    for e in entries:   # fixed display order; rank shown as a chip
        gap = e["v"] - lead
        sub = "leads" if e["rank"] == 1 else f"+{gap:.4f} behind"
        cells.append(
            f'<div class="score {"lead" if e["rank"] == 1 else ""}">'
            f'<div class="sr"><span class="rank r{e["rank"]}">#{e["rank"]}</span>'
            f'<span class="sname"><i class="sw {e["cls"]}"></i>{e["name"]}</span></div>'
            f'<div class="sv">{e["v"]:.4f}</div>'
            f'<div class="ss">{sub}</div></div>')
    n_txt = f"{n_common} common markets" if n_common else "common markets"
    return (f'<div class="scores">{"".join(cells)}</div>'
            f'<div class="score-note">Brier score on the same {n_txt} — lower is better. '
            f'Every prediction graded against the station reading each market settles on.</div>')


def build_payload(d: dict, series: dict) -> dict:
    days, span = _fmt_span(d)
    G = lambda k, dflt="—": str(d.get(k, dflt))
    try:
        model_wins = float(d["br_model"]) < float(d["br_market"])
        skill = (1.0 - float(d["br_model"]) / float(d["br_market"])) * 100.0
        skill_txt = f"{skill:+.1f}%".replace("-", "−")
    except Exception:
        model_wins, skill_txt = False, "—"
    if model_wins:
        takeaway = ("Our calibrated forecast now predicts settlements <b>more accurately than the "
                    "market price</b> — the lines below have crossed.")
        edge_chip = '<span class="chip good">MODEL AHEAD</span>'
    else:
        takeaway = ("The market still out-predicts the model — shown on purpose. <b>No edge is "
                    "claimed until the model line drops below the market line</b>; meanwhile the "
                    "candidate edges below are walked forward on paper.")
        edge_chip = '<span class="chip warn">MARKET LEADS</span>'

    # paper-book running total from the equity series
    eq = series.get("equity") or []
    book_net = f"{eq[-1]['v']:+.2f}u".replace("-", "−") if eq else "—"
    hb = series.get("heartbeat") or []
    runs_today = str(hb[-1]["n"]) if hb else "—"

    commit_dt = _last_commit_dt()
    now = datetime.now(timezone.utc)
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through_iso": commit_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_collect_iso": series.get("last_collect_iso"),
        "bind": {
            "UPDATED": commit_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "GATE_STATUS": G("gate_status"), "GATE_MKTS": G("gate_mkts"), "GATE_BETS": G("gate_bets"),
            "GATE_MKTS_THR": G("gate_mkts_thr", "150"),
            "DATA_DAYS": days, "SPAN": span,
            "SB_WR": G("sb_wr"), "SB_GRADED": G("sb_graded"), "SB_ENTRIES": G("sb_entries"),
            "BR_MARKET": G("br_market"), "BR_ENS": G("br_ens"), "BR_MODEL": G("br_model"),
            "N_MKTS": G("n_mkts"), "CRPS_MODEL": G("crps_model"), "CRPS_ENS": G("crps_ens"),
            "SB_FULL": G("sb_full"), "SB_CORE": G("sb_core"), "SB_AWAIT": G("sb_await"),
            "E3_ROI": G("e3_roi"), "E3_N": G("e3_n"),
            "SKILL": skill_txt, "BOOK_NET": book_net, "RUNS_TODAY": runs_today,
        },
        "html": {
            "EDGE_CHIP": edge_chip,
            "TAKEAWAY": takeaway,
            "SCOREBOARD": _scoreboard_html(d, series.get("n_common")),
            "CITIES_HTML": _cities_html(series.get("city", []), d),
        },
        "series": series,
    }


def _cities_html(city_series: list, d: dict) -> str:
    # freshness comes from data_status; fall back gracefully if absent
    status = _run(["data_status.py"]) if not hasattr(_cities_html, "_cache") else _cities_html._cache
    _cities_html._cache = status
    lag = {n: int(v) for n, v in re.findall(r"(\w+)\s+latest obs\s+[\d-]+\s+\((\d+)d behind", status)}
    briers = {r["city"]: r for r in (city_series or [])}
    out = []
    for key in CITY_ORDER:
        disp, station = CITY_META[key]
        days = lag.get(key)
        chip = ('<span class="chip">no truth yet</span>' if days is None
                else f'<span class="chip {"good" if days <= 14 else "warn"}">truth {days}d behind</span>')
        b = briers.get(key)
        stat = (f'<div class="cb mono">n={b["n"]} · mkt {b["market"]:.3f} · mdl {b["model"]:.3f}</div>'
                if b else '<div class="cb mono">—</div>')
        out.append(f'<div class="city"><div class="cn">{disp} <span class="cc mono">{station}</span></div>'
                   f'{stat}{chip}</div>')
    return "\n      ".join(out)


def render_shell(payload: dict) -> str:
    # Escape "<" in the inline copy so a fragment can never break out of the <script> tag.
    inline = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.replace("%%PAYLOAD%%", inline)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(gather(), compute_series())
    out.write_text(render_shell(payload), encoding="utf-8")
    data_out = out.parent / "data.json"
    data_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) + {data_out} ({data_out.stat().st_size} bytes)")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediction Markets Bot — Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* Committed dark "mission control" — a markets terminal, deliberately single-theme. */
  :root {
    color-scheme: dark;
    --bg: #0a0e16; --panel: #101623; --panel-2: #141b2b; --border: #1d2740; --border-2: #2b3a5c;
    --ink: #e9eef7; --ink-2: #9aa8bf; --ink-3: #5f6d85;
    --accent: #4cc9f0; --accent-dim: #2b7d99;
    --market: #c98500; --model: #1e93c4; --ens: #bb62b0;   /* validated CVD-safe on dark */
    --market-hi: #f0b449; --model-hi: #4cc9f0; --ens-hi: #d98ccf; /* brighter text/ends variants */
    --good: #3ddc84; --good-bg: rgba(61,220,132,.09);
    --warn: #f0b449; --warn-bg: rgba(240,180,73,.09);
    --bad: #f2635f; --bad-bg: rgba(242,99,95,.09);
    --grid-line: #1a2338;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Roboto, sans-serif;
    --shadow: 0 1px 0 rgba(255,255,255,.02) inset, 0 8px 28px rgba(0,0,0,.45);
  }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body { margin: 0; background: radial-gradient(1100px 420px at 50% -140px, rgba(76,201,240,.07), transparent 65%), var(--bg);
         color: var(--ink); font-family: var(--sans); -webkit-font-smoothing: antialiased; line-height: 1.5; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 26px 22px 70px; opacity: 0; transform: translateY(8px); transition: opacity .5s ease, transform .5s ease; }
  body.ready .wrap { opacity: 1; transform: none; }
  .mono, .sv, .kv, td, th { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .eyebrow { font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; }
  h1, h2 { text-wrap: balance; }

  /* ── top status bar ── */
  .topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 18px; padding: 12px 16px; margin-bottom: 20px;
            background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }
  .brand { display: flex; align-items: center; gap: 12px; margin-right: auto; }
  .brand-mark { width: 36px; height: 36px; border-radius: 9px; flex: none; background: linear-gradient(145deg, var(--accent), var(--accent-dim));
                display: grid; place-items: center; font-size: 12.5px; letter-spacing: -1px; color: #06222e; font-weight: 700; font-family: var(--mono); }
  .brand h1 { margin: 0; font-size: 16.5px; font-weight: 700; letter-spacing: -.01em; font-family: var(--mono); }
  .brand p { margin: 1px 0 0; font-size: 11.5px; color: var(--ink-2); }
  .tb-item { text-align: right; font-family: var(--mono); font-size: 11px; color: var(--ink-3); line-height: 1.55; }
  .tb-item b { color: var(--ink-2); font-weight: 600; display: block; font-size: 12px; }
  .pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px 5px 10px; border-radius: 100px; font-size: 11.5px; font-weight: 650;
          font-family: var(--mono); letter-spacing: .04em; border: 1px solid transparent; white-space: nowrap; }
  .pill.good { color: var(--good); background: var(--good-bg); border-color: color-mix(in srgb, var(--good) 30%, transparent); }
  .pill.warn { color: var(--warn); background: var(--warn-bg); border-color: color-mix(in srgb, var(--warn) 30%, transparent); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex: none; }
  .dot.live { animation: pulse 2.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 55%, transparent); } 50% { box-shadow: 0 0 0 5px transparent; } }
  .live-ago { color: var(--good); }

  .chip { display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .03em;
          padding: 3px 8px; border-radius: 5px; border: 1px solid var(--border-2); color: var(--ink-2); white-space: nowrap; }
  .chip.good { color: var(--good); border-color: color-mix(in srgb, var(--good) 34%, transparent); background: var(--good-bg); }
  .chip.warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 34%, transparent); background: var(--warn-bg); }
  .chip.bad  { color: var(--bad);  border-color: color-mix(in srgb, var(--bad) 34%, transparent);  background: var(--bad-bg); }
  .chip.accent { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 38%, transparent); background: color-mix(in srgb, var(--accent) 10%, transparent); }

  section { margin-top: 22px; }
  .sec-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
  .sec-head h2 { margin: 0; font-size: 13.5px; font-weight: 650; letter-spacing: .01em; font-family: var(--mono); text-transform: uppercase; }
  .sec-head h2::before { content: "// "; color: var(--ink-3); font-weight: 500; }
  .sec-head .note { font-size: 12px; color: var(--ink-3); }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }

  /* ── hero verdict ── */
  .hero { background: linear-gradient(180deg, var(--panel-2), var(--panel)); border: 1px solid var(--border-2); border-radius: 14px;
          padding: 20px 22px 18px; box-shadow: var(--shadow); }
  .hero-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
  .hero-head h2 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: -.01em; }
  .scores { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .score { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 13px 15px; }
  .score.lead { border-color: color-mix(in srgb, var(--good) 40%, var(--border)); background: linear-gradient(180deg, color-mix(in srgb, var(--good) 6%, var(--panel)), var(--panel)); }
  .sr { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .rank { font-family: var(--mono); font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; letter-spacing: .05em; }
  .rank.r1 { color: var(--good); background: var(--good-bg); }
  .rank.r2 { color: var(--ink-2); background: rgba(154,168,191,.1); }
  .rank.r3 { color: var(--ink-3); background: rgba(95,109,133,.12); }
  .sname { font-size: 12px; color: var(--ink-2); font-weight: 600; display: inline-flex; align-items: center; gap: 7px; }
  .sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .sw.market { background: var(--market-hi); } .sw.model { background: var(--model-hi); } .sw.ens { background: var(--ens-hi); }
  .sv { font-size: 30px; font-weight: 700; letter-spacing: -.02em; line-height: 1; }
  .score.lead .sv { color: var(--good); }
  .ss { font-family: var(--mono); font-size: 11px; color: var(--ink-3); margin-top: 6px; }
  .score-note { font-size: 12px; color: var(--ink-3); margin-top: 12px; }
  .takeaway { font-size: 13px; color: var(--ink-2); margin: 12px 0 0; max-width: 84ch; }
  .takeaway b { color: var(--ink); font-weight: 640; }

  /* ── KPI band ── */
  .kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 14px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 11px 13px; box-shadow: var(--shadow); }
  .kpi .label { font-size: 10px; color: var(--ink-3); font-weight: 600; margin-bottom: 6px; font-family: var(--mono); letter-spacing: .08em; text-transform: uppercase; }
  .kpi .big { font-family: var(--mono); font-size: 19px; font-weight: 700; letter-spacing: -.02em; line-height: 1.05; }
  .kpi .big.g { color: var(--good); }
  .kpi .big small { font-size: 11px; color: var(--ink-3); font-weight: 500; }
  .kpi .sub { font-size: 10.5px; color: var(--ink-3); margin-top: 4px; font-family: var(--mono); }

  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 2px 0 10px; align-items: center; }
  .lg { display: inline-flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--ink-2); font-weight: 550; }
  .lg i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .lgnote { margin-left: auto; color: var(--ink-3); font-size: 11.5px; }
  .toggle { display: inline-flex; border: 1px solid var(--border-2); border-radius: 7px; overflow: hidden; margin-left: auto; }
  .toggle button { appearance: none; border: 0; background: transparent; color: var(--ink-3); font-family: var(--mono); font-size: 11px;
                   font-weight: 600; padding: 5px 12px; cursor: pointer; letter-spacing: .03em; }
  .toggle button.on { background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); }
  .toggle button:hover:not(.on) { color: var(--ink-2); }

  .chartwrap { position: relative; width: 100%; }
  svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
  .gridline { stroke: var(--grid-line); stroke-width: 1; }
  .axis { fill: var(--ink-3); font-family: var(--mono); font-size: 10.5px; }
  .serieslabel { font-family: var(--mono); font-size: 11px; font-weight: 650; }
  .tip { position: absolute; pointer-events: none; opacity: 0; transition: opacity .12s; background: var(--panel-2); border: 1px solid var(--border-2);
         border-radius: 8px; padding: 8px 10px; font-size: 11.5px; box-shadow: 0 10px 30px rgba(0,0,0,.5); z-index: 5; min-width: 122px; }
  .tip .tt { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); margin-bottom: 5px; }
  .tip .tr { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .tip .tr span { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); }
  .tip .tr i { width: 8px; height: 8px; border-radius: 2px; }
  .tip .tr b { color: var(--ink); font-weight: 680; }

  /* ── tables ── */
  .tblwrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; font-size: 10px; font-weight: 600; color: var(--ink-3); letter-spacing: .08em; text-transform: uppercase;
       padding: 7px 10px; border-bottom: 1px solid var(--border-2); white-space: nowrap; }
  td { padding: 7.5px 10px; border-bottom: 1px solid var(--border); color: var(--ink-2); white-space: nowrap; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(76,201,240,.03); }
  td.num, th.num { text-align: right; }
  td b { color: var(--ink); font-weight: 650; }
  .gatebar { position: relative; width: 92px; height: 6px; border-radius: 3px; background: var(--border); display: inline-block; vertical-align: middle; overflow: hidden; }
  .gatebar i { position: absolute; inset: 0 auto 0 0; background: var(--accent); border-radius: 3px; }
  .winner-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; vertical-align: middle; }

  .strip { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
           background: var(--panel); box-shadow: var(--shadow); }
  .strip > div { padding: 12px 15px; border-right: 1px solid var(--border); }
  .strip > div:last-child { border-right: none; }
  .strip .n { font-family: var(--mono); font-size: 19px; font-weight: 700; }
  .strip .n.pos { color: var(--good); }
  .strip .k { font-size: 10.5px; color: var(--ink-3); margin-top: 3px; font-family: var(--mono); }
  .lead-p { font-size: 12.5px; color: var(--ink-2); margin: 0 0 12px; max-width: 84ch; }
  .lead-p b { color: var(--ink); font-weight: 640; }

  .cities { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .city { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 13px; box-shadow: var(--shadow); }
  .city .cn { font-size: 13px; font-weight: 650; margin-bottom: 4px; }
  .city .cc { font-size: 10px; color: var(--ink-3); font-weight: 500; }
  .city .cb { font-size: 10.5px; color: var(--ink-3); margin-bottom: 9px; }

  footer { margin-top: 34px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 11.5px; color: var(--ink-3); line-height: 1.7; }
  footer b { color: var(--ink-2); font-weight: 600; }
  footer .row { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 10px; font-family: var(--mono); font-size: 11px; }
  #errbar { display: none; margin-bottom: 14px; padding: 9px 13px; border-radius: 9px; font-size: 12px; color: var(--warn); background: var(--warn-bg);
            border: 1px solid color-mix(in srgb, var(--warn) 30%, transparent); }
  .flash { animation: flashbg 1.1s ease; }
  @keyframes flashbg { 0%, 12% { background: color-mix(in srgb, var(--accent) 24%, transparent); } 100% { background: transparent; } }

  @media (max-width: 900px) {
    .kpis { grid-template-columns: repeat(3, 1fr); }
    .scores { grid-template-columns: 1fr; }
    .grid2 { grid-template-columns: 1fr; }
    .strip { grid-template-columns: repeat(2, 1fr); }
    .strip > div:nth-child(2) { border-right: none; }
    .strip > div:nth-child(1), .strip > div:nth-child(2) { border-bottom: 1px solid var(--border); }
    .cities { grid-template-columns: repeat(2, 1fr); }
    .tb-item.optional { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { *, .chart * { animation: none !important; transition: none !important; } .wrap { opacity: 1; transform: none; } }
</style>

<div class="wrap">
  <div id="errbar">Couldn't reach the live data feed — showing the last snapshot. Retrying…</div>

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">▲▼</div>
      <div><h1>prediction-markets-bot</h1><p>Autonomous Polymarket weather tracker · 5 cities · runs unattended in the cloud</p></div>
    </div>
    <span class="pill good" id="statuspill"><span class="dot live"></span><span id="statustext">CHECKING…</span></span>
    <div class="tb-item optional"><b class="mono" id="utcclock">--:--:-- UTC</b>system time</div>
    <div class="tb-item"><b class="mono live-ago" data-ago>refreshing…</b>page data</div>
    <div class="tb-item"><b class="mono" data-bind="UPDATED">—</b>data through</div>
  </div>

  <div class="hero">
    <div class="hero-head">
      <span class="eyebrow">The verdict</span>
      <h2>Who predicts the weather markets best?</h2>
      <span data-bind-html="EDGE_CHIP"></span>
    </div>
    <div data-bind-html="SCOREBOARD"></div>
    <p class="takeaway" data-bind-html="TAKEAWAY"></p>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Collector</div><div class="big" id="collectorbig">—</div><div class="sub" id="collectorsub">last snapshot</div></div>
    <div class="kpi"><div class="label">Runs today</div><div class="big"><span data-bind="RUNS_TODAY">—</span><small> /12</small></div><div class="sub">2-hourly cycles</div></div>
    <div class="kpi"><div class="label">Sample gate</div><div class="big g"><span data-bind="GATE_STATUS">—</span></div><div class="sub"><span data-bind="GATE_MKTS">—</span> mkts · <span data-bind="GATE_BETS">—</span> bets</div></div>
    <div class="kpi"><div class="label">Data span</div><div class="big"><span data-bind="DATA_DAYS">—</span><small> days</small></div><div class="sub" data-bind="SPAN">—</div></div>
    <div class="kpi"><div class="label">Model skill vs market</div><div class="big"><span data-bind="SKILL">—</span></div><div class="sub">Brier, + = model better</div></div>
    <div class="kpi"><div class="label">Paper book P&amp;L</div><div class="big g"><span data-bind="BOOK_NET">—</span></div><div class="sub"><span data-bind="SB_GRADED">—</span> settled · taker-conservative</div></div>
  </div>

  <section>
    <div class="card">
      <div class="legend">
        <span class="lg"><i style="background:var(--market-hi)"></i>Market price</span>
        <span class="lg"><i style="background:var(--model-hi)"></i>Our model</span>
        <span class="lg"><i style="background:var(--ens-hi)"></i>Raw ensemble</span>
        <div class="toggle" id="accmode">
          <button data-mode="acc" class="on">CUMULATIVE</button><button data-mode="roll">ROLLING 60</button>
        </div>
      </div>
      <div class="chartwrap"><div id="c_acc"></div></div>
      <div class="legend" style="margin:8px 0 0"><span class="lgnote" id="accnote">Brier score to date — lower = more accurate. The model must cross below the market line before anything trades.</span></div>
    </div>
  </section>

  <section class="grid2">
    <div class="card">
      <div class="sec-head" style="margin-bottom:6px"><h2>Accuracy by city</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--market-hi)"></i>Market</span><span class="lg"><i style="background:var(--model-hi)"></i>Model</span><span class="lgnote">Brier — lower = better</span></div>
      <div class="chartwrap"><div id="c_city"></div></div>
    </div>
    <div class="card">
      <div class="sec-head" style="margin-bottom:6px"><h2>Model calibration</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--model-hi)"></i>Model · dot size = sample</span><span class="lgnote">on the diagonal = perfectly calibrated</span></div>
      <div class="chartwrap"><div id="c_calib"></div></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>Where the edge might be</h2><span class="note">per-bucket accuracy + the pre-registered forward gates — nothing trades real money until a gate passes</span></div>
    <div class="grid2">
      <div class="card">
        <div class="tblwrap"><table id="t_buckets"><thead><tr>
          <th>Bucket</th><th class="num">n</th><th class="num">Model</th><th class="num">Market</th><th class="num">Δ</th><th>Forward gate</th>
        </tr></thead><tbody></tbody></table></div>
      </div>
      <div class="card">
        <div class="legend"><span class="lg"><i style="background:var(--accent)"></i>Paper book equity</span><span class="lgnote">net units / 1 contract per entry · taker fees paid</span></div>
        <div class="chartwrap"><div id="c_equity"></div></div>
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="eyebrow">Candidate edge · forward paper trial</span></div>
    <div class="sec-head" style="margin-bottom:10px"><h2>Structure paper book</h2><span class="chip accent">PAPER ONLY — NO REAL MONEY</span></div>
    <p class="lead-p">A <b>model-free</b> strategy that ignores the forecast entirely: sell over-priced long-shot temperature bins, buy clear favorites, settle against real station readings. Early signal is strong — it stays paper until it clears its pre-registered forward test.</p>
    <div class="strip">
      <div><div class="n pos"><span data-bind="SB_WR">—</span>%</div><div class="k">win rate · <span data-bind="SB_GRADED">—</span> settled</div></div>
      <div><div class="n pos" data-bind="SB_FULL">—</div><div class="k">edge / contract · full band</div></div>
      <div><div class="n pos" data-bind="SB_CORE">—</div><div class="k">edge / contract · core band</div></div>
      <div><div class="n" data-bind="SB_AWAIT">—</div><div class="k">awaiting settlement</div></div>
    </div>
    <p class="lead-p" style="margin-top:10px;margin-bottom:0">A second signal — the model's most selective bucket — shows <b><span data-bind="E3_ROI">—</span> ROI on <span data-bind="E3_N">—</span> bets</b>, still in-sample. Nothing goes live until a bucket logs <b>40+ forward bets</b> beating the market.</p>
  </section>

  <section class="grid2">
    <div class="card">
      <div class="sec-head" style="margin-bottom:6px"><h2>Recent settlements</h2><span class="note">latest graded markets · ● = closer to the outcome</span></div>
      <div class="tblwrap"><table id="t_recent"><thead><tr>
        <th>Date</th><th>City</th><th>Bin</th><th class="num">Model</th><th class="num">Market</th><th>Result</th><th>Closer</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="card">
      <div class="sec-head" style="margin-bottom:6px"><h2>Track record &amp; heartbeat</h2></div>
      <div class="legend"><span class="lg"><i style="background:var(--accent)"></i>Gradable markets</span><span class="lgnote">dashed = <span data-bind="GATE_MKTS_THR">150</span>-market gate</span></div>
      <div class="chartwrap"><div id="c_growth"></div></div>
      <div class="legend" style="margin-top:14px"><span class="lg"><i style="background:var(--accent)"></i>Collector runs / day</span><span class="lgnote">last 30 days</span></div>
      <div class="chartwrap"><div id="c_hb"></div></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>Stations</h2><span class="note">per-city sample, accuracy and settlement-truth freshness</span></div>
    <div class="cities" data-bind-html="CITIES_HTML"></div>
  </section>

  <footer>
    <div class="row"><span><b>PIPELINE</b> collect 2h · truth-eval daily · retrain on demand</span><span><b>HOST</b> GitHub Actions (free)</span><span><b>STORE</b> git</span><span><b>PAGE</b> static + client-side live fetch</span></div>
    <div><b>Method:</b> every prediction is graded against the actual weather-station reading each market settles on — never against the forecast grid it came from. No performance is claimed until a pre-committed sample gate is met (now cleared); the honest headline is that the market still leads. All trading shown is paper; no real orders are placed.</div>
  </footer>
</div>

<script id="D0" type="application/json">%%PAYLOAD%%</script>
<script>
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";
  var D = {};                 // current series
  var lastSeriesJSON = "";    // skip redundant redraws
  var lastGen = null;         // generated_at ISO — "refreshed Xm ago" clock
  var lastCollect = null;     // last_collect_iso — OPERATIONAL pill
  var prevBind = null;        // previous scalars → flash only what changed
  var accMode = "acc";        // cumulative vs rolling toggle
  var css = function (v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); };
  function el(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function niceTop(v) {
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
    var W = Math.max(host.clientWidth || 700, 320), H = W < 520 ? 220 : (opts.h || 280), P = { l: 46, r: 56, t: 14, b: 28 };
    var series = opts.series.filter(function (s) { return s.points.some(function (p) { return p != null; }); });
    if (!series.length) { host.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; return; }
    var n = opts.xLabels.length;
    var vals = series.flatMap(function (s) { return s.points.filter(function (v) { return v != null; }); });
    var yMin = opts.yMin != null ? opts.yMin : Math.min(0, Math.min.apply(null, vals));
    var yMax = opts.yMax != null ? opts.yMax : niceTop(Math.max.apply(null, vals) * 1.08);
    if (yMax <= yMin) yMax = yMin + 1;
    // Optional value-scaled x (opts.xVals + opts.xDomain) — used by the calibration plot so the
    // y=x diagonal is geometrically true; default is even index spacing.
    var xd0 = 0, xd1 = 1;
    if (opts.xVals) {
      xd0 = opts.xDomain ? opts.xDomain[0] : Math.min.apply(null, opts.xVals);
      xd1 = opts.xDomain ? opts.xDomain[1] : Math.max.apply(null, opts.xVals);
    }
    var xOfV = function (v) { return P.l + (W - P.l - P.r) * (v - xd0) / ((xd1 - xd0) || 1); };
    var xOf = function (i) {
      if (opts.xVals) return xOfV(opts.xVals[i]);
      return P.l + (n <= 1 ? 0 : (W - P.l - P.r) * i / (n - 1));
    };
    var yOf = function (v) { return P.t + (H - P.t - P.b) * (1 - (v - yMin) / (yMax - yMin)); };
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    var ticks = 4;
    for (var g = 0; g <= ticks; g++) {
      var yv = yMin + (yMax - yMin) * g / ticks, yy = yOf(yv);
      svg.appendChild(el("line", { class: "gridline", x1: P.l, y1: yy, x2: W - P.r, y2: yy }));
      svg.appendChild(el("text", { class: "axis", x: P.l - 8, y: yy + 3.5, "text-anchor": "end" }, opts.yFmt ? opts.yFmt(yv) : yv.toFixed(2)));
    }
    if (yMin < 0) { // emphasized zero line for P&L-style charts
      svg.appendChild(el("line", { x1: P.l, y1: yOf(0), x2: W - P.r, y2: yOf(0), stroke: css("--ink-3"), "stroke-width": 1, opacity: .6 }));
    }
    if (opts.diagonal) {
      var dx1 = opts.xVals ? xOfV(xd0) : xOf(0), dx2 = opts.xVals ? xOfV(xd1) : xOf(n - 1);
      var dy1 = yOf(opts.xVals ? xd0 : 0), dy2 = yOf(opts.xVals ? xd1 : yMax);
      svg.appendChild(el("line", { x1: dx1, y1: dy1, x2: dx2, y2: dy2, stroke: css("--ink-3"), "stroke-width": 1, "stroke-dasharray": "4 4", opacity: .55 }));
    }
    if (opts.refLine && opts.refLine.y <= yMax) {
      var ry = yOf(opts.refLine.y);
      svg.appendChild(el("line", { x1: P.l, y1: ry, x2: W - P.r, y2: ry, stroke: css("--warn"), "stroke-width": 1.2, "stroke-dasharray": "5 4", opacity: .85 }));
      svg.appendChild(el("text", { class: "axis", x: P.l + 4, y: ry - 5, fill: css("--warn") }, opts.refLine.label));
    }
    if (opts.xVals && opts.xTicks) {
      opts.xTicks.forEach(function (tv) {
        svg.appendChild(el("text", { class: "axis", x: xOfV(tv), y: H - 8, "text-anchor": "middle" },
                           opts.xTickFmt ? opts.xTickFmt(tv) : String(tv)));
      });
    } else {
      // skip a regular tick when it sits within ~a label-width of the always-shown final tick
      var every = Math.ceil(n / 6);
      for (var i = 0; i < n; i++) if ((i % every === 0 && xOf(n - 1) - xOf(i) > 58) || i === n - 1) {
        svg.appendChild(el("text", { class: "axis", x: xOf(i), y: H - 8, "text-anchor": i === n - 1 ? "end" : "middle" }, opts.xLabels[i]));
      }
    }
    var endLabels = [];
    series.forEach(function (s) {
      var pts = s.points.map(function (v, i) { return v == null ? null : [xOf(i), yOf(v)]; }).filter(Boolean);
      if (opts.area && s.fill) {
        var dd = "M" + pts.map(function (p) { return p[0] + "," + p[1]; }).join(" L ");
        dd += " L " + pts[pts.length - 1][0] + "," + yOf(Math.max(yMin, 0)) + " L " + pts[0][0] + "," + yOf(Math.max(yMin, 0)) + " Z";
        svg.appendChild(el("path", { d: dd, fill: s.color, "fill-opacity": .1, stroke: "none" }));
      }
      if (opts.dots) {
        pts.forEach(function (p, pi) {
          var r = opts.sizes ? opts.sizes[pi] : 4.5;
          svg.appendChild(el("circle", { cx: p[0], cy: p[1], r: r, fill: s.color, "fill-opacity": .85, stroke: css("--panel"), "stroke-width": 1.5 }));
        });
      } else {
        svg.appendChild(el("polyline", { points: pts.map(function (p) { return p[0] + "," + p[1]; }).join(" "), fill: "none", stroke: s.color, "stroke-width": s.thick || 2.2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
        var last = pts[pts.length - 1];
        svg.appendChild(el("circle", { cx: last[0], cy: last[1], r: 3.6, fill: s.color }));
        var lbl = el("text", { class: "serieslabel", x: last[0] + 7, y: last[1] + 3.5, fill: s.labelColor || s.color }, opts.endFmt ? opts.endFmt(s) : "");
        svg.appendChild(lbl);
        endLabels.push({ el: lbl, y: last[1] });
      }
    });
    // de-overlap the series end labels (push apart to >= 13px vertical separation)
    endLabels.sort(function (a, b) { return a.y - b.y; });
    for (var li = 1; li < endLabels.length; li++) {
      if (endLabels[li].y - endLabels[li - 1].y < 13) {
        endLabels[li].y = endLabels[li - 1].y + 13;
        endLabels[li].el.setAttribute("y", endLabels[li].y + 3.5);
      }
    }
    host.appendChild(svg);
    if (opts.dots) return;
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    var cross = el("line", { class: "gridline", y1: P.t, y2: H - P.b, "stroke-dasharray": "3 3", opacity: 0 }); svg.appendChild(cross);
    var focus = series.map(function (s) { var c = el("circle", { r: 4, fill: s.color, stroke: css("--panel"), "stroke-width": 1.5, opacity: 0 }); svg.appendChild(c); return c; });
    svg.appendChild(el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }));
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
    var W = Math.max(host.clientWidth || 700, 320), H = W < 520 ? 220 : (opts.h || 280), P = { l: 46, r: 14, t: 12, b: opts.showN === false ? 24 : 40 };
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
      var x0 = P.l + gw * gi, bw = Math.min(26, gw / (keys.length + 1)), inner = bw * keys.length + (keys.length - 1) * 4;
      var start = x0 + (gw - inner) / 2;
      keys.forEach(function (k, ki) {
        var v = grp[k.key], x = start + ki * (bw + 4), y = yOf(v);
        var bar = el("rect", { x: x, y: y, width: bw, height: Math.max(0, yOf(0) - y), rx: 3, fill: k.color });
        bar.addEventListener("mousemove", function () {
          tip.innerHTML = '<div class="tt">' + grp.label + (showN ? ' · n=' + grp.n : '') + '</div>' + keys.map(function (kk) { return '<div class="tr"><span><i style="background:' + kk.color + '"></i>' + kk.name + '</span><b>' + tf(grp[kk.key]) + '</b></div>'; }).join("");
          tip.style.opacity = 1; var hr = host.getBoundingClientRect();
          tip.style.left = Math.min(Math.max((x0 + gw / 2) / W * hr.width - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = "6px";
        });
        bar.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
        svg.appendChild(bar);
      });
      if ((gi % lblEvery === 0 && (opts.groups.length - 1 - gi) * gw > 48) || gi === opts.groups.length - 1) {
        svg.appendChild(el("text", { class: "axis", x: x0 + gw / 2, y: H - (showN ? 22 : 8), "text-anchor": "middle" }, grp.label));
        if (showN) svg.appendChild(el("text", { class: "axis", x: x0 + gw / 2, y: H - 8, "text-anchor": "middle", opacity: .7 }, "n=" + grp.n));
      }
    });
    host.appendChild(svg);
  }

  // ---- tables ----
  function renderBuckets() {
    var tb = document.querySelector("#t_buckets tbody");
    if (!tb) return;
    var rows = D.buckets || [], gate = D.gate_fwd_n || 40, nd = D.nom_date || "Jul 12";
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="6" style="color:var(--ink-3)">No bucket data yet.</td></tr>'; return; }
    tb.innerHTML = rows.map(function (r) {
      var d = r.market - r.model;
      var chip = d > 0.0005 ? '<span class="chip good">model +' + d.toFixed(3) + '</span>'
               : d < -0.0005 ? '<span class="chip bad">mkt +' + (-d).toFixed(3) + '</span>'
               : '<span class="chip">even</span>';
      var gateCell = "—";
      if (r.nom) {
        var pct = Math.min(100, r.fwd_n / gate * 100);
        var ahead = r.fwd_model != null && r.fwd_market != null && r.fwd_model <= r.fwd_market;
        gateCell = '<span class="gatebar"><i style="width:' + pct.toFixed(0) + '%"></i></span> '
                 + '<span class="mono" style="font-size:11px">' + r.fwd_n + "/" + gate + "</span>"
                 + (r.fwd_n >= 3 ? (ahead ? ' <span class="chip good">on track</span>' : ' <span class="chip warn">behind</span>') : "");
      }
      return "<tr><td><b>" + esc(r.b) + "</b>" + (r.nom ? ' <span class="chip accent">NOMINATED</span>' : "") + "</td>"
           + '<td class="num">' + r.n + "</td>"
           + '<td class="num">' + r.model.toFixed(3) + "</td>"
           + '<td class="num">' + r.market.toFixed(3) + "</td>"
           + '<td class="num">' + chip + "</td>"
           + "<td>" + gateCell + "</td></tr>";
    }).join("");
    var note = document.querySelector("#t_buckets"); // caption note under table
    if (note && !document.getElementById("bnote")) {
      var p = document.createElement("div"); p.id = "bnote";
      p.style.cssText = "font-size:11px;color:var(--ink-3);margin-top:8px";
      p.textContent = "Forward gate: " + gate + "+ markets graded after " + nd + " nomination with model ≤ market Brier. Buckets = city × lead time.";
      note.parentNode.appendChild(p);
    }
  }
  function renderRecent() {
    var tb = document.querySelector("#t_recent tbody");
    if (!tb) return;
    var rows = D.recent || [];
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="7" style="color:var(--ink-3)">No settlements yet.</td></tr>'; return; }
    tb.innerHTML = rows.map(function (r) {
      var mErr = Math.abs(r.model - r.out), kErr = Math.abs(r.market - r.out);
      var closer = mErr < kErr ? '<span class="winner-dot" style="background:var(--model-hi)"></span> model'
                 : kErr < mErr ? '<span class="winner-dot" style="background:var(--market-hi)"></span> market'
                 : "tie";
      return "<tr><td>" + esc(r.d) + "</td><td>" + esc(r.city) + "</td><td><b>" + esc(r.bin) + "</b></td>"
           + '<td class="num">' + (r.model * 100).toFixed(0) + "%</td>"
           + '<td class="num">' + (r.market * 100).toFixed(0) + "%</td>"
           + "<td>" + (r.out ? '<span class="chip good">YES</span>' : '<span class="chip">NO</span>') + "</td>"
           + '<td style="font-size:11px">' + closer + "</td></tr>";
    }).join("");
  }

  // ===== chart render (reads global D) =====
  function draw() {
    ["c_acc", "c_city", "c_calib", "c_growth", "c_hb", "c_equity"].forEach(function (id) { var h = document.getElementById(id); if (h) h.innerHTML = ""; });
    document.querySelectorAll(".chartwrap .tip").forEach(function (t) { t.remove(); });

    var acc = (accMode === "roll" ? D.roll : D.acc) || [];
    var note = document.getElementById("accnote");
    if (note) note.textContent = accMode === "roll"
      ? "Trailing 60-market Brier — recent form. Lower = more accurate; watch whether the model's recent window closes the gap."
      : "Brier score to date — lower = more accurate. The model must cross below the market line before anything trades.";
    if (acc.length) lineChart("c_acc", {
      h: 290, xLabels: acc.map(function (r) { return r.t; }),
      yFmt: function (v) { return v.toFixed(2); }, tipFmt: function (v) { return v.toFixed(4); },
      tipSuffix: function (i) { return "  ·  n=" + acc[i].n; },
      endFmt: function (s) { var last = s.points.filter(function (v) { return v != null; }).slice(-1)[0]; return last != null ? last.toFixed(3) : ""; },
      series: [
        { name: "Market", color: css("--market"), labelColor: css("--market-hi"), points: acc.map(function (r) { return r.market; }) },
        { name: "Model", color: css("--model"), labelColor: css("--model-hi"), points: acc.map(function (r) { return r.model; }) },
        { name: "Ensemble", color: css("--ens"), labelColor: css("--ens-hi"), points: acc.map(function (r) { return r.ens; }) }
      ]
    });

    var city = D.city || [];
    barChart("c_city", {
      h: 280, yFmt: function (v) { return v.toFixed(2); },
      groups: city.map(function (r) { return { label: r.city, n: r.n, market: r.market, model: r.model }; }),
      series: [{ key: "market", name: "Market", color: css("--market") }, { key: "model", name: "Model", color: css("--model") }]
    });

    var calib = D.calib || [];
    if (calib.length) {
      var maxN = Math.max.apply(null, calib.map(function (r) { return r.n; }));
      lineChart("c_calib", {
        h: 280, dots: true, diagonal: true, yMin: 0, yMax: 1,
        xLabels: calib.map(function (r) { return r.p.toFixed(2); }),
        xVals: calib.map(function (r) { return r.p; }), xDomain: [0, 1],
        xTicks: [0, 0.25, 0.5, 0.75, 1],
        xTickFmt: function (v) { return v === 0 ? "0" : v === 1 ? "1" : v.toFixed(2); },
        yFmt: function (v) { return v.toFixed(1); },
        sizes: calib.map(function (r) { return 3 + 6 * Math.sqrt(r.n / maxN); }),
        series: [{ name: "Model", color: css("--model-hi"), points: calib.map(function (r) { return r.f; }) }]
      });
    } else { var hc = document.getElementById("c_calib"); if (hc) hc.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; }

    var eq = D.equity || [];
    if (eq.length) lineChart("c_equity", {
      h: 264, area: true, xLabels: eq.map(function (r) { return r.t; }),
      yFmt: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "u"; },
      tipFmt: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "u"; },
      tipSuffix: function (i) { return "  ·  sh " + eq[i].sh.toFixed(2) + " / fav " + eq[i].fav.toFixed(2); },
      endFmt: function (s) { var last = s.points.slice(-1)[0]; return last != null ? (last >= 0 ? "+" : "") + last.toFixed(2) + "u" : ""; },
      series: [{ name: "Net units", color: css("--accent"), points: eq.map(function (r) { return r.v; }), fill: true, thick: 2.4 }]
    });
    else { var he = document.getElementById("c_equity"); if (he) he.innerHTML = '<div style="color:var(--ink-3);font-size:12px;padding:20px 0">No settled paper entries yet.</div>'; }

    var gr = D.growth || [];
    if (gr.length) lineChart("c_growth", {
      h: 210, area: true, xLabels: gr.map(function (r) { return r.t; }),
      yFmt: function (v) { return String(Math.round(v)); },
      tipFmt: function (v) { return Math.round(v); },
      refLine: D.gate_line ? { y: D.gate_line, label: "gate " + D.gate_line } : null,
      endFmt: function (s) { var last = s.points.slice(-1)[0]; return last != null ? String(last) : ""; },
      series: [{ name: "Markets", color: css("--accent"), points: gr.map(function (r) { return r.m; }), fill: true }]
    });

    var hb = D.heartbeat || [];
    barChart("c_hb", {
      h: 150, showN: false, tipFmt: function (v) { return String(Math.round(v)); }, yFmt: function (v) { return String(Math.round(v)); },
      groups: hb.map(function (r) { return { label: r.d.slice(5), n: r.n, runs: r.n }; }),
      series: [{ key: "runs", name: "Runs", color: css("--accent") }]
    });

    renderBuckets();
    renderRecent();
  }

  // ===== live data layer =====
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function flash(e) { if (reduceMotion) return; e.classList.remove("flash"); void e.offsetWidth; e.classList.add("flash"); }
  function applyBind(bind, html) {
    var b = bind || {}, first = prevBind === null;
    Object.keys(b).forEach(function (k) {
      var val = String(b[k]);
      document.querySelectorAll('[data-bind="' + k + '"]').forEach(function (e) {
        if (e.textContent === val) return;
        e.textContent = val;
        if (!first && prevBind[k] !== undefined && String(prevBind[k]) !== val) flash(e);
      });
    });
    prevBind = b;
    var h = html || {};
    Object.keys(h).forEach(function (k) {
      document.querySelectorAll('[data-bind-html="' + k + '"]').forEach(function (e) { if (e.innerHTML !== h[k]) e.innerHTML = h[k]; });
    });
  }
  function fmtAgo(iso) {
    var t = Date.parse(iso);
    if (isNaN(t)) return null;
    var s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 45) return "just now";
    if (s < 5400) return Math.max(1, Math.round(s / 60)) + "m ago";
    if (s < 172800) return Math.round(s / 3600) + "h ago";
    return Math.round(s / 86400) + "d ago";
  }
  function tickClock() {
    var elm = document.querySelector("[data-ago]");
    if (elm && lastGen) elm.textContent = "refreshed " + (fmtAgo(lastGen) || "—");
    var uc = document.getElementById("utcclock");
    if (uc) uc.textContent = new Date().toISOString().slice(11, 19) + " UTC";
    // OPERATIONAL pill: green if the collector snapshotted within the last 5h (cadence is 2h)
    var pill = document.getElementById("statuspill"), st = document.getElementById("statustext");
    if (pill && st) {
      if (lastCollect) {
        var hrs = (Date.now() - Date.parse(lastCollect)) / 36e5;
        var ok = hrs < 5;
        pill.className = "pill " + (ok ? "good" : "warn");
        st.textContent = ok ? "OPERATIONAL" : "COLLECTOR STALE";
      } else { st.textContent = "OPERATIONAL"; }
    }
    var cb = document.getElementById("collectorbig"), cs = document.getElementById("collectorsub");
    if (cb && lastCollect) { cb.textContent = fmtAgo(lastCollect) || "—"; cs.textContent = "last market snapshot"; }
  }
  function apply(P) {
    if (!P || !P.bind) return false;
    applyBind(P.bind, P.html);
    lastGen = P.generated_at || null;
    lastCollect = P.last_collect_iso || (P.series && P.series.last_collect_iso) || null;
    var sj = JSON.stringify(P.series || {});
    if (sj !== lastSeriesJSON) { D = P.series || {}; lastSeriesJSON = sj; draw(); }
    document.body.classList.add("ready");
    tickClock();
    return true;
  }
  function applyInline() {
    try { return apply(JSON.parse(document.getElementById("D0").textContent || "{}")); }
    catch (e) { document.body.classList.add("ready"); return false; }
  }
  function load() {
    fetch("data.json?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("http " + r.status); return r.json(); })
      .then(function (P) { apply(P); document.getElementById("errbar").style.display = "none"; })
      .catch(function () { if (!document.body.classList.contains("ready")) document.getElementById("errbar").style.display = "block"; });
  }

  // accuracy-chart toggle
  document.getElementById("accmode").addEventListener("click", function (ev) {
    var b = ev.target.closest("button");
    if (!b || b.classList.contains("on")) return;
    accMode = b.getAttribute("data-mode");
    this.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x === b); });
    draw();
  });

  applyInline();                    // instant first paint from the embedded copy
  load();                           // then refresh from the network (no-op offline / file://)
  setInterval(load, 120000);        // pick up the next publish while the tab stays open
  setInterval(tickClock, 1000);     // UTC clock + ago clock + status pill
  var _rt; window.addEventListener("resize", function () { clearTimeout(_rt); _rt = setTimeout(draw, 180); });
})();
</script>
"""


if __name__ == "__main__":
    main()
