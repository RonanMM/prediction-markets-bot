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
    # Leg2 favourites — often "0 graded" this early; capture so the UI can say "pending" honestly.
    fav = re.search(r"Leg2 favorite core.*?n=(\d+)\s+wr\s+(\d+)%.*?taker\s+([+\-][\d.]+)", book)
    d["sb_fav_graded"] = fav.group(1) if fav else "0"
    # Leg 1b moderate-shoulder FORWARD gate (pre-reg 2026-07-23) — forward-only progress.
    mod = re.search(r"Leg1b moderate.*?FORWARD gate[^:]*:\s*n=(\d+)\s+taker\s+([+\-][\d.]+)\s+\[([^\]]+)\]",
                    book, re.DOTALL)
    if mod:
        d["sb_mod_fwd_n"], d["sb_mod_fwd"] = mod.group(1), mod.group(2)
        gm = re.match(r"(\d+)/(\d+)", mod.group(3))
        d["sb_mod_need"] = gm.group(2) if gm else "80"
        d["sb_mod_pass"] = "1" if "GATE" in mod.group(3) else "0"
    # dispersion monitor — per-month spread calibration std(z) (1.0 = honest, >1.15 overconfident)
    d["disp"] = [{"m": m, "z": float(z)} for m, _n, z in
                 re.findall(r"Tmax (\d{4}-\d{2})\s+(\d+)\s+([\d.]+)", oos)]
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

        # verdict scoreboard — Brier on the common set, all-time AND trailing 60 (the toggle)
        cs = c.sort_values("td")

        def _sb(w):
            return {"market": round(float(w["b_mkt"].mean()), 4),
                    "ens": round(float(w["b_ens"].mean()), 4),
                    "model": round(float(w["b_model"].mean()), 4), "n": int(len(w))}
        out["score"] = {"all": _sb(cs), "recent": _sb(cs.tail(60))}

        # ROI at production params (IN-SAMPLE, noisy) — model AND ensemble, for the honest
        # accuracy-vs-profit panel. The ensemble's positive ROI while it LOSES on Brier is the
        # exact "127% ROI" mirage this project warns about; surfaced only with that caveat,
        # never as an edge claim. Same betting policy for both (MIN_EDGE + positive Kelly).
        try:
            import config as _cfg

            def _roi(df, recent=False):
                b = df[(df["abs_edge"].astype(float) >= _cfg.MIN_EDGE) & (df["kelly"].astype(float) > 0)].copy()
                if recent:   # rolling last 60 BETS — ROI's natural unit, so the count matches the toggle
                    b = b.assign(_td=pd.to_datetime(b["target_date"], errors="coerce")).sort_values("_td").tail(60)
                r = ev._roi_at_production(b)
                return {"roi": round(float(r["roi"]), 4), "bets": int(r["bets"]), "wins": int(r["wins"])}
            out["roi"] = {"model": {"all": _roi(cal), "recent": _roi(cal, True)},
                          "ens":   {"all": _roi(ens), "recent": _roi(ens, True)}}
        except Exception as e:
            out["roi_error"] = f"{type(e).__name__}: {e}"

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

        # accuracy by city — model, market, AND raw ensemble Brier (real per-city ordering varies)
        ens2 = ens.copy()
        ens2["b_ens"] = (ens2["forecast_prob"] - ens2["outcome"]) ** 2
        ecity = ens2.groupby("city")["b_ens"].mean()
        city = []
        for cty, g in cal.groupby("city"):
            row = {"city": str(cty), "model": round(float(g["b_model"].mean()), 4),
                   "market": round(float(g["b_mkt"].mean()), 4), "n": int(len(g))}
            if str(cty) in ecity.index:
                row["ens"] = round(float(ecity.loc[str(cty)]), 4)
            city.append(row)
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

        # recent settlements feed — the last 60 graded markets, newest first
        rec = cal.sort_values("td").tail(60)
        out["recent"] = [{
            "d": pd.Timestamp(r.td).strftime("%b %-d"),
            "city": str(r.city),
            "bin": _bin_label(str(r.question)),
            "model": round(float(r.forecast_prob), 3),
            "market": round(float(getattr(r, mkt_col)), 3),
            "out": int(r.outcome),
        } for r in rec.itertuples()][::-1]

        # who was closer, last 60 common markets — per city (recent window ⇒ no Hong Kong)
        wc = cs.tail(60).copy()
        wc["mwin"] = wc["b_model"] < wc["b_mkt"]
        won = [{"city": str(k), "mwin": int(g["mwin"].sum()), "n": int(len(g))}
               for k, g in wc.groupby("city")]
        won.sort(key=lambda r: r["n"], reverse=True)
        out["woncity"] = {"rows": won, "mwin": int(wc["mwin"].sum()), "n": int(len(wc))}

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
    series["disp"] = d.get("disp", [])
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
            "SB_FAV_GRADED": G("sb_fav_graded", "0"),
            "SB_MOD_FWD_N": G("sb_mod_fwd_n", "0"), "SB_MOD_FWD": G("sb_mod_fwd", "—"),
            "SB_MOD_NEED": G("sb_mod_need", "80"), "SB_MOD_PASS": G("sb_mod_pass", "0"),
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
<title>prediction-markets-bot — live</title>
<style>
  :root{
    color-scheme:dark;
    --page:#0d0d0d; --surf:#1a1a19; --surf2:#201f1e;
    --line:rgba(255,255,255,.10); --grid:#2c2c2a; --axis:#3a3a37;
    --ink:#ffffff; --ink2:#c3c2b7; --muted:#8b897f; --faint:#5a584f;
    --market:#3987e5; --model:#e66767; --ens:#8b897f;
    --good:#22c55e; --warn:#fab219; --bad:#e05252;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Roboto,sans-serif;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html{background:var(--page)}
  body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;line-height:1.5;font-feature-settings:"tnum"}
  .wrap{max-width:980px;margin:0 auto;padding:0 24px 80px;opacity:0;transition:opacity .5s ease}
  body.ready .wrap{opacity:1}
  .mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

  /* masthead */
  .mast{display:flex;justify-content:space-between;align-items:center;padding:26px 0 16px;border-bottom:1px solid var(--line);flex-wrap:wrap;gap:10px}
  .mark{font-size:15px;font-weight:600} .mark b{color:var(--muted);font-weight:400}
  .subm{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin-top:5px}
  .meta{text-align:right;font-family:var(--mono);font-size:10.5px;color:var(--muted);line-height:1.7}
  .pill{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--good);letter-spacing:.03em}
  .pill.warn{color:var(--warn)}
  .dot{width:7px;height:7px;border-radius:50%;background:currentColor;display:inline-block}
  .dot.live{animation:pulse 2.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,currentColor 55%,transparent)}50%{box-shadow:0 0 0 5px transparent}}

  /* sections */
  section{padding-top:38px}
  .shd{display:flex;align-items:baseline;gap:13px;margin-bottom:16px}
  .shd .n{font-family:var(--mono);font-size:11px;color:var(--faint)}
  .shd h2{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0}
  .shd .r{margin-left:auto;font-size:10.5px;color:var(--faint);text-align:right}

  /* verdict */
  .eyebrow{font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
  .verdict{font-size:26px;line-height:1.24;font-weight:650;letter-spacing:-.015em;margin:11px 0 16px;max-width:24ch}
  .verdict em{font-style:normal}
  .vhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:11px;flex-wrap:wrap;gap:8px}
  .vhead .lbl{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
  .seg{display:inline-flex;background:var(--surf2);border:1px solid var(--line);border-radius:7px;padding:3px}
  .seg button{font-family:var(--sans);font-size:11px;color:var(--muted);background:none;border:0;padding:6px 13px;border-radius:5px;cursor:pointer}
  .seg button.on{background:var(--market);color:#fff;font-weight:600}
  .rank{background:var(--surf);border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .rrow{display:grid;grid-template-columns:26px 1fr 92px 88px;align-items:center;gap:12px;padding:13px 16px;border-top:1px solid var(--line)}
  .rrow:first-child{border-top:0}
  .rrow.lead{background:linear-gradient(90deg,rgba(57,135,229,.10),transparent 60%)}
  .rrow .i{font-family:var(--mono);font-size:12px;color:var(--faint)} .rrow.lead .i{color:var(--market)}
  .sw{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:9px;vertical-align:middle}
  .sw.market{background:var(--market)} .sw.model{background:var(--model)} .sw.ens{background:var(--ens)}
  .rrow .who{font-size:13.5px;color:var(--ink2)} .rrow.lead .who{color:var(--ink);font-weight:600}
  .tagl{font-size:8.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--market);border:1px solid rgba(57,135,229,.5);border-radius:3px;padding:2px 6px;margin-left:9px}
  .rrow .val{text-align:right;font-family:var(--mono);font-size:17px} .rrow.lead .val{color:var(--good)}
  .rrow .d{text-align:right;font-family:var(--mono);font-size:12px;color:var(--bad)} .rrow .d.z{color:var(--faint)}
  .caveat{display:none;margin-top:10px;font-size:11px;line-height:1.5;color:var(--warn);background:rgba(250,178,25,.07);border:1px solid rgba(250,178,25,.28);border-radius:7px;padding:9px 12px}
  .caveat.show{display:block} .caveat b{color:#f0c24a}
  .roiwarn{margin-top:12px;font-size:11px;line-height:1.55;color:var(--warn);background:rgba(250,178,25,.07);border:1px solid rgba(250,178,25,.28);border-radius:7px;padding:10px 12px} .roiwarn b{color:#f0c24a}
  table.cmp td:first-child{color:var(--ink2);font-weight:500} table.cmp .rowlbl small{color:var(--faint);font-weight:400}

  .gloss{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}
  .gc{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:13px 14px;border-top:2px solid var(--faint)}
  .gc.b{border-top-color:var(--market)} .gc.g{border-top-color:var(--ens)} .gc.r{border-top-color:var(--model)}
  .gc .h{font-size:12px;font-weight:600;display:flex;align-items:center;gap:8px} .gc .h .sw{margin:0}
  .gc p{font-size:11px;line-height:1.5;color:var(--muted);margin:7px 0 0}
  .brierdef{font-size:10.5px;color:var(--faint);margin-top:10px;font-family:var(--mono)} .brierdef b{color:var(--muted);font-weight:400}
  .lede{font-size:13px;line-height:1.65;color:var(--ink2);margin:18px 2px 0;max-width:74ch} .lede b{color:var(--ink);font-weight:600} .lede .m{font-family:var(--mono);color:var(--ink)}

  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}
  .kpi{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:14px 15px;border-left:2px solid var(--faint)}
  .kpi.g{border-left-color:var(--good)} .kpi.r{border-left-color:var(--model)} .kpi.b{border-left-color:var(--market)}
  .kpi .k{font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
  .kpi .v{font-size:23px;font-weight:650;font-family:var(--mono);margin-top:7px} .kpi .v.g{color:var(--good)} .kpi .v.r{color:var(--model)}
  .kpi .s{font-size:10px;color:var(--faint);margin-top:3px}

  .cwrap{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .panel{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:16px 17px}
  .find{font-size:14px;font-weight:600;letter-spacing:-.01em;margin-bottom:2px}
  .findsub{font-size:10.5px;color:var(--muted);margin-bottom:12px}
  .leg{display:flex;gap:14px;font-size:10px;color:var(--ink2);margin-bottom:6px;flex-wrap:wrap} .leg span{display:flex;align-items:center;gap:5px} .leg i{width:11px;height:3px;border-radius:2px;display:inline-block}
  .cap{font-size:10.5px;line-height:1.55;color:var(--muted);margin-top:11px} .cap b{color:var(--ink2);font-weight:500}
  .statrow{display:flex;gap:20px;margin-top:14px;padding-top:13px;border-top:1px solid var(--line);flex-wrap:wrap}
  .st .k{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)} .st .v{font-size:16px;font-family:var(--mono);margin-top:4px} .st .v small{font-size:10px;color:var(--faint)}

  .chartwrap{position:relative;width:100%}
  svg.chart{width:100%;height:auto;display:block;overflow:visible}
  .gridline{stroke:var(--grid);stroke-width:1}
  .axis{fill:var(--faint);font-family:var(--sans);font-size:10px}
  .serieslabel{font-family:var(--mono);font-size:10.5px;font-weight:650}
  .tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;background:var(--surf2);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:11.5px;box-shadow:0 10px 30px rgba(0,0,0,.5);z-index:5;min-width:120px}
  .tip .tt{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-bottom:5px}
  .tip .tr{display:flex;align-items:center;justify-content:space-between;gap:12px;font-family:var(--mono)}
  .tip .tr span{display:inline-flex;align-items:center;gap:6px;color:var(--ink2)} .tip .tr i{width:8px;height:8px;border-radius:2px} .tip .tr b{color:var(--ink)}

  table.data{width:100%;border-collapse:collapse;font-size:12.5px}
  table.data th{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:500;text-align:left;padding:0 12px 9px 0;border-bottom:1px solid var(--line);white-space:nowrap}
  table.data th.num,table.data td.num{text-align:right;font-family:var(--mono)}
  table.data td{padding:9px 12px 9px 0;border-bottom:1px solid var(--line);color:var(--ink2);white-space:nowrap}
  table.data tr:last-child td{border-bottom:0}
  td.city{color:var(--ink);font-weight:500} .pos{color:var(--good)} .neg{color:var(--model)}
  .pill2{font-size:9px;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:3px;border:1px solid var(--line);color:var(--muted)}
  .pill2.on{color:var(--good);border-color:rgba(34,197,94,.45);background:rgba(34,197,94,.08)}
  .pill2.warn{color:var(--warn);border-color:rgba(250,178,25,.4);background:rgba(250,178,25,.08)}
  .paperflag{font-size:9px;letter-spacing:.13em;text-transform:uppercase;color:var(--warn);border:1px solid rgba(250,178,25,.4);background:rgba(250,178,25,.08);border-radius:3px;padding:3px 9px}
  .gatebar{position:relative;width:80px;height:6px;border-radius:3px;background:var(--surf2);border:1px solid var(--line);display:inline-block;vertical-align:middle;overflow:hidden}
  .gatebar i{position:absolute;inset:0 auto 0 0;background:var(--good)}
  .tblscroll{max-height:340px;overflow-y:auto}

  .wonbars{display:flex;flex-direction:column;gap:9px;margin-top:2px}
  .wb{display:flex;align-items:center;gap:10px;font-size:11px} .wb .nm{width:70px;color:var(--ink2)}
  .wb .bar{flex:1;height:15px;background:var(--surf2);border-radius:3px;overflow:hidden;display:flex}
  .wb .bar .mk{background:var(--model);height:100%} .wb .bar .mkt{background:var(--market);height:100%}
  .wb .c{font-family:var(--mono);font-size:10px;color:var(--muted);width:46px;text-align:right}

  .cities{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
  .city{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:12px 13px}
  .city .cn{font-size:13px;font-weight:600} .city .cc{font-size:10px;color:var(--faint);font-weight:400}
  .city .cb{font-size:10px;color:var(--muted);margin:6px 0 8px;font-family:var(--mono)}

  footer{margin-top:46px;padding-top:16px;border-top:1px solid var(--line);font-size:10px;line-height:1.8;color:var(--faint);font-family:var(--mono)} footer b{color:var(--muted);font-weight:400}
  #errbar{display:none;margin:14px 0;padding:9px 13px;border-radius:8px;font-size:12px;color:var(--warn);background:rgba(250,178,25,.07);border:1px solid rgba(250,178,25,.3)}
  .flash{animation:flashbg 1.1s ease}
  @keyframes flashbg{0%,12%{background:color-mix(in srgb,var(--market) 22%,transparent)}100%{background:transparent}}
  @media(max-width:720px){.cwrap,.gloss{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.cities{grid-template-columns:repeat(2,1fr)}}
  @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.wrap{opacity:1}}
</style>

<div class="wrap">
  <div id="errbar">Couldn't reach the live data feed — showing the last snapshot. Retrying…</div>

  <div class="mast">
    <div><div class="mark">prediction-markets-bot <b>/ weather</b></div><div class="subm">Polymarket temperature markets · 5 cities · autonomous, cloud-run</div></div>
    <div class="meta">
      <div><span class="pill" id="statuspill"><span class="dot live"></span><span id="statustext">live</span></span> · <span data-ago>refreshing…</span></div>
      <div><span data-bind="N_MKTS">—</span> markets · through <span data-bind="UPDATED">—</span></div>
    </div>
  </div>

  <!-- 00 VERDICT -->
  <section style="padding-top:32px">
    <div class="eyebrow">The verdict · <span id="wLabel">the full track record</span></div>
    <div class="verdict" id="verdictLine">Who forecasts the weather markets best?</div>

    <div class="vhead">
      <span class="lbl">Window</span>
      <span class="seg" id="winseg"><button class="on" data-win="all">All time</button><button data-win="recent">Last 60</button></span>
    </div>
    <div class="rank" id="rank"></div>
    <div class="caveat" id="cav"><b>Small-sample warning.</b> 60 markets is a noisy window — one clean settlement swings it, and recent form is where this project has been fooled before. Read the all-time number for the verdict.</div>

    <div class="gloss">
      <div class="gc b"><div class="h"><span class="sw market"></span>Market price</div><p>Polymarket's live YES price — the crowd's money-weighted probability. The benchmark to beat.</p></div>
      <div class="gc g"><div class="h"><span class="sw ens"></span>Raw ensemble</div><p>A 122-member weather ensemble (ICON + GFS + ECMWF). Its raw spread becomes a probability — no post-processing.</p></div>
      <div class="gc r"><div class="h"><span class="sw model"></span>Calibrated model</div><p>Our forecast: a multi-model blend, then EMOS — a statistical correction of bias &amp; spread against each station's history.</p></div>
    </div>
    <div class="brierdef"><b>Brier score</b> = mean squared error of the probabilities. 0 = perfect · lower = more accurate.</div>

    <div class="panel" style="margin-top:16px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:12px">
        <div>
          <div class="find" style="font-size:13px">Accuracy vs. profit — why we trust the Brier, not the ROI</div>
          <div class="findsub" id="roisub" style="margin-bottom:0">the same forecasters, scored two ways · all-time</div>
        </div>
        <span class="seg" id="roiseg" style="flex:none"><button class="on" data-rwin="all">All time</button><button data-rwin="recent">Last 60</button></span>
      </div>
      <div style="overflow-x:auto"><table class="data cmp" id="t_roi"><thead><tr>
        <th>Metric</th><th class="num">Market</th><th class="num">Ensemble</th><th class="num">Model</th>
      </tr></thead><tbody></tbody></table></div>
      <div class="roiwarn" id="roiwarn"></div>
    </div>

    <div class="kpis">
      <div class="kpi b"><div class="k">Markets graded</div><div class="v" data-bind="N_MKTS">—</div><div class="s">settlement truth</div></div>
      <div class="kpi g"><div class="k">Sample gate</div><div class="v g" data-bind="GATE_STATUS">—</div><div class="s"><span data-bind="GATE_MKTS_THR">240</span> required</div></div>
      <div class="kpi r"><div class="k">Model − market</div><div class="v r" id="k_gap">—</div><div class="s">Brier gap</div></div>
      <div class="kpi"><div class="k">Live exposure</div><div class="v">$0</div><div class="s">paper only</div></div>
    </div>
    <p class="lede" data-bind-html="TAKEAWAY"></p>
  </section>

  <!-- 01 ACCURACY -->
  <section>
    <div class="shd"><span class="n">01</span><h2>Accuracy</h2><span class="r">over time · by city · by temperature</span></div>
    <div class="cwrap">
      <div class="panel">
        <div class="find">The market has led all season.</div>
        <div class="findsub">Running Brier as markets settle · lower = better</div>
        <div class="leg"><span><i style="background:var(--market)"></i>Market</span><span><i style="background:var(--model)"></i>Model</span><span><i style="background:var(--ens)"></i>Ensemble</span></div>
        <div class="chartwrap"><div id="c_acc"></div></div>
        <div class="cap"><b>The model (red) must cross below the market (blue) and stay.</b> It hasn't, once.</div>
      </div>
      <div class="panel">
        <div class="find">The model loses to the market in every city.</div>
        <div class="findsub">Brier per city · market vs ensemble vs model</div>
        <div class="leg"><span><i style="background:var(--market);border-radius:50%"></i>Market</span><span><i style="background:var(--ens);border-radius:50%"></i>Ensemble</span><span><i style="background:var(--model);border-radius:50%"></i>Model</span></div>
        <div class="chartwrap"><div id="c_city"></div></div>
        <div class="cap"><b>Each row is one city.</b> The model dot (red) sits right of the market (blue) everywhere — worse in all five.</div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="statrow" style="border-top:0;padding-top:0;margin-top:0">
        <div class="st"><div class="k">Temp accuracy · CRPS</div><div class="v" style="color:var(--good)"><span data-bind="CRPS_MODEL">—</span> <small>model</small></div></div>
        <div class="st"><div class="k">vs ensemble</div><div class="v"><span data-bind="CRPS_ENS">—</span> <small>°C err</small></div></div>
        <div class="st" style="flex:1;min-width:240px"><div class="k">what it means</div><div class="v" style="font-size:11.5px;font-family:var(--sans);color:var(--ink2);line-height:1.5">On the raw temperature the calibration <b style="color:var(--good)">does help</b> — the model's forecast is sharper than the ensemble's. It just doesn't beat the market's bin <em>prices</em>.</div></div>
      </div>
    </div>
  </section>

  <!-- 02 DIAGNOSTICS -->
  <section>
    <div class="shd"><span class="n">02</span><h2>Model diagnostics</h2><span class="r">calibration · overconfidence · buckets</span></div>
    <div class="cwrap">
      <div class="panel">
        <div class="find">The model's long-shot odds are too low.</div>
        <div class="findsub">Real outcome rate minus what the model said, by confidence bin</div>
        <div class="leg"><span><i style="background:var(--market)"></i>happened more than it said</span><span><i style="background:var(--model)"></i>happened less</span></div>
        <div class="chartwrap"><div id="c_calib"></div></div>
        <div class="cap"><b>Read:</b> a bin the model calls a long shot happens more often than it claimed (blue, up); its mid-confidence calls land a bit less (red, down). Off at the edges only.</div>
      </div>
      <div class="panel">
        <div class="find">The overconfidence was a spring thing.</div>
        <div class="findsub">Spread calibration by month · 1.0 = honest, &gt;1.15 = overconfident</div>
        <div class="chartwrap"><div id="c_disp"></div></div>
        <div class="cap"><b>Read:</b> overconfident in spring's volatile weather (amber), calibrated by summer (green). <b>So we don't widen the tails</b> — that would over-correct the calm months.</div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="find">Only a couple of buckets beat the market — and only in-sample.</div>
      <div class="findsub">A "bucket" = a city × lead-time slice. Where the model loses it is not nominated; the ones that win must prove it forward before any real order.</div>
      <div style="overflow-x:auto"><table class="data" id="t_buckets" style="margin-top:4px"><thead><tr>
        <th>Bucket</th><th class="num">n</th><th class="num">Model</th><th class="num">Market</th><th class="num">Δ</th><th>Forward gate</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <!-- 03 PAPER BOOK -->
  <section>
    <div class="shd"><span class="n">03</span><h2>Paper book</h2><span class="r">model-free structure legs</span> <span class="paperflag">Paper — no real money</span></div>
    <p class="lede" style="margin:0 2px 14px">A <b>separate book</b> from the model above — it bets on market <b>structure</b>, not the weather. <b>Leg 1</b> sells over-priced 5–35¢ shoulder bins; <b>Leg 2</b> buys 65–85¢ YES-favourites &gt;12h before close. Independent mispricings, each gated on its own before a single real order. <b>Leg 1b</b> refines Leg 1 to the over-priced 10–25¢ sub-band (pre-registered 2026-07-23, forward-only).</p>
    <div class="cwrap">
      <div class="panel">
        <div class="find" id="sb_title">The shoulder leg — model-free, in-sample.</div>
        <div class="findsub">Running net units · taker fees paid</div>
        <div class="chartwrap"><div id="c_equity"></div></div>
        <div class="statrow">
          <div class="st"><div class="k">Net units</div><div class="v" id="sb_net" data-bind="BOOK_NET">—</div></div>
          <div class="st"><div class="k">Settled</div><div class="v" data-bind="SB_GRADED">—</div></div>
          <div class="st"><div class="k">Win rate</div><div class="v"><span data-bind="SB_WR">—</span>%</div></div>
          <div class="st"><div class="k">Awaiting</div><div class="v" data-bind="SB_AWAIT">—</div></div>
        </div>
      </div>
      <div class="panel">
        <div class="find">Each leg proves itself before real money.</div>
        <div class="findsub">A leg must clear a pre-registered forward target — settlements holding positive expectancy out-of-sample — before it can trade. In-sample profit doesn't count.</div>
        <table class="data" style="margin-top:2px">
          <tr><th>Leg</th><th class="num">Graded</th><th class="num">Edge / contract</th><th>Status</th></tr>
          <tr><td class="city">1 · sell shoulder</td><td class="num" data-bind="SB_GRADED">—</td><td class="num" id="sb_full_cell" data-bind="SB_FULL">—</td><td><span class="pill2 warn">paper</span></td></tr>
          <tr><td class="city">2 · buy favourite</td><td class="num" id="leg2n">—</td><td class="num" id="leg2edge">—</td><td><span class="pill2" id="leg2status">pending</span></td></tr>
          <tr><td class="city">1b · moderate [10–25¢]</td><td class="num" id="modn">—</td><td class="num" id="modedge">—</td><td><span class="pill2" id="modstatus">forward</span></td></tr>
        </table>
        <p class="cap" style="margin-top:14px"><b>Nothing is live.</b> A second signal — the model's most selective bucket — shows <b><span data-bind="E3_ROI">—</span> on <span data-bind="E3_N">—</span> bets</b>, still in-sample. No real orders until a gate passes.</p>
      </div>
    </div>
  </section>

  <!-- 04 ENTRIES -->
  <section>
    <div class="shd"><span class="n">04</span><h2>Recent settlements</h2><span class="r">last 60 · every graded market</span></div>
    <div class="cwrap" style="grid-template-columns:1.7fr 1fr">
      <div class="panel">
        <div class="tblscroll"><table class="data" id="t_recent"><thead><tr>
          <th>Date</th><th>City</th><th>Bin</th><th class="num">Mdl</th><th class="num">Mkt</th><th>Result</th><th>Closer</th>
        </tr></thead><tbody></tbody></table></div>
      </div>
      <div class="panel">
        <div class="find" style="font-size:13px" id="wonhd">Nearer the outcome, last 60</div>
        <div class="findsub">one vote per market · the metric that most flatters the model</div>
        <div class="wonbars" id="c_won"></div>
        <div class="cap" style="margin-top:14px"><span style="color:var(--model)">■</span> model nearer &nbsp; <span style="color:var(--market)">■</span> market nearer. This counts each market once, so a hair-closer call and a lucky call both score 1 — the model wins ~half these coin-flips yet still <b>loses on Brier and ROI</b>, because when it's wrong it's wrong big. Not an edge. Recent window has no Hong Kong (21-day truth lag).</div>
      </div>
    </div>
  </section>

  <!-- 05 HEALTH -->
  <section>
    <div class="shd"><span class="n">05</span><h2>System health</h2><span class="r">collection · data freshness</span></div>
    <div class="kpis" style="margin-top:0">
      <div class="kpi g"><div class="k">Collector</div><div class="v" id="collectorbig">—</div><div class="s">last snapshot · 2-hourly</div></div>
      <div class="kpi"><div class="k">Runs today</div><div class="v"><span data-bind="RUNS_TODAY">—</span><span style="font-size:12px;color:var(--faint)"> /12</span></div><div class="s">collection cycles</div></div>
      <div class="kpi"><div class="k">Data span</div><div class="v"><span data-bind="DATA_DAYS">—</span><span style="font-size:12px;color:var(--faint)"> d</span></div><div class="s" data-bind="SPAN">—</div></div>
      <div class="kpi"><div class="k">Data through</div><div class="v" style="font-size:14px" data-bind="UPDATED">—</div><div class="s">last commit</div></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="find" style="font-size:13px">Settlement-truth freshness</div>
      <div class="findsub">each market grades against the reading it settles on · Hong Kong lags ~3 weeks by design</div>
      <div class="cities" data-bind-html="CITIES_HTML"></div>
    </div>
  </section>

  <footer>
    <div><b>Truth</b> NWS CLI / IEM METAR / HKO · settlement-faithful grading · every prediction graded against the station reading it settles on, never the forecast grid</div>
    <div><b>Discipline</b> gate met, market still leads · nothing trades real money until its pre-registered forward gate passes · all trading shown is paper</div>
  </footer>
</div>

<script id="D0" type="application/json">%%PAYLOAD%%</script>
<script>
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";
  var D = {};
  var lastSeriesJSON = "";
  var lastGen = null, lastCollect = null, prevBind = null;
  var win = "all";       // verdict scoreboard window
  var roiWin = "all";    // ROI box window — independent of the verdict toggle
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
    var mag = Math.pow(10, Math.floor(Math.log10(v))), n = v / mag;
    var step = n <= 1 ? 1 : n <= 1.5 ? 1.5 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 3 ? 3 : n <= 5 ? 5 : 10;
    return step * mag;
  }
  // ---- line/area chart (tooltips + crosshair) ----
  function lineChart(id, opts) {
    var host = document.getElementById(id);
    if (!host) return;
    var W = Math.max(host.clientWidth || 640, 320), H = W < 520 ? 210 : (opts.h || 260), P = { l: 40, r: 52, t: 14, b: 26 };
    var series = opts.series.filter(function (s) { return s.points.some(function (p) { return p != null; }); });
    if (!series.length) { host.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; return; }
    var n = opts.xLabels.length;
    var vals = series.flatMap(function (s) { return s.points.filter(function (v) { return v != null; }); });
    var yMin = opts.yMin != null ? opts.yMin : Math.min(0, Math.min.apply(null, vals));
    var yMax = opts.yMax != null ? opts.yMax : niceTop(Math.max.apply(null, vals) * 1.08);
    if (yMax <= yMin) yMax = yMin + 1;
    var xd0 = 0, xd1 = 1;
    if (opts.xVals) { xd0 = opts.xDomain ? opts.xDomain[0] : Math.min.apply(null, opts.xVals); xd1 = opts.xDomain ? opts.xDomain[1] : Math.max.apply(null, opts.xVals); }
    var xOfV = function (v) { return P.l + (W - P.l - P.r) * (v - xd0) / ((xd1 - xd0) || 1); };
    var xOf = function (i) { if (opts.xVals) return xOfV(opts.xVals[i]); return P.l + (n <= 1 ? 0 : (W - P.l - P.r) * i / (n - 1)); };
    var yOf = function (v) { return P.t + (H - P.t - P.b) * (1 - (v - yMin) / (yMax - yMin)); };
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    var ticks = 4;
    for (var g = 0; g <= ticks; g++) {
      var yv = yMin + (yMax - yMin) * g / ticks, yy = yOf(yv);
      svg.appendChild(el("line", { class: "gridline", x1: P.l, y1: yy, x2: W - P.r, y2: yy }));
      svg.appendChild(el("text", { class: "axis", x: P.l - 8, y: yy + 3.5, "text-anchor": "end" }, opts.yFmt ? opts.yFmt(yv) : yv.toFixed(2)));
    }
    if (yMin < 0) svg.appendChild(el("line", { x1: P.l, y1: yOf(0), x2: W - P.r, y2: yOf(0), stroke: css("--axis"), "stroke-width": 1 }));
    var every = Math.ceil(n / 6);
    for (var i = 0; i < n; i++) if ((i % every === 0 && xOf(n - 1) - xOf(i) > 52) || i === n - 1)
      svg.appendChild(el("text", { class: "axis", x: xOf(i), y: H - 8, "text-anchor": i === n - 1 ? "end" : "middle" }, opts.xLabels[i]));
    var endLabels = [];
    series.forEach(function (s) {
      var pts = s.points.map(function (v, i) { return v == null ? null : [xOf(i), yOf(v)]; }).filter(Boolean);
      if (opts.area && s.fill) {
        var dd = "M" + pts.map(function (p) { return p[0] + "," + p[1]; }).join(" L ");
        dd += " L " + pts[pts.length - 1][0] + "," + yOf(Math.max(yMin, 0)) + " L " + pts[0][0] + "," + yOf(Math.max(yMin, 0)) + " Z";
        svg.appendChild(el("path", { d: dd, fill: s.color, "fill-opacity": .1, stroke: "none" }));
      }
      svg.appendChild(el("polyline", { points: pts.map(function (p) { return p[0] + "," + p[1]; }).join(" "), fill: "none", stroke: s.color, "stroke-width": s.thick || 2.2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "none" }));
      var last = pts[pts.length - 1];
      svg.appendChild(el("circle", { cx: last[0], cy: last[1], r: 3, fill: s.color }));
      var lbl = el("text", { class: "serieslabel", x: last[0] + 7, y: last[1] + 3.5, fill: s.color }, opts.endFmt ? opts.endFmt(s) : "");
      svg.appendChild(lbl); endLabels.push({ el: lbl, y: last[1] });
    });
    endLabels.sort(function (a, b) { return a.y - b.y; });
    for (var li = 1; li < endLabels.length; li++) if (endLabels[li].y - endLabels[li - 1].y < 13) { endLabels[li].y = endLabels[li - 1].y + 13; endLabels[li].el.setAttribute("y", endLabels[li].y + 3.5); }
    host.appendChild(svg);
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    var cross = el("line", { class: "gridline", y1: P.t, y2: H - P.b, "stroke-dasharray": "3 3", opacity: 0 }); svg.appendChild(cross);
    var focus = series.map(function (s) { var c = el("circle", { r: 4, fill: s.color, stroke: css("--surf"), "stroke-width": 1.5, opacity: 0 }); svg.appendChild(c); return c; });
    svg.appendChild(el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }));
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect(), xv = (ev.clientX - r.left) / r.width * W;
      var i = Math.round((xv - P.l) / ((W - P.l - P.r) / (n - 1))); i = Math.max(0, Math.min(n - 1, i));
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
      var hr = host.getBoundingClientRect(), px = xOf(i) / W * hr.width;
      tip.style.left = Math.min(Math.max(px - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = "6px";
    });
    svg.addEventListener("mouseleave", function () { tip.style.opacity = 0; cross.setAttribute("opacity", 0); focus.forEach(function (c) { c.setAttribute("opacity", 0); }); });
  }
  // ---- dumbbell (per-city market/ens/model) ----
  function dumbbell(id, rows) {
    var host = document.getElementById(id);
    if (!host || !rows.length) { if (host) host.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">No data yet.</div>'; return; }
    var W = Math.max(host.clientWidth || 640, 320), rh = 30, P = { l: 78, r: 46, t: 8, b: 22 }, H = P.t + P.b + rows.length * rh;
    var all = rows.flatMap(function (r) { return [r.market, r.model, r.ens].filter(function (v) { return v != null; }); });
    var lo = Math.min.apply(null, all) * 0.9, hi = Math.max.apply(null, all) * 1.05;
    var xOf = function (v) { return P.l + (W - P.l - P.r) * (v - lo) / ((hi - lo) || 1); };
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    [lo + (hi - lo) * .25, lo + (hi - lo) * .5, lo + (hi - lo) * .75].forEach(function (gv) {
      svg.appendChild(el("line", { class: "gridline", x1: xOf(gv), y1: P.t, x2: xOf(gv), y2: H - P.b }));
      svg.appendChild(el("text", { class: "axis", x: xOf(gv), y: H - 8, "text-anchor": "middle" }, gv.toFixed(2)));
    });
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    rows.forEach(function (r, i) {
      var y = P.t + rh * i + rh / 2;
      svg.appendChild(el("text", { class: "axis", x: 0, y: y + 3.5, fill: css("--ink2") }, r.city));
      var xs = [r.market, r.ens, r.model].filter(function (v) { return v != null; }).map(xOf);
      svg.appendChild(el("line", { x1: Math.min.apply(null, xs), y1: y, x2: Math.max.apply(null, xs), y2: y, stroke: css("--axis"), "stroke-width": 1.5 }));
      [["market", r.market, css("--market")], ["ens", r.ens, css("--ens")], ["model", r.model, css("--model")]].forEach(function (m) {
        if (m[1] == null) return;
        var c = el("circle", { cx: xOf(m[1]), cy: y, r: 4, fill: m[2] });
        c.addEventListener("mousemove", function () {
          tip.innerHTML = '<div class="tt">' + esc(r.city) + ' · n=' + r.n + '</div>'
            + '<div class="tr"><span><i style="background:' + css("--market") + '"></i>Market</span><b>' + r.market.toFixed(3) + '</b></div>'
            + (r.ens != null ? '<div class="tr"><span><i style="background:' + css("--ens") + '"></i>Ensemble</span><b>' + r.ens.toFixed(3) + '</b></div>' : '')
            + '<div class="tr"><span><i style="background:' + css("--model") + '"></i>Model</span><b>' + r.model.toFixed(3) + '</b></div>';
          tip.style.opacity = 1; var hr = host.getBoundingClientRect();
          tip.style.left = Math.min(Math.max(xOf(m[1]) / W * hr.width - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = (y / H * hr.height + 8) + "px";
        });
        c.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
        svg.appendChild(c);
      });
    });
    host.appendChild(svg);
  }
  // ---- diverging bars (calibration gap) ----
  function divergingBars(id, rows) {
    var host = document.getElementById(id);
    if (!host || !rows.length) { if (host) host.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">Not enough graded data yet.</div>'; return; }
    var W = Math.max(host.clientWidth || 640, 320), H = 210, P = { t: 22, b: 34 }, mid = P.t + (H - P.t - P.b) / 2;
    var maxG = Math.max.apply(null, rows.map(function (r) { return Math.abs(r.f - r.p); })) || 0.1;
    var half = (H - P.t - P.b) / 2;
    var gw = W / rows.length, bw = Math.min(38, gw * 0.5);
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    svg.appendChild(el("text", { class: "axis", x: W, y: P.t + 4, "text-anchor": "end" }, "happened more"));
    svg.appendChild(el("text", { class: "axis", x: W, y: H - P.b + 12, "text-anchor": "end" }, "happened less"));
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    rows.forEach(function (r, i) {
      var gap = r.f - r.p, h = Math.abs(gap) / maxG * half, x = gw * i + (gw - bw) / 2;
      var y = gap >= 0 ? mid - h : mid, col = gap >= 0 ? css("--market") : css("--model");
      var bar = el("rect", { x: x, y: y, width: bw, height: Math.max(2, h), rx: 4, fill: col });
      bar.addEventListener("mousemove", function () {
        tip.innerHTML = '<div class="tt">model said ' + (r.p * 100).toFixed(0) + '%</div><div class="tr"><span>actually happened</span><b>' + (r.f * 100).toFixed(0) + '%</b></div><div class="tr"><span>markets</span><b>' + r.n + '</b></div>';
        tip.style.opacity = 1; var hr = host.getBoundingClientRect();
        tip.style.left = Math.min(Math.max((gw * i + gw / 2) / W * hr.width - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = "6px";
      });
      bar.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
      svg.appendChild(bar);
      svg.appendChild(el("text", { class: "axis", x: gw * i + gw / 2, y: H - 8, "text-anchor": "middle" }, (r.p * 100).toFixed(0) + "%"));
    });
    svg.appendChild(el("line", { x1: 0, y1: mid, x2: W, y2: mid, stroke: css("--axis"), "stroke-width": 1 }));
    host.appendChild(svg);
  }
  // ---- dispersion bars (std z by month, ref 1.0) ----
  function dispChart(id, rows) {
    var host = document.getElementById(id);
    if (!host || !rows.length) { if (host) host.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">No data yet.</div>'; return; }
    var W = Math.max(host.clientWidth || 640, 320), H = 210, P = { l: 30, r: 12, t: 16, b: 26 };
    var yMax = Math.max(2, niceTop(Math.max.apply(null, rows.map(function (r) { return r.z; }))));
    var yOf = function (v) { return P.t + (H - P.t - P.b) * (1 - v / yMax); };
    var gw = (W - P.l - P.r) / rows.length, bw = Math.min(48, gw * 0.62);
    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    [0, 1, 2].forEach(function (yv) { if (yv > yMax) return; svg.appendChild(el("line", { class: "gridline", x1: P.l, y1: yOf(yv), x2: W - P.r, y2: yOf(yv) })); svg.appendChild(el("text", { class: "axis", x: P.l - 6, y: yOf(yv) + 3.5, "text-anchor": "end" }, yv.toFixed(1))); });
    var monN = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var tip = document.createElement("div"); tip.className = "tip"; host.parentNode.appendChild(tip);
    rows.forEach(function (r, i) {
      var x = P.l + gw * i + (gw - bw) / 2, y = yOf(r.z), col = r.z > 1.15 ? css("--warn") : css("--good");
      var bar = el("rect", { x: x, y: y, width: bw, height: Math.max(2, yOf(0) - y), rx: 4, fill: col });
      bar.addEventListener("mousemove", function () { tip.innerHTML = '<div class="tt">' + r.m + '</div><div class="tr"><span>std(z)</span><b>' + r.z.toFixed(2) + '</b></div>'; tip.style.opacity = 1; var hr = host.getBoundingClientRect(); tip.style.left = Math.min(Math.max((P.l + gw * i + gw / 2) / W * hr.width - tip.offsetWidth / 2, 0), hr.width - tip.offsetWidth) + "px"; tip.style.top = "6px"; });
      bar.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
      svg.appendChild(bar);
      svg.appendChild(el("text", { class: "axis", x: x + bw / 2, y: y - 5, "text-anchor": "middle", fill: css("--muted") }, r.z.toFixed(2)));
      var mn = monN[parseInt(r.m.slice(5), 10)] || r.m;
      svg.appendChild(el("text", { class: "axis", x: P.l + gw * i + gw / 2, y: H - 8, "text-anchor": "middle" }, mn));
    });
    var ry = yOf(1.0);
    svg.appendChild(el("line", { x1: P.l, y1: ry, x2: W - P.r, y2: ry, stroke: css("--good"), "stroke-width": 1, "stroke-dasharray": "4 3", opacity: .75 }));
    svg.appendChild(el("text", { class: "axis", x: W - P.r, y: ry - 4, "text-anchor": "end", fill: css("--good") }, "1.0"));
    host.appendChild(svg);
  }

  // ---- verdict scoreboard (reads D.score, driven by the window toggle) ----
  function renderScore() {
    var sc = D.score;
    var host = document.getElementById("rank"), gapEl = document.getElementById("k_gap"), vl = document.getElementById("verdictLine"), wl = document.getElementById("wLabel");
    if (!host || !sc) return;
    var s = sc[win] || sc.all;
    var rows = [["market", "Market price", s.market], ["ens", "Raw ensemble", s.ens], ["model", "Calibrated model", s.model]];
    var lead = Math.min(s.market, s.ens, s.model);
    var ordered = rows.slice().sort(function (a, b) { return a[2] - b[2]; });
    var rankOf = {}; ordered.forEach(function (r, i) { rankOf[r[0]] = i + 1; });
    host.innerHTML = rows.map(function (r) {
      var isLead = r[2] === lead, gap = r[2] - lead;
      return '<div class="rrow' + (isLead ? ' lead' : '') + '"><span class="i">0' + rankOf[r[0]] + '</span>'
        + '<span class="who"><span class="sw ' + r[0] + '"></span>' + r[1] + (isLead ? '<span class="tagl">Leads</span>' : '') + '</span>'
        + '<span class="val">' + r[2].toFixed(3) + '</span>'
        + '<span class="d' + (isLead ? ' z' : '') + '">' + (isLead ? '—' : '+' + gap.toFixed(3)) + '</span></div>';
    }).join("");
    var mg = s.model - s.market;
    if (gapEl) gapEl.textContent = (mg >= 0 ? "+" : "−") + Math.abs(mg).toFixed(3);
    var leadName = ordered[0][0];
    if (vl) vl.innerHTML = leadName === "model"
      ? 'Our model now forecasts the weather markets <em style="color:var(--good)">best</em>.'
      : 'The <em style="color:var(--market)">market</em> forecasts the weather better than our model.';
    if (wl) wl.textContent = win === "recent" ? "the last " + s.n + " settled markets" : "all " + s.n + " common markets";
    document.getElementById("cav").classList.toggle("show", win === "recent");
    document.querySelectorAll("#winseg button").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-win") === win); });
  }
  function renderRoi() {
    var tb = document.querySelector("#t_roi tbody"), warn = document.getElementById("roiwarn"), sub = document.getElementById("roisub"), seg = document.getElementById("roiseg");
    if (!tb) return;
    var sc = D.score || {}, s = sc[roiWin] || sc.all || {};
    var rm = (D.roi && D.roi.model) ? (D.roi.model[roiWin] || D.roi.model.all) : null;
    var re = (D.roi && D.roi.ens) ? (D.roi.ens[roiWin] || D.roi.ens.all) : null;
    function b(v) { return v == null ? "—" : v.toFixed(3); }
    function pct(o) { var x = o.roi; return (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + "%"; }
    function roiCell(o) {
      if (!o || o.bets == null || o.bets === 0) return '<td class="num" style="color:var(--faint)">—</td>';
      // positive ROI is muted (never green — a worse forecaster's profit is not "good"); negative is red
      var cls = o.roi < 0 ? ' neg' : '', col = o.roi < 0 ? '' : ' style="color:var(--muted)"';
      return '<td class="num' + cls + '"' + col + '>' + pct(o) + ' <small style="color:var(--faint)">· ' + o.bets + ' bets</small></td>';
    }
    var recent = roiWin === "recent";
    var brierLbl = 'Brier · accuracy <small>(' + (recent ? 'last 60 markets' : 'all markets') + ')</small>';
    var roiLbl = 'ROI · in-sample <small>(' + (recent ? 'last 60 bets' : 'all bets placed') + ')</small>';
    tb.innerHTML =
      '<tr><td class="rowlbl">' + brierLbl + '</td>'
      + '<td class="num pos">' + b(s.market) + '</td><td class="num">' + b(s.ens) + '</td><td class="num">' + b(s.model) + '</td></tr>'
      + '<tr><td class="rowlbl">' + roiLbl + '</td>'
      + '<td class="num" style="color:var(--faint)">—</td>' + roiCell(re) + roiCell(rm) + '</tr>';
    if (sub) sub.textContent = "the same forecasters, scored two ways · " + (recent ? "last 60" : "all-time");
    if (seg) seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x.getAttribute("data-rwin") === roiWin); });
    if (warn) warn.innerHTML = '<b>Why ROI is not the scoreboard.</b> The ensemble is <b>less accurate than the market</b> (worse Brier) yet can show a positive ROI — bet-selection and sizing luck on a small in-sample set, not a real edge. It <b>swings wildly between windows and each data refresh</b> (all-time vs recent, and run to run), the tell-tale of noise rather than skill. Accuracy (Brier) is the honest verdict; the market still wins it.';
  }
  function renderWon() {
    var host = document.getElementById("c_won"), hd = document.getElementById("wonhd"), w = D.woncity;
    if (!host || !w) return;
    var pctAll = w.n ? Math.round(w.mwin / w.n * 100) : 0;
    hd.innerHTML = 'Nearer the outcome: model ' + w.mwin + ' / ' + w.n + ' <span style="color:var(--faint);font-weight:400">(' + pctAll + '% — a coin-flip)</span>';
    host.innerHTML = (w.rows || []).map(function (r) {
      var pct = r.n ? Math.round(r.mwin / r.n * 100) : 0;
      return '<div class="wb"><span class="nm">' + esc(r.city) + '</span><span class="bar"><span class="mk" style="width:' + pct + '%"></span><span class="mkt" style="width:' + (100 - pct) + '%"></span></span><span class="c">' + r.mwin + '/' + r.n + '</span></div>';
    }).join("");
  }
  function renderBuckets() {
    var tb = document.querySelector("#t_buckets tbody");
    if (!tb) return;
    var rows = D.buckets || [], gate = D.gate_fwd_n || 40;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="6" style="color:var(--faint)">No bucket data yet.</td></tr>'; return; }
    tb.innerHTML = rows.map(function (r) {
      var d = r.market - r.model;
      var dCell = d > 0.0005 ? '<span class="pos">−' + d.toFixed(3) + '</span>' : d < -0.0005 ? '<span class="neg">+' + (-d).toFixed(3) + '</span>' : '0.000';
      var gateCell;
      if (r.nom) {
        var pct = Math.min(100, r.fwd_n / gate * 100);
        gateCell = '<span class="gatebar"><i style="width:' + pct.toFixed(0) + '%"></i></span> <span class="mono" style="font-size:10.5px;color:var(--good)">nominated · ' + r.fwd_n + '/' + gate + '</span>';
      } else gateCell = '<span class="pill2">loses — off</span>';
      return '<tr><td class="city">' + esc(r.b) + '</td><td class="num">' + r.n + '</td><td class="num ' + (d > 0.0005 ? 'pos' : '') + '">' + r.model.toFixed(3) + '</td><td class="num">' + r.market.toFixed(3) + '</td><td class="num">' + dCell + '</td><td>' + gateCell + '</td></tr>';
    }).join("");
  }
  function renderRecent() {
    var tb = document.querySelector("#t_recent tbody");
    if (!tb) return;
    var rows = D.recent || [];
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="7" style="color:var(--faint)">No settlements yet.</td></tr>'; return; }
    tb.innerHTML = rows.map(function (r) {
      var mErr = Math.abs(r.model - r.out), kErr = Math.abs(r.market - r.out);
      var closer = mErr < kErr ? '<span class="pos">Model ✓</span>' : kErr < mErr ? '<span class="neg">Market ✓</span>' : 'tie';
      return '<tr><td class="mono">' + esc(r.d) + '</td><td class="city">' + esc(r.city) + '</td><td>' + esc(r.bin) + '</td>'
        + '<td class="num">' + (r.model * 100).toFixed(0) + '%</td><td class="num">' + (r.market * 100).toFixed(0) + '%</td>'
        + '<td>' + (r.out ? 'Yes' : 'No') + '</td><td>' + closer + '</td></tr>';
    }).join("");
  }
  function renderLegs() {
    var favN = (D.sb_fav_graded != null) ? D.sb_fav_graded : (prevBind && prevBind.SB_FAV_GRADED) || "0";
  }

  function draw() {
    ["c_acc", "c_city", "c_calib", "c_disp", "c_equity"].forEach(function (id) { var h = document.getElementById(id); if (h) h.innerHTML = ""; });
    document.querySelectorAll(".chartwrap .tip").forEach(function (t) { t.remove(); });

    var acc = D.acc || [];
    if (acc.length) lineChart("c_acc", {
      h: 240, xLabels: acc.map(function (r) { return r.t; }),
      yFmt: function (v) { return v.toFixed(2); }, tipFmt: function (v) { return v.toFixed(4); },
      tipSuffix: function (i) { return "  ·  n=" + acc[i].n; },
      endFmt: function (s) { var last = s.points.filter(function (v) { return v != null; }).slice(-1)[0]; return last != null ? last.toFixed(3) : ""; },
      series: [
        { name: "Market", color: css("--market"), points: acc.map(function (r) { return r.market; }) },
        { name: "Model", color: css("--model"), points: acc.map(function (r) { return r.model; }) },
        { name: "Ensemble", color: css("--ens"), dash: "4 3", thick: 1.6, points: acc.map(function (r) { return r.ens; }) }
      ]
    });

    var city = (D.city || []).slice().sort(function (a, b) { return a.market - b.market; });
    dumbbell("c_city", city.map(function (r) { return { city: (r.city === "HongKong" ? "Hong Kong" : r.city === "NYC" ? "New York" : r.city), market: r.market, model: r.model, ens: r.ens, n: r.n }; }));

    divergingBars("c_calib", D.calib || []);
    dispChart("c_disp", D.disp || []);

    var eq = D.equity || [];
    var eqCol = (eq.length && eq[eq.length - 1].v < 0) ? css("--model") : css("--good");
    if (eq.length) lineChart("c_equity", {
      h: 210, area: true, xLabels: eq.map(function (r) { return r.t; }),
      yFmt: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "u"; }, tipFmt: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "u"; },
      endFmt: function (s) { var last = s.points.slice(-1)[0]; return last != null ? (last >= 0 ? "+" : "") + last.toFixed(2) + "u" : ""; },
      series: [{ name: "Net units", color: eqCol, points: eq.map(function (r) { return r.v; }), fill: true, thick: 2.2 }]
    });
    else { var he = document.getElementById("c_equity"); if (he) he.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">No settled paper entries yet.</div>'; }

    renderScore();
    renderRoi();
    renderWon();
    renderBuckets();
    renderRecent();
  }

  // ===== live layer =====
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
    // Sign-aware colours + data-driven titles — a "finding" must never contradict the number.
    // BOOK_NET / SB_FULL strings use "−" (U+2212) for negatives (Python formats +/− then swaps).
    function _neg(v) { return v != null && (String(v).charAt(0) === "−" || String(v).charAt(0) === "-"); }
    if (b.BOOK_NET != null) {
      var neg = _neg(b.BOOK_NET);
      var netEl = document.getElementById("sb_net"); if (netEl) netEl.style.color = neg ? "var(--model)" : "var(--good)";
      var tEl = document.getElementById("sb_title");
      if (tEl) tEl.textContent = neg ? "The shoulder leg is underwater — and in-sample." : "The shoulder leg is up — but in-sample.";
    }
    if (b.SB_FULL != null) { var fc = document.getElementById("sb_full_cell"); if (fc) fc.style.color = _neg(b.SB_FULL) ? "var(--model)" : "var(--good)"; }
    // Leg 2 favourites — honest "pending" until it has graded entries
    var favN = b.SB_FAV_GRADED;
    if (favN != null) {
      var n2 = document.getElementById("leg2n"), e2 = document.getElementById("leg2edge"), s2 = document.getElementById("leg2status");
      if (n2) n2.textContent = favN;
      if (parseInt(favN, 10) > 0) { if (s2) { s2.textContent = "paper"; s2.className = "pill2 warn"; } }
      else { if (e2) e2.textContent = "—"; if (s2) { s2.textContent = "pending"; s2.className = "pill2"; } }
    }
    // Leg 1b moderate-shoulder forward gate — progress n/need, edge only once it has entries.
    if (b.SB_MOD_FWD_N != null) {
      var mn = document.getElementById("modn"), me = document.getElementById("modedge"), ms = document.getElementById("modstatus");
      var fn = parseInt(b.SB_MOD_FWD_N, 10) || 0, pass = b.SB_MOD_PASS === "1";
      if (mn) mn.textContent = b.SB_MOD_FWD_N + "/" + (b.SB_MOD_NEED || "80");
      if (me) me.textContent = fn > 0 ? b.SB_MOD_FWD : "—";
      if (ms) { ms.textContent = pass ? "gate ✓" : "forward"; ms.className = pass ? "pill2 on" : "pill2"; }
    }
    prevBind = b;
    var h = html || {};
    Object.keys(h).forEach(function (k) { document.querySelectorAll('[data-bind-html="' + k + '"]').forEach(function (e) { if (e.innerHTML !== h[k]) e.innerHTML = h[k]; }); });
  }
  function fmtAgo(iso) {
    var t = Date.parse(iso); if (isNaN(t)) return null;
    var s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 45) return "just now"; if (s < 5400) return Math.max(1, Math.round(s / 60)) + "m ago"; if (s < 172800) return Math.round(s / 3600) + "h ago"; return Math.round(s / 86400) + "d ago";
  }
  function tickClock() {
    var elm = document.querySelector("[data-ago]"); if (elm && lastGen) elm.textContent = "updated " + (fmtAgo(lastGen) || "—");
    var pill = document.getElementById("statuspill"), st = document.getElementById("statustext");
    if (pill && st) { if (lastCollect) { var hrs = (Date.now() - Date.parse(lastCollect)) / 36e5, ok = hrs < 5; pill.className = "pill " + (ok ? "" : "warn"); st.textContent = ok ? "live" : "collector stale"; } }
    var cb = document.getElementById("collectorbig"); if (cb && lastCollect) cb.textContent = fmtAgo(lastCollect) || "—";
  }
  function apply(P) {
    if (!P || !P.bind) return false;
    applyBind(P.bind, P.html);
    lastGen = P.generated_at || null;
    lastCollect = P.last_collect_iso || (P.series && P.series.last_collect_iso) || null;
    var sj = JSON.stringify(P.series || {});
    if (sj !== lastSeriesJSON) { D = P.series || {}; lastSeriesJSON = sj; draw(); }
    document.body.classList.add("ready"); tickClock(); return true;
  }
  function applyInline() { try { return apply(JSON.parse(document.getElementById("D0").textContent || "{}")); } catch (e) { document.body.classList.add("ready"); return false; } }
  function load() {
    fetch("data.json?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("http " + r.status); return r.json(); })
      .then(function (P) { apply(P); document.getElementById("errbar").style.display = "none"; })
      .catch(function () { if (!document.body.classList.contains("ready")) document.getElementById("errbar").style.display = "block"; });
  }
  document.getElementById("winseg").addEventListener("click", function (ev) {
    var b = ev.target.closest("button"); if (!b || b.getAttribute("data-win") === win) return;
    win = b.getAttribute("data-win"); renderScore();
  });
  document.getElementById("roiseg").addEventListener("click", function (ev) {
    var b = ev.target.closest("button"); if (!b || b.getAttribute("data-rwin") === roiWin) return;
    roiWin = b.getAttribute("data-rwin"); renderRoi();
  });
  applyInline(); load();
  setInterval(load, 120000); setInterval(tickClock, 1000);
  var _rt; window.addEventListener("resize", function () { clearTimeout(_rt); _rt = setTimeout(draw, 180); });
})();
</script>"""


if __name__ == "__main__":
    main()
