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
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
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

# ── CAPTURE-TIER cities (added 2026-08-04) ────────────────────────────────────────────────
# The seven US cities where Polymarket and Kalshi resolve on the SAME station and differ only
# in the RULER (Wunderground hourly-METAR max vs NWS CLI 1-minute max). We capture prices and
# truth for them; we do NOT forecast or model them, so they have no model-vs-market Brier and
# CANNOT go in CITY_ORDER — `_missing_cities` would refuse to publish while they have no
# gradable markets, freezing the live dashboard. This panel reports COLLECTION HEALTH only.
CAPTURE_META = {
    "Los Angeles":   ("KLAX", "KXHIGHLAX"),
    "Austin":        ("KAUS", "KXHIGHAUS"),
    "Atlanta":       ("KATL", "KXHIGHTATL"),
    "Houston":       ("KHOU", "KXHIGHTHOU"),
    "Miami":         ("KMIA", "KXHIGHMIA"),
    "Seattle":       ("KSEA", "KXHIGHTSEA"),
    "San Francisco": ("KSFO", "KXHIGHTSFO"),
}


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
    # NOTE: the structure book is NOT run as a subprocess and scraped any more — `_book_binds`
    # computes those scalars from the graded frame. See its docstring for the two silent
    # blank-outs that coupling caused.
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


def config_forward_min() -> int:
    """The pre-registered forward-bet floor (non-binding since 2026-07-28 — the
    clustered interval is what actually gates; see evaluate_oos._mde)."""
    try:
        import config as cfg
        return int(getattr(cfg, "E3_FORWARD_MIN_BETS", 40))
    except Exception:
        return 40


def _series_error(payload) -> "str | None":
    """The exception compute_series swallowed, if any.

    compute_series wraps its whole body in one try/except and records the failure in
    series["error"]. Nothing checked it, so on 2026-07-28 a NameError silently dropped
    every panel computed after the failure point — "Recent settlements" published as
    "No settlements yet" while the workflow stayed green. Same shape as the truth
    outage that _missing_cities was written for: a partial page must fail loudly."""
    try:
        err = (payload or {}).get("series", {}).get("error")
    except AttributeError:
        return None
    return str(err) if err else None


def _pooled_gap(w, a: str = "b_model", b: str = "b_mkt") -> dict:
    """Paired `a`-minus-`b` Brier gap + clustered 95% interval. THE powered test.

    Defaults to model-minus-market. Also used for model-minus-ENSEMBLE, which the guide calls
    "the number that matters most in the entire report" (a calibrator that cannot beat the physics
    it calibrates is doing damage) and which had only ever been published as two separate means —
    the unpaired form this function exists to replace. Model-vs-ensemble needs the common set,
    since the ensemble structurally cannot price Tmin bins.

    Takes whatever frame it is given and scores every row, so hand it ALL gradable markets —
    this question needs the model probability, the traded price and the outcome, and nothing
    else. Restricting to the ensemble-paired common set (what the scoreboard needs) threw away
    140 of 401 markets, 68 of them Tmin bins the ensemble structurally cannot price, and 18.7%
    of the precision — for a comparison the ensemble takes no part in. It also disagreed with
    evaluate_oos, the arbiter, which has always used the full set: +0.026 here vs +0.0211 there.

    Paired per market so the market's own difficulty cancels; clustered by city-day because
    every bin for a city on a date settles on ONE weather outcome.
    """
    import stats_util

    iv = stats_util.interval(
        w[a] - w[b],
        stats_util.cluster_key(w.assign(target_date=w["td"].dt.strftime("%Y-%m-%d"))))
    return {"gap": round(float(iv["mean"]), 4),
            "lo": round(float(iv["mean"] - 1.96 * iv["se"]), 4),
            "hi": round(float(iv["mean"] + 1.96 * iv["se"]), 4),
            "n": int(iv["n"]), "clusters": int(iv["n_clusters"])}


def _city_rows(c) -> list[dict]:
    """Per-city Brier for market / ensemble / model, all on the SAME markets.

    `c` is the common-set frame (markets every forecaster scored) carrying b_mkt/b_ens/b_model.
    Scoring the three from their own trackers is what published London as an ensemble win on
    2026-07-28: the ensemble drew an easier 94-market subset while the row read n=115, and the
    ordering reversed (0.1197 vs 0.1290 unpaired; 0.1259 vs 0.1228 paired). One row, one market
    set, one n — the chart draws these three on a single connected line, so they have to be
    comparable or the line is a lie.

    `degenerate` marks a city whose graded markets ALL settled the same way. Brier there is
    mean(p^2) with no discrimination in it — lowest-numbers-wins, and predicting 0.0 everywhere
    scores a perfect 0 — so the row is not an accuracy comparison and must not render as one.
    That is the Hong Kong case (11 markets, 0 YES, ensemble 'beat' the market by stating lower
    numbers).
    """
    rows = []
    for cty, g in c.groupby("city"):
        y = g["outcome"].astype(float)
        rows.append({
            "city": str(cty), "n": int(len(g)),
            "market": round(float(g["b_mkt"].mean()), 4),
            "ens": round(float(g["b_ens"].mean()), 4),
            "model": round(float(g["b_model"].mean()), 4),
            "degenerate": bool(y.nunique() < 2),
            "yes": int(y.sum()),
        })
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows


def compute_series() -> dict:
    out: dict = {"acc": [], "roll": [], "city": [], "calib": [], "growth": [],
                 "heartbeat": [], "buckets": [], "recent": [], "equity": [], "capture": [],
                 "bk_equity": [], "bk_cities": [], "bk_cs_date": ""}
    # Capture-tier coverage FIRST, and outside the eval machinery: it reads only collected files,
    # so it must still render if anything downstream fails. capture_coverage never raises.
    out["capture"] = capture_coverage()
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

        # POOLED paired gap — the powered test. Per-market model-minus-market difference (the
        # market's own difficulty cancels), clustered by city-day because every bin for a city on
        # a date settles on ONE weather outcome. Splitting into 15 buckets costs ~15x this sample,
        # which is why the per-bucket rows cannot conclude anything for years and this can.
        try:
            # On ALL gradable markets (`cal`), NOT the ensemble-paired common set the scoreboard
            # needs — see _pooled_gap. The tile and the 3-way scoreboard above it therefore run on
            # different samples ON PURPOSE, and both label their own n; this is the one comparison
            # that gains nothing from the ensemble's presence and loses 35% of the data to it.
            # Matches evaluate_oos, the arbiter.
            allm = cal.sort_values("td")
            out["pooled"] = {"all": _pooled_gap(allm), "recent": _pooled_gap(allm.tail(60))}
            # Model vs the ENSEMBLE it calibrates, paired and clustered. On the common set: the
            # ensemble prices no Tmin bins, so `cal` would compare it on markets it never scored.
            out["pooled"]["vs_ens"] = _pooled_gap(cs, "b_model", "b_ens")
        except Exception:
            pass

        # ROI at production params (IN-SAMPLE, noisy) — model AND ensemble, for the honest
        # accuracy-vs-profit panel. The ensemble's positive ROI while it LOSES on Brier is the
        # exact "127% ROI" mirage this project warns about; surfaced only with that caveat,
        # never as an edge claim. Same betting policy for both (MIN_EDGE + positive Kelly).
        try:
            import config as _cfg

            def _roi(df):
                b = df[(df["abs_edge"].astype(float) >= _cfg.MIN_EDGE) & (df["kelly"].astype(float) > 0)].copy()
                r = ev._roi_at_production(b)
                return {"roi": round(float(r["roi"]), 4), "bets": int(r["bets"]), "wins": int(r["wins"])}
            # "Recent" ROI is scoped to the SAME last-60 common markets the Brier toggle uses, so
            # both "Last 60" views mean one thing. Bets are those placed WITHIN that market window
            # (so recent bet counts are < 60 — the last 60 markets, of which N were bet).
            recent_cids = set(cs.tail(60)["condition_id"])
            out["roi"] = {
                "model": {"all": _roi(cal), "recent": _roi(cal[cal["condition_id"].isin(recent_cids)])},
                "ens":   {"all": _roi(ens), "recent": _roi(ens[ens["condition_id"].isin(recent_cids)])},
            }
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

        # accuracy by city — on the COMMON set `c`, exactly like the scoreboard above. Reading
        # model/market off `cal` and the ensemble off `ens` compared different market sets in one
        # row; see _city_rows.
        out["city"] = _city_rows(c)

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
            # 2026-07-28: every bucket is nominated, each with its OWN forward clock, so the
            # forward slice must be per-bucket rather than one global date. LIVE_BUCKETS now
            # means gate-PASSED (empty), which is not the same thing as "under test".
            noms = dict(getattr(cfg, "E3_NOMINATIONS", {}))
            nominated = set(noms) or set(getattr(cfg, "LIVE_BUCKETS", set()))
            default_nom = str(getattr(cfg, "E3_NOMINATION_DATE", "2026-07-12"))
        except Exception:
            noms, nominated, default_nom = {}, set(), "2026-07-12"
        buckets = []
        if "bucket" in cal.columns:
            for b, g in cal.groupby("bucket"):
                if len(g) < 3:
                    continue
                fwd = g[g["td"] > pd.Timestamp(noms.get(str(b), default_nom))]
                row = {"b": str(b), "n": int(len(g)),
                       "model": round(float(g["b_model"].mean()), 4),
                       "market": round(float(g["b_mkt"].mean()), 4),
                       "nom": str(b) in nominated,
                       "since": noms.get(str(b), default_nom),
                       "fwd_n": int(len(fwd))}
                if len(fwd) >= 3:
                    row["fwd_model"] = round(float(fwd["b_model"].mean()), 4)
                    row["fwd_market"] = round(float(fwd["b_mkt"].mean()), 4)
                buckets.append(row)
            buckets.sort(key=lambda r: (not r["nom"], -r["n"]))
        out["buckets"] = buckets
        out["gate_fwd_n"] = config_forward_min()

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

    # paper-book equity curve — settled by shoulder_book's OWN grading + fee math (taker-
    # conservative: half-spread crossed on entry, 0.05·p·(1−p) fee), cumulative by target date.
    # Reads the SHARED graded frame (`_book_graded`); it used to re-derive grading inline, a
    # second copy that could drift from the report it sits beside.
    try:
        # Same function as the breadth panel, so the two charts carry identical semantics: Leg 1
        # (all shoulder) and Leg 1b (moderate band), ROI on deployed capital, Leg 2 excluded.
        _, graded = _book_graded()
        out["equity"] = roi_series(graded)
    except Exception as e:
        out["eq_error"] = f"{type(e).__name__}: {e}"

    # Breadth equity curve + per-city breakdown. The breadth book carries ~10× the inventory of
    # the 5-city book and had no chart at all — only two summary rows — so a reader could not see
    # whether its edge accrued steadily or came from a handful of days.
    try:
        out["bk_equity"] = _breadth_equity()
        out["bk_cities"] = _breadth_cities()
        import shoulder_book_breadth as _bb
        out["bk_cs_date"] = _bb.CITYSEL_PREREG_DATE   # the per-city note cites the frozen date
    except Exception as e:
        out["bk_series_error"] = f"{type(e).__name__}: {e}"

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


def _collect_lag_hours(last_collect_iso, generated_at_iso):
    """Hours between the newest market snapshot and the BUILD — i.e. how far behind the
    collector was when this payload was made. Returns None when unknown.

    Measured at build time on purpose. The pill used to compare the newest snapshot against the
    LIVE browser clock, which silently added the dashboard's own publish delay to the collector's
    lag: on 2026-07-27 it read "collector stale" (6.7h) when the collector had run 3h earlier and
    the dashboard simply had not rebuilt since. Publish age is a separate number — the page
    computes that one from generated_at, where it belongs."""
    if not last_collect_iso or not generated_at_iso:
        return None
    try:
        a = datetime.fromisoformat(str(last_collect_iso).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(generated_at_iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (b - a).total_seconds() / 3600.0


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

@lru_cache(maxsize=1)
def _book_graded():
    """`(parsed, graded)` for the 5-city paper book — settled ONCE per build.

    Cached because three consumers need the same frame (panel scalars, equity curve, per-city
    table) and grading walks every stored price path. Returns empty frames on any failure so
    every consumer degrades to "—" rather than the build dying."""
    try:
        import shoulder_book as sb
        return sb.graded_book()
    except Exception:
        import pandas as pd
        return pd.DataFrame(), pd.DataFrame()


@lru_cache(maxsize=1)
def _breadth_graded():
    """`(book, graded)` for the breadth book, OFFLINE — frozen settlements only, no network.

    The daily truth-eval job does the settlement lookups and commits them; the dashboard must
    never depend on a live Gamma call to render."""
    import pandas as pd
    try:
        import shoulder_book_breadth as bb
        book = bb._load_book()
        if book.empty:
            return book, pd.DataFrame()
        return book, bb.grade_book(book=book, lookup=False)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _sig(v, nd: int = 3) -> str:
    """Signed fixed-point with a typographic minus, the form every panel cell expects."""
    return f"{v:+.{nd}f}".replace("-", "−")


def _book_binds() -> dict:
    """5-city structure-book scalars, computed STRUCTURALLY from the graded frame.

    Until 2026-08-04 these were regex-scraped out of `shoulder_book.py --report`'s printed text.
    That coupling broke twice, silently, and both times the panel fell back to its zero/"—"
    defaults while the workflow stayed green:

      * 2026-07-27 — the new "(54 city-days)" and "CI[...]" columns blanked Win rate and the
        Leg 1 edge;
      * 2026-08-02 — the GATE_MIN_DATES amendment appended "(12/30 dates)" to the Leg 1b line,
        which the gate regex anchored past. The public page then showed Leg 1b as **0/80
        forward** for two days while the book actually stood at **78/80, +0.025**.

    A hand-written fixture in the test suite passed through BOTH, because the fixture was a
    second copy of the report format rather than the producer's real output — a test that must
    be manually re-synced with its producer is not a regression test. Reading the numbers
    directly removes the coupling; `test_book_binds_match_the_printed_report` now pins the two
    representations together for whatever still prints.
    """
    b = {"SB_ENTRIES": "—", "SB_GRADED": "—", "SB_AWAIT": "—", "SB_WR": "—",
         "SB_FULL_N": "—", "SB_FULL": "—", "SB_CORE": "—", "SB_FAV_GRADED": "0",
         "SB_MOD_FWD_N": "0", "SB_MOD_FWD": "—", "SB_MOD_NEED": "80", "SB_MOD_PASS": "0"}
    try:
        import shoulder_book as sb
        parsed, graded = _book_graded()
        if parsed.empty:
            return b
        b.update(SB_ENTRIES=str(len(parsed)), SB_GRADED=str(len(graded)),
                 SB_AWAIT=str(len(parsed) - len(graded)))
        if graded.empty:
            return b
        sh = graded[graded["leg"] == "shoulder"]
        fav = graded[graded["leg"] == "favorite"]
        b["SB_FAV_GRADED"] = str(len(fav))
        if len(sh):
            b.update(SB_FULL_N=str(len(sh)), SB_WR=f"{sh['side_won'].mean() * 100:.0f}",
                     SB_FULL=_sig(sh["net_edge"].mean()))
            core = sh[sh["band"] == "core"]
            if len(core):
                b["SB_CORE"] = _sig(core["net_edge"].mean())
            st = sb.moderate_gate_stats(sh)
            if st:
                f = st["forward"]
                b.update(SB_MOD_FWD_N=str(f["n"]), SB_MOD_NEED=str(sb.GATE_MOD[0]),
                         SB_MOD_FWD=_sig(f["taker"]) if f["n"] else "—",
                         SB_MOD_PASS="1" if f.get("gate_pass") else "0")
    except Exception as e:
        b["SB_ERR"] = str(e)[:80]
    return b


def _gate_bind_set(prefix: str, st: dict, need_n: int = 80) -> dict:
    """The six bind strings one pre-registered forward gate needs, in the shape the panel's
    leg-table row reads. `st` is a `moderate_gate_stats` result ({} or None before any forward
    entry grades). One builder for every gate: the breadth panel now shows four, and four
    hand-rolled copies is how the maker cell ends up reading the forward slice in one row and the
    context slice in another."""
    d = {f"{prefix}_N": "0", f"{prefix}_NEED": str(need_n), f"{prefix}_NET": "—",
         f"{prefix}_PASS": "0", f"{prefix}_MAKER": "—", f"{prefix}_MAKER_N": "0"}
    if not st:
        return d
    f = st["forward"]
    d.update({f"{prefix}_N": str(f["n"]),
              f"{prefix}_NET": (f"{f['taker']:+.4f}".replace("-", "−") if f["n"] else "—"),
              f"{prefix}_PASS": "1" if f.get("gate_pass") else "0",
              # maker from the FORWARD slice, matching the taker cell beside it.
              f"{prefix}_MAKER_N": str(f.get("maker_n", 0)),
              f"{prefix}_MAKER": (f"{f['maker']:+.4f}".replace("-", "−")
                                  if f.get("maker_n") else "—")})
    return d


def _breadth_binds() -> dict:
    """Breadth structure-book stats, read OFFLINE from the committed CSV (no network — the
    daily truth-eval job does the settlement lookups and freezes them). Returns bind strings
    with sensible defaults so the panel renders even before the first entry."""
    b = {"BK_ENTRIES": "0", "BK_CITIES": "0", "BK_GRADED": "0", "BK_AWAIT": "0",
         "BK_FULL_N": "0", "BK_FULL_NET": "—", "BK_WR": "—",
         "BK_FULL_MAKER": "—", "BK_FULL_MAKER_N": "0",
         "BK_CS_A_CITIES": "0", "BK_CS_B_CITIES": "0", "BK_CS_DATE": "", "BK_CS_MISSING": ""}
    for p in ("BK_MOD", "BK_CS_A", "BK_CS_B"):
        b.update(_gate_bind_set(p, None))
    try:
        import shoulder_book_breadth as bb
        book, graded = _breadth_graded()                  # offline: frozen settlements only
        if book.empty:
            return b
        # Count CANONICAL cities: the venue's Seoul rename otherwise reads as a 50th city.
        b.update(BK_ENTRIES=str(len(book)),
                 BK_CITIES=str(bb.canonicalize_cities(book)["city"].nunique()),
                 BK_GRADED=str(len(graded)), BK_AWAIT=str(len(book) - len(graded)))
        if not graded.empty:
            sh = graded[graded["leg"] == "shoulder"]
            if len(sh):
                b.update(BK_FULL_N=str(len(sh)),
                         BK_WR=f"{sh['side_won'].mean() * 100:.0f}",
                         BK_FULL_NET=f"{sh['net_edge'].mean():+.4f}".replace("-", "−"))
                # Publish the full-band MAKER number too. It is currently negative (adverse
                # selection: a resting sell-YES fills when the price ticks back up), which is
                # exactly why it must be shown rather than dashed out.
                if "maker_filled" in sh.columns:
                    fl = sh[sh["maker_filled"].astype(bool)]
                    if len(fl):
                        mnet = (fl["side_won"].astype(float)
                                - fl["entry_side_price"].astype(float)).mean()
                        b.update(BK_FULL_MAKER_N=str(len(fl)),
                                 BK_FULL_MAKER=f"{mnet:+.4f}".replace("-", "−"))
            need_mod, _ = bb.GATE_MOD_BREADTH
            b.update(_gate_bind_set(
                "BK_MOD", bb.moderate_gate_stats(graded, prereg_date=bb.BREADTH_PREREG_DATE),
                need_mod))
            # City-selection gates (pre-registered 2026-08-04). The sets are FROZEN literals in
            # shoulder_book_breadth — read, never recomputed here, or the published gate would
            # re-fit itself on every 2-hourly build and could never fail.
            need_cs, _ = bb.GATE_CITYSEL
            missing = []
            for key, sel in (("BK_CS_A", bb.CITYSEL_A), ("BK_CS_B", bb.CITYSEL_B)):
                b.update(_gate_bind_set(key, bb.moderate_gate_stats(
                    graded, prereg_date=bb.CITYSEL_PREREG_DATE, lo=bb.BAND_LO, hi=bb.BAND_HI,
                    gate=bb.GATE_CITYSEL, cities=sel), need_cs))
                b[f"{key}_CITIES"] = str(len(sel))
                missing += [c for c in bb.citysel_missing(book, sel) if c not in missing]
            b["BK_CS_DATE"] = bb.CITYSEL_PREREG_DATE
            # A frozen label that stops matching shrinks the registered set in silence. Publish it.
            b["BK_CS_MISSING"] = ", ".join(missing)
    except Exception as e:
        b["BK_ERR"] = str(e)[:80]
    return b


# A city needs this many graded shoulder entries before it gets a row. Below it the per-city
# interval is so wide the row carries no information at all, and a 50-row table of noise invites
# exactly the cherry-picking the panel's own caption warns against.
BK_CITY_MIN_N = 20


def roi_series(graded, mod_lo: float = 0.10, mod_hi: float = 0.25) -> list:
    """Cumulative RETURN ON CAPITAL by target date for Leg 1 (all shoulder) and Leg 1b (the
    gated [mod_lo, mod_hi) sub-band), plus each day's win rate and the payoff's break-even.

    Shared by BOTH books so the two panels cannot drift apart — they were separate copies for
    about an hour and the 5-city one was already aggregating a different set of legs than the
    breadth one.

    Why ROI and not summed units. Until 2026-08-04 these charts plotted a cumulative sum of
    per-contract edge. That quantity is exactly `n × mean` — the breadth headline +38.0u was 1344
    contracts × 2.83¢ — so its SLOPE tracked how many markets the collector picked up, not how
    good the bets were: widening the city list would have steepened the line at identical edge.
    It read as a performance chart while behaving like a volume chart. ROI on deployed capital is
    invariant to inventory; only a change in edge moves it.

    Leg 2 (favourites) is excluded: these are the shoulder legs, and mixing a 4-contract leg into
    the same line would put two different trades on one axis.
    """
    import pandas as pd
    import config
    import shoulder_book as sb
    if graded is None or graded.empty or "leg" not in graded.columns:
        return []
    sh = graded[graded["leg"] == "shoulder"].copy()
    if sh.empty:
        return []
    yes = sh["entry_yes_price"].astype(float)
    df = pd.DataFrame({
        "t": pd.to_datetime(sh["target_date"], errors="coerce"),
        "net": sh["net_edge"].astype(float),
        # capital actually laid out per contract, on the same taker-conservative basis as the
        # P&L: the entry price plus the half-spread crossed on entry.
        "cost": sh["entry_side_price"].astype(float) + config.HALF_SPREAD,
        "won": sh["side_won"].astype(bool),
        "inmod": (yes >= mod_lo) & (yes < mod_hi),
    }).dropna(subset=["t"])
    if df.empty:
        return []
    for c in ("net", "cost"):
        df[f"m{c}"] = df[c].where(df["inmod"], 0.0)
    g = df.groupby("t").agg(net=("net", "sum"), cost=("cost", "sum"),
                            mnet=("mnet", "sum"), mcost=("mcost", "sum"),
                            n=("net", "size"), wr=("won", "mean")).sort_index()
    mn = df[df["inmod"]].groupby("t").agg(mwr=("won", "mean"), mn=("won", "size"))
    cum = g[["net", "cost", "mnet", "mcost"]].cumsum()

    def _roi(num, den):
        return round(float(num / den * 100.0), 3) if den else 0.0

    be = sb.breakeven_win_rate(df["net"], df["won"])
    mbe = sb.breakeven_win_rate(df.loc[df["inmod"], "net"], df.loc[df["inmod"], "won"])
    out = []
    for t in g.index:
        c = cum.loc[t]
        out.append({"t": t.strftime("%b %-d"),
                    "roi": _roi(c["net"], c["cost"]),
                    "mroi": _roi(c["mnet"], c["mcost"]),
                    "wr": round(float(g["wr"].loc[t]) * 100, 2),
                    "mwr": (round(float(mn["mwr"].loc[t]) * 100, 2) if t in mn.index else None),
                    "n": int(g["n"].loc[t]),
                    "mn": (int(mn["mn"].loc[t]) if t in mn.index else 0),
                    # constant on every point so the JS can draw them as reference lines
                    "be": round(be * 100, 2) if be is not None else None,
                    "mbe": round(mbe * 100, 2) if mbe is not None else None,
                    # kept for the tooltip only — never the y-axis again
                    "u": round(float(c["net"]), 2), "mu": round(float(c["mnet"]), 2),
                    "cap": round(float(c["cost"]), 2)})
    return out


def _breadth_equity() -> list:
    """Breadth book's ROI curve — see `roi_series`."""
    _, graded = _breadth_graded()
    return roi_series(graded)


def _breadth_cities() -> list:
    """Per-city shoulder-band edge across the breadth book. DESCRIPTIVE ONLY.

    Answers "does any city sell shoulders better than the others?" — but the honest answer is
    almost certainly "no, and this table cannot tell you". With ~50 cities, ~2-3 clear a 95% CI
    by chance under a true null. The payload therefore ships `sig_n` (cities whose interval
    excludes zero) beside `expected_sig` (0.05 × cities) so the page can print the comparison
    rather than leaving the reader to eyeball a sorted list and pick a winner. See
    shoulder_book.per_city_stats."""
    import shoulder_book as sb
    _, graded = _breadth_graded()
    if graded.empty or "leg" not in graded.columns:
        return []
    sh = graded[graded["leg"] == "shoulder"]
    return sb.per_city_stats(sh, min_n=BK_CITY_MIN_N)


def _f4(v) -> str:
    """Format a Brier for display, tolerating the parsed-string fallback."""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def build_payload(d: dict, series: dict) -> dict:
    days, span = _fmt_span(d)
    series["disp"] = d.get("disp", [])
    G = lambda k, dflt="—": str(d.get(k, dflt))
    # Headline accuracy comes from the PAIRED common set (score.all); the parsed
    # br_* values measure the model over more markets than the ensemble and flatter it.
    paired = ((series.get("score") or {}).get("all")) or {}
    br = {k: (paired.get(k) if paired.get(k) is not None else d.get(f"br_{k}"))
          for k in ("market", "model", "ens")}
    try:
        model_wins = float(br["model"]) < float(br["market"])
        skill = (1.0 - float(br["model"]) / float(br["market"])) * 100.0
        skill_txt = f"{skill:+.1f}%".replace("-", "−")
    except Exception:
        model_wins, skill_txt = False, "—"
    takeaway = ("Brier score is the scoreboard (lower = more accurate). <b>No real money is "
                "traded until a forecaster's Brier falls below the market's</b> — the table above "
                "is the current standing; the candidate edges below are walked forward on paper.")
    edge_chip = ""

    # Paper-book headline: RETURN ON CAPITAL, not summed units. "+7.45u" was 305 contracts ×
    # 2.4¢ — a number that grows with inventory alone, so it flattered the book every time the
    # collector widened. ROI is invariant to how many tickets were bought.
    eq = series.get("equity") or []
    book_net = f"{eq[-1]['roi']:+.2f}%".replace("-", "−") if eq else "—"
    # Leg 1b's return on capital was plotted but never stated. The panel showed ONE headline
    # number beside a TWO-line chart, so the moderate band — the leg that is actually gated, and
    # the one running above the full band — had no figure a reader could quote.
    book_net_mod = (f"{eq[-1]['mroi']:+.2f}%".replace("-", "−")
                    if eq and eq[-1].get("mroi") is not None else "—")
    book_be = (f"{eq[-1]['be']:.1f}" if eq and eq[-1].get("be") is not None else "—")
    book_cap = f"${eq[-1]['cap']:,.0f}" if eq else "—"
    # Name the five cities. This panel and the breadth panel below run the SAME legs over
    # different universes, and only the breadth one said which ("every Polymarket weather city").
    book_cities = ", ".join(CITY_META[c][0] for c in CITY_ORDER)
    hb = series.get("heartbeat") or []
    runs_today = str(hb[-1]["n"]) if hb else "—"

    commit_dt = _last_commit_dt()
    now = datetime.now(timezone.utc)
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at": generated_at,
        "data_through_iso": commit_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_collect_iso": series.get("last_collect_iso"),
        # collector lag AS OF THIS BUILD — never mixed with how old the publish itself is
        "collect_lag_hours": _collect_lag_hours(series.get("last_collect_iso"), generated_at),
        "bind": {
            "UPDATED": commit_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "GATE_STATUS": G("gate_status"), "GATE_MKTS": G("gate_mkts"), "GATE_BETS": G("gate_bets"),
            "GATE_MKTS_THR": G("gate_mkts_thr", "150"),
            "DATA_DAYS": days, "SPAN": span,
            "BR_MARKET": _f4(br["market"]), "BR_ENS": _f4(br["ens"]), "BR_MODEL": _f4(br["model"]),
            "N_MKTS": G("n_mkts"), "CRPS_MODEL": G("crps_model"), "CRPS_ENS": G("crps_ens"),
            "SKILL": skill_txt, "BOOK_NET": book_net, "RUNS_TODAY": runs_today,
            "BOOK_BE": book_be, "BOOK_CAP": book_cap,
            "BOOK_NET_MOD": book_net_mod, "BOOK_CITIES": book_cities,
            "BOOK_NCITIES": str(len(CITY_ORDER)),
            # SB_* (5-city book) and BK_* (breadth) both come from their graded frames — never
            # from a printed report. SB_FULL_N is Leg 1 only; SB_GRADED counts every leg.
            **_book_binds(),
            **_breadth_binds(),
        },
        "html": {
            "EDGE_CHIP": edge_chip,
            "TAKEAWAY": takeaway,
            "CITIES_HTML": _cities_html(series.get("city", []), d),
            "CAPTURE_HTML": _capture_html(series.get("capture", [])),
        },
        "series": series,
    }


def capture_coverage() -> list:
    """Per-city collection health for the CAPTURE tier. Never raises.

    This answers one question — "is the new pipeline actually alive?" — and deliberately reports
    NO accuracy figures: these cities have no model, and until their markets resolve there is
    nothing to score either. Reporting a Brier here would be inventing one.

    Exception-safe by contract. `main()` refuses to publish when `compute_series` raises
    (`_series_error`), so a missing or malformed capture file must degrade to zeroes rather than
    take the whole dashboard down with it — the five-city page must keep publishing regardless.
    """
    import pandas as pd

    rows = []
    for city, (station, series_ticker) in CAPTURE_META.items():
        rec = {"city": city, "station": station, "series": series_ticker,
               "pm_markets": 0, "pm_snaps": 0, "kal_markets": 0, "kal_books": 0,
               "kal_candles": 0, "graded": 0, "first_target": None}
        try:
            slug = re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")

            pm = PKG / "data" / "polymarket" / f"{slug}_snapshots.csv"
            if pm.exists():
                d = pd.read_csv(pm, low_memory=False)
                rec["pm_snaps"] = int(len(d))
                if "condition_id" in d.columns:
                    rec["pm_markets"] = int(d["condition_id"].nunique())
                dates = pd.to_datetime(d.get("end_date_iso"), errors="coerce", utc=True)
                if dates is not None and dates.notna().any():
                    rec["first_target"] = str(dates.min().date())

            for key, fname in (("kal_markets", "markets"), ("kal_books", "books"),
                               ("kal_candles", "candles")):
                p = PKG / "data" / "kalshi" / f"{slug}_{fname}.csv"
                if p.exists():
                    rec[key] = int(len(pd.read_csv(p, low_memory=False)))
        except Exception as exc:                      # noqa: BLE001 — must never block publish
            sys.stderr.write(f"::warning::capture coverage for {city} unavailable: {exc}\n")
        rows.append(rec)
    return rows


def _capture_html(rows: list) -> str:
    """The capture-tier panel. Shows collection health, never accuracy."""
    if not rows:
        return '<div class="cb mono">capture tier not configured</div>'
    out = []
    for r in rows:
        live = r["pm_markets"] > 0 and r["kal_markets"] > 0
        chip = ('<span class="chip good">both venues</span>' if live
                else '<span class="chip warn">awaiting data</span>')
        awaiting = ('<span class="chip">awaiting first resolution</span>'
                    if r["graded"] == 0 else
                    f'<span class="chip good">{r["graded"]} graded</span>')
        out.append(
            f'<div class="city"><div class="cn">{r["city"]} '
            f'<span class="cc mono">{r["station"]}</span></div>'
            f'<div class="cb mono">PM {r["pm_markets"]} mkts · '
            f'KAL {r["kal_markets"]}/{r["kal_books"]}/{r["kal_candles"]}</div>'
            f'{chip}{awaiting}</div>')
    return "\n      ".join(out)


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


def _missing_cities(payload: dict) -> list:
    """Expected cities (CITY_ORDER) that have NO gradable markets in this build — the signature
    of a truth-source outage (e.g. IEM 503 dropping Seoul/London). A city missing here silently
    corrupts every downstream number (Brier/ROI/buckets/win-rate), so the caller must refuse to
    publish rather than overwrite the last good dashboard with a partial one."""
    present = {c.get("city") for c in payload.get("series", {}).get("city", [])}
    return [c for c in CITY_ORDER if c not in present]


GUIDE_SRC = Path(__file__).resolve().parent / "guide.html"
_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def guide_values(payload: dict) -> dict:
    """Live figures for the guide's `{{KEY}}` placeholders, from the SAME payload the dashboard
    renders — so the two pages cannot state different numbers for the same quantity.

    Only figures that are claims about the PRESENT belong here. The chronology's per-incident
    numbers ("4 of 60 audited markets were graded backwards") are historical facts with a date on
    them and stay hard-coded: re-deriving them from today's data would silently rewrite the record
    of what was measured at the time, which is the opposite of a post-mortem's job.
    """
    b, s = payload.get("bind", {}), payload.get("series", {})
    pooled = s.get("pooled") or {}
    g, r, e = pooled.get("all", {}), pooled.get("recent", {}), pooled.get("vs_ens", {})
    sc = (s.get("score") or {}).get("all", {})

    def sig(v, nd=4):
        return "—" if v is None else f"{v:+.{nd}f}".replace("-", "−")

    def ci(d):
        return "—" if not d else f"[{sig(d.get('lo'))}, {sig(d.get('hi'))}]"

    def num(v, nd=4):
        # NOT f"{v:.4f}" on a defaulted NaN — that renders the literal string "nan" into the
        # published prose, which reads as a typo rather than as missing data.
        return "—" if v is None else f"{float(v):.{nd}f}"

    v = {
        "COMPILED": (payload.get("generated_at") or "")[:10] or "—",
        # headline sample + the powered model-vs-market test
        "MKTS": str(g.get("n", "—")), "CITYDAYS": str(g.get("clusters", "—")),
        "GAP": sig(g.get("gap")), "GAP_CI": ci(g),
        "GAP60": sig(r.get("gap")), "GAP60_CI": ci(r), "N60": str(r.get("n", "—")),
        "GAP60_DAYS": str(r.get("clusters", "—")),
        # three-way Brier — the COMMON set, which is a different (smaller) sample on purpose
        "COMMON": str(sc.get("n", "—")),
        "BR_MARKET": num(sc.get("market")), "BR_ENS": num(sc.get("ens")),
        "BR_MODEL": num(sc.get("model")),
        # model vs the ensemble it calibrates — paired, the guide's central Family-A question
        "VSENS": sig(e.get("gap")), "VSENS_CI": ci(e),
        "VSENS_N": str(e.get("n", "—")), "VSENS_DAYS": str(e.get("clusters", "—")),
        "CRPS_MODEL": str(b.get("CRPS_MODEL", "—")), "CRPS_ENS": str(b.get("CRPS_ENS", "—")),
        # structure books
        "SB_GRADED": str(b.get("SB_GRADED", "—")), "SB_FULL": str(b.get("SB_FULL", "—")),
        "SB_FULL_N": str(b.get("SB_FULL_N", "—")), "SB_WR": str(b.get("SB_WR", "—")),
        "SB_MOD_N": str(b.get("SB_MOD_FWD_N", "—")), "SB_MOD": str(b.get("SB_MOD_FWD", "—")),
        "BOOK_NET": str(b.get("BOOK_NET", "—")), "BOOK_NET_MOD": str(b.get("BOOK_NET_MOD", "—")),
        "BOOK_BE": str(b.get("BOOK_BE", "—")),
        "BK_CITIES": str(b.get("BK_CITIES", "—")), "BK_GRADED": str(b.get("BK_GRADED", "—")),
        "BK_FULL_N": str(b.get("BK_FULL_N", "—")), "BK_FULL_NET": str(b.get("BK_FULL_NET", "—")),
        "BK_MOD_N": str(b.get("BK_MOD_N", "—")), "BK_MOD_NET": str(b.get("BK_MOD_NET", "—")),
        "BK_WR": str(b.get("BK_WR", "—")),
        "BK_FULL_MAKER": str(b.get("BK_FULL_MAKER", "—")),
    }
    return v


def render_guide(template: str, values: dict) -> str:
    """Substitute `{{KEY}}` in the guide, and REFUSE on any placeholder left unresolved.

    The guide publishes as static HTML with no JS, so an unsubstituted `{{GAP}}` would sit on the
    public page as literal braces — the quiet, plausible-looking wrongness this project keeps
    paying for. Failing here stops the publish and keeps the last good copy instead.
    """
    out = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)
    left = sorted(set(_PLACEHOLDER.findall(out)))
    if left:
        raise KeyError(f"guide has unresolved placeholders: {', '.join(left)}")
    return out


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(gather(), compute_series())
    # Completeness guard: never publish a degraded dashboard. A transient truth outage that drops
    # whole cities must FAIL LOUDLY (red run + last-good copy preserved), not silently ship half a
    # dataset. Override with DASHBOARD_ALLOW_PARTIAL=1 only for deliberate local/partial builds.
    # A swallowed compute_series exception means panels are missing but the page still renders
    # and the workflow still exits 0 — exactly how the 2026-07-28 NameError shipped an empty
    # "Recent settlements" table. Treat it like a truth outage: refuse, keep the last good copy.
    serr = _series_error(payload)
    if serr and os.environ.get("DASHBOARD_ALLOW_PARTIAL") != "1":
        sys.stderr.write(
            f"::error::dashboard build INCOMPLETE — compute_series failed with {serr}. Panels "
            f"after the failure point would publish empty. Refusing; last good copy kept. Set "
            f"DASHBOARD_ALLOW_PARTIAL=1 to override.\n")
        sys.exit(1)
    missing = _missing_cities(payload)
    if missing and os.environ.get("DASHBOARD_ALLOW_PARTIAL") != "1":
        sys.stderr.write(
            f"::error::dashboard build INCOMPLETE — no gradable markets for {missing} "
            f"(likely a truth-source outage). Refusing to publish a degraded dashboard; the last "
            f"good published copy is kept. Set DASHBOARD_ALLOW_PARTIAL=1 to override.\n")
        sys.exit(1)
    out.write_text(render_shell(payload), encoding="utf-8")
    data_out = out.parent / "data.json"
    data_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # The guide is rendered HERE rather than copied by the workflow, so its live figures come from
    # this same payload. Two hand-maintained copies of one statistic drifting apart, with only one
    # of them right, is this repo's most-repeated defect.
    guide_out = out.parent / "guide.html"
    guide_out.write_text(
        render_guide(GUIDE_SRC.read_text(encoding="utf-8"), guide_values(payload)),
        encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes) + {data_out} ({data_out.stat().st_size} "
          f"bytes) + {guide_out} ({guide_out.stat().st_size} bytes)")


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

  .guidelink{display:inline-block;margin:0 0 16px;padding:8px 14px;border:1px solid var(--line);border-radius:3px;color:var(--ink);text-decoration:none;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;transition:border-color .15s,color .15s}  .guidelink:hover,.guidelink:focus-visible{border-color:var(--good);color:var(--good)}
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
    <div class="eyebrow">Forecast accuracy · <span id="wLabel">the full track record</span></div>
    <div class="verdict" id="verdictLine">Brier score by forecaster — lower is more accurate; the leader is flagged.</div>

    <div class="vhead">
      <span class="lbl">Window</span>
      <span class="seg" id="winseg"><button class="on" data-win="all">All time</button><button data-win="recent">Last 60</button></span>
    </div>
    <div class="rank" id="rank"></div>
    <div class="cap" id="pooledLine" style="margin-top:10px"></div>
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
          <div class="find" style="font-size:13px">Accuracy (Brier) vs. profit (ROI)</div>
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
      <div class="kpi r"><div class="k">Model − market</div><div class="v r" id="k_gap">—</div><div class="s" id="k_gap_ci">Brier gap</div></div>
      <div class="kpi"><div class="k">Live exposure</div><div class="v">$0</div><div class="s">paper only</div></div>
    </div>
    <p class="lede" data-bind-html="TAKEAWAY"></p>
  </section>

  <!-- 01 ACCURACY -->
  <section>
    <div class="shd"><span class="n">01</span><h2>Accuracy</h2><span class="r">over time · by city · by temperature</span></div>
    <div class="cwrap">
      <div class="panel">
        <div class="find">Brier score over time</div>
        <div class="findsub">Running Brier as markets settle · lower = better</div>
        <div class="leg"><span><i style="background:var(--market)"></i>Market</span><span><i style="background:var(--model)"></i>Model</span><span><i style="background:var(--ens)"></i>Ensemble</span></div>
        <div class="chartwrap"><div id="c_acc"></div></div>
        <div class="cap">Running Brier score as markets settle — market, model, ensemble. Lower is more accurate.</div>
      </div>
      <div class="panel">
        <div class="find">Brier score by city</div>
        <div class="findsub">Brier per city · market vs ensemble vs model</div>
        <div class="leg"><span><i style="background:var(--market);border-radius:50%"></i>Market</span><span><i style="background:var(--ens);border-radius:50%"></i>Ensemble</span><span><i style="background:var(--model);border-radius:50%"></i>Model</span></div>
        <div class="chartwrap"><div id="c_city"></div></div>
        <div class="cap">Each row is one city — market, ensemble, and model Brier <b>on the same markets</b> (the set all three scored), so the three dots are directly comparable. Further left is more accurate. A row marked <b>†</b> had every market settle NO: Brier carries no accuracy signal there, since the lowest numbers win by default.</div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="statrow" style="border-top:0;padding-top:0;margin-top:0">
        <div class="st"><div class="k">Temp accuracy · CRPS</div><div class="v" id="crps_v"><span data-bind="CRPS_MODEL">—</span> <small>model</small></div></div>
        <div class="st"><div class="k">vs ensemble</div><div class="v"><span data-bind="CRPS_ENS">—</span> <small>°C err</small></div><div class="s" id="crps_note">—</div></div>
        <div class="st" style="flex:1;min-width:240px"><div class="k">what it measures</div><div class="v" style="font-size:11.5px;font-family:var(--sans);color:var(--ink2);line-height:1.5">CRPS is the temperature-forecast error in °C (lower = sharper). It scores the point forecast against the station reading — not the market's bin prices.</div></div>
      </div>
    </div>
  </section>

  <!-- 02 DIAGNOSTICS -->
  <section>
    <div class="shd"><span class="n">02</span><h2>Model diagnostics</h2><span class="r">calibration · overconfidence · buckets</span></div>
    <div class="cwrap">
      <div class="panel">
        <div class="find">Model calibration by confidence bin</div>
        <div class="findsub">Real outcome rate minus what the model said, by confidence bin</div>
        <div class="leg"><span><i style="background:var(--market)"></i>happened more than it said</span><span><i style="background:var(--model)"></i>happened less</span></div>
        <div class="chartwrap"><div id="c_calib"></div></div>
        <div class="cap">Realized outcome rate minus the model's predicted rate, by confidence bin. Blue = happened more often than predicted; red = less. On the line = calibrated.</div>
      </div>
      <div class="panel">
        <div class="find">Forecast spread calibration, by month</div>
        <div class="findsub">Spread calibration by month · 1.0 = honest, &gt;1.15 = overconfident</div>
        <div class="chartwrap"><div id="c_disp"></div></div>
        <div class="cap">Realized error divided by the forecast's stated spread, by month. 1.0 = calibrated; above 1.0 (amber) = spread too narrow; below 1.0 = too wide.</div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="find">Accuracy by bucket, with forward gates</div>
      <div class="findsub">A "bucket" = a city × lead-time slice. Δ = market − model Brier. Nominated buckets carry a forward gate that must pass before any real order.</div>
      <div style="overflow-x:auto"><table class="data" id="t_buckets" style="margin-top:4px"><thead><tr>
        <th>Bucket</th><th class="num">n</th><th class="num">Model</th><th class="num">Market</th><th class="num">Δ</th><th>Forward gate</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <!-- 03 PAPER BOOK -->
  <section>
    <div class="shd"><span class="n">03</span><h2>Paper book</h2><span class="r">model-free structure legs · <span data-bind="BOOK_NCITIES">5</span> modelled cities</span> <span class="paperflag">Paper — no real money</span></div>
    <p class="lede" style="margin:0 2px 14px">A <b>separate book</b> from the model above — it bets on market <b>structure</b>, not the weather. <b>Leg 1</b> sells over-priced 5–35¢ shoulder bins; <b>Leg 2</b> buys 65–85¢ YES-favourites &gt;12h before close. Independent mispricings, each gated on its own before a single real order. <b>Leg 1b</b> refines Leg 1 to the over-priced 10–25¢ sub-band (pre-registered 2026-07-23, forward-only).</p>
    <div class="cwrap">
      <div class="panel">
        <div class="find" id="sb_title">Shoulder book · return on capital · <span data-bind="BOOK_NCITIES">5</span> modelled cities</div>
        <div class="findsub"><b><span data-bind="BOOK_CITIES">—</span></b> only — the same legs run across every weather city in the breadth book below, which is the wider cross-check. Cumulative return on the capital actually laid out · taker fees paid. <b>Not</b> a running total of units — that number grows with how many tickets we buy, so it climbs even when the edge doesn't. The early points sit on a handful of contracts and swing wildly; the line settling downward is the sample growing, <b>not</b> the edge improving.</div>
        <div class="chartwrap"><div id="c_equity"></div></div>
        <div class="leg" style="margin-top:8px"><span><i style="background:var(--good)"></i>Leg 1 · all shoulder [5–35¢]</span><span><i style="background:var(--market)"></i>Leg 1b · moderate [10–25¢]</span></div>
        <div class="statrow">
          <div class="st"><div class="k">Return on capital <span class="dim">1 / 1b</span></div><div class="v"><span id="sb_net" data-bind="BOOK_NET">—</span> <span class="dim" style="font-size:11px">/ <span id="sb_net_mod" data-bind="BOOK_NET_MOD">—</span></span></div></div>
          <div class="st"><div class="k">Capital laid out</div><div class="v" data-bind="BOOK_CAP">—</div></div>
          <div class="st"><div class="k">Win rate <span class="dim">/ breakeven</span></div><div class="v"><span data-bind="SB_WR">—</span>% <span class="dim" style="font-size:11px">/ <span data-bind="BOOK_BE">—</span>%</span></div></div>
          <div class="st"><div class="k">Settled</div><div class="v" data-bind="SB_GRADED">—</div></div>
        </div>
      </div>
      <div class="panel">
        <div class="find">Each leg proves itself before real money.</div>
        <div class="findsub">A leg must clear a pre-registered forward target — settlements holding positive expectancy out-of-sample — before it can trade. In-sample profit doesn't count.</div>
        <table class="data" style="margin-top:2px">
          <tr><th>Leg</th><th class="num">Graded</th><th class="num">Edge <span class="dim">$/contract</span></th><th>Status</th></tr>
          <tr><td class="city">1 · sell shoulder</td><td class="num" data-bind="SB_FULL_N">—</td><td class="num" id="sb_full_cell" data-bind="SB_FULL">—</td><td><span class="pill2 warn">paper</span></td></tr>
          <tr><td class="city">2 · buy favourite</td><td class="num" id="leg2n">—</td><td class="num" id="leg2edge">—</td><td><span class="pill2" id="leg2status">pending</span></td></tr>
          <tr><td class="city">1b · moderate [10–25¢]</td><td class="num" id="modn">—</td><td class="num" id="modedge">—</td><td><span class="pill2" id="modstatus">forward</span></td></tr>
        </table>
        <p class="cap" style="margin-top:14px"><b>Nothing is live — this is a paper book.</b> Each leg trades real money only after its own pre-registered forward gate passes. In-sample profit doesn't count; the honest measure is the settled edge per contract above.</p>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="find">Breadth book · every Polymarket weather city</div>
      <div class="findsub">The same model-free structure legs, extended from 5 cities to every active weather market (~51), graded on Polymarket's own settlement. A separately pre-registered forward gate (2026-07-23) that cross-validates the 5-city book at ~10× the inventory.</div>
      <div class="statrow">
        <div class="st"><div class="k">Entries</div><div class="v" data-bind="BK_ENTRIES">—</div></div>
        <div class="st"><div class="k">Cities</div><div class="v" data-bind="BK_CITIES">—</div></div>
        <div class="st"><div class="k">Settled</div><div class="v" data-bind="BK_GRADED">—</div></div>
        <div class="st"><div class="k">Awaiting</div><div class="v" data-bind="BK_AWAIT">—</div></div>
      </div>
      <table class="data" style="margin-top:10px">
        <tr><th>Leg</th><th class="num">Forward</th><th class="num">Taker <span class="dim">$/contract</span></th><th class="num">Maker <span class="dim">$/contract</span></th><th>Status</th></tr>
        <tr><td class="city">1b · moderate [10–25¢] · all cities</td><td class="num" id="bkmodn">—</td><td class="num" id="bkmodedge">—</td><td class="num" id="bkmodmaker">—</td><td><span class="pill2" id="bkmodstatus">forward</span></td></tr>
        <tr><td class="city">1 · sell shoulder [5–35¢] · all cities</td><td class="num" id="bkfulln">—</td><td class="num" id="bkfulledge" data-bind="BK_FULL_NET">—</td><td class="num" id="bkfullmaker" data-bind="BK_FULL_MAKER">—</td><td><span class="pill2 warn">paper</span></td></tr>
        <tr><td class="city">1d · shoulder, <span id="bkcsan">31</span> cities that were winning</td><td class="num" id="bkcsan_n">—</td><td class="num" id="bkcsaedge">—</td><td class="num" id="bkcsamaker">—</td><td><span class="pill2" id="bkcsastatus">forward</span></td></tr>
        <tr><td class="city">1e · shoulder, best <span id="bkcsbn">12</span> cities</td><td class="num" id="bkcsbn_n">—</td><td class="num" id="bkcsbedge">—</td><td class="num" id="bkcsbmaker">—</td><td><span class="pill2" id="bkcsbstatus">forward</span></td></tr>
      </table>
      <p class="cap" style="margin-top:8px"><b>1d and 1e are falsification tests, not tips.</b> Both drop the cities that had been losing — in-sample that lifts return on capital from <b>+0.80%</b> to <b>+2.21%</b> and <b>+4.38%</b>. But reshuffling which city each city-day belongs to, destroying city identity while keeping every price and outcome, and applying the identical rule yields <b>+2.15%</b> — so almost the whole gain is the arithmetic of selecting on the numbers you then report. Two more checks agree: per-city spread (sd 0.0252) barely exceeds what pure noise predicts (0.0243), and city edge in the first half of the sample predicts the second half at <b>ρ = +0.14</b>, sign agreement 49% — a coin flip. The two sets were frozen on <span data-bind="BK_CS_DATE">2026-08-04</span> and only entries recorded after that date are graded here. <span id="bkcsmiss"></span></p>

      <div class="findsub" style="margin-top:20px">Return on capital laid out · all cities · taker fees paid. The moderate band runs <b>above</b> the full band because the rest of the shoulder — mainly the 25–35¢ core — loses; the whole apparent edge sits in 10–25¢. Ten target days is not a track record.</div>
      <div class="chartwrap"><div id="c_bkequity"></div></div>
      <div class="leg" style="margin-top:8px"><span><i style="background:var(--good)"></i>All shoulder [5–35¢]</span><span><i style="background:var(--market)"></i>Moderate band [10–25¢] only</span></div>

      <div class="findsub" style="margin-top:22px">Daily win rate vs the rate this payoff needs to break even</div>
      <div class="chartwrap"><div id="c_bkwr"></div></div>
      <div class="leg" style="margin-top:8px"><span><i style="background:var(--market)"></i>Win rate · moderate band</span><span><i style="background:var(--warn)"></i>Break-even</span></div>
      <p class="cap" style="margin-top:8px">This is a <b>short-volatility</b> trade: it wins small and often, and loses big and rarely — measured over the breadth book, a win pays <b>+14.7¢</b> and a loss costs <b>−82.9¢</b>, a 5.6× ratio. So the win rate alone says nothing; only its distance above the break-even line does, and on the full band that gap is <b>0.7 percentage points</b>. A payoff shaped like this draws a smooth rising line right up until the tail lands — the smoothness above is not evidence, it is the signature.</p>

      <div class="findsub" style="margin-top:22px">By city · shoulder band [5–35¢]</div>
      <div id="bkcitynote" class="cap" style="margin:6px 0 10px"></div>
      <div id="bkcities"></div>
      <p class="cap" style="margin-top:12px"><b>Paper — no real money.</b> <b>Maker net</b> = filled-only, no spread/fee, rebate excluded (after the 2026-07-27 fill-detector fix the maker edge is <b>not</b> established: 5-city +0.056 CI [−0.018,+0.129], breadth at comparable liquidity −0.007 [−0.028,+0.015] — both consistent with zero; thin books &lt;1k liquidity are significantly negative). Forward-only: only entries recorded on/after 2026-07-23 count toward the gate, so the hypothesis is never graded on the data that suggested it.</p>
    </div>
  </section>

  <!-- 04 ENTRIES -->
  <section>
    <div class="shd"><span class="n">04</span><h2>Recent settlements</h2><span class="r">last 60 · every graded market</span></div>
    <div class="panel">
      <div class="tblscroll"><table class="data" id="t_recent"><thead><tr>
        <th>Date</th><th>City</th><th>Bin</th><th class="num">Mdl</th><th class="num">Mkt</th><th>Result</th><th>Closer</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <!-- 05 HEALTH -->
  <section>
    <div class="shd"><span class="n">05</span><h2>System health</h2><span class="r">collection · data freshness</span></div>
    <div class="kpis" style="margin-top:0">
      <div class="kpi g"><div class="k">Collector</div><div class="v" id="collectorbig">—</div><div class="s">last snapshot · hourly cron</div></div>
      <div class="kpi"><div class="k">Runs today</div><div class="v"><span data-bind="RUNS_TODAY">—</span><span style="font-size:12px;color:var(--faint)"> /24</span></div><div class="s">cycles · GitHub drops some</div></div>
      <div class="kpi"><div class="k">Data span</div><div class="v"><span data-bind="DATA_DAYS">—</span><span style="font-size:12px;color:var(--faint)"> d</span></div><div class="s" data-bind="SPAN">—</div></div>
      <div class="kpi"><div class="k">Data through</div><div class="v" style="font-size:14px" data-bind="UPDATED">—</div><div class="s">last commit</div></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="find" style="font-size:13px">Settlement-truth freshness</div>
      <div class="findsub">each market grades against the reading it settles on · Hong Kong lags ~3 weeks by design</div>
      <div class="cities" data-bind-html="CITIES_HTML"></div>
    </div>

    <div class="panel" style="margin-top:14px">
      <div class="find" style="font-size:13px">Cross-venue capture (new 2026-08-04)</div>
      <div class="findsub">seven US cities where Polymarket and Kalshi resolve on the SAME station, differing only in the ruler &middot; collection health only &mdash; these cities are deliberately not modelled, so there is no accuracy figure to report yet</div>
      <div class="cities" data-bind-html="CAPTURE_HTML"></div>
    </div>
  </section>

  <footer>
    <a class="guidelink" href="./guide.html">Read the guide — every model we tried, and where each failed</a>
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
  var lastGen = null, lastCollect = null, collectLagH = null, prevBind = null;
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
      // A city whose graded markets ALL settled the same way carries no discrimination: Brier
      // is mean(p^2), so the lowest numbers "win" by construction. Dim it and mark it rather
      // than letting it read as an accuracy result.
      var dead = !!r.degenerate, dim = dead ? 0.32 : 1;
      svg.appendChild(el("text", { class: "axis", x: 0, y: y + 3.5, fill: css("--ink2"), opacity: dim }, r.city + (dead ? " †" : "")));
      var xs = [r.market, r.ens, r.model].filter(function (v) { return v != null; }).map(xOf);
      svg.appendChild(el("line", { x1: Math.min.apply(null, xs), y1: y, x2: Math.max.apply(null, xs), y2: y, stroke: css("--axis"), "stroke-width": 1.5, opacity: dim }));
      [["market", r.market, css("--market")], ["ens", r.ens, css("--ens")], ["model", r.model, css("--model")]].forEach(function (m) {
        if (m[1] == null) return;
        var c = el("circle", { cx: xOf(m[1]), cy: y, r: 4, fill: m[2], opacity: dim });
        c.addEventListener("mousemove", function () {
          tip.innerHTML = '<div class="tt">' + esc(r.city) + ' · n=' + r.n + '</div>'
            + '<div class="tr"><span><i style="background:' + css("--market") + '"></i>Market</span><b>' + r.market.toFixed(3) + '</b></div>'
            + (r.ens != null ? '<div class="tr"><span><i style="background:' + css("--ens") + '"></i>Ensemble</span><b>' + r.ens.toFixed(3) + '</b></div>' : '')
            + '<div class="tr"><span><i style="background:' + css("--model") + '"></i>Model</span><b>' + r.model.toFixed(3) + '</b></div>'
            + (dead ? '<div class="tr" style="color:var(--faint)">† all ' + r.n + ' settled NO — no accuracy signal; lowest numbers win by default</div>' : '');
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
    // Pooled paired gap + clustered interval. The ranking alone cannot distinguish a real
    // gap from sampling noise, so the interval is what makes this a finding. Rendered here on
    // the live path, alongside the gap KPI whose number this interval belongs to.
    var pl = document.getElementById("pooledLine"), ci = document.getElementById("k_gap_ci");
    var P = (D.pooled || {})[win] || null;
    var sgn = function (x) { return (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(4); };
    if (P && isFinite(P.gap)) {
      var worse = P.lo > 0, better = P.hi < 0;
      // The interval lives in the KPI box that already shows this number - a point estimate
      // beside its own uncertainty, rather than the same figure twice in two places.
      if (ci) ci.textContent = "95% CI [" + sgn(P.lo) + ", " + sgn(P.hi) + "]";
      if (pl) {
        pl.innerHTML = worse
          ? "Across all " + P.n + " markets the model is <b>measurably worse</b> than the market: "
            + "the whole interval sits above zero, so this is a real gap, not sampling noise."
          : better
          ? "Across all " + P.n + " markets the model is <b>measurably better</b> than the market: "
            + "the whole interval sits below zero."
          : "Model and market are <b>indistinguishable</b> here - the interval includes zero, so "
            + "the ranking above could be chance.";
        pl.innerHTML += " (" + P.clusters + " independent city-days; bins settling on one day "
          + "count once, since they share a single weather outcome.)";
        // The ranking above is a THREE-way comparison, so it can only use markets the ensemble
        // also scored; this test needs no ensemble and uses every gradable market. Different n
        // on purpose - say so, or the two look like they contradict each other.
        var nRank = (D.score || {})[win] ? D.score[win].n : null;
        if (nRank && nRank !== P.n) {
          pl.innerHTML += " <span style='color:var(--faint)'>Model vs. market needs no ensemble, "
            + "so it runs on all " + P.n + " gradable markets; the three-way ranking above is "
            + "limited to the " + nRank + " that the ensemble also priced.</span>";
        }
      }
    } else {
      if (ci) ci.textContent = "Brier gap";
      if (pl) pl.textContent = "";
    }
    // The KPI number and the interval printed beneath it MUST come from ONE computation. This
    // was s.model - s.market (the three-way scoreboard, 261 markets) while k_gap_ci showed the
    // pooled test's interval (401) — the tile read "+0.026  95% CI [+0.0068,+0.0353]", a point
    // estimate sitting outside its own stated interval. Prefer the pooled gap; fall back to the
    // scoreboard difference only when the pooled block is unavailable.
    var mg = (P && isFinite(P.gap)) ? P.gap : (s.model - s.market);
    if (gapEl) gapEl.textContent = (mg >= 0 ? "+" : "−") + Math.abs(mg).toFixed(3);
    // verdictLine is a neutral static description; the ranked table (leader flagged) shows the standing.
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
    var roiLbl = 'ROI · in-sample <small>(' + (recent ? 'last 60 markets' : 'all bets placed') + ')</small>';
    tb.innerHTML =
      '<tr><td class="rowlbl">' + brierLbl + '</td>'
      + '<td class="num pos">' + b(s.market) + '</td><td class="num">' + b(s.ens) + '</td><td class="num">' + b(s.model) + '</td></tr>'
      + '<tr><td class="rowlbl">' + roiLbl + '</td>'
      + '<td class="num" style="color:var(--faint)">—</td>' + roiCell(re) + roiCell(rm) + '</tr>';
    if (sub) sub.textContent = "the same forecasters, scored two ways · " + (recent ? "last 60" : "all-time");
    if (seg) seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x.getAttribute("data-rwin") === roiWin); });
    if (warn) warn.innerHTML = '<b>ROI is in-sample and noisy.</b> On the same bets, a forecaster with a worse Brier can still show a positive ROI — bet-selection and sizing luck on a small set. It <b>swings between windows and each data refresh</b> (all-time vs recent, run to run). The Brier row above is the ranking; ROI is context, not the scoreboard.';
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
      } else gateCell = '<span class="pill2">not nominated</span>';
      return '<tr><td class="city">' + esc(r.b) + '</td><td class="num">' + r.n + '</td><td class="num ' + (d > 0.0005 ? 'pos' : '') + '">' + r.model.toFixed(3) + '</td><td class="num">' + r.market.toFixed(3) + '</td><td class="num">' + dCell + '</td><td>' + gateCell + '</td></tr>';
    }).join("");
  }
  // Per-city breadth breakdown. Deliberately printed WITH the chance-expectation line: sorting
  // ~50 cities by edge and reading the top is a multiple-comparisons trap, and this table sits
  // beside gates that exist to stop exactly that. See shoulder_book.per_city_stats.
  function renderBkCities() {
    var host = document.getElementById("bkcities"), note = document.getElementById("bkcitynote");
    if (!host) return;
    var rows = D.bk_cities || [];
    if (!rows.length) { host.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:14px 0">No city has enough settled shoulder entries yet.</div>'; if (note) note.innerHTML = ""; return; }
    var sig = rows.filter(function (r) { return r.sig; }).length;
    var exp = (0.05 * rows.length);
    var med = rows.map(function (r) { return r.dates; }).sort(function (a, b) { return a - b; })[Math.floor(rows.length / 2)];
    if (note) note.innerHTML = '<b>Descriptive only — not a city-selection rule.</b> ' + rows.length +
      ' cities with ≥20 settled entries. <b>' + sig + '</b> show a 95% interval clear of zero; <b>~' +
      exp.toFixed(1) + '</b> would do so by chance alone if no city had any edge — so the spread down ' +
      'this column is what noise looks like, not a ranking. The median city spans just <b>' + med +
      ' target days</b>, far short of the 30 a gate requires, which is the other reason no row here ' +
      'is evidence. Any city that looks good would need its own forward gate, declared before the ' +
      'fact, exactly like Leg 1b — which is what <b>legs 1d and 1e above now are</b>: the two ' +
      'obvious cuts of this table, frozen on ' + (D.bk_cs_date || "2026-08-04") + ' and graded ' +
      'only on entries recorded after that date.';
    host.innerHTML = '<table class="data"><tr><th>City</th><th class="num">Settled</th><th class="num">Days</th><th class="num">Win</th>' +
      '<th class="num">Taker <span class="dim">$/contract</span></th><th class="num">95% CI</th></tr>' +
      rows.map(function (r) {
        var col = r.taker < 0 ? "var(--model)" : "var(--good)";
        var ci = '<span style="color:var(--muted)">[' + fmtS(r.lo, 3) + ', ' + fmtS(r.hi, 3) + ']</span>';
        if (r.sig) ci += ' <span class="pill2" style="margin-left:6px">≠0</span>';
        return '<tr><td class="city">' + esc(r.city) + '</td><td class="num">' + r.n + '</td>' +
          '<td class="num" style="color:var(--muted)">' + r.dates + '</td>' +
          '<td class="num">' + (r.wr * 100).toFixed(0) + '%</td>' +
          '<td class="num" style="color:' + col + '">' + fmtS(r.taker, 4) + '</td>' +
          '<td class="num">' + ci + '</td></tr>';
      }).join("") + '</table>';
  }
  function fmtS(v, nd) {
    if (v == null || !isFinite(v)) return "—";
    return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(nd);
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
    ["c_acc", "c_city", "c_calib", "c_disp", "c_equity", "c_bkequity", "c_bkwr"].forEach(function (id) { var h = document.getElementById(id); if (h) h.innerHTML = ""; });
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
    dumbbell("c_city", city.map(function (r) { return { city: (r.city === "HongKong" ? "Hong Kong" : r.city === "NYC" ? "New York" : r.city), market: r.market, model: r.model, ens: r.ens, n: r.n, degenerate: r.degenerate }; }));

    divergingBars("c_calib", D.calib || []);
    dispChart("c_disp", D.disp || []);

    // y-axes below are RETURN ON CAPITAL (%), never summed units: a cumulative unit total is
    // n x mean, so its slope tracks how many markets we collected, not how good the bets were.
    function pct(v) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1) + "%"; }
    function pct2(v) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "%"; }
    function lastOf(s) { var l = s.points.slice(-1)[0]; return l != null ? pct2(l) : ""; }

    var eq = D.equity || [];
    if (eq.length) lineChart("c_equity", {
      h: 210, xLabels: eq.map(function (r) { return r.t; }),
      yFmt: pct, tipFmt: pct2, endFmt: lastOf,
      series: [
        { name: "Leg 1 · all shoulder", color: css("--good"), points: eq.map(function (r) { return r.roi; }), thick: 2.2 },
        { name: "Leg 1b · moderate", color: css("--market"), points: eq.map(function (r) { return r.mroi; }), thick: 1.8 }
      ]
    });
    else { var he = document.getElementById("c_equity"); if (he) he.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">No settled paper entries yet.</div>'; }

    var bkeq = D.bk_equity || [];
    if (bkeq.length) lineChart("c_bkequity", {
      h: 210, xLabels: bkeq.map(function (r) { return r.t; }),
      yFmt: pct, tipFmt: pct2, endFmt: lastOf,
      series: [
        { name: "All shoulder", color: css("--good"), points: bkeq.map(function (r) { return r.roi; }), thick: 2.2 },
        { name: "Moderate [10-25c]", color: css("--market"), points: bkeq.map(function (r) { return r.mroi; }), thick: 1.8 }
      ]
    });
    else { var hb2 = document.getElementById("c_bkequity"); if (hb2) hb2.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">No settled breadth entries yet.</div>'; }

    // Daily win rate against break-even. The gap between the two lines IS the edge; the win rate
    // on its own is unreadable for a payoff that loses 5.6x what it wins.
    var mbe = bkeq.length ? bkeq[0].mbe : null;
    // Scale TIGHTLY around the data. The default axis starts at 0, which pushes an 83-90% line
    // and an 84.9% break-even into the top tenth of the frame — drawn on top of each other, so
    // the gap between them (the entire edge, and the entire point of this chart) is invisible.
    var wrv = bkeq.map(function (r) { return r.mwr; }).filter(function (v) { return v != null; });
    if (mbe != null) wrv = wrv.concat([mbe]);
    var wrLo = wrv.length ? Math.floor(Math.min.apply(null, wrv) - 1.5) : 0;
    var wrHi = wrv.length ? Math.ceil(Math.max.apply(null, wrv) + 1.5) : 100;
    if (bkeq.length && mbe != null) lineChart("c_bkwr", {
      h: 180, xLabels: bkeq.map(function (r) { return r.t; }), yMin: wrLo, yMax: wrHi,
      yFmt: function (v) { return v.toFixed(0) + "%"; }, tipFmt: function (v) { return v.toFixed(1) + "%"; },
      endFmt: function (s) { var l = s.points.slice(-1)[0]; return l != null ? l.toFixed(1) + "%" : ""; },
      series: [
        { name: "Win rate", color: css("--market"), points: bkeq.map(function (r) { return r.mwr; }), thick: 2.2 },
        { name: "Break-even", color: css("--warn"), dash: "4 3", thick: 1.4, points: bkeq.map(function () { return mbe; }) }
      ]
    });
    else { var hw = document.getElementById("c_bkwr"); if (hw) hw.innerHTML = '<div style="color:var(--faint);font-size:12px;padding:20px 0">Not enough settled entries to compute a break-even.</div>'; }

    renderBkCities();
    renderScore();
    renderRoi();
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
      // net-units value stays sign-coloured (green +, red −) — the number shows good/bad, not a title.
      var netEl = document.getElementById("sb_net"); if (netEl) netEl.style.color = _neg(b.BOOK_NET) ? "var(--model)" : "var(--good)";
      var netMod = document.getElementById("sb_net_mod");
      if (netMod && b.BOOK_NET_MOD && b.BOOK_NET_MOD !== "—") netMod.style.color = _neg(b.BOOK_NET_MOD) ? "var(--model)" : "var(--good)";
    }
    // CRPS: green ONLY when the model actually beats the ensemble (lower is better).
    var cv = document.getElementById("crps_v");
    if (cv && b.CRPS_MODEL && b.CRPS_ENS) {
      var cm = parseFloat(b.CRPS_MODEL), ce = parseFloat(b.CRPS_ENS);
      cv.style.color = (isFinite(cm) && isFinite(ce)) ? (cm < ce ? "var(--good)" : "var(--model)") : "";
    }
    var cnote = document.getElementById("crps_note");
    if (cnote && b.CRPS_MODEL && b.CRPS_ENS) {
      var m2 = parseFloat(b.CRPS_MODEL), e2 = parseFloat(b.CRPS_ENS);
      cnote.textContent = (m2 < e2 ? "model sharper" : "ensemble sharper");
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
      if (me) { me.textContent = fn > 0 ? b.SB_MOD_FWD : "—";
        me.style.color = fn > 0 ? (_neg(b.SB_MOD_FWD) ? "var(--model)" : "var(--good)") : ""; }
      if (ms) { ms.textContent = pass ? "gate ✓" : "forward"; ms.className = pass ? "pill2 on" : "pill2"; }
    }
    // Breadth book — all-cities Leg 1b forward gate + shoulder leg, mirrors the 5-city panel.
    if (b.BK_MOD_N != null) {
      var bn = document.getElementById("bkmodn"), be = document.getElementById("bkmodedge"),
          bs = document.getElementById("bkmodstatus");
      var bfn = parseInt(b.BK_MOD_N, 10) || 0, bpass = b.BK_MOD_PASS === "1";
      if (bn) bn.textContent = b.BK_MOD_N + "/" + (b.BK_MOD_NEED || "80");
      if (be) { be.textContent = bfn > 0 ? b.BK_MOD_NET : "—"; be.style.color = _neg(b.BK_MOD_NET) ? "var(--model)" : (bfn > 0 ? "var(--good)" : ""); }
      var bmk = document.getElementById("bkmodmaker"), mkn = parseInt(b.BK_MOD_MAKER_N, 10) || 0;
      if (bmk) { bmk.textContent = mkn > 0 ? b.BK_MOD_MAKER : "—"; bmk.style.color = _neg(b.BK_MOD_MAKER) ? "var(--model)" : (mkn > 0 ? "var(--good)" : ""); }
      if (bs) { bs.textContent = bpass ? "gate ✓" : "forward"; bs.className = bpass ? "pill2 on" : "pill2"; }
      // City-selection gates 1d/1e — same row shape, filled from the same bind set.
      [["BK_CS_A", "bkcsa", "BK_CS_A_CITIES", "bkcsan"],
       ["BK_CS_B", "bkcsb", "BK_CS_B_CITIES", "bkcsbn"]].forEach(function (g) {
        var p = g[0], id = g[1];
        var cn = document.getElementById(g[3]); if (cn) cn.textContent = b[g[2]] || "—";
        var n = document.getElementById(id + "n_n"), e = document.getElementById(id + "edge"),
            mk = document.getElementById(id + "maker"), s = document.getElementById(id + "status");
        var fn = parseInt(b[p + "_N"], 10) || 0, ps = b[p + "_PASS"] === "1";
        var mn = parseInt(b[p + "_MAKER_N"], 10) || 0;
        if (n) n.textContent = (b[p + "_N"] || "0") + "/" + (b[p + "_NEED"] || "80");
        if (e) { e.textContent = fn > 0 ? b[p + "_NET"] : "—";
          e.style.color = fn > 0 ? (_neg(b[p + "_NET"]) ? "var(--model)" : "var(--good)") : ""; }
        if (mk) { mk.textContent = mn > 0 ? b[p + "_MAKER"] : "—";
          mk.style.color = mn > 0 ? (_neg(b[p + "_MAKER"]) ? "var(--model)" : "var(--good)") : ""; }
        if (s) { s.textContent = ps ? "gate ✓" : "forward"; s.className = ps ? "pill2 on" : "pill2"; }
      });
      var miss = document.getElementById("bkcsmiss");
      if (miss) miss.innerHTML = b.BK_CS_MISSING
        ? '<b style="color:var(--model)">Registered set has shrunk — no recent entries for: '
          + b.BK_CS_MISSING + '. This gate no longer measures the set that was frozen.</b>' : "";
      var bfl = document.getElementById("bkfulln"); if (bfl) bfl.textContent = b.BK_FULL_N;
      var bfe = document.getElementById("bkfulledge"); if (bfe) bfe.style.color = _neg(b.BK_FULL_NET) ? "var(--model)" : (b.BK_FULL_NET && b.BK_FULL_NET !== "—" ? "var(--good)" : "");
      var bfm = document.getElementById("bkfullmaker");
      if (bfm) bfm.style.color = _neg(b.BK_FULL_MAKER) ? "var(--model)" : (b.BK_FULL_MAKER && b.BK_FULL_MAKER !== "—" ? "var(--good)" : "");
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
    // Two INDEPENDENT lags: the collector's (measured at build time, server-side) and this
    // page's own publish age (live clock vs generated_at). Adding them together is what made a
    // late rebuild read as "collector stale" on 2026-07-27.
    if (pill && st) {
      var pubH = lastGen ? (Date.now() - Date.parse(lastGen)) / 36e5 : null;
      var colH = (collectLagH === null || collectLagH === undefined) ? null : collectLagH;
      var label = "live", ok = true;
      if (colH !== null && colH >= 5) { label = "collector stale"; ok = false; }
      else if (pubH !== null && pubH >= 5) { label = "dashboard stale"; ok = false; }
      if (colH !== null || pubH !== null) { pill.className = "pill " + (ok ? "" : "warn"); st.textContent = label; }
    }
    var cb = document.getElementById("collectorbig");
    // Build-time lag: live clock - frozen snapshot would add this page's own publish delay
    // and report a 1h-old collector as "4h ago" whenever the rebuild is late.
    if (cb) { if (collectLagH !== null && collectLagH !== undefined) {
        cb.textContent = (collectLagH < 1.5 ? Math.round(collectLagH * 60) + "m" : collectLagH.toFixed(1) + "h") + " behind";
      } else { cb.textContent = "—"; } }   // unknown lag shows nothing, never the browser-clock number
  }
  function apply(P) {
    if (!P || !P.bind) return false;
    applyBind(P.bind, P.html);
    lastGen = P.generated_at || null;
    lastCollect = P.last_collect_iso || (P.series && P.series.last_collect_iso) || null;
    collectLagH = (typeof P.collect_lag_hours === "number") ? P.collect_lag_hours : null;
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
