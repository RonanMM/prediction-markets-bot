# Breadth Structure Book Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the model-free shoulder/favorite structure paper book to every active Polymarket weather city, graded off Polymarket's own settlement, under a separately pre-registered forward gate.

**Architecture:** A new standalone module `shoulder_book_breadth.py` pulls all weather-tag temperature markets live from Gamma, records shoulder-sell / favorite-buy paper entries (deduped, append-only) to `output/shoulder_paper_breadth.csv`, and grades them by fetching each market's resolved outcome (`GET /markets/{id}`). It reuses `shoulder_book`'s bands and fee model unchanged. The only edit to existing code is an additive `prereg_date` kwarg on `shoulder_book.moderate_gate_stats`.

**Tech Stack:** Python 3, pandas, `http_util.get_json` (requests-based; Gamma-safe UA), existing `shoulder_book` / `fetch_polymarket` helpers.

## Global Constraints

- Run tests from repo root: `pytest -o addopts="" tests/ -v`.
- HTTP goes through `http_util.get_json(url, params, label)` — never raw urllib (its default UA gets 403 from Gamma).
- Reuse from `shoulder_book`: `BAND_LO, BAND_HI, CORE_LO, FAV_LO, FAV_HI, FAV_CORE_HI, FAV_MIN_HOURS_TO_END, _net_edge, moderate_gate_stats, MOD_LO, MOD_HI`. Reuse from `fetch_polymarket`: `parse_question`, `parse_question_date` (import via the same paths `shoulder_book` uses).
- All functions that fetch or use "now" take an injectable argument (`fetch=`, `now_utc=`, `bins=`) defaulting to the live/real value, so tests run offline.
- No change to `shoulder_book.py` behavior except the additive kwarg; the 5-city report must stay identical.
- Directly on master (approved). The module is additive and inert until `main.py` calls it.

---

### Task 1: Additive `prereg_date` kwarg on `moderate_gate_stats`

**Files:**
- Modify: `src/polymarket_weather/shoulder_book.py` (`moderate_gate_stats`, ~line 225)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces: `moderate_gate_stats(graded: pd.DataFrame, prereg_date: str = MOD_PREREG_DATE) -> dict` — unchanged output shape `{"context": {...}, "forward": {...}}`; `forward` counts entries with `entered_at_utc >= prereg_date`.

- [ ] **Step 1: Write the failing test**

```python
def test_moderate_gate_prereg_date_kwarg():
    import shoulder_book as sb
    import pandas as pd
    # two in-band [0.10,0.25) entries, one before and one on a custom prereg date
    graded = pd.DataFrame([
        {"entry_yes_price": 0.15, "entered_at_utc": "2026-07-20T00:00:00+00:00",
         "side_won": True,  "entry_side_price": 0.85},
        {"entry_yes_price": 0.15, "entered_at_utc": "2026-08-01T00:00:00+00:00",
         "side_won": False, "entry_side_price": 0.85},
    ])
    # default date (2026-07-23): forward = the Aug-01 row only
    assert sb.moderate_gate_stats(graded)["forward"]["n"] == 1
    # custom later date: forward = only the Aug-01 row still (>= 2026-08-01)
    assert sb.moderate_gate_stats(graded, prereg_date="2026-08-01")["forward"]["n"] == 1
    # custom earlier date: both rows are forward
    assert sb.moderate_gate_stats(graded, prereg_date="2026-07-01")["forward"]["n"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py::test_moderate_gate_prereg_date_kwarg -v`
Expected: FAIL (`moderate_gate_stats() got an unexpected keyword argument 'prereg_date'`).

- [ ] **Step 3: Implement**

In `shoulder_book.py`, change the signature and the one `prereg` line:

```python
def moderate_gate_stats(graded: pd.DataFrame, prereg_date: str = MOD_PREREG_DATE) -> dict:
    ...
    prereg = pd.Timestamp(prereg_date, tz="UTC")
```

(Everything else in the function is unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py::test_moderate_gate_prereg_date_kwarg -v`
Expected: PASS. Also run the existing `test_moderate_gate_stats` — still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/shoulder_book.py tests/test_polymarket_weather.py
git commit -m "shoulder_book: additive prereg_date kwarg on moderate_gate_stats"
```

---

### Task 2: `shoulder_book_breadth.py` skeleton — constants + title/bin parsing

**Files:**
- Create: `src/polymarket_weather/shoulder_book_breadth.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces:
  - `BREADTH_PREREG_DATE = "2026-07-23"`, `GATE_MOD_BREADTH = (80, 0.03)`
  - `parse_event_title(title: str) -> tuple[str,str,str] | None` → `(kind, city, date_str)` where `kind ∈ {"max","min"}`
  - `bins_from_event(event: dict) -> list[dict]` → per-bin dicts `{condition_id, market_id, city, kind, date_str, question, yes, liquidity, end}` (`end` = pandas UTC Timestamp; `yes`/`liquidity` floats). Skips bins with unparseable price.

- [ ] **Step 1: Write the failing test**

```python
def test_parse_event_title_and_bins():
    import shoulder_book_breadth as b
    assert b.parse_event_title("Highest temperature in Paris on July 23?") == ("max", "Paris", "July 23")
    assert b.parse_event_title("Lowest temperature in New York City on July 3?") == ("min", "New York City", "July 3")
    assert b.parse_event_title("Who wins the election?") is None

    ev = {"title": "Highest temperature in Paris on July 23?", "endDate": "2026-07-23T22:00:00Z",
          "markets": [
              {"conditionId": "0xAAA", "id": "111", "question": "Highest temperature in Paris on July 23 (30-31°C)?",
               "groupItemTitle": "30-31", "outcomePrices": "[\"0.18\", \"0.82\"]", "liquidityNum": 5000},
              {"conditionId": "0xBBB", "id": "222", "question": "…(bad)…",
               "groupItemTitle": "x", "outcomePrices": "not-json", "liquidityNum": 10},
          ]}
    bins = b.bins_from_event(ev)
    assert len(bins) == 1                      # bad-price bin skipped
    r = bins[0]
    assert r["city"] == "Paris" and r["kind"] == "max"
    assert r["condition_id"] == "0xAAA" and r["market_id"] == "111"
    assert abs(r["yes"] - 0.18) < 1e-9 and abs(r["liquidity"] - 5000) < 1e-9
    assert str(r["end"].tz) == "UTC"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement**

```python
"""shoulder_book_breadth.py — the model-free structure paper book (shoulder sell + favorite
buy), extended to EVERY active Polymarket weather city.

Discovery is live from Gamma (weather tag). Grading is against Polymarket's OWN settlement
(the resolved market's terminal pinned outcome via /markets/{id}) — for a model-free
structure trade the market's settlement IS the P&L ground truth, so no weather-truth feed
or forecast model is needed. Entries are append-only + deduped on (condition_id, leg) in
output/shoulder_paper_breadth.csv. Bands and the taker-fee model are imported unchanged
from shoulder_book; the moderate [10,25)¢ gate is separately PRE-REGISTERED
(BREADTH_PREREG_DATE) so the 5-city stream is never contaminated. No real orders, no edge
claim — a clean forward measurement across the full city universe.
"""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import config
from http_util import get_json
from fetch_polymarket import parse_question, parse_question_date
from shoulder_book import (BAND_LO, BAND_HI, CORE_LO, FAV_LO, FAV_HI, FAV_CORE_HI,
                           FAV_MIN_HOURS_TO_END, _net_edge, moderate_gate_stats, MOD_LO, MOD_HI)

GAMMA = "https://gamma-api.polymarket.com"
BREADTH_PREREG_DATE = "2026-07-23"
GATE_MOD_BREADTH    = (80, 0.03)
PREDAY_HOURS        = 24          # tz-free "pre-day": > this many hours to market end
_PIN                = 0.03        # terminal price within this of 0/1 counts as settled

_OUT = Path(__file__).resolve().parent.parent.parent / "output" / "shoulder_paper_breadth.csv"
_BCOLS = ["entered_at_utc", "city", "condition_id", "market_id", "question", "target_date",
          "entry_yes_price", "liquidity", "leg", "side", "entry_side_price", "band",
          "settled_outcome"]

_TITLE = re.compile(r"^(Highest|Lowest) temperature in (.+?) on (.+?)\??$")

def parse_event_title(title: str):
    m = _TITLE.match((title or "").strip())
    if not m:
        return None
    return ("max" if m.group(1) == "Highest" else "min"), m.group(2).strip(), m.group(3).strip()

def _yes_of(mk: dict):
    try:
        op = json.loads(mk.get("outcomePrices", "[]"))
        return float(op[0]) if op else None
    except Exception:
        return None

def bins_from_event(event: dict) -> list[dict]:
    p = parse_event_title(event.get("title", ""))
    if not p:
        return []
    kind, city, date_str = p
    end = pd.to_datetime(event.get("endDate"), utc=True, errors="coerce")
    out = []
    for mk in event.get("markets", []):
        yes = _yes_of(mk)
        if yes is None:
            continue
        try:
            liq = float(mk.get("liquidityNum") or mk.get("liquidity") or 0.0)
        except Exception:
            liq = 0.0
        out.append(dict(condition_id=mk.get("conditionId"), market_id=str(mk.get("id")),
                        city=city, kind=kind, date_str=date_str,
                        question=mk.get("question") or event.get("title", ""),
                        yes=yes, liquidity=liq, end=end))
    return out
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polymarket_weather/shoulder_book_breadth.py tests/test_polymarket_weather.py
git commit -m "breadth book: module skeleton — title/bin parsing + constants"
```

---

### Task 3: Live discovery — `fetch_weather_bins()`

**Files:**
- Modify: `src/polymarket_weather/shoulder_book_breadth.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `bins_from_event` (Task 2).
- Produces: `fetch_weather_events(fetch=get_json) -> list[dict]` (paged weather-tag temperature events); `fetch_weather_bins(fetch=get_json) -> list[dict]` (flattened bins).

- [ ] **Step 1: Write the failing test** (inject a fake pager — no network)

```python
def test_fetch_weather_bins_injected():
    import shoulder_book_breadth as b
    page = [{"title": "Highest temperature in Paris on July 23?", "endDate": "2026-07-23T22:00:00Z",
             "markets": [{"conditionId": "0xAAA", "id": "1", "question": "q",
                          "outcomePrices": "[\"0.2\",\"0.8\"]", "liquidityNum": 100}]},
            {"title": "Not a temperature market", "markets": []}]
    calls = {"n": 0}
    def fake(url, params, label="API"):
        calls["n"] += 1
        return page if params.get("offset", 0) == 0 else []   # one page then empty
    bins = b.fetch_weather_bins(fetch=fake)
    assert len(bins) == 1 and bins[0]["city"] == "Paris"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`fetch_weather_bins` undefined).

- [ ] **Step 3: Implement**

```python
def fetch_weather_events(fetch=get_json) -> list[dict]:
    out, off = [], 0
    while True:
        page = fetch(f"{GAMMA}/events",
                     {"tag_slug": "weather", "active": "true", "closed": "false",
                      "limit": 100, "offset": off}, "Gamma")
        page = page if isinstance(page, list) else (page or {}).get("data", [])
        if not page:
            break
        out += [e for e in page if parse_event_title(e.get("title", ""))]
        if len(page) < 100:
            break
        off += 100
        if off > 3000:      # safety cap
            break
    return out

def fetch_weather_bins(fetch=get_json) -> list[dict]:
    bins = []
    for ev in fetch_weather_events(fetch=fetch):
        bins.extend(bins_from_event(ev))
    return bins
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "breadth book: live weather-event discovery (paged, injectable)"`

---

### Task 4: `scan_and_record_breadth()` — recording, bands, dedup, endDate day logic

**Files:**
- Modify: `src/polymarket_weather/shoulder_book_breadth.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces: `scan_and_record_breadth(bins=None, now_utc=None, out_path=_OUT, fetch=get_json) -> int` — appends new deduped entries, returns count added. `_load_book(path)` / helper returns a DataFrame with `_BCOLS`.

- [ ] **Step 1: Write the failing test** (offline; a tmp CSV)

```python
def test_scan_and_record_breadth(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd
    from datetime import datetime, timezone
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    end_next_day = pd.Timestamp("2026-07-23T22:00:00Z")   # ~34h out -> pre-day
    end_soon     = pd.Timestamp("2026-07-22T20:00:00Z")   # ~8h out  -> NOT pre-day
    def mk(cid, yes, end, q="Highest temperature in Paris on July 23 (30-31°C)?"):
        return dict(condition_id=cid, market_id=cid[2:], city="Paris", kind="max",
                    date_str="July 23", question=q, yes=yes, liquidity=5000, end=end)
    bins = [
        mk("0xSH", 0.15, end_next_day),   # shoulder [5,35), pre-day -> recorded (No)
        mk("0xFV", 0.70, end_next_day),   # favorite [65,85), >12h  -> recorded (Yes)
        mk("0xLATE", 0.15, end_soon),     # shoulder band but <24h  -> NOT recorded as shoulder
        mk("0xMID", 0.50, end_next_day),  # neither band            -> nothing
    ]
    out = tmp_path / "breadth.csv"
    n = b.scan_and_record_breadth(bins=bins, now_utc=now, out_path=out)
    df = pd.read_csv(out)
    legs = set(zip(df["condition_id"], df["leg"]))
    assert ("0xSH", "shoulder") in legs
    assert ("0xFV", "favorite") in legs
    assert ("0xLATE", "shoulder") not in legs
    assert df[df["condition_id"] == "0xMID"].empty
    # dedup: running again adds nothing
    assert b.scan_and_record_breadth(bins=bins, now_utc=now, out_path=out) == 0
    # recorded sides/prices
    sh = df[df["condition_id"] == "0xSH"].iloc[0]
    assert sh["side"] == "No" and abs(sh["entry_side_price"] - 0.85) < 1e-6
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`scan_and_record_breadth` undefined).

- [ ] **Step 3: Implement**

```python
def _load_book(path=_OUT) -> pd.DataFrame:
    if Path(path).exists():
        df = pd.read_csv(path)
        for c in _BCOLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    return pd.DataFrame(columns=_BCOLS)

def scan_and_record_breadth(bins=None, now_utc=None, out_path=_OUT, fetch=get_json) -> int:
    if bins is None:
        bins = fetch_weather_bins(fetch=fetch)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    book = _load_book(out_path)
    known = set(zip(book["condition_id"], book["leg"]))
    added = []
    for r in bins:
        end = r["end"]
        if pd.isna(end):
            continue
        yes = float(r["yes"])
        hours_to_end = (pd.Timestamp(end) - pd.Timestamp(now_utc)).total_seconds() / 3600.0
        tgt = parse_question_date(r["question"], pd.Timestamp(end).tz_localize(None)) \
              or pd.Timestamp(end).date()
        base = {"entered_at_utc": now_utc.isoformat(), "city": r["city"],
                "condition_id": r["condition_id"], "market_id": r["market_id"],
                "question": r["question"], "target_date": str(tgt),
                "entry_yes_price": round(yes, 4), "liquidity": round(float(r["liquidity"]), 2),
                "settled_outcome": ""}
        cid = r["condition_id"]
        if (cid, "shoulder") not in known and hours_to_end > PREDAY_HOURS and BAND_LO <= yes < BAND_HI:
            added.append({**base, "leg": "shoulder", "side": "No",
                          "entry_side_price": round(1.0 - yes, 4),
                          "band": "core" if yes >= CORE_LO else "outer"})
            known.add((cid, "shoulder"))
        if (cid, "favorite") not in known and hours_to_end > FAV_MIN_HOURS_TO_END and FAV_LO <= yes < FAV_HI:
            added.append({**base, "leg": "favorite", "side": "Yes",
                          "entry_side_price": round(yes, 4),
                          "band": "fav_core" if yes < FAV_CORE_HI else "fav_outer"})
            known.add((cid, "favorite"))
    if added:
        book = pd.concat([book, pd.DataFrame(added)], ignore_index=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        book.reindex(columns=_BCOLS).to_csv(out_path, index=False)
    return len(added)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "breadth book: scan_and_record_breadth (bands, dedup, endDate day logic)"`

---

### Task 5: `settlement_outcome()` + `grade_book()` — settlement grading, frozen once set

**Files:**
- Modify: `src/polymarket_weather/shoulder_book_breadth.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces:
  - `settlement_outcome(market_id, fetch=get_json) -> int | None` (1 YES-won / 0 NO-won / None unsettled)
  - `grade_book(book=None, out_path=_OUT, fetch=get_json) -> pd.DataFrame` — fills+freezes `settled_outcome`, persists, returns graded frame with `side_won`, `net_edge`.

- [ ] **Step 1: Write the failing test**

```python
def test_settlement_outcome_and_freeze(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd
    def fake_market(state):
        return {"111": {"closed": True,  "outcomePrices": "[\"1\",\"0\"]"},   # YES won
                "222": {"closed": True,  "outcomePrices": "[\"0\",\"1\"]"},   # NO won
                "333": {"closed": False, "outcomePrices": "[\"0.5\",\"0.5\"]"}}[state]
    def fetch(url, params=None, label="API"):
        mid = url.rstrip("/").split("/")[-1]
        return fake_market(mid)
    assert b.settlement_outcome("111", fetch=fetch) == 1
    assert b.settlement_outcome("222", fetch=fetch) == 0
    assert b.settlement_outcome("333", fetch=fetch) is None

    # grade_book fills settled_outcome once and freezes it
    book = pd.DataFrame([
        {**{c: "" for c in b._BCOLS}, "condition_id": "0xA", "market_id": "111",
         "leg": "shoulder", "side": "No", "entry_side_price": 0.85,
         "entry_yes_price": 0.15, "entered_at_utc": "2026-07-24T00:00:00+00:00"},
    ])
    out = tmp_path / "breadth.csv"
    book.reindex(columns=b._BCOLS).to_csv(out, index=False)
    g = b.grade_book(out_path=out, fetch=fetch)
    assert int(g.iloc[0]["settled_outcome"]) == 1
    # side "No" with YES-won => side lost
    assert bool(g.iloc[0]["side_won"]) is False
    # freeze: a fetch that would now say 0 must NOT overwrite the persisted 1
    def fetch2(url, params=None, label="API"):
        return {"closed": True, "outcomePrices": "[\"0\",\"1\"]"}
    g2 = b.grade_book(out_path=out, fetch=fetch2)
    assert int(g2.iloc[0]["settled_outcome"]) == 1
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (undefined).

- [ ] **Step 3: Implement**

```python
def _market_dict(resp):
    if isinstance(resp, list):
        return resp[0] if resp else None
    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        return resp["data"][0] if resp["data"] else None
    return resp if isinstance(resp, dict) else None

def settlement_outcome(market_id, fetch=get_json):
    if not market_id or str(market_id) in ("", "nan", "None"):
        return None
    resp = fetch(f"{GAMMA}/markets/{market_id}", None, "Gamma")
    mk = _market_dict(resp)
    if not mk or not mk.get("closed"):
        return None
    try:
        prices = [float(x) for x in json.loads(mk.get("outcomePrices", "[]"))]
    except Exception:
        return None
    if not prices or max(prices) < 1.0 - _PIN:
        return None
    return 1 if prices[0] >= 1.0 - _PIN else 0

def _is_unset(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan", "None")

def grade_book(book=None, out_path=_OUT, fetch=get_json) -> pd.DataFrame:
    if book is None:
        book = _load_book(out_path)
    if book.empty:
        return book
    changed = False
    for i, r in book.iterrows():
        if _is_unset(r.get("settled_outcome")):
            o = settlement_outcome(r.get("market_id"), fetch=fetch)
            if o is not None:
                book.at[i, "settled_outcome"] = o
                changed = True
    if changed:
        book.reindex(columns=_BCOLS).to_csv(out_path, index=False)
    graded = book[book["settled_outcome"].map(lambda v: not _is_unset(v))].copy()
    if graded.empty:
        return graded
    graded["settled_outcome"] = graded["settled_outcome"].astype(float).astype(int)
    graded["side_won"] = (graded["settled_outcome"] == 1) == (graded["side"] == "Yes")
    graded["net_edge"] = _net_edge(graded["side_won"], graded["entry_side_price"].astype(float))
    return graded
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "breadth book: settlement grading via /markets/{id}, frozen once observed"`

---

### Task 6: `report_breadth()` + CLI

**Files:**
- Modify: `src/polymarket_weather/shoulder_book_breadth.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces: `report_breadth(out_path=_OUT, fetch=get_json) -> None` (prints legs + Leg 1b breadth gate via `moderate_gate_stats(graded, prereg_date=BREADTH_PREREG_DATE)`); `__main__` with `--record` (scan) / default (report).

- [ ] **Step 1: Write the failing test** (report runs and computes the breadth gate forward-only, offline)

```python
def test_report_breadth_gate_forward_only(tmp_path, capsys):
    import shoulder_book_breadth as b
    import pandas as pd
    # two in-band [10,25) shoulder entries already settled; one before, one after prereg
    rows = []
    for cid, entered, won in [("0xA", "2026-07-01T00:00:00+00:00", 1),
                              ("0xB", "2026-07-25T00:00:00+00:00", 1)]:
        rows.append({**{c: "" for c in b._BCOLS}, "condition_id": cid, "market_id": cid,
                     "leg": "shoulder", "side": "No", "entry_yes_price": 0.15,
                     "entry_side_price": 0.85, "entered_at_utc": entered, "settled_outcome": won})
    out = tmp_path / "breadth.csv"
    pd.DataFrame(rows).reindex(columns=b._BCOLS).to_csv(out, index=False)
    # no-op fetch (already settled)
    b.report_breadth(out_path=out, fetch=lambda *a, **k: None)
    text = capsys.readouterr().out
    assert "BREADTH" in text.upper()
    # forward gate counts ONLY the post-2026-07-23 entry
    graded = b.grade_book(out_path=out, fetch=lambda *a, **k: None)
    stats = b.moderate_gate_stats(graded, prereg_date=b.BREADTH_PREREG_DATE)
    assert stats["forward"]["n"] == 1
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`report_breadth` undefined).

- [ ] **Step 3: Implement**

```python
def _leg_line(graded, mask, label):
    sub = graded[mask]
    if sub.empty:
        print(f"  {label}: 0 graded")
        return
    print(f"  {label}: n={len(sub)}  win={sub['side_won'].mean():.1%}  "
          f"net taker={sub['net_edge'].mean():+.4f}/share")

def report_breadth(out_path=_OUT, fetch=get_json) -> None:
    book = _load_book(out_path)
    if book.empty:
        print("BREADTH structure book: no paper entries yet.")
        return
    graded = grade_book(book=book, out_path=out_path, fetch=fetch)
    ncities = book["city"].nunique()
    print(f"BREADTH STRUCTURE PAPER BOOK: {len(book)} entries across {ncities} cities "
          f"({len(graded)} graded, {len(book) - len(graded)} awaiting settlement)")
    if graded.empty:
        return
    yes = graded["entry_yes_price"].astype(float)
    sh = graded["leg"] == "shoulder"
    _leg_line(graded, sh, "Leg1 shoulder [5,35)")
    _leg_line(graded, sh & (yes >= CORE_LO), "Leg1 core   [20,35)")
    _leg_line(graded, graded["leg"] == "favorite", "Leg2 favorite [65,85)")
    st = moderate_gate_stats(graded, prereg_date=BREADTH_PREREG_DATE)
    if st:
        c, f = st["context"], st["forward"]
        need_n, need_e = GATE_MOD_BREADTH
        mark = "PASS" if f.get("gate_pass") else "pending"
        print(f"  Leg1b moderate [10,25) — pre-registered {BREADTH_PREREG_DATE}")
        print(f"    context (all graded):  n={c['n']}  wr {c['wr']:.1%}  taker {c['taker']:+.4f}")
        print(f"    FORWARD gate:          n={f['n']}/{need_n}  taker {f['taker']:+.4f} "
              f"v +{need_e:.3f}  [{mark}]")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="scan live markets and record entries")
    a = ap.parse_args()
    if a.record:
        print(f"recorded {scan_and_record_breadth()} new breadth entries")
    else:
        report_breadth()
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS. Then run the FULL suite: `pytest -o addopts="" tests/ -v` — all green.
- [ ] **Step 5: Commit** — `git commit -m "breadth book: report_breadth + CLI (legs + pre-registered Leg1b gate)"`

---

### Task 7: Wire into `main.py` collector hook + live smoke test

**Files:**
- Modify: `src/polymarket_weather/main.py` (~line 161, right after the shoulder hook)

**Interfaces:**
- Consumes: `scan_and_record_breadth` (Task 4).

- [ ] **Step 1: Add the hook** (after the existing `shoulder_book` try/except at main.py:153-161)

```python
    # Breadth structure book (all Polymarket weather cities) — model-free, settlement-graded.
    try:
        from shoulder_book_breadth import scan_and_record_breadth
        nb = scan_and_record_breadth()
        if nb:
            logger.info("Breadth structure book: recorded %d new entries.", nb)
    except Exception as e:
        logger.warning("Breadth book scan failed: %s", e)
```

- [ ] **Step 2: Live smoke test** (real network, records into the real CSV)

Run from `src/polymarket_weather/`:
```bash
python shoulder_book_breadth.py --record && python shoulder_book_breadth.py
```
Expected: records N>0 entries across many cities, then the report prints the leg lines and a `Leg1b … FORWARD gate: n=…/80 … [pending]` (forward count small — today's entries only).

- [ ] **Step 3: Verify the CSV**

```bash
python -c "import pandas as pd; d=pd.read_csv('../../output/shoulder_paper_breadth.csv'); print(len(d),'rows,',d['city'].nunique(),'cities'); print(d['leg'].value_counts().to_dict())"
```
Expected: rows across many untracked cities, legs ∈ {shoulder, favorite}.

- [ ] **Step 4: Commit** (module + first data)

```bash
git add src/polymarket_weather/main.py output/shoulder_paper_breadth.csv
git commit -m "collector: record the breadth structure book each cycle (main.py hook) + first snapshot"
```

---

## Self-Review

- **Spec coverage:** discovery (T2–3), recording+bands+dedup+endDate logic (T4), settlement grading frozen (T5), separate pre-registered gate (T1 kwarg + T6 wiring), report (T6), cloud wiring via main.py hook (T7), tests for each (T1–6). All spec §3 items covered.
- **Placeholders:** none — every step has runnable code/commands.
- **Type consistency:** bin dict keys (`condition_id/market_id/city/kind/date_str/question/yes/liquidity/end`) are produced in T2 and consumed unchanged in T3–4; `_BCOLS` defined in T2 and used in T4–6; `moderate_gate_stats(prereg_date=)` defined T1, used T6; `settlement_outcome`/`grade_book` signatures consistent T5→T6.
