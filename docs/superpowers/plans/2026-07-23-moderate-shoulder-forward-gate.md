# Moderate-Shoulder Forward Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-registered, forward-only "moderate shoulder" gate (Leg 1b) to `shoulder_book.py` — a report-time refinement that sells only the over-priced YES `[0.10, 0.25)` band.

**Architecture:** Purely additive. No change to recording (`scan_and_record`) or the CSV schema. A new pure function `moderate_gate_stats(graded)` filters the existing graded shoulder entries to the moderate band and splits them into *context* (all) and *forward* (entered on/after the pre-registration date) — the forward slice is the pre-registered taker gate. `report()` prints the result; the module docstring records the pre-registration.

**Tech Stack:** Python 3.11, pandas, pytest.

## Global Constraints

- **No schema/recording change.** Do NOT touch `scan_and_record`, `_COLS`, or the CSV. Copy verbatim: band `MOD_LO, MOD_HI = 0.10, 0.25`; forward clock `MOD_PREREG_DATE = "2026-07-23"`; gate `GATE_MOD = (80, 0.03)` (min graded FORWARD entries, min mean net **taker** $/share).
- **Forward-only counting is the honesty core:** the gate counts an entry only if `entered_at_utc >= MOD_PREREG_DATE` (UTC). Pre-reg entries are context, never gate.
- **Taker-gated; maker reported, not gated.** No real orders, no edge claim — this only starts a clean forward measurement.
- **Run tests from the repository root:** `pytest -o addopts="" tests/ -v`.
- **Every commit message ends with:**
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 1: Constants, pre-registration docstring, and the `moderate_gate_stats` function

**Files:**
- Modify: `src/polymarket_weather/shoulder_book.py` (docstring ~line 28; constants after line 77; new function before `def report()`)
- Test: `tests/test_polymarket_weather.py` (append a new test)

**Interfaces:**
- Produces: `MOD_LO, MOD_HI = 0.10, 0.25`; `MOD_PREREG_DATE = "2026-07-23"`; `GATE_MOD = (80, 0.03)`; and
  `moderate_gate_stats(graded: pd.DataFrame) -> dict`. Returns `{}` if `graded` is empty or missing any of
  `{entry_yes_price, entered_at_utc, side_won, entry_side_price}`; otherwise
  `{"context": {n, wr, taker, maker_n, maker}, "forward": {n, wr, taker, maker_n, maker, gate_pass}}`.
- Consumes: existing module-level `_net_edge(side_won, side_price)`.

- [ ] **Step 1: Add the pre-registration line to the module docstring**

Edit `src/polymarket_weather/shoulder_book.py`. Find:

```
  (Leg 2 outer [75,85)¢ is recorded and reported but carries no gate.)
```

Replace with:

```
  (Leg 2 outer [75,85)¢ is recorded and reported but carries no gate.)
  Leg 1b moderate yes∈[10,25)¢: ≥80 graded FORWARD entries AND mean net taker ≥ +3¢/share
    (§10f, pre-reg 2026-07-23) — a report-time refinement of Leg 1. Discovered in-sample
    (n=109): the full band nets ~0 because it lumps the fat-tail-UNDER-priced deep band
    [5,10)¢ with the OVER-priced moderate band. Sells only the moderate band; gate counts
    only entries recorded on/after the pre-reg date, so it is never graded on the discovery
    data. Maker reported, not gated.
```

- [ ] **Step 2: Add the constants**

Edit `src/polymarket_weather/shoulder_book.py`. Find:

```python
GATE_CORE_MAKER = (80, 0.03)   # (min FILLED core entries, min mean maker net $/share)

_SNAP_WINDOW_MIN = 60             # a "run" = rows within this window of the newest row
```

Replace with:

```python
GATE_CORE_MAKER = (80, 0.03)   # (min FILLED core entries, min mean maker net $/share)

# Leg 1b (moderate shoulder) — pre-registered 2026-07-23. A REPORT-TIME refinement of Leg 1,
# not a new recorded leg. Sell only the OVER-priced moderate band; the deep band [0.05,0.10) is
# fat-tail-UNDER-priced (excluded) and the core [0.25,0.35) is fair (excluded). Discovered
# in-sample (n=109, where the full band nets ~0 by lumping the two). The gate counts FORWARD
# entries only (entered_at_utc >= MOD_PREREG_DATE), so the hypothesis is never graded on the data
# that generated it. Taker-gated; maker reported, not gated. Threshold shape mirrors GATE_CORE
# (chosen by analogy, NOT tuned to the observed ~+3¢).
MOD_LO, MOD_HI  = 0.10, 0.25
MOD_PREREG_DATE = "2026-07-23"     # forward clock (UTC): gate counts entries entered_at_utc >= this
GATE_MOD        = (80, 0.03)       # (min graded FORWARD entries, min mean net taker $/share)

_SNAP_WINDOW_MIN = 60             # a "run" = rows within this window of the newest row
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_polymarket_weather.py`:

```python
def test_moderate_gate_stats():
    """Leg 1b: report-time moderate-shoulder [0.10,0.25) gate, forward-only (entered >=
    MOD_PREREG_DATE). Verifies band filter, forward-date filter, and taker gate math."""
    import pandas as pd
    from shoulder_book import moderate_gate_stats, MOD_LO, MOD_HI, MOD_PREREG_DATE, GATE_MOD
    assert (MOD_LO, MOD_HI) == (0.10, 0.25)
    assert MOD_PREREG_DATE == "2026-07-23"
    need_n, need_e = GATE_MOD

    def rows(n, yes, won, entered):
        # a shoulder SELL: side='No', entry_side_price = 1 - yes
        return pd.DataFrame({
            "entry_yes_price": [yes] * n,
            "entered_at_utc": [entered] * n,
            "side_won": [won] * n,
            "entry_side_price": [round(1.0 - yes, 4)] * n,
        })

    PRE = "2026-01-01T00:00:00+00:00"    # before pre-reg
    POST = "2026-12-31T00:00:00+00:00"   # on/after pre-reg

    # band + forward filters: deep (0.07) and core (0.30) excluded; PRE entries context-only
    df = pd.concat([
        rows(3, 0.15, True, POST),   # in band, forward
        rows(2, 0.07, True, POST),   # deep band -> excluded
        rows(4, 0.30, True, POST),   # core band -> excluded
        rows(5, 0.15, True, PRE),    # in band but pre-reg -> context only
    ], ignore_index=True)
    s = moderate_gate_stats(df)
    assert s["context"]["n"] == 8        # in-band: 3 forward + 5 pre
    assert s["forward"]["n"] == 3        # in-band AND forward
    assert s["forward"]["gate_pass"] is False   # n=3 < 80

    # gate PASSES: >=80 forward in-band winners (taker per win >> +0.03)
    sp = moderate_gate_stats(rows(need_n, 0.12, True, POST))
    assert sp["forward"]["n"] == need_n
    assert sp["forward"]["taker"] >= need_e
    assert sp["forward"]["gate_pass"] is True

    # gate FAILS on edge: >=80 forward in-band losers (taker very negative)
    sl = moderate_gate_stats(rows(need_n, 0.12, False, POST))
    assert sl["forward"]["n"] == need_n
    assert sl["forward"]["gate_pass"] is False

    # empty / missing columns -> {}
    assert moderate_gate_stats(pd.DataFrame()) == {}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py::test_moderate_gate_stats -v`
Expected: FAIL — `ImportError: cannot import name 'moderate_gate_stats'`.

- [ ] **Step 5: Implement `moderate_gate_stats`**

Edit `src/polymarket_weather/shoulder_book.py`. Find the line `def report() -> None:` and insert this function immediately BEFORE it:

```python
def moderate_gate_stats(graded: pd.DataFrame) -> dict:
    """Leg 1b — the moderate-shoulder [MOD_LO, MOD_HI) refinement of Leg 1, computed at REPORT
    TIME from existing shoulder entries (no new recording). Returns 'context' (all graded
    in-band, incl. the in-sample discovery sample) and 'forward' (entries entered on/after
    MOD_PREREG_DATE only — the pre-registered, forward-only gate). Taker-gated; maker reported,
    not gated. Returns {} if graded is empty or missing a required column.

    graded needs: entry_yes_price, entered_at_utc, side_won, entry_side_price (optional: maker_filled).
    """
    need = {"entry_yes_price", "entered_at_utc", "side_won", "entry_side_price"}
    if graded.empty or not need.issubset(graded.columns):
        return {}
    yes = graded["entry_yes_price"].astype(float)
    inband = graded[(yes >= MOD_LO) & (yes < MOD_HI)].copy()
    inband["_taker"] = _net_edge(inband["side_won"], inband["entry_side_price"].astype(float))
    entered = pd.to_datetime(inband["entered_at_utc"], utc=True, errors="coerce")
    prereg = pd.Timestamp(MOD_PREREG_DATE, tz="UTC")

    def _agg(sub: pd.DataFrame) -> dict:
        d = {"n": int(len(sub)),
             "wr": float(sub["side_won"].mean()) if len(sub) else 0.0,
             "taker": float(sub["_taker"].mean()) if len(sub) else 0.0,
             "maker_n": 0, "maker": 0.0}
        if "maker_filled" in sub.columns and len(sub):
            f = sub[sub["maker_filled"].astype(bool)]
            d["maker_n"] = int(len(f))
            if len(f):
                d["maker"] = float((f["side_won"].astype(float)
                                    - f["entry_side_price"].astype(float)).mean())
        return d

    need_n, need_e = GATE_MOD
    forward = _agg(inband[entered >= prereg])
    forward["gate_pass"] = bool(forward["n"] >= need_n and forward["taker"] >= need_e)
    return {"context": _agg(inband), "forward": forward}


```

(Keep the two blank lines before `def report()`.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest -o addopts="" tests/test_polymarket_weather.py::test_moderate_gate_stats -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite (nothing else broke)**

Run: `pytest -o addopts="" tests/ -v`
Expected: all pass (previous count + 1).

- [ ] **Step 8: Commit**

```bash
git add src/polymarket_weather/shoulder_book.py tests/test_polymarket_weather.py
git commit -m "feat: pre-register Leg 1b moderate-shoulder forward gate (constants + stats fn)

Report-time refinement of shoulder_book Leg 1: sell only the over-priced moderate band
YES [0.10,0.25), excluding the fat-tail-under-priced deep band and the fair core.
moderate_gate_stats splits graded shoulder entries into context (all) and forward
(entered >= 2026-07-23) — the forward taker gate (80 @ +0.03). Pre-registration recorded
in the docstring. Unit-tested (band + forward-date + gate math).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wire `moderate_gate_stats` into the report

**Files:**
- Modify: `src/polymarket_weather/shoulder_book.py` (inside `report()`, after the Leg 1 core line ~267)

**Interfaces:**
- Consumes: `moderate_gate_stats`, `GATE_MOD`, `MOD_PREREG_DATE` from Task 1; the existing `graded` DataFrame in `report()` (has columns `leg, side_won, entry_side_price, entry_yes_price, entered_at_utc, maker_filled`).
- Produces: two printed report lines (no return value).

- [ ] **Step 1: Add the report block**

Edit `src/polymarket_weather/shoulder_book.py`. Find:

```python
    _line("Leg1 shoulder core [20,35)¢", sh[sh["band"] == "core"], GATE_CORE, GATE_CORE_MAKER)
```

Replace with:

```python
    _line("Leg1 shoulder core [20,35)¢", sh[sh["band"] == "core"], GATE_CORE, GATE_CORE_MAKER)
    # Leg 1b — moderate shoulder [0.10,0.25): report-time refinement, FORWARD-only gate.
    mod = moderate_gate_stats(sh)
    if mod:
        c, f = mod["context"], mod["forward"]
        need_n, need_e = GATE_MOD
        gate = ("✅MOD-GATE" if f["gate_pass"]
                else f"{f['n']}/{need_n}@{f['taker']:+.3f}v{need_e:+.3f}")
        print(f"  Leg1b moderate shoulder [10,25)¢ — pre-reg {MOD_PREREG_DATE}")
        print(f"    context (incl. in-sample): n={c['n']} wr {c['wr']:.0%} "
              f"taker {c['taker']:+.3f} | maker {c['maker_n']}/{c['n']} {c['maker']:+.3f}")
        print(f"    FORWARD gate (entered≥pre-reg): n={f['n']} taker {f['taker']:+.3f} "
              f"[{gate}] | maker {f['maker_n']}/{f['n']} {f['maker']:+.3f}")
```

- [ ] **Step 2: Verify the report prints correctly and the gate does NOT inherit the in-sample entries**

Run: `cd src/polymarket_weather && python fetch_historical_truth.py >/dev/null 2>&1; python shoulder_book.py --report`
Expected: output includes a `Leg1b moderate shoulder [10,25)¢ — pre-reg 2026-07-23` block. The **context** line shows a non-trivial `n` (~48, the in-sample band subset). The **FORWARD gate** line shows a **small** `n` (0 or a low count of entries entered on/after 2026-07-23) — it must NOT equal the context n. This confirms forward-only counting.

- [ ] **Step 3: Run the full suite**

Run (from repo root): `pytest -o addopts="" tests/ -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/polymarket_weather/shoulder_book.py
git commit -m "feat: print Leg 1b moderate-shoulder gate in shoulder_book report

Adds the context + forward-gate lines to report() via moderate_gate_stats. Forward
gate reads only entries entered >= pre-reg date, so it starts near 0/80 rather than
inheriting the in-sample discovery entries.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- New constants (§3.1) → Task 1 Step 2. ✓
- Report block context + forward lines (§3.2) → Task 2 Step 1. ✓
- Forward-only counting (§3.3) → `moderate_gate_stats` forward filter (Task 1 Step 5) + verified (Task 2 Step 2). ✓
- Docstring pre-registration §10f (§3.4) → Task 1 Step 1. ✓
- Unit test: band, forward-date, gate math (§4) → Task 1 Step 3. ✓
- Success criteria: forward gate does not inherit 109 (§5) → Task 2 Step 2 verification. ✓
- Out of scope: no `scan_and_record`/schema change, no maker gate → honored (no such steps). ✓

**Placeholder scan:** none — all code and commands are concrete.

**Type consistency:** `moderate_gate_stats` returns `{"context":…, "forward":…}` with keys `n, wr, taker, maker_n, maker` (+ `gate_pass` on forward); Task 2 reads exactly those keys (`c['n']`, `f['taker']`, `f['gate_pass']`, etc.). `GATE_MOD` unpacked as `(need_n, need_e)` in both the function and the report — consistent.
