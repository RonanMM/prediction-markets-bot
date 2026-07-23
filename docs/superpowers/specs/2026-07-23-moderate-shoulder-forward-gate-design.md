# Design — Pre-registered "moderate shoulder" forward gate

**Date:** 2026-07-23
**Status:** Approved design; ready for implementation plan
**Component:** `src/polymarket_weather/shoulder_book.py`
**Author:** Claude (Opus 4.8) with Ronan

---

## 1. Motivation

The structure paper book's Leg 1 (shoulder sell, YES priced `[0.05, 0.35)` pre-day)
shows **no aggregate edge** on its fullest honest sample: 109 graded entries, taker
net **−0.001/share**, win-rate 83.5% ≈ breakeven 82.9% (t = −0.03). It looked like
+6.6¢ earlier only because a stale-local-truth subset (17 entries) was graded.

A read-only meta-analysis of the 109 entries found the aggregate is zero because the
`[0.05, 0.35)` band **lumps two opposite mispricings together**:

| YES band | n | Priced | Realized | Mispricing | Reading |
|---|---|---|---|---|---|
| 0.05–0.10 (deep) | 29 | 7.3% | 10.3% | **+3.1pp** | UNDER-priced → selling loses |
| 0.10–0.15 | 18 | 13.0% | 5.6% | −7.4pp | over-priced → sell |
| 0.15–0.20 | 15 | 17.3% | 13.3% | −4.0pp | over-priced → sell |
| 0.20–0.25 | 15 | 22.3% | 20.0% | −2.3pp | over-priced → sell |
| 0.25–0.35 | 30 | 30.8% | 30.0% | −0.8pp | fair |

The structure **triangulates** across three independent sources, which is what
distinguishes it from a lucky in-sample slice:
1. **Market data** (above): deep shoulders hit *more* than priced; moderate hit *less*.
2. **Known market structure:** the favorite-longshot bias, with the literature's
   refinement that it is strongest in *moderate* longshots while the deepest can be
   *under*-priced.
3. **Our own model meta-analysis:** the `[0, 0.1)` bins realize ~16% vs ~4% predicted —
   the identical fat-tail effect that under-prices deep shoulders here.

## 2. Hypothesis being pre-registered

> Selling the **moderate** shoulder band **YES ∈ [0.10, 0.25)** pre-day (buying NO) is
> positive-EV after real taker costs, because that band is systematically over-priced;
> the deep band `[0.05, 0.10)` is excluded (fat-tail under-pricing) and the fair core
> `[0.25, 0.35)` is excluded (no edge).

This is a **discovered-in-sample** hypothesis. It is **not** an edge claim. It must be
validated on entries recorded **after** the pre-registration date before any real order.

## 3. Design

Purely additive. **No change** to `scan_and_record`, no new leg rows, no schema/column
change. The existing `shoulder` leg already records every qualifying pre-day sell with
`entry_yes_price` and `entered_at_utc`; the moderate leg is a **report-time filter** over
those entries plus a pre-registered gate.

### 3.1 New constants (pre-registered, dated — cannot be moved post-hoc)

```python
# Leg 1b (moderate shoulder) — pre-registered 2026-07-23. A report-time refinement of
# Leg 1: sell ONLY the over-priced moderate band, excluding the fat-tail-under-priced
# deep band and the fair core. Discovered in-sample (n=109); gated forward-only.
MOD_LO, MOD_HI      = 0.10, 0.25        # YES-price band to sell (buy NO)
MOD_PREREG_DATE     = "2026-07-23"      # forward clock: gate counts entries entered on/after this
GATE_MOD            = (80, 0.03)        # (min graded FORWARD entries, min mean net taker $/share)
```

`GATE_MOD` mirrors the shape of the existing `GATE_CORE = (80, 0.03)` — a bar chosen by
analogy, **not** tuned to the observed in-sample number (~+3¢).

### 3.2 New report block

Add a block after the existing Leg 1 full/core lines:

```
Leg1b moderate shoulder [0.10,0.25) — pre-registered 2026-07-23
  context (all graded, incl. in-sample discovery): n=..  wr ..%  taker ±..  maker ../.. ±..
  FORWARD gate (entered >= 2026-07-23):             n=../80  taker ±.. v +0.030  [PASS/pending]  maker ../.. ±..
```

- The **context** line is all graded moderate-band entries — clearly labelled so it is
  never mistaken for the gate. Provides continuity with the discovery sample.
- The **forward** line is the pre-registered gate: only entries with
  `entered_at_utc >= MOD_PREREG_DATE`, graded, in the moderate band. It reports n/80,
  the mean net **taker** edge vs the +0.03 threshold, and a PASS marker only when both
  `n >= 80` and `edge >= 0.03`.
- **Maker** (fills, mean maker net) is reported on both lines as upside — **not gated**.

### 3.3 Forward-only counting (the honesty core)

The gate counts an entry only if `entered_at_utc >= MOD_PREREG_DATE`. Entries recorded
before today — including the 109 that generated the hypothesis — are **context only**,
never gate. This guarantees the hypothesis is never graded on the data that produced it.

### 3.4 Pre-registration note in the module docstring

Add a `§10f` entry to the `shoulder_book.py` module docstring recording: the hypothesis,
the band `[0.10, 0.25)`, the mechanism (deep under-priced by fat tails, moderate
over-priced), the discovery date and sample (n=109), and the forward gate — so the
pre-registration is permanent and auditable, exactly like Legs 1 and 2.

## 4. Testing

Add a unit test to `tests/test_polymarket_weather.py` using synthetic entries spanning
bands and dates:
- band filter selects only `entry_yes_price ∈ [0.10, 0.25)`;
- forward-date filter excludes `entered_at_utc < MOD_PREREG_DATE`;
- gate math: `n` counts forward-graded entries; PASS iff `n >= 80 and mean_net >= 0.03`;
  pending otherwise.

Run: `pytest -o addopts="" tests/ -v` from the repo root.

## 5. Success criteria

- Report prints the new Leg1b block with a correctly-computed forward gate.
- Forward gate reads `n=0/80` today (no entries recorded on/after 2026-07-23 yet) or the
  true small forward count — i.e. it does **not** inherit the 109 in-sample entries.
- All existing tests still pass; new test passes.
- Pre-registration recorded in the docstring.

## 6. Out of scope

- No change to `scan_and_record` (recording), no new leg rows, no new CSV columns.
- No maker gate (maker reported only).
- **No real orders and no edge claim.** This only starts a clean forward measurement.
- No retuning of the existing Leg 1 / Leg 2 gates.
