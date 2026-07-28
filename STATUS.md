# Where things actually stand (plain English)

Updated **2026-07-28**. For the full edge strategy and evidence see `docs/EDGE_MEGAPLAN.md`;
for the technical reference see `CLAUDE.md`. Live numbers, anytime, from
`src/polymarket_weather/`:

```bash
python data_status.py          # sample-size gate progress
python evaluate_oos.py         # the arbiter: model vs market vs ensemble + per-bucket gates
python audit_settlements.py    # is our grading faithful to how markets actually settled?
python shoulder_book.py --report   # the structure paper book vs its gates
```

## What this project does
It studies Polymarket "what will the temperature be?" markets in five cities (NYC, Chicago,
London, Seoul, Hong Kong). Two possible ways to make money: (1) predict the weather better
than the market's price does, or (2) find structural mispricings in how the market prices the
outcome bins, no forecasting required. Everything is measured against **settlement truth** —
what the market actually paid out on — behind pre-committed sample-size gates, because this
project has been burned by flattering measurements four separate times.

## The four broken rulers (all found, all fixed)
1. **Self-grading (2026-06):** bets were graded against the same forecast grid that produced
   them → the fictional "127.5% ROI". Fixed by grading against station observations.
2. **Corrupted truth feed (2026-07-03):** Meteostat was up to 9 °C off recent official
   readings. Replaced with NWS climate reports / METAR summaries / HKO.
3. **~25 code bugs (2026-07):** the biggest silently voided 83% of US markets from grading;
   others leaked future data into the backtest and over-widened the model's distributions.
   All fixed; evaluation regenerated (`docs/BUGFIX_EXECUTION_REPORT.md`).
4. **Wrong settlement source (2026-07-12):** the markets don't resolve on the NWS climate
   report at all — they resolve on **wunderground.com** pages, which can differ by one degree
   exactly at bin boundaries. 4 of 60 audited markets had been graded *backwards*. Truth for
   NYC/Chicago is now reconstructed the way Wunderground computes it (`wu_truth.py`), and
   `audit_settlements.py` permanently checks our grades against how markets actually settled
   (currently 60/61; the one standing miss is a Seoul data-source subtlety).

## Verdict on the forecasting strategy: no edge — it stays OFF

### The pooled test (2026-07-28) — the one that actually has the sample
Every earlier per-bucket hunt split the data 15 ways and left each test far too weak to
conclude anything. Pooled across every city and horizon, on settlement-faithful labels and the
RAW tradeable price:

```
n = 401 markets over 171 independent city-days
model-minus-market Brier gap  +0.0211   95% CI [+0.0068, +0.0353]   t = 2.90
→ the interval sits ENTIRELY ABOVE zero: the model is WORSE than the market, decisively
```

That is not "no edge found yet" — it is a positive finding that the model is measurably worse
by ~2 Brier points. Three independent things agree: (1) this pooled interval; (2) only 4 of 15
buckets even have a negative point estimate where chance alone predicts ~7.5; (3) a null-world
simulation (model recentred to exactly market-equal, real sizes and correlations, 5,000 runs)
produces a best-looking bucket of −0.046 median, while our real best is only −0.016 — the
observed "pockets" are weaker than what pure noise generates.

**Why the per-bucket gates will not settle this.** At the observed dispersion a bucket needs
~418 forward bets to resolve a 0.02 Brier edge (~3 years at current volume); 67 for a 0.05 edge.
The pre-committed 40-bet floor is therefore NOT the binding constraint — at n=40 only an edge
larger than ~0.065 could ever clear the interval, and no bucket has ever shown that. The gates
stay running (they cost nothing and would catch something real) but they are a lottery ticket,
not a plan. `evaluate_oos.py` prints the pooled test first for this reason.

The pre-committed gate is met (240 gradable markets / 354 graded bets as of 2026-07-13), so
this is a real verdict, not a small-sample tease. Accuracy (Brier, lower = better):

| predictor | Brier |
|---|---|
| The market price | **0.128** |
| A plain weather ensemble | 0.160 |
| Our calibrated model | 0.166 |

Backtest ROI at production parameters is **−8.7% over 213 bets** — note this number swings
several points a day as fresh bets resolve (it was −22.5% the day before); the Brier gap above
is the stable signal, and it says the market clearly out-predicts the model.

**Why a ~53% win rate still loses money:** win rate only matters relative to the price paid.
Our bets average ~51¢ per $1 of payout (stake-weighted 54¢), so break-even is ~54–55% after
spread and fee — and the sides we buy win almost exactly as often as their prices already
implied (market said 51.0%, realized ~50–53%): no informational edge, so costs make it
negative. The bleed is concentrated in 70–85¢ entries (win 71% vs 78.5% needed) and cheap
longshots (win 10.2% vs 13.7% needed); only the 85¢+ favorites beat their price. The
market's prices are almost perfectly calibrated on exactly the bets we flag — our big
disagreements with it are our own errors. Two earlier "pockets" of apparent model edge (NYC
same-day especially) evaporated under corrected labels: most of that edge was the mislabeled
boundary days. Two marginal pockets (Seoul and Chicago next-day) remain nominated, paper-only,
and must prove themselves on future graded bets before any real size. **Superseded
2026-07-28:** the hand-picked pair was replaced by ALL 15 city×horizon buckets under test, each
with its own forward clock and a Bonferroni-corrected threshold (`config.E3_NOMINATIONS`), so no
bucket is graded on the data that selected it and testing many cannot manufacture a winner.
One known reason for model weakness is fixable: it was trained against the old truth feed, so
retraining against settlement truth is the top queued model task.

## The promising part: market-structure edges (no forecasting involved)
Calibrating the market across **every** bin (not just ones our model flagged) against
settlement truth found one robust behavioral mispricing, visible from both sides:

- **Bins priced 20–35¢ the day before are overpriced** — they realize ~19% → selling collects
  ≈ +8¢ per share (n=147).
- **Their mirror image, 65–75¢ favorites, are underpriced** — they realize ~81% → buying
  collects ≈ +7.7¢ per share net (n=143). High favorites (85–97¢) are priced *correctly* —
  the edge is specifically the "boring modest favorite" band that crowds under-buy while
  over-buying plausible-looking longshots next door.

Also verified (2026-07-13): Polymarket's real fee schedule — **makers pay nothing and earn
rebates; takers pay at most 1.25¢/share on weather** (0.05·p·(1−p)) — far kinder than the 2%
our backtests conservatively assumed, and a strong argument for maker-first execution.

A **paper book** (`shoulder_book.py`) now records both legs automatically every collection
cycle and grades itself against settlement truth — 15 entries recorded as of 2026-07-13, first
gradings land ~a day after each target date. Pre-registered gates before any real order:
shoulder band ≥150 graded entries at ≥+2¢/share net (core ≥80 at ≥+3¢); favorites core ≥80 at
≥+3¢. In-sample this book earns ~8–11% per position with 1–2 day turnover, ~4–5 entries/day —
the nearest-term realistic path to positive ROI, if the forward numbers hold.

### ⚠️ 2026-07-27: the full-band gate was MET and it is NOT evidence — still no real money
On 2026-07-27 Leg 1's full band hit its pre-registered gate: **n=150, mean +0.0234 ≥ +0.020**,
forward-only (recording starts 2026-07-12, the pre-registration date, with zero prior entries).
It does not establish an edge, for two independent reasons:

1. **It is statistically indistinguishable from zero.** Over 54 independent city-days the
   clustered 95% CI is **[−0.023, +0.070]** (t≈1.0). The gate thresholds were point estimates
   with no power requirement; per-bet dispersion is ~0.345, so detecting a true 2¢ edge at 80%
   power needs ~1,760 *independent* bets. "Gate met" and "no edge" look identical at n=150.
2. **It fails to replicate out-of-sample.** The breadth book (`shoulder_book_breadth.py`, 49
   cities, independent of the 5-city stream) grades the same band at **+0.003 [−0.008, +0.015]**
   on 603 entries / 98 city-days — a tight zero. Worse, the **core band flips sign**: +0.049 in
   the 5-city book vs **−0.028** in breadth. Two samples disagreeing about which band carries
   the effect is what noise looks like, not a structural mispricing.

**Gate amendment (pre-registered 2026-07-27, tightening only.)** Every gate now also requires
its clustered 95% CI to exclude zero over ≥30 independent city-days — a city-day is one weather
outcome, so it, not a bin, is the unit of independence. No existing threshold was moved and
nothing that failed the old gate can pass the new one. Under the amendment **no gate is
currently passed**, which is the honest state. (Note: clustering here *tightens* intervals
rather than widening them, because bins on one day are mutually exclusive — the correction is
about using the right unit, not about being conservative.)

The one number still worth watching is breadth `outer` [5,20)¢ at +0.023 (t≈2.1) — but it is
one of ~4 bands examined and all its graded entries fall in a 3-day window, so treat it as a
hypothesis the accruing book will test, not a finding.

## Bottom line
1. The evaluation machinery is now trustworthy end-to-end: settlement-faithful labels, a
   permanent settlement audit, honest costs, pre-committed gates.
2. **Forecast betting: no edge, off.** Revisit after the settlement-truth retrain and the
   same-day obs overhaul (`docs/EDGE_MEGAPLAN.md` Book A / W1).
3. **Structure betting: unproven, still paper.** The full-band gate was met on 2026-07-27 and
   does not count — CI spans zero and the 4×-larger breadth book replicates it at ~0 with the
   core band's sign flipped (see above). Gates are now power-aware. Real money only when a gate
   passes *under the amendment*; none does.
4. Everything above regenerates from the four commands at the top of this file.
