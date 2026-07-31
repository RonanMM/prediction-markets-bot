# Research archaeology — what was tried, what was never tried, what is running

**Compiled 2026-07-31** (read-only pass; no project file modified except this one).

**Sources mined:** `CLAUDE.md`, `STATUS.md`, `docs/EDGE_MEGAPLAN.md` (§0–§12b),
`docs/BUGFIX_EXECUTION_REPORT.md`, all six `docs/superpowers/specs/*.md` and six `plans/*.md`,
`scratch/meta_analysis_report.md`, `src/polymarket_weather/guide.html` (the public
12-model post-mortem), `shoulder_book.py`, `shoulder_book_breadth.py`, `bet_selection.py`,
`config.py`, the four GitHub workflows, `git log --oneline --all` (348 commits, 7 local +
14 remote branches), and the 14 project memory files.

**Live numbers re-measured today** rather than quoted from docs (`evaluate_oos.py`,
`data_status.py`, `audit_settlements.py`, `shoulder_book.py --report`,
`shoulder_book_breadth.py`, plus a read-only band decomposition and clustered bootstrap of
`output/shoulder_paper_breadth.csv`). See Appendix 1. Several documented numbers have moved
materially — flagged in §D.

---

# A. TESTED AND REJECTED — the do-not-propose-again list

Each entry: what it was → what killed it → the number.

## A1. Forecasting the temperature better than the price (the whole original premise)
- **Pooled, paired, clustered model-minus-market Brier gap: +0.0183, 95% CI [+0.0045, +0.0321],
  n=421 markets / 178 city-days, t=2.60** (measured today). Interval entirely above zero =
  positive finding that the model is *worse*, not absence of evidence.
- Earlier readings, same conclusion: +0.0259 [+0.0084,+0.0434] (n=261/143, 2026-07-28);
  +0.0211 [+0.0068,+0.0353] (n=401/171).
- Scoreboard today: **market 0.1184 < model 0.1367 ≈ ensemble 0.1376**. ROI −11.4% on 421 bets.
- **The unlock was pairing, not sample size** (`pooled-verdict-2026-07-28`): same 261 markets,
  two separate averages SE 0.0195 (can't tell) → paired per market SE 0.0086 (conclusive).
  *Any future comparison must be paired per market and clustered by city-day.*

## A2. EMOS v1 (ERA5-reanalysis-trained calibrator) — retired 2026-07
Trained against reanalysis, i.e. data unavailable at decision time. **Understated live forecast
error 2–3× at betting leads.** Produced the overconfident tails that lost on Brier.

## A3. Random Forest calibrator — removed
"Verified net-negative on held-out data and deleted. More flexibility bought more overfitting."
(guide.html A3). First appearance of *added model complexity did not help*.

## A4. QRF v1 (quantile forest, moment-matched to Student-t) — M1 FAIL
**QRF 0.1713 vs ensemble 0.1685.** Root cause: collapsing the learned quantiles back to a
Student-t was lossy in all five cities (raw quantiles 3–11% better on CRPS) — and the self-gate
scored the collapse, wrongly failing London/Chicago.

## A5. QRF v2 (empirical CDF served directly) — M1 FAIL, worse than v1
**QRF 0.1820 vs ensemble 0.1767.** Re-measured today on the committed tracker: **0.1684 vs
1628 ensemble on the same 111 markets → still FAIL**; paired CRPS 1.1863 vs 1.0835.
The fix *worked on its own terms* — per-city holdout CRPS now beats the ensemble in HK
(0.698 v 1.206), Seoul (0.771 v 0.823), NYC (0.975 v 1.009) — and the market-facing gate still
failed. **The single most valuable finding in the catalogue: better temperature forecasts do not
produce better bin probabilities.** QRF is merged, eval-only, self-gated off; memory says
explicitly *do not re-run the retrain to "get the answer" — it is in.*

## A6. Intraday conditioning as an *edge* (Family C1) — premise falsified 2026-07-27
The mechanism works (by 17:00 local, sigma ≈0.4 °C, obs coefficient c≈0.95). The edge does not:
| same-day regime | n | model | market | gap |
|---|---|---|---|---|
| running max **known** | 67 | 0.165 | 0.117 | **+0.049** |
| morning, not yet known | 112 | 0.136 | 0.109 | +0.027 |
The model does **worse** relative to the market exactly when it has the extra information.
The running max is public; the market prices it at least as well, faster.
**This killed the planned QRF-on-intraday-snapshots successor before a line was written.**

## A7. W2 — sigma inflation on model/market disagreement (2026-07-11)
`sigma_bet = sigma·min(1+γ|p_model−p_mkt|, 1.8)`. **Brier improved 0.1612 → 0.1538 at γ=4;
ROI did NOT (−20.3% → −21/−23%).** Failed its pre-registered ROI leg. Lesson: *widening cannot
fix a centre error.* Demoted to a sizing-honesty layer (γ≈2); never the main defense.

## A8. Static sigma widening / regime-free tail fattening
`evaluate_oos.py:602,615` — "a static sigma widen fixes spring but over-widens summer —
validated OOS to hurt". Confirmed live today: Tmax std(z) 1.78 in March, 1.76 in May, **0.98 in
July** (calibrated). A single global widen is wrong in both seasons.

## A9. E1 — max-selection between predictors (`our_prob = max(ml, ensemble)`)
Systematic upward bias masquerading as edge; corrupts the forecast before any evaluation sees it.
Replaced with a mean; measured edge dropped, which was the point.

## A10. E3 per-bucket "pockets" (hand-picked) — the pockets were noise
- In-sample nomination NYC|same-day + Seoul|1d showed **+16.8% ROI / 48 bets**. First
  regeneration shrank it to +2.1%/55. Settlement-truth labels then **killed NYC|same-day
  outright** (0.084 → 0.175 vs market 0.120; ON-book −21.5%): its edge *was* the mislabeled
  boundary days.
- **Null-world simulation (5,000 runs, model forced exactly market-equal):** best-looking bucket
  median **−0.046**, 5th pct −0.184; our real best is only **−0.016**. Ranking crowns the
  *smallest* bucket — HongKong|2d+ (n=5) wins 33.9% of null runs, London|1d (n=48) 4.4%.
- Today: **zero of 15 buckets has an interval below zero; four are significantly worse.**

## A11. Bet selection — 32 pre-registered subsets, Phase A and Phase C (2026-07-29)
- **Phase A (production universe):** 4 of 32 negative, **every one with |gap| < its own in-sample
  MDE**. Flat ROI negative in 31/32. Best was `bucket@Seoul|1d` −0.0412 (own MDE 0.0540, n=13);
  its forward E3 gap on overlapping data is **−0.0060 (n=33)**. Held-out never spent.
- **Phase C (filters off, universe 2.3× larger: 1156 markets vs 504):** 7/32 negative; the one
  candidate clearing the uncorrected test — `bucket@HongKong|same-day`, CI upper **−0.0004** — is
  a degenerate artifact (**0 YES of 13**, so Brier = mean(p²) and lowest numbers win by
  construction), fails Bonferroni (needs z=3.16; gap+3.16·se = +0.0322), has 11 city-days vs the
  floor of 30, and **loses money (flat ROI −0.018)**.
- **ROI cannot be the test statistic here**: the held-out ROI interval is ~46 pp wide; a +3.3%
  Kelly ROI reversed to −4.5% at flat stakes. The three highest-ROI candidates all had *worse*
  Brier than the market.

## A12. NBM (US National Blend of Models) — tested, not selected
Professional station-specific guidance for KLGA/KORD lost, and not narrowly: **KORD lead-1 1.94 °C
vs our blend 1.19 °C.** A useful negative — *the US-city deficit is not mean-forecast quality.*

## A13. Blend-model shopping: CMA and BOM — rejected
Neutral or harmful in the 2026-07-03 per-city sweep. (Adopted: GEM broadly, Météo-France for
NYC/London, AIFS for Seoul/NYC/London, JMA Seoul-only.) Megaplan §7 explicitly deprioritizes
further blend shopping.

## A14. The hybrid "market centre + our shape" test — shape adds nothing
n=86: market 0.1417, model 0.1842, **market-implied mu + our sigma/nu 0.1406**, 50/50 centre blend
0.1559. The market's implied sigma (~1.0 °C London / 1.2 °C Seoul) is *tighter than ours and
right*. **There is no cheap "better tails" edge around the market's centre.**

## A15. Post-lock settlement lag (§10c) — mostly efficient
10 of 11 markets with tick data sat within a **median 0.1¢** of terminal in the 24 h after the
outcome was physically locked. The 11th was our own mislabeled NYC 07-03 row.

## A16. Naive favourite bias ("90¢ favourites are really 95¢") — dead
Favourites priced **85–97¢ realize 0.920 vs 0.922 priced (n=238)** — ~breakeven after spread.
50–65¢ "favourites" are negative (the label is noise); mid-day favourites tested negative.

## A17. Optimizer re-tuning (`optimizer.py`, `optimizer_full.py`)
Coordinate-ascent + OOS-split grid searches both endorse the existing config rather than moving
off it. **"Do not re-tune; the bottleneck is data, not parameters."** (Caveat: those searches
predate the settlement-truth corrections — Phase C tested the filters-off case and found nothing.)

## A18. Shoulder-book Leg 1 full band — gate MET 2026-07-27, and it is NOT evidence
n=150, mean **+0.0234 ≥ +0.020** threshold, genuinely forward-only. Rejected as evidence for two
independent reasons: (1) **clustered 95% CI [−0.023, +0.070], t≈1.0** over 54 city-days —
"met" and "no edge" indistinguishable; detecting a true 2¢ edge at 80% power needs **~1,760
independent bets**; (2) **fails to replicate** — breadth put the same band at +0.003
[−0.008,+0.015] and *flipped the sign of the core band* (+0.049 → −0.028).
→ Gate amendment (tightening-only, 2026-07-27): every gate now also needs a clustered 95% CI
excluding zero over ≥30 city-days.

## A19. Maker-side "the edge is bigger as a maker" — unproven, and negative where significant
The original +5.8¢ maker number came from a **fill detector that was nearly blind** (median ONE
later price print per entry; 36% had none). Rebuilt on dense CLOB history: full-band maker
+0.072 → **+0.056, CI [−0.018,+0.129], t=1.49 — not significant**. Breadth at ≥1k liquidity:
**−0.0065 [−0.028,+0.015]**. Re-measured today, breadth maker is negative in every band
(full −0.021, core −0.053, moderate −0.001). The one **significant** maker result is a warning:
**thin books (<$1k liquidity) lose −0.064 [−0.124,−0.005]** — don't make markets where nobody
trades.

## A20. Meteostat as truth feed — corrupt, replaced
Up to **9 °C** from official readings on recent dates (2026-07-03 audit). Legacy/reference only.

## A21. NWS CLI as settlement truth — the fourth broken ruler
Markets resolve on **wunderground.com**, whose daily extremes are hourly-METAR max/min over the
local calendar day. **4 of 60 audited settlements were graded backwards.** Where both sources
exist they disagreed on **99 of 174 days**, flipping **18 of 573 rows (3.1%)**.
→ Corollary that pre-emptively kills an untested idea: see B9.

## A22. Grading against the forecast grid — the "127.5% ROI" mirage
Prediction and "truth" shared the grid's error. Station-truth grading roughly **halved** measured
ROI (48% → 22% on one 64–76 bet sample).

## A23. Nine broken measuring instruments, all flattering
June: self-grading. 07-03: corrupt truth feed. July: ~25 code bugs (largest silently voided
**83% of US markets**). 07-12: wrong settlement source. 07-20: correct grader silently unused in
3 of 4 workflows. 07-27: gate passing on a point estimate. 07-27: near-blind fill detector.
07-28: scoreboard comparing different market sets (model 0.148 not 0.1369 on equal footing).
07-28: a comparison frozen for days. Plus 07-30: **a truncated obs download flipped the headline
verdict** (+0.0178 CI above zero → +0.0122 CI spanning zero) on a green run.
**Six of nine were silent. Every single correction moved the answer against the model.**

---

# B. PROPOSED BUT NEVER EXECUTED — ranked by expected value today

The ranking weighs: (i) does the blocker still bind, (ii) does the 11× discovery fix (2026-07-30,
24 → 264 markets/cycle) unblock it, (iii) does it survive what §A already established.

---

## B1. ⭐ The deep/outer shoulder band [5,20)¢ was DELIBERATELY EXCLUDED from every gate — and the evidence it was excluded on has now reversed, significantly

**Where written:** `docs/superpowers/specs/2026-07-23-moderate-shoulder-forward-gate-design.md`
§1–§2; `shoulder_book.py` `MOD_LO, MOD_HI = 0.10, 0.25`.

**What it proposed (and did):** pre-register the *moderate* band [10,25)¢ only, on the argument
that the two ends behave oppositely. The deep band was excluded with an explicit rationale:

> "0.05–0.10 (deep) | n=29 | priced 7.3% | realized 10.3% | **+3.1pp UNDER-priced → selling
> loses**" … "the deepest can be *under*-priced" (favourite-longshot literature) … "our own model
> meta-analysis: the [0,0.1) bins realize ~16% vs ~4% predicted — the identical fat-tail effect."

**Why it stalled:** it did not stall — it was *decided against*, on **n=29 entries** from the
5-city book.

**Does the blocker still hold? No — it has reversed with 19× the sample.** Measured today on
`shoulder_paper_breadth.csv` (all 1,873 graded entries are forward of `BREADTH_PREREG_DATE`),
clustered by city-day, with a 2,000-draw cluster bootstrap as a check on the skewed loss
distribution:

| band (YES ¢) | n | city-days | win | mean net taker | clustered 95% CI | bootstrap CI |
|---|---:|---:|---:|---:|---|---|
| full [5,35) | 1829 | 294 | 84.8% | +0.0051 | [−0.0014,+0.0116] | [−0.0012,+0.0112] |
| **outer [5,20)** | **1128** | **293** | **93.8%** | **+0.0327** | **[+0.0202,+0.0453]** | [+0.0198,+0.0455] |
| **deep [5,10)** | **555** | **254** | **97.5%** | **+0.0317** | **[+0.0192,+0.0442]** | [+0.0186,+0.0431] |
| [10,20) | 573 | 262 | 90.2% | +0.0338 | [+0.0115,+0.0561] | [+0.0126,+0.0559] |
| moderate [10,25) — *the gated band* | 786 | 287 | 87.3% | +0.0248 | [+0.0060,+0.0436] | [+0.0053,+0.0424] |
| core [20,35) | 701 | 291 | 70.3% | **−0.0394** | **[−0.0624,−0.0164]** | [−0.0619,−0.0144] |
| [25,35) | 488 | 269 | — | −0.0569 | [−0.0887,−0.0251] | — |

Three things follow. (1) The deep band is **over**-priced here, not under-priced — realized ~2.5%
against ~7.5% priced — the exact opposite of the n=29 finding the exclusion rests on. (2) t≈5.1
for outer and ≈5.0 for deep survives Bonferroni for the ~7 bands examined (z=2.69). (3) The core
band's exclusion is **vindicated and now significant** (−0.039, CI below zero), which is also why
the full band reads ~0: it is +0.033 and −0.039 cancelling.

**Honest caveats (do not skip):** the band split is **post-hoc** — outer/deep was not
pre-registered, and STATUS.md already flagged breadth `outer` as "one of ~4 bands examined". The
data covers only **2026-07-23 → 07-30, eight days of summer**, and calm weather structurally
favours shoulder-*selling* (the outcome lands near the mode, so the sold bins don't hit) — this
is a live seasonality confound with no shoulder season in the sample. And the edge is **~3.2¢
against an assumed `HALF_SPREAD = 0.01`** at a ~94¢ NO price, which is a guess (see B2).

**Action:** pre-register a **new, dated, tightening-only** breadth gate for `[5,20)¢` today with
(n, effect, clustered CI) stated in advance, and start the forward clock now. At the measured
accrual of ~300 breadth entries/day (~230 graded/day), an n=500 forward deep-band sample lands in
**about a week**, not a quarter. This is the cheapest well-powered question in the project.

---

## B2. ⭐ Best bid/ask + order-book depth in the snapshot store — proposed 3 times, never built, and it is now the binding blocker on B1

**Where written:** `EDGE_MEGAPLAN.md` §6 (data additions table: *"Best bid/ask + depth per bin in
snapshots — Book C, E1/E2 — we currently store only mid/last; spread assumptions are guesses until
this lands"*), §3 Book C prereqs, §10b caveats ("mid-price fills overstate what a taker gets").

**Verified never built:** `data/polymarket/*_snapshots.csv` columns are
`fetched_at_utc, city, condition_id, question, active, closed, end_date_iso, start_date_iso,
volume_usdc, volume_24h_usdc, liquidity_usdc, market_slug, outcome_probs_json, clob_token_ids_json`.
No bid, no ask, no depth. `config.HALF_SPREAD = 0.01` still carries the comment *"tune once real
order-book data exists"*.

**Why it stalled:** sequenced behind everything else (§7 item 8) and nobody needed it while no
strategy was close to positive.

**Does the blocker still hold? No — and half the code already exists.**
`data_loader.check_orderbook_vwap()` already pulls the **CLOB L2 book** and computes a
size-weighted fill price; it is called only at bet time (`engine.py:753`) and nothing is
persisted. Recording best bid/ask (and top-of-book size) per bin per collect cycle is a
column-addition to an append-only CSV plus one call already written.

**Why it matters now:** B1's edge is 3.2¢/share on a NO side trading near 94¢. If the true
half-spread at that price is 2–3¢ rather than 1¢, the entire result is an artifact of a
hard-coded constant. **This is the single highest-leverage missing measurement in the repo**, and
it is a measurement, not a model.

---

## B3. ⭐ Kalshi cross-market archiver — ranked #3 in the queue on 2026-07-12, never started

**Where written:** `EDGE_MEGAPLAN.md` §10e-1 and §9 queue row 3 — *"Kalshi archiver — start
collecting NOW, analysis later | **next new-data item**"*. Rationale: Kalshi runs the same city
temperature contracts settling on the same reports, with a different (more professional?) crowd
and a free API. Uses: leading signal, cross-book arbitrage when bins diverge, second calibration
reference. *"Cheapest new data source with the highest ceiling."*

**Verified never started:** the string `kalshi` appears in exactly one file in the repo —
EDGE_MEGAPLAN.md. No fetcher, no data directory, no workflow step.

**Why it stalled:** displaced by W0 (settlement truth) on 2026-07-12 and never re-surfaced; the
project then spent two weeks proving the model has no edge.

**Does the blocker still hold? No.** There was never a technical blocker — it is a free public
API and the archiving cost is one more step in `collect.yml`. And it is the *only* item in the
entire corpus that satisfies the project's own stated re-entry condition:

> STATUS.md: *"Revisit only on a genuinely new information source, not a thirteenth modelling
> approach."*

Every dollar of the last month went into thirteenth modelling approaches. This is the one
genuinely-new-information item, and the cost of *starting the archive* is decoupled from the cost
of analysing it — an archive not started today can never be analysed in September.

---

## B4. ⭐ Book C — the coherence / basket-arbitrage book (no forecast view). Blocked on partition coverage, which the discovery fix just removed.

**Where written:** `EDGE_MEGAPLAN.md` §3 Book C, §7 item 8, §8 gate (*"≥30 executed paper baskets
with Σ-mispricing ≥ 3¢ after depth-checked fills"*).

**What it proposed:** when Σ(YES asks) over an exhaustive bin partition < 0.97, buying the whole
set is near-riskless (exactly one bin must resolve YES); when Σ(YES bids) > 1.03, selling the set
is. Plus relative value *within* the PMF against the market's own centre (never against the
centre — §1e says the centre is right). α5 already *measures* bin-sum deviation but only awards a
score bonus.

**Why it stalled:** listed prereqs were depth data (B2) and E2 execution plumbing.

**Does the blocker still hold? Partially — and the *unstated* blocker just vanished.** A basket
trade requires seeing **every bin of an event simultaneously**. Until 2026-07-30 discovery
returned ~24 markets/cycle across five cities — a fraction of one city-day's ~11 bins — so Σ over
the partition was not merely noisy, it was **not computable**. Tag discovery now returns 264/cycle
from a single `/events?tag_slug=weather` pass that yields **whole events with their full market
lists**, i.e. complete partitions by construction. The breadth book already consumes exactly this
feed across ~50 cities.

Two independent supports: the breadth scan measured a **median +4% overround** on every weather
book, and `pmf_sum_dev` has a **median 0.595** on the training frame (bet-selection spec §5) —
incoherence is large and everywhere. Depth (B2) remains a genuine prereq for claiming the basket
is *fillable*; measuring how often the partition sums outside [0.97, 1.03] costs nothing and
needs no depth.

---

## B5. Bin-addition mechanics — never built; the data thinness that made it undetectable is gone

**Where written:** `EDGE_MEGAPLAN.md` §10e-6. *"When a forecast shifts enough that Polymarket adds
a new bin to the event, sibling prices reprice mechanically and lag; detectable from our snapshots
(new condition_ids appearing mid-event)."*

**Why it stalled:** untested, un-ranked, and — decisively — **44% of collected markets had exactly
one snapshot**, so "a new condition_id appearing mid-event" was indistinguishable from "we finally
saw this market". Detection was impossible.

**Does the blocker still hold? No.** With 264 markets/cycle and full event enumeration, a new bin
is now visibly an *addition to an event we were already tracking*. Model-free, offline-testable on
data already committed going forward, and structurally adjacent to B1/B4 (it is a repricing of the
shoulders).

---

## B6. Draw a validation split FORWARD — the fix was written down and the clock was never started

**Where written:** `EDGE_MEGAPLAN.md` §12a. *"The fix, for any future validation: draw the split
FORWARD. Pre-register the rule, start the clock today, and validate on markets that settle
afterwards."* Memory `retrospective-splits-unsound` says the same.

**Why it stalled:** written as the closing note of a negative result; the session ended.

**Does the blocker still hold? No — it is pure discipline, costs nothing, and B1 needs it today.**
Note also that Phase A's held-out set (209 markets / 75 city-days) is **still unspent** —
`output/holdout_log.jsonl` does not exist. It remains a live, uncontaminated one-shot for any
*newly* pre-registered rule, although §12a correctly observes that a retrospectively-drawn split
is not defensible here because the standing gates report on those markets daily.

---

## B7. W4 — the resolution-mechanics audit remainder (cheap, and history says it moves the number)

**Where written:** `EDGE_MEGAPLAN.md` §4 W4, §7 item 5.

**Two specific unfinished items:**
1. Root-cause the `lte`/`gte` question-type bucket: **n=13 but the largest per-row gap of any
   qtype, +0.13 / +0.06.**
2. Root-cause the alignment-shaped outliers, e.g. **Chicago 2026-03-22: mu 9.9 °C vs actual
   20.6 °C at lead 0.5** — a 10.7 °C "miss" that looks like a wrong-day or stale-forecast row.

**Why it stalled:** superseded by W0 (which handled the *source* question) and never revisited;
W4 was §7 item 5 and the queue never got there.

**Does the blocker still hold? No — both are offline, hours of work.** Precedent strongly favours
doing it: every prior measurement correction moved the verdict against the model, and the
dispersion monitor still reads Tmax std(z)=1.78 in March. A single day-boundary error at the tails
is worth many Brier points.

---

## B8. W3 — regime features for the two known biases (Chicago lake-breeze, Seoul flow-conditional anchor)

**Where written:** `EDGE_MEGAPLAN.md` §4 W3, §7 item 6. Chicago: onshore E/NE component at KORD
hours 10–18 + lake-vs-air differential (NOAA GLERL) + weak gradient → cap/skew Tmax. Seoul: choose
or blend the Bucheon-corridor vs RKSI anchor by forecast wind direction.

**Why it stalled:** §7 item 6, never reached. Verified never built (`glerl`, `buoy`, `lake_breeze`
appear nowhere in the codebase).

**Does the blocker still hold? The data blocker no longer does — the *evidence* blocker does.**
It is fully backtestable offline on the 2022→now archives that exist. But: Chicago|same-day is
today's worst bucket (**+0.0563, CI [+0.0142,+0.0984]**, i.e. significantly worse) *and* the
market's Chicago Brier of 0.0818–0.0856 is the sharpest number on the board — so this attacks a
real defect against the hardest counterparty. A12 (NBM losing by 0.75 °C) already established the
US deficit is not mean-forecast quality. **Rank: medium-low; a correctness/curiosity item, not an
edge plan.**

---

## B9. 1-minute ASOS boundary sniping — never built, and W0 appears to have quietly falsified it

**Where written:** `EDGE_MEGAPLAN.md` §10e-2, ranked #2 of the untested list. *"Hourly-METAR
watchers can't see between-obs spikes; the CLI/WU max can. On days the 1-min series crosses a bin
boundary the hourly series didn't, we know the settlement moved and most of the market doesn't."*

**Why it stalled:** folded into Book A / W1, which never happened.

**Does the blocker still hold? The premise appears to be wrong.** W0 established that markets
settle on **wunderground.com**, whose daily extremes are the **hourly-METAR** max/min over the
local calendar day — that is precisely why `wu_truth.py` exists and why it fixed 3 of 4 audit
misses. If settlement is computed from hourly METARs, a 1-minute spike the hourly series missed
**does not move the settlement**, so there is nothing to snipe. The idea was written on
2026-07-12, hours before the reasoning that undercuts it. Flagging rather than deleting: it is
worth one explicit check against `resolution_url` semantics before anyone spends a week on it.

---

## B10. Book B — the run-release event trader

**Where written:** `EDGE_MEGAPLAN.md` §3 Book B, §6 (release-aligned fetch scheduler), §7 item 7,
§8 gate (≥100 logged release signals; post-signal drift ≥ 2× modeled cost).

**Why it stalled:** needs W1's minute-cadence infrastructure, which was never built.

**Does the blocker still hold? Yes, and it got worse.** The honesty check in §3 already says this
is ≈ breakeven at taker costs and 2 h cadence, with the bet being that minute cadence turns +1–2¢
gross into +2–4¢ net. The execution substrate is now **GitHub Actions**, whose schedules are
best-effort — measured 3h49 gaps and dashboard runs firing ~3 h late. A latency strategy cannot
live there. Reviving it means a different hosting substrate first, which is a real project.
**Rank: low unless the substrate changes.**

---

## B11. W1 items 1 & 4 — 10–15 min obs polling and the heating-curve nowcast v2

**Where written:** `EDGE_MEGAPLAN.md` §4 W1 (items 1 and 4), §7 item 2 — *"the single biggest
lever"*.

**Why it stalled:** superseded when Book A's premise was tested.

**Does the blocker still hold? Mostly yes.** A6 falsified the *informational* half: the market
prices the running max at least as well as we do. Note the falsification was of the **modelling**
advantage — the **freshness** half (poll every 10–15 min instead of hourly) was never separately
tested. But per B10, freshness cannot be delivered on the current substrate, and the guide's own
synthesis says being able to read a public feed is "table stakes… you are competing on speed."
**Rank: low. Do not revive as a model change.**

---

## B12. Bimodal-ensemble mixture pricing

**Where written:** `EDGE_MEGAPLAN.md` §10e-3. On frontal-timing days the 122-member ensemble is
bimodal and the true PMF two-humped; a mixture PMF prices between-hump bins below and hump bins
above a unimodal market. Requires storing member-level (or quantile) ensemble output plus a
dip-test trigger.

**Why it stalled:** needs a data change — `fetch_ensemble.py` stores mean/std, not members.

**Does the blocker still hold? Yes (data), and the thesis is weakened.** §1e already showed there
is no shape edge around the market's centre, and A5 showed a genuinely better-shaped distribution
(empirical QRF, 3 of 5 cities beating the ensemble on CRPS) still priced bins **worse**. The one
distinguishing argument left is that §1e only ruled out a better *unimodal* shape. **Rank: low.**

---

## B13. Upstream-station nowcasting (GLERL buoys, KMA AWS between the Incheon coast and RKSI)

**Where written:** `EDGE_MEGAPLAN.md` §10e-5. Sea/lake-breeze fronts announce themselves at
shoreline stations 1–2 h before they cap the resolution station; nobody trading these books watches
those feeds. Feeds W3.

**Why it stalled:** new feeds + W3 never started.

**Does the blocker still hold? Yes (new data ingestion), and A6 argues against.** The one genuine
distinction from A6: shoreline-buoy data is *not* the feed everyone else reads, so the
information-parity objection does not automatically apply. **Rank: low-medium; the most defensible
of the remaining forecast-side ideas precisely because it is non-public-by-habit.**

---

## B14. Chicago → NYC synoptic sequencing

**Where written:** `EDGE_MEGAPLAN.md` §10e-7. The same front hits KORD ~a day before KLGA;
Chicago's *realized* outcome updates NYC's next-day distribution before NYC's crowd reacts.
Explicitly *"testable offline from truth + leads archives."*

**Why it stalled:** last item on an untested list.

**Does the blocker still hold? No — zero new data required.** It is a correlation study on files
already on disk (`*_historical_actuals.csv`, `*_historical_leads*.csv`) and could be answered in an
afternoon. Weak prior (NWP models already ingest the upstream observation), but it is the cheapest
remaining forecast-side test. **Rank: medium-low.**

---

## B15. E2 — maker-first execution / paper CLOB plumbing

**Where written:** `EDGE_MEGAPLAN.md` §5 E2, §7 item 4; prereq of Books A/B/C and of any go-live.

**Verified never built:** `engine.py:805` still `raise NotImplementedError("Wire in
py_clob_client. See class docstring.")`.

**Does the blocker still hold?** Nothing technical. But A19 removed the *reason* — the maker
thesis is unproven overall and significantly **negative** in thin books. Fee reality (E1,
verified) still favours makers: takers pay ≤1.25¢/share (0.05·p·(1−p)), makers pay **zero** and
share a 25% rebate pool. **Rank: required before any real order, not a source of edge.**

---

## B16. E4 — sizing v2 (market-as-evidence posterior replacing scalar `SHRINK_WEIGHT`)

**Where written:** `EDGE_MEGAPLAN.md` §5 E4. **Why it stalled:** depends on W2, which failed (A7).
**Still blocked, and moot** — "the bottleneck was never sizing" (§5 E4's own words).

---

## B17. `SHRINK_WEIGHT` — recommended by every sweep since June, never actually set

`config.SHRINK_WEIGHT = 1.0` (a no-op). Today's sweep recommends **w = 0.15, Brier 0.1177** —
which is *below* the pure market's 0.1184, the only sign anywhere that the model carries a sliver
of information the price lacks. The 2026-07-22 meta-analysis recommended **w = 0.00**; the
2026-07-13 bugfix report recommended **w = 0**. The recommendation has never been actioned.
**Honest reading:** a 0.0007 Brier improvement is well inside noise, and CLAUDE.md correctly frames
shrinking as damage limitation, not edge. **Rank: low, but it is a genuinely unexecuted written
proposal and it is one config line.**

---

## B18. `FEE_RATE = 0.02` migration in the model backtesters — "queued" since 2026-07-13

**Where written:** `EDGE_MEGAPLAN.md` §5 E1; `config.py:119` *"FEE_RATE is queued (their 2%
assumption overstates costs, especially near-extreme prices)."* The real schedule was verified
2026-07-13. The model backtesters still charge 2% of winnings — conservative, so it makes the
model look *worse*, which is why nobody hurried. **Cheap; do it before quoting any model ROI
again.** Note the structure books already use the verified fee.

---

## B19. §10c passive settlement-lag scanner

**Where written:** `EDGE_MEGAPLAN.md` §10c — *"Keep as a passive scanner (alert when
|price − known outcome| > 3¢ after day-end), not a book."* Never built.
**Blocker:** none; and with breadth grading off Polymarket settlements across ~50 cities plus 11×
discovery, the sample for this is now ~300 markets/day instead of 11. **Rank: medium-low, cheap.**

---

## B20. The `is_stale` and intraday selectors that bet-selection could not test — now testable, and α8 was broken anyway

**Where written:** `bet-selection-design.md` §5 "Deliberately excluded": *"`is_stale` (22 on
train) and intraday conditioning (12 on train). Too thin to test."*

**Two things changed.** (1) Volume: at 264 markets/cycle these populations grow ~11×. (2) More
important — **α8 was not merely thin, it was wrong**: `signals.market_staleness` returns
`is_stale=False, hours_since_move=0.0` (the *maximum-freshness* reading) whenever fewer than 2
snapshots exist, which was **44% of all markets**. Every result that ever involved α8 — including
the optimizer's endorsement of `STALE_HOURS` / `STALE_MOVE_THRESHOLD` — was computed on a signal
that encoded absence of evidence as evidence of activity. α1 (momentum) and α7 (convergence)
"mostly could not compute" for the same reason.
**Action:** this is not a strategy proposal so much as a *re-measurement obligation* — the α-signal
layer has never once been evaluated on non-degenerate data.

---

## B21. Roadmap leftovers (memory `raincheck-model-roadmap`, 2026-07-04)
- **Tmin blend re-sweep** on min targets (Tmin currently inherits the Tmax-adopted per-city model
  sets as a prior). Never run.
- **Seasonal blend re-sweeps** (AIFS archive starts mid-2024; GEM missed NYC's margin; BOM tiny
  window). Never run.
- **Live intraday effect check by `sigma_source`** — superseded and answered by A6.
**Rank: low; A1/A5 make blend refinement a rounding error.**

---

## B22. Deferred engineering that is still open (BUGFIX_EXECUTION_REPORT §7)
`F8` (extract the triplicated per-bin eval block in `engine.analyse_city`, needs a characterization
test), `F6` (collapse the 4 diverged previous-runs leads fetchers into one chunker), `F7 remainder`
(~9 ad-hoc slug call sites), `F11`/`F14` (Gamma pagination fan-out — **partly obsoleted by tag
discovery**, and per-bin re-filtering in `market_staleness`). Zero behaviour change; listed for
completeness. Also still open from §6: **`output/all_bins.csv` is unreadable/malformed** (file
untouched since May 14).

---

## B23. Decided-against, recorded so nobody re-proposes it
- **Expand the *forecast model* to new cities** (§10e-4 universe expansion). Executed for the
  *structure* book only; the breadth memory is explicit: *"Do NOT expand the forecast model to new
  cities — it loses to the market anyway."*
- **A faster HKO endpoint** (§6 data table: *"HK: faster HKO daily feed or drop HK"*). Superseded
  by a standing decision (Ronan, 2026-07-28): grading must use the source Polymarket actually
  resolves on, for every city. HKO publishes in **monthly batches ~3 weeks after month end**, so HK
  is ungradable for the whole current month by design — a frozen HK truth file is expected, not
  broken. (Confirmed live: HK truth latest 2026-06-30, 31 days behind.)
- **Book A as its own paper book** with the ≥60-same-day-bets gate (§3/§8). Subsumed into E3 and
  then falsified by A6.

---

# C. LIVE AND RUNNING — every pre-registered gate and its distance from passing

All gates now carry the **2026-07-27 power amendment**: (n, effect) **and** a clustered 95% CI
excluding zero over **≥30 independent city-days**. Amendments are tightening-only and dated.

## C1. Structure paper book — 5 cities (`shoulder_book.py`, `output/shoulder_paper.csv`)
280 entries, 176 graded, 104 awaiting truth. Accrual **jumped from ~12/day to 43/day on
2026-07-30** (the discovery fix) — the gates below now fill ~3.5× faster than the documents assume.

| leg / band | gate (n, effect) | current | verdict | binding constraint |
|---|---|---|---|---|
| Leg1 shoulder full [5,35)¢ | 150, +0.020 | n=175, 66 city-days, **+0.034** CI [−0.009,+0.076] | **n ✅ effect ✅ CI ❌** | significance |
| Leg1 core [20,35)¢ | 80, +0.030 | n=76, 51 city-days, +0.059 CI [−0.019,+0.138] | pending | n (4 short) *and* CI |
| **Leg1b moderate [10,25)¢** (pre-reg 2026-07-23, forward-only) | 80, +0.030 | **n=25/80**, +0.034 | pending | **n** — ~2 weeks at the new rate |
| Leg2 favourite core [65,75)¢ | 80, +0.030 | **n=1**, −0.671, 1 city-day | pending | structurally starved (see note) |
| Leg1 core, maker | 80 fills, +0.030 | 44/76 filled, +0.115 | pending | n |

**Leg2 is starved by construction:** it requires a YES price in [65,85)¢ **and** >12 h to local day
end — 1 graded entry in 2.5 weeks (47 in breadth). Either the window is wrong or this leg cannot be
measured on this cadence; worth a design decision rather than more waiting.

## C2. Breadth structure book — ~50 cities (`shoulder_book_breadth.py`)
2,384 entries across 50 cities; 1,873 graded, 511 awaiting settlement; ~300 entries/day. All
entries are forward of `BREADTH_PREREG_DATE = 2026-07-23`, so its sample is clean by construction.

| leg / band | gate | current | verdict |
|---|---|---|---|
| **Leg1b moderate [10,25)¢** (`GATE_MOD_BREADTH = (80, 0.03)`) | 80, +0.030, CI>0, ≥30 city-days | **n=806/80 ✅ · 287 city-days ✅ · CI [+0.0080,+0.0447] ✅ · effect +0.0263 ❌** | **pending on the effect threshold alone, short by 0.4¢** |
| Leg1 full [5,35)¢ | (reported, no breadth gate) | n=1829, +0.0051 CI [−0.0014,+0.0116] | flat zero |
| Leg1 core [20,35)¢ | (reported) | n=701, **−0.0394 CI [−0.0624,−0.0164]** | **significantly negative** |
| Leg2 favourite [65,85)¢ | (reported) | n=44, −0.0215 CI [−0.150,+0.107] | no information |
| maker, all bands | (reported, never gated) | full −0.021 · core −0.053 · moderate −0.001 | negative |

**This is the closest anything in the project has come to a legitimate pass.** Three of four gate
conditions are met with room to spare; the effect estimate sits 0.0037 below the pre-registered
+0.030, with a CI upper bound of +0.0447. The threshold **must not be lowered** (tightening-only
rule). The honest read: it either drifts up as sample accrues, or it settles as a real-but-smaller
edge that the pre-registered bar deliberately refuses to trade. Note also that the un-gated
neighbouring bands (B1) are stronger than the gated one.

## C3. E3 forward gates — all 15 city × horizon buckets
Rule: ≥40 forward bets, ≥30 city-days, gap interval entirely below zero at **z = 2.94**
(Bonferroni for 15 tests). `LIVE_BUCKETS` is **empty** and now means *gate-passed*, not
*someone liked it*.

| bucket | clock since | forward n | gap | can see (MDE) | verdict |
|---|---|---:|---:|---:|---|
| Seoul\|1d | 2026-07-12 | 33 | −0.0060 | 0.049 | pending (33/40) |
| Chicago\|1d | 2026-07-12 | 4 | −0.0607 | 0.126 | pending |
| London\|same-day | 2026-07-28 | 4 | −0.1299 | 0.165 | pending |
| London\|1d | 2026-07-28 | 4 | +0.0657 | 0.184 | pending |
| NYC\|same-day | 2026-07-28 | 3 | −0.0176 | — | pending |
| Chicago\|same-day | 2026-07-28 | 3 | +0.0650 | — | pending |
| Seoul\|same-day | 2026-07-28 | 2 | −0.0713 | — | pending |
| London\|2d+ | 2026-07-28 | 1 | −0.0800 | — | pending |
| 7 others (Seoul\|2d+, Chicago\|2d+, NYC\|1d, NYC\|2d+, HK×3) | 2026-07-28 | 0 | — | — | no forward bets |

**These are lottery tickets, by the project's own arithmetic.** At the observed dispersion a bucket
needs **~418 forward bets to resolve a 0.02 edge (~3 years at the old volume)**; 67 for a 0.05
edge. At n=40 only an edge >~0.065 could ever clear — larger than any gap ever observed.
The 40-bet floor is decorative; the interval is the constraint. They cost nothing to keep running.
*(The 11× discovery fix should compress the timeline materially, but nobody has re-derived the
estimate at the new rate — worth doing, since STATUS.md explicitly says the ~3-year figures are now
pessimistic.)*

## C4. QRF M1 gate — FAIL, parked
`QRF 0.1684 > ENSEMBLE 0.1628` on the same 111 markets (measured today). Merged, eval-only
(`--predictor qrf`), self-gated off, never touches production. Memory is explicit: do not re-run
the retrain.

## C5. Standing guards
- **Pre-committed sample gate: MET** — 421/150 gradable markets, 639/100 gradable bets.
- **Settlement audit: 129/133 = 97.0%**, floor 95%. Standing misses: HongKong 06-08, HongKong
  06-11, Seoul 03-24, Seoul 07-19.
- `dashboard.yml` now runs the audit **before** building, so a degraded ruler blocks publishing.
- `build_dashboard._missing_cities` refuses to publish if any `CITY_ORDER` city is ungradable.
- `fetch_station_obs` guards: a failed year-chunk keeps the existing CSV; a refetch may never
  shrink or go backwards in latest observation.

---

# D. INTERNAL CONTRADICTIONS AND STALE CLAIMS

Ordered by how likely each is to mislead a new proposal.

### D1. W0.2 ("retrain against settlement truth") is described as the top queued task — it has already shipped
- `CLAUDE.md`: *"retraining EMOS/Tmin/intraday against settlement truth is the **top queued model
  task** (W0.2)"* — and again at the E3/W0 note.
- `STATUS.md`: *"retraining against settlement truth is the top queued model task."*
- `EDGE_MEGAPLAN.md` §9 row 4 and the READ-THIS-FIRST header: *"top queued work = W0.2."*
- `.github/workflows/retrain.yml` header: *"CLAUDE.md flags a pending settlement-truth retrain
  (W0.2)."*
- **Reality:** `train_calibrator.py:159-161`, `train_intraday.py:71-73` and `train_qrf.py:148-155`
  all call `settlement_truth.load_training_truth(slug)` with the comment *"settlement-faithful
  target (W0.2)"*, and `retrain.yml` rebuilds `{slug}_settlement_actuals.csv` in-runner.
  **W0.2 is done.** Anyone reading the docs would propose work that already exists.

### D2. E3 nominations — CLAUDE.md still names two hand-picked buckets
`CLAUDE.md`: *"current nominations Seoul|1d + Chicago|1d, forward clock restarted 2026-07-12"*
and *"E3 per-bucket selective aggression … `config.LIVE_BUCKETS`"*.
**Reality:** `config.E3_NOMINATIONS` holds **all 15** buckets with per-bucket clocks and a
Bonferroni-corrected threshold; `LIVE_BUCKETS` is an **empty set** and its meaning changed to
*gate-passed*. STATUS.md records the supersession; CLAUDE.md does not.

### D3. "The model loses to the raw ensemble it is built on" — no longer true unpaired, still true paired
`guide.html` A2 and the QRF spec §0 both headline *"it loses to the raw physics it is built on"*
(0.150 vs 0.142 / 0.1417 vs 0.1360). **Today: model 0.1367 ≤ ensemble 0.1376 → the EDGE CHECK's
ensemble leg reads PASS.** But on the *paired* 274-market common set the ensemble is still ahead
(0.1376 vs model 0.1400), and paired CRPS still favours the ensemble (1.1037 vs 1.1989).
The claim needs the word "paired" attached wherever it appears, or it reads as simply false
against the arbiter's own output.

### D4. `retrain.yml` contradicts itself inside one file
Line 8: *"**workflow_dispatch ONLY (no schedule)** … To make it automatic later, add a `schedule:`
block."* Line 37: `schedule: - cron: "20 4 * * 0"`. CLAUDE.md correctly says weekly. The file's own
docstring is stale.

### D5. Breadth-book numbers in STATUS.md are a week old and the conclusion has moved
`STATUS.md` (and the 2026-07-27 memory): breadth full band **+0.003 [−0.008,+0.015] on 603 entries
/ 98 city-days**; *"the one number still worth watching is breadth `outer` [5,20)¢ at +0.023
(t≈2.1) — but it is one of ~4 bands and all its graded entries fall in a 3-day window."*
**Today: 1,873 graded / 294 city-days. Outer [5,20)¢ = +0.0327 [+0.0202,+0.0453], t≈5.1;
deep [5,10)¢ = +0.0317 [+0.0192,+0.0442]; core [20,35)¢ = −0.0394 [−0.0624,−0.0164]
(now significantly negative); moderate [10,25)¢ = +0.0248 CI above zero.** The "hypothesis to
watch" has become the strongest measured result in the project, and no document says so.

### D6. The deep-shoulder band is described as under-priced in the spec and over-priced in the data
The moderate-gate spec excludes [5,10)¢ because *"deep shoulders hit more than priced"*
(n=29: 7.3% priced, 10.3% realized). The breadth memory repeats it: *"deep [5,10)¢ shoulders are
UNDER-priced by fat tails and cancel the moderate gains"* and *"deep loses on both (maker −11.2¢)"*.
**Breadth, n=555 / 254 city-days: deep [5,10)¢ realizes ~2.5% against ~7.5% priced, taker
+0.0317 CI [+0.0192,+0.0442].** These cannot both be right. The n=29 sample loses.

### D7. Capture-rate arithmetic disagrees across three documents
`STATUS.md` §5: *"we were only collecting **3%** of the markets"*. Memory
`discovery-volume-ceiling`: *"~**9%**"* (24/264 for the five cities). Spec §1: *"roughly 3%"*
(tag-wide: 264 city markets out of 1,994 weather markets). Both are defensible with different
denominators, but they are used interchangeably. The 11× multiplier is the reliable figure.

### D8. Settlement-audit misses are misattributed
`STATUS.md`: *"the standing misses are **Seoul/London** tenths-of-a-degree boundary cases."*
**Live: 2× HongKong (06-08, 06-11) + 2× Seoul (03-24, 07-19). No London.** The HK misses are
`lowest temperature` rows, i.e. a Tmin/HKO pattern, not the documented tenths-of-a-degree story —
worth an actual look rather than a footnote.

### D9. Headline scoreboard numbers differ across every document (expected, but unlabelled)
CLAUDE.md 0.128/0.160/0.166 (2026-07-13, n=240) · STATUS.md 0.128/0.160/0.166 + pooled +0.0178 ·
guide.html 0.1158/0.1360/0.1417 (n=261) · meta-analysis 0.1213/0.1502/0.1546 (n=264, station not
settlement truth) · **live today 0.1184/0.1376/0.1367 (n=421)**. The memory file already warns
*"numbers drift daily — always re-run `evaluate_oos.py`."* The risk is a proposal anchored to a
version that also carried a different *conclusion* (e.g. "model beats ensemble" flipped twice).

### D10. Minor
- `CLAUDE.md` says `collect.yml` commits *"`data/polymarket`, `data/weather`, `shoulder_paper.csv`"*
  — `output/shoulder_paper_breadth.csv` is also written each cycle (2,384 rows).
- `EDGE_MEGAPLAN.md` §6 data-bug 2 (*"`output/all_bins.csv` is unreadable/malformed"*) is still
  open; the file is untouched since 2026-05-14.
- The megaplan's READ-THIS-FIRST header is dated 2026-07-13 and describes §9–§10 as "current",
  while §12/§12a/§12b (2026-07-29) sit below it unreferenced.

---

# Appendix 1 — everything measured live for this report (2026-07-31)

```
data_status.py     421/150 gradable markets · 639/100 gradable bets · GATE MET
                   HK truth 31d behind (monthly batch, expected); others 1–2d
audit_settlements  129/133 = 97.0% (floor 95%); misses HK 06-08, HK 06-11, Seoul 03-24, Seoul 07-19
evaluate_oos       MODEL 0.1367 · MARKET 0.1184 · ENSEMBLE 0.1376 · QRF 0.1684 (M1 FAIL v 0.1628)
                   POOLED gap +0.0183 [+0.0045,+0.0321] t=2.60 over 178 city-days
                   ROI −11.4% / 421 bets · shrink sweep w*=0.15 (Brier 0.1177)
                   dispersion: Tmax std(z) 1.33 pooled — 1.78 Mar, 1.76 May, 1.08 Jun, 0.98 Jul
shoulder_book      280 entries (176 graded); full +0.034 CI[−0.009,+0.076]; core +0.059 n=76;
                   Leg1b forward 25/80 +0.034; Leg2 n=1
breadth            2384 entries / 50 cities (1873 graded); Leg1b forward 806/80 +0.0263
                   CI[+0.0080,+0.0447] v +0.030 → pending on effect size only
band decomposition + 2000-draw clustered bootstrap → see B1 table
accrual            5-city book 12/day → 43/day on 2026-07-30; breadth ~300/day
```

# Appendix 2 — the three rules that survive everything above

1. **Read the paired, clustered Brier gap. Never the ROI.** ROI's interval here is ~46 pp wide and
   swings 14 points overnight on one settlement.
2. **A pre-registered threshold prevents cherry-picking; it says nothing about power.** State every
   gate as (n, effect, **significance**), cluster on the city-day, and require out-of-sample
   replication before a pass counts.
3. **Assume the next measurement correction will also cost you.** Ten broken instruments, ten
   corrections, all in the same direction. Six of them were silent — a green run with a plausible
   wrong number is this project's dominant failure mode.
