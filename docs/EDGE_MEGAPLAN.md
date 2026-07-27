# Raincheck — Edge Megaplan (2026-07-11)

A strategy for actually gaining edge, grounded in a fresh decomposition of the honest, gate-met
evaluation (n=211 station-graded markets, 302 bets, Brier: market **0.128** < model **0.163**).
Companion docs: `STATUS.md` (plain-English state), `CLAUDE.md` (technical reference),
`docs/BUGFIX_EXECUTION_REPORT.md` (the 2026-07 bug-fix pass this work builds on).

Everything below is driven by new evidence produced on 2026-07-10/11 from the graded tracker,
the CLOB price histories, and the snapshot store (analysis scripts summarized in §11).

> **⚠️ READ THIS FIRST — state as of 2026-07-13.** This document is chronological; §0–§8 hold
> the 2026-07-10/11 evidence and several of their headline numbers were later corrected by the
> **settlement-truth fix (§10a)** — most notably, the "NYC same-day beats the market" pocket
> was a grading artifact and is dead. The living state (2026-07-13): market Brier **0.128** <
> ensemble 0.160 < model 0.166 (n=240); model book OFF (ROI −8.7% and day-to-day noisy — the
> Brier gap is the verdict); E3 nominations Seoul|1d + Chicago|1d
> (paper, forward gates from 2026-07-12); the **two-leg structure paper book** (§10b shoulders
> + §10d favorites, real fees verified §5-E1) is the nearest-term positive-ROI candidate; top
> queued work = W0.2 settlement-truth retrain, then §9's order. Sections §9–§10 are current.
> Progress snapshot 2026-07-13: gate 240 markets / 354 bets (met); settlement audit **60/61**;
> structure book 15 paper entries (0 graded yet); forward gates Seoul|1d 1/40, Chicago|1d 0/40.

---

## 0. TL;DR — the strategy in five sentences

1. The decomposition shows the market is **almost perfectly calibrated on exactly the bets we
   flag** — our disagreements with it are mostly our own mu (center) errors, so *generic*
   "forecast better, bet the difference" is a dead end at our current skill.
2. But the loss is **not uniform**: NYC same-day already **beats** the market (Brier 0.084 vs
   0.114, n=27), London same-day forecast-only is at parity, while Chicago same-day
   (2.1 °C mean-forecast error) and the *current* intraday conditioning in Seoul/London are the
   big drags. Edge is a **pockets** game, not a global game.
3. Therefore: stop betting the whole surface; run three narrow **books** — (A) a same-day
   *nowcast sniper* built on fresh station obs (the NYC recipe, industrialized), (B) a
   *run-release event trader* that acts in the minutes after new NWP runs land, (C) a
   *coherence/structure* book that arbs bin-sum incoherence without taking a forecast view.
4. In parallel, stop the bleeding globally: make sigma **adverse-selection aware** (realized z-std
   on flagged bets is 1.41 — our sigma is honest on average days but 40% too small on the days we
   bet), and bet only in buckets with a demonstrated, pre-registered edge.
5. Execution economics get their own workstream: verify the real Polymarket fee (config assumes
   2% on winnings) and switch to **maker-first** order placement — on a ~53%-win book, spread +
   fee assumptions are worth several ROI points on their own.

---

## 1. What the decomposition established (evidence, 2026-07-10/11)

### 1a. The market is calibrated *on our flagged set*; our deviations are noise
Splitting the 211 graded markets by the direction of disagreement:

| direction (abs(model−mkt)>0.05) | n | mean model | mean market | realized |
|---|---|---|---|---|
| model **above** market | 71 | 0.273 | 0.112 | **0.070** |
| model **below** market | 140 | 0.123 | 0.264 | **0.257** |

In both directions the realized frequency lands on the **market's** number, almost exactly.
And the bigger the disagreement, the worse we do: at |Δ|∈(0.05,0.1] we win 69% of rows (Brier
tie); at (0.2,0.35] we win 34%; at >0.35 we win 22%. **Big flags are our errors, not theirs.**

### 1b. Losses are center misses, not tail events — and flagging *selects* our bad days
- Median |bin − mu|/sigma over the 30 worst rows: **1.15** (center misplacement, not >2σ tails).
- Realized z = (mu − actual)/sigma on the flagged set: mean ≈ 0, but **z-std = 1.41**
  (Chicago 1.68, Seoul 1.52, NYC 1.33, London 1.07). Sigma is honest unconditionally (that's how
  it was fit) but conditional on "we disagree ≥6 pts with a calibrated market", our error is ~40%
  bigger than sigma claims. This *is* the calibration-table anomaly (model says 0.04 → realized
  0.19; model says 0.24–0.34 → realized 0.10–0.19). It's adverse selection, and it's fixable
  (§4, W2) — not by forecasting better, but by pricing our own uncertainty honestly *conditional
  on having flagged*.

### 1c. The gap by slice (gap = model Brier − market Brier; positive = we lose)

| slice | n | gap/row | note |
|---|---|---|---|
| Chicago same-day | 17 | **+0.114** | worst bucket; same-day mu MAE 2.07 °C (London: 0.76!) |
| Seoul same-day (intraday rows) | 10 | **+0.141** | intraday conditioning misfiring |
| London 1d | 23 | +0.041 | |
| Seoul 3d | 7 | +0.125 | March forecast busts, small n |
| **NYC same-day** | 27 | **−0.031** | we BEAT the market; intraday subset n=5: −0.099 |
| London same-day (non-intraday) | 26 | −0.004 | parity |
| Seoul same-day (non-intraday) | 13 | +0.006 | parity |

- Same-day is ~46% of graded markets (97/211) — the biggest book by volume.
- By local hour, the same-day gap is worst at **14:00–18:00 local** (+0.078/row): exactly the
  hours where the outcome is being decided by observations we hold 1–3 h stale (2 h collector
  cadence) while market participants watch live. The market's same-day advantage looks like
  **fresher eyes, not better meteorology** — a bot can out-fresh humans.
- Intraday-sourced rows overall (n=20): the WORST sigma_source bucket everywhere except NYC.
  The most-promising lever is currently mis-tuned (diagnosis in §4 W1).
- Chicago same-day errors are warm-biased on hot days (e.g. 07-05 mu 27.7 vs actual 24.4;
  06-10 mu 35.1 vs 32.8) — classic **lake-breeze cap** days. NYC (KLGA) has the sea-breeze
  analog but currently works; Chicago needs the regime feature (§4 W3).

### 1d. Timing: the market absorbs forecast news in ~2–6 h; our flags carry no lead
- After our flags, the market does NOT converge toward us (52–56% toward, signed convergence ≈ 0,
  negative at 24 h). Our signal contains ~nothing the market later learns. (206 matched events.)
- After a ≥0.5 °C forecast shift between consecutive 2 h snapshots, sibling-bin prices drift in
  the implied direction by +1–2¢ over 12–24 h (1–2 °C shifts: +2.3¢) — real but ≈ round-trip
  cost at taker economics, and partly forecast-revision autocorrelation. At a **2 h** cadence
  there is no exploitable latency; minutes-after-release there may be (§5 Book B).
- CLOB prices change about hourly (median inter-change gap 1.0 h at 1 h API fidelity) — these are
  active books during the day, not dead ones.

### 1e. The hybrid experiment: market center + our shape ≈ the market. Shape adds nothing.
On 86 graded markets where sibling bins allowed extracting the market-implied center:

| predictor | Brier |
|---|---|
| market (raw price) | 0.1417 |
| model (as-is) | 0.1842 |
| **market-implied mu + model sigma/nu** | **0.1406** |
| 50/50 center blend + model sigma | 0.1559 |

The market's prices are already consistent with (its center + a sane dispersion): its implied
sigma is ~1.0 °C (London) / 1.2 °C (Seoul) at these mostly short-lead snapshots — *tighter* than
ours and right. Model-market center disagreement averages 1.9 °C, and the market's center wins.
**Conclusion: there is no cheap "better tails" edge around the market's center. Either bring a
better center in specific pockets, or don't take a level view at all.**

---

## 2. The strategic reframe

The old frame — "forecast the temperature better than the market, everywhere, then Kelly" — is
falsified at our current skill (§1a, §1e). The new frame:

> **Bet only where we are structurally faster or structurally cleaner than the marginal market
> participant; elsewhere, don't trade.**

Three structural advantages a bot can actually have over this market's participants:
1. **Freshness** — machines don't sleep and can poll obs/models at minute cadence
   (Book A: same-day nowcast; Book B: run-release events).
2. **Coherence** — humans price bins one at a time; a machine prices the whole PMF
   (Book C: bin-sum and relative-value structure).
3. **Honest uncertainty** — we can measure our own conditional error and refuse bets the market
   is better at (W2 + per-bucket gating: the "edge of knowing you have no edge").

Everything in §3–§6 serves one of those three.

---

## 3. The three books (what we actually trade)

### Book A — Same-day nowcast sniper (highest conviction; NYC proves the ceiling)
**Thesis:** from late morning the day's max is progressively "locked in" by observations. With
minute-fresh METAR/ASOS and a proper heating-curve nowcast, Tmax at 14–18 h local is predictable
to a few tenths °C while bins still trade at 10–40¢. The market's afternoon sharpness is humans
refreshing dashboards; we can be systematically fresher. Evidence: NYC same-day already beats
the market with today's half-broken plumbing (§1c); the same-day gap is concentrated exactly in
the stale-obs hours; the market's implied sigma (~1 °C) leaves room for a sub-0.5 °C nowcaster.
**Trades:** afternoon Tmax bins (and morning Tmin bins — the overnight low locks in even
earlier; London Tmin sigma@12 ≈ 0.7 °C already) in NYC, Chicago, London, Seoul.
**Prereqs:** W1 (obs latency + intraday overhaul), W3 (Chicago lake-breeze), E2 (maker-first).
**Gate (pre-registered):** live paper-trade ≥60 same-day bets; require model Brier ≤ market
Brier − 0.01 on those bets before real size.

### Book B — Run-release event trader
**Thesis:** new NWP runs land on a known schedule (ECMWF 00Z ~07:00 UTC, 12Z ~19:00 UTC; GFS
~+3.5 h per cycle; HRRR hourly for US). The market absorbs forecast shifts in ~2–6 h (§1d).
A scanner that compares the *new* run's Tmax vs the *pre-release* market within minutes of
release, and trades only shifts ≥1 °C toward a specific bin, captures the 1–3¢ absorption drift
— and, unlike Book A, needs no forecasting skill at all, only speed. Exit into absorption
(hours), don't hold to resolution.
**Honesty check:** at taker costs this was ≈ breakeven at 2 h cadence; the bet is that
minute-cadence + maker entry turns +1–2¢ gross into +2–4¢ net. Small edge, high frequency,
low correlation with Book A.
**Gate:** 4 weeks of logged signals (no orders) → measured post-signal drift must exceed
2× assumed costs before any capital.

### Book C — Coherence/structure book (no forecast view)
**Thesis:** sibling bins are priced by different people at different times; the PMF often sums
to ≠ 1 (α5 already measures this; today it's only a score bonus). When Σ(YES asks) < 0.97 across
an exhaustive partition, buying the set is a near-riskless basket (one bin must resolve YES);
when Σ(YES bids) > 1.03, selling the set is. Also relative-value *within* the PMF: enforce
monotone, unimodal-ish mass around the market's own center and trade the outlier bin against
its neighbors — never against the center (§1e says the center is right).
**Prereqs:** order-book depth data (we currently store only mid/last price — need best bid/ask
per bin in the snapshot store), E2 execution plumbing. Strictly liquidity-gated
(`COHERENCE_MIN_LIQ` already exists).
**Sizing:** small but nearly market-neutral; the payoff is also informational — it tells us
which bins are lazily priced (feeds Book A/B targeting).

**Explicitly NOT a book:** long-lead (2 d+) value betting on our forecast level. §1 says we
lose there; the per-bucket gate keeps it off until some future model change flips its bucket.

---

## 4. Model workstreams (make the pockets winnable)

### W1 — Obs pipeline + intraday overhaul (serves Book A; the single biggest lever)
The intraday conditioner is the right idea currently mis-tuned: outside NYC it *adds* Brier
(§1c). Diagnosis targets, in order:
1. **Obs freshness at serve:** collector runs every 2 h; METARs are published ~hourly with ~5–10
   min latency (IEM/aviationweather), and KLGA/KORD have 1–5-min ASOS feeds on IEM. During
   station-local 09:00–20:00, poll obs every 10–15 min and re-price. This alone converts the
   14–18 h "stale hours" loss (§1c) into our best hours.
2. **Per-hour sigma honesty:** the failure cases (Seoul 06-28: intraday mu 29.3±1.0, actual
   27.0; London 07-03: mu 25.4±0.56, actual 27) are *morning/midday* hours where the per-hour
   regression trusts the forecast with far too small a sigma. Apply the per-lead trick per hour:
   floor each hour's sigma at its **honest holdout residual std**, never gated; self-gate the
   *mean* only. Require the obs coefficient (c_h) to be materially informative before the
   regression replaces EMOS at that hour.
3. **Local-hour audit:** verify train↔serve hour alignment for BST (London DST) and the
   "last completed hour" convention post-C4/C5; one off-by-one hour misapplies every fit.
4. **Heating-curve nowcast (v2 of intraday):** replace per-hour linear fits with a remaining-heat
   model: ΔT_remaining ~ f(local hour, month, cloud/shortwave forecast for the rest of day, wind,
   running-max trajectory), fit as quantile regressions on 3+ years of hourly obs
   (`fetch_station_obs` history) × archived hourly forecasts. This is where sub-0.5 °C afternoon
   sigma comes from, physically.
5. **Tmin mirror:** same for the morning lock-in of the overnight low (ceiling logic exists).

### W2 — Adverse-selection-aware dispersion — **TESTED 2026-07-11: partial negative**
Sigma is right on average and 41% too small on flagged days (§1b). The replay
(`w2_replay.py`, frozen n=220 set) tested sigma_bet = sigma · min(1+γ·|p_model−p_mkt|, 1.8):
- Brier improves monotonically (0.1612 → 0.1538 at γ=4) and the [0,0.1) calibration bucket
  moves 0.039→0.19 to 0.051→0.16 — the distribution gets more honest…
- …but **ROI does NOT improve** (−20.3% → −21/−23% across γ; bets only drop 206→172). Widening
  sigma cannot fix a *center* problem: the surviving flags are still the same wrong-center book.
  **Fails its §8 pre-registered ROI criterion → do not ship as the bleeding-stopper.**
Disposition: keep a moderate γ≈2 as a *sizing-honesty* layer only (applied at the edge/Kelly
stage like `SHRINK_WEIGHT`, with the tracker still storing the PURE model prob for eval
integrity), and let **E3 bucket gating carry the defense** (tested same day, see §5 E3).

### W3 — Regime features for the two known biases
- **Chicago lake-breeze cap:** detector = forecast onshore component (E/NE) at KORD hours 10–18
  + Lake Michigan water-vs-air differential (NOAA GLERL buoy data) + weak synoptic gradient →
  cap/skew the Tmax distribution (or add the regime flag to the EMOS mean/sigma). Backtestable
  today against 2022→now archived leads + truth.
- **Seoul flow-conditional anchor:** Bucheon-vs-RKSI anchor chosen (or blended) by forecast
  wind direction — the fixed inland anchor wins on average but should lose on onshore days.
  Test on archived leads for both points (needs an airport-cell previous-runs backfill; the
  fetcher exists).

### W4 — Resolution-mechanics audit (protects the tails; cheap)
- Verify the **climate-day boundary** per station (CLI = local *standard* midnight for KLGA/KORD;
  METAR daily for EGLC/RKSI as IEM computes it) and the whole-degree rounding conventions vs
  `pmf`'s half-width integration, in both °F and °C cities.
- Root-cause the outliers that look like alignment bugs, e.g. Chicago 2026-03-22 (mu 9.9 °C,
  actual 20.6 °C at lead 0.5 — a 10.7 °C "miss" more likely a wrong-day/stale-forecast row) and
  the `lte`/`gte` bucket (n=13 but the largest per-row gap of any qtype: +0.13/+0.06).
  One systematic day-boundary error at the tails is many Brier points.

### W5 — Regime-conditional tails (after W2; ablate against it)
Widen tails on high-volatility days using features we already store: run-to-run drift
(`total_drift`), ensemble spread, frontal-passage signature (large hour-to-hour forecast temp
gradient). Only keep if it adds on top of W2 — they overlap (both widen when uncertain).

---

## 5. Execution & economics workstreams

### E1 — Fee/reward truth — **VERIFIED 2026-07-13**
Polymarket moved to a maker-taker schedule on 2026-07-01: weather taker fee =
**0.05·p·(1−p) per share** (max 1.25¢ at 50¢, ~0.24¢ at 95¢), **makers pay nothing** and the
Maker Rebates Program returns **25%** of collected weather taker fees to makers daily
(help.polymarket.com "Trading fees"; docs.polymarket.us/fees). Encoded as
`config.taker_fee_per_share()`; `shoulder_book.py` settles with it. The legacy
`FEE_RATE = 0.02` (2% of winnings) stays in the model-book backtesters as a conservative
overstatement — migrating them is queued. Consequence: maker-first execution (E2) is
decisively favored, and near-extreme prices are far cheaper to trade than assumed.

### E2 — Maker-first execution
Never cross the spread on entry by default: post at our price (or 1 tick inside), TTL-cancel on
any forecast/obs update (we are the informed side intraday post-W1; resting quotes must be
pulled when our signal moves). Taker only for Book B releases and Book A late-afternoon
"lock-in" bets where immediacy is the edge. Paper mode first via `py-clob-client`.

### E3 — Per-bucket selective aggression (the kill-switch layer) — **now the primary defense**
Define buckets = (book × city × lead/hour band). Each bucket trades only while its rolling
out-of-sample Brier beats the market's on the same bets (pre-registered n per bucket, e.g. 60).
Buckets start in paper mode. This layer is what lets us run aggressive experiments without
re-learning the −20% lesson.

**Frozen-set preview (2026-07-11, n=220 — IN-SAMPLE nomination, NOT a result):** restricting
the book to the two buckets whose model Brier beats the market — NYC|same-day (0.084 vs 0.114,
n=27) and Seoul|1d (0.119 vs 0.128, n=21) — flips ROI from **−20.3% (206 bets)** to **+16.8%
(48 bets, 68.8% win)**. These buckets were chosen on the same data that grades them, so this
number only *nominates* the hypothesis; the forward paper-trade gate (§8) confirms or kills it.
All other city×lead buckets grade off (Chicago|2d+ and HK|2d+ pass but with n=9 and n=1).
*Update 2026-07-12:* the first regeneration (n=236, E2 date-fix active) already shrank the
ON-book to **+2.1% ROI / 55 bets** (NYC|same-day 0.110 vs 0.121, Seoul|1d 0.123 vs 0.126) —
the in-sample margin was partly noise, exactly what the forward gate is for.
*Update 2026-07-12 (settlement-faithful labels, W0):* **NYC|same-day is DEAD** — its edge was
concentrated on exactly the boundary days our old labels got wrong (corrected: 0.175 vs market
0.120; old ON-book −21.5% ROI). Nominations revised to the corrected set's only passers,
**Seoul|1d (0.123 vs 0.126) and Chicago|1d (0.106 vs 0.124)** — both marginal, both paper-only,
forward clock restarted (`E3_NOMINATION_DATE = 2026-07-12`). A nomination that flips three
times in two days is the strongest argument yet for trusting nothing but the forward gate.

### E4 — Sizing v2
Replace the scalar `SHRINK_WEIGHT` with the W2 market-as-evidence posterior, keep fractional
Kelly + group/portfolio caps. Kelly inputs must use post-W2 probabilities (they're the honest
ones). No other sizing work — the bottleneck was never sizing.

---

## 6. Data & infrastructure additions

| addition | serves | note |
|---|---|---|
| METAR/ASOS polling 10–15 min during local day | Book A / W1 | IEM 1–5-min ASOS for KLGA/KORD; aviationweather for EGLC/RKSI |
| Run-release-aligned fetch scheduler | Book B | poll Open-Meteo minutes after ECMWF/GFS/HRRR/ICON publish times, not on a flat 2 h clock |
| Best bid/ask + depth per bin in snapshots | Book C, E1/E2 | we currently store only mid/last; spread assumptions are guesses until this lands |
| Lake Michigan / NYC harbor water temp | W3 | NOAA GLERL / NDBC buoys |
| NWS point forecast + forecast discussion (US) | W3/analysis | if the market herds on NWS, our blend-vs-NWS delta marks exactly where we can win or must stand down |
| HK: faster HKO daily feed or drop HK | hygiene | truth lag 40 d; n=4 graded; not worth much effort |

**Data bugs found during this analysis (fix soon, they bite silently):**
1. `*_price_history.csv` timestamps are epoch-seconds ÷1000 → all dates read 1970-01-21
   (fetcher treats seconds as ms). Recoverable (×1000), but fix the fetcher and backfill.
2. `output/all_bins.csv` is unreadable/malformed (pandas can't parse it).
3. CLOB history retention: only ~10 days of ticks came back for July — if the API window is
   short, we should snapshot order books ourselves going forward (see depth item above).

---

## 7. Sequencing (what to do, in order)

| # | work | effort | why now |
|---|---|---|---|
| 1 | W2 adverse-selection sigma + E3 bucket gating | days | stops the bleeding; makes everything after it safe to paper-trade |
| 2 | W1 obs/intraday overhaul (items 1–3, then 4) | 1–2 wk build | Book A is the only pocket with *demonstrated* upside (NYC); backtest can't test it — must go live-paper |
| 3 | Book A live paper in all cities | 3–4 wk measure | pre-registered gate: ≥60 same-day bets, model ≤ market − 0.01 Brier |
| 4 | E1 fee/reward audit + E2 maker plumbing | parallel, days | pure economics; needed by every book |
| 5 | W4 resolution-mechanics audit | 1–2 days | cheap; protects tails; explains the grotesque outliers |
| 6 | W3 Chicago lake-breeze + Seoul flow anchor | ~1 wk | backtestable offline on existing archives |
| 7 | Book B release scanner (signals-only month first) | ~1 wk | needs W1's cadence infra |
| 8 | Book C structure bot | ~1 wk | needs depth data + E2 |

Explicitly deprioritized: more blend-model shopping (NBM/CMA/BOM negatives stand), optimizer
re-tuning (bottleneck is signal, not params), long-lead value betting (falsified for now), HK.

## 8. Pre-registered success criteria (so we can't fool ourselves again)

- **Global:** no live capital until some bucket passes its gate; `evaluate_oos.py` stays the
  arbiter and grows a `--by-bucket` view (market Brier vs model Brier per bucket, paired).
- **Book A:** ≥60 live-paper same-day bets, model Brier ≤ market − 0.01 → smallest real size;
  re-check at 150.
- **Book B:** ≥100 logged release signals; mean post-signal aligned drift ≥ 2× modeled cost.
- **Book C:** ≥30 executed (paper) baskets with Σ-mispricing ≥ 3¢ after depth-checked fills.
- **W2:** on the frozen 2026-07-10 graded set, calibration table's [0,0.1) bucket realized rate
  inside [pred−0.05, pred+0.05], and backtest ROI at production params improves ≥ 8 pp with bet
  count within ±40% (it must reprice bets, not just veto everything).
  **→ TESTED 2026-07-11: FAILED the ROI leg** (Brier improves, ROI doesn't — see §4 W2).
  W2 demoted to a sizing-honesty layer; E3 bucket gating is the defense.
- **E3 ON-buckets (NYC|same-day, Seoul|1d):** in-sample nomination +16.8% ROI/48 bets. Confirm
  forward: ≥40 new graded bets per bucket with model Brier ≤ market Brier before any real size;
  a bucket that drops below market Brier on its rolling last-60 goes back to paper.
- **Any new model change:** ships only with a paired Brier/CRPS win on the holdout AND no
  degradation of the per-bucket table. Gate constants stay in `data_status.py`.
- **POWER AMENDMENT (pre-registered 2026-07-27, tightening only — applies to every gate here).**
  A gate now passes only when its **clustered 95% CI excludes zero** over ≥30 independent
  city-days, in addition to its existing (n, effect) thresholds. No threshold was moved, and
  nothing failing an old gate can pass the new one (`shoulder_book.gate_verdict`,
  `GATE_MIN_CLUSTERS`; 7 tests). **Why:** on 2026-07-27 §10b's full band MET its gate
  (n=150, +0.0234 ≥ +0.020) with CI **[−0.023, +0.070]** — met and no-edge were
  indistinguishable — while the independent breadth book (49 cities, 603 graded, 98 city-days)
  put the same band at **+0.003 [−0.008, +0.015]** and **flipped the sign of the core band**
  (+0.049 → −0.028). A pre-registered threshold stops post-hoc cherry-picking but says nothing
  about POWER: detecting a true 2¢ edge at 80% power needs ~1,760 independent bets. State
  future gates as (n, effect, significance), cluster on the true unit of independence (a
  city-day = one weather outcome, not a bin), and require out-of-sample replication before a
  pass counts. **Gate amendments are tightening-only and dated; a threshold is never loosened
  to fit a result.** Under the amendment no structure gate currently passes.

## 9. Re-sequenced queue after the second wave (2026-07-12)

The §10a grading audit changes the order of §7: nothing outranks fixing the arbiter's labels.

| # | work | status / supersedes |
|---|---|---|
| 1 | **W0: WU-faithful truth channel + re-grade** (§10a) | ✅ DONE 2026-07-12 (audit 59/60); spawned **W0.2: retrain EMOS/Tmin/intraday against settlement targets** — now the top model task |
| 2 | Shoulder-premium PAPER book (§10b) | ✅ DONE 2026-07-12 (`shoulder_book.py`, auto-records each cycle; gates §10b) |
| 3 | Kalshi archiver (§10e-1) — start collecting NOW, analysis later | next new-data item |
| 4 | W0.2 settlement-target retrain, then E3 forward gates (running, revised nominations) | replaces the dead NYC nomination path |
| 5 | W1 obs/intraday overhaul → Book A live paper | unchanged |
| 6 | 1-min ASOS + boundary-day settlement model (§10a-2, §10e-2) | folds into Book A/W1 |
| 7 | Everything else in §7's order | |

## 10. Second wave — new edge veins (2026-07-12, with first test results)

Ideas not derived from the handoff's leads; each attacks the market's *mechanics* rather than
out-forecasting it. Three were tested same-day on stored data.

### 10a. ⚠️ Resolution-source fidelity — TESTED, and it's a real problem AND a real edge
The markets resolve on **wunderground.com** station pages (`resolution_anchors.resolution_url`),
not on the NWS CLI / IEM METAR our truth feed reads. Auditing our grades against *actual
settlements* (last pinned post-day snapshot price as the settlement proxy):
**56/60 agree — 4/60 (6.7%) of our labels are WRONG**, all at whole-degree boundaries
(Chicago 05-28 72.0°F; NYC 06-27 Tmin 69.0°F; NYC 07-03 Tmin 79→WU said 80; Seoul 03-24 14.0°C).
Consequences, in order:
1. **Eval integrity:** ~7% label noise poisons every Brier/ROI number (and likely *understates*
   the market, which prices the WU reading exactly). Fix before trusting any gate: add a WU-
   faithful truth channel (reconstruct WU's daily min/max algorithm — data window, rounding,
   local-clock vs LST midnight — or scrape the page the oracle reads), grade against it,
   regenerate. This supersedes W4 and jumps to the FRONT of the queue.
2. **Boundary-day edge:** whoever models the WU-vs-CLI divergence (tie rounding, the midnight
   window, WU's provisional-then-revised values) knows the *settlement* distribution better than
   traders watching METARs/NWS. Every bin is 1°F wide — boundary days are common, and they are
   exactly the days markets stay uncertain longest.
3. **Settlement risk control:** never hold a boundary-day position into resolution on CLI logic.

**→ SHIPPED 2026-07-12** (`wu_truth.py` + `audit_settlements.py`, wired into
`grading.fetch_actual_weather` with CLI fallback + glitch guard): the hourly-METAR
reconstruction over the local calendar day matched the real settlement in all three US
disagreement cases; the audit now reads **59/60 (98.3%)**. The remaining miss is Seoul
2026-03-24 (WU likely ingests SYNOP tenths for RKSI — hourly METARs can't reproduce it), so
RKSI/EGLC stay on the IEM-daily feed with the boundary risk documented.
**Corrected-label fallout (why this ranked #1):** market Brier drops to **0.1224**, model rises
to 0.1646, and — critically — **NYC|same-day's "edge" was a grading artifact** (0.084→0.1746
vs market 0.1202; the old ON-book's +16.8% becomes −21.5%). Under settlement labels the
calibrator also now loses the paired check to the raw ensemble (0.171 vs 0.159) —
**W0.2 follow-up: retrain EMOS (and Tmin/intraday fits) against settlement-faithful targets**,
i.e. wu-reconstructed actuals for NYC/Chicago across the whole 2022→now training archive.

### 10b. Shoulder-bin premium (full-book calibration) — TESTED, promising
First-ever calibration of the market over ALL bins (not just model-flagged): n=885 pre-day
priced bins, 50 city-dates. Deep tails (≤2¢) and favorites (35–65¢) are fairly priced, but the
**"shoulder" bins at 5–35¢ are overpriced by +2 to +6¢/share** (5–10¢: price 0.076 vs realized
0.053; 10–20¢: 0.149 vs 0.126; 20–35¢: 0.271 vs 0.211, n=147). Interpretation: retail spreads
mass across "plausible-looking" neighbors of the favorite; the overround concentrates in the
shoulders. Structure trade: **sell the shoulders / buy the mode** (or buy the NO basket where
the book sums > 1) — no forecast skill required. Caveats: bins within a market are correlated
(true n ≈ 50 dates, the 20–35¢ gap is ~1.8 se), and mid-price fills overstate what a taker
gets — needs the E1/E2 depth data and a maker-first sizing pass. Pre-register: ≥150 forward
shoulder-bin observations, sell-edge ≥ +2¢ after spread.

**→ SHIPPED as a PAPER book 2026-07-12** (`shoulder_book.py`, auto-recording each collector
cycle via main.py; append-only `output/shoulder_paper.csv`; grades via the settlement-faithful
truth). Re-tested under corrected labels the premium *sharpens and localizes*: **20–35¢ bins
+8.1¢/share** (price 0.271 vs realized 0.190, n=147), 5–10¢ +2.3¢, while 10–20¢ is dead (−0.5¢)
and favorites are slightly *under*priced (−2.0¢) — coherent with mass migrating from mode to
shoulders. Pre-registered forward gates (declared before any forward data): full band
[5,35)¢ ≥150 graded entries at ≥ +2¢ net; core [20,35)¢ ≥80 at ≥ +3¢ net. No real orders until
a gate passes; the 10–20¢ segment is recorded for honesty but expected ~zero.

### 10c. Post-lock settlement lag — TESTED, mostly efficient (one live counterexample)
In the 24 h after the outcome is physically locked, 10/11 markets with tick data sat within a
median 0.1¢ of terminal — no systematic free money at this sample size. The 11th was the NYC
07-03 WU-divergence market (§10a), i.e. the "lag" was actually OUR label being wrong. Keep as a
passive scanner (alert when |price − known outcome| > 3¢ after day-end), not a book.

### 10d. Favorite-longshot structure — TESTED 2026-07-13 (the "easy wins" hypothesis, refined)
Hypothesis (user): favorites priced ~90% are really ~95% because nobody bothers with easy wins.
Tested on a 1,646-row panel (every market × time-band last snapshot, settlement-faithful labels):
- **The naive version is dead:** favorites priced 85–97¢ realize almost exactly their price
  (pre-day 0.920 realized vs 0.922 priced, n=238) — after spread they're ~breakeven.
- **The refined version is real:** the **65–75¢ favorite band is underpriced everywhere** —
  pre-day realized 0.807 vs 0.710 (n=109), day-early 0.824 vs 0.701 (n=34); ≈ **+7.7¢/share
  net at verified taker fees**. This is the mirror image of the §10b shoulders (a 70¢ favorite
  IS a 30¢ shoulder seen from the other side) — two independent measurements of one mispricing:
  the market over-buys "plausible" 25–35¢ outcomes and under-buys their complements.
- Near-coin-flips (50–65¢ favorites) are NEGATIVE (that "favorite" label is noise) and mid-day
  favorites tested negative — the leg is restricted to >12 h before the local day ends.
- Deep favorites (97–99.5¢) show +0.6¢ raw pre-day; at the verified fee (~0.05–0.2¢ up there)
  this is a *maker-only* settlement-carry candidate — parked behind E2.
**→ SHIPPED 2026-07-13** as Leg 2 of `shoulder_book.py` (now a two-leg *structure* book): buys
YES-side favorites at 65–85¢ (>12 h to day end), settles with the real 0.05·p·(1−p) taker fee.
Pre-registered gate (declared before any forward data): core [65,75)¢ ≥80 graded entries at
≥ +3¢ net; outer [75,85)¢ tracked without a gate.

### 10e. Untested but high-plausibility (ranked)
1. **Kalshi cross-market** — Kalshi runs the same city temp contracts settling on the same
   reports, with a different (more pro?) crowd and a free API. Use as leading signal, cross-book
   arb when bins diverge, and as a second calibration reference. Cheapest new data source with
   the highest ceiling; start archiving it immediately alongside our snapshots.
2. **1-minute ASOS sniping (US)** — hourly-METAR watchers can't see between-obs spikes; the
   CLI/WU max can. IEM serves 1–5-min ASOS for KLGA/KORD. On days the 1-min series crosses a
   bin boundary the hourly series didn't, we know the settlement moved and most of the market
   doesn't. Pairs with Book A infrastructure (W1).
3. **Bimodal-ensemble mixture pricing** — the hybrid test (§1e) only rules out a better
   *unimodal* shape. On frontal-timing days the 122-member ensemble is bimodal and the true PMF
   two-humped; a mixture PMF prices between-hump bins below and hump bins above a unimodal
   market. Requires storing member-level (or quantile) ensemble output, then a dip-test trigger.
4. **Universe expansion** — Polymarket lists more weather cities (Miami, Austin, Phoenix, LA,
   Dallas…) with thinner, softer books; our whole stack (CLI/WU truth, EMOS, buckets)
   generalizes. Rank all weather markets by sharpness (spread × volume × tick rate) and enter
   the softest three first. The edge may be *where we play*, not how well.
5. **Upstream-station nowcasting** — lake-breeze/sea-breeze fronts announce themselves at
   shoreline stations (GLERL buoys, KMA AWS between Incheon coast and RKSI) 1–2 h before they
   cap the resolution station; nobody trading these books watches those feeds. Feeds W3.
6. **Bin-addition mechanics** — when a forecast shifts enough that Polymarket adds a new bin to
   the event, sibling prices reprice mechanically and lag; detectable from our snapshots
   (new condition_ids appearing mid-event).
7. **Chicago→NYC synoptic sequencing** — the same front hits KORD ~a day before KLGA; Chicago's
   *realized* outcome updates NYC's next-day distribution before NYC's crowd reacts. Testable
   offline from truth + leads archives.

## 11. Provenance of the numbers in this doc

All from 2026-07-10/11 analysis of `output/opportunities_evaluation_calibrated.csv` (graded via
`grading.resolves_yes` against station truth), `data/polymarket/*_snapshots.csv`,
`data/polymarket/*_price_history.csv` (timestamps recovered ×1000), using `evaluate_oos.py`
helpers. One-off scripts (scratchpad, not committed): Brier-gap decomposition, same-day/intraday
split, flag-convergence lead-lag, forecast-shift→price-drift event study, and the
market-center+model-shape hybrid. Headline eval numbers match `python evaluate_oos.py` output on
that date (model 0.1633, market 0.1278, ensemble 0.1659; ROI −20.2% on 199 bets).
