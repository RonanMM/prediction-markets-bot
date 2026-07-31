# Empirical opportunity scan — model-free structure in the Raincheck market data

Date: 2026-07-31. All work read-only against `src/polymarket_weather/data/` and `output/`.
Scripts: `scratch/rc_lib.py`, `a0_structure.py`, `a1_coherence.py`, `a1b_coherence_detail.py`,
`a2_flb.py`, `a3_a4_a5.py`, `a4b_drift.py`, `a6_hk_grading.py`, `a6b_shoulder_hk.py`,
`a7_breadth_bands.py`, `a7b_breadth_stress.py`, `a7c_breadth_exec.py`.

Inference is `stats_util.interval` / `cluster_key` throughout — **clustered by city-day**, the
repo's own estimator. Grading goes through `grading.resolves_yes` (settlement-faithful, with
`wu_truth` for NYC/Chicago) except where the breadth book's own `settled_outcome` (Polymarket's
terminal pinned price) is the stated truth channel. Costs are always the repo's:
`config.HALF_SPREAD = 0.01` crossed on entry plus the **verified** weather taker fee
`0.05·p·(1−p)` per share; the legacy 2%-of-payout model is reported where it changes a verdict.

---

## 0. Data inventory and one prerequisite finding

| source | span | size |
|---|---|---|
| `*_snapshots.csv` (5 cities) | 2026-03-17 .. 07-30 | 6,969 rows / 2,714 markets (median **2** snapshots per market) |
| `*_price_history.csv` (CLOB, hourly) | 2026-07-01 .. 07-30 | 94,897 Yes-leg rows / 1,791 markets |
| `output/shoulder_paper.csv` (5-city book) | entries 07-12 .. 07-30 | 280 entries, 175 graded |
| `output/shoulder_paper_breadth.csv` (49 cities) | entries 07-23 .. 07-29 | 2,384 entries, 1,829 graded shoulder |

**Bin-set completeness is the binding constraint on all coherence work.** Of 547
(city, target-date, kind) books in the 5-city snapshot history, only **40 have a complete
partition** (an `X or below` bin, a contiguous tiling, and an `X or higher` bin). Median capture
is **4–5 bins**. This is the documented discovery ceiling. Days collected after the 2026-07-30
fix are complete (11 bins). Everything below that needs exhaustiveness is therefore built on
~26 books over 5 target dates; everything that needs only *mutual exclusivity* uses the full
sample.

A partition must contain exactly one winning bin. On fully-graded books with ≥4 bins:

| city | books | mean bins | mean YES per book |
|---|---|---|---|
| Chicago | 38 | 6.8 | 0.79 |
| London | 68 | 7.2 | 0.69 |
| New York City | 70 | 6.5 | 0.67 |
| Seoul | 65 | 6.7 | 0.72 |
| **Hong Kong** | **14** | **6.6** | **0.21** |

Four cities sit at 0.67–0.79 (below 1.0 because the capture is incomplete). Hong Kong is a
different animal — see §6, which must be read before any Hong Kong number in this report or
anywhere else in the repo.

---

## 1. PMF coherence violations — REAL BUT NOT TRADEABLE

### 1.1 How big is the violation?

The right estimator depends on direction:

* **Over-round (Σp > 1) → buy NO on every bin.** Needs only mutual exclusivity, so it is valid
  on an incomplete capture, and incompleteness biases the measured Σp *downward*. Worst case
  (the winner is in our subset) payout = n−1 against an outlay of n − Σp.
  `guaranteed net = (Σp − 1) − n·h − Σ fee(pᵢ)`
* **Under-round (Σp < 1) → buy YES on every bin.** Needs exhaustiveness. Only the 26 complete
  books qualify.

On the **26 exhaustive books** (11 bins each, one observation per book, 15 city-days,
5 distinct target dates):

| lead | books | over-round Σp−1 | clustered 95% CI |
|---|---|---|---|
| 0–6 h | 4 | +0.0035 | [+0.0020, +0.0050] |
| 6–12 h | 6 | +0.0164 | [−0.0122, +0.0451] |
| 12–24 h | 9 | +0.0197 | [+0.0029, +0.0365] |
| 24–48 h | 16 | +0.0287 | [+0.0134, +0.0440] |
| 48–96 h | 14 | **+0.4406** | [+0.1469, +0.7343] ← artifact, see 1.3 |
| **pooled** | **26** | **+0.0223** | **[+0.0075, +0.0371]** (median +0.0215) |

So a full weather book carries a **~2.2 % over-round**, converging to ~0.35 % at resolution.
That is a real, statistically-distinguishable-from-zero vig. It is also the smallest number
in this report.

### 1.2 The dutch book is dead on arrival

Selling an 11-bin book costs 11 × 1 ¢ half-spread ≈ **11 ¢**, plus ~4 ¢ of taker fee, against a
**2.2 ¢** over-round. Measured directly:

| feed | baskets | net-profitable at h = 0.01 | city-days |
|---|---|---|---|
| snapshot cycles (Mar–Jul) | 1,680 | **0** | 0 |
| CLOB hourly (July) | 10,621 | 58 | 25 |
| CLOB, excluding books < 2 h old | — | 14 | 9 |

The break-even half-spread at the 99th-percentile basket is **0.0086** — you would need
sub-1 ¢ execution on every leg simultaneously. Under-round is worse: only 3 of 411 exhaustive
observations were buy-all-YES profitable, all three on 2026-07-30 fresh books.

### 1.3 The 58 "profitable" baskets are a quote artifact, not an arbitrage

They cluster at market-open hours on freshly-listed books. Concretely, London Tmin for
2026-08-01 at 05:00 UTC on 07-30:

```
13 or below  0.4650      17°C  0.3600
14°C         0.3550      18°C  0.3550
15°C         0.3650      19°C  0.3550
16°C         0.3850      20°C  0.1400   ... Σ = 2.93 over 11 bins
```

Six bins quoted at 0.355–0.385 is not a market view; it is the midpoint of a book with no
two-sided depth (bid ~0.01 / ask ~0.70 → mid 0.355). By 08:00 the sum is 1.6, by 14:00 it is
1.0. **The largest measured "arbitrages" occur exactly where the 1 ¢ half-spread assumption is
least defensible.** This same artifact reappears in §3 and §7 and is the most important
methodological caveat in this document.

**Verdict: statistically real (+0.0223, CI excludes 0), nowhere near large enough to trade.**

---

## 2. Favorite–longshot bias — NO USABLE PATTERN

Model-free: price in, graded outcome out. `p − q > 0` means the band is overpriced.
Last pre-resolution snapshot, Hong Kong excluded (§6), n = 1,928 bins / **270 city-days**:

| price band | n | city-days | mean p | realized | p − q | clustered 95% CI | sell net | net CI |
|---|---|---|---|---|---|---|---|---|
| ≤ 0.01 | 834 | 225 | 0.0029 | 0.0024 | +0.0005 | [−0.0028, +0.0038] | −0.0096 | [−0.0129, −0.0063] |
| 0.01–0.02 | 156 | 107 | 0.0146 | 0.0192 | −0.0047 | [−0.0258, +0.0165] | −0.0154 | [−0.0365, +0.0057] |
| 0.02–0.05 | 201 | 136 | 0.0347 | 0.0249 | +0.0099 | [−0.0115, +0.0312] | −0.0018 | [−0.0232, +0.0195] |
| 0.05–0.10 | 155 | 111 | 0.0746 | 0.0516 | +0.0230 | [−0.0123, +0.0583] | +0.0095 | [−0.0258, +0.0448] |
| 0.10–0.20 | 159 | 112 | 0.1441 | 0.1069 | +0.0372 | [−0.0103, +0.0846] | +0.0210 | [−0.0264, +0.0684] |
| 0.20–0.35 | 236 | 152 | 0.2733 | 0.2331 | +0.0403 | [−0.0101, +0.0906] | +0.0204 | [−0.0299, +0.0708] |
| 0.35–0.50 | 103 | 79 | 0.4128 | 0.4563 | −0.0435 | [−0.1290, +0.0420] | −0.0656 | [−0.1511, +0.0199] |
| 0.50–0.65 | 26 | 26 | 0.5678 | 0.4615 | +0.1063 | [−0.0896, +0.3021] | +0.0841 | [−0.1118, +0.2800] |
| 0.65–0.80 | 16 | 14 | 0.7380 | 0.6875 | +0.0505 | [−0.1702, +0.2711] | +0.0309 | [−0.1896, +0.2514] |
| 0.90–1.00 | 40 | 38 | 0.9806 | 0.9250 | +0.0556 | [−0.0247, +0.1360] | +0.0447 | [−0.0356, +0.1250] |
| **POOLED** | **1,928** | **270** | 0.1144 | 0.1032 | **+0.0112** | **[+0.0041, +0.0183]** | | |

Repeated at CLOB prices 6 h / 24 h / 48 h before resolution: pooled +0.0082 [−0.0002, +0.0167],
+0.0062 [−0.0025, +0.0149], +0.0160 [+0.0007, +0.0313].

**Findings:**

1. **There is no favorite–longshot bias in the textbook sense.** The classic pattern is monotone
   overpricing of longshots. Here the *extreme* longshots are the best-priced band in the table
   (p−q = +0.0005 at a mean price of 0.29 ¢). Every individual band's CI spans zero.
2. The pooled +0.0112 [+0.0041, +0.0183] is simply §1's over-round, distributed over bins.
   It is real and it is a per-bin effect of ~1 pp.
3. **The overpricing that exists sits in the 5–35 ¢ shoulder — which is already `shoulder_book`.**
   This is a replication, not a discovery.
4. **The cheap end is untradeable by construction.** At a 0.3 ¢ price the entire prize is 0.3 ¢
   against a 1 ¢ half-spread; the sell CI is *entirely negative* [−0.0129, −0.0063]. Any strategy
   that "sells cheap tails" loses to the spread before it meets the weather.
5. Favorites are not underpriced. The 90–100 ¢ band shows +0.0556 (i.e. mildly *over*priced) on
   38 city-days with a CI spanning zero — the opposite sign to `shoulder_book`'s Leg-2 hypothesis
   and consistent with its 1 graded entry telling us nothing.

**Verdict: null. The only real effect is the uniform vig, and no band's net is distinguishable
from zero.**

---

## 3. Tmax / Tmin and adjacent-day coherence — CLEAN NULL

Tmin ≤ Tmax always, so for any `a < b` the events `{Tmax ≤ a}` and `{Tmin ≥ b}` are mutually
exclusive and `p(A) + p(B) ≤ 1`. Cumulative probabilities were rebuilt from each side's bin
prices (`lte` bin plus the tiling, and symmetrically for `gte`).

**On sane books** (each side's own bins summing ≤ 1.05 — i.e. excluding the §1.3 quote
artifact): 5,006 (city, date, hour) states, **1,801 exclusive pairs tested, 35 gross violations,
maximum excess 1.2 %, and ZERO net-profitable after two legs of half-spread and fee.** All 35
are the residual single-book over-round leaking into the cumulative sum, not a genuine
inconsistency between the two markets.

Soft check: implied `E[Tmax] − E[Tmin]` from normalized bin prices over 922 states / 85
city-days — median **5.84 °C**, minimum **1.89 °C**, **fraction ≤ 0: 0.0000**. The market never
prices a diurnal range at or below zero, and never gets close.

Including the fresh-listing books produces 117 "violations" over 5 city-days with a headline
`P(Tmax ≤ 31) = 1.561` — a number above 1 is proof the input is a wide-mid artifact, not a
market. Worth recording as a trap: a naive scan of this exact form reports 63 "profitable
arbitrages" with median net +8.7 ¢, every one of them fictitious.

**Verdict: the market's joint Tmax/Tmin pricing is coherent. Nothing here.**

---

## 4. Price dynamics — NO MOMENTUM, ONE REAL LEAD GRADIENT

### 4.1 Momentum / mean reversion: nothing

Hourly CLOB bin-price changes, 46,408 observations over 151 city-days:

| subset | n | corr(Δₜ, Δₜ₊₁) | E[Δₜ·Δₜ₊₁] | clustered 95% CI |
|---|---|---|---|---|
| all | 46,408 | −0.0376 | −0.000045 | [−0.000090, +0.000000] |
| price 0.05–0.35 | 15,999 | −0.0220 | −0.000035 | [−0.000090, +0.000020] |
| > 24 h to end | 32,924 | −0.0572 | −0.000030 | [−0.000048, −0.000012] |
| ≤ 24 h to end | 13,484 | −0.0290 | −0.000080 | [−0.000229, +0.000070] |

Slight mean reversion, detectable at long lead, economically nil (the implied reversal is a
small fraction of one cent — orders of magnitude below the tick, let alone the spread).

Sorting bins by their trailing 24 h price change into quintiles and asking whether the *outcome*
deviates from the price: all five quintiles' CIs span zero (most extreme +0.0336
[−0.0222, +0.0895]). **The price path carries no exploitable information about the outcome.**

### 4.2 The one robust gradient: over-pricing decays with lead

One observation per bin per lead band (so each bin counts once), Hong Kong excluded:

| lead | bins | city-days | mean p | realized | p − q | clustered 95% CI | sell net | net CI |
|---|---|---|---|---|---|---|---|---|
| 48–72 h | 794 | 104 | 0.1333 | 0.1020 | **+0.0313** | **[+0.0150, +0.0476]** | +0.0162 | [−0.0000, +0.0325] |
| 24–48 h | 1,088 | 111 | 0.1192 | 0.1149 | +0.0043 | [−0.0049, +0.0134] | −0.0100 | [−0.0191, −0.0008] |
| 12–24 h | 686 | 110 | 0.1279 | 0.1385 | −0.0106 | [−0.0240, +0.0027] | −0.0242 | [−0.0376, −0.0109] |
| 6–12 h | 367 | 99 | 0.1374 | 0.1499 | −0.0124 | [−0.0311, +0.0062] | −0.0247 | [−0.0433, −0.0060] |
| 0–6 h | 216 | 86 | 0.1367 | 0.1389 | −0.0022 | [−0.0171, +0.0126] | −0.0129 | [−0.0278, +0.0020] |

This is the same phenomenon as §1's over-round decay, seen per-bin instead of per-book. It is
monotone and the sign flips: **bins are overpriced ~3 pp two-to-three days out and slightly
*under*priced inside 24 h.**

Controlling for the §1.3 artifact (requiring the book to be ≥ 6 h old at observation) the gross
effect survives — +0.0292 [+0.0050, +0.0535] — but the sample falls to 53 city-days and the
**net** is +0.0151 [−0.0090, +0.0393]. At age ≥ 12 h: gross +0.0309 [−0.0011, +0.0629] on 25
city-days, below `MIN_CLUSTERS`.

**Verdict: the gross lead gradient is statistically real. Net of costs it is not
distinguishable from zero in any specification, and the specification that removes the
known artifact drops below the repo's own minimum cluster count.**

---

## 5. The thin-market cohort — a pure forward test

The 2026-07-30 discovery fix is visible in the collector:

| day | cycles | markets/day | markets per cycle |
|---|---|---|---|
| 07-27 | 14 | 61 | 7.3 |
| 07-28 | 21 | 79 | 9.0 |
| 07-29 | 19 | 76 | 8.6 |
| **07-30** | 12 | **266** | **51.1** |

The clean comparison is *within* 2026-07-30 — all markets visible that day, split by whether we
had ever seen them before (this removes any time trend):

| cohort | n | median volume | IQR | median 24 h volume | median liquidity |
|---|---|---|---|---|---|
| previously visible | 51 | **$4,182** | $1,140 – $10,489 | $2,661 | $3,892 |
| newly visible | 215 | **$610** | $227 – $1,656 | $498 | $2,722 |

The new cohort is **~6.9× thinner by cumulative volume and ~5.3× by 24 h volume**. Liquidity
(Gamma's book-depth proxy) falls only 1.4×, which is why liquidity filters alone never caught
this — `MIN_LIQUIDITY = 1000` would admit most of the new cohort.

**Can we say anything preliminary? Essentially no.** Of 215 new-cohort markets, **30** have a
graded outcome, with target dates 2026-07-30 .. 08-01. That is one day of settlements. This is
a forward test and nothing more.

This matters more than it looks, because of §7: the one live positive result in this whole scan
lives *entirely* inside cohorts of this kind.

---

## 6. STRUCTURAL BUG: Hong Kong grading is wrong, and it is armed to fire ~2026-08-21

`resolution_anchors.RESOLUTION_ANCHORS["Hong Kong"]["resolution_unit"] = "0.1 °C"`, while the
market questions are whole-degree (`"Will the lowest temperature in Hong Kong be 25°C on July
27?"`) and HKO truth is published at 0.1 °C. `pmf.resolves_yes_temp` grades an `exact` bin with
`actual == round(temp, 1)` — so a 25 °C bin resolves YES only if the station reads *exactly*
25.0 °C. HK bins therefore almost never resolve YES.

Evidence, three independent ways:

1. **Realized rate.** 0 / 100 graded HK Tmax exact bins resolve YES; 1 / 45 Tmin.
2. **The partition test.** HK books with ≥ 4 bins average **0.21** winners per book against
   0.67–0.79 for the other four cities. Regrading whole-°C gives **0.64** — right in line.
3. **The settlement audit cannot see it.** `audit_settlements` on the 56 settled HK markets:
   only 3 ever settled YES. The repo rule catches 1 of 3, whole-°C catches 2 of 3, and
   **overall agreement is identical (96.43 %) under both rules** because the statistic is
   dominated by trivially-correct NOs. Per-city: Chicago 10/10, London 19/19, NYC 27/27,
   HongKong 54/56, Seoul 19/21 — overall 96.99 %, comfortably above the 95 % floor. Applying
   whole-°C to every city degrades Chicago (0.90) and NYC (0.963) and leaves London/Seoul/HK
   unchanged, so the fix belongs in the Hong Kong anchor, not globally.

**The landmine.** Hong Kong is currently *not* contaminating anything, because HKO's monthly
batch means 0 of the 68 HK entries in `shoulder_paper.csv` are graded yet. When the July batch
lands (~2026-08-21), **68 of 280 entries — 24 % of the book — will grade at once, and under
this rule a sell book wins nearly all of them.** `shoulder_book`'s Leg-1 full-band gate is
currently at n = 175 / +0.034 CI [−0.009, +0.076] against a 150-entry, +2 ¢ threshold. A
one-step injection of ~68 near-certain wins is exactly the kind of event that trips a
pre-registered gate on an artifact.

This is the repo's documented failure mode verbatim: a green run, a plausible number, flattering
the strategy. **Recommend fixing the Hong Kong `resolution_unit` before 2026-08-21**, and
re-running `audit_settlements.py` afterwards to confirm the other four cities are untouched.

---

## 7. The breadth-book band decomposition — reproduced, then falsified on replication

Claim under test: deep shoulder [5,10) ¢ nets **+0.0317** CI [+0.0192, +0.0442] (n = 555 / 254
city-days) while core [20,35) ¢ is significantly **negative** at −0.0394, the two cancelling to
a ~zero full band.

### 7.1 It reproduces

Recomputed with `shoulder_book._net_edge` and `stats_util` on `shoulder_paper_breadth.csv`
(1,829 graded shoulder entries, Polymarket's own `settled_outcome`):

| band | n | city-days | mean p | realized | taker net | clustered 95% CI |
|---|---|---|---|---|---|---|
| deep [5,10) ¢ | 535 | 249 | 0.0686 | 0.0262 | **+0.0297** | **[+0.0167, +0.0427]** |
| moderate [10,20) ¢ | 578 | 265 | 0.1444 | 0.0934 | +0.0352 | [+0.0130, +0.0573] |
| core [20,35) ¢ | 716 | 291 | 0.2754 | 0.2933 | **−0.0375** | **[−0.0598, −0.0153]** |
| **full [5,35) ¢** | 1,829 | 294 | 0.1735 | 0.1520 | +0.0051 | [−0.0014, +0.0116] |
| pre-registered [10,25) ¢ | 806 | 287 | 0.1669 | 0.1241 | +0.0263 | [+0.0080, +0.0447] |

Small differences from the quoted figures (n = 535 here vs 555 as claimed) are a re-run on a
different state of the file: `shoulder_paper_breadth.csv` was rewritten at 13:15 UTC today by
another process backfilling `settled_outcome` on 316 rows. **The graded set is moving under any
analysis of it** — worth pinning a copy before quoting numbers. The arithmetic and the
cancellation story are nonetheless correct.

It also survives the checks I expected it to fail:

* **Not a clustering artifact.** City-day [+0.0167, +0.0427], city (g=49) [+0.0181, +0.0412],
  target-date (g=6) [+0.0164, +0.0429] — all three agree, and the between-date standard
  deviation of the daily mean (0.0165) matches the city-day SE (0.0066 vs 0.0067).
* **Not a grading-selection artifact.** Grading coverage is 78.9 / 77.8 / 78.2 % across the
  three bands, and every ungraded row is one of the two unresolved future target dates
  (07-31, 08-01). `max_yes_after > 0.9` reproduces the realized YES rate per band exactly.
* **Internally stable.** Alphabetical half-splits by city: +0.0349 and +0.0256.

### 7.2 Five reasons it should not be traded

**(a) Six calendar days.** The target dates are 2026-07-25 … 07-30. The "294 independent
city-days" are 49 cities × 6 days. Clustering on date gives **g = 6 < `stats_util.MIN_CLUSTERS`
= 30** — by the repo's own written rule, that interval is "not meaningful". The city-day
interval is only valid if the effect carries no common daily component; it happens to look that
way here, but six draws cannot establish that.

**(b) The split is post-hoc, and post-hoc in the worst possible direction.** The band constants
`BAND_LO=0.05 / CORE_LO=0.20 / BAND_HI=0.35` pre-date the data, and all breadth entries post-date
`BREADTH_PREREG_DATE = 2026-07-23`, so nothing here is graded on its discovery sample. But the
**[5,10) ¢ band was never the pre-registered hypothesis — and the repo pre-registered the
opposite sign for it.** `shoulder_book`'s §10f docstring (pre-reg 2026-07-23) states the deep
band [5,10) ¢ is *fat-tail UNDER-priced* and that the edge lives in the moderate band. The
actual pre-registered band, [10,25) ¢, lands at **+0.0263 — below its own +0.03 gate
threshold**. Selecting [5,10) ¢ after seeing the breadth numbers is a new hypothesis wearing an
old registration's clothes.

**(c) It does not replicate anywhere we have history.** Three attempts, two of them formally
inconsistent with the claim:

| sample | dates | n | city-days | deep-band net | 95% CI |
|---|---|---|---|---|---|
| breadth, 44 NEW cities | 6 | 433 | 220 | **+0.0345** | [+0.0213, +0.0477] |
| breadth, original 5 cities | 6 | 102 | 29 | +0.0092 | [−0.0287, +0.0470] |
| 5-city `shoulder_paper` (weather truth) | 18 | 46 | 36 | −0.0098 | [−0.0845, +0.0650] |
| **5-city CLOB, leads > 24 h** | **~30** | **402** | **105** | **−0.0076** | **[−0.0281, +0.0128]** |

The last row is the decisive one: an independent sample with 105 city-days over ~30 distinct
dates, priced from traded CLOB history and graded against settlement-faithful weather truth,
puts the deep band at **−0.0076 [−0.0281, +0.0128]**. The claimed +0.0297 lies **outside** that
interval, and the two intervals do not overlap ([+0.0167, +0.0427] vs [−0.0281, +0.0128]).
The core band flips too: −0.0375 in breadth against **+0.0593** in the 5-city book.

**(d) The effect lives entirely in the thinnest cohort.** Deep-band median liquidity is $1,175,
with 37 % below `config.MIN_LIQUIDITY`. The 44 new cities are the §5 cohort — ~7× thinner than
anything this project has previously measured, and the one place where a posted mid is least
likely to be a hittable price. §1.3 shows what those mids can look like.

**(e) It dies at 3 ¢ of slippage.** Deep-band NO win rate 0.9738 against a recorded NO entry of
0.9314:

| extra slippage | net | clustered 95% CI |
|---|---|---|
| +0.01 (as reported) | +0.0297 | [+0.0167, +0.0427] |
| +0.02 | +0.0201 | [+0.0071, +0.0331] |
| +0.03 | +0.0106 | [−0.0024, +0.0236] |
| +0.05 | −0.0085 | [−0.0215, +0.0045] |

Equivalently: **the recorded 6.9 ¢ mid must be genuinely sellable at ≥ 3.9 ¢.** Under the legacy
2 %-of-payout fee model the reported figure is already only +0.0129 [+0.0002, +0.0257], and dies
at 2 ¢ of slippage.

### 7.3 What is actually shared between the two samples

Both agree on a **lead** gradient, not a price-band one. §4.2 (5 cities, CLOB, weather truth)
finds bins overpriced ~3 pp at 48–72 h and underpriced inside 24 h; §1 finds the book over-round
decaying from 2.9 % at 24–48 h to 0.35 % at resolution. The breadth book enters only at
`PREDAY_HOURS = 24`, i.e. entirely in the overpriced regime. **The plausible common cause is
"early books are over-round", not "cheap bins are mispriced"** — and the over-round is ~2 ¢
across an 11-bin book, which is the size of the thing, and the size of the spread.

**Verdict: statistically real within its six-day window; NOT established as tradeable; NOT
replicated on the sample with the most history, where the point estimate has the opposite sign
and the intervals do not overlap.** Recommend it stays paper, that the pre-registered [10,25) ¢
band remains the gated one, and that the deep band gets its own forward clock starting now
rather than inheriting credit for the week that suggested it.

---

## 8. Summary table

| # | finding | effect | clustered 95% CI | city-day clusters | survives costs? | real? | tradeable? |
|---|---|---|---|---|---|---|---|
| 1 | book over-round (full books) | +0.0223 | [+0.0075, +0.0371] | 15 (5 dates) | no — 11 legs × 1 ¢ vs 2.2 ¢ | yes | **no** |
| 1 | mechanical dutch book | 0/1,680 snapshot, 14/10,621 CLOB ex-fresh | — | 9 | no | artifact | **no** |
| 2 | favorite–longshot, pooled | +0.0112 | [+0.0041, +0.0183] | 270 | it *is* the vig | yes | **no** |
| 2 | any individual price band | — | all span 0 | 14–225 | no | no | **no** |
| 2 | extreme longshots (< 1 ¢) | +0.0005 | [−0.0028, +0.0038] | 225 | sell CI entirely negative | no | **no** |
| 3 | Tmax/Tmin joint incoherence | 35/1,801 pairs, max 1.2 % | — | — | 0 profitable | **no** | **no** |
| 4 | price momentum / reversion | corr −0.038 | [−9e-5, 0] | 151 | ~0 economically | marginal | **no** |
| 4 | lead gradient, 48–72 h gross | +0.0313 | [+0.0150, +0.0476] | 104 | net +0.0162 [−0.0000, +0.0325] | yes | **not proven** |
| 5 | new-cohort volume vs old | 6.9× thinner (median $610 vs $4,182) | — | 1 day graded | — | yes | forward test only |
| 6 | Hong Kong grading | 0.21 vs 0.64 winners/book | — | — | — | **BUG** | fix before 08-21 |
| 7 | breadth deep band [5,10) ¢ | +0.0297 | [+0.0167, +0.0427] | 249 (6 dates) | dies at 3 ¢ slippage | **within-window yes** | **not replicated** |
| 7 | same band, 5-city CLOB | −0.0076 | [−0.0281, +0.0128] | 105 (~30 dates) | — | contradicts | **no** |

## 9. Recommendations

1. **Fix `RESOLUTION_ANCHORS["Hong Kong"]["resolution_unit"]` before ~2026-08-21.** This is the
   only item here with a deadline. Re-run `audit_settlements.py` after (the other four cities
   must stay at 100/100/96/90 %).
2. **Do not promote the deep band on the current evidence.** Give it its own forward clock from
   today, keep [10,25) ¢ as the gated hypothesis, and require replication on the 5 cities.
3. **Instrument the spread.** Every negative result in this document is decided by
   `HALF_SPREAD = 0.01`, a placeholder the code itself flags as "tune once real order-book data
   exists". Recording top-of-book bid/ask alongside the mid would convert three "not proven"
   verdicts into answers, and is the single highest-value data addition available.
4. **Treat the mid as untrustworthy for books < 6 h old.** §1.3's 2.93-sum book would have
   produced fictitious arbitrage signals in at least three of the analyses above.
5. **Do not re-run coherence work on pre-2026-07-30 data.** Only 40 of 547 historical books are
   complete; the post-fix feed makes this analysis possible for the first time, and it needs
   weeks of accumulation, not a re-analysis.
