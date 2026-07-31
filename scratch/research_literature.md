# Non-forecasting edge in prediction markets — literature sweep + independent measurement

**Date:** 2026-07-31 · **Scope:** sources of edge that do NOT require out-forecasting the consensus.
**Status of the forecasting thread:** closed negative (pooled paired Brier gap +0.0183, CI [+0.0045, +0.0321], n=421 / 178 city-days). Nothing below assumes it reopens.

This report does two things. Part A is the literature sweep. Part B is an **independent measurement I ran on this repo's own stored snapshots** (6,969 rows, 2026-03-17 → 2026-07-30, graded against settlement-faithful truth via `grading.resolves_yes`) because the literature turned out to make a *falsifiable, horizon-dependent* prediction about exactly the trade this project is already running. Part B is the load-bearing part.

---

## 0. Bottom line, ranked

| # | Candidate | Verdict | Net EV |
|---|---|---|---|
| 1 | **Horizon restriction on the structure book** (sell shoulders ONLY pre-day; never same-day) | **Strongest finding. Confirmed by two independent datasets + my own measurement.** | Free — it is a *restriction* on an existing book, costs nothing to adopt |
| 2 | **Core [25,35)¢ pre-day short** | Only cell whose clustered CI excludes zero (+7.2¢/leg, 122 clusters) — but **failed to replicate** in the repo's own 49-city breadth book | Unproven; needs the replication, not more of the same sample |
| 3 | **Maker-side book-overround harvest** | Overround is real (+2.8¢/book median) but **taker fee is 3.6¢** — maker-only or dead | Small but structurally sound; requires order-book data we don't collect |
| 4 | **Kalshi as a second consensus + settlement-ruler basis** | Cheapest new data source; NOT an arbitrage (different rulers) | Positive option value, low cost |
| 5 | Deep [5,10)¢ short | **Does not reproduce** (+0.0023, CI spans zero); *negative* same-day (−0.086) | ~Zero. Do not trade |
| 6 | Moderate [10,25)¢ short (currently pre-registered as Leg 1b) | Measures **−0.0063**, CI spans zero | ~Zero. The pre-registration looks like noise-fitting |
| 7 | Favorite leg (buy 65–85¢ pre-day) | **No inventory** — n=3 markets in 4 months | Untradeable as specified |
| 8 | Coherence / dutch-book arbitrage | Dead on modern venues (3.6s median life, ~15 shares) | Negative |
| 9 | Weather-derivative risk premium (CDD/HDD) | Does not transfer — no hedging demand in daily bins | Zero |
| 10 | Round-number / tick clustering | Execution detail, not edge | Zero |

**The single most important sentence in this report:** every band-level "edge" this project has measured has changed sign or magnitude on re-measurement — §10b (2026-07-12) said core +8.1¢/deep +2.3¢/moderate dead; the code comment (2026-07-23) says deep is *under*-priced and moderate is the good one; the breadth book (2026-07-27) flipped core from +0.049 to −0.028; I measure moderate at −0.006 and deep at +0.002. **A quantity that changes sign every time you measure it is noise.** The horizon effect (#1) is the only structure that has held up across every cut.

---

# PART A — Literature sweep

## A1. Favorite-longshot bias: what the real-money evidence actually says

### The modern real-money studies (these are the ones that count)

**Kalshi — Bürgi, Deng & Whelan, "Makers and Takers: The Economics of the Kalshi Prediction Market"** (CEPR DP20631 / GWU WP 2026-001, 2026). Transaction-level data on **>300,000 contracts**.
- "Low-price contracts win far less often than required to break even, while high-price contracts win more often and yield small positive returns."
- Longshots **below $0.30: bias −0.077** (p<0.001), realized win rate 4.0%.
- **Takers lose ~32% on average; makers lose ~10%.** Both groups show the longshot pattern; takers lose substantially more.
- Fed/interest-rate markets show **near-perfect calibration and no significant bias** — the bias is category-specific, not universal.
- URLs: https://cepr.org/publications/dp20631 · https://www2.gwu.edu/~forcpgm/2026-001.pdf · https://ideas.repec.org/p/pra/mprapa/126350.html

> **Read that maker/taker number carefully.** *Both* sides lose on average. This is a negative-sum venue. The finding is not "sell longshots and win" — it is "everyone loses, longshot buyers lose most." Those are different claims and only the first is tradeable.

**Polymarket — Qin & Yang, "Polymarket-v1 Database"** (arXiv:2606.04217, 2026). 1.20bn trades, 1.30m markets, $61bn notional, 2022-11-21 → 2026-04-28.
- Realized return defined as `payout − price`.
- Price decile 0.00–0.10 (avg price 0.052): **mean return −0.0023** (n=34.9m)
- Price decile 0.40–0.50 (avg 0.451): **+0.0026**
- Price decile 0.90–1.00 (avg 0.952): **+0.0056**
- **Returns are NOT adjusted for fees or spread.**
- Weather is not a standalone category (subsumed under "Sci-Tech").
- URL: https://arxiv.org/html/2606.04217v1

> **The arithmetic that kills the naive version.** −0.0023 on a 5.2¢ contract is −4.4% of stake for the buyer, so **+0.0023/share for the seller**. Polymarket's weather taker fee is `0.05·p·(1−p)` = **0.0024/share at p=0.05**. The whole Polymarket-wide longshot premium is consumed by the fee to the last basis point, before you cross a single spread. Selling deep longshots on the pooled Polymarket evidence is *exactly* break-even.

**Dissent — Reichenbach & Walther, "Exploring Decentralized Prediction Markets"** (SSRN 5910522, Dec 2025), 478m Polymarket trades: documents a tendency to overtrade the default and "Yes" option but **finds no evidence of a general longshot bias.** Two large Polymarket studies, opposite conclusions. Treat the Polymarket FLB as contested.

### The horizon result — the most useful thing in the sweep

**Le, "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets"** (arXiv:2602.19520, 2026). Kalshi (64.7m trades / 210,608 contracts) + Polymarket (227.6m trades / 116,000 contracts), through 2025-12-31.

Model: `logit(P(y=1)) = a + b·logit(p)`. **b > 1 ⇒ prices compressed toward 50% (classic FLB — longshots overpriced). b < 1 ⇒ prices too extreme (anti-FLB — longshots UNDERpriced).**

**Weather markets specifically** — Kalshi only (Polymarket weather coverage "negligible"), **26,911 markets / 4.4m trades / 279.1m contracts**, base rate 24%:

| horizon | slope b | implication for a longshot SELLER |
|---|---|---|
| 0–1 h | **0.69** | strongly adverse |
| 1–3 h | 0.84 | adverse |
| 3–6 h | 0.74 | adverse |
| 6–12 h | 0.87 | adverse |
| 12–24 h | 0.91 | adverse |
| 24–48 h | 0.97 | ~neutral |
| **2 d – 1 w** | **1.20** | **favourable** |
| 1 w – 1 mo | 1.20 | favourable |
| 1 mo + | 1.37 | favourable |

The paper's own words: weather shows "overconfidence at short horizons (slopes 0.69–0.97 within 48 hours), where prices are too extreme," converging to the universal underconfidence pattern beyond 2 days.

**This is a sign flip at ~48 hours, on 26,911 weather markets, from a venue and instrument entirely independent of ours.** Raincheck's structure book currently trades both regimes with one sign. Part B tests this prediction on our own data and it holds.

Caveats, stated plainly: Kalshi weather contracts are **threshold-exceedance** ("above/below X", base rate 24%), not 1°F bins; a single logit slope cannot represent a mid-band (20–35¢) effect; and the 0–1 h slope of 0.69 is suspicious — logit recalibration is dominated by extreme observations, and near-resolved contracts plus any settlement-label noise will manufacture a low slope. I would trust the **monotone gradient** and the **~48 h crossing**, not the individual short-horizon coefficients.

### Where the bias concentrates, and why (the deep-vs-core question)

The classical literature says FLB is **monotone in odds** — worst at the longest prices. Horse racing: return at 100/1+ is about **−61%** vs **−23%** betting randomly (Wikipedia/Thaler-Ziemba lineage; https://en.wikipedia.org/wiki/Favourite-longshot_bias). Pinnacle soccer, 12,084 matches: favourites −3.64%, underdogs −26.08%.

**But there is a strong deflationary explanation, and it matters for us.**

**Data Golf, "The Favourite-Longshot Bias is not a bias"** (https://datagolf.com/fav-longshot-not-a-bias). On **27,150 Pinnacle soccer matches**, the bookmaker allocates margin **roughly equally in absolute terms** (~0.9% per selection) rather than proportionally — because proportional margin at extreme odds would either exceed 100% or be unprofitable. Equal *absolute* margin on a small price is a large *relative* margin. **The declining-return-with-odds pattern falls out mechanically from rational price-setting, with no bettor irrationality required.** A heterogeneous-agent model reproduces it; the "representative bettor with risk-loving preferences" story is not needed.

Implication for Polymarket weather: there is no bookmaker setting a margin, but there **is** a structural analogue — the **minimum tick**. A bin whose fair value is 0.4¢ cannot be quoted at 0.4¢ on a 1¢ grid. Rounding at the bottom of the book manufactures apparent overpricing of deep longshots that is **not harvestable**, because you cannot sell below the tick either, and the spread there is ~100% of the contract's value. This is the leading candidate explanation for why the deep band looks mispriced at mid and measures zero when you charge realistic costs (Part B).

Also relevant: **Green, Lee & Rothschild, "The Favorite-Longshot Midas"** (https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2018/08/The-Favorite-Longshot-Midas.pdf) — argues the price-setter, not bettor demand, generates the bias. Same deflationary direction.

### Documented failure modes of selling longshots

1. **The premium is the fee.** Shown arithmetically above on Polymarket-wide data.
2. **Spreads are widest exactly where the claimed edge is.** Anatomy of a Decentralized Prediction Market (arXiv:2604.24366) Table 1, median quoted spread by mid-price: **0.00–0.10 → 1,818 bps** (p25 1,176, p75 4,000); 0.10–0.20 → **1,339 bps**; 0.40–0.50 → **400 bps**; 0.50–0.60 → **400 bps**. Paper's words: "climbs to 1,300–1,800 bps for markets trading below 0.10... an order of magnitude wider" than IEM. *(Units are bps of mid — they cannot be probability points, since an 18-point spread at a 0.05 mid is impossible. At a 7.5¢ mid that is ~1.0–1.35¢ full spread, ~0.5–0.7¢ half — so this repo's `HALF_SPREAD = 0.01` is actually conservative there, and roughly right at the core band.)* The paper attributes the longshot spread premium to a **liquidity-provision constraint**: "low-probability binary contracts have a bounded upside and an asymmetric downside for the maker" — i.e. the wide spread is the maker's compensation for exactly the risk you are proposing to take.
3. **Tail risk and collateral.** Selling a 7.5¢ YES on Polymarket means buying the NO at 92.5¢ — you post **92.5¢ of collateral to earn a claimed 3¢**. Per-bet sd ≈ √(p(1−p)) ≈ 0.26 against a ~0.03 mean: a **Sharpe of ~0.12 per bet**. Practitioners size this at a small fraction of Kelly precisely because the loss distribution is 1-in-13 for the full stake, and because estimation error in p at 7.5¢ is proportionally enormous.
4. **Order-flow toxicity rises when noise traders leave.** Polymarket-v1's fee-reform event study (taker fees activated by category: crypto Jan 2026, sports Feb 2026, **politics/news/others March 2026**): True VPIN **+0.01594** (t=15.43), Gibbs spread **+0.00805**, both p<0.001. Weather taker fees started **2026-03-30** — inside this repo's sample window. Any edge measured on pre-reform data is measured on a different market.
5. **Niche + wide-spread markets select for informed specialists.** Polymarket-v1: Gibbs spread coefficient **−4.1280** (p<0.001) predicting Brier — wider spreads correlate with *lower* forecast error, a selection effect favouring informed specialists in niche markets. Weather is a niche, wide-spread market. This cuts *against* us and is consistent with the model having lost.

### Nobody has published FLB on Polymarket/Kalshi **weather bins** specifically

Closest is Le (2026) on 26,911 Kalshi weather *threshold* markets (above). The Polymarket weather "strategy" material is all marketing content (laikalabs, botforkalshi, tradetheoutcome, polytraderbot) — treat as zero evidentiary weight. The one semi-primary artifact is a Medium teardown of an actual on-chain wallet, `0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11` ("automatedAItradingbot"): ~$88k cumulative, 3,029 predictions, most profitable entries **$0.03–$0.10** (+$18k over 445 positions), heavy losses above $0.50 (−$26k), record 349 wins / 1,537 losses in a 2,000-trade sample. That wallet is **BUYING** cheap longshots, not selling them — the exact opposite of the strategy under consideration, and it is reportedly profitable. n=1, unverified, but worth knowing the loudest public weather-trading example runs the other way.

## A2. Coherence / dutch-book arbitrage — dead on modern venues

**Arbitrage Analysis in Polymarket NBA Markets** (arXiv:2605.00864): **75m order-book snapshots**, 173 games, polled every 3.6–5.5 s.
- Single-market arb (prices not summing to $1): **7 executable episodes across 3,042 markets = 0.0001% of time in arbitrage**, total capped profit **$210.19**, **median duration 3.6 seconds**.
- Combinatorial arb: 290 episodes, median return **101 bps**, median duration 16 s; 17.2% lasted ≤4 s.
- **76.9% of combinatorial opportunities constrained to an average executable size of 14.8 shares.**
- Conclusion: "risk-free extraction strictly confined to the retail tier," bounded by liquidity, not by pricing efficiency.

**Unravelling the Probabilistic Forest** (arXiv:2508.03474, AFT 2025): ~**$40m** of arbitrage profit historically extracted across market-rebalancing and combinatorial types.

**Verdict for us:** an opportunity with a 3.6-second median life is unreachable by an hourly collector. Historical $40m totals are a fact about 2022–2024, not about a strategy available now. Polymarket's **NegRisk adapter** (https://docs.polymarket.com/advanced/neg-risk) does make the mechanics clean — a NO share in any market converts to 1 YES in every other, no stated conversion fee — which matters for **capital efficiency** of candidate #3, but it does not create opportunities.

## A3. Market-maker economics in thin event markets

- **Polymarket fees:** taker `fee = C · rate · p · (1−p)`; weather max **$1.25 per 100 shares** ⇒ rate = **0.05**, peaking at p=0.50 and vanishing at the extremes. **Makers pay zero** and receive **15–25% of collected taker fees** rebated daily (weather ≈25%). Sources: help.polymarket.com "Trading fees" (13364478), "Maker Rebates Program" (13364471), docs.polymarket.us/fees. *This repo's `config.taker_fee_per_share` is correct; the legacy `FEE_RATE = 0.02` overstates costs and is already flagged in config.*
- **Adverse selection on liquid Polymarket books is near zero.** Anatomy paper, Glosten-Harris on top-100 markets: median effective half-spread **−0.0003**, median adverse-selection component **0.0**. Depth is *uniform*, not top-of-book concentrated (median L1/L10 = **0.137** vs 0.10 for a perfectly uniform grid). Median Herfindahl **0.031** ⇒ ~**32 effective makers per market**.
- **Critical methodological warning from the same paper:** the public WebSocket feed infers trade direction with only **~59% accuracy** vs on-chain ground truth (Lee-Ready gets ~80% on equities). The effective half-spread **flips sign on 67% of markets** and Kyle's λ on 60% when you substitute real direction. *Any microstructure signal built from Polymarket's public feed is probably measuring its own inference error.*
- **Depth decays into resolution:** within-category log-log slope **0.55** (t=3.85) ⇒ ~6% less mean depth per 10× reduction in seconds-to-close.
- **Optimal Market Making in Prediction Markets** (arXiv:2607.17991): optimal vs myopic quoting over 10,000 simulated paths — PnL sd **28.11 → 10.34** (−63%), terminal inventory **49.37 → 15.23** (−69%), 5% VaR **32.41 → 4.20** (−87%), at a **0.6% cost in mean PnL**. Spreads should widen near settlement for prices near 50¢ where settlement risk peaks. Inventory control is worth far more than edge here.

## A4. Mean reversion / overreaction

**Dujava (QuantPedia, 2026-04-17), "Exploiting Mean-Reversion in Decentralized Prediction Markets."** Zero-spread Sharpes of +2.97 / +1.58 / +1.72 collapse under a **10 bps** spread; the best zero-spread variant (X5_Y1) becomes the **worst** under friction (Sharpe +2.97 → **−2.60**). Conclusion: mean-reversion alpha "requires passive limit-order placement" to survive.

**External validity is close to nil** — the three contracts studied are "Will Jesus Christ return in 2025?", "Will China invade Taiwan in 2025?", "Will the US confirm that aliens exist in 2025?". These are joke/no-news contracts whose price movement *is* noise by construction. I would not carry a single number from this study to weather. It is included because its *cost-sensitivity* conclusion is the general lesson, and because it points the same direction as Le's short-horizon weather slopes.

## A5. Weather derivatives (CDD/HDD) — does not transfer

Cao & Wei (2004), *J. Futures Markets* 24:1065-1089, establish a significant market price of weather risk; Benth & Lempa (2024, *Applied Stochastic Models*) treat CDD/HDD hedging; CME weather volumes up 260% vs 2022. **The premium exists because energy utilities have genuine hedging demand for monthly aggregate indices and pay to transfer that risk.** No participant needs to hedge "NYC high is exactly 73°F tomorrow." There is no hedger, therefore no risk premium to collect. **Drop this line entirely.**

---

# PART B — Independent measurement on this repo's data

**Method.** All 5 city snapshot files (6,969 rows, 2026-03-17 → 2026-07-30). Target date via `pmf.parse_question_date(q, ref_date=snapshot_date)`; outcome via `grading.resolves_yes` (settlement-faithful — WU reconstruction for NYC/Chicago, station feeds elsewhere). **5,058 of 6,969 rows graded, 2,117 distinct markets, 320 city-days.** Hours-to-local-day-end computed per city. **One observation per market per horizon stratum** (the last snapshot in the stratum) — no pseudo-replication within a stratum. **All CIs clustered on city-day**, per this project's own rule. Costs charged: `HALF_SPREAD` crossed on entry + `config.taker_fee_per_share` at the execution price. `net` is short-YES taker P&L per share.

### B1. Short-YES net edge by band and horizon

| band | >48 h | 24–48 h | 12–24 h | <12 h |
|---|---|---|---|---|
| (0.02,0.05] | −0.0027 | −0.0151 | −0.0117 | **−0.0949** |
| (0.05,0.10] | +0.0013 | +0.0156 | −0.0102 | **−0.0855** |
| (0.10,0.15] | −0.0232 | −0.0160 | +0.0082 | — |
| (0.15,0.20] | +0.0358 | +0.0145 | — | — |
| (0.20,0.25] | −0.0293 | +0.0604 | +0.0564 | — |
| (0.25,0.35] | +0.0683 | +0.0756 | +0.0832 | — |
| (0.35,0.50] | **+0.1945** | −0.0141 | −0.0643 | — |

**Every clustered 95% CI in this table contains zero except one:** (0.35,0.50] at >48 h, +0.1945 [+0.0457, +0.3434], n=32 / 28 clusters. That is **one significant cell out of ~25 tested** — precisely the false-positive count you expect at α=0.05 — and it **flips sign at 24–48 h and again at 12–24 h**. Discard it.

### B2. The horizon gradient — the finding that replicates

The deep bands go monotonically from ~zero pre-day to **strongly negative same-day**: (0.02,0.05] runs −0.003 → −0.015 → −0.012 → **−0.095**; (0.05,0.10] runs +0.001 → +0.016 → −0.010 → **−0.086**. Raw (pre-cost) edges show the same: +0.0085 at >48 h → **−0.0836** at <12 h.

**Selling cheap bins on the day is a clear, large loser.** That is exactly Le (2026)'s Kalshi weather slope going 1.20 (>2 d) → 0.69 (<1 h), reproduced on a different venue, a different contract type (1°F bins vs threshold contracts), and a different sample. **Two independent confirmations of one horizon-dependent sign flip is the most robust result in this entire sweep.**

### B3. Pre-day (≥24 h) portfolio construction, one observation per market

n = 1,442 markets over 258 city-days.

| band | legs | city-days | mean/leg | clustered 95% CI |
|---|---|---|---|---|
| full [5,35)¢ | 575 | 205 | **+0.0327** | [+0.0026, +0.0628] |
| moderate [10,25)¢ | 250 | 155 | **−0.0063** | [−0.0565, +0.0440] |
| **core [25,35)¢** | 166 | 122 | **+0.0716** | **[+0.0070, +0.1363]** |
| deep [5,10)¢ | 159 | 109 | **+0.0023** | [−0.0391, +0.0437] |

Aggregated per city-day, core [25,35)¢ gives **+0.0945 [+0.0148, +0.1742]** — the only construction significant at both the leg and the city-day level.

### B4. Three corrections to the working understanding

1. **The "+0.0317 for deep [5,10)¢" figure is a mislabel.** I measure deep [5,10)¢ at **+0.0023** (CI [−0.0391, +0.0437]) — indistinguishable from zero. The number **+0.0327** is my **full-band [5,35)¢** result. The reported +0.0317 almost certainly *is* the full band, attributed to the wrong sub-band. This matters because the full band's edge is entirely carried by [25,35)¢: full = +0.033, and with [25,35) removed the remainder is ≈0.
2. **The currently pre-registered Leg 1b is the wrong band.** `shoulder_book.py` pre-registered (2026-07-23) `MOD_LO, MOD_HI = 0.10, 0.25` as "the OVER-priced moderate band," excluding deep as "fat-tail-UNDER-priced" and core [0.25,0.35) as "fair." My measurement inverts all three labels: moderate is **−0.006** (dead), deep is **+0.002** (dead), core is **+0.072** (the only live one). The 2026-07-23 refinement was fitted to n=109 in-sample and does not survive re-measurement on 4× the data.
3. **Leg 2 (buy 65–85¢ favorites) has no inventory pre-day.** Across four months: n=11 markets in [0.50,0.65), **n=3** in [0.65,0.75), **zero** above 0.75. Weather books almost never contain a pre-day favorite above 50¢ — the mass sits in a 25–45¢ mode plus a long tail of ≤2¢ bins (across 49 complete books: 343 of 537 legs priced ≤0.02, carrying just 2.2% of book mass). `shoulder_book.py`'s own comment already says Leg 2 is a same-day phenomenon; this confirms it quantitatively.

### B5. Book overround — the coherence trade, measured

49 complete books (9–11 legs each, both tails present):

- median Σ(YES) = **1.0280**; mean 1.0239; 44% of books > 1.02; IQR [0.9993, 1.0435]
- median **gross** overround **+2.80¢** per $1 of event notional
- median **taker fee to sell every leg: 3.62¢**
- median **NET as a taker: +0.39¢**, positive on only **55%** of books
- ROI on capital: **0.28% gross, 0.039% net taker**; median book liquidity $36,910

**The overround is real and the taker fee eats it.** This is a maker-only trade or it is nothing. It also explains the shoulder premium's origin: if a book carries only 2.8¢ of total overround, a claimed +7¢ on one leg means other legs must be *under*-priced by an offsetting amount — i.e. selling a shoulder is not harvesting overround, it is a **directional bet that the modal bin wins**, with correspondingly higher variance. The variance-minimal expression of the same view is a **paired trade within one book** (sell the [25,35)¢ shoulder, buy the mode) at constant total mass, not two separately-recorded legs.

*Caveat that applies to every number in B5:* `outcome_probs_json` from Gamma is a mid/last price, not an executable bid. The true harvestable overround is strictly lower.

---

# PART C — Ranked candidates

## 1. Horizon restriction on the structure book — sell shoulders pre-day only

**Edge source (one sentence):** the market's price-to-truth calibration slope in weather contracts is a known, monotone function of time-to-resolution that crosses 1.0 at ~48 hours — so the same short-shoulder trade has opposite expected sign before and inside the day, and trading both regimes with one sign averages an edge against a loss.

**Mechanism & effect size.** Le (2026, arXiv:2602.19520), 26,911 Kalshi weather markets: slope 1.20 at 2 d–1 w vs 0.69–0.97 within 48 h. My Part B replication on Polymarket bins: deep-band short-YES net runs +0.001 (>48 h) → −0.086 (<12 h); raw +0.0085 → −0.0836.

**Data needed.** None new. Already have `entered_at_utc`, target date, city, and a settlement-faithful grader.

**Cheapest honest test.** Add an hours-to-local-day-end stratum to `shoulder_book.py` and pre-register that all Leg 1 gates count **only** entries with ≥24 h (ideally ≥48 h) to day end, and that a <12 h stratum is recorded with the **opposite** predicted sign as a falsification check. This is a *tightening* of existing gates, so it is admissible under the repo's tightening-only rule.

**Sample size.** Zero additional — it reclassifies data already collected (1,012 rows at >48 h, 1,717 at 24–48 h, 1,056 at <12 h). The falsification check needs ~100 same-day city-days to confirm the negative sign; already have 162.

**Why it might fail here.** Kalshi weather is threshold-exceedance with a 24% base rate, not 1°F bins — different instrument, possibly different crowd composition. And my own confirmation shares its truth pipeline with the thing being tested; a truth-source regression would move both together (cf. the 2026-07-30 obs-truncation incident).

---

## 2. Core [25,35)¢ pre-day short — the only live cell, and it has already failed one replication

**Edge source:** retail spreads probability mass across plausible-looking neighbours of the modal bin, so the bins just outside the mode carry more price than realized frequency; a seller of those bins collects the difference without any view on the temperature.

**Effect size.** Mine: **+0.0716/leg net of half-spread and taker fee, 166 legs / 122 city-day clusters, CI [+0.0070, +0.1363]**; +0.0945/city-day [+0.0148, +0.1742]. Repo §10b (2026-07-12): +8.1¢ (price 0.271 vs realized 0.190, n=147). Consistent.

**But — the replication failed.** The repo's own breadth book (independent, 49 cities, 603 graded entries, 98 clusters) **flipped the core band's sign from +0.049 to −0.028**, and put the full band at +0.003 [−0.008, +0.015]. Two samples, opposite signs. The gate-power amendment (2026-07-27) records the honest reading: the full-band gate was nominally MET at n=150 / +0.0234 with clustered CI **[−0.023, +0.070]**, i.e. "gate met" and "no edge" were indistinguishable.

**Data needed.** Order-book depth (bid/ask, not Gamma mid) to know whether the premium is inside the spread. `clob_token_ids_json` is already stored, so a CLOB book poller is a small addition to the collector.

**Cheapest honest test.** Do **not** collect more of the same five cities — that re-measures the sample that already produced the number. Run the pre-registered forward gate on the breadth universe (49 cities), where clusters accumulate ~10× faster, and require the core band to clear a clustered CI > 0 **there**.

**Sample size.** At the measured +7¢ with per-bet sd 0.444, 2σ needs **~120 independent city-days (~8 weeks at 15/wk)**. At the pre-registered +3¢ threshold it needs **~878 (~59 weeks)** — and ~1,760 for 80% power, which is the repo's own figure. **The gate as written is unreachable in 3 months on 5 cities.** It is reachable on the breadth universe.

**Why it might fail here.** (a) It is one cell out of ~25 I tested and ~3 the repo has tested, and the winning cell has moved every time; (b) Gamma mid ≠ executable bid; (c) depth — the NBA study measured 14.8 shares of executable size on constrained opportunities; (d) the March 30 2026 weather-fee activation is a regime break inside the sample.

---

## 3. Maker-side overround harvest

**Edge source:** the book's YES prices sum to more than 1, so a maker who is resting on the sell side of every leg is being paid a structural premium for supplying the mass the crowd wants to buy — a service fee, not a forecast.

**Effect size (mine, Part B5).** Median gross **+2.80¢/book**; taker fee to capture it **3.62¢** ⇒ net **+0.39¢** as taker. As a maker: fee 0, plus ~25% weather rebate. Anatomy paper: median adverse-selection component **0.0** on top-100 markets, ~32 effective makers/market.

**Data needed.** CLOB order-book snapshots. The decisive statistic is **Σ(best bid) across a complete book**: if Σbid > 1 the premium is takeable today; if Σbid < 1 < Σask it lives entirely inside the spread and only a maker gets it.

**Cheapest honest test.** Poll the CLOB book for one city's complete book every 15 min for 2 weeks and tabulate Σbid and Σask. This is a **pure data question, not a trading question** — no positions, no gates, and it simultaneously calibrates `HALF_SPREAD` (currently a hardcoded 0.01 with a "tune once real order-book data exists" comment).

**Sample size.** ~2 weeks of book snapshots. Trivial.

**Why it might fail here.** Requires ~10 simultaneous fills; you get filled first on the leg that is about to win (adverse selection is not zero on weather books — the near-zero measurement was on top-100 markets, which weather is not); depth decays 6% per 10× reduction in seconds-to-close; and 0.28% gross ROI on capital per event is a thin carry that only compounds if capital recycles fast, which NegRisk conversion helps but does not solve.

---

## 4. Kalshi as a second consensus + the settlement-ruler basis

**Edge source:** Kalshi runs the same cities' daily temperature contracts but settles on the **NWS Daily Climate Report over the local-standard-time day**, while Polymarket settles on **wunderground.com** hourly-METAR extremes over the local calendar day — this project owns the only validated model of when those two rulers disagree (`wu_truth.py`, 59/60 settlement audit), so it knows the settlement distribution better than a trader watching either feed alone.

**Effect size.** Kalshi source confirmed at help.kalshi.com/en/articles/13823837-weather-markets ("the final climate report issued by the National Weather Service... local standard time"). Repo-measured divergence: **4/60 settlements (6.7%)** graded backwards pre-W0; **18/573 rows (3.1%)**, 15 market-days flipped verdict when the ruler changed.

**Data needed.** Kalshi's public API (free). Nothing else — both truth channels already exist locally.

**Cheapest honest test.** Start archiving Kalshi temperature quotes alongside the existing hourly Polymarket snapshots. Two questions come free: (i) does Kalshi's price lead Polymarket's (cross-correlation at 1 h lags)? — a *borrowed consensus* is not out-forecasting the consensus; (ii) on days where the two rulers would give different bins, does either book price the difference?

**Sample size.** Question (i) needs ~4–6 weeks of paired quotes. **Question (ii) is fatally underpowered and should be flagged as such:** ruler-divergence days are ~3% of city-days ⇒ ~5 in 178, ~6 more in 3 months. **Treat the ruler basis as a risk control (never hold a boundary-day position into resolution), not as a book.**

**Why it might fail here.** Kalshi's contracts are threshold-exceedance, not 1°F bins, so there is no clean leg-for-leg mapping; capital sits on two venues; and if Kalshi leads, the lead is likely inside the ~2–6 h window in which this repo already measured that the market absorbs forecast news (§1d).

---

## 5–7. Rejected on this repo's own data

- **Deep [5,10)¢ short:** +0.0023 [−0.0391, +0.0437] pre-day; **−0.0855 same-day**. Not a trade. The Polymarket-wide arithmetic (premium 0.0023 vs fee 0.0024) says the same. Do not trade it, and do not trade it same-day in either direction without a separate hypothesis.
- **Moderate [10,25)¢ short (current Leg 1b):** −0.0063 [−0.0565, +0.0440]. The band should be re-pre-registered as [25,35)¢ or the leg retired. Note the tightening-only rule: swapping bands is *not* a tightening, so this must be a **new** leg with a **new** forward clock, not an amendment.
- **Favorite leg, buy 65–85¢ pre-day:** n=3 markets in 4 months. Untradeable as specified; only reachable as a same-day book.

## 8–10. Rejected from the literature

- **Coherence/dutch-book arbitrage:** 0.0001% of time in arbitrage, 3.6 s median duration, ~15 shares. Unreachable at hourly polling, uneconomic at any polling rate we would build.
- **Weather-derivative risk premium:** no hedging demand exists in daily bins. Zero transfer.
- **Round-number / tick clustering:** real (returns divisible by 5 ticks over-represented; retail trades cluster on integers) but it is an execution consideration for maker placement, not an edge. Fold into candidate #3's quote placement if #3 proceeds.

---

# PART D — Power, and what 3 months can and cannot settle

At **~15 independent city-days/week ⇒ ~195 city-days in 3 months**. A city-day resolves on one temperature scalar, so it is the unit of independence, not the bin.

| effect | price | sd/bet | n for 2σ | weeks @15/wk |
|---|---|---|---|---|
| +0.0317 | 0.075 | 0.263 | 276 | 18.4 |
| +0.0230 | 0.075 | 0.263 | 525 | 35.0 |
| **+0.0810** | **0.271** | **0.444** | **120** | **8.0** |
| +0.0300 (the pre-registered core gate) | 0.271 | 0.444 | 878 | 58.5 |
| +0.0200 | 0.150 | 0.357 | 1,275 | 85.0 |

**Explicitly flagged as unreachable in ~3 months on the 5-city universe:** the [25,35)¢ gate at its pre-registered +3¢ threshold (≈59 weeks to 2σ, ≈2 years at 80% power — matching the repo's own ~1,760-bet figure); any deep-band effect at the +2.3¢ magnitude (≈35 weeks); anything in [10,25)¢ (≈85 weeks); and the ruler-divergence basis (~11 usable days ever).

**Reachable in ~3 months:** the horizon restriction (needs no new data at all); the Σbid/Σask book question (2 weeks); Kalshi lead-lag (4–6 weeks); and the [25,35)¢ core band **if and only if** it is run on the 49-city breadth universe, where clusters accumulate ~10× faster.

---

# Sources

- Bürgi, Deng & Whelan, *Makers and Takers: The Economics of the Kalshi Prediction Market* — https://cepr.org/publications/dp20631 · https://www2.gwu.edu/~forcpgm/2026-001.pdf · https://ideas.repec.org/p/pra/mprapa/126350.html · https://www.karlwhelan.com/sports-betting-kalshi-prediction-market/
- Qin & Yang, *Polymarket-v1 Database*, arXiv:2606.04217 — https://arxiv.org/html/2606.04217v1
- Le, *Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets*, arXiv:2602.19520 — https://arxiv.org/html/2602.19520v1
- Reichenbach & Walther, *Exploring Decentralized Prediction Markets*, SSRN 5910522 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522
- *The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book*, arXiv:2604.24366 — https://arxiv.org/html/2604.24366v1
- *Arbitrage Analysis in Polymarket NBA Markets*, arXiv:2605.00864 — https://arxiv.org/html/2605.00864v1
- *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets*, arXiv:2508.03474 (AFT 2025) — https://arxiv.org/abs/2508.03474
- *Optimal Market Making in Prediction Markets*, arXiv:2607.17991 — https://arxiv.org/html/2607.17991
- Dujava, *Exploiting Mean-Reversion in Decentralized Prediction Markets* (QuantPedia, 2026) — https://quantpedia.com/exploiting-mean-reversion-in-decentralized-prediction-markets-evidence-from-polymarket-binary-contracts/
- *Systematic Edges in Prediction Markets* (QuantPedia) — https://quantpedia.com/systematic-edges-in-prediction-markets/
- Data Golf, *The Favourite-Longshot Bias is not a bias* — https://datagolf.com/fav-longshot-not-a-bias
- Green, Lee & Rothschild, *The Favorite-Longshot Midas* — https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2018/08/The-Favorite-Longshot-Midas.pdf
- *Favourite-longshot bias* (overview + horse-racing magnitudes) — https://en.wikipedia.org/wiki/Favourite-longshot_bias
- Cao & Wei (2004), *Weather derivatives valuation and market price of weather risk*, J. Futures Markets 24:1065-1089 — http://www.yorku.ca/mcao/cao_wei_weather_JFM
- Benth & Lempa (2024), *Hedging temperature risk with CDD and HDD temperature futures* — https://onlinelibrary.wiley.com/doi/full/10.1002/asmb.2815
- Polymarket docs — Negative Risk Markets: https://docs.polymarket.com/advanced/neg-risk · Trading Fees (help art. 13364478) · Maker Rebates (help art. 13364471) · https://docs.polymarket.us/fees
- Kalshi Help Center, *Weather Markets* — https://help.kalshi.com/en/articles/13823837-weather-markets

**Low-weight / marketing sources consulted and discounted:** laikalabs.ai, botforkalshi.com, startpolymarket.com, tradetheoutcome.com, polytraderbot.com, medium.com/mountain-movers, pillarlabai.com. Used only for the on-chain wallet identifier in A1; no quantitative claim in this report rests on them.
