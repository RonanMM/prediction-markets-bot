# Market landscape scan — where structural edge might live that is NOT "forecast the weather better"

Compiled 2026-07-31. All numbers below are from primary sources queried live (Kalshi public API,
Polymarket Gamma/CLOB API) or measured from this repo's own data, unless a web citation is given.
Web-sourced claims are marked `[web]` and should be treated as weaker than the API/measured ones.

---

## 0. READ THIS FIRST — two blocking constraints

### 0a. Venue access for a UK-based operator

| venue | UK retail access | basis |
|---|---|---|
| **Kalshi** | **NO** — UK is on the restricted-jurisdiction list. Kalshi holds a CFTC DCM licence, which authorises US persons; onboarding requires SSN + US address and blocks at country-of-residence before KYC upload. `[web]` | Kalshi Member Agreement / help centre |
| **Polymarket (global, polymarket.com)** | **NO** — UK IPs geoblocked since the Gambling Commission action; FCA's 2019 retail binary-options ban also bites. `[web]` | multiple 2026 sources |
| **Polymarket US (QCX LLC, polymarket.us)** | **NO** — US-only CFTC DCM, separate order book from global. `[web]` | docs.polymarket.us |

**Consequence:** *none* of the trading ideas below are cleanly executable from the UK today. That is a
legal question for the operator, not one I can resolve, and it should be settled **before** any
engineering. What is unambiguously fine is the **data** side: Kalshi's market-data API is fully
public and unauthenticated (verified below — no account, no KYC, no US nexus), so Kalshi as an
*information source* is available regardless of trading eligibility.

This reranks everything. The report is therefore split into:
- **Track A — data/analysis only** (legal from anywhere, actionable now)
- **Track B — requires a tradeable account** (gated on 0a)

### 0b. Our documented fee model is stale in the prose, correct in the code

`CLAUDE.md` says "2% fee on winning payout". The **live** Gamma `feeSchedule` on every Polymarket
weather market is:

```json
{"exponent": 1, "rate": 0.05, "takerOnly": true, "rebateRate": 0.25}   // feeType: "weather_fees"
```

i.e. **taker fee = 0.05·p·(1−p) per share; makers pay ZERO; 25 % of collected taker fees are
rebated to makers.** `shoulder_book.py` already uses `0.05·p·(1−p)` and is right. The prose in
CLAUDE.md is what is wrong and should be corrected — it materially understates how much cheaper
maker-side execution is than taker-side.

---

## 1. Kalshi — full inventory (the coordinator's priority item)

### 1.1 API access — verified live, no auth

Every call below returned 200 with **no API key, no account, no KYC**, from this machine:

| endpoint | works unauth? | returns |
|---|---|---|
| `GET /trade-api/v2/series?category=Climate and Weather` | ✅ | 291 weather series w/ `settlement_sources` (name + URL) |
| `GET /trade-api/v2/markets?series_ticker=…&status=open` | ✅ | strikes, `yes_bid/ask`, sizes, `volume_fp`, `open_interest_fp`, full `rules_primary` |
| `GET /trade-api/v2/markets/{ticker}/orderbook` | ✅ | full depth ladder both sides |
| `GET /trade-api/v2/markets/trades?ticker=…` | ✅ | public tape incl. taker side |
| `GET /trade-api/v2/series/{s}/markets/{m}/candlesticks` | ✅ | **1-minute** OHLC + `yes_bid`/`yes_ask` + volume + OI |
| `GET /trade-api/v2/events?series_ticker=…&status=settled` | ✅ | event list; `result` field gives the settlement |

Base host: `https://api.elections.kalshi.com`.

**Historical depth — the important caveat.** The *events* index goes back to **2021-08-06**
(1,816 settled `KXHIGHNY` events). But the *market objects* — and therefore the candlesticks, which
can only be addressed via a market ticker — are only served for roughly the **last ~2 months**.
Probed boundary on 2026-07-31:

```
KXHIGHNY-26APR15  → 0 markets
KXHIGHNY-26MAY15  → 0 markets
KXHIGHNY-26MAY25  → 6 markets   ← boundary
KXHIGHNY-26JUN15  → 6 markets + 37 hourly candles
KXHIGHNY-26JUL30  → 6 markets + 1,556 one-minute candles
```

**So: you get ~2 months of free backfill and must archive forward from there.** This is exactly the
"start collecting NOW" call the megaplan made on 2026-07-12 and never executed — every day of delay
since then is a day of history that has now aged out of the ~2-month window. The cost of the
archiver is trivial (one unauthenticated GET loop; no key to rotate, no rate-limit auth tier).

### 1.2 What Kalshi actually lists

291 series in `Climate and Weather`. The tradeable-daily core:

**Daily high temperature — 20 cities, all NWS Climatological Report (CLI):**
`KXHIGHNY` (Central Park), `KXHIGHCHI` (**Chicago Midway**), `KXHIGHLAX`, `KXHIGHMIA`, `KXHIGHAUS`
(Austin-Bergstrom), `KXHIGHDEN`, `KXHIGHPHIL`, `KXHIGHTDC` (DCA), `KXHIGHTPHX`, `KXHIGHTSEA`,
`KXHIGHTBOS`, `KXHIGHTDAL` (DFW), `KXHIGHTATL`, `KXHIGHTHOU` (Hobby), `KXHIGHTLV`, `KXHIGHTMIN`,
`KXHIGHTNOLA`, `KXHIGHTOKC`, `KXHIGHTSATX`, `KXHIGHTSFO`.

**Daily low temperature —** parallel `KXLOWT*` set, same stations, same CLI source.

**Structure:** 6 markets per city-day — a `less` floor, four exclusive **2 °F** `between` bins, and a
`greater` cap. E.g. `KXHIGHTSEA-26JUL31`: `80° or below / 81–82 / 83–84 / 85–86 / 87–88 / 89° or above`.

**Other weather:** hourly directional temperature (`KXTEMPNYCH`, `KXTEMPCHIH`, … — resolved by **The
Weather Company**, *not* NWS); daily/monthly rain (`KXRAINNYC` $2.2 M lifetime over 53 events,
`KXRAIN` "where will it rain" $1.1 M / 11 events, `KXRAINNYCM`/`KXRAINCHIM` monthly ~$250 K each);
snow (`KXSNOWNY*`, seasonal, currently dormant); plus long-dated climate (Arctic ice, global temp,
CO₂), natural-disaster and earthquake series.

### 1.3 Liquidity — Kalshi's daily temp books are 5–20× Polymarket's

Volume per settled city-day (contracts ≈ $1 notional), 2026-07-28…30:

| series | vol / city-day | series | vol / city-day |
|---|---:|---|---:|
| KXHIGHLAX | **531 K – 733 K** | KXHIGHTSFO | 62 K – 79 K |
| KXHIGHMIA | 165 K – 220 K | KXHIGHTATL | 63 K – 98 K |
| KXHIGHNY | 173 K – 273 K | KXHIGHTDAL | 64 K – 74 K |
| KXHIGHCHI | 100 K – 163 K | KXLOWTNYC | 39 K – 77 K |
| KXHIGHTPHX | 94 K – 126 K | KXHIGHTDC | 18 K – 34 K |
| KXHIGHTSEA | 86 K – 144 K | KXHIGHTNOLA | 14 K – 18 K |

Median quoted spread on open daily-temp markets: **1–2.5 ¢**. Compare Polymarket NYC daily-high
**event** volume: ~$26 K/24 h; LA ~$32 K/24 h. Kalshi NYC is ~8× deeper, Kalshi LA ~20× deeper.

### 1.4 Overlap with our five cities — and the resolution-source problem

| our city | Polymarket station + ruler | Kalshi equivalent | overlap? |
|---|---|---|---|
| **NYC** | **KLGA LaGuardia**, Wunderground | **KNYC Central Park**, NWS CLI | different **station** *and* different ruler |
| **Chicago** | **KORD O'Hare**, Wunderground | **KMDW Midway**, NWS CLI | different **station** *and* different ruler |
| London | EGLC, Wunderground | — none — | no Kalshi market |
| Seoul | RKSI, Wunderground | — none — | no Kalshi market |
| Hong Kong | HKO, weather.gov.hk | — none — | no Kalshi market |

**Only 2 of our 5 cities overlap at all, and in both the physical station is different.** Central
Park vs LaGuardia routinely differ by several °F (park vs waterfront tarmac); O'Hare vs Midway
likewise. This is **basis risk, not arbitrage** — precisely the trap the coordinator flagged.

### 1.5 The much better overlap set — 7 SAME-STATION city pairs

Polymarket has expanded well past our five. Cross-referencing every Polymarket US weather market's
resolution URL against Kalshi's `rules_primary`:

| city | Polymarket station (WU) | Kalshi station (CLI) | same station? | 2 °F bin edges aligned? |
|---|---|---|---|---|
| **Los Angeles** | KLAX | KLAX | ✅ | ✅ (both even-start) |
| **Austin** | KAUS | KAUS Bergstrom | ✅ | ✅ |
| **Atlanta** | KATL | KATL | ✅ | ✅ |
| **Houston** | KHOU Hobby | KHOU Hobby | ✅ | ✅ |
| **Miami** | KMIA | KMIA | ✅ | ✗ off by 1 °F |
| **Seattle** | KSEA | KSEA | ✅ | ✗ off by 1 °F |
| **San Francisco** | KSFO | KSFO | ✅ | ✗ off by 1 °F |
| Chicago | KORD | KMDW | ✗ | ✅ |
| NYC | KLGA | KNYC | ✗ | ✅ |
| Dallas | KDAL Love | KDFW | ✗ | ✅ |
| Denver | KBKF Buckley | KDEN | ✗ | ✗ |

Bin edges float with the forecast, so alignment must be re-checked per day — but the **station**
mapping is fixed and is the part that matters.

### 1.6 The ruler gap, measured — 1,668 station-days, not an assertion

For the 7 same-station pairs the only difference is the **ruler**: Polymarket reads Wunderground
(max over the *hourly* METARs, station-local calendar day); Kalshi reads the NWS CLI (max over
*1-minute* ASOS data, local-**standard**-time day). I measured this directly on our own archives
(`{slug}_obs_hourly.csv` vs `{slug}_historical_actuals.csv`, 2022-01-01 → 2026-07-28, rounded to
whole °F exactly as the markets read it):

| station | n days | P(CLI ≥ WU) | P(CLI > WU) | P(CLI < WU) | mean CLI−WU |
|---|---:|---:|---:|---:|---:|
| **KLGA** | 1,668 | **99.40 %** | 55.6 % | 0.60 % | **+0.66 °F** |
| **KORD** | 1,670 | **99.40 %** | 50.4 % | 0.60 % | **+0.61 °F** |

Distribution of CLI−WU at KORD: `{−1: 10, 0: 818, +1: 684, +2: 140, +3: 15, +4: 2, +5: 1}`.

Two things follow, and they are the analytical core of this report:

1. **The CLI reading dominates the WU reading 99.4 % of the time.** `WU_max ≤ CLI_max` almost
   surely, because hourly METARs are a strict subsample of the 1-minute record. So for any
   threshold T, `P(WU ≥ T) ≤ P(CLI ≥ T)` — a *one-sided* relationship, not a symmetric one.
2. **The fair cross-venue basis is not zero, it is ≈ +0.6 °F.** Anyone treating "same city, same
   day" as equivalent is systematically wrong at bin boundaries; with 2 °F bins a +0.6 °F mean shift
   moves ~25–30 % of the probability mass across an edge.

We are unusually well placed to exploit this: the project has already built and validated
`wu_truth.py` and been burned four times learning the difference. **The ruler-conversion function is
an asset nobody else appears to have bothered to measure.** Caveat: measured at KLGA/KORD only; it
must be re-measured per station before use (IEM serves both feeds for all seven, ~1 h of fetching).

### 1.7 Kalshi fees vs Polymarket

| | taker | maker | notes |
|---|---|---|---|
| **Polymarket global (weather)** | `0.05·p·(1−p)` per share | **0** | +25 % of taker fees rebated to makers; `takerOnly: true` (from live `feeSchedule`) |
| **Polymarket US** | `0.06·p·(1−p)`, cap $1.50/100 @ 50 ¢ | **−0.0125·p·(1−p)** (maker is *paid*) | eff. 2026-07-01 `[web]` |
| **Kalshi** | `ceil(0.07·p·(1−p)·100)/100` ≈ $0.0175 @ 50 ¢ | ≈ 25 % of taker (~$0.0044 @ 50 ¢) | makers **pay**, they are not paid `[web]` |

**Polymarket is the cheaper venue on both sides** — materially so for makers (0 or negative vs a
positive maker fee on Kalshi).

---

## 2. Polymarket's weather category — it is now ~50 cities, not 5

Live Gamma enumeration (`tag_slug=weather`, open events, 2026-07-31): **224 open events / 2,014
markets**, of which **135 events are daily-high-temperature** and 22 daily-low.

Daily-temp 24 h volume ≈ **$2.54 M** across the high-temp events alone. Cities and 24 h volume:

| tier | cities (24 h vol) |
|---|---|
| **$100 K+** | Shanghai 236 K, Hong Kong 218 K, Wellington 165 K, Tokyo 130 K, Guangzhou 115 K |
| **$60–100 K** | Shenzhen 97 K, Seoul 97 K, Chengdu 96 K, Qingdao 47 K*, Beijing 76 K, London 74 K, Madrid 69 K, Milan 68 K, Amsterdam 64 K, Munich 48 K, Paris 62 K, Taipei 81 K, Busan 81 K |
| **$20–60 K** | Singapore, Wuhan, Chongqing, Helsinki, Kuala Lumpur, Jeddah, Cape Town, Karachi, Manila, Ankara, Lucknow, Tel Aviv, Warsaw, Moscow, Istanbul, São Paulo, Buenos Aires, Toronto, Mexico City, NYC 37 K, Miami 34 K, Dallas 30 K, Chicago 25 K, Denver, Austin, Atlanta, LA 45 K, SF, Houston, Seattle |
| **$0 (new listings)** | **Jinan, Zhengzhou** |

**Structure:** every daily-temp event is **11 exclusive bins**, `negRisk: true`, `orderMinSize: 5`,
tick `0.001`, `rewardsMaxSpread: 4.5 ¢`.

**Resolution sources** (extracted from each market's description — all verified, not assumed):
- **Wunderground station page** for ~46 cities (`EHAM`, `KATL`, `KAUS`, `ZBAA`, `SAEZ`, `RKPK`,
  `FACT`, `ZUUU`, `KORD`, `ZUCK`, `KDAL`(Love), `KBKF`(Buckley), `ZGGG`, `EFHK`, `KHOU`(Hobby),
  `ZSJN`, `OPKC`, `WMKK`, `EGLC`, `KLAX`, `VILK`, `LEMD`, `RPLL`, `MMMX`, `KMIA`, `LIMC`, `EDDM`,
  `KLGA`, `MPMG`, `LFPB`(Le Bourget), `ZSQD`, `KSFO`, `SBGR`, `KSEA`, `RKSI`, `ZSPD`, `ZGSZ`,
  `WSSS`, `RCSS`, `RJTT`(Haneda), `CYYZ`, `EPWA`, `NZWN`, `ZHHH`, `ZHCC`, `LTAC`)
- **`weather.gov/wrh/timeseries`** for Moscow (`UUWW`), Istanbul (`LTFM`), Tel Aviv (`LLBG`) — a
  *different ruler again* (NOAA timeseries page, not WU). Any expansion into these three needs its
  own truth channel; do not assume the WU reconstruction applies.
- **HKO `climat.htm`** for Hong Kong (as we already handle).

**Non-temperature markets on Polymarket are thin.** Ranked by 24 h volume, everything that is not a
daily temp bin: Super Typhoon Dolphin $13 K, weekly earthquake-count $5 K, **Precipitation in
Seattle in July $2.3 K**, hottest-year ranking $2.2 K, **Precipitation in NYC in July $0.6 K**,
**Precipitation in London in July $0.4 K**, Mt. Washington wind speed $19. There is **no daily
"will it rain" market on Polymarket** — precipitation exists only as *monthly aggregates* with
three-figure daily volume.

---

## 3. Liquidity rewards — the largest un-exploited structural payment I found

Summing `clobRewards.rewardsDailyRate` across all open Polymarket weather temperature markets:

> **$16,198 / day**, spread over 491 market-reward entries.

Per city-day it concentrates in the 2–3 near-the-money bins:

```
Highest temperature in NYC on July 31   (event 24 h vol $25.7 K)
  82-83°F   $19/day    84-85°F  $188/day    86-87°F  $165/day    88-89°F   $28/day
Highest temperature in LA on July 31    (event 24 h vol $32.0 K)
  76-77°F  $142/day    78-79°F  $245/day    80-81°F   $11/day
```

Reward-band terms: quote within **4.5 ¢** of the size-adjusted midpoint, minimum **100 shares** on
the incentivised bins (20 on the rest); paid daily at 00:00 UTC; score is proportional share with a
quadratic penalty for distance from mid `[web]`.

Measured book depth on the LA 78-79 °F bin (the $245/day one), live CLOB:

```
BIDS  0.49×264  0.48×1009  0.47×316  0.46×200        → ≈ $1.5 K within the reward band
ASKS  0.51×16   0.52×878   0.53×1107  0.55×70        → ≈ $1.5 K within the reward band
```

So a **$1,000/side quote is a ~40 % share of a $245/day pool on a market that lives ~2 days**, on
top of a **zero** maker fee and a 25 % rebate on taker fees. The nominal rate on capital is
extraordinary — which is exactly why it should be assumed to be competed away or to carry a hidden
cost. The hidden cost has a name: adverse selection. See §5 #3.

---

## 4. NegRisk mechanics

Every Polymarket daily-temp event is `negRisk: true` — 11 mutually exclusive bins under the
NegRiskAdapter, where `1 NO share in outcome i = 1 YES share in every other outcome + 1 USDC` `[web]`.
Two mechanical constraints follow:

- **Convert arb:** if Σ(YES asks over all 11 bins) < $1 (plus 11 legs of taker fee), buy the set —
  exactly one must pay $1.
- **NO-side arb:** if Σ(NO bids) > $10 (= n−1), sell the set.

The α5 signal already *scores* PMF incoherence but does not *trade* it, and CLAUDE.md notes α5 was
liquidity-guarded precisely because incoherence is usually thinness. That guard is right for a
scoring signal and wrong for an execution signal — a convert arb doesn't care why the sum is off.

---

## 5. RANKED OPPORTUNITIES

---

### #1 — Kalshi as a second consensus, transferred through a measured station+ruler function
**Track A (data only — no Kalshi account, no KYC, legal from the UK).**

1. **Edge source in one sentence.** Kalshi's daily-temperature books are 5–20× deeper and quoted by
   a different, CFTC-regulated, US crowd, so their price contains a consensus that the Polymarket
   price for the *same station* demonstrably does not — and we can carry it across the venue
   boundary because we have measured the ruler-conversion function (P(CLI ≥ WU) = 99.4 %, mean
   +0.6 °F, n = 1,668) that nobody else has bothered to build.
2. **What's tradeable, where, what source, what fee.** You trade **Polymarket** daily-temp bins as
   usual (WU station page, taker `0.05·p·(1−p)`, maker 0). Kalshi is *input only*. Best targets are
   the **7 same-station pairs** (LAX, AUS, ATL, HOU, MIA, SEA, SFO — §1.5), where the only gap is
   the ruler. NYC/Chicago also work but need a second, larger station-to-station transfer
   (KNYC→KLGA, KMDW→KORD) on top of the ruler transfer.
3. **Needs a forecast edge?** **No.** It needs *someone else's* forecast plus a historical transfer
   function. This is the one candidate that satisfies the project's own stated re-entry condition —
   "revisit forecast alpha only on a genuinely new information source" — because Kalshi's price
   series is genuinely new information, not a better model of the same information.
4. **Cheapest honest test.** (a) Build the archiver *this week* — an unauthenticated GET loop over
   ~20 series' open markets + orderbook, appended alongside the existing hourly collect cycle; grab
   the ~2 months of backfill that is still in the window before it ages out. (b) Fetch IEM METAR
   hourly + NWS CLI daily for the 7 same-station airports and re-measure the CLI−WU transfer per
   station (~1 h). (c) After 4–6 weeks, run a **paired Brier** of `transfer(Kalshi price)` vs the
   contemporaneous Polymarket price on the same market-days, against our existing settlement-faithful
   labels — reusing the `evaluate_oos.py` pairing machinery verbatim, with a pre-registered forward
   gate exactly like the E3 buckets. No new statistics to invent.
5. **Why it might fail for us.** (i) Somebody is probably already arbing these two venues, so the
   transferred price may equal the Polymarket price and the gap is zero — that is the null and the
   test is designed to find it fast. (ii) The transfer adds variance; a sharper prior degraded by a
   noisy conversion can end up *worse* than the local price. (iii) The two venues' books close at
   different times, so a naive as-of join leaks lookahead — the join must be timestamped, and this
   is exactly the class of silent bug (green run, flattering number) this project has hit nine times.
   (iv) Only 2 of our current 5 cities overlap, and both on the *wrong* stations — the payoff is
   concentrated in cities we do not yet trade.

---

### #2 — Cross-venue dominance spread (WU ≤ CLI), same-station cities
**Track B — BLOCKED for a UK operator (needs a funded Kalshi account). Listed second because if the
legal read in §0a is wrong, this is the best thing in the report.**

1. **Edge source in one sentence.** `WU_max ≤ CLI_max` on 99.40 % of 1,668 station-days, so Kalshi's
   "≥ T" contract *dominates* Polymarket's "≥ T" contract on the same station — any state that pays
   the Polymarket leg also pays the Kalshi leg — which makes any price inversion between them a
   near-riskless spread rather than a directional view.
2. **What's tradeable.** Buy Kalshi YES(≥T) / sell Polymarket YES(≥T) whenever
   `P_poly(≥T) − P_kalshi(≥T) > costs`, on LAX/AUS/ATL/HOU (aligned bins — cleanest) and
   MIA/SEA/SFO (1 °F offset — needs CDF-ladder composition). Kalshi taker `0.07·p(1−p)`, Polymarket
   taker `0.05·p(1−p)`; both legs settle T+1 morning.
3. **Needs a forecast edge?** **No — none at all.** It is a pure inequality between two rulers.
4. **Cheapest honest test.** Free and immediate: archive both venues' books for the 7 pairs and
   count how often the inversion appears, how large, and how deep. Zero capital required to measure.
   Then re-measure P(CLI ≥ WU) at the seven airports themselves (currently only proven at KLGA/KORD).
5. **Why it might fail.** (i) **Legality — a UK resident cannot hold the Kalshi leg.** (ii) Capital
   is double-posted across two venues with no cross-margin, so the return on *total* capital is
   roughly half the headline. (iii) The 0.6 % violation tail is small but not zero, and the CLI day
   is local-*standard*-time while WU's is the local calendar day — a genuine (if rare) boundary
   break, mostly harmless for Tmax which peaks mid-afternoon, more dangerous for Tmin. (iv) The
   inversion is the most obvious free money on two public APIs; assume latency-advantaged bots
   already take it and that what is left at minute-latency is the residue.

---

### #3 — Market-make Polymarket weather for the liquidity rewards + zero maker fee
**Track B (Polymarket account) — but no Kalshi dependency and no forecast dependency.**

1. **Edge source in one sentence.** The exchange pays **$16,198/day** to whoever holds a quote within
   4.5 ¢ of mid on weather bins, and charges makers **nothing** — a payment that exists outside the
   price and is therefore not something the price can be "right" about.
2. **What's tradeable.** Resting two-sided limit orders on the 2–3 near-the-money bins of any of ~50
   Polymarket weather events, min 100 shares, spread ≤ 4.5 ¢. Fee: **0** as maker, plus a 25 % share
   of taker fees collected. Resolution source is the same WU station pages we already grade against.
3. **Needs a forecast edge?** **No** — but it needs a *fair-value* estimate good enough not to be
   picked off, which is a weaker requirement than beating the market. Our calibrated PMF is already
   close to the market (Brier 0.166 vs 0.128); that gap is fatal for *taking* and much less fatal for
   *quoting around mid*.
4. **Cheapest honest test.** Pure paper, no capital: extend the existing `shoulder_book.py` pattern —
   at each collect cycle record the quotes you *would* have rested (bin, side, price, size), then
   reconstruct fills from the CLOB trade tape and settle at truth. Compare `reward accrual + rebate`
   against `realised adverse selection`. The whole thing is offline arithmetic on data we already
   pull, plus the reward-rate field we now know exists.
5. **Why it might fail.** (i) **Adverse selection is the entire question** — you get filled precisely
   when someone knows more, and on same-day markets the running METAR max makes "knowing more"
   trivially easy; the mitigation (quote pre-day only, pull on the target day) also cuts you off from
   the fattest reward pools. (ii) Reward share is proportional and dilutes the moment anyone else
   sizes up. (iii) Inventory: with 11 correlated bins per event you accumulate a temperature position
   whether you want one or not, and a single modest-capital operator has no hedge for it. (iv) It
   needs a live order-management loop with cancel/replace at minute cadence — a genuinely different
   piece of engineering from the current append-only collector, and a new class of failure (stale
   quote left resting through a forecast move).

---

### #4 — Trade the negRisk coherence constraint instead of merely scoring it
**Track B (Polymarket only).**

1. **Edge source in one sentence.** In a `negRisk` event exactly one of 11 bins pays $1, so
   Σ(YES asks) < $1 is a *mechanical* violation that requires no opinion about the weather whatsoever.
2. **What's tradeable.** Buy the full 11-bin set on any Polymarket daily-temp event when
   Σ(asks) + Σ(fees) < $1 (or sell the NO set when Σ(NO bids) > $10). ~157 open temp events ×
   2 sides × 50 cities is a large surface. Fee: 11 legs × `0.05·p·(1−p)` — small at extreme prices,
   which is where incoherence tends to live.
3. **Needs a forecast edge?** **No — zero.**
4. **Cheapest honest test.** **Free, offline, today.** We already store months of per-bin snapshots.
   Reconstruct Σ(best ask) per event per snapshot, subtract the 11-leg fee, and count violations by
   size and by the min depth across legs. That answers "does this exist, how big, how often, and how
   much size" without a single new API call. If the answer is "never, or 0.1 ¢ on 5 shares", drop it
   in an afternoon.
5. **Why it might fail.** (i) Almost certainly already harvested by faster bots — minute latency is
   the wrong weapon for the most-watched free-money pattern on the venue. (ii) Fill size is capped by
   the *thinnest* leg, so a 1 ¢ violation on a bin with 5 shares of depth is worth 5 ¢. (iii) Partial
   fills convert a riskless set into an outright position — this needs atomic-ish execution logic we
   don't have. (iv) Our snapshots are hourly, so a historical study *understates* transient violations
   and *overstates* persistent ones; treat any positive result as needing live confirmation.

---

### #5 — Universe expansion: run the existing structure book across ~50 cities
**Track B (Polymarket only).**

1. **Edge source in one sentence.** None — and that is the point: this multiplies whatever edge the
   already-measured structure legs have (the 65–75 ¢ favourite band realised 0.807 vs 0.710 priced,
   ≈ +7.7 ¢/share net) across ~10× the market count, so it buys **statistical power and capacity**,
   not a new signal.
2. **What's tradeable.** `shoulder_book.py`'s two legs, unchanged, on all ~50 Polymarket weather
   cities. WU station pages for 46 of them; **Moscow/Istanbul/Tel Aviv use `weather.gov/wrh/timeseries`
   and need their own truth channel before they can be graded at all.**
3. **Needs a forecast edge?** **No** — the structure legs are explicitly model-free.
4. **Cheapest honest test.** Discovery already sees these (264 markets/cycle post-fix). Add the new
   cities to `resolution_anchors.py`, stand up truth feeds (IEM METAR daily covers most of the ICAO
   list), and let the *existing* pre-registered gates run forward. Do **not** re-derive the shoulder
   result on the new cities and call it replication if you also tune on them.
5. **Why it might fail.** (i) The breadth book has already been tried once — the 2026-07-27 verdict
   was "full-band gate MET but underpowered, CI spans zero, **not replicated in breadth**". More
   cities did not rescue it then. (ii) Thin books mean the measured +7.7 ¢ may not be *fillable* at
   size; the 24 h volume on the mid-tier cities is $20–60 K across 11 bins. (iii) 46 new
   station-truth channels is 46 new opportunities for the silent-failure bug class that has bitten
   this project nine times. (iv) Grading lag varies by country and some feeds will be as awkward as
   HKO's monthly batch.

---

### #6 — "Max is locked" scanner on same-day Polymarket bins
**Track B (Polymarket only).**

1. **Edge source in one sentence.** Polymarket resolves on the *hourly* METAR max, so the settlement
   value is fully determined by 24 public observations and becomes effectively **known several hours
   before the day ends** — while the book may still price the settled bin below 100 ¢.
2. **What's tradeable.** Same-day bins on any WU-resolved city, after the diurnal peak has passed and
   the running hourly max can no longer be beaten. Taker `0.05·p·(1−p)` — negligible near 99 ¢.
3. **Needs a forecast edge?** **Almost none** — only the trivial "temperature does not exceed its
   afternoon peak after sunset", which our intraday fits already quantify (17:00 local σ ≈ 0.4 °C,
   c ≈ 0.95).
4. **Cheapest honest test.** Offline on existing data: join `{slug}_obs_hourly.csv` to our snapshot
   archive, and for each late-day snapshot compare the price of the bin containing the running max to
   1.0. Megaplan §10c tried a version of this and found the apparent lag was *our labels being
   wrong* — but that was **before** W0 fixed the ruler, so the negative result no longer stands and
   the re-test is cheap.
5. **Why it might fail.** (i) It is the most obvious edge in the market and the residual is probably
   1–2 ¢ of spread, not free money. (ii) Positive results here have already been shown once to be an
   artifact of a bad ruler — treat any repeat with maximum suspicion and grade only with `wu_truth`.
   (iii) METAR delivery latency and SPECI/corrected observations can move a "locked" max. (iv) Buying
   at 97 ¢ risks 97 to make 3; a single bad grade wipes 30 good ones.

---

### #7 — Non-temperature weather (precipitation, snow, wind) — **DROP**

1. **Edge source.** I cannot state one crisply, which by the brief's own rule means drop.
2. **What exists.** Polymarket has **no daily rain market at all** — only monthly aggregates
   ("Precipitation in NYC in July", 24 h volume **$552**; London **$389**; Seattle **$2,320**) and
   novelty wind (Mt. Washington, **$19**). Kalshi has real daily rain (`KXRAINNYC`, $2.2 M lifetime)
   but that is Track B *and* Kalshi *and* US-only.
3. **Forecast edge needed?** **Yes, heavily** — and it cuts **against** us. The premise "consensus NWP
   handles precipitation worse, so the market is softer" is half right: the market is softer *because
   the physics is harder*, and our error grows at least as fast as theirs. Our entire stack (EMOS,
   Gaussian/Student-t PMFs, CRPS, per-lead σ floors) is built for a continuous unimodal variable;
   precipitation is a mixed discrete–continuous variable with a point mass at zero, and none of the
   calibration machinery transfers.
4. **Test.** Not worth designing one.
5. **Failure.** Volume is three figures per day. Even a real edge could not be sized.

---

### #8 — CME weather futures (HDD/CDD) as a leading signal — **DROP**

1. **Edge source.** None usable. CME degree-day contracts are **monthly and seasonal cumulative**
   indices ($20 × index) — they carry information about a month's aggregate warmth, essentially none
   about whether Thursday's LaGuardia max lands in 84–85 °F or 86–87 °F.
2. **Availability.** Daily settlements are distributed via **CME DataMine**, which is a paid data
   product `[web]`. Volumes in the weather complex are thin and institutional.
3. **Forecast edge needed?** Moot.
4. **Test.** None worth running.
5. **Failure.** Wrong time resolution, wrong granularity, non-free data, and a venue requiring
   futures-account standing we plausibly do not have.

---

## 6. Who else is doing this

Searched GitHub, blogs and aggregators. Findings, all treated skeptically:

- **`suislanchez/polymarket-kalshi-weather-bot`** — trades Kalshi `KXHIGH` + Polymarket off 31-member
  GFS ensembles, 8 % edge threshold, fractional Kelly. README headlines "highest profits $1.8k".
  **No verifiable track record**, and the repo explicitly calls itself "a simulation tool for
  educational purposes". Notably it contains **no discussion of the resolution-source difference** —
  i.e. it treats Kalshi and Polymarket "NYC" as the same event when they are Central Park vs
  LaGuardia. That is the single most common error in this space and it is in the most-starred repo.
- **`ImMike/polymarket-arbitrage`** — cross-venue scanner over 10,000+ markets, generic, no weather
  specialisation, no published results.
- A cottage industry of Kalshi-weather content sites (`minutetemp.com`, `wethr.net`,
  `outcomeedge.live`, `betterweatherbettor.com`, `vweatherstation.com`, `botforkalshi.com`,
  `pillarlabai.com`) selling forecast-consensus dashboards and "playbooks". All sell the *forecast*
  angle. **None publishes an audited or independently verifiable P&L.**

**Read-through:** the public field is doing exactly the thing this project has already proven does
not work (beat the consensus with an ensemble), and is doing it *without* handling the ruler and
station problem. Our four-times-burned resolution discipline is the genuine comparative advantage,
and nothing I found suggests anyone else has built it.

---

## 7. Suggested order of work

| # | action | cost | gated on |
|---|---|---|---|
| 1 | **Settle §0a with the operator.** Every Track B item is void without it. | 0 | — |
| 2 | **Build the Kalshi archiver now.** Free, unauthenticated, and ~2 months of backfill is expiring daily. Pure data collection — legal irrespective of §0a. | ~half a day | — |
| 3 | **Offline negRisk coherence study** on the snapshots we already hold. Answers #4 with no new data. | ~an afternoon | — |
| 4 | Re-measure CLI−WU at the 7 same-station airports from IEM. | ~1 hour | — |
| 5 | Paper the maker/reward book (#3) using the `shoulder_book.py` pattern. | ~2 days | §0a for the live step |
| 6 | Pre-register the #1 forward gate before looking at any cross-venue number. | ~1 hour | archiver has 4–6 weeks |

**Two standing warnings.**
- The eval trackers carry no grade column; grading happens at read time. A cross-venue join is a new
  read-time surface, and every one of the nine documented silent failures in this project produced a
  green run and a plausible wrong number that flattered us. Whatever join is built must be verified
  against what renders, not against a passing workflow.
- Pre-register before measuring. The bet-selection search (32 subsets) and the breadth book both
  produced "promising" numbers that dissolved under their own MDE. Nothing here is exempt.

---

## Sources

Primary (queried live, 2026-07-31): `api.elections.kalshi.com/trade-api/v2/{series,markets,events,orderbook,candlesticks,trades}`;
`gamma-api.polymarket.com/events?tag_slug=weather`; `clob.polymarket.com/book`.
Measured: `src/polymarket_weather/data/weather/{new_york_city,chicago}_{obs_hourly,historical_actuals}.csv`.

Web:
- [Kalshi Weather Markets — Help Center](https://help.kalshi.com/en/articles/13823837-weather-markets)
- [Kalshi Liquidity Incentive Program](https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program)
- [Kalshi Fee Schedule (July 2026)](https://kalshi.com/docs/kalshi-fee-schedule.pdf) · [Market Math: Kalshi fees 2026](https://marketmath.io/blog/kalshi-fees-guide-2026)
- [Polymarket US Fee Schedule](https://docs.polymarket.us/fees) · [Polymarket Liquidity Rewards](https://docs.polymarket.com/market-makers/liquidity-rewards) · [Polymarket NegRisk overview](https://docs.polymarket.com/developers/neg-risk/overview)
- [Kalshi restricted countries incl. UK](https://laikalabs.ai/prediction-markets/kalshi-legal-supported-restricted-countries) · [Kalshi UK access attempt](https://hotminute.co.uk/2026/04/29/i-tried-to-use-kalshi-from-the-uk-in-2026-heres-what-worked-and-what-didnt/)
- [Polymarket blocked in the UK](https://startpolymarket.com/countries/united-kingdom/) · [Is Polymarket legal in the UK](https://predmarket.io/blog/polymarket-legal-uk)
- [suislanchez/polymarket-kalshi-weather-bot](https://github.com/suislanchez/polymarket-kalshi-weather-bot) · [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage)
- [CME Weather Futures fact card](https://www.cmegroup.com/trading/weather/files/weather-fact-card.pdf) · [CME Daily Settlements](https://www.cmegroup.com/market-data/daily-settlements.html)
