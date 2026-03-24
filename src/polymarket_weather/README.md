# Polymarket Weather Tracker

Tracks prediction market probabilities on Polymarket for temperature markets and
compares them against real-world weather forecasts to find pricing inefficiencies.

---

## Project Structure

```
polymarket_weather/
├── config.py                   # Cities, API endpoints, paths, color scheme
├── fetch_polymarket.py         # Gamma API + CLOB price history
├── fetch_weather.py            # Open-Meteo forecast (daily + hourly)
├── fetch_ensemble.py           # Open-Meteo ensemble forecast (40 model members)
├── processing.py               # Storage, dedup, implied temp calculation
├── visualization.py            # All 3 plot types
├── main.py                     # CLI entry point
├── polymarket_weather_analysis.py  # Inefficiency analyzer + betting bot
├── requirements.txt
└── data/
    ├── polymarket/             # {city}_snapshots.csv, {city}_price_history.csv
    └── weather/                # {city}_daily.csv, {city}_hourly.csv, {city}_ensemble.csv
└── plots/                      # PNG outputs
└── logs/                       # Per-run log files
└── output/                     # opportunities_v4.csv and analysis plots
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Recommended Daily Workflow

Run these two commands each day — the first fetches fresh data, the second finds bets:

```bash
# 1. Fetch latest weather forecasts + ensemble model data (keep market data from last full run)
python main.py --skip-polymarket

# 2. Analyze for opportunities and show what to bet right now
python polymarket_weather_analysis.py --data_dir ./data --no_plots --live --bankroll 1000
```

To also refresh Polymarket data (do this less frequently — it's slower):
```bash
python main.py
```

---

## main.py — Data Collection

Fetches and stores all data. Run once daily or on demand.

```bash
# Full pipeline: fetch Polymarket markets + weather forecasts + ensemble, then plot
python main.py

# Weather + ensemble only (faster, no Polymarket API calls)
python main.py --skip-polymarket

# Polymarket only
python main.py --skip-weather

# Re-generate plots from stored data without fetching anything
python main.py --plots-only

# Print market vs forecast summary without fetching or plotting
python main.py --summary-only

# Specific cities only
python main.py --cities Seoul London "New York City"

# Verbose logging
python main.py -v
```

### Console summary output (per city):
```
  City: London
  ────────────────────────────────────────────────────
  Market : What will be the highest temperature in London on 2026-03-20?
  Volume : $12,500 USDC
  Market implied temp : +21.3°C
  Forecast max temp   : +14.2°C  (2026-03-20)
  Difference          : +7.1°C  ▲ (market ABOVE forecast)
```

### Plots (saved to `plots/`):
- **Plot 1** (`{city}_plot1_forecast_vs_market.png`): Forecast line vs market-implied temp scatter
- **Plot 2** (`{city}_{cid}_plot2_dynamics.png`): Outcome probabilities + 24h volume over time
- **Plot 3** (`{city}_plot3_deltas.png`): Forecast change vs implied temp shift (delta view)
- **Efficiency signal** (`efficiency_signal_all_cities.png`): Cross-city bar chart of differences

---

## APIs Used

| API | Endpoint | Auth |
|-----|----------|------|
| Polymarket Gamma | `https://gamma-api.polymarket.com` | None (read-only) |
| Polymarket CLOB | `https://clob.polymarket.com` | None (read-only) |
| Open-Meteo Forecast | `https://api.open-meteo.com/v1/forecast` | None (free) |
| Open-Meteo Ensemble | `https://ensemble-api.open-meteo.com/v1/ensemble` | None (free) |

No API keys required for data collection. Execution (placing real bets) requires a `POLYMARKET_PRIVATE_KEY`.

---

## Inefficiency Analyzer (`polymarket_weather_analysis.py`)

This is the core betting engine. It compares Polymarket's implied probability distribution
across temperature bins against a meteorological forecast distribution, finds where they
diverge, and sizes bets using the Kelly criterion.

### The Core Idea

Each Polymarket temperature market is a question like *"Will the high in London on March 26 be exactly 14°C?"*
There are typically 10–20 such markets for the same city and date, each covering a different
temperature bin. Together they form an implied probability distribution.

The script builds its own probability distribution from the weather forecast and compares the two.
Where the market underprices a bin (forecast says 30% chance, market says 15%), it recommends
betting **Yes**. Where the market overprices a bin (forecast says 5%, market says 25%), it
recommends betting **No**.

### Why Markets Are Often Mispriced

Polymarket weather markets imply a spread of ~0.4–0.6°C around the most likely bin,
while real forecast uncertainty is 1.0–2.5°C depending on how far ahead the forecast is.
This means:

- Markets are **overconfident** — they put too much probability on the exact forecast bin
- Adjacent bins are **underpriced** — they're more likely than the market believes
- A forecast shift (e.g., a 24h update moving the predicted high from 15°C to 17°C) often
  takes hours to be reflected in market prices — creating a window to bet the updated view

### How Forecast Uncertainty Is Calculated

The script uses the **ICON-Seamless ensemble model** (~39 members) fetched from Open-Meteo.
Running `main.py --skip-polymarket` populates `data/weather/{city}_ensemble.csv`.

Each ensemble member is an independent model run with slightly different initial conditions.
The spread across members (`ens_std`) gives a data-driven estimate of forecast uncertainty.
When ensemble data is not available, the script falls back to a fixed lookup table:

| Days ahead | σ fallback (°C) |
|------------|----------------|
| 0–1 | 0.7–1.0 |
| 2–3 | 1.5–2.0 |
| 4–5 | 2.0–2.8 |
| >5 | 3.2 |

The ensemble estimate is significantly more accurate — at day 5, the ensemble σ is typically
~1.1°C vs. the 2.8°C fallback, which means the fallback produces too many false edges.
Always run `main.py --skip-polymarket` before the analyzer.

The uncertainty is modelled as a **Student-t distribution** (not a plain Gaussian) to give
heavier tails — the actual degrees-of-freedom parameter is fitted from the ensemble's p10/p90
spread, so extreme temperature outcomes get appropriately higher probability.

### Step-by-Step Pipeline

**Step 1 — Group snapshots**

Each row in the snapshot CSV is one Polymarket market. Rows are grouped by
`(city, target_date, fetch_time)` so that all temperature bins for the same city/date
at the same moment in time are analysed together.

**Step 2 — Parse questions**

`parse_question()` uses regex to extract the temperature and condition type:

| Pattern example | Condition | How it's evaluated |
|---|---|---|
| `be 15°C on` | `exact` | bin [14.5, 15.5]°C |
| `be 15°C or higher` | `gte` | P(temp ≥ 15°C) |
| `be 15°C or lower` | `lte` | P(temp ≤ 15°C) |
| `between 55–60°F` | `range` | converted to °C bin |
| Fahrenheit single values | `exact` | bin ±0.28°C (half a °F degree) |

**Step 3 — Build the market probability distribution**

The raw `yes_prob` values across all exact bins don't sum to 1 — they're independent
markets. The script normalises them into a proper probability distribution:

```
market_pmf[t] = yes_prob[t] / sum(yes_prob for all exact bins)
```

**Step 4 — Build the forecast probability distribution**

Using the ensemble mean as the centre (μ) and ensemble std as the spread (σ), the script
computes the probability that the actual temperature falls in each bin:

```
forecast_prob[t] = CDF(t + 0.5, μ, σ, ν) − CDF(t − 0.5, μ, σ, ν)
```

where CDF is the Student-t cumulative distribution. For `gte`/`lte` markets, it
computes the tail probability directly rather than a single-degree bin.

**Step 5 — Compute edge**

```
edge = forecast_prob[t] − market_prob[t]
```

- Positive edge → forecast thinks this bin is underpriced → bet **Yes**
- Negative edge → forecast thinks this bin is overpriced → bet **No**

The absolute value must exceed `min_edge` (default 7%) to be considered actionable.

**Step 6 — Score with 9 alpha signals**

Raw edge is adjusted by a composite score from nine signals:

| Signal | What it measures |
|--------|-----------------|
| α1 Momentum | Whether the forecast has been trending in the direction of the bet over recent days |
| α2 Diurnal spread | Extra uncertainty from day/night temperature range (wide range = more σ) |
| α3 Student-t tails | Whether the distribution needs heavy tails (extreme temps more likely than Gaussian says) |
| α4 Constrained PMF | Whether the market's own gte/lte markets are consistent with its exact-bin prices |
| α5 Internal consistency | Whether the market PMF sums to a sensible value (close to 1 = coherent market) |
| α6 Volume recency | Whether recent volume suggests informed traders are already closing the gap |
| α7 Forecast convergence | Whether the forecast has been stable recently (converging = higher confidence) |
| α8 Market staleness | Whether the market hasn't updated in a suspiciously long time given how far out it is |
| α9 Correlated bets | Group Kelly cap — prevents over-betting when multiple bins on the same city/date all look good |

**Step 7 — Kelly sizing**

```
b     = ((1 − market_price) / market_price) × (1 − 0.02 fee)
kelly = 0.25 × (b × forecast_prob − (1 − forecast_prob)) / b
kelly = clip(kelly, 0, 0.20)   # per-bet cap
```

The 0.25 factor is fractional Kelly — it bets a quarter of the theoretically optimal
amount to reduce variance. The 20% per-bet cap and 40% total portfolio cap prevent
catastrophic over-exposure.

---

## Running the Analyzer

### Standard dry run (see what it would bet, no real money):
```bash
python polymarket_weather_analysis.py --data_dir ./data --no_plots --live --bankroll 1000
```

### What `--live` does

Without `--live`, the bot uses prices from your stored snapshot CSVs, which may be
hours old. With `--live`, it fetches the current price for every candidate market
directly from the Gamma API and re-verifies that the edge still exists right now
before recommending a bet. Always use `--live` for real decisions.

### Custom thresholds:
```bash
python polymarket_weather_analysis.py \
    --data_dir data \
    --min_edge 0.08 \
    --min_liq  500 \
    --bankroll 2000 \
    --live
```

### Specific cities only:
```bash
python polymarket_weather_analysis.py --data_dir data --cities london chicago --live
```

### All CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Root data folder |
| `--cities` | all discovered | Space-separated city names to analyse |
| `--output_dir` | `./output` | Where to write CSVs and plots |
| `--min_edge` | `0.07` | Minimum edge to consider (0.07 = 7 percentage points) |
| `--min_liq` | `400` | Minimum market liquidity in USDC |
| `--bankroll` | `1000` | Your bankroll in USDC for Kelly sizing |
| `--live` | off | Re-fetch current prices from Gamma API before recommending bets |
| `--execute` | off | Place real orders (requires `POLYMARKET_PRIVATE_KEY` env var). Implies `--live` |
| `--no_plots` | off | Skip matplotlib output (faster) |

---

## Reading the Bot Output

```
Side    Size$    Edge      EV$      Liq$     Mom  Sig       Question
────  ───────  ──────  ───────  ────────  ──────  ────────  ────────────────
Yes   $ 38.00  64.0%  $ 73.74  $    757   -0.02  ensemble  Will the highest temperature in Seoul be 16°C…  (+0.0d)
No    $ 38.00  29.6%  $ 16.08  $  1,150   -0.10  ensemble  Will the highest temperature in Seoul be 13°C…  (+0.0d)
```

| Column | Meaning |
|--------|---------|
| **Side** | Bet direction — Yes (temp will hit that value) or No (it won't) |
| **Size$** | Dollar amount to stake from your bankroll |
| **Edge** | Gap between forecast probability and market price. 64% means forecast assigns 74% chance, market is priced at 10% |
| **EV$** | Expected profit on this bet: `edge × size` (assumes perfect model calibration) |
| **Liq$** | Market liquidity available — how much you can bet before moving the price |
| **Mom** | Momentum (α1): positive = forecast trending warmer, negative = trending cooler |
| **Sig** | Sigma source: `ensemble` = live model data (preferred), `nwp_tabl` = fallback table |
| **(+Nd)** | Days until market resolves |

The **same day/city can have both Yes and No bets on different questions** — e.g. betting Yes on
"Seoul 16°C today" and No on "Seoul 13°C today" are two independent markets that both reflect
the same forecast view (actual temp ~16°C).

The summary line shows total bets, total exposure (capped at 40% of bankroll), and
aggregate expected EV. Treat high EV numbers with scepticism — they assume the model
is perfectly calibrated, which it isn't.

---

## Output Files

| File | Description |
|------|-------------|
| `output/opportunities_v4.csv` | All filtered opportunities with edge, Kelly, alpha scores, and sigma source |
| `output/all_bins.csv` | Every (city, date, snapshot, bin) record with forecast and market probs |
| `output/distribution_comparison.png` | Forecast PMF vs market PMF for the most dislocated snapshots |
| `output/drift_{city}.png` | How the forecast temperature evolved over time for each target date |
| `output/edge_landscape.png` | Edge distribution, horizon scatter, EV and Kelly histograms |

---

## Architecture

### Data flow
```
Gamma API        → fetch_polymarket.py  → data/polymarket/{city}_snapshots.csv
Open-Meteo       → fetch_weather.py     → data/weather/{city}_daily.csv
                                                      {city}_hourly.csv
Open-Meteo ens.  → fetch_ensemble.py   → data/weather/{city}_ensemble.csv
                                                   ↓
                              processing.py  (dedup + implied temp)
                                                   ↓
                         visualization.py   (plots)
                  polymarket_weather_analysis.py  (alpha signals + Kelly sizing)
```

### Storage
All data is **append-only CSV** (never overwrites past records).
Deduplication keys:
- Market snapshots: `(condition_id, fetched_at_utc)`
- Weather daily: `(city, date_local, fetched_at_utc)`
- Ensemble: `(city, date_local, fetched_at_utc)`
- Price history: `(token_id, timestamp_utc)`

### Timezone handling
- Open-Meteo returns local timestamps (`timezone=auto`)
- All records store both `_local` and `_utc` timestamps
- Analysis and plots use UTC throughout
- City timezone definitions live in `config.CITIES`

---

## Adding More Cities

Edit `CITIES` in `config.py`:

```python
CITIES["Tokyo"] = {
    "timezone": ZoneInfo("Asia/Tokyo"),
    "lat": 35.6762,
    "lon": 139.6503,
    "search_terms": ["Tokyo", "tokyo"],
}
```

No other changes required — all pipeline steps iterate over `CITIES`.

---

## Going Live (Placing Real Bets)

**This places real money. Use with caution.**

1. Set your Polymarket private key:
   ```bash
   export POLYMARKET_PRIVATE_KEY=your_key_here
   ```

2. Run with `--execute`:
   ```bash
   python polymarket_weather_analysis.py --data_dir ./data --no_plots --live --execute --bankroll 500
   ```

The `--execute` flag implies `--live` (always re-verifies prices before placing any order).
Orders are placed via the Polymarket CLOB API using `py_clob_client`:

```python
from py_clob_client.client import ClobClient
client = ClobClient("https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
client.create_market_order(OrderArgs(
    token_id=YES_TOKEN_ID,
    price=market_price,
    size=size_usdc / market_price,
    side=BUY,
))
```

---

## Jupyter Notebook (`polymarket_weather_notebook.ipynb`)

Interactive companion to the analysis script with Plotly charts for exploration.

Key cells:
- **Forecast drift** — how `temp_max_c` evolves over time for each target date (spot late forecast revisions)
- **Market probability evolution** — P(Yes) over time per market (spot stale or suddenly-moving markets)
- **Opportunity detection** — run the full analyzer interactively with configurable thresholds
- **Forecast vs market scatter** — points above the 45° diagonal are underpriced (bet Yes), below are overpriced (bet No)
- **Historical calibration** — checks model accuracy on resolved markets
