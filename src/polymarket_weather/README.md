# Polymarket Weather Tracker

Tracks prediction market probabilities on Polymarket for temperature markets and
compares them against real-world weather forecasts over time.

---

## Project Structure

```
polymarket_weather/
├── config.py              # Cities, API endpoints, paths, color scheme
├── fetch_polymarket.py    # Gamma API + CLOB price history
├── fetch_weather.py       # Open-Meteo forecast (daily + hourly)
├── processing.py          # Storage, dedup, implied temp calculation
├── visualization.py       # All 3 plot types
├── main.py                # CLI entry point
├── requirements.txt
└── data/
    ├── polymarket/        # {city}_snapshots.csv, {city}_price_history.csv
    └── weather/           # {city}_daily.csv, {city}_hourly.csv
└── plots/                 # PNG outputs
└── logs/                  # Per-run log files
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Full daily run (fetch + store + plot + summary):
```bash
python main.py
```

### Specific cities only:
```bash
python main.py --cities Seoul London "New York City"
```

### Weather only (skip Polymarket fetch):
```bash
python main.py --skip-polymarket
```

### Re-generate plots from existing stored data:
```bash
python main.py --plots-only
```

### Print summary only:
```bash
python main.py --summary-only
```

### Verbose logging:
```bash
python main.py -v
```

---

## Output

### Console summary (per city):
```
  City: London
  ────────────────────────────────────────────────────
  Market : What will be the highest temperature in London on 2026-03-20?
  Volume : $12,500 USDC
  Market implied temp : +21.3°C
  Forecast max temp   : +14.2°C  (2026-03-17)
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
| Open-Meteo | `https://api.open-meteo.com/v1/forecast` | None (free) |

No API keys required.

---

## Inefficiency Analyzer (`polymarket_weather_analysis.py`)

This script searches for pricing inefficiencies between Polymarket's temperature
markets and NWP-calibrated weather forecasts. The core idea is that Polymarket
markets form a **categorical distribution** across discrete temperature bins for
each (city, date), and this distribution is often overconfident or stale relative
to the best available forecast.

### Key Insight

Polymarket weather markets imply a market σ of ~0.4–0.6 °C around the modal bin,
while NWP verification statistics show true forecast uncertainty is σ ≈ 1.5–2.5 °C
at 2–4 days ahead. This systematic mismatch means:

- The market **overprices** the exact modal temperature bin.
- The market **underprices** adjacent bins.
- A recent forecast shift (e.g., a 24 h update) may not yet be reflected in prices.

### Pipeline

```
Polymarket snapshots  ──┐
                         ├─► build_day_snapshot()
Open-Meteo daily     ──┘        │
                                 ▼
                        Reconstruct market PMF  (normalized over exact bins)
                        Build forecast PMF      (Gaussian, NWP-calibrated σ)
                                 │
                                 ├─► KL divergence  (overall dislocation score)
                                 └─► Per-bin edge   (forecast_prob − market_prob)
                                              │
                                              ▼
                                   filter_opportunities()
                                   Kelly sizing → ranked bets
```

#### Step 1 — Group snapshots

Each snapshot CSV row is one Polymarket market for one temperature bin (e.g.
"Will the high in London on 2026-03-22 be exactly 15 °C?"). Rows are grouped by
`(target_date, fetch_time_bucket)` with a 10-minute floor, so every group
represents all available bins for the same city and target date at roughly the
same moment in time.

#### Step 2 — Parse questions

`parse_question()` uses regex to extract the temperature condition and value from
the market question string. It handles:

| Pattern | Condition |
|---------|-----------|
| `be 15 °C on` | `exact` |
| `be 15 °C or higher` | `gte` |
| `be 15 °C or lower/below` | `lte` |
| `between 55-60 °F` | `range` |
| Fahrenheit variants of the above | converted to °C |

#### Step 3 — Reconstruct market PMF

The individual Polymarket `yes_prob` values (0–1) for each bin are **not** a
proper probability distribution — they do not sum to 1. For the `exact` bins,
the script normalises them so they form a valid PMF:

```
market_pmf[t] = yes_prob[t] / sum(yes_prob for all exact bins)
```

#### Step 4 — Build forecast PMF

For the same set of temperature bins, the script computes a Gaussian-based
probability using Open-Meteo's forecast `temp_max_c` as the mean (μ) and an
NWP-calibrated sigma (σ) that grows with the forecast horizon:

| Days ahead | σ (°C) |
|-----------|--------|
| 0 | 0.8 |
| 1 | 1.2 |
| 2 | 1.6 |
| 3 | 2.0 |
| 4 | 2.4 |
| 5 | 2.8 |
| >5 | 3.2 |

Each `exact` bin at temperature `t` gets probability:

```
forecast_raw[t] = Φ(t + 0.5, μ, σ) − Φ(t − 0.5, μ, σ)
```

These are then normalised to form `forecast_pmf`.

#### Step 5 — Compute edge and KL divergence

For every exact bin:

```
edge = forecast_pmf[t] − market_pmf[t]
```

Positive edge → our model thinks this bin is underpriced → bet **Yes**.  
Negative edge → overpriced → bet **No**.

The **KL divergence** KL(forecast ‖ market) measures the overall dislocation of
the entire distribution for the snapshot. It acts as a summary signal for how
stale or miscalibrated the market is on that day.

#### Step 6 — Filter and size bets

`filter_opportunities()` keeps records where:
- `|edge| ≥ min_edge` (default 7 pp, overridden by `--min_edge`)
- `liquidity ≥ min_liq` (default $400 USDC, overridden by `--min_liq`)

Kelly fraction sizing:

```
b = (1 / their_prob) − 1      # decimal odds
kelly = 0.25 × (b × our_prob − (1 − our_prob)) / b
kelly = clip(kelly, 0, 0.15)  # cap at 15% of bankroll
```

The 0.25 fractional Kelly factor and 15% hard cap limit variance.

### Running the Analyzer

```bash
# Basic run (all cities, 7% edge threshold, $1000 bankroll):
python3 polymarket_weather_analysis.py --data_dir data

# Custom thresholds:
python3 polymarket_weather_analysis.py \
    --data_dir data \
    --min_edge 0.08 \
    --min_liq  500 \
    --bankroll 2000

# Specific cities only:
python3 polymarket_weather_analysis.py \
    --data_dir data \
    --cities london chicago

# Skip plot generation:
python3 polymarket_weather_analysis.py --data_dir data --no_plots
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Root data folder (must contain `polymarket/` and `weather/` sub-dirs) |
| `--cities` | all discovered | Whitespace-separated list of cities to analyze |
| `--output_dir` | `./output` | Where to write CSVs and PNG plots |
| `--min_edge` | `0.07` | Minimum edge fraction (e.g. `0.08` = 8 pp) |
| `--min_liq` | `400` | Minimum liquidity in USDC |
| `--bankroll` | `1000` | Bankroll in USDC used for Kelly sizing |
| `--no_plots` | off | Skip all matplotlib output |

### Outputs

| File | Description |
|------|-------------|
| `output/all_bins.csv` | Every (city, date, snapshot, bin) record with forecast and market probs |
| `output/opportunities_v2.csv` | Filtered subset with Kelly fractions and EV estimates |
| `output/distribution_comparison.png` | Bar chart comparing forecast PMF vs market PMF for the 6 most dislocated snapshots |
| `output/drift_{city}.png` | Forecast temp-max evolution over time for a city |
| `output/edge_landscape.png` | 6-panel overview: edge distribution, horizon scatter, mode-shift, EV and Kelly histograms, liquidity scatter |

### `WeatherBettingBot` scaffold

The `WeatherBettingBot` class simulates trade execution in dry-run mode and prints
a formatted order list. To go live, replace `_execute()` with calls to the
Polymarket CLOB API:

```python
from py_clob_client.client import ClobClient
client = ClobClient("https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
client.create_market_order(OrderArgs(
    token_id=YES_TOKEN_ID,
    price=market_prob,
    size=size_usdc / market_prob,
    side=BUY,
))
```

---

## Jupyter Notebook (`polymarket_weather_notebook.ipynb`)

The notebook is a fully interactive companion to the analysis script. It shares
the same core functions via import and adds interactive Plotly charts for
exploration. Below is a cell-by-cell guide.

### Cell 1 — Install dependencies

Runs `pip install` for pandas, numpy, matplotlib, scipy, seaborn, plotly, and
ipywidgets. Safe to re-run; exits silently if packages are already present.

### Cell 2 — Imports and config

Imports all core functions from `polymarket_weather_analysis.py`. Key settings:

```python
DATA_DIR   = Path('./data')   # ← change if your data lives elsewhere
OUTPUT_DIR = Path('./output')
```

Auto-discovers available cities by globbing `*_snapshots.csv` under `DATA_DIR`.

### Cell 3 — Load and inspect raw data (Section 1)

Calls `load_snapshots()` and `load_daily()` for every discovered city. Displays
the first 10 rows of the snapshot table for the first city, showing the question
text, yes_prob, liquidity, end date, and fetch time.

### Cell 4 — View daily forecast sample (Section 1 continued)

Shows the first 10 rows of the daily weather forecast table for the same city.

### Cell 5 — Forecast drift (Section 2)

`plot_drift_interactive()` calls `forecast_drift_analysis()` and renders an
interactive Plotly line chart showing how `temp_max_c` evolves for the last 8
target dates as new forecasts are fetched over time. Each target date is a
separate coloured trace. Use this chart to spot:

- Large late-breaking forecast revisions (potential market lag edge)
- Dates where the forecast converged early vs. stayed volatile

### Cell 6 — Market probability evolution (Section 3)

`plot_market_evo_interactive()` tracks P(Yes) over time for the 10 most-updated
condition IDs per city. Hover to see the full question text. Use this to identify:

- Markets that moved sharply (possible informed trading or forecast reaction)
- Markets that barely moved (potentially stale liquidity)

### Cell 7 — Opportunity detection (Section 4)

Calls `analyze_city()` for every city with configurable thresholds:

```python
MIN_EDGE      = 0.08   # 8 pp minimum edge
MIN_LIQUIDITY = 500    # USDC
```

Concatenates results into `combined`. Prints a count per city and total.

### Cell 8 — Opportunity table (Section 4 continued)

Displays the top 30 opportunities sorted by descending edge, showing forecast and
market probabilities, condition type, Kelly fraction, and liquidity.

### Cell 9 — Forecast vs market scatter (Section 5)

Interactive Plotly scatter of `forecast_prob` vs `market_prob`. Points above the
45° diagonal line are underpriced (bet Yes); points below are overpriced (bet No).
Dot size encodes edge magnitude. Hover shows city, date, days ahead, and question.

### Cell 10 — Edge vs horizon scatter (Section 6)

Shows whether edge magnitude correlates with forecast horizon. Expectation: older
snapshots at longer horizons should show more dislocation because the market is
slower to update. A dashed yellow line marks the minimum edge threshold. Also
prints the Pearson correlation between `days_ahead` and `edge`.

### Cell 11 — Anomaly deep-dive (Section 7)

Filters for `edge ≥ 20%` (configurable via `threshold_anomaly`). These are the
highest-confidence opportunities where the forecast and market are most divergent.
Displays question text, Kelly fraction, and liquidity for manual review.

### Cell 12 — Bot dry-run simulation (Section 8)

Instantiates `WeatherBettingBot` and calls `bot.run(combined)`, which:

1. Deduplicates to the most recent snapshot per `condition_id`.
2. Skips markets already settled (`days_ahead ≤ 0`).
3. Computes Kelly size for each opportunity.
4. Prints a formatted order table (dry-run, no real trades).
5. Reports total exposure and expected value.

After the run, the bet log is displayed as a DataFrame with size, probabilities,
and edge per order.

### Cell 13 — Historical calibration (Section 9)

Attempts to estimate model accuracy on resolved markets. It identifies markets
where the final `yes_prob` is 0 or 1 (settled), reconstructs the forecast
probability at snapshot time, and checks whether the model's directional call
matched the resolution. Reports overall accuracy and a calibration table.

### Cell 14 — Export (Section 10)

Saves the full opportunity DataFrame to `output/opportunities.csv`, sorted by
descending edge.

---

## Architecture

### Market discovery
`fetch_polymarket.py` searches the Gamma API `/markets` endpoint using combinations
of keywords (`"highest temperature"`, `"temperature in"`, etc.) and per-city
search terms. Each match is de-duplicated by `conditionId`.

### Implied temperature
Outcome labels (e.g. `"20–22°C"`, `">86°F"`, `"above 30°C"`) are parsed with regex
into `(low, high)` ranges, converted to °C if needed, then combined as a
probability-weighted average:

```
E[T] = Σ( midpoint_i × P_i ) / Σ( P_i )
```

### Timezone handling
- Open-Meteo returns local timestamps (with `timezone=auto`)
- All dates are converted to UTC before storage using `zoneinfo`
- Plots use UTC on the x-axis
- Per-city timezone information is stored in `config.CITIES`

### Storage
All data is **append-only CSV** (never overwrites past history).
De-duplication is enforced on:
- Market snapshots: `(condition_id, fetched_at_utc)`
- Weather daily: `(city, date_local, fetched_at_utc)`
- Price history: `(token_id, timestamp_utc)`

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

---

## Extending the Pipeline

- **New data source**: add a `fetch_*.py` module + corresponding `save_*` / `load_*`
  helpers in `processing.py`
- **New plot type**: add a function in `visualization.py` + call from `generate_all_plots()`
- **Automation**: schedule `python main.py` once daily via cron or systemd timer
