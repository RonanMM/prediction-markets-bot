# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raincheck is a Polymarket weather prediction market tracker that compares market-implied temperatures against meteorological forecasts to identify pricing inefficiencies. The active project code lives in `src/polymarket_weather/`, not in `src/raincheck/` (which is a near-empty PyScaffold skeleton).

## Commands

All commands run from `src/polymarket_weather/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline: fetch Polymarket + weather, print summary, generate plots
python main.py

# Skip fetches, just re-generate plots from stored CSVs
python main.py --plots-only

# Print market vs forecast summary without fetching or plotting
python main.py --summary-only

# Target specific cities
python main.py --cities London Seoul "New York City"

# Run the parameter sweep optimizer (reads pre-made CSV, very fast)
python optimizer.py

# Run the full-pipeline grid search optimizer (runs engine back-to-front, slow)
python optimizer_full.py

# Run the inefficiency analyzer
python polymarket_weather_analysis.py --data_dir ./data [--min_edge 0.06] [--min_liq 1000] [--bankroll 1000]

# Interactive notebook
jupyter notebook polymarket_weather_notebook.ipynb
```

```bash
# Run tests (from repo root)
pytest tests/ -v

# Run with tox
tox
```

## Architecture

### Data Flow

```
Gamma API → fetch_polymarket.py → data/polymarket/*.csv
Open-Meteo API → fetch_weather.py → data/weather/*.csv
                                         ↓
                    processing.py (dedup + implied temp computation)
                                         ↓
              visualization.py (PNG plots) + polymarket_weather_analysis.py (alpha signals)
```

### Key Design Decisions

- **Append-only CSVs**: Data is never overwritten. `processing.py` deduplicates on read using composite keys (e.g., `condition_id + fetched_at_utc` for market snapshots). This makes all fetches idempotent.
- **No database**: Pure CSV storage, loaded into pandas for analysis.
- **Dual timezone storage**: All records store both `_local` and `_utc` timestamps; plots use UTC.
- **Stateless fetchers**: `fetch_polymarket.py` and `fetch_weather.py` have no side effects beyond returning dicts; `processing.py` handles persistence.

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `config.py` | Single source of truth: city definitions (lat/lon, timezone, search terms), API endpoints, storage paths, retry/timeout settings |
| `fetch_polymarket.py` | Gamma API (market search/details) + CLOB API (price history). Extracts outcome probabilities and temperature bins from market questions. |
| `fetch_weather.py` | Open-Meteo 16-day forecast. Returns daily (temp_max, temp_min, precip) and hourly (temp, precip_prob) data. |
| `processing.py` | CSV append + dedup logic; computes probability-weighted implied temperature from outcome labels using regex for ranges like "20-22°C", ">86°F". |
| `visualization.py` | Three plot types per city: (1) forecast vs market-implied temp over time, (2) per-market outcome probabilities + volume, (3) delta view. Plus cross-city efficiency bar chart. |
| `polymarket_weather_analysis.py` | Advanced analyzer: reconstructs market PMF from bins, builds Gaussian forecast PMF with horizon-calibrated σ (0.7–3.2°C), computes KL divergence, per-bin edge, Kelly sizing. Outputs `output/opportunities_v2.csv`. |

### Inefficiency Analysis (polymarket_weather_analysis.py)

The analyzer implements 9 alpha signals (α1–α9) covering forecast momentum, min/max spread, Student-t tail modeling, constrained PMF fitting, internal consistency, volume recency, forecast convergence, market update lag, and correlated bet grouping. These have been modularized and split out into `engine.py`, `signals.py`, `models.py`, `pmf.py`, `reports.py`, and `data_loader.py`.

Filtering defaults: `min_edge=6%`, `min_liquidity=$1000` (optimized from $400 to avoid noisy, illiquid slippage). Kelly sizing uses a `0.50` fraction with a strict `8%` cap per market (`MAX_KELLY_PER_BET`) and a `20%` cap per city/date group to protect the bankroll.

### APIs Used

- **Gamma API** (`gamma-api.polymarket.com`): Market search and details — no auth required.
- **CLOB API** (`clob.polymarket.com`): Token price history — no auth required. SSL warnings are suppressed in `fetch_polymarket.py`.
- **Open-Meteo** (`api.open-meteo.com`): Free weather forecasts — no API key needed.

All fetches use retry logic with exponential backoff (`RETRY_ATTEMPTS=3`, `REQUEST_TIMEOUT=20s` from `config.py`).

### Ensemble Forecast

`fetch_ensemble.py` calls `ensemble-api.open-meteo.com` for the ICON-Seamless model (~39 members). Each day's `ens_std` replaces the hardcoded NWP sigma lookup in the analysis. Run `main.py` (or manually call `fetch_ensemble.fetch_all_cities()`) to populate `data/weather/{city}_ensemble.csv`. Without ensemble data the analysis falls back to the `NWP_PARAMS` lookup table.

The impact: day-5 ICON-Seamless spread (σ≈1.1°C) vs the old hardcoded 2.8°C — the lookup was nearly 3× too wide at medium range, creating many false edges.

### Adding a New City

Edit `config.py` — add an entry to `CITIES` with `timezone`, `lat`, `lon`, and `search_terms`. No other changes required; all pipeline steps iterate over `CITIES`.
