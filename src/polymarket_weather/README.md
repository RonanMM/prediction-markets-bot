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
