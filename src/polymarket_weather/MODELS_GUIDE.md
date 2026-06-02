# Polymarket Weather Analyzer: The Two Engines Explained

When running the inefficiency analysis for weather markets, this bot utilizes the **Student-t mathematical formula** to calculate probabilities. However, it can feed two completely different sets of data into that formula. 

These represent our two "engines" or "modes".

---

## 1. The ML Engine (Ensemble Data)
**What it is:** This is the default, primary engine. It uses Open-Meteo's machine learning-enhanced **Ensemble Forecast** (which runs roughly 40 to 120 live simulations simultaneously).
**Why use it:** It detects **real-time uncertainty**. If a storm is highly unpredictable today, the ML data will widen the spread/variance. It makes the bot much smarter and prevents overconfident bets.

### How to run the ML Engine
By default, the analysis script will look for the ML ensemble data. As long as you have fetched the data, it will use the ML engine automatically.

**Step 1: Fetch the ML ensemble data**
You can do this by running the full pipeline:
```bash
python main.py
```
*(Alternatively, you can just run `python fetch_ensemble.py` if you only want to update the ensemble data).*

**Step 2: Run the analyzer**
```bash
python polymarket_weather_analysis.py --data_dir ./data
```
*Note: In the output, you will see `Ensemble loaded` and the `Sig` column will say `ensemble`.*

---

## 2. The Non-ML Baseline Engine (NWP Table)
**What it is:** This is the fallback engine. Instead of live machine-learning simulations, it relies on a **hardcoded lookup table** (`NWP_PARAMS`) to determine the variance.
**Why use it:** You generally shouldn't use this for live betting, as it assumes that every "5-day forecast" has the exact same uncertainty, ignoring real-world weather patterns. It is mostly used as a baseline to see how much the ML engine is saving you from bad bets.

### How to run the Non-ML Baseline Engine
The bot is programmed to automatically fall back to the Non-ML Baseline if it *cannot find* the ML ensemble data files. To force the bot to run in this mode, you just have to temporarily hide or delete the ensemble data.

**Step 1: Hide the ML ensemble data**
Rename the ensemble CSV files so the script can't find them:
```bash
for file in data/weather/*_ensemble.csv; do mv "$file" "${file}.bak"; done
```

**Step 2: Run the analyzer**
```bash
python polymarket_weather_analysis.py --data_dir ./data
```
*Note: In the output, you will see `Ensemble not available — falling back to NWP_PARAMS table` and the `Sig` column will say `nwp_tabl`.*

**Step 3: Restore the ML ensemble data**
When you are done testing, don't forget to restore the ensemble files so the bot uses the ML engine again!
```bash
for file in data/weather/*_ensemble.csv.bak; do mv "$file" "${file%.bak}"; done
```
