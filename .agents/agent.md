# Raincheck Workspace Rules & Customizations

This file outlines the workspace-specific rules, parameters, and architecture layout for AI agents working on the Raincheck weather betting bot.

## Project Structure & Architecture
- **Active Code Base**: All active weather bot code resides in `src/polymarket_weather/` (NOT in `src/raincheck/` which is a PyScaffold skeleton).
- **Refactored Modular Engine**: Do not write monolithic analysis code in `polymarket_weather_analysis.py`. Use the following modular files:
  * [engine.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/engine.py): Main simulation loop, `WeatherBettingBot`, and Kelly sizing logic.
  * [signals.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/signals.py): Alpha signal metrics (momentum, convergence, staleness, volume).
  * [pmf.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/pmf.py): PMF reconstruction and probability parsing rules.
  * [models.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/models.py): Opportunity and MarketBin dataclasses.
  * [reports.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/reports.py): Formatting print reports and plotting wrappers.
  * [data_loader.py](file:///Users/ronanmulligan/Documents/GitHub/raincheck/src/polymarket_weather/data_loader.py): Loaders for snapshots and weather forecasts.

## Calibrated Execution Rules
Always respect the optimized parameters found during grid search:
1. **`MIN_LIQUIDITY = 1000`**: Only bet in markets with at least $1000 USDC active volume to avoid slippage.
2. **`KELLY_FRACTION = 0.50`**: Scale normal bets using a 0.50 multiplier.
3. **`MAX_KELLY_PER_BET = 0.08`**: Never bet more than 8% of the bankroll on a single market, regardless of confidence.
4. **`MAX_KELLY_PER_GROUP = 0.20`**: Max 20% exposure per city/date group.
5. **`STALE_HOURS = 4`**: Penalize pricing data that hasn't moved in 4 hours.

## Unit Testing
- All tests must go under `tests/test_polymarket_weather.py`.
- Run tests via pytest from the workspace root:
  ```bash
  pytest -o addopts="" tests/ -v
  ```
