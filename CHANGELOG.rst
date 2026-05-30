=========
Changelog
=========

Version 0.1
===========

- Feature A added
- FIX: nasty bug #1729 fixed
- Added `--disable_ml` flag to `polymarket_weather_analysis.py` to toggle between ML Calibrator and legacy Student-t models.
- Fixed a pathing bug causing `joblib.load()` to fail to find the `models/` directory, which previously caused the ML model to silently fall back to the ensemble mode.
- Updated `polymarket_weather_analysis.py` to save separate output CSVs for ML vs Ensemble for head-to-head live evaluation.
- Updated `historical_backtester.py` to accept custom start dates and target CSV paths via command line arguments.
