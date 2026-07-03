#!/bin/bash
# raincheck_validate.sh — one-command end-to-end validation of the model.
#
# Runs the full honest loop: fetch station truth + archive features, train the EMOS calibrators,
# regenerate BOTH eval trackers (EMOS model + pure ensemble), then run the arbiter (evaluate_oos.py)
# and the sample-gate tracker (data_status.py).
#
# Requires network access to Meteostat + Open-Meteo (the fetch steps). Portable: it locates the repo
# from this script's own path and uses ${PYTHON:-python3}, so it is not tied to any one machine.
# Note: gradability is bounded by Meteostat's ~3-week publishing lag — recent bets won't grade yet.
#
#   ./scripts/raincheck_validate.sh            # full run
#   PYTHON=python ./scripts/raincheck_validate.sh

set -u
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG="$REPO/src/polymarket_weather"   # in-package code runs from here (relative data paths)

cd "$PKG" || { echo "cd FAILED: $PKG"; exit 1; }

echo "── 1/6 station truth (Meteostat) ─────────────────────────────────────────"
"$PYTHON" fetch_historical_truth.py    || { echo "truth fetch failed (network?)"; exit 1; }
echo "── 2/6 archive features (Open-Meteo) ─────────────────────────────────────"
"$PYTHON" fetch_historical_features.py || { echo "feature fetch failed (network?)"; exit 1; }
echo "── 3/6 train EMOS calibrators ────────────────────────────────────────────"
"$PYTHON" train_calibrator.py          || { echo "training failed"; exit 1; }
echo "── 4/6 regenerate eval tracker (EMOS model) ──────────────────────────────"
"$PYTHON" polymarket_weather_analysis.py --data_dir ./data --no_plots
echo "── 5/6 regenerate eval tracker (pure ensemble baseline) ──────────────────"
"$PYTHON" polymarket_weather_analysis.py --data_dir ./data --no_plots --disable-calibrator
echo "── 6/6 arbiter + gate ────────────────────────────────────────────────────"
"$PYTHON" evaluate_oos.py
"$PYTHON" data_status.py
