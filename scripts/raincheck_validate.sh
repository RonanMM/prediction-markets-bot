#!/bin/bash
# raincheck_validate.sh — one-command end-to-end validation of the model.
#
# Runs the full honest loop: fetch station truth (resolution-faithful sources: NWS CLI /
# IEM METAR / HKO) + per-lead archived forecasts (Open-Meteo Previous Runs), train the
# per-lead EMOS v2 calibrators, regenerate BOTH eval trackers (calibrated model + pure
# ensemble), then run the arbiter (evaluate_oos.py) and the sample-gate tracker
# (data_status.py).
#
# Requires network access to IEM/HKO + Open-Meteo. Portable: locates the repo from this
# script's own path and uses ${PYTHON:-python3}. Truth lag is now ~1 day for
# KLGA/KORD/EGLC/RKSI (HKO lags ~1 month; HK bets grade once HKO publishes).
#
#   ./scripts/raincheck_validate.sh            # full run
#   PYTHON=python ./scripts/raincheck_validate.sh

set -u
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG="$REPO/src/polymarket_weather"   # in-package code runs from here (relative data paths)

cd "$PKG" || { echo "cd FAILED: $PKG"; exit 1; }

echo "── 1/7 station truth (NWS CLI / IEM METAR / HKO) ─────────────────────────"
"$PYTHON" fetch_historical_truth.py    || { echo "truth fetch failed (network?)"; exit 1; }
echo "── 2/7 per-lead archived forecasts (Open-Meteo Previous Runs) ────────────"
"$PYTHON" fetch_historical_leads.py    || { echo "leads fetch failed (network?)"; exit 1; }
echo "── 3/7 multi-model per-lead archives ─────────────────────────────────────"
"$PYTHON" fetch_historical_leads_mm.py || { echo "mm leads fetch failed (network?)"; exit 1; }
echo "── 4/7 train per-lead EMOS v2 calibrators ────────────────────────────────"
"$PYTHON" train_calibrator.py          || { echo "training failed"; exit 1; }
echo "── 5/7 regenerate eval tracker (calibrated model) ────────────────────────"
"$PYTHON" polymarket_weather_analysis.py --data_dir ./data --no_plots
echo "── 6/7 regenerate eval tracker (pure ensemble baseline) ──────────────────"
"$PYTHON" polymarket_weather_analysis.py --data_dir ./data --no_plots --disable-calibrator
echo "── 7/7 arbiter + gate ────────────────────────────────────────────────────"
"$PYTHON" evaluate_oos.py
"$PYTHON" data_status.py
