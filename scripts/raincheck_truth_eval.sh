#!/bin/bash
# raincheck_truth_eval.sh — refresh station truth, then regenerate the eval tracker
# (Handoff Step 1, actions 2-3). Run daily, AFTER collection. Order matters:
# refresh truth first so newly-published station obs make resolved markets gradable,
# THEN regenerate output/opportunities_evaluation_calibrated.csv from the grown snapshot history.

set -u
PYTHON="/Users/ronanmulligan/.pyenv/versions/3.11.9/bin/python3"
REPO="/Users/ronanmulligan/Documents/GitHub/raincheck"
PKG="$REPO/src/polymarket_weather"
LOG="$REPO/logs/truth_eval.log"

mkdir -p "$REPO/logs"
cd "$PKG" || { echo "$(date '+%F %T') cd FAILED $PKG" >>"$LOG"; exit 1; }

echo "$(date '+%F %T') ── truth: fetch_historical_truth.py start" >>"$LOG"
"$PYTHON" fetch_historical_truth.py >>"$LOG" 2>&1
echo "$(date '+%F %T') ── truth: exit $?" >>"$LOG"

echo "$(date '+%F %T') ── eval: polymarket_weather_analysis.py start" >>"$LOG"
"$PYTHON" polymarket_weather_analysis.py --data_dir ./data >>"$LOG" 2>&1
echo "$(date '+%F %T') ── eval: exit $?" >>"$LOG"

# Append a one-line progress snapshot vs the pre-committed gate so the log doubles as a tracker.
"$PYTHON" data_status.py 2>&1 | grep -E "GRADABLE|gradable (markets|bets)|GATE" >>"$LOG"
echo "$(date '+%F %T') ── done" >>"$LOG"
