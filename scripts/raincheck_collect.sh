#!/bin/bash
# raincheck_collect.sh — append one live market+forecast snapshot (Handoff Step 1, action 1).
# Idempotent: data is append-only and deduped on read, so re-runs are safe.
# Designed for launchd/cron; uses an ABSOLUTE python path because those run with a minimal PATH.

set -u
PYTHON="/Users/ronanmulligan/.pyenv/versions/3.11.9/bin/python3"
REPO="/Users/ronanmulligan/Documents/GitHub/raincheck"
PKG="$REPO/src/polymarket_weather"          # in-package code must run from here (relative data paths)
LOG="$REPO/logs/collect.log"

mkdir -p "$REPO/logs"
cd "$PKG" || { echo "$(date '+%F %T') cd FAILED $PKG" >>"$LOG"; exit 1; }

echo "$(date '+%F %T') ── collect: main.py start" >>"$LOG"
"$PYTHON" main.py >>"$LOG" 2>&1
echo "$(date '+%F %T') ── collect: main.py exit $?" >>"$LOG"
