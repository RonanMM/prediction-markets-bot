# Raincheck scheduled collection (Handoff Step 1)

Forward-accumulates the out-of-sample track record by running the pipeline on a schedule.
All data is append-only and deduped on read, so every job is idempotent and safe to re-run.

## What runs when (local launchd agents)

| Agent (`~/Library/LaunchAgents/`)   | Script                          | Schedule            | Does |
|-------------------------------------|---------------------------------|---------------------|------|
| `com.raincheck.collect.plist`       | `scripts/raincheck_collect.sh`  | 06:00 & 18:00 daily | `main.py` — append one market+forecast snapshot |
| `com.raincheck.truth-eval.plist`    | `scripts/raincheck_truth_eval.sh` | 07:00 daily       | `fetch_historical_truth.py` → `polymarket_weather_analysis.py` → append a gate-progress line |

Logs: `logs/collect.log`, `logs/truth_eval.log` (and `*.launchd.log` for launchd's own stdout/stderr).
Both are gitignored.

## ⚠️ Required one-time setup: Full Disk Access (macOS TCC)

This repo lives under `~/Documents`, which macOS protects (TCC). A launchd-spawned process is
**not** granted access by default, so the jobs fail with `Operation not permitted` until you grant
Full Disk Access to the interpreter launchd uses:

1. Open **System Settings → Privacy & Security → Full Disk Access**.
2. Click **+**, press **⌘⇧G**, enter **`/bin/bash`**, add it, and toggle it **on**.
   (`/bin/bash` is what the launchd agents execute. Granting FDA here is broad — see the tighter
   alternative below if you prefer.)
3. Reload the agents (see Manage). Verify the next run writes to `logs/collect.log`.

**Tighter alternative (avoids granting FDA to all of bash):** move this repo out of `~/Documents`
(e.g. to `~/dev/raincheck` or `~/GitHub/raincheck`), update the absolute paths in the two scripts
and two plists, and reload. Non-protected locations need no FDA grant at all.

## Manage

```bash
UID_N=$(id -u)

# Load / reload (re-run after editing a plist or granting FDA)
launchctl bootout    gui/$UID_N ~/Library/LaunchAgents/com.raincheck.collect.plist 2>/dev/null
launchctl bootstrap  gui/$UID_N ~/Library/LaunchAgents/com.raincheck.collect.plist
launchctl bootout    gui/$UID_N ~/Library/LaunchAgents/com.raincheck.truth-eval.plist 2>/dev/null
launchctl bootstrap  gui/$UID_N ~/Library/LaunchAgents/com.raincheck.truth-eval.plist

# Run once now (test)
launchctl kickstart -k gui/$UID_N/com.raincheck.collect
launchctl kickstart -k gui/$UID_N/com.raincheck.truth-eval

# Status (column 2 = last exit code; 0 = OK)
launchctl list | grep raincheck

# Disable
launchctl bootout gui/$UID_N ~/Library/LaunchAgents/com.raincheck.collect.plist
launchctl bootout gui/$UID_N ~/Library/LaunchAgents/com.raincheck.truth-eval.plist
```

## Check progress toward the gate

```bash
cd src/polymarket_weather && python data_status.py
```

No ROI/win-rate conclusions until the pre-committed gate is met (≥150 gradable markets AND
≥100 gradable bets — see CLAUDE.md). The usual bottleneck is Meteostat's ~2–3 week publishing
lag, not market resolution.
