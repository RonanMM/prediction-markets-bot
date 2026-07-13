# Polymarket Weather Tracker

Tracks Polymarket temperature markets in five cities (NYC, Chicago, London, Seoul, Hong Kong),
compares them against calibrated weather forecasts, and — more importantly — measures honestly
whether any of it constitutes an edge. It currently does **not** for the forecasting strategy
(the market out-predicts the model; see `../../STATUS.md`); the live experiments are
market-structure paper books and per-bucket forward gates.

**Canonical docs** (this README is deliberately thin):
- `../../STATUS.md` — plain-English current state and verdicts
- `../../CLAUDE.md` — technical reference: architecture, module map, model stack, commands
- `../../docs/EDGE_MEGAPLAN.md` — the edge strategy, all tested hypotheses, pre-registered gates
- `../../scripts/README.md` — scheduled collection (launchd + GitHub Actions)

## Quick start

```bash
pip install -r requirements.txt

# collect a snapshot (markets + forecasts + obs; also feeds the paper books)
python main.py

# the honest-evaluation loop
python data_status.py            # sample-size gate progress
python evaluate_oos.py           # model vs market vs ensemble + per-bucket forward gates
python audit_settlements.py     # grading vs ACTUAL market settlements (keep ≥95%)
python shoulder_book.py --report # structure paper book vs its pre-registered gates
```

Unit tests run from the repo root: `pytest -o addopts="" tests/ -v`.

## Ground rules

1. **Settlement truth only.** Markets resolve on wunderground.com station pages; grading goes
   through `wu_truth.py` / `grading.py` accordingly. Never grade against a forecast grid.
2. **Gates before money.** Nothing trades real funds until its pre-registered forward gate
   passes (`config.LIVE_BUCKETS` for the model book, `shoulder_book.py` for structure legs).
3. **Append-only data.** Fetchers append, `processing.py` dedupes on read; a snapshot you
   didn't record is gone forever, so keep the 2-hourly collectors running.
