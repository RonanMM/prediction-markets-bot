# Where things actually stand (plain English)

This is the honest, no-jargon summary of what this project does and whether it works.
For the technical detail see `CLAUDE.md`. For live numbers run
`cd src/polymarket_weather && python data_status.py` and `python evaluate_oos.py`.

## What this project does
It bets on Polymarket "what will the temperature be?" markets. It uses weather forecasts to
estimate the true odds, and places a bet when the market price looks wrong. The only question
that matters: **does it predict the weather better than the betting market does?** If yes, it
makes money. If no, it doesn't.

## The big correction
An earlier version claimed **127.5% ROI**. That number was measured with a broken ruler.

To know if a bet won you need the *real* temperature that day. The old code took that "real"
temperature from **the same forecast it used to place the bet** — i.e. it graded its own
homework, comparing its prediction against its own prediction. Naturally it looked accurate.

The fix ("station truth" grading): grade every bet against the **actual weather-station reading**
— the official thermometer the market pays out on. Graded honestly, the ROI roughly **halved**,
and the old "we're winning" story no longer holds up.

## What the honest evaluation shows right now
`evaluate_oos.py` asks the real question: is the model's guess more accurate than just trusting
the market price? Accuracy is scored by **Brier score** — lower = better.

| Predictor | Brier (lower is better) |
|---|---|
| **Our model** | **0.166** |
| The market price | 0.157 |
| A simpler weather method (ensemble) | also beats our model |

**Our model is currently the least accurate of the three — the market is beating it.** So even
though past bets show ~+34.6% "profit", that profit is **luck on a small sample, not skill**:
you can't reliably beat a market you can't out-predict. The fancy machine-learning layer may even
be *hurting* versus the plain forecast.

## Why there's no final verdict yet
Only ~64 bets have been honestly graded so far — far too few to conclude anything (could be a
fluke either way). So a rule was set **in advance**: draw no conclusion until **≥150 graded
markets and ≥100 bets**. Until then every report prints a "⚠️ PREVIEW — don't act on this" banner.

The reason we're stuck at ~64: weather stations **publish their official data ~3 weeks late**, so
recent bets can't be graded yet. Automatic daily collection is now running to pile up data over
time (see `scripts/README.md`).

## Bottom line
1. The old "127% ROI / we're winning" claim was an illusion — now corrected.
2. There is **no proof of an edge**, and early signs suggest the model isn't beating the market.
3. **You can't decide yet** — more weeks of data are needed; collection is automated.
4. When the data is ready, run `python evaluate_oos.py`. If the model's Brier finally drops
   **below the market's (0.157)**, that's a real edge. If not, the fix is the *model*, not the
   bet sizing (the bet-sizing parameters are already validated as near-optimal).
