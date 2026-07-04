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

## The second broken ruler (found & fixed 2026-07-03)
The "actual weather-station reading" itself turned out to be wrong. The old truth feed
(Meteostat) was publishing values up to **9 °C off** the official station reports for recent
weeks (e.g. LaGuardia hit 91 °F on 2026-06-05; Meteostat said ~76 °F — and still did a month
later), and its Hong Kong feed never matched the Observatory the market actually pays out on.

Truth now comes from the same sources the markets resolve against: official NWS climate reports
for New York/Chicago, METAR station summaries for London/Seoul, and the Hong Kong Observatory's
own data service. Two bonuses: labels are correct, and they publish within ~a day (was ~3 weeks),
so bets grade almost immediately.

## The model rebuild (2026-07-03)
The audit also found the forecast model was **2–3× overconfident**: it was trained on "what
happened" data (reanalysis) but used to predict 1–3 days ahead, where errors are much larger. It
was claiming near-certainty on outcomes that missed ~15% of the time. The calibrator was retrained
on **real archived forecasts at each lead time** (4.5 years of them) against the corrected truth,
per city and per days-ahead, using a multi-model forecast blend (ECMWF+GFS+ICON, +JMA for Seoul).

## What the honest evaluation shows right now
`evaluate_oos.py` asks the real question: is the model's guess more accurate than just trusting
the market price? Accuracy is scored by **Brier score** — lower = better.

| Predictor | Brier (lower is better) |
|---|---|
| The market price | **0.110** |
| **Our model (rebuilt)** | 0.143 |
| A simpler weather method (ensemble) | 0.156 |

The rebuilt model now clearly beats the simple ensemble (it used to lose to it) and is the best
temperature forecaster we have — but **the market still beats the model on the bets the model
flags**. When our model strongly disagrees with the market, the market is usually the one that's
right. So: still no proof of an edge; the gap is closing but not closed.

## Why there's no final verdict yet
The pre-committed rule stands: no conclusion until **>=150 graded markets and >=100 bets**. The
bets half of the gate is now met (~106 graded); markets are at ~78/150 and filling fast now that
truth publishes daily instead of every 3 weeks.

## Bottom line
1. Two broken rulers found and fixed: self-graded outcomes (2026-06) and a corrupted truth feed
   (2026-07). Every number before these fixes was unreliable.
2. The model itself is now honestly calibrated and beats its own baseline — but **not yet the
   market**. No edge is proven.
3. The most promising next steps are written up in CLAUDE.md/memory: use the NWS "National Blend"
   station forecasts for the US cities, and condition same-day bets on the temperature already
   observed that day (the market prices this in; our model currently doesn't see it).
4. When the gate is met, run `python evaluate_oos.py`. If the model's Brier drops below the
   market's, that's a real edge. If not, the fix is the *model*, not the bet sizing.
