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

## The third broken ruler: ~25 code bugs (found & fixed 2026-07)
A whole-repo code review found ~25 bugs, several of which were *hiding the real answer*. The
biggest: **"between X-Y°F" markets — about 83% of the US-city markets — were silently never
priced or graded** (a parsing/routing gap), so the graded set was small and skewed. Others that
distorted the numbers: a **look-ahead leak** in the backtest (it could read a forecast run from
*after* the bet was placed), predictive probabilities that were **over-dispersed** (a standard
deviation was used as a Student-t scale, ~40% too wide), a **grading↔pricing mismatch** on the
same markets, and **dishonest cost accounting** in the optimizer. These are now fixed and the
whole evaluation was regenerated from clean inputs. Full detail: `docs/BUGFIX_EXECUTION_REPORT.md`.

## What the honest evaluation shows now — the gate is MET
Fixing the voided range markets pulled ~140 previously-invisible markets into the graded set, so
the pre-committed sample gate is finally satisfied:

    gradable markets   211 / 150   [MET]
    gradable bets      302 / 100   [MET]

There is now a real verdict. `evaluate_oos.py` asks: is the model's guess more accurate than just
trusting the market price? Accuracy is Brier score — lower = better.

| Predictor | Brier (lower is better) |
|---|---|
| The market price | **0.128** |
| Our model (rebuilt + fixed) | 0.163 |
| A simpler weather method (ensemble) | 0.166 |

Two things are true at once:
- **The model beats its own baseline.** Against the raw ensemble it wins on Brier (0.163 vs 0.166)
  and on the paired temperature-calibration score (CRPS 1.28 vs 1.33). The multi-model blend, the
  per-lead calibration, and the dispersion fix make a genuinely better *forecaster* than the ensemble.
- **The model does NOT beat the market.** Market Brier 0.128 is clearly below model Brier 0.163.
  The "how much to trust the model" sweep lands on **zero** (pure market beats pure model), and
  betting the model at production sizing returns **−20% ROI over 199 graded bets**. When the model
  strongly disagrees with the price, the price is usually right.

## Bottom line (2026-07 — gate met)
1. Earlier numbers were unreliable for three separate reasons now fixed: self-graded outcomes,
   a corrupted truth feed, and ~25 code bugs. The evaluation is finally trustworthy.
2. **Verdict: no edge over the market.** The gate is met, the model is honestly calibrated and
   beats its ensemble baseline — but it is not a better predictor than Polymarket, so it loses
   money. This is a real go/no-go answer, not "too little data."
3. The fixes did not create edge; they removed the bugs that were hiding the truth. As a *bettor*
   against this market, the current model has no advantage. The place to improve is the *model's
   accuracy* (same-day intraday conditioning, NWS National Blend for the US cities — the deferred
   C4/C5 work), not the bet sizing.
4. Re-check anytime with `python evaluate_oos.py` and `python data_status.py`. If model Brier ever
   drops below market Brier on the graded set, that's a real edge; until then, don't bet it live.
