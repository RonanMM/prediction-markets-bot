import pytest
import pandas as pd
from pathlib import Path

# Import path is set centrally in conftest.py (source root = src/polymarket_weather).
from config import CITY_NAMES, KELLY_FRACTION, FEE_RATE
from pmf import parse_question
from models import Opportunity
from engine import _kelly_size
import grading
from grading import native_round, resolves_yes

def test_city_names_mapping():
    """Verify that city name mappings are correct and resolve key bugs."""
    assert CITY_NAMES["new_york_city"] == "NYC"
    assert CITY_NAMES["new_york"] == "NYC"
    assert CITY_NAMES["london"] == "London"
    assert CITY_NAMES["seoul"] == "Seoul"
    assert CITY_NAMES["hong_kong"] == "HongKong"
    assert CITY_NAMES["chicago"] == "Chicago"

def test_parse_question_celsius():
    """Verify that Celsius questions are parsed correctly."""
    # Test GTE
    parsed = parse_question("Will the temperature be 35°C or higher on June 23?")
    assert parsed is not None
    assert parsed["condition"] == "gte"
    assert parsed["temp_c"] == 35.0
    assert parsed["half_width"] == 0.5

    # Test Exact
    parsed = parse_question("Will the temperature be 30°C on June 23?")
    assert parsed is not None
    assert parsed["condition"] == "exact"
    assert parsed["temp_c"] == 30.0
    assert parsed["half_width"] == 0.5

    # Test LTE
    parsed = parse_question("Will the temperature be 25°C or lower on June 23?")
    assert parsed is not None
    assert parsed["condition"] == "lte"
    assert parsed["temp_c"] == 25.0
    assert parsed["half_width"] == 0.5

def test_parse_question_fahrenheit():
    """Verify that Fahrenheit questions are parsed and converted to Celsius correctly."""
    # 95°F = 35°C
    parsed = parse_question("Will the temperature be 95°F or higher?")
    assert parsed is not None
    assert parsed["condition"] == "gte"
    assert pytest.approx(parsed["temp_c"]) == 35.0
    assert pytest.approx(parsed["half_width"]) == 0.2777777

    # 50°F = 10°C
    parsed = parse_question("Will the temperature be 50°F on June 23?")
    assert parsed is not None
    assert parsed["condition"] == "exact"
    assert pytest.approx(parsed["temp_c"]) == 10.0

def test_kelly_sizing():
    """Verify Kelly bet size calculations and caps."""
    from types import SimpleNamespace
    
    # Create a mock opportunity using SimpleNamespace
    opp = SimpleNamespace(
        their_prob=0.50,       # Polymarket price = 50c
        our_prob=0.80,         # Our true probability = 80%
    )

    # With their_prob = 0.50, our_prob = 0.80, fee = 0.02, kelly_fraction = 0.50
    # effective payout odds b = ((1 - 0.5) / 0.5) * (1 - 0.02) = 1.0 * 0.98 = 0.98
    # raw kelly = kelly_fraction * (b * p - q) / b = 0.50 * (0.98 * 0.80 - 0.20) / 0.98
    # raw kelly = 0.50 * (0.784 - 0.20) / 0.98 = 0.50 * 0.584 / 0.98 = 0.298
    # Since raw_kelly > MAX_KELLY_PER_BET (0.08), it should be capped at 0.08.
    
    size = _kelly_size(opp, kelly_fraction=0.50, fee=0.02)
    assert size == 0.08  # Capped at MAX_KELLY_PER_BET (0.08)

    # Let's test a smaller edge that won't hit the cap
    opp.our_prob = 0.55  # 55% vs 50% price
    # raw kelly = 0.50 * (0.98 * 0.55 - 0.45) / 0.98 = 0.50 * (0.539 - 0.45) / 0.98 = 0.50 * 0.089 / 0.98 = 0.0454
    size_small = _kelly_size(opp, kelly_fraction=0.50, fee=0.02)
    assert pytest.approx(size_small, abs=0.001) == 0.0454


def test_native_round_units():
    """Temperatures must round onto the market's NATIVE resolution grid, not always °C."""
    # whole °F: the °C value of a whole-°F threshold round-trips back to that °F.
    assert native_round((35 - 32) * 5 / 9, "whole °F") == 35
    # 2.2 °C is 35.96 °F -> 36 °F (NOT 2). Proves we round in °F, not °C.
    assert native_round(2.2, "whole °F") == 36
    # whole °C (default) and tenths-of-°C grids.
    assert native_round(2.2, "whole °C") == 2
    assert native_round(2.6, "whole °C") == 3
    assert native_round(25.64, "0.1 °C") == 25.6


def test_resolves_yes_fahrenheit_boundary(monkeypatch):
    """A °F market where °C-rounding would have flipped the outcome (the bug this fixes)."""
    bin_c = (35 - 32) * 5 / 9  # 35 °F expressed in °C (~1.667)
    q = "Will the highest temperature in Chicago be 35°F or below on March 23?"
    # Station reads 2.2 °C = 35.96 °F -> 36 °F. 36 <= 35 is False -> resolves NO.
    monkeypatch.setattr(grading, "fetch_actual_weather", lambda *a, **k: 2.2)
    assert resolves_yes("Chicago", "2026-03-23", q, bin_c) is False
    # (Old °C grading: round(2.2)=2 <= round(1.667)=2 -> would have wrongly said YES.)
    # And a clear case below the threshold: 0.5 °C = 32.9 °F -> 33 °F <= 35 -> YES.
    monkeypatch.setattr(grading, "fetch_actual_weather", lambda *a, **k: 0.5)
    assert resolves_yes("Chicago", "2026-03-23", q, bin_c) is True


def test_resolves_yes_celsius_and_missing(monkeypatch):
    """Whole-°C market grades in °C; returns None when station truth is unavailable."""
    q = "Will the highest temperature in London be 20°C or higher on June 25?"
    monkeypatch.setattr(grading, "fetch_actual_weather", lambda *a, **k: 19.6)  # ->20 °C
    assert resolves_yes("London", "2026-06-25", q, 20.0) is True
    monkeypatch.setattr(grading, "fetch_actual_weather", lambda *a, **k: None)
    assert resolves_yes("London", "2026-06-25", q, 20.0) is None


# ── EMOS calibrator (WS2) ────────────────────────────────────────────────────

def test_emos_mean_and_sigma_math():
    """Mean is a linear calibration; sigma trusts ensemble spread but never drops below the floor."""
    from predictors.emos import emos_mean, emos_sigma
    params = {"a": 1.0, "b": 0.9, "c_sin": 0.0, "c_cos": 0.0, "s0_floor": 1.5, "s1": 1.0}
    # mean = 1 + 0.9*20 = 19.0 (seasonal terms zero)
    assert pytest.approx(emos_mean(params, 20.0, 182)) == 19.0
    # low ensemble spread -> floored to s0_floor (fixes overconfidence)
    assert pytest.approx(emos_sigma(params, 0.8, 0.0)) == 1.5
    # high ensemble spread -> trusted, plus diurnal boost
    assert pytest.approx(emos_sigma(params, 3.0, 0.25)) == 3.25


def _tiny_frames():
    daily = pd.DataFrame({
        "date_local":     pd.to_datetime(["2026-07-02"]),
        "fetched_at_utc": pd.to_datetime(["2026-07-01T00:00:00Z"]),
        "temp_max_c":     [20.0],
        "temp_min_c":     [12.0],
    })
    ens = pd.DataFrame({
        "date_local":     pd.to_datetime(["2026-07-02"]),
        "fetched_at_utc": pd.to_datetime(["2026-07-01T00:00:00Z"]),
        "ens_mean": [19.5], "ens_std": [1.0], "ens_p10": [18.0], "ens_p25": [19.0],
        "ens_median": [19.5], "ens_p75": [20.0], "ens_p90": [21.0], "ens_spread": [3.0],
        "n_members": [30],
    })
    return daily, ens


def test_emos_predictor_uses_deterministic_mean(monkeypatch):
    """With trained params, EMOS calibrates the DETERMINISTIC forecast (no train/serve skew)."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params",
                        lambda slug: {"a": 1.0, "b": 0.9, "c_sin": 0.0, "c_cos": 0.0,
                                      "s0_floor": 1.5, "s1": 1.0})
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "emos"
    assert pytest.approx(dist.mu) == 19.0          # 1 + 0.9*20 (deterministic 20, not ens_mean 19.5)
    assert pytest.approx(dist.sigma) == 1.5        # max(1.0, 1.5) floor, no diurnal boost


def test_emos_predictor_falls_back_without_params(monkeypatch):
    """No trained params -> defer to the pure ensemble predictor."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug: None)
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "ensemble"


def test_emos_self_gates_when_calibration_worse(monkeypatch):
    """Self-gating (Fix #2): if the honest holdout says calibration did NOT beat the raw forecast
    (holdout_rmse_calibrated >= holdout_rmse_raw — e.g. London, NYC), EMOS defers to the pure
    ensemble rather than injecting a mean that hurt out of sample."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params",
                        lambda slug: {"a": 1.0, "b": 0.9, "c_sin": 0.0, "c_cos": 0.0,
                                      "s0_floor": 1.5, "s1": 1.0,
                                      "holdout_rmse_calibrated": 1.20,
                                      "holdout_rmse_raw": 0.98})
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "ensemble"          # calibration didn't help -> pure ensemble


def test_emos_applies_when_calibration_helps(monkeypatch):
    """The gate keeps EMOS where calibration DID beat the raw forecast on the holdout
    (holdout_rmse_calibrated < holdout_rmse_raw — e.g. Seoul, Chicago, Hong Kong)."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params",
                        lambda slug: {"a": 1.0, "b": 0.9, "c_sin": 0.0, "c_cos": 0.0,
                                      "s0_floor": 1.5, "s1": 1.0,
                                      "holdout_rmse_calibrated": 1.20,
                                      "holdout_rmse_raw": 1.70})
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "emos"              # calibration helped -> keep EMOS


# ── Shrink-to-market weight sweep (WS3) ──────────────────────────────────────

def test_best_shrink_weight_picks_the_more_accurate_source():
    """The sweep should lean toward whichever of model/market predicts the outcomes better."""
    from evaluate_oos import _best_shrink_weight
    y = [1, 0, 1, 0, 1, 0]
    # Market nails it, model is a coin flip -> best w should be 0 (trust the market).
    mkt_good = pd.DataFrame({"outcome": y,
                             "forecast_prob": [0.5] * 6,
                             "market_prob_raw": [0.95, 0.05, 0.95, 0.05, 0.95, 0.05]})
    best_w, _, _, _ = _best_shrink_weight(mkt_good)
    assert best_w == 0.0
    # Model nails it, market is a coin flip -> best w should be 1 (trust the model).
    model_good = pd.DataFrame({"outcome": y,
                               "forecast_prob": [0.95, 0.05, 0.95, 0.05, 0.95, 0.05],
                               "market_prob_raw": [0.5] * 6})
    best_w, _, _, _ = _best_shrink_weight(model_good)
    assert best_w == 1.0


# ── Guarded coherence bonus (WS6) ────────────────────────────────────────────

def _opp_for_scoring(liquidity, pmf_sum_dev):
    from types import SimpleNamespace
    return SimpleNamespace(
        abs_edge=0.10, bet_side="Yes", ema_momentum=0.0, pmf_sum_dev=pmf_sum_dev,
        volume_recency=0.0, forecast_var=0.0, is_stale=False,
        liquidity=liquidity, days_ahead=1.0)


def test_coherence_bonus_requires_liquidity():
    """The α5 incoherence bonus fires only when the market is liquid enough to actually trade."""
    from signals import score_opportunity
    # Liquid: incoherent (sum!=1) scores HIGHER than coherent — bonus active.
    assert score_opportunity(_opp_for_scoring(5000, 0.5)) > \
           score_opportunity(_opp_for_scoring(5000, 0.0))
    # Illiquid: incoherence earns NO bonus (thin market, won't fill) — scores identical.
    assert score_opportunity(_opp_for_scoring(500, 0.5)) == \
           score_opportunity(_opp_for_scoring(500, 0.0))

