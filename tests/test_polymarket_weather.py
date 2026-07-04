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


# ── EMOS calibrator v2 (per-lead) ────────────────────────────────────────────

def _v2_fit(use_cal=True, a=1.0, sigma=1.5):
    return {"a": a, "b": 0.9, "c_sin": 0.0, "c_cos": 0.0,
            "use_calibrated_mean": use_cal, "sigma": sigma, "nu": 8.0,
            "holdout_rmse_calibrated": 1.2, "holdout_rmse_raw": 1.7}


def _v2_params(input_kind="mm_mean", use_cal=True, sigma=1.5):
    return {"version": 2, "input": input_kind,
            "mm_models": ["ecmwf", "gfs"],
            "leads": {"1": {"mm_mean": _v2_fit(use_cal, a=1.0, sigma=sigma),
                            "mm_proxy": _v2_fit(use_cal, a=3.0, sigma=sigma),
                            "best_match": _v2_fit(use_cal, a=2.0, sigma=sigma)}}}


def test_emos_mean_math_and_gating():
    """Calibrated mean is linear; a holdout-gated fit returns the raw input unchanged."""
    from predictors.emos import emos_mean
    assert pytest.approx(emos_mean(_v2_fit(use_cal=True), 20.0, 182)) == 19.0   # 1 + 0.9*20
    assert pytest.approx(emos_mean(_v2_fit(use_cal=False), 20.0, 182)) == 20.0  # gated -> raw


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
        "n_members": [30], "ens_min_mean": [11.0], "ens_min_std": [0.8],
    })
    return daily, ens


def test_emos_v2_prefers_exact_mm_then_proxy(monkeypatch):
    """With daily_mm data the EXACT multi-model mean is used with the mm_mean fit;
    without it, the live ensemble mean is used WITH THE PROXY FIT (its own sigma) —
    never the full-blend coefficients."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    mm = pd.DataFrame({
        "date_local":     pd.to_datetime(["2026-07-02"]),
        "fetched_at_utc": pd.to_datetime(["2026-07-01T00:00:00Z"]),
        "tmax_ecmwf": [21.0], "tmax_gfs": [23.0],
    })
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens, mm_df=mm)
    assert dist.source == "emos_v2"
    assert pytest.approx(dist.mu) == 1.0 + 0.9 * 22.0   # exact mm mean (21+23)/2, mm_mean fit
    assert pytest.approx(dist.sigma) == 1.5             # max(ens_std 1.0 + boost 0, floor 1.5)
    assert pytest.approx(dist.nu) == 8.0
    # no mm_df -> ensemble mean with the mm_proxy fit (a=3.0)
    dist2 = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert pytest.approx(dist2.mu) == 3.0 + 0.9 * 19.5


def test_emos_v2_falls_back_to_deterministic_with_matching_fit(monkeypatch):
    """No ensemble data -> the deterministic input is used WITH the best_match fit
    (never the mm_mean coefficients), and the sigma floor still applies."""
    from predictors import emos as emos_mod
    daily, _ = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean", sigma=5.0))
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, None)
    assert dist.source == "emos_v2"
    assert pytest.approx(dist.mu) == 2.0 + 0.9 * 20.0   # deterministic 20 with the best_match fit
    assert dist.sigma >= 5.0                            # per-lead floor binds over the NWP table


def test_emos_v2_falls_back_without_params(monkeypatch):
    """No trained params -> defer to the pure ensemble predictor."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": None)
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "ensemble"


def test_emos_v2_gates_mean_but_keeps_sigma_floor(monkeypatch):
    """Where the holdout gated the mean correction off, the raw input mean is served —
    but the per-lead sigma floor is NEVER gated away (it is the main overconfidence fix)."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params",
                        lambda slug, kind="max": _v2_params("mm_mean", use_cal=False))
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert dist.source == "emos_v2"
    assert pytest.approx(dist.mu) == 19.5     # raw ens_mean, no correction
    assert pytest.approx(dist.sigma) == 1.5   # floor still applied


def test_censored_bin_probabilities():
    """With a floor f (running observed max), T = max(f, Z): bins entirely below f get 0,
    the bin containing f absorbs the point mass P(Z <= f), thresholds below f are certain."""
    from pmf import _bin_prob, _cdf
    mu, sigma, nu = 20.0, 1.0, 30.0   # ~Gaussian
    # bin far below the floor: impossible
    assert _bin_prob(16.0, mu, sigma, nu, 0.5, floor=19.0) == 0.0
    # threshold already reached: P(T >= 18) = 1 - F_T(17.5) = 1
    assert 1.0 - _cdf(17.5, mu, sigma, nu, floor=19.0) == 1.0
    # the floor bin picks up the collapsed mass: sum over all bins ≈ 1
    bins = [ _bin_prob(t, mu, sigma, nu, 0.5, floor=19.0) for t in range(15, 27) ]
    assert abs(sum(bins) - 1.0) < 0.01
    # and the floor bin (19) is strictly larger than without the floor
    assert _bin_prob(19.0, mu, sigma, nu, 0.5, floor=19.0) > _bin_prob(19.0, mu, sigma, nu, 0.5)


def _obs_frame(date_str, temps_by_hour):
    return pd.DataFrame({
        "valid_local": pd.to_datetime([f"{date_str} {h:02d}:51" for h in temps_by_hour]),
        "date_local":  [date_str] * len(temps_by_hour),
        "temp_c":      list(temps_by_hour.values()),
    })


def test_emos_v2_intraday_conditioning(monkeypatch):
    """Same-station-local-day bet with obs: the per-hour fit replaces mu/sigma and the
    distribution is floored at the running max; next-day bets are untouched."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    monkeypatch.setattr(emos_mod, "_load_intraday",
                        lambda slug, kind="max": {"version": 1, "nu": 6.0,
                                      "hours": {"13": {"a": 0.5, "b": 0.4, "c": 0.6,
                                                       "sigma": 0.7}}})
    obs = _obs_frame("2026-07-02", {9: 17.0, 11: 18.5, 12: 19.2})
    # fetch at 13:30 local London (UTC+1 in July) on the target day itself
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T12:30:00Z"),
        0.0, daily, ens, obs_df=obs)
    assert dist.source == "emos_v2_intraday"
    assert pytest.approx(dist.floor) == 19.2                      # running max
    assert pytest.approx(dist.mu) == 0.5 + 0.4 * 20.0 + 0.6 * 19.2  # a + b·det + c·M
    assert pytest.approx(dist.sigma) == 0.7
    assert pytest.approx(dist.nu) == 6.0
    # same obs, but a NEXT-day target: no conditioning, no floor
    daily2 = daily.copy(); daily2["date_local"] = pd.to_datetime(["2026-07-03"])
    ens2 = ens.copy(); ens2["date_local"] = pd.to_datetime(["2026-07-03"])
    dist2 = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-03"), pd.Timestamp("2026-07-02T12:30:00Z"),
        1.0, daily2, ens2, obs_df=obs)
    assert dist2.source == "emos_v2"
    assert dist2.floor is None


def test_emos_v2_intraday_floor_without_hour_fit(monkeypatch):
    """Hours the trainer gated off still get the floor (pure information, costs nothing) —
    but keep the unconditioned mu/sigma."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    monkeypatch.setattr(emos_mod, "_load_intraday",
                        lambda slug, kind="max": {"version": 1, "nu": 6.0, "hours": {}})
    obs = _obs_frame("2026-07-02", {9: 17.0})
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T12:30:00Z"),
        0.0, daily, ens, obs_df=obs)
    assert dist.source == "emos_v2"
    assert pytest.approx(dist.floor) == 17.0
    assert pytest.approx(dist.mu) == 3.0 + 0.9 * 19.5   # unchanged proxy-fit calibration


def test_emos_v2_nbm_as_of_join(monkeypatch):
    """input=nbm -> the latest NBM run AVAILABLE at fetch_time is used with the nbm fit;
    runs published later must not leak into the backtest."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    params = _v2_params("nbm")
    params["leads"]["1"]["nbm"] = _v2_fit(a=4.0)
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": params)
    nbm = pd.DataFrame({
        "avail_utc": pd.to_datetime(["2026-07-01T08:00:00Z", "2026-07-01T14:00:00Z"], utc=True),
        "date_local": ["2026-07-02", "2026-07-02"],
        "nbm_tmax_txn_c": [24.0, 26.0],
        "nbm_tmax_tmp_c": [23.0, 25.0],
    })
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "New York City", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens, nbm_df=nbm)
    assert dist.source == "emos_v2"
    # only the 08:00Z run was available at 12:00Z; txn (24.0) preferred over tmp (23.0)
    assert pytest.approx(dist.mu) == 4.0 + 0.9 * 24.0
    # without NBM data the chain falls back to the exact/proxy/deterministic inputs
    dist2 = emos_mod.EMOSPredictor().predict_distribution(
        "New York City", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens)
    assert pytest.approx(dist2.mu) == 3.0 + 0.9 * 19.5   # mm_proxy fit on ens_mean


def test_emos_v2_min_distribution_and_ceiling(monkeypatch):
    """kind='min' -> the Tmin params price off the ensemble MIN stats with their own
    fit; same-day obs set a CEILING (Tmin cannot end above the running min); and with
    no trained Tmin params the predictor returns None (engine skips min bins)."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params",
                        lambda slug, kind="max": _v2_params("mm_mean") if kind == "min" else None)
    monkeypatch.setattr(emos_mod, "_load_intraday", lambda slug, kind="max": None)
    # next-day min bet: proxy fit (a=3.0) on ens_min_mean 11.0; sigma floored at 1.5
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens, kind="min")
    assert dist.source == "emos_v2_min"
    assert pytest.approx(dist.mu) == 3.0 + 0.9 * 11.0
    assert pytest.approx(dist.sigma) == 1.5      # max(ens_min_std 0.8, sigma_lead 1.5)
    assert dist.ceiling is None and dist.floor is None
    # same-day with obs: ceiling = running MIN of the day's observations
    obs = _obs_frame("2026-07-02", {6: 9.5, 9: 11.0, 12: 15.0})
    dist2 = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T12:30:00Z"),
        0.0, daily, ens, obs_df=obs, kind="min")
    assert pytest.approx(dist2.ceiling) == 9.5
    assert dist2.floor is None
    # no trained min params -> None (never price min bins off a Tmax distribution)
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": None)
    assert emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        1.0, daily, ens, kind="min") is None


def test_censored_ceiling_probabilities():
    """Ceiling censoring: bins entirely above the ceiling get 0; P(Tmin <= t) = 1 for
    thresholds at/above it; total mass is conserved."""
    from pmf import _bin_prob, _cdf
    mu, sigma, nu = 10.0, 1.0, 30.0
    assert _bin_prob(13.0, mu, sigma, nu, 0.5, ceiling=9.0) == 0.0
    assert _cdf(9.5, mu, sigma, nu, ceiling=9.0) == 1.0
    bins = [_bin_prob(t, mu, sigma, nu, 0.5, ceiling=9.0) for t in range(4, 15)]
    assert abs(sum(bins) - 1.0) < 0.01
    assert _bin_prob(9.0, mu, sigma, nu, 0.5, ceiling=9.0) > _bin_prob(9.0, mu, sigma, nu, 0.5)


def test_emos_v2_uses_nearest_trained_lead(monkeypatch):
    """days_ahead beyond the trained leads clamps to the nearest available lead entry."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-01T12:00:00Z"),
        6.0, daily, ens)   # only lead "1" is trained
    assert dist.source == "emos_v2"
    assert pytest.approx(dist.mu) == 3.0 + 0.9 * 19.5


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

