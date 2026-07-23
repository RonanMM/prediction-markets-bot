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
    distribution is floored at the running max; next-day bets are untouched.
    C4: at 13:30 local the last COMPLETED hour is 12, so the hour-12 fit is the one applied."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    monkeypatch.setattr(emos_mod, "_load_intraday",
                        lambda slug, kind="max": {"version": 1, "nu": 6.0,
                                      "hours": {"12": {"a": 0.5, "b": 0.4, "c": 0.6,
                                                       "sigma": 0.7}}})
    obs = _obs_frame("2026-07-02", {9: 17.0, 11: 18.5, 12: 19.2})
    # fetch at 13:30 local London (UTC+1 in July) on the target day itself
    dist = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T12:30:00Z"),
        0.0, daily, ens, obs_df=obs)
    assert dist.source == "emos_v2_intraday"
    assert pytest.approx(dist.floor) == 19.2                      # running max through 13:30
    # a + b·(lead-1 forecast) + c·(running max through the completed hour 12)
    assert pytest.approx(dist.mu) == 0.5 + 0.4 * 20.0 + 0.6 * 19.2
    assert pytest.approx(dist.sigma) == 0.7
    assert pytest.approx(dist.nu) == 6.0


def test_intraday_uses_last_completed_hour(monkeypatch):
    """C4: a fit for the CURRENT partial hour is not applied mid-hour (its M was trained through
    the hour's end); it applies only once that hour completes — while the floor still binds."""
    from predictors import emos as emos_mod
    daily, ens = _tiny_frames()
    monkeypatch.setattr(emos_mod, "_load_params", lambda slug, kind="max": _v2_params("mm_mean"))
    monkeypatch.setattr(emos_mod, "_load_intraday",
                        lambda slug, kind="max": {"version": 1, "nu": 6.0,
                                      "hours": {"13": {"a": 0.5, "b": 0.4, "c": 0.6, "sigma": 0.7}}})
    obs = _obs_frame("2026-07-02", {9: 17.0, 11: 18.5, 12: 19.2, 13: 19.8})
    # 13:30 local — hour 13 is still in progress, so the hour-13 fit must NOT be applied yet…
    d_partial = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T12:30:00Z"),
        0.0, daily, ens, obs_df=obs)
    assert d_partial.source == "emos_v2"                 # no intraday fit applied
    assert pytest.approx(d_partial.floor) == 19.2        # …but the floor still binds (obs through 13:30)
    # 14:30 local — hour 13 is now complete, so its fit applies with M through hour 13 (19.8).
    d_done = emos_mod.EMOSPredictor().predict_distribution(
        "London", pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-02T13:30:00Z"),
        0.0, daily, ens, obs_df=obs)
    assert d_done.source == "emos_v2_intraday"
    assert pytest.approx(d_done.mu) == 0.5 + 0.4 * 20.0 + 0.6 * 19.8
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


# ── Phase 0 regression guards (quick wins & safety) ──────────────────────────

def test_fetch_polymarket_get_verifies_tls():
    """F1: the Polymarket price feed must not disable TLS verification (verify=False) nor
    suppress InsecureRequestWarning — it drives real-money bet sizing."""
    src = (Path(__file__).resolve().parents[1] /
           "src/polymarket_weather/fetch_polymarket.py").read_text()
    assert "verify=False" not in src
    assert "disable_warnings" not in src


def test_dedup_uses_full_key_even_when_col_missing(tmp_path):
    """B3: a dedup column missing on one side must NOT silently disable dedup. A second
    append of the same logical rows appends zero new rows."""
    from processing import _append_csv
    path = tmp_path / "d.csv"
    recs = [{"city": "london", "date_local": "2026-07-01", "fetched_at_utc": "2026-07-01T00:00Z"}]
    n1 = _append_csv(path, recs, dedup_cols=["city", "date_local", "fetched_at_utc"])
    n2 = _append_csv(path, recs, dedup_cols=["city", "date_local", "fetched_at_utc"])
    assert n1 == 1 and n2 == 0


def test_processing_has_no_dead_load_ensemble():
    """F9: processing.load_ensemble was dead (engine uses data_loader.load_ensemble); removed."""
    import processing
    assert not hasattr(processing, "load_ensemble")


def test_no_meteostat_in_active_grading_docstring():
    """F10: grading truth is NWS CLI / IEM METAR / HKO — the module docstring must not
    still attribute it to the retired/corrupted Meteostat feed."""
    import grading
    assert "meteostat" not in (grading.__doc__ or "").lower()


# ── Phase 1 regression guards (append-freeze) ────────────────────────────────

def test_append_csv_unions_new_columns(tmp_path):
    """B1: a fetcher that starts emitting a new column must NOT have it silently dropped;
    _append_csv widens the file (old rows NA-backfilled) and the new column persists."""
    from processing import _append_csv
    path = tmp_path / "mm.csv"
    _append_csv(path, [{"city": "london", "date_local": "2026-07-01",
                        "fetched_at_utc": "t0", "tmax_ecmwf": "20"}],
                dedup_cols=["city", "date_local", "fetched_at_utc"])
    # Later fetch adds tmax_gem (the exact class of column the freeze dropped).
    _append_csv(path, [{"city": "london", "date_local": "2026-07-02",
                        "fetched_at_utc": "t1", "tmax_ecmwf": "21", "tmax_gem": "22"}],
                dedup_cols=["city", "date_local", "fetched_at_utc"])
    df = pd.read_csv(path)
    assert "tmax_gem" in df.columns                     # new column survived
    assert set(df["date_local"].astype(str)) == {"2026-07-01", "2026-07-02"}
    # Old row NA for the new column; new row carries its value.
    assert pd.isna(df.loc[df["date_local"] == "2026-07-01", "tmax_gem"]).all()
    assert float(df.loc[df["date_local"] == "2026-07-02", "tmax_gem"].iloc[0]) == 22.0


def test_append_csv_preserves_old_columns_when_new_row_lacks_them(tmp_path):
    """B1: a new row missing an existing column must not drop that column from the file."""
    from processing import _append_csv
    path = tmp_path / "mm.csv"
    _append_csv(path, [{"city": "london", "date_local": "2026-07-01",
                        "fetched_at_utc": "t0", "tmax_ecmwf": "20", "tmax_gem": "22"}],
                dedup_cols=["city", "date_local", "fetched_at_utc"])
    _append_csv(path, [{"city": "london", "date_local": "2026-07-02",
                        "fetched_at_utc": "t1", "tmax_ecmwf": "21"}],
                dedup_cols=["city", "date_local", "fetched_at_utc"])
    df = pd.read_csv(path)
    assert "tmax_gem" in df.columns


def test_ensure_schema_widens_and_is_idempotent(tmp_path):
    """B2: ensure_schema adds missing columns (NA) once, then is a no-op."""
    from processing import ensure_schema
    path = tmp_path / "ens.csv"
    pd.DataFrame([{"city": "london", "date_local": "2026-07-01", "ens_mean": "20"}]).to_csv(
        path, index=False)
    assert ensure_schema(path, ["ens_min_mean", "ens_min_std"]) is True
    cols = set(pd.read_csv(path, nrows=0).columns)
    assert {"ens_min_mean", "ens_min_std"} <= cols
    assert "ens_mean" in cols                            # existing column preserved
    assert ensure_schema(path, ["ens_min_mean", "ens_min_std"]) is False  # idempotent


# ── Phase 2 regression guards (truth-feed integrity, F2) ─────────────────────

def test_truth_sanitize_keeps_min_only_rows():
    """F2: a day with valid Tmin but missing Tmax is KEPT (Tmin markets grade off it),
    not dropped as the old temp_max_c-only dropna did."""
    from fetch_historical_truth import _sanitize_truth
    df = pd.DataFrame([
        {"date_local": "2026-07-01", "temp_max_c": 30.0, "temp_min_c": 20.0, "source": "x"},
        {"date_local": "2026-07-02", "temp_max_c": None, "temp_min_c": 21.0, "source": "x"},
    ])
    out = _sanitize_truth(df, min_expected_rows=1)
    assert out is not None
    assert set(out["date_local"]) == {"2026-07-01", "2026-07-02"}
    assert float(out.loc[out["date_local"] == "2026-07-02", "temp_min_c"].iloc[0]) == 21.0


def test_truth_sanity_rejects_bad_min():
    """F2: an implausible temp_min_c fails the sanity gate (old gate checked max only)."""
    from fetch_historical_truth import _sanitize_truth
    df = pd.DataFrame([
        {"date_local": "2026-07-01", "temp_max_c": 30.0, "temp_min_c": -999.0, "source": "x"},
    ])
    assert _sanitize_truth(df, min_expected_rows=1) is None


# ── Phase 3 regression guards (calibration) ──────────────────────────────────

def test_student_t_scale_is_std():
    """C1: sigma is a standard DEVIATION; _t_scale converts it to the t-scale so the served
    distribution's std equals sigma (the old code passed sigma straight in as the scale,
    over-dispersing by sqrt(nu/(nu-2)))."""
    import numpy as np
    from pmf import _t_scale
    for nu in (4.0, 5.0, 8.0, 15.0):
        scale = _t_scale(1.3, nu)
        realized_std = scale * np.sqrt(nu / (nu - 2.0))
        assert abs(realized_std - 1.3) < 1e-9
    assert _t_scale(1.3, 2.0) == 1.3          # nu<=2: undefined variance → sigma-as-scale, no crash


def _ens_row(**over):
    row = {"date_local": pd.Timestamp("2026-07-05"),
           "fetched_at_utc": pd.Timestamp("2026-07-05 06:00", tz="UTC"),
           "ens_mean": 25.0, "ens_std": 1.5, "ens_p10": 23.0, "ens_p90": 27.0,
           "ens_spread": 4.0, "n_members": 30, "ens_min_mean": 15.0, "ens_min_std": 1.0}
    row.update(over)
    return pd.DataFrame([row])


def test_get_ensemble_params_is_as_of_only():
    """D1: a row fetched AFTER fetch_time must never be selected (no look-ahead leak)."""
    from predictors.ensemble import get_ensemble_params
    ens = _ens_row(fetched_at_utc=pd.Timestamp("2026-07-05 22:00", tz="UTC"))
    assert get_ensemble_params(ens, pd.Timestamp("2026-07-05"),
                               pd.Timestamp("2026-07-03 12:00", tz="UTC")) is None


def test_get_ensemble_params_min_survives_bad_max():
    """C6: a bad Tmax std no longer discards a valid Tmin (Tmin serving keeps its stats)."""
    from predictors.ensemble import get_ensemble_params
    ens = _ens_row(ens_std=float("nan"))
    p = get_ensemble_params(ens, pd.Timestamp("2026-07-05"),
                            pd.Timestamp("2026-07-05 12:00", tz="UTC"))
    assert p is not None
    assert p["ens_mean"] is None and p["ens_std"] is None
    assert p["ens_min_mean"] == 15.0 and p["ens_min_std"] == 1.0


def test_get_ensemble_params_rejects_nan_mean():
    """E4: a NaN mean must not propagate; with both max and min unusable, return None."""
    from predictors.ensemble import get_ensemble_params
    ens = _ens_row(ens_mean=float("nan"), ens_min_mean=float("nan"))
    assert get_ensemble_params(ens, pd.Timestamp("2026-07-05"),
                               pd.Timestamp("2026-07-05 12:00", tz="UTC")) is None


# ── Phase 4 regression guards (range-bin coverage) ───────────────────────────

_RANGE_Q = "Will the highest temperature in NYC be between 62-63°F on March 20?"


def test_marketbin_carries_range_endpoints():
    """A1: parse_question emits range endpoints and MarketBin must retain them (they were
    dropped at construction, losing the info needed to price/grade the range)."""
    from models import MarketBin
    from pmf import parse_question, _f2c
    p = parse_question(_RANGE_Q)
    assert p["condition"] == "range"
    b = MarketBin("c", _RANGE_Q, p["condition"], p["temp_c"], p["half_width"], 0.5,
                  1000, 0, 0, temp_lo=p.get("temp_lo"), temp_hi=p.get("temp_hi"))
    assert abs(b.temp_lo - _f2c(62)) < 1e-9 and abs(b.temp_hi - _f2c(63)) < 1e-9
    pe = parse_question("Will the temperature be 30°C on June 1?")
    be = MarketBin("c", "q", pe["condition"], pe["temp_c"], pe["half_width"], 0.5, 1000, 0, 0)
    assert be.temp_lo is None and be.temp_hi is None      # non-range bins have no endpoints


def test_condition_prob_range_honors_rounding():
    """A3: range prob widens by ±half_width to match the whole-°F rounding set; the old
    _cdf(hi)-_cdf(lo) (a 1°F window) understated it by ~2x."""
    from pmf import parse_question, _condition_prob, _cdf
    p = parse_question(_RANGE_Q)
    mu = p["temp_c"]
    widened = _condition_prob(p, mu, 1.0, 8.0)
    naive = _cdf(p["temp_hi"], mu, 1.0, 8.0) - _cdf(p["temp_lo"], mu, 1.0, 8.0)
    assert widened > naive and widened > 0.4


def test_resolves_yes_range_landmine():
    """A5: a 63°F reading resolves YES for 'between 62-63°F' — the old exact== of the
    banker's-rounded midpoint (round(62.5)=62) graded it NO."""
    from pmf import parse_question, resolves_yes_temp, _f2c
    from grading import native_round
    p = parse_question(_RANGE_Q)
    assert resolves_yes_temp(p, native_round(_f2c(63), "whole °F"), "whole °F", native_round) is True
    assert resolves_yes_temp(p, native_round(_f2c(64), "whole °F"), "whole °F", native_round) is False
    assert resolves_yes_temp(p, native_round(_f2c(61), "whole °F"), "whole °F", native_round) is False


def test_resolves_yes_direction_no_more_than_and_exceed():
    """A5: grading direction matches pmf.parse_question — 'no more than' is lte and 'exceed'
    is gte (the old substring scan flipped 'no more than' to gte and mis-graded 'exceed')."""
    from pmf import parse_question, resolves_yes_temp, _f2c
    from grading import native_round
    lo = parse_question("Will the temperature be no more than 20°C on May 1?")
    assert lo["condition"] == "lte"
    assert resolves_yes_temp(lo, 20, "whole °C", native_round) is True
    assert resolves_yes_temp(lo, 21, "whole °C", native_round) is False
    hi = parse_question("Will the temperature exceed 75°F on May 1?")
    assert hi["condition"] == "gte"
    assert resolves_yes_temp(hi, native_round(_f2c(76), "whole °F"), "whole °F", native_round) is True
    assert resolves_yes_temp(hi, native_round(_f2c(74), "whole °F"), "whole °F", native_round) is False


def test_reconstruct_pmf_range_only():
    """A4: a range-only market (US cities) now builds a non-empty PMF backbone instead of
    returning {} (which silently zeroed the coherence signal)."""
    from models import MarketBin
    from pmf import parse_question, reconstruct_pmf

    def mk(qtext, yp):
        p = parse_question(qtext)
        return MarketBin("c", qtext, p["condition"], p["temp_c"], p["half_width"], yp,
                         1000, 0, 0, temp_lo=p.get("temp_lo"), temp_hi=p.get("temp_hi"))
    bins = [mk("be between 60-61°F on May 1?", 0.5), mk("be between 62-63°F on May 1?", 0.5)]
    mkt, fc, cons = reconstruct_pmf(bins, mu=16.5, sigma=1.5, nu=8.0)
    assert len(mkt) == 2 and len(fc) == 2


def test_all_tracker_questions_parse():
    """A5 audit: every distinct question in the committed snapshots must parse, so grading's
    parse-based path applies and never silently falls back to exact==. Skips if no snapshots."""
    import glob
    from pmf import parse_question
    files = glob.glob(str(Path(__file__).resolve().parents[1] /
                          "src/polymarket_weather/data/polymarket/*_snapshots.csv"))
    if not files:
        pytest.skip("no snapshot CSVs present")
    unparsed = 0
    for f in files:
        for q in pd.read_csv(f, usecols=["question"])["question"].dropna().astype(str).unique():
            if parse_question(q) is None:
                unparsed += 1
    assert unparsed == 0


# ── Phase 5 regression guards (engine / live path) ───────────────────────────

def test_days_from_now_clamps_past_targets():
    """E1: days-from-now is clamped at 0 so a same-day/past target is never negative (which
    would drop it from the `>= min_days` filter — the same-day intraday-bet bug)."""
    from engine import _days_from_now
    from types import SimpleNamespace
    assert _days_from_now(SimpleNamespace(target_date="2000-01-01")) == 0.0


def test_kelly_size_nonfinite_returns_zero():
    """E4: a NaN our_prob must never produce a NaN Kelly (np.clip(nan)=nan) — size 0 instead."""
    from engine import _kelly_size
    from types import SimpleNamespace
    assert _kelly_size(SimpleNamespace(their_prob=0.5, our_prob=float("nan"))) == 0.0


def test_f_unit_sniff_ignores_month_names():
    """E6: the °F unit sniff is digit-anchored, so 'February'/'of' in a °C question are not
    misread as Fahrenheit (which shrank the adjacency threshold)."""
    from engine import _F_UNIT_RE
    assert _F_UNIT_RE.search("be between 62-63°F on March 20?")
    assert not _F_UNIT_RE.search("be 15°C on February 5?")
    assert not _F_UNIT_RE.search("forecast of 15°C tomorrow")


def test_adjacent_bin_filter_collapses_same_side_neighbours():
    """E6: same-side adjacent °C bins (exactly 1.0 apart) are collapsed (old strict `<` never
    fired); non-adjacent bins are kept."""
    from engine import apply_group_kelly_cap
    from types import SimpleNamespace

    def opp(temp, q, score):
        return SimpleNamespace(group_key="g", alpha_score=score, bet_side="Yes",
                               bin_temp_c=temp, question=q, kelly=0.05)
    adj = apply_group_kelly_cap([opp(14.0, "be 14°C on Feb 5?", 0.9),
                                 opp(15.0, "be 15°C on Feb 5?", 0.8)], bankroll=1000)
    assert len(adj) == 1
    far = apply_group_kelly_cap([opp(14.0, "be 14°C on Feb 5?", 0.9),
                                 opp(17.0, "be 17°C on Feb 5?", 0.8)], bankroll=1000)
    assert len(far) == 2


def test_momentum_uses_selected_column():
    """F4: momentum tracks the requested series — Tmin markets must use temp_min_c drift, not
    the daily-max drift; a missing column degrades to neutral without crashing."""
    from signals import compute_momentum
    df = pd.DataFrame([
        {"date_local": pd.Timestamp("2026-07-05"), "fetched_at_utc": pd.Timestamp("2026-07-01", tz="UTC"),
         "temp_max_c": 30.0, "temp_min_c": 20.0},
        {"date_local": pd.Timestamp("2026-07-05"), "fetched_at_utc": pd.Timestamp("2026-07-02", tz="UTC"),
         "temp_max_c": 33.0, "temp_min_c": 20.5},
    ])
    ft = pd.Timestamp("2026-07-03", tz="UTC")
    td = pd.Timestamp("2026-07-05")
    assert abs(compute_momentum(df, td, ft, col="temp_max_c")["total_drift"] - 3.0) < 1e-9
    assert abs(compute_momentum(df, td, ft, col="temp_min_c")["total_drift"] - 0.5) < 1e-9
    assert compute_momentum(df, td, ft, col="nope")["n_snaps"] == 0


def test_opportunity_carries_model_prob_for_live_reshrink():
    """E3: the pre-shrink model probability and shrink weight are stored on the Opportunity so
    live re-verification can re-shrink toward the fresh price instead of the stale snapshot."""
    from models import Opportunity
    assert "model_prob_side" in Opportunity.__dataclass_fields__
    assert "shrink_weight" in Opportunity.__dataclass_fields__


# ── Phase 6 regression guards (backtest economics + arbiter) ─────────────────

def test_settle_bet_crosses_spread_and_fee():
    """D6/D7: honest settlement crosses HALF_SPREAD on entry and pays FEE_RATE on the win, so
    a winning bet nets strictly LESS than the retired (payout-size)*0.98 economics."""
    from backtest_common import settle_bet
    win = settle_bet(0.5, True, 100.0)
    retired = (100.0 / 0.5 - 100.0) * 0.98
    assert win < retired
    assert settle_bet(0.5, False, 100.0) == -100.0


def test_single_kelly_matches_engine():
    """D6: the shared Kelly is byte-for-byte engine._kelly_size (so all tools size identically)."""
    from backtest_common import single_kelly
    from engine import _kelly_size
    from types import SimpleNamespace
    for our, their in [(0.6, 0.4), (0.3, 0.5), (0.9, 0.85), (0.5, 0.5)]:
        a = single_kelly(our, their)
        b = _kelly_size(SimpleNamespace(our_prob=our, their_prob=their))
        assert abs(a - b) < 1e-12


def test_apply_caps_group_then_portfolio():
    """D6: per-group cap (MAX_KELLY_PER_GROUP) then portfolio cap (MAX_TOTAL_KELLY)."""
    from backtest_common import apply_caps
    k = apply_caps([0.15, 0.15], ["g", "g"])           # 0.30 in one group -> 0.20
    assert abs(k.sum() - 0.20) < 1e-9 and abs(k[0] - 0.10) < 1e-9
    k2 = apply_caps([0.30, 0.30, 0.30], ["a", "b", "c"])  # 3×0.20=0.60 -> portfolio 0.40
    assert abs(k2.sum() - 0.40) < 1e-9


def test_crps_by_key_separates_max_and_min(tmp_path, monkeypatch):
    """D2: a Tmax and a Tmin market on the same city-date are scored as SEPARATE keys (the old
    groupby.last() spliced one row's floor onto the other → a chimera)."""
    import evaluate_oos
    monkeypatch.setattr(evaluate_oos, "fetch_actual_weather", lambda c, d, q="": 25.0)
    df = pd.DataFrame([
        {"city": "London", "target_date": "2026-07-05", "fetched_at": "2",
         "question": "Will the highest temperature be 26°C on July 5?",
         "forecast_mu": 26.0, "forecast_sigma": 1.5, "forecast_nu": 8.0},
        {"city": "London", "target_date": "2026-07-05", "fetched_at": "2",
         "question": "Will the lowest temperature be 18°C on July 5?",
         "forecast_mu": 18.0, "forecast_sigma": 1.0, "forecast_nu": 8.0},
    ])
    p = tmp_path / "cal.csv"
    df.to_csv(p, index=False)
    assert {k[2] for k in evaluate_oos._crps_by_key(p)} == {"max", "min"}


# ── E2 regression guards (question-date target) ──────────────────────────────

def test_parse_question_date_basic_and_year_wrap():
    """E2: the target date is read from the question, with the year inferred from the reference
    (endDateIso) so a December/January wrap resolves correctly."""
    from datetime import date
    from pmf import parse_question_date
    assert parse_question_date("… be 21°C on March 18?", date(2026, 3, 19)) == date(2026, 3, 18)
    assert parse_question_date("… be 5°C on January 2?", date(2025, 12, 30)) == date(2026, 1, 2)
    assert parse_question_date("… highest on Mar 3rd?", date(2026, 3, 1)) == date(2026, 3, 3)
    assert parse_question_date("Will it be hot tomorrow?", date(2026, 3, 1)) is None


def test_question_date_is_noop_outside_hong_kong():
    """E2 audit guard: for the collected snapshots, the question date must equal endDateIso for
    every city EXCEPT Hong Kong (whose endDateIso is a day off). Skips if no snapshots present."""
    import glob
    from pmf import parse_question_date
    files = glob.glob(str(Path(__file__).resolve().parents[1] /
                          "src/polymarket_weather/data/polymarket/*_snapshots.csv"))
    if not files:
        pytest.skip("no snapshot CSVs present")
    for f in files:
        if "hong_kong" in f:
            continue
        df = pd.read_csv(f, usecols=["question", "end_date_iso"]).dropna(subset=["end_date_iso"])
        end = pd.to_datetime(df["end_date_iso"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        disagree = sum(
            1 for q, en in zip(df["question"], end)
            if not pd.isna(en) and parse_question_date(q, en) is not None
            and pd.Timestamp(parse_question_date(q, en)) != en
        )
        assert disagree == 0, f"{f}: {disagree} question/endDateIso disagreements (expected 0)"



def test_bucket_key_bands_and_live_eligibility():
    """E3 (edge megaplan): buckets are 'City|band' with same-day ≤0.5d, 1d ≤1.5d, else 2d+.
    LIVE_BUCKETS holds execution-eligible buckets; membership is exact-string, so bucket_key
    output must match the config entries verbatim."""
    from config import bucket_key, LIVE_BUCKETS
    assert bucket_key("NYC", 0.0) == "NYC|same-day"
    assert bucket_key("NYC", 0.5) == "NYC|same-day"
    assert bucket_key("Seoul", 0.51) == "Seoul|1d"
    assert bucket_key("Seoul", 1.5) == "Seoul|1d"
    assert bucket_key("London", 1.51) == "London|2d+"
    assert bucket_key("Chicago", 6.0) == "Chicago|2d+"
    # every nominated bucket must be reconstructible via bucket_key (guards against drift
    # between the config strings and the key format, without pinning the data-driven list)
    for bk in LIVE_BUCKETS:
        city, band = bk.split("|")
        d = {"same-day": 0.3, "1d": 1.0, "2d+": 3.0}[band]
        assert bucket_key(city, d) == bk


def test_epoch_to_utc_iso_seconds_and_ms():
    """CLOB history `t` is epoch SECONDS (the old ÷1000 bug collapsed 2026 onto 1970-01-21);
    millisecond inputs (>1e12) must still resolve to the same instant."""
    from fetch_polymarket import _epoch_to_utc_iso
    s = 1_783_000_000            # 2026-07-02T13:46:40+00:00
    assert _epoch_to_utc_iso(s) == "2026-07-02T13:46:40+00:00"
    assert _epoch_to_utc_iso(s * 1000) == _epoch_to_utc_iso(s)
    assert not _epoch_to_utc_iso(s).startswith("1970")


def test_wu_truth_matches_actual_settlements():
    """W0 (megaplan §10a): markets resolve on Wunderground (hourly-METAR extremes over the
    local calendar day), not the NWS CLI. These three markets settled OPPOSITE to the CLI-based
    grade and the WU reconstruction must match the real settlement. Data comes from the
    committed obs CSVs; skip if absent."""
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "src/polymarket_weather/data/weather"
    if not (base / "new_york_city_obs_hourly.csv").exists():
        pytest.skip("obs CSVs not present")
    # NYC 2026-07-03: CLI Tmin 79°F, WU/hourly 80°F -> '80-81°F' settled YES
    assert resolves_yes("NYC", "2026-07-03",
                        "Will the lowest temperature in New York City be between 80-81°F on July 3?",
                        26.9) is True
    # NYC 2026-06-27: CLI Tmin 69°F, WU/hourly 70°F -> '68-69°F' settled NO
    assert resolves_yes("NYC", "2026-06-27",
                        "Will the lowest temperature in New York City be between 68-69°F on June 27?",
                        20.3) is False
    # Chicago 2026-05-28: CLI Tmax 72°F, WU/hourly 71°F -> '72-73°F' settled NO
    assert resolves_yes("Chicago", "2026-05-28",
                        "Will the highest temperature in Chicago be between 72-73°F on May 28?",
                        22.5) is False


def test_wu_truth_fallback_and_scope():
    """Non-validated cities return None from wu_daily_extreme (grading falls back to the
    historical-actuals feed); unknown dates return None rather than a fabricated extreme."""
    from wu_truth import wu_daily_extreme
    assert wu_daily_extreme("Seoul", "2026-03-24", "max") is None      # not WU-validated
    assert wu_daily_extreme("London", "2026-07-03", "max") is None     # not WU-validated
    assert wu_daily_extreme("NYC", "1999-01-01", "max") is None        # before obs coverage


def test_shoulder_band_rule():
    """§10b/§10e pre-registered bands: shoulder full [0.05, 0.35) core [0.20, 0.35);
    favorite [0.65, 0.85) core [0.65, 0.75), entries only >12h before local day end."""
    from shoulder_book import (BAND_LO, BAND_HI, CORE_LO,
                               FAV_LO, FAV_CORE_HI, FAV_HI, FAV_MIN_HOURS_TO_END)
    assert (BAND_LO, BAND_HI, CORE_LO) == (0.05, 0.35, 0.20)
    assert (FAV_LO, FAV_CORE_HI, FAV_HI) == (0.65, 0.75, 0.85)
    assert FAV_MIN_HOURS_TO_END == 12.0
    inside = [0.05, 0.19, 0.20, 0.349]
    outside = [0.049, 0.35, 0.60, 0.01]
    assert all(BAND_LO <= p < BAND_HI for p in inside)
    assert not any(BAND_LO <= p < BAND_HI for p in outside)


def test_verified_weather_taker_fee():
    """E1 (verified 2026-07-13): weather taker fee = 0.05·p·(1−p) per share — 1.25¢ max at
    p=0.5, symmetric, tiny at the extremes; makers pay zero."""
    from config import taker_fee_per_share, MAKER_FEE, WEATHER_TAKER_RATE
    assert WEATHER_TAKER_RATE == 0.05 and MAKER_FEE == 0.0
    assert pytest.approx(taker_fee_per_share(0.5)) == 0.0125
    assert pytest.approx(taker_fee_per_share(0.7)) == taker_fee_per_share(0.3)
    assert pytest.approx(taker_fee_per_share(0.7)) == 0.0105
    assert taker_fee_per_share(0.99) < 0.001
    assert taker_fee_per_share(0.0) == 0.0 and taker_fee_per_share(1.0) == 0.0


def test_maker_fill_is_conservative_trade_through():
    """A resting maker order fills only if a LATER price trades through it from the taker side:
    SELL YES fills when price ticks back UP to the ask; BUY YES fills when it ticks DOWN to the bid."""
    from shoulder_book import maker_filled
    # SELL YES (side='No') posted at 0.27 — fills if a later snapshot >= 0.27, not if it only falls
    assert maker_filled("No", 0.27, [0.24, 0.28, 0.10]) is True
    assert maker_filled("No", 0.27, [0.24, 0.20, 0.05]) is False   # only fell → never lifted
    # BUY YES (side='Yes') posted at 0.70 — fills if a later snapshot <= 0.70, not if it only rises
    assert maker_filled("Yes", 0.70, [0.72, 0.68, 0.81]) is True
    assert maker_filled("Yes", 0.70, [0.75, 0.80, 0.90]) is False  # only rose → bid never hit
    assert maker_filled("No", 0.27, []) is False                    # no later prices → no fill


def test_backbone_stats_handles_range_only_markets():
    """Range-only (NYC/Chicago) markets must NOT be pinned at pmf_dev=1.0 / mode=mu.

    Regression for the α5 backbone bug: keying coherence + market mode off `exact` bins
    alone made every US-city (range-only) row report maximal fake incoherence and zero
    model-vs-market mode disagreement.
    """
    from engine import _backbone_stats
    from models import MarketBin
    def rb(t, y):
        return MarketBin("c", "between X-Y°F", "range", t, 0.278, y, 5000, 0, 0,
                         temp_lo=t - 0.3, temp_hi=t + 0.3)
    bins = [rb(29.0, 0.30), rb(30.0, 0.45), rb(31.0, 0.20)]   # sums to 0.95
    dev, mode = _backbone_stats(bins, mu_fallback=99.0)
    assert abs(dev - 0.05) < 1e-9        # |0.95 - 1| — NOT 1.0
    assert mode == 30.0                  # priciest bin, NOT the model mean
    dev0, mode0 = _backbone_stats([], mu_fallback=99.0)
    assert dev0 == 0.0 and mode0 == 99.0  # empty ⇒ unknown coherence, mode falls back


def test_reconstruct_pmf_consistency_nan_without_constraints():
    """A market with no gte/lte bins reports NaN coherence (unchecked), not a fake 1.0."""
    import math
    from pmf import reconstruct_pmf
    from models import MarketBin
    bins = [MarketBin("c", "q", "range", 30.0, 0.278, 0.5, 5000, 0, 0,
                      temp_lo=29.7, temp_hi=30.3)]
    _, _, cons = reconstruct_pmf(bins, 30.0, 1.5, 8.0)
    assert math.isnan(cons)


def test_moderate_gate_stats():
    """Leg 1b: report-time moderate-shoulder [0.10,0.25) gate, forward-only (entered >=
    MOD_PREREG_DATE). Verifies band filter, forward-date filter, and taker gate math."""
    import pandas as pd
    from shoulder_book import moderate_gate_stats, MOD_LO, MOD_HI, MOD_PREREG_DATE, GATE_MOD
    assert (MOD_LO, MOD_HI) == (0.10, 0.25)
    assert MOD_PREREG_DATE == "2026-07-23"
    need_n, need_e = GATE_MOD

    def rows(n, yes, won, entered):
        # a shoulder SELL: side='No', entry_side_price = 1 - yes
        return pd.DataFrame({
            "entry_yes_price": [yes] * n,
            "entered_at_utc": [entered] * n,
            "side_won": [won] * n,
            "entry_side_price": [round(1.0 - yes, 4)] * n,
        })

    PRE = "2026-01-01T00:00:00+00:00"    # before pre-reg
    POST = "2026-12-31T00:00:00+00:00"   # on/after pre-reg

    # band + forward filters: deep (0.07) and core (0.30) excluded; PRE entries context-only
    df = pd.concat([
        rows(3, 0.15, True, POST),   # in band, forward
        rows(2, 0.07, True, POST),   # deep band -> excluded
        rows(4, 0.30, True, POST),   # core band -> excluded
        rows(5, 0.15, True, PRE),    # in band but pre-reg -> context only
    ], ignore_index=True)
    s = moderate_gate_stats(df)
    assert s["context"]["n"] == 8        # in-band: 3 forward + 5 pre
    assert s["forward"]["n"] == 3        # in-band AND forward
    assert s["forward"]["gate_pass"] is False   # n=3 < 80

    # gate PASSES: >=80 forward in-band winners (taker per win >> +0.03)
    sp = moderate_gate_stats(rows(need_n, 0.12, True, POST))
    assert sp["forward"]["n"] == need_n
    assert sp["forward"]["taker"] >= need_e
    assert sp["forward"]["gate_pass"] is True

    # gate FAILS on edge: >=80 forward in-band losers (taker very negative)
    sl = moderate_gate_stats(rows(need_n, 0.12, False, POST))
    assert sl["forward"]["n"] == need_n
    assert sl["forward"]["gate_pass"] is False

    # empty / missing columns -> {}
    assert moderate_gate_stats(pd.DataFrame()) == {}


# ── Breadth structure book (all Polymarket weather cities) ──────────────────────

def test_moderate_gate_prereg_date_kwarg():
    """The additive prereg_date kwarg re-bases the forward clock without touching the default."""
    import shoulder_book as sb
    import pandas as pd
    graded = pd.DataFrame([
        {"entry_yes_price": 0.15, "entered_at_utc": "2026-07-20T00:00:00+00:00",
         "side_won": True,  "entry_side_price": 0.85},
        {"entry_yes_price": 0.15, "entered_at_utc": "2026-08-01T00:00:00+00:00",
         "side_won": False, "entry_side_price": 0.85},
    ])
    # default date (2026-07-23): forward = the Aug-01 row only
    assert sb.moderate_gate_stats(graded)["forward"]["n"] == 1
    # custom later date: forward = only the Aug-01 row (>= 2026-08-01)
    assert sb.moderate_gate_stats(graded, prereg_date="2026-08-01")["forward"]["n"] == 1
    # custom earlier date: both rows are forward
    assert sb.moderate_gate_stats(graded, prereg_date="2026-07-01")["forward"]["n"] == 2


def test_parse_event_title_and_bins():
    import shoulder_book_breadth as b
    assert b.parse_event_title("Highest temperature in Paris on July 23?") == ("max", "Paris", "July 23")
    assert b.parse_event_title("Lowest temperature in New York City on July 3?") == ("min", "New York City", "July 3")
    assert b.parse_event_title("Who wins the election?") is None

    ev = {"title": "Highest temperature in Paris on July 23?", "endDate": "2026-07-23T22:00:00Z",
          "markets": [
              {"conditionId": "0xAAA", "id": "111",
               "question": "Highest temperature in Paris on July 23 (30-31°C)?",
               "groupItemTitle": "30-31", "outcomePrices": "[\"0.18\", \"0.82\"]", "liquidityNum": 5000},
              {"conditionId": "0xBBB", "id": "222", "question": "…(bad)…",
               "groupItemTitle": "x", "outcomePrices": "not-json", "liquidityNum": 10},
          ]}
    bins = b.bins_from_event(ev)
    assert len(bins) == 1                      # bad-price bin skipped
    r = bins[0]
    assert r["city"] == "Paris" and r["kind"] == "max"
    assert r["condition_id"] == "0xAAA" and r["market_id"] == "111"
    assert abs(r["yes"] - 0.18) < 1e-9 and abs(r["liquidity"] - 5000) < 1e-9
    assert str(r["end"].tz) == "UTC"


def test_fetch_weather_bins_injected():
    import shoulder_book_breadth as b
    page = [{"title": "Highest temperature in Paris on July 23?", "endDate": "2026-07-23T22:00:00Z",
             "markets": [{"conditionId": "0xAAA", "id": "1", "question": "q",
                          "outcomePrices": "[\"0.2\",\"0.8\"]", "liquidityNum": 100}]},
            {"title": "Not a temperature market", "markets": []}]

    def fake(url, params, label="API"):
        return page if params.get("offset", 0) == 0 else []   # one page then empty

    bins = b.fetch_weather_bins(fetch=fake)
    assert len(bins) == 1 and bins[0]["city"] == "Paris"


def test_scan_and_record_breadth(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd
    from datetime import datetime, timezone
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    end_next_day = pd.Timestamp("2026-07-23T22:00:00Z")   # ~34h out -> pre-day
    end_soon     = pd.Timestamp("2026-07-22T20:00:00Z")   # ~8h out  -> NOT pre-day

    def mk(cid, yes, end, q="Highest temperature in Paris on July 23 (30-31°C)?"):
        return dict(condition_id=cid, market_id=cid[2:], city="Paris", kind="max",
                    date_str="July 23", question=q, yes=yes, liquidity=5000, end=end)

    bins = [
        mk("0xSH", 0.15, end_next_day),   # shoulder [5,35), pre-day -> recorded (No)
        mk("0xFV", 0.70, end_next_day),   # favorite [65,85), >12h  -> recorded (Yes)
        mk("0xLATE", 0.15, end_soon),     # shoulder band but <24h  -> NOT recorded as shoulder
        mk("0xMID", 0.50, end_next_day),  # neither band            -> nothing
    ]
    out = tmp_path / "breadth.csv"
    n = b.scan_and_record_breadth(bins=bins, now_utc=now, out_path=out)
    assert n == 2
    df = pd.read_csv(out)
    legs = set(zip(df["condition_id"], df["leg"]))
    assert ("0xSH", "shoulder") in legs
    assert ("0xFV", "favorite") in legs
    assert ("0xLATE", "shoulder") not in legs
    assert df[df["condition_id"] == "0xMID"].empty
    # dedup: running again adds nothing
    assert b.scan_and_record_breadth(bins=bins, now_utc=now, out_path=out) == 0
    # recorded sides/prices
    sh = df[df["condition_id"] == "0xSH"].iloc[0]
    assert sh["side"] == "No" and abs(sh["entry_side_price"] - 0.85) < 1e-6
    fv = df[df["condition_id"] == "0xFV"].iloc[0]
    assert fv["side"] == "Yes" and abs(fv["entry_side_price"] - 0.70) < 1e-6


def test_settlement_outcome_and_freeze(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd

    def fetch(url, params=None, label="API"):
        mid = url.rstrip("/").split("/")[-1]
        return {"111": {"closed": True,  "outcomePrices": "[\"1\",\"0\"]"},   # YES won
                "222": {"closed": True,  "outcomePrices": "[\"0\",\"1\"]"},   # NO won
                "333": {"closed": False, "outcomePrices": "[\"0.5\",\"0.5\"]"}}[mid]

    assert b.settlement_outcome("111", fetch=fetch) == 1
    assert b.settlement_outcome("222", fetch=fetch) == 0
    assert b.settlement_outcome("333", fetch=fetch) is None

    # grade_book fills settled_outcome once and freezes it
    book = pd.DataFrame([
        {**{c: "" for c in b._BCOLS}, "condition_id": "0xA", "market_id": "111",
         "leg": "shoulder", "side": "No", "entry_side_price": 0.85,
         "entry_yes_price": 0.15, "entered_at_utc": "2026-07-24T00:00:00+00:00"},
    ])
    out = tmp_path / "breadth.csv"
    book.reindex(columns=b._BCOLS).to_csv(out, index=False)
    g = b.grade_book(out_path=out, fetch=fetch)
    assert int(g.iloc[0]["settled_outcome"]) == 1
    # side "No" with YES-won => side lost
    assert bool(g.iloc[0]["side_won"]) is False
    # freeze: a fetch that would now say 0 must NOT overwrite the persisted 1
    def fetch2(url, params=None, label="API"):
        return {"closed": True, "outcomePrices": "[\"0\",\"1\"]"}
    g2 = b.grade_book(out_path=out, fetch=fetch2)
    assert int(g2.iloc[0]["settled_outcome"]) == 1


def test_report_breadth_gate_forward_only(tmp_path, capsys):
    import shoulder_book_breadth as b
    import pandas as pd
    rows = []
    for cid, entered, won in [("0xA", "2026-07-01T00:00:00+00:00", 1),
                              ("0xB", "2026-07-25T00:00:00+00:00", 1)]:
        rows.append({**{c: "" for c in b._BCOLS}, "condition_id": cid, "market_id": cid,
                     "leg": "shoulder", "side": "No", "entry_yes_price": 0.15,
                     "entry_side_price": 0.85, "entered_at_utc": entered, "settled_outcome": won})
    out = tmp_path / "breadth.csv"
    pd.DataFrame(rows).reindex(columns=b._BCOLS).to_csv(out, index=False)
    b.report_breadth(out_path=out, fetch=lambda *a, **k: None)   # no-op fetch (already settled)
    text = capsys.readouterr().out
    assert "BREADTH" in text.upper()
    # forward gate counts ONLY the post-2026-07-23 entry
    graded = b.grade_book(out_path=out, fetch=lambda *a, **k: None)
    stats = b.moderate_gate_stats(graded, prereg_date=b.BREADTH_PREREG_DATE)
    assert stats["forward"]["n"] == 1


def test_breadth_maker_path_and_fill(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd
    from datetime import datetime, timezone, timedelta
    end = pd.Timestamp("2026-07-25T22:00:00Z")   # far future -> pre-day, >12h to end

    def mk(cid, yes):
        return dict(condition_id=cid, market_id=cid[2:], city="Paris", kind="max",
                    date_str="July 25", question="Highest temperature in Paris on July 25 (30-31°C)?",
                    yes=yes, liquidity=5000, end=end)

    out = tmp_path / "b.csv"
    t1 = datetime(2026, 7, 23, 0, 0, tzinfo=timezone.utc)
    # cycle 1: record a shoulder sell (0.15) and a favorite buy (0.70)
    b.scan_and_record_breadth(bins=[mk("0xSH", 0.15), mk("0xFV", 0.70)], now_utc=t1, out_path=out)
    d = pd.read_csv(out)
    assert d.loc[d.condition_id == "0xSH", "max_yes_after"].isna().all()   # entry cycle: no path yet

    # cycle 2: shoulder ticks UP through the resting sell (0.20 >= 0.15 -> fills);
    #          favorite ticks DOWN through the resting bid (0.60 <= 0.70 -> fills)
    t2 = t1 + timedelta(hours=2)
    n2 = b.scan_and_record_breadth(bins=[mk("0xSH", 0.20), mk("0xFV", 0.60)], now_utc=t2, out_path=out)
    assert n2 == 0   # dedup: no new entries, only path updates
    d = pd.read_csv(out)
    assert float(d.loc[d.condition_id == "0xSH", "max_yes_after"].iloc[0]) == 0.20
    assert float(d.loc[d.condition_id == "0xFV", "min_yes_after"].iloc[0]) == 0.60

    def fetch(url, params=None, label="API"):
        mid = url.rstrip("/").split("/")[-1]
        return {"SH": {"closed": True, "outcomePrices": "[\"0\",\"1\"]"},   # NO won -> shoulder wins
                "FV": {"closed": True, "outcomePrices": "[\"1\",\"0\"]"}}[mid]  # YES won -> favorite wins
    g = b.grade_book(out_path=out, fetch=fetch)
    assert bool(g[g.condition_id == "0xSH"].iloc[0]["maker_filled"]) is True
    assert bool(g[g.condition_id == "0xFV"].iloc[0]["maker_filled"]) is True

    # a shoulder whose price NEVER rose to entry does NOT fill as maker
    out2 = tmp_path / "b2.csv"
    b.scan_and_record_breadth(bins=[mk("0xNF", 0.15)], now_utc=t1, out_path=out2)
    b.scan_and_record_breadth(bins=[mk("0xNF", 0.11)], now_utc=t2, out_path=out2)   # only falls
    g2 = b.grade_book(out_path=out2, fetch=lambda u, p=None, l="API": {"closed": True, "outcomePrices": "[\"0\",\"1\"]"})
    assert bool(g2.iloc[0]["maker_filled"]) is False
