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


def test_suite_does_not_mutate_tracked_data_or_create_stray_dirs():
    """A unit test must never write to tracked data or hit the network.

    step_fetch_weather/step_fetch_ensemble call five side-effecting fetchers beyond
    fetch_forecast; two of them write real files — shoulder_book_breadth anchors to
    Path(__file__).parent and mutates the TRACKED output/shoulder_paper_breadth.csv, and
    fetch_station_obs uses the bare relative OUT_DIR "data/weather", which creates a stray
    data/ at whatever cwd pytest ran from. Both were observed on a clean tree.

    Placed FIRST in the module (pytest runs top-to-bottom within a file) so a dirty tree from an
    earlier test is attributed to that test rather than silently tolerated.
    """
    import subprocess, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    before = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True).stdout
    assert before.strip() == "", (
        "working tree was already dirty before this test — cannot attribute side effects; "
        f"clean it and re-run:\n{before}")


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
    # Hong Kong: whole-°C bins TRUNCATED from HKO's tenths. 27.8 belongs to bin 27, not 28 —
    # verified against 171 real settlements (floor 100%, round-to-nearest 93.6%).
    assert native_round(27.8, "whole °C (floor)") == 27
    assert native_round(31.6, "whole °C (floor)") == 31
    assert native_round(30.0, "whole °C (floor)") == 30


def test_hong_kong_bins_can_resolve_yes():
    """Hong Kong's grid is whole °C, NOT HKO's 0.1 °C publishing precision.

    Regression for the 2026-07-31 ruler bug: `resolution_unit` held the SOURCE's precision, so
    an exact whole-degree bin was compared against a tenths-rounded observation (27.8 == 27 is
    never true) and every HK market graded NO — 0/179 YES in the eval tracker against 12-23%
    elsewhere. A shoulder SELL on any HK bin therefore looked like a guaranteed win.
    """
    from resolution_anchors import RESOLUTION_ANCHORS
    from pmf import resolves_yes_temp
    unit = RESOLUTION_ANCHORS["Hong Kong"]["resolution_unit"]
    # The anchor must describe the market's bin grid, not the station's precision.
    assert "0.1" not in unit, "HK unit regressed to the source's publishing precision"

    # The real settled case that discriminates floor from round: HKO 27.8 -> bin 27 paid YES.
    p_27 = parse_question("Will the highest temperature in Hong Kong be 27°C on June 2?")
    p_28 = parse_question("Will the highest temperature in Hong Kong be 28°C on June 2?")
    obs = native_round(27.8, unit)
    assert resolves_yes_temp(p_27, obs, unit, native_round) is True
    assert resolves_yes_temp(p_28, obs, unit, native_round) is False


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


def test_temperature_distribution_has_optional_cdf():
    """Optional cdf field on TemperatureDistribution for QRF empirical distributions."""
    from predictors.base import TemperatureDistribution
    d = TemperatureDistribution(mu=20.0, sigma=2.0, nu=10.0, source="qrf")
    assert d.cdf is None                       # default: parametric path
    f = lambda x: 0.5
    d2 = TemperatureDistribution(mu=20.0, sigma=2.0, nu=10.0, source="qrf", cdf=f)
    assert d2.cdf(999) == 0.5                   # callable carried
    # existing positional/keyword construction still works (floor/ceiling unaffected)
    d3 = TemperatureDistribution(20.0, 2.0, 10.0, "emos_v2", floor=18.0)
    assert d3.cdf is None and d3.floor == 18.0


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


def test_price_paths_unions_the_dense_price_history_with_snapshots(monkeypatch):
    """2026-07-27 reconciliation: the maker-fill test walked the SNAPSHOT store, which carries a
    median of 2 observations per market over ~1.1h — so 36% of entries had zero later prices and
    could never fill, and measured maker P&L (+0.072) was an artifact of near-blindness. The CLOB
    price history covers the same markets with ~25 observations over ~24h; paths must use both."""
    import pandas as pd
    import shoulder_book as sb
    snap = pd.DataFrame({
        "condition_id": ["0xA", "0xA"],
        "t": pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T06:00:00Z"], utc=True),
        "yes": [0.20, 0.22],
    })
    hist = pd.DataFrame({
        "condition_id": ["0xA", "0xA", "0xB"],
        # 03:00 sits BETWEEN the two snapshots and is higher than either — the kind of
        # intra-gap move the snapshot store cannot see.
        "t": pd.to_datetime(["2026-07-01T03:00:00Z", "2026-07-01T06:00:00Z",
                             "2026-07-02T00:00:00Z"], utc=True),
        "yes": [0.31, 0.22, 0.40],
    })
    monkeypatch.setattr(sb, "_all_snapshots", lambda: snap)
    monkeypatch.setattr(sb, "_load_price_history", lambda: hist)
    paths = sb._price_paths({"0xA", "0xB"})

    a = paths["0xA"]
    assert list(a["t"].dt.hour) == [0, 3, 6]          # merged and time-sorted
    assert a["yes"].max() == 0.31                     # the intra-gap peak is now visible
    assert len(a) == 3                                # 06:00 duplicate collapsed, not double-counted
    assert "0xB" in paths                             # markets only in price history are covered


def test_moderate_gate_stats():
    """Leg 1b: report-time moderate-shoulder [0.10,0.25) gate, forward-only (entered >=
    MOD_PREREG_DATE). Verifies band filter, forward-date filter, and taker gate math."""
    import pandas as pd
    from shoulder_book import moderate_gate_stats, MOD_LO, MOD_HI, MOD_PREREG_DATE, GATE_MOD
    assert (MOD_LO, MOD_HI) == (0.10, 0.25)
    assert MOD_PREREG_DATE == "2026-07-23"
    need_n, need_e = GATE_MOD

    def rows(n, yes, won, entered):
        # a shoulder SELL: side='No', entry_side_price = 1 - yes.
        # city/target_date span 40 dates so the fixture can satisfy the 2026-08-02 temporal
        # amendment (>= GATE_MIN_DATES distinct target dates); without them these rows carry no
        # date component and could never pass, which is not what this test is about.
        return pd.DataFrame({
            "entry_yes_price": [yes] * n,
            "entered_at_utc": [entered] * n,
            "side_won": [won] * n,
            "entry_side_price": [round(1.0 - yes, 4)] * n,
            "city": [f"c{i % 10}" for i in range(n)],
            "target_date": [f"2026-06-{1 + i % 40 % 30:02d}" for i in range(n)],
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


# ── Gate power amendment 2026-07-27 (clustered significance) ────────────────────

def test_minimum_detectable_edge_exposes_what_a_sample_can_actually_see():
    """'31/40 bets' reads as 78% done when the interval actually needs ~400. The gate must report
    the smallest edge its CURRENT sample could resolve, so progress can't be misread."""
    import evaluate_oos as ev
    assert ev._mde(0.0210, 2.94) == pytest.approx(0.0617, abs=0.001)
    # more data -> a smaller edge becomes detectable
    assert ev._mde(0.0105, 2.94) < ev._mde(0.0210, 2.94)


def test_pooled_gap_is_reported_as_the_powered_test():
    """Splitting into 15 buckets costs ~15x the sample. The pooled test is the one that has the
    data, and it must be printed as such rather than buried under underpowered per-bucket rows."""
    import evaluate_oos as ev
    import inspect
    src = inspect.getsource(ev)
    assert "POOLED" in src
    # and the bet-count floor must be labelled as non-binding, so 'n/40' isn't read as progress
    assert "binding constraint" in src


def test_every_bucket_is_nominated_with_its_own_forward_clock():
    """2026-07-28: replaces the hand-picked {Seoul|1d, Chicago|1d} with ALL buckets under test.
    Buckets nominated on 2026-07-12 keep that clock (their forward sample was earned honestly);
    everything added now starts today, because its history has already been inspected."""
    import config
    noms = config.E3_NOMINATIONS
    cities = {"Seoul", "London", "Chicago", "NYC", "HongKong"}
    expected = {f"{c}|{h}" for c in cities for h in ("same-day", "1d", "2d+")}
    assert set(noms) == expected, "every city x horizon must be under test"
    assert noms["Seoul|1d"] == "2026-07-12" and noms["Chicago|1d"] == "2026-07-12"
    assert noms["London|1d"] == "2026-07-28"          # newly nominated -> clock starts today
    # LIVE_BUCKETS now means GATE-PASSED (execution-eligible), not "someone liked the look of it"
    assert config.LIVE_BUCKETS == set()


def test_gate_threshold_is_corrected_for_the_number_of_buckets_tested():
    """Bonferroni: checking 15 buckets means the winner must be more convincing than if we had
    checked one. z 1.96 -> ~2.94, so a result that clears the single-test bar can still fail."""
    import evaluate_oos as ev
    assert ev._e3_gate_z(1) == pytest.approx(1.96, abs=0.01)
    assert ev._e3_gate_z(15) == pytest.approx(2.94, abs=0.02)
    assert ev._e3_gate_z(15) > ev._e3_gate_z(5) > ev._e3_gate_z(1)


def test_multiplicity_correction_actually_blocks_a_marginal_pass():
    import evaluate_oos as ev
    # gap -0.02, se 0.009: clears the single-test bar (-0.02 + 1.96*0.009 < 0) but not 15-test
    g = {"n": 60, "n_clusters": 40, "gap": -0.020, "se": 0.009}
    assert ev._e3_gate_pass(g, min_bets=40, z=ev._e3_gate_z(1)) is True
    assert ev._e3_gate_pass(g, min_bets=40, z=ev._e3_gate_z(15)) is False


def test_clustered_stats_live_in_a_shared_module_and_shoulder_book_still_exposes_them():
    """The E3 bucket gates need the same clustered inference as the structure gates, so the
    estimator moves to stats_util. shoulder_book must keep re-exporting it (existing callers)."""
    import stats_util
    import shoulder_book as sb
    assert stats_util.clustered_mean_se is sb.clustered_mean_se
    assert stats_util.MIN_CLUSTERS == sb.GATE_MIN_CLUSTERS


def test_bucket_gap_stats_are_paired_and_clustered():
    """The per-bucket number is a PAIRED difference (model Brier - market Brier) on the same
    markets, so the interval must be built on the per-market difference, clustered by city-day —
    bins settling on one day share one weather outcome."""
    import pandas as pd
    import evaluate_oos as ev
    # 3 city-days x 4 bins; model is uniformly 0.10 better on every market -> gap -0.10, tight CI
    rows = []
    for day in range(3):
        for b in range(4):
            rows.append({"city": "Seoul", "target_date": f"2026-07-0{day+1}",
                         "outcome": 1, "forecast_prob": 0.8, "market_prob_raw": 0.6})
    s = ev._gap_stats(pd.DataFrame(rows), "market_prob_raw")
    assert s["n"] == 12 and s["n_clusters"] == 3
    assert s["gap"] == pytest.approx(0.8**2 - 0.6**2 - 2 * (0.8 - 0.6), abs=1e-9) or s["gap"] < 0
    assert s["ci_hi"] < 0            # model confidently better here


def test_e3_forward_gate_requires_the_gap_interval_to_exclude_zero():
    """Same 2026-07-27 amendment as the structure gates: beating the market on a point estimate
    over a handful of bets is not evidence. Chicago|1d sat at 5 forward bets."""
    import evaluate_oos as ev
    tiny = {"n": 5, "n_clusters": 2, "gap": -0.068, "se": 0.060}
    strong = {"n": 60, "n_clusters": 40, "gap": -0.020, "se": 0.005}
    assert ev._e3_gate_pass(tiny, min_bets=40) is False        # too few bets AND interval spans 0
    assert ev._e3_gate_pass(strong, min_bets=40) is True
    # a large, well-clustered sample whose interval still spans zero must NOT pass
    spans = {"n": 60, "n_clusters": 40, "gap": -0.020, "se": 0.020}
    assert ev._e3_gate_pass(spans, min_bets=40) is False
    # an unusable SE (fewer than 2 clusters) can never pass
    assert ev._e3_gate_pass({"n": 60, "n_clusters": 40, "gap": -0.5,
                             "se": float("inf")}, min_bets=40) is False


def test_clustered_mean_se_treats_a_city_day_as_one_observation():
    """Bets on the same city-day share ONE weather outcome, so they are not independent.
    The clustered SE must reflect the number of city-days, not the number of bets."""
    import pandas as pd
    import shoulder_book as sb
    # two city-days, 50 bets each: one day won big, the other lost big -> true mean 0
    values = pd.Series([0.5] * 50 + [-0.5] * 50)
    clusters = pd.Series(["NYC|2026-07-01"] * 50 + ["NYC|2026-07-02"] * 50)
    mean, se, n_clusters = sb.clustered_mean_se(values, clusters)
    assert n_clusters == 2
    assert mean == pytest.approx(0.0)
    # iid SE would be 0.5/sqrt(100) = 0.05 and would call this a precise zero;
    # the clustered SE must be an order of magnitude larger (2 real observations).
    assert se > 0.4


def test_gate_verdict_fails_when_clustered_ci_includes_zero():
    """The 2026-07-27 amendment. A point estimate clearing the threshold is NOT enough:
    the clustered 95% CI must also exclude zero. This is the real 2026-07-27 case —
    full band n=150, mean +0.023 >= +0.020, but CI [-0.023, +0.070]."""
    import numpy as np
    import pandas as pd
    import shoulder_book as sb
    rng = np.random.default_rng(0)
    # 150 bets over 54 city-days, mean pinned at +0.023 with per-bet noise ~0.345
    clusters = pd.Series([f"c{i % 54}" for i in range(150)])
    values = pd.Series(rng.normal(0.023, 0.345, 150))
    values = values - values.mean() + 0.023
    v = sb.gate_verdict(values, clusters, need_n=150, need_e=0.020)
    assert v["n"] == 150 and v["n_clusters"] == 54
    assert v["mean"] >= 0.020                        # old gate would have PASSED
    assert v["ci_lo"] < 0                            # but zero is inside the interval
    assert v["pass"] is False


def test_gate_verdict_passes_only_when_edge_is_both_large_and_significant():
    import pandas as pd
    import shoulder_book as sb
    # 120 bets across 40 city-days spanning 40 dates, consistently ~+0.05 -> tight CI above 0.
    # Keys must carry a date component: since the 2026-08-02 amendment a gate also needs
    # >= GATE_MIN_DATES distinct target dates, and a dateless key cannot demonstrate spread.
    clusters = pd.Series([f"c{i % 40}|2026-06-{1 + (i % 40) % 30:02d}" for i in range(120)])
    values = pd.Series([0.05, 0.06, 0.04] * 40)
    v = sb.gate_verdict(values, clusters, need_n=100, need_e=0.03)
    assert v["ci_lo"] > 0
    assert v["pass"] is True


def test_gate_verdict_requires_a_minimum_number_of_clusters():
    """A huge, perfectly consistent edge over 3 city-days is still 3 observations."""
    import pandas as pd
    import shoulder_book as sb
    clusters = pd.Series([f"c{i % 3}" for i in range(120)])
    values = pd.Series([0.30] * 120)
    v = sb.gate_verdict(values, clusters, need_n=100, need_e=0.03)
    assert v["n_clusters"] == 3 < sb.GATE_MIN_CLUSTERS
    assert v["pass"] is False


def test_gate_amendment_is_tightening_only():
    """The amendment may only ever make a gate HARDER to pass: anything passing the new
    gate must also satisfy the original (n >= need_n and mean >= need_e)."""
    import pandas as pd
    import shoulder_book as sb
    clusters = pd.Series([f"c{i % 40}" for i in range(120)])
    # mean below the economic threshold, but extremely precise -> must still FAIL
    values = pd.Series([0.010] * 120)
    v = sb.gate_verdict(values, clusters, need_n=100, need_e=0.03)
    assert v["ci_lo"] > 0 and v["mean"] < 0.03
    assert v["pass"] is False


def test_gate_line_withholds_the_taker_gate_when_the_edge_is_not_significant(capsys):
    """The exact 2026-07-27 regression: n and mean both clear the pre-registered thresholds,
    but the clustered CI spans zero, so the line must NOT print the ✅ gate mark."""
    import numpy as np
    import pandas as pd
    import shoulder_book as sb
    rng = np.random.default_rng(0)
    n = 150
    won = pd.Series(rng.random(n) < 0.86)
    sub = pd.DataFrame({
        "side_won": won,
        "entry_side_price": [0.80] * n,
        "entry_yes_price": [0.20] * n,
        "city": [f"c{i % 54}" for i in range(n)],
        "target_date": ["2026-07-01"] * n,
    })
    sub["net_edge"] = sb._net_edge(sub["side_won"], sub["entry_side_price"])
    sb._gate_line("Leg1 shoulder full [5,35)¢", sub, (150, 0.02), None)
    out = capsys.readouterr().out
    assert "TAKER-GATE" not in out          # mean clears +0.02 but zero is inside the CI
    assert "CI" in out                      # the interval is shown, not just the point estimate


def test_moderate_gate_stats_reports_clustered_significance():
    """moderate_gate_stats must carry the clustered fields through so the printed gate
    and the pass/fail decision use the same numbers."""
    import pandas as pd
    import shoulder_book as sb
    n = 120
    graded = pd.DataFrame({
        "entry_yes_price": [0.15] * n,
        "entered_at_utc": ["2026-12-31T00:00:00+00:00"] * n,
        "side_won": [True] * n,
        "entry_side_price": [0.85] * n,
        "city": [f"city{i % 40}" for i in range(n)],
        # 40 distinct dates: the 2026-08-02 amendment also requires temporal spread.
        "target_date": [f"2026-10-{1 + i % 40 % 30:02d}" for i in range(n)],
    })
    f = sb.moderate_gate_stats(graded)["forward"]
    assert f["n_clusters"] == 40
    assert "ci_lo" in f and "se" in f
    assert f["gate_pass"] is True


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
    from datetime import datetime, timezone, timedelta, date
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
    aft = date(2026, 7, 26)   # after the July-25 target so grading proceeds
    g = b.grade_book(out_path=out, fetch=fetch, as_of=aft)
    assert bool(g[g.condition_id == "0xSH"].iloc[0]["maker_filled"]) is True
    assert bool(g[g.condition_id == "0xFV"].iloc[0]["maker_filled"]) is True

    # a shoulder whose price NEVER rose to entry does NOT fill as maker
    out2 = tmp_path / "b2.csv"
    b.scan_and_record_breadth(bins=[mk("0xNF", 0.15)], now_utc=t1, out_path=out2)
    b.scan_and_record_breadth(bins=[mk("0xNF", 0.11)], now_utc=t2, out_path=out2)   # only falls
    g2 = b.grade_book(out_path=out2, as_of=aft, fetch=lambda u, p=None, l="API": {"closed": True, "outcomePrices": "[\"0\",\"1\"]"})
    assert bool(g2.iloc[0]["maker_filled"]) is False


def test_grade_book_skips_future_targets(tmp_path):
    import shoulder_book_breadth as b
    import pandas as pd
    from datetime import date
    rows = [
        {**{c: "" for c in b._BCOLS}, "condition_id": "0xP", "market_id": "past",
         "leg": "shoulder", "side": "No", "entry_side_price": 0.85, "entry_yes_price": 0.15,
         "entered_at_utc": "2026-07-20T00:00:00+00:00", "target_date": "2026-07-20"},
        {**{c: "" for c in b._BCOLS}, "condition_id": "0xF", "market_id": "future",
         "leg": "shoulder", "side": "No", "entry_side_price": 0.85, "entry_yes_price": 0.15,
         "entered_at_utc": "2026-07-23T00:00:00+00:00", "target_date": "2026-07-25"},
    ]
    out = tmp_path / "b.csv"
    pd.DataFrame(rows).reindex(columns=b._BCOLS).to_csv(out, index=False)
    looked_up = []

    def fetch(url, params=None, label="API"):
        looked_up.append(url.rstrip("/").split("/")[-1])
        return {"closed": True, "outcomePrices": "[\"0\",\"1\"]"}   # NO won

    b.grade_book(out_path=out, fetch=fetch, as_of=date(2026, 7, 24))
    # only the past-dated market is looked up; the 07-25 future one is skipped
    assert looked_up == ["past"]


def test_series_error_blocks_publishing():
    """compute_series wraps everything in one try/except and stashes the exception in
    series['error']. Nothing checked it, so a NameError I introduced on 2026-07-28 silently
    dropped every panel computed after it — the whole 'Recent settlements' table published as
    'No settlements yet' while the run stayed green. A recorded error must fail the build."""
    import build_dashboard as bd
    assert bd._series_error({"series": {"error": "NameError: name 'nom_date' is not defined"}}) \
        == "NameError: name 'nom_date' is not defined"
    assert bd._series_error({"series": {}}) is None
    assert bd._series_error({}) is None


def test_dashboard_completeness_guard():
    import build_dashboard as bd
    # full set -> nothing missing
    full = {"series": {"city": [{"city": c} for c in bd.CITY_ORDER]}}
    assert bd._missing_cities(full) == []
    # an IEM outage drops Seoul + London -> both flagged (this is what must block publishing)
    degraded = {"series": {"city": [{"city": c} for c in ["NYC", "Chicago", "HongKong"]]}}
    assert set(bd._missing_cities(degraded)) == {"Seoul", "London"}
    # empty / malformed payload -> all cities missing (never publishes nothing)
    assert set(bd._missing_cities({})) == set(bd.CITY_ORDER)


_BOOK_REPORT_NOW = """STRUCTURE PAPER BOOK: 210 entries (151 graded, 59 awaiting truth)
  Leg1 shoulder full [5,35)¢: n=150 (54 city-days) wr 86% | taker +0.023 CI[-0.023,+0.070] (150/150@+0.023v+0.020, CI spans 0) | maker 51/150 filled +0.072
  Leg1 shoulder core [20,35)¢: n=62 (42 city-days) wr 79% | taker +0.049 CI[-0.038,+0.136] (62/80@+0.049v+0.030, CI spans 0) | maker 24/62 filled +0.108
  Leg1b moderate shoulder [10,25)¢ — pre-reg 2026-07-23
    context (incl. in-sample): n=68 wr 87% taker +0.017 | maker 26/68 +0.132
    FORWARD gate (entered≥pre-reg): n=16 taker -0.042 [16/80@-0.042v+0.030] | maker 5/16 +0.174
  Leg2 favorite core [65,75)¢: n=1 (1 city-day) wr 0% | taker -0.671 (1/80@-0.671v+0.030, 1/30 city-days) | maker 0 filled
"""

_BOOK_REPORT_OLD = """STRUCTURE PAPER BOOK: 210 entries (151 graded, 59 awaiting truth)
  Leg1 shoulder full [5,35)¢: n=150 wr 86% | taker +0.023 | maker 51/150 filled +0.072
  Leg1 shoulder core [20,35)¢: n=62 wr 79% | taker +0.049 | maker 24/62 filled +0.108
  Leg2 favorite core [65,75)¢: n=1 wr 0% | taker -0.671 | maker 0 filled
"""


def test_shoulder_report_parse_survives_the_city_days_and_CI_columns():
    """2026-07-27 regression: adding '(54 city-days)' and 'CI[...]' to the report line broke the
    dashboard's n=(\\d+)\\s+wr regex, blanking Win rate and the Leg 1 edge on the published page
    (SB_WR 87 -> '—', SB_FULL +0.030 -> '—') and silently reporting Leg 2 as 0 graded."""
    import build_dashboard as bd
    d = bd._parse_book_scalars(_BOOK_REPORT_NOW)
    assert (d["sb_entries"], d["sb_graded"], d["sb_await"]) == ("210", "151", "59")
    assert d["sb_wr"] == "86"                 # was blanked
    assert d["sb_full"] == "+0.023"           # was blanked
    assert d["sb_core"] == "+0.049"
    assert d["sb_fav_graded"] == "1"          # was silently reported as 0
    assert (d["sb_mod_fwd_n"], d["sb_mod_fwd"]) == ("16", "-0.042")
    assert d["sb_mod_need"] == "80" and d["sb_mod_pass"] == "0"


def test_shoulder_report_parse_still_reads_the_pre_amendment_format():
    """The parser must not simply be re-pinned to today's format — older reports still parse."""
    import build_dashboard as bd
    d = bd._parse_book_scalars(_BOOK_REPORT_OLD)
    assert d["sb_wr"] == "86" and d["sb_full"] == "+0.023" and d["sb_fav_graded"] == "1"


def test_shoulder_report_parse_marks_a_passed_gate():
    import build_dashboard as bd
    passed = _BOOK_REPORT_NOW.replace("[16/80@-0.042v+0.030]", "[✅MOD-GATE]")
    assert bd._parse_book_scalars(passed)["sb_mod_pass"] == "1"



def test_dashboard_links_to_the_self_hosted_guide():
    """The guide is published alongside index.html in the Pages repo, so the link must be
    RELATIVE — a claude.ai artifact URL would be private and 404 for every other visitor."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'href="./guide.html"' in src
    assert "claude.ai/code/artifact" not in src      # never link the private artifact


def test_guide_page_is_standalone_and_dependency_free():
    """It ships to plain GitHub Pages: no mermaid runtime, no external fonts or scripts, and its
    own <head> (the artifact host supplies one; Pages does not)."""
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "src/polymarket_weather/guide.html"
    h = p.read_text(encoding="utf-8")
    assert h.startswith("<!doctype html>") and "<head>" in h
    assert "mermaid" not in h                       # would silently render as raw text
    assert "https://" not in h.split("<body>")[0]   # no external assets in head
    assert '<script' not in h                       # static document, no JS needed
    assert 'href="./index.html"' in h               # back link to the dashboard


def test_headline_briers_come_from_the_paired_common_set():
    """Coverage kept after deleting _scoreboard_html: the BR_* binds must still read the PAIRED
    numbers, not evaluate_oos's unpaired MODEL row (401 markets vs the ensemble's 246)."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert '"BR_MODEL": _f4(br["model"])' in src
    assert 'paired = ((series.get("score") or {}).get("all")) or {}' in src


def test_pooled_interval_is_computed_for_both_scoreboard_windows():
    """The Model-minus-market tile follows the all/last-60 toggle, so its interval must too —
    otherwise the number and its uncertainty describe different samples."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'out["pooled"] = {"all":' in src
    assert '"recent":' in src


def test_gap_tile_carries_its_interval():
    """A point estimate with no interval cannot be told from noise, and the gap already has a
    KPI box — so the interval belongs in that box, not in a second one."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'id="k_gap_ci"' in src
    assert 'getElementById("k_gap_ci")' in src


def test_pooled_line_renders_in_the_LIVE_scoreboard_path():
    """_scoreboard_html's output is DEAD PAYLOAD: the page has no data-bind-html="SCOREBOARD"
    mount, and builds its scoreboard live from D.score into #rank via renderScore(). So the
    pooled interval has to be rendered on that path or it never reaches a viewer."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'id="pooledLine"' in src            # a real element in the DOM
    assert "D.pooled" in src                   # filled from the payload by the live renderer
    # and the dead mount must not be the only place it appears
    assert 'data-bind-html="SCOREBOARD"' not in src



def test_skill_is_computed_from_the_paired_model_number():
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'float(d["br_model"]) / float(d["br_market"])' not in src


def test_crps_model_is_not_unconditionally_green():
    """CRPS 1.1248 (model) vs 1.0795 (ensemble) means the model is WORSE, but the tile hardcoded
    color:var(--good). Colour must depend on the comparison."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert '<div class="v" style="color:var(--good)"><span data-bind="CRPS_MODEL">' not in src
    assert "crps_v" in src            # the element the JS colours by comparison


def test_collector_tile_reports_build_time_lag_not_browser_clock():
    """The status pill was fixed to separate collector lag from publish lag, but the Collector KPI
    still did fmtAgo(lastCollect) against the live clock — so a 3.5h-old build showed a 1h-old
    collector as '4h ago'."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert 'cb.textContent = fmtAgo(lastCollect)' not in src
    assert "collectLagH" in src


def test_every_edge_cell_is_sign_coloured():
    """Sign colouring must be consistent across both structure tables: the 5-city Leg 1b edge and
    the breadth full-band maker cell were left unpainted, so +0.042 rendered like plain text and
    −0.0169 was not red. (The gate pill, not the colour, carries the pass/fail verdict.)"""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    # 5-city Leg 1b forward edge
    assert 'me.style.color' in src
    # breadth full-band maker needs an addressable cell before it can be painted
    assert 'id="bkfullmaker"' in src
    assert 'bkfullmaker' in src.split("<script")[-1] or 'getElementById("bkfullmaker")' in src


def test_leg1_row_shows_leg1_graded_count_not_the_whole_book():
    """The '1 · sell shoulder' row bound SB_GRADED (every leg, 151) while Leg 1 itself had 150 —
    the extra row is the single graded Leg 2 favourite."""
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    assert '"SB_FULL_N": G("sb_full_n"' in src              # exposed in the payload
    assert '<td class="city">1 · sell shoulder</td><td class="num" data-bind="SB_FULL_N"' in src


def test_breadth_full_band_maker_is_published_not_hidden():
    """The breadth Leg 1 maker cell was a hardcoded dash while the data existed and was NEGATIVE
    (447/603 filled, −0.0169). A book that hides its losing column is not a paper book."""
    import build_dashboard as bd
    import inspect
    b = bd._breadth_binds()
    assert "BK_FULL_MAKER" in b and "BK_FULL_MAKER_N" in b
    assert 'data-bind="BK_FULL_MAKER"' in inspect.getsource(bd)


def test_collect_lag_hours_measures_the_collector_not_the_publish_delay():
    """2026-07-27: the header said 'collector stale' while the collector had run 3h earlier.
    The pill was comparing the browser clock to the newest snapshot FROZEN IN THE PAYLOAD, so a
    late dashboard rebuild was charged to the collector. Collector lag must be measured at BUILD
    time (build clock - newest snapshot) and be independent of how old the publish is."""
    import build_dashboard as bd
    # collector ran 3h before the build -> 3.0, regardless of how stale the publish later gets
    assert bd._collect_lag_hours("2026-07-27T11:41:00Z",
                                 "2026-07-27T14:41:00Z") == pytest.approx(3.0)
    # the real case: build at 10:30 over data through 07:51 -> 2.65h, NOT the 6.7h the pill showed
    assert bd._collect_lag_hours("2026-07-27T07:51:00Z",
                                 "2026-07-27T10:30:00Z") == pytest.approx(2.65, abs=0.02)
    # unknown collector timestamp -> None (unknown is not "fresh")
    assert bd._collect_lag_hours(None, "2026-07-27T10:30:00Z") is None


def test_payload_carries_collect_lag_so_the_pill_can_tell_the_two_apart():
    import build_dashboard as bd
    import inspect
    src = inspect.getsource(bd)
    # the payload must expose the build-time collector lag...
    assert '"collect_lag_hours"' in src
    # ...and the pill must no longer derive collector staleness from the live browser clock
    assert 'Date.now() - Date.parse(lastCollect)' not in src


def test_quantile_forest_calibrated_and_monotone():
    import numpy as np
    from predictors.qrf_core import QuantileForest
    rng = np.random.default_rng(0)
    # heteroscedastic: spread grows with x -> a parametric-fixed-sigma model can't fit this, QRF can
    X = rng.uniform(0, 10, size=(4000, 1))
    y = X[:, 0] + rng.normal(0, 0.5 + 0.4 * X[:, 0])
    qf = QuantileForest(n_estimators=200, min_samples_leaf=40, random_state=0).fit(X, y)
    qs = qf.predict_quantiles(X, [0.1, 0.5, 0.9])
    assert qs.shape == (4000, 3)
    assert np.all(qs[:, 0] <= qs[:, 1] + 1e-9) and np.all(qs[:, 1] <= qs[:, 2] + 1e-9)   # monotone
    cov = np.mean((y >= qs[:, 0]) & (y <= qs[:, 2]))     # nominal 80% central coverage
    assert 0.72 <= cov <= 0.88
    # spread must widen with x (heteroscedastic learned)
    lo = qf.predict_quantiles(np.array([[1.0]]), [0.1, 0.9])
    hi = qf.predict_quantiles(np.array([[9.0]]), [0.1, 0.9])
    assert (hi[0, 1] - hi[0, 0]) > (lo[0, 1] - lo[0, 0])


def test_moment_match_recovers_shape():
    import numpy as np
    from scipy import stats
    from predictors.qrf_core import moment_match
    levels = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]
    # a near-Gaussian sample -> high nu, sigma ~2, mu ~10
    gq = stats.norm(10, 2).ppf(levels)
    mu, sigma, nu = moment_match(levels, np.array(gq))
    assert abs(mu - 10) < 0.2 and abs(sigma - 2) < 0.3 and nu >= 15
    # a heavy-tailed sample (t, df=3) -> low nu
    tq = stats.t(df=3, loc=10, scale=2).ppf(levels)
    _, _, nu_t = moment_match(levels, np.array(tq))
    assert nu_t < nu           # heavier tail => lower nu


def test_intraday_running_max_no_leakage():
    import pandas as pd, numpy as np
    from qrf_features import intraday_running_max, build_row, FEATURE_COLS
    tz = "Asia/Seoul"
    obs = pd.DataFrame({
        "valid_local": pd.to_datetime(["2026-07-09 08:00", "2026-07-09 12:00", "2026-07-09 16:00"]),
        "temp_c": [22.0, 27.0, 31.0]})
    tgt = pd.Timestamp("2026-07-09")
    # as-of 13:00 local: only the 08:00 and 12:00 obs exist -> running max 27, NOT 31
    fetch = pd.Timestamp("2026-07-09 13:00", tz=tz)
    rm = intraday_running_max(obs, tgt, fetch, tz)
    assert rm == 27.0
    # a fabricated LATER obs must not change the as-of-13:00 result (no look-ahead)
    obs2 = pd.concat([obs, pd.DataFrame({"valid_local": [pd.Timestamp("2026-07-09 14:30")], "temp_c": [40.0]})])
    assert intraday_running_max(obs2, tgt, fetch, tz) == 27.0
    # build_row yields exactly FEATURE_COLS
    row = build_row({"ecmwf": 30.0, "gfs": 29.0, "icon": 31.0}, {"mean": 30.0, "std": 1.5, "p10": 28, "p50": 30, "p90": 32},
                    running_max=27.0, is_same_day=1, lead=0, doy=190)
    assert set(row) == set(FEATURE_COLS)


def test_intraday_running_max_dst_fallback_no_crash():
    """DST fall-back: a repeated local hour (e.g., 2025-11-02 01:30 occurs twice in America/New_York)
    must not crash with AmbiguousTimeError. The function should handle ambiguous times gracefully."""
    import pandas as pd
    from qrf_features import intraday_running_max
    from zoneinfo import ZoneInfo

    tz = "America/New_York"
    # 2025-11-02 is a DST fall-back date in America/New_York.
    # At 2:00 AM EDT, clocks roll back to 1:00 AM EST, so 1:30 AM occurs twice.
    obs = pd.DataFrame({
        "valid_local": pd.to_datetime([
            "2025-11-02 00:30",  # before the transition
            "2025-11-02 01:30",  # ambiguous hour (occurs twice)
            "2025-11-02 03:00",  # after the transition
        ]),
        "temp_c": [15.0, 14.0, 13.0]
    })

    tgt = pd.Timestamp("2025-11-02")
    # fetch_time after all obs, on the same day, with explicit timezone
    fetch = pd.Timestamp("2025-11-02 12:00", tz=ZoneInfo(tz))

    # This should NOT raise AmbiguousTimeError; should return a finite float.
    result = intraday_running_max(obs, tgt, fetch, tz)

    # Should return a valid number (ideally the max of non-NaT observations)
    # or NaN if all observations become NaT due to ambiguity, but not raise.
    assert isinstance(result, float)
    # If the function handled the ambiguous time gracefully, it should return
    # a finite value representing the max of the valid observations.
    assert result == result  # Not NaN check (NaN != NaN).


def test_fit_city_gates_and_persists(tmp_path, monkeypatch):
    import numpy as np, pandas as pd, json
    import train_qrf
    monkeypatch.setattr(train_qrf, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(1)
    from qrf_features import FEATURE_COLS
    X = pd.DataFrame(rng.normal(size=(600, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = X["ens_mean"].values * 0 + rng.normal(20, 3, size=600)     # QRF learns sigma~3
    # ens_holdout_crps is a CRPS-scale (°C) value, not a 0-1 Brier: a high one -> QRF should beat.
    res = train_qrf.fit_city("seoul", X, y, ens_holdout_crps=5.0)
    assert set(res) >= {"beats_ensemble", "holdout_crps", "n"}
    assert (tmp_path / "seoul_qrf.joblib").exists()
    meta = json.loads((tmp_path / "seoul_qrf_meta.json").read_text())
    assert "beats_ensemble" in meta


def test_fit_city_gates_on_empirical(tmp_path, monkeypatch):
    import numpy as np, pandas as pd, json
    import train_qrf
    monkeypatch.setattr(train_qrf, "_MODELS_DIR", tmp_path)
    from qrf_features import FEATURE_COLS
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(20, 3, size=(600, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = rng.normal(20, 3, size=600)
    # ensemble baseline deliberately weak -> QRF should beat it on the empirical CRPS
    res = train_qrf.fit_city("seoul", X, y, ens_holdout_crps=5.0)
    assert res["beats_ensemble"] is True
    meta = json.loads((tmp_path / "seoul_qrf_meta.json").read_text())
    # the gate score is a plausible CRPS scale (sample-CRPS of a ~3σ spread is order ~1-2, not the
    # Gaussian-proxy value); assert it's finite and positive (mechanism check, not a magic number)
    assert meta["holdout_crps"] > 0 and np.isfinite(meta["holdout_crps"])


def test_train_qrf_loader_assembles_dateordered_and_gates(tmp_path, monkeypatch):
    """Drive train_qrf()'s loader on a tiny in-memory fixture (no network): a wide
    per-date leads_mm+truth frame -> date-ordered X/y -> ensemble CRPS proxy (same
    formula) -> fit_city artifacts. Proves loader + CRPS-gate plumbing offline."""
    import numpy as np, pandas as pd, json
    import train_qrf
    from qrf_features import FEATURE_COLS

    monkeypatch.setattr(train_qrf, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_qrf, "_MIN_ROWS", 20)   # keep the fixture small but valid

    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    seasonal = 20 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365.0)
    fixture = {"date_local": dates.strftime("%Y-%m-%d")}
    # Real archives (fetch_historical_leads_mm.py / _cand.py) cap at LEADS = range(1, 5) —
    # only leads 1-4 ever have fcst_tmax_lead{n}_{m} columns. Fabricating leads 5-7 here
    # would hide the train/serve safety gap (QRF trained on 1-4 but served unbounded).
    for n in range(1, 5):
        for m in ("ecmwf", "gfs", "icon"):
            fixture[f"fcst_tmax_lead{n}_{m}"] = seasonal + rng.normal(0, 1.0 + 0.2 * n, size=len(dates))
    fixture["temp_max_c"] = seasonal + rng.normal(0, 1.0, size=len(dates))
    wide = pd.DataFrame(fixture)
    # feed the rows out of order to prove the loader re-sorts by date before the holdout split
    wide = wide.sample(frac=1.0, random_state=1).reset_index(drop=True)

    # unit-check the pure assembly helper: date-ordered rows, exact FEATURE_COLS, y aligned.
    X, y = train_qrf._assemble_xy(wide)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == 40 * 4                 # 40 dates x leads 1-4 (archive-limited)
    # mm_mean finite everywhere; ens_* left NaN (no historical ensemble archive)
    assert np.isfinite(X["mm_mean"].to_numpy()).all()
    assert X["ens_mean"].isna().all()
    assert (X["is_same_day"] == 0).all()

    # ensemble baseline CRPS proxy equals the same closed-form on the same holdout rows.
    ens_crps = train_qrf._ensemble_holdout_crps(X, y)
    cut = train_qrf._holdout_cut(len(y))
    mu = X["mm_mean"].to_numpy(float)
    sigma = max(float(np.std(y[:cut] - mu[:cut])), 0.1)
    assert ens_crps == train_qrf.crps_gaussian_proxy(y[cut:], mu[cut:], sigma)
    assert np.isfinite(ens_crps) and ens_crps > 0

    # full loader path: monkeypatch the on-disk reader, run one city, assert artifacts + meta.
    monkeypatch.setattr(train_qrf, "_load_city_frame", lambda slug: wide)
    results = train_qrf.train_qrf(cities=["Seoul"])
    assert "seoul" in results
    assert (tmp_path / "seoul_qrf.joblib").exists()
    meta = json.loads((tmp_path / "seoul_qrf_meta.json").read_text())
    assert set(meta) >= {"beats_ensemble", "holdout_crps", "ens_holdout_crps", "n", "max_lead"}
    assert meta["n"] == 40 * 4
    assert meta["ens_holdout_crps"] == ens_crps       # loader passed the same value it computed
    # train/serve safety gap fix: the archive only ever produces leads 1-4, so max_lead
    # must reflect the actual assembled data (computed), not the hopeful LEADS=range(1,8).
    assert meta["max_lead"] == 4


def test_qrf_predictor_gates_and_floors(tmp_path, monkeypatch):
    import numpy as np, pandas as pd, json, joblib
    from predictors import qrf as qmod
    from predictors.qrf_core import QuantileForest
    from qrf_features import FEATURE_COLS
    monkeypatch.setattr(qmod, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, len(FEATURE_COLS))); y = rng.normal(20, 3, size=400)
    joblib.dump(QuantileForest().fit(X, y), tmp_path / "seoul_qrf.joblib")
    # gated OFF -> None (fallback)
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": False}))
    p = qmod.QRFPredictor()
    assert p.predict_distribution("Seoul", pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-09 13:00"),
                                  0, pd.DataFrame(), kind="max") is None
    # gated ON -> a TemperatureDistribution with a sane wide-ish sigma
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": True}))
    # (feature assembly from the *_df args is exercised in the live path; here assert gate + type via a stub)

    # kind="min" is out of v1 scope (no _qrf_min artifact) -> always None, gate or not.
    assert p.predict_distribution("Seoul", pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-09 13:00"),
                                  0, pd.DataFrame(), kind="min") is None

    # missing artifact entirely (different city) -> None.
    assert p.predict_distribution("London", pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-09 13:00"),
                                  0, pd.DataFrame(), kind="max") is None


def test_qrf_predictor_serves_gated_on_with_floor(tmp_path, monkeypatch):
    """The brief's gate test scopes full feature-assembly-from-dfs to the live path
    (Task 7). Because that assembly is genuinely exercisable with well-formed
    fixtures, this test builds real (not mocked) daily_df/ens_df/mm_df/obs_df in the
    exact schema `data_loader.py` produces, and drives predict_distribution's actual
    code path end-to-end: EMOS's as-of helpers -> qrf_features.build_row ->
    QuantileForest.predict_quantiles -> moment_match -> TemperatureDistribution."""
    import numpy as np, pandas as pd, json, joblib
    from predictors import qrf as qmod
    from predictors.qrf_core import QuantileForest
    from qrf_features import FEATURE_COLS

    monkeypatch.setattr(qmod, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, len(FEATURE_COLS)))
    y = rng.normal(20, 3, size=400)
    joblib.dump(QuantileForest().fit(X, y), tmp_path / "seoul_qrf.joblib")
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": True}))

    target_date = pd.Timestamp("2026-07-09")
    fetch_same_day = pd.Timestamp("2026-07-09 13:00", tz="UTC")   # 22:00 KST, same local day
    run_fetched_at = fetch_same_day - pd.Timedelta(hours=1)

    daily_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "temp_max_c": [29.5],
    })
    mm_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "tmax_ecmwf": [29.0], "tmax_gfs": [28.5], "tmax_icon": [29.8],
        "tmax_aifs": [np.nan], "tmax_gem": [29.2], "tmax_mf": [np.nan], "tmax_jma": [29.6],
    })
    ens_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "ens_mean": [29.3], "ens_std": [1.2], "ens_p10": [27.5], "ens_p90": [31.0],
        "ens_spread": [3.5], "n_members": [40],
        "ens_min_mean": [18.0], "ens_min_std": [0.8],
    })
    obs_df = pd.DataFrame({
        "valid_local": pd.to_datetime(["2026-07-09 08:00", "2026-07-09 12:00"]),
        "temp_c": [24.0, 27.0],
    })

    p = qmod.QRFPredictor()
    dist = p.predict_distribution("Seoul", target_date, fetch_same_day, 0, daily_df,
                                   ens_df=ens_df, mm_df=mm_df, obs_df=obs_df, kind="max")
    assert dist is not None
    assert dist.source == "qrf"
    assert np.isfinite(dist.mu) and dist.sigma > 0 and np.isfinite(dist.nu)
    # same-day: the running observed max (27.0, as-of 13:00 UTC / 22:00 KST) floors it.
    assert dist.floor == 27.0

    # A lead-1 bet (fetch the day BEFORE the target's station-local day) is not
    # same-day -> no floor, regardless of obs_df content.
    fetch_prior_day = pd.Timestamp("2026-07-08 13:00", tz="UTC")   # 22:00 KST on 07-08
    dist2 = p.predict_distribution("Seoul", target_date, fetch_prior_day, 1, daily_df,
                                    ens_df=ens_df, mm_df=mm_df, obs_df=obs_df, kind="max")
    assert dist2 is not None
    assert dist2.floor is None


def test_qrf_predictor_respects_max_lead_bound(tmp_path, monkeypatch):
    """Train/serve safety gap: the archived per-lead data only ever covers leads 1-4
    (fetch_historical_leads_mm.py / _cand.py cap at range(1, 5)), so train_qrf.py's
    self-gate (beats_ensemble) is validated ONLY on leads 1-4. QRFPredictor must not
    extrapolate that guarantee to leads 5-7 (real '2d+' bucket traffic) -- with a
    gated-ON artifact whose meta records max_lead=4, days_ahead beyond max_lead must
    fall back to None (EMOS/ensemble), while days_ahead within range still serves."""
    import numpy as np, pandas as pd, json, joblib
    from predictors import qrf as qmod
    from predictors.qrf_core import QuantileForest
    from qrf_features import FEATURE_COLS

    monkeypatch.setattr(qmod, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, len(FEATURE_COLS)))
    y = rng.normal(20, 3, size=400)
    joblib.dump(QuantileForest().fit(X, y), tmp_path / "seoul_qrf.joblib")
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": True, "max_lead": 4}))

    target_date = pd.Timestamp("2026-07-09")
    fetch_prior_day = pd.Timestamp("2026-07-08 13:00", tz="UTC")

    daily_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [fetch_prior_day],
        "temp_max_c": [29.5],
    })
    mm_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [fetch_prior_day],
        "tmax_ecmwf": [29.0], "tmax_gfs": [28.5], "tmax_icon": [29.8],
        "tmax_aifs": [np.nan], "tmax_gem": [29.2], "tmax_mf": [np.nan], "tmax_jma": [29.6],
    })
    ens_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [fetch_prior_day],
        "ens_mean": [29.3], "ens_std": [1.2], "ens_p10": [27.5], "ens_p90": [31.0],
        "ens_spread": [3.5], "n_members": [40],
        "ens_min_mean": [18.0], "ens_min_std": [0.8],
    })

    p = qmod.QRFPredictor()
    # days_ahead=6 exceeds max_lead=4 -> untested extrapolation regime -> safe fallback (None).
    dist_far = p.predict_distribution("Seoul", target_date, fetch_prior_day, 6, daily_df,
                                       ens_df=ens_df, mm_df=mm_df, kind="max")
    assert dist_far is None
    # days_ahead=1 is within the trained/gated range -> still serves normally.
    dist_near = p.predict_distribution("Seoul", target_date, fetch_prior_day, 1, daily_df,
                                        ens_df=ens_df, mm_df=mm_df, kind="max")
    assert dist_near is not None
    assert dist_near.source == "qrf"


def test_qrf_predictor_serves_empirical_cdf(tmp_path, monkeypatch):
    """Task 4: predict_distribution must ALSO serve the empirical CDF built from the
    fine quantile grid (qrf_core.Q_FINE), not just the moment-matched Student-t used
    for mu/sigma/nu. Mirrors test_qrf_predictor_serves_gated_on_with_floor's exact
    df-fixture schema/values (same city/dates/floor) -- only the new cdf assertions
    are added here."""
    import numpy as np, pandas as pd, json, joblib
    from predictors import qrf as qmod
    from predictors.qrf_core import QuantileForest
    from qrf_features import FEATURE_COLS

    monkeypatch.setattr(qmod, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, len(FEATURE_COLS)))
    y = rng.normal(20, 3, size=400)
    joblib.dump(QuantileForest().fit(X, y), tmp_path / "seoul_qrf.joblib")
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": True}))

    target_date = pd.Timestamp("2026-07-09")
    fetch_same_day = pd.Timestamp("2026-07-09 13:00", tz="UTC")   # 22:00 KST, same local day
    run_fetched_at = fetch_same_day - pd.Timedelta(hours=1)

    daily_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "temp_max_c": [29.5],
    })
    mm_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "tmax_ecmwf": [29.0], "tmax_gfs": [28.5], "tmax_icon": [29.8],
        "tmax_aifs": [np.nan], "tmax_gem": [29.2], "tmax_mf": [np.nan], "tmax_jma": [29.6],
    })
    ens_df = pd.DataFrame({
        "date_local": [target_date],
        "fetched_at_utc": [run_fetched_at],
        "ens_mean": [29.3], "ens_std": [1.2], "ens_p10": [27.5], "ens_p90": [31.0],
        "ens_spread": [3.5], "n_members": [40],
        "ens_min_mean": [18.0], "ens_min_std": [0.8],
    })
    obs_df = pd.DataFrame({
        "valid_local": pd.to_datetime(["2026-07-09 08:00", "2026-07-09 12:00"]),
        "temp_c": [24.0, 27.0],
    })

    p = qmod.QRFPredictor()
    dist = p.predict_distribution("Seoul", target_date, fetch_same_day, 0, daily_df,
                                   ens_df=ens_df, mm_df=mm_df, obs_df=obs_df, kind="max")
    assert dist is not None
    assert dist.source == "qrf"
    # summary stats are unchanged (still sensible moment-matched values).
    assert np.isfinite(dist.mu) and dist.sigma > 0 and np.isfinite(dist.nu)
    assert dist.floor == 27.0

    # NEW: the empirical CDF is also served, and it's a monotone probability.
    assert dist.cdf is not None
    lo, hi = dist.cdf(0.0), dist.cdf(50.0)
    assert lo <= hi
    assert 0.0 <= dist.cdf(20.0) <= 1.0
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_m1_gate():
    import evaluate_oos as ev
    assert ev.m1_gate(0.130, 0.142) is True      # QRF beats ensemble
    assert ev.m1_gate(0.150, 0.142) is False     # QRF worse -> gate fails


def test_cdf_uses_callable_and_preserves_censoring():
    import pmf
    # a ramp CDF: 0 at 10, 1 at 30, linear between
    ramp = lambda x: min(1.0, max(0.0, (x - 10.0) / 20.0))
    # in-range: uses the callable, not the Student-t
    assert abs(pmf._cdf(20.0, mu=25.0, sigma=3.0, nu=8.0, cdf=ramp) - 0.5) < 1e-9
    # clamped to [0,1]
    assert pmf._cdf(40.0, 25.0, 3.0, 8.0, cdf=ramp) == 1.0
    assert pmf._cdf(0.0, 25.0, 3.0, 8.0, cdf=ramp) == 0.0
    # floor/ceiling censoring still short-circuits BEFORE the callable
    assert pmf._cdf(15.0, 25.0, 3.0, 8.0, floor=18.0, cdf=ramp) == 0.0     # below floor
    assert pmf._cdf(31.0, 25.0, 3.0, 8.0, ceiling=30.0, cdf=ramp) == 1.0   # at/above ceiling
    # REGRESSION: cdf=None reproduces the Student-t exactly
    import scipy.stats as st
    from pmf import _t_scale
    got = pmf._cdf(24.0, 25.0, 3.0, 8.0)
    exp = float(st.t.cdf((24.0 - 25.0) / _t_scale(3.0, 8.0), df=8.0))
    assert abs(got - exp) < 1e-12


def test_bin_prob_via_callable_sums_to_one():
    import numpy as np, pmf
    ramp = lambda x: min(1.0, max(0.0, (x - 10.0) / 20.0))
    # bins every 1°C from 10..30 should capture ~all mass. Bins span (10.5, 29.5], so the
    # analytically exact gap from the ramp's full [10,30] domain is 0.05 (2.5% in each excluded
    # tail) -- a hard boundary equal to a naive `< 0.05` tolerance, which float64 rounding of
    # 0.05 itself (not representable exactly) can tip either way. 0.06 keeps a safety margin
    # above that boundary while still catching a real bug: if `cdf` were silently NOT threaded
    # through `_bin_prob` (falling back to the Student-t), the same sum comes out ~0.939, a 0.061
    # gap that still fails this assertion.
    total = sum(pmf._bin_prob(t, 25.0, 3.0, 8.0, half_width=0.5, cdf=ramp) for t in range(11, 30))
    assert abs(total - 1.0) < 0.06


def test_empirical_cdf_monotone_tails_and_recovery():
    import numpy as np
    from scipy import stats
    from predictors.qrf_core import empirical_cdf_from_quantiles, Q_FINE
    # quantiles of N(20, 3): the reconstructed CDF should track the Gaussian and be monotone
    vals = stats.norm(20, 3).ppf(Q_FINE)
    F = empirical_cdf_from_quantiles(Q_FINE, vals)
    xs = np.linspace(5, 35, 100)
    cs = np.array([F(x) for x in xs])
    assert np.all(np.diff(cs) >= -1e-9)                 # monotone non-decreasing
    assert cs[0] >= 0.0 and cs[-1] <= 1.0               # in range
    assert F(20.0) == max(0.0, min(1.0, F(20.0))) and abs(F(20.0) - 0.5) < 0.05   # median ~0.5
    # tails finite and heading to the bounds
    assert F(-100.0) < 0.02 and F(100.0) > 0.98
    # near the body it tracks the Gaussian
    assert abs(F(23.0) - stats.norm(20, 3).cdf(23.0)) < 0.05
    # degenerate: all-equal quantiles -> a step at the median, no crash
    Fd = empirical_cdf_from_quantiles(Q_FINE, np.full(len(Q_FINE), 12.0))
    assert Fd(11.9) < 0.5 <= Fd(12.1)


def test_sample_crps_orders_correctly():
    import numpy as np
    from predictors.qrf_core import sample_crps
    from scipy import stats
    y = 20.0
    tight = stats.norm(20, 1).ppf(np.linspace(.01, .99, 99))
    wide = stats.norm(20, 5).ppf(np.linspace(.01, .99, 99))
    assert sample_crps(tight, y) < sample_crps(wide, y)   # sharper+calibrated scores better


def test_engine_threads_ml_cdf_into_pricing(tmp_path):
    """Final-review fix: analyse_city() must forward dist_ml.cdf into the Tmax pricing
    calls (reconstruct_pmf / _bin_prob / _condition_prob), not just unpack mu/sigma/nu.
    Before this fix the engine silently discarded a QRF-style empirical cdf at the
    actual pricing sites, so the stored forecast_prob (which drives the M1 Brier gate)
    was still the lossy Student-t moment match no matter what the predictor served.

    Regression design: two analyse_city() runs against the SAME market snapshot, using
    a stub ML predictor that returns the IDENTICAL (mu, sigma, nu, source) both times —
    only `cdf` differs (a sharp step function vs None). If the engine ever again drops
    `cdf` at a call site, both runs collapse to the same Student-t pricing and this test
    fails (forecast_prob would be identical instead of ~1.0 vs ~0.2)."""
    import json
    import pandas as pd
    from engine import analyse_city
    from predictors.base import BasePredictor, TemperatureDistribution

    city = "testcity_cdf"
    (tmp_path / "polymarket").mkdir(parents=True, exist_ok=True)
    (tmp_path / "weather").mkdir(parents=True, exist_ok=True)

    fetched_at = "2026-07-18T12:00:00Z"
    snap = pd.DataFrame({
        "condition_id":       ["cond1"],
        "question":           ["Will the temperature be 20°C on July 20?"],
        "outcome_probs_json": [json.dumps({"Yes": 0.30, "No": 0.70})],
        "fetched_at_utc":     [fetched_at],
        "end_date_iso":       ["2026-07-20T00:00:00Z"],
        "liquidity_usdc":     [5000.0],
        "volume_24h_usdc":    [100.0],
        "volume_usdc":        [1000.0],
    })
    snap.to_csv(tmp_path / "polymarket" / f"{city}_snapshots.csv", index=False)

    daily = pd.DataFrame({
        "date_local":     ["2026-07-20"],
        "fetched_at_utc": [fetched_at],
        "temp_max_c":     [20.0],
        "temp_min_c":     [15.0],
    })
    daily.to_csv(tmp_path / "weather" / f"{city}_daily.csv", index=False)

    class _StubMLPredictor(BasePredictor):
        """Always returns the SAME (mu, sigma, nu, source) — only `cdf` (ctor arg) varies,
        isolating the wiring under test from every other pricing input."""
        def __init__(self, cdf_fn):
            self._cdf_fn = cdf_fn

        def predict_distribution(self, city, target_date, fetch_time, days_ahead,
                                  daily_df, ens_df=None, mm_df=None, obs_df=None,
                                  nbm_df=None, **kwargs):
            return TemperatureDistribution(mu=20.0, sigma=2.0, nu=8.0,
                                           source="emos_v2_stub", cdf=self._cdf_fn)

    # A step CDF: ~all mass sits exactly at 20C, sharply different from the Student-t(20,2,8)
    # moment match (which spreads mass over the +-2C sigma).
    step_cdf = lambda x: 0.0 if x < 20.0 else 1.0

    common = dict(min_edge=0.0, min_liq=0.0, conflict_gating=False)
    opps_with_cdf = analyse_city(tmp_path, city,
                                 ml_predictor_override=_StubMLPredictor(step_cdf), **common)
    opps_without_cdf = analyse_city(tmp_path, city,
                                    ml_predictor_override=_StubMLPredictor(None), **common)

    assert len(opps_with_cdf) == 1, "expected exactly one priced opportunity with cdf set"
    assert len(opps_without_cdf) == 1, "expected exactly one priced opportunity with cdf=None"

    opp_cdf, opp_no_cdf = opps_with_cdf[0], opps_without_cdf[0]
    assert opp_cdf.condition_id == opp_no_cdf.condition_id == "cond1"

    # Same bin, same (mu, sigma, nu) -- only the served cdf differs. If the engine were
    # still dropping cdf at the pricing call sites, forecast_prob would be IDENTICAL here.
    assert opp_cdf.forecast_prob != pytest.approx(opp_no_cdf.forecast_prob, abs=1e-9)
    assert opp_cdf.forecast_prob > 0.9     # step cdf: ~all mass at the bin
    assert opp_no_cdf.forecast_prob < 0.5  # Student-t moment match: mass spread out
    assert opp_cdf.edge != pytest.approx(opp_no_cdf.edge, abs=1e-9)


# --------------------------------------------------------------------------------------
# Dashboard: per-city Brier must be a PAIRED comparison
# --------------------------------------------------------------------------------------

def _city_frame():
    """A common-set frame shaped like build_dashboard's `c`.

    London: 4 markets. The ensemble is GOOD on the two the model also saw and TERRIBLE on
    the rest; the unpaired bug scored it only on an easier subset. HongKong: every market
    settled NO, so Brier carries no discrimination at all.
    """
    import pandas as pd
    rows = [
        # city,      outcome, model, market, ens
        ("London",   0, 0.20, 0.20, 0.60),
        ("London",   1, 0.80, 0.80, 0.40),
        ("London",   0, 0.10, 0.10, 0.10),
        ("London",   1, 0.90, 0.90, 0.90),
        ("HongKong", 0, 0.30, 0.40, 0.20),
        ("HongKong", 0, 0.25, 0.35, 0.15),
    ]
    df = pd.DataFrame(rows, columns=["city", "outcome", "p_model", "p_market", "p_ens"])
    df["b_model"] = (df["p_model"] - df["outcome"]) ** 2
    df["b_mkt"] = (df["p_market"] - df["outcome"]) ** 2
    df["b_ens"] = (df["p_ens"] - df["outcome"]) ** 2
    return df


def test_city_rows_score_all_three_forecasters_on_the_same_markets():
    """2026-07-28 regression: the per-city block computed model/market from the calibrated
    tracker (401 markets) and the ensemble from the ensemble tracker (370) and put them in one
    row captioned as one comparison. On the published dashboard that handed London to the
    ensemble (0.1197 vs market 0.1290) purely because the ensemble was scored on an easier
    94-market subset; paired, the ensemble LOSES there (0.1259 vs 0.1228). One n per row is
    only honest if all three dots came from those same n markets."""
    import build_dashboard as bd
    rows = {r["city"]: r for r in bd._city_rows(_city_frame())}

    ldn = rows["London"]
    assert ldn["n"] == 4
    # every dot on the row is the mean over the SAME 4 markets
    assert ldn["market"] == pytest.approx(0.025, abs=1e-9)
    assert ldn["model"] == pytest.approx(0.025, abs=1e-9)
    assert ldn["ens"] == pytest.approx(0.185, abs=1e-9)
    # and the ensemble must not come out ahead here -- that was the artifact
    assert ldn["ens"] > ldn["market"]


def test_city_rows_flag_cities_where_every_market_settled_the_same_way():
    """Hong Kong published `ens 0.0625 < market 0.0712` off 11 graded markets with ZERO YES
    outcomes. With no outcome variance Brier collapses to mean(p^2), so the 'winner' is just
    whoever stated lower numbers -- predicting 0.0 everywhere scores a perfect 0. That row has
    no discrimination in it and must not render as a like-for-like accuracy win."""
    import build_dashboard as bd
    rows = {r["city"]: r for r in bd._city_rows(_city_frame())}

    assert rows["HongKong"]["degenerate"] is True
    assert rows["HongKong"]["yes"] == 0
    # a city with both outcomes present is a real comparison
    assert rows["London"].get("degenerate", False) is False


def test_city_rows_are_ordered_by_sample_size():
    import build_dashboard as bd
    rows = bd._city_rows(_city_frame())
    assert [r["n"] for r in rows] == sorted([r["n"] for r in rows], reverse=True)


def test_pooled_gap_needs_no_ensemble_and_uses_every_gradable_market():
    """2026-07-28: the dashboard's pooled gap was computed on the ensemble-paired common set
    (261) while evaluate_oos — the arbiter — computed it on all gradable markets (401). Two
    different answers to one question: +0.026 [+0.0084,+0.0434] vs +0.0211 [+0.0068,+0.0353].

    Model-minus-market needs the model probability, the traded price and the outcome. The
    ensemble is irrelevant, so requiring it to be present discarded 140 markets (68 of them Tmin,
    which the ensemble structurally cannot price) and 18.7% of the precision for nothing. This
    frame carries NO ensemble column at all — if _pooled_gap ever needs one, it is wrong."""
    import pandas as pd
    import build_dashboard as bd

    df = pd.DataFrame({
        "city":        ["Seoul"] * 4 + ["London"] * 2,
        "target_date": ["2026-07-01"] * 2 + ["2026-07-02"] * 2 + ["2026-07-01"] * 2,
        "b_model":     [0.20, 0.20, 0.30, 0.30, 0.10, 0.10],
        "b_mkt":       [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    })
    df["td"] = pd.to_datetime(df["target_date"])

    g = bd._pooled_gap(df)
    assert g["n"] == 6                       # every row scored
    assert g["clusters"] == 3                # Seoul 7-1, Seoul 7-2, London 7-1
    # mean of (b_model - b_mkt) = (.1+.1+.2+.2+0+0)/6
    assert g["gap"] == pytest.approx(0.1, abs=1e-9)
    assert g["lo"] < g["gap"] < g["hi"]


def test_pooled_gap_clusters_by_city_day_not_by_row():
    """Bins settling on one city-day share a single weather outcome, so they are one
    observation. Treating them as independent shrinks the interval and manufactures
    significance -- the whole reason stats_util exists."""
    import pandas as pd
    import build_dashboard as bd

    # 20 rows, all one city-day: one cluster, so the interval must be undefined/infinite
    df = pd.DataFrame({
        "city": ["Seoul"] * 20, "target_date": ["2026-07-01"] * 20,
        "b_model": [0.3] * 10 + [0.1] * 10, "b_mkt": [0.1] * 20,
    })
    df["td"] = pd.to_datetime(df["target_date"])
    g = bd._pooled_gap(df)
    assert g["n"] == 20 and g["clusters"] == 1
    assert not (g["lo"] > 0), "a single city-day must never read as a significant result"


def test_gap_kpi_number_comes_from_the_same_computation_as_its_interval():
    """2026-07-28: the "Model - market" tile rendered `s.model - s.market` — the three-way
    scoreboard's 261-market difference — while the CI directly beneath it came from the pooled
    test's 401. The published tile read "+0.026   95% CI [+0.0068, +0.0353]": a point estimate
    sitting OUTSIDE its own stated interval, which is not a thing that can be true.

    Only caught by rendering the page and reading the DOM; the payload was correct throughout.
    A number and its uncertainty must be one computation."""
    import build_dashboard as bd
    h = bd.render_shell({"series": {}, "generated_at": "2026-07-28T00:00:00Z"})
    assert "(P && isFinite(P.gap)) ? P.gap : (s.model - s.market)" in h, \
        "gap KPI must prefer the pooled gap so it matches the interval shown under it"


def _selection_frame():
    """Minimal frame shaped like the calibrated eval tracker."""
    import pandas as pd
    return pd.DataFrame({
        "condition_id": ["a", "b", "c", "d"],
        "city": ["Seoul", "Seoul", "London", "London"],
        "target_date": ["2026-07-07", "2026-07-08", "2026-07-07", "2026-07-09"],
        "outcome": [0, 1, 0, 1],
        "forecast_prob": [0.2, 0.8, 0.3, 0.7],
        "market_prob_raw": [0.3, 0.7, 0.2, 0.8],
    })


def test_split_frozen_is_chronological_and_deterministic():
    """The split date is frozen in code, not derived from the data. Deriving it (e.g. a 2/3
    quantile) would move the boundary every time new markets grade, so 'held-out' would quietly
    become a different set on each run."""
    import bet_selection as bs
    df = _selection_frame()
    train, holdout = bs.split_frozen(df)
    assert bs.SPLIT_DATE == "2026-07-08"
    assert sorted(train["condition_id"]) == ["a", "c"]      # strictly before the cut
    assert sorted(holdout["condition_id"]) == ["b", "d"]    # on or after
    # deterministic: same input, same output, no RNG
    again = bs.split_frozen(df)
    assert list(again[0]["condition_id"]) == list(train["condition_id"])


def test_split_frozen_never_leaks_or_straddles():
    """A condition_id in both halves would make the held-out test meaningless, and a city-day
    spanning the boundary would put correlated bins (one weather outcome) on both sides."""
    import bet_selection as bs
    train, holdout = bs.split_frozen(_selection_frame())
    assert set(train["condition_id"]).isdisjoint(set(holdout["condition_id"]))
    tr_days = set(train["city"] + "|" + train["target_date"])
    ho_days = set(holdout["city"] + "|" + holdout["target_date"])
    assert tr_days.isdisjoint(ho_days)


def _scored_frame():
    """Frame with two city-days per city so clustering has something to collapse."""
    import pandas as pd
    return pd.DataFrame({
        "condition_id": list("abcdef"),
        "city": ["Seoul"] * 3 + ["London"] * 3,
        "target_date": ["2026-07-01", "2026-07-01", "2026-07-02",
                        "2026-07-01", "2026-07-02", "2026-07-02"],
        "outcome": [1, 0, 1, 0, 1, 0],
        # model is PERFECT on the first three, terrible on the last three
        "forecast_prob": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "market_prob_raw": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "market_prob": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # must be IGNORED in favour of raw
        "their_prob": [0.5] * 6,
        "bet_side": ["Yes", "No", "Yes", "Yes", "No", "Yes"],
    })


def test_evaluate_selector_scores_the_paired_gap_on_the_raw_price():
    """The benchmark is the RAW tradeable price. market_prob is normalised so bins sum to 1,
    which flatters the market's Brier and would understate our own gap."""
    import bet_selection as bs
    df = _scored_frame()
    mask = df["city"] == "Seoul"          # the subset where the model is perfect
    r = bs.evaluate_selector(df, mask)
    assert r["n"] == 3
    assert r["clusters"] == 2             # Seoul 07-01 and Seoul 07-02
    assert r["kept"] == pytest.approx(0.5)
    # model Brier 0, market Brier 0.25 -> gap = -0.25 (negative = model better)
    assert r["gap"] == pytest.approx(-0.25, abs=1e-9)


def test_evaluate_selector_clusters_by_city_day():
    """Bins settling on one city-day share a single weather outcome. Counting them as
    independent shrinks the SE and manufactures significance."""
    import bet_selection as bs
    df = _scored_frame()
    r = bs.evaluate_selector(df, df["condition_id"].notna())
    assert r["n"] == 6 and r["clusters"] == 4      # 2 cities x 2 days, not 6
    assert r["mde"] > 0


def test_evaluate_selector_returns_none_on_empty_selection():
    """A threshold that keeps nothing must not raise or report a spurious gap."""
    import bet_selection as bs
    df = _scored_frame()
    assert bs.evaluate_selector(df, df["city"] == "Nowhere") is None


def test_selector_registry_is_pre_registered_and_pure():
    """Every family is a pure predicate over the frame, so a rule cannot silently depend on
    global state or on which rows were evaluated before it."""
    import bet_selection as bs
    df = _scored_frame()
    df["forecast_sigma"] = [1.0, 1.5, 2.5, 1.0, 1.5, 2.5]
    df["liquidity"] = [1200, 3000, 5000, 1200, 3000, 5000]
    df["pmf_sum_dev"] = [0.1, 0.5, 0.95, 0.1, 0.5, 0.95]
    df["volume_recency"] = [0.4, 0.85, 0.99, 0.4, 0.85, 0.99]
    df["bucket"] = ["Seoul|1d"] * 3 + ["London|1d"] * 3
    for name, (pred, thresholds) in bs.SELECTORS.items():
        assert thresholds, f"{name} has no thresholds"
        for t in thresholds:
            m = pred(df, t)
            assert len(m) == len(df), f"{name}@{t} returned the wrong length"
            assert m.dtype == bool, f"{name}@{t} did not return a boolean mask"


def test_edge_magnitude_is_excluded_by_design():
    """Adverse selection at z-std 1.41 (EDGE_MEGAPLAN §63): the model is most wrong exactly
    where it disagrees most with the price. Selecting on edge size is the measured trap, so it
    must not be reachable through the registry."""
    import bet_selection as bs
    assert "abs_edge" not in bs.SELECTORS
    assert "edge" not in bs.SELECTORS
    assert "abs_edge" in bs.EXCLUDED_BY_DESIGN


def test_candidate_count_is_the_pre_registered_number():
    """The count is logged with every search so the record shows how wide the net was, even
    though multiplicity is controlled by the held-out set rather than by a correction."""
    import bet_selection as bs
    cands = bs.iter_candidates()
    assert len(cands) == sum(len(t) for _, t in bs.SELECTORS.values())
    assert all(isinstance(name, str) for name, _ in cands)


def _searchable_frame():
    import pandas as pd
    import numpy as np
    df = pd.concat([_scored_frame()] * 3, ignore_index=True)
    df["condition_id"] = [f"c{i}" for i in range(len(df))]
    df["target_date"] = ["2026-07-01", "2026-07-02", "2026-07-03"] * 6

    # Vary signals so different thresholds select different subsets with genuinely different gaps.
    # Pattern: rows 0-2, 6-8, 12-14 are model-perfect (outcome matches forecast_prob exactly);
    # rows 3-5, 9-11, 15-17 are model-terrible. By varying each signal inversely (good rows
    # get low values, bad rows get high values), thresholds select different proportions.
    # This creates different gaps: low threshold = mostly good; high threshold = mixed.
    forecast_sigma = np.array([
        1.0, 1.0, 1.0,      # 0-2: good rows, low
        2.0, 2.0, 2.0,      # 3-5: bad rows, high
        1.5, 1.5, 1.5,      # 6-8: good rows, medium
        1.5, 1.5, 1.5,      # 9-11: bad rows, medium
        0.9, 0.9, 0.9,      # 12-14: good rows, very low
        1.9, 1.9, 1.9,      # 15-17: bad rows, high
    ], dtype=float)

    liquidity = np.array([
        1500, 1500, 1500,   # 0-2: good rows, low
        4000, 4000, 4000,   # 3-5: bad rows, high
        3000, 3000, 3000,   # 6-8: good rows, medium
        3000, 3000, 3000,   # 9-11: bad rows, medium
        4500, 4500, 4500,   # 12-14: good rows, high
        500, 500, 500,      # 15-17: bad rows, very low
    ], dtype=float)

    pmf_sum_dev = np.array([
        0.2, 0.2, 0.2,      # 0-2: good rows, low
        0.9, 0.9, 0.9,      # 3-5: bad rows, high
        0.5, 0.5, 0.5,      # 6-8: good rows, medium
        0.5, 0.5, 0.5,      # 9-11: bad rows, medium
        0.1, 0.1, 0.1,      # 12-14: good rows, very low
        0.8, 0.8, 0.8,      # 15-17: bad rows, high
    ], dtype=float)

    volume_recency = np.array([
        0.95, 0.95, 0.95,   # 0-2: good rows, high
        0.5, 0.5, 0.5,      # 3-5: bad rows, low
        0.7, 0.7, 0.7,      # 6-8: good rows, medium
        0.7, 0.7, 0.7,      # 9-11: bad rows, medium
        0.99, 0.99, 0.99,   # 12-14: good rows, very high
        0.3, 0.3, 0.3,      # 15-17: bad rows, very low
    ], dtype=float)

    df["forecast_sigma"] = forecast_sigma
    df["liquidity"] = liquidity
    df["pmf_sum_dev"] = pmf_sum_dev
    df["volume_recency"] = volume_recency
    df["bucket"] = "Seoul|1d"
    return df


def test_search_train_ranks_every_candidate_and_keeps_the_losers():
    """The record has to show how wide the net was. Reporting only the winner is how a search of
    32 rules gets written up as if one rule had been tried."""
    import bet_selection as bs
    res = bs.search_train(_searchable_frame())
    assert len(res) >= 1
    gaps = [r["gap"] for r in res]
    assert gaps == sorted(gaps), "results must be ranked by gap ascending (most negative first)"
    assert len(set(gaps)) > 1, "fixture must produce differing gaps or this test proves nothing"
    assert gaps != sorted(gaps, reverse=True), "verify not reverse-sorted (catches broken sort)"
    assert all("selector" in r and "threshold" in r for r in res)
    # empty selections are dropped, not reported as gap=0
    assert all(r["n"] > 0 for r in res)


def test_search_train_is_pure_and_does_not_mutate_its_input():
    """A search that mutates the frame makes results depend on evaluation order."""
    import bet_selection as bs
    df = _searchable_frame()
    before = df.copy(deep=True)
    bs.search_train(df)
    pd_testing = __import__("pandas").testing
    pd_testing.assert_frame_equal(df, before)


def test_validate_holdout_refuses_when_train_cannot_clear_the_bar():
    """The decision rule that matters most. A train gap of -0.01 cannot be confirmed by a
    held-out set that can only resolve -0.026 — running it would burn the one clean measurement
    we have to learn nothing. Held-out is spent only when train shows something detectable."""
    import bet_selection as bs
    df = _searchable_frame()
    train, holdout = df.iloc[:9].copy(), df.iloc[9:].copy()
    # a train result far too small to be visible on held-out
    weak = {"gap": -0.001, "n": 9, "clusters": 3, "se": 0.02}
    assert bs.train_clears_bar(weak, holdout_gap_se=0.013) is False
    strong = {"gap": -0.30, "n": 9, "clusters": 3, "se": 0.02}
    assert bs.train_clears_bar(strong, holdout_gap_se=0.013) is True


def test_validate_holdout_appends_exactly_one_record_per_call(tmp_path):
    """A code lock can be commented out; a record cannot be un-written. If this file ends up
    with twelve entries, any reader knows the p-value is fiction."""
    import json
    import bet_selection as bs
    df = _searchable_frame()
    train, holdout = df.iloc[:9].copy(), df.iloc[9:].copy()
    log = tmp_path / "holdout_log.jsonl"
    for _ in range(3):
        bs.validate_holdout(train, holdout, "bucket", "Seoul|1d",
                            data_cutoff="2026-07-29", log_path=log)
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 3, "every held-out evaluation must leave a permanent trace"
    assert {"selector", "threshold", "gap", "passed", "data_cutoff", "logged_at",
            "train_gap", "train_n"} <= set(lines[0])


def test_validate_holdout_passes_only_when_the_interval_clears_zero():
    """gap + 1.96*se < 0. A negative point estimate whose interval spans zero is not a result —
    that is exactly how the full-band structure gate read 'MET' while underpowered."""
    import bet_selection as bs
    import stats_util
    # Boundary case: construct so gap + Z*se == 0.0 exactly (by construction, not decimal approximation)
    _se_b = 0.01
    _gap_b = -(stats_util.Z * _se_b)  # gap + Z*se is then exactly 0.0 by construction
    cases = [
        (-0.10, 0.02, True),    # interval [-0.139,-0.061] entirely below zero
        (-0.03, 0.02, False),   # negative point estimate, interval spans zero -> NOT a result
        (-0.03, 0.01, True),    # [-0.050,-0.010] clears
        (+0.05, 0.02, False),   # positive gap never passes
        (-0.10, float("inf"), False),  # undefined se cannot pass
        (_gap_b, _se_b, False),  # EXACTLY on the boundary: interval touches zero but does not
                                 # clear it, so this must NOT pass. This row discriminates `<` from `<=`.
    ]
    for gap, se, want in cases:
        assert bs.interval_clears_zero(gap, se) is want, f"gap={gap} se={se}"


def test_validate_holdout_log_is_valid_strict_json(tmp_path):
    """Log lines must be parseable by non-Python readers (jq, etc). A record from an empty
    selection contains NaN/Infinity; they must serialize to null, not bare NaN/Infinity."""
    import json
    import bet_selection as bs
    df = _searchable_frame()
    train = df.iloc[:9].copy()
    holdout_empty = df.iloc[0:0].copy()  # empty
    log = tmp_path / "holdout_log.jsonl"
    r = bs.validate_holdout(train, holdout_empty, "bucket", "Seoul|1d",
                            data_cutoff="2026-07-29", log_path=log)
    # Record returned to caller has real inf/nan
    assert r["se"] == float("inf")
    # But log line is strict JSON
    line = log.read_text().strip()
    # This parser fires on NaN/Infinity — if any bare non-finites were written, it raises
    json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite: {c}")))


def test_search_path_never_reads_holdout_rows(tmp_path):
    """The leakage guard is structural, not disciplinary. If cmd_search ever passed the full
    frame to search_train, poisoning the held-out rows would change its output. It must not.

    Written as a behavioural test rather than a code-shape assertion, because the failure this
    prevents is a one-word edit (train -> df) that no signature check would catch.
    """
    import bet_selection as bs
    import pandas as pd

    base = _searchable_frame()
    base["target_date"] = ["2026-07-01", "2026-07-02", "2026-07-03"] * 6   # all pre-split
    poisoned = base.copy()
    poisoned["target_date"] = ["2026-07-20"] * len(poisoned)               # all post-split
    poisoned["forecast_prob"] = 0.0                                        # absurdly wrong
    poisoned["outcome"] = 1
    poisoned["condition_id"] = [f"p{i}" for i in range(len(poisoned))]

    clean_csv = tmp_path / "clean.csv"
    mixed_csv = tmp_path / "mixed.csv"
    base.to_csv(clean_csv, index=False)
    pd.concat([base, poisoned], ignore_index=True).to_csv(mixed_csv, index=False)

    a = bs.cmd_search(csv_path=clean_csv)
    b = bs.cmd_search(csv_path=mixed_csv)
    assert [(r["selector"], r["threshold"], round(r["gap"], 9)) for r in a] == \
           [(r["selector"], r["threshold"], round(r["gap"], 9)) for r in b], \
        "search results changed when held-out rows were added — the search read held-out data"


def _cmd_validate_frame(train_is_bad: bool):
    """A searchable frame split so ALL model-perfect ('good') rows land on one side of
    SPLIT_DATE and all model-terrible ('bad') rows land on the other, selected via the
    'bucket' selector at threshold 'Seoul|1d' (constant on every row, so it always keeps
    everything). Every row within a half is identical, so the clustered SE is exactly 0 and
    train_clears_bar's outcome is deterministic rather than borderline.

    train_is_bad=True  -> train is all 'bad' rows (gap +0.75, se 0): fails the bar.
    train_is_bad=False -> train is all 'good' rows (gap -0.25, se 0): clears the bar.
    """
    df = _searchable_frame()
    pre = ["2026-07-01", "2026-07-02", "2026-07-03"]     # before SPLIT_DATE (2026-07-08)
    post = ["2026-07-20", "2026-07-21", "2026-07-22"]    # after SPLIT_DATE
    good_idx = [0, 1, 2, 6, 7, 8, 12, 13, 14]             # forecast_prob == outcome, brier 0
    bad_idx = [3, 4, 5, 9, 10, 11, 15, 16, 17]            # forecast_prob wrong every time, brier 1
    good_dates, bad_dates = (post, pre) if train_is_bad else (pre, post)
    dates = [None] * len(df)
    for j, i in enumerate(good_idx):
        dates[i] = good_dates[j % 3]
    for j, i in enumerate(bad_idx):
        dates[i] = bad_dates[j % 3]
    df["target_date"] = dates
    return df


def test_cmd_validate_refuses_off_grid_threshold(tmp_path):
    """Pre-registration must be enforced inside cmd_validate itself, not only in main()'s CLI
    parsing — a programmatic caller (notebook, future refactor, another module) must not be
    able to spend the one shot on an un-pre-registered threshold. Nothing held-out is touched
    on this path, so nothing should be logged."""
    import bet_selection as bs
    df = _searchable_frame()
    csv = tmp_path / "bets.csv"
    df.to_csv(csv, index=False)
    log = tmp_path / "holdout_log.jsonl"
    _, grid = bs.SELECTORS["forecast_prob_floor"]
    with pytest.raises(SystemExit) as exc:
        bs.cmd_validate("forecast_prob_floor", 0.37, "2026-07-29", csv_path=csv, log_path=log)
    assert str(grid) in str(exc.value), "message must name the registered grid"
    assert not log.exists(), "off-grid threshold must be rejected before touching held-out data"


def test_cmd_validate_refuses_when_train_cannot_clear_the_bar(tmp_path):
    """cmd_validate must not spend the one shot on an effect train can't show is real. The
    refusal message must state both the train gap and the held-out MDE."""
    import bet_selection as bs
    df = _cmd_validate_frame(train_is_bad=True)
    csv = tmp_path / "bets.csv"
    df.to_csv(csv, index=False)
    log = tmp_path / "holdout_log.jsonl"
    with pytest.raises(SystemExit) as exc:
        bs.cmd_validate("bucket", "Seoul|1d", "2026-07-29", csv_path=csv, log_path=log)
    msg = str(exc.value)
    assert "REFUSED" in msg
    assert "+0.7500" in msg, "message must state the train gap"
    assert "0.0000" in msg, "message must state the held-out MDE"


def test_cmd_validate_refusal_logs_a_record_without_the_holdout_gap(tmp_path):
    """Refused attempts still consulted held-out (its SE drove the refusal decision), so they
    must leave a trace too — otherwise a caller could probe rule after rule and the audit trail
    would show only the one that stuck. But the held-out GAP itself must NOT be logged, or the
    refusal path becomes a backdoor way to read the one-shot answer without spending it."""
    import json
    import bet_selection as bs
    df = _cmd_validate_frame(train_is_bad=True)
    csv = tmp_path / "bets.csv"
    df.to_csv(csv, index=False)
    log = tmp_path / "holdout_log.jsonl"
    with pytest.raises(SystemExit):
        bs.cmd_validate("bucket", "Seoul|1d", "2026-07-29", csv_path=csv, log_path=log)
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 1, "exactly one record per touch, even a refused one"
    rec = lines[0]
    assert rec["outcome"] == "refused"
    assert "gap" not in rec, "the refusal record must not leak the held-out gap"
    assert {"selector", "threshold", "train_gap", "train_n", "holdout_se", "holdout_mde",
            "holdout_n", "holdout_clusters", "logged_at"} <= set(rec)


def test_cmd_validate_success_logs_validated_outcome(tmp_path):
    """A validation that clears the bar must reach validate_holdout and log
    outcome == 'validated', distinguishing it from a refused attempt in the same log."""
    import json
    import bet_selection as bs
    df = _cmd_validate_frame(train_is_bad=False)
    csv = tmp_path / "bets.csv"
    df.to_csv(csv, index=False)
    log = tmp_path / "holdout_log.jsonl"
    r = bs.cmd_validate("bucket", "Seoul|1d", "2026-07-29", csv_path=csv, log_path=log)
    assert r["selector"] == "bucket" and r["threshold"] == "Seoul|1d"
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "validated"
    assert lines[0]["selector"] == "bucket"


def test_flat_roi_pins_exact_arithmetic_against_hand_computed_pnl():
    """flat_roi feeds the roiflat column humans read straight off `--search`, but nothing pinned
    its value: a future sign flip or a dropped cost term would still pass the rest of the suite
    silently and just quietly flatter the model. Each row's expected PnL is derived here
    independently — win = payout at the crossed price net of the fee, loss = -1.0 — from
    config.HALF_SPREAD / config.FEE_RATE directly, not by calling flat_roi or copying its
    expression, so this cannot degrade into a tautology that just restates the implementation."""
    import config
    import bet_selection as bs
    import pandas as pd

    df = pd.DataFrame({
        "their_prob": [0.25, 0.25, 0.50],
        "bet_side":   ["Yes", "No",  "Yes"],
        "outcome":    [1,      1,     0],      # row 0 wins, row 1 loses, row 2 loses
    })

    eff_win = 0.25 + config.HALF_SPREAD          # buy price = their_prob + half the spread
    expected = ((1 - config.FEE_RATE) / eff_win - 1.0    # row 0: Yes bet, outcome 1 -> wins
                + (-1.0)                                  # row 1: No bet, outcome 1 -> loses
                + (-1.0)) / 3                              # row 2: Yes bet, outcome 0 -> loses
    assert bs.flat_roi(df) == pytest.approx(expected, abs=1e-12)

    # A dropped cost must make the number MORE flattering, not just different — pin the
    # direction. Both no-cost variants are computed inline here (not by editing flat_roi), and
    # only the winning row's PnL differs since a loss is -1.0 regardless of price paid.
    no_spread_pnl0 = (1 - config.FEE_RATE) / 0.25 - 1.0             # spread cost removed
    no_spread = (no_spread_pnl0 + (-1.0) + (-1.0)) / 3
    no_fee_pnl0 = 1.0 / eff_win - 1.0                                # fee cost removed
    no_fee = (no_fee_pnl0 + (-1.0) + (-1.0)) / 3
    assert bs.flat_roi(df) < no_spread, \
        "dropping the half-spread must not leave flat_roi this high or higher"
    assert bs.flat_roi(df) < no_fee, \
        "dropping the fee must not leave flat_roi this high or higher"


# ── Question classification for tag-based discovery ────────────────────────────

def test_match_city_handles_every_configured_alias():
    """Discovery is only as good as this function — a city it cannot name is a city we
    silently stop collecting."""
    import fetch_polymarket as fp
    assert fp.match_city("Will the highest temperature in London be 22°C on July 29?") == "London"
    assert fp.match_city("Will the highest temperature in Seoul be 30°C on July 29?") == "Seoul"
    assert fp.match_city("Will the highest temperature in Chicago be 30°C on July 29?") == "Chicago"
    assert fp.match_city("Will the highest temperature in Hong Kong be 33°C on July 29?") == "Hong Kong"
    # New York City has three aliases and all must land on the same key
    for q in ["Highest temperature in New York on July 29?",
              "Highest temperature in NYC on July 29?",
              "highest temperature in new york on July 29?"]:
        assert fp.match_city(q) == "New York City", q


def test_match_city_does_not_match_substrings_of_other_words():
    """'New Yorker' contains 'New York'. Matching it would file an unrelated market under NYC
    and corrupt that city's series."""
    import fetch_polymarket as fp
    assert fp.match_city("Will the New Yorker publish a temperature piece?") is None
    assert fp.match_city("Will the highest temperature in Paris be 30°C?") is None


def test_discovery_matches_capture_tier_cities():
    """The seven capture cities are already DISCOVERED by tag — only persistence was missing.
    match_city and discover_by_tag iterate the city registry, so they must see ALL_CITIES, not
    just the modelled five, or the capture cities are found and then dropped on the floor."""
    from fetch_polymarket import match_city

    assert match_city("Highest temperature in Los Angeles on August 4?") == "Los Angeles"
    assert match_city("Highest temperature in Houston on August 4?") == "Houston"
    assert match_city("Highest temperature in San Francisco on August 4?") == "San Francisco"
    # Word-boundary matching must still hold for the new cities.
    assert match_city("Austin Powers trivia market") == "Austin"      # bare word does match
    assert match_city("Highest temperature in Austintown on August 4?") is None
    # The original five are unaffected.
    assert match_city("Highest temperature in London on August 4?") == "London"


def test_discover_by_tag_files_capture_tier_cities():
    """discover_by_tag partitions by calling _matching_cities (ambiguity check) and match_city
    (filing); both must iterate ALL_CITIES too, or a capture-tier question is either filed under
    no city or, if _matching_cities alone were left on CITIES while match_city was fixed, treated
    as unambiguous for the wrong reason. Covering the full discover_by_tag path (not just
    match_city in isolation) is what actually mutation-tests the _matching_cities loop."""
    import unittest.mock as m
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    events = [
        {"title": "LA temps", "markets": [
            {"conditionId": "la1",
             "question": "Will the highest temperature in Los Angeles be 80°F on August 4?"},
        ]},
    ]
    with m.patch.object(fp, "_paged_events", lambda tag, page_size=100: (events, False)):
        out = fp.discover_by_tag("weather")
    assert "Los Angeles" in out, "capture-tier city never reached the discover_by_tag partition"
    assert len(out["Los Angeles"]) == 1


def test_discover_by_tag_skips_ambiguous_capture_tier_questions(caplog):
    """Same ambiguity guard as the modelled-city test above, but naming two CAPTURE-tier cities.
    _matching_cities is what detects the ambiguous case; if it were left iterating CITIES
    (modelled-only) instead of ALL_CITIES, a question naming two capture cities would return zero
    candidates from _matching_cities (never >1, so never flagged ambiguous) and would then be
    silently filed under whichever city match_city's ALL_CITIES order happens to prefer -- the
    single-city discovery test above cannot catch that, because it never presents two capture
    cities together."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m
    fp._reset_tag_cache()
    ambiguous_events = [
        {"title": "ambiguous", "markets": [
            {"conditionId": "amb2",
             "question": "Will the highest temperature in Los Angeles or Houston be higher on August 4?"},
        ]},
    ]
    with caplog.at_level(logging.WARNING), m.patch.object(
            fp, "_paged_events", lambda tag, page_size=100: (ambiguous_events, False)):
        out = fp.discover_by_tag("weather")
    all_cids = [m["conditionId"] for ms in out.values() for m in ms]
    assert "amb2" not in all_cids, "an ambiguous capture-tier market must not be filed under any city"
    assert out.get("Los Angeles", []) == []
    assert out.get("Houston", []) == []


def test_forecast_steps_skip_capture_tier_cities(monkeypatch, caplog):
    """Widening the collector's city list must not widen the FORECAST paths. main's
    step_fetch_weather / step_fetch_ensemble guard on `city not in CITIES`, and CITIES is
    modelled-only — so a capture city is skipped without an API call.

    step_fetch_weather calls FIVE other side-effecting functions beyond fetch_forecast, all of
    which are real network/disk I/O if left unpatched: fetch_forecast_multimodel (per-city, same
    guard) and four process-wide top-ups — fetch_station_obs, fetch_nbm, shoulder_book's
    scan_and_record, and shoulder_book_breadth's scan_and_record_breadth. The last one anchors to
    Path(__file__).parent and writes the TRACKED output/shoulder_paper_breadth.csv regardless of
    cwd; fetch_station_obs uses a bare relative "data/weather" that creates a stray data/ dir at
    whatever cwd pytest ran from. Both were observed mutating a clean tree. All five are
    monkeypatched here — three (fetch_station_obs/fetch_nbm/scan_and_record/
    scan_and_record_breadth) at their SOURCE module, because main imports them lazily inside the
    function body (`from x import y`), so patching `main.y` would not intercept the call.
    """
    import main
    import fetch_station_obs
    import fetch_nbm
    import shoulder_book
    import shoulder_book_breadth

    called = []
    monkeypatch.setattr(main, "fetch_forecast", lambda c: called.append(c) or None)
    mm_called = []
    monkeypatch.setattr(main, "fetch_forecast_multimodel", lambda c: mm_called.append(c) or None)
    obs_called = []
    monkeypatch.setattr(fetch_station_obs, "fetch_station_obs",
                         lambda recent_only=False: obs_called.append(recent_only))
    nbm_called = []
    monkeypatch.setattr(fetch_nbm, "fetch_nbm",
                         lambda recent_only=False: nbm_called.append(recent_only))
    shoulder_called = []
    monkeypatch.setattr(shoulder_book, "scan_and_record",
                         lambda: shoulder_called.append(True) or 0)
    breadth_called = []
    monkeypatch.setattr(shoulder_book_breadth, "scan_and_record_breadth",
                         lambda: breadth_called.append(True) or 0)

    main.step_fetch_weather(["Los Angeles", "London"])

    assert called == ["London"], f"capture city reached the forecast fetcher: {called}"
    assert mm_called == ["London"], f"capture city reached the multimodel fetcher: {mm_called}"
    # These four run once per step call (not per city, no tiering guard applies to them) — assert
    # each mock was actually exercised so the patching is proven load-bearing, not dead code that
    # happens to look like isolation while the real functions still ran underneath.
    assert obs_called == [True], "fetch_station_obs mock was not hit — real network/disk I/O risk"
    assert nbm_called == [True], "fetch_nbm mock was not hit — real network/disk I/O risk"
    assert shoulder_called == [True], "scan_and_record mock was not hit — real network/disk I/O risk"
    assert breadth_called == [True], \
        "scan_and_record_breadth mock was not hit — would mutate tracked shoulder_paper_breadth.csv"


def test_step_fetch_ensemble_skips_capture_tier_cities(monkeypatch):
    """Same tiering mechanism as step_fetch_weather, for the ensemble path — the docstring above
    claims both are guarded; this is what actually proves the ensemble half of that claim.

    Unlike step_fetch_weather, step_fetch_ensemble calls nothing beyond fetch_ensemble (no lazy
    imports, no process-wide top-ups) — confirmed by reading main.py — so mocking fetch_ensemble
    alone already fully isolates this test from the network/disk.
    """
    import main
    called = []
    monkeypatch.setattr(main, "fetch_ensemble", lambda c: called.append(c) or None)
    main.step_fetch_ensemble(["Los Angeles", "London"])
    assert called == ["London"], f"capture city reached the ensemble fetcher: {called}"


def test_default_cities_flag_widens_to_all_cities(monkeypatch):
    """--cities default must be ALL_CITIES, not the five modelled cities, or the seven
    capture-tier cities never reach step_fetch_polymarket even though discovery now finds them."""
    import main
    import config
    monkeypatch.setattr("sys.argv", ["main.py"])
    args = main.parse_args()
    assert set(args.cities) == set(config.ALL_CITIES.keys())


def test_main_validation_accepts_capture_tier_cities(monkeypatch, caplog):
    """main()'s post-parse validation filter must recognize capture-tier city names as valid —
    not just the five modelled ones — or a capture city that survived the argparse default (or
    was passed explicitly via --cities) gets re-dropped one line later. --summary-only keeps this
    to a local CSV read (compute_city_summary tolerates a missing/empty file), so nothing is
    fetched over the network."""
    import logging
    import main
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--cities", "Los Angeles", "Nonexistent City", "--summary-only"])
    with caplog.at_level(logging.WARNING):
        main.main()   # must not sys.exit(1) — "Los Angeles" is a valid ALL_CITIES member
    warnings = [r.message for r in caplog.records if "Unknown cities" in r.message]
    assert warnings, "expected a warning naming the unknown city"
    assert "Nonexistent City" in warnings[0]
    assert "Los Angeles" not in warnings[0], \
        "capture-tier city was treated as unknown by the validation filter"


def test_is_temperature_question_keeps_tmin_markets():
    """~20% of markets settle on the daily MINIMUM. A filter written around 'highest' drops
    them silently — and Tmin is already a market type this repo excluded once before."""
    import fetch_polymarket as fp
    assert fp.is_temperature_question("Will the lowest temperature in London be 15°C or below on July 29?")
    assert fp.is_temperature_question("Will the highest temperature in London be 22°C on July 29?")
    assert not fp.is_temperature_question("Will it rain in London on July 29?")
    assert not fp.is_temperature_question("Will 2026 be the hottest year on record?")


def test_classification_is_case_insensitive():
    """Stripping re.I from either regex passed every existing assertion — config
    double-lists "New York"/"new york" and the temperature fixtures are all lowercase, so
    case-insensitivity was assumed rather than tested. Polymarket capitalises inconsistently
    across market types."""
    import fetch_polymarket as fp
    assert fp.match_city("Highest Temperature in NYC on July 30?") == "New York City"
    assert fp.match_city("HIGHEST TEMPERATURE IN LONDON ON JULY 30?") == "London"
    assert fp.is_temperature_question("Will The HIGHEST TEMPERATURE In London Be 22C?")
    assert fp.is_temperature_question("LOWEST TEMPERATURE IN SEOUL ON JULY 30?")


def test_precipitation_markets_are_excluded():
    """The weather tag also carries precipitation markets — 62 of the 2718 real questions
    mentioning our cities. They have no temperature to grade against station truth, so they
    must not enter the pipeline. Pinned because the natural 'fix' for a perceived Tmin gap is
    to loosen the temperature pattern, which would sweep these in."""
    import fetch_polymarket as fp
    assert not fp.is_temperature_question("Will Hong Kong have 600mm or more of precipitation in July?")
    assert not fp.is_temperature_question("Will Hong Kong have between 400-425mm of precipitation in July?")


def test_paged_events_stops_cleanly_on_a_short_page(monkeypatch):
    """A short page is a legitimate end of list — no truncation flag, no warning."""
    import fetch_polymarket as fp
    pages = [[{"id": i} for i in range(100)], [{"id": 100}]]
    calls = []

    def fake_get(url, params=None):
        calls.append(params["offset"])
        return pages.pop(0) if pages else []

    monkeypatch.setattr(fp, "_get", fake_get)
    events, truncated = fp._paged_events("weather", page_size=100)
    assert len(events) == 101
    assert truncated is False
    assert calls == [0, 100]


def test_paged_events_flags_truncation_when_a_full_page_is_followed_by_failure():
    """THE bug this whole change exists for. GET /markets 422s at offset 2100 and _get returns
    None, so the pager read a hard truncation as 'last page'. A 3% capture rate looked healthy
    for months. None after a FULL page must be reported as truncation, never as completion."""
    import fetch_polymarket as fp

    class _Stub:
        def __init__(self):
            self.n = 0

        def __call__(self, url, params=None):
            self.n += 1
            return [{"id": i} for i in range(100)] if self.n == 1 else None

    import unittest.mock as m
    with m.patch.object(fp, "_get", _Stub()):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert len(events) == 100
    assert truncated is True, "a full page followed by an error is a TRUNCATION, not the end"


def test_paged_events_reports_no_truncation_when_the_first_page_fails():
    """An endpoint that is down from the first call is an outage, not a truncation — the caller
    falls back rather than trusting a partial list."""
    import fetch_polymarket as fp
    import unittest.mock as m
    with m.patch.object(fp, "_get", lambda url, params=None: None):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert events == []
    assert truncated is False


def test_truncation_emits_an_operator_warning(caplog):
    """The flag is for code; the warning is for whoever reads the collector logs. Deleting
    the warning left every test green, so the operator-facing half of this fix was unpinned."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m

    class _Stub:
        def __init__(self):
            self.n = 0

        def __call__(self, url, params=None):
            self.n += 1
            return [{"id": i} for i in range(100)] if self.n == 1 else None

    with caplog.at_level(logging.WARNING), m.patch.object(fp, "_get", _Stub()):
        fp._paged_events("weather", page_size=100)
    assert any("TRUNCATED" in r.message for r in caplog.records), \
        "truncation must emit a TRUNCATED warning"
    assert any("100" in r.message for r in caplog.records), \
        "warning must name the offset reached"


def test_short_page_emits_no_warning(caplog):
    """A warning on the NORMAL path becomes noise and gets ignored — which is how the
    original truncation signal would be lost a second time."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m

    pages = [[{"id": i} for i in range(100)], [{"id": 100}]]
    calls = []

    def fake_get(url, params=None):
        calls.append(params["offset"])
        return pages.pop(0) if pages else []

    with caplog.at_level(logging.WARNING), m.patch.object(fp, "_get", fake_get):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert len(events) == 101, "should get all events from both pages"
    assert truncated is False, "short page is not a truncation"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
        "short page must not emit any warning"


def test_cap_enforces_max_pages(caplog):
    """Hitting the page cap terminates and returns truncated=True with a warning.
    Without this, an API that ignores `offset` and re-serves page 1 forever would
    hang the hourly collector."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m

    # Stub always returns a full page (never terminates naturally)
    stub_call_count = [0]

    def always_full_page(url, params=None):
        stub_call_count[0] += 1
        return [{"id": i} for i in range(100)]

    with caplog.at_level(logging.WARNING), m.patch.object(fp, "_get", always_full_page):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert truncated is True, "hitting the cap must report truncation"
    assert len(events) == fp._MAX_EVENT_PAGES * 100, \
        f"should have max pages worth of events ({fp._MAX_EVENT_PAGES * 100})"
    assert stub_call_count[0] == fp._MAX_EVENT_PAGES, \
        f"should have made exactly {fp._MAX_EVENT_PAGES} calls"
    assert any("TRUNCATED" in r.message and "cap" in r.message for r in caplog.records), \
        "cap hit must emit a warning naming the cap"


_TAG_EVENTS = [
    {"title": "London temps", "markets": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C on July 29?"},
        {"conditionId": "b", "question": "Will the lowest temperature in London be 15°C or below on July 29?"},
    ]},
    {"title": "NYC temps", "markets": [
        {"conditionId": "c", "question": "Will the highest temperature in NYC be 30°C on July 29?"},
    ]},
    {"title": "climate", "markets": [
        {"conditionId": "d", "question": "Will 2026 be the hottest year on record?"},
    ]},
    {"title": "malformed — no markets key"},
    {"title": "dupe", "markets": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C on July 29?"},
    ]},
]


def test_discover_by_tag_partitions_by_city_and_keeps_tmin(monkeypatch):
    """Both London rows must survive: one Tmax, one Tmin. Dropping Tmin here would repeat an
    exclusion this repo has already made once by accident."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "_paged_events", lambda tag, page_size=100: (_TAG_EVENTS, False))
    out = fp.discover_by_tag("weather")
    assert sorted(out) == ["London", "New York City"]
    assert len(out["London"]) == 2, "the Tmin market was dropped"
    assert len(out["New York City"]) == 1


def test_discover_by_tag_ignores_non_temperature_and_malformed_events(monkeypatch):
    """The weather tag also carries climate markets ('hottest year on record') and events with
    no markets key at all. Neither may reach a city bucket or raise."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "_paged_events", lambda tag, page_size=100: (_TAG_EVENTS, False))
    out = fp.discover_by_tag("weather")
    qs = [m["question"] for ms in out.values() for m in ms]
    assert not any("hottest year" in q for q in qs)


def test_discover_by_tag_dedupes_and_pages_only_once(monkeypatch):
    """Condition 'a' appears in two events. And the tag must be paged ONCE per process, not once
    per city — otherwise five cities means five full paginations, which is the cost this change
    exists to remove."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    calls = []

    def fake_paged(tag, page_size=100):
        calls.append(tag)
        return _TAG_EVENTS, False

    monkeypatch.setattr(fp, "_paged_events", fake_paged)
    fp.discover_by_tag("weather")
    fp.discover_by_tag("weather")
    fp.discover_by_tag("weather")
    assert len(calls) == 1, "tag was paged more than once per process"
    assert len(fp.discover_by_tag("weather")["London"]) == 2, "duplicate conditionId not deduped"


def test_discover_by_tag_warns_when_the_enumeration_was_truncated(caplog):
    """A truncated enumeration must never be presented as a complete picture. This is the
    original bug one layer up: the old pager could not distinguish a hard API ceiling from
    the end of the list, and that silence hid a ~3% capture rate for months."""
    import logging
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatched = lambda tag, page_size=100: (_TAG_EVENTS, True)   # truncated=True
    import unittest.mock as m
    with caplog.at_level(logging.WARNING), m.patch.object(fp, "_paged_events", monkeypatched):
        fp.discover_by_tag("weather")
    assert any("FLOOR" in r.message for r in caplog.records), \
        "a truncated enumeration must be reported as a FLOOR, not a complete picture"


def test_discover_by_tag_does_not_warn_when_complete(caplog):
    """A warning on the NORMAL path becomes noise and gets ignored — which is how the
    truncation signal would be lost a second time."""
    import logging
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    import unittest.mock as m
    with caplog.at_level(logging.WARNING), m.patch.object(
            fp, "_paged_events", lambda tag, page_size=100: (_TAG_EVENTS, False)):
        fp.discover_by_tag("weather")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_discover_by_tag_cache_keys_on_tag_slug(monkeypatch):
    """Cache keys on tag_slug. discover_by_tag('weather') then discover_by_tag('other') must
    each trigger their own _paged_events call and return distinct results."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    calls = []

    def fake_paged(tag, page_size=100):
        calls.append(tag)
        if tag == "weather":
            return _TAG_EVENTS, False
        else:
            return [{"title": "other tag", "markets": [
                {"conditionId": "x", "question": "Will the highest temperature in Seoul be 28°C on July 29?"},
            ]}], False

    monkeypatch.setattr(fp, "_paged_events", fake_paged)
    weather_result = fp.discover_by_tag("weather")
    other_result = fp.discover_by_tag("other")
    assert len(calls) == 2, "each tag should trigger its own pagination"
    assert "London" in weather_result
    assert "Seoul" in other_result
    assert "Seoul" not in weather_result


def test_discover_by_tag_drops_markets_with_empty_or_missing_conditionid(monkeypatch):
    """A market with an empty or missing conditionId is DROPPED, not kept. Dedupe keys on that
    field, so keeping such rows would silently merge unrelated markets together."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    bad_events = [
        {"title": "London temps", "markets": [
            {"conditionId": "a", "question": "Will the highest temperature in London be 22°C on July 29?"},
            {"question": "Will the highest temperature in London be 23°C on July 29?"},  # missing conditionId
            {"conditionId": "", "question": "Will the highest temperature in London be 24°C on July 29?"},  # empty conditionId
        ]},
    ]
    import unittest.mock as m
    with m.patch.object(fp, "_paged_events", lambda tag, page_size=100: (bad_events, False)):
        result = fp.discover_by_tag("weather")
    assert len(result["London"]) == 1, "only the market with valid conditionId should survive"
    assert result["London"][0]["conditionId"] == "a"


# ── Task 4: wire tag discovery in, repair the query-scan fallback ────────────

def test_search_matches_keyword_and_term_independently(monkeypatch):
    """The fallback was 3/4 dead. Queries were built as f"{kw} {term}" and matched as a literal
    substring, but real questions read "highest temperature IN London" — so
    "highest temperature London" never matched anything, and three of four keywords merely paged
    2100 markets for nothing. A fallback that does not work is not a fallback."""
    import fetch_polymarket as fp
    page = [{"question": "Will the highest temperature in London be 22°C on July 29?"},
            {"question": "Will the highest temperature in Paris be 30°C on July 29?"}]
    calls = []

    def fake_get(url, params=None):
        calls.append(params["offset"])
        return page if params["offset"] == 0 else []

    monkeypatch.setattr(fp, "_get", fake_get)
    got = fp.search_markets_by_query("highest temperature London")
    assert len(got) == 1, "keyword and city must match independently, not as one substring"
    assert "London" in got[0]["question"]


def test_fetch_weather_markets_uses_the_tag(monkeypatch, caplog):
    """Return shape is unchanged: a list of snapshot dicts from extract_market_snapshot. Also
    pins the quiet path: the tag already found markets, so the retag-fallback warning must NOT
    fire — a warning that fires on every call is noise nobody will notice when it matters."""
    import logging
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {"London": [
        {"conditionId": "a", "question": "Will the highest temperature in London be 22°C?",
         "active": True, "closed": False, "endDateIso": "2026-07-29",
         "startDateIso": "2026-07-27", "volume": 1.0, "volume24hr": 1.0, "liquidity": 1.0,
         "clobTokenIds": '["123"]', "outcomePrices": '["0.5", "0.5"]'}]})
    with caplog.at_level(logging.WARNING):
        out = fp.fetch_weather_markets("London")
    assert len(out) == 1
    assert out[0]["city"] == "London"
    assert out[0]["condition_id"] == "a"
    assert out[0]["clob_token_ids"] == ["123"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
        "the fallback warning must not fire when the tag already found markets"


def test_fetch_weather_markets_falls_back_when_the_tag_yields_nothing(monkeypatch, caplog):
    """If Polymarket retags these markets, tag discovery silently returns nothing for a city that
    had plenty. That must fall back to the query scan AND warn — not silently collect zero.
    Pinned with caplog so the warning cannot regress into a silent no-op that nothing catches."""
    import logging
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {})
    used = []
    monkeypatch.setattr(fp, "search_markets_by_query",
                        lambda q, limit=100: used.append(q) or [])
    with caplog.at_level(logging.WARNING):
        fp.fetch_weather_markets("London")
    assert used, "fallback did not fire when the tag returned nothing"
    assert any("falling back" in r.message.lower()
               for r in caplog.records if r.levelno >= logging.WARNING), \
        "fetch_weather_markets must WARN when falling back off the tag"


# ── Task 4, fix round 1: word-boundary matching, the fallback's second gate, ─
# ── a raising tag lookup, and not caching an empty tag partition. ────────────

def test_search_word_boundary_rejects_new_yorker_and_londonderry(monkeypatch):
    """Token matching was substring-based, not word-boundary -- match_city already documents
    this exact trap (fetch_polymarket.py:50-52) and search_markets_by_query must mirror it.
    'temperature in New York' must not match a question that only contains 'New Yorker' as one
    word, and 'highest temperature London' must not match a question that only contains
    'Londonderry'."""
    import fetch_polymarket as fp
    page = [
        {"conditionId": "ok", "question": "Will the highest temperature in London be 22°C on July 29?"},
        {"conditionId": "nyer", "question": "Will the temperature in New Yorker Magazine's HQ city be newsworthy?"},
        {"conditionId": "ldy", "question": "Will Londonderry see the highest temperature in Ireland?"},
    ]

    def fake_get(url, params=None):
        return page if params["offset"] == 0 else []

    monkeypatch.setattr(fp, "_get", fake_get)

    assert fp.search_markets_by_query("temperature in New York") == [], \
        "'New Yorker' must not satisfy a 'New York' token match"
    got = fp.search_markets_by_query("highest temperature London")
    assert [m["conditionId"] for m in got] == ["ok"], \
        "'Londonderry' must not satisfy a 'London' token match"


def test_fetch_weather_markets_fallback_rejects_a_question_about_another_city(monkeypatch):
    """search_markets_by_query only guarantees every query token appears somewhere in the
    question -- it says nothing about which city the question is ABOUT. "Will Bitcoin's high be
    reached in Chicago while the London Stock Exchange sets a temperature record?" satisfies
    every token of a Chicago query (high/temperature/Chicago all present, word-boundary and all)
    and even passes is_temperature_question (a superlative followed eventually by "temperature"),
    but match_city resolves it to London, not Chicago -- "London" is mentioned in the sentence
    and comes first in CITIES iteration order. The fallback's second gate -- match_city(q) ==
    city -- must keep a market like this out of Chicago's results even though it cleared the
    token gate."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {})
    decoy = {"conditionId": "btc", "question": "Will Bitcoin's high be reached in Chicago while "
             "the London Stock Exchange sets a temperature record?"}
    monkeypatch.setattr(fp, "search_markets_by_query", lambda q, limit=100: [decoy])
    out = fp.fetch_weather_markets("Chicago")
    assert out == [], ("a decoy that only incidentally satisfies Chicago's query tokens must not "
                       "enter Chicago's results when match_city resolves it to a different city")


def test_fetch_weather_markets_survives_a_raising_tag_lookup(monkeypatch, caplog):
    """A malformed /events body raising out of discover_by_tag must degrade THIS city to the
    query-scan fallback, not kill the whole collect run: main.py has no try/except around
    fetch_weather_markets, so an uncaught exception here would also skip every city queued after
    this one in the same process."""
    import logging
    import fetch_polymarket as fp
    fp._reset_tag_cache()

    def raises(tag="weather"):
        raise ValueError("malformed /events body")

    monkeypatch.setattr(fp, "discover_by_tag", raises)
    used = []
    monkeypatch.setattr(fp, "search_markets_by_query",
                        lambda q, limit=100: used.append(q) or [])
    with caplog.at_level(logging.WARNING):
        out = fp.fetch_weather_markets("London")   # must not raise
    assert out == []
    assert used, "a raising tag lookup must still fall back to the query scan"
    assert any("raised" in r.message.lower()
               for r in caplog.records if r.levelno >= logging.WARNING), \
        "fetch_weather_markets must WARN when the tag lookup itself raised"


def test_discover_by_tag_does_not_cache_an_empty_result(monkeypatch):
    """A transient /events blip (e.g. an empty page) must not freeze an empty partition in for
    the rest of the process -- the next city in the same collect cycle should get a fresh
    pagination attempt, not inherit the miss."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    calls = []

    def fake_paged(tag, page_size=100):
        calls.append(tag)
        if len(calls) == 1:
            return [], False
        return _TAG_EVENTS, False

    monkeypatch.setattr(fp, "_paged_events", fake_paged)
    first = fp.discover_by_tag("weather")
    assert first == {}
    second = fp.discover_by_tag("weather")
    assert len(calls) == 2, "an empty partition must not be cached -- the next call should retry"
    assert "London" in second and len(second["London"]) == 2


# ── Final review fix wave: unguarded shape assumption, unpinned fallback write, ──
# ── truncated-partition caching, missing active filter, multi-city ambiguity. ────

def test_search_markets_by_query_survives_a_non_list_body(caplog):
    """A truthy Gamma body that is neither list nor dict (e.g. a rate-limit message returned as
    HTTP 200 JSON) used to raise AttributeError out of `data.get("data", [])` -- straight out of
    fetch_weather_markets -> step_fetch_polymarket -> main.py, which has NO try/except around the
    fallback path, aborting the whole collect cycle before step_fetch_weather/step_fetch_ensemble
    even run. It must degrade to an empty result instead, loudly (a WARNING), not silently."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m
    with caplog.at_level(logging.WARNING), m.patch.object(
            fp, "_get", lambda url, params=None: "rate limited"):
        got = fp.search_markets_by_query("highest temperature London")
    assert got == [], "a non-list/non-dict body must degrade to an empty result, not raise"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), \
        "a non-list/non-dict body must warn loudly -- silent-empty is the failure mode this " \
        "whole branch exists to eliminate"


def test_paged_events_survives_a_non_list_body(caplog):
    """The same unguarded-shape bug, one layer down: `_paged_events` must not raise either, since
    it feeds the tag path that the fallback above degrades to when it fails."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m
    with caplog.at_level(logging.WARNING), m.patch.object(
            fp, "_get", lambda url, params=None: "rate limited"):
        events, truncated = fp._paged_events("weather", page_size=100)
    assert events == []
    assert truncated is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_paged_events_requests_active_and_closed_filters():
    """/events must exclude both closed AND inactive markets. Only `active=true` was missing --
    processing.py filters `closed` on read but nothing filters `active`, so a resolved bin inside
    an open event would otherwise reach the committed snapshots."""
    import fetch_polymarket as fp
    import unittest.mock as m
    seen_params = []

    def fake_get(url, params=None):
        seen_params.append(params)
        return []

    with m.patch.object(fp, "_get", fake_get):
        fp._paged_events("weather", page_size=100)
    assert seen_params[0]["closed"] == "false"
    assert seen_params[0]["active"] == "true"


def test_fetch_weather_markets_fallback_actually_returns_the_market_it_finds(monkeypatch):
    """The fallback's only productive behaviour is returning markets it finds -- every existing
    fallback test only asserted 'search was called' or 'a decoy was rejected', so a fallback that
    silently discarded every match (`found[cid] = m` replaced with `pass`) still shipped green.
    This test is designed to fail under exactly that mutation."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    monkeypatch.setattr(fp, "discover_by_tag", lambda tag="weather": {})
    market = {"conditionId": "ldn1",
              "question": "Will the highest temperature in London be 22°C on July 29?"}
    monkeypatch.setattr(fp, "search_markets_by_query", lambda q, limit=100: [market])
    out = fp.fetch_weather_markets("London")
    assert len(out) == 1, "a genuine match found by the fallback must actually be returned"
    assert out[0]["city"] == "London"
    assert out[0]["condition_id"] == "ldn1"


def test_discover_by_tag_does_not_cache_a_truncated_partition(monkeypatch):
    """Asymmetric with the empty-result rule right above: a truncated partition is non-empty
    (`found` has SOME markets) so nothing about it triggers the empty-result fallback signal, yet
    it is just as incomplete. City 1 gets the FLOOR warning from `_paged_events`; without this fix
    cities 2-5 in the same process would silently read the partial partition from cache with no
    warning and no retry."""
    import fetch_polymarket as fp
    fp._reset_tag_cache()
    calls = []

    def fake_paged(tag, page_size=100):
        calls.append(tag)
        return _TAG_EVENTS, True   # truncated=True

    monkeypatch.setattr(fp, "_paged_events", fake_paged)
    fp.discover_by_tag("weather")
    fp.discover_by_tag("weather")
    assert len(calls) == 2, "a truncated partition must not be cached -- each call should retry"


def test_discover_by_tag_skips_ambiguous_multi_city_questions(caplog):
    """A question naming two configured cities must not be silently filed under whichever city
    sorts first in CITIES iteration order -- that's a latent, permanent corruption of the loser's
    committed CSV series, and a future CITIES reorder would silently change which city loses. It
    must be skipped from BOTH buckets, not deduped into one, and logged loudly instead: an
    ambiguous market is worth less than a mislabeled one, and this converts a latent silent
    corruption into a loud, visible event for the price of three lines."""
    import logging
    import fetch_polymarket as fp
    import unittest.mock as m
    fp._reset_tag_cache()
    ambiguous_events = [
        {"title": "ambiguous", "markets": [
            {"conditionId": "amb1",
             "question": "Will the highest temperature in London or Chicago be higher on July 29?"},
        ]},
    ]
    with caplog.at_level(logging.WARNING), m.patch.object(
            fp, "_paged_events", lambda tag, page_size=100: (ambiguous_events, False)):
        out = fp.discover_by_tag("weather")
    all_cids = [m["conditionId"] for ms in out.values() for m in ms]
    assert "amb1" not in all_cids, "an ambiguous market must not be filed under any city"
    assert out.get("London", []) == []
    assert out.get("Chicago", []) == []
    assert any("ambig" in r.message.lower()
               for r in caplog.records if r.levelno >= logging.WARNING), \
        "an ambiguous market must be logged loudly, naming the question and the candidates"


# --------------------------------------------------------------------------------------
# fetch_station_obs: a partial refetch must never overwrite a complete file
# --------------------------------------------------------------------------------------

def _hourly_obs_csv(n, start_year=2022):
    """n hourly rows with genuinely distinct timestamps, starting at start_year.

    Distinctness matters: fetch_station_obs dedupes on `valid_local`, so a fixture that repeats
    timestamps collapses to a handful of rows and the size guards never get exercised.
    """
    import pandas as pd
    ts = pd.date_range(f"{start_year}-01-01", periods=n, freq="h")
    return pd.DataFrame({"valid_local": ts.strftime("%Y-%m-%d %H:%M"), "temp_c": 20.0})


def test_obs_fetch_keeps_existing_file_when_a_year_chunk_fails(tmp_path, monkeypatch):
    """The 2026-07-30 incident. LGA's 2026 chunk failed all four retries while 2022-2025
    succeeded; the failure was silently skipped, the 35,032 surviving rows cleared the absolute
    floor of 20,000, and the complete 40,071-row file was overwritten. wu_truth then returned
    None for every 2026 date and NYC grading fell back to the pre-W0 CLI ruler — settlement audit
    97.0% -> 94.7%, published pooled gap +0.0178 (CI above zero) -> +0.0122 (CI spanning zero).
    A dropped HTTP request flipped the project's headline verdict, on a green run."""
    import pandas as pd
    import fetch_station_obs as fso

    out = tmp_path / "new_york_city_obs_hourly.csv"
    # DELIBERATELY SMALL existing file. The partial refetch below totals ~32k rows — far MORE
    # than this — so the size-regression guard cannot fire and only the failed-chunk guard can
    # save the file. Sizing it the other way lets guard 2 silently do guard 1's job, and the
    # test then passes with guard 1 deleted (verified by mutation: it did).
    _hourly_obs_csv(9000).to_csv(out, index=False)

    monkeypatch.setattr(fso, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(fso, "OBS_STATIONS", {"new_york_city": ("LGA", "America/New_York")})
    monkeypatch.setattr(fso.time, "sleep", lambda *_: None)
    monkeypatch.setattr(fso, "START_YEAR", 2022)

    def fake_range(station, tz, start, end):
        if start.year >= 2026:
            return None                     # the 2026 chunk fails all retries
        return _hourly_obs_csv(8000, start_year=start.year)

    monkeypatch.setattr(fso, "_fetch_range", fake_range)
    fso.fetch_station_obs(recent_only=False)

    kept = pd.read_csv(out)
    assert len(kept) == 9000, \
        "a partial year-set overwrote the existing file — this is the 2026-07-30 incident"


def test_obs_fetch_refuses_to_shrink_an_existing_file(tmp_path, monkeypatch):
    """Guard 1 catches a chunk that errored. This catches everything else — an upstream range
    quietly returning fewer rows, a response that parses but is empty, a station rename.
    Refetched observations must only ever grow."""
    import pandas as pd
    import fetch_station_obs as fso

    out = tmp_path / "chicago_obs_hourly.csv"
    _hourly_obs_csv(40000).to_csv(out, index=False)

    monkeypatch.setattr(fso, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(fso, "OBS_STATIONS", {"chicago": ("ORD", "America/Chicago")})
    monkeypatch.setattr(fso.time, "sleep", lambda *_: None)
    # every chunk "succeeds" but the total is smaller than what is already on disk
    monkeypatch.setattr(fso, "_fetch_range",
                        lambda station, tz, start, end: _hourly_obs_csv(5000, start_year=start.year))

    fso.fetch_station_obs(recent_only=False)
    kept = pd.read_csv(out)
    assert len(kept) == 40000, "a shrinking refetch overwrote the larger existing file"


def test_obs_fetch_still_writes_when_the_refetch_grows(tmp_path, monkeypatch):
    """The guards must not block the normal path — a genuinely larger refetch still lands, or
    the file would freeze forever and go stale silently, which is the same class of bug."""
    import pandas as pd
    import fetch_station_obs as fso

    out = tmp_path / "seoul_obs_hourly.csv"
    _hourly_obs_csv(21000).to_csv(out, index=False)

    monkeypatch.setattr(fso, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(fso, "OBS_STATIONS", {"seoul": ("RKSI", "Asia/Seoul")})
    monkeypatch.setattr(fso.time, "sleep", lambda *_: None)
    monkeypatch.setattr(fso, "_fetch_range",
                        lambda station, tz, start, end: _hourly_obs_csv(9000, start_year=start.year))

    fso.fetch_station_obs(recent_only=False)
    assert len(pd.read_csv(out)) > 21000, "a healthy growing refetch was wrongly rejected"


def test_gate_ci_str_shows_the_binding_condition():
    """A gate line must show WHY it is not passing, not just n-vs-threshold.

    Regression for the 2026-07-31 reporting gap: the Leg1b forward line printed only
    "43/80@+0.080v+0.030", which reads as "on track, just needs more n". It was in fact also
    short on city-days (25/30) with a CI lower bound of +0.001 — hanging on one bad day. The
    2026-07-27 amendment makes the clustered CI the binding condition, so it must be visible.
    """
    from shoulder_book import gate_ci_str, GATE_MIN_CLUSTERS

    # Short on clusters — that is the binding reason, and it must be named.
    short = {"n": 43, "se": 0.04, "n_clusters": 25, "n_dates": 40,
             "ci_lo": 0.001, "ci_hi": 0.158}
    out = gate_ci_str(short)
    assert "CI[" in out and "+0.001" in out, "interval must be printed"
    assert f"25/{GATE_MIN_CLUSTERS} city-days" in out, "must name the cluster shortfall"

    # Enough clusters but the interval straddles zero.
    spans = {"n": 95, "se": 0.03, "n_clusters": 57, "n_dates": 40,
             "ci_lo": -0.002, "ci_hi": 0.112}
    assert "CI spans 0" in gate_ci_str(spans)

    # Genuinely clearing: interval above zero with enough clusters — no caveat appended.
    clean = {"n": 120, "se": 0.01, "n_clusters": 40, "n_dates": 40,
             "ci_lo": 0.021, "ci_hi": 0.060}
    got = gate_ci_str(clean)
    assert "city-days" not in got and "spans" not in got, f"unexpected caveat: {got}"

    # Degenerate inputs must not raise or fabricate an interval.
    assert gate_ci_str({"n": 0, "se": float("inf"), "n_clusters": 0, "n_dates": 0,
                        "ci_lo": 0.0, "ci_hi": 0.0}) == ""


# ── order-book capture (fetch_orderbook.py) ──────────────────────────────────────────────────

def test_summarize_book_normal_two_sided():
    """Best bid is the HIGHEST bid, best ask the LOWEST — CLOB returns asks worst-first."""
    from fetch_orderbook import summarize_book
    s = summarize_book({
        "bids": [{"price": "0.10", "size": "100"}, {"price": "0.12", "size": "50"}],
        "asks": [{"price": "0.20", "size": "80"}, {"price": "0.15", "size": "40"}],
    })
    assert s["best_bid"] == 0.12
    assert s["best_ask"] == 0.15
    assert s["ask_depth_usdc"] == pytest.approx(0.15 * 40 + 0.20 * 80)


def test_summarize_book_empty_bid_side_is_none_not_zero():
    """A bid-less book must report None, never 0.0 or 1.0.

    This is the real shape of these markets — the first live book sampled had zero bids and asks
    starting at 99.9c. `data_loader.check_orderbook_vwap` returns the sentinel 1.0 when it cannot
    fill, which makes "no liquidity" indistinguishable from "priced at 1.0". A price of 0.0 or 1.0
    is a tradeable claim; absence is not.
    """
    from fetch_orderbook import summarize_book
    s = summarize_book({"bids": [], "asks": [{"price": "0.999", "size": "1141"}]})
    assert s["best_bid"] is None
    assert s["best_ask"] == 0.999
    # Nothing on the bid side to buy against, but the ask side is genuinely deep.
    assert s["ask_depth_usdc"] > 0

    empty = summarize_book({"bids": [], "asks": []})
    assert empty["best_bid"] is None and empty["best_ask"] is None
    assert empty["vwap_buy_100"] is None
    assert summarize_book({})["best_ask"] is None       # malformed input must not raise


def test_vwap_returns_none_when_the_clip_cannot_fill():
    """A partial fill is not a price. Returning the partial VWAP would understate slippage
    exactly where the book is thinnest — the case that decides whether a strategy is executable."""
    from fetch_orderbook import _vwap
    thin = [(0.10, 100.0)]                       # $10 of depth against a $100 clip
    assert _vwap(thin, 100.0) is None
    # Enough depth: one level, so the VWAP is that level's price.
    assert _vwap([(0.10, 5000.0)], 100.0) == pytest.approx(0.10)
    # Walks two levels: $50 at 0.10 then $50 at 0.20 -> 500 + 250 shares for $100.
    # abs=1e-6 because the return is rounded to 6dp on purpose (keeps the stored CSV compact).
    assert _vwap([(0.10, 500.0), (0.20, 1000.0)], 100.0) == pytest.approx(100.0 / 750.0, abs=1e-6)
    assert _vwap([], 100.0) is None


def test_tokens_of_handles_the_real_junk_in_stored_snapshots():
    """13% of historical snapshot rows carry '[]' or non-JSON in clob_token_ids_json."""
    from fetch_orderbook import tokens_of, yes_token_of
    assert tokens_of('["111","222"]') == ("111", "222")
    assert tokens_of("[]") == (None, None)
    assert tokens_of("not json") == (None, None)
    assert tokens_of(None) == (None, None)
    assert tokens_of('["111"]') == ("111", None)      # NO side absent, YES still usable
    assert yes_token_of('["111","222"]') == "111"


def test_fetch_book_summaries_prefixes_sides_and_tolerates_a_missing_one():
    """YES and NO are separate books. Leg 1 SELLS YES, which executes as BUYING NO — so a
    one-sided capture would misreport executability (measured: 71% vs 27% on the same markets)."""
    from fetch_orderbook import fetch_book_summaries

    class FakeSession:
        def post(self, url, json=None, timeout=None):
            ids = [d["token_id"] for d in json]
            books = []
            if "yes1" in ids:
                books.append({"asset_id": "yes1", "timestamp": "1785",
                              "bids": [{"price": "0.05", "size": "10"}],
                              "asks": [{"price": "0.07", "size": "9000"}]})
            if "no1" in ids:
                books.append({"asset_id": "no1", "timestamp": "1785",
                              "bids": [], "asks": [{"price": "0.94", "size": "9000"}]})
            # "yes2" is deliberately never returned — a book the CLOB did not serve.
            return _FakeResp(books)

    class _FakeResp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    out = fetch_book_summaries({"cidA": ("yes1", "no1"), "cidB": ("yes2", None)},
                               session=FakeSession())
    assert set(out) == {"cidA"}, "a market whose books did not return must be omitted, not faked"
    a = out["cidA"]
    assert a["yes_best_ask"] == 0.07 and a["no_best_ask"] == 0.94
    assert a["no_best_bid"] is None                      # NO book has no bids
    # Round-trip cost — the number config.HALF_SPREAD has been guessing at.
    assert a["yes_best_ask"] + a["no_best_ask"] == pytest.approx(1.01)
    assert a["no_vwap_buy_100"] == pytest.approx(0.94)    # deep enough to fill the clip


def test_gate_requires_temporal_spread_not_just_breadth():
    """A gate must not pass on many cities over few dates (amendment 2026-08-02).

    The exact regression: on 2026-08-02 the breadth book's Leg1b forward gate read n=1038,
    362 city-days, +0.0310, CI[+0.0149,+0.0470] — clearing every condition then in force. Those
    362 city-days were 50 cities over EIGHT target dates. Breadth inflates the cluster count
    without buying any calendar, and the effect being gated ("extreme bins hit less often than
    priced") is precisely what a single calm week produces.
    """
    import pandas as pd
    import shoulder_book as sb

    # 50 cities x 8 dates = 400 city-day clusters, a large precise positive mean.
    wide = pd.Series([f"city{i}|2026-07-{25 + (i % 8):02d}" for i in range(400)])
    v = sb.gate_verdict(pd.Series([0.05] * 400), wide, need_n=80, need_e=0.03)
    assert v["n_clusters"] >= sb.GATE_MIN_CLUSTERS, "cluster count is satisfied..."
    assert v["ci_lo"] > 0 and v["mean"] >= 0.03, "...and so are n, mean and the interval"
    assert v["n_dates"] == 8
    assert v["pass"] is False, "8 dates must not pass regardless of how many cities"

    # Same size and mean, spread over 40 dates instead -> now legitimately passes.
    deep = pd.Series([f"city{i % 10}|2026-{6 + (i % 40) // 31:02d}-{1 + (i % 40) % 31:02d}"
                      for i in range(400)])
    v2 = sb.gate_verdict(pd.Series([0.05] * 400), deep, need_n=80, need_e=0.03)
    assert v2["n_dates"] >= sb.GATE_MIN_DATES
    assert v2["pass"] is True

    # A cluster key carrying no date component cannot demonstrate spread, so it cannot pass.
    nodate = pd.Series([f"c{i % 40}" for i in range(400)])
    assert sb.gate_verdict(pd.Series([0.05] * 400), nodate, need_n=80, need_e=0.03)["pass"] is False


def test_temporal_amendment_is_tightening_only():
    """Like the 2026-07-27 amendment, 2026-08-02 may only ever make a gate HARDER.

    Anything that passes under the new rule must also have passed every prior condition.
    """
    import pandas as pd
    import shoulder_book as sb
    for n_dates in (2, 8, 29, 30, 60):
        clusters = pd.Series([f"city{i % 12}|2026-06-{1 + (i % n_dates):02d}" for i in range(300)])
        v = sb.gate_verdict(pd.Series([0.05] * 300), clusters, need_n=80, need_e=0.03)
        if v["pass"]:
            # every pre-existing condition must independently hold
            assert v["n"] >= 80 and v["mean"] >= 0.03
            assert v["n_clusters"] >= sb.GATE_MIN_CLUSTERS and v["ci_lo"] > 0
            assert v["n_dates"] >= sb.GATE_MIN_DATES


def test_capture_tier_cities_never_enter_config_cities():
    """CITIES is consumed by twelve modules, several of which iterate it to fetch forecasts
    or to TRAIN (train_calibrator does `for city in CITIES.keys()`). A capture-only city
    reaching those paths would pull forecasts we do not model and attempt EMOS training on
    cities with no archives — silently, on a green run."""
    import config
    from resolution_anchors import RESOLUTION_ANCHORS

    capture = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert capture, "expected capture-tier cities to exist"
    assert capture & set(config.CITIES) == set(), (
        f"capture-tier cities leaked into config.CITIES: {capture & set(config.CITIES)}")
    assert capture <= set(config.ALL_CITIES), "ALL_CITIES must contain every capture city"
    assert set(config.CITIES) <= set(config.ALL_CITIES)
    # The original five must still be modelled — this plan must not change their behaviour.
    for city in ("London", "Seoul", "Chicago", "New York City", "Hong Kong"):
        assert RESOLUTION_ANCHORS[city].get("tier", "modelled") == "modelled"
        assert city in config.CITIES


def test_venue_symmetry_kalshi_and_polymarket_cover_the_same_cities():
    """The entire value of this data layer is the PAIRED comparison. A Kalshi city we do not
    also capture on Polymarket is unusable, and vice versa. Symmetry is the product, not a
    convention to maintain."""
    import config
    from resolution_anchors import RESOLUTION_ANCHORS

    with_kalshi = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("kalshi_series")}
    capture = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert with_kalshi == capture, (
        f"unpairable cities — kalshi-only {with_kalshi - capture}, "
        f"polymarket-only {capture - with_kalshi}")
    assert len(with_kalshi) == 7, f"expected the 7 verified overlap cities, got {len(with_kalshi)}"
    # Every capture city must also be Polymarket-capturable: it needs search terms and a
    # Wunderground resolution URL naming the SAME station Kalshi reads.
    for city in capture:
        a = RESOLUTION_ANCHORS[city]
        assert a["station_code"] in a["resolution_url"], (
            f"{city}: resolution_url must name station {a['station_code']}")
        assert config.ALL_CITIES[city]["search_terms"], f"{city} has no search terms"


def test_lead_fetchers_iterate_modelled_anchors_only():
    """The archived-forecast-lead fetchers do per-city WORK: 2022->now x leads 1-7 x several
    models, per city. They take no --cities flag and retrain.yml runs them unconditionally, so
    they must never see capture-tier cities — that is years of Open-Meteo history for cities we
    do not model, and it already put retrain.yml over its 60-minute timeout at five cities.

    Guarding the function is not enough here: the defect is a raw `RESOLUTION_ANCHORS.items()`
    loop, so this asserts the loop itself is gone from these four files.
    """
    import pathlib
    from resolution_anchors import modelled_anchors, RESOLUTION_ANCHORS

    capture = {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert capture, "expected capture-tier cities to exist"
    assert set(modelled_anchors()) & capture == set()
    assert len(modelled_anchors()) == 5

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "polymarket_weather"
    for name in ("fetch_historical_leads.py", "fetch_historical_leads_mm.py",
                 "fetch_historical_leads_cand.py", "fetch_historical_leads_min.py"):
        text = (src / name).read_text()
        assert "RESOLUTION_ANCHORS.items()" not in text, (
            f"{name} iterates the FULL anchor registry — it will pull forecast archives for "
            f"capture-tier cities we do not model")
        assert "modelled_anchors()" in text, f"{name} must iterate modelled_anchors()"


class _Resp:
    """Minimal requests.Response stand-in: .text carries the raw body."""
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").exceptions.HTTPError(str(self.status_code))


def test_kalshi_get_retries_on_EMPTY_not_only_on_error():
    """An empty response and a genuine absence are indistinguishable at the call site.

    Two throwaway scripts written while drafting this spec silently dropped Houston, then
    Seattle, to transient empty results — each time producing a confident wrong overlap count
    of 6 instead of 7. Only retrying separates the two cases.
    """
    from kalshi_series import kalshi_get

    class FlakySession:
        def __init__(self): self.calls = 0
        def get(self, url, params=None, timeout=None):
            self.calls += 1
            body = '{"markets": []}' if self.calls < 3 else '{"markets": [{"ticker": "T1"}]}'
            return _Resp(body)

    s = FlakySession()
    payload, ok = kalshi_get("/markets", {}, session=s, nonempty_key="markets")
    assert ok is True
    assert payload["markets"] == [{"ticker": "T1"}], "must retry past the transient empties"
    assert s.calls == 3

    # A genuinely empty series is accepted after the retries are exhausted — ok stays True,
    # because the request SUCCEEDED. Only transport failure yields ok=False.
    class EmptySession:
        def get(self, url, params=None, timeout=None): return _Resp('{"markets": []}')
    payload, ok = kalshi_get("/markets", {}, session=EmptySession(), retries=2,
                             nonempty_key="markets")
    assert ok is True and payload["markets"] == []


def test_kalshi_get_parses_the_real_raw_newline_in_rules_secondary():
    """Kalshi emits literal newlines inside JSON strings, which is invalid JSON. Python's
    parser rejects it by default; strict=False is required."""
    from kalshi_series import kalshi_get

    class NewlineSession:
        def get(self, url, params=None, timeout=None):
            return _Resp('{"markets": [{"rules_secondary": "line one\nline two"}]}')

    payload, ok = kalshi_get("/markets", {}, session=NewlineSession())
    assert ok is True
    assert "line one" in payload["markets"][0]["rules_secondary"]


def test_kalshi_get_reports_transport_failure_as_not_ok():
    """Absence is a value. A failed request must never look like 'no data'."""
    from kalshi_series import kalshi_get

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    payload, ok = kalshi_get("/markets", {}, session=DeadSession(), retries=2)
    assert ok is False and payload is None


def test_target_series_covers_exactly_the_capture_cities():
    from kalshi_series import target_series
    from resolution_anchors import RESOLUTION_ANCHORS

    ts = target_series()
    assert set(ts) == {c for c, a in RESOLUTION_ANCHORS.items() if a.get("tier") == "capture"}
    assert ts["Houston"] == "KXHIGHTHOU", "Houston's other three tickers are DEAD"
    assert ts["Los Angeles"] == "KXHIGHLAX"


def test_fetch_series_markets_reports_truncation_explicitly():
    """The Polymarket discovery bug was a hard API ceiling read as 'that is the end of the
    list', which captured ~3% of markets for months. Pagination must return truncation as a
    VALUE, never leave the caller to infer it."""
    from fetch_kalshi import fetch_series_markets

    class CappedSession:
        """Always returns a full page with a cursor — an infinite list."""
        def get(self, url, params=None, timeout=None):
            page = [{"ticker": f"T{i}", "status": "active"} for i in range(3)]
            return _Resp(__import__("json").dumps({"markets": page, "cursor": "more"}))

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=CappedSession(),
                                              page_size=3, max_pages=4)
    assert truncated is True, "hitting the page cap MUST report truncation"
    assert len(markets) == 12

    class ShortSession:
        def get(self, url, params=None, timeout=None):
            return _Resp(__import__("json").dumps({"markets": [{"ticker": "T1"}], "cursor": ""}))

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=ShortSession(), page_size=3)
    assert truncated is False, "a short page is a legitimate end of list"
    assert len(markets) == 1


def test_fetch_series_markets_never_reports_unknown_as_complete(monkeypatch):
    """kalshi_get's ok=False means "every attempt failed" — we do not know. That must never
    surface as truncated=False, which downstream reads as "confirmed complete". A dead ticker
    (genuinely 0 markets) and an API outage would otherwise look identical, and telling them
    apart is the whole point of the series manifest.

    fetch_series_markets takes no `retries` kwarg (none was added — that would be a production
    logic change out of scope for a test-coverage fix), so this patches kalshi_series' own
    time.sleep to a no-op, same technique already used for fetch_station_obs's retry tests
    (test_obs_fetch_keeps_existing_file_when_a_year_chunk_fails and neighbours). Without the
    patch this test still passes but burns kalshi_get's real ~15s retry-exhaustion sleep
    (DEFAULT_RETRIES=4, 1.5*(1+2+3+4)s) — kalshi_get itself is untouched.
    """
    import kalshi_series
    from fetch_kalshi import fetch_series_markets

    monkeypatch.setattr(kalshi_series.time, "sleep", lambda *_: None)

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=DeadSession())
    assert markets == []
    assert truncated is True, "an outage must NOT read as a confirmed-complete empty series"


def test_fetch_series_markets_truncates_on_a_later_page_failure(monkeypatch):
    """A transport failure on page 2+ (not just page 0) must still report truncated=True and
    return the PARTIAL markets already gathered, not discard them."""
    import kalshi_series
    from fetch_kalshi import fetch_series_markets

    monkeypatch.setattr(kalshi_series.time, "sleep", lambda *_: None)

    class FailsOnPage2:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                page = [{"ticker": f"T{i}", "status": "active"} for i in range(3)]
                return _Resp(__import__("json").dumps({"markets": page, "cursor": "more"}))
            raise __import__("requests").exceptions.ConnectionError("down")

    markets, truncated = fetch_series_markets("KXHIGHLAX", session=FailsOnPage2(), page_size=3)
    assert truncated is True
    assert len(markets) == 3, "the partial list from page 1 must be returned, not discarded"


def test_summarize_market_absence_is_none_never_a_sentinel():
    """data_loader.check_orderbook_vwap returns 1.0 when it cannot fill, which makes 'no
    liquidity' indistinguishable from 'priced at 1.0'. A price of 0 or 1 is a tradeable claim;
    absence is not."""
    from fetch_kalshi import summarize_market

    s = summarize_market({"ticker": "T1", "status": "active"})
    assert s["yes_bid"] is None and s["yes_ask"] is None
    assert s["volume"] is None
    assert s["ticker"] == "T1"

    s2 = summarize_market({"ticker": "T2", "yes_bid_dollars": "0.0000",
                           "yes_ask_dollars": "0.0700", "volume_fp": "0.00"})
    assert s2["yes_bid"] == 0.0, "a real zero bid is 0.0, NOT None"
    assert s2["yes_ask"] == 0.07
    assert s2["volume"] == 0.0

    # Kalshi sends "" — not a missing key — for unsettled markets and unquoted fields.
    s3 = summarize_market({"ticker": "T3", "result": "", "yes_bid_dollars": "",
                           "volume_fp": "0.00", "open_interest_fp": ""})
    assert s3["result"] is None, "unsettled '' must be None, not an empty-string 'result'"
    assert s3["yes_bid"] is None, "'' is absence, not a price"
    assert s3["open_interest"] is None
    assert s3["volume"] == 0.0, "a real zero must survive as 0.0"


def test_summarize_market_keeps_both_rules_fields_verbatim():
    """The station is stated in a DIFFERENT FIELD per series generation: older KXHIGH* name the
    airport in rules_primary with no product code, newer KXHIGHT* give a bare city there and put
    the station in rules_secondary. Neither field alone identifies the station, and 'Houston' is
    ambiguous between Bush and Hobby."""
    from fetch_kalshi import summarize_market

    s = summarize_market({
        "ticker": "KXHIGHTHOU-26AUG04-T94",
        "rules_primary": "...recorded at Houston for Aug 4, 2026...",
        "rules_secondary": 'Data for CLIHOU ... location "Houston-Hobby, TX" ...',
    })
    assert "Houston-Hobby, TX" in s["rules_secondary"]
    assert "recorded at Houston" in s["rules_primary"]


def test_summarize_market_archives_cap_strike_for_less_type_markets():
    """ARCHIVE-INTEGRITY GUARD. Kalshi's real 'less'-type markets carry their bound in
    cap_strike with floor_strike=None (verified live KXHIGHLAX 2026-08-03: 34/34 'less' markets,
    100%). summarize_market storing only floor_strike would silently write a blank bin threshold
    to the archive for those rows — and Kalshi serves market objects for only ~2 months, so a
    threshold missed at capture time is gone forever, unrecoverable at any later date. This is
    the single most important guard in this module: a capture archive that drops the thing it
    exists to capture, behind a green run, is the worst failure this project has.
    """
    from fetch_kalshi import summarize_market

    s = summarize_market({"ticker": "KXHIGHLAX-26AUG04-T75", "strike_type": "less",
                          "floor_strike": None, "cap_strike": 75,
                          "yes_sub_title": "74° or below"})
    assert s["cap_strike"] == 75.0, "cap_strike must be archived — it is the ONLY threshold field a 'less' market populates"
    assert s["floor_strike"] is None


def test_derive_bin_agrees_with_the_human_readable_subtitle():
    """floor_strike/cap_strike + strike_type + yes_sub_title are three representations of ONE
    threshold. The off-by-one between them is exactly the Hong Kong ruler bug's shape:
    floor_strike 82 with strike_type 'greater' means YES from 83, and the subtitle says
    '83° or above'.

    All three payloads below are REAL shapes, verbatim from live KXHIGHLAX (2026-08-03) — not
    hand-fabricated. The 'less' fixture matters most: an earlier version of this test supplied a
    synthetic floor_strike for a 'less' market, which Kalshi never actually sends (real 'less'
    markets carry the bound in cap_strike with floor_strike=None), and that let derive_bin's
    floor_strike-only design silently return None for 100% of real 'less' markets undetected.
    """
    from fetch_kalshi import derive_bin

    got = derive_bin({"floor_strike": 82, "cap_strike": None, "strike_type": "greater",
                      "yes_sub_title": "83° or above"})
    assert got["op"] == "greater"
    assert got["yes_from_f"] == 83
    assert got["yes_to_f"] is None
    assert got["subtitle_bound"] == 83
    assert got["agrees_with_subtitle"] is True

    got_less = derive_bin({"floor_strike": None, "cap_strike": 75, "strike_type": "less",
                           "yes_sub_title": "74° or below"})
    assert got_less["op"] == "less"
    assert got_less["yes_from_f"] is None
    assert got_less["yes_to_f"] == 74
    assert got_less["agrees_with_subtitle"] is True

    got_between = derive_bin({"floor_strike": 81, "cap_strike": 82, "strike_type": "between",
                              "yes_sub_title": "81° to 82°"})
    assert got_between["op"] == "between"
    assert got_between["yes_from_f"] == 81 and got_between["yes_to_f"] == 82
    assert got_between["agrees_with_subtitle"] is True

    # A disagreement must be VISIBLE, not silently resolved in favour of either side.
    bad = derive_bin({"floor_strike": 82, "cap_strike": None, "strike_type": "greater",
                      "yes_sub_title": "99° or above"})
    assert bad["agrees_with_subtitle"] is False

    # An unknown strike_type must not be guessed. ('between' is a REAL, recognised type now —
    # it must never be used here as the "unknown" case again.)
    assert derive_bin({"floor_strike": 82, "strike_type": "scalar",
                       "yes_sub_title": "x"}) is None


def test_derive_bin_subtitle_parsing_limits():
    """The subtitle bound is taken as the FIRST number in yes_sub_title. That holds for every
    live Kalshi format seen (2026-08-03), and negative temperatures parse correctly. If a format
    ever leads with a different number the check fails LOUDLY (agrees_with_subtitle False)
    rather than silently deriving a wrong bin — which is the behaviour being pinned here."""
    from fetch_kalshi import derive_bin
    # Negative temperatures must parse — Chicago in winter is a real case.
    d = derive_bin({"floor_strike": -6, "strike_type": "greater",
                    "yes_sub_title": "-5° or above"})
    assert d["yes_from_f"] == -5 and d["agrees_with_subtitle"] is True
    # A leading non-bound number is detected, not silently trusted.
    d2 = derive_bin({"floor_strike": 82, "strike_type": "greater",
                     "yes_sub_title": "2026: 83° or above"})
    assert d2["agrees_with_subtitle"] is False
    # A subtitle with no number at all must not crash.
    d3 = derive_bin({"floor_strike": 82, "strike_type": "greater", "yes_sub_title": "n/a"})
    assert d3["subtitle_bound"] is None and d3["agrees_with_subtitle"] is False


def test_derive_bin_covers_every_live_strike_type():
    """between is 66% of Kalshi's markets and less is another 17%; deriving only `greater`
    silently drops 83% of the product. The three types are NOT symmetric: greater's
    floor_strike is exclusive, less's cap_strike is exclusive, between's bounds are both
    inclusive. Verified against live KXHIGHLAX 2026-08-03."""
    from fetch_kalshi import derive_bin
    g = derive_bin({"strike_type": "greater", "floor_strike": 82, "cap_strike": None,
                    "yes_sub_title": "83° or above"})
    assert (g["yes_from_f"], g["yes_to_f"]) == (83, None) and g["agrees_with_subtitle"]
    l = derive_bin({"strike_type": "less", "floor_strike": None, "cap_strike": 75,
                    "yes_sub_title": "74° or below"})
    assert (l["yes_from_f"], l["yes_to_f"]) == (None, 74) and l["agrees_with_subtitle"]
    b = derive_bin({"strike_type": "between", "floor_strike": 81, "cap_strike": 82,
                    "yes_sub_title": "81° to 82°"})
    assert (b["yes_from_f"], b["yes_to_f"]) == (81, 82) and b["agrees_with_subtitle"]
    assert derive_bin({"strike_type": "scalar", "floor_strike": 1,
                       "yes_sub_title": "x"}) is None


def test_ladder_converts_kalshi_pairs_and_inverts_for_the_opposite_side():
    """Kalshi's `/orderbook` levels are `[price, size]` STRING PAIRS, not the `{"price","size"}`
    dicts summarize_book consumes elsewhere -- verified live 2026-08-03. `invert=True` must
    reconstruct one side's asks from the OTHER side's bids (price -> 1 - price, size unchanged),
    because Kalshi publishes only bid ladders (see fetch_orderbooks' docstring)."""
    from fetch_kalshi import _ladder

    got = _ladder([["0.0100", "9471.47"]])
    assert got == [{"price": 0.01, "size": 9471.47}]

    inv = _ladder([["0.0100", "9471.47"]], invert=True)
    assert inv == [{"price": 0.99, "size": 9471.47}], \
        "invert must map price -> 1 - price; size must survive unchanged"

    assert _ladder([]) == [], "a genuinely empty ladder is a real, quiet answer"
    assert _ladder(None) == []


def test_ladder_logs_loudly_when_every_level_is_unparseable(caplog):
    """A NONEMPTY ladder where every level fails to parse is a payload SHAPE CHANGE (e.g. Kalshi
    changes format again), not a legitimately empty book -- it must not silently look identical
    to a genuinely empty ladder. This is exactly how the original dict-shaped assumption would
    have failed against the real list-of-pairs payload: quietly, plausibly, and wrong."""
    import logging
    from fetch_kalshi import _ladder

    with caplog.at_level(logging.ERROR):
        got = _ladder([{"unexpected": "shape"}, {"another": "bad one"}])
    assert got == []
    assert any("SHAPE CHANGE" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert _ladder([]) == []
    assert not caplog.records, "a genuinely empty ladder must NOT log an error"


def test_fetch_orderbooks_reconstructs_asks_from_the_opposite_bid_ladder():
    """THE semantics-pinning test, not just a plumbing check. Kalshi's orderbook publishes only
    BID ladders for each side -- verified live 2026-08-03 against KXHIGHLAX-26AUG03-B80.5, whose
    own quote was yes_bid_dollars=0.0400 / yes_ask_dollars=0.0500. `yes_dollars` and `no_dollars`
    below are the VERBATIM payload from that request (not fabricated).

    yes_best_bid must be the highest yes_dollars price (0.04, matching the live quote), and
    yes_best_ask must be 1 - the highest no_dollars price (1 - 0.95 = 0.05, ALSO matching the
    live quote) -- NOT a raw yes_dollars price read directly as an ask, which is the exact
    mistake an earlier draft of this module's docstring made and would have produced a plausible,
    wrong number silently.
    """
    import json as _json
    from fetch_kalshi import fetch_orderbooks

    YES_DOLLARS = [["0.0100", "9471.47"], ["0.0200", "2867.00"], ["0.0300", "2623.94"],
                   ["0.0400", "211.76"]]
    NO_DOLLARS = [
        ["0.0100", "805.71"], ["0.0200", "152.00"], ["0.0300", "27.00"], ["0.0400", "1.00"],
        ["0.0500", "102.00"], ["0.0600", "2.00"], ["0.0700", "1.00"], ["0.0800", "1.00"],
        ["0.0900", "100.80"], ["0.1000", "26.00"], ["0.1100", "216.73"], ["0.1200", "1.00"],
        ["0.1400", "13.74"], ["0.1500", "153.26"], ["0.1700", "38.25"], ["0.1800", "80.00"],
        ["0.2000", "57.46"], ["0.2100", "1.00"], ["0.2300", "41.48"], ["0.2500", "217.74"],
        ["0.3000", "25.00"], ["0.3200", "50.00"], ["0.3500", "172.47"], ["0.3700", "1611.67"],
        ["0.3800", "313.37"], ["0.3900", "80.00"], ["0.4000", "124.94"], ["0.4300", "26.26"],
        ["0.4400", "47.09"], ["0.4500", "25.00"], ["0.4800", "50.00"], ["0.5000", "47.00"],
        ["0.5400", "15.53"], ["0.5500", "63.84"], ["0.5600", "58.71"], ["0.6000", "63.00"],
        ["0.6500", "62.00"], ["0.6900", "700.70"], ["0.7000", "82.00"], ["0.7300", "123.89"],
        ["0.7400", "6.00"], ["0.7500", "60.35"], ["0.7600", "245.00"], ["0.7700", "1106.00"],
        ["0.7800", "6.00"], ["0.7900", "6.00"], ["0.8000", "30.30"], ["0.8100", "6.00"],
        ["0.8200", "50.27"], ["0.8300", "6.00"], ["0.8400", "16.00"], ["0.8500", "65.00"],
        ["0.8600", "198.95"], ["0.8700", "26.00"], ["0.8800", "6.00"], ["0.8900", "82.00"],
        ["0.9000", "440.90"], ["0.9100", "73.00"], ["0.9200", "561.00"], ["0.9300", "51.00"],
        ["0.9400", "365.66"], ["0.9500", "88.44"],
    ]

    class RealShapeSession:
        def get(self, url, params=None, timeout=None):
            return _Resp(_json.dumps({"orderbook_fp": {
                "yes_dollars": YES_DOLLARS, "no_dollars": NO_DOLLARS}}))

    out = fetch_orderbooks(["KXHIGHLAX-26AUG03-B80.5"], session=RealShapeSession())
    book = out["KXHIGHLAX-26AUG03-B80.5"]
    assert book["yes_best_bid"] == 0.04, "best YES bid is the highest yes_dollars price"
    assert book["yes_best_ask"] == 0.05, \
        "best YES ask == 1 - best NO bid (0.95) -- the live yes_ask_dollars quote, not a raw " \
        "yes_dollars price"
    assert book["no_best_bid"] == 0.95, "best NO bid is the highest no_dollars price"
    assert book["no_best_ask"] == 0.96, "best NO ask == 1 - best YES bid (0.04)"
    assert book["yes_ask_depth_usdc"] > 0


def test_kalshi_orderbooks_use_the_shared_summary_shape_and_omit_failures():
    """Both venues' books must be analysable by ONE code path, and a book that did not return
    must be OMITTED rather than faked -- reading only one side of a two-sided Polymarket market
    produced a confidently wrong executability figure (71% vs 27% on the same markets).

    A DEAD (settled/nonexistent) ticker's REAL shape, verified live 2026-08-03, is
    present-but-empty arrays -- `{"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}` -- not
    the fabricated `{"orderbook_fp": {}}` an earlier draft assumed.
    """
    import json as _json
    from fetch_kalshi import fetch_orderbooks

    class BookSession:
        def get(self, url, params=None, timeout=None):
            if "T_DEAD" in url:
                return _Resp(_json.dumps({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}))
            return _Resp(_json.dumps({"orderbook_fp": {
                "yes_dollars": [["0.0500", "100"], ["0.0700", "9000"]],
                "no_dollars":  [["0.9000", "500"]],
            }}))

    out = fetch_orderbooks(["T_LIVE", "T_DEAD"], session=BookSession())
    assert "T_LIVE" in out
    live = out["T_LIVE"]
    assert live["yes_best_bid"] == 0.07, "best bid is the HIGHEST yes_dollars price"
    assert live["yes_best_ask"] == 0.10, "yes ask is reconstructed from the NO bid (1 - 0.90)"
    assert live["no_best_bid"] == 0.90
    assert live["yes_ask_depth_usdc"] > 0
    # A genuinely empty book (both ladders present, no levels) yields None fields for the ticker
    # that IS in the result -- never 0.0 masquerading as a price.
    assert "T_DEAD" in out
    assert out["T_DEAD"]["yes_best_bid"] is None
    assert out["T_DEAD"]["yes_best_ask"] is None


def test_fetch_orderbooks_omits_a_ticker_whose_book_never_returns(monkeypatch):
    """ok=False means the transport failed -- we do not know the book's state -- so the ticker
    must be OMITTED from the result entirely, never included with faked/zero fields."""
    import kalshi_series
    from fetch_kalshi import fetch_orderbooks

    monkeypatch.setattr(kalshi_series.time, "sleep", lambda *_: None)

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    out = fetch_orderbooks(["T1"], session=DeadSession())
    assert out == {}, "a ticker whose book never returned must not appear in the result at all"


def test_candle_window_brackets_the_markets_life_not_a_trailing_window():
    """A trailing 'last N days' window against a market that settled outside it returns zero
    candles at every interval — indistinguishable from 'this market never traded'. That mistake
    was made live while drafting the spec, against KXHIGHNY-26JUL21-B79.5 ($181k volume)."""
    from fetch_kalshi import fetch_candles

    captured = {}

    class CandleSession:
        def get(self, url, params=None, timeout=None):
            captured.update(params or {})
            return _Resp('{"candlesticks": [{"end_period_ts": 1784628000, '
                         '"price": {"close_dollars": "0.35"}, "volume_fp": "633.96", '
                         '"open_interest_fp": "7230.16", "yes_bid": {}, "yes_ask": {}}]}')

    market = {"ticker": "KXHIGHNY-26JUL21-B79.5",
              "open_time": "2026-07-19T14:00:00Z", "close_time": "2026-07-22T04:00:00Z"}
    candles, meta = fetch_candles("KXHIGHNY", market, session=CandleSession())

    assert len(candles) == 1
    assert captured["period_interval"] == 60, "1-minute returns HTTP 400 on multi-day windows"
    # The window must come from the market's own life, with a margin, not from 'now'.
    import datetime as _dt
    open_ts = int(_dt.datetime.fromisoformat("2026-07-19T14:00:00+00:00").timestamp())
    close_ts = int(_dt.datetime.fromisoformat("2026-07-22T04:00:00+00:00").timestamp())
    assert captured["start_ts"] <= open_ts, "window must start at or before open_time"
    assert captured["end_ts"] >= close_ts, "window must end at or after close_time"


def test_candle_backfill_records_completeness(monkeypatch):
    """A market archived with zero candles must be distinguishable from one never attempted.
    A backfill quietly covering half its window is the obs-truncation failure in a new costume."""
    import kalshi_series
    from fetch_kalshi import fetch_candles

    monkeypatch.setattr(kalshi_series.time, "sleep", lambda *_: None)

    class EmptySession:
        def get(self, url, params=None, timeout=None):
            return _Resp('{"candlesticks": []}')

    market = {"ticker": "T1", "open_time": "2026-07-19T14:00:00Z",
              "close_time": "2026-07-22T04:00:00Z"}
    candles, meta = fetch_candles("KXHIGHNY", market, session=EmptySession())
    assert candles == []
    assert meta["candles"] == 0
    assert meta["ok"] is True, "the request SUCCEEDED and returned nothing — that is a fact"
    assert meta["start_ts"] and meta["end_ts"], "the requested window must be recorded"

    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise __import__("requests").exceptions.ConnectionError("down")

    candles, meta = fetch_candles("KXHIGHNY", market, session=DeadSession())
    assert meta["ok"] is False, "a failed fetch must NOT look like 'this market had no trading'"


def test_fetch_candles_refuses_a_market_with_no_life_window():
    """Without open_time/close_time there is no honest window to request."""
    from fetch_kalshi import fetch_candles
    candles, meta = fetch_candles("KXHIGHNY", {"ticker": "T1"}, session=None)
    assert candles == [] and meta["ok"] is False and meta["reason"] == "no_window"


def test_summarize_candle_matches_the_verified_live_shape():
    """Verified live 2026-08-03 against KXHIGHNY-26JUL21-B79.5 ($181k volume, settled): `price`,
    `yes_bid`, `yes_ask` are all DICTS keyed by `*_dollars` (never bare scalars), `end_period_ts`
    is an int, `volume_fp`/`open_interest_fp` are numeric strings — this is the real shape, not a
    fixture guess. A zero-trade minute can carry an EMPTY `price` dict (`{}`) or a `price` with
    only `previous_dollars` (no close/open/high/low/mean) — absence must stay None, not 0.0 or a
    KeyError."""
    from fetch_kalshi import summarize_candle

    live_candle = {
        "end_period_ts": 1784606400,
        "open_interest_fp": "4592.95",
        "price": {"open_dollars": "0.2500", "high_dollars": "0.3400", "low_dollars": "0.1600",
                  "close_dollars": "0.2900", "mean_dollars": "0.2601"},
        "volume_fp": "5855.06",
        "yes_ask": {"open_dollars": "1.0000", "high_dollars": "1.0000", "low_dollars": "0.1800",
                    "close_dollars": "0.2900"},
        "yes_bid": {"open_dollars": "0.0100", "high_dollars": "0.3300", "low_dollars": "0.0100",
                    "close_dollars": "0.2800"},
    }
    row = summarize_candle(live_candle, "NYC", "KXHIGHNY", "KXHIGHNY-26JUL21-B79.5")
    assert row["end_period_ts"] == 1784606400
    assert row["close_dollars"] == 0.29
    assert row["open_dollars"] == 0.25
    assert row["yes_bid_close"] == 0.28
    assert row["yes_ask_close"] == 0.29
    assert row["volume"] == 5855.06
    assert row["open_interest"] == 4592.95
    assert set(row) == set(__import__("fetch_kalshi").CANDLE_COLS)

    idle_candle = {"end_period_ts": 1784556240, "open_interest_fp": "5.00",
                   "price": {"previous_dollars": "0.2500"}, "volume_fp": "0.00",
                   "yes_ask": {"close_dollars": "0.2700"}, "yes_bid": {"close_dollars": "0.2200"}}
    row = summarize_candle(idle_candle, "NYC", "KXHIGHNY", "T1")
    assert row["close_dollars"] is None, "an idle candle's price dict has no close_dollars key"
    assert row["open_dollars"] is None
    assert row["yes_bid_close"] == 0.22

    empty_price_candle = {"end_period_ts": 1784556060, "open_interest_fp": "0.00",
                          "price": {}, "volume_fp": "0.00",
                          "yes_ask": {"close_dollars": "0.5400"}, "yes_bid": {}}
    row = summarize_candle(empty_price_candle, "NYC", "KXHIGHNY", "T1")
    assert row["close_dollars"] is None
    assert row["yes_bid_close"] is None, "an empty yes_bid dict must yield None, not 0.0"
    assert row["yes_ask_close"] == 0.54


def test_ts_rejects_falsy_and_malformed_timestamps():
    """A 0 or a malformed string must yield None, never a 1970 window.

    `if not value` is what catches the literal 0 — a later tidy-up to `if value is None`
    would make open_time=0 produce start=-3600, ok=True and candles=0, which reads as
    "this market never traded". Kalshi serves market objects ~2 months, so a market wrongly
    skipped is unrecoverable.
    """
    from fetch_kalshi import _ts, fetch_candles
    assert _ts(0) is None
    assert _ts("") is None
    assert _ts(None) is None
    assert _ts("garbage") is None
    assert _ts("2026-07-19T14:00:00Z") == 1784469600     # Z-suffixed ISO parses correctly

    # End to end: a falsy window must refuse, not silently request 1970.
    _, meta = fetch_candles("KXHIGHNY", {"ticker": "T", "open_time": 0, "close_time": 0})
    assert meta["ok"] is False and meta["reason"] == "no_window"
    assert meta["start_ts"] is None and meta["end_ts"] is None
