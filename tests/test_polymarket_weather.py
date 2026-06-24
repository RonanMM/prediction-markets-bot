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

