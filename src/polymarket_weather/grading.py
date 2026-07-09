"""grading.py — resolve weather markets against the RESOLUTION-STATION observation.

A market settles on its named station's daily reading (e.g. Incheon RKSI for Seoul),
NOT a forecast grid cell. This grades backtests against that station truth, read from
`data/weather/<slug>_historical_actuals.csv` (resolution-faithful station truth from
NWS CLI / IEM METAR / HKO, written by `fetch_historical_truth.py`; stations named in
`resolution_anchors.py`). It replaces the old per-script grid-based `fetch_actual_weather`,
which fetched a forecast cell — wrong in general, and especially for Seoul whose forecast
point (Bucheon) is deliberately offset from the resolution station (Incheon).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from resolution_anchors import RESOLUTION_ANCHORS

_TRUTH_DIR = Path(__file__).resolve().parent / "data" / "weather"

# canonical city name + aliases (e.g. "NYC", "HongKong") -> data-file slug
_SLUG = {}
# city name + aliases -> the market's resolution_unit (e.g. "whole °F", "0.1 °C").
_UNIT = {}
for _city, _a in RESOLUTION_ANCHORS.items():
    _s = _city.replace(" ", "_").lower()
    _SLUG[_city] = _s
    _UNIT[_city] = _a.get("resolution_unit", "whole °C")
    for _alias in _a.get("aliases", []):
        _SLUG[_alias] = _s
        _UNIT[_alias] = _a.get("resolution_unit", "whole °C")


def native_round(temp_c, unit):
    """Round a °C temperature onto the market's native resolution grid.

    Markets settle in the units printed on the question, not in °C, so a °F market must be
    converted to °F and rounded to a whole degree BEFORE the threshold comparison — rounding
    in °C (the old behaviour) can flip boundary days. Units come from `resolution_unit` in the
    anchor: 'whole °F', '0.1 °C' (tenths), or 'whole °C' (default).
    """
    if unit and "°F" in unit:
        return round(temp_c * 9.0 / 5.0 + 32.0)   # whole °F
    if unit and unit.strip().startswith("0.1"):
        return round(temp_c, 1)                    # tenths of a °C
    return round(temp_c)                           # whole °C


def resolves_yes(city, target_date, question, bin_temp_c):
    """Did the market resolve YES? True / False, or None if station truth is unavailable.

    Grades the resolution-station observation against the question's threshold, both rounded to
    the market's NATIVE unit (see `native_round`). Direction is read from the question text:
    'higher/above/more/at least' => >=, 'lower/below/less/at most' => <=, else exact ==.
    """
    actual_c = fetch_actual_weather(city, target_date, question)
    if actual_c is None:
        return None
    unit = _UNIT.get(city, "whole °C")
    actual_n = native_round(actual_c, unit)
    thresh_n = native_round(float(bin_temp_c), unit)
    q = str(question).lower()
    if "higher" in q or "above" in q or "more" in q or "at least" in q:
        return actual_n >= thresh_n
    if "lower" in q or "below" in q or "less" in q or "at most" in q:
        return actual_n <= thresh_n
    return actual_n == thresh_n


@lru_cache(maxsize=None)
def _truth(slug, kind):
    """Map date_local -> station temp (°C) for kind in {'max','min'}; {} if unavailable."""
    col = "temp_max_c" if kind == "max" else "temp_min_c"
    path = _TRUTH_DIR / f"{slug}_historical_actuals.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if col not in df.columns:
        return {}
    df = df.dropna(subset=[col])
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.strftime("%Y-%m-%d")
    return dict(zip(df["date_local"], df[col]))


def fetch_actual_weather(city, target_date, question=""):
    """Resolution-station observed temperature (°C) for the date, or None if unknown.

    Uses the daily MIN for 'lowest temperature' markets, else the daily MAX.
    Drop-in replacement for the old grid grader: `question` is optional (defaults to MAX).
    Returns None beyond the truth file's coverage (unresolved date) so callers skip it.
    """
    slug = _SLUG.get(city)
    if slug is None:
        return None
    kind = "min" if "lowest" in str(question).lower() else "max"
    val = _truth(slug, kind).get(str(target_date))
    return None if val is None else float(val)
