"""grading.py — resolve weather markets against the RESOLUTION-STATION observation.

A market settles on its named station's daily reading (e.g. Incheon RKSI for Seoul),
NOT a forecast grid cell. This grades backtests against that station truth, read from
`data/weather/<slug>_historical_actuals.csv` (resolution-faithful station truth from
NWS CLI / IEM METAR / HKO, written by `fetch_historical_truth.py`; stations named in
`resolution_anchors.py`). It replaces the old per-script grid-based `fetch_actual_weather`,
which fetched a forecast cell — wrong in general, and especially for Seoul whose forecast
point (Bucheon) is deliberately offset from the resolution station (Incheon).

W0 (2026-07-12): for NYC/Chicago the primary truth is now the WUNDERGROUND-style
reconstruction from hourly METARs (`wu_truth.py`) — the markets resolve on the WU page, whose
extremes can differ from the CLI by 1°F at bin boundaries (4/60 settlements audited wrong
before this fix; see docs/EDGE_MEGAPLAN.md §10a and audit_settlements.py).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from resolution_anchors import RESOLUTION_ANCHORS
from pmf import parse_question, resolves_yes_temp
import wu_truth
from wu_truth import wu_daily_extreme

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

    A5: direction/outcome is derived from pmf.parse_question — the SAME parser the engine
    prices from — so grading and pricing can never diverge (this fixes 'between X-Y°F' range
    grading and the 'no more than'/'exceed'/'reach' direction bugs the old substring scan had).
    Observation and threshold are both rounded to the market's NATIVE unit (see `native_round`).
    If parse_question cannot classify the question, fall back to the legacy substring scan on
    bin_temp_c (NOT exact ==) so already-graded rows keep their outcome.
    """
    actual_c = fetch_actual_weather(city, target_date, question)
    if actual_c is None:
        return None
    unit = _UNIT.get(city, "whole °C")
    actual_n = native_round(actual_c, unit)

    parsed = parse_question(str(question))
    if parsed is not None:
        return resolves_yes_temp(parsed, actual_n, unit, native_round)

    # Legacy fallback for questions parse_question cannot classify.
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
    """SETTLEMENT-faithful observed temperature (°C) for the date, or None if unknown.

    Uses the daily MIN for 'lowest temperature' markets, else the daily MAX.
    W0 (edge megaplan §10a): the markets resolve on wunderground.com, whose daily extremes are
    the hourly-METAR extremes over the local calendar day — NOT the NWS CLI's 1-minute/LST-day
    reading our historical-actuals feed stores. For the settlement-validated US cities the
    WU-style reconstruction (`wu_truth.wu_daily_extreme`) is primary and the CLI value is the
    fallback plus a glitch guard (a lone bad METAR must not fabricate an extreme; if the two
    sources diverge implausibly, keep the CLI). Other cities are unchanged.
    Returns None beyond the truth coverage (unresolved date) so callers skip it.
    """
    slug = _SLUG.get(city)
    if slug is None:
        return None
    kind = "min" if "lowest" in str(question).lower() else "max"
    cli = _truth(slug, kind).get(str(target_date))
    cli = None if cli is None else float(cli)
    wu = wu_daily_extreme(city, str(target_date), kind)
    if wu is not None and (cli is None or abs(wu - cli) <= wu_truth.SANITY_MAX_DIVERGENCE_C):
        return wu
    return cli
