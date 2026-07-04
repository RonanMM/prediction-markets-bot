import re
import numpy as np
from typing import Optional
from scipy.stats import t as student_t
from models import MarketBin

# QUESTION PARSER
# ══════════════════════════════════════════════════════════════════════════════

# ── Celsius patterns ────────────────────────────────────────────────────────
_RE_GTE_C   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:higher|above|more)",   re.I)
_RE_LTE_C   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:lower|below|less)",    re.I)
_RE_EXACT_C = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*C\s+on\b",                         re.I)
# "at least X°C", "exceed X°C", "reach X°C", "X°C or above/higher"
_RE_GTE_C_2 = re.compile(
    r"(?:at\s+least|exceed[s]?|reach(?:es)?)\s+(-?\d+(?:\.\d+)?)\s*°?\s*C", re.I)
_RE_GTE_C_3 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:above|higher|more)",        re.I)
_RE_LTE_C_2 = re.compile(
    r"(?:at\s+most|no\s+more\s+than)\s+(-?\d+(?:\.\d+)?)\s*°?\s*C",   re.I)
_RE_LTE_C_3 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*C\s+or\s+(?:below|lower|less)",         re.I)

# ── Fahrenheit patterns ──────────────────────────────────────────────────────
_RE_GTE_F   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:higher|above|more)",   re.I)
_RE_LTE_F   = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:lower|below|less)",    re.I)
_RE_EXACT_F = re.compile(
    r"be\s+(-?\d+(?:\.\d+)?)\s*°?\s*F\s+on\b",                         re.I)
_RE_RANGE_F = re.compile(r"between\s+(\d+)-(\d+)\s*°?\s*F",            re.I)
_RE_GTE_F_2 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:above|higher|more)",        re.I)
_RE_LTE_F_2 = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F\s+or\s+(?:below|lower|less)",         re.I)

# 0.5°F converted to °C — used as bin half-width for Fahrenheit markets
_HALF_WIDTH_F_IN_C = 0.5 * 5.0 / 9.0   # ≈ 0.2778°C
_HALF_WIDTH_C      = 0.5                # standard Celsius bin


def _f2c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def parse_question(q: str) -> Optional[dict]:
    """
    Returns: {
        condition:  'exact'|'gte'|'lte'|'range',
        temp_c:     float,          # target / midpoint in °C
        half_width: float,          # bin half-width in °C (0.5 for °C, 0.278 for °F)
        temp_lo:    float (range only),
        temp_hi:    float (range only),
    }
    Tries progressively broader patterns so the most specific match wins.
    """
    # ── Celsius gte ───────────────────────────────────────────────────────────
    for pat in (_RE_GTE_C, _RE_GTE_C_2, _RE_GTE_C_3):
        m = pat.search(q)
        if m:
            return {"condition": "gte", "temp_c": float(m.group(1)),
                    "half_width": _HALF_WIDTH_C}
    # ── Celsius lte ───────────────────────────────────────────────────────────
    for pat in (_RE_LTE_C, _RE_LTE_C_2, _RE_LTE_C_3):
        m = pat.search(q)
        if m:
            return {"condition": "lte", "temp_c": float(m.group(1)),
                    "half_width": _HALF_WIDTH_C}
    # ── Celsius exact ─────────────────────────────────────────────────────────
    m = _RE_EXACT_C.search(q)
    if m:
        return {"condition": "exact", "temp_c": float(m.group(1)),
                "half_width": _HALF_WIDTH_C}
    # ── Fahrenheit range ──────────────────────────────────────────────────────
    m = _RE_RANGE_F.search(q)
    if m:
        lo, hi = _f2c(float(m.group(1))), _f2c(float(m.group(2)))
        return {"condition": "range", "temp_c": (lo + hi) / 2,
                "temp_lo": lo, "temp_hi": hi, "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit gte ────────────────────────────────────────────────────────
    for pat in (_RE_GTE_F, _RE_GTE_F_2):
        m = pat.search(q)
        if m:
            return {"condition": "gte", "temp_c": _f2c(float(m.group(1))),
                    "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit lte ────────────────────────────────────────────────────────
    for pat in (_RE_LTE_F, _RE_LTE_F_2):
        m = pat.search(q)
        if m:
            return {"condition": "lte", "temp_c": _f2c(float(m.group(1))),
                    "half_width": _HALF_WIDTH_F_IN_C}
    # ── Fahrenheit exact ──────────────────────────────────────────────────────
    m = _RE_EXACT_F.search(q)
    if m:
        return {"condition": "exact", "temp_c": _f2c(float(m.group(1))),
                "half_width": _HALF_WIDTH_F_IN_C}
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ENGINE — Student-t with adaptive sigma
# ══════════════════════════════════════════════════════════════════════════════


def _cdf(x: float, mu: float, sigma: float, nu: float,
         floor: float = None, ceiling: float = None) -> float:
    """Student-t CDF; falls back to Gaussian when nu>30.

    Censoring (same-day bets):
      * `floor` f (Tmax markets: the running observed daily max) — T = max(f, Z):
        F_T(x) = 0 below the floor; a point mass of F_Z(f) sits AT the floor.
      * `ceiling` c (Tmin markets: the running observed daily min) — T = min(c, Z):
        F_T(x) = 1 at/above the ceiling; the point mass P(Z >= c) sits AT the ceiling.
    """
    import scipy.stats
    if floor is not None and x < floor:
        return 0.0
    if ceiling is not None and x >= ceiling:
        return 1.0
    if nu > 30:
        return float(scipy.stats.norm.cdf(x, loc=mu, scale=sigma))
    return float(student_t.cdf((x - mu) / sigma, df=nu))


def _bin_prob(temp_c: float, mu: float, sigma: float, nu: float,
              half_width: float = 0.5, floor: float = None,
              ceiling: float = None) -> float:
    """P(temp_c - half_width < actual <= temp_c + half_width).
    Use half_width=0.278 for Fahrenheit markets (0.5°F in °C)."""
    upper_bound = temp_c + half_width
    lower_bound = temp_c - half_width
    return (_cdf(upper_bound, mu, sigma, nu, floor, ceiling)
            - _cdf(lower_bound, mu, sigma, nu, floor, ceiling))


def _condition_prob(parsed: dict, mu: float, sigma: float, nu: float,
                    floor: float = None, ceiling: float = None) -> float:
    c  = parsed["condition"]
    t  = parsed["temp_c"]
    hw = parsed.get("half_width", 0.5)
    if c == "exact":  return _bin_prob(t, mu, sigma, nu, hw, floor, ceiling)
    if c == "gte":    return 1.0 - _cdf(t - hw, mu, sigma, nu, floor, ceiling)
    if c == "lte":    return _cdf(t + hw, mu, sigma, nu, floor, ceiling)
    if c == "range":  return (_cdf(parsed["temp_hi"], mu, sigma, nu, floor, ceiling) -
                              _cdf(parsed["temp_lo"], mu, sigma, nu, floor, ceiling))
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_pmf(bins: list[MarketBin],
                    mu: float, sigma: float, nu: float,
                    temps_range: tuple = (-5, 40),
                    floor: float = None,
                    ceiling: float = None) -> tuple[dict, dict, float]:
    """
    Build market PMF and forecast PMF over a shared temperature support,
    respecting gte/lte bins as constraints.

    Returns:
        market_pmf  : {temp_c -> normalised probability}
        forecast_pmf: {temp_c -> normalised probability}
        consistency : how well bins sum to 1.0 (1.0 = perfect)
    """
    exact = [b for b in bins if b.condition == "exact"]
    gte   = [b for b in bins if b.condition == "gte"]
    lte   = [b for b in bins if b.condition == "lte"]

    # ── Step 1: exact bins → raw market PMF backbone ──────────────────────
    if not exact:
        return {}, {}, 0.0

    raw_mkt  = {b.temp_c: b.yes_prob for b in exact}
    raw_sum  = sum(raw_mkt.values())
    if raw_sum < 0.05:
        return {}, {}, 0.0

    # ── Step 2: measure internal consistency ──────────────────────────────
    # gte bin P(≥T) should ≈ sum of exact bins at temps ≥ T
    consistency_err = 0.0
    n_constraints   = 0
    for b in gte:
        implied = sum(v for k, v in raw_mkt.items() if k >= b.temp_c)
        consistency_err += abs(implied - b.yes_prob)
        n_constraints   += 1
    for b in lte:
        implied = sum(v for k, v in raw_mkt.items() if k <= b.temp_c)
        consistency_err += abs(implied - b.yes_prob)
        n_constraints   += 1

    consistency = 1.0 - (consistency_err / max(n_constraints, 1))

    # ── Step 3: estimate missing bin mass ─────────────────────────────────
    # All observed exact temps
    obs_temps = sorted(raw_mkt.keys())

    # Expected total from boundary bins
    max_gte_temp = max((b.temp_c for b in gte), default=None)
    min_lte_temp = min((b.temp_c for b in lte), default=None)

    # Upper residual: P(> max_exact_temp) from gte bin or forecast tail
    if max_gte_temp is not None:
        max_gte_bin = max(gte, key=lambda b: b.temp_c)
        upper_residual = max(0.0, max_gte_bin.yes_prob -
                             sum(v for k, v in raw_mkt.items() if k >= max_gte_temp))
    else:
        # Estimate from forecast: probability beyond highest exact bin
        upper_residual = max(0.0, 1.0 - _cdf(max(obs_temps) + 0.5, mu, sigma, nu, floor, ceiling))

    # Lower residual: P(< min_exact_temp) from lte bin or forecast tail
    if min_lte_temp is not None:
        min_lte_bin = min(lte, key=lambda b: b.temp_c)
        lower_residual = max(0.0, min_lte_bin.yes_prob -
                             sum(v for k, v in raw_mkt.items() if k <= min_lte_temp))
    else:
        lower_residual = max(0.0, _cdf(min(obs_temps) - 0.5, mu, sigma, nu, floor, ceiling))

    # Total available probability for exact bins + estimated residuals
    total_prob = raw_sum + upper_residual + lower_residual
    total_prob = max(total_prob, raw_sum)  # can't be less than what we see

    # ── Step 4: normalise market PMF ──────────────────────────────────────
    market_pmf = {k: v / total_prob for k, v in raw_mkt.items()}

    # ── Step 5: forecast PMF over same support ────────────────────────────
    fc_raw = {t: _bin_prob(t, mu, sigma, nu, floor=floor, ceiling=ceiling) for t in obs_temps}
    fc_sum = sum(fc_raw.values()) + upper_residual + lower_residual
    fc_sum = max(fc_sum, sum(fc_raw.values()))
    forecast_pmf = {t: v / fc_sum for t, v in fc_raw.items()}

    return market_pmf, forecast_pmf, max(0.0, min(1.0, consistency))


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

