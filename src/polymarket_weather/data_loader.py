import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import json

def discover_cities(data_dir: Path) -> list[str]:
    """Cities that have any market snapshots — legacy file OR daily partitions.

    This decides which cities the ENTIRE analysis runs on. Globbing only `*_snapshots.csv` after
    the 2026-09-03 daily-partition migration returns nothing, and the run prints "No *_snapshots
    .csv files found" and exits 0 — the whole pipeline doing nothing, successfully.
    """
    import re
    data_dir = data_dir / "polymarket"
    names = set()
    for f in data_dir.glob("*_snapshots*.csv"):
        stem = f.stem
        # `{city}_snapshots` or `{city}_snapshots_YYYY-MM-DD`
        m = re.match(r"^(.+)_snapshots(?:_\d{4}-\d{2}-\d{2})?$", stem)
        if m:
            names.add(m.group(1))
    return sorted(names)


def _parse_yes(val) -> float:
    try:
        d = json.loads(str(val).replace("'", '"'))
        return float(d.get("Yes", d.get("yes", np.nan)))
    except Exception:
        return np.nan


def load_snapshots(data_dir: Path, city: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "polymarket" / f"{city}_snapshots.csv")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["end_date_iso"]   = pd.to_datetime(df["end_date_iso"],   utc=True, errors="coerce")
    df["yes_prob"]       = df["outcome_probs_json"].apply(_parse_yes)
    df["city"]           = city
    # deduplicate: keep freshest row per (condition_id, fetched_bucket)
    df["fetch_bucket"]   = df["fetched_at_utc"].dt.floor("10min")
    df = (df.sort_values("fetched_at_utc")
            .groupby(["condition_id", "fetch_bucket"], as_index=False)
            .last())
    return df


def load_daily(data_dir: Path, city: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "weather" / f"{city}_daily.csv")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    return df


def load_hourly(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    p = data_dir / "weather" / f"{city}_hourly.csv"
    # load_partitioned, NOT read_csv — the hourly forecast is one file per UTC day and the
    # legacy name no longer exists, so `p.exists()` would return None forever, silently.
    from processing import load_partitioned, partitioned_available
    if not partitioned_available(p):
        return None
    df = load_partitioned(p)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["datetime_utc"]   = pd.to_datetime(df["datetime_utc"],   utc=True, errors="coerce")
    return df


def load_daily_mm(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    """Load per-model deterministic daily Tmax forecasts ({slug}_daily_mm.csv) if
    collected. This is the exact serving input for multi-model-mean calibrations;
    older snapshots without it fall back to the ensemble-mean proxy."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    p = data_dir / "weather" / f"{slug}_daily_mm.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    for col in df.columns:
        if col.startswith("tmax_") or col.startswith("tmin_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_nbm(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    """Load NBM station guidance ({slug}_nbm.csv, US cities only). Rows are runtime-
    stamped with avail_utc, so callers can as-of join against any snapshot fetch time —
    a backtest only ever sees runs that were genuinely available."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    p = data_dir / "weather" / f"{slug}_nbm.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["avail_utc"] = pd.to_datetime(df["avail_utc"], utc=True)
    df["date_local"] = df["date_local"].astype(str)
    for col in ("nbm_tmax_txn_c", "nbm_tmax_tmp_c"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_obs_hourly(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    """Load hourly station observations ({slug}_obs_hourly.csv) for intraday
    conditioning of same-day bets. None if not collected (e.g. Hong Kong)."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    p = data_dir / "weather" / f"{slug}_obs_hourly.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["valid_local"] = pd.to_datetime(df["valid_local"])
    df["date_local"] = df["valid_local"].dt.strftime("%Y-%m-%d")
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    return df.dropna(subset=["temp_c"])


def load_ensemble(data_dir: Path, city: str) -> Optional[pd.DataFrame]:
    """Load ensemble CSV if available. Returns None if not yet fetched."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    p = data_dir / "weather" / f"{slug}_ensemble.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True)
    df["date_local"]     = pd.to_datetime(df["date_local"]).dt.normalize()
    for col in ["ens_mean", "ens_std", "ens_p10", "ens_p25",
                "ens_median", "ens_p75", "ens_p90", "ens_spread",
                "ens_min_mean", "ens_min_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def fetch_live_prices(condition_ids: list[str]) -> dict[str, float]:
    """
    Fetch current Yes-side prices for a list of condition_ids from the Gamma API.
    Returns {condition_id: current_yes_prob}.
    Uses GET /markets/{condition_id} per market (Gamma API doesn't support bulk lookup).
    """
    import requests as _req
    import time as _time

    if not condition_ids:
        return {}

    prices: dict[str, float] = {}

    for cid in condition_ids:
        try:
            resp = _req.get(
                "https://gamma-api.polymarket.com/markets",
                params={"condition_ids": cid},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue
            mkt = data[0] if isinstance(data, list) else data
            raw = mkt.get("outcomePrices")
            if not raw:
                continue
            probs = json.loads(raw) if isinstance(raw, str) else raw
            yes_p = float(probs[0])
            prices[cid] = yes_p
            _time.sleep(0.1)   # polite: ~10 req/s
        except Exception as exc:
            print(f"  [WARN] live price fetch failed for {cid[:16]}…: {exc}")

    return prices


def check_orderbook_vwap(condition_id: str, bet_side: str, target_size_usdc: float) -> float:
    """
    Gets the token_id for the bet_side, pulls the CLOB L2 orderbook,
    and calculates the effective average price (VWAP) for the target size.
    Returns the VWAP price, or 1.0 if insufficient liquidity.
    """
    import requests as _req
    import time as _time
    import json as _json
    try:
        _time.sleep(0.2)
        
        # 1. Get Token ID from Gamma
        gamma_req = _req.get("https://gamma-api.polymarket.com/markets", params={"condition_ids": condition_id})
        gamma_req.raise_for_status()
        gamma_resp = gamma_req.json()
        
        if not gamma_resp: 
            print(f"  [DEBUG] Gamma API returned empty for condition {condition_id[:8]}...")
            return 1.0
        
        market = gamma_resp[0] if isinstance(gamma_resp, list) else gamma_resp
        
        raw_outcomes = market.get("outcomes", "[]")
        outcomes = _json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        
        raw_tokens = market.get("clobTokenIds", "[]")
        clob_tokens = _json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        
        token_id = None
        for i, outcome in enumerate(outcomes):
            if outcome == bet_side and i < len(clob_tokens):
                token_id = clob_tokens[i]
                break
                
        if not token_id: 
            print(f"  [DEBUG] No token_id found for side '{bet_side}' in condition {condition_id[:8]}...")
            return 1.0
        
        _time.sleep(0.2)
        
        # 2. Get Order Book from CLOB
        clob_req = _req.get("https://clob.polymarket.com/book", params={"token_id": token_id})
        clob_req.raise_for_status()
        clob_resp = clob_req.json()
        
        asks = clob_resp.get("asks", [])
        if not asks:
            print(f"  [DEBUG] CLOB returned 0 asks (empty orderbook) for token {token_id[:8]}...")
            return 1.0
            
        asks.sort(key=lambda x: float(x["price"]))
        
        filled_usdc = 0.0
        total_shares = 0.0
        
        for ask in asks:
            price = float(ask["price"])
            size_shares = float(ask["size"])
            cost_usdc = size_shares * price
            
            if filled_usdc + cost_usdc >= target_size_usdc:
                needed_usdc = target_size_usdc - filled_usdc
                total_shares += needed_usdc / price
                filled_usdc += needed_usdc
                break
            else:
                total_shares += size_shares
                filled_usdc += cost_usdc
        
        if filled_usdc < target_size_usdc or total_shares == 0:
            print(f"  [DEBUG] Not enough liquidity to fill ${target_size_usdc:.2f}. Only found ${filled_usdc:.2f}")
            return 1.0  
            
        return round(target_size_usdc / total_shares, 4)
    except Exception as e:
        print(f"  [WARN] Orderbook check failed for {condition_id[:8]}...: {e}")
        return 1.0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════


