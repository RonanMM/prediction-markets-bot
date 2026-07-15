import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from models import Opportunity, MarketBin
from config import (MIN_EDGE, MIN_LIQUIDITY, MIN_MARKET_PRICE, KELLY_FRACTION,
                    FEE_RATE, MAX_KELLY_PER_BET, MAX_KELLY_PER_GROUP, MAX_TOTAL_KELLY, CITY_NAMES,
                    SHRINK_WEIGHT, LIVE_BUCKETS, bucket_key)
from signals import score_opportunity, compute_momentum, forecast_convergence, volume_recency_signal, market_staleness
from predictors.emos import EMOSPredictor
from predictors.ensemble import EnsemblePredictor
from predictors.nwp_fallback import spread_sigma_boost
from pmf import parse_question, parse_question_date, _bin_prob, _condition_prob, reconstruct_pmf
from data_loader import (load_snapshots, load_daily, load_daily_mm, load_ensemble,
                         load_nbm, load_obs_hourly, fetch_live_prices, check_orderbook_vwap)

logger = logging.getLogger(__name__)

# A digit immediately before an optional degree sign then a word-boundary F — matches
# "62°F"/"63 F"/"48F" but NOT month names ("February") or "of" (E6 unit sniff).
_F_UNIT_RE = re.compile(r"[0-9]\s*°?\s*F\b")

def _days_from_now(o: Opportunity) -> float:
    from datetime import datetime, timezone
    td = pd.to_datetime(o.target_date).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (td - now).total_seconds() / 86400.0)

def _backbone_stats(market_bins: list[MarketBin], mu_fallback: float) -> tuple[float, float]:
    """α5 coherence deviation + market mode over the DISJOINT backbone (exact + range).

    Keying these off `exact` bins alone was a bug: US-city markets ("between X-Y°F") are
    RANGE-only, so `exact_bins` was empty, giving pmf_dev = |0 - 1| = 1.0 (maximal fake
    incoherence → always-on α5 bonus) and market_mode_c = mu (mode_shift_c collapsed to 0)
    on every NYC/Chicago row. "No backbone bins" is UNKNOWN coherence, not maximal → 0.0.
    """
    backbone = [b for b in market_bins if b.condition in ("exact", "range")]
    if not backbone:
        return 0.0, mu_fallback
    pmf_dev = abs(sum(b.yes_prob for b in backbone) - 1.0)
    market_mode_c = max(backbone, key=lambda b: b.yes_prob).temp_c
    return pmf_dev, market_mode_c


def _kelly_size(opp: Opportunity, kelly_fraction: float = KELLY_FRACTION, fee: float = FEE_RATE) -> float:
    """
    Fractional Kelly with fee adjustment.

    On Polymarket you pay `their_prob` per share and receive $1 if correct.
    With a fee on the net profit, effective payout per share = (1 - fee).
    So net odds b = ((1 - their_prob) / their_prob) * (1 - fee).
    """
    p = opp.their_prob
    if not (np.isfinite(p) and np.isfinite(opp.our_prob)):   # E4: never size a NaN opportunity
        return 0.0
    if p <= 1e-4 or p >= (1.0 - 1e-4):
        return 0.0
    b = ((1.0 - p) / p) * (1.0 - fee)
    if b <= 0:
        return 0.0
    q   = opp.our_prob
    raw = kelly_fraction * (b * q - (1.0 - q)) / b
    return float(np.clip(raw, 0.0, MAX_KELLY_PER_BET))


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def apply_group_kelly_cap(opps: list[Opportunity],
                           bankroll: float) -> list[Opportunity]:
    """
    For each (city, target_date) group, scale down kelly fractions so total
    group exposure ≤ MAX_KELLY_PER_GROUP * bankroll.
    Also flags bets that are near-redundant (betting Yes on 14°C and 15°C and ≥14°C).
    """
    from collections import defaultdict

    groups: dict[str, list[Opportunity]] = defaultdict(list)
    for o in opps:
        groups[o.group_key].append(o)

    result = []
    for gkey, group in groups.items():
        # Sort by alpha_score descending
        group.sort(key=lambda x: x.alpha_score, reverse=True)

        # Remove near-duplicate bets: same side, adjacent temp bins.
        # Key on actual bin temperature, using correct Celsius or Fahrenheit threshold.
        filtered = []
        for o in group:
            # E6: sniff the unit with a digit-anchored regex (the old `"f" in question` matched
            # "February"/"of"), and use `<= spacing` (the old strict `< spacing` never fired,
            # since adjacent bins are exactly one unit apart: 1.0°C or 5/9°C for a 1°F step).
            is_f = bool(_F_UNIT_RE.search(o.question))
            threshold = (5.0 / 9.0) if is_f else 1.0
            if any(prev.bet_side == o.bet_side
                   and abs(prev.bin_temp_c - o.bin_temp_c) <= threshold + 1e-9
                   for prev in filtered):
                continue
            filtered.append(o)

        # Scale kelly to group cap
        total_raw_kelly = sum(o.kelly for o in filtered)
        if total_raw_kelly > MAX_KELLY_PER_GROUP:
            scale = MAX_KELLY_PER_GROUP / total_raw_kelly
            for o in filtered:
                o.kelly = round(o.kelly * scale, 4)

        result.extend(filtered)

    return result


def apply_portfolio_cap(opps: list[Opportunity]) -> list[Opportunity]:
    """
    Hard cap on total portfolio Kelly across all groups.
    After group caps are applied, total exposure can still far exceed bankroll
    (e.g. 5 cities × 7 dates × 20% = 700%). This enforces MAX_TOTAL_KELLY.
    """
    total = sum(o.kelly for o in opps)
    if total > MAX_TOTAL_KELLY:
        scale = MAX_TOTAL_KELLY / total
        for o in opps:
            o.kelly = round(o.kelly * scale, 4)
    return opps


def _create_opportunity(
    b: MarketBin, city: str, target_date_ts, fetch_time, days_fwd: float,
    mu: float, sigma: float, nu: float, s_boost: float, sigma_source: str,
    f_prob_raw: float, m_prob_raw: float, m_prob: float, edge: float, bet_side: str,
    our_prob: float, their_prob: float, mom: dict, fc_var: float,
    vol_rec: float, stale: dict, consistency: float, pmf_dev: float,
    exact_bins: list, market_mode_c: float, kelly_fraction: float,
    floor: float = None, ceiling: float = None,
    temp_lo_c: float = None, temp_hi_c: float = None,
    model_prob_side: float = None, shrink_weight: float = 1.0,
) -> Opportunity:
    opp = Opportunity(
        city           = CITY_NAMES.get(city, city),
        city_raw       = city,
        condition_id   = b.condition_id,
        question       = b.question[:80],
        target_date    = str(target_date_ts.date()),
        fetched_at     = fetch_time,
        days_ahead     = round(days_fwd, 2),
        forecast_mu    = round(mu, 2),
        forecast_sigma = round(sigma, 3),
        forecast_nu    = round(nu, 1),
        sigma_boost    = round(s_boost, 3),
        sigma_source   = sigma_source,
        forecast_prob  = round(f_prob_raw, 4),
        market_prob    = round(m_prob, 4),
        market_prob_raw= round(b.yes_prob, 4),
        edge           = round(edge, 4),
        abs_edge       = round(abs(edge), 4),
        bet_side       = bet_side,
        our_prob       = round(our_prob, 4),
        their_prob     = round(their_prob, 4),
        ema_momentum   = round(mom["ema_momentum"], 4),
        total_drift    = round(mom["total_drift"], 3),
        last_delta     = round(mom["last_delta"], 3),
        forecast_var   = round(fc_var, 4),
        volume_recency = round(vol_rec, 3),
        market_mkt_mom = stale["market_momentum"],
        hours_since_move=stale["hours_since_move"],
        is_stale       = stale["is_stale"],
        pmf_consistency= round(consistency, 4),
        pmf_sum_dev    = round(pmf_dev, 4),
        liquidity      = b.liquidity,
        volume_24h     = b.volume_24h,
        n_exact_bins   = len(exact_bins),
        market_mode_c  = market_mode_c,
        mode_shift_c   = round(mu - market_mode_c, 2),
        bin_temp_c     = b.temp_c,
        group_key      = f"{city}|{target_date_ts.date()}",
        forecast_floor = None if floor is None else round(float(floor), 2),
        forecast_ceiling = None if ceiling is None else round(float(ceiling), 2),
        temp_lo_c      = None if temp_lo_c is None else round(float(temp_lo_c), 3),
        temp_hi_c      = None if temp_hi_c is None else round(float(temp_hi_c), 3),
        model_prob_side= None if model_prob_side is None else round(float(model_prob_side), 4),
        shrink_weight  = round(float(shrink_weight), 4),
    )
    opp.alpha_score = score_opportunity(opp)
    opp.kelly       = _kelly_size(opp, kelly_fraction=kelly_fraction)
    opp.ev_per_dollar = round(our_prob / their_prob - 1.0, 4)
    opp.bucket        = bucket_key(opp.city, opp.days_ahead)
    opp.live_eligible = opp.bucket in LIVE_BUCKETS
    return opp

def analyse_city(data_dir: Path, city: str,
                 min_edge: float = MIN_EDGE,
                 min_liq:  float = MIN_LIQUIDITY,
                 use_calibrator: bool = True,
                 kelly_fraction: float = KELLY_FRACTION,
                 conflict_gating: bool = True,
                 shrink_weight: float = SHRINK_WEIGHT) -> list[Opportunity]:

    try:
        snap_df  = load_snapshots(data_dir, city)
        daily_df = load_daily(data_dir, city)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return []

    # Load ensemble — used to replace the hardcoded sigma lookup table.
    # Falls back gracefully if not yet fetched (run main.py to populate).
    ens_df = load_ensemble(data_dir, city)
    if ens_df is not None and not ens_df.empty:
        print(f"  Ensemble loaded: {len(ens_df)} rows ({ens_df['fetched_at_utc'].max()})")
    else:
        print("  Ensemble not available — falling back to NWP_PARAMS table")

    # Live deterministic multi-model forecasts — the exact serving input for the
    # calibrated predictor's multi-model mean; older snapshots fall back to ens_mean.
    mm_df = load_daily_mm(data_dir, city)

    # Hourly station observations — intraday conditioning of same-day bets
    # (running-max floor + per-hour regression). None where not collected (Hong Kong).
    obs_df = load_obs_hourly(data_dir, city)

    # NBM station guidance (US cities) — runtime-stamped, as-of joined per snapshot.
    nbm_df = load_nbm(data_dir, city)

    results: list[Opportunity] = []

    # Group by (target_date, fetch_bucket) to process each snapshot window chronologically.
    # E2: the market resolves on the day NAMED in the question, which endDateIso can be a day off
    # from (audited: only the 32 Hong Kong rows in the collected data — every other city is a
    # no-op). Derive the target from the question, falling back to endDateIso when unparseable.
    end_norm = snap_df["end_date_iso"].dt.normalize()
    if getattr(end_norm.dt, "tz", None) is not None:
        end_norm = end_norm.dt.tz_localize(None)
    if "question" in snap_df.columns:
        derived = pd.to_datetime([
            (parse_question_date(q, en) or en)
            for q, en in zip(snap_df["question"], end_norm)
        ])
    else:
        derived = pd.DatetimeIndex(end_norm)
    # parse_question_date returns naive dates, but everything downstream (the days_ahead
    # subtraction vs fetched_at_utc, the as-of joins) expects the pre-E2 tz-aware UTC midnight
    # that end_date_iso used to provide — restore it, or naive−aware subtraction raises.
    snap_df["end_date_norm"] = derived.tz_localize("UTC")
    snap_df = snap_df.sort_values("fetched_at_utc")
    groups = snap_df.groupby(["end_date_norm", "fetch_bucket"], sort=False)
    print(f"  {city}: {len(groups)} (date × snapshot) groups, "
          f"{snap_df['condition_id'].nunique()} unique markets", flush=True)

    # Primary (calibrated) predictor:
    #   use_calibrator off → pure ensemble (primary == secondary, so the averaging is a no-op)
    #   default            → EMOS / Nonhomogeneous Regression (calibrated ensemble post-processing)
    ml_predictor = EMOSPredictor() if use_calibrator else EnsemblePredictor()
    ens_predictor = EnsemblePredictor()

    for (target_date_ts, fetch_bucket_ts), group in groups:
        # Representative fetch time
        fetch_time = group["fetched_at_utc"].max()

        # Days ahead
        if pd.isna(target_date_ts):
            continue
        days_fwd = max(0.0,
                       (target_date_ts.to_pydatetime() -
                        fetch_time.to_pydatetime()).total_seconds() / 86400)

        s_boost = spread_sigma_boost(daily_df, target_date_ts, fetch_time)

        # ── Predict distribution parameters ────────────────────────────────
        try:
            dist_ml = ml_predictor.predict_distribution(
                city=city,
                target_date=target_date_ts,
                fetch_time=fetch_time,
                days_ahead=days_fwd,
                daily_df=daily_df,
                ens_df=ens_df,
                mm_df=mm_df,
                obs_df=obs_df,
                nbm_df=nbm_df
            )
            mu_ml = dist_ml.mu
            sigma_ml = dist_ml.sigma
            nu_ml = dist_ml.nu
            sigma_source = dist_ml.source
        except Exception as exc:
            # E5: don't silently swallow — a systematic predictor failure (renamed column,
            # corrupt models/{slug}_emos.json, schema drift) would otherwise be indistinguishable
            # from a genuine no-edge day. Log/count it, then skip this (date × snapshot) group.
            logger.warning("%s: predictor failed for target %s (%s: %s) — skipping group",
                           city, target_date_ts, type(exc).__name__, exc)
            continue

        try:
            dist_ens = ens_predictor.predict_distribution(
                city=city,
                target_date=target_date_ts,
                fetch_time=fetch_time,
                days_ahead=days_fwd,
                daily_df=daily_df,
                ens_df=ens_df
            )
            mu_ens = dist_ens.mu
            sigma_ens = dist_ens.sigma
            nu_ens = dist_ens.nu
        except Exception:
            mu_ens, sigma_ens, nu_ens = mu_ml, sigma_ml, nu_ml

        # Tmin distribution for "lowest temperature" markets (EMOS-only — the raw
        # ensemble predictor has no min support; None => min bins are skipped rather
        # than priced off a Tmax distribution).
        dist_min = None
        if isinstance(ml_predictor, EMOSPredictor):
            try:
                dist_min = ml_predictor.predict_distribution(
                    city=city, target_date=target_date_ts, fetch_time=fetch_time,
                    days_ahead=days_fwd, daily_df=daily_df, ens_df=ens_df,
                    mm_df=mm_df, obs_df=obs_df, kind="min")
            except Exception:
                dist_min = None

        # Keep default mu, sigma, nu for PMF reconstruction and backward compatibility
        mu = mu_ml
        sigma = sigma_ml
        nu = nu_ml

        # ── Alpha signals ──────────────────────────────────────────────────
        mom    = compute_momentum(daily_df, target_date_ts, fetch_time)   # α1
        fc_var = forecast_convergence(daily_df, target_date_ts, fetch_time)  # α7

        # ── Parse all bins in this group ───────────────────────────────────
        # "Lowest temperature" markets settle on the daily MIN and are routed to the
        # Tmin distribution (min_bins); everything else is a Tmax market. They must
        # never share a distribution or a PMF group.
        market_bins: list[MarketBin] = []
        min_bins: list[MarketBin] = []
        for _, row in group.iterrows():
            question = str(row.get("question", ""))
            parsed = parse_question(question)
            if parsed is None:
                continue
            yp = row["yes_prob"]
            if np.isnan(yp):
                continue
            bin_ = MarketBin(
                condition_id = str(row["condition_id"]),
                question     = str(row.get("question", "")),
                condition    = parsed["condition"],
                temp_c       = parsed["temp_c"],
                half_width   = parsed.get("half_width", 0.5),
                yes_prob     = float(yp),
                liquidity    = float(row.get("liquidity_usdc", 0) or 0),
                volume_24h   = float(row.get("volume_24h_usdc", 0) or 0),
                volume_total = float(row.get("volume_usdc", 0) or 0),
                temp_lo      = parsed.get("temp_lo"),   # None unless condition=='range'
                temp_hi      = parsed.get("temp_hi"),
            )
            if "lowest" in question.lower():
                min_bins.append(bin_)
            else:
                market_bins.append(bin_)

        if not market_bins and not min_bins:
            continue

        # ── Tmin ("lowest temperature") markets ────────────────────────────
        # Priced purely from the calibrated Tmin distribution (no raw-ensemble
        # averaging or conflict gating — there is no ensemble Tmin baseline), with
        # ceiling censoring on same-day bets (Tmin cannot end above the running min).
        if min_bins and dist_min is not None:
            # F4: Tmin markets score α1 momentum / α7 convergence off the MIN forecast series.
            mom_min = compute_momentum(daily_df, target_date_ts, fetch_time, col="temp_min_c")
            fc_var_min = forecast_convergence(daily_df, target_date_ts, fetch_time, col="temp_min_c")
            mu_mn, sg_mn, nu_mn = dist_min.mu, dist_min.sigma, dist_min.nu
            _, _, consistency_mn = reconstruct_pmf(
                min_bins, mu_mn, sg_mn, nu_mn, ceiling=dist_min.ceiling)
            exact_mn = [b for b in min_bins if b.condition == "exact"]
            pmf_dev_mn = abs(sum(b.yes_prob for b in exact_mn) - 1.0) if exact_mn else 0.0
            mode_mn = (max(exact_mn, key=lambda b: b.yes_prob).temp_c
                       if exact_mn else mu_mn)
            for b in min_bins:
                if b.liquidity < min_liq:
                    continue
                if b.condition == "exact":
                    f_prob = _bin_prob(b.temp_c, mu_mn, sg_mn, nu_mn, b.half_width,
                                       ceiling=dist_min.ceiling)
                elif b.condition in ("gte", "lte", "range"):
                    parsed = {"condition": b.condition, "temp_c": b.temp_c,
                              "half_width": b.half_width,
                              "temp_lo": b.temp_lo, "temp_hi": b.temp_hi}
                    f_prob = _condition_prob(parsed, mu_mn, sg_mn, nu_mn,
                                             ceiling=dist_min.ceiling)
                else:
                    continue

                m_prob_raw = b.yes_prob
                if m_prob_raw < MIN_MARKET_PRICE or m_prob_raw > (1.0 - MIN_MARKET_PRICE):
                    continue

                bet_side = "Yes" if (f_prob - m_prob_raw) > 0 else "No"
                our_prob_model = f_prob if bet_side == "Yes" else (1.0 - f_prob)
                their_prob = m_prob_raw if bet_side == "Yes" else (1.0 - m_prob_raw)
                our_prob = shrink_weight * our_prob_model + (1.0 - shrink_weight) * their_prob
                f_prob_raw = our_prob if bet_side == "Yes" else (1.0 - our_prob)
                edge = our_prob - their_prob
                if not np.isfinite(edge) or abs(edge) < min_edge:
                    continue

                vol_rec = volume_recency_signal(
                    group[group["condition_id"] == b.condition_id].iloc[0]
                    if len(group[group["condition_id"] == b.condition_id]) else pd.Series())
                stale = market_staleness(snap_df, b.condition_id, fetch_time, days_fwd)

                opp = _create_opportunity(
                    b, city, target_date_ts, fetch_time, days_fwd,
                    mu_mn, sg_mn, nu_mn, 0.0, dist_min.source,
                    f_prob_raw, m_prob_raw, m_prob_raw, edge, bet_side,
                    our_prob, their_prob, mom_min, fc_var_min, vol_rec, stale,
                    consistency_mn, pmf_dev_mn, exact_mn, mode_mn, kelly_fraction,
                    ceiling=dist_min.ceiling,
                    temp_lo_c=b.temp_lo, temp_hi_c=b.temp_hi,
                    model_prob_side=our_prob_model, shrink_weight=shrink_weight,
                )
                results.append(opp)

        if not market_bins:
            continue

        # ── Reconstruct PMF (α4/5) ─────────────────────────────────────────
        market_pmf, forecast_pmf, consistency = reconstruct_pmf(
            market_bins, mu, sigma, nu, floor=dist_ml.floor)

        exact_bins = [b for b in market_bins if b.condition == "exact"]
        # α5 coherence + market mode use the full disjoint backbone (exact + range), matching
        # reconstruct_pmf. exact_bins stays exact-only below (it drives the exact-bin pricing
        # loop and n_exact_bins, which is legitimately 0 for range-only markets).
        pmf_dev, market_mode_c = _backbone_stats(market_bins, mu)

        # ── Per-bin edge computation ───────────────────────────────────────
        # Use RAW probabilities:
        #   f_prob_raw = our model's P(temp = X)  from Student-t distribution
        #   m_prob_raw = b.yes_prob               the actual Polymarket price
        # Do NOT use normalized PMF values for edge/Kelly — those are for the
        for b in exact_bins:
            if b.liquidity < min_liq:
                continue

            # Compute raw probabilities from both models. C2: the ensemble side honors the
            # SAME same-day floor as the ML side — the running observed max is a model-free
            # fact, so an uncensored ensemble must not veto a bet the floor already settled.
            f_prob_ml = _bin_prob(b.temp_c, mu_ml, sigma_ml, nu_ml, b.half_width,
                                  floor=dist_ml.floor)
            f_prob_ens = _bin_prob(b.temp_c, mu_ens, sigma_ens, nu_ens, b.half_width,
                                   floor=dist_ml.floor)

            m_prob_raw = b.yes_prob   # actual market price you pay

            # Skip near-settled markets (price approaching 0 or 1)
            if m_prob_raw < MIN_MARKET_PRICE or m_prob_raw > (1.0 - MIN_MARKET_PRICE):
                continue

            # Compute edges and sides for both models
            edge_ml = f_prob_ml - m_prob_raw
            edge_ens = f_prob_ens - m_prob_raw
            
            bet_side_ml = "Yes" if edge_ml > 0 else "No"
            bet_side_ens = "Yes" if edge_ens > 0 else "No"

            # Conflict Gating: skip if ML and Ensemble bet sides disagree
            if conflict_gating and bet_side_ml != bet_side_ens:
                continue

            bet_side = bet_side_ml

            # Sizing: AVERAGE the two model probabilities for the bet side (ML vs Ensemble).
            # Never take the max — max-selection cherry-picks the more optimistic model and
            # systematically inflates edge, which oversizes Kelly and worsens calibration.
            # Exception: EMOS v2 already consumes the ensemble mean AND spread with honest
            # per-lead dispersion; averaging it with the raw (underdispersed) ensemble would
            # re-thin the tails, so the calibrated distribution stands alone.
            our_prob_ml = f_prob_ml if bet_side == "Yes" else (1.0 - f_prob_ml)
            our_prob_ens = f_prob_ens if bet_side == "Yes" else (1.0 - f_prob_ens)
            our_prob_model = (our_prob_ml if sigma_source.startswith("emos_v2")
                              else 0.5 * (our_prob_ml + our_prob_ens))
            their_prob = m_prob_raw if bet_side == "Yes" else (1.0 - m_prob_raw)
            # Shrink toward the market (w=1 → pure model). The market out-predicts the model, so
            # only deviate from the price in proportion to shrink_weight.
            our_prob = shrink_weight * our_prob_model + (1.0 - shrink_weight) * their_prob

            # Reconstruct f_prob_raw (our effective probability) and edge for Kelly sizing
            f_prob_raw = our_prob if bet_side == "Yes" else (1.0 - our_prob)
            edge = our_prob - their_prob

            # Minimum edge check on the (averaged, shrunk) probability
            if not np.isfinite(edge) or abs(edge) < min_edge:
                continue

            # PMF-normalized values kept for reporting/consistency signal
            m_prob = market_pmf.get(b.temp_c, m_prob_raw)
            # f_prob = forecast_pmf.get(b.temp_c, f_prob_raw)  # Removed unused variable

            # α6, α8
            vol_rec = volume_recency_signal(
                group[group["condition_id"] == b.condition_id].iloc[0]
                if len(group[group["condition_id"] == b.condition_id]) else pd.Series())
            stale   = market_staleness(snap_df, b.condition_id, fetch_time, days_fwd)

            opp = _create_opportunity(
                b, city, target_date_ts, fetch_time, days_fwd,
                mu, sigma, nu, s_boost, sigma_source,
                f_prob_raw, b.yes_prob, m_prob, edge, bet_side,
                our_prob, their_prob, mom, fc_var, vol_rec, stale,
                consistency, pmf_dev, exact_bins, market_mode_c, kelly_fraction,
                floor=dist_ml.floor,
                model_prob_side=our_prob_model, shrink_weight=shrink_weight,
            )
            results.append(opp)

        # ── Boundary + range bins (gte / lte / range) — direct CDF comparison ──
        # 'range' ("between X-Y°F") is the dominant US-city format; it was previously dropped
        # by both pricing loops, silently voiding ~83% of NYC/Chicago markets (A2).
        for b in market_bins:
            if b.condition not in ("gte", "lte", "range"):
                continue
            if b.liquidity < min_liq:
                continue

            parsed  = {"condition": b.condition, "temp_c": b.temp_c, "half_width": b.half_width,
                       "temp_lo": b.temp_lo, "temp_hi": b.temp_hi}
            f_prob_ml = _condition_prob(parsed, mu_ml, sigma_ml, nu_ml, floor=dist_ml.floor)
            f_prob_ens = _condition_prob(parsed, mu_ens, sigma_ens, nu_ens, floor=dist_ml.floor)  # C2

            m_prob_raw = b.yes_prob

            # Skip near-settled markets
            if m_prob_raw < MIN_MARKET_PRICE or m_prob_raw > (1.0 - MIN_MARKET_PRICE):
                continue

            # Compute edges and sides
            edge_ml = f_prob_ml - m_prob_raw
            edge_ens = f_prob_ens - m_prob_raw

            bet_side_ml = "Yes" if edge_ml > 0 else "No"
            bet_side_ens = "Yes" if edge_ens > 0 else "No"

            # Conflict Gating: skip if ML and Ensemble bet sides disagree
            if conflict_gating and bet_side_ml != bet_side_ens:
                continue

            bet_side = bet_side_ml

            # Sizing: AVERAGE the two model probabilities (never max — see exact-bin note above),
            # then shrink toward the market by shrink_weight. EMOS v2 stands alone (see above).
            our_prob_ml = f_prob_ml if bet_side == "Yes" else (1.0 - f_prob_ml)
            our_prob_ens = f_prob_ens if bet_side == "Yes" else (1.0 - f_prob_ens)
            our_prob_model = (our_prob_ml if sigma_source.startswith("emos_v2")
                              else 0.5 * (our_prob_ml + our_prob_ens))
            their_prob = m_prob_raw if bet_side == "Yes" else (1.0 - m_prob_raw)
            our_prob = shrink_weight * our_prob_model + (1.0 - shrink_weight) * their_prob

            f_prob_raw = our_prob if bet_side == "Yes" else (1.0 - our_prob)
            edge = our_prob - their_prob

            # Minimum edge check
            if not np.isfinite(edge) or abs(edge) < min_edge:
                continue

            vol_rec = volume_recency_signal(
                group[group["condition_id"] == b.condition_id].iloc[0]
                if len(group[group["condition_id"] == b.condition_id]) else pd.Series())
            stale   = market_staleness(snap_df, b.condition_id, fetch_time, days_fwd)

            opp = _create_opportunity(
                b, city, target_date_ts, fetch_time, days_fwd,
                mu, sigma, nu, s_boost, sigma_source,
                f_prob_raw, m_prob_raw, m_prob_raw, edge, bet_side,
                our_prob, their_prob, mom, fc_var, vol_rec, stale,
                consistency, pmf_dev, exact_bins, market_mode_c, kelly_fraction,
                floor=dist_ml.floor,
                temp_lo_c=b.temp_lo, temp_hi_c=b.temp_hi,
                model_prob_side=our_prob_model, shrink_weight=shrink_weight,
            )
            results.append(opp)

    return results




class WeatherBettingBot:
    """
    Weather prediction market betting bot.

    Modes:
      dry_run=True   — print orders, don't execute (default)
      dry_run=False  — execute via py_clob_client (requires POLYMARKET_PRIVATE_KEY)

    live_mode=True   — recalculate days_ahead from NOW and re-verify prices via
                       the Gamma API before placing any order. Always use this
                       for real-money execution.

    Execution:
      pip install py-clob-client
      export POLYMARKET_PRIVATE_KEY="0x..."

      from py_clob_client.client import ClobClient
      from py_clob_client.clob_types import OrderArgs, BUY

      client = ClobClient(
          host="https://clob.polymarket.com",
          key=os.environ["POLYMARKET_PRIVATE_KEY"],
          chain_id=137,   # Polygon
      )
      # YES bet: buy YES token at price=their_prob, size=usdc/price shares
      client.create_market_order(OrderArgs(
          token_id=YES_TOKEN_ID,   # from clob_token_ids_json field
          price=their_prob,
          size=size_usdc / their_prob,
          side=BUY,
      ))
    """

    def __init__(self, bankroll: float = 1000.0,
                 dry_run: bool = True, live_mode: bool = False,
                 kelly_fraction: float = KELLY_FRACTION):
        self.bankroll  = bankroll
        self.dry_run   = dry_run
        self.live_mode = live_mode
        self.kelly_fraction = kelly_fraction
        self.log: list[dict] = []

    def run(self, opps: list[Opportunity], min_edge: float = MIN_EDGE,
            min_days: float = 0.0):
        if not opps:
            print("\n  No tradeable opportunities.")
            return

        now = datetime.now(timezone.utc)

        # ── Step 1: filter to future markets using CURRENT time ───────────
        # days_ahead stored in each Opportunity was computed from the snapshot
        # fetch_time, which may be hours or days old. Recalculate from NOW.
        def _days_from_now(o: Opportunity) -> float:
            # E1: clamp at 0 like the module-level helper. WITHOUT the clamp a same-day market
            # (target = midnight UTC) goes negative after 00:00 UTC and is dropped by the
            # `>= min_days` filter below, so same-day intraday bets never execute (live or dry).
            target = pd.Timestamp(o.target_date, tz="UTC")
            return max(0.0, (target - pd.Timestamp(now)).total_seconds() / 86400)

        candidate = [
            o for o in opps
            if _days_from_now(o) >= min_days and o.kelly > 0
        ]
        if min_days > 0:
            n_skipped = sum(1 for o in opps if 0 < _days_from_now(o) < min_days)
            if n_skipped:
                print(f"\n  [min_days={min_days}] Skipped {n_skipped} markets "
                      f"resolving within {min_days:.0f}d")

        if not candidate:
            print("\n  No future opportunities with positive Kelly.")
            return

        # ── Step 2 (live_mode): fetch current market prices and re-verify ─
        if self.live_mode:
            print(f"\n  [LIVE] Fetching current prices for {len(candidate)} markets …")
            cids        = list({o.condition_id for o in candidate})
            live_prices = fetch_live_prices(cids)
            verified    = []
            for o in candidate:
                live_p = live_prices.get(o.condition_id)
                if live_p is None:
                    print(f"  [SKIP] {o.condition_id[:16]}… — no live price")
                    continue
                if live_p < MIN_MARKET_PRICE or live_p > (1.0 - MIN_MARKET_PRICE):
                    print(f"  [SKIP] {o.question[:50]} — near-settled ({live_p:.2%})")
                    continue
                # E3: RE-SHRINK toward the fresh live price. The stored forecast_prob/our_prob
                # blended the model with the STALE snapshot price (w·model + (1-w)·snapshot), so
                # at shrink_weight<1 the live edge/Kelly must be recomputed from the pre-shrink
                # model probability against the current price — not read off the stale blend.
                their_live = live_p if o.bet_side == "Yes" else (1.0 - live_p)
                w  = o.shrink_weight if o.shrink_weight is not None else 1.0
                mp = o.model_prob_side if o.model_prob_side is not None else o.our_prob
                our_live  = w * mp + (1.0 - w) * their_live
                live_edge = our_live - their_live
                if abs(live_edge) < min_edge:
                    print(f"  [SKIP] {o.question[:50]} — edge closed "
                          f"(was {o.abs_edge:.1%}, now {abs(live_edge):.1%})")
                    continue
                # Update the opportunity with the fresh price and re-shrunk probability.
                o.market_prob_raw = round(live_p, 4)
                o.our_prob        = round(our_live, 4)
                o.their_prob      = round(their_live, 4)
                o.forecast_prob   = round(our_live if o.bet_side == "Yes" else 1.0 - our_live, 4)
                o.edge            = round(live_edge, 4)
                o.abs_edge        = round(abs(live_edge), 4)
                o.kelly           = _kelly_size(o, kelly_fraction=self.kelly_fraction)
                o.ev_per_dollar   = round(o.our_prob / o.their_prob - 1.0, 4)
                verified.append(o)
            candidate = verified
            print(f"  [LIVE] {len(candidate)} markets still have edge after price check\n")

        # ── Step 3: group cap (α9) ────────────────────────────────────────
        candidate = apply_group_kelly_cap(candidate, self.bankroll)

        # ── Step 4: total portfolio cap ───────────────────────────────────
        candidate = apply_portfolio_cap(candidate)

        # ── Step 5: sort and deduplicate — one bet per condition_id ───────
        candidate.sort(key=lambda o: o.alpha_score, reverse=True)
        seen:  set[str]          = set()
        final: list[Opportunity] = []
        for o in candidate:
            if o.condition_id not in seen:
                final.append(o)
                seen.add(o.condition_id)

        # ── Step 6: print and execute ────────────────────────────────────
        print(f"\n{'─'*78}")
        mode = ("DRY RUN" if self.dry_run else "⚠  LIVE")
        live_tag = " + live-price-verified" if self.live_mode else ""
        print(f"  WeatherBettingBot [{mode}{live_tag}]  bankroll=${self.bankroll:,.0f}")
        print(f"  Fee model: {FEE_RATE:.0%}  |  Max exposure: {MAX_TOTAL_KELLY:.0%} bankroll")
        print(f"{'─'*78}")
        print(f"  {'Side':4s}  {'Size$':>7}  {'Edge':>6}  {'EV$':>7}  "
              f"{'Liq$':>8}  {'Mom':>6}  {'Sig':8s}  Question")
        print(f"  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*7}  "
              f"{'─'*8}  {'─'*6}  {'─'*8}  {'─'*70}")

        total_size = 0.0
        total_ev   = 0.0
        for o in final:
            size = round(self.bankroll * o.kelly, 2)
            if size < 1.0:
                continue

            if self.live_mode:
                # For a "Yes" bet, we are buying Yes tokens, so we check the ask side of the Yes token book.
                # For a "No" bet, we are buying No tokens. The price of a No token is (1 - yes_price).
                # So, buying a No token at 0.70 is equivalent to selling a Yes token at 0.30.
                # We check the VWAP for our side and adjust the effective market probability.
                vwap_price = check_orderbook_vwap(o.condition_id, o.bet_side, size)
                
                # The price from the book is for the token we are buying (e.g. price of "No" token)
                # We need to convert it back to the equivalent "Yes" probability for our edge calculation.
                effective_market_yes_prob = vwap_price if o.bet_side == "Yes" else 1.0 - vwap_price
                real_edge = o.forecast_prob - effective_market_yes_prob

                if (o.bet_side == "Yes" and real_edge < min_edge) or (o.bet_side == "No" and real_edge > -min_edge):
                    print(f"  [SKIP] {o.question[:45]}… — Slippage killed edge (VWAP: {vwap_price:.2f})")
                    continue
                
                # Update the opportunity with the true, slippage-adjusted price
                o.their_prob = vwap_price
                o.abs_edge   = round(abs(real_edge), 4)
                o.ev_per_dollar = round(o.our_prob / o.their_prob - 1.0, 4)

            ev    = round(o.ev_per_dollar * size, 2)
            days  = round(_days_from_now(o), 1)
            order = {
                "condition_id":  o.condition_id,
                "question":      o.question,
                "bet_side":      o.bet_side,
                "size_usdc":     size,
                "market_prob":   o.their_prob,
                "forecast_prob": o.our_prob,
                "edge":          o.abs_edge,
                "ev_dollar":     ev,
                "alpha_score":   o.alpha_score,
                "days_to_expiry":days,
                "sigma_source":  o.sigma_source,
                "is_stale":      o.is_stale,
                "live_verified": self.live_mode,
                "timestamp_utc": now.isoformat(),
            }
            self._execute(order)
            total_size += size
            total_ev   += ev
            print(f"  {o.bet_side:4s}  ${size:>6.2f}  "
                  f"{o.abs_edge:>5.1%}  ${ev:>6.2f}  "
                  f"${o.liquidity:>7,.0f}  "
                  f"{o.ema_momentum:>+6.2f}  "
                  f"{o.sigma_source[:8]:8s}  "
                  f"{o.question[:70]}…  (+{days:.1f}d)")

        print(f"{'─'*78}")
        print(f"  Bets     : {len(self.log)}")
        print(f"  Exposure : ${total_size:,.2f}  ({total_size/self.bankroll:.1%} of bankroll)")
        print(f"  Exp. EV  : ${total_ev:+,.2f}  ({total_ev/max(total_size,1)*100:.1f}% ROI)")

    def _execute(self, order: dict):
        self.log.append(order)
        if not self.dry_run:
            raise NotImplementedError("Wire in py_clob_client. See class docstring.")

