"""train_intraday.py — per-hour conditioning of daily Tmax on the running observed max.

Same-day markets are priced by participants who can see the day's observations; a pure
forecast can't compete after mid-morning. This trains, per city and per local hour h:

    Tmax = a_h + b_h·fcst + c_h·M_h + eps,   sigma_h = std(holdout eps)

where M_h is the running daily max observed through hour h (routine METARs,
fetch_station_obs.py) and fcst is the day-ahead forecast (lead-1 previous-runs archive —
the same variable family the serving forecast comes from). Serving additionally floors
the predictive distribution at M_h (Tmax can never be below what was already observed):
T = max(M_h, Z), Z ~ t(mu_h, sigma_h, nu).

Expected shape (and the built-in sanity check): c_h grows toward 1 and sigma_h shrinks
toward ~0.3 °C through the afternoon; conditioning adds nothing before ~09 local. Hours
where the conditioned fit does NOT beat the forecast-only fit on the temporal holdout
are dropped from the params, so serving self-gates to the unconditioned path there.

Output: models/{slug}_intraday.json
Run from src/polymarket_weather/:   python train_intraday.py
"""
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CITIES_WITH_OBS = ["new_york_city", "chicago", "london", "seoul"]
HOURS = range(5, 24)      # local hours worth conditioning on (obs before ~05 rarely bind)
HOLDOUT_FRAC = 0.20
MIN_ROWS = 300


def _fit_nu(residuals: np.ndarray) -> float:
    try:
        from scipy.stats import kurtosis
        ex = float(kurtosis(residuals, fisher=True, bias=False))
    except Exception:
        return 7.0
    if ex <= 0.05:
        return 30.0
    return float(np.clip(4.0 + 6.0 / ex, 4.0, 30.0))


# Target specs: max conditions on the running MAX (a floor for Tmax); min conditions on
# the running MIN (a ceiling for Tmin — by mid-morning the overnight low is usually in).
TARGETS = {
    "max": {"truth_col": "temp_max_c", "fcst_file": "data/weather/{slug}_historical_leads.csv",
            "fcst_col": "fcst_tmax_lead1", "run_agg": "max", "out": "{slug}_intraday.json"},
    "min": {"truth_col": "temp_min_c", "fcst_file": "data/weather/{slug}_historical_leads_min.csv",
            "fcst_col": "fcst_tmin_lead1", "run_agg": "min", "out": "{slug}_intraday_min.json"},
}


def train_intraday():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for slug in CITIES_WITH_OBS:
        for kind, spec in TARGETS.items():
            _train_one(slug, kind, spec)


def _train_one(slug: str, kind: str, spec: dict):
        try:
            from settlement_truth import load_training_truth
            obs = pd.read_csv(f"data/weather/{slug}_obs_hourly.csv")
            truth = load_training_truth(slug)   # settlement-faithful target (W0.2); CLI fallback
            leads = pd.read_csv(spec["fcst_file"].format(slug=slug))
        except Exception as e:
            logger.error(f"{slug} [{kind}]: missing inputs ({e})")
            return

        truth_col, fcst_col = spec["truth_col"], spec["fcst_col"]
        obs["valid_local"] = pd.to_datetime(obs["valid_local"])
        obs["date_local"] = obs["valid_local"].dt.strftime("%Y-%m-%d")
        obs["hour"] = obs["valid_local"].dt.hour

        # Running max (or min) through each hour of each local day.
        if spec["run_agg"] == "max":
            hourly_run = (obs.groupby(["date_local", "hour"])["temp_c"].max()
                            .groupby(level=0).cummax().rename("run_val").reset_index())
        else:
            hourly_run = (obs.groupby(["date_local", "hour"])["temp_c"].min()
                            .groupby(level=0).cummin().rename("run_val").reset_index())

        base = (truth[["date_local", truth_col]]
                .merge(leads[["date_local", fcst_col]], on="date_local")
                .dropna())

        per_hour = {}
        resid_pool = []
        for h in HOURS:
            m = base.merge(hourly_run[hourly_run["hour"] == h], on="date_local").dropna()
            m = m.sort_values("date_local").reset_index(drop=True)
            if len(m) < MIN_ROWS:
                continue
            # Truth must respect the censoring bound; drop the few rows where unit
            # rounding or a bad METAR violates it by >0.6 °C (data error, not signal).
            if spec["run_agg"] == "max":
                m = m[m[truth_col] >= m["run_val"] - 0.6]
            else:
                m = m[m[truth_col] <= m["run_val"] + 0.6]

            y = m[truth_col].to_numpy(dtype=float)
            X = np.column_stack([
                np.ones(len(m)),
                m[fcst_col].to_numpy(dtype=float),
                m["run_val"].to_numpy(dtype=float),
            ])
            cut = int(len(m) * (1.0 - HOLDOUT_FRAC))
            beta, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
            resid = y[cut:] - X[cut:] @ beta
            rmse_cond = float(np.sqrt(np.mean(resid ** 2)))

            # Forecast-only baseline on the same split — the self-gate.
            Xf = X[:, :2]
            beta_f, *_ = np.linalg.lstsq(Xf[:cut], y[:cut], rcond=None)
            rmse_fcst = float(np.sqrt(np.mean((y[cut:] - Xf[cut:] @ beta_f) ** 2)))
            if rmse_cond >= rmse_fcst:
                continue   # conditioning doesn't help at this hour — serve unconditioned

            per_hour[str(h)] = {
                "a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                "sigma": float(np.std(resid)),
                "n_train": int(cut), "n_holdout": int(len(m) - cut),
                "holdout_rmse_conditioned": rmse_cond,
                "holdout_rmse_forecast_only": rmse_fcst,
            }
            resid_pool.append(resid)

        if not per_hour:
            logger.warning(f"{slug} [{kind}]: no hour beat the forecast-only baseline — skipping")
            return

        params = {
            "version": 1,
            "kind": kind,
            "nu": _fit_nu(np.concatenate(resid_pool)),
            "hours": per_hour,
        }
        out = os.path.join(MODELS_DIR, spec["out"].format(slug=slug))
        with open(out, "w") as fh:
            json.dump(params, fh, indent=2)

        hs = sorted(int(k) for k in per_hour)
        s12 = per_hour.get("12", {}).get("sigma")
        s17 = per_hour.get("17", {}).get("sigma")
        logger.info(f"{slug} [{kind}]: hours {hs[0]}–{hs[-1]} kept ({len(hs)}), "
                    f"sigma@12 {s12 if s12 is None else round(s12, 2)} "
                    f"sigma@17 {s17 if s17 is None else round(s17, 2)} → {out}")


if __name__ == "__main__":
    train_intraday()
