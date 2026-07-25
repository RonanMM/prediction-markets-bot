# Serve Empirical QRF Distribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the QRF forecaster's raw empirical distribution (via an optional CDF callable) instead of the lossy Student-t moment-match, without changing behavior for any other predictor.

**Architecture:** `TemperatureDistribution` gains an optional `cdf` callable. `pmf._cdf` uses it when present (keeping floor/ceiling censoring); all bin/PMF math flows through it unchanged. `QRFPredictor` builds a semi-parametric empirical CDF (99-quantile body + Gaussian tails) and sets it. `train_qrf.fit_city` re-gates on the empirical-CDF CRPS. `cdf=None` (EMOS/ensemble) is byte-identical to today.

**Tech Stack:** Python 3, numpy, scipy, scikit-learn, pandas, joblib. No new dependencies.

## Global Constraints

- Run tests from repo root: `pytest -o addopts="" tests/ -v`.
- **Additivity is the load-bearing invariant:** with `cdf=None`, `pmf._cdf`, `_bin_prob`, `_condition_prob`, `reconstruct_pmf`, and every existing predictor MUST behave byte-identically to before. A regression test enforces this. EMOS/ensemble must never be touched behaviorally.
- The `cdf` callable is the CDF of the **uncensored** variable Z; `floor`/`ceiling` censoring is applied ON TOP by `_cdf` exactly as for the Student-t (return 0.0 below floor, 1.0 at/above ceiling, before consulting `cdf`).
- `moment_match` is RETAINED — QRF still reports `(mu, sigma, nu)` as summary stats; they just no longer drive bins when `cdf` is set.
- Interfaces (verified): `TemperatureDistribution(mu, sigma, nu, source, floor=None, ceiling=None)` in `predictors/base.py`; `pmf._cdf(x, mu, sigma, nu, floor=None, ceiling=None)`; `QRFPredictor.predict_distribution(... kind="max") -> TemperatureDistribution | None`; `QuantileForest.predict_quantiles(X, q)`; `_Q`/`moment_match` in `predictors/qrf_core.py`.
- Scope: Tmax only, 5 cities. `min_lead` guard and M2 are out of scope.

---

### Task 1: `cdf` field on `TemperatureDistribution`

**Files:**
- Modify: `src/polymarket_weather/predictors/base.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces: `TemperatureDistribution` dataclass gains `cdf: Callable[[float], float] | None = None` as the last field (after `ceiling`).

- [ ] **Step 1: Write the failing test**

```python
def test_temperature_distribution_has_optional_cdf():
    from predictors.base import TemperatureDistribution
    d = TemperatureDistribution(mu=20.0, sigma=2.0, nu=10.0, source="qrf")
    assert d.cdf is None                       # default: parametric path
    f = lambda x: 0.5
    d2 = TemperatureDistribution(mu=20.0, sigma=2.0, nu=10.0, source="qrf", cdf=f)
    assert d2.cdf(999) == 0.5                   # callable carried
    # existing positional/keyword construction still works (floor/ceiling unaffected)
    d3 = TemperatureDistribution(20.0, 2.0, 10.0, "emos_v2", floor=18.0)
    assert d3.cdf is None and d3.floor == 18.0
```

- [ ] **Step 2: Run to verify it fails** — `pytest ...::test_temperature_distribution_has_optional_cdf` → FAIL (unexpected kwarg `cdf`).

- [ ] **Step 3: Implement** — in `predictors/base.py`, add to the dataclass (after `ceiling`):

```python
    from typing import Callable   # at top of file with the other imports
    ...
    cdf: "Callable[[float], float] | None" = None   # optional empirical CDF of the uncensored Z;
                                                     # when set, pmf uses it instead of the Student-t
```

(Use the existing import style in the file; add `from typing import Callable` if absent.)

- [ ] **Step 4: Run to verify it passes** — PASS. Full suite green (dataclass change is additive).
- [ ] **Step 5: Commit** — `git commit -m "qrf-empirical: optional cdf field on TemperatureDistribution"`

---

### Task 2: `pmf._cdf` (+ callers) honor an optional `cdf`

**Files:**
- Modify: `src/polymarket_weather/pmf.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces: `_cdf(x, mu, sigma, nu, floor=None, ceiling=None, cdf=None)`; `_bin_prob(..., cdf=None)`; `_condition_prob(parsed, mu, sigma, nu, floor=None, ceiling=None, cdf=None)`; `reconstruct_pmf(..., cdf=None)` — all pass `cdf` down to `_cdf`.

- [ ] **Step 1: Write the failing test**

```python
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
    # bins every 1°C from 10..30 should capture ~all mass
    total = sum(pmf._bin_prob(t, 25.0, 3.0, 8.0, half_width=0.5, cdf=ramp) for t in range(11, 30))
    assert abs(total - 1.0) < 0.05
```

- [ ] **Step 2: Run to verify it fails** — FAIL (unexpected kwarg `cdf`).

- [ ] **Step 3: Implement** — in `pmf.py`:
  - `_cdf`: add `cdf=None` param; keep the `floor`/`ceiling` short-circuits unchanged; then:
    ```python
    if cdf is not None:
        return min(1.0, max(0.0, float(cdf(x))))
    return float(student_t.cdf((x - mu) / _t_scale(sigma, nu), df=nu))
    ```
  - `_bin_prob`: add `cdf=None`; forward to both `_cdf(...)` calls.
  - `_condition_prob`: add `cdf=None`; forward to every `_cdf`/`_bin_prob` call inside it.
  - `reconstruct_pmf`: add `cdf=None`; forward to the `_condition_prob`/`_bin_prob`/`_cdf` calls it makes. (Read the function body and thread `cdf` through every call site.)

- [ ] **Step 4: Run to verify it passes** — both new tests PASS; run full suite (regression: all existing pmf tests unchanged).
- [ ] **Step 5: Commit** — `git commit -m "qrf-empirical: pmf._cdf and callers honor an optional cdf (censoring preserved)"`

---

### Task 3: `build_empirical_cdf` — semi-parametric CDF (empirical body + Gaussian tails)

**Files:**
- Modify: `src/polymarket_weather/predictors/qrf_core.py` (add the builder + a sample-CRPS helper if absent)
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Produces:
  - `Q_FINE: list[float]` = `[0.01, 0.02, …, 0.99]` (99 levels).
  - `empirical_cdf_from_quantiles(levels: list[float], values: np.ndarray) -> Callable[[float], float]` — monotone CDF: linear-interp body between `values[0]`..`values[-1]`; Gaussian tail beyond each outer knot fit to the two outermost knots on that side; clamped [0,1]; degenerate (near-constant values) → tiny-σ step at the median.
  - `sample_crps(samples: np.ndarray, y: float) -> float` — sorted-sample CRPS (reuse if a helper already exists; else add).

- [ ] **Step 1: Write the failing test**

```python
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
    tight = stats.norm(20, 1).ppf(np.linspace(.01,.99,99))
    wide  = stats.norm(20, 5).ppf(np.linspace(.01,.99,99))
    assert sample_crps(tight, y) < sample_crps(wide, y)   # sharper+calibrated scores better
```

- [ ] **Step 2: Run to verify it fails** — FAIL (undefined).

- [ ] **Step 3: Implement** (append to `qrf_core.py`)

```python
Q_FINE = [round(0.01 * k, 2) for k in range(1, 100)]     # 0.01..0.99


def sample_crps(samples, y):
    s = np.sort(np.asarray(samples, float)); n = len(s)
    if n == 0:
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    i = np.arange(1, n + 1)
    e2 = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * s)    # E|X-X'| via sorted identity
    return float(e1 - 0.5 * e2)


def empirical_cdf_from_quantiles(levels, values):
    lv = np.asarray(levels, float); v = np.asarray(values, float)
    order = np.argsort(v); v = v[order]; lv = lv[order]
    lo_v, lo_p, hi_v, hi_p = v[0], lv[0], v[-1], lv[-1]
    spread = hi_v - lo_v
    if spread < 1e-6:                                     # degenerate -> step at the median
        med = float(np.median(v))
        return lambda x: 0.0 if x < med else 1.0
    # Gaussian tail params: solve mu,sigma so the Gaussian CDF hits the two outer knots per side.
    from scipy.stats import norm
    def _tail(p1, x1, p2, x2):
        z1, z2 = norm.ppf(p1), norm.ppf(p2)
        sig = (x2 - x1) / (z2 - z1) if abs(z2 - z1) > 1e-9 else max(spread, 1e-3)
        mu = x1 - z1 * sig
        return mu, max(sig, 1e-3)
    lmu, lsig = _tail(lv[0], v[0], lv[1], v[1])           # left tail
    rmu, rsig = _tail(lv[-2], v[-2], lv[-1], v[-1])       # right tail
    def F(x):
        if x <= lo_v:
            return float(min(lo_p, norm.cdf(x, lmu, lsig)))
        if x >= hi_v:
            return float(max(hi_p, norm.cdf(x, rmu, rsig)))
        return float(np.interp(x, v, lv))                 # monotone body
    return F
```

- [ ] **Step 4: Run to verify it passes** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "qrf-empirical: build_empirical_cdf (empirical body + Gaussian tails) + sample_crps"`

---

### Task 4: `QRFPredictor` serves the empirical CDF

**Files:**
- Modify: `src/polymarket_weather/predictors/qrf.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `empirical_cdf_from_quantiles`, `Q_FINE` (T3); `QuantileForest.predict_quantiles`; the existing feature assembly in `predictors/qrf.py`.
- Produces: gated-ON `predict_distribution` returns `TemperatureDistribution(mu, sigma, nu, "qrf", floor=floor, cdf=<empirical>)`; gated-OFF / missing / `days_ahead > max_lead` → `None` (unchanged).

- [ ] **Step 1: Write the failing test** (mirror the existing `test_qrf_predictor_serves_gated_on_with_floor` fixture style)

```python
def test_qrf_predictor_serves_empirical_cdf(tmp_path, monkeypatch):
    import numpy as np, json, joblib
    from predictors import qrf as qmod
    from predictors.qrf_core import QuantileForest
    from qrf_features import FEATURE_COLS
    from predictors.base import TemperatureDistribution
    monkeypatch.setattr(qmod, "_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(3)
    X = rng.normal(20, 3, size=(400, len(FEATURE_COLS))); y = rng.normal(20, 3, size=400)
    joblib.dump(QuantileForest(n_estimators=60, min_samples_leaf=20).fit(X, y), tmp_path / "seoul_qrf.joblib")
    (tmp_path / "seoul_qrf_meta.json").write_text(json.dumps({"beats_ensemble": True, "max_lead": 4}))
    # reuse whatever fixture df-builders the existing gated-on test uses; call predict_distribution
    # for a lead-1 Tmax bet and assert:
    dist = _call_qrf_lead1(qmod.QRFPredictor())      # helper mirrors test_qrf_predictor_serves_gated_on_with_floor
    assert isinstance(dist, TemperatureDistribution)
    assert dist.source == "qrf" and dist.cdf is not None
    # the served cdf is a monotone probability
    assert dist.cdf(0.0) <= dist.cdf(50.0)
    assert 0.0 <= dist.cdf(20.0) <= 1.0
```

(If a shared `_call_qrf_lead1` helper doesn't exist, inline the same df fixtures the current
gated-on test builds — do NOT invent new schemas.)

- [ ] **Step 2: Run to verify it fails** — FAIL (`dist.cdf is None`, because the predictor doesn't set it yet).

- [ ] **Step 3: Implement** — in `predictors/qrf.py` `predict_distribution`, where it currently computes `q7 = predict_quantiles(_Q)` → `moment_match` → returns the Student-t distribution: additionally predict `q_fine = predict_quantiles(Q_FINE)` for the same feature row, build `cdf = empirical_cdf_from_quantiles(Q_FINE, q_fine)`, keep the `moment_match` `(mu,sigma,nu)` as summary, and return `TemperatureDistribution(mu, sigma, nu, "qrf", floor=floor, cdf=cdf)`. (One `predict_quantiles` call can request `Q_FINE` and read `_Q` from it, or make two calls — prefer one call over the fine grid and derive the 7-level summary from it to avoid double leaf-weighting.)

- [ ] **Step 4: Run to verify it passes** — PASS; full suite green (existing QRF tests: `cdf` is now set, but `mu/sigma/nu`/floor/gate behavior unchanged — update any existing assertion that asserted `cdf is None` for QRF, none should).
- [ ] **Step 5: Commit** — `git commit -m "qrf-empirical: QRFPredictor serves the empirical CDF (moment-match kept as summary)"`

---

### Task 5: `fit_city` re-gates on the empirical-CDF CRPS

**Files:**
- Modify: `src/polymarket_weather/train_qrf.py`
- Test: `tests/test_polymarket_weather.py`

**Interfaces:**
- Consumes: `QuantileForest.predict_quantiles`, `Q_FINE`, `sample_crps` (T3).
- Produces: `fit_city`'s `holdout` (persisted as `meta["holdout_crps"]` and compared to `ens_holdout_crps`) is now the mean **empirical-CDF sample-CRPS** over the holdout rows, not `crps_gaussian_proxy` on the collapsed `(mu,sigma)`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails or passes** — this may PASS on the gate boolean (weak ensemble) but the intent is the *score path* changed. Make the test meaningful by asserting the score is computed via the empirical path: temporarily assert the new score differs from the old `crps_gaussian_proxy(y_holdout, mm[:,0], mm[:,1])` value (they will differ because one is analytic-Gaussian on the collapse, the other is sample-CRPS on the empirical). If the harness makes that awkward, keep the finite/positive + beats-weak-ensemble assertions and rely on the implementation review to confirm the path.

- [ ] **Step 3: Implement** — in `fit_city`, replace:
  ```python
  q = qf.predict_quantiles(X[cut:], _Q)
  mm = np.array([moment_match(_Q, q[i]) for i in range(len(q))])
  holdout = crps_gaussian_proxy(y[cut:], mm[:, 0], mm[:, 1])
  ```
  with an empirical-CDF holdout CRPS:
  ```python
  qf_gate = qf                      # already fit on X[:cut]
  qfine = qf_gate.predict_quantiles(X[cut:], Q_FINE)
  holdout = float(np.mean([sample_crps(qfine[i], y[cut:][i]) for i in range(len(qfine))]))
  ```
  Keep everything else (the full-data refit `qf_full`, the meta write, `beats_ensemble = holdout <= ens_holdout_crps`) unchanged. Import `Q_FINE`, `sample_crps` from `predictors.qrf_core`.

  **Note for the implementer:** `_ensemble_holdout_crps` (the baseline) still uses `crps_gaussian_proxy`. The two are different CRPS estimators (sample vs analytic-Gaussian), so the comparison is slightly apples-to-oranges — but both are honest CRPS-scale, and the *real* ensemble bar is `evaluate_oos`'s M1 gate, not this self-gate. Leave `_ensemble_holdout_crps` as-is (a fully-consistent estimator swap is a larger change; the self-gate is only a serving switch, and the diagnostic showed raw beats the proxy by margins far exceeding the estimator gap). Record this in the report.

- [ ] **Step 4: Run to verify it passes** — PASS; full suite green (Task-4-of-original `test_fit_city_gates_and_persists` may need its assertion updated if it asserted the old score magnitude — update it to the finite/positive check).
- [ ] **Step 5: Commit** — `git commit -m "qrf-empirical: fit_city self-gate scores the empirical-CDF CRPS"`

---

### Task 6: Full-suite regression sweep + serving smoke test

**Files:**
- Test/verify only (no new production code unless a regression surfaces).

- [ ] **Step 1:** Run the FULL suite: `pytest -o addopts="" tests/ -q`. All green. Confirm the additivity invariant held (no existing pmf/EMOS/ensemble test changed behavior).
- [ ] **Step 2: Offline serving smoke test** — with the committed `models/seoul_qrf.joblib` (present locally) and Seoul's local archives, in `src/polymarket_weather/`:
  ```bash
  PYTHONPATH=. python -c "
  import json, joblib, numpy as np
  import train_qrf as T
  from predictors.qrf_core import empirical_cdf_from_quantiles, Q_FINE, sample_crps, moment_match, _Q if False else None
  from predictors.qrf_core import empirical_cdf_from_quantiles as E, Q_FINE as QF, sample_crps as S
  df=T._load_city_frame('seoul'); X,y=T._assemble_xy(df)
  Xv=X[T.FEATURE_COLS].to_numpy(float); cut=T._holdout_cut(len(y))
  qf=joblib.load('models/seoul_qrf.joblib')
  qfine=qf.predict_quantiles(Xv[cut:cut+120], QF)
  emp=np.mean([S(qfine[i], y[cut:][i]) for i in range(len(qfine))])
  print('seoul empirical holdout CRPS (120 rows):', round(float(emp),3))
  "
  ```
  Expected: an empirical CRPS in the ballpark of the diagnostic's Seoul RAW (~0.68), clearly below the collapsed self-gate value (~0.73) — confirming the serving path realizes the diagnostic gain. Record the number.
- [ ] **Step 3: Commit** (if anything changed) — else note "verification only, no code change".

---

## Self-Review

- **Spec coverage:** cdf field (T1), pmf plumbing + regression (T2), empirical CDF builder + sample_crps (T3), QRFPredictor serves it (T4), self-gate re-scored (T5), full regression + serving smoke (T6). All spec §1–§4 covered.
- **Placeholder scan:** the only fixture-dependent spot is T4's `_call_qrf_lead1` — explicitly instructs reuse of the existing gated-on test's df fixtures, not invention. Everything else is complete code.
- **Type consistency:** `Q_FINE`/`sample_crps`/`empirical_cdf_from_quantiles` defined T3, consumed T4/T5/T6; `cdf` field defined T1, consumed T2/T4; `_cdf(..., cdf=None)` defined T2, used by the served distribution's bins; `TemperatureDistribution(...)` matches base.py.
- **Additivity guard:** the regression assertion in T2 (`cdf=None` reproduces the Student-t to 1e-12) plus the full-suite sweep in T6 protect the "EMOS/ensemble untouched" invariant.
