"""qrf_core.py — Quantile Regression Forest (Meinshausen 2006) on scikit-learn RandomForest.
Fit a standard regression forest; at predict, read the empirical distribution of the training
targets that share leaves with the query (weighted by 1/leaf-size per tree) and return quantiles.
No parametric shape is assumed in the LEARNING — the conditional spread/tails are data-driven."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as _stats


class QuantileForest:
    def __init__(self, n_estimators=300, min_samples_leaf=30, random_state=0):
        self.rf = RandomForestRegressor(n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        random_state=random_state, n_jobs=-1)
        self._y = None
        self._train_leaves = None      # (n_train, n_trees) leaf ids

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.rf.fit(X, y)
        self._y = y
        self._train_leaves = self.rf.apply(X)      # leaf id per (sample, tree)
        return self

    def predict_quantiles(self, X, q):
        X = np.asarray(X, float)
        q = np.asarray(q, float)
        test_leaves = self.rf.apply(X)             # (n_test, n_trees)
        n_trees = self._train_leaves.shape[1]
        out = np.empty((X.shape[0], len(q)))
        for i in range(X.shape[0]):
            # weight each training sample by how often it shares the query's leaf, normalised
            # per tree by that leaf's training size (the QRF weighting).
            w = np.zeros(self._y.shape[0])
            for t in range(n_trees):
                same = self._train_leaves[:, t] == test_leaves[i, t]
                c = same.sum()
                if c:
                    w[same] += 1.0 / c
            w /= n_trees
            out[i] = _weighted_quantile(self._y, w, q)
        return out


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(q, cw, v)


def moment_match(q_levels, q_values):
    """Convert quantiles to Student-t parameters (mu, sigma, nu).

    Given quantile levels and their values from a QRF (or any distribution),
    extract Student-t parameters by:
    - mu = median (50th quantile)
    - sigma = spread from the central 68% interval (84th - 16th quantiles)
    - nu = degrees of freedom chosen so the outer/inner tail ratio matches
    """
    q_levels = np.asarray(q_levels, float)
    q_values = np.asarray(q_values, float)

    def qv(p):
        """Get quantile value at the closest available level."""
        return float(q_values[int(np.argmin(np.abs(q_levels - p)))])

    mu = qv(0.5)
    sigma = max((qv(0.84) - qv(0.16)) / 2.0, 1e-3)

    # empirical tail ratio: outer span / inner span
    emp = (qv(0.95) - qv(0.05)) / max(qv(0.75) - qv(0.25), 1e-6)

    # Student-t theoretical ratio as a function of nu; pick the nu whose ratio matches.
    grid = np.array([3, 4, 5, 6, 8, 10, 15, 20, 30, 40], float)
    ratios = np.array([(_stats.t(df=n).ppf(0.95) - _stats.t(df=n).ppf(0.05)) /
                       (_stats.t(df=n).ppf(0.75) - _stats.t(df=n).ppf(0.25)) for n in grid])
    nu = float(grid[int(np.argmin(np.abs(ratios - emp)))])

    return mu, sigma, nu


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
    """Semi-parametric CDF: linear-interp body between the outer knots, Gaussian tail beyond
    each outer knot fit to that side's two outermost knots. Clamped to [0,1]; the degenerate
    near-constant case collapses to a step at the median rather than crashing."""
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
