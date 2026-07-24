"""qrf_core.py — Quantile Regression Forest (Meinshausen 2006) on scikit-learn RandomForest.
Fit a standard regression forest; at predict, read the empirical distribution of the training
targets that share leaves with the query (weighted by 1/leaf-size per tree) and return quantiles.
No parametric shape is assumed in the LEARNING — the conditional spread/tails are data-driven."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
