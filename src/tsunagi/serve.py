"""A model + serving function that consumes named columns (what a real scoring service would do)."""
from __future__ import annotations
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .contract import _numeric


def make_data(seed=0):
    d = load_breast_cancer()
    X, y = d.data, d.target
    names = [n.replace(" ", "_") for n in d.feature_names]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    cols = {n: X[:, i].astype(float) for i, n in enumerate(names)}
    # two derived categorical columns
    for src, new in (("mean_radius", "size_tier"), ("worst_texture", "texture_tier")):
        q = np.quantile(cols[src], [1 / 3, 2 / 3])
        cols[new] = np.array(["low", "mid", "high"], dtype=object)[np.digitize(cols[src], q)]
    return cols, y


CATS = {"size_tier": ["low", "mid", "high"], "texture_tier": ["low", "mid", "high"]}


class Service:
    def __init__(self, cols, y, seed=0, kind="rf"):
        self.num = [k for k in cols if k not in CATS]
        self.median = {k: float(np.nanmedian(cols[k])) for k in self.num}
        X = self._matrix(cols)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1).fit(X, y)
        self.importance = dict(zip(self.num + [f"{c}={v}" for c in CATS for v in CATS[c]], rf.feature_importances_))
        self.model = rf if kind == "rf" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y)

    def _matrix(self, cols):
        n = len(next(iter(cols.values())))
        parts = []
        for k in self.num:
            if k in cols:
                x = _numeric(np.asarray(cols[k], dtype=object) if cols[k].dtype == object else cols[k])
            else:
                x = np.full(n, np.nan)              # missing column -> imputed
            x = np.where(np.isnan(x), self.median[k], x)
            parts.append(x[:, None])
        for c, vals in CATS.items():
            col = cols.get(c, np.array([None] * n, dtype=object))
            for v in vals:
                parts.append(np.array([x == v for x in col], dtype=float)[:, None])
        return np.hstack(parts)

    def predict(self, cols):
        return self.model.predict(self._matrix(cols))

    def top_numeric(self, k=3):
        return [n for n, _ in sorted(((n, self.importance[n]) for n in self.num), key=lambda t: -t[1])[:k]]

    def bottom_numeric(self, k=3):
        return [n for n, _ in sorted(((n, self.importance[n]) for n in self.num), key=lambda t: t[1])[:k]]
