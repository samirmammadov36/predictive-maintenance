from __future__ import annotations

from itertools import product
import time

from sklearn.ensemble import RandomForestRegressor

from ..evaluation import regression_metrics


def train_random_forest(X_train, y_train, X_val, y_val, seed: int = 42):
    search = {
        "n_estimators": [250, 450],
        "max_depth": [None, 16],
        "min_samples_leaf": [1, 2],
    }
    rows = []
    best = None
    best_key = (float("inf"), float("inf"))
    for n_estimators, max_depth, min_samples_leaf in product(
        search["n_estimators"], search["max_depth"], search["min_samples_leaf"]
    ):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        pred = model.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        row = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "train_seconds": train_seconds,
            **metrics,
        }
        rows.append(row)
        key = (metrics["rmse"], metrics["mae"])
        if key < best_key:
            best_key = key
            best = model
    return best, rows
