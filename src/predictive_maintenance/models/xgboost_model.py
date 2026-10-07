from __future__ import annotations

import time
import xgboost as xgb

from ..evaluation import regression_metrics


def train_xgboost(X_train, y_train, X_val, y_val, seed: int = 42):
    configs = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "learning_rate": 0.10, "subsample": 0.85, "colsample_bytree": 0.9},
    ]
    rows = []
    best = None
    best_key = (float("inf"), float("inf"))
    for cfg in configs:
        model = xgb.XGBRegressor(
            n_estimators=1500,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            reg_lambda=1.0,
            early_stopping_rounds=30,
            **cfg,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_seconds = time.perf_counter() - start
        pred = model.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        row = {**cfg, "best_iteration": getattr(model, "best_iteration", None), "train_seconds": train_seconds, **metrics}
        rows.append(row)
        key = (metrics["rmse"], metrics["mae"])
        if key < best_key:
            best_key = key
            best = model
    return best, rows
