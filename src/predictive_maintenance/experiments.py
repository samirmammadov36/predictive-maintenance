from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from .artifacts import load_joblib
from .constants import CYCLE_COLUMN, ID_COLUMN, RUL_TRUTH_COLUMN
from .evaluation import regression_metrics
from .features import build_tabular_features
from .masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from .models.lstm import LSTMConfig, RULLSTM, predict_lstm
from .sequences import build_prefix_window, scale_sequence_frame


def _recover(prefix, bundle, method: str):
    if method == "median":
        return bundle.medians.median_fill(prefix, bundle.selected_sensors)
    if method == "forward_fill":
        return bundle.medians.causal_forward_fill(prefix, bundle.selected_sensors)
    if method == "none":
        return prefix.copy()
    raise ValueError(f"Unknown recovery method: {method}")


def _tabular_input(prefix, bundle):
    feat, _ = build_tabular_features(prefix, bundle.selected_sensors, bundle.rolling_window)
    return bundle.tabular_scaler.transform(feat.iloc[[-1]][bundle.feature_columns])


def _sequence_input(prefix, bundle):
    scaled = scale_sequence_frame(prefix, bundle.selected_sensors, bundle.sequence_scaler)
    return np.expand_dims(build_prefix_window(scaled, bundle.selected_sensors, bundle.lstm_window), axis=0)


def _load_models(bundle, artifact_dir: Path):
    rf = load_joblib(artifact_dir / "models/random_forest.joblib")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(artifact_dir / "models/xgboost.json")
    lstm_meta = load_joblib(artifact_dir / "models/lstm_meta.joblib")
    cfg = LSTMConfig(**lstm_meta["config"])
    lstm = RULLSTM(cfg.input_size, cfg.hidden_units, cfg.dense_units)
    lstm.load_state_dict(torch.load(artifact_dir / "models/lstm.pt", map_location="cpu"))
    lstm.eval()
    return {"random_forest": rf, "xgboost": xgb_model, "lstm": lstm}


def _predict(model_name, model, prefix, bundle):
    start = time.perf_counter()
    if model_name in {"random_forest", "xgboost"}:
        pred = float(model.predict(_tabular_input(prefix, bundle))[0])
    else:
        pred = float(predict_lstm(model, _sequence_input(prefix, bundle))[0])
    return max(pred, 0.0), time.perf_counter() - start


def _corrupt(prefix, bundle, scenario: str, seed: int | None):
    if scenario == "clean":
        return prefix.copy(), None
    if scenario in {"random_10", "random_20"}:
        low, high = nested_random_masks(prefix, bundle.selected_sensors, int(seed), 0.10, 0.20)
        mask = low if scenario == "random_10" else high
        return apply_sensor_mask(prefix, mask, bundle.selected_sensors), None
    if scenario == "short_gap":
        mask, sensor = short_gap_mask(prefix, bundle.selected_sensors, "sensor_11", 5)
        return apply_sensor_mask(prefix, mask, bundle.selected_sensors), sensor
    raise ValueError(scenario)


def run_experiments(bundle, artifact_dir: str | Path = "artifacts", random_seeds=(7, 17, 27), return_predictions: bool = False):
    artifact_dir = Path(artifact_dir)
    models = _load_models(bundle, artifact_dir)
    target_map = bundle.official_test_targets.set_index(ID_COLUMN)[RUL_TRUTH_COLUMN].to_dict()
    rows = []
    prediction_rows = []
    scenario_specs = [("clean", None, "none")]
    for scenario in ("random_10", "random_20"):
        for seed in random_seeds:
            for recovery in ("median", "forward_fill"):
                scenario_specs.append((scenario, seed, recovery))
    for recovery in ("median", "forward_fill"):
        scenario_specs.append(("short_gap", None, recovery))

    for scenario, seed, recovery in scenario_specs:
        per_model_true = {name: [] for name in models}
        per_model_pred = {name: [] for name in models}
        per_model_time = {name: 0.0 for name in models}
        gap_sensor = None
        for engine_id in sorted(bundle.official_test_df[ID_COLUMN].unique()):
            prefix = bundle.official_test_df[bundle.official_test_df[ID_COLUMN] == engine_id].sort_values(CYCLE_COLUMN).copy()
            corrupted, chosen_gap_sensor = _corrupt(prefix, bundle, scenario, seed)
            gap_sensor = chosen_gap_sensor or gap_sensor
            recovered = _recover(corrupted, bundle, recovery)
            if recovered[bundle.selected_sensors].isna().any().any():
                raise AssertionError("Missing values remain after recovery")
            y = float(target_map[int(engine_id)])
            for name, model in models.items():
                pred, elapsed = _predict(name, model, recovered, bundle)
                per_model_true[name].append(y)
                per_model_pred[name].append(pred)
                per_model_time[name] += elapsed
                prediction_rows.append(
                    {
                        "model": name,
                        "scenario": scenario,
                        "seed": seed,
                        "recovery": recovery,
                        "engine_id": int(engine_id),
                        "rul_true": y,
                        "rul_pred": pred,
                        "error": pred - y,
                        "abs_error": abs(pred - y),
                    }
                )
        for name in models:
            metrics = regression_metrics(per_model_true[name], per_model_pred[name])
            rows.append(
                {
                    "model": name,
                    "scenario": scenario,
                    "seed": seed,
                    "recovery": recovery,
                    "gap_sensor": gap_sensor,
                    "n_engines": len(per_model_true[name]),
                    "prediction_seconds": per_model_time[name],
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    clean = result[result["scenario"] == "clean"][["model", "mae", "rmse"]].rename(columns={"mae": "clean_mae", "rmse": "clean_rmse"})
    result = result.merge(clean, on="model", how="left")
    result["delta_mae"] = result["mae"] - result["clean_mae"]
    result["delta_rmse"] = result["rmse"] - result["clean_rmse"]
    predictions = pd.DataFrame(prediction_rows)
    if return_predictions:
        return result, predictions
    return result


def aggregate_random_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["model", "scenario", "recovery"], dropna=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
        )
        .reset_index()
    )
