from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_joblib
from .constants import CYCLE_COLUMN, ID_COLUMN, RUL_TRUTH_COLUMN
from .evaluation import (
    FD001_ENGINE_IDS, regression_metrics, validate_official_test_inputs, validate_official_test_predictions,
)
from .features import build_tabular_features
from .masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from .models.lstm import predict_lstm
from .models.lstm_checkpoint import load_lstm_checkpoint
from .prepared import official_test_raw_frame
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
    import xgboost as xgb

    rf = load_joblib(artifact_dir / "models/random_forest.joblib")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(artifact_dir / "models/xgboost.json")
    lstm = load_lstm_checkpoint(bundle, artifact_dir)
    return {"random_forest": rf, "xgboost": xgb_model, "lstm": lstm}


def _predict(model_name, model, prefix, bundle):
    start = time.perf_counter()
    if model_name in {"random_forest", "xgboost"}:
        output = model.predict(_tabular_input(prefix, bundle))
    else:
        output = predict_lstm(model, _sequence_input(prefix, bundle))
    try:
        if np.iscomplexobj(output):
            raise ValueError("Complex prediction")
        values = np.asarray(output, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} must return exactly one finite real prediction") from exc
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError(f"{model_name} must return exactly one finite prediction for the observed prefix")
    return max(float(values[0]), 0.0), time.perf_counter() - start


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


def run_experiments(
    bundle, artifact_dir: str | Path = "artifacts", random_seeds=(7, 17, 27), return_predictions: bool = False,
    *, expected_engine_ids=FD001_ENGINE_IDS,
):
    from numbers import Integral

    seeds = tuple(random_seeds)
    if (len(seeds) != 3 or set(seeds) != {7, 17, 27}
            or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) for seed in seeds)):
        raise ValueError("Official-test evaluation requires exactly random seeds 7, 17 and 27")
    raw = official_test_raw_frame(bundle)
    manifest = validate_official_test_inputs(raw, bundle.official_test_targets, bundle.selected_sensors, expected_engine_ids)
    if bundle.lstm_window != 30:
        raise ValueError("Official-test LSTM coverage requires the fixed 30-cycle window")
    artifact_dir = Path(artifact_dir)
    models = _load_models(bundle, artifact_dir)
    rows = []
    prediction_rows = []
    scenario_specs = [("clean", None, "median")]
    for scenario in ("random_10", "random_20"):
        for seed in seeds:
            for recovery in ("median", "forward_fill"):
                scenario_specs.append((scenario, seed, recovery))
    for recovery in ("median", "forward_fill"):
        scenario_specs.append(("short_gap", None, recovery))

    for scenario, seed, recovery in scenario_specs:
        scenario_predictions = []
        per_model_time = {name: 0.0 for name in models}
        gap_sensor = None
        for engine_id in manifest.index:
            prefix = raw[raw[ID_COLUMN] == engine_id].sort_values(CYCLE_COLUMN).copy()
            corrupted, chosen_gap_sensor = _corrupt(prefix, bundle, scenario, seed)
            gap_sensor = chosen_gap_sensor or gap_sensor
            recovered = _recover(corrupted, bundle, recovery)
            if not recovered[[ID_COLUMN, CYCLE_COLUMN]].equals(prefix[[ID_COLUMN, CYCLE_COLUMN]]):
                raise ValueError("Recovery must preserve observed engine/cycle rows and chronology")
            if not np.isfinite(recovered[bundle.selected_sensors].to_numpy(dtype=float)).all():
                raise ValueError("Non-finite sensor values remain after recovery")
            y = float(manifest.loc[engine_id, RUL_TRUTH_COLUMN])
            for name, model in models.items():
                pred, elapsed = _predict(name, model, recovered, bundle)
                if not np.isfinite(pred):
                    raise ValueError(f"{name} returned a non-finite official-test prediction")
                pred = max(float(pred), 0.0)
                per_model_time[name] += elapsed
                scenario_predictions.append(
                    {
                        "model": name,
                        "scenario": scenario,
                        "seed": seed,
                        "recovery": recovery,
                        "engine_id": int(engine_id),
                        "cycle": int(recovered[CYCLE_COLUMN].iloc[-1]),
                        "observed_history_length": len(prefix),
                        "sequence_window": bundle.lstm_window if name == "lstm" else None,
                        "sequence_history_length": min(len(prefix), bundle.lstm_window) if name == "lstm" else None,
                        "padding_cycles": max(0, bundle.lstm_window - len(prefix)) if name == "lstm" else None,
                        "rul_true": y,
                        "rul_pred": pred,
                        "error": pred - y,
                        "abs_error": abs(pred - y),
                    }
                )
        for name in models:
            checked = validate_official_test_predictions(
                pd.DataFrame([row for row in scenario_predictions if row["model"] == name]), manifest,
            )
            prediction_rows.extend(checked.to_dict("records"))
            metrics = regression_metrics(checked[RUL_TRUTH_COLUMN], checked.rul_pred)
            coverage = {}
            if name == "lstm":
                lengths = manifest.observed_history_length
                coverage = {
                    "n_padded_engines": int((lengths < bundle.lstm_window).sum()),
                    "n_unpadded_engines": int((lengths >= bundle.lstm_window).sum()),
                    "sequence_window": bundle.lstm_window,
                    "min_observed_history_length": int(lengths.min()),
                    "max_observed_history_length": int(lengths.max()),
                }
            rows.append(
                {
                    "model": name,
                    "scenario": scenario,
                    "seed": seed,
                    "recovery": recovery,
                    "gap_sensor": gap_sensor,
                    "n_engines": len(checked),
                    "prediction_seconds": per_model_time[name],
                    **metrics,
                    **coverage,
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
            delta_rmse_std=("delta_rmse", "std"),
            delta_mae_mean=("delta_mae", "mean"),
            delta_mae_std=("delta_mae", "std"),
        )
        .reset_index()
    )
