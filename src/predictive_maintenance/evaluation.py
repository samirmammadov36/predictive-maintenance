from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN, RUL_TRUTH_COLUMN, SENSOR_COLUMNS

FD001_ENGINE_IDS = tuple(range(1, 101))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def _positive_integer_column(values: pd.Series, name: str) -> None:
    if values.isna().any() or values.map(
        lambda value: isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
    ).any():
        raise ValueError(f"{name} must contain finite positive integers")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any() or (numeric % 1 != 0).any():
        raise ValueError(f"{name} must contain finite positive integers")


def validate_official_test_inputs(raw, targets, sensor_columns, expected_engine_ids=FD001_ENGINE_IDS):
    """Build a label-aligned coverage manifest; native sensor NaNs remain raw."""
    expected = tuple(expected_engine_ids)
    if (not expected or len(set(expected)) != len(expected)
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0 for value in expected)):
        raise ValueError("Expected official-test engine IDs must be unique positive integers")
    if (not isinstance(sensor_columns, (list, tuple)) or not sensor_columns
            or any(sensor not in SENSOR_COLUMNS for sensor in sensor_columns)
            or len(set(sensor_columns)) != len(sensor_columns)):
        raise ValueError("Selected sensors must be a unique ordered FD001 sensor list")
    for frame, name, columns in (
        (raw, "raw official-test inputs", [ID_COLUMN, CYCLE_COLUMN, *sensor_columns]),
        (targets, "official-test labels", [ID_COLUMN, CYCLE_COLUMN, RUL_TRUTH_COLUMN]),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError(f"Invalid {name}: require a non-empty DataFrame with unique columns")
        if any(column not in frame for column in columns):
            raise ValueError(f"Missing columns in {name}: {columns}")
        _positive_integer_column(frame[ID_COLUMN], f"{name} engine IDs")
        _positive_integer_column(frame[CYCLE_COLUMN], f"{name} cycles")
        if set(frame[ID_COLUMN]) != set(expected):
            raise ValueError(f"{name} must contain exactly the expected {len(expected)} engine identities")
    splits = []
    for frame in (raw, targets):
        if "dataset_split" in frame:
            if frame.dataset_split.isna().any() or frame.dataset_split.nunique() != 1:
                raise ValueError("Official-test inputs must contain one non-missing dataset_split identity")
            split = frame.dataset_split.iloc[0]
            if isinstance(split, str) and not split.strip():
                raise ValueError("Official-test dataset_split identity must not be empty")
            splits.append(split)
    if len(splits) == 2 and splits[0] != splits[1]:
        raise ValueError("Official-test input/label dataset_split identity mismatch")
    if not raw.index.is_unique or raw.duplicated([ID_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError("Raw official-test row labels and engine/cycle pairs must be unique")
    if targets[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate official-test labels for an engine")
    try:
        if np.iscomplexobj(raw[list(sensor_columns)].to_numpy()):
            raise ValueError("Complex sensors")
        measurements = raw[list(sensor_columns)].to_numpy(dtype=float, na_value=np.nan)
        truth = targets[RUL_TRUTH_COLUMN].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("Official-test sensors and labels must be numeric") from exc
    if np.isinf(measurements).any():
        raise ValueError("Raw official-test sensors may contain NaN, but not infinity")
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("Official-test RUL labels must be finite and non-negative; do not cap them")
    manifest = raw.groupby(ID_COLUMN)[CYCLE_COLUMN].agg(cycle="max", observed_history_length="size")
    manifest = manifest.loc[sorted(expected)].copy()
    ordered_labels = targets.set_index(ID_COLUMN).loc[manifest.index]
    if not np.array_equal(manifest[CYCLE_COLUMN], ordered_labels[CYCLE_COLUMN]):
        raise ValueError("Official-test label cycle does not match the final observed cycle")
    manifest[RUL_TRUTH_COLUMN] = ordered_labels[RUL_TRUTH_COLUMN].to_numpy(dtype=float)
    return manifest


def validate_official_test_predictions(predictions, manifest):
    """Require one final-cycle prediction per engine, align labels and clamp negatives."""
    required = [ID_COLUMN, CYCLE_COLUMN, RUL_TRUTH_COLUMN, "rul_pred"]
    if (not isinstance(predictions, pd.DataFrame) or not predictions.columns.is_unique
            or any(column not in predictions for column in required)):
        raise ValueError("Official-test predictions require engine_id, cycle, rul_true and rul_pred")
    if (len(predictions) != len(manifest) or predictions[ID_COLUMN].duplicated().any()
            or set(predictions[ID_COLUMN]) != set(manifest.index)):
        raise ValueError("Official-test prediction coverage mismatch: require exactly one prediction per expected engine")
    _positive_integer_column(predictions[ID_COLUMN], "Prediction engine IDs")
    _positive_integer_column(predictions[CYCLE_COLUMN], "Prediction cycles")
    ordered = predictions.set_index(ID_COLUMN).loc[manifest.index].copy()
    if not np.array_equal(ordered[CYCLE_COLUMN], manifest[CYCLE_COLUMN]):
        raise ValueError("Prediction cycle does not match the final observed cycle")
    if not np.array_equal(ordered[RUL_TRUTH_COLUMN], manifest[RUL_TRUTH_COLUMN]):
        raise ValueError("Prediction labels do not match official uncapped RUL labels")
    try:
        values = ordered.rul_pred.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Official-test predictions must be finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Official-test predictions must be finite before negative clamping")
    ordered["rul_pred"] = np.maximum(values, 0.0)
    if "error" in ordered:
        ordered["error"] = ordered.rul_pred - ordered[RUL_TRUTH_COLUMN]
    if "abs_error" in ordered:
        ordered["abs_error"] = (ordered.rul_pred - ordered[RUL_TRUTH_COLUMN]).abs()
    return ordered.reset_index()
