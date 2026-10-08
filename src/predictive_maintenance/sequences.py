from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .constants import CYCLE_COLUMN, ID_COLUMN, SENSOR_COLUMNS


def _validate_sensor_columns(sensor_columns: list[str]) -> None:
    if not isinstance(sensor_columns, (list, tuple)) or not sensor_columns:
        raise ValueError("sensor_columns must be a non-empty ordered list")
    if any(not isinstance(column, str) or column not in SENSOR_COLUMNS for column in sensor_columns):
        raise ValueError("sensor_columns must contain valid FD001 sensor names only")
    if len(set(sensor_columns)) != len(sensor_columns):
        raise ValueError("sensor_columns must not contain duplicate names")


def _sequence_sensor_frame(df: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    _validate_sensor_columns(sensor_columns)
    if df.empty:
        raise ValueError("Sequence scaling input must not be empty")
    missing = [column for column in sensor_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    if df.columns[df.columns.duplicated()].isin(sensor_columns).any():
        raise ValueError("Selected sensor columns have duplicate DataFrame labels")
    sensors = df[list(sensor_columns)]
    try:
        if np.iscomplexobj(sensors.to_numpy()):
            raise ValueError
        values = sensors.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Selected sensor columns must contain finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Selected sensor columns must contain finite numeric values")
    return sensors


def validate_sequence_scaler(scaler: StandardScaler, sensor_columns: list[str]) -> None:
    """Check the fitted input schema without changing scaler statistics."""
    _validate_sensor_columns(sensor_columns)
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Sequence scaler must be a StandardScaler")
    try:
        check_is_fitted(scaler)
    except ValueError as exc:
        raise ValueError("Sequence scaler must be fitted before use") from exc
    if scaler.n_features_in_ != len(sensor_columns):
        raise ValueError("Sensor count does not match the fitted sequence scaler")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != list(sensor_columns):
        raise ValueError("Sensor order does not match the fitted sequence scaler")


def fit_sequence_scaler(train_df: pd.DataFrame, sensor_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(_sequence_sensor_frame(train_df, sensor_columns))
    return scaler


def scale_sequence_frame(df: pd.DataFrame, sensor_columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    sensors = _sequence_sensor_frame(df, sensor_columns)
    validate_sequence_scaler(scaler, sensor_columns)
    out = df.copy()
    out[list(sensor_columns)] = scaler.transform(sensors)
    return out


def _validate_window(window: int) -> None:
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("window must be a positive integer")


def _ordered_observations(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    _validate_window(window)
    if df.empty:
        raise ValueError("Sequence input must not be empty")
    identity_columns = [ID_COLUMN]
    if "dataset_split" in df.columns:
        identity_columns.insert(0, "dataset_split")
    for column in identity_columns:
        if column not in df.columns or df[column].isna().any():
            raise ValueError(f"Engine identity requires non-missing {column} values")
    if CYCLE_COLUMN not in df.columns:
        raise ValueError("Sequence input requires cycle values")
    try:
        cycles = pd.to_numeric(df[CYCLE_COLUMN], errors="raise")
        if np.iscomplexobj(cycles) or df[CYCLE_COLUMN].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError
        numeric_cycles = cycles.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cycle values must be finite positive integers") from exc
    valid_cycles = (
        np.isfinite(numeric_cycles)
        & (numeric_cycles > 0)
        & (numeric_cycles == np.floor(numeric_cycles))
    )
    if not valid_cycles.all():
        raise ValueError("cycle values must be finite positive integers")
    ordered = df.copy()
    ordered[CYCLE_COLUMN] = cycles
    if ordered.duplicated([*identity_columns, CYCLE_COLUMN]).any():
        raise ValueError("Duplicate cycle values within an engine identity")
    return ordered.sort_values([*identity_columns, CYCLE_COLUMN]), identity_columns


def _sensor_array(df: pd.DataFrame, sensor_columns: list[str]) -> np.ndarray:
    try:
        if np.iscomplexobj(df[sensor_columns].to_numpy()):
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            values = df[sensor_columns].to_numpy(dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Sensor values must be finite numeric values; impute before sequence construction") from exc
    if not np.isfinite(values).all():
        raise ValueError("Sensor values must be finite in float32; impute before sequence construction")
    return values


def _window_from_values(values: np.ndarray, end_index: int, window: int) -> np.ndarray:
    start = max(0, end_index - window + 1)
    chunk = values[start : end_index + 1]
    if len(chunk) < window:
        pad = np.repeat(chunk[[0]], window - len(chunk), axis=0)
        chunk = np.vstack([pad, chunk])
    return chunk


def left_padded_window(values: np.ndarray, end_index: int, window: int) -> np.ndarray:
    _validate_window(window)
    values = np.asarray(values)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Sensor observations must be a non-empty two-dimensional array")
    try:
        finite = np.isfinite(values).all()
    except TypeError as exc:
        raise ValueError("Sensor values must be finite numeric values") from exc
    if not finite:
        raise ValueError("Sensor values must be finite; impute before sequence construction")
    if (
        isinstance(end_index, (bool, np.bool_))
        or not isinstance(end_index, Integral)
        or not 0 <= end_index < len(values)
    ):
        raise ValueError("end_index must identify an available observation")
    return _window_from_values(values, end_index, window)


def build_training_windows(
    df: pd.DataFrame,
    sensor_columns: list[str],
    target_column: str,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    ordered, identity_columns = _ordered_observations(df, window)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _, engine in ordered.groupby(identity_columns, sort=False, observed=True):
        values = _sensor_array(engine, sensor_columns)
        targets = engine[target_column].to_numpy(dtype=np.float32)
        for i in range(len(engine)):
            xs.append(_window_from_values(values, i, window))
            ys.append(float(targets[i]))
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def build_prefix_window(
    prefix_df: pd.DataFrame,
    sensor_columns: list[str],
    window: int = 30,
) -> np.ndarray:
    """Return the final window of an already-truncated, imputed engine prefix."""
    ordered, identity_columns = _ordered_observations(prefix_df, window)
    if len(ordered[identity_columns].drop_duplicates()) != 1:
        raise ValueError("Prefix input must contain exactly one engine identity")
    values = _sensor_array(ordered, sensor_columns)
    return _window_from_values(values, len(values) - 1, window)
