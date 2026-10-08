from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import CYCLE_COLUMN, ID_COLUMN


def fit_sequence_scaler(train_df: pd.DataFrame, sensor_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[sensor_columns])
    return scaler


def scale_sequence_frame(df: pd.DataFrame, sensor_columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[sensor_columns] = scaler.transform(out[sensor_columns])
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
