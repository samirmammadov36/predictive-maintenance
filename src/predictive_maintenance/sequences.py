from __future__ import annotations

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


def left_padded_window(values: np.ndarray, end_index: int, window: int) -> np.ndarray:
    start = max(0, end_index - window + 1)
    chunk = values[start : end_index + 1]
    if len(chunk) < window:
        pad = np.repeat(chunk[[0]], window - len(chunk), axis=0)
        chunk = np.vstack([pad, chunk])
    return chunk


def build_training_windows(
    df: pd.DataFrame,
    sensor_columns: list[str],
    target_column: str,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _, engine in df.sort_values([ID_COLUMN, CYCLE_COLUMN]).groupby(ID_COLUMN, sort=False):
        values = engine[sensor_columns].to_numpy(dtype=np.float32)
        targets = engine[target_column].to_numpy(dtype=np.float32)
        for i in range(len(engine)):
            xs.append(left_padded_window(values, i, window))
            ys.append(float(targets[i]))
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def build_prefix_window(
    prefix_df: pd.DataFrame,
    sensor_columns: list[str],
    window: int = 30,
) -> np.ndarray:
    values = prefix_df.sort_values(CYCLE_COLUMN)[sensor_columns].to_numpy(dtype=np.float32)
    return left_padded_window(values, len(values) - 1, window)
