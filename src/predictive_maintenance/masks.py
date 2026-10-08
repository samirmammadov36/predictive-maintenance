from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN, SENSOR_COLUMNS


def _engine_rng_seed(seed: int, engine_id: int) -> int:
    return int(seed + 1009 * int(engine_id))


def _positive_integer_values(values: pd.Series, name: str) -> pd.Series:
    invalid_types = values.map(
        lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
    )
    if values.isna().any() or invalid_types.any():
        raise ValueError(f"{name} values must be finite positive integers")
    try:
        numeric = pd.to_numeric(values, errors="raise")
        valid = np.isfinite(numeric) & (numeric > 0) & (numeric % 1 == 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} values must be finite positive integers") from exc
    if not valid.all():
        raise ValueError(f"{name} values must be finite positive integers")
    return numeric


def _validate_prefix(prefix_df: pd.DataFrame, sensor_columns: list[str]) -> int:
    if not isinstance(prefix_df, pd.DataFrame) or prefix_df.empty:
        raise ValueError("Mask construction requires a non-empty observed prefix")
    if not prefix_df.columns.is_unique or not prefix_df.index.is_unique:
        raise ValueError("Prefix row indices and column labels must be unique")
    if not isinstance(sensor_columns, (list, tuple)) or not sensor_columns:
        raise ValueError("sensor_columns must be a non-empty ordered sensor list")
    if any(not isinstance(sensor, str) or sensor not in SENSOR_COLUMNS for sensor in sensor_columns):
        raise ValueError("sensor_columns must contain selected FD001 sensor names only")
    if len(set(sensor_columns)) != len(sensor_columns):
        raise ValueError("sensor_columns must not contain duplicate sensors")
    missing_sensors = [sensor for sensor in sensor_columns if sensor not in prefix_df.columns]
    if missing_sensors:
        raise ValueError(f"Selected sensor columns are missing: {missing_sensors}")
    identity = [ID_COLUMN]
    if "dataset_split" in prefix_df.columns:
        identity.insert(0, "dataset_split")
    if any(column not in prefix_df.columns for column in identity):
        raise ValueError("Prefix must contain engine_id")
    if prefix_df[identity].isna().any().any():
        raise ValueError("Engine identity values must not be missing")
    if "dataset_split" in identity and prefix_df["dataset_split"].map(
        lambda value: isinstance(value, str) and not value.strip()
    ).any():
        raise ValueError("dataset_split values must not be missing")
    if len(prefix_df[identity].drop_duplicates()) != 1:
        raise ValueError("Mask construction expects exactly one engine identity")
    engine_ids = _positive_integer_values(prefix_df[ID_COLUMN], ID_COLUMN)
    if CYCLE_COLUMN not in prefix_df.columns:
        raise ValueError("Prefix must contain cycle values")
    cycles = _positive_integer_values(prefix_df[CYCLE_COLUMN], CYCLE_COLUMN)
    if cycles.duplicated().any():
        raise ValueError("Duplicate cycle values within the engine prefix")
    if not cycles.is_monotonic_increasing:
        raise ValueError("Prefix cycles must be in chronological order")
    return int(engine_ids.iloc[0])


def nested_random_masks(
    prefix_df: pd.DataFrame,
    sensor_columns: list[str],
    seed: int,
    low_rate: float = 0.10,
    high_rate: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nested boolean masks; low-rate cells are always a subset of high-rate cells."""
    engine_id = _validate_prefix(prefix_df, sensor_columns)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, rate in (("low_rate", low_rate), ("high_rate", high_rate)):
        if (
            isinstance(rate, (bool, np.bool_))
            or not isinstance(rate, Real)
            or not np.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError(f"{name} must be a finite number between 0 and 1")
    if low_rate > high_rate:
        raise ValueError("low_rate must not exceed high_rate")
    rng = np.random.default_rng(_engine_rng_seed(seed, engine_id))
    n_cells = len(prefix_df) * len(sensor_columns)
    order = rng.permutation(n_cells)
    low_n = int(round(low_rate * n_cells))
    high_n = int(round(high_rate * n_cells))
    low_flat = np.zeros(n_cells, dtype=bool)
    high_flat = np.zeros(n_cells, dtype=bool)
    low_flat[order[:low_n]] = True
    high_flat[order[:high_n]] = True
    index = prefix_df.index
    low = pd.DataFrame(low_flat.reshape(len(prefix_df), len(sensor_columns)), index=index, columns=sensor_columns)
    high = pd.DataFrame(high_flat.reshape(len(prefix_df), len(sensor_columns)), index=index, columns=sensor_columns)
    return low, high


def short_gap_mask(
    prefix_df: pd.DataFrame,
    sensor_columns: list[str],
    preferred_sensor: str = "sensor_11",
    gap_cycles: int = 5,
) -> tuple[pd.DataFrame, str]:
    _validate_prefix(prefix_df, sensor_columns)
    if (
        isinstance(gap_cycles, (bool, np.bool_))
        or not isinstance(gap_cycles, Integral)
        or gap_cycles <= 0
    ):
        raise ValueError("gap_cycles must be a positive integer")
    if not isinstance(preferred_sensor, str) or preferred_sensor not in SENSOR_COLUMNS:
        raise ValueError("preferred_sensor must be an FD001 sensor name")
    sensor = preferred_sensor if preferred_sensor in sensor_columns else sensor_columns[0]
    mask = pd.DataFrame(False, index=prefix_df.index, columns=sensor_columns)
    tail_index = prefix_df.index[-min(gap_cycles, len(prefix_df)) :]
    mask.loc[tail_index, sensor] = True
    return mask, sensor


def apply_sensor_mask(df: pd.DataFrame, mask: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    _validate_prefix(df, sensor_columns)
    if not isinstance(mask, pd.DataFrame):
        raise ValueError("mask must be a boolean DataFrame")
    if not mask.index.is_unique or not mask.columns.is_unique:
        raise ValueError("Mask row indices and column labels must be unique")
    if len(mask.index) != len(df.index) or not df.index.isin(mask.index).all():
        raise ValueError("Mask row labels must exactly match the observed prefix")
    if len(mask.columns) != len(sensor_columns) or not pd.Index(sensor_columns).isin(mask.columns).all():
        raise ValueError("Mask columns must exactly match the selected sensor columns")
    if mask.isna().any().any() or not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes):
        raise ValueError("Mask cells must be boolean values without missing values")
    out = df.copy()
    aligned = mask.reindex(index=out.index, columns=sensor_columns)
    out.loc[:, sensor_columns] = out[sensor_columns].mask(aligned)
    return out
