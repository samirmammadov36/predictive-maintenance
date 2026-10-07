from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ID_COLUMN


def _engine_rng_seed(seed: int, engine_id: int) -> int:
    return int(seed + 1009 * int(engine_id))


def nested_random_masks(
    prefix_df: pd.DataFrame,
    sensor_columns: list[str],
    seed: int,
    low_rate: float = 0.10,
    high_rate: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nested boolean masks; low-rate cells are always a subset of high-rate cells."""
    engine_ids = prefix_df[ID_COLUMN].unique()
    if len(engine_ids) != 1:
        raise ValueError("Mask generation expects exactly one engine prefix")
    rng = np.random.default_rng(_engine_rng_seed(seed, int(engine_ids[0])))
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
    sensor = preferred_sensor if preferred_sensor in sensor_columns else sensor_columns[0]
    mask = pd.DataFrame(False, index=prefix_df.index, columns=sensor_columns)
    tail_index = prefix_df.index[-min(gap_cycles, len(prefix_df)) :]
    mask.loc[tail_index, sensor] = True
    return mask, sensor


def apply_sensor_mask(df: pd.DataFrame, mask: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    aligned = mask.reindex(index=out.index, columns=sensor_columns, fill_value=False)
    out.loc[:, sensor_columns] = out[sensor_columns].mask(aligned)
    return out
