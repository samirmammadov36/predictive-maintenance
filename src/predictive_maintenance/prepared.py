from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class PreparedBundle:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    official_test_df: pd.DataFrame
    official_test_targets: pd.DataFrame
    selected_sensors: list[str]
    feature_columns: list[str]
    train_engine_ids: list[int]
    validation_engine_ids: list[int]
    validation_prefix_cycles: dict[int, int]
    medians: object
    sequence_scaler: object
    tabular_scaler: object
    rolling_window: int
    lstm_window: int
