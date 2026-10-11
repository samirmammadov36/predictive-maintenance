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
    official_test_raw_df: pd.DataFrame | None = None


def official_test_raw_frame(bundle) -> pd.DataFrame:
    raw = getattr(bundle, "official_test_raw_df", None)
    if not isinstance(raw, pd.DataFrame):
        raise ValueError(
            "Prepared bundle lacks raw official-test inputs. Regenerate with `python scripts/prepare_data.py` "
            "in the project .venv using original FD001 files; median-imputed official_test_df cannot be used as raw."
        )
    return raw
