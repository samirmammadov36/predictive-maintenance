from __future__ import annotations

import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN


def build_tabular_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    rolling_window: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Create causal tabular features shared by Random Forest and XGBoost."""
    ordered = df.sort_values([ID_COLUMN, CYCLE_COLUMN]).copy()
    feature_cols: list[str] = [CYCLE_COLUMN, *sensor_columns]

    for sensor in sensor_columns:
        roll_col = f"{sensor}_mean_{rolling_window}"
        diff_col = f"{sensor}_diff_{rolling_window}"

        ordered[roll_col] = (
            ordered.groupby(ID_COLUMN, sort=False)[sensor]
            .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
        )

        def causal_diff(s: pd.Series) -> pd.Series:
            arr = s.to_numpy()
            out = []
            for i, value in enumerate(arr):
                ref_idx = max(0, i - rolling_window)
                out.append(float(value - arr[ref_idx]))
            return pd.Series(out, index=s.index)

        ordered[diff_col] = ordered.groupby(ID_COLUMN, sort=False)[sensor].transform(causal_diff)
        feature_cols.extend([roll_col, diff_col])

    return ordered, feature_cols
