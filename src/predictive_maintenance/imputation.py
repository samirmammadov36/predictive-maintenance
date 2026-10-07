from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .constants import ID_COLUMN


@dataclass
class TrainingMedians:
    values: dict[str, float]

    @classmethod
    def fit(cls, train_df: pd.DataFrame, sensor_columns: list[str]) -> "TrainingMedians":
        medians = train_df[sensor_columns].median(axis=0)
        if medians.isna().any():
            bad = medians[medians.isna()].index.tolist()
            raise ValueError(f"Cannot fit medians; all values missing for: {bad}")
        return cls({k: float(v) for k, v in medians.items()})

    def median_fill(self, df: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        out[sensor_columns] = out[sensor_columns].fillna(self.values)
        return out

    def causal_forward_fill(self, df: pd.DataFrame, sensor_columns: list[str]) -> pd.DataFrame:
        """Forward-fill within each engine only, then use training medians as fallback."""
        out = df.copy()
        out[sensor_columns] = (
            out.groupby(ID_COLUMN, sort=False)[sensor_columns]
            .ffill()
            .fillna(self.values)
        )
        return out
