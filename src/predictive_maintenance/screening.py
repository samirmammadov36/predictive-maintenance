from __future__ import annotations

import pandas as pd


def screen_sensors(
    train_df: pd.DataFrame,
    sensor_columns: list[str],
    unique_ratio_threshold: float = 0.0005,
    dominant_fraction_threshold: float = 0.995,
) -> tuple[list[str], pd.DataFrame]:
    """Conservatively remove constant/near-constant sensors using training rows only."""
    rows = []
    n = len(train_df)
    selected: list[str] = []
    for col in sensor_columns:
        series = train_df[col].dropna()
        nunique = int(series.nunique(dropna=True))
        unique_ratio = nunique / max(n, 1)
        dominant_fraction = float(series.value_counts(normalize=True, dropna=False).iloc[0]) if len(series) else 1.0
        is_constant = nunique <= 1
        is_near_constant = unique_ratio <= unique_ratio_threshold and dominant_fraction >= dominant_fraction_threshold
        remove = is_constant or is_near_constant
        rows.append(
            {
                "sensor": col,
                "nunique": nunique,
                "unique_ratio": unique_ratio,
                "dominant_fraction": dominant_fraction,
                "removed": bool(remove),
                "reason": "constant" if is_constant else ("near_constant" if is_near_constant else "kept"),
            }
        )
        if not remove:
            selected.append(col)
    return selected, pd.DataFrame(rows)
