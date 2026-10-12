"""Reusable visualization and inspection-queue helpers for the FD001 demo.

This module intentionally contains no Streamlit code and no model-loading logic.
It can therefore be reused by the final dashboard regardless of which model
inference adapter is selected by the integration layer.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from matplotlib.figure import Figure

from .constants import CYCLE_COLUMN, ID_COLUMN

RUL_PREDICTION_COLUMN = "rul_pred"

# These sentences mirror the scope agreed in the project plan and can be shown
# directly by the Streamlit integration layer.
HEALTH_STATUS_NOTE = (
    "Health labels are configurable display rules derived from predicted RUL; "
    "they are not a separately trained classifier or validated maintenance thresholds."
)
QUEUE_NOTE = (
    "Engines can have different final observed cycles. The inspection queue is "
    "a demonstration, not a synchronized live fleet or an evaluated inspection policy."
)


def health_label(
    rul: float,
    *,
    healthy_threshold: float = 60.0,
    warning_threshold: float = 30.0,
) -> str:
    """Map a predicted RUL value to the demo health label.

    Project display rule:
    - RUL > 60 -> Healthy
    - 30 < RUL <= 60 -> Warning
    - RUL <= 30 -> Critical
    """
    if healthy_threshold <= warning_threshold:
        raise ValueError("healthy_threshold must be greater than warning_threshold")

    value = float(rul)
    if value > healthy_threshold:
        return "Healthy"
    if value > warning_threshold:
        return "Warning"
    return "Critical"


def observed_prefix(
    df: pd.DataFrame,
    engine_id: int,
    observed_cycle: int,
    *,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> pd.DataFrame:
    """Return only one engine's history visible at ``observed_cycle``.

    Future rows are never returned. This is the basic replay-safety boundary for
    dashboard charts and prediction inputs.
    """
    _require_columns(df, [id_column, cycle_column])
    if observed_cycle < 1:
        raise ValueError("observed_cycle must be at least 1")

    engine = df[df[id_column] == engine_id].sort_values(cycle_column)
    if engine.empty:
        raise ValueError(f"Engine {engine_id} not found")

    prefix = engine[engine[cycle_column] <= observed_cycle].copy()
    if prefix.empty:
        first_cycle = int(engine[cycle_column].min())
        raise ValueError(
            f"observed_cycle={observed_cycle} is before the first available cycle "
            f"({first_cycle}) for engine {engine_id}"
        )

    return prefix.reset_index(drop=True)


def top_k_queue(
    predictions: pd.DataFrame,
    k: int,
    *,
    id_column: str = ID_COLUMN,
    rul_column: str = RUL_PREDICTION_COLUMN,
    healthy_threshold: float = 60.0,
    warning_threshold: float = 30.0,
) -> pd.DataFrame:
    """Return the K engines with the lowest predicted RUL.

    Ties are broken by engine ID, matching the project demo specification.
    Input is expected to contain one final-snapshot prediction per engine.
    """
    if k not in {5, 10}:
        raise ValueError("k must be 5 or 10 for the project demonstration")
    _require_columns(predictions, [id_column, rul_column])

    queue = predictions.copy()
    if queue[id_column].duplicated().any():
        duplicates = sorted(queue.loc[queue[id_column].duplicated(keep=False), id_column].unique().tolist())
        raise ValueError(f"Expected one prediction per engine; duplicate engine IDs: {duplicates}")
    if queue[rul_column].isna().any():
        raise ValueError(f"{rul_column} contains missing values")

    queue = queue.sort_values([rul_column, id_column], ascending=[True, True]).head(k).copy()
    queue.insert(0, "rank", range(1, len(queue) + 1))
    queue["health_status"] = queue[rul_column].map(
        lambda value: health_label(
            value,
            healthy_threshold=healthy_threshold,
            warning_threshold=warning_threshold,
        )
    )
    return queue.reset_index(drop=True)


def plot_sensor_history(
    prefix: pd.DataFrame,
    sensor_columns: str | Iterable[str],
    *,
    cycle_column: str = CYCLE_COLUMN,
) -> Figure:
    """Plot one or more sensor histories from an already-causal engine prefix."""
    sensors = [sensor_columns] if isinstance(sensor_columns, str) else list(sensor_columns)
    if not sensors:
        raise ValueError("At least one sensor column is required")
    _require_columns(prefix, [cycle_column, *sensors])
    if prefix.empty:
        raise ValueError("prefix must contain at least one row")

    ordered = prefix.sort_values(cycle_column)
    fig = Figure()
    ax = fig.subplots()
    for sensor in sensors:
        ax.plot(ordered[cycle_column], ordered[sensor], label=sensor)

    ax.set_title("Observed sensor history")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor value")
    if len(sensors) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_rul_history(
    rul_history: pd.DataFrame,
    *,
    cycle_column: str = CYCLE_COLUMN,
    rul_column: str = RUL_PREDICTION_COLUMN,
) -> Figure:
    """Plot predicted RUL versus observed cycle for replay history."""
    _require_columns(rul_history, [cycle_column, rul_column])
    if rul_history.empty:
        raise ValueError("rul_history must contain at least one row")
    if rul_history[rul_column].isna().any():
        raise ValueError(f"{rul_column} contains missing values")

    ordered = rul_history.sort_values(cycle_column)
    fig = Figure()
    ax = fig.subplots()
    ax.plot(ordered[cycle_column], ordered[rul_column], marker="o")
    ax.set_title("Predicted Remaining Useful Life")
    ax.set_xlabel("Observed cycle")
    ax.set_ylabel("Predicted RUL (cycles)")
    fig.tight_layout()
    return fig


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
