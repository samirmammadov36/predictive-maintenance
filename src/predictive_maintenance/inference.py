"""Reusable causal inference helpers for the predictive-maintenance dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN
from predictive_maintenance.dashboard import observed_prefix
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.masks import (
    apply_sensor_mask,
    nested_random_masks,
    short_gap_mask,
)
from predictive_maintenance.models.lstm import predict_lstm
from predictive_maintenance.sequences import (
    build_prefix_window,
    scale_sequence_frame,
)


def corrupt_prefix(
    prefix: pd.DataFrame,
    bundle,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    """Apply a deterministic missing-sensor scenario to one visible prefix."""
    sensors = bundle.selected_sensors

    if scenario == "Clean":
        return prefix.copy()

    if scenario in {"Random 10%", "Random 20%"}:
        low_mask, high_mask = nested_random_masks(prefix, sensors, seed)
        mask = low_mask if scenario == "Random 10%" else high_mask

    elif scenario == "Five-cycle sensor gap":
        mask, _ = short_gap_mask(
            prefix,
            sensors,
            preferred_sensor="sensor_11",
            gap_cycles=5,
        )

    else:
        raise ValueError(f"Unknown missingness scenario: {scenario}")

    return apply_sensor_mask(prefix, mask, sensors)


def recover_prefix(
    prefix: pd.DataFrame,
    bundle,
    recovery: str,
) -> pd.DataFrame:
    """Recover missing sensor values without using future information."""
    if recovery == "Training median":
        return bundle.medians.median_fill(
            prefix,
            bundle.selected_sensors,
        )

    if recovery == "Causal forward fill":
        return bundle.medians.causal_forward_fill(
            prefix,
            bundle.selected_sensors,
        )

    raise ValueError(f"Unknown recovery method: {recovery}")


def predict_prefix(
    model_name: str,
    model,
    prefix: pd.DataFrame,
    bundle,
) -> float:
    """Predict RUL using only the rows supplied in the causal prefix."""
    if prefix.empty:
        raise ValueError("At least one observed cycle is required.")

    if model_name in {"Random Forest", "XGBoost"}:
        features, _ = build_tabular_features(
            prefix,
            bundle.selected_sensors,
            bundle.rolling_window,
        )

        latest = features.iloc[[-1]][bundle.feature_columns]
        matrix = bundle.tabular_scaler.transform(latest)

        prediction = float(model.predict(matrix)[0])

    elif model_name == "LSTM":
        scaled = scale_sequence_frame(
            prefix,
            bundle.selected_sensors,
            bundle.sequence_scaler,
        )

        window = build_prefix_window(
            scaled,
            bundle.selected_sensors,
            bundle.lstm_window,
        )

        prediction = float(
            predict_lstm(
                model,
                np.expand_dims(window, axis=0),
            )[0]
        )

    elif model_name == "Median baseline":
        prediction = float(model.predict(1)[0])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return max(0.0, prediction)


def predict_with_scenario(
    model_name: str,
    model,
    prefix: pd.DataFrame,
    bundle,
    scenario: str,
    recovery: str,
    seed: int,
) -> float:
    """Apply missingness and recovery, then predict RUL for one prefix."""
    corrupted = corrupt_prefix(
        prefix,
        bundle,
        scenario,
        seed,
    )

    recovered = recover_prefix(
        corrupted,
        bundle,
        recovery,
    )

    if recovered[bundle.selected_sensors].isna().any().any():
        raise ValueError(
            "Missing sensor values remain after recovery."
        )

    return predict_prefix(
        model_name,
        model,
        recovered,
        bundle,
    )


def build_replay(
    engine: pd.DataFrame,
    observed_cycle: int,
    model_name: str,
    model,
    bundle,
    scenario: str,
    recovery: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return visible history and causal cycle-by-cycle RUL predictions."""
    visible = observed_prefix(
        engine,
        int(engine[ID_COLUMN].iloc[0]),
        observed_cycle,
    )

    rows = []

    for cycle in visible[CYCLE_COLUMN].tolist():
        # Only use information available at this exact replay cycle.
        cycle_prefix = visible[
            visible[CYCLE_COLUMN] <= cycle
        ].copy()

        prediction = predict_with_scenario(
            model_name,
            model,
            cycle_prefix,
            bundle,
            scenario,
            recovery,
            seed,
        )

        rows.append(
            {
                CYCLE_COLUMN: int(cycle),
                "rul_pred": prediction,
            }
        )

    return visible, pd.DataFrame(rows)