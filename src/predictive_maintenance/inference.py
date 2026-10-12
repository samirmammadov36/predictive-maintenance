"""Saved LSTM inference with an explicit boundary before sensor processing."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN
from .imputation import TrainingMedians
from .masks import _positive_integer_values, apply_sensor_mask, nested_random_masks, short_gap_mask
from .models.lstm import predict_lstm
from .models.lstm_checkpoint import load_lstm_checkpoint
from .sequences import build_prefix_window, scale_sequence_frame


@dataclass(frozen=True)
class LSTMInferenceResult:
    rul_pred: float
    engine_id: int
    dataset_split: str | None
    cycle: int
    observed_history_length: int
    sequence_window: int
    padding_cycles: int
    chart_prefix: pd.DataFrame
    sensor_mask: pd.DataFrame


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _visible_prefix(raw, sensors, engine_id, observed_cycle, dataset_split):
    if not isinstance(raw, pd.DataFrame) or raw.empty or not raw.columns.is_unique:
        raise ValueError("Raw trajectories require a non-empty DataFrame with unique columns")
    columns = [ID_COLUMN, CYCLE_COLUMN, *sensors]
    if any(column not in raw for column in columns):
        raise ValueError(f"Raw trajectories require columns: {columns}")
    if dataset_split is not None and (not isinstance(dataset_split, str) or not dataset_split.strip()):
        raise ValueError("dataset_split must be a non-empty string")
    selected = raw[ID_COLUMN].eq(engine_id).fillna(False)
    if not selected.any():
        raise ValueError(f"Engine {engine_id} was not found")
    resolved_split = None
    if "dataset_split" in raw:
        columns.insert(0, "dataset_split")
        splits = raw.loc[selected, "dataset_split"]
        if dataset_split is None:
            if splits.isna().any() or splits.nunique() != 1:
                raise ValueError("Ambiguous or missing engine identity; supply dataset_split")
            dataset_split = splits.iloc[0]
            if not isinstance(dataset_split, str) or not dataset_split.strip():
                raise ValueError("dataset_split must identify a non-empty string namespace")
        selected &= raw["dataset_split"].eq(dataset_split).fillna(False)
        resolved_split = dataset_split
    elif dataset_split is not None:
        raise ValueError("dataset_split was supplied but raw trajectories have no such column")
    if not selected.any():
        raise ValueError(f"Engine identity ({dataset_split}, {engine_id}) was not found")
    # Inspect cycle metadata to establish visibility, never hidden sensor values or targets.
    positions = np.flatnonzero(selected.to_numpy(dtype=bool))
    cycles = _positive_integer_values(raw.iloc[positions][CYCLE_COLUMN], CYCLE_COLUMN)
    visible = cycles.to_numpy() <= observed_cycle
    positions = positions[visible]
    if not len(positions):
        raise ValueError("Observed cutoff produces an empty visible engine prefix")
    prefix = raw.iloc[positions][columns].copy(deep=True)
    prefix[CYCLE_COLUMN] = cycles.to_numpy()[visible]
    prefix = prefix.sort_values(CYCLE_COLUMN)
    try:
        if np.iscomplexobj(prefix[sensors].to_numpy()):
            raise ValueError("Complex sensor values")
        values = prefix[sensors].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Visible raw sensors must be numeric, with native NaNs allowed") from exc
    if np.isinf(values).any():
        raise ValueError("Visible raw sensors may contain NaN, but not infinity")
    # Normalize only visible measurements; future object-valued rows cannot alter chart dtypes.
    prefix[sensors] = values
    return prefix, resolved_split


class LSTMInferenceAdapter:
    """Load once; predict from explicitly bounded raw engine trajectories."""

    def __init__(self, bundle, artifact_dir: str | Path = "artifacts"):
        self._model = load_lstm_checkpoint(bundle, artifact_dir)
        self._sensors = tuple(bundle.selected_sensors)
        self._window = bundle.lstm_window
        self._scaler = copy.deepcopy(bundle.sequence_scaler)
        if not isinstance(bundle.medians, TrainingMedians):
            raise ValueError("LSTM inference requires fitted TrainingMedians")
        self._medians = copy.deepcopy(bundle.medians)
        try:
            medians = np.asarray([self._medians.values[sensor] for sensor in self._sensors], dtype=float)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Training medians must contain every frozen sensor") from exc
        if not np.isfinite(medians).all():
            raise ValueError("Training medians must be finite for every frozen sensor")

    def predict(
        self, raw_trajectories: pd.DataFrame, engine_id: int, observed_cycle: int, *,
        dataset_split: str | None = None, scenario: str = "clean", recovery: str = "median",
        seed: int | None = None,
    ) -> LSTMInferenceResult:
        engine_id = _positive_integer(engine_id, "engine_id")
        observed_cycle = _positive_integer(observed_cycle, "observed_cycle")
        if not isinstance(scenario, str) or scenario not in {"clean", "random_10", "random_20", "short_gap"}:
            raise ValueError("Unknown inference scenario")
        if not isinstance(recovery, str) or recovery not in {"median", "forward_fill"}:
            raise ValueError("Recovery must be median or forward_fill")
        if scenario == "clean" and recovery != "median":
            raise ValueError("Clean inference requires training-median recovery")
        if scenario in {"random_10", "random_20"}:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed not in {7, 17, 27}:
                raise ValueError("Random inference requires seed 7, 17 or 27")
        elif seed is not None:
            raise ValueError("seed applies only to random missingness scenarios")
        sensors = list(self._sensors)
        prefix, resolved_split = _visible_prefix(
            raw_trajectories, sensors, engine_id, observed_cycle, dataset_split,
        )
        if scenario in {"random_10", "random_20"}:
            low, high = nested_random_masks(prefix, sensors, int(seed))
            mask = low if scenario == "random_10" else high
        elif scenario == "short_gap":
            mask, _ = short_gap_mask(prefix, sensors)
        else:
            mask = pd.DataFrame(False, index=prefix.index, columns=sensors)
        # Also validates clean visible identities/cycles/indices through the shared mask helper.
        corrupted = apply_sensor_mask(prefix, mask, sensors)
        if recovery == "median":
            recovered = self._medians.median_fill(corrupted, sensors)
        else:
            recovered = self._medians.causal_forward_fill(corrupted, sensors)
        scaled = scale_sequence_frame(recovered, sensors, self._scaler)
        window = build_prefix_window(scaled, sensors, self._window)
        output = predict_lstm(self._model, np.expand_dims(window, axis=0))
        try:
            if np.iscomplexobj(output):
                raise ValueError("Complex prediction")
            values = np.asarray(output, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("LSTM must return one finite real RUL prediction") from exc
        if len(values) != 1 or not np.isfinite(values).all():
            raise ValueError("LSTM must return one finite RUL prediction before negative clamping")
        return LSTMInferenceResult(
            rul_pred=max(float(values[0]), 0.0), engine_id=engine_id, dataset_split=resolved_split,
            cycle=int(prefix[CYCLE_COLUMN].iloc[-1]), observed_history_length=len(prefix),
            sequence_window=self._window, padding_cycles=max(0, self._window - len(prefix)),
            chart_prefix=recovered.copy(deep=True), sensor_mask=mask.copy(deep=True),
        )
