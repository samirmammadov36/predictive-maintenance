"""Saved-model inference with a shared explicit boundary before sensor processing."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .constants import CYCLE_COLUMN, ID_COLUMN
from .features import build_tabular_features
from .imputation import TrainingMedians
from .masks import _positive_integer_values, apply_sensor_mask, nested_random_masks, short_gap_mask
from .models.baseline import MedianRULBaseline
from .sequences import build_prefix_window, scale_sequence_frame, _validate_sensor_columns


def predict_lstm(*args, **kwargs):
    from .models.lstm import predict_lstm as predict
    return predict(*args, **kwargs)


def load_lstm_checkpoint(*args, **kwargs):
    from .models.lstm_checkpoint import load_lstm_checkpoint as load
    return load(*args, **kwargs)


@dataclass(frozen=True)
class InferenceResult:
    rul_pred: float
    engine_id: int
    dataset_split: str | None
    cycle: int
    observed_history_length: int
    sequence_window: int | None
    padding_cycles: int
    chart_prefix: pd.DataFrame
    sensor_mask: pd.DataFrame


LSTMInferenceResult = InferenceResult  # Preserve the existing public result name.


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



def _finite_rul(output, label: str) -> float:
    try:
        if np.iscomplexobj(output):
            raise ValueError("Complex prediction")
        values = np.asarray(output, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must return one finite real RUL prediction") from exc
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError(f"{label} must return one finite RUL prediction before negative clamping")
    return max(float(values[0]), 0.0)


class _PrefixInferenceAdapter:
    """Shared raw boundary, masking/recovery and result contract; no fitting."""
    _window = None
    _label = "Saved model"

    def _freeze_recovery(self, bundle):
        _validate_sensor_columns(bundle.selected_sensors)
        self._sensors = tuple(bundle.selected_sensors)
        if not isinstance(bundle.medians, TrainingMedians):
            raise ValueError("Saved inference requires fitted TrainingMedians")
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
    ) -> InferenceResult:
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
        corrupted = apply_sensor_mask(prefix, mask, sensors)
        if recovery == "median":
            recovered = self._medians.median_fill(corrupted, sensors)
        else:
            recovered = self._medians.causal_forward_fill(corrupted, sensors)
        if not np.isfinite(recovered[sensors].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite sensor values remain after recovery")
        prediction = _finite_rul(self._model_output(recovered), self._label)
        return InferenceResult(
            rul_pred=prediction, engine_id=engine_id, dataset_split=resolved_split,
            cycle=int(prefix[CYCLE_COLUMN].iloc[-1]), observed_history_length=len(prefix),
            sequence_window=self._window, padding_cycles=max(0, self._window - len(prefix)) if self._window else 0,
            chart_prefix=recovered.copy(deep=True), sensor_mask=mask.copy(deep=True),
        )


class LSTMInferenceAdapter(_PrefixInferenceAdapter):
    """Load only through the existing checkpoint compatibility checks."""
    _label = "LSTM"

    def __init__(self, bundle, artifact_dir: str | Path = "artifacts"):
        self._model = load_lstm_checkpoint(bundle, artifact_dir)
        self._window = bundle.lstm_window
        self._scaler = copy.deepcopy(bundle.sequence_scaler)
        self._freeze_recovery(bundle)

    def _model_output(self, recovered):
        sensors = list(self._sensors)
        scaled = scale_sequence_frame(recovered, sensors, self._scaler)
        window = build_prefix_window(scaled, sensors, self._window)
        return predict_lstm(self._model, np.expand_dims(window, axis=0))


def _read_schema(root: Path) -> dict:
    path = root / "preprocessing/schema.json"
    if not path.is_file():
        raise ValueError(f"Required preprocessing schema is missing: {path}; regenerate preparation")
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read preprocessing schema: {path}") from exc
    if not isinstance(schema, dict):
        raise ValueError("Preprocessing schema must be a dictionary")
    return schema


class _TabularInferenceAdapter(_PrefixInferenceAdapter):
    def _freeze_tabular(self, bundle, root):
        self._freeze_recovery(bundle)
        self._rolling_window = _positive_integer(bundle.rolling_window, "rolling_window")
        self._features = tuple(bundle.feature_columns)
        # Ask the shared feature builder for its schema only; this is never fitted or saved.
        probe = pd.DataFrame({ID_COLUMN: [1], CYCLE_COLUMN: [1], **{s: [0.0] for s in self._sensors}})
        _, expected = build_tabular_features(probe, list(self._sensors), self._rolling_window)
        if list(self._features) != expected:
            raise ValueError("Frozen tabular feature order does not match shared feature generation")
        schema = _read_schema(Path(root))
        if schema.get("selected_sensors") != list(self._sensors) or schema.get("feature_columns") != list(self._features):
            raise ValueError("Preprocessing schema and bundle sensor/feature order mismatch")
        self._scaler = copy.deepcopy(bundle.tabular_scaler)
        if not isinstance(self._scaler, StandardScaler):
            raise ValueError("Tabular scaler must be a fitted StandardScaler")
        try:
            check_is_fitted(self._scaler)
        except ValueError as exc:
            raise ValueError("Tabular scaler must be fitted") from exc
        self._check_feature_schema(self._scaler.n_features_in_, getattr(self._scaler, "feature_names_in_", None), "Tabular scaler")
        for name in ("mean_", "var_", "scale_"):
            values = np.asarray(getattr(self._scaler, name, None), dtype=float)
            if values.shape != (len(self._features),) or not np.isfinite(values).all():
                raise ValueError(f"Tabular scaler fitted {name} statistics must be finite")
        if (self._scaler.scale_ <= 0).any() or (self._scaler.var_ < 0).any():
            raise ValueError("Tabular scaler fitted statistics are invalid")
        if not np.isfinite(self._scaler.n_samples_seen_).all() or (np.asarray(self._scaler.n_samples_seen_) <= 0).any():
            raise ValueError("Tabular scaler must have fitted training observations")

    def _check_feature_schema(self, count, names, label):
        if count != len(self._features):
            raise ValueError(f"{label} input feature count mismatch")
        if names is not None and list(names) != list(self._features):
            raise ValueError(f"{label} input feature order mismatch")

    def _model_output(self, recovered):
        features, columns = build_tabular_features(recovered, list(self._sensors), self._rolling_window)
        if columns != list(self._features):
            raise ValueError("Tabular feature generation changed after loading")
        matrix = self._scaler.transform(features.iloc[[-1]][list(self._features)])
        if not np.isfinite(matrix).all():
            raise ValueError("Scaled tabular features must be finite")
        if self._named_model:
            matrix = pd.DataFrame(matrix, columns=self._features)
        return self._model.predict(matrix)


class RFInferenceAdapter(_TabularInferenceAdapter):
    _label = "Random Forest"

    def __init__(self, bundle, artifact_dir: str | Path = "artifacts"):
        self._freeze_tabular(bundle, artifact_dir)
        self._model = joblib.load(Path(artifact_dir) / "models/random_forest.joblib")
        if not isinstance(self._model, RandomForestRegressor):
            raise ValueError("Random Forest artifact must be a fitted RandomForestRegressor")
        try:
            check_is_fitted(self._model)
        except ValueError as exc:
            raise ValueError("Random Forest model must be fitted") from exc
        names = getattr(self._model, "feature_names_in_", None)
        self._check_feature_schema(self._model.n_features_in_, names, self._label)
        self._named_model = names is not None


class XGBoostInferenceAdapter(_TabularInferenceAdapter):
    _label = "XGBoost"

    def __init__(self, bundle, artifact_dir: str | Path = "artifacts"):
        self._freeze_tabular(bundle, artifact_dir)
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ValueError("XGBoost inference requires xgboost>=2.0,<4 in the project .venv") from exc
        path = Path(artifact_dir) / "models/xgboost.json"
        if not path.is_file():
            raise ValueError(f"Required XGBoost model is missing: {path}")
        self._model = XGBRegressor()
        try:
            self._model.load_model(path)
            booster = self._model.get_booster()
        except Exception as exc:
            raise ValueError(f"Cannot load fitted XGBoost regressor: {path}") from exc
        self._check_feature_schema(booster.num_features(), booster.feature_names, self._label)
        self._named_model = booster.feature_names is not None


class MedianBaselineInferenceAdapter(_PrefixInferenceAdapter):
    _label = "Median baseline"

    def __init__(self, bundle, artifact_dir: str | Path = "artifacts"):
        self._freeze_recovery(bundle)
        self._model = joblib.load(Path(artifact_dir) / "models/baseline.joblib")
        if not isinstance(self._model, MedianRULBaseline) or self._model.value_ is None:
            raise ValueError("Median baseline artifact must be a fitted MedianRULBaseline")
        _finite_rul([self._model.value_], self._label)

    def _model_output(self, recovered):
        return self._model.predict(1)


ADAPTERS = {"lstm": LSTMInferenceAdapter, "random_forest": RFInferenceAdapter,
            "xgboost": XGBoostInferenceAdapter, "median_baseline": MedianBaselineInferenceAdapter}


def load_inference_adapter(bundle, artifact_dir, model_name="lstm"):
    if model_name not in ADAPTERS:
        raise ValueError(f"Unknown saved inference model: {model_name}")
    return ADAPTERS[model_name](bundle, artifact_dir)
