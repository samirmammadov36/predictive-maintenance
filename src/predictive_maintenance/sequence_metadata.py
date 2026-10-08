"""Sequence input provenance and compatibility checks; no model-weight validation."""
from __future__ import annotations

from numbers import Integral, Real
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import ID_COLUMN
from .sequences import validate_sequence_scaler

SCHEMA_VERSION = 1
PADDING_RULE = "repeat_first_imputed_observation"


def _engine_ids(values, name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must contain actual engine IDs")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integer engine IDs")
    ids = sorted(int(value) for value in values)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate engine IDs")
    return ids


def _split_identity(split: dict) -> dict:
    if not isinstance(split, dict):
        raise ValueError("Split information must be a dictionary")
    seed = split.get("seed")
    fraction = split.get("validation_prefix_fraction")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Split seed must be a non-negative integer")
    if isinstance(fraction, bool) or not isinstance(fraction, Real) or not 0 < fraction < 1:
        raise ValueError("Validation prefix fraction must be between zero and one")
    train = _engine_ids(split.get("train_engines"), "Training split")
    validation = _engine_ids(split.get("validation_engines"), "Validation split")
    if set(train) & set(validation):
        raise ValueError("Training and validation split engines overlap")
    return {"seed": int(seed), "train_engines": train, "validation_engines": validation,
            "validation_prefix_fraction": float(fraction)}


def _scaler_description(scaler: StandardScaler) -> dict:
    names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None
    return {"type": f"{type(scaler).__module__}.{type(scaler).__name__}",
            "feature_names": names, "feature_count": int(scaler.n_features_in_)}


def validate_sequence_input_metadata(
    metadata: dict,
    scaler: StandardScaler,
    sensor_columns: list[str],
    window: int,
    expected_split: dict,
) -> None:
    """Check input compatibility and declared provenance, not model weights or scaler means."""
    validate_sequence_scaler(scaler, sensor_columns)
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported sequence metadata schema version")
    if metadata.get("dataset") != "FD001" or metadata.get("padding_rule") != PADDING_RULE:
        raise ValueError("Sequence metadata dataset or padding rule mismatch")
    if metadata.get("sensor_columns") != list(sensor_columns):
        raise ValueError("Sequence metadata sensor order mismatch")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("Sequence window must be a positive integer")
    if type(metadata.get("sequence_window")) is not int or metadata["sequence_window"] != window:
        raise ValueError("Sequence metadata window mismatch")
    if _split_identity(metadata.get("split")) != _split_identity(expected_split):
        raise ValueError("Sequence metadata split identity mismatch")
    if metadata.get("scaler") != _scaler_description(scaler):
        raise ValueError("Sequence metadata scaler schema mismatch")
    rows = metadata.get("training_row_count")
    if type(rows) is not int or rows <= 0 or np.ndim(scaler.n_samples_seen_) != 0 or scaler.n_samples_seen_ != rows:
        raise ValueError("Sequence metadata training row count does not match scaler fit")
    cap = metadata.get("fitting_rul_cap")
    if type(cap) is not int or cap <= 0:
        raise ValueError("Sequence metadata fitting RUL cap must be a positive integer")
    source = metadata.get("training_source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("Sequence metadata requires the training source path")
    checksum = source.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None:
        raise ValueError("Sequence metadata requires the training source SHA-256 checksum")


def build_sequence_input_metadata(
    train_df: pd.DataFrame,
    sensor_columns: list[str],
    scaler: StandardScaler,
    *,
    dataset: str,
    window: int,
    split_info: dict,
    rul_cap: int,
    training_source_manifest: dict,
) -> dict:
    """Build JSON-ready metadata from the actual preparation inputs."""
    split = _split_identity(split_info)
    if ID_COLUMN not in train_df or train_df[ID_COLUMN].isna().any():
        raise ValueError("Training data requires non-missing engine IDs")
    actual_ids = _engine_ids(train_df[ID_COLUMN].unique().tolist(), "Training data")
    if actual_ids != split["train_engines"]:
        raise ValueError("Training data engine IDs do not match the declared training split")
    if not isinstance(training_source_manifest, dict):
        raise ValueError("Training source manifest must be a dictionary")
    validate_sequence_scaler(scaler, sensor_columns)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "sensor_columns": list(sensor_columns),
        "sequence_window": window,
        "padding_rule": PADDING_RULE,
        "split": split,
        "fitting_rul_cap": rul_cap,
        "training_row_count": len(train_df),
        "scaler": _scaler_description(scaler),
        "training_source": {key: training_source_manifest.get(key) for key in ("path", "sha256")},
    }
    validate_sequence_input_metadata(metadata, scaler, sensor_columns, window, split)
    return metadata
