"""LSTM checkpoint provenance and compatibility for the existing artifact paths."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
from numbers import Integral

import numpy as np
import torch
from torch import nn

from ..artifacts import dump_joblib, dump_json, save_torch_state
from ..constants import CYCLE_COLUMN, ID_COLUMN
from ..data import file_manifest, validation_prefixes
from ..sequence_metadata import validate_sequence_input_metadata
from .lstm import LSTMConfig, RULLSTM

VALIDATION_PREFIX_FRACTION = 0.70

SELECTION = {"source": "clean_validation_prefix", "primary_metric": "rmse", "tie_breaker": "mae"}


def _read_metadata(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Required LSTM/input metadata is missing: {path}; legacy compatibility is unverified")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read LSTM/input metadata: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"LSTM/input metadata must be a dictionary: {path}")
    return metadata


def read_lstm_input_metadata(bundle, artifact_dir: str | Path) -> dict:
    """Validate Task 9's contract against the supplied bundle and split manifest."""
    prep = Path(artifact_dir) / "preprocessing"
    metadata = _read_metadata(prep / "sequence_input_metadata.json")
    split = _read_metadata(prep / "split_manifest.json")
    if (split.get("train_engines") != sorted(bundle.train_engine_ids)
            or split.get("validation_engines") != sorted(bundle.validation_engine_ids)):
        raise ValueError("LSTM input bundle and split manifest engine identities mismatch")
    declared_split = metadata.get("split")
    if not isinstance(declared_split, dict):
        raise ValueError("LSTM input metadata requires a split identity")
    validate_sequence_input_metadata(
        metadata, bundle.sequence_scaler, bundle.selected_sensors, bundle.lstm_window,
        {**split, "validation_prefix_fraction": VALIDATION_PREFIX_FRACTION},
    )
    prefix_cycles = getattr(bundle, "validation_prefix_cycles", None)
    if not isinstance(prefix_cycles, dict) or set(prefix_cycles) != set(bundle.validation_engine_ids):
        raise ValueError("LSTM validation prefix keys must exactly match validation engines")
    # Offline provenance only: inspect engine/cycle metadata, never hidden future sensors.
    trajectories = bundle.validation_df[[ID_COLUMN, CYCLE_COLUMN]].sort_values([ID_COLUMN, CYCLE_COLUMN])
    expected_prefixes = validation_prefixes(
        trajectories, bundle.validation_engine_ids, prefix_fraction=VALIDATION_PREFIX_FRACTION,
    )
    for engine_id, endpoint in expected_prefixes.items():
        supplied = prefix_cycles[engine_id]
        if isinstance(supplied, (bool, np.bool_)) or not isinstance(supplied, Integral) or supplied != endpoint:
            raise ValueError(f"LSTM validation prefix endpoint mismatch for engine {engine_id}: expected {endpoint}")
    if bundle.lstm_window != 30:
        raise ValueError("Scoped LSTM sequence window must be 30")
    return metadata


def _architecture(cfg: LSTMConfig) -> dict:
    if type(cfg.input_size) is not int or cfg.input_size <= 0 or cfg.hidden_units != 32 or cfg.dense_units != 16:
        raise ValueError("Scoped LSTM architecture requires positive input_size, hidden_units=32, dense_units=16")
    for name in ("batch_size", "max_epochs", "patience", "seed"):
        value = getattr(cfg, name)
        minimum = 0 if name == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid LSTM checkpoint configuration: {name}")
    if (isinstance(cfg.learning_rate, bool) or not isinstance(cfg.learning_rate, (int, float))
            or not np.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0):
        raise ValueError("Invalid LSTM checkpoint configuration: learning_rate")
    return {
        "type": "RULLSTM", "input_size": cfg.input_size, "hidden_units": 32,
        "dense_units": 16, "output_units": 1, "dense_activation": "ReLU",
        "output_activation": "linear", "num_layers": 1, "bidirectional": False,
        "sequence_window": 30,
    }


def _validate_selection(metadata: dict) -> None:
    record = metadata.get("selected_validation")
    epoch = metadata.get("selected_epoch")
    if (type(epoch) is not int or epoch <= 0 or not isinstance(record, dict)
            or type(record.get("epoch")) is not int or record["epoch"] != epoch):
        raise ValueError("LSTM selected epoch and validation record mismatch")
    for metric in ("rmse", "mae"):
        value = record.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"LSTM selected validation {metric} must be finite and non-negative")


def save_lstm_checkpoint(model, bundle, artifact_dir, cfg, training_meta, input_metadata) -> None:
    """Save restored weights and their selected validation/input provenance."""
    current_input = read_lstm_input_metadata(bundle, artifact_dir)
    if current_input != input_metadata:
        raise ValueError("LSTM preprocessing metadata changed during training")
    architecture = _architecture(cfg)
    if cfg.input_size != len(bundle.selected_sensors):
        raise ValueError("LSTM configuration input size does not match sensor order")
    if (not isinstance(model, RULLSTM) or model.lstm.input_size != cfg.input_size
            or model.lstm.hidden_size != 32 or model.lstm.num_layers != 1 or model.lstm.bidirectional
            or len(model.head) != 3 or not isinstance(model.head[1], nn.ReLU)
            or model.head[0].in_features != 32 or model.head[0].out_features != 16
            or model.head[2].in_features != 16 or model.head[2].out_features != 1):
        raise ValueError("LSTM model does not match the fixed checkpoint architecture")
    _validate_selection(training_meta)
    if (training_meta["selected_epoch"] > cfg.max_epochs
            or any(training_meta.get(key) != value for key, value in asdict(cfg).items())):
        raise ValueError("LSTM selected checkpoint and training configuration mismatch")
    metadata = {
        "schema_version": 1, "architecture": architecture, "config": asdict(cfg),
        "selection": SELECTION.copy(), "selected_epoch": training_meta["selected_epoch"],
        "selected_validation": copy.deepcopy(training_meta["selected_validation"]),
        "sequence_input_metadata": copy.deepcopy(input_metadata),
    }
    model_dir = Path(artifact_dir) / "models"
    checkpoint = model_dir / "lstm.pt"
    save_torch_state(model, checkpoint)
    metadata["checkpoint_sha256"] = file_manifest(checkpoint).sha256
    dump_json(metadata, model_dir / "lstm_checkpoint_metadata.json")
    # Preserve the existing diagnostic/config artifact for existing consumers.
    dump_joblib({"config": asdict(cfg), "meta": training_meta}, model_dir / "lstm_meta.joblib")


def load_lstm_checkpoint(bundle, artifact_dir: str | Path) -> RULLSTM:
    """Load only a checkpoint with verified declared input/architecture compatibility."""
    model_dir = Path(artifact_dir) / "models"
    metadata = _read_metadata(model_dir / "lstm_checkpoint_metadata.json")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("Unsupported LSTM checkpoint metadata schema version")
    current_input = read_lstm_input_metadata(bundle, artifact_dir)
    if metadata.get("sequence_input_metadata") != current_input:
        raise ValueError("LSTM checkpoint preprocessing metadata mismatch (sensor order, window or provenance)")
    if not isinstance(metadata.get("config"), dict) or set(metadata["config"]) != set(LSTMConfig.__dataclass_fields__):
        raise ValueError("Invalid LSTM checkpoint configuration metadata")
    try:
        cfg = LSTMConfig(**metadata["config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid LSTM checkpoint configuration metadata") from exc
    if metadata.get("architecture") != _architecture(cfg) or cfg.input_size != len(bundle.selected_sensors):
        raise ValueError("LSTM checkpoint architecture metadata mismatch")
    if metadata.get("selection") != SELECTION:
        raise ValueError("LSTM checkpoint selection must use clean validation-prefix RMSE and MAE")
    _validate_selection(metadata)
    if metadata["selected_epoch"] > cfg.max_epochs:
        raise ValueError("LSTM selected epoch exceeds configured maximum epochs")
    checkpoint = model_dir / "lstm.pt"
    if not checkpoint.is_file() or metadata.get("checkpoint_sha256") != file_manifest(checkpoint).sha256:
        raise ValueError("LSTM checkpoint file does not match its metadata checksum")
    model = RULLSTM(cfg.input_size, cfg.hidden_units, cfg.dense_units)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("LSTM checkpoint weights are incompatible with the fixed architecture") from exc
    model.eval()
    return model
