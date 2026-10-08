from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.data import manifests_as_dict, split_development_engines
from predictive_maintenance.sequence_metadata import build_sequence_input_metadata, validate_sequence_input_metadata
from predictive_maintenance.sequences import fit_sequence_scaler, scale_sequence_frame, validate_sequence_scaler


@pytest.fixture
def prepared_input(tmp_path):
    train = pd.DataFrame({"engine_id": [2, 9, 2], "cycle": [1, 1, 2],
                          "sensor_1": [1.0, 3.0, 5.0], "sensor_2": [10.0, 14.0, 18.0]})
    columns = ["sensor_2", "sensor_1"]
    split = {"seed": 17, "train_engines": [2, 9], "validation_engines": [5],
             "validation_prefix_fraction": 0.65}
    source = tmp_path / "synthetic_source.txt"
    source.write_text("Synthetic unit-test fixture; not real FD001 data.\n", encoding="utf-8")
    manifest = manifests_as_dict([source])[0]
    scaler = fit_sequence_scaler(train, columns)
    metadata = build_sequence_input_metadata(
        train, columns, scaler, dataset="FD001", window=30, split_info=split,
        rul_cap=111, training_source_manifest=manifest,
    )
    return train, columns, split, manifest, scaler, metadata


def test_sequence_scaler_save_load_and_metadata_json_round_trip(prepared_input, tmp_path):
    train, columns, split, manifest, scaler, metadata = prepared_input
    scaler_path = tmp_path / "synthetic_scaler.joblib"
    metadata_path = tmp_path / "synthetic_sequence_metadata.json"
    joblib.dump(scaler, scaler_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded_scaler = joblib.load(scaler_path)
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert loaded_metadata == metadata
    validate_sequence_input_metadata(loaded_metadata, loaded_scaler, columns, 30, split)
    held_out = train.assign(sensor_1=1e9, sensor_2=-1e9, dataset_split="official_test")
    pd.testing.assert_frame_equal(scale_sequence_frame(held_out, columns, scaler),
                                  scale_sequence_frame(held_out, columns, loaded_scaler))
    assert loaded_metadata["dataset"] == "FD001"
    assert loaded_metadata["schema_version"] == 1
    assert loaded_metadata["sensor_columns"] == ["sensor_2", "sensor_1"]
    assert loaded_metadata["padding_rule"] == "repeat_first_imputed_observation"
    assert loaded_metadata["split"] == split
    assert loaded_metadata["fitting_rul_cap"] == 111
    assert loaded_metadata["training_row_count"] == 3
    assert loaded_metadata["scaler"]["feature_names"] == columns
    assert loaded_metadata["scaler"]["feature_count"] == 2
    assert loaded_metadata["training_source"] == {key: manifest[key] for key in ("path", "sha256")}
    assert loaded_metadata["training_source"]["sha256"] == hashlib.sha256(Path(manifest["path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("mismatch", ["sensors", "window", "seed", "train_ids", "validation_ids", "prefix_fraction"])
def test_sequence_metadata_rejects_incompatible_supplied_inputs(prepared_input, mismatch):
    _, columns, split, _, scaler, metadata = prepared_input
    columns = columns.copy()
    split = copy.deepcopy(split)
    window = 30
    if mismatch == "sensors":
        columns.reverse()
    elif mismatch == "window":
        window = 29
    elif mismatch == "seed":
        split["seed"] = 42
    elif mismatch == "train_ids":
        split["train_engines"] = [2, 8]
    elif mismatch == "validation_ids":
        split["validation_engines"] = [6]
    else:
        split["validation_prefix_fraction"] = 0.70
    with pytest.raises(ValueError, match="order|window|split identity"):
        validate_sequence_input_metadata(metadata, scaler, columns, window, split)


@pytest.mark.parametrize("field,value,message", [
    ("schema_version", 99, "schema version"),
    ("dataset", "FD002", "dataset or padding"),
    ("padding_rule", "zero", "dataset or padding"),
    ("sensor_columns", ["sensor_1", "sensor_2"], "sensor order"),
    ("training_row_count", 4, "training row count"),
    ("scaler", {"type": "StandardScaler", "feature_count": 99, "feature_names": []}, "scaler schema"),
    ("training_source", {"path": "synthetic", "sha256": "invalid"}, "SHA-256"),
])
def test_sequence_metadata_rejects_corrupted_fields(prepared_input, field, value, message):
    _, columns, split, _, scaler, metadata = prepared_input
    metadata = copy.deepcopy(metadata)
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        validate_sequence_input_metadata(metadata, scaler, columns, 30, split)


def test_sequence_metadata_checks_actual_80_engine_training_identity(tmp_path):
    development = pd.DataFrame({"engine_id": np.repeat(np.arange(1, 101), 2), "cycle": [1, 2] * 100})
    development["sensor_1"] = development.engine_id.astype(float)
    development["sensor_2"] = development.cycle.astype(float)
    train_ids, val_ids = split_development_engines(development, seed=42)
    assert len(train_ids) == 80 and len(val_ids) == 20
    # Extreme hidden engines are present before selection, but must never reach fitting.
    development.loc[development.engine_id.isin(val_ids), "sensor_1"] = 1e12
    train = development[development.engine_id.isin(train_ids)].copy()
    columns = ["sensor_1", "sensor_2"]
    scaler = fit_sequence_scaler(train, columns)
    np.testing.assert_allclose(scaler.mean_, [np.mean(train_ids), 1.5])
    np.testing.assert_allclose(scaler.var_, [np.var(train_ids), 0.25])
    source = tmp_path / "synthetic_development.txt"
    source.write_text("Synthetic 100-engine test fixture", encoding="utf-8")
    manifest = manifests_as_dict([source])[0]
    split = {"seed": 42, "train_engines": train_ids, "validation_engines": val_ids,
             "validation_prefix_fraction": 0.70}
    metadata = build_sequence_input_metadata(train, columns, scaler, dataset="FD001", window=30,
                                             split_info=split, rul_cap=125, training_source_manifest=manifest)
    assert metadata["split"]["train_engines"] == train_ids
    assert metadata["split"]["validation_engines"] == val_ids
    assert metadata["training_row_count"] == 160
    contaminated = pd.concat([train, development[development.engine_id == val_ids[0]]])
    with pytest.raises(ValueError, match="Training data engine IDs"):
        build_sequence_input_metadata(contaminated, columns, scaler, dataset="FD001", window=30,
                                      split_info=split, rul_cap=125, training_source_manifest=manifest)


def test_sequence_scaler_fallback_without_feature_names_and_wrong_count(prepared_input):
    train, columns, _, _, _, _ = prepared_input
    unnamed_scaler = StandardScaler().fit(train[columns].to_numpy())
    validate_sequence_scaler(unnamed_scaler, columns)
    one_sensor_scaler = fit_sequence_scaler(train, ["sensor_1"])
    with pytest.raises(ValueError, match="Sensor count"):
        validate_sequence_scaler(one_sensor_scaler, columns)
    with pytest.raises(ValueError, match="must be fitted"):
        validate_sequence_scaler(StandardScaler(), columns)


def test_sequence_scaling_rejects_empty_input(prepared_input):
    train, columns, _, _, scaler, _ = prepared_input
    with pytest.raises(ValueError, match="empty"):
        fit_sequence_scaler(train.iloc[:0], columns)
    with pytest.raises(ValueError, match="empty"):
        scale_sequence_frame(train.iloc[:0], columns, scaler)
