"""Synthetic checkpoint tests; all generated artifacts stay in pytest temporary directories."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.data import add_training_rul, file_manifest
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.models import lstm as lstm_module
from predictive_maintenance.models.lstm import LSTMConfig, RULLSTM, predict_lstm, train_lstm
from predictive_maintenance.models.lstm_checkpoint import (
    load_lstm_checkpoint, read_lstm_input_metadata, save_lstm_checkpoint,
)
from predictive_maintenance.prefixes import make_validation_sequence_matrix
from predictive_maintenance.sequence_metadata import build_sequence_input_metadata
from predictive_maintenance.sequences import fit_sequence_scaler


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path):
    train = pd.DataFrame({
        "engine_id": [2, 2, 2, 9, 9, 9], "cycle": [1, 2, 3] * 2,
        "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "sensor_2": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    })
    train = add_training_rul(train)
    validation = add_training_rul(pd.DataFrame({
        "engine_id": [5] * 4, "cycle": [1, 2, 3, 4],
        "sensor_1": [7.0, 8.0, 9.0, 10.0], "sensor_2": [22.0, 24.0, 26.0, 28.0],
    }))
    sensors = ["sensor_2", "sensor_1"]
    scaler = fit_sequence_scaler(train, sensors)
    engineered, feature_columns = build_tabular_features(train, sensors, 10)
    bundle = SimpleNamespace(
        train_df=train, validation_df=validation, selected_sensors=sensors,
        sequence_scaler=scaler, lstm_window=30, train_engine_ids=[2, 9],
        validation_engine_ids=[5], validation_prefix_cycles={5: 2}, rolling_window=10,
        feature_columns=feature_columns, tabular_scaler=StandardScaler().fit(engineered[feature_columns]),
        # These are deliberately unusable for evaluation; training must never read them.
        official_test_df=object(), official_test_targets=object(),
    )
    prep = tmp_path / "artifacts/preprocessing"
    prep.mkdir(parents=True)
    source = tmp_path / "synthetic_training_source.txt"
    source.write_text("Synthetic unit-test data; not real FD001 observations.", encoding="utf-8")
    split = {"seed": 42, "train_engines": [2, 9], "validation_engines": [5],
             "validation_prefix_fraction": 0.70}
    metadata = build_sequence_input_metadata(
        train, sensors, scaler, dataset="FD001", window=30, split_info=split,
        rul_cap=125, training_source_manifest=asdict(file_manifest(source)),
    )
    (prep / "sequence_input_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (prep / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    return bundle, tmp_path / "artifacts", metadata



@pytest.mark.parametrize("length,endpoint", [(1, 1), (4, 2), (5, 3), (10, 7)])
def test_input_metadata_checks_each_chronological_validation_prefix(prepared, length, endpoint):
    bundle, root, metadata = prepared
    second = bundle.validation_df.iloc[[0] * length].copy()
    second["engine_id"] = 6
    second["cycle"] = np.arange(1, length + 1)
    bundle.validation_df = pd.concat([bundle.validation_df, second], ignore_index=True).sample(frac=1, random_state=17)
    bundle.validation_engine_ids = [5, 6]
    bundle.validation_prefix_cycles = {5: 2, 6: endpoint}
    metadata["split"]["validation_engines"] = [5, 6]
    prep = root / "preprocessing"
    (prep / "sequence_input_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    split_path = prep / "split_manifest.json"
    split = json.loads(split_path.read_text())
    split["validation_engines"] = [5, 6]
    split_path.write_text(json.dumps(split), encoding="utf-8")
    before = bundle.validation_df.copy(deep=True)
    assert read_lstm_input_metadata(bundle, root) == metadata
    assert bundle.validation_prefix_cycles == {5: 2, 6: endpoint}
    pd.testing.assert_frame_equal(bundle.validation_df, before)
    # Changing the second engine must fail even when the first engine remains valid.
    bundle.validation_prefix_cycles[6] = endpoint + 1
    with pytest.raises(ValueError, match="prefix endpoint mismatch for engine 6"):
        read_lstm_input_metadata(bundle, root)


@pytest.mark.parametrize("fraction", [0.65, 0.80])
def test_input_metadata_rejects_altered_declared_prefix_fraction(prepared, fraction):
    bundle, root, metadata = prepared
    metadata["split"]["validation_prefix_fraction"] = fraction
    (root / "preprocessing/sequence_input_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="split identity mismatch"):
        read_lstm_input_metadata(bundle, root)


@pytest.mark.parametrize("endpoint", [1, 3])
def test_input_metadata_rejects_altered_actual_prefix_endpoint(prepared, endpoint):
    bundle, root, _ = prepared
    bundle.validation_prefix_cycles[5] = endpoint
    with pytest.raises(ValueError, match="prefix endpoint mismatch for engine 5: expected 2"):
        read_lstm_input_metadata(bundle, root)


@pytest.mark.parametrize("prefixes", [{}, {5: 2, 6: 2}])
def test_input_metadata_rejects_missing_or_extra_prefix_engine_keys(prepared, prefixes):
    bundle, root, _ = prepared
    bundle.validation_prefix_cycles = prefixes
    with pytest.raises(ValueError, match="prefix keys must exactly match validation engines"):
        read_lstm_input_metadata(bundle, root)


def test_offline_prefix_check_does_not_expose_hidden_future_sensors(prepared):
    bundle, root, metadata = prepared
    before_X, before_y, before_engines = make_validation_sequence_matrix(bundle)
    bundle.validation_df.loc[bundle.validation_df.cycle > 2, bundle.selected_sensors] = np.nan
    assert read_lstm_input_metadata(bundle, root) == metadata
    after_X, after_y, after_engines = make_validation_sequence_matrix(bundle)
    np.testing.assert_array_equal(before_X, after_X)
    np.testing.assert_array_equal(before_y, after_y)
    assert before_engines == after_engines == [(5, 2)]


def _controlled_validation(monkeypatch, metrics):
    captured = []
    real_predict = lstm_module._predict
    iterator = iter(metrics)

    def predict(model, X, device):
        captured.append(copy.deepcopy(model.state_dict()))
        return real_predict(model, X, device)

    monkeypatch.setattr(lstm_module, "_predict", predict)
    monkeypatch.setattr(lstm_module, "regression_metrics", lambda y, pred: dict(zip(("rmse", "mae"), next(iterator))))
    return captured


@pytest.mark.parametrize("metrics,epoch,length", [
    ([(10, 8), (8, 7), (9, 6), (9, 5)], 2, 4),
    ([(10, 8), (10, 6), (10, 7), (10, 7)], 2, 4),
    ([(10, 8)] * 3, 1, 3),
    ([(10, 8), (11, 7), (10, 6), (10, 6), (10, 6)], 3, 5),
    ([(10, 8), (10 - 5e-9, 9), (11, 8), (11, 8)], 2, 4),
])
def test_selected_epoch_restored_weights_and_patience_agree(monkeypatch, metrics, epoch, length):
    captured = _controlled_validation(monkeypatch, metrics)
    X = np.arange(120, dtype=np.float32).reshape(2, 30, 2) / 100
    y = np.array([2.0, 4.0], dtype=np.float32)
    model, history, meta = train_lstm(X, y, X, y, LSTMConfig(2, max_epochs=8, patience=2))
    assert len(history) == length
    assert meta["selected_epoch"] == epoch
    assert meta["selected_validation"] == history[epoch - 1]
    assert meta["selected_validation"] == min(history, key=lambda row: (row["rmse"], row["mae"]))
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, captured[epoch - 1][name], rtol=0, atol=0)
    assert any(not torch.equal(captured[-1][name], weight) for name, weight in captured[epoch - 1].items())


@pytest.mark.parametrize("array_input", [True, False])
def test_defaults_architecture_and_real_synthetic_validation_metrics(array_input):
    cfg = LSTMConfig(2)
    assert (cfg.hidden_units, cfg.dense_units, cfg.batch_size, cfg.learning_rate, cfg.max_epochs, cfg.patience) == (32, 16, 64, 0.001, 50, 5)
    X = np.ones((2, 30, 2), dtype=np.float32)
    y = np.array([2.0, 4.0], dtype=np.float32)
    supplied = X if array_input else X.tolist()
    model, history, meta = train_lstm(supplied, y, supplied, y, LSTMConfig(2, max_epochs=1))
    pred = np.maximum(predict_lstm(model, X), 0)
    assert model.lstm.hidden_size == 32 and model.head[0].out_features == 16
    assert model.head[-1].out_features == 1 and isinstance(model.head[-1], torch.nn.Linear)
    assert model(torch.as_tensor(X)).shape == (2,)
    assert meta["selected_epoch"] == 1
    assert history[0]["mae"] == pytest.approx(np.mean(np.abs(y - pred)))
    assert history[0]["rmse"] == pytest.approx(np.mean((y - pred) ** 2) ** 0.5)


def _save_fixture(prepared):
    bundle, root, metadata = prepared
    cfg = LSTMConfig(2)
    torch.manual_seed(42)
    model = RULLSTM(2)
    record = {"epoch": 1, "rmse": 8.0, "mae": 6.0, "train_mse": 1.0}
    training_meta = {**asdict(cfg), "selected_epoch": 1, "selected_validation": record,
                     "device": "cpu", "train_seconds": 0.0}
    save_lstm_checkpoint(model, bundle, root, cfg, training_meta, metadata)
    return model, training_meta


def test_save_load_prediction_parity_and_provenance(prepared):
    model, training_meta = _save_fixture(prepared)
    bundle, root, input_metadata = prepared
    before = copy.deepcopy(bundle.sequence_scaler.__dict__)
    loaded = load_lstm_checkpoint(bundle, root)
    X = np.random.default_rng(17).normal(size=(3, 30, 2)).astype(np.float32)
    np.testing.assert_array_equal(predict_lstm(model, X), predict_lstm(loaded, X))
    metadata = json.loads((root / "models/lstm_checkpoint_metadata.json").read_text())
    assert metadata["sequence_input_metadata"] == input_metadata
    assert metadata["sequence_input_metadata"]["sensor_columns"] == ["sensor_2", "sensor_1"]
    assert metadata["selected_epoch"] == 1
    assert metadata["selected_validation"] == training_meta["selected_validation"]
    assert metadata["checkpoint_sha256"] == file_manifest(root / "models/lstm.pt").sha256
    legacy = joblib.load(root / "models/lstm_meta.joblib")
    assert legacy["config"] == asdict(LSTMConfig(2)) and legacy["meta"] == training_meta
    for name in ["mean_", "var_", "scale_", "feature_names_in_"]:
        np.testing.assert_array_equal(bundle.sequence_scaler.__dict__[name], before[name])
    assert not loaded.training


@pytest.mark.parametrize("problem,message", [
    ("sensor_order_checkpoint", "preprocessing"), ("sensor_order_bundle", "Sensor order"),
    ("window_checkpoint", "preprocessing"), ("window_bundle", "window"),
    ("architecture_hidden", "architecture"), ("architecture_output", "architecture"),
    ("architecture_activation", "architecture"), ("configuration_hidden", "architecture"),
    ("split_snapshot", "preprocessing"), ("bundle_split", "engine identities"),
    ("input_split_manifest", "split identity"), ("checksum", "checksum"),
    ("selection_source", "clean validation-prefix"), ("selected_epoch", "selected epoch"),
    ("schema", "schema version"), ("missing_config", "configuration"),
    ("missing_checkpoint_metadata", "legacy compatibility is unverified"),
    ("missing_input_metadata", "metadata is missing"), ("malformed_input_split", "split identity"),
    ("weights_shape", "weights are incompatible"),
])
def test_incompatible_checkpoint_or_inputs_fail_clearly(prepared, problem, message):
    _save_fixture(prepared)
    bundle, root, _ = prepared
    path = root / "models/lstm_checkpoint_metadata.json"
    metadata = json.loads(path.read_text())
    if problem == "sensor_order_checkpoint":
        metadata["sequence_input_metadata"]["sensor_columns"].reverse()
    elif problem == "sensor_order_bundle":
        bundle.selected_sensors.reverse()
    elif problem == "window_checkpoint":
        metadata["sequence_input_metadata"]["sequence_window"] = 29
    elif problem == "window_bundle":
        bundle.lstm_window = 29
    elif problem.startswith("architecture_"):
        field = {"architecture_hidden": "hidden_units", "architecture_output": "output_units",
                 "architecture_activation": "output_activation"}[problem]
        metadata["architecture"][field] = "sigmoid" if field == "output_activation" else 64
    elif problem == "configuration_hidden":
        metadata["config"]["hidden_units"] = 64
    elif problem == "split_snapshot":
        metadata["sequence_input_metadata"]["split"]["seed"] = 17
    elif problem == "bundle_split":
        bundle.train_engine_ids = [2, 8]
    elif problem == "input_split_manifest":
        split_path = root / "preprocessing/split_manifest.json"
        split = json.loads(split_path.read_text())
        split["seed"] = 17
        split_path.write_text(json.dumps(split))
    elif problem == "checksum":
        with (root / "models/lstm.pt").open("ab") as stream:
            stream.write(b"different checkpoint")
    elif problem == "selection_source":
        metadata["selection"]["source"] = "official_test"
    elif problem == "selected_epoch":
        metadata["selected_epoch"] = 2
    elif problem == "schema":
        metadata["schema_version"] = 99
    elif problem == "missing_config":
        metadata["config"].pop("dense_units")
    elif problem == "missing_checkpoint_metadata":
        path.unlink()
    elif problem == "missing_input_metadata":
        (root / "preprocessing/sequence_input_metadata.json").unlink()
    elif problem == "malformed_input_split":
        input_path = root / "preprocessing/sequence_input_metadata.json"
        current = json.loads(input_path.read_text())
        current["split"] = None
        input_path.write_text(json.dumps(current))
    else:
        torch.save({}, root / "models/lstm.pt")
        metadata["checkpoint_sha256"] = file_manifest(root / "models/lstm.pt").sha256
    if problem != "missing_checkpoint_metadata":
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_lstm_checkpoint(bundle, root)


def test_saving_rejects_input_contract_changed_during_training(prepared):
    bundle, root, metadata = prepared
    changed = copy.deepcopy(metadata)
    changed["training_source"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="changed during training"):
        save_lstm_checkpoint(RULLSTM(2), bundle, root, LSTMConfig(2), {}, changed)
    assert not (root / "models/lstm.pt").exists()


class _OtherModelStub:
    def load_model(self, path):
        self.loaded_path = path

    def save_model(self, path):
        Path(path).write_text("Synthetic non-LSTM test stub", encoding="utf-8")


def test_existing_saved_model_path_uses_validated_lstm_loader(prepared, monkeypatch):
    model, _ = _save_fixture(prepared)
    bundle, root, _ = prepared
    # Isolate optional XGBoost/RF dependencies: this test verifies only the LSTM load path.
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=_OtherModelStub))
    path = ROOT / "src/predictive_maintenance/experiments.py"
    spec = importlib.util.spec_from_file_location("predictive_maintenance._test_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_joblib", lambda path: object())
    models = module._load_models(bundle, root)
    X = np.ones((1, 30, 2), dtype=np.float32)
    np.testing.assert_array_equal(predict_lstm(model, X), predict_lstm(models["lstm"], X))
    (root / "models/lstm_checkpoint_metadata.json").unlink()
    with pytest.raises(ValueError, match="legacy compatibility is unverified"):
        module._load_models(bundle, root)


def test_training_script_summary_saved_weights_and_epoch_agree(prepared, monkeypatch):
    bundle, root, _ = prepared
    cfg = {"project": {"seed": 42}, "lstm": {
        "hidden_units": 32, "dense_units": 16, "learning_rate": 0.001,
        "batch_size": 64, "max_epochs": 4, "patience": 2,
    }}
    # Only LSTM training/saving is real; config loading and other model training are isolated.
    monkeypatch.setitem(sys.modules, "predictive_maintenance.config", SimpleNamespace(load_config=lambda path: cfg))
    monkeypatch.setitem(sys.modules, "predictive_maintenance.models.xgboost_model", SimpleNamespace(
        train_xgboost=lambda *args: (_OtherModelStub(), [{"rmse": 5.0, "mae": 4.0}]),
    ))
    spec = importlib.util.spec_from_file_location("_test_train_models", ROOT / "scripts/train_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", root.parent)
    monkeypatch.setattr(module, "train_random_forest", lambda *args: (SimpleNamespace(), [{"rmse": 5.0, "mae": 4.0}]))
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    captured = _controlled_validation(monkeypatch, [(10, 8), (10, 6), (11, 5), (11, 5)])
    module.main()
    summary = json.loads((root.parent / "results/metrics/validation_selection_summary.json").read_text())
    metadata = json.loads((root / "models/lstm_checkpoint_metadata.json").read_text())
    diagnostic = joblib.load(root / "models/lstm_meta.joblib")
    history = pd.read_csv(root.parent / "results/metrics/lstm_training_history.csv")
    assert summary["lstm_best"] == metadata["selected_validation"] == diagnostic["meta"]["selected_validation"]
    assert metadata["selected_epoch"] == diagnostic["meta"]["selected_epoch"] == 2
    assert summary["lstm_best"]["epoch"] == 2
    assert (summary["lstm_best"]["rmse"], summary["lstm_best"]["mae"]) == (10, 6)
    assert history.epoch.tolist() == [1, 2, 3, 4]
    loaded = load_lstm_checkpoint(bundle, root)
    for name, weight in loaded.state_dict().items():
        torch.testing.assert_close(weight, captured[1][name], rtol=0, atol=0)
