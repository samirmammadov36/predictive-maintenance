"""Actual saved LSTM inference on synthetic fixtures; artifacts remain in tmp_path."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance import inference
from predictive_maintenance.data import file_manifest
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.masks import nested_random_masks, short_gap_mask
from predictive_maintenance.models.lstm import LSTMConfig, RULLSTM, predict_lstm
from predictive_maintenance.models.lstm_checkpoint import save_lstm_checkpoint
from predictive_maintenance.sequence_metadata import build_sequence_input_metadata
from predictive_maintenance.sequences import fit_sequence_scaler

CONFIGURATIONS = [("clean", "median", None)] + [
    (scenario, recovery, seed) for scenario in ("random_10", "random_20")
    for seed in (7, 17, 27) for recovery in ("median", "forward_fill")
] + [("short_gap", recovery, None) for recovery in ("median", "forward_fill")]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def artifacts(tmp_path):
    def create(sensors=None, bias=50.0):
        sensors = ["sensor_2", "sensor_11"] if sensors is None else sensors
        train = pd.DataFrame({"engine_id": [2, 2, 9, 9], "cycle": [1, 2, 1, 2],
                              "sensor_11": [80.0, 100.0, 120.0, 180.0], "sensor_2": [10.0, 20.0, 30.0, 80.0]})
        validation = pd.DataFrame({"engine_id": [5] * 4, "cycle": [1, 2, 3, 4]})
        scaler = fit_sequence_scaler(train, sensors)
        bundle = SimpleNamespace(
            train_df=train, validation_df=validation, selected_sensors=sensors,
            sequence_scaler=scaler, medians=TrainingMedians.fit(train, sensors), lstm_window=30,
            train_engine_ids=[2, 9], validation_engine_ids=[5], validation_prefix_cycles={5: 2},
        )
        root = tmp_path / "artifacts"
        prep = root / "preprocessing"
        prep.mkdir(parents=True, exist_ok=True)
        source = tmp_path / "synthetic_source.txt"
        source.write_text("Synthetic test data only; not actual FD001 training.", encoding="utf-8")
        split = {"seed": 42, "train_engines": [2, 9], "validation_engines": [5], "validation_prefix_fraction": 0.70}
        metadata = build_sequence_input_metadata(
            train, sensors, scaler, dataset="FD001", window=30, split_info=split,
            rul_cap=125, training_source_manifest=asdict(file_manifest(source)),
        )
        (prep / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
        (prep / "sequence_input_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        cfg = LSTMConfig(len(sensors))
        torch.manual_seed(42)
        model = RULLSTM(len(sensors))
        with torch.no_grad():
            model.head[-1].bias.fill_(bias)
        # Synthetic selection record exercises serialization; no model training is performed here.
        training_meta = {**asdict(cfg), "selected_epoch": 1, "selected_validation": {"epoch": 1, "rmse": 8.0, "mae": 6.0}}
        save_lstm_checkpoint(model, bundle, root, cfg, training_meta, metadata)
        return bundle, root, model
    return create


@pytest.fixture
def saved(artifacts):
    bundle, root, model = artifacts()
    return bundle, root, model, inference.LSTMInferenceAdapter(bundle, root)


@pytest.fixture
def raw():
    rows = []
    for engine, length in [(1, 45), (2, 9)]:
        for cycle in range(1, length + 1):
            rows.append({"engine_id": engine, "cycle": cycle, "sensor_11": float(cycle * 10 + (900 if engine == 2 else 0)),
                         "setting_1": 3.0, "sensor_2": float(cycle + (200 if engine == 2 else 20)),
                         "rul_true": 999999.0, "rul_target": -999999.0})
    frame = pd.DataFrame(rows, index=100 + 3 * np.arange(len(rows)))
    frame.loc[(frame.engine_id == 1) & (frame.cycle == 1), ["sensor_11", "sensor_2"]] = np.nan
    frame.loc[(frame.engine_id == 1) & (frame.cycle == 3), "sensor_11"] = np.nan
    return frame.sample(frac=1, random_state=17)


def _capture_inputs(adapter):
    inputs = []
    adapter._model.register_forward_pre_hook(lambda model, args: inputs.append(args[0].detach().cpu().numpy().copy()))
    return inputs


def _assert_same(left, right):
    for field in ("rul_pred", "engine_id", "dataset_split", "cycle", "observed_history_length", "sequence_window", "padding_cycles"):
        assert getattr(left, field) == getattr(right, field)
    pd.testing.assert_frame_equal(left.chart_prefix, right.chart_prefix)
    pd.testing.assert_frame_equal(left.sensor_mask, right.sensor_mask)


@pytest.mark.parametrize("length", [1, 5, 29, 30, 31])
def test_actual_saved_model_history_padding_sensor_order_and_parity(saved, raw, length):
    bundle, _, original_model, adapter = saved
    inputs = _capture_inputs(adapter)
    result = adapter.predict(raw, 1, length)
    assert result.cycle == length and result.engine_id == 1 and result.dataset_split is None
    assert result.observed_history_length == length and result.sequence_window == 30
    assert result.padding_cycles == max(0, 30 - length)
    assert list(result.chart_prefix) == ["engine_id", "cycle", "sensor_2", "sensor_11"]
    measurements = np.array([[25.0 if cycle == 1 else cycle + 20.0,
                              110.0 if cycle in (1, 3) else cycle * 10.0] for cycle in range(1, length + 1)])
    np.testing.assert_array_equal(result.chart_prefix[["sensor_2", "sensor_11"]], measurements)
    scaled = ((measurements - [35.0, 120.0]) / np.sqrt([725.0, 1400.0])).astype(np.float32)
    expected = scaled[-30:]
    if length < 30:
        expected = np.concatenate([np.repeat(scaled[[0]], 30 - length, axis=0), expected])
    assert inputs[0].shape == (1, 30, 2) and inputs[0].dtype == np.float32
    np.testing.assert_allclose(inputs[0][0], expected, rtol=1e-6, atol=1e-7)
    assert np.isfinite(result.rul_pred) and result.rul_pred > 0
    assert result.rul_pred == max(float(predict_lstm(original_model, expected[None])[0]), 0)
    assert not result.sensor_mask.to_numpy().any()
    assert result.chart_prefix.cycle.tolist() == list(range(1, length + 1))


@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
@pytest.mark.parametrize("cutoff", [1, 5, 29, 30, 31])
def test_future_perturbation_actual_prediction_mask_chart_and_padding(saved, raw, scenario, recovery, seed, cutoff):
    _, _, _, adapter = saved
    inputs = _capture_inputs(adapter)
    options = {"scenario": scenario, "recovery": recovery, "seed": seed}
    before = adapter.predict(raw, 1, cutoff, **options)
    changed = raw.copy(deep=True)
    future = (changed.engine_id == 1) & (changed.cycle > cutoff)
    changed.loc[future, "sensor_11"] = np.inf
    changed.loc[future, "sensor_2"] = -999999.0
    changed.loc[:, ["rul_true", "rul_target"]] = np.nan
    # The future-only object value also changes the physical column dtype; visible charts stay numeric.
    appended = pd.DataFrame({"engine_id": [1], "cycle": [46], "sensor_11": ["hidden future sensor"],
                             "sensor_2": [np.nan], "setting_1": [99.0], "rul_true": [0.0], "rul_target": [0.0]}, index=[99999])
    changed = pd.concat([changed, appended])
    after = adapter.predict(changed, 1, cutoff, **options)
    _assert_same(before, after)
    np.testing.assert_array_equal(inputs[0], inputs[1])
    assert after.chart_prefix.cycle.max() <= cutoff and len(after.chart_prefix) == cutoff
    assert not {"rul_true", "rul_target", "setting_1"} & set(after.chart_prefix)
    assert not {"rul_true", "rul_target", "setting_1"} & set(after.sensor_mask)


@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
def test_identical_options_repeat_and_other_engine_cannot_affect_inference(saved, raw, scenario, recovery, seed):
    _, _, _, adapter = saved
    options = {"scenario": scenario, "recovery": recovery, "seed": seed}
    before = adapter.predict(raw, 1, 8, **options)
    _assert_same(before, adapter.predict(raw, 1, 8, **options))
    changed = raw.copy(deep=True)
    changed.loc[changed.engine_id == 2, ["sensor_11", "sensor_2"]] = np.inf
    _assert_same(before, adapter.predict(changed, 1, 8, **options))


@pytest.mark.parametrize("recovery", ["median", "forward_fill"])
def test_native_nans_raw_mask_then_recovery_with_frozen_sensor_fallback(saved, raw, recovery):
    _, _, _, adapter = saved
    result = adapter.predict(raw, 1, 8, scenario="short_gap", recovery=recovery)
    assert result.sensor_mask.sensor_11.tolist() == [False] * 3 + [True] * 5
    assert not result.sensor_mask.sensor_2.any()
    expected = [110.0, 20.0] + ([110.0] * 6 if recovery == "median" else [20.0] * 6)
    assert result.chart_prefix.sensor_11.tolist() == expected
    # Prefilling native cycle-3 NaN with median would incorrectly propagate 110 into this gap.
    assert result.chart_prefix.sensor_2.iloc[0] == 25.0
    assert np.isfinite(result.chart_prefix[["sensor_2", "sensor_11"]]).all().all()


@pytest.mark.parametrize("seed", [7, 17, 27])
def test_random_masks_match_existing_visible_prefix_and_remain_nested(saved, raw, seed):
    _, _, _, adapter = saved
    prefix = raw[(raw.engine_id == 1) & (raw.cycle <= 5)].sort_values("cycle")
    low, high = nested_random_masks(prefix, ["sensor_2", "sensor_11"], seed)
    result_low = adapter.predict(raw, 1, 5, scenario="random_10", seed=seed)
    result_high = adapter.predict(raw, 1, 5, scenario="random_20", seed=seed)
    pd.testing.assert_frame_equal(result_low.sensor_mask, low)
    pd.testing.assert_frame_equal(result_high.sensor_mask, high)
    assert result_low.sensor_mask.to_numpy().sum() == 1 and result_high.sensor_mask.to_numpy().sum() == 2
    assert not (result_low.sensor_mask & ~result_high.sensor_mask).to_numpy().any()
    forwarded = adapter.predict(raw, 1, 5, scenario="random_20", seed=seed, recovery="forward_fill")
    pd.testing.assert_frame_equal(forwarded.sensor_mask, result_high.sensor_mask)


@pytest.mark.parametrize("length", [1, 5, 29, 30, 31])
def test_short_gap_uses_first_frozen_sensor_when_sensor_11_absent(artifacts, raw, length):
    bundle, root, _ = artifacts(["sensor_2"])
    adapter = inference.LSTMInferenceAdapter(bundle, root)
    result = adapter.predict(raw, 1, length, scenario="short_gap", recovery="forward_fill")
    assert list(result.sensor_mask) == ["sensor_2"]
    assert result.sensor_mask.sensor_2.tolist() == [False] * max(0, length - 5) + [True] * min(5, length)
    assert list(result.chart_prefix) == ["engine_id", "cycle", "sensor_2"]
    assert result.padding_cycles == max(0, 30 - length)


def test_composite_identity_with_overlapping_row_indices_and_ambiguous_selection(saved, raw):
    _, _, _, adapter = saved
    one = raw[raw.engine_id == 1].copy()
    other = one.copy()
    one["dataset_split"] = "official_test"
    other["dataset_split"] = "development"
    other[["sensor_11", "sensor_2"]] = 900000.0
    combined = pd.concat([other, one])  # Row indices legitimately overlap across identities.
    with pytest.raises(ValueError, match="Ambiguous"):
        adapter.predict(combined, 1, 5)
    result = adapter.predict(combined, 1, 5, dataset_split="official_test")
    reference = adapter.predict(one, 1, 5)  # One unambiguous namespace can be inferred.
    _assert_same(result, reference)
    assert result.dataset_split == "official_test"
    assert list(result.chart_prefix) == ["dataset_split", "engine_id", "cycle", "sensor_2", "sensor_11"]
    other_result = adapter.predict(combined, 1, 5, dataset_split="development")
    assert other_result.chart_prefix.sensor_11.tolist() == [900000.0] * 5
    assert other_result.dataset_split == "development"


def test_cutoff_between_available_cycles_reports_actual_final_cycle(saved, raw):
    _, _, _, adapter = saved
    raw = raw[~((raw.engine_id == 1) & raw.cycle.isin([2, 4, 6]))]
    result = adapter.predict(raw, 1, 4)
    assert result.cycle == 3 and result.observed_history_length == 2 and result.padding_cycles == 28
    final = adapter.predict(raw, 1, 999)
    assert final.cycle == 45 and final.observed_history_length == 42 and final.padding_cycles == 0


def test_no_mutation_no_refit_load_once_and_private_preprocessing_snapshot(artifacts, raw, monkeypatch):
    bundle, root, _ = artifacts()
    real_load = inference.load_lstm_checkpoint
    calls = []
    def load(*args):
        calls.append(args)
        return real_load(*args)
    monkeypatch.setattr(inference, "load_lstm_checkpoint", load)
    adapter = inference.LSTMInferenceAdapter(bundle, root)
    original_raw = raw.copy(deep=True)
    original_scaler = copy.deepcopy(bundle.sequence_scaler.__dict__)
    original_medians = bundle.medians.values.copy()
    original_weights = copy.deepcopy(adapter._model.state_dict())
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("Inference refitted scaler"))
    monkeypatch.setattr(TrainingMedians, "fit", lambda *args, **kwargs: pytest.fail("Inference refitted medians"))
    result = adapter.predict(raw, 1, 5, scenario="short_gap", recovery="forward_fill")
    pd.testing.assert_frame_equal(raw, original_raw)
    assert bundle.medians.values == original_medians
    for name in ["mean_", "var_", "scale_", "feature_names_in_"]:
        np.testing.assert_array_equal(getattr(bundle.sequence_scaler, name), original_scaler[name])
    for name, weights in adapter._model.state_dict().items():
        torch.testing.assert_close(weights, original_weights[name], rtol=0, atol=0)
    # The loaded adapter is independent of caller mutations and does not re-read artifact metadata per call.
    bundle.selected_sensors.reverse()
    bundle.lstm_window = 3
    bundle.sequence_scaler.mean_.fill(99999)
    bundle.medians.values = {sensor: -99999 for sensor in bundle.medians.values}
    (root / "models/lstm_checkpoint_metadata.json").unlink()
    _assert_same(result, adapter.predict(raw, 1, 5, scenario="short_gap", recovery="forward_fill"))
    assert len(calls) == 1
    result.chart_prefix.loc[:, "sensor_2"] = -99999
    result.sensor_mask.loc[:, :] = False
    again = adapter.predict(raw, 1, 5, scenario="short_gap", recovery="forward_fill")
    assert (again.chart_prefix.sensor_2 > 0).all() and again.sensor_mask.sensor_11.all()
    pd.testing.assert_frame_equal(raw, original_raw)


@pytest.mark.parametrize("options,message", [
    ({"engine_id": 0}, "engine_id"), ({"engine_id": -1}, "engine_id"), ({"engine_id": True}, "engine_id"),
    ({"engine_id": 1.0}, "engine_id"), ({"engine_id": None}, "engine_id"), ({"engine_id": 999}, "not found"),
    ({"observed_cycle": 0}, "observed_cycle"), ({"observed_cycle": -1}, "observed_cycle"),
    ({"observed_cycle": 1.5}, "observed_cycle"), ({"observed_cycle": True}, "observed_cycle"),
    ({"observed_cycle": np.nan}, "observed_cycle"), ({"scenario": "S1"}, "scenario"),
    ({"scenario": []}, "scenario"), ({"recovery": "backward_fill"}, "Recovery"),
    ({"recovery": None}, "Recovery"), ({"recovery": "forward_fill"}, "Clean"),
    ({"scenario": "random_10"}, "seed"), ({"scenario": "random_20", "seed": 8}, "seed"),
    ({"scenario": "random_10", "seed": 7.0}, "seed"), ({"scenario": "random_10", "seed": True}, "seed"),
    ({"seed": 7}, "only to random"), ({"scenario": "short_gap", "seed": 7}, "only to random"),
    ({"dataset_split": ""}, "dataset_split"), ({"dataset_split": []}, "dataset_split"),
    ({"dataset_split": "official_test"}, "no such column"),
])
def test_invalid_options_fail_clearly(saved, raw, options, message):
    _, _, _, adapter = saved
    args = {"engine_id": 1, "observed_cycle": 5, **options}
    with pytest.raises(ValueError, match=message):
        adapter.predict(raw, **args)


@pytest.mark.parametrize("problem,message", [
    ("empty", "non-empty"), ("missing_sensor", "require columns"), ("duplicate_column", "unique columns"),
    ("duplicate_cycle", "Duplicate cycle"), ("duplicate_index", "indices"), ("invalid_cycle", "positive integers"),
    ("nan_cycle", "positive integers"), ("bool_cycle", "positive integers"),
    ("infinite_sensor", "infinity"), ("bad_sensor", "numeric"), ("complex_sensor", "numeric"),
    ("empty_visible", "empty visible"), ("missing_split", "missing engine identity"),
    ("blank_split", "non-empty string namespace"), ("wrong_split", "not found"),
])
def test_invalid_visible_inputs_fail_clearly(saved, raw, problem, message):
    _, _, _, adapter = saved
    options = {}
    if problem == "empty":
        raw = raw.iloc[:0]
    elif problem == "missing_sensor":
        raw = raw.drop(columns="sensor_11")
    elif problem == "duplicate_column":
        raw = pd.concat([raw, raw[["sensor_11"]]], axis=1)
    elif problem in {"duplicate_cycle", "duplicate_index"}:
        row = raw[(raw.engine_id == 1) & (raw.cycle == 2)].copy()
        if problem == "duplicate_cycle":
            row.index = [99999]
        else:
            row["cycle"] = 4
        raw = pd.concat([raw, row])
    elif problem in {"invalid_cycle", "nan_cycle", "bool_cycle"}:
        raw["cycle"] = raw.cycle.astype(object)
        raw.loc[(raw.engine_id == 1) & (raw.cycle == 2), "cycle"] = {"invalid_cycle": 1.5, "nan_cycle": np.nan, "bool_cycle": True}[problem]
    elif problem in {"infinite_sensor", "bad_sensor", "complex_sensor"}:
        raw["sensor_11"] = raw.sensor_11.astype(object)
        raw.loc[(raw.engine_id == 1) & (raw.cycle == 2), "sensor_11"] = {"infinite_sensor": np.inf, "bad_sensor": "bad", "complex_sensor": 1 + 2j}[problem]
    elif problem == "empty_visible":
        raw = raw[~((raw.engine_id == 1) & (raw.cycle == 1))]
        options["observed_cycle"] = 1
    else:
        raw["dataset_split"] = {"missing_split": None, "blank_split": " ", "wrong_split": "test"}[problem]
        if problem == "wrong_split":
            options["dataset_split"] = "development"
    with pytest.raises(ValueError, match=message):
        adapter.predict(raw, 1, options.pop("observed_cycle", 5), **options)


@pytest.mark.parametrize("problem,message", [
    ("sensor_order", "Sensor order"), ("window", "window"), ("architecture", "architecture"),
    ("missing_metadata", "legacy compatibility is unverified"), ("missing_input_metadata", "metadata is missing"),
    ("checksum", "checksum"), ("prefix_fraction", "split identity"),
    ("missing_median", "every frozen sensor"), ("infinite_median", "finite"),
])
def test_loading_rejects_incompatible_or_missing_artifacts(artifacts, problem, message):
    bundle, root, _ = artifacts()
    path = root / "models/lstm_checkpoint_metadata.json"
    if problem == "sensor_order":
        bundle.selected_sensors.reverse()
    elif problem == "window":
        bundle.lstm_window = 29
    elif problem == "architecture":
        metadata = json.loads(path.read_text())
        metadata["architecture"]["hidden_units"] = 64
        path.write_text(json.dumps(metadata), encoding="utf-8")
    elif problem == "missing_metadata":
        path.unlink()
    elif problem == "missing_input_metadata":
        (root / "preprocessing/sequence_input_metadata.json").unlink()
    elif problem == "checksum":
        with (root / "models/lstm.pt").open("ab") as stream:
            stream.write(b"Different synthetic checkpoint")
    elif problem == "prefix_fraction":
        path = root / "preprocessing/sequence_input_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["split"]["validation_prefix_fraction"] = 0.8
        path.write_text(json.dumps(metadata), encoding="utf-8")
    elif problem == "missing_median":
        bundle.medians.values.pop("sensor_2")
    else:
        bundle.medians.values["sensor_2"] = np.inf
    with pytest.raises(ValueError, match=message):
        inference.LSTMInferenceAdapter(bundle, root)


@pytest.mark.parametrize("bias", [-10.0, np.nan, np.inf, -np.inf])
def test_actual_model_output_is_finite_before_negative_clamping(artifacts, raw, bias):
    bundle, root, _ = artifacts(bias=bias)
    adapter = inference.LSTMInferenceAdapter(bundle, root)
    if np.isfinite(bias):
        assert adapter.predict(raw, 1, 5).rul_pred == 0.0
    else:
        with pytest.raises(ValueError, match="finite.*before negative clamping"):
            adapter.predict(raw, 1, 5)
