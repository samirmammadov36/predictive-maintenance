"""Synthetic evaluator tests; generated preparation artifacts stay in tmp_path."""
from __future__ import annotations

import copy
import importlib.util
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

from predictive_maintenance import experiments
from predictive_maintenance.data import add_training_rul
from predictive_maintenance.evaluation import (
    FD001_ENGINE_IDS, validate_official_test_inputs, validate_official_test_predictions,
)
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from predictive_maintenance.prepared import PreparedBundle, official_test_raw_frame
from predictive_maintenance.sequences import fit_sequence_scaler


class _TabularSpy:
    def __init__(self, constant=None):
        self.constant = constant
        self.inputs = []

    def predict(self, X):
        self.inputs.append(np.asarray(X).copy())
        return np.full(len(X), self.constant) if self.constant is not None else X[:, 0]


class _SequenceSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.inputs = []

    def forward(self, X):
        self.inputs.append(X.detach().cpu().numpy().copy())
        return X[:, -1, 0] + self.anchor * 0


def _bundle(sensors=None):
    sensors = ["sensor_11", "sensor_2"] if sensors is None else sensors
    train = add_training_rul(pd.DataFrame({
        "engine_id": [21, 21, 22, 22], "cycle": [1, 2, 1, 2],
        "sensor_11": [80.0, 100.0, 120.0, 140.0], "sensor_2": [10.0, 20.0, 30.0, 40.0],
    }))
    medians = TrainingMedians.fit(train, sensors)
    rows = []
    for engine, length in [(1, 1), (2, 7), (3, 31)]:
        for cycle in range(1, length + 1):
            rows.append({"engine_id": engine, "cycle": cycle, "sensor_2": float(engine * 10 + cycle),
                         "setting_1": float(engine), "sensor_11": float(200 + cycle if engine == 3 else cycle * 10)})
    raw = pd.DataFrame(rows, index=100 + 3 * np.arange(len(rows)))
    raw.loc[(raw.engine_id == 1) | ((raw.engine_id == 2) & (raw.cycle == 2)), ["sensor_11", "sensor_2"]] = np.nan
    raw["dataset_split"] = "official_test"
    raw = raw.sample(frac=1, random_state=42)
    targets = pd.DataFrame({"engine_id": [3, 1, 2], "cycle": [31, 1, 7], "rul_true": [5.0, 200.0, 150.0],
                            "dataset_split": "official_test"})
    engineered, features = build_tabular_features(train, sensors, 10)
    return PreparedBundle(
        train, train.copy(), medians.median_fill(raw, sensors), targets, sensors, features,
        [21, 22], [23], {23: 1}, medians, fit_sequence_scaler(train, sensors),
        StandardScaler().fit(engineered[features]), 10, 30, official_test_raw_df=raw,
    )


@pytest.fixture
def observed_run(monkeypatch):
    bundle = _bundle()
    models = {"random_forest": _TabularSpy(-3.0), "xgboost": _TabularSpy(), "lstm": _SequenceSpy()}
    monkeypatch.setattr(experiments, "_load_models", lambda *args: models)
    context, masks, recovered, model_frames = {}, {}, {}, {}
    corrupt_original, recover_original, predict_original = experiments._corrupt, experiments._recover, experiments._predict

    def corrupt(prefix, bundle, scenario, seed):
        context.update(scenario=scenario, seed=seed, engine=int(prefix.engine_id.iloc[0]))
        output, sensor = corrupt_original(prefix, bundle, scenario, seed)
        key = (scenario, seed, context["engine"])
        masks.setdefault(key, []).append((prefix.copy(deep=True), output.copy(deep=True)))
        return output, sensor

    def recover(prefix, bundle, method):
        context["recovery"] = method
        output = recover_original(prefix, bundle, method)
        key = (context["scenario"], context["seed"], method, context["engine"])
        recovered[key] = output.copy(deep=True)
        return output

    def predict(name, model, prefix, bundle):
        key = (context["scenario"], context["seed"], context["recovery"], context["engine"])
        model_frames.setdefault(key, []).append((name, prefix.copy(deep=True)))
        return predict_original(name, model, prefix, bundle)

    monkeypatch.setattr(experiments, "_corrupt", corrupt)
    monkeypatch.setattr(experiments, "_recover", recover)
    monkeypatch.setattr(experiments, "_predict", predict)
    return bundle, models, masks, recovered, model_frames


def _run(bundle):
    return experiments.run_experiments(bundle, return_predictions=True, expected_engine_ids=(1, 2, 3))


def test_raw_mask_recovery_shared_inputs_and_all_configurations(observed_run, monkeypatch):
    bundle, models, masks, recovered, model_frames = observed_run
    before = copy.deepcopy(bundle)
    # Poison the compatibility frame: it must never be mistaken for raw inputs.
    bundle.official_test_df.loc[:, bundle.selected_sensors] = 777777.0
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("Evaluation refitted preprocessing"))
    monkeypatch.setattr(TrainingMedians, "fit", lambda *args, **kwargs: pytest.fail("Evaluation refitted medians"))
    results, predictions = _run(bundle)
    assert len(results) == 45 and len(predictions) == 135
    expected = {("clean", None, "median"), ("short_gap", None, "median"), ("short_gap", None, "forward_fill")}
    expected |= {(scenario, seed, recovery) for scenario in ["random_10", "random_20"]
                 for seed in [7, 17, 27] for recovery in ["median", "forward_fill"]}
    for name, group in results.groupby("model"):
        actual = {(row.scenario, None if pd.isna(row.seed) else int(row.seed), row.recovery) for row in group.itertuples()}
        assert actual == expected and len(group) == 15
        assert group.n_engines.tolist() == [3] * 15
    # Each configuration/engine shares one recovered sensor frame across all three models.
    for frames in model_frames.values():
        assert {name for name, _ in frames} == set(models)
        for _, frame in frames[1:]:
            pd.testing.assert_frame_equal(frame, frames[0][1])
    for (scenario, seed, engine), entries in masks.items():
        assert len(entries) == (1 if scenario == "clean" else 2)
        for original, corrupted in entries:
            assert original.cycle.is_monotonic_increasing
            if scenario.startswith("random"):
                low, high = nested_random_masks(original, bundle.selected_sensors, seed)
                assert not (low & ~high).to_numpy().any()
                expected_mask = low if scenario == "random_10" else high
                pd.testing.assert_frame_equal(corrupted, apply_sensor_mask(original, expected_mask, bundle.selected_sensors))
            elif scenario == "short_gap":
                mask, sensor = short_gap_mask(original, bundle.selected_sensors)
                assert sensor == "sensor_11"
                pd.testing.assert_frame_equal(corrupted, apply_sensor_mask(original, mask, bundle.selected_sensors))
        if len(entries) == 2:
            pd.testing.assert_frame_equal(entries[0][1], entries[1][1])
    # Native cycle-2 NaN precedes the S3 gap (cycles 3..7). Raw FF uses cycle 1, not a prefilled median.
    gap_ff = recovered[("short_gap", None, "forward_fill", 2)]
    assert gap_ff.sensor_11.tolist() == [10.0] * 7
    assert bundle.medians.values["sensor_11"] == 110.0
    assert recovered[("short_gap", None, "median", 2)].sensor_11.tolist() == [10.0] + [110.0] * 6
    assert recovered[("clean", None, "median", 1)].sensor_11.tolist() == [110.0]
    pd.testing.assert_frame_equal(bundle.official_test_raw_df, before.official_test_raw_df)
    pd.testing.assert_frame_equal(bundle.train_df, before.train_df)
    assert bundle.medians.values == before.medians.values
    for name in ["mean_", "var_", "scale_", "feature_names_in_"]:
        np.testing.assert_array_equal(getattr(bundle.sequence_scaler, name), getattr(before.sequence_scaler, name))
    for inputs in models["lstm"].inputs:
        assert inputs.shape == (1, 30, 2) and np.isfinite(inputs).all()


def test_scoring_uncapped_labels_final_cycles_negative_clamp_and_coverage(observed_run):
    bundle, _, _, _, _ = observed_run
    results, predictions = _run(bundle)
    assert set(predictions.loc[predictions.engine_id == 1, "rul_true"]) == {200.0}
    assert set(predictions.loc[predictions.engine_id == 2, "rul_true"]) == {150.0}
    assert predictions.groupby("engine_id").cycle.unique().apply(list).to_dict() == {1: [1], 2: [7], 3: [31]}
    assert (predictions.rul_pred >= 0).all()
    assert predictions.loc[predictions.model == "random_forest", "rul_pred"].tolist() == [0.0] * 45
    rf = results[results.model == "random_forest"]
    np.testing.assert_allclose(rf.mae, 355.0 / 3)
    np.testing.assert_allclose(rf.rmse, np.sqrt((200 ** 2 + 150 ** 2 + 5 ** 2) / 3))
    lstm = results[results.model == "lstm"]
    assert lstm.n_padded_engines.tolist() == [2] * 15 and lstm.n_unpadded_engines.tolist() == [1] * 15
    assert lstm.min_observed_history_length.tolist() == [1] * 15
    assert lstm.max_observed_history_length.tolist() == [31] * 15
    for engine, history, used, padding in [(1, 1, 1, 29), (2, 7, 7, 23), (3, 31, 30, 0)]:
        rows = predictions[(predictions.model == "lstm") & (predictions.engine_id == engine)]
        assert rows.observed_history_length.tolist() == [history] * 15
        assert rows.sequence_history_length.tolist() == [used] * 15
        assert rows.padding_cycles.tolist() == [padding] * 15 and rows.sequence_window.tolist() == [30] * 15
    # Actual clean predictions identify the final observation used by each input path.
    clean_predictions = predictions[predictions.scenario == "clean"]
    np.testing.assert_allclose(
        clean_predictions.loc[clean_predictions.model == "lstm", "rul_pred"],
        [0.0, 0.0, (231.0 - 110.0) / np.sqrt(500.0)], rtol=1e-6,
    )
    np.testing.assert_allclose(
        clean_predictions.loc[clean_predictions.model == "xgboost", "rul_pred"],
        [0.0, 11.0, 59.0],
    )
    clean = results[results.scenario == "clean"].set_index("model")
    for row in results.itertuples():
        assert row.delta_mae == pytest.approx(row.mae - clean.loc[row.model, "mae"])
        assert row.delta_rmse == pytest.approx(row.rmse - clean.loc[row.model, "rmse"])
    aggregated = experiments.aggregate_random_results(results)
    for row in aggregated[aggregated.scenario.str.startswith("random")].itertuples():
        source = results[(results.model == row.model) & (results.scenario == row.scenario) & (results.recovery == row.recovery)]
        assert len(source) == 3
        assert row.mae_mean == pytest.approx(source.mae.mean()) and row.mae_std == pytest.approx(source.mae.std())
        assert row.rmse_mean == pytest.approx(source.rmse.mean()) and row.rmse_std == pytest.approx(source.rmse.std())
        assert row.delta_mae_mean == pytest.approx(source.delta_mae.mean())
        assert row.delta_rmse_std == pytest.approx(source.delta_rmse.std())


def test_unsorted_runner_prefixes_are_causal_and_never_cross_engines(observed_run):
    bundle, _, _, recovered, _ = observed_run
    raw = bundle.official_test_raw_df
    raw.loc[raw.engine_id == 1, "sensor_11"] = 9990.0
    raw.loc[(raw.engine_id == 2) & (raw.cycle == 1), "sensor_11"] = np.nan
    _run(bundle)
    first = recovered[("random_10", 7, "forward_fill", 1)]
    second = recovered[("random_10", 7, "forward_fill", 2)]
    assert first.sensor_11.iloc[-1] == 9990.0
    assert second.sensor_11.iloc[0] == 110.0  # No previous-engine value or later observed sensor.
    assert second.sensor_11.iloc[1] == 110.0
    assert second.engine_id.tolist() == [2] * 7 and second.cycle.tolist() == list(range(1, 8))


def test_short_gap_fallback_and_single_cycle_padding(monkeypatch):
    bundle = _bundle(["sensor_2"])
    sequence = _SequenceSpy()
    monkeypatch.setattr(experiments, "_load_models", lambda *args: {"lstm": sequence})
    results, predictions = _run(bundle)
    assert results.loc[results.scenario == "short_gap", "gap_sensor"].tolist() == ["sensor_2"] * 2
    assert len(predictions) == 45
    for inputs in sequence.inputs:
        assert inputs.shape == (1, 30, 1)
    # Engine 1 has one imputed observation, so every element repeats it.
    np.testing.assert_array_equal(sequence.inputs[0], np.zeros((1, 30, 1), dtype=np.float32))


@pytest.mark.parametrize("old_style", ["none", "missing_attribute"])
def test_old_bundles_fail_actionably_before_loading_models(monkeypatch, old_style):
    bundle = _bundle()
    if old_style == "none":
        bundle.official_test_raw_df = None
    else:
        del bundle.official_test_raw_df
    monkeypatch.setattr(experiments, "_load_models", lambda *args: pytest.fail("Models loaded before raw-input validation"))
    with pytest.raises(ValueError, match="Regenerate with.*scripts/prepare_data.py.*cannot be used as raw"):
        _run(bundle)


def test_default_official_evaluation_requires_exactly_100_engines(monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(experiments, "_load_models", lambda *args: pytest.fail("Models loaded before coverage validation"))
    with pytest.raises(ValueError, match="expected 100 engine identities"):
        experiments.run_experiments(bundle)


@pytest.mark.parametrize("seeds", [(7, 17), (7, 17, 17), (7, 17, 29), (7.0, 17, 27)])
def test_missingness_seed_contract_is_fixed(seeds):
    with pytest.raises(ValueError, match="exactly random seeds"):
        experiments.run_experiments(_bundle(), random_seeds=seeds, expected_engine_ids=(1, 2, 3))


@pytest.mark.parametrize("problem,message", [
    ("missing_label", "expected 3 engine"), ("extra_label", "expected 3 engine"),
    ("duplicate_label", "Duplicate official-test labels"), ("label_cycle", "final observed cycle"),
    ("missing_raw_engine", "expected 3 engine"), ("duplicate_cycle", "engine/cycle"),
    ("mixed_split", "dataset_split"), ("label_split", "identity mismatch"),
    ("infinite_sensor", "infinity"), ("nan_label", "RUL labels must be finite"),
])
def test_input_coverage_and_labels_are_validated(problem, message):
    bundle = _bundle()
    raw, labels = bundle.official_test_raw_df.copy(), bundle.official_test_targets.copy()
    if problem == "missing_label":
        labels = labels[labels.engine_id != 2]
    elif problem == "extra_label":
        labels = pd.concat([labels, labels.iloc[[0]].assign(engine_id=4)], ignore_index=True)
    elif problem == "duplicate_label":
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    elif problem == "label_cycle":
        labels.loc[labels.engine_id == 2, "cycle"] = 6
    elif problem == "missing_raw_engine":
        raw = raw[raw.engine_id != 2]
    elif problem == "duplicate_cycle":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    elif problem == "mixed_split":
        raw.loc[raw.index[0], "dataset_split"] = "development"
    elif problem == "label_split":
        labels["dataset_split"] = "development"
    elif problem == "infinite_sensor":
        raw.loc[raw.index[0], "sensor_11"] = np.inf
    else:
        labels.loc[labels.engine_id == 1, "rul_true"] = np.nan
    with pytest.raises(ValueError, match=message):
        validate_official_test_inputs(raw, labels, bundle.selected_sensors, (1, 2, 3))


@pytest.mark.parametrize("problem,message", [
    ("missing", "coverage mismatch"), ("extra", "coverage mismatch"), ("duplicate", "coverage mismatch"),
    ("wrong_cycle", "final observed cycle"), ("capped_label", "uncapped RUL labels"),
    ("nan", "finite before"), ("inf", "finite before"), ("negative_inf", "finite before"),
])
def test_prediction_coverage_and_alignment_failures(problem, message):
    bundle = _bundle()
    manifest = validate_official_test_inputs(bundle.official_test_raw_df, bundle.official_test_targets, bundle.selected_sensors, (1, 2, 3))
    predictions = manifest.reset_index().assign(rul_pred=[1.0, 2.0, 3.0])
    if problem == "missing":
        predictions = predictions.iloc[:-1]
    elif problem == "extra":
        predictions.loc[2, "engine_id"] = 4
    elif problem == "duplicate":
        predictions.loc[2, "engine_id"] = 1
    elif problem == "wrong_cycle":
        predictions.loc[1, "cycle"] = 6
    elif problem == "capped_label":
        predictions.loc[0, "rul_true"] = 125.0
    else:
        predictions.loc[0, "rul_pred"] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[problem]
    with pytest.raises(ValueError, match=message):
        validate_official_test_predictions(predictions, manifest)


@pytest.mark.parametrize("output", [[], [1.0, 2.0], [np.nan], [np.inf], [-np.inf]])
def test_model_must_emit_exactly_one_finite_output(output):
    bundle = _bundle()
    prefix = bundle.medians.median_fill(bundle.official_test_raw_df[lambda frame: frame.engine_id == 1], bundle.selected_sensors)
    model = SimpleNamespace(predict=lambda X: np.asarray(output))
    with pytest.raises(ValueError, match="exactly one finite"):
        experiments._predict("random_forest", model, prefix, bundle)


def test_saved_bundle_roundtrip_preserves_native_raw_nans(tmp_path):
    original = _bundle()
    path = tmp_path / "synthetic_bundle.joblib"
    joblib.dump(original, path)
    loaded = joblib.load(path)
    pd.testing.assert_frame_equal(official_test_raw_frame(loaded), original.official_test_raw_df)
    assert loaded.official_test_raw_df[loaded.selected_sensors].isna().any().any()
    assert not loaded.official_test_df[loaded.selected_sensors].isna().any().any()


def test_real_entrypoint_enforces_100_engine_requirement(tmp_path, monkeypatch):
    bundle = _bundle()
    prep = tmp_path / "artifacts/preprocessing"
    prep.mkdir(parents=True)
    joblib.dump(bundle, prep / "prepared_bundle.joblib")
    spec = importlib.util.spec_from_file_location("_test_run_checkpoint_experiments", ROOT / "scripts/run_checkpoint_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "_load_models", lambda *args: pytest.fail("Real entrypoint bypassed engine coverage check"))
    with pytest.raises(ValueError, match="expected 100 engine identities"):
        module.main()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("label_count", [99, 100, 101])
def test_preparation_stores_raw_inputs_and_rejects_extra_missing_labels(tmp_path, monkeypatch, label_count):
    cfg = {"project": {"dataset": "NASA C-MAPSS FD001", "seed": 42, "rul_cap": 125,
                       "validation_fraction": 0.2, "validation_prefix_fraction": 0.7},
           "data": {"raw_dir": "data/raw", "processed_dir": "data/processed", "train_file": "synthetic_train.txt",
                    "test_file": "synthetic_test.txt", "test_rul_file": "synthetic_RUL.txt"},
           "features": {"rolling_window": 10}, "lstm": {"window": 30},
           "screening": {"near_constant_unique_ratio": 0.0005, "near_constant_dominant_fraction": 0.995}}
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    for filename in ["synthetic_train.txt", "synthetic_test.txt", "synthetic_RUL.txt"]:
        (raw_dir / filename).write_text("Synthetic test fixture, not real FD001 data.", encoding="utf-8")
    train = pd.DataFrame({"engine_id": np.repeat(np.arange(1, 101), 2), "cycle": [1, 2] * 100})
    train["sensor_11"] = train.engine_id.astype(float) + train.cycle
    train["sensor_2"] = train.engine_id.astype(float) * 2 + train.cycle
    official = train[train.cycle == 1].copy().reset_index(drop=True)
    official.loc[official.engine_id == 2, "sensor_11"] = np.nan
    labels = pd.Series(np.arange(200, 200 + label_count, dtype=float), index=range(1, label_count + 1))
    monkeypatch.setitem(sys.modules, "predictive_maintenance.config", SimpleNamespace(load_config=lambda path: cfg))
    spec = importlib.util.spec_from_file_location("_test_prepare_data", ROOT / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "read_cmapss_file", lambda path: train.copy() if path.name == "synthetic_train.txt" else official.copy())
    monkeypatch.setattr(module, "read_test_rul", lambda path: labels.copy())
    monkeypatch.setattr(module, "screen_sensors", lambda *args: (["sensor_11", "sensor_2"], pd.DataFrame({"sensor": ["sensor_11", "sensor_2"]})))
    if label_count != 100:
        with pytest.raises(ValueError, match="exactly the 100 official-test"):
            module.main()
        assert not (tmp_path / "artifacts/preprocessing/prepared_bundle.joblib").exists()
        return
    module.main()
    bundle = joblib.load(tmp_path / "artifacts/preprocessing/prepared_bundle.joblib")
    pd.testing.assert_frame_equal(bundle.official_test_raw_df, official)
    assert pd.isna(bundle.official_test_raw_df.loc[1, "sensor_11"])
    assert bundle.official_test_df.loc[1, "sensor_11"] == bundle.medians.values["sensor_11"]
    assert bundle.official_test_targets.rul_true.max() == 299
    assert len(bundle.train_engine_ids) == 80 and len(bundle.validation_engine_ids) == 20
    np.testing.assert_allclose(bundle.sequence_scaler.mean_, bundle.train_df[bundle.selected_sensors].mean().to_numpy())
