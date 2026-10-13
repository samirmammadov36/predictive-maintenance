"""Actual saved RF/XGBoost/baseline inference and common replay contracts."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_lstm_inference import artifacts, raw, single_cpu_thread, CONFIGURATIONS
from predictive_maintenance import inference
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.models.baseline import MedianRULBaseline
from predictive_maintenance.replay import (
    MODEL_ARTIFACTS, artifact_fingerprint, load_replay_resources, build_model_replay, build_inspection_queue,
)

MODELS = ("random_forest", "xgboost", "median_baseline")
ALL_MODELS = (*MODELS, "lstm")


@pytest.fixture
def saved_models(artifacts, raw):
    def create(named=False, targets=None):
        bundle, root, lstm = artifacts()
        bundle.rolling_window = 10
        features, bundle.feature_columns = build_tabular_features(bundle.train_df, bundle.selected_sensors, 10)
        bundle.tabular_scaler = StandardScaler().fit(features[bundle.feature_columns])
        matrix = bundle.tabular_scaler.transform(features[bundle.feature_columns])
        if named:
            matrix = pd.DataFrame(matrix, columns=bundle.feature_columns)
        targets = np.array([70., 60., 40., 30.]) if targets is None else np.asarray(targets)
        rf = RandomForestRegressor(n_estimators=8, max_depth=3, random_state=42, n_jobs=1).fit(matrix, targets)
        xgb = XGBRegressor(n_estimators=5, max_depth=2, objective="reg:squarederror", random_state=42, n_jobs=1).fit(matrix, targets)
        baseline = MedianRULBaseline().fit(targets)
        joblib.dump(rf, root / "models/random_forest.joblib")
        xgb.save_model(root / "models/xgboost.json")
        joblib.dump(baseline, root / "models/baseline.joblib")
        bundle.official_test_raw_df = raw.copy(deep=True)
        bundle.official_test_df = raw.fillna(777777.)  # Must never reach inference as raw.
        joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
        (root / "preprocessing/schema.json").write_text(json.dumps({
            "selected_sensors": bundle.selected_sensors, "feature_columns": bundle.feature_columns,
        }), encoding="utf-8")
        return bundle, root, {"random_forest": rf, "xgboost": xgb, "median_baseline": baseline, "lstm": lstm}
    return create


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("named", [False, True])
def test_saved_prediction_parity_frozen_feature_order_and_early_histories(saved_models, model_name, named):
    bundle, root, originals = saved_models(named)
    adapter = inference.load_inference_adapter(bundle, root, model_name)
    for cutoff in (1, 5, 29, 30, 31):
        result = adapter.predict(bundle.official_test_raw_df, 1, cutoff)
        assert result.cycle == cutoff and result.observed_history_length == cutoff
        assert result.sequence_window is None and result.padding_cycles == 0
        assert list(result.chart_prefix) == ["engine_id", "cycle", "sensor_2", "sensor_11"]
        assert result.chart_prefix.sensor_2.iloc[0] == 25.
        assert result.chart_prefix.sensor_11.iloc[0] == 110.
        assert result.chart_prefix.cycle.max() == cutoff
        if model_name == "median_baseline":
            expected = originals[model_name].predict(1)[0]
        else:
            features, _ = build_tabular_features(result.chart_prefix, bundle.selected_sensors, 10)
            matrix = bundle.tabular_scaler.transform(features.iloc[[-1]][bundle.feature_columns])
            if named:
                matrix = pd.DataFrame(matrix, columns=bundle.feature_columns)
            expected = originals[model_name].predict(matrix)[0]
        assert result.rul_pred == max(float(expected), 0.)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
def test_shared_masks_recovery_future_and_other_engine_isolation(saved_models, model_name, scenario, recovery, seed):
    bundle, root, _ = saved_models()
    resources = load_replay_resources(root, model_name)
    captured = []
    original_predict = resources.adapter._model.predict
    def model_predict(*args, **kwargs):
        captured.append(np.asarray(args[0]).copy())
        return original_predict(*args, **kwargs)
    resources.adapter._model.predict = model_predict
    options = dict(scenario=scenario, recovery=recovery, seed=seed)
    before, history = build_model_replay(resources, (None, 1), 8, **options)
    lstm = inference.LSTMInferenceAdapter(bundle, root).predict(bundle.official_test_raw_df, 1, 8, **options)
    pd.testing.assert_frame_equal(before.sensor_mask, lstm.sensor_mask)
    pd.testing.assert_frame_equal(before.chart_prefix, lstm.chart_prefix)
    altered = bundle.official_test_raw_df.copy(deep=True)
    altered.loc[(altered.engine_id == 1) & (altered.cycle > 8), "sensor_11"] = np.inf
    altered.loc[altered.engine_id == 2, "sensor_2"] = -999999.
    altered[["rul_true", "rul_target"]] = np.nan
    original = resources.raw_trajectories
    changed = type(resources)(resources.adapter, altered, resources.engine_cycles)
    after, altered_history = build_model_replay(changed, (None, 1), 8, **options)
    assert before.rul_pred == after.rul_pred
    pd.testing.assert_frame_equal(before.chart_prefix, after.chart_prefix)
    pd.testing.assert_frame_equal(before.sensor_mask, after.sensor_mask)
    pd.testing.assert_frame_equal(history, altered_history)
    assert history.cycle.tolist() == list(range(1, 9))
    assert len(captured) == 16
    for left, right in zip(captured[:8], captured[8:]):
        np.testing.assert_array_equal(left, right)
    if model_name == 'median_baseline':
        assert before.rul_pred == 50. and all(value == 1 for value in captured)
    pd.testing.assert_frame_equal(resources.raw_trajectories, original)
    if scenario == "short_gap":
        expected = [110., 20.] + ([110.] * 6 if recovery == "median" else [20.] * 6)
        assert before.chart_prefix.sensor_11.tolist() == expected
        if model_name != 'median_baseline':
            values = [8., 28., 20., 25., 3., 31.25, -90.] if recovery == 'forward_fill' else [8., 28., 110., 25., 3., 98.75, 0.]
            np.testing.assert_allclose(bundle.tabular_scaler.inverse_transform(captured[7])[0], values)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_each_replay_call_has_own_cutoff_and_queue_uses_all_final_snapshots(saved_models, model_name):
    bundle, root, _ = saved_models()
    one = bundle.official_test_raw_df.assign(dataset_split="official_test")
    other = one.assign(dataset_split="development", sensor_11=np.inf)
    late = one[one.engine_id == 2].assign(engine_id=3, cycle=lambda df: df.cycle + 50)
    bundle.official_test_raw_df = pd.concat([one, other, late])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    resources = load_replay_resources(root, model_name)
    actual, calls = resources.adapter.predict, []
    def predict(raw, engine, cycle, **options):
        result = actual(raw, engine, cycle, **options)
        calls.append((engine, cycle, result))
        return result
    resources.adapter.predict = predict
    result, history = build_model_replay(resources, ("official_test", 1), 5, scenario="short_gap", recovery="forward_fill")
    assert [(e, c) for e, c, _ in calls] == [(1, c) for c in range(1, 6)]
    assert [point.sensor_mask.sensor_11.sum() for _, _, point in calls] == [1, 2, 3, 4, 5]
    assert result.chart_prefix.cycle.max() == history.cycle.max() == 5
    calls.clear()
    queue = build_inspection_queue(resources, "official_test", 5)
    assert [(e, c) for e, c, _ in calls] == [(1, 45), (2, 9), (3, 59)]
    assert set(queue.engine_id) == {1, 2, 3}
    assert set(queue.dataset_split) == {"official_test"}
    build_model_replay(resources, ("official_test", 1), 1)
    pd.testing.assert_frame_equal(queue, build_inspection_queue(resources, "official_test", 5))


@pytest.mark.parametrize("model_name", MODELS)
def test_no_refit_mutation_and_private_preprocessing_snapshot(saved_models, model_name, monkeypatch):
    bundle, root, _ = saved_models()
    frame = bundle.official_test_raw_df.copy(deep=True)
    state = copy.deepcopy(bundle.tabular_scaler.__dict__)
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("Inference refitted scaler"))
    monkeypatch.setattr(TrainingMedians, "fit", lambda *a, **k: pytest.fail("Inference refitted medians"))
    adapter = inference.load_inference_adapter(bundle, root, model_name)
    before = adapter.predict(frame, 1, 5, scenario="short_gap", recovery="forward_fill")
    for name in ("mean_", "var_", "scale_", "feature_names_in_"):
        np.testing.assert_array_equal(getattr(bundle.tabular_scaler, name), state[name])
    bundle.selected_sensors.reverse()
    bundle.feature_columns.reverse()
    bundle.tabular_scaler.mean_.fill(99999.)
    bundle.medians.values = {}
    after = adapter.predict(frame, 1, 5, scenario="short_gap", recovery="forward_fill")
    assert before.rul_pred == after.rul_pred
    pd.testing.assert_frame_equal(before.chart_prefix, after.chart_prefix)
    pd.testing.assert_frame_equal(frame, bundle.official_test_raw_df)
    assert np.isfinite(state["mean_"]).all()


@pytest.mark.parametrize("model_name", ("random_forest", "xgboost"))
@pytest.mark.parametrize("problem", ("schema_missing", "order", "schema", "scaler_unfit", "scaler_order", "scaler_nan", "model_count", "model_order"))
def test_concrete_tabular_incompatibilities_fail(saved_models, model_name, problem):
    bundle, root, _ = saved_models(named=True)
    if problem == "schema_missing":
        (root / "preprocessing/schema.json").unlink()
    elif problem == "order":
        bundle.feature_columns.reverse()
    elif problem == "schema":
        (root / "preprocessing/schema.json").write_text("{}", encoding="utf-8")
    elif problem == "scaler_unfit":
        bundle.tabular_scaler = StandardScaler()
    elif problem == "scaler_order":
        bundle.tabular_scaler.feature_names_in_ = bundle.tabular_scaler.feature_names_in_[::-1]
    elif problem == "scaler_nan":
        bundle.tabular_scaler.scale_[0] = np.nan
    else:
        columns = bundle.feature_columns[:-1] if problem == "model_count" else bundle.feature_columns[::-1]
        X = pd.DataFrame(np.arange(4 * len(columns)).reshape(4, -1), columns=columns)
        if model_name == "random_forest":
            model = RandomForestRegressor(n_estimators=2, n_jobs=1).fit(X, [1, 2, 3, 4])
            joblib.dump(model, root / "models/random_forest.joblib")
        else:
            model = XGBRegressor(n_estimators=2, n_jobs=1).fit(X, [1, 2, 3, 4])
            model.save_model(root / "models/xgboost.json")
    with pytest.raises(ValueError, match="schema|order|fitted|finite|count"):
        inference.load_inference_adapter(bundle, root, model_name)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("output", ([], [1, 2], [np.nan], [np.inf], [-np.inf], [1 + 2j], [-2.]))
def test_finite_before_clamp_on_loaded_models(saved_models, model_name, output, monkeypatch):
    bundle, root, _ = saved_models()
    adapter = inference.load_inference_adapter(bundle, root, model_name)
    monkeypatch.setattr(adapter._model, "predict", lambda *a, **k: output)
    if output == [-2.]:
        assert adapter.predict(bundle.official_test_raw_df, 1, 5).rul_pred == 0.
    else:
        with pytest.raises(ValueError, match="finite.*(prediction|before)"):
            adapter.predict(bundle.official_test_raw_df, 1, 5)


@pytest.mark.parametrize("model_name", MODELS)
def test_only_selected_artifacts_and_optional_xgboost_dependency(saved_models, model_name, monkeypatch):
    bundle, root, _ = saved_models()
    required = set(MODEL_ARTIFACTS[model_name])
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in required:
            path.unlink()
    if model_name != "xgboost":
        monkeypatch.setitem(sys.modules, "xgboost", None)
    resources = load_replay_resources(root, model_name)
    assert resources.adapter.predict(resources.raw_trajectories, 1, 1).rul_pred >= 0
    assert {name for name, _ in artifact_fingerprint(root, model_name)} == required
    for name in required:
        path = root / name
        original = path.read_bytes()
        fingerprint = artifact_fingerprint(root, model_name)
        path.write_bytes(original + b" ")
        assert fingerprint != artifact_fingerprint(root, model_name)
        path.write_bytes(original)
    model_path = next(name for name in required if name.startswith("models/"))
    (root / model_path).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_replay_resources(root, model_name)

@pytest.mark.parametrize("model_name", MODELS)
def test_actual_saved_negative_predictions_are_clamped(saved_models, model_name):
    bundle, root, _ = saved_models(targets=[-2., -3., -4., -5.])
    adapter = inference.load_inference_adapter(bundle, root, model_name)
    assert adapter.predict(bundle.official_test_raw_df, 1, 5).rul_pred == 0.


@pytest.mark.parametrize("model_name", ("lstm", "random_forest", "median_baseline"))
def test_missing_xgboost_package_does_not_block_other_models(saved_models, model_name, monkeypatch):
    _, root, _ = saved_models()
    (root / "models/xgboost.json").unlink()
    monkeypatch.setitem(sys.modules, "xgboost", None)
    resources = load_replay_resources(root, model_name)
    assert resources.adapter.predict(resources.raw_trajectories, 1, 1).rul_pred >= 0.


@pytest.mark.parametrize("value", (None, np.nan, np.inf))
def test_baseline_rejects_unfitted_or_nonfinite_artifact(saved_models, value):
    bundle, root, models = saved_models()
    models["median_baseline"].value_ = value
    joblib.dump(models["median_baseline"], root / "models/baseline.joblib")
    with pytest.raises(ValueError, match="fitted|finite"):
        inference.load_inference_adapter(bundle, root, "median_baseline")

def test_existing_lstm_loader_predictor_keyword_calls_and_public_aliases(saved_models):
    bundle, root, _ = saved_models()
    model = inference.load_lstm_checkpoint(bundle=bundle, artifact_dir=root)
    X = np.zeros((1, 30, len(bundle.selected_sensors)), dtype=np.float32)
    assert inference.predict_lstm(model=model, X=X).shape == (1,)
    from predictive_maintenance.replay import LSTMReplayResources, ReplayResources, build_lstm_replay
    assert inference.LSTMInferenceResult is inference.InferenceResult
    assert LSTMReplayResources is ReplayResources and build_lstm_replay is build_model_replay
