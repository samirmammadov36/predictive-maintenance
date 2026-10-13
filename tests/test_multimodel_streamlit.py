"""Actual multi-model AppTest inference, plots, final queues and selected-artifact caching."""
from __future__ import annotations
from pathlib import Path
import sys
import copy

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app
from test_saved_inference import saved_models, artifacts, raw, single_cpu_thread, MODELS, ALL_MODELS
from test_streamlit_lstm import clear_application_resource_cache, _application, _healthy_app
from predictive_maintenance.inference import _PrefixInferenceAdapter
from predictive_maintenance.replay import load_replay_resources, build_inspection_queue

LABELS = {value: key for key, value in app.MODEL_NAMES.items()}
OPTIONS = [("clean", "median", None), ("short_gap", "forward_fill", None), ("random_20", "forward_fill", 17)]


@pytest.fixture
def plots(monkeypatch):
    captured = {"sensor": [], "rul": [], "calls": [], "downloads": []}
    for kind, name in (("sensor", "plot_sensor_history"), ("rul", "plot_rul_history")):
        original = getattr(app, name)
        def plot(*args, _original=original, _kind=kind, **kwargs):
            figure = _original(*args, **kwargs)
            captured[_kind].append([(line.get_xdata().copy(), line.get_ydata().copy()) for line in figure.axes[0].lines])
            return figure
        monkeypatch.setattr(app, name, plot)
    original = _PrefixInferenceAdapter.predict
    def predict(self, raw, engine, cycle, **options):
        captured["calls"].append((type(self).__name__, engine, cycle, options.copy()))
        return original(self, raw, engine, cycle, **options)
    monkeypatch.setattr(_PrefixInferenceAdapter, "predict", predict)
    original_download = app.st.download_button
    def download(label, data, **kwargs):
        captured["downloads"].append(data)
        return original_download(label, data, **kwargs)
    monkeypatch.setattr(app.st, "download_button", download)
    return captured


def select_options(at, scenario, recovery, seed):
    at.selectbox(key="scenario").select(next(k for k, v in app.SCENARIOS.items() if v == scenario)).run()
    if scenario != "clean":
        at.selectbox(key="recovery").select(next(k for k, v in app.RECOVERIES.items() if v == recovery)).run()
    if seed is not None:
        at.selectbox(key="seed").select(seed).run()
    return at


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("scenario,recovery,seed", OPTIONS)
def test_actual_model_controls_independent_replay_plots_and_future_perturbation(saved_models, plots, model_name, scenario, recovery, seed):
    bundle, root, _ = saved_models()
    at = _application(root)
    at.selectbox(key="model").select(LABELS[model_name]).run()
    at.select_slider[0].set_value(5).run()
    select_options(at, scenario, recovery, seed)
    plots["calls"].clear()
    at.run()
    _healthy_app(at)
    assert [cycle for _, _, cycle, _ in plots["calls"]] == [1, 2, 3, 4, 5]
    assert all(options == dict(dataset_split=None, scenario=scenario, recovery=recovery, seed=seed)
               for _, _, _, options in plots["calls"])
    resources = load_replay_resources(root, model_name)
    expected = [resources.adapter.predict(resources.raw_trajectories, 1, cycle, scenario=scenario, recovery=recovery, seed=seed)
                for cycle in range(1, 6)]
    np.testing.assert_array_equal(plots["rul"][-1][0][1], [result.rul_pred for result in expected])
    np.testing.assert_array_equal(plots["sensor"][-1][0][1], expected[-1].chart_prefix.sensor_2)
    assert plots["sensor"][-1][0][0].tolist() == plots["rul"][-1][0][0].tolist() == [1, 2, 3, 4, 5]
    before = {kind: copy.deepcopy(plots[kind][-1]) for kind in ("sensor", "rul")}
    prediction = at.metric[0].value
    bundle.official_test_raw_df.loc[(bundle.official_test_raw_df.engine_id == 1) & (bundle.official_test_raw_df.cycle > 5), "sensor_11"] = np.inf
    bundle.official_test_raw_df.loc[bundle.official_test_raw_df.engine_id == 2, "sensor_2"] = -99999.
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at.run()
    _healthy_app(at)
    assert at.metric[0].value == prediction
    for kind in before:
        for (x, y), (new_x, new_y) in zip(before[kind], plots[kind][-1]):
            np.testing.assert_array_equal(x, new_x)
            np.testing.assert_array_equal(y, new_y)
    if model_name == "median_baseline":
        assert len(set(plots["rul"][-1][0][1])) == 1
        assert any("constant fitted training-target median" in caption.value for caption in at.caption)


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("k", [5, 10])
def test_all_model_final_queue_top_k_ties_csv_and_slider_independence(saved_models, plots, model_name, k):
    bundle, root, _ = saved_models(targets=[50.] * 4)
    source = bundle.official_test_raw_df[lambda df: df.engine_id == 2].sort_values("cycle")
    one = pd.concat([source.iloc[:engine % 9 + 1].assign(engine_id=engine, sensor_11=110., sensor_2=25.)
                     for engine in range(1, 13)], ignore_index=True).assign(dataset_split="official_test")
    other = one.assign(dataset_split="development", sensor_11=np.inf)
    bundle.official_test_raw_df = pd.concat([one, other], ignore_index=True)
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at = _application(root)
    at.selectbox(key="model").select(LABELS[model_name]).run()
    at.selectbox(key="engine").select(("official_test", 1)).run()
    at.select_slider[0].set_value(2).run()
    at.selectbox(key="queue_k").select(k).run()
    plots["calls"].clear()
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    queue = at.dataframe[-1].value
    assert [(e, c) for _, e, c, _ in plots["calls"][-12:]] == [(e, e % 9 + 1) for e in range(1, 13)]
    assert queue.engine_id.tolist() == list(range(1, k + 1))
    assert set(queue.dataset_split) == {"official_test"}
    assert queue["rank"].tolist() == list(range(1, k + 1))
    assert plots["downloads"][-1] == queue.to_csv(index=False).encode("utf-8")
    at.select_slider[0].set_value(1).run()
    assert len(at.dataframe) == 1
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    pd.testing.assert_frame_equal(queue, at.dataframe[-1].value)
    assert plots["sensor"][-1][0][0].tolist() == plots["rul"][-1][0][0].tolist() == [1]


@pytest.mark.parametrize("model_name", MODELS)
def test_one_observation_model_path_no_invalid_slider(saved_models, model_name):
    bundle, root, _ = saved_models()
    bundle.official_test_raw_df = bundle.official_test_raw_df[lambda df: (df.engine_id == 1) & (df.cycle == 1)]
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at = _application(root)
    at.selectbox(key="model").select(LABELS[model_name]).run()
    _healthy_app(at)
    assert not at.select_slider and not at.selectbox(key="queue_k").proto.disabled


def test_switching_models_options_artifacts_and_cache_isolation(saved_models, plots, monkeypatch):
    bundle, root, models = saved_models()
    original, loads = app.load_replay_resources, []
    def load(path, model_name="lstm"):
        loads.append(model_name)
        return original(path, model_name)
    monkeypatch.setattr(app, "load_replay_resources", load)
    at = _application(root)
    for model in MODELS:
        at.selectbox(key="model").select(LABELS[model]).run()
        at.select_slider[0].set_value(5).run()
        _healthy_app(at)
        expected = load_replay_resources(root, model).adapter.predict(bundle.official_test_raw_df, 1, 5).rul_pred
        assert float(at.metric[0].value) == pytest.approx(expected, abs=0.005)
    assert loads == ["lstm", *MODELS]
    models["median_baseline"].value_ += 7.
    joblib.dump(models["median_baseline"], root / "models/baseline.joblib")
    at.run()
    _healthy_app(at)
    assert loads[-1] == "median_baseline" and len(loads) == 5
    assert float(at.metric[0].value) == 57.
    (root / "models/xgboost.json").unlink()
    at.selectbox(key="model").select("Random Forest").run()
    _healthy_app(at)  # unrelated missing model does not invalidate RF resources
    at.selectbox(key="model").select("XGBoost").run()
    assert not at.exception and at.error and not at.metric and not at.dataframe
    at.selectbox(key="model").select("LSTM").run()
    _healthy_app(at)


@pytest.mark.parametrize("model_name", MODELS)
def test_missing_selected_model_never_shows_stale_or_dummy_prediction(saved_models, model_name):
    _, root, _ = saved_models()
    at = _application(root)
    at.selectbox(key="model").select(LABELS[model_name]).run()
    _healthy_app(at)
    name = {"random_forest": "random_forest.joblib", "xgboost": "xgboost.json", "median_baseline": "baseline.joblib"}[model_name]
    (root / "models" / name).unlink()
    at.run()
    assert not at.exception and len(at.error) == 1 and not at.metric and not at.dataframe
    assert name in at.error[0].value
