"""AppTest controls exercise the actual saved LSTM path and Matplotlib line data."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app
from test_lstm_inference import artifacts, raw, single_cpu_thread, CONFIGURATIONS
from test_lstm_replay import replay_artifacts
from predictive_maintenance.inference import LSTMInferenceAdapter
from predictive_maintenance.models.lstm import LSTMConfig
from predictive_maintenance.models.lstm_checkpoint import save_lstm_checkpoint
from predictive_maintenance.replay import load_replay_resources


@pytest.fixture(autouse=True)
def clear_application_resource_cache():
    app._cached_resources.clear()
    yield
    app._cached_resources.clear()


@pytest.fixture
def rendered(monkeypatch):
    plots, calls = {"sensor": [], "rul": []}, []
    for kind, name in [("sensor", "plot_sensor_history"), ("rul", "plot_rul_history")]:
        actual = getattr(app, name)
        def capture(*args, _actual=actual, _kind=kind, **kwargs):
            figure = _actual(*args, **kwargs)
            plots[_kind].append([(line.get_xdata().copy(), line.get_ydata().copy()) for line in figure.axes[0].lines])
            return figure
        monkeypatch.setattr(app, name, capture)
    actual_predict = LSTMInferenceAdapter.predict
    def predict(self, raw, engine, cycle, **options):
        calls.append((engine, cycle, options.copy()))
        return actual_predict(self, raw, engine, cycle, **options)
    monkeypatch.setattr(LSTMInferenceAdapter, "predict", predict)
    return plots, calls


def _application(root):
    return AppTest.from_string(f"from app import run_app\nrun_app({str(root)!r})", default_timeout=30).run()


def _healthy_app(at):
    assert not at.exception and not at.error
    assert len(at.metric) == 2


def test_streamlit_start_slider_real_inference_and_actual_plot_data(replay_artifacts, rendered):
    _, root, _ = replay_artifacts
    plots, calls = rendered
    at = _application(root)
    _healthy_app(at)
    assert at.metric[1].value == "Warning"
    assert at.selectbox(key="recovery").options == ["Training median"]
    assert all(widget.key != "seed" for widget in at.selectbox)
    assert not at.selectbox(key="queue_k").proto.disabled
    calls.clear()
    at.select_slider[0].set_value(5).run()
    _healthy_app(at)
    assert [cycle for _, cycle, _ in calls] == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(plots["sensor"][-1][0][0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(plots["sensor"][-1][0][1], [25.0, 22.0, 23.0, 24.0, 25.0])
    np.testing.assert_array_equal(plots["rul"][-1][0][0], [1, 2, 3, 4, 5])
    assert float(at.metric[0].value) == pytest.approx(plots["rul"][-1][0][1][-1], abs=0.005)
    assert any("display rules" in element.value for element in at.caption)
    assert any("recovered measurements" in element.value for element in at.caption)


@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
def test_control_mapping_and_each_prefix_specific_prediction(replay_artifacts, rendered, scenario, recovery, seed):
    _, root, _ = replay_artifacts
    plots, calls = rendered
    at = _application(root)
    at.select_slider[0].set_value(5).run()
    label = next(label for label, value in app.SCENARIOS.items() if value == scenario)
    at.selectbox(key="scenario").select(label).run()
    if scenario != "clean":
        at.selectbox(key="recovery").select(next(label for label, value in app.RECOVERIES.items() if value == recovery)).run()
    if seed is not None:
        at.selectbox(key="seed").select(seed).run()
    calls.clear()
    at.run()
    _healthy_app(at)
    assert [cycle for _, cycle, _ in calls] == [1, 2, 3, 4, 5]
    assert all(options == {"dataset_split": None, "scenario": scenario, "recovery": recovery, "seed": seed} for _, _, options in calls)
    resources = load_replay_resources(root)
    expected = [resources.adapter.predict(resources.raw_trajectories, 1, cycle, scenario=scenario, recovery=recovery, seed=seed).rul_pred
                for cycle in range(1, 6)]
    np.testing.assert_array_equal(plots["rul"][-1][0][1], expected)


def test_one_cycle_engine_avoids_invalid_slider_and_pads(replay_artifacts, rendered):
    bundle, root, _ = replay_artifacts
    bundle.official_test_raw_df = bundle.official_test_raw_df[lambda frame: (frame.engine_id == 1) & (frame.cycle == 1)]
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    plots, calls = rendered
    at = _application(root)
    _healthy_app(at)
    assert not at.select_slider and [cycle for _, cycle, _ in calls] == [1]
    assert any("one available observation" in element.value for element in at.caption)
    assert "29" in at.json[0].value
    assert plots["sensor"][-1][0][0].tolist() == [1] and plots["rul"][-1][0][0].tolist() == [1]


def test_engine_options_do_not_reuse_stale_predictions_or_plots(replay_artifacts, rendered):
    _, root, _ = replay_artifacts
    plots, calls = rendered
    at = _application(root)
    at.select_slider[0].set_value(5).run()
    before = plots["rul"][-1][0][1].copy()
    calls.clear()
    at.selectbox(key="engine").select((None, 2)).run()
    at.select_slider[0].set_value(3).run()
    _healthy_app(at)
    assert all(engine == 2 for engine, _, _ in calls)
    np.testing.assert_array_equal(plots["sensor"][-1][0][1], [201.0, 202.0, 203.0])
    assert plots["rul"][-1][0][0].tolist() == [1, 2, 3]
    assert not np.array_equal(before[:3], plots["rul"][-1][0][1])


def test_future_raw_artifact_changes_reload_but_preserve_both_plots_at_cutoff(replay_artifacts, rendered):
    bundle, root, _ = replay_artifacts
    plots, calls = rendered
    at = _application(root)
    at.selectbox(key="scenario").select("S3 Short gap").run()
    at.selectbox(key="recovery").select("Causal forward fill").run()
    at.select_slider[0].set_value(5).run()
    before = {kind: [(x.copy(), y.copy()) for x, y in plots[kind][-1]] for kind in plots}
    prediction = at.metric[0].value
    raw = bundle.official_test_raw_df.copy()
    raw.loc[(raw.engine_id == 1) & (raw.cycle > 5), ["sensor_11", "sensor_2"]] = np.inf
    bundle.official_test_raw_df = pd.concat([raw, raw[(raw.engine_id == 1) & (raw.cycle == 45)].assign(cycle=46).set_axis([99999])])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at.run()
    at.select_slider[0].set_value(5).run()
    _healthy_app(at)
    assert at.metric[0].value == prediction
    for kind in plots:
        for (old_x, old_y), (new_x, new_y) in zip(before[kind], plots[kind][-1]):
            np.testing.assert_array_equal(old_x, new_x)
            np.testing.assert_array_equal(old_y, new_y)
            assert max(new_x) <= 5


def test_checkpoint_load_cached_across_slider_and_invalidated_by_changed_weights(replay_artifacts, rendered, monkeypatch):
    bundle, root, model = replay_artifacts
    actual_load, loads = app.load_replay_resources, []
    def load(directory):
        loads.append(directory)
        return actual_load(directory)
    monkeypatch.setattr(app, "load_replay_resources", load)
    at = _application(root)
    at.select_slider[0].set_value(5).run()
    before = float(at.metric[0].value)
    assert len(loads) == 1
    metadata = json.loads((root / "models/lstm_checkpoint_metadata.json").read_text())
    cfg = LSTMConfig(**metadata["config"])
    with torch.no_grad():
        model.head[-1].bias.add_(20)
    save_lstm_checkpoint(model, bundle, root, cfg,
                         {**asdict(cfg), "selected_epoch": metadata["selected_epoch"], "selected_validation": metadata["selected_validation"]},
                         metadata["sequence_input_metadata"])
    at.run()
    _healthy_app(at)
    assert len(loads) == 2 and float(at.metric[0].value) == pytest.approx(before + 20, abs=0.01)


@pytest.mark.parametrize("problem", ["bundle", "checkpoint_metadata", "incompatible", "legacy_raw"])
def test_missing_incompatible_artifacts_show_message_without_prediction_or_exception(replay_artifacts, problem):
    bundle, root, _ = replay_artifacts
    if problem == "bundle":
        (root / "preprocessing/prepared_bundle.joblib").unlink()
    elif problem == "checkpoint_metadata":
        (root / "models/lstm_checkpoint_metadata.json").unlink()
    elif problem == "legacy_raw":
        del bundle.official_test_raw_df
        joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    else:
        path = root / "models/lstm_checkpoint_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["architecture"]["hidden_units"] = 64
        path.write_text(json.dumps(metadata), encoding="utf-8")
    at = _application(root)
    assert not at.exception and len(at.error) == 1 and not at.metric
    assert "Saved LSTM replay is unavailable" in at.error[0].value
    assert any("No automatic training or download" in element.value for element in at.info)


@pytest.mark.parametrize("model", ["Median", "Random Forest", "XGBoost"])
def test_other_model_controls_report_integration_status_without_dummy_outputs(replay_artifacts, model):
    _, root, _ = replay_artifacts
    at = _application(root)
    at.selectbox(key="model").select(model).run()
    assert not at.exception and not at.error and not at.metric
    assert any(f"Saved inference for {model} is not connected" in element.value for element in at.info)
    assert at.selectbox(key="queue_k").proto.disabled


def test_namespace_engine_selector_keeps_identical_ids_isolated(replay_artifacts, rendered):
    bundle, root, _ = replay_artifacts
    one = bundle.official_test_raw_df[lambda frame: frame.engine_id == 1].copy()
    other = one.copy()
    one["dataset_split"] = "official_test"
    other["dataset_split"] = "development"
    other[["sensor_11", "sensor_2"]] = 9999.0
    bundle.official_test_raw_df = pd.concat([one, other])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    plots, calls = rendered
    at = _application(root)
    at.selectbox(key="engine").select(("official_test", 1)).run()
    calls.clear()
    at.select_slider[0].set_value(3).run()
    _healthy_app(at)
    assert all(options["dataset_split"] == "official_test" for _, _, options in calls)
    np.testing.assert_array_equal(plots["sensor"][-1][0][1], [25.0, 22.0, 23.0])


def test_returning_to_clean_clears_random_seed_and_forward_fill_options(replay_artifacts, rendered):
    _, root, _ = replay_artifacts
    _, calls = rendered
    at = _application(root)
    at.select_slider[0].set_value(5).run()
    at.selectbox(key="scenario").select("S2 Random 20%").run()
    at.selectbox(key="recovery").select("Causal forward fill").run()
    at.selectbox(key="seed").select(27).run()
    calls.clear()
    at.selectbox(key="scenario").select("S0 Clean").run()
    _healthy_app(at)
    assert at.selectbox(key="recovery").value == "Training median"
    assert all(widget.key != "seed" for widget in at.selectbox)
    assert all(options["scenario"] == "clean" and options["recovery"] == "median" and options["seed"] is None
               for _, _, options in calls)


def test_removed_artifact_cannot_leave_cached_prediction_on_screen(replay_artifacts):
    _, root, _ = replay_artifacts
    at = _application(root)
    _healthy_app(at)
    (root / "models/lstm_checkpoint_metadata.json").unlink()
    at.run()
    assert not at.exception and len(at.error) == 1 and not at.metric
    assert "lstm_checkpoint_metadata.json" in at.error[0].value
