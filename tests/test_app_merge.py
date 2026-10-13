"""Merge regressions: actual saved LSTM queues and executable manifest parity."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import app
from test_lstm_inference import artifacts, raw, single_cpu_thread, CONFIGURATIONS
from test_lstm_replay import replay_artifacts
from test_streamlit_lstm import rendered, clear_application_resource_cache, _application, _healthy_app
from test_official_test_evaluation import observed_run
from predictive_maintenance import experiments
from predictive_maintenance.replay import load_replay_resources
from run_experiments import build_evaluation_manifest


def test_manifest_matches_executable_recoveries_and_separate_baseline(observed_run):
    bundle, *_ = observed_run
    metrics = experiments.run_experiments(bundle, expected_engine_ids=(1, 2, 3))
    plan = build_evaluation_manifest()
    normalize = lambda seed: None if seed == "" or pd.isna(seed) else int(seed)
    planned = {(row["model"], row["scenario"], normalize(row["seed"]), row["recovery"]) for row in plan[1:]}
    executed = {(row.model, row.scenario, normalize(row.seed), row.recovery) for row in metrics.itertuples()}
    assert planned == executed and len(planned) == 45
    assert len(plan) == 46 and plan[0]["model"] == "median_baseline"


@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
def test_lstm_queue_uses_raw_final_prefix_for_every_engine(replay_artifacts, scenario, recovery, seed):
    bundle, root, _ = replay_artifacts
    late = bundle.official_test_raw_df[lambda frame: frame.engine_id == 2].assign(
        engine_id=3, cycle=lambda frame: frame.cycle + 50,
    )
    bundle.official_test_raw_df = pd.concat([bundle.official_test_raw_df, late], ignore_index=True)
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    resources = load_replay_resources(root)
    original_raw = resources.raw_trajectories.copy(deep=True)
    actual = resources.adapter.predict
    calls = []
    def predict(raw, engine, cycle, **options):
        calls.append((engine, cycle, options))
        return actual(raw, engine, cycle, **options)
    resources.adapter.predict = predict
    options = {"scenario": scenario, "recovery": recovery, "seed": seed}
    queue = app.build_lstm_queue(resources, None, 5, **options)
    assert [(engine, cycle) for engine, cycle, _ in calls] == [(1, 45), (2, 9), (3, 59)]
    assert all(opts == {"dataset_split": None, **options} for _, _, opts in calls)
    assert len(queue) == 3 and set(queue.engine_id) == {1, 2, 3}
    assert queue.set_index("engine_id").observed_cycle.to_dict() == {1: 45, 2: 9, 3: 59}
    for row in queue.itertuples():
        assert row.rul_pred == actual(
            resources.raw_trajectories, row.engine_id, row.observed_cycle, **options,
        ).rul_pred
    assert queue.rul_pred.is_monotonic_increasing
    pd.testing.assert_frame_equal(resources.raw_trajectories, original_raw)


def test_queue_namespace_and_final_snapshot_endpoints(replay_artifacts):
    bundle, root, _ = replay_artifacts
    one = bundle.official_test_raw_df.copy()
    one["dataset_split"] = "official_test"
    other = one.assign(dataset_split="development", sensor_11=np.inf, sensor_2=np.inf)
    later = one[one.engine_id == 2].assign(engine_id=3, cycle=lambda frame: frame.cycle + 50)
    bundle.official_test_raw_df = pd.concat([one, other, later])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    queue = app.build_lstm_queue(load_replay_resources(root), "official_test", 10)
    assert set(queue.engine_id) == {1, 2, 3}
    assert queue.dataset_split.tolist() == ["official_test"] * 3
    assert queue.set_index("engine_id").observed_cycle.to_dict() == {1: 45, 2: 9, 3: 59}


@pytest.mark.parametrize("bias", [np.nan, np.inf, -np.inf])
def test_queue_rejects_actual_saved_nonfinite_weights_before_clamping(artifacts, raw, bias):
    bundle, root, _ = artifacts(bias=bias)
    bundle.official_test_raw_df = raw
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    with pytest.raises(ValueError, match="finite.*before negative clamping"):
        app.build_lstm_queue(load_replay_resources(root), None, 5)


@pytest.mark.parametrize("k", [5, 10])
def test_app_queue_actual_predictions_download_and_no_stale_results(replay_artifacts, rendered, monkeypatch, k):
    bundle, root, _ = replay_artifacts
    # More than ten engines with different final cycles prove K limits and snapshot scoring.
    original = bundle.official_test_raw_df[lambda frame: frame.engine_id == 2].sort_values("cycle")
    bundle.official_test_raw_df = pd.concat([
        original.iloc[:engine % 9 + 1].assign(
            engine_id=engine, sensor_11=lambda frame: frame.sensor_11 + engine,
        ) for engine in range(1, 13)
    ], ignore_index=True)
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    downloads = []
    actual_download = app.st.download_button
    def download(label, data, **kwargs):
        downloads.append(data)
        return actual_download(label, data, **kwargs)
    monkeypatch.setattr(app.st, "download_button", download)
    at = _application(root)
    at.select_slider[0].set_value(2).run()
    at.selectbox(key="queue_k").select(k).run()
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    queue = at.dataframe[-1].value
    resources = load_replay_resources(root)
    expected = app.build_lstm_queue(resources, None, k)
    pd.testing.assert_frame_equal(queue, expected)
    assert len(queue) == k and queue["rank"].tolist() == list(range(1, k + 1))
    assert downloads[-1] == expected.to_csv(index=False).encode("utf-8")
    endpoints = resources.raw_trajectories.groupby("engine_id").cycle.max().to_dict()
    assert all(row.observed_cycle == endpoints[row.engine_id] for row in queue.itertuples())
    # Slider changes only bounded replay/charts, never queue values or membership.
    at.select_slider[0].set_value(1).run()
    assert len(at.dataframe) == 1
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    pd.testing.assert_frame_equal(at.dataframe[-1].value, expected)
    plots, _ = rendered
    assert plots["sensor"][-1][0][0].tolist() == [1]
    assert plots["rul"][-1][0][0].tolist() == [1]
    # Options and artifacts recompute rather than retain previous queue results.
    at.selectbox(key="scenario").select("S3 Short gap").run()
    _healthy_app(at)
    assert len(at.dataframe) == 1
    at.button(key="calculate_queue").click().run()
    expected_gap = app.build_lstm_queue(resources, None, k, scenario="short_gap")
    pd.testing.assert_frame_equal(at.dataframe[-1].value, expected_gap)
    bundle.official_test_raw_df.loc[bundle.official_test_raw_df.cycle > 1, "sensor_2"] += 1.0
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at.run()
    assert len(at.dataframe) == 1
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    expected_changed = app.build_lstm_queue(load_replay_resources(root), None, k, scenario="short_gap")
    pd.testing.assert_frame_equal(at.dataframe[-1].value, expected_changed)
    assert downloads[-1] == expected_changed.to_csv(index=False).encode("utf-8")
    at.selectbox(key="model").select("Random Forest").run()
    assert not at.exception and at.error and not at.metric and not at.dataframe
    assert not at.button and not at.selectbox(key="queue_k").proto.disabled
    assert "schema.json" in at.error[0].value

@pytest.mark.parametrize("k", [5, 10])
def test_final_queue_predicts_all_engines_and_breaks_actual_model_ties_by_engine_id(replay_artifacts, k):
    bundle, root, _ = replay_artifacts
    source = bundle.official_test_raw_df[lambda frame: frame.engine_id == 2].sort_values("cycle")
    bundle.official_test_raw_df = pd.concat([
        source.iloc[:engine % 9 + 1].assign(engine_id=engine, sensor_11=110.0, sensor_2=25.0)
        for engine in range(1, 13)
    ], ignore_index=True)
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    resources = load_replay_resources(root)
    # Reverse insertion order to exercise tie-breaking rather than input ordering.
    resources = replace(resources, engine_cycles=dict(reversed(list(resources.engine_cycles.items()))))
    actual, calls = resources.adapter.predict, []
    def predict(raw, engine, cycle, **options):
        calls.append((engine, cycle))
        return actual(raw, engine, cycle, **options)
    resources.adapter.predict = predict
    queue = app.build_lstm_queue(resources, None, k)
    endpoints = resources.raw_trajectories.groupby("engine_id").cycle.max().to_dict()
    assert calls == [(engine, endpoints[engine]) for engine in range(12, 0, -1)]
    assert len(calls) == 12 and queue.rul_pred.nunique() == 1
    assert queue.engine_id.tolist() == list(range(1, k + 1))
    assert queue["rank"].tolist() == list(range(1, k + 1))


def test_final_queue_scenario_recovery_and_seed_change_actual_predictions(replay_artifacts):
    _, root, _ = replay_artifacts
    resources = load_replay_resources(root)
    options = [
        {"scenario": "clean"},
        {"scenario": "short_gap", "recovery": "median"},
        {"scenario": "short_gap", "recovery": "forward_fill"},
        *[{"scenario": "random_20", "recovery": "forward_fill", "seed": seed} for seed in (7, 17, 27)],
    ]
    values = [app.build_lstm_queue(resources, None, 5, **option).set_index("engine_id").rul_pred
              for option in options]
    assert not values[0].equals(values[1])
    assert not values[1].equals(values[2])
    assert len({tuple(value.loc[[1, 2]]) for value in values[3:]}) > 1


def test_app_queue_namespace_does_not_follow_replay_slider(replay_artifacts, rendered):
    bundle, root, _ = replay_artifacts
    one = bundle.official_test_raw_df.assign(dataset_split="official_test")
    other = one.assign(dataset_split="development", sensor_11=np.inf, sensor_2=np.inf)
    late = one[one.engine_id == 2].assign(engine_id=3, cycle=lambda frame: frame.cycle + 50)
    bundle.official_test_raw_df = pd.concat([one, other, late])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    at = _application(root)
    at.selectbox(key="engine").select(("official_test", 1)).run()
    at.select_slider[0].set_value(5).run()
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    before = at.dataframe[-1].value.copy()
    assert set(before.engine_id) == {1, 2, 3}
    assert before.dataset_split.tolist() == ["official_test"] * 3
    assert before.set_index("engine_id").observed_cycle.to_dict() == {1: 45, 2: 9, 3: 59}
    at.select_slider[0].set_value(1).run()
    at.button(key="calculate_queue").click().run()
    _healthy_app(at)
    pd.testing.assert_frame_equal(before, at.dataframe[-1].value)


def test_queue_rejects_absent_namespace(replay_artifacts):
    _, root, _ = replay_artifacts
    with pytest.raises(ValueError, match="No engines in the selected namespace"):
        app.build_lstm_queue(load_replay_resources(root), "absent", 5)
