"""Real saved-adapter replay on reused synthetic fixture patterns."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_lstm_inference import artifacts, raw, single_cpu_thread, CONFIGURATIONS
from predictive_maintenance.dashboard import plot_sensor_history, plot_rul_history
from predictive_maintenance.replay import (
    REPLAY_ARTIFACTS, artifact_fingerprint, available_engine_cycles, build_lstm_replay, load_replay_resources,
)


@pytest.fixture
def replay_artifacts(artifacts, raw):
    bundle, root, original_model = artifacts()
    bundle.official_test_raw_df = raw.copy(deep=True)
    bundle.official_test_df = raw.fillna(777777.0)  # Compatibility frame must never supply raw replay input.
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    return bundle, root, original_model


@pytest.mark.parametrize("cutoff", [1, 5, 29, 30, 31])
def test_replay_early_cycles_padding_and_each_independent_cutoff(replay_artifacts, cutoff):
    _, root, _ = replay_artifacts
    resources = load_replay_resources(root)
    actual_predict = resources.adapter.predict
    calls = []
    def predict(raw, engine, cycle, **options):
        result = actual_predict(raw, engine, cycle, **options)
        calls.append((cycle, result))
        return result
    resources.adapter.predict = predict
    result, history = build_lstm_replay(resources, (None, 1), cutoff, scenario="short_gap", recovery="forward_fill")
    assert [cycle for cycle, _ in calls] == list(range(1, cutoff + 1))
    assert history.cycle.tolist() == list(range(1, cutoff + 1))
    assert history.rul_pred.tolist() == [result.rul_pred for _, result in calls]
    for cycle, point in calls:
        assert point.observed_history_length == cycle and point.padding_cycles == max(0, 30 - cycle)
        assert point.sensor_mask.sensor_11.sum() == min(5, cycle)
    assert result.cycle == cutoff and result.rul_pred == history.rul_pred.iloc[-1]
    if cutoff >= 2:
        assert calls[1][1].chart_prefix.sensor_11.tolist() == [110.0, 110.0]


@pytest.mark.parametrize("scenario,recovery,seed", CONFIGURATIONS)
def test_replay_future_perturbation_preserves_actual_sensor_and_rul_plot_lines(replay_artifacts, scenario, recovery, seed):
    bundle, root, _ = replay_artifacts
    options = {"scenario": scenario, "recovery": recovery, "seed": seed}
    resources = load_replay_resources(root)
    before_result, before_history = build_lstm_replay(resources, (None, 1), 5, **options)
    raw = bundle.official_test_raw_df.copy(deep=True)
    raw.loc[(raw.engine_id == 1) & (raw.cycle > 5), ["sensor_11", "sensor_2"]] = np.inf
    raw.loc[raw.engine_id == 2, ["sensor_11", "sensor_2"]] = -999999.0
    raw = pd.concat([raw, raw[(raw.engine_id == 1) & (raw.cycle == 45)].assign(cycle=46).set_axis([99999])])
    bundle.official_test_raw_df = raw
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    changed = load_replay_resources(root)
    after_result, after_history = build_lstm_replay(changed, (None, 1), 5, **options)
    assert before_result.rul_pred == after_result.rul_pred
    pd.testing.assert_frame_equal(before_result.chart_prefix, after_result.chart_prefix)
    pd.testing.assert_frame_equal(before_history, after_history)
    figures = [plot_sensor_history(before_result.chart_prefix, "sensor_11"), plot_sensor_history(after_result.chart_prefix, "sensor_11"),
               plot_rul_history(before_history), plot_rul_history(after_history)]
    try:
        for left, right in [(figures[0], figures[1]), (figures[2], figures[3])]:
            np.testing.assert_array_equal(left.axes[0].lines[0].get_xdata(), right.axes[0].lines[0].get_xdata())
            np.testing.assert_array_equal(left.axes[0].lines[0].get_ydata(), right.axes[0].lines[0].get_ydata())
            assert max(left.axes[0].lines[0].get_xdata()) == 5
    finally:
        for figure in figures:
            figure.clear()


@pytest.mark.parametrize("name", REPLAY_ARTIFACTS)
def test_content_fingerprint_detects_changes_in_every_consumed_artifact(replay_artifacts, name):
    _, root, _ = replay_artifacts
    before = artifact_fingerprint(root)
    path = root / name
    with path.open("ab") as stream:
        stream.write(b" ")
    after = artifact_fingerprint(root)
    assert before != after
    assert [key for (key, old), (_, new) in zip(before, after) if old != new] == [name]


def test_composite_choices_and_replay_are_namespace_isolated(replay_artifacts):
    bundle, root, _ = replay_artifacts
    one = bundle.official_test_raw_df[lambda frame: frame.engine_id == 1].copy()
    other = one.copy()
    one["dataset_split"] = "official_test"
    other["dataset_split"] = "development"
    other[["sensor_11", "sensor_2"]] = 9999.0
    bundle.official_test_raw_df = pd.concat([other, one])
    joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
    resources = load_replay_resources(root)
    assert set(resources.engine_cycles) == {("official_test", 1), ("development", 1)}
    result, history = build_lstm_replay(resources, ("official_test", 1), 3)
    assert result.chart_prefix.dataset_split.unique().tolist() == ["official_test"]
    assert result.chart_prefix.sensor_11.tolist() == [110.0, 20.0, 110.0]
    assert history.cycle.tolist() == [1, 2, 3]


@pytest.mark.parametrize("problem", ["missing", "legacy_raw", "incompatible"])
def test_replay_loading_rejects_missing_legacy_or_incompatible_artifacts(replay_artifacts, problem):
    bundle, root, _ = replay_artifacts
    if problem == "missing":
        (root / "preprocessing/prepared_bundle.joblib").unlink()
        error, message = FileNotFoundError, "prepared_bundle"
    elif problem == "legacy_raw":
        del bundle.official_test_raw_df
        joblib.dump(bundle, root / "preprocessing/prepared_bundle.joblib")
        error, message = ValueError, "Regenerate"
    else:
        path = root / "models/lstm_checkpoint_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["architecture"]["hidden_units"] = 64
        path.write_text(json.dumps(metadata), encoding="utf-8")
        error, message = ValueError, "architecture"
    with pytest.raises(error, match=message):
        load_replay_resources(root)


def test_metadata_choices_reject_missing_identity_and_duplicate_cycles(raw):
    invalid = raw.copy()
    invalid["dataset_split"] = None
    with pytest.raises(ValueError, match="non-empty strings"):
        available_engine_cycles(invalid)
    with pytest.raises(ValueError, match="Duplicate"):
        available_engine_cycles(pd.concat([raw, raw.iloc[[0]]]))
