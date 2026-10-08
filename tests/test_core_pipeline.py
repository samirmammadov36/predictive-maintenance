from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN, RUL_TARGET_COLUMN, RUL_TRUTH_COLUMN
from predictive_maintenance.data import add_training_rul, split_development_engines, validation_prefixes
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from predictive_maintenance.sequences import (
    build_prefix_window, build_training_windows, fit_sequence_scaler, left_padded_window, scale_sequence_frame,
)


def _frame() -> pd.DataFrame:
    rows = []
    for engine in (1, 2, 3, 4, 5):
        for cycle in range(1, 7):
            rows.append(
                {
                    ID_COLUMN: engine,
                    CYCLE_COLUMN: cycle,
                    "sensor_1": float(engine * 10 + cycle),
                    "sensor_2": float(cycle * 2),
                }
            )
    return pd.DataFrame(rows)


def test_rul_generation():
    df = add_training_rul(_frame(), cap=3)
    e1 = df[df.engine_id == 1]
    assert e1[RUL_TRUTH_COLUMN].tolist() == [5, 4, 3, 2, 1, 0]
    assert e1[RUL_TARGET_COLUMN].tolist() == [3, 3, 3, 2, 1, 0]


def test_engine_split_has_no_overlap_and_is_reproducible():
    df = _frame()
    a_train, a_val = split_development_engines(df, validation_fraction=0.4, seed=42)
    b_train, b_val = split_development_engines(df, validation_fraction=0.4, seed=42)
    assert a_train == b_train and a_val == b_val
    assert not (set(a_train) & set(a_val))
    assert len(a_train) + len(a_val) == df.engine_id.nunique()


def test_validation_prefix_is_before_failure():
    df = _frame()
    prefixes = validation_prefixes(df, [1], prefix_fraction=0.70)
    assert prefixes[1] == 4
    assert prefixes[1] < df[df.engine_id == 1].cycle.max()


def test_causal_forward_fill_does_not_cross_engines():
    df = _frame()
    medians = TrainingMedians.fit(df[df.engine_id <= 3], ["sensor_1", "sensor_2"])
    corrupted = df.copy()
    corrupted.loc[(corrupted.engine_id == 1) & (corrupted.cycle == 3), "sensor_1"] = np.nan
    corrupted.loc[(corrupted.engine_id == 2) & (corrupted.cycle == 1), "sensor_1"] = np.nan
    filled = medians.causal_forward_fill(corrupted, ["sensor_1", "sensor_2"])
    assert filled.loc[(filled.engine_id == 1) & (filled.cycle == 3), "sensor_1"].iloc[0] == 12.0
    # Engine 2 cycle 1 has no past value, so it must use the training median, not engine 1's last value.
    assert filled.loc[(filled.engine_id == 2) & (filled.cycle == 1), "sensor_1"].iloc[0] == medians.values["sensor_1"]


def test_tabular_features_are_causal():
    df = _frame()[lambda x: x.engine_id == 1]
    engineered, cols = build_tabular_features(df, ["sensor_1"], rolling_window=3)
    # cycle 2: mean uses cycles 1-2 only
    row2 = engineered[engineered.cycle == 2].iloc[0]
    assert row2["sensor_1_mean_3"] == np.mean([11.0, 12.0])
    # 3-cycle difference at cycle 2 uses earliest available cycle 1
    assert row2["sensor_1_diff_3"] == 1.0
    assert "sensor_1_mean_3" in cols


def test_nested_masks_are_deterministic_and_nested():
    prefix = _frame()[lambda x: x.engine_id == 1].copy()
    low1, high1 = nested_random_masks(prefix, ["sensor_1", "sensor_2"], seed=7)
    low2, high2 = nested_random_masks(prefix, ["sensor_1", "sensor_2"], seed=7)
    assert low1.equals(low2) and high1.equals(high2)
    assert ((low1 & ~high1).to_numpy().sum()) == 0


def test_short_gap_masks_tail_only():
    prefix = _frame()[lambda x: x.engine_id == 1].copy()
    mask, sensor = short_gap_mask(prefix, ["sensor_1", "sensor_2"], preferred_sensor="sensor_2", gap_cycles=2)
    assert sensor == "sensor_2"
    assert mask["sensor_2"].sum() == 2
    corrupted = apply_sensor_mask(prefix, mask, ["sensor_1", "sensor_2"])
    assert corrupted.tail(2)["sensor_2"].isna().all()
    assert corrupted.head(4)["sensor_2"].notna().all()


def test_sequence_windows_never_mix_engines_and_pad_left():
    df = add_training_rul(_frame(), cap=5).sample(frac=1, random_state=42)
    X, y = build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN, window=4)
    assert X.shape == (len(df), 4, 2)
    assert X.dtype == y.dtype == np.float32
    cycle_windows = [[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
    for engine in range(1, 6):
        for observation, cycles in enumerate(cycle_windows):
            index = (engine - 1) * 6 + observation
            expected = [[engine * 10 + cycle, cycle * 2] for cycle in cycles]
            np.testing.assert_array_equal(X[index], np.array(expected, dtype=np.float32))
            assert y[index] == 5 - observation


def _sequence_frame(length: int, engine_id: int = 1, offset: int = 0) -> pd.DataFrame:
    cycles = np.arange(1, length + 1)
    return pd.DataFrame({
        ID_COLUMN: engine_id,
        CYCLE_COLUMN: cycles,
        "sensor_1": (offset + cycles).astype(float),
        "sensor_2": (offset + 100 + cycles).astype(float),
        RUL_TARGET_COLUMN: offset + 1000 + cycles,
    })


@pytest.mark.parametrize("length,expected_cycles", [
    (1, [1] * 30),
    (5, [1] * 25 + [1, 2, 3, 4, 5]),
    (29, [1] + list(range(1, 30))),
    (30, list(range(1, 31))),
    (31, list(range(2, 32))),
])
def test_sequence_history_lengths_and_final_window(length, expected_cycles):
    df = _sequence_frame(length)
    expected = np.array([[cycle, 100 + cycle] for cycle in expected_cycles], dtype=np.float32)
    X, y = build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN)
    prefix = build_prefix_window(df, ["sensor_1", "sensor_2"])
    assert X.shape == (length, 30, 2)
    assert prefix.shape == (30, 2)
    assert X.dtype == y.dtype == prefix.dtype == np.float32
    np.testing.assert_array_equal(X[-1], expected)
    np.testing.assert_array_equal(prefix, expected)
    np.testing.assert_array_equal(y, np.arange(1001, 1001 + length, dtype=np.float32))


def test_sequence_composite_identity_keeps_same_engine_id_separate():
    development = _sequence_frame(3, offset=10).assign(dataset_split="development")
    official_test = _sequence_frame(3, offset=100).assign(dataset_split="official_test")
    df = pd.concat([official_test, development], ignore_index=True).sample(frac=1, random_state=7)
    X, y = build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN, window=3)
    cycle_windows = [[1, 1, 1], [1, 1, 2], [1, 2, 3]]
    assert X.shape == (6, 3, 2)
    for identity_index, offset in enumerate((10, 100)):
        for observation, cycles in enumerate(cycle_windows):
            index = identity_index * 3 + observation
            expected = [[offset + cycle, offset + 100 + cycle] for cycle in cycles]
            np.testing.assert_array_equal(X[index], np.array(expected, dtype=np.float32))
            assert y[index] == offset + 1001 + observation
    np.testing.assert_array_equal(
        build_prefix_window(official_test, ["sensor_1", "sensor_2"], window=3),
        np.array([[101, 201], [102, 202], [103, 203]], dtype=np.float32),
    )


@pytest.mark.parametrize("composite", [False, True])
def test_sequence_prefix_rejects_mixed_engine_identities(composite):
    first = _sequence_frame(2)
    second = _sequence_frame(2, engine_id=1 if composite else 2, offset=100)
    if composite:
        first = first.assign(dataset_split="development")
        second = second.assign(dataset_split="official_test")
    df = pd.concat([first, second], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one engine identity"):
        build_prefix_window(df, ["sensor_1", "sensor_2"])


def test_sequence_unsorted_rows_are_chronological_and_do_not_mutate_input():
    df = _sequence_frame(4).iloc[[3, 1, 0, 2]]
    original = df.copy(deep=True)
    X, y = build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN, window=3)
    expected = np.array([
        [[1, 101], [1, 101], [1, 101]],
        [[1, 101], [1, 101], [2, 102]],
        [[1, 101], [2, 102], [3, 103]],
        [[2, 102], [3, 103], [4, 104]],
    ], dtype=np.float32)
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(y, np.array([1001, 1002, 1003, 1004], dtype=np.float32))
    np.testing.assert_array_equal(build_prefix_window(df, ["sensor_1", "sensor_2"], window=3), expected[-1])
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("physical_order", [["sensor_1", "sensor_2"], ["sensor_2", "sensor_1"]])
def test_sequence_explicit_sensor_order_overrides_dataframe_order(physical_order):
    df = _sequence_frame(3)[[ID_COLUMN, CYCLE_COLUMN, *physical_order, RUL_TARGET_COLUMN]]
    X, y = build_training_windows(df, ["sensor_2", "sensor_1"], RUL_TARGET_COLUMN, window=3)
    expected = np.array([[101, 1], [102, 2], [103, 3]], dtype=np.float32)
    np.testing.assert_array_equal(X[-1], expected)
    np.testing.assert_array_equal(build_prefix_window(df, ["sensor_2", "sensor_1"], window=3), expected)
    assert y[-1] == 1003


@pytest.mark.parametrize("column", [ID_COLUMN, "dataset_split"])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_sequence_rejects_missing_engine_identity(column, missing):
    df = _sequence_frame(2).assign(dataset_split="development")
    df[column] = df[column].astype(object)
    df.loc[0, column] = missing
    with pytest.raises(ValueError, match=f"non-missing {column}"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN)
    with pytest.raises(ValueError, match=f"non-missing {column}"):
        build_prefix_window(df, ["sensor_1"])


def test_sequence_rejects_missing_engine_id_column():
    df = _sequence_frame(2).drop(columns=ID_COLUMN)
    with pytest.raises(ValueError, match="non-missing engine_id"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN)
    with pytest.raises(ValueError, match="non-missing engine_id"):
        build_prefix_window(df, ["sensor_1"])


@pytest.mark.parametrize("composite", [False, True])
def test_sequence_rejects_duplicate_cycles_within_identity(composite):
    df = _sequence_frame(3)
    if composite:
        df = df.assign(dataset_split="development")
    df.loc[2, CYCLE_COLUMN] = 2
    with pytest.raises(ValueError, match="Duplicate cycle"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN)
    with pytest.raises(ValueError, match="Duplicate cycle"):
        build_prefix_window(df, ["sensor_1"])


@pytest.mark.parametrize("invalid", [0, -1, 1.5, np.nan, np.inf, -np.inf, True, "invalid", 1 + 2j])
def test_sequence_rejects_invalid_cycles(invalid):
    df = _sequence_frame(2)
    df[CYCLE_COLUMN] = df[CYCLE_COLUMN].astype(object)
    df.loc[0, CYCLE_COLUMN] = invalid
    with pytest.raises(ValueError, match="cycle values must be finite positive integers"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN)
    with pytest.raises(ValueError, match="cycle values must be finite positive integers"):
        build_prefix_window(df, ["sensor_1"])


@pytest.mark.parametrize("with_columns", [False, True])
def test_sequence_rejects_empty_input(with_columns):
    df = _sequence_frame(1).iloc[:0] if with_columns else pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        build_prefix_window(df, ["sensor_1"])
    with pytest.raises(ValueError, match="empty"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN)


@pytest.mark.parametrize("window", [0, -1, 1.5, 30.0, True, np.bool_(True), "30", None, np.nan, np.inf])
def test_sequence_rejects_invalid_window_sizes(window):
    df = _sequence_frame(2)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN, window=window)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        build_prefix_window(df, ["sensor_1"], window=window)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        left_padded_window(np.array([[1.0], [2.0]], dtype=np.float32), 1, window)


def test_sequence_accepts_numpy_integer_window_and_integral_cycles():
    df = _sequence_frame(2)
    df[CYCLE_COLUMN] = df[CYCLE_COLUMN].astype(float)
    expected = np.array([[1], [1], [2]], dtype=np.float32)
    X, _ = build_training_windows(df, ["sensor_1"], RUL_TARGET_COLUMN, window=np.int64(3))
    np.testing.assert_array_equal(X[-1], expected)
    np.testing.assert_array_equal(build_prefix_window(df, ["sensor_1"], window=np.int64(3)), expected)
    np.testing.assert_array_equal(left_padded_window(np.array([[1], [2]], dtype=np.float32), np.int64(1), np.int64(3)), expected)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_sequence_rejects_nonfinite_sensor_values(invalid):
    df = _sequence_frame(2)
    df.loc[0, "sensor_1"] = invalid
    with pytest.raises(ValueError, match="Sensor values must be finite"):
        build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN)
    with pytest.raises(ValueError, match="Sensor values must be finite"):
        build_prefix_window(df, ["sensor_1", "sensor_2"])
    with pytest.raises(ValueError, match="Sensor values must be finite"):
        left_padded_window(df[["sensor_1", "sensor_2"]].to_numpy(), 1, 30)


@pytest.mark.parametrize("cycle", [5, 30, 35])
def test_sequence_causality_and_explicit_prefix_truncation(cycle):
    df = _sequence_frame(45)
    changed = df.copy()
    changed.loc[changed.cycle > cycle, ["sensor_1", "sensor_2"]] += 10000
    sensors = ["sensor_1", "sensor_2"]
    original_X, original_y = build_training_windows(df, sensors, RUL_TARGET_COLUMN)
    changed_X, changed_y = build_training_windows(changed, sensors, RUL_TARGET_COLUMN)
    np.testing.assert_array_equal(original_X[:cycle], changed_X[:cycle])
    assert original_y[cycle - 1] == changed_y[cycle - 1] == 1000 + cycle
    prefix = changed[changed.cycle <= cycle]
    np.testing.assert_array_equal(build_prefix_window(prefix, sensors), original_X[cycle - 1])
    np.testing.assert_array_equal(original_X[cycle - 1, -1], np.array([cycle, 100 + cycle], dtype=np.float32))
    # Without explicit truncation, the supplied frame ends at cycle 45.
    np.testing.assert_array_equal(
        build_prefix_window(changed, sensors)[-1],
        np.array([10045, 10145], dtype=np.float32),
    )


def test_sequence_scaler_uses_training_statistics_and_never_refits(monkeypatch):
    train = pd.DataFrame({"engine_id": [2, 9, 2], "cycle": [1, 1, 2],
                          "sensor_1": [1.0, 3.0, 5.0], "sensor_2": [10.0, 14.0, 18.0]})
    columns = ["sensor_2", "sensor_1"]
    scaler = fit_sequence_scaler(train, columns)
    np.testing.assert_allclose(scaler.mean_, [14.0, 3.0])
    np.testing.assert_allclose(scaler.var_, [32.0 / 3.0, 8.0 / 3.0])
    assert scaler.n_samples_seen_ == 3
    assert list(scaler.feature_names_in_) == columns
    before = {name: getattr(scaler, name).copy() for name in ("mean_", "var_", "scale_", "feature_names_in_")}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Transform must never refit the scaler")

    monkeypatch.setattr(scaler, "fit", forbidden_fit)
    for extreme in (1e9, -1e12):
        held_out = train.assign(sensor_1=extreme, sensor_2=-extreme, dataset_split="held_out", rul_true=999)
        original = held_out.copy(deep=True)
        transformed = scale_sequence_frame(held_out, columns, scaler)
        np.testing.assert_allclose(transformed[columns], np.tile(
            [(-extreme - 14.0) / np.sqrt(32.0 / 3.0), (extreme - 3.0) / np.sqrt(8.0 / 3.0)], (3, 1)))
        pd.testing.assert_frame_equal(transformed.drop(columns=columns), original.drop(columns=columns))
        pd.testing.assert_frame_equal(held_out, original)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(scaler, name), values)
    assert scaler.n_samples_seen_ == 3


@pytest.mark.parametrize("columns,message", [
    ([], "non-empty"),
    (["sensor_1", "sensor_1"], "duplicate"),
    (["sensor_3"], "Missing sensor"),
    (["engine_id"], "valid FD001 sensor"),
    (["setting_1"], "valid FD001 sensor"),
    (["sensor_22"], "valid FD001 sensor"),
    ([None], "valid FD001 sensor"),
    ([1], "valid FD001 sensor"),
    ([""], "valid FD001 sensor"),
    ("sensor_1", "non-empty"),
])
def test_sequence_scaling_rejects_invalid_sensor_columns(columns, message):
    df = _sequence_frame(2)
    scaler = fit_sequence_scaler(df, ["sensor_1", "sensor_2"])
    with pytest.raises(ValueError, match=message):
        fit_sequence_scaler(df, columns)
    with pytest.raises(ValueError, match=message):
        scale_sequence_frame(df, columns, scaler)


def test_sequence_scaling_rejects_reordered_sensors():
    df = _sequence_frame(2)
    scaler = fit_sequence_scaler(df, ["sensor_2", "sensor_1"])
    with pytest.raises(ValueError, match="Sensor order"):
        scale_sequence_frame(df, ["sensor_1", "sensor_2"], scaler)


def test_sequence_scaling_rejects_ambiguous_dataframe_columns():
    df = _sequence_frame(2)
    scaler = fit_sequence_scaler(df, ["sensor_1"])
    duplicated = pd.concat([df, df[["sensor_1"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate DataFrame labels"):
        fit_sequence_scaler(duplicated, ["sensor_1"])
    with pytest.raises(ValueError, match="duplicate DataFrame labels"):
        scale_sequence_frame(duplicated, ["sensor_1"], scaler)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "invalid"])
def test_sequence_scaling_rejects_invalid_sensor_measurements(invalid):
    df = _sequence_frame(2)
    scaler = fit_sequence_scaler(df, ["sensor_1"])
    df["sensor_1"] = df["sensor_1"].astype(object)
    df.loc[0, "sensor_1"] = invalid
    with pytest.raises(ValueError, match="finite numeric"):
        fit_sequence_scaler(df, ["sensor_1"])
    with pytest.raises(ValueError, match="finite numeric"):
        scale_sequence_frame(df, ["sensor_1"], scaler)
