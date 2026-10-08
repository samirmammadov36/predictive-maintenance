from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask

SENSORS = ["sensor_11", "sensor_2", "sensor_5"]
SEEDS = [7, 17, 27]


def _prefix(length=6):
    cycles = np.arange(1, length + 1)
    frame = pd.DataFrame({
        "engine_id": np.full(length, 3), "cycle": cycles,
        "sensor_5": 500.0 + cycles, "setting_1": 10.0 + cycles,
        "sensor_2": 200.0 + cycles, "sensor_11": 1100.0 + cycles,
        "rul_true": 100 - cycles, "rul_target": 100 - cycles,
    }, index=10 + 7 * np.arange(length))
    frame.attrs["units"] = "cycles"
    return frame


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("length,sensors,expected", [
    (1, SENSORS[:1], (0, 0)), (5, SENSORS[:1], (0, 1)),
    (15, SENSORS[:1], (2, 3)), (25, SENSORS[:1], (2, 5)),
    (3, SENSORS, (1, 2)), (10, SENSORS, (3, 6)),
])
def test_random_counts_nesting_reproducibility(seed, length, sensors, expected):
    prefix = _prefix(length)
    before = prefix.copy(deep=True)
    low, high = nested_random_masks(prefix, sensors, seed)
    again_low, again_high = nested_random_masks(prefix, sensors, seed)
    assert (low.to_numpy().sum(), high.to_numpy().sum()) == expected
    assert not (low & ~high).to_numpy().any()
    pd.testing.assert_frame_equal(low, again_low)
    pd.testing.assert_frame_equal(high, again_high)
    assert low.index.equals(prefix.index) and low.columns.tolist() == sensors
    assert all(dtype == bool for dtype in low.dtypes)
    pd.testing.assert_frame_equal(prefix, before)
    assert prefix.attrs == before.attrs


@pytest.mark.parametrize("seed,low_cells,high_cells", [
    (7, [(38, "sensor_2")], [(17, "sensor_2"), (38, "sensor_2")]),
    (17, [(24, "sensor_11")], [(24, "sensor_11"), (38, "sensor_2")]),
    (27, [(31, "sensor_11")], [(31, "sensor_11"), (31, "sensor_2")]),
])
def test_existing_random_mask_positions_are_preserved(seed, low_cells, high_cells):
    # Fixed values captured from the committed algorithm before this change.
    low, high = nested_random_masks(_prefix(5), SENSORS[:2], seed)
    for mask, expected in [(low, low_cells), (high, high_cells)]:
        actual = [(row, sensor) for row in mask.index for sensor in mask.columns if mask.loc[row, sensor]]
        assert actual == expected


@pytest.mark.parametrize("length,hidden_cycles", [
    (1, [1]), (4, [1, 2, 3, 4]), (5, [1, 2, 3, 4, 5]),
    (6, [2, 3, 4, 5, 6]), (30, [26, 27, 28, 29, 30]),
])
@pytest.mark.parametrize("sensors,expected_sensor", [
    (["sensor_5", "sensor_11", "sensor_2"], "sensor_11"),
    (["sensor_5", "sensor_2"], "sensor_5"),
])
def test_gap_preferred_fallback_and_observed_tail(length, hidden_cycles, sensors, expected_sensor):
    prefix = _prefix(length)
    before = prefix.copy(deep=True)
    mask, sensor = short_gap_mask(prefix, sensors)
    assert sensor == expected_sensor
    assert prefix.loc[mask[sensor], "cycle"].tolist() == hidden_cycles
    assert mask.to_numpy().sum() == len(hidden_cycles)
    corrupted = apply_sensor_mask(prefix, mask, sensors)
    assert corrupted.loc[mask[sensor], sensor].isna().all()
    pd.testing.assert_series_equal(corrupted.loc[~mask[sensor], sensor], prefix.loc[~mask[sensor], sensor])
    untouched = [column for column in prefix.columns if column != sensor]
    pd.testing.assert_frame_equal(corrupted[untouched], prefix[untouched])
    pd.testing.assert_frame_equal(prefix, before)
    assert corrupted.attrs == prefix.attrs == before.attrs


def test_custom_gap_arguments_and_numpy_integer_seed():
    prefix = _prefix()
    mask, sensor = short_gap_mask(prefix, SENSORS, preferred_sensor="sensor_2", gap_cycles=np.int64(2))
    assert sensor == "sensor_2"
    assert prefix.loc[mask[sensor], "cycle"].tolist() == [5, 6]
    expected = nested_random_masks(prefix, SENSORS, 7)
    actual = nested_random_masks(prefix, SENSORS, np.int64(7))
    for left, right in zip(expected, actual):
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("seed", SEEDS)
def test_shared_masks_can_be_reused_without_mutation(seed):
    prefix = _prefix(10).assign(dataset_split="official_test")
    original = prefix.copy(deep=True)
    low, high = nested_random_masks(prefix, SENSORS, seed)
    gap, _ = short_gap_mask(prefix, SENSORS)
    for mask in [low, high, gap]:
        mask_before = mask.copy(deep=True)
        expected = apply_sensor_mask(prefix, mask, SENSORS)
        # Consumer names demonstrate reuse; no model or recovery integration is claimed.
        for model in ["RF", "XGBoost", "LSTM"]:
            for recovery in ["training_median", "causal_forward_fill"]:
                actual = apply_sensor_mask(prefix, mask, SENSORS)
                pd.testing.assert_frame_equal(actual, expected, obj=f"{model}/{recovery}")
                np.testing.assert_array_equal(actual[SENSORS].isna(), mask)
        pd.testing.assert_frame_equal(mask, mask_before)
    pd.testing.assert_frame_equal(prefix, original)
    assert prefix.attrs == original.attrs


def test_exact_label_alignment_with_reordered_mask_and_nondefault_indices():
    prefix = _prefix(5)
    prefix.index = [90, 10, 70, 30, 50]
    mask = pd.DataFrame(False, index=prefix.index, columns=SENSORS)
    mask.loc[10, "sensor_5"] = True
    mask.loc[50, "sensor_11"] = True
    reordered = mask.loc[[50, 90, 30, 70, 10], list(reversed(SENSORS))]
    result = apply_sensor_mask(prefix, reordered, SENSORS)
    assert result.index.tolist() == [90, 10, 70, 30, 50]
    assert result.columns.tolist() == prefix.columns.tolist()
    assert pd.isna(result.loc[10, "sensor_5"])
    assert pd.isna(result.loc[50, "sensor_11"])
    assert result.loc[10, "sensor_11"] == 1102.0
    assert result.loc[50, "sensor_5"] == 505.0
    np.testing.assert_array_equal(result[SENSORS].isna(), mask)
    pd.testing.assert_frame_equal(result.drop(columns=SENSORS), prefix.drop(columns=SENSORS))


def test_composite_identity_keeps_existing_seed_algorithm():
    prefix = _prefix()
    plain = nested_random_masks(prefix, SENSORS, 17)
    for split in ["development", "official_test"]:
        composite = nested_random_masks(prefix.assign(dataset_split=split), SENSORS, 17)
        for actual, expected in zip(composite, plain):
            pd.testing.assert_frame_equal(actual, expected)


def _call(operation, prefix, sensors=SENSORS):
    if operation == "random":
        return nested_random_masks(prefix, sensors, 7)
    if operation == "gap":
        return short_gap_mask(prefix, sensors)
    mask = pd.DataFrame(False, index=prefix.index, columns=SENSORS)
    return apply_sensor_mask(prefix, mask, sensors)


@pytest.mark.parametrize("operation", ["random", "gap", "apply"])
@pytest.mark.parametrize("problem,message", [
    ("empty", "non-empty"), ("mixed", "one engine identity"),
    ("mixed_split", "one engine identity"), ("missing_id", "identity"),
    ("missing_id_column", "engine_id"), ("missing_split", "identity"),
    ("blank_split", "dataset_split"), ("duplicate_cycle", "Duplicate cycle"),
    ("unsorted", "chronological"), ("missing_cycle", "cycle"),
    ("duplicate_index", "unique"), ("duplicate_columns", "unique"),
])
def test_invalid_prefix_rejected_by_all_operations(operation, problem, message):
    prefix = _prefix()
    if problem == "empty":
        prefix = prefix.iloc[:0]
    elif problem == "mixed":
        prefix.loc[prefix.index[-1], "engine_id"] = 4
    elif problem == "mixed_split":
        prefix["dataset_split"] = ["train"] * 5 + ["test"]
    elif problem == "missing_id":
        prefix["engine_id"] = pd.Series([3] * 5 + [pd.NA], index=prefix.index, dtype="Int64")
    elif problem == "missing_id_column":
        prefix = prefix.drop(columns="engine_id")
    elif problem == "missing_split":
        prefix["dataset_split"] = ["train"] * 5 + [None]
    elif problem == "blank_split":
        prefix["dataset_split"] = " "
    elif problem == "duplicate_cycle":
        prefix.loc[prefix.index[-1], "cycle"] = 5
    elif problem == "unsorted":
        prefix = prefix.iloc[[0, 2, 1, 3, 4, 5]]
    elif problem == "missing_cycle":
        prefix = prefix.drop(columns="cycle")
    elif problem == "duplicate_index":
        prefix.index = [1, 1, 2, 3, 4, 5]
    else:
        prefix = pd.concat([prefix, prefix[["sensor_2"]]], axis=1)
    with pytest.raises(ValueError, match=message):
        _call(operation, prefix)


@pytest.mark.parametrize("operation", ["random", "gap", "apply"])
@pytest.mark.parametrize("column", ["engine_id", "cycle"])
@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf, -np.inf, None, True, 1 + 0j, np.complex64(1), "bad"])
def test_invalid_identity_and_cycle_values(operation, column, value):
    prefix = _prefix()
    prefix[column] = prefix[column].astype(object)
    prefix.loc[prefix.index[0], column] = value
    with pytest.raises(ValueError):
        _call(operation, prefix)


def test_nullable_missing_cycle_is_rejected():
    prefix = _prefix()
    prefix["cycle"] = pd.Series([1, 2, 3, 4, 5, pd.NA], dtype="Int64", index=prefix.index)
    with pytest.raises(ValueError, match="cycle.*finite positive integers"):
        nested_random_masks(prefix, SENSORS, 7)


@pytest.mark.parametrize("operation", ["random", "gap", "apply"])
@pytest.mark.parametrize("sensors", [[], None, "sensor_11", {"sensor_11"}, ["sensor_2", "sensor_2"],
    [None], ["cycle"], ["engine_id"], ["setting_1"], ["rul_target"], ["sensor_99"], ["sensor_1"]])
def test_invalid_sensor_lists(operation, sensors):
    with pytest.raises(ValueError, match="sensor"):
        _call(operation, _prefix(), sensors)


@pytest.mark.parametrize("seed", [-1, 7.0, 1.5, True, np.bool_(False), "7", None, np.nan, np.inf])
def test_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed"):
        nested_random_masks(_prefix(), SENSORS, seed)


@pytest.mark.parametrize("low,high", [(-0.1, 0.2), (0.1, 1.1), (0.3, 0.2), (np.nan, 0.2),
    (0.1, np.inf), (True, 0.2), (0.1, False), ("0.1", 0.2), (0.1, None)])
def test_invalid_rates(low, high):
    with pytest.raises(ValueError, match="rate"):
        nested_random_masks(_prefix(), SENSORS, 7, low, high)


@pytest.mark.parametrize("low,high", [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_valid_endpoint_rates(low, high):
    masks = nested_random_masks(_prefix(5), SENSORS, 7, low, high)
    assert [mask.to_numpy().sum() for mask in masks] == [int(low * 15), int(high * 15)]


@pytest.mark.parametrize("gap", [0, -1, 5.0, 1.5, True, np.bool_(False), "5", None, np.nan, np.inf])
def test_invalid_gap_lengths(gap):
    with pytest.raises(ValueError, match="gap_cycles"):
        short_gap_mask(_prefix(), SENSORS, gap_cycles=gap)


@pytest.mark.parametrize("sensor", [None, "cycle", "sensor_99", 11])
def test_invalid_preferred_sensor(sensor):
    with pytest.raises(ValueError, match="preferred_sensor"):
        short_gap_mask(_prefix(), SENSORS, preferred_sensor=sensor)


@pytest.mark.parametrize("problem,message", [
    ("missing_row", "row labels"), ("extra_row", "row labels"), ("wrong_row", "row labels"),
    ("duplicate_row", "unique"), ("missing_sensor", "columns"), ("extra_sensor", "columns"),
    ("duplicate_sensor", "unique"), ("integer", "boolean"), ("object", "boolean"),
    ("string", "boolean"), ("missing_bool", "boolean"), ("series", "DataFrame"),
])
def test_malformed_masks_are_not_silently_filled(problem, message):
    prefix = _prefix()
    mask = pd.DataFrame(False, index=prefix.index, columns=SENSORS)
    if problem == "missing_row":
        mask = mask.iloc[:-1]
    elif problem == "extra_row":
        mask.loc[999] = False
    elif problem == "wrong_row":
        mask.index = [999, *mask.index[1:]]
    elif problem == "duplicate_row":
        mask.index = [mask.index[0], *mask.index[:-1]]
    elif problem == "missing_sensor":
        mask = mask.drop(columns="sensor_2")
    elif problem == "extra_sensor":
        mask["setting_1"] = False
    elif problem == "duplicate_sensor":
        mask = pd.concat([mask, mask[["sensor_2"]]], axis=1)
    elif problem in ["integer", "object", "string"]:
        mask = mask.astype({"integer": int, "object": object, "string": str}[problem])
    elif problem == "missing_bool":
        mask = mask.astype("boolean")
        mask.iloc[0, 0] = pd.NA
    else:
        mask = mask.iloc[:, 0]
    with pytest.raises(ValueError, match=message):
        apply_sensor_mask(prefix, mask, SENSORS)


def test_existing_raw_missingness_and_nullable_boolean_mask_are_allowed():
    prefix = _prefix()
    prefix.loc[prefix.index[0], "sensor_2"] = np.nan
    low, _ = nested_random_masks(prefix, SENSORS, 7)
    result = apply_sensor_mask(prefix, low.astype("boolean"), SENSORS)
    assert pd.isna(result.loc[prefix.index[0], "sensor_2"])
    pd.testing.assert_frame_equal(result.drop(columns=SENSORS), prefix.drop(columns=SENSORS))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cutoff", [1, 4, 7])
def test_explicit_prefix_is_isolated_from_future_sensor_changes(seed, cutoff):
    trajectory = _prefix(10)
    changed = trajectory.copy(deep=True)
    changed.loc[changed.cycle > cutoff, SENSORS] = -999999.0
    prefix = trajectory.loc[trajectory.cycle <= cutoff].copy()
    changed_prefix = changed.loc[changed.cycle <= cutoff].copy()
    for left, right in zip(nested_random_masks(prefix, SENSORS, seed), nested_random_masks(changed_prefix, SENSORS, seed)):
        pd.testing.assert_frame_equal(left, right)
        pd.testing.assert_frame_equal(apply_sensor_mask(prefix, left, SENSORS), apply_sensor_mask(changed_prefix, right, SENSORS))
    gap, sensor = short_gap_mask(prefix, SENSORS)
    changed_gap, changed_sensor = short_gap_mask(changed_prefix, SENSORS)
    pd.testing.assert_frame_equal(gap, changed_gap)
    assert sensor == changed_sensor == "sensor_11"
    assert gap.index.equals(prefix.index)
