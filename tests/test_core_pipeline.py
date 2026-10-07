from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN, RUL_TARGET_COLUMN, RUL_TRUTH_COLUMN
from predictive_maintenance.data import add_training_rul, split_development_engines, validation_prefixes
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask
from predictive_maintenance.sequences import build_training_windows


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
    df = add_training_rul(_frame(), cap=5)
    X, y = build_training_windows(df, ["sensor_1", "sensor_2"], RUL_TARGET_COLUMN, window=4)
    assert X.shape == (len(df), 4, 2)
    first = X[0]
    assert np.all(first == np.array([[11.0, 2.0]] * 4, dtype=np.float32))
    assert y[0] == 5
