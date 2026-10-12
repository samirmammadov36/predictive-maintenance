from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.artifacts import dump_joblib, load_joblib
from predictive_maintenance.constants import (
    CYCLE_COLUMN,
    ID_COLUMN,
    RAW_COLUMNS,
    RUL_TARGET_COLUMN,
    RUL_TRUTH_COLUMN,
)
from predictive_maintenance.data import (
    add_training_rul,
    official_test_targets,
    read_cmapss_file,
    read_test_rul,
)
from predictive_maintenance.features import build_tabular_features


def _raw_row(engine_id: int, cycle: int, base: float) -> list[float]:
    # Official FD001 rows contain engine id, cycle, three operating settings,
    # and 21 sensor values: 26 columns in total.
    settings = [base + 0.1, base + 0.2, base + 0.3]
    sensors = [base + i for i in range(1, 22)]
    return [engine_id, cycle, *settings, *sensors]


def test_cmapss_loader_uses_official_schema_and_sorts_rows(tmp_path):
    path = tmp_path / "train_FD001.txt"
    rows = [
        _raw_row(2, 1, 20.0),
        _raw_row(1, 2, 12.0),
        _raw_row(1, 1, 11.0),
    ]
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )

    frame = read_cmapss_file(path)

    assert frame.columns.tolist() == RAW_COLUMNS
    assert frame[[ID_COLUMN, CYCLE_COLUMN]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
    assert pd.api.types.is_integer_dtype(frame[ID_COLUMN])
    assert pd.api.types.is_integer_dtype(frame[CYCLE_COLUMN])


def test_training_rul_keeps_uncapped_truth_and_caps_fitting_target():
    frame = pd.DataFrame(
        {
            ID_COLUMN: [1, 1, 1, 1, 1],
            CYCLE_COLUMN: [1, 2, 3, 4, 5],
        }
    )

    labeled = add_training_rul(frame, cap=2)

    assert labeled[RUL_TRUTH_COLUMN].tolist() == [4, 3, 2, 1, 0]
    assert labeled[RUL_TARGET_COLUMN].tolist() == [2.0, 2.0, 2.0, 1.0, 0.0]


def test_official_test_rul_maps_one_based_file_rows_to_final_engine_cycles(tmp_path):
    test_frame = pd.DataFrame(
        {
            ID_COLUMN: [1, 1, 2, 2, 2],
            CYCLE_COLUMN: [1, 2, 1, 2, 3],
        }
    )
    rul_path = tmp_path / "RUL_FD001.txt"
    rul_path.write_text("7\n12\n", encoding="utf-8")

    rul = read_test_rul(rul_path)
    targets = official_test_targets(test_frame, rul)

    assert rul.index.tolist() == [1, 2]
    assert targets.to_dict("records") == [
        {ID_COLUMN: 1, CYCLE_COLUMN: 2, RUL_TRUTH_COLUMN: 7},
        {ID_COLUMN: 2, CYCLE_COLUMN: 3, RUL_TRUTH_COLUMN: 12},
    ]


def test_random_forest_pipeline_predictions_survive_joblib_round_trip(tmp_path):
    rows = []
    for engine_id in (1, 2, 3):
        for cycle in range(1, 7):
            rows.append(
                {
                    ID_COLUMN: engine_id,
                    CYCLE_COLUMN: cycle,
                    "sensor_2": float(engine_id * 5 + cycle),
                    "sensor_3": float(engine_id + cycle * 0.5),
                }
            )

    frame = add_training_rul(pd.DataFrame(rows), cap=5)
    engineered, feature_columns = build_tabular_features(
        frame, ["sensor_2", "sensor_3"], rolling_window=3
    )
    scaler = StandardScaler().fit(engineered[feature_columns])
    matrix = scaler.transform(engineered[feature_columns])
    target = engineered[RUL_TARGET_COLUMN].to_numpy(dtype=np.float32)

    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
    model.fit(matrix, target)
    expected = model.predict(matrix)

    model_path = tmp_path / "random_forest.joblib"
    dump_joblib(model, model_path)
    reloaded = load_joblib(model_path)
    actual = reloaded.predict(matrix)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert reloaded.n_features_in_ == len(feature_columns)
