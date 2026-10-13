import numpy as np
import pandas as pd

import predictive_maintenance.inference as inference
from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN
from predictive_maintenance.sequences import build_prefix_window


def test_replay_uses_only_history_available_at_each_cycle(monkeypatch):
    engine = pd.DataFrame(
        {
            ID_COLUMN: [1, 1, 1, 1, 1],
            CYCLE_COLUMN: [1, 2, 3, 4, 5],
        }
    )

    received_prefixes = []

    def fake_predict_with_scenario(
        model_name,
        model,
        prefix,
        bundle,
        scenario,
        recovery,
        seed,
    ):
        received_prefixes.append(prefix[CYCLE_COLUMN].tolist())
        return float(len(prefix))

    monkeypatch.setattr(
        inference,
        "predict_with_scenario",
        fake_predict_with_scenario,
    )

    _, replay = inference.build_replay(
        engine=engine,
        observed_cycle=5,
        model_name="Random Forest",
        model=None,
        bundle=None,
        scenario="Clean",
        recovery="Training median",
        seed=7,
    )

    assert received_prefixes == [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]

    assert replay[CYCLE_COLUMN].tolist() == [1, 2, 3, 4, 5]
    assert replay["rul_pred"].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_lstm_prefix_window_handles_early_cycles():
    sensors = ["sensor_1", "sensor_2"]

    frame = pd.DataFrame(
    {
        CYCLE_COLUMN: [1, 2, 3],
        "sensor_1": [10.0, 20.0, 30.0],
        "sensor_2": [1.0, 2.0, 3.0],
    }
)

    window = build_prefix_window(
        frame,
        sensors,
        window=5,
    )

    assert window.shape == (5, 2)

    expected = np.array(
        [
            [10.0, 1.0],
            [10.0, 1.0],
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
        ]
    )

    np.testing.assert_allclose(window, expected)