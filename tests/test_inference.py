import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    ("history_length", "expected_first_cycle", "expected_last_cycle"),
    [
        (1, 1, 1),
        (29, 1, 29),
        (30, 1, 30),
        (31, 2, 31),
    ],
)
def test_lstm_prefix_window_uses_real_30_cycle_boundary(
    history_length,
    expected_first_cycle,
    expected_last_cycle,
):
    sensors = ["sensor_1", "sensor_2"]

    frame = pd.DataFrame(
        {
            CYCLE_COLUMN: list(range(1, history_length + 1)),
            "sensor_1": [
                float(cycle)
                for cycle in range(1, history_length + 1)
            ],
            "sensor_2": [
                float(cycle * 10)
                for cycle in range(1, history_length + 1)
            ],
        }
    )

    window = build_prefix_window(
        frame,
        sensors,
        window=30,
    )

    assert window.shape == (30, 2)

    # The newest observation must always be the final row.
    assert window[-1, 0] == expected_last_cycle

    if history_length < 30:
        # Short histories are left-padded using the first observation.
        assert window[0, 0] == expected_first_cycle
        assert window[30 - history_length - 1, 0] == expected_first_cycle

    else:
        # Histories >= 30 keep only the latest 30 observations.
        assert window[0, 0] == expected_first_cycle