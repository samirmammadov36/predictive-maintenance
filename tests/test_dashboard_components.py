from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.dashboard import (
    health_label,
    observed_prefix,
    plot_rul_history,
    plot_sensor_history,
    top_k_queue,
)


def _history() -> pd.DataFrame:
    rows = []
    for engine_id in (1, 2):
        for cycle in range(1, 7):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "sensor_2": float(engine_id * 100 + cycle),
                    "sensor_3": float(cycle * 2),
                }
            )
    return pd.DataFrame(rows)


def test_health_label_boundaries_match_demo_rules():
    assert health_label(61) == "Healthy"
    assert health_label(60) == "Warning"
    assert health_label(31) == "Warning"
    assert health_label(30) == "Critical"
    assert health_label(0) == "Critical"


def test_health_label_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="healthy_threshold"):
        health_label(50, healthy_threshold=30, warning_threshold=60)


def test_observed_prefix_never_exposes_future_rows_or_other_engines():
    prefix = observed_prefix(_history(), engine_id=1, observed_cycle=4)
    assert prefix["engine_id"].unique().tolist() == [1]
    assert prefix["cycle"].tolist() == [1, 2, 3, 4]
    assert int(prefix["cycle"].max()) == 4


def test_observed_prefix_rejects_unknown_engine():
    with pytest.raises(ValueError, match="not found"):
        observed_prefix(_history(), engine_id=999, observed_cycle=4)


def test_top_k_queue_sorts_by_rul_then_engine_id_and_adds_status():
    predictions = pd.DataFrame(
        {
            "engine_id": [9, 4, 2, 8, 1, 7, 6],
            "rul_pred": [40.0, 15.0, 15.0, 70.0, 5.0, 20.0, 80.0],
            "observed_cycle": [100, 120, 90, 85, 140, 110, 95],
        }
    )

    queue = top_k_queue(predictions, k=5)

    assert queue["engine_id"].tolist() == [1, 2, 4, 7, 9]
    assert queue["rank"].tolist() == [1, 2, 3, 4, 5]
    assert queue["health_status"].tolist() == ["Critical", "Critical", "Critical", "Critical", "Warning"]


def test_top_k_queue_rejects_duplicate_engine_predictions():
    predictions = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 3, 4, 5],
            "rul_pred": [10, 11, 12, 13, 14, 15],
        }
    )
    with pytest.raises(ValueError, match="duplicate engine IDs"):
        top_k_queue(predictions, k=5)


def test_sensor_history_plot_uses_only_rows_supplied_in_prefix():
    prefix = observed_prefix(_history(), engine_id=1, observed_cycle=3)
    fig = plot_sensor_history(prefix, "sensor_2")
    try:
        line = fig.axes[0].lines[0]
        assert list(line.get_xdata()) == [1, 2, 3]
        assert max(line.get_xdata()) == 3
    finally:
        fig.clear()


def test_rul_history_plot_orders_cycles_before_plotting():
    rul_history = pd.DataFrame(
        {
            "cycle": [3, 1, 2],
            "rul_pred": [80.0, 100.0, 90.0],
        }
    )
    fig = plot_rul_history(rul_history)
    try:
        line = fig.axes[0].lines[0]
        assert list(line.get_xdata()) == [1, 2, 3]
        assert list(line.get_ydata()) == [100.0, 90.0, 80.0]
    finally:
        fig.clear()
