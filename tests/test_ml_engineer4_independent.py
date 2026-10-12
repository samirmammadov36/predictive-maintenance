"""Independent regression tests for ML Engineer 4 reusable contracts."""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model_interface import PredictionAdapter
from run_experiments import build_evaluation_manifest, write_manifest
from predictive_maintenance.dashboard import health_label, observed_prefix, top_k_queue
from predictive_maintenance.evaluation import regression_metrics
from predictive_maintenance.models.baseline import MedianRULBaseline
from validation import (
    validate_cycle_order,
    validate_engine_split,
    validate_target_alignment,
)


def test_manifest_covers_expected_matrix_and_recovery_methods():
    rows = build_evaluation_manifest()
    assert len(rows) == 46
    assert rows[0] == {
        "model": "median_baseline",
        "scenario": "clean",
        "seed": "",
        "recovery": "baseline",
    }

    learned_rows = rows[1:]
    assert len(learned_rows) == 45
    clean_rows = [row for row in learned_rows if row["scenario"] == "clean"]
    assert len(clean_rows) == 3
    assert all(row["recovery"] == "median" and row["seed"] == "" for row in clean_rows)
    assert {row["model"] for row in learned_rows} == {
        "random_forest", "xgboost", "lstm"
    }
    assert {row["scenario"] for row in learned_rows} == {
        "clean", "random_10", "random_20", "short_gap"
    }
    random_rows = [
        row for row in learned_rows
        if row["scenario"] in {"random_10", "random_20"}
    ]
    assert len(random_rows) == 36
    assert {row["seed"] for row in random_rows} == {7, 17, 27}
    assert {row["recovery"] for row in random_rows} == {
        "median", "forward_fill"
    }


def test_manifest_writer_includes_recovery_column(tmp_path):
    target = tmp_path / "manifest.csv"
    rows = write_manifest(target)
    with target.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(rows) == 46
    assert "recovery" in saved[0]


def test_engine_split_rejects_overlap():
    assert validate_engine_split([1, 2], [3, 4]) is True
    with pytest.raises(ValueError, match="leakage"):
        validate_engine_split([1, 2], [2, 3])


def test_cycle_order_rejects_non_increasing_cycles():
    assert validate_cycle_order([1, 1, 2], [1, 2, 1]) is True
    with pytest.raises(ValueError, match="Cycle order"):
        validate_cycle_order([1, 1], [2, 2])


def test_target_alignment_checks_lengths_and_values():
    assert validate_target_alignment([1, 2], [5, 5], [4, 3]) is True
    with pytest.raises(ValueError, match="alignment"):
        validate_target_alignment([1, 2], [5, 5], [4, 2])
    with pytest.raises(ValueError, match="equal length"):
        validate_target_alignment([1], [5, 5], [4])


def test_median_baseline_requires_fit_and_predicts_constant_median():
    baseline = MedianRULBaseline()
    with pytest.raises(RuntimeError, match="not been fitted"):
        baseline.predict(2)
    assert baseline.fit([1, 3, 9]) is baseline
    np.testing.assert_allclose(baseline.predict(3), [3, 3, 3])


class NegativePredictor:
    def predict(self, values):
        return [-2, 4.5]


def test_prediction_adapter_clamps_negative_predictions():
    assert PredictionAdapter(NegativePredictor()).predict([0, 1]) == [0.0, 4.5]


def test_prediction_adapter_rejects_model_without_predict():
    with pytest.raises(TypeError, match="predict"):
        PredictionAdapter(object()).predict([1])


def test_regression_metrics_clip_negative_rul_and_calculate_scores():
    metrics = regression_metrics([1, 3], [-5, 5])
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx((2.5) ** 0.5)


@pytest.mark.parametrize(
    ("rul", "expected"),
    [
        (61, "Healthy"),
        (60, "Warning"),
        (31, "Warning"),
        (30, "Critical"),
        (0, "Critical"),
    ],
)
def test_health_labels_follow_configured_boundaries(rul, expected):
    assert health_label(rul) == expected


def test_health_label_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="greater"):
        health_label(50, healthy_threshold=30, warning_threshold=60)


def test_observed_prefix_never_includes_future_cycles():
    frame = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2],
            "cycle": [1, 2, 3, 1],
            "sensor_1": [10, 20, 30, 40],
        }
    )
    prefix = observed_prefix(frame, engine_id=1, observed_cycle=2)
    assert prefix["cycle"].tolist() == [1, 2]
    assert prefix["sensor_1"].tolist() == [10, 20]


def test_observed_prefix_rejects_unknown_engine_and_invalid_cutoff():
    frame = pd.DataFrame({"engine_id": [1], "cycle": [1]})
    with pytest.raises(ValueError, match="not found"):
        observed_prefix(frame, engine_id=99, observed_cycle=1)
    with pytest.raises(ValueError, match="at least 1"):
        observed_prefix(frame, engine_id=1, observed_cycle=0)


def test_top_k_queue_orders_lowest_rul_and_breaks_ties_by_engine_id():
    frame = pd.DataFrame(
        {"engine_id": [8, 2, 5], "rul_pred": [10.0, 10.0, 40.0]}
    )
    queue = top_k_queue(frame, 5)
    assert queue["engine_id"].tolist() == [2, 8, 5]
    assert queue["rank"].tolist() == [1, 2, 3]
    assert queue["health_status"].tolist() == [
        "Critical", "Critical", "Warning"
    ]


def test_top_k_queue_rejects_invalid_k_duplicate_ids_and_missing_predictions():
    with pytest.raises(ValueError, match="k must be"):
        top_k_queue(pd.DataFrame({"engine_id": [1], "rul_pred": [5]}), 3)
    with pytest.raises(ValueError, match="duplicate"):
        top_k_queue(
            pd.DataFrame({"engine_id": [1, 1], "rul_pred": [5, 6]}), 5
        )
    with pytest.raises(ValueError, match="missing"):
        top_k_queue(
            pd.DataFrame({"engine_id": [1], "rul_pred": [np.nan]}), 5
        )
