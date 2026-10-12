"""Verify Task 29 data/target mapping and saved Random Forest consistency.

Run from the repository root after data preparation and model training:

    python scripts/verify_data_rf_pipeline.py

The script uses the generated, gitignored preprocessing/model artifacts and compares
fresh Random Forest clean-test metrics with the committed clean comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN, RUL_TRUTH_COLUMN
from predictive_maintenance.evaluation import regression_metrics
from predictive_maintenance.features import build_tabular_features


PREPARED_PATH = ROOT / "artifacts/preprocessing/prepared_bundle.joblib"
RF_MODEL_PATH = ROOT / "artifacts/models/random_forest.joblib"
SELECTION_PATH = ROOT / "results/metrics/validation_selection_summary.json"
CLEAN_COMPARISON_PATH = ROOT / "results/metrics/clean_comparison.csv"
OUTPUT_PATH = ROOT / "results/metrics/data_rf_verification.json"


def _require(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required generated artifacts are missing. Run `python scripts/download_data.py`, "
            "`python scripts/prepare_data.py`, and `python scripts/train_models.py` first. "
            "Missing: " + ", ".join(missing)
        )


def _validate_target_mapping(bundle) -> dict[str, int]:
    test_df = bundle.official_test_df.sort_values([ID_COLUMN, CYCLE_COLUMN]).copy()
    targets = bundle.official_test_targets.sort_values(ID_COLUMN).copy()

    engine_ids = sorted(map(int, test_df[ID_COLUMN].unique()))
    target_ids = sorted(map(int, targets[ID_COLUMN].unique()))
    if engine_ids != target_ids:
        raise AssertionError("Official test target engine IDs do not match official test engine IDs")
    if targets[ID_COLUMN].duplicated().any():
        raise AssertionError("Official test targets contain duplicate engine IDs")
    if targets[RUL_TRUTH_COLUMN].isna().any():
        raise AssertionError("Official test targets contain missing RUL values")

    max_cycles = test_df.groupby(ID_COLUMN)[CYCLE_COLUMN].max().astype(int)
    target_cycles = targets.set_index(ID_COLUMN)[CYCLE_COLUMN].astype(int)
    if not target_cycles.equals(max_cycles.loc[target_cycles.index]):
        raise AssertionError("Official test targets are not attached to each engine's final observed cycle")

    return {
        "official_test_engines": len(engine_ids),
        "official_target_rows": len(targets),
    }


def _predict_clean_random_forest(bundle, model) -> tuple[np.ndarray, np.ndarray, int]:
    target_map = bundle.official_test_targets.set_index(ID_COLUMN)[RUL_TRUTH_COLUMN].to_dict()
    y_true: list[float] = []
    y_pred: list[float] = []
    raw_negative_predictions = 0

    for engine_id in sorted(bundle.official_test_df[ID_COLUMN].unique()):
        prefix = (
            bundle.official_test_df[bundle.official_test_df[ID_COLUMN] == engine_id]
            .sort_values(CYCLE_COLUMN)
            .copy()
        )
        features, _ = build_tabular_features(
            prefix, bundle.selected_sensors, bundle.rolling_window
        )
        latest = features.iloc[[-1]][bundle.feature_columns]
        matrix = bundle.tabular_scaler.transform(latest)
        raw_prediction = float(model.predict(matrix)[0])
        raw_negative_predictions += int(raw_prediction < 0.0)
        prediction = max(0.0, raw_prediction)

        y_true.append(float(target_map[int(engine_id)]))
        y_pred.append(prediction)

    return (
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
        raw_negative_predictions,
    )


def main() -> None:
    _require([PREPARED_PATH, RF_MODEL_PATH, SELECTION_PATH, CLEAN_COMPARISON_PATH])

    bundle = joblib.load(PREPARED_PATH)
    model = joblib.load(RF_MODEL_PATH)
    target_checks = _validate_target_mapping(bundle)

    if len(bundle.feature_columns) != getattr(model, "n_features_in_", -1):
        raise AssertionError(
            "Saved Random Forest feature count does not match the frozen preprocessing schema"
        )
    scaler_features = getattr(bundle.tabular_scaler, "n_features_in_", -1)
    if len(bundle.feature_columns) != scaler_features:
        raise AssertionError("Saved tabular scaler feature count does not match the frozen schema")

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["random_forest_selected"]
    params = model.get_params()
    expected_params = {
        "n_estimators": int(selection["n_estimators"]),
        "max_depth": selection["max_depth"],
        "min_samples_leaf": int(selection["min_samples_leaf"]),
    }
    actual_params = {key: params[key] for key in expected_params}
    if actual_params != expected_params:
        raise AssertionError(
            f"Saved Random Forest parameters {actual_params} do not match selected configuration {expected_params}"
        )

    y_true, y_pred, raw_negative_predictions = _predict_clean_random_forest(bundle, model)
    computed = regression_metrics(y_true, y_pred)

    comparison = pd.read_csv(CLEAN_COMPARISON_PATH)
    rf_rows = comparison[comparison["model"] == "random_forest"]
    if len(rf_rows) != 1:
        raise AssertionError("clean_comparison.csv must contain exactly one random_forest row")
    expected_metrics = rf_rows.iloc[0]

    atol = 1e-6
    for metric in ("mae", "rmse"):
        if not np.isclose(computed[metric], float(expected_metrics[metric]), rtol=0.0, atol=atol):
            raise AssertionError(
                f"Fresh Random Forest {metric}={computed[metric]:.12f} does not match "
                f"committed {metric}={float(expected_metrics[metric]):.12f}"
            )

    report = {
        "status": "passed",
        **target_checks,
        "selected_sensor_count": len(bundle.selected_sensors),
        "feature_count": len(bundle.feature_columns),
        "random_forest_parameters": actual_params,
        "validation_mae": float(selection["mae"]),
        "validation_rmse": float(selection["rmse"]),
        "official_clean_mae": float(computed["mae"]),
        "official_clean_rmse": float(computed["rmse"]),
        "raw_negative_predictions_clamped": raw_negative_predictions,
        "verification": [
            "official test labels map one-to-one to official test engines",
            "target rows use each engine's final observed cycle",
            "saved RF feature count matches the frozen preprocessing schema",
            "saved RF hyperparameters match validation_selection_summary.json",
            "fresh saved-model clean metrics match clean_comparison.csv",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Verification report written to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
