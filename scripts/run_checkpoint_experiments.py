from __future__ import annotations

from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.experiments import aggregate_random_results, run_experiments
from predictive_maintenance.models.baseline import MedianRULBaseline
from predictive_maintenance.evaluation import regression_metrics


def main() -> None:
    bundle = joblib.load(ROOT / "artifacts/preprocessing/prepared_bundle.joblib")
    results, predictions = run_experiments(bundle, ROOT / "artifacts", random_seeds=(7, 17, 27), return_predictions=True)
    out = ROOT / "results/metrics"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    aggregate_random_results(results).to_csv(out / "metrics_aggregated.csv", index=False)

    baseline = joblib.load(ROOT / "artifacts/models/baseline.joblib")
    truth = bundle.official_test_targets["rul_true"].to_numpy(dtype=float)
    baseline_metrics = regression_metrics(truth, baseline.predict(len(truth)))
    baseline_row = {"model": "median_baseline", "scenario": "clean", **baseline_metrics}
    import pandas as pd
    pd.DataFrame([baseline_row]).to_csv(out / "baseline.csv", index=False)
    clean_models = results[results["scenario"] == "clean"][["model", "mae", "rmse"]]
    comparison = pd.concat([clean_models, pd.DataFrame([baseline_row])[["model", "mae", "rmse"]]], ignore_index=True).sort_values("rmse")
    comparison.to_csv(out / "clean_comparison.csv", index=False)
    print(results.to_string(index=False))
    print("\nWrote metrics.csv, predictions.csv, metrics_aggregated.csv, baseline.csv and clean_comparison.csv")


if __name__ == "__main__":
    main()
