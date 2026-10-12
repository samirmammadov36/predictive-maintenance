"""Build the shared manifest for the executable robustness evaluation.

The learned-model matrix mirrors src/predictive_maintenance/experiments.py:
- clean data, with no recovery step;
- random 10% and 20% missingness for seeds 7, 17 and 27, each with
  median and causal forward-fill recovery;
- a five-cycle sensor gap with both recovery methods.

The median baseline is evaluated separately on the clean official-test set by
scripts/run_checkpoint_experiments.py, so it appears once in this manifest.
"""

import csv
from pathlib import Path

RANDOM_SEEDS = (7, 17, 27)
RANDOM_SCENARIOS = ("random_10", "random_20")
RECOVERY_METHODS = ("median", "forward_fill")
LEARNED_MODELS = ("random_forest", "xgboost", "lstm")


def build_evaluation_manifest(models=LEARNED_MODELS, random_seeds=RANDOM_SEEDS):
    """Return a plan aligned with the executable experiment runner.

    Returns 46 rows by default: 45 learned-model evaluations and one clean
    median-baseline evaluation.
    """
    rows = [
        {
            "model": "median_baseline",
            "scenario": "clean",
            "seed": "",
            "recovery": "baseline",
        }
    ]

    for model in models:
        rows.append(
            {
                "model": model,
                "scenario": "clean",
                "seed": "",
                "recovery": "none",
            }
        )
        for scenario in RANDOM_SCENARIOS:
            for seed in random_seeds:
                for recovery in RECOVERY_METHODS:
                    rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "seed": seed,
                            "recovery": recovery,
                        }
                    )
        for recovery in RECOVERY_METHODS:
            rows.append(
                {
                    "model": model,
                    "scenario": "short_gap",
                    "seed": "",
                    "recovery": recovery,
                }
            )

    return rows


def write_manifest(path="experiment_manifest.csv"):
    """Write the shared evaluation plan as a CSV file."""
    rows = build_evaluation_manifest()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "scenario", "seed", "recovery"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    write_manifest()
