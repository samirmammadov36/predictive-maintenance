from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "results/metrics/metrics.csv"
if not metrics_path.exists():
    raise SystemExit("Run scripts/run_checkpoint_experiments.py first")

df = pd.read_csv(metrics_path)
clean = df[df.scenario == "clean"][["model", "mae", "rmse"]].sort_values("rmse")
print("=== CLEAN MODEL COMPARISON ===")
print(clean.to_string(index=False))
print("\n=== ROBUSTNESS SUMMARY ===")
robust = (
    df[df.scenario != "clean"]
    .groupby(["model", "scenario", "recovery"], dropna=False)
    .agg(rmse_mean=("rmse", "mean"), delta_rmse_mean=("delta_rmse", "mean"))
    .reset_index()
    .sort_values(["scenario", "model", "rmse_mean"])
)
print(robust.to_string(index=False))
