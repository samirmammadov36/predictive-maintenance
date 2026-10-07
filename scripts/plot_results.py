from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
metrics_dir = ROOT / "results/metrics"
plot_dir = ROOT / "results/plots"
plot_dir.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(metrics_dir / "metrics.csv")
predictions = pd.read_csv(metrics_dir / "predictions.csv")
comparison = pd.read_csv(metrics_dir / "clean_comparison.csv")

plt.figure(figsize=(7, 4))
plt.bar(comparison["model"], comparison["rmse"])
plt.ylabel("RMSE (cycles)")
plt.title("Clean FD001 final-snapshot comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(plot_dir / "clean_rmse_comparison.png", dpi=160)
plt.close()

robust = metrics[metrics["scenario"] != "clean"].copy()
summary = robust.groupby(["model", "scenario", "recovery"], dropna=False)["delta_rmse"].mean().reset_index()
summary["label"] = summary["scenario"] + " / " + summary["recovery"].astype(str)
for model in sorted(summary.model.unique()):
    part = summary[summary.model == model]
    plt.figure(figsize=(9, 4))
    plt.bar(part["label"], part["delta_rmse"])
    plt.ylabel("RMSE increase from clean (cycles)")
    plt.title(f"{model}: robustness under missing sensor measurements")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{model}_robustness_delta_rmse.png", dpi=160)
    plt.close()

clean_pred = predictions[predictions["scenario"] == "clean"]
for model in sorted(clean_pred.model.unique()):
    part = clean_pred[clean_pred.model == model]
    plt.figure(figsize=(5, 5))
    plt.scatter(part["rul_true"], part["rul_pred"], s=18)
    lim = max(part["rul_true"].max(), part["rul_pred"].max())
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"{model}: clean predictions")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{model}_clean_true_vs_pred.png", dpi=160)
    plt.close()

print(f"Plots written to {plot_dir}")
