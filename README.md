# Predictive Maintenance — NASA C-MAPSS FD001

Remaining Useful Life (RUL) prediction using Random Forest, XGBoost and a compact LSTM, with a median baseline and robustness tests under artificial sensor missingness.

## ML Engineer 4 deliverables

This branch contains repository setup, leakage-safe split contracts, integrity checks, the median baseline, a common experiment manifest, metric comparison utilities and the Streamlit control layer.

## Reproducibility contract

- Dataset: NASA C-MAPSS FD001.
- Engine split: 80/20 with seed 42.
- Validation history: first 70% of each held-out engine trajectory.
- Engine ID is metadata only and never a model feature.
- RUL is measured in operating cycles.
- Clean models are frozen before missingness experiments.
- Random 10% and 20% missingness use seeds 7, 17 and 27.
- Missing sensors are masked before imputation and feature generation.
- Forward fill is causal and falls back to training medians.
- Predictions are clamped to zero.
- Dashboard labels: RUL > 60 Healthy, 30 < RUL <= 60 Warning, RUL <= 30 Critical.

## Run

Generate the shared experiment manifest with:
    python run_experiments.py

Run the dashboard with:
    streamlit run app.py

Model-specific training and preprocessing modules can consume config.yaml, split_manifest.json, validation.py and model_interface.py.

## Scope

The project evaluates FD001 RUL prediction and robustness to artificial missing sensor measurements. It does not claim a new ML algorithm or real-world maintenance thresholds.

## Samir LSTM implementation and run path

See [Samir's task summary and acceptance evidence](docs/ml_engineer3_summary.md) for Tasks 3/9/10/15/19/23/27/31 and the linked sequence, missingness, checkpoint, evaluation and replay contracts.

Supply the original `data/raw/train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt`, with dependencies from `requirements.txt` available in the project .venv. From the repository, the executable path is:

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_data.py
& '.\.venv\Scripts\python.exe' scripts/train_models.py
& '.\.venv\Scripts\python.exe' scripts/run_checkpoint_experiments.py
& '.\.venv\Scripts\python.exe' scripts/plot_results.py
& '.\.venv\Scripts\python.exe' scripts/checkpoint_report.py
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Preparation writes the bundle, training medians/scalers and input/split/source metadata under `artifacts/preprocessing`. Training fits the baseline and all three models, writes checkpoints/sidecars under `artifacts/models` and validation logs under `results/metrics`. Evaluation loads frozen models and raw official-test prefixes, writes 45 metric rows plus a separate baseline, predictions and seed aggregates; plotting/report scripts consume those generated results. The LSTM application needs the prepared raw-input bundle and compatible LSTM checkpoint/metadata, not precomputed metric tables.

Root `run_experiments.py` only writes a coarse `experiment_manifest.csv`; it does not execute predictions or enumerate both recovery methods. Root `comparison.py` only merges exported CSV columns. Neither replaces `scripts/run_checkpoint_experiments.py`.

The current application connects **LSTM only**. Other model adapters and top-K queue assembly remain pending team work; unsupported selections display their integration status. The previously reported **634 passed** suite used synthetic/regression fixtures and was not rerun for this documentation review. Real FD001 preparation/training/evaluation and real-checkpoint browser acceptance remain unverified; historical metric tables are not evidence for the current checkpoint. These commands document the run path and were not executed during this review.
