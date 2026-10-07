# 75% Checkpoint — ML Pipeline and Robustness Experiment

This checkpoint adds the executable Phases 1–5 ML pipeline while preserving the existing team-owned root modules (`app.py`, `baseline.py`, `comparison.py`, `model_interface.py`, `run_experiments.py`, `validation.py`, etc.).

## What is implemented

- Official NASA C-MAPSS FD001 loading/downloading
- RUL target generation (uncapped truth; capped training target at 125)
- 80/20 engine-level split, seed 42
- 70% causal validation prefix per held-out engine
- training-only sensor screening, medians, and scalers
- causal forward fill and training-median recovery
- shared RF/XGBoost causal features
- 30-cycle LSTM windows
- median baseline, Random Forest, XGBoost, compact LSTM
- clean evaluation and sensor-missingness stress tests
- 10% / 20% random missingness, seeds 7/17/27
- five-cycle short sensor gap
- MAE/RMSE aggregation and plots

The existing root `run_experiments.py` creates the team experiment manifest. The executable model stress test is intentionally named `scripts/run_checkpoint_experiments.py` to avoid overwriting that teammate-owned module.

## Environment

Recommended: Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run order

```powershell
python scripts/download_data.py
pytest -q
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/run_checkpoint_experiments.py
python scripts/plot_results.py
python scripts/checkpoint_report.py
```

Generated raw data and trained model/preprocessing artifacts are ignored by Git. Small result CSVs and plots under `results/` may be committed as checkpoint evidence.

## Current known limitation

The compact LSTM showed strong validation-prefix performance but poor official-test generalization, with many high-RUL engine predictions saturating around roughly 52 cycles. This is documented as remaining model-improvement work for the final 25%; the checkpoint does not hide or retune against the official test set.

## Remaining final 25%

- full Streamlit inference integration and replay/queue behavior
- LSTM validation/model-selection improvement
- final QA and end-to-end acceptance test
- final README/report/presentation polishing
