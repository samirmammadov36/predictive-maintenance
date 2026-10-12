# ML Engineer 4 — Implementation, Verification & Handover Report

**Repository:** `samirmammadov36/predictive-maintenance`<br>
**Dataset:** NASA C-MAPSS FD001<br>
**Prepared:** 2026-10-12<br>
**Implementation branch:** `feature/lale-ml-engineer-4-completion`

## Executive summary

The ML Engineer 4 work is organized around eight deliverable areas. The main code gap identified in the previous review was that `app.py` exposed controls but did not connect them to saved models and inference. This branch replaces that shell with a Streamlit inference workflow that loads the prepared bundle and trained model artifacts, predicts Remaining Useful Life (RUL) from the selected engine's observed prefix, supports sensor-missingness scenarios and recovery methods, displays sensor/RUL history, and calculates a downloadable top-K inspection queue.

**Important verification boundary:** this is a code-level handover report. The GitHub connector was used to commit changes, but this environment has not executed the project in a Python environment with the dataset and model artifacts. Do not treat the application as runtime-tested or claim that the test suite passed until the commands in the QA section have been run.

## Task-by-task status

| Task area | Status | Evidence / notes |
|---|---|---|
| 4. Repository structure and configuration | DONE | Existing `config.yaml`, `src/predictive_maintenance/`, scripts, results, and tests define a reproducible project structure. |
| 11. Deterministic split and validation-prefix manifest | DONE in pipeline | `scripts/prepare_data.py` persists the seed, train engine IDs, validation engine IDs, and per-engine validation prefix cycles to `artifacts/preprocessing/split_manifest.json`. This generated file depends on running data preparation and is not expected to exist in a clean clone. |
| 12. Data-integrity report | DONE in pipeline | `scripts/prepare_data.py` writes `data/processed/data_integrity_report.json`, including split overlap and missingness checks, and writes hashes/sizes to the data manifest. Run the preparation step to generate these artifacts. |
| 16. Median baseline and prediction adapter | DONE | Median baseline implementation exists at `src/predictive_maintenance/models/baseline.py`; the repository root also contains the common `PredictionAdapter` contract in `model_interface.py`. The dashboard baseline inference path has been connected to the baseline's count-based `predict(n)` API. |
| 20. Experiment manifest and experiment execution | DONE as separate entry points | `python run_experiments.py` writes the shared manifest. `python scripts/run_checkpoint_experiments.py` executes the model robustness experiment and writes metrics/predictions/aggregates. These are separate by design; the first writes a plan, the second runs the evaluation. |
| 24. Metrics and model comparison | IMPLEMENTED; regenerate and verify | The experiment runner writes `metrics.csv`, `predictions.csv`, `metrics_aggregated.csv`, `baseline.csv`, and `clean_comparison.csv`. Regenerate these from the current artifacts before presenting results as final. |
| 28. Streamlit application | IMPLEMENTED; runtime validation pending | `app.py` now loads prepared data and saved models, supports model selection, observed-cycle cutoff, clean/random 10%/random 20%/short-gap scenarios, median/causal-forward-fill recovery, sensor history, RUL replay, health labels, and a downloadable top-5/top-10 queue. |
| 32. End-to-end QA and final handover | PARTIAL until executed | This report and the runbook document the pipeline and known boundaries. Automated test execution, clean-clone smoke test, model artifact generation, dashboard launch, and final metrics review remain to be performed in a Python runtime. |

## Streamlit behavior added

- Loads the prepared preprocessing bundle and model artifacts from `artifacts/`.
- Provides model selection for XGBoost, Random Forest, LSTM, and the median baseline.
- Lets a user select an engine and observed cycle. Prediction and replay are restricted to the selected visible history.
- Supports clean data, random 10% and 20% sensor missingness, and a five-cycle sensor gap.
- Supports training-median replacement or causal per-engine forward fill.
- Displays the selected sensor's observed history, RUL prediction history, current predicted RUL, and a configurable demo health label.
- Calculates a fleet inspection queue on request, sorts by predicted RUL, and provides a CSV download.

Health thresholds are presentation rules (RUL > 60 = Healthy, 30 < RUL <= 60 = Warning, RUL <= 30 = Critical). They are not a separately trained classifier or validated real-world maintenance thresholds. The dataset is simulated degradation data; missingness is artificially injected for robustness evaluation.

## Reproducible runbook

Use Python 3.11 and run commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_models.py
python run_experiments.py
python scripts/run_checkpoint_experiments.py
python scripts/plot_results.py
python scripts/checkpoint_report.py
streamlit run app.py
```

If the raw FD001 data is already available under `data/raw/`, the download command may be skipped. The dashboard requires the prepared bundle and trained model artifacts; if they are missing, it displays instructions for generating them.

## QA acceptance checklist

- [ ] Install all requirements in a clean Python 3.11 environment.
- [ ] Run `pytest -q` and save the actual terminal output.
- [ ] Run the download/preparation/training/experiment pipeline without errors.
- [ ] Confirm `data_integrity_report.json` reports no train/validation engine overlap and no missing selected sensor values after recovery.
- [ ] Confirm the metrics files and clean comparison are regenerated from the same run.
- [ ] Launch `streamlit run app.py` and test all four model choices.
- [ ] Test all four sensor scenarios and both recovery methods.
- [ ] Move the observed-cycle slider and verify the replay chart never includes cycles beyond the cutoff.
- [ ] Generate top-5 and top-10 queues and confirm CSV download works.
- [ ] Review official test metrics and document LSTM generalization limitations without tuning on the official test labels.

## Known limitations and scientific guardrails

1. The code changes have not been runtime-executed in this session; passing tests are not asserted.
2. The dashboard's model artifacts are generated locally by the training pipeline and may be absent from a fresh clone because model/data artifacts are not committed.
3. The compact LSTM's official-test generalization has previously been identified as an improvement area in `PROJECT_STATUS.md`. Any model changes must be selected using validation prefixes, not official test labels.
4. Dashboard health labels and the inspection queue are a demonstration, not a live industrial monitoring service or a validated maintenance policy.
5. Metric CSVs already in the repository are evidence from an earlier run; regenerate them after any training or evaluation change before citing them as the final results.

## Final assessment

The previously missing application integration has been implemented in this branch, and the split manifest, integrity report, baseline, experiment runner, and metric output areas are covered by the existing pipeline. The remaining condition for marking **every task fully DONE** is execution-based verification: run tests and the full pipeline, launch the dashboard, complete the acceptance checklist, and update this report with actual outputs and results.

## Integration with Samir's safeguards (2026-10-13)

The merged application keeps the LSTM-only validated loading path from `LSTMInferenceAdapter` and raw official-test inputs from `official_test_raw_frame()`. Each RUL replay point generates its own cutoff-specific mask/recovery; charts use only recovered visible sensors and bounded predictions. Resource hashes invalidate changed artifacts, and finite-output checks precede clamping. One-observation histories are supported.

The shared top-K/CSV UI uses LSTM final observed snapshots: predict every engine in the selected namespace at its own final observed cycle, then rank by ascending predicted RUL with engine ID breaking ties for K=5/10. Snapshot cycle lengths differ; the selected-engine replay slider does not affect queue predictions or membership. Queue results are not cached. RF, XGBoost and median-baseline controls remain available but show a pending Task 25 adapter message, with no unverified fallback inference or queue. The earlier multi-model completion statements above describe the pre-integration branch; current all-model acceptance remains pending.

The root experiment manifest now lists 45 learned-model configurations plus one separate baseline, with clean learned-model recovery set to training median for native NaNs. Task 29 verification code/tests/report and independent QA tests/workflow are preserved. Historical Task 29 metrics are not a reproduction in this merge session. Real FD001 preparation/training/evaluation and real-checkpoint browser acceptance remain unverified.

Integration verification used `.venv\Scripts\python.exe -I -B -m pytest -q -p no:cacheprovider`: **686 passed**, no failures/skips. `pip check` passed. This was synthetic/regression verification, including the retained Task 29 tests; it was not a real FD001 preparation/training/evaluation run or real-checkpoint browser acceptance. No packages were installed; PyYAML and XGBoost are absent from the local .venv, which still blocks the corresponding real-data executable paths.
