# ML Engineer 4 — Implementation, Verification & Handover Report

**Repository:** `samirmammadov36/predictive-maintenance`  
**Dataset:** NASA C-MAPSS FD001  
**Final QA date:** 2026-10-13

## Executive summary

The ML Engineer 4 work covers repository integration, validation, baseline support, experiment execution, model comparison, Streamlit integration, and final end-to-end QA.

The project now provides a complete predictive-maintenance prototype for Remaining Useful Life (RUL) estimation using NASA C-MAPSS FD001.

The final implementation includes:

- reproducible FD001 preprocessing
- engine-level train/validation isolation
- training-only preprocessing statistics
- median RUL baseline
- Random Forest
- XGBoost
- compact LSTM
- missing-sensor robustness experiments
- reusable inference
- causal cycle-by-cycle replay
- Streamlit dashboard
- Top-K fleet inspection queue
- automated verification
- final runtime QA

The previously incomplete application integration has been completed. The Streamlit application loads the prepared preprocessing bundle and trained model artifacts, predicts RUL using only the selected engine's currently visible history, supports missing-sensor scenarios and recovery methods, displays sensor and RUL history, and provides a downloadable inspection queue.

The complete project pipeline and dashboard were successfully executed and manually verified during final QA.

---

## Task-by-task status

| Task area | Status | Evidence / notes |
| --- | --- | --- |
| 4. Repository structure and configuration | DONE | `config.yaml`, `src/predictive_maintenance/`, scripts, tests, results, documentation, and application entry points provide the shared project structure. |
| 11. Deterministic split and validation-prefix manifest | DONE | `scripts/prepare_data.py` regenerated the preprocessing artifacts using an 80/20 engine split with seed 42. Final run confirmed 80 training engines, 20 validation engines, and no train/validation overlap. |
| 12. Data-integrity verification | DONE | Final preparation completed with 16,779 training rows, 3,852 validation rows, 13,096 test rows, 15 selected sensors, and no missing selected-sensor values after median recovery. |
| 16. Median baseline and prediction adapter | DONE | Median baseline implementation is available under `src/predictive_maintenance/models/`. Baseline inference was also manually verified through Streamlit. |
| 20. Experiment manifest and execution | DONE | `python run_experiments.py` regenerated the experiment manifest. `python scripts/run_checkpoint_experiments.py` successfully executed the full robustness evaluation. |
| 24. Metrics and model comparison | DONE | `metrics.csv`, `predictions.csv`, `metrics_aggregated.csv`, `baseline.csv`, and `clean_comparison.csv` were regenerated from the final pipeline run. |
| 28. Streamlit application | DONE and runtime verified | The application successfully loads preprocessing/model artifacts and supports model selection, observed-cycle replay, missingness scenarios, recovery methods, sensor visualization, health labels, Top-K queue generation, and CSV export. |
| 32. End-to-end QA and final handover | DONE | Data preparation, model training, experiments, plots, reports, verification scripts, 45 automated tests, Streamlit launch, and manual dashboard QA all completed successfully. |

---

## Final preprocessing verification

The final `prepare_data.py` run produced:

```text
train_engines: 80
validation_engines: 20
official_test_engines: 100
selected_sensor_count: 15
train_validation_overlap: []
train_rows: 16779
validation_rows: 3852
test_rows: 13096
missing_after_median_train: 0
missing_after_median_validation: 0
missing_after_median_test: 0
```

The selected sensor set contains:

```text
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

This confirms that preprocessing completed without train/validation engine overlap or unrecovered missing selected-sensor values.

---

## Final model training verification

All three learned models were retrained from the regenerated preprocessing artifacts.

### Random Forest

Selected configuration:

```text
n_estimators = 250
max_depth = None
min_samples_leaf = 1
```

Validation performance:

```text
MAE  = 17.0276
RMSE = 22.5376
```

### XGBoost

Selected configuration:

```text
max_depth = 6
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
best_iteration = 42
```

Validation performance:

```text
MAE  = 17.7184
RMSE = 23.3740
```

### LSTM

The compact LSTM uses:

```text
30-cycle window
32 LSTM hidden units
16-unit Dense layer
1 RUL output
Adam learning rate = 0.001
batch size = 64
maximum epochs = 50
early-stopping patience = 5
```

Model configurations were selected using validation-prefix performance. Official test labels were not used for tuning.

---

## Official clean model comparison

The final regenerated clean official-test results were:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| XGBoost | 15.15 | 19.59 |
| Random Forest | 15.25 | 21.30 |
| LSTM | 36.17 | 45.23 |

XGBoost achieved the strongest clean official-test performance.

The compact LSTM generalized substantially worse than the tabular models. This result is retained and reported rather than tuning the architecture against official test labels.

---

## Missing-sensor robustness evaluation

The final experiment evaluated the frozen learned models under:

- Clean
- Random 10% missingness
- Random 20% missingness
- Five-cycle `sensor_11` gap

Random missingness was evaluated with deterministic seeds:

```text
7
17
27
```

Two recovery methods were compared:

- training-median replacement
- causal forward fill with training-median fallback

The final experiment generated:

```text
3 clean evaluations
36 random-missingness evaluations
6 short-gap evaluations
```

for a total of:

```text
45 learned-model evaluation rows
```

before aggregation.

The median RUL baseline is reported separately.

The final robustness results show that causal forward fill generally preserved performance better than median replacement under the more severe missingness scenarios, especially for XGBoost and the five-cycle sensor gap.

---

## Data / Random Forest verification

The dedicated verification script completed with:

```text
status: passed
official_test_engines: 100
official_target_rows: 100
selected_sensor_count: 15
feature_count: 46
```

Verified Random Forest configuration:

```text
n_estimators: 250
max_depth: None
min_samples_leaf: 1
```

Official clean Random Forest performance:

```text
MAE  = 15.24804
RMSE = 21.30231
```

The verification confirmed that:

- official test labels map one-to-one to official test engines
- target rows use each engine's final observed cycle
- the saved Random Forest feature count matches the frozen preprocessing schema
- saved Random Forest hyperparameters match the validation-selection record
- fresh saved-model clean metrics match `clean_comparison.csv`

The verification report is written to:

```text
results/metrics/data_rf_verification.json
```

---

## Streamlit behavior

The Streamlit dashboard:

- loads the prepared preprocessing bundle and saved model artifacts
- supports XGBoost, Random Forest, LSTM, and the median baseline
- allows engine selection
- allows observed-cycle replay cutoff
- supports Clean, Random 10%, Random 20%, and five-cycle sensor-gap scenarios
- supports training-median and causal-forward-fill recovery
- supports deterministic random-mask seeds 7, 17, and 27
- displays predicted RUL
- displays Healthy / Warning / Critical status
- displays sensor history under the selected availability scenario
- displays causal cycle-by-cycle RUL replay
- calculates Top-5 or Top-10 inspection queues
- sorts the queue by ascending predicted RUL
- provides inspection queue CSV download

Health thresholds are presentation rules:

```text
RUL > 60          → Healthy
30 < RUL <= 60    → Warning
RUL <= 30         → Critical
```

They are not a separately trained classifier or validated real-world maintenance thresholds.

---

## Replay causality verification

Replay was manually verified during final Streamlit QA.

For a fixed engine, model, sensor scenario, recovery method, and random-mask seed:

1. a historical cycle prediction was recorded;
2. the observed-cycle cutoff was extended farther into the engine history;
3. the earlier replay prediction was checked again.

The historical prediction remained unchanged.

This confirms that replay predictions use only information available at each replay cycle and do not change when later observations become visible.

Automated tests also cover replay causality.

---

## LSTM early-cycle verification

The LSTM was manually tested with the Clean scenario at observed cycles:

```text
1
29
30
31
```

All four cases produced valid predictions without runtime errors.

These cases verify the important 30-cycle sequence boundary:

- cycle 1 uses left padding
- cycle 29 requires one additional padding row
- cycle 30 provides one complete window
- cycle 31 uses the latest 30 observed cycles

Automated tests additionally verify the same 1, 29, 30, and 31-cycle boundary behavior.

---

## Final automated test result

The final test suite was executed after preprocessing, training, experiment, inference, and replay integration work.

Result:

```text
45 passed in 3.33s
```

The tests cover areas including:

- data loading
- RUL target generation
- train/validation split isolation
- imputation
- sensor masks
- feature generation
- sequence generation
- dashboard helpers
- Random Forest pipeline verification
- reusable inference behavior
- causal replay
- LSTM short-history padding
- 30-cycle LSTM boundary handling

---

## Final runtime QA

The following end-to-end workflow was completed successfully:

```text
raw FD001 data
→ preprocessing
→ model training
→ experiment manifest
→ robustness evaluation
→ result aggregation
→ plot generation
→ experiment report
→ RF/data verification
→ automated tests
→ Streamlit dashboard
```

Manual Streamlit QA verified:

- XGBoost clean inference
- XGBoost missing-sensor inference
- Random Forest inference
- LSTM inference
- median baseline inference
- Random 10% scenario
- Random 20% scenario
- five-cycle sensor gap
- training-median recovery
- causal-forward-fill recovery
- deterministic seeds 7, 17, and 27
- causal replay behavior
- sensor-history behavior under missingness
- health-status rules
- Top-5 inspection queue
- Top-10 inspection queue
- ascending queue ordering
- inspection queue CSV download

No blocking runtime errors were observed during final acceptance testing.

---

## Reproducible runbook

Use Python 3.11 and run commands from the repository root.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run automated tests:

```bash
python -m pytest -q
```

Download FD001 if raw data is not already available:

```bash
python scripts/download_data.py
```

Prepare the preprocessing artifacts:

```bash
python scripts/prepare_data.py
```

Train the models:

```bash
python scripts/train_models.py
```

Generate the shared experiment manifest:

```bash
python run_experiments.py
```

Run the robustness experiment:

```bash
python scripts/run_checkpoint_experiments.py
```

Generate plots:

```bash
python scripts/plot_results.py
```

Generate the experiment summary:

```bash
python scripts/checkpoint_report.py
```

Verify the Data / Random Forest pipeline:

```bash
python scripts/verify_data_rf_pipeline.py
```

Launch Streamlit:

```bash
python -m streamlit run app.py
```

If raw FD001 data is already present under `data/raw/`, the download step may be skipped.

The Streamlit dashboard requires the generated preprocessing bundle and trained model artifacts.

---

## QA acceptance checklist

- [x] Install project requirements in the working Python environment.
- [x] Regenerate preprocessing artifacts from raw FD001.
- [x] Confirm 80 training engines and 20 validation engines with no overlap.
- [x] Confirm no missing selected-sensor values remain after median recovery.
- [x] Retrain and save Random Forest, XGBoost, and LSTM.
- [x] Regenerate the experiment manifest.
- [x] Run all 45 learned-model robustness evaluations.
- [x] Regenerate metrics, predictions, aggregates, baseline, and clean-comparison outputs.
- [x] Regenerate result plots.
- [x] Regenerate the checkpoint experiment summary.
- [x] Run the Data / Random Forest verification script successfully.
- [x] Run the automated test suite: **45 passed**.
- [x] Launch the Streamlit dashboard.
- [x] Test all four model choices.
- [x] Test all four sensor-availability scenarios.
- [x] Test both recovery methods.
- [x] Verify deterministic seeds 7, 17, and 27.
- [x] Verify replay predictions do not use future information.
- [x] Verify LSTM early-cycle behavior at cycles 1, 29, 30, and 31.
- [x] Generate Top-5 and Top-10 inspection queues.
- [x] Confirm queue ordering by ascending predicted RUL.
- [x] Confirm inspection queue CSV download.
- [x] Document the LSTM generalization limitation without tuning against official test labels.

---

## Known limitations and scientific guardrails

1. FD001 is simulated turbofan degradation data and should not be presented as a real production fleet.

2. The missing-sensor scenarios are artificially injected stress tests. They do not represent measured production sensor-failure probabilities.

3. Model and preprocessing artifacts are generated locally by the pipeline and may not exist in a fresh clone until the preparation and training commands are executed.

4. The compact LSTM generalized substantially worse than Random Forest and XGBoost on the official clean test set. The official test labels were not used for additional architecture or hyperparameter tuning.

5. Healthy, Warning, and Critical are configurable dashboard display rules, not a separately trained classifier or validated maintenance thresholds.

6. The Top-K inspection queue is a demonstration of RUL-based prioritization. It is not a validated maintenance scheduling policy or synchronized live fleet.

7. Results should not be generalized beyond the experimental scope without additional real-world validation.

---

## Final assessment

The predictive-maintenance implementation is complete and has passed the final project acceptance workflow.

The previously missing application integration is now connected through reusable inference, and the complete data, preprocessing, training, evaluation, robustness, replay, dashboard, and verification paths have been executed successfully.

Final evidence includes:

```text
45 automated tests passed
100 official test engines evaluated
15 selected sensors
46 tabular features
45 learned-model robustness evaluation rows
successful Streamlit runtime QA
successful causal replay verification
successful Top-K and CSV verification
```

All required project components are ready for final demonstration and submission.