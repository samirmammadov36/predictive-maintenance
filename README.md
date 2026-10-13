# Predictive Maintenance — NASA C-MAPSS FD001

Machine-learning prototype for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS FD001 dataset.

The project compares:

- Random Forest
- XGBoost
- Compact LSTM
- Median RUL baseline

In addition to clean-data performance, the project evaluates how the frozen models behave when sensor measurements are artificially missing and compares two causal recovery methods.

A Streamlit dashboard provides engine replay, RUL prediction, sensor visualization, health-status display rules, and a Top-K inspection queue.

---

## Project objective

The goal is to estimate how many operating cycles remain before an engine reaches the end of its simulated trajectory.

RUL is measured in **operating cycles**, not hours or calendar days.

The main experimental contribution is a reproducible robustness comparison under artificial sensor missingness.

The project does not claim a new machine-learning algorithm.

---

## Dataset

Dataset: **NASA C-MAPSS FD001**

The project uses:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Training RUL is calculated as:

```text
RUL = final engine cycle - current cycle
```

The uncapped RUL truth is preserved.

For model fitting, the training target is capped at:

```text
125 cycles
```

The official FD001 test set contains 100 engines and is evaluated using the provided official RUL labels.

---

## Leakage-safe validation

The project uses an engine-level split:

```text
80 training engines
20 validation engines
seed = 42
```

Rows from the same engine are never divided between training and validation.

For each validation engine, only the first:

```text
floor(70% of full trajectory length)
```

is visible to the model.

The hidden future is used only to calculate the true validation RUL.

`engine_id` is metadata and is never used as a model feature.

---

## Sensor selection

Sensor screening is performed using only the 80 training engines.

15 sensors are retained:

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

Six constant sensors are removed:

```text
sensor_1
sensor_5
sensor_10
sensor_16
sensor_18
sensor_19
```

---

## Tabular features

Random Forest and XGBoost use the same causal feature representation.

The final input contains 46 features:

```text
1 cycle feature
15 current sensor measurements
15 causal 10-cycle rolling means
15 causal 10-cycle differences
```

Operating settings are excluded from this project version.

Features are generated using only information available up to the current observed cycle.

---

## LSTM input

The LSTM uses 30-cycle sequences containing the 15 selected sensors.

Architecture:

```text
LSTM: 32 hidden units
Dense: 16 units
Output: 1 RUL value
```

Training configuration:

```text
window = 30
batch size = 64
Adam learning rate = 0.001
maximum epochs = 50
early-stopping patience = 5
```

Short histories are left-padded using the first available imputed observation.

Histories longer than 30 cycles use only the latest 30 observations.

Sequences never cross engine boundaries.

---

## Missing-sensor robustness experiment

Four sensor-availability scenarios are evaluated.

### S0 — Clean

Original observed sensor measurements.

### S1 — Random 10%

10% of selected sensor measurements are hidden.

### S2 — Random 20%

20% of selected sensor measurements are hidden.

### S3 — Five-cycle sensor gap

`sensor_11` is hidden during the final five observed cycles of the current prefix.

Random scenarios use deterministic seeds:

```text
7, 17, 27
```

---

## Missing-value recovery

Two recovery methods are compared.

### Training median

Each missing sensor value is replaced by the median calculated from the training engines only.

### Causal forward fill

A missing value is replaced using the most recent earlier observation from the same engine.

If no previous value exists, the training median is used as fallback.

The project never uses:

- backward fill
- future interpolation
- test-derived medians

The processing order is:

```text
visible prefix
→ sensor mask
→ imputation
→ feature generation / scaling
→ model prediction
```

---

## Model selection

Models are selected using clean validation-prefix performance.

Primary selection metric:

```text
RMSE
```

Tie-breaker:

```text
MAE
```

After validation, model configurations are frozen before official test evaluation and missingness experiments.

The official test labels are not used for model tuning.

---

## Clean official test results

| Model | MAE | RMSE |
| --- | ---: | ---: |
| XGBoost | 15.15 | 19.59 |
| Random Forest | 15.25 | 21.30 |
| LSTM | 36.17 | 45.23 |

XGBoost achieved the strongest clean official-test performance in this project.

The compact LSTM generalized substantially worse than the tabular models. This result is reported as-is rather than tuning the model against the official test labels.

---

## Robustness findings

The frozen models are evaluated under the same missing-sensor scenarios using the same official test population.

For XGBoost, causal forward fill generally preserved performance better than training-median replacement under the more severe missingness scenarios.

For example, under Random 20% missingness, forward fill kept XGBoost RMSE close to its clean result, while median replacement caused a noticeably larger degradation.

The experiment is intended as a controlled robustness stress test. Artificial missingness should not be interpreted as real production sensor-failure data.

---

## Streamlit dashboard

The local Streamlit application supports:

- XGBoost
- Random Forest
- LSTM
- Median baseline
- engine selection
- observed-cycle replay cutoff
- sensor-availability scenario selection
- missing-value recovery selection
- random-mask seed selection
- predicted RUL
- Healthy / Warning / Critical display status
- sensor-history visualization
- causal RUL replay
- Top-5 or Top-10 fleet inspection queue
- inspection queue CSV download

Replay predictions are causal.

For example, a prediction displayed for cycle 30 uses only information from cycles 1–30. Extending the replay slider farther into the engine history does not change the previously generated cycle-30 prediction.

Health display rules:

```text
RUL > 60          → Healthy
30 < RUL <= 60    → Warning
RUL <= 30         → Critical
```

These are demonstration rules only. They are not a separately trained classifier or validated maintenance thresholds.

---

## Repository structure

```text
.
├── app.py
├── config.yaml
├── run_experiments.py
├── requirements.txt
├── PROJECT_STATUS.md
├── experiment_manifest.csv
│
├── src/
│   └── predictive_maintenance/
│       ├── artifacts.py
│       ├── config.py
│       ├── constants.py
│       ├── dashboard.py
│       ├── data.py
│       ├── evaluation.py
│       ├── experiments.py
│       ├── features.py
│       ├── imputation.py
│       ├── inference.py
│       ├── masks.py
│       ├── prefixes.py
│       ├── prepared.py
│       ├── screening.py
│       ├── sequences.py
│       └── models/
│           ├── baseline.py
│           ├── lstm.py
│           ├── random_forest.py
│           └── xgboost_model.py
│
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── run_checkpoint_experiments.py
│   ├── plot_results.py
│   ├── checkpoint_report.py
│   ├── verify_data_rf_pipeline.py
│   └── write_sensor_screening_report.py
│
├── tests/
├── docs/
├── notebooks/
├── artifacts/
├── data/
└── results/
```

Generated raw data, preprocessing artifacts, trained model files, local environments, and secrets are intentionally excluded from Git.

---

## Environment setup

Python 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Run the tests

From the repository root:

```bash
python -m pytest -q
```

Current verified test-suite status:

```text
45 passed
```

The tests cover areas including:

- target generation
- split integrity
- imputation
- deterministic missingness masks
- feature generation
- sequence generation
- dashboard utilities
- RF/data verification
- causal replay
- LSTM short-history padding
- 30-cycle LSTM boundary behavior

---

## Full pipeline

Run the following commands from the repository root.

### 1. Download FD001

```bash
python scripts/download_data.py
```

If the raw FD001 files already exist under `data/raw/`, this step may be skipped.

### 2. Prepare the data

```bash
python scripts/prepare_data.py
```

This generates the frozen split, selected sensors, medians, scalers, feature schema, validation prefixes, and other preprocessing artifacts.

### 3. Train the models

```bash
python scripts/train_models.py
```

This trains and saves:

- median baseline
- Random Forest
- XGBoost
- compact LSTM

### 4. Generate the experiment manifest

```bash
python run_experiments.py
```

### 5. Run the robustness experiment

```bash
python scripts/run_checkpoint_experiments.py
```

This generates:

- `metrics.csv`
- `predictions.csv`
- `metrics_aggregated.csv`
- `baseline.csv`
- `clean_comparison.csv`

### 6. Generate plots

```bash
python scripts/plot_results.py
```

Plots are written under:

```text
results/plots
```

### 7. Generate the experiment summary

```bash
python scripts/checkpoint_report.py
```

### 8. Verify the Data / Random Forest pipeline

```bash
python scripts/verify_data_rf_pipeline.py
```

The verification checks:

```text
100 official test engines
100 official target rows
15 selected sensors
46 tabular features
saved Random Forest configuration consistency
official clean metric consistency
```

### 9. Launch the dashboard

```bash
python -m streamlit run app.py
```

Using `python -m streamlit` ensures that Streamlit is launched from the currently active Python environment.

---

## Expected evaluation output

The robustness experiment produces:

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

The median baseline is reported separately.

All official model evaluations use the same 100-engine test population.

---

## Reusable inference

Reusable inference functionality is implemented in:

```text
src/predictive_maintenance/inference.py
```

It handles:

- preprocessing bundle loading
- saved model loading
- deterministic sensor corruption
- missing-value recovery
- Random Forest inference
- XGBoost inference
- LSTM inference
- baseline inference
- scenario-aware prediction
- causal cycle-by-cycle replay

The Streamlit application uses this shared inference layer instead of duplicating model-loading and prediction logic inside the UI.

---

## Documentation

Additional technical documentation is available under `docs/`, including documentation for:

- sensor screening
- preprocessing and XGBoost
- Data / Random Forest verification
- LSTM sequence generation
- LSTM inference and replay
- robustness scenarios
- integration and QA

See `PROJECT_STATUS.md` for the final project status.

---

## Final QA status

The full project pipeline and dashboard were successfully validated end-to-end.

Automated verification:

```text
45 passed
```

The following runtime workflow was successfully completed:

```text
raw FD001 data
→ preprocessing
→ model training
→ experiment generation
→ robustness evaluation
→ plots and reports
→ verification scripts
→ Streamlit dashboard
```

Manual Streamlit QA verified:

- XGBoost inference
- Random Forest inference
- LSTM inference, including early-cycle behavior at cycles 1, 29, 30, and 31
- median baseline inference
- Clean, Random 10%, Random 20%, and five-cycle sensor-gap scenarios
- training-median and causal forward-fill recovery
- deterministic random-mask seeds 7, 17, and 27
- causal replay behavior
- sensor-history behavior under missingness
- health-status display rules
- Top-5 and Top-10 inspection queues
- ascending RUL ordering in the inspection queue
- inspection queue CSV download

No blocking runtime errors were observed during the final acceptance test.

The project is ready for demonstration and submission.

---

## Limitations

- FD001 is a simulated turbofan degradation dataset.
- Artificial missingness is a controlled robustness experiment, not real production sensor-failure data.
- The dashboard health labels are demonstration rules, not validated industrial maintenance thresholds.
- The inspection queue is a prioritization demonstration, not a validated maintenance scheduling policy.
- The compact LSTM generalized substantially worse than Random Forest and XGBoost in this experiment.
- The project does not claim a novel machine-learning algorithm.
- Results should not be generalized beyond the experimental scope without additional validation.