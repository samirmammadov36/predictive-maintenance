# ML Engineer 2 — Implementation and Verification Summary

## Responsibilities

This document covers the ML Engineer 2 work for the NASA C-MAPSS FD001 predictive-maintenance project:

- initial exploratory data analysis (EDA)
- training-only sensor screening
- missing-value recovery
- shared preprocessing contract
- XGBoost training and validation selection
- XGBoost clean and missing-sensor evaluation
- dashboard visualization and inspection-queue components

The executable notebook, source modules, tests, metric tables, and plots are the primary evidence. This document summarizes how those pieces fit together.

## Initial EDA

`notebooks/01_initial_eda.ipynb` records the raw FD001 structure, engine counts, missing-value checks, trajectory lengths, sensor variability, and example sensor histories.

The EDA is descriptive. It does not use official test RUL labels to choose preprocessing or model settings. Final sensor screening is performed later on the 80 training engines only.

## Sensor Screening

Sensor screening is implemented in `src/predictive_maintenance/screening.py` and is fitted only on the training-engine portion of the fixed 80/20 development split.

A sensor is removed when it is constant, or when it meets both configured near-constant conditions:

- unique-value ratio <= `0.0005`
- dominant-value fraction >= `0.995`

The frozen selected sensor list contains 15 sensors:

`sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

The following 6 sensors are removed because they are constant in the training-only screening data:

`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`

`sensor_6` has very low variability, but it does not satisfy the configured removal rule and is therefore retained.

The pipeline writes the full screening evidence to `data/processed/sensor_screening.csv` and freezes the selected sensor list in `artifacts/preprocessing/schema.json`. Running `python scripts/write_sensor_screening_report.py` converts that generated evidence into the commit-friendly `docs/sensor_screening_report.md`.

Validation and official test rows are not used to decide which sensors to keep.

Because `sensor_11` is retained, it is used directly for the five-cycle short-gap missingness scenario.

## Shared Tabular Features

Random Forest and XGBoost use the same causal tabular feature schema implemented in `src/predictive_maintenance/features.py`:

- current operating cycle
- current values of the selected sensors
- causal 10-cycle rolling mean for every selected sensor (`min_periods=1`)
- causal 10-cycle difference for every selected sensor

When fewer than 10 earlier cycles are available, the difference uses the earliest available observation. Engine ID is metadata and is not a model feature. Operating settings are excluded from this MVP.

## Missing-Value Recovery

Two recovery strategies are implemented in `src/predictive_maintenance/imputation.py`:

1. **Training median replacement** — each missing sensor value is replaced with that sensor's median fitted only on the 80 training engines.
2. **Causal per-engine forward fill** — a missing value uses the most recent earlier value from the same engine; if no earlier value exists, it falls back to the training median.

Backward fill, future interpolation, and test-derived medians are intentionally not used because they would introduce future/test information.

## Processing Contract

For missing-sensor robustness experiments, the processing order is fixed:

1. take the currently observed engine prefix
2. apply the artificial sensor mask to raw selected sensors
3. recover missing values using the selected recovery method
4. recompute causal tabular/sequence features
5. apply scalers fitted only on training engines
6. generate a non-negative RUL prediction with the frozen model

The models are not retrained for individual missingness scenarios.

## XGBoost Model Selection

XGBoost is trained as a regression model for Remaining Useful Life. Candidate configurations are compared only on clean validation prefixes. RMSE is the primary selection metric and MAE is the tie-breaker.

The selected validation configuration recorded in `results/metrics/validation_selection_summary.json` is:

- `max_depth = 6`
- `learning_rate = 0.05`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- best iteration: `42`
- validation MAE: `17.72` cycles
- validation RMSE: `23.37` cycles

The selected model and preprocessing are frozen before official test evaluation. Official test labels are not used to tune the model.

## Official Clean Test Result

On the official FD001 test population of 100 engines, XGBoost produced the strongest clean result among the three trained models:

- MAE: **15.15 cycles**
- RMSE: **19.59 cycles**

The clean comparison is stored in `results/metrics/clean_comparison.csv`.

## Missing-Sensor Robustness

XGBoost is evaluated under the agreed stress scenarios:

- random 10% missing sensor measurements, seeds 7/17/27
- random 20% missing sensor measurements, seeds 7/17/27
- a five-cycle short sensor gap

Both median replacement and causal forward fill are evaluated on identical masks. The aggregated XGBoost RMSE results are:

| Scenario | Recovery | RMSE | Change from clean RMSE |
| --- | --- | ---: | ---: |
| Clean | none | 19.59 | 0.00 |
| Random 10% | forward fill | 19.86 | +0.26 |
| Random 10% | median | 20.06 | +0.46 |
| Random 20% | forward fill | 19.65 | +0.05 |
| Random 20% | median | 21.40 | +1.80 |
| Short gap | forward fill | 19.95 | +0.36 |
| Short gap | median | 21.42 | +1.82 |

Small negative or near-zero changes under individual random masks are treated as finite-sample variation, not evidence that missing data improves the model. Overall, causal forward fill preserves XGBoost performance better than median replacement in the more severe 20% and short-gap cases.

FD001 contains simulated degradation data and the injected missingness is an artificial robustness stress test.

## Dashboard Components

Reusable Task 26 components live in `src/predictive_maintenance/dashboard.py`. They deliberately contain no model-loading or Streamlit assembly code so the final application owner can reuse them with the shared inference layer.

The module provides:

- `observed_prefix(...)` — returns only one engine's rows up to the selected observed cycle so downstream replay charts and prediction inputs can be restricted to the available history
- `plot_sensor_history(...)` — plots one or more sensor histories from the causal prefix
- `plot_rul_history(...)` — plots predicted RUL across observed replay cycles
- `health_label(...)` — applies the agreed display rules:
  - RUL > 60 -> Healthy
  - 30 < RUL <= 60 -> Warning
  - RUL <= 30 -> Critical
- `top_k_queue(...)` — returns K = 5 or 10 engines with the lowest predicted RUL, with ties broken by engine ID

Health labels are display rules, not a trained classifier or validated maintenance thresholds. The top-K queue is a demonstration and not an evaluated inspection policy or synchronized live-fleet view.

## Verification

`tests/test_dashboard_components.py` verifies:

- health-label boundaries at 30 and 60 cycles
- invalid threshold ordering
- replay prefixes never include another engine or a future cycle
- top-K ordering by predicted RUL and engine-ID tie-breaking
- duplicate-engine protection in final-snapshot queues
- sensor-history plots contain only supplied causal-prefix cycles
- RUL-history plots order cycles before visualization

The core pipeline tests continue to verify target generation, engine-level split isolation, causal forward fill, causal features, deterministic/nested masks, short-gap masking, and LSTM sequence boundaries/padding.
