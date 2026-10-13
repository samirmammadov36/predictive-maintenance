# Final Project Status

## Project state

The core predictive-maintenance implementation for NASA C-MAPSS FD001 is complete.

Implemented components:

- FD001 data loading and target generation
- Training-only sensor screening
- Training-median and causal forward-fill recovery
- Leakage-safe preprocessing
- Random Forest model
- XGBoost model
- Compact LSTM model
- Missing-sensor robustness experiments
- Reusable inference layer
- Causal cycle-by-cycle replay
- Streamlit dashboard
- Health-status display rules
- Top-K inspection queue
- Automated tests and verification scripts
- Final technical documentation

## Model evaluation

The frozen models were evaluated on the same official FD001 test population.

Clean official test performance:

- XGBoost: MAE ≈ 15.15, RMSE ≈ 19.59
- Random Forest: MAE ≈ 15.25, RMSE ≈ 21.30
- LSTM: MAE ≈ 36.17, RMSE ≈ 45.23

XGBoost achieved the strongest clean performance in this project.

## Missing-sensor robustness

The models were also evaluated under:

- clean sensor input
- random 10% sensor missingness
- random 20% sensor missingness
- a five-cycle gap in `sensor_11`

Two recovery methods were compared:

- training-median replacement
- causal forward fill with training-median fallback

For XGBoost, causal forward fill generally preserved performance better under the more severe missingness scenarios.

## Verification status

Current automated test suite:

- 45 tests passing

Additional verification includes:

- official test-label mapping
- final observed-cycle scoring
- saved Random Forest configuration consistency
- feature-schema consistency
- replay causality
- LSTM short-history padding
- 30-cycle LSTM window boundaries

## Final QA status

The final end-to-end runtime QA was completed successfully.

Verified steps:

- raw FD001 data available and loaded successfully
- preprocessing artifacts regenerated successfully
- all models retrained and saved successfully
- experiment manifest regenerated
- all 45 learned-model robustness evaluations completed
- result tables and plots regenerated
- Data / Random Forest verification passed
- automated test suite: **45 passed**
- Streamlit application launched successfully
- XGBoost, Random Forest, LSTM, and median baseline inference verified
- Clean, Random 10%, Random 20%, and five-cycle gap scenarios verified
- training-median and causal forward-fill recovery verified
- deterministic seeds 7, 17, and 27 verified
- replay causality manually verified
- LSTM early-cycle behavior verified
- Top-5 and Top-10 inspection queues verified
- inspection queue CSV export verified

The project is ready for the final demonstration and submission.

## Known limitations

- FD001 is simulated turbofan degradation data.
- Artificial missingness is a stress-test scenario, not observed production sensor failure.
- Healthy/Warning/Critical labels are dashboard display rules, not validated maintenance thresholds.
- The inspection queue is a prioritization demonstration, not a validated maintenance scheduling policy.
- The compact LSTM generalized worse than the tabular models in this experiment.
- The project does not claim a new machine-learning algorithm.
- Results should not be generalized beyond the experimental scope without additional validation.