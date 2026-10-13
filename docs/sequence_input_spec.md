# LSTM Sequence, Inference and Replay Specification

## Input and sequence structure

The LSTM uses the selected sensor measurements only. `engine_id` is metadata and is never used as a model feature.

- Window length: **30 cycles**
- Input size: **15 selected sensors**
- Output: one predicted Remaining Useful Life (RUL) value
- Sequence type: sequence-to-one regression
- Training target: capped RUL at the final cycle of each training window
- Engine boundaries are always preserved; sequences never contain observations from multiple engines

## Scaling

Sensor values are standardized using a `StandardScaler`.

The scaler is fitted only on the selected sensor columns from the **80 training engines**. Validation and test data are transformed using this frozen training-fitted scaler.

This prevents validation or test information from leaking into preprocessing.

## Short-history handling

If fewer than 30 cycles are available, the sequence is left-padded using the first available **imputed** observation.

Examples:

- 1 observed cycle → first observation repeated until a 30-cycle window is created
- 29 observed cycles → one padding row is added
- 30 observed cycles → no padding is required
- More than 30 cycles → only the latest 30 observations are retained

Automated tests verify the behavior at 1, 29, 30, and 31 observed cycles.

## LSTM architecture

The compact LSTM model uses:

- LSTM layer: **32 hidden units**
- Dense hidden layer: **16 units**
- Output layer: **1 RUL value**
- Optimizer: Adam
- Learning rate: **0.001**
- Batch size: **64**
- Maximum epochs: **50**
- Early-stopping patience: **5**

The selected model is frozen after validation and is not retrained for individual missingness scenarios.

## Validation protocol

Validation is performed at the engine level.

For each held-out validation engine, the visible history ends at:

`floor(0.70 × full trajectory length)`

Only this visible prefix is supplied to the model. The hidden future is used only by the evaluator to determine the true remaining life.

The model therefore cannot access observations occurring after the validation cutoff.

## Missing-sensor scenarios

The LSTM is evaluated under the same sensor-availability scenarios as the tabular models:

- **Clean** — original observed sensor values
- **Random 10%** — 10% of selected sensor measurements are hidden
- **Random 20%** — 20% of selected sensor measurements are hidden
- **Five-cycle sensor gap** — `sensor_11` is hidden during the final five cycles of the current observed prefix

Random scenarios use deterministic seeds **7, 17, and 27**.

The processing order is:

`visible prefix → sensor mask → imputation → scaling → 30-cycle sequence → LSTM prediction`

Missing values are recovered using either:

- training-median replacement, or
- causal per-engine forward fill with training-median fallback

No backward fill, future interpolation, or test-derived statistics are used.

## Causal inference and replay

LSTM inference uses only the selected engine's history available at the current observed cycle.

For replay, every historical prediction is generated from its own prefix.

For example, the prediction shown at cycle 30 uses only cycles 1–30, even if the dashboard slider is currently positioned at cycle 80.

This prevents future information from influencing earlier replay predictions.

The replay behavior is covered by automated tests.

## Official test evaluation

For each of the 100 official FD001 test engines:

1. use the complete observed test prefix;
2. apply the selected missingness and recovery configuration;
3. build the final 30-cycle sequence;
4. generate one RUL prediction;
5. compare it with the official uncapped RUL label.

Negative predictions are clamped to zero before scoring.

## Clean LSTM result

On the official clean test population, the LSTM achieved approximately:

- **MAE: 36.17 cycles**
- **RMSE: 45.23 cycles**

The LSTM performed worse than Random Forest and XGBoost on the official clean test set.

This result is reported as-is. The official test labels were not used for additional tuning or model selection.

## Limitations

The compact LSTM architecture was intentionally kept fixed for this project.

The project does not claim that this architecture is optimal for FD001. Its weaker official test performance indicates that the selected sequence model generalized less effectively than the tabular models in this experiment.

The LSTM is therefore included as a comparative modeling approach rather than being presented as the best-performing model.