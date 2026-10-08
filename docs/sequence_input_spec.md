# LSTM sequence input specification (checkpoint version)

- Input: selected sensor measurements only, in the supplied `sensor_columns` order; engine identity is metadata and never a model feature.
- Window length: 30 cycles.
- Sequence-to-one target: capped training RUL at the final cycle of each training window.
- Short histories: left-pad by repeating the first available **imputed** observation within that engine until the window is full; no zero padding or padding mask.
- Scaling: `StandardScaler` fitted on selected sensor columns from the 80 training engines only.
- Boundaries: a sequence may never cross an engine boundary. If `dataset_split` is present, identity is `(dataset_split, engine_id)`; otherwise `engine_id` alone preserves compatibility with separate dataset frames.
- Caller responsibility: impute sensors before sequence construction and explicitly truncate to the observed prefix before calling `build_prefix_window()`. It returns the final window of the supplied prefix and never infers a cutoff.
- Validation: observations are sorted by cycle within each identity. Empty input, missing identity values, duplicate cycles within an identity, non-finite/non-positive/fractional cycles, non-finite sensor values (including float32 overflow), and non-positive/non-integer window sizes are rejected with `ValueError`. Prefix input must contain exactly one engine identity.
- Validation: exactly one causal prefix per held-out engine, ending at floor(70% of its full trajectory length); the hidden future is used only by the evaluator to calculate true RUL.
- Official test: use each engine's full observed test prefix and compare one prediction with its official RUL label.

- Scaling schema: fitting receives only the 80 selected training engines. Sensor names must be valid, unique, present and ordered; transform checks fitted sensor order/count and never refits the scaler. Scaling inputs must already contain finite measurements.
- Preparation additionally writes `artifacts/preprocessing/sequence_input_metadata.json`: schema version, FD001 identity, ordered sensors, window/padding, actual split IDs and seed, validation prefix fraction, fitting RUL cap, training row count, scaler schema and the training source SHA-256 from `data_manifest.json`. Existing bundle/scaler/schema/split outputs remain unchanged.
- `validate_sequence_input_metadata()` compares the saved input contract with the supplied scaler, sensor order, window and expected split. It checks declared provenance and fitted row count; it does not authenticate scaler statistics, verify model weights or establish model/preprocessing pairing.
