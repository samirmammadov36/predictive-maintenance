# LSTM sequence input specification (checkpoint version)

- Input: selected sensor measurements only; engine ID is metadata and never a model feature.
- Window length: 30 cycles.
- Sequence-to-one target: capped training RUL at the final cycle of each training window.
- Short histories: left-pad with the first available **imputed** observation.
- Scaling: `StandardScaler` fitted on selected sensor columns from the 80 training engines only.
- Boundaries: a sequence may never cross an engine boundary.
- Validation: exactly one causal prefix per held-out engine, ending at floor(70% of its full trajectory length); the hidden future is used only by the evaluator to calculate true RUL.
- Official test: use each engine's full observed test prefix and compare one prediction with its official RUL label.
