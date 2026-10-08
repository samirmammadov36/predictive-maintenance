# FD001 sensor missingness specification

Masks operate on selected raw sensors in one already-truncated observed engine prefix. The caller must truncate before calling these functions; they do not infer a cutoff or access hidden future rows. Rows must have finite positive integer cycles in increasing chronological order, without duplicates. Unique row indices are required; their labels need not be numerically sorted.

Engine identity is `(dataset_split, engine_id)` when `dataset_split` is present, otherwise `engine_id` for existing separate dataset frames. Empty or mixed-engine prefixes and missing identities are rejected. Selected sensors must form a non-empty, unique, ordered list of present FD001 sensor columns. Identifiers, cycles, operating settings and targets cannot be selected for masking.

- **S0:** clean input; no artificial mask.
- **S1:** hide `round(0.10 * selected_sensor_cells)` cells.
- **S2:** hide `round(0.20 * selected_sensor_cells)` cells.
- Random scenarios use seeds **7, 17, 27**. Python `round` preserves the existing nearest-integer, ties-to-even behavior. One permutation supplies both masks, so S1 is a subset of S2 for the same prefix, seed and frozen sensor order. The existing RNG seed remains `seed + 1009 * engine_id`; composite identity validation does not change the RNG algorithm. Masks for different prefix lengths need not be prefix-stable.
- **S3:** hide `sensor_11` in the final five observed cycles. If absent from the frozen selected sensor list, hide its first retained sensor. Histories shorter than five cycles hide all available trailing cycles. Existing explicit preferred-sensor and positive integer gap arguments remain supported.

Generate each scenario mask once and reuse the same boolean DataFrame across all models and both recovery methods (training medians and causal forward fill). Application requires exactly the prefix row labels and selected sensor labels, unique labels, and boolean cells with no missing values. A reordered mask with the same complete labels is aligned by label; missing, extra or ambiguous labels are rejected instead of filled with `False`. Prefix rows are never silently reordered, and input frames and masks are not mutated. Natural missing raw sensor values are allowed; these functions do not impute them.

Processing order: **truncate ? mask raw selected sensors ? impute ? features/scaling/sequences ? predict**. Random seeds must be non-negative integers; rates must be finite numbers satisfying `0 <= low_rate <= high_rate <= 1`; gap lengths must be positive integers.

These safeguards and synthetic tests validate mask construction and application. They do not establish experiment-runner integration, recovery correctness, model evaluation correctness or replay causality. In particular, the runner's use of already-imputed prepared frames remains a separate raw-before-imputation integration issue.
