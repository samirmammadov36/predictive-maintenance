# Samir — ML Engineer 3: implementation and acceptance evidence

This summary covers Tasks 3, 9, 10, 15, 19, 23, 27 and 31 for NASA C-MAPSS FD001. Review baseline: `feature/samir` at `d9a98b6d7836244bd7ba9e1f3e016a0638673e8a`. RUL is in **operating cycles**.

At the Task 31 documentation-review baseline, the full suite was **previously reported as 634 passed** before publication of Task 27 part 2. It was not rerun for this documentation review. Existing test bodies, code and `config.yaml` were inspected now; test names below identify synthetic/regression evidence, not successful real FD001 execution.

## Task responsibilities

| Task | Implementation and responsibility | Implementation / execution status |
| --- | --- | --- |
| 3 | `sequences.py`, [sequence input contract](sequence_input_spec.md): selected sensor inputs, frozen order, 30-observation window, first-imputed-observation padding | Implemented; final documentation consistency reviewed; real inputs unverified |
| 9 | `scripts/prepare_data.py`, `sequences.py`, `sequence_metadata.py`: training-only scaler, ordered schema, split/source metadata | Implemented and covered synthetically; real preparation unverified |
| 10 | `sequences.py`: chronological arrays, engine boundaries, final-observation target alignment, invalid-input rejection | Implemented and covered synthetically |
| 15 | `sequences.py`, `masks.py`, [missingness contract](missingness_spec.md): windows and deterministic nested masks | Implemented and covered synthetically; real robustness execution unverified |
| 19 | `models/lstm.py`, `models/lstm_checkpoint.py`, `scripts/train_models.py`: fixed LSTM, early stopping, checkpoint selection/save/load | Implemented with synthetic optimization and serialization evidence; full FD001 training unverified |
| 23 | `prepared.py`, `evaluation.py`, `experiments.py`, preparation and `scripts/run_checkpoint_experiments.py`: raw-prefix evaluation and official engine coverage | Implemented; synthetic runner tests use instrumented models; real evaluation unverified |
| 27 | `inference.py`, `replay.py`, `app.py`, shared `dashboard.py` plots: validated saved LSTM, explicit cutoff, Streamlit replay | LSTM adapter/application implemented and covered with actual synthetic saved checkpoints and AppTest; real-checkpoint browser acceptance unverified |
| 31 | This summary and the five linked specifications: boundaries, masks, metadata, coverage, causal replay and acceptance mapping | Documentation/evidence review complete; real-data acceptance remains open |

Source paths in this table are relative to `src/predictive_maintenance/` unless prefixed by `scripts/` or named `app.py`.

## Fixed training and validation contract

`config.yaml` sets `split.train_fraction=0.8`, `project.seed=42`, `split.validation_prefix_fraction=0.70`, `project.rul_cap=125`, and both sequence/LSTM windows to 30. Split whole engines from the 100 development trajectories into 80 training and 20 validation engines; never split individual rows. Screening, medians and both scalers are fitted on the training-engine frame only.

`data.add_training_rul()` preserves uncapped `rul_true = final_cycle - cycle` and caps only `rul_target` for fitting at 125. Each validation engine contributes one clean prefix ending at chronological observation `max(1, floor(0.70 * trajectory row count))`; its endpoint is that observation's cycle. Hidden future engine/cycle metadata supports offline prefix-provenance checking and evaluator truth, while hidden sensor rows never enter validation model inputs.

The fixed `RULLSTM` has one unidirectional LSTM layer with 32 hidden units, Dense 16 with ReLU, and one linear output. Defaults are batch size 64, Adam learning rate 0.001, at most 50 epochs, patience 5 and seed 42. Each epoch is selected by clean validation-prefix `(RMSE, MAE)`: lower RMSE wins, then lower MAE on an exact RMSE tie; an exact tie in both keeps the earlier checkpoint. A MAE tie-break improvement resets patience; a tie in both consumes patience. Restored weights, selected epoch, sidecar and training summary use the same selection record. Official test results never participate in selection, refitting or missingness retraining.

## Inputs, masks and recovery

LSTM inputs contain only selected sensors in the frozen metadata order; engine IDs, cycles, settings and targets are not LSTM features. The actual retained list comes from training-only screening and saved metadata; historical sensor lists are not substituted for missing artifacts.

`build_training_windows()` sorts within engine identity and aligns each target to its window's final observation. Identity is `(dataset_split, engine_id)` when available, otherwise `engine_id` for separate frames. `build_prefix_window()` accepts exactly one already-truncated, imputed prefix and returns its final window. Both produce float32 sensor arrays. For 1/5/29 observations, padding lengths are 29/25/1; 30/31 observations need no padding, and the 31-observation window uses observations 2..31. Padding repeats the first available recovered observation after saved scaling; it never crosses engines. No zero padding or padding mask is introduced. Empty prefixes, missing identities, duplicate/invalid cycles, invalid windows and non-finite sensor values fail.

The evaluation/inference order is **raw observed prefix -> selected-sensor mask -> recovery -> saved features/scaling/sequences -> prediction**. Native NaNs remain raw until recovery. S0 adds no artificial missingness and uses training medians. S1/S2 hide `round(0.10 * cells)` / `round(0.20 * cells)`, using Python ties-to-even rounding, seeds 7/17/27 and the existing `seed + 1009 * engine_id` RNG. The 10% mask is nested in 20% for the same prefix, seed and sensor order, and identical masks are used across models and recovery methods. Selected cells include cells already containing native NaNs; mask counts are not counts of newly missing values.

S3 hides sensor_11 in the final five observed rows, or the first frozen sensor if sensor_11 was screened out; shorter prefixes hide their available tail. Missingness uses training-median recovery or causal forward fill with training-median fallback. Mask helpers require one chronological prefix and exact boolean row/sensor labels. The shared forward-fill helper groups by engine_id and relies on caller chronology/isolation: the runner and adapter supply one sorted engine namespace at a time. Do not apply that helper directly to unsorted or combined namespaces. No backward fill or future interpolation is used.

## Evaluation, checkpoint loading and replay

[Official evaluation](official_test_evaluation_spec.md) consumes `official_test_raw_frame(bundle)`, never the median-imputed compatibility frame `official_test_df`. Legacy bundles without explicit raw inputs fail with regeneration instructions. The real entrypoint requires engine IDs 1..100, exactly one uncapped finite non-negative official label per engine, and one finite prediction aligned to each final observed cycle. Negative finite predictions are clamped to zero; missing/extra/duplicate identities, label/cycle mismatches and non-finite outputs fail.

The three trained models each have 15 configurations: 1 clean, 12 random (2 rates x 3 seeds x 2 recoveries), 2 short-gap. This yields 45 metric rows, plus a separate clean median baseline. MAE/RMSE, clean deltas and random-seed mean/sample-standard-deviation retain their meanings. LSTM predictions record history/window/used-history/padding lengths; metrics record evaluated, padded/unpadded engine counts and min/max observed history. Short histories are retained.

[Checkpoint loading](lstm_checkpoint_spec.md) reuses Task 9 metadata: schema/dataset/padding, ordered sensors, window, actual split IDs/seed, prefix fraction, fitting cap, row count, scaler schema and training-source checksum. `load_lstm_checkpoint()` checks bundle/split agreement, independently enforces fraction 0.70 and exact validation-prefix keys/endpoints, compares the saved input snapshot, enforces fixed architecture, validates selection/configuration and checkpoint SHA-256, then strictly loads weights. Missing legacy metadata fails; none is fabricated. These checks establish declared compatibility and accidental file pairing, not authentication or independent reproduction of scaler/median statistics or training.

[Inference/replay](lstm_inference_spec.md) selects an engine namespace, retains `cycle <= observed_cycle`, then sorts before sensor processing. Hidden future sensor values, targets, settings and other engines do not enter prediction or charts. `chart_prefix` is a copied recovered, unscaled visible sensor frame containing only identity/cycle/selected sensors. Sensor charts may display the full visible history; model windows use its final 30 observations.

Every RUL replay point at cycle c calls the actual saved adapter with its own cutoff c, mask and recovery; no points after the selected cutoff are computed. App controls map clean to median/no seed and random scenarios to 7/17/27. One-observation histories avoid an invalid slider. Resources are cached by absolute artifact path and hashes of all five consumed files; replay results are recomputed for engine/options/cutoff changes. Missing/incompatible artifacts show errors without substitute predictions. Health labels are shared display rules. Only LSTM is connected; other model controls report pending Task 25 integration. The merged LSTM queue ranks engines within the selected namespace at the selected cutoff and offers CSV download; no queue results are cached.

## Acceptance evidence

Test names are existing functions, including their parameterized cases. No document-wording tests were added. Paths below are under `tests/`.

| Requirement / task | Existing test module and exact names |
| --- | --- |
| Engine split and RUL truth/cap (3/9/19) | `test_core_pipeline.py::test_engine_split_has_no_overlap_and_is_reproducible`; `test_rul_generation`; `test_sequence_metadata.py::test_sequence_metadata_checks_actual_80_engine_training_identity` |
| Training-only statistics, extreme held-out values, no refit (9) | `test_core_pipeline.py::test_sequence_scaler_uses_training_statistics_and_never_refits`; `test_official_test_evaluation.py::test_preparation_stores_raw_inputs_and_rejects_extra_missing_labels` |
| Sensor order, float32, 1/5/29/30/31 padding/final windows (3/10/31) | `test_core_pipeline.py::test_sequence_history_lengths_and_final_window`; `test_sequence_explicit_sensor_order_overrides_dataframe_order`; `test_lstm_inference.py::test_actual_saved_model_history_padding_sensor_order_and_parity` |
| Boundaries and every target alignment (3/10/31) | `test_core_pipeline.py::test_sequence_windows_never_mix_engines_and_pad_left`; `test_sequence_composite_identity_keeps_same_engine_id_separate`; `test_sequence_prefix_rejects_mixed_engine_identities` |
| Chronology and invalid inputs (10/31) | `test_core_pipeline.py::test_sequence_unsorted_rows_are_chronological_and_do_not_mutate_input`; `test_sequence_rejects_duplicate_cycles_within_identity`; `test_sequence_rejects_invalid_cycles`; `test_sequence_rejects_missing_engine_identity`; `test_sequence_rejects_empty_input`; `test_sequence_rejects_invalid_window_sizes`; `test_sequence_rejects_nonfinite_sensor_values` |
| Window causality / explicit caller truncation (3/10/31) | `test_core_pipeline.py::test_sequence_causality_and_explicit_prefix_truncation` |
| Metadata/save-load/schema enforcement (9) | `test_sequence_metadata.py::test_sequence_scaler_save_load_and_metadata_json_round_trip`; `test_sequence_metadata_rejects_incompatible_supplied_inputs`; `test_core_pipeline.py::test_sequence_scaling_rejects_reordered_sensors` |
| Reproducibility, counts/nesting, reuse and alignment (15/31) | `test_sensor_masks.py::test_random_counts_nesting_reproducibility`; `test_existing_random_mask_positions_are_preserved`; `test_shared_masks_can_be_reused_without_mutation`; `test_exact_label_alignment_with_reordered_mask_and_nondefault_indices`; `test_malformed_masks_are_not_silently_filled` |
| S3 short histories and frozen fallback (15/31) | `test_sensor_masks.py::test_gap_preferred_fallback_and_observed_tail`; `test_lstm_inference.py::test_short_gap_uses_first_frozen_sensor_when_sensor_11_absent` |
| Mask prefix isolation / invalid prefix rejection (15/31) | `test_sensor_masks.py::test_explicit_prefix_is_isolated_from_future_sensor_changes`; `test_invalid_prefix_rejected_by_all_operations`; `test_invalid_identity_and_cycle_values` |
| Fixed architecture/defaults and selection/patience/restoration (19) | `test_lstm_checkpoint.py::test_defaults_architecture_and_real_synthetic_validation_metrics`; `test_selected_epoch_restored_weights_and_patience_agree`; `test_training_script_summary_saved_weights_and_epoch_agree` |
| Checkpoint parity, incompatible/legacy rejection (19/27) | `test_lstm_checkpoint.py::test_save_load_prediction_parity_and_provenance`; `test_incompatible_checkpoint_or_inputs_fail_clearly`; `test_existing_saved_model_path_uses_validated_lstm_loader` |
| Independent 0.70 fraction and actual endpoints/keys (19) | `test_lstm_checkpoint.py::test_input_metadata_checks_each_chronological_validation_prefix`; `test_input_metadata_rejects_altered_declared_prefix_fraction`; `test_input_metadata_rejects_altered_actual_prefix_endpoint`; `test_input_metadata_rejects_missing_or_extra_prefix_engine_keys`; `test_offline_prefix_check_does_not_expose_hidden_future_sensors` |
| Raw native NaNs, shared masks, 15/45 configurations, no refit (23) | `test_official_test_evaluation.py::test_raw_mask_recovery_shared_inputs_and_all_configurations`; `test_saved_bundle_roundtrip_preserves_native_raw_nans`; `test_old_bundles_fail_actionably_before_loading_models` |
| Causal recovery and engine isolation (23/27/31) | `test_official_test_evaluation.py::test_unsorted_runner_prefixes_are_causal_and_never_cross_engines`; `test_lstm_inference.py::test_native_nans_raw_mask_then_recovery_with_frozen_sensor_fallback` |
| Official 100 engines, label/cycle/output failures (23) | `test_official_test_evaluation.py::test_default_official_evaluation_requires_exactly_100_engines`; `test_real_entrypoint_enforces_100_engine_requirement`; `test_input_coverage_and_labels_are_validated`; `test_prediction_coverage_and_alignment_failures`; `test_model_must_emit_exactly_one_finite_output` |
| Uncapped truth, clamp, coverage/deltas/seed aggregation (23) | `test_official_test_evaluation.py::test_scoring_uncapped_labels_final_cycles_negative_clamp_and_coverage`; `test_short_gap_fallback_and_single_cycle_padding` |
| Actual saved inference future perturbation / other engine isolation (27/31) | `test_lstm_inference.py::test_future_perturbation_actual_prediction_mask_chart_and_padding`; `test_identical_options_repeat_and_other_engine_cannot_affect_inference`; `test_composite_identity_with_overlapping_row_indices_and_ambiguous_selection` |
| No mutation/refit, snapshot and finite-before-clamp (27) | `test_lstm_inference.py::test_no_mutation_no_refit_load_once_and_private_preprocessing_snapshot`; `test_actual_model_output_is_finite_before_negative_clamping` |
| Each replay cutoff and actual plotted X/Y future isolation (27/31) | `test_lstm_replay.py::test_replay_early_cycles_padding_and_each_independent_cutoff`; `test_replay_future_perturbation_preserves_actual_sensor_and_rul_plot_lines` |
| Streamlit actual inference, controls, early cycles and both plots (27/31) | `test_streamlit_lstm.py::test_streamlit_start_slider_real_inference_and_actual_plot_data`; `test_control_mapping_and_each_prefix_specific_prediction`; `test_one_cycle_engine_avoids_invalid_slider_and_pads`; `test_future_raw_artifact_changes_reload_but_preserve_both_plots_at_cutoff` |
| App cache/options/namespaces/errors and pending model status (27) | `test_streamlit_lstm.py::test_checkpoint_load_cached_across_slider_and_invalidated_by_changed_weights`; `test_engine_options_do_not_reuse_stale_predictions_or_plots`; `test_namespace_engine_selector_keeps_identical_ids_isolated`; `test_missing_incompatible_artifacts_show_message_without_prediction_or_exception`; `test_other_model_controls_report_integration_status_without_dummy_outputs` |

Runner tests instrument RF/XGBoost/LSTM outputs to verify data flow and scoring; they do not prove real model quality or all-model artifact loading. Checkpoint selection tests combine real synthetic optimization with controlled validation metrics and isolate optional non-LSTM training/config loading. Inference and AppTest tests use actual temporary saved RULLSTM checkpoints and inspect model inputs/Matplotlib line arrays. The synthetic fixtures are not real FD001 artifacts.

## Required files, generated artifacts and actual run path

From the repository, supply original `data/raw/train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt`. They represent 100 development engines, 100 observed official-test prefixes and 100 final-prefix RUL labels. Use project dependencies within `requirements.txt`; [README run commands](../README.md#samir-lstm-implementation-and-run-path) distinguish the executable scripts from root utilities.

| Producer | Expected outputs |
| --- | --- |
| `scripts/prepare_data.py` | `data/processed/sensor_screening.csv`, `data_integrity_report.json`; `artifacts/preprocessing/prepared_bundle.joblib` (training/validation frames, medians/scalers and explicit raw official-test field), `training_medians.joblib`, `sequence_scaler.joblib`, `tabular_scaler.joblib`, `schema.json`, `split_manifest.json`, `data_manifest.json`, `sequence_input_metadata.json` |
| `scripts/train_models.py` | `artifacts/models/baseline.joblib`, `random_forest.joblib`, `xgboost.json`, `lstm.pt`, `lstm_meta.joblib`, `lstm_checkpoint_metadata.json`; `results/metrics/random_forest_validation_search.csv`, `xgboost_validation_search.csv`, `lstm_training_history.csv`, `validation_selection_summary.json` |
| `scripts/run_checkpoint_experiments.py` | `results/metrics/metrics.csv`, `predictions.csv`, `metrics_aggregated.csv`, `baseline.csv`, `clean_comparison.csv` |
| `scripts/plot_results.py` | `results/plots/clean_rmse_comparison.png`, model-specific `*_robustness_delta_rmse.png` and `*_clean_true_vs_pred.png` |

The current review found only `.gitkeep` in `data/raw`, `artifacts/preprocessing` and `artifacts/models`. Real preparation, checkpoint compatibility, full training, official evaluation and real-checkpoint browser acceptance remain **unverified**. The existing .venv has no PyYAML or XGBoost distribution, so real preparation/training/all-model evaluation also require dependency setup; none was installed here.

Historical CSVs/plots and teammate reports describe recorded earlier results. Their existence or a commit does not establish that they came from the current checkpoint and preprocessing. A clean real-data run must retain the actual manifests, compatible checkpoint sidecars, training log, coverage/predictions, metrics and browser acceptance evidence. Do not relabel historical scores as current results.

## Remaining teammate dependencies

| Owner | Dependency / impact |
| --- | --- |
| Rafiq — data/RF | Verify original FD001 files/labels and shared split; provide the frozen RF artifact and later RF inference adapter |
| Cahid — features/imputation/XGBoost/visualizations | Keep causal recovery/features and frozen schema compatible; provide XGBoost artifacts/adapter; reuse existing plots with bounded data |
| Lalə — runner/comparison/Streamlit/end-to-end QA | Provide the missing Task 25 adapters and their non-LSTM queue paths; preserve root manifest alignment with the executable 15-configuration matrix; coordinate real-data/browser acceptance with Samir |
| Samir — LSTM | Verify real training/selection/checkpoint/evaluation and replay acceptance after data/dependencies are available; preserve the explicit-cutoff and chart-prefix interfaces |

No shared interface or production code changed in this documentation review. Any future shared-interface change needs agreement on namespace identity, raw-prefix recovery order, bounded chart data and artifact provenance.

Integration verification used `.venv\Scripts\python.exe -I -B -m pytest -q -p no:cacheprovider`: **686 passed**, no failures/skips. `pip check` passed. This was synthetic/regression verification, including the retained Task 29 tests; it was not a real FD001 preparation/training/evaluation run or real-checkpoint browser acceptance. No packages were installed; PyYAML and XGBoost are absent from the local .venv, which still blocks the corresponding real-data executable paths.
