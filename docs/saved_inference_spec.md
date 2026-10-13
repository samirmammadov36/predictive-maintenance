# Saved-model inference and multi-model replay (Task 25)

`RFInferenceAdapter`, `XGBoostInferenceAdapter`, `MedianBaselineInferenceAdapter` and the existing `LSTMInferenceAdapter` accept `(bundle, artifact_dir)` and are Streamlit-independent. `load_inference_adapter(bundle, artifact_dir, model_name="lstm")` dispatches `random_forest`, `xgboost`, `median_baseline` or `lstm`. XGBoost is imported only by its adapter. LSTM exclusively calls the existing `load_lstm_checkpoint()`; its compatibility checks are not bypassed.

All expose `predict(raw_trajectories, engine_id, observed_cycle, *, dataset_split=None, scenario="clean", recovery="median", seed=None)`. The shared boundary is select identity -> truncate cycle <= observed_cycle -> sort -> validate visible sensors -> shared mask -> recovery. Composite namespaces, invalid identities/cycles/cutoffs/options, native NaNs and no-mutation behavior follow the [LSTM contract](lstm_inference_spec.md).

RF/XGBoost then call the shared causal `build_tabular_features()`, select the final row in frozen feature order, apply the saved training-only scaler and predict once. Features remain current cycle, selected sensors and causal rolling means/differences; engine ID/settings/targets are excluded. No preprocessing fitting, training or tuning occurs.

The baseline calls `predict(1)`: its fitted training-target median is constant across engines, cycles and missingness options, although recovered sensor charts/masks reflect those options. The UI states this explicitly.

`InferenceResult` contains identity, actual final visible cycle, history length, finite non-negative scalar `rul_pred`, copied `chart_prefix` (recovered, unscaled identity/cycle/selected sensors only) and copied boolean `sensor_mask`. Non-sequence models return `sequence_window=None`, `padding_cycles=0`; LSTM retains its 30-cycle first-imputed-observation padding. `LSTMInferenceResult` remains a compatibility alias. One finite real output is required before clamping negatives.

## Consumed artifacts and compatibility limits

All paths below are under artifacts/. Medians/scalers come from the bundle; standalone exports and historical metric/selection tables are not consumed.

| Model | Files loaded and fingerprinted |
| --- | --- |
| LSTM | preprocessing/prepared_bundle.joblib, preprocessing/sequence_input_metadata.json, preprocessing/split_manifest.json, models/lstm.pt, models/lstm_checkpoint_metadata.json |
| RF | preprocessing/prepared_bundle.joblib, preprocessing/schema.json, models/random_forest.joblib |
| XGBoost | preprocessing/prepared_bundle.joblib, preprocessing/schema.json, models/xgboost.json |
| Median baseline | preprocessing/prepared_bundle.joblib, models/baseline.joblib |

RF/XGBoost check bundle sensor/feature order against `schema.json` and the shared feature builder's schema for the saved rolling window. A tiny schema-only probe is never fitted or saved as data/provenance. The scaler must be a fitted StandardScaler with matching count/names when present, finite mean/variance/scale, positive scale and fitted observation counts. Model input counts must match; saved names are compared exactly when available. RF requires a fitted RandomForestRegressor; XGBoost loads the saved regressor and checks booster count/names. Baseline requires a fitted MedianRULBaseline with a finite value. Each adapter freezes private preprocessing copies.

Current RF/XGBoost training uses scaled arrays, so model feature names may be absent. Counts cannot detect a same-width model trained in another column order. There is no RF/XGBoost checkpoint/preprocessing snapshot or pairing checksum like LSTM. These checks/fingerprints **do not prove model/preprocessing pairing**, training-only provenance, authenticity or independent reproduction of scaler/median statistics. Baseline has no sidecar authenticating its fitting targets. No missing provenance is fabricated. Publish coherent artifacts and perform real-data acceptance before presenting results as verified.

Missing selected model/schema files, concrete incompatibilities and legacy bundles without explicit raw official-test inputs fail clearly. The application never substitutes historical/dummy predictions or automatically prepares/downloads/trains.

## Replay, application and queues

`load_replay_resources(artifact_dir, model_name="lstm")` loads only the selected adapter and gets raw inputs exclusively through `official_test_raw_frame()`. `artifact_fingerprint(artifact_dir, model_name="lstm")` hashes exactly that model's consumed files. App cache keys include model, absolute path and hashes; changes/disappearance invalidate or reject loading, and files changed during loading are rejected. Missing LSTM artifacts do not block RF/baseline. Missing XGBoost package/files do not block the other paths.

`build_model_replay(resources, identity, observed_cycle, **options)` calls the adapter separately at each available cycle up to the cutoff, generating each point's own prefix/mask/recovery. Sensor and RUL plots receive bounded data only. `LSTMReplayResources` and `build_lstm_replay` remain compatible aliases of the shared interfaces; defaults remain LSTM.

`build_inspection_queue(resources, dataset_split, k, **options)` predicts **every engine** in the selected official-test namespace at its own final observed cycle before top-K selection. Sort ascending RUL, break ties by engine ID, K=5/10. Different final cycles make this a demonstration queue, not a synchronized fleet. Replay slider changes do not affect queue values/membership. No replay/queue results are cached; recalculate after UI reruns. CSV matches the displayed queue. `app.build_lstm_queue` remains an alias.

All four app selections connect to actual predictions, health display rules, recovered sensor/RUL charts, replay and queue/CSV. Clean forces median/no seed; random scenarios use 7/17/27; missingness supports median/causal forward fill. One-observation engines avoid an invalid slider. Provisional inline inference helpers were retired after reusable-adapter coverage; no parallel inference pipeline remains.

## Acceptance boundary

`tests/test_saved_inference.py` uses small fitted/saved RF, XGBoost and baseline artifacts with the existing synthetic LSTM fixture. Tests cover parity, frozen order, native NaNs/recovery, actual model input matrices and manually calculated causal features, future/engine isolation, per-cutoff replay, final-snapshot coverage, no refit/mutation, concrete schema/artifact mismatches, finite-before-clamp and artifact/dependency isolation.

`tests/test_multimodel_streamlit.py` exercises actual adapters using AppTest and inspects Matplotlib X/Y arrays and CSV bytes. It checks controls, independent cutoffs, bounded charts, complete namespace queues, top-5/top-10/ties, slider independence, early histories, model/options/artifact switching and missing artifacts. Existing LSTM checkpoint/inference/replay, masks/evaluation and Task 29 tests remain evidence. Synthetic fitted models/selection records are not real FD001 artifacts.

Real FD001 preparation, full training/evaluation, real artifact pairing/metrics, clean-clone execution and real-checkpoint browser acceptance remain unverified. Training architecture/config, historical results and Task 29 evidence are unchanged.

Final Task 25 verification: `.venv\Scripts\python.exe -I -B -m pytest -q -p no:cacheprovider` completed with **816 passed**, no failures/skips. Focused adapter/LSTM, AppTest/replay and strengthened adapter checks passed before the final run. Only XGBoost 3.4.1 was installed into the existing project .venv; existing NumPy/SciPy satisfied its dependencies. This is synthetic/regression evidence, not a real FD001 or real-checkpoint browser acceptance result.
