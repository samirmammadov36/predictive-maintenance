"""Validated LSTM dashboard/replay; other model adapters await Task 25."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.dashboard import (
    HEALTH_STATUS_NOTE, QUEUE_NOTE, health_label, plot_rul_history, plot_sensor_history, top_k_queue,
)
from predictive_maintenance.replay import artifact_fingerprint, build_lstm_replay, load_replay_resources
from predictive_maintenance.inference import _positive_integer, _visible_prefix
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.masks import apply_sensor_mask, nested_random_masks, short_gap_mask

SCENARIOS = {"S0 Clean": "clean", "S1 Random 10%": "random_10",
             "S2 Random 20%": "random_20", "S3 Short gap": "short_gap"}
RECOVERIES = {"Training median": "median", "Causal forward fill": "forward_fill"}


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_resources(artifact_dir: str, fingerprint: tuple):
    resources = load_replay_resources(artifact_dir)
    if artifact_fingerprint(artifact_dir) != fingerprint:
        raise ValueError("Saved artifacts changed while loading; retry after preparation finishes")
    return resources




# Retained teammate preparation/prediction helpers for the future Task 25 adapter.
# The application does not enable these non-LSTM paths without artifact compatibility checks.
def corrupt_prefix(prefix: pd.DataFrame, bundle, scenario: str, seed: int) -> pd.DataFrame:
    """Apply a deterministic sensor-missingness scenario to the visible prefix only."""
    sensors = bundle.selected_sensors
    if scenario == "Clean":
        return prefix.copy()
    if scenario in {"Random 10%", "Random 20%"}:
        low_mask, high_mask = nested_random_masks(prefix, sensors, seed)
        mask = low_mask if scenario == "Random 10%" else high_mask
    elif scenario == "Five-cycle sensor gap":
        mask, _ = short_gap_mask(prefix, sensors, preferred_sensor="sensor_11", gap_cycles=5)
    else:
        raise ValueError(f"Unknown missingness scenario: {scenario}")
    return apply_sensor_mask(prefix, mask, sensors)


def recover_prefix(prefix: pd.DataFrame, bundle, recovery: str) -> pd.DataFrame:
    if recovery == "Training median":
        return bundle.medians.median_fill(prefix, bundle.selected_sensors)
    if recovery == "Causal forward fill":
        return bundle.medians.causal_forward_fill(prefix, bundle.selected_sensors)
    raise ValueError(f"Unknown recovery method: {recovery}")


def predict_prefix(model_name: str, model, prefix: pd.DataFrame, bundle) -> float:
    """Retained non-LSTM helper; it does not verify saved artifact provenance."""
    if prefix.empty:
        raise ValueError("At least one observed cycle is required.")
    if model_name in {"Random Forest", "XGBoost"}:
        features, _ = build_tabular_features(prefix, bundle.selected_sensors, bundle.rolling_window)
        latest = features.iloc[[-1]][bundle.feature_columns]
        output = model.predict(bundle.tabular_scaler.transform(latest))
    elif model_name in {"Median", "Median baseline"}:
        output = model.predict(1)
    elif model_name == "LSTM":
        raise ValueError("LSTM requires the validated LSTMInferenceAdapter")
    else:
        raise ValueError(f"Unknown prediction model: {model_name}")
    if np.iscomplexobj(output):
        raise ValueError("Prediction must be one finite real value before negative clamping")
    values = np.asarray(output, dtype=float).reshape(-1)
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError("Prediction must be one finite value before negative clamping")
    return max(0.0, float(values[0]))


def predict_with_scenario(model_name, model, prefix, bundle, scenario, recovery, seed) -> float:
    corrupted = corrupt_prefix(prefix, bundle, scenario, seed)
    recovered = recover_prefix(corrupted, bundle, "Training median" if scenario == "Clean" else recovery)
    if not np.isfinite(recovered[bundle.selected_sensors].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite sensor values remain after recovery.")
    return predict_prefix(model_name, model, recovered, bundle)


def build_replay(engine, observed_cycle, model_name, model, bundle, scenario, recovery, seed):
    """Retained helper: truncate and regenerate masks separately at every cycle."""
    if engine.empty or engine.engine_id.nunique() != 1:
        raise ValueError("Replay requires one non-empty engine trajectory")
    visible, _ = _visible_prefix(
        engine, bundle.selected_sensors, int(engine.engine_id.iloc[0]),
        _positive_integer(observed_cycle, "observed_cycle"), None,
    )
    rows = []
    for cycle in visible["cycle"].tolist():
        cycle_prefix = visible[visible["cycle"] <= cycle].copy()
        corrupted = corrupt_prefix(cycle_prefix, bundle, scenario, seed)
        recovered = recover_prefix(corrupted, bundle, "Training median" if scenario == "Clean" else recovery)
        rows.append({"cycle": int(cycle), "rul_pred": predict_prefix(model_name, model, recovered, bundle)})
    return recovered, pd.DataFrame(rows)

def build_lstm_queue(resources, dataset_split, k, *, scenario="clean", recovery="median", seed=None):
    """Rank every engine in one namespace at its own final observed cycle."""
    rows = []
    for (split, engine), cycles in resources.engine_cycles.items():
        if split != dataset_split:
            continue
        result = resources.adapter.predict(
            resources.raw_trajectories, engine, cycles[-1], dataset_split=split,
            scenario=scenario, recovery=recovery, seed=seed,
        )
        if not np.isfinite(result.rul_pred):
            raise ValueError("Queue predictions must be finite before negative clamping")
        rows.append({"dataset_split": split, "engine_id": engine,
                     "observed_cycle": result.cycle, "rul_pred": max(float(result.rul_pred), 0.0)})
    if not rows:
        raise ValueError("No engines in the selected namespace")
    return top_k_queue(pd.DataFrame(rows), k=k)

def run_app(artifact_dir: str | Path | None = None):
    st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
    st.title("Predictive Maintenance — FD001")
    st.caption("NASA C-MAPSS FD001 demonstration: RUL is in operating cycles, not validated maintenance thresholds.")
    model = st.selectbox("Model", ["Median", "Random Forest", "XGBoost", "LSTM"], index=3, key="model")
    scenario = SCENARIOS[st.selectbox("Missingness scenario", list(SCENARIOS), key="scenario")]
    recovery_options = ["Training median"] if scenario == "clean" else list(RECOVERIES)
    recovery = RECOVERIES[st.selectbox("Recovery method", recovery_options, key="recovery")]
    seed = st.selectbox("Random seed", [7, 17, 27], key="seed") if scenario in {"random_10", "random_20"} else None
    k = st.selectbox("Inspection queue K", [5, 10], disabled=model != "LSTM", key="queue_k")
    if model != "LSTM":
        st.info(f"Saved inference for {model} is not connected in this application yet. "
                "The reusable Task 25 adapter and its artifact/prefix checks are pending; "
                "replay and queue are unavailable for this model.")
        return
    root = Path(artifact_dir) if artifact_dir is not None else ROOT / "artifacts"
    try:
        root = root.resolve()
        resources = _cached_resources(str(root), artifact_fingerprint(root))
        identity = st.selectbox(
            "Engine", list(resources.engine_cycles), key="engine",
            format_func=lambda value: f"{value[0]} / Engine {value[1]}" if value[0] is not None else f"Engine {value[1]}",
        )
        cycles = resources.engine_cycles[identity]
        if len(cycles) == 1:
            observed_cycle = cycles[0]
            st.caption(f"Observed cycle: {observed_cycle} (one available observation)")
        else:
            observed_cycle = st.select_slider(
                "Observed cycle", options=cycles, value=cycles[0], key=f"observed_cycle:{identity!r}",
            )
        with st.spinner("Computing prefix-only LSTM replay..."):
            result, history = build_lstm_replay(
                resources, identity, observed_cycle, scenario=scenario, recovery=recovery, seed=seed,
            )
    except Exception as exc:
        # Artifact deserialization, compatibility and inference failures must not produce substitute predictions.
        st.error(f"Saved LSTM replay is unavailable: {exc}")
        st.info("Verify the prepared raw-input bundle and matching LSTM checkpoint/metadata. No automatic training or download is performed.")
        return
    st.metric("Predicted RUL (cycles)", f"{result.rul_pred:.2f}")
    st.metric("Demo health label", health_label(result.rul_pred))
    st.caption(HEALTH_STATUS_NOTE)
    st.write({"actual_visible_cycle": result.cycle, "observed_history_length": result.observed_history_length,
              "sequence_window": result.sequence_window, "padding_cycles": result.padding_cycles})
    sensors = [column for column in result.sensor_mask.columns]
    sensor = st.selectbox("Sensor", sensors, key="sensor")
    st.caption("Sensor history shows recovered measurements in original sensor units, using only the visible prefix.")
    for figure in (plot_sensor_history(result.chart_prefix, sensor), plot_rul_history(history)):
        try:
            st.pyplot(figure)
        finally:
            figure.clear()

    st.subheader("RUL replay")
    st.dataframe(history.tail(10), hide_index=True)
    with st.expander("Fleet inspection queue"):
        st.write(QUEUE_NOTE)
        st.caption(
            "LSTM queue ranks every engine in the selected namespace at its own final observed cycle. "
            "Snapshot cycle lengths differ; the replay slider does not affect this demonstration queue. "
            "Click again after changing scenario/recovery/seed or artifacts; "
            "queue results are not cached."
        )
        if st.button("Calculate fleet queue", type="primary", key="calculate_queue"):
            try:
                queue = build_lstm_queue(
                    resources, identity[0], k,
                    scenario=scenario, recovery=recovery, seed=seed,
                )
            except Exception as exc:
                st.error(f"LSTM inspection queue is unavailable: {exc}")
            else:
                st.dataframe(queue, hide_index=True)
                st.download_button(
                    "Download inspection queue CSV", queue.to_csv(index=False).encode("utf-8"),
                    file_name=f"inspection_queue_top_{k}.csv", mime="text/csv", key="download_queue",
                )


if __name__ == "__main__":
    run_app()
