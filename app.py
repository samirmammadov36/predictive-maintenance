"""Saved-model dashboard with per-prefix replay and final-snapshot inspection queues."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.dashboard import (
    HEALTH_STATUS_NOTE, QUEUE_NOTE, health_label, plot_rul_history, plot_sensor_history,
)
from predictive_maintenance.replay import (
    artifact_fingerprint, build_model_replay, load_replay_resources,
    build_inspection_queue as build_lstm_queue,
)

MODEL_NAMES = {"Median": "median_baseline", "Random Forest": "random_forest", "XGBoost": "xgboost", "LSTM": "lstm"}

SCENARIOS = {"S0 Clean": "clean", "S1 Random 10%": "random_10",
             "S2 Random 20%": "random_20", "S3 Short gap": "short_gap"}
RECOVERIES = {"Training median": "median", "Causal forward fill": "forward_fill"}


@st.cache_resource(show_spinner=False, max_entries=8)
def _cached_resources(artifact_dir: str, fingerprint: tuple, model_name: str = "lstm"):
    resources = load_replay_resources(artifact_dir) if model_name == "lstm" else load_replay_resources(artifact_dir, model_name)
    if artifact_fingerprint(artifact_dir, model_name) != fingerprint:
        raise ValueError("Saved artifacts changed while loading; retry after preparation finishes")
    return resources


def run_app(artifact_dir: str | Path | None = None):
    st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
    st.title("Predictive Maintenance — FD001")
    st.caption("NASA C-MAPSS FD001 demonstration: RUL is in operating cycles, not validated maintenance thresholds.")
    model = st.selectbox("Model", ["Median", "Random Forest", "XGBoost", "LSTM"], index=3, key="model")
    scenario = SCENARIOS[st.selectbox("Missingness scenario", list(SCENARIOS), key="scenario")]
    recovery_options = ["Training median"] if scenario == "clean" else list(RECOVERIES)
    recovery = RECOVERIES[st.selectbox("Recovery method", recovery_options, key="recovery")]
    seed = st.selectbox("Random seed", [7, 17, 27], key="seed") if scenario in {"random_10", "random_20"} else None
    k = st.selectbox("Inspection queue K", [5, 10], key="queue_k")
    model_name = MODEL_NAMES[model]
    root = Path(artifact_dir) if artifact_dir is not None else ROOT / "artifacts"
    try:
        root = root.resolve()
        resources = _cached_resources(str(root), artifact_fingerprint(root, model_name), model_name)
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
        with st.spinner(f"Computing prefix-only {model} replay..."):
            result, history = build_model_replay(
                resources, identity, observed_cycle, scenario=scenario, recovery=recovery, seed=seed,
            )
    except Exception as exc:
        # Artifact deserialization, compatibility and inference failures must not produce substitute predictions.
        st.error(f"Saved {model} replay is unavailable: {exc}")
        st.info("Verify the prepared raw-input bundle and required selected-model artifacts/schema. No automatic training or download is performed.")
        return
    st.metric("Predicted RUL (cycles)", f"{result.rul_pred:.2f}")
    st.metric("Demo health label", health_label(result.rul_pred))
    st.caption(HEALTH_STATUS_NOTE)
    if model == "Median":
        st.caption("The baseline is the constant fitted training-target median; sensor missingness does not change its RUL prediction.")
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
            "The selected model ranks every engine in the selected namespace at its own final observed cycle. "
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
                st.error(f"{model} inspection queue is unavailable: {exc}")
            else:
                st.dataframe(queue, hide_index=True)
                st.download_button(
                    "Download inspection queue CSV", queue.to_csv(index=False).encode("utf-8"),
                    file_name=f"inspection_queue_top_{k}.csv", mime="text/csv", key="download_queue",
                )


if __name__ == "__main__":
    run_app()
