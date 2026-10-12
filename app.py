"""Saved LSTM replay; other model and queue integration remains team-owned."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.dashboard import (
    HEALTH_STATUS_NOTE, health_label, plot_rul_history, plot_sensor_history,
)
from predictive_maintenance.replay import artifact_fingerprint, build_lstm_replay, load_replay_resources

SCENARIOS = {"S0 Clean": "clean", "S1 Random 10%": "random_10",
             "S2 Random 20%": "random_20", "S3 Short gap": "short_gap"}
RECOVERIES = {"Training median": "median", "Causal forward fill": "forward_fill"}


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_resources(artifact_dir: str, fingerprint: tuple):
    resources = load_replay_resources(artifact_dir)
    if artifact_fingerprint(artifact_dir) != fingerprint:
        raise ValueError("Saved artifacts changed while loading; retry after preparation finishes")
    return resources


def run_app(artifact_dir: str | Path | None = None):
    st.title("Predictive Maintenance — FD001")
    model = st.selectbox("Model", ["Median", "Random Forest", "XGBoost", "LSTM"], index=3, key="model")
    scenario = SCENARIOS[st.selectbox("Missingness scenario", list(SCENARIOS), key="scenario")]
    recovery_options = ["Training median"] if scenario == "clean" else list(RECOVERIES)
    recovery = RECOVERIES[st.selectbox("Recovery method", recovery_options, key="recovery")]
    seed = st.selectbox("Random seed", [7, 17, 27], key="seed") if scenario in {"random_10", "random_20"} else None
    st.selectbox("Inspection queue K", [5, 10], disabled=True, key="queue_k")
    st.caption("Inspection queue integration is pending.")
    if model != "LSTM":
        st.info(f"Saved inference for {model} is not connected in this application yet.")
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


if __name__ == "__main__":
    run_app()
