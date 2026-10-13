"""Streamlit dashboard for causal NASA C-MAPSS FD001 RUL inference."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from predictive_maintenance.constants import CYCLE_COLUMN, ID_COLUMN
from predictive_maintenance.dashboard import (
    HEALTH_STATUS_NOTE,
    QUEUE_NOTE,
    health_label,
    plot_rul_history,
    plot_sensor_history,
    top_k_queue,
)
from predictive_maintenance.deployment import ensure_deployment_artifacts
from predictive_maintenance.inference import (
    build_replay,
    corrupt_prefix,
    load_prepared_bundle,
    load_saved_models,
    predict_with_scenario,
)


PREPARED_PATH = ROOT / "artifacts/preprocessing/prepared_bundle.joblib"
MODEL_DIR = ROOT / "artifacts/models"


@st.cache_resource
def load_bundle():
    return load_prepared_bundle(PREPARED_PATH)


@st.cache_resource
def load_models():
    return load_saved_models(MODEL_DIR)


def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("Predictive Maintenance — Remaining Useful Life")

    st.caption(
        "NASA C-MAPSS FD001 demonstration. "
        "Predictions estimate remaining operating cycles; "
        "they are not validated real-world maintenance thresholds."
    )

    try:
        ensure_deployment_artifacts(ROOT)

        bundle = load_bundle()
        models = load_models()

    except (
        RuntimeError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        st.error(str(exc))

        st.info(
            "If running locally, install requirements and run: "
            "python scripts/download_data.py → "
            "python scripts/prepare_data.py → "
            "python scripts/train_models.py"
        )

        st.stop()

    engine_ids = sorted(
        map(
            int,
            bundle.official_test_df[ID_COLUMN].unique(),
        )
    )

    model_name = st.sidebar.selectbox(
        "Prediction model",
        [
            "XGBoost",
            "Random Forest",
            "LSTM",
            "Median baseline",
        ],
    )

    scenario = st.sidebar.selectbox(
        "Sensor availability",
        [
            "Clean",
            "Random 10%",
            "Random 20%",
            "Five-cycle sensor gap",
        ],
    )

    recovery = st.sidebar.selectbox(
        "Missing-value recovery",
        [
            "Causal forward fill",
            "Training median",
        ],
    )

    seed = st.sidebar.selectbox(
        "Random-mask seed",
        [7, 17, 27],
    )

    engine_id = st.sidebar.selectbox(
        "Engine ID",
        engine_ids,
    )

    engine = (
        bundle.official_test_df[
            bundle.official_test_df[ID_COLUMN] == engine_id
        ]
        .sort_values(CYCLE_COLUMN)
        .copy()
    )

    max_cycle = int(engine[CYCLE_COLUMN].max())

    observed_cycle = st.sidebar.slider(
        "Observed cycle (replay cutoff)",
        min_value=int(engine[CYCLE_COLUMN].min()),
        max_value=max_cycle,
        value=max_cycle,
    )

    k = st.sidebar.radio(
        "Inspection queue size",
        [5, 10],
        horizontal=True,
    )

    model = models[model_name]

    visible, replay = build_replay(
        engine,
        observed_cycle,
        model_name,
        model,
        bundle,
        scenario,
        recovery,
        int(seed),
    )

    current_prediction = float(
        replay.iloc[-1]["rul_pred"]
    )

    status = health_label(
        current_prediction
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Engine",
        str(engine_id),
    )

    metric_cols[1].metric(
        "Observed cycle",
        str(observed_cycle),
    )

    metric_cols[2].metric(
        "Predicted RUL",
        f"{current_prediction:.1f} cycles",
    )

    metric_cols[3].metric(
        "Display status",
        status,
    )

    # Show sensor history using the selected availability scenario.
    corrupted_visible = corrupt_prefix(
        visible,
        bundle,
        scenario,
        int(seed),
    )

    st.subheader(
        "Sensor history under selected availability scenario"
    )

    sensor = st.selectbox(
        "Sensor to inspect",
        bundle.selected_sensors,
    )

    st.pyplot(
        plot_sensor_history(
            corrupted_visible,
            sensor,
        ),
        clear_figure=True,
    )

    st.subheader("RUL replay")

    st.pyplot(
        plot_rul_history(replay),
        clear_figure=True,
    )

    st.dataframe(
        replay.tail(10),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(HEALTH_STATUS_NOTE)

    st.caption(
        f"Scenario: {scenario} · "
        f"Recovery: {recovery} · "
        f"Mask seed: {seed}. "
        "All predictions and charts are restricted "
        "to the selected observed cycle."
    )

    with st.expander("Fleet inspection queue"):
        st.write(QUEUE_NOTE)

        if st.button(
            "Calculate fleet queue",
            type="primary",
        ):
            progress = st.progress(0)
            queue_rows = []

            for index, fleet_engine_id in enumerate(
                engine_ids,
                start=1,
            ):
                fleet_engine = (
                    bundle.official_test_df[
                        bundle.official_test_df[
                            ID_COLUMN
                        ]
                        == fleet_engine_id
                    ]
                    .sort_values(CYCLE_COLUMN)
                    .copy()
                )

                last_cycle = int(
                    fleet_engine[CYCLE_COLUMN].max()
                )

                pred = predict_with_scenario(
                    model_name,
                    model,
                    fleet_engine,
                    bundle,
                    scenario,
                    recovery,
                    int(seed),
                )

                queue_rows.append(
                    {
                        ID_COLUMN: fleet_engine_id,
                        "observed_cycle": last_cycle,
                        "rul_pred": pred,
                    }
                )

                progress.progress(
                    index / len(engine_ids)
                )

            queue = top_k_queue(
                pd.DataFrame(queue_rows),
                k=k,
            )

            st.dataframe(
                queue,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download inspection queue CSV",
                queue.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"inspection_queue_top_{k}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()