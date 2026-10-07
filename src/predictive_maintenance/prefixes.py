from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN
from .data import prefix_for_engine, true_rul_at_cycle
from .features import build_tabular_features
from .sequences import build_prefix_window, scale_sequence_frame


def make_validation_tabular_matrix(bundle):
    xs, ys, meta = [], [], []
    for engine_id in bundle.validation_engine_ids:
        end_cycle = bundle.validation_prefix_cycles[int(engine_id)]
        prefix = prefix_for_engine(bundle.validation_df, int(engine_id), end_cycle)
        engineered, _ = build_tabular_features(prefix, bundle.selected_sensors, bundle.rolling_window)
        row = engineered.iloc[[-1]][bundle.feature_columns]
        xs.append(bundle.tabular_scaler.transform(row)[0])
        ys.append(true_rul_at_cycle(bundle.validation_df, int(engine_id), end_cycle))
        meta.append((int(engine_id), int(end_cycle)))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), meta


def make_validation_sequence_matrix(bundle):
    xs, ys, meta = [], [], []
    for engine_id in bundle.validation_engine_ids:
        end_cycle = bundle.validation_prefix_cycles[int(engine_id)]
        prefix = prefix_for_engine(bundle.validation_df, int(engine_id), end_cycle)
        scaled = scale_sequence_frame(prefix, bundle.selected_sensors, bundle.sequence_scaler)
        xs.append(build_prefix_window(scaled, bundle.selected_sensors, bundle.lstm_window))
        ys.append(true_rul_at_cycle(bundle.validation_df, int(engine_id), end_cycle))
        meta.append((int(engine_id), int(end_cycle)))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), meta
