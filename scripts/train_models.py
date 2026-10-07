from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.artifacts import dump_joblib, dump_json
from predictive_maintenance.config import load_config
from predictive_maintenance.constants import RUL_TARGET_COLUMN
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.models.baseline import MedianRULBaseline
from predictive_maintenance.models.lstm import LSTMConfig, train_lstm
from predictive_maintenance.models.random_forest import train_random_forest
from predictive_maintenance.models.xgboost_model import train_xgboost
from predictive_maintenance.prefixes import make_validation_sequence_matrix, make_validation_tabular_matrix
from predictive_maintenance.sequences import build_training_windows, scale_sequence_frame


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    bundle = joblib.load(ROOT / "artifacts/preprocessing/prepared_bundle.joblib")
    model_dir = ROOT / "artifacts/models"
    result_dir = ROOT / "results/metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    engineered_train, _ = build_tabular_features(bundle.train_df, bundle.selected_sensors, bundle.rolling_window)
    X_train_tab = bundle.tabular_scaler.transform(engineered_train[bundle.feature_columns])
    y_train_tab = engineered_train[RUL_TARGET_COLUMN].to_numpy(dtype=np.float32)
    X_val_tab, y_val, _ = make_validation_tabular_matrix(bundle)

    baseline = MedianRULBaseline().fit(y_train_tab)
    dump_joblib(baseline, model_dir / "baseline.joblib")

    rf, rf_log = train_random_forest(X_train_tab, y_train_tab, X_val_tab, y_val, cfg["project"]["seed"])
    dump_joblib(rf, model_dir / "random_forest.joblib")
    pd.DataFrame(rf_log).to_csv(result_dir / "random_forest_validation_search.csv", index=False)

    xgb_model, xgb_log = train_xgboost(X_train_tab, y_train_tab, X_val_tab, y_val, cfg["project"]["seed"])
    xgb_model.save_model(model_dir / "xgboost.json")
    pd.DataFrame(xgb_log).to_csv(result_dir / "xgboost_validation_search.csv", index=False)

    scaled_train = scale_sequence_frame(bundle.train_df, bundle.selected_sensors, bundle.sequence_scaler)
    X_train_seq, y_train_seq = build_training_windows(
        scaled_train,
        bundle.selected_sensors,
        RUL_TARGET_COLUMN,
        bundle.lstm_window,
    )
    X_val_seq, y_val_seq, _ = make_validation_sequence_matrix(bundle)
    lstm_cfg = LSTMConfig(
        input_size=len(bundle.selected_sensors),
        hidden_units=cfg["lstm"]["hidden_units"],
        dense_units=cfg["lstm"]["dense_units"],
        learning_rate=cfg["lstm"]["learning_rate"],
        batch_size=cfg["lstm"]["batch_size"],
        max_epochs=cfg["lstm"]["max_epochs"],
        patience=cfg["lstm"]["patience"],
        seed=cfg["project"]["seed"],
    )
    lstm, lstm_history, lstm_meta = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, lstm_cfg)
    torch.save(lstm.state_dict(), model_dir / "lstm.pt")
    dump_joblib({"config": {k: v for k, v in lstm_meta.items() if k in LSTMConfig.__dataclass_fields__}, "meta": lstm_meta}, model_dir / "lstm_meta.joblib")
    pd.DataFrame(lstm_history).to_csv(result_dir / "lstm_training_history.csv", index=False)

    summary = {
        "random_forest_selected": min(rf_log, key=lambda r: (r["rmse"], r["mae"])),
        "xgboost_selected": min(xgb_log, key=lambda r: (r["rmse"], r["mae"])),
        "lstm_best": min(lstm_history, key=lambda r: (r["rmse"], r["mae"])),
        "note": "Models selected on clean validation-prefix RMSE with MAE as tie-breaker; official test data was not used for tuning.",
    }
    dump_json(summary, result_dir / "validation_selection_summary.json")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
