from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.artifacts import dump_joblib, dump_json
from predictive_maintenance.config import load_config
from predictive_maintenance.constants import SENSOR_COLUMNS
from predictive_maintenance.data import (
    add_training_rul,
    manifests_as_dict,
    official_test_targets,
    read_cmapss_file,
    read_test_rul,
    split_development_engines,
    validation_prefixes,
)
from predictive_maintenance.features import build_tabular_features
from predictive_maintenance.imputation import TrainingMedians
from predictive_maintenance.prepared import PreparedBundle
from predictive_maintenance.screening import screen_sensors
from predictive_maintenance.sequences import fit_sequence_scaler


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    raw = ROOT / cfg["data"]["raw_dir"]
    processed = ROOT / cfg["data"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    prep_dir = ROOT / "artifacts/preprocessing"
    prep_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw / cfg["data"]["train_file"]
    test_path = raw / cfg["data"]["test_file"]
    rul_path = raw / cfg["data"]["test_rul_file"]
    missing = [str(p) for p in (train_path, test_path, rul_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw FD001 files. Run `python scripts/download_data.py` first. Missing: " + ", ".join(missing)
        )

    full_train = add_training_rul(read_cmapss_file(train_path), cap=cfg["project"]["rul_cap"])
    official_test = read_cmapss_file(test_path)
    test_rul = read_test_rul(rul_path)
    test_targets = official_test_targets(official_test, test_rul)

    train_ids, val_ids = split_development_engines(
        full_train,
        cfg["project"]["validation_fraction"],
        cfg["project"]["seed"],
    )
    train_df = full_train[full_train.engine_id.isin(train_ids)].copy()
    val_df = full_train[full_train.engine_id.isin(val_ids)].copy()
    prefix_cycles = validation_prefixes(val_df, val_ids, cfg["project"]["validation_prefix_fraction"])

    selected, screening_report = screen_sensors(
        train_df,
        SENSOR_COLUMNS,
        cfg["screening"]["near_constant_unique_ratio"],
        cfg["screening"]["near_constant_dominant_fraction"],
    )
    if not selected:
        raise RuntimeError("Sensor screening removed every sensor")
    screening_report.to_csv(processed / "sensor_screening.csv", index=False)

    medians = TrainingMedians.fit(train_df, selected)
    train_clean = medians.median_fill(train_df, selected)
    val_clean = medians.median_fill(val_df, selected)
    test_clean = medians.median_fill(official_test, selected)

    sequence_scaler = fit_sequence_scaler(train_clean, selected)

    engineered_train, feature_columns = build_tabular_features(
        train_clean,
        selected,
        cfg["features"]["rolling_window"],
    )
    tabular_scaler = StandardScaler().fit(engineered_train[feature_columns])

    bundle = PreparedBundle(
        train_df=train_clean,
        validation_df=val_clean,
        official_test_df=test_clean,
        official_test_targets=test_targets,
        selected_sensors=selected,
        feature_columns=feature_columns,
        train_engine_ids=train_ids,
        validation_engine_ids=val_ids,
        validation_prefix_cycles=prefix_cycles,
        medians=medians,
        sequence_scaler=sequence_scaler,
        tabular_scaler=tabular_scaler,
        rolling_window=cfg["features"]["rolling_window"],
        lstm_window=cfg["lstm"]["window"],
    )
    dump_joblib(bundle, prep_dir / "prepared_bundle.joblib")
    dump_joblib(medians, prep_dir / "training_medians.joblib")
    dump_joblib(sequence_scaler, prep_dir / "sequence_scaler.joblib")
    dump_joblib(tabular_scaler, prep_dir / "tabular_scaler.joblib")

    split_manifest = {
        "seed": cfg["project"]["seed"],
        "train_engines": train_ids,
        "validation_engines": val_ids,
        "validation_prefix_cycles": prefix_cycles,
    }
    dump_json(split_manifest, prep_dir / "split_manifest.json")
    dump_json({"selected_sensors": selected, "feature_columns": feature_columns}, prep_dir / "schema.json")
    dump_json(manifests_as_dict([train_path, test_path, rul_path]), prep_dir / "data_manifest.json")

    checks = {
        "train_engines": len(train_ids),
        "validation_engines": len(val_ids),
        "official_test_engines": int(official_test.engine_id.nunique()),
        "selected_sensor_count": len(selected),
        "selected_sensors": selected,
        "train_validation_overlap": sorted(set(train_ids) & set(val_ids)),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(official_test),
        "missing_after_median_train": int(train_clean[selected].isna().sum().sum()),
        "missing_after_median_validation": int(val_clean[selected].isna().sum().sum()),
        "missing_after_median_test": int(test_clean[selected].isna().sum().sum()),
    }
    dump_json(checks, processed / "data_integrity_report.json")
    print(json.dumps(checks, indent=2))
    print(f"Prepared bundle written to {prep_dir / 'prepared_bundle.joblib'}")


if __name__ == "__main__":
    main()
