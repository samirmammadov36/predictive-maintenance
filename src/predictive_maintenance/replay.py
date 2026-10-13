"""Streamlit-independent selected-model loading, per-cutoff replay and final-snapshot queues."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .constants import CYCLE_COLUMN, ID_COLUMN
from .data import file_manifest
from .inference import InferenceResult, load_inference_adapter, _positive_integer
from .masks import _positive_integer_values
from .prepared import official_test_raw_frame

REPLAY_ARTIFACTS = (
    "preprocessing/prepared_bundle.joblib", "preprocessing/sequence_input_metadata.json",
    "preprocessing/split_manifest.json", "models/lstm.pt", "models/lstm_checkpoint_metadata.json",
)

MODEL_ARTIFACTS = {
    "lstm": REPLAY_ARTIFACTS,
    "random_forest": ("preprocessing/prepared_bundle.joblib", "preprocessing/schema.json", "models/random_forest.joblib"),
    "xgboost": ("preprocessing/prepared_bundle.joblib", "preprocessing/schema.json", "models/xgboost.json"),
    "median_baseline": ("preprocessing/prepared_bundle.joblib", "models/baseline.joblib"),
}


@dataclass(frozen=True)
class ReplayResources:
    adapter: object
    raw_trajectories: pd.DataFrame
    engine_cycles: dict[tuple[str | None, int], tuple[int, ...]]


def artifact_fingerprint(artifact_dir: str | Path, model_name: str = "lstm") -> tuple[tuple[str, str], ...]:
    """Hash every file actually consumed by bundle/checkpoint loading."""
    root = Path(artifact_dir)
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unknown saved inference model: {model_name}")
    return tuple((name, file_manifest(root / name).sha256) for name in MODEL_ARTIFACTS[model_name])


def available_engine_cycles(raw: pd.DataFrame) -> dict:
    """Inspect identity/cycle metadata only, for engine and slider choices."""
    if not isinstance(raw, pd.DataFrame) or raw.empty or not raw.columns.is_unique:
        raise ValueError("Raw official-test trajectories must be a non-empty DataFrame with unique columns")
    columns = [ID_COLUMN, CYCLE_COLUMN]
    has_split = "dataset_split" in raw
    if has_split:
        columns.insert(0, "dataset_split")
    if any(column not in raw for column in columns):
        raise ValueError("Raw official-test trajectories require engine_id and cycle")
    metadata = raw[columns].copy()
    for column in (ID_COLUMN, CYCLE_COLUMN):
        metadata[column] = _positive_integer_values(metadata[column], column)
    if has_split and metadata.dataset_split.map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise ValueError("Raw dataset_split identities must be non-empty strings")
    if metadata.duplicated(columns).any():
        raise ValueError("Duplicate engine/cycle observations in raw official-test trajectories")
    if not has_split:
        metadata["dataset_split"] = None
    choices = {}
    for (split, engine), frame in metadata.groupby(["dataset_split", ID_COLUMN], dropna=False, observed=True, sort=True):
        identity = (split if has_split else None, int(engine))
        choices[identity] = tuple(sorted(int(cycle) for cycle in frame[CYCLE_COLUMN]))
    return choices


def load_replay_resources(artifact_dir: str | Path, model_name: str = "lstm") -> ReplayResources:
    root = Path(artifact_dir)
    bundle = joblib.load(root / "preprocessing/prepared_bundle.joblib")
    raw = official_test_raw_frame(bundle)
    choices = available_engine_cycles(raw)
    return ReplayResources(load_inference_adapter(bundle, root, model_name), raw, choices)


def build_model_replay(
    resources: ReplayResources, identity: tuple[str | None, int], observed_cycle: int, *,
    scenario: str = "clean", recovery: str = "median", seed: int | None = None,
) -> tuple[InferenceResult, pd.DataFrame]:
    """Predict each available history point independently; never reuse a recovered full prefix."""
    observed_cycle = _positive_integer(observed_cycle, "observed_cycle")
    if identity not in resources.engine_cycles:
        raise ValueError("Selected engine identity is unavailable")
    cycles = [cycle for cycle in resources.engine_cycles[identity] if cycle <= observed_cycle]
    if not cycles:
        raise ValueError("Observed cutoff produces an empty replay history")
    split, engine = identity
    rows = []
    for cycle in cycles:
        result = resources.adapter.predict(
            resources.raw_trajectories, engine, cycle, dataset_split=split,
            scenario=scenario, recovery=recovery, seed=seed,
        )
        rows.append({"cycle": result.cycle, "rul_pred": result.rul_pred})
    return result, pd.DataFrame(rows)

# Public LSTM names remain aliases of the shared interfaces.
LSTMReplayResources = ReplayResources
build_lstm_replay = build_model_replay


def build_inspection_queue(resources, dataset_split, k, *, scenario="clean", recovery="median", seed=None):
    """Predict every selected-namespace final snapshot before top-K ranking."""
    from .dashboard import top_k_queue
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
