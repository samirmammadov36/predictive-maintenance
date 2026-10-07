from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    CYCLE_COLUMN,
    ID_COLUMN,
    RAW_COLUMNS,
    RUL_TARGET_COLUMN,
    RUL_TRUTH_COLUMN,
)


@dataclass(frozen=True)
class FileManifest:
    path: str
    size_bytes: int
    sha256: str


def file_manifest(path: str | Path) -> FileManifest:
    path = Path(path)
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return FileManifest(str(path), path.stat().st_size, h.hexdigest())


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a raw C-MAPSS train/test text file with the official 26-column layout."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns, found {df.shape[1]} in {path}")
    if df.isna().all(axis=1).any():
        df = df.dropna(how="all")
    df[ID_COLUMN] = df[ID_COLUMN].astype(int)
    df[CYCLE_COLUMN] = df[CYCLE_COLUMN].astype(int)
    return df.sort_values([ID_COLUMN, CYCLE_COLUMN]).reset_index(drop=True)


def add_training_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    out = df.copy()
    final_cycle = out.groupby(ID_COLUMN)[CYCLE_COLUMN].transform("max")
    out[RUL_TRUTH_COLUMN] = (final_cycle - out[CYCLE_COLUMN]).astype(int)
    out[RUL_TARGET_COLUMN] = out[RUL_TRUTH_COLUMN].clip(upper=cap).astype(float)
    return out


def read_test_rul(path: str | Path) -> pd.Series:
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    values = pd.to_numeric(values, errors="raise").astype(int).reset_index(drop=True)
    values.index = np.arange(1, len(values) + 1)
    values.index.name = ID_COLUMN
    values.name = RUL_TRUTH_COLUMN
    return values


def official_test_targets(test_df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """Return one official target row per test engine at its final observed cycle."""
    final_rows = (
        test_df.sort_values([ID_COLUMN, CYCLE_COLUMN])
        .groupby(ID_COLUMN, as_index=False)
        .tail(1)[[ID_COLUMN, CYCLE_COLUMN]]
        .copy()
    )
    final_rows[RUL_TRUTH_COLUMN] = final_rows[ID_COLUMN].map(rul)
    if final_rows[RUL_TRUTH_COLUMN].isna().any():
        missing = final_rows.loc[final_rows[RUL_TRUTH_COLUMN].isna(), ID_COLUMN].tolist()
        raise ValueError(f"Missing official test RUL for engines: {missing}")
    return final_rows.reset_index(drop=True)


def split_development_engines(
    train_df: pd.DataFrame,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    engines = np.array(sorted(train_df[ID_COLUMN].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engines)
    n_val = int(round(len(engines) * validation_fraction))
    val = sorted(map(int, shuffled[:n_val]))
    tr = sorted(map(int, shuffled[n_val:]))
    if set(tr) & set(val):
        raise AssertionError("Train/validation engine overlap detected")
    return tr, val


def validation_prefixes(
    df: pd.DataFrame,
    engine_ids: Iterable[int],
    prefix_fraction: float = 0.70,
) -> dict[int, int]:
    """Map each validation engine to the last visible cycle of a fixed causal prefix."""
    prefix_cycles: dict[int, int] = {}
    for engine_id in engine_ids:
        engine = df[df[ID_COLUMN] == engine_id]
        if engine.empty:
            raise ValueError(f"Engine {engine_id} not found")
        full_length = len(engine)
        visible_rows = max(1, int(np.floor(prefix_fraction * full_length)))
        prefix_cycles[int(engine_id)] = int(engine.iloc[visible_rows - 1][CYCLE_COLUMN])
    return prefix_cycles


def prefix_for_engine(df: pd.DataFrame, engine_id: int, end_cycle: int | None = None) -> pd.DataFrame:
    engine = df[df[ID_COLUMN] == engine_id].sort_values(CYCLE_COLUMN)
    if end_cycle is not None:
        engine = engine[engine[CYCLE_COLUMN] <= end_cycle]
    if engine.empty:
        raise ValueError(f"No rows for engine={engine_id}, end_cycle={end_cycle}")
    return engine.copy().reset_index(drop=True)


def true_rul_at_cycle(full_df: pd.DataFrame, engine_id: int, cycle: int) -> int:
    max_cycle = int(full_df.loc[full_df[ID_COLUMN] == engine_id, CYCLE_COLUMN].max())
    return max_cycle - int(cycle)


def manifests_as_dict(paths: Iterable[str | Path]) -> list[dict]:
    return [asdict(file_manifest(p)) for p in paths]
