from __future__ import annotations

import json
from pathlib import Path
import joblib
import torch


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_joblib(obj, path: str | Path) -> None:
    joblib.dump(obj, ensure_parent(path))


def load_joblib(path: str | Path):
    return joblib.load(path)


def dump_json(obj, path: str | Path) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def save_torch_state(model, path: str | Path) -> None:
    torch.save(model.state_dict(), ensure_parent(path))
