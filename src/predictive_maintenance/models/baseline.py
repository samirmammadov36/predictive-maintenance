from __future__ import annotations

import numpy as np


class MedianRULBaseline:
    def __init__(self) -> None:
        self.value_: float | None = None

    def fit(self, y) -> "MedianRULBaseline":
        self.value_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("Baseline has not been fitted")
        return np.full(n, self.value_, dtype=float)
