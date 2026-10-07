from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation import regression_metrics


@dataclass
class LSTMConfig:
    input_size: int
    hidden_units: int = 32
    dense_units: int = 16
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 5
    seed: int = 42


class RULLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_units: int = 32, dense_units: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_units, dense_units), nn.ReLU(), nn.Linear(dense_units, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


def _predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy()


def train_lstm(X_train, y_train, X_val, y_val, cfg: LSTMConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RULLSTM(cfg.input_size, cfg.hidden_units, cfg.dense_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    best_state = None
    best_rmse = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        val_pred = _predict(model, X_val, device)
        metrics = regression_metrics(y_val, val_pred)
        history.append({"epoch": epoch, "train_mse": float(np.mean(batch_losses)), **metrics})
        if metrics["rmse"] < best_rmse - 1e-8:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"train_seconds": time.perf_counter() - start, "device": device, **asdict(cfg)}


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device.type
    return _predict(model, X, device)
