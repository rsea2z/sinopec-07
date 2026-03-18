from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEQUENCE_FEATURES = [
    "收盘价",
    "成交量",
    "持仓量",
    "return_1d",
    "intraday_range",
    "open_close_spread",
    "volume_oi_ratio",
]


@dataclass
class DeepResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        return self.net(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_rows: int, test_rows: int) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100),
        "train_rows": float(train_rows),
        "test_rows": float(test_rows),
        "eval_rows": float(mask.sum()),
    }


def build_sequence_data(dataset: pd.DataFrame, target_col: str, lookback: int = 20) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    df = dataset.copy().sort_values("date").reset_index(drop=True)
    features = list(dict.fromkeys([col for col in SEQUENCE_FEATURES if col in df.columns]))
    select_cols = list(dict.fromkeys(["date", "收盘价", target_col] + features))
    usable = df[select_cols].copy()
    usable[features] = usable[features].apply(pd.to_numeric, errors="coerce")
    usable[features] = usable[features].ffill().fillna(0.0)
    usable[target_col] = pd.to_numeric(usable[target_col], errors="coerce")
    usable["base_close"] = pd.to_numeric(usable["收盘价"], errors="coerce")
    usable["target_return"] = usable[target_col] / usable["base_close"] - 1.0

    X_list: list[np.ndarray] = []
    y_return: list[float] = []
    y_price: list[float] = []
    base_close: list[float] = []
    dates: list[pd.Timestamp] = []

    values = usable[features].to_numpy(dtype=float)
    target_returns = usable["target_return"].to_numpy(dtype=float)
    target_prices = usable[target_col].to_numpy(dtype=float)
    base_prices = usable["base_close"].to_numpy(dtype=float)
    date_values = usable["date"].to_numpy()

    for end_idx in range(lookback - 1, len(usable)):
        start_idx = end_idx - lookback + 1
        window = values[start_idx : end_idx + 1]
        if not np.isfinite(window).all():
            continue
        if not np.isfinite(target_returns[end_idx]):
            continue
        X_list.append(window)
        y_return.append(target_returns[end_idx])
        y_price.append(target_prices[end_idx])
        base_close.append(base_prices[end_idx])
        dates.append(date_values[end_idx])

    X = np.stack(X_list)
    y_ret = np.array(y_return, dtype=np.float32)
    y_price_arr = np.array(y_price, dtype=np.float32)
    base_close_arr = np.array(base_close, dtype=np.float32)

    split_idx = int(len(X) * 0.8)
    train_X = X[:split_idx]
    test_X = X[split_idx:]
    train_y = y_ret[:split_idx]
    test_y = y_ret[split_idx:]

    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    payload = {
        "train_X": train_X,
        "test_X": test_X,
        "train_y": train_y,
        "test_y": test_y,
        "test_price": y_price_arr[split_idx:],
        "test_base_close": base_close_arr[split_idx:],
        "test_dates": np.array(dates[split_idx:]),
        "feature_count": train_X.shape[-1],
        "sequence_length": train_X.shape[1],
        "train_rows": len(train_X),
        "test_rows": len(test_X),
    }
    meta = {"features": features}
    return payload, meta


def _train_torch_model(
    model: nn.Module,
    train_X: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 64,
) -> nn.Module:
    dataset = SequenceDataset(train_X, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model


def _predict_price(model: nn.Module, test_X: np.ndarray, base_close: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_X, dtype=torch.float32, device=device)
        pred_return = model(x).detach().cpu().numpy()
    return base_close * (1.0 + pred_return)


def run_mlp_experiment(dataset: pd.DataFrame, target_col: str, lookback: int = 20, epochs: int = 25) -> DeepResult:
    payload, _ = build_sequence_data(dataset, target_col, lookback=lookback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(payload["sequence_length"] * payload["feature_count"])
    model = _train_torch_model(model, payload["train_X"], payload["train_y"], device=device, epochs=epochs)
    preds = _predict_price(model, payload["test_X"], payload["test_base_close"], device=device)
    metrics = _compute_metrics(payload["test_price"], preds, payload["train_rows"], payload["test_rows"])
    metrics.update({"feature_count": float(payload["feature_count"]), "lookback": float(payload["sequence_length"])})
    pred_df = pd.DataFrame({"date": payload["test_dates"], "actual": payload["test_price"], "predicted": preds})
    return DeepResult(metrics=metrics, predictions=pred_df)


def run_lstm_experiment(dataset: pd.DataFrame, target_col: str, lookback: int = 20, epochs: int = 25) -> DeepResult:
    payload, _ = build_sequence_data(dataset, target_col, lookback=lookback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(payload["feature_count"])
    model = _train_torch_model(model, payload["train_X"], payload["train_y"], device=device, epochs=epochs)
    preds = _predict_price(model, payload["test_X"], payload["test_base_close"], device=device)
    metrics = _compute_metrics(payload["test_price"], preds, payload["train_rows"], payload["test_rows"])
    metrics.update({"feature_count": float(payload["feature_count"]), "lookback": float(payload["sequence_length"])})
    pred_df = pd.DataFrame({"date": payload["test_dates"], "actual": payload["test_price"], "predicted": preds})
    return DeepResult(metrics=metrics, predictions=pred_df)


def run_transformer_experiment(dataset: pd.DataFrame, target_col: str, lookback: int = 20, epochs: int = 25) -> DeepResult:
    payload, _ = build_sequence_data(dataset, target_col, lookback=lookback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(payload["feature_count"])
    model = _train_torch_model(model, payload["train_X"], payload["train_y"], device=device, epochs=epochs)
    preds = _predict_price(model, payload["test_X"], payload["test_base_close"], device=device)
    metrics = _compute_metrics(payload["test_price"], preds, payload["train_rows"], payload["test_rows"])
    metrics.update({"feature_count": float(payload["feature_count"]), "lookback": float(payload["sequence_length"])})
    pred_df = pd.DataFrame({"date": payload["test_dates"], "actual": payload["test_price"], "predicted": preds})
    return DeepResult(metrics=metrics, predictions=pred_df)
