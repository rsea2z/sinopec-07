from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import RESULTS_DIR, TRACK_CONFIGS
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import fit_naive_persistence, time_split_three_way
from residual_search import rmse, mae, prepare_numeric, select_topk_by_corr


SEQ_COLS = [
    "return_1d",
    "intraday_range",
    "open_close_spread",
    "volume_oi_ratio",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_5",
    "close_lag_10",
    "close_lag_20",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "return_lag_20",
    "close_ma_5",
    "close_ma_10",
    "close_ma_20",
    "close_std_5",
    "close_std_10",
    "close_std_20",
    "volume_ma_5",
    "volume_ma_10",
    "volume_ma_20",
]


class FusionDataset(Dataset):
    def __init__(self, seq_x: np.ndarray, static_x: np.ndarray, y: np.ndarray) -> None:
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


class AttentionFusionRegressor(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, d_model: int = 64, nhead: int = 4) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        seq_h = self.encoder(self.seq_proj(seq_x))[:, -1, :]
        static_h = self.static_net(static_x)
        gate = self.gate(torch.cat([seq_h, static_h], dim=1))
        fused = torch.cat([seq_h * gate, static_h * (1.0 - gate)], dim=1)
        return self.head(fused).squeeze(-1)


def build_px_fusion_data(dataset: pd.DataFrame, target_col: str, lookback: int = 20, top_k: int = 20):
    df = dataset.copy().sort_values("date").reset_index(drop=True)
    seq_cols = [c for c in SEQ_COLS if c in df.columns]

    X_numeric, y_price, close_price, residual = prepare_numeric(df, target_col)
    static_candidates = [c for c in X_numeric.columns if c not in seq_cols and c not in {"开盘价", "最高价", "最低价", "收盘价", "结算价", "成交量", "持仓量"}]
    static_candidates_df = X_numeric[static_candidates].copy()
    selected_static = select_topk_by_corr(static_candidates_df, residual, top_k=top_k)

    work = df[["date", "收盘价", target_col] + seq_cols].copy()
    work[seq_cols] = work[seq_cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
    static_df = static_candidates_df[selected_static].apply(pd.to_numeric, errors="coerce")

    seq_rows = []
    static_rows = []
    target_returns = []
    target_prices = []
    base_prices = []
    dates = []

    for idx in range(lookback - 1, len(work)):
        seq_window = work.loc[idx - lookback + 1 : idx, seq_cols].to_numpy(dtype=float)
        static_row = static_df.iloc[idx].to_numpy(dtype=float)
        target_price = pd.to_numeric(work.iloc[idx][target_col], errors="coerce")
        base_price = pd.to_numeric(work.iloc[idx]["收盘价"], errors="coerce")
        if not np.isfinite(seq_window).all():
            continue
        if not np.isfinite(target_price) or not np.isfinite(base_price) or base_price == 0:
            continue
        seq_rows.append(seq_window)
        static_rows.append(static_row)
        target_returns.append(target_price / base_price - 1.0)
        target_prices.append(target_price)
        base_prices.append(base_price)
        dates.append(work.iloc[idx]["date"])

    seq_x = np.array(seq_rows, dtype=np.float32)
    static_x = np.array(static_rows, dtype=np.float32)
    y_ret = np.array(target_returns, dtype=np.float32)
    y_price = np.array(target_prices, dtype=np.float32)
    base_prices = np.array(base_prices, dtype=np.float32)
    dates = np.array(dates)

    n = len(seq_x)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_seq = seq_x[:train_end]
    val_seq = seq_x[train_end:val_end]
    test_seq = seq_x[val_end:]
    train_static = static_x[:train_end]
    val_static = static_x[train_end:val_end]
    test_static = static_x[val_end:]
    train_y = y_ret[:train_end]
    val_y = y_ret[train_end:val_end]
    test_y = y_ret[val_end:]

    seq_mean = train_seq.mean(axis=(0, 1), keepdims=True)
    seq_std = train_seq.std(axis=(0, 1), keepdims=True)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)
    train_seq = (train_seq - seq_mean) / seq_std
    val_seq = (val_seq - seq_mean) / seq_std
    test_seq = (test_seq - seq_mean) / seq_std

    imp = SimpleImputer(strategy="median")
    train_static = imp.fit_transform(train_static)
    val_static = imp.transform(val_static)
    test_static = imp.transform(test_static)
    static_mean = train_static.mean(axis=0, keepdims=True)
    static_std = train_static.std(axis=0, keepdims=True)
    static_std = np.where(static_std < 1e-6, 1.0, static_std)
    train_static = (train_static - static_mean) / static_std
    val_static = (val_static - static_mean) / static_std
    test_static = (test_static - static_mean) / static_std

    return {
        "train_seq": train_seq,
        "val_seq": val_seq,
        "test_seq": test_seq,
        "train_static": train_static,
        "val_static": val_static,
        "test_static": test_static,
        "train_y": train_y,
        "val_y": val_y,
        "test_y": test_y,
        "val_price": y_price[train_end:val_end],
        "test_price": y_price[val_end:],
        "val_base": base_prices[train_end:val_end],
        "test_base": base_prices[val_end:],
        "val_dates": dates[train_end:val_end],
        "test_dates": dates[val_end:],
        "selected_static": selected_static,
        "lookback": lookback,
    }


def train_model(payload: dict[str, object], epochs: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionFusionRegressor(
        seq_dim=payload["train_seq"].shape[-1],
        static_dim=payload["train_static"].shape[-1],
    ).to(device)
    train_ds = FusionDataset(payload["train_seq"], payload["train_static"], payload["train_y"])
    val_seq = torch.tensor(payload["val_seq"], dtype=torch.float32, device=device)
    val_static = torch.tensor(payload["val_static"], dtype=torch.float32, device=device)
    val_y = payload["val_y"]

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for seq_x, static_x, y in loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(seq_x, static_x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(val_seq, val_static).detach().cpu().numpy()
        val_rmse = rmse(val_y, pred_val)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val, device


def predict_prices(model, device, seq_x, static_x, base_close):
    model.eval()
    with torch.no_grad():
        seq_t = torch.tensor(seq_x, dtype=torch.float32, device=device)
        static_t = torch.tensor(static_x, dtype=torch.float32, device=device)
        pred_ret = model(seq_t, static_t).detach().cpu().numpy()
    return base_close * (1.0 + pred_ret)


def main():
    config = TRACK_CONFIGS["track2"]
    symbol = "PX"
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for horizon in [5, 10, 20]:
        dataset = build_feature_dataset(
            config["trading"][symbol],
            config["main_contract_labels"][symbol],
            config["fundamentals"],
            horizon,
        )
        target_col = f"target_t_plus_{horizon}"

        payload = build_px_fusion_data(dataset, target_col=target_col, lookback=20, top_k=20)
        model, val_ret_rmse, device = train_model(payload, epochs=30)
        dl_val_pred = predict_prices(model, device, payload["val_seq"], payload["val_static"], payload["val_base"])
        dl_test_pred = predict_prices(model, device, payload["test_seq"], payload["test_static"], payload["test_base"])

        train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)
        naive_val = fit_naive_persistence(train_df, val_df, target_col)
        naive_test = fit_naive_persistence(pd.concat([train_df, val_df]).sort_values("date"), test_df, target_col)
        naive_val_pred = naive_val.predictions["predicted"].to_numpy(dtype=float)
        actual_val = naive_val.predictions["actual"].to_numpy(dtype=float)
        naive_pred = naive_test.predictions["predicted"].to_numpy(dtype=float)
        actual = naive_test.predictions["actual"].to_numpy(dtype=float)

        aligned_val_len = min(len(dl_val_pred), len(naive_val_pred), len(actual_val))
        dl_val_pred = dl_val_pred[-aligned_val_len:]
        naive_val_pred = naive_val_pred[-aligned_val_len:]
        actual_val = actual_val[-aligned_val_len:]

        best_alpha = 0.0
        best_val_rmse = rmse(actual_val, naive_val_pred)
        for alpha in np.linspace(0.0, 1.0, 21):
            blended_val = (1.0 - alpha) * naive_val_pred + alpha * dl_val_pred
            score = rmse(actual_val, blended_val)
            if score < best_val_rmse:
                best_val_rmse = score
                best_alpha = float(alpha)

        aligned_len = min(len(dl_test_pred), len(naive_pred), len(actual))
        naive_pred = naive_pred[-aligned_len:]
        actual = actual[-aligned_len:]
        dl_test_pred = dl_test_pred[-aligned_len:]
        final_pred = (1.0 - best_alpha) * naive_pred + best_alpha * dl_test_pred

        pd.DataFrame(
            {
                "date": naive_test.predictions["date"].tail(aligned_len).values,
                "actual": actual,
                "naive_pred": naive_pred,
                "attention_fusion_pred": dl_test_pred,
                "blended_pred": final_pred,
            }
        ).to_csv(pred_dir / f"px_attention_fusion_h{horizon}.csv", index=False, encoding="utf-8-sig")

        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "selected_static_count": len(payload["selected_static"]),
                "selected_static_preview": "|".join(payload["selected_static"][:12]),
                "alpha_dl": best_alpha,
                "val_blended_rmse": best_val_rmse,
                "test_naive_rmse": rmse(actual, naive_pred),
                "test_dl_rmse": rmse(actual, dl_test_pred),
                "test_blended_rmse": rmse(actual, final_pred),
                "test_blended_gain": rmse(actual, naive_pred) - rmse(actual, final_pred),
                "test_blended_mae": mae(actual, final_pred),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(metric_dir / "px_attention_fusion_summary.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
