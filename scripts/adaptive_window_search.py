from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import HORIZONS, RESULTS_DIR, TRACK_CONFIGS
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import (
    fit_naive_persistence,
    fit_random_forest_direct,
    fit_random_forest_return,
    fit_ridge_regressor,
    time_split_three_way,
)
from residual_search import fit_predict_residual_et, prepare_numeric, rmse, mae, select_topk_by_corr


WINDOWS = [None, 252, 504, 756]
ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
MIN_WINDOW_ROWS = 80


def recent_slice(df: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if window is None or len(df) <= window:
        return df.copy()
    return df.tail(window).copy().reset_index(drop=True)


def fit_predict_candidate(
    model_name: str,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    if model_name == "ridge_direct":
        return fit_ridge_regressor(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "random_forest_direct":
        return fit_random_forest_direct(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "random_forest_return":
        return fit_random_forest_return(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "residual_et":
        train_X, _, _, train_residual = prepare_numeric(train_df, target_col)
        pred_X, _, pred_close, _ = prepare_numeric(pred_df, target_col)
        cols = select_topk_by_corr(train_X, train_residual, top_k=40)
        if not cols:
            return pred_close
        residual_pred = fit_predict_residual_et(train_X, train_residual, pred_X, cols)
        return pred_close + residual_pred
    raise ValueError(f"Unknown model: {model_name}")


def search_task(track_key: str, symbol: str, horizon: int) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    config = TRACK_CONFIGS[track_key]
    dataset = build_feature_dataset(
        config["trading"][symbol],
        config["main_contract_labels"][symbol],
        config["fundamentals"],
        horizon,
    )
    target_col = f"target_t_plus_{horizon}"
    train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)

    naive_val = fit_naive_persistence(train_df, val_df, target_col)
    naive_val_pred = naive_val.predictions["predicted"].to_numpy(dtype=float)
    val_actual = naive_val.predictions["actual"].to_numpy(dtype=float)
    naive_val_rmse = rmse(val_actual, naive_val_pred)

    search_rows: list[dict[str, object]] = []
    best = {
        "model_name": "naive",
        "window": "all",
        "alpha_ml": 0.0,
        "val_rmse": naive_val_rmse,
    }

    for model_name in ["ridge_direct", "random_forest_direct", "random_forest_return", "residual_et"]:
        for window in WINDOWS:
            windowed_train = recent_slice(train_df, window)
            if len(windowed_train) < MIN_WINDOW_ROWS:
                continue
            try:
                ml_val_pred = fit_predict_candidate(model_name, windowed_train, val_df, target_col)
            except Exception:
                continue
            model_val_rmse = rmse(val_actual, ml_val_pred)
            window_label = "all" if window is None else str(window)
            search_rows.append(
                {
                    "track": track_key,
                    "symbol": symbol,
                    "horizon": horizon,
                    "stage": "model",
                    "candidate": model_name,
                    "window": window_label,
                    "alpha_ml": 1.0,
                    "rmse": model_val_rmse,
                }
            )
            for alpha in ALPHAS:
                blend = (1.0 - alpha) * naive_val_pred + alpha * ml_val_pred
                blend_rmse = rmse(val_actual, blend)
                search_rows.append(
                    {
                        "track": track_key,
                        "symbol": symbol,
                        "horizon": horizon,
                        "stage": "blend",
                        "candidate": model_name,
                        "window": window_label,
                        "alpha_ml": alpha,
                        "rmse": blend_rmse,
                    }
                )
                if blend_rmse < best["val_rmse"]:
                    best = {
                        "model_name": model_name,
                        "window": window_label,
                        "alpha_ml": alpha,
                        "val_rmse": blend_rmse,
                    }

    train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
    test_actual = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float)
    naive_test_pred = fit_naive_persistence(train_val_df, test_df, target_col).predictions["predicted"].to_numpy(dtype=float)

    if best["model_name"] == "naive":
        ml_test_pred = naive_test_pred.copy()
        train_rows_used = len(train_val_df)
    else:
        window = None if best["window"] == "all" else int(best["window"])
        windowed_train_val = recent_slice(train_val_df, window)
        train_rows_used = len(windowed_train_val)
        ml_test_pred = fit_predict_candidate(best["model_name"], windowed_train_val, test_df, target_col)

    final_pred = (1.0 - best["alpha_ml"]) * naive_test_pred + best["alpha_ml"] * ml_test_pred

    result_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual": test_actual,
            "naive_pred": naive_test_pred,
            "ml_pred": ml_test_pred,
            "adaptive_pred": final_pred,
        }
    )
    summary = {
        "track": track_key,
        "track_label": config["label"],
        "symbol": symbol,
        "horizon": horizon,
        "selected_model": best["model_name"],
        "window": best["window"],
        "alpha_ml": best["alpha_ml"],
        "val_rmse": best["val_rmse"],
        "test_naive_rmse": rmse(test_actual, naive_test_pred),
        "test_rmse": rmse(test_actual, final_pred),
        "test_rmse_gain": rmse(test_actual, naive_test_pred) - rmse(test_actual, final_pred),
        "test_mae": mae(test_actual, final_pred),
        "train_rows_used": train_rows_used,
        "test_rows": len(test_df),
    }
    return summary, search_rows, result_df


def main() -> None:
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    searches: list[dict[str, object]] = []

    for track_key, config in TRACK_CONFIGS.items():
        for symbol in config["trading"]:
            for horizon in HORIZONS:
                summary, search_rows, result_df = search_task(track_key, symbol, horizon)
                summaries.append(summary)
                searches.extend(search_rows)
                result_df.to_csv(
                    pred_dir / f"adaptive_window_{track_key}_{symbol}_h{horizon}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    summary_df = pd.DataFrame(summaries).sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    search_df = pd.DataFrame(searches)
    summary_df.to_csv(metric_dir / "adaptive_window_summary.csv", index=False, encoding="utf-8-sig")
    search_df.to_csv(metric_dir / "adaptive_window_search.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
