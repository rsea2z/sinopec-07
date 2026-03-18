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


MODEL_REGISTRY = {
    "ridge_direct": fit_ridge_regressor,
    "random_forest_direct": fit_random_forest_direct,
    "random_forest_return": fit_random_forest_return,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def main() -> None:
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    val_search_rows: list[dict[str, object]] = []

    for track_key, config in TRACK_CONFIGS.items():
        for symbol, trading_path in config["trading"].items():
            marker = config["main_contract_labels"][symbol]
            for horizon in HORIZONS:
                dataset = build_feature_dataset(trading_path, marker, config["fundamentals"], horizon)
                target_col = f"target_t_plus_{horizon}"
                train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)

                naive_val = fit_naive_persistence(train_df, val_df, target_col)
                naive_val_pred = naive_val.predictions["predicted"].to_numpy(dtype=float)
                naive_val_true = naive_val.predictions["actual"].to_numpy(dtype=float)
                naive_val_rmse = rmse(naive_val_true, naive_val_pred)

                best_model_name = None
                best_model_fn = None
                best_model_val_rmse = np.inf
                best_model_val_pred = None

                for model_name, model_fn in MODEL_REGISTRY.items():
                    val_result = model_fn(train_df, val_df, target_col)
                    val_pred = val_result.predictions["predicted"].to_numpy(dtype=float)
                    val_true = val_result.predictions["actual"].to_numpy(dtype=float)
                    current_rmse = rmse(val_true, val_pred)
                    val_search_rows.append(
                        {
                            "track": track_key,
                            "symbol": symbol,
                            "horizon": horizon,
                            "stage": "validation_model",
                            "candidate": model_name,
                            "rmse": current_rmse,
                        }
                    )
                    if current_rmse < best_model_val_rmse:
                        best_model_name = model_name
                        best_model_fn = model_fn
                        best_model_val_rmse = current_rmse
                        best_model_val_pred = val_pred

                best_alpha = 0.0
                best_blend_val_rmse = naive_val_rmse
                for alpha in np.linspace(0.0, 1.0, 21):
                    blend_pred = (1.0 - alpha) * naive_val_pred + alpha * best_model_val_pred
                    blend_rmse = rmse(naive_val_true, blend_pred)
                    val_search_rows.append(
                        {
                            "track": track_key,
                            "symbol": symbol,
                            "horizon": horizon,
                            "stage": "validation_blend",
                            "candidate": best_model_name,
                            "alpha_ml": alpha,
                            "rmse": blend_rmse,
                        }
                    )
                    if blend_rmse < best_blend_val_rmse:
                        best_blend_val_rmse = blend_rmse
                        best_alpha = float(alpha)

                train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
                naive_test = fit_naive_persistence(train_val_df, test_df, target_col)
                ml_test = best_model_fn(train_val_df, test_df, target_col)
                naive_test_pred = naive_test.predictions["predicted"].to_numpy(dtype=float)
                ml_test_pred = ml_test.predictions["predicted"].to_numpy(dtype=float)
                final_pred = (1.0 - best_alpha) * naive_test_pred + best_alpha * ml_test_pred
                actual = ml_test.predictions["actual"].to_numpy(dtype=float)
                naive_test_rmse = rmse(actual, naive_test_pred)
                final_rmse = rmse(actual, final_pred)
                final_mae = mae(actual, final_pred)

                result_df = pd.DataFrame(
                    {
                        "date": ml_test.predictions["date"],
                        "actual": actual,
                        "naive_pred": naive_test_pred,
                        "ml_pred": ml_test_pred,
                        "ensemble_pred": final_pred,
                    }
                )
                result_df.to_csv(
                    pred_dir / f"competition_{track_key}_{symbol}_h{horizon}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                rows.append(
                    {
                        "track": track_key,
                        "track_label": config["label"],
                        "symbol": symbol,
                        "horizon": horizon,
                        "selected_model": best_model_name,
                        "alpha_ml": best_alpha,
                        "val_naive_rmse": naive_val_rmse,
                        "val_best_ml_rmse": best_model_val_rmse,
                        "val_best_blend_rmse": best_blend_val_rmse,
                        "test_naive_rmse": naive_test_rmse,
                        "test_rmse": final_rmse,
                        "test_rmse_gain": naive_test_rmse - final_rmse,
                        "test_mae": final_mae,
                        "train_rows": len(train_val_df),
                        "test_rows": len(test_df),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    search = pd.DataFrame(val_search_rows)
    summary.to_csv(metric_dir / "competition_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    search.to_csv(metric_dir / "competition_ensemble_search.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
