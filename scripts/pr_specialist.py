from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import RESULTS_DIR, TRACK_CONFIGS
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import (
    fit_naive_persistence,
    fit_random_forest_direct,
    fit_random_forest_return,
    fit_ridge_regressor,
    time_split_three_way,
)
from residual_search import fit_predict_residual_et, prepare_numeric, rmse, mae, select_topk_by_corr


HORIZONS = [5, 10, 20]
WINDOWS = [None, 80, 120, 160]
ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
MIN_WINDOW_ROWS = 60


def recent_slice(df: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if window is None or len(df) <= window:
        return df.copy()
    return df.tail(window).copy().reset_index(drop=True)


def fit_candidate(model_name: str, train_df: pd.DataFrame, pred_df: pd.DataFrame, target_col: str) -> np.ndarray:
    if model_name == "naive":
        return fit_naive_persistence(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "ridge":
        return fit_ridge_regressor(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "rf_direct":
        return fit_random_forest_direct(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "rf_return":
        return fit_random_forest_return(train_df, pred_df, target_col).predictions["predicted"].to_numpy(dtype=float)
    if model_name == "residual_et":
        train_X, _, _, train_residual = prepare_numeric(train_df, target_col)
        pred_X, _, pred_close, _ = prepare_numeric(pred_df, target_col)
        cols = select_topk_by_corr(train_X, train_residual, top_k=20)
        if not cols:
            return pred_close
        residual_pred = fit_predict_residual_et(train_X, train_residual, pred_X, cols)
        return pred_close + residual_pred
    raise ValueError(model_name)


def build_candidates(train_df: pd.DataFrame, pred_df: pd.DataFrame, target_col: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["naive"] = fit_candidate("naive", train_df, pred_df, target_col)
    for model_name in ["ridge", "rf_direct", "rf_return", "residual_et"]:
        for window in WINDOWS:
            windowed = recent_slice(train_df, window)
            if len(windowed) < MIN_WINDOW_ROWS:
                continue
            label = "all" if window is None else str(window)
            try:
                out[f"{model_name}_{label}"] = fit_candidate(model_name, windowed, pred_df, target_col)
            except Exception:
                continue
    return out


def main() -> None:
    config = TRACK_CONFIGS["track2"]
    symbol = "PR"
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        dataset = build_feature_dataset(
            config["trading"][symbol],
            config["main_contract_labels"][symbol],
            config["fundamentals"],
            horizon,
        )
        target_col = f"target_t_plus_{horizon}"
        train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)

        val_candidates = build_candidates(train_df, val_df, target_col)
        val_actual = pd.to_numeric(val_df[target_col], errors="coerce").to_numpy(dtype=float)
        best = {
            "name_a": "naive",
            "name_b": "naive",
            "alpha_b": 0.0,
            "val_rmse": rmse(val_actual, val_candidates["naive"]),
        }

        candidate_names = list(val_candidates.keys())
        for name_a in candidate_names:
            for name_b in candidate_names:
                pred_a = val_candidates[name_a]
                pred_b = val_candidates[name_b]
                for alpha_b in ALPHAS:
                    blend = (1.0 - alpha_b) * pred_a + alpha_b * pred_b
                    score = rmse(val_actual, blend)
                    search_rows.append(
                        {
                            "symbol": symbol,
                            "horizon": horizon,
                            "name_a": name_a,
                            "name_b": name_b,
                            "alpha_b": alpha_b,
                            "val_rmse": score,
                        }
                    )
                    if score < best["val_rmse"]:
                        best = {"name_a": name_a, "name_b": name_b, "alpha_b": alpha_b, "val_rmse": score}

        train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
        test_candidates = build_candidates(train_val_df, test_df, target_col)
        test_actual = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float)
        naive_test_pred = test_candidates["naive"]
        final_pred = (1.0 - best["alpha_b"]) * test_candidates[best["name_a"]] + best["alpha_b"] * test_candidates[best["name_b"]]

        out_df = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "actual": test_actual,
                "naive_pred": naive_test_pred,
                "specialist_pred": final_pred,
            }
        )
        out_df.to_csv(pred_dir / f"pr_specialist_h{horizon}.csv", index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "blend_a": best["name_a"],
                "blend_b": best["name_b"],
                "alpha_b": best["alpha_b"],
                "val_rmse": best["val_rmse"],
                "test_naive_rmse": rmse(test_actual, naive_test_pred),
                "test_rmse": rmse(test_actual, final_pred),
                "test_rmse_gain": rmse(test_actual, naive_test_pred) - rmse(test_actual, final_pred),
                "test_mae": mae(test_actual, final_pred),
                "candidate_count": len(test_candidates),
            }
        )

    summary = pd.DataFrame(summary_rows)
    search = pd.DataFrame(search_rows)
    summary.to_csv(metric_dir / "pr_specialist_summary.csv", index=False, encoding="utf-8-sig")
    search.to_csv(metric_dir / "pr_specialist_search.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
