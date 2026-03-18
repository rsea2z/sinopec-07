from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import TRACK_CONFIGS, RESULTS_DIR
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import (
    fit_naive_persistence,
    fit_random_forest_direct,
    fit_random_forest_return,
    time_split_three_way,
)
from residual_search import (
    fit_predict_residual_et,
    prepare_numeric,
    rmse,
    mae,
    select_topk_by_corr,
)


PX_HORIZONS = [5, 10, 20]
ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


def main() -> None:
    config = TRACK_CONFIGS["track2"]
    symbol = "PX"
    trading_path = config["trading"][symbol]
    marker = config["main_contract_labels"][symbol]
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []

    for horizon in PX_HORIZONS:
        dataset = build_feature_dataset(trading_path, marker, config["fundamentals"], horizon)
        target_col = f"target_t_plus_{horizon}"
        train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)

        naive_val = fit_naive_persistence(train_df, val_df, target_col)
        naive_val_pred = naive_val.predictions["predicted"].to_numpy(dtype=float)
        val_actual = naive_val.predictions["actual"].to_numpy(dtype=float)

        rf_direct_val = fit_random_forest_direct(train_df, val_df, target_col)
        rf_return_val = fit_random_forest_return(train_df, val_df, target_col)

        train_X, _, _, train_residual = prepare_numeric(train_df, target_col)
        val_X, _, val_close, _ = prepare_numeric(val_df, target_col)
        cols = select_topk_by_corr(train_X, train_residual, top_k=40)
        residual_pred_val = fit_predict_residual_et(train_X, train_residual, val_X, cols)
        residual_val_pred = val_close + residual_pred_val

        candidates = {
            "naive": naive_val_pred,
            "rf_direct": rf_direct_val.predictions["predicted"].to_numpy(dtype=float),
            "rf_return": rf_return_val.predictions["predicted"].to_numpy(dtype=float),
            "residual_et": residual_val_pred,
        }

        best = {
            "name_a": "naive",
            "name_b": "naive",
            "alpha_b": 0.0,
            "val_rmse": rmse(val_actual, naive_val_pred),
        }

        candidate_names = list(candidates.keys())
        for name_a in candidate_names:
            for name_b in candidate_names:
                pred_a = candidates[name_a]
                pred_b = candidates[name_b]
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
        naive_test = fit_naive_persistence(train_val_df, test_df, target_col)
        rf_direct_test = fit_random_forest_direct(train_val_df, test_df, target_col)
        rf_return_test = fit_random_forest_return(train_val_df, test_df, target_col)

        tv_X, _, _, tv_residual = prepare_numeric(train_val_df, target_col)
        test_X, test_y, test_close, _ = prepare_numeric(test_df, target_col)
        cols = select_topk_by_corr(tv_X, tv_residual, top_k=40)
        residual_pred_test = fit_predict_residual_et(tv_X, tv_residual, test_X, cols)
        residual_test_pred = test_close + residual_pred_test

        test_candidates = {
            "naive": naive_test.predictions["predicted"].to_numpy(dtype=float),
            "rf_direct": rf_direct_test.predictions["predicted"].to_numpy(dtype=float),
            "rf_return": rf_return_test.predictions["predicted"].to_numpy(dtype=float),
            "residual_et": residual_test_pred,
        }

        final_pred = (1.0 - best["alpha_b"]) * test_candidates[best["name_a"]] + best["alpha_b"] * test_candidates[best["name_b"]]
        naive_test_pred = test_candidates["naive"]
        result_df = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "actual": test_y,
                "naive_pred": naive_test_pred,
                "rf_direct_pred": test_candidates["rf_direct"],
                "rf_return_pred": test_candidates["rf_return"],
                "residual_et_pred": test_candidates["residual_et"],
                "specialist_pred": final_pred,
            }
        )
        result_df.to_csv(pred_dir / f"px_specialist_h{horizon}.csv", index=False, encoding="utf-8-sig")

        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "blend_a": best["name_a"],
                "blend_b": best["name_b"],
                "alpha_b": best["alpha_b"],
                "val_rmse": best["val_rmse"],
                "test_naive_rmse": rmse(test_y, naive_test_pred),
                "test_rmse": rmse(test_y, final_pred),
                "test_rmse_gain": rmse(test_y, naive_test_pred) - rmse(test_y, final_pred),
                "test_mae": mae(test_y, final_pred),
            }
        )

    summary = pd.DataFrame(rows)
    pd.DataFrame(search_rows).to_csv(metric_dir / "px_specialist_search.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(metric_dir / "px_specialist_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
