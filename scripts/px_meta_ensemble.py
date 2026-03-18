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
    time_split_three_way,
)
from px_attention_fusion import build_px_fusion_data, predict_prices, train_model
from residual_search import fit_predict_residual_et, prepare_numeric, rmse, mae, select_topk_by_corr


ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


def build_specialist_candidate_predictions(dataset: pd.DataFrame, target_col: str):
    train_df, val_df, test_df = time_split_three_way(dataset, target_col=target_col)

    naive_val = fit_naive_persistence(train_df, val_df, target_col).predictions.rename(columns={"predicted": "naive"})
    rf_direct_val = fit_random_forest_direct(train_df, val_df, target_col).predictions.rename(columns={"predicted": "rf_direct"})
    rf_return_val = fit_random_forest_return(train_df, val_df, target_col).predictions.rename(columns={"predicted": "rf_return"})

    train_X, _, _, train_residual = prepare_numeric(train_df, target_col)
    val_X, _, val_close, _ = prepare_numeric(val_df, target_col)
    cols = select_topk_by_corr(train_X, train_residual, top_k=40)
    residual_val = fit_predict_residual_et(train_X, train_residual, val_X, cols)
    residual_val_df = pd.DataFrame({"date": val_df["date"].values, "actual": val_df[target_col].values, "residual_et": val_close + residual_val})

    train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
    naive_test = fit_naive_persistence(train_val_df, test_df, target_col).predictions.rename(columns={"predicted": "naive"})
    rf_direct_test = fit_random_forest_direct(train_val_df, test_df, target_col).predictions.rename(columns={"predicted": "rf_direct"})
    rf_return_test = fit_random_forest_return(train_val_df, test_df, target_col).predictions.rename(columns={"predicted": "rf_return"})

    tv_X, _, _, tv_residual = prepare_numeric(train_val_df, target_col)
    test_X, _, test_close, _ = prepare_numeric(test_df, target_col)
    cols = select_topk_by_corr(tv_X, tv_residual, top_k=40)
    residual_test = fit_predict_residual_et(tv_X, tv_residual, test_X, cols)
    residual_test_df = pd.DataFrame({"date": test_df["date"].values, "actual": test_df[target_col].values, "residual_et": test_close + residual_test})

    val_merged = naive_val[["date", "actual", "naive"]].merge(
        rf_direct_val[["date", "rf_direct"]], on="date"
    ).merge(
        rf_return_val[["date", "rf_return"]], on="date"
    ).merge(
        residual_val_df[["date", "residual_et"]], on="date"
    )

    test_merged = naive_test[["date", "actual", "naive"]].merge(
        rf_direct_test[["date", "rf_direct"]], on="date"
    ).merge(
        rf_return_test[["date", "rf_return"]], on="date"
    ).merge(
        residual_test_df[["date", "residual_et"]], on="date"
    )
    return val_merged, test_merged


def build_attention_predictions(dataset: pd.DataFrame, target_col: str):
    payload = build_px_fusion_data(dataset, target_col=target_col, lookback=20, top_k=20)
    model, _, device = train_model(payload, epochs=30)
    val_pred = predict_prices(model, device, payload["val_seq"], payload["val_static"], payload["val_base"])
    test_pred = predict_prices(model, device, payload["test_seq"], payload["test_static"], payload["test_base"])

    val_df = pd.DataFrame({"date": payload["val_dates"], "actual": payload["val_price"], "attention": val_pred})
    test_df = pd.DataFrame({"date": payload["test_dates"], "actual": payload["test_price"], "attention": test_pred})
    return val_df, test_df


def main() -> None:
    config = TRACK_CONFIGS["track2"]
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    search_rows = []
    for horizon in [5, 10, 20]:
        dataset = build_feature_dataset(
            config["trading"]["PX"],
            config["main_contract_labels"]["PX"],
            config["fundamentals"],
            horizon,
        )
        target_col = f"target_t_plus_{horizon}"
        val_base, test_base = build_specialist_candidate_predictions(dataset, target_col)
        val_att, test_att = build_attention_predictions(dataset, target_col)

        val_df = val_base.merge(val_att[["date", "attention"]], on="date", how="inner").dropna()
        test_df = test_base.merge(test_att[["date", "attention"]], on="date", how="inner").dropna()

        candidates = ["naive", "rf_direct", "rf_return", "residual_et", "attention"]
        best = {
            "name_a": "naive",
            "name_b": "naive",
            "alpha_b": 0.0,
            "val_rmse": rmse(val_df["actual"].to_numpy(float), val_df["naive"].to_numpy(float)),
        }

        for name_a in candidates:
            for name_b in candidates:
                pred_a = val_df[name_a].to_numpy(dtype=float)
                pred_b = val_df[name_b].to_numpy(dtype=float)
                for alpha_b in ALPHAS:
                    blend = (1.0 - alpha_b) * pred_a + alpha_b * pred_b
                    score = rmse(val_df["actual"].to_numpy(dtype=float), blend)
                    search_rows.append(
                        {
                            "symbol": "PX",
                            "horizon": horizon,
                            "name_a": name_a,
                            "name_b": name_b,
                            "alpha_b": alpha_b,
                            "val_rmse": score,
                        }
                    )
                    if score < best["val_rmse"]:
                        best = {"name_a": name_a, "name_b": name_b, "alpha_b": alpha_b, "val_rmse": score}

        final_pred = (1.0 - best["alpha_b"]) * test_df[best["name_a"]].to_numpy(dtype=float) + best["alpha_b"] * test_df[best["name_b"]].to_numpy(dtype=float)
        actual = test_df["actual"].to_numpy(dtype=float)
        naive_pred = test_df["naive"].to_numpy(dtype=float)

        output = test_df.copy()
        output["meta_pred"] = final_pred
        output.to_csv(pred_dir / f"px_meta_ensemble_h{horizon}.csv", index=False, encoding="utf-8-sig")

        rows.append(
            {
                "symbol": "PX",
                "horizon": horizon,
                "blend_a": best["name_a"],
                "blend_b": best["name_b"],
                "alpha_b": best["alpha_b"],
                "val_rmse": best["val_rmse"],
                "test_naive_rmse": rmse(actual, naive_pred),
                "test_rmse": rmse(actual, final_pred),
                "test_rmse_gain": rmse(actual, naive_pred) - rmse(actual, final_pred),
                "test_mae": mae(actual, final_pred),
                "eval_rows": len(test_df),
            }
        )

    summary = pd.DataFrame(rows)
    pd.DataFrame(search_rows).to_csv(metric_dir / "px_meta_ensemble_search.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(metric_dir / "px_meta_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
