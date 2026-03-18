from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import HORIZONS, RESULTS_DIR, TRACK_CONFIGS
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import time_split_three_way


TOP_K_LIST = [5, 10, 20, 40, 80]
ALPHAS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def prepare_numeric(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [c for c in df.columns if c not in {"date", target_col}]
    X = df[feature_cols].replace({pd.NA: np.nan}).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["收盘价"], errors="coerce").to_numpy(dtype=float)
    residual = y - close
    valid_cols = [c for c in X.columns if X[c].notna().any()]
    return X[valid_cols], y, close, residual


def select_topk_by_corr(X: pd.DataFrame, residual: np.ndarray, top_k: int) -> list[str]:
    target = pd.Series(residual)
    scores: list[tuple[str, float]] = []
    for col in X.columns:
        s = X[col]
        valid = s.notna() & target.notna()
        if valid.sum() < 20:
            continue
        corr = s[valid].corr(target[valid])
        if pd.notna(corr):
            scores.append((col, abs(float(corr))))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in scores[:top_k]]
    if "收盘价" in X.columns and "收盘价" not in selected:
        selected = ["收盘价"] + selected[:-1] if selected else ["收盘价"]
    return selected


def fit_predict_residual_ridge(
    train_X: pd.DataFrame,
    train_residual: np.ndarray,
    pred_X: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    valid = np.isfinite(train_residual)
    train_X = train_X.loc[valid, columns]
    train_residual = train_residual[valid]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_X)
    X_pred = imputer.transform(pred_X[columns])
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    model.fit(X_train, train_residual)
    return model.predict(X_pred)


def fit_predict_residual_et(
    train_X: pd.DataFrame,
    train_residual: np.ndarray,
    pred_X: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    valid = np.isfinite(train_residual)
    train_X = train_X.loc[valid, columns]
    train_residual = train_residual[valid]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_X)
    X_pred = imputer.transform(pred_X[columns])
    model = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train, train_residual)
    return model.predict(X_pred)


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

    train_X, train_y, train_close, train_residual = prepare_numeric(train_df, target_col)
    val_X, val_y, val_close, _ = prepare_numeric(val_df, target_col)
    test_X, test_y, test_close, _ = prepare_numeric(test_df, target_col)

    naive_val = val_close
    naive_val_rmse = rmse(val_y, naive_val)

    search_rows: list[dict[str, object]] = []
    best = {
        "model_family": "naive",
        "top_k": 0,
        "alpha_ml": 0.0,
        "val_rmse": naive_val_rmse,
        "selected_features": [],
    }

    for model_family, predictor in [("ridge_residual", fit_predict_residual_ridge), ("extratrees_residual", fit_predict_residual_et)]:
        for top_k in TOP_K_LIST:
            cols = select_topk_by_corr(train_X, train_residual, top_k)
            if not cols:
                continue
            val_residual_pred = predictor(train_X, train_residual, val_X, cols)
            val_ml_pred = val_close + val_residual_pred
            val_ml_rmse = rmse(val_y, val_ml_pred)
            search_rows.append(
                {
                    "track": track_key,
                    "symbol": symbol,
                    "horizon": horizon,
                    "stage": "model",
                    "candidate": model_family,
                    "top_k": top_k,
                    "alpha_ml": 1.0,
                    "rmse": val_ml_rmse,
                }
            )
            for alpha in ALPHAS:
                blend = (1.0 - alpha) * naive_val + alpha * val_ml_pred
                blend_rmse = rmse(val_y, blend)
                search_rows.append(
                    {
                        "track": track_key,
                        "symbol": symbol,
                        "horizon": horizon,
                        "stage": "blend",
                        "candidate": model_family,
                        "top_k": top_k,
                        "alpha_ml": alpha,
                        "rmse": blend_rmse,
                    }
                )
                if blend_rmse < best["val_rmse"]:
                    best = {
                        "model_family": model_family,
                        "top_k": top_k,
                        "alpha_ml": alpha,
                        "val_rmse": blend_rmse,
                        "selected_features": cols,
                    }

    train_val_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
    train_val_X, train_val_y, train_val_close, train_val_residual = prepare_numeric(train_val_df, target_col)
    columns = best["selected_features"]
    if columns:
        predictor = fit_predict_residual_ridge if best["model_family"] == "ridge_residual" else fit_predict_residual_et
        test_residual_pred = predictor(train_val_X, train_val_residual, test_X, columns)
        ml_test_pred = test_close + test_residual_pred
    else:
        ml_test_pred = test_close.copy()

    ensemble_pred = (1.0 - best["alpha_ml"]) * test_close + best["alpha_ml"] * ml_test_pred
    result_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual": test_y,
            "naive_pred": test_close,
            "ml_pred": ml_test_pred,
            "ensemble_pred": ensemble_pred,
        }
    )

    summary = {
        "track": track_key,
        "track_label": config["label"],
        "symbol": symbol,
        "horizon": horizon,
        "selected_model": best["model_family"],
        "top_k": best["top_k"],
        "alpha_ml": best["alpha_ml"],
        "val_rmse": best["val_rmse"],
        "test_naive_rmse": rmse(test_y, test_close),
        "test_rmse": rmse(test_y, ensemble_pred),
        "test_rmse_gain": rmse(test_y, test_close) - rmse(test_y, ensemble_pred),
        "test_mae": mae(test_y, ensemble_pred),
        "train_rows": len(train_val_df),
        "test_rows": len(test_df),
        "selected_features": "|".join(columns[:20]),
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
                    pred_dir / f"residual_{track_key}_{symbol}_h{horizon}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    summary_df = pd.DataFrame(summaries).sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    search_df = pd.DataFrame(searches)
    summary_df.to_csv(metric_dir / "residual_search_summary.csv", index=False, encoding="utf-8-sig")
    search_df.to_csv(metric_dir / "residual_search_grid.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
