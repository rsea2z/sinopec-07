from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame


def time_split(df: pd.DataFrame, target_col: str, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = df.loc[df[target_col].notna()].copy()
    split_idx = int(len(usable) * train_ratio)
    if split_idx <= 0 or split_idx >= len(usable):
        raise ValueError("Dataset is too small for time split.")
    return usable.iloc[:split_idx].copy(), usable.iloc[split_idx:].copy()


def time_split_three_way(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = df.loc[df[target_col].notna()].copy()
    n = len(usable)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Dataset is too small for three-way split.")
    return usable.iloc[:train_end].copy(), usable.iloc[train_end:val_end].copy(), usable.iloc[val_end:].copy()


def _build_metric_dict(y_test: pd.Series, preds: np.ndarray, train_rows: int, test_rows: int, feature_count: int) -> dict[str, float]:
    y_true = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float)
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
        "feature_count": float(feature_count),
        "eval_rows": float(mask.sum()),
    }


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [col for col in train_df.columns if col not in {"date", target_col}]
    X_train = train_df[feature_cols].replace({pd.NA: np.nan})
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].replace({pd.NA: np.nan})
    y_test = test_df[target_col]

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    valid_feature_cols = [col for col in X_train.columns if X_train[col].notna().any()]
    X_train = X_train[valid_feature_cols]
    X_test = X_test[valid_feature_cols]
    return X_train, X_test, y_train, y_test


def fit_naive_persistence(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
    preds = test_df["收盘价"].to_numpy(dtype=float)
    y_test = test_df[target_col]
    metrics = _build_metric_dict(y_test, preds, len(train_df), len(test_df), 1)
    pred_df = pd.DataFrame({"date": test_df["date"].values, "actual": y_test.values, "predicted": preds})
    return TrainResult(metrics=metrics, predictions=pred_df)


def fit_ridge_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
    X_train, X_test, y_train, y_test = _prepare_features(train_df, test_df, target_col)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = _build_metric_dict(y_test, preds, len(train_df), len(test_df), X_train.shape[1])
    pred_df = pd.DataFrame({"date": test_df["date"].values, "actual": y_test.values, "predicted": preds})
    return TrainResult(metrics=metrics, predictions=pred_df)


def fit_random_forest_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
    X_train, X_test, y_train, y_test = _prepare_features(train_df, test_df, target_col)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train_imp, y_train)
    preds = model.predict(X_test_imp)

    metrics = _build_metric_dict(y_test, preds, len(train_df), len(test_df), X_train.shape[1])

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual": y_test.values,
            "predicted": preds,
        }
    )
    return TrainResult(metrics=metrics, predictions=pred_df)


def fit_random_forest_return(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
    X_train, X_test, _, y_test = _prepare_features(train_df, test_df, target_col)
    base_close_train = train_df["收盘价"].to_numpy(dtype=float)
    base_close_test = test_df["收盘价"].to_numpy(dtype=float)
    y_train_return = train_df[target_col].to_numpy(dtype=float) / base_close_train - 1.0

    valid_train_mask = np.isfinite(y_train_return)
    X_train = X_train.loc[valid_train_mask].copy()
    y_train_return = y_train_return[valid_train_mask]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train_imp, y_train_return)
    pred_return = model.predict(X_test_imp)
    preds = base_close_test * (1.0 + pred_return)

    metrics = _build_metric_dict(y_test, preds, len(train_df), len(test_df), X_train.shape[1])
    pred_df = pd.DataFrame({"date": test_df["date"].values, "actual": y_test.values, "predicted": preds})
    return TrainResult(metrics=metrics, predictions=pred_df)


def fit_baseline_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
    return fit_random_forest_direct(train_df, test_df, target_col)
