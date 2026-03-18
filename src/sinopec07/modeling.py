from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


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


def fit_baseline_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> TrainResult:
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

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train_imp, y_train)
    preds = model.predict(X_test_imp)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, preds)),
        "mape": float(np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100),
        "train_rows": float(len(train_df)),
        "test_rows": float(len(test_df)),
        "feature_count": float(len(valid_feature_cols)),
    }

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual": y_test.values,
            "predicted": preds,
        }
    )
    return TrainResult(metrics=metrics, predictions=pred_df)
