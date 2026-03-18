from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import align_fundamental_to_calendar, normalize_trading_data, normalize_wide_fundamental


PRICE_COLUMNS = ["开盘价", "最高价", "最低价", "收盘价", "结算价", "成交量", "持仓量"]
LAGS = (1, 2, 3, 5, 10, 20)
ROLL_WINDOWS = (5, 10, 20)


def build_target_frame(trading_path: Path, main_contract_marker: str) -> pd.DataFrame:
    df = normalize_trading_data(trading_path, main_contract_marker)
    keep_cols = ["date", "contract_label"] + [col for col in PRICE_COLUMNS if col in df.columns]
    out = df[keep_cols].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["收盘价"].pct_change(fill_method=None)
    out["intraday_range"] = (out["最高价"] - out["最低价"]) / out["收盘价"].replace(0, np.nan)
    out["open_close_spread"] = (out["收盘价"] - out["开盘价"]) / out["开盘价"].replace(0, np.nan)
    out["volume_oi_ratio"] = out["成交量"] / out["持仓量"].replace(0, np.nan)

    for lag in LAGS:
        out[f"close_lag_{lag}"] = out["收盘价"].shift(lag)
        out[f"return_lag_{lag}"] = out["收盘价"].pct_change(lag, fill_method=None)
        out[f"volume_lag_{lag}"] = out["成交量"].shift(lag)

    for window in ROLL_WINDOWS:
        out[f"close_ma_{window}"] = out["收盘价"].rolling(window).mean()
        out[f"close_std_{window}"] = out["收盘价"].rolling(window).std()
        out[f"volume_ma_{window}"] = out["成交量"].rolling(window).mean()

    return out


def build_feature_dataset(
    trading_path: Path,
    main_contract_marker: str,
    fundamental_paths: list[Path],
    horizon: int,
) -> pd.DataFrame:
    target = build_target_frame(trading_path, main_contract_marker)
    dataset = add_price_features(target)

    for factor_path in fundamental_paths:
        factor_df = normalize_wide_fundamental(factor_path)
        aligned = align_fundamental_to_calendar(dataset[["date"]], factor_df)
        aligned = aligned.drop(columns=["date"])
        dataset = pd.concat([dataset.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)

    dataset[f"target_t_plus_{horizon}"] = dataset["收盘价"].shift(-horizon)
    dataset = dataset.drop(columns=["contract_label"])
    dataset = dataset.sort_values("date").reset_index(drop=True)
    return dataset
