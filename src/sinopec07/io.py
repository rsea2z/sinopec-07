from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


CSV_ENCODINGS = ("utf-8", "gbk", "gb18030", "latin1")


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    raise RuntimeError(f"Failed to read csv: {path}") from last_error


def parse_mixed_date(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y/%m", "%Y-%m"):
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        if parsed.notna().sum() == len(cleaned):
            return parsed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(cleaned, errors="coerce")


def clean_numeric_frame(df: pd.DataFrame, skip_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    skip = set(skip_columns)
    for column in out.columns:
        if column in skip:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            continue
        series = out[column].astype(str).str.replace(",", "", regex=False).str.strip()
        series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() > 0:
            out[column] = converted
    return out


def normalize_trading_data(path: Path, main_contract_marker: str) -> pd.DataFrame:
    raw = read_csv_auto(path)
    date_col = next(col for col in raw.columns if "日期" in str(col))
    code_col = raw.columns[1]
    raw[date_col] = parse_mixed_date(raw[date_col])
    raw = raw.loc[raw[date_col].notna()].copy()
    raw = clean_numeric_frame(raw, skip_columns=[date_col, code_col])

    codes = raw[code_col].astype(str)
    if "主力合约" in main_contract_marker:
        mask = codes.str.contains(main_contract_marker, na=False)
    else:
        mask = codes.eq(main_contract_marker)

    selected = raw.loc[mask].copy()
    selected = selected.rename(columns={date_col: "date", code_col: "contract_label"})
    selected = selected.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    selected["target_close"] = selected["收盘价"]
    return selected.reset_index(drop=True)


def normalize_wide_fundamental(path: Path) -> pd.DataFrame:
    raw = read_csv_auto(path)
    date_col = raw.columns[0]
    raw[date_col] = parse_mixed_date(raw[date_col])
    raw = raw.loc[raw[date_col].notna()].copy()
    raw = clean_numeric_frame(raw, skip_columns=[date_col])
    raw = raw.rename(columns={date_col: "date"})
    raw = raw.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    prefix = path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    rename_map = {col: f"{prefix}__{col}" for col in raw.columns if col != "date"}
    return raw.rename(columns=rename_map).reset_index(drop=True)


def align_fundamental_to_calendar(calendar: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty or factor_df["date"].nunique() <= 1:
        return calendar[["date"]].copy()
    aligned = pd.merge_asof(
        calendar[["date"]].sort_values("date"),
        factor_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    return aligned
