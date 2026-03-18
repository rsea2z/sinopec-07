from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import HORIZONS, RESULTS_DIR, TRACK_CONFIGS
from .features import build_feature_dataset
from .modeling import fit_baseline_regressor, time_split


def ensure_results_dirs() -> None:
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)


def run_single_target(track_key: str, symbol: str, save_dataset: bool = False) -> list[dict[str, object]]:
    config = TRACK_CONFIGS[track_key]
    trading_path = config["trading"][symbol]
    marker = config["main_contract_labels"][symbol]
    fundamentals = config["fundamentals"]

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        dataset = build_feature_dataset(trading_path, marker, fundamentals, horizon)
        target_col = f"target_t_plus_{horizon}"
        train_df, test_df = time_split(dataset, target_col=target_col)
        result = fit_baseline_regressor(train_df, test_df, target_col=target_col)

        if save_dataset:
            dataset_path = RESULTS_DIR / "datasets" / f"{track_key}_{symbol}_h{horizon}.csv"
            dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

        pred_path = RESULTS_DIR / "predictions" / f"{track_key}_{symbol}_h{horizon}.csv"
        result.predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")

        row = {
            "track": track_key,
            "track_label": config["label"],
            "symbol": symbol,
            "horizon": horizon,
            **result.metrics,
        }
        rows.append(row)

    return rows


def run_all_tracks(save_dataset: bool = False) -> pd.DataFrame:
    ensure_results_dirs()
    all_rows: list[dict[str, object]] = []
    for track_key, config in TRACK_CONFIGS.items():
        for symbol in config["trading"]:
            all_rows.extend(run_single_target(track_key, symbol, save_dataset=save_dataset))
    summary = pd.DataFrame(all_rows).sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "metrics" / "baseline_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def audit_track_shapes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for track_key, config in TRACK_CONFIGS.items():
        for symbol, trading_path in config["trading"].items():
            rows.append(
                {
                    "track": track_key,
                    "symbol": symbol,
                    "trading_path": str(trading_path),
                    "fundamental_count": len(config["fundamentals"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics" / "track_inventory.csv", index=False, encoding="utf-8-sig")
    return df

