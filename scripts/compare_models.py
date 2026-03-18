from __future__ import annotations

import sys
from pathlib import Path

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
    time_split,
)


MODEL_REGISTRY = {
    "naive_persistence": fit_naive_persistence,
    "ridge_direct": fit_ridge_regressor,
    "random_forest_direct": fit_random_forest_direct,
    "random_forest_return": fit_random_forest_return,
}


def main() -> None:
    rows: list[dict[str, object]] = []
    for track_key, config in TRACK_CONFIGS.items():
        for symbol, trading_path in config["trading"].items():
            marker = config["main_contract_labels"][symbol]
            for horizon in HORIZONS:
                dataset = build_feature_dataset(trading_path, marker, config["fundamentals"], horizon)
                target_col = f"target_t_plus_{horizon}"
                train_df, test_df = time_split(dataset, target_col=target_col)
                for model_name, model_fn in MODEL_REGISTRY.items():
                    result = model_fn(train_df, test_df, target_col=target_col)
                    rows.append(
                        {
                            "track": track_key,
                            "track_label": config["label"],
                            "symbol": symbol,
                            "horizon": horizon,
                            "model": model_name,
                            **result.metrics,
                        }
                    )

    out = pd.DataFrame(rows).sort_values(["track", "symbol", "horizon", "model"]).reset_index(drop=True)
    best = (
        out.sort_values(["track", "symbol", "horizon", "rmse", "mae"])
        .groupby(["track", "symbol", "horizon"], as_index=False)
        .first()
    )

    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "metrics" / "model_comparison.csv", index=False, encoding="utf-8-sig")
    best.to_csv(RESULTS_DIR / "metrics" / "best_models.csv", index=False, encoding="utf-8-sig")

    print("=== Best Models By Track / Symbol / Horizon ===")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
