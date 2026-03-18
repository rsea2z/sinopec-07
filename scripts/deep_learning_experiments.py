from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import RESULTS_DIR, TRACK_CONFIGS
from sinopec07.deep_learning import run_lstm_experiment, run_mlp_experiment, run_transformer_experiment
from sinopec07.features import build_feature_dataset
from sinopec07.modeling import fit_naive_persistence, time_split


EXPERIMENTS = [
    ("track1", "Brent", 5),
    ("track2", "PTA", 5),
    ("track2", "PTA", 10),
    ("track3", "PE", 5),
    ("track3", "PE", 10),
]


def main() -> None:
    rows: list[dict[str, object]] = []
    pred_dir = RESULTS_DIR / "predictions"
    metric_dir = RESULTS_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    for track_key, symbol, horizon in EXPERIMENTS:
        config = TRACK_CONFIGS[track_key]
        dataset = build_feature_dataset(
            config["trading"][symbol],
            config["main_contract_labels"][symbol],
            config["fundamentals"],
            horizon,
        )
        target_col = f"target_t_plus_{horizon}"
        train_df, test_df = time_split(dataset, target_col=target_col)
        naive = fit_naive_persistence(train_df, test_df, target_col=target_col)
        mlp = run_mlp_experiment(dataset, target_col=target_col, lookback=20, epochs=20)
        lstm = run_lstm_experiment(dataset, target_col=target_col, lookback=20, epochs=20)
        transformer = run_transformer_experiment(dataset, target_col=target_col, lookback=20, epochs=20)

        for model_name, result in [
            ("naive_persistence", naive),
            ("mlp_return", mlp),
            ("lstm_return", lstm),
            ("transformer_return", transformer),
        ]:
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
            result.predictions.to_csv(
                pred_dir / f"deep_{track_key}_{symbol}_h{horizon}_{model_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    out = pd.DataFrame(rows).sort_values(["track", "symbol", "horizon", "model"]).reset_index(drop=True)
    out.to_csv(metric_dir / "deep_learning_comparison.csv", index=False, encoding="utf-8-sig")
    best = (
        out.sort_values(["track", "symbol", "horizon", "rmse", "mae"])
        .groupby(["track", "symbol", "horizon"], as_index=False)
        .first()
    )
    best.to_csv(metric_dir / "deep_learning_best.csv", index=False, encoding="utf-8-sig")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
