from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.config import RESULTS_DIR, TRACK_CONFIGS
from sinopec07.features import build_feature_dataset, build_target_frame


def main() -> None:
    rows: list[dict[str, object]] = []
    for track_key, config in TRACK_CONFIGS.items():
        for symbol, trading_path in config["trading"].items():
            marker = config["main_contract_labels"][symbol]
            target = build_target_frame(trading_path, marker)
            base_row = {
                "track": track_key,
                "symbol": symbol,
                "target_rows": len(target),
                "target_start": target["date"].min().date(),
                "target_end": target["date"].max().date(),
                "fundamental_count": len(config["fundamentals"]),
            }
            for horizon in (5, 10, 20):
                dataset = build_feature_dataset(trading_path, marker, config["fundamentals"], horizon)
                target_col = f"target_t_plus_{horizon}"
                usable = dataset[target_col].notna().sum()
                feature_cols = [col for col in dataset.columns if col not in {"date", target_col}]
                missing_ratio = float(dataset[feature_cols].isna().mean().mean())
                rows.append(
                    {
                        **base_row,
                        "horizon": horizon,
                        "dataset_rows": len(dataset),
                        "usable_rows": int(usable),
                        "feature_count": len(feature_cols),
                        "avg_missing_ratio": missing_ratio,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "metrics" / "dataset_audit.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
