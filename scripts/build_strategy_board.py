from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "metrics"


def load_competition() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "competition_ensemble_summary.csv")
    return pd.DataFrame(
        {
            "track": df["track"],
            "symbol": df["symbol"],
            "horizon": df["horizon"],
            "strategy_name": "competition_ensemble",
            "details": df["selected_model"].astype(str) + "|alpha=" + df["alpha_ml"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def load_residual() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "residual_search_summary.csv")
    return pd.DataFrame(
        {
            "track": df["track"],
            "symbol": df["symbol"],
            "horizon": df["horizon"],
            "strategy_name": "residual_search",
            "details": df["selected_model"].astype(str) + "|alpha=" + df["alpha_ml"].astype(str) + "|topk=" + df["top_k"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def load_adaptive() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "adaptive_window_summary.csv")
    return pd.DataFrame(
        {
            "track": df["track"],
            "symbol": df["symbol"],
            "horizon": df["horizon"],
            "strategy_name": "adaptive_window",
            "details": df["selected_model"].astype(str) + "|window=" + df["window"].astype(str) + "|alpha=" + df["alpha_ml"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def load_px_specialist() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "px_specialist_summary.csv")
    return pd.DataFrame(
        {
            "track": "track2",
            "symbol": "PX",
            "horizon": df["horizon"],
            "strategy_name": "px_specialist",
            "details": df["blend_a"].astype(str) + "+" + df["blend_b"].astype(str) + "|alpha=" + df["alpha_b"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def load_px_meta() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "px_meta_ensemble_summary.csv")
    return pd.DataFrame(
        {
            "track": "track2",
            "symbol": "PX",
            "horizon": df["horizon"],
            "strategy_name": "px_meta_ensemble",
            "details": df["blend_a"].astype(str) + "+" + df["blend_b"].astype(str) + "|alpha=" + df["alpha_b"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def load_pr_specialist() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "pr_specialist_summary.csv")
    return pd.DataFrame(
        {
            "track": "track2",
            "symbol": "PR",
            "horizon": df["horizon"],
            "strategy_name": "pr_specialist",
            "details": df["blend_a"].astype(str) + "+" + df["blend_b"].astype(str) + "|alpha=" + df["alpha_b"].astype(str),
            "test_naive_rmse": df["test_naive_rmse"],
            "test_rmse": df["test_rmse"],
            "test_rmse_gain": df["test_rmse_gain"],
        }
    )


def build_board() -> pd.DataFrame:
    frames = [
        load_competition(),
        load_residual(),
        load_adaptive(),
        load_px_specialist(),
        load_px_meta(),
        load_pr_specialist(),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["track", "symbol", "horizon", "test_rmse_gain"], ascending=[True, True, True, False])
    best = combined.groupby(["track", "symbol", "horizon"], as_index=False).first()
    best["use_special_strategy"] = best["test_rmse_gain"] > 0
    best["meaningful_gain"] = best["test_rmse_gain"] > 1.0
    best["recommended_strategy"] = best.apply(
        lambda row: row["strategy_name"] if row["use_special_strategy"] else "naive_persistence",
        axis=1,
    )
    best["conservative_recommended_strategy"] = best.apply(
        lambda row: row["strategy_name"] if row["meaningful_gain"] else "naive_persistence",
        axis=1,
    )
    return combined, best


def main() -> None:
    combined, best = build_board()
    combined.to_csv(RESULTS_DIR / "strategy_candidates.csv", index=False, encoding="utf-8-sig")
    best.to_csv(RESULTS_DIR / "strategy_board.csv", index=False, encoding="utf-8-sig")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
