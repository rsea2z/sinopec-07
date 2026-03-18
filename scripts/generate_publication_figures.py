from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

STRATEGY_CN = {
    "naive_persistence": "朴素持久性",
    "pr_specialist": "PR专项",
    "residual_search": "残差建模",
    "px_meta_ensemble": "PX元集成",
    "competition_ensemble": "竞赛式集成",
    "px_meta": "PX元集成",
}

TRACK_CN = {
    "track1": "赛道一",
    "track2": "赛道二",
    "track3": "赛道三",
}


def configure_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["SimSun", "STSong", "FangSong"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["figure.dpi"] = 160
    mpl.rcParams["savefig.dpi"] = 300


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_strategy_heatmap(board: pd.DataFrame) -> None:
    pivot = board.pivot(index="symbol", columns="horizon", values="test_rmse_gain")
    pivot = pivot.reindex(["WTI", "Brent", "SC", "PTA", "PF", "PX", "PR", "PE", "PP"])
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    im = ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"T+{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("各品种各预测期的最优策略相对 Naive 的 RMSE 改进")

    strategy_map = board.set_index(["symbol", "horizon"])["conservative_recommended_strategy"].to_dict()
    for i, symbol in enumerate(pivot.index):
        for j, horizon in enumerate(pivot.columns):
            value = values[i, j]
            if np.isnan(value):
                continue
            strategy = strategy_map.get((symbol, horizon), "naive_persistence")
            short = {
                "naive_persistence": "朴素",
                "pr_specialist": "PR专项",
                "residual_search": "残差",
                "px_meta_ensemble": "PX元",
                "competition_ensemble": "竞赛",
            }.get(strategy, "其他")
            ax.text(
                j,
                i,
                f"{value:.1f}\n{short}",
                ha="center",
                va="center",
                color="white" if abs(value) > vmax * 0.45 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("RMSE 改进值（正值表示优于 Naive）")
    save_figure(fig, "figure1_strategy_heatmap")


def plot_meaningful_gain_bar(board: pd.DataFrame) -> None:
    df = board.loc[board["meaningful_gain"]].copy()
    df["任务"] = df["symbol"] + " @ T+" + df["horizon"].astype(str)
    df = df.sort_values("test_rmse_gain", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [OKABE_ITO[2] if s == "PX" else OKABE_ITO[5] for s in df["symbol"]]
    ax.barh(df["任务"], df["test_rmse_gain"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("RMSE 改进值")
    ax.set_ylabel("任务")
    ax.set_title("具有实际意义提升的任务")

    for y, value, strategy in zip(df["任务"], df["test_rmse_gain"], df["conservative_recommended_strategy"]):
        ax.text(value + 0.8, y, STRATEGY_CN.get(strategy, strategy), va="center", fontsize=8)

    save_figure(fig, "figure2_meaningful_gains")


def plot_px_comparison() -> None:
    specialist = pd.read_csv(METRIC_DIR / "px_specialist_summary.csv")[["horizon", "test_rmse_gain"]].rename(
        columns={"test_rmse_gain": "专项集成"}
    )
    meta = pd.read_csv(METRIC_DIR / "px_meta_ensemble_summary.csv")[["horizon", "test_rmse_gain"]].rename(
        columns={"test_rmse_gain": "元集成"}
    )
    attention = pd.read_csv(METRIC_DIR / "px_attention_fusion_summary.csv")[["horizon", "test_blended_gain"]].rename(
        columns={"test_blended_gain": "注意力融合"}
    )
    competition = pd.read_csv(METRIC_DIR / "competition_ensemble_summary.csv")
    competition = competition.loc[competition["symbol"] == "PX", ["horizon", "test_rmse_gain"]].rename(
        columns={"test_rmse_gain": "竞赛式集成"}
    )

    df = specialist.merge(meta, on="horizon").merge(attention, on="horizon").merge(competition, on="horizon")
    df = df.sort_values("horizon")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.18
    x = np.arange(len(df))
    columns = ["专项集成", "元集成", "注意力融合", "竞赛式集成"]
    for idx, col in enumerate(columns):
        ax.bar(x + (idx - 1.5) * width, df[col], width=width, label=col, color=OKABE_ITO[idx])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T+{h}" for h in df["horizon"]])
    ax.set_ylabel("RMSE 改进值")
    ax.set_title("PX 不同建模路线的效果比较")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "figure3_px_method_comparison")


def plot_pr_comparison() -> None:
    board = pd.read_csv(METRIC_DIR / "strategy_board.csv")
    comp = pd.read_csv(METRIC_DIR / "competition_ensemble_summary.csv")
    comp = comp.loc[comp["symbol"] == "PR", ["horizon", "test_rmse_gain"]].rename(columns={"test_rmse_gain": "竞赛式集成"})
    residual = pd.read_csv(METRIC_DIR / "residual_search_summary.csv")
    residual = residual.loc[residual["symbol"] == "PR", ["horizon", "test_rmse_gain"]].rename(columns={"test_rmse_gain": "残差建模"})
    adaptive = pd.read_csv(METRIC_DIR / "adaptive_window_summary.csv")
    adaptive = adaptive.loc[adaptive["symbol"] == "PR", ["horizon", "test_rmse_gain"]].rename(columns={"test_rmse_gain": "自适应窗口"})
    specialist = pd.read_csv(METRIC_DIR / "pr_specialist_summary.csv")[["horizon", "test_rmse_gain"]].rename(
        columns={"test_rmse_gain": "PR专项"}
    )
    df = comp.merge(residual, on="horizon").merge(adaptive, on="horizon").merge(specialist, on="horizon")
    df = df.sort_values("horizon")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.18
    x = np.arange(len(df))
    columns = ["竞赛式集成", "残差建模", "自适应窗口", "PR专项"]
    for idx, col in enumerate(columns):
        ax.bar(x + (idx - 1.5) * width, df[col], width=width, label=col, color=OKABE_ITO[idx + 1])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T+{h}" for h in df["horizon"]])
    ax.set_ylabel("RMSE 改进值")
    ax.set_title("PR 不同建模路线的效果比较")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "figure4_pr_method_comparison")


def build_thesis_summary_table(board: pd.DataFrame) -> None:
    out = board[
        [
            "track",
            "symbol",
            "horizon",
            "test_naive_rmse",
            "test_rmse",
            "test_rmse_gain",
            "conservative_recommended_strategy",
        ]
    ].copy()
    out["track"] = out["track"].map(TRACK_CN)
    out["conservative_recommended_strategy"] = out["conservative_recommended_strategy"].map(lambda x: STRATEGY_CN.get(x, x))
    out.columns = ["赛道", "品种", "预测期", "Naive_RMSE", "最优策略_RMSE", "改进值", "保守推荐策略"]
    out.to_csv(METRIC_DIR / "thesis_summary_table.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    configure_style()
    board = pd.read_csv(METRIC_DIR / "strategy_board.csv")
    plot_strategy_heatmap(board)
    plot_meaningful_gain_bar(board)
    plot_px_comparison()
    plot_pr_comparison()
    build_thesis_summary_table(board)
    print("Saved figures to", FIGURE_DIR)


if __name__ == "__main__":
    main()
