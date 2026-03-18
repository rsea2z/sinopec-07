from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"
PREDICTION_DIR = ROOT / "results" / "predictions"

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

STRATEGY_SHORT = {
    "naive_persistence": "朴素",
    "pr_specialist": "PR专项",
    "residual_search": "残差",
    "px_meta_ensemble": "PX元",
    "competition_ensemble": "竞赛",
}

TRACK_CN = {
    "track1": "赛道一",
    "track2": "赛道二",
    "track3": "赛道三",
}

CHAIN_CN = {
    "track1": "原油链",
    "track2": "聚酯链",
    "track3": "塑料链",
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
    ax.set_title("各品种各预测期相对 Naive 的 RMSE 改进")

    strategy_map = board.set_index(["symbol", "horizon"])["conservative_recommended_strategy"].to_dict()
    for i, symbol in enumerate(pivot.index):
        for j, horizon in enumerate(pivot.columns):
            value = values[i, j]
            if np.isnan(value):
                continue
            strategy = strategy_map.get((symbol, horizon), "naive_persistence")
            ax.text(
                j,
                i,
                f"{value:.1f}\n{STRATEGY_SHORT.get(strategy, '其他')}",
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
    ax.set_title("具有实质意义提升的任务")

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


def plot_track_horizon_gain(board: pd.DataFrame) -> None:
    agg = (
        board.groupby(["track", "horizon"], as_index=False)["test_rmse_gain"]
        .mean()
        .sort_values(["track", "horizon"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    horizons = sorted(agg["horizon"].unique())
    x = np.arange(3)
    width = 0.22
    for idx, horizon in enumerate(horizons):
        sub = agg.loc[agg["horizon"] == horizon].set_index("track").reindex(["track1", "track2", "track3"])
        ax.bar(
            x + (idx - 1) * width,
            sub["test_rmse_gain"],
            width=width,
            label=f"T+{horizon}",
            color=OKABE_ITO[idx],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["原油链", "聚酯链", "塑料链"])
    ax.set_ylabel("平均 RMSE 改进值")
    ax.set_title("不同赛道与预测期上的平均改进水平")
    ax.legend(frameon=False)
    save_figure(fig, "figure5_track_horizon_gain")


def plot_data_coverage() -> None:
    df = pd.read_csv(METRIC_DIR / "dataset_audit.csv")
    df = df.loc[df["horizon"] == 5, ["track", "symbol", "usable_rows", "feature_count", "avg_missing_ratio"]].copy()
    df = df.sort_values("usable_rows", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax1, ax2 = axes

    ax1.barh(df["symbol"], df["usable_rows"], color=OKABE_ITO[1])
    ax1.set_xlabel("可用样本数")
    ax1.set_ylabel("品种")
    ax1.set_title("各品种可用样本规模")

    color_map = {"track1": OKABE_ITO[0], "track2": OKABE_ITO[2], "track3": OKABE_ITO[5]}
    ax2.scatter(
        df["feature_count"],
        df["avg_missing_ratio"],
        s=df["usable_rows"] / 2,
        c=[color_map[t] for t in df["track"]],
        alpha=0.85,
    )
    for _, row in df.iterrows():
        ax2.text(row["feature_count"] + 1, row["avg_missing_ratio"], row["symbol"], fontsize=8, va="center")
    ax2.set_xlabel("特征数")
    ax2.set_ylabel("平均缺失率")
    ax2.set_title("特征维度与缺失水平")
    save_figure(fig, "figure6_data_coverage")


def plot_prediction_cases() -> None:
    px20 = pd.read_csv(PREDICTION_DIR / "px_meta_ensemble_h20.csv").tail(35).copy()
    pr5 = pd.read_csv(PREDICTION_DIR / "pr_specialist_h5.csv").tail(35).copy()

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=False)
    ax1, ax2 = axes

    ax1.plot(pd.to_datetime(px20["date"]), px20["actual"], label="实际值", color=OKABE_ITO[7], linewidth=1.8)
    ax1.plot(pd.to_datetime(px20["date"]), px20["naive"], label="Naive", color=OKABE_ITO[1], linewidth=1.4)
    ax1.plot(pd.to_datetime(px20["date"]), px20["meta_pred"], label="PX元集成", color=OKABE_ITO[2], linewidth=1.8)
    ax1.set_title("PX @ T+20 测试集预测曲线")
    ax1.set_ylabel("收盘价")
    ax1.legend(frameon=False, ncol=3)

    ax2.plot(pd.to_datetime(pr5["date"]), pr5["actual"], label="实际值", color=OKABE_ITO[7], linewidth=1.8)
    ax2.plot(pd.to_datetime(pr5["date"]), pr5["naive_pred"], label="Naive", color=OKABE_ITO[1], linewidth=1.4)
    ax2.plot(pd.to_datetime(pr5["date"]), pr5["specialist_pred"], label="PR专项", color=OKABE_ITO[5], linewidth=1.8)
    ax2.set_title("PR @ T+5 测试集预测曲线")
    ax2.set_ylabel("收盘价")
    ax2.legend(frameon=False, ncol=3)
    save_figure(fig, "figure7_prediction_cases")


def plot_strategy_distribution(board: pd.DataFrame) -> None:
    counts = board["conservative_recommended_strategy"].value_counts()
    labels = [STRATEGY_CN.get(k, k) for k in counts.index]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = [OKABE_ITO[7], OKABE_ITO[5], OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[0]][: len(labels)]
    ax.bar(labels, counts.values, color=colors)
    ax.set_ylabel("任务数量")
    ax.set_title("保守策略板中的方法分布")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)
    save_figure(fig, "figure8_strategy_distribution")


def plot_workflow_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.24, 0.17, 0.48, "多源数据输入\n交易数据\n+ 基本面数据"),
        (0.225, 0.24, 0.17, 0.48, "对齐与特征工程\n主力抽取\n+ 多频对齐"),
        (0.43, 0.24, 0.17, 0.48, "基线与候选模型\nNaive / RF\n残差 / 深度学习"),
        (0.635, 0.24, 0.17, 0.48, "验证集选择\n窗口搜索\n+ 集成搜索"),
        (0.84, 0.24, 0.14, 0.48, "最终策略板\nPX/PR专项\n其余回退Naive"),
    ]
    colors = [OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5], OKABE_ITO[7]]
    for (x, y, w, h, text), color in zip(boxes, colors):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color + "22",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.6, transform=ax.transAxes)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = 0.50
        ax.annotate(
            "",
            xy=(x2 - 0.01, y),
            xytext=(x1 + 0.01, y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
        )
    ax.set_title("本文研究框架与实验闭环", fontsize=13)
    save_figure(fig, "figure9_workflow_diagram")


def plot_simple_task_ranking(board: pd.DataFrame) -> None:
    df = board.copy()
    df["任务"] = df["symbol"] + " @ T+" + df["horizon"].astype(str)
    df = df.sort_values("test_rmse_gain", ascending=True)

    colors = [OKABE_ITO[5] if v < 0 else OKABE_ITO[2] for v in df["test_rmse_gain"]]
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.barh(df["任务"], df["test_rmse_gain"], color=colors)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("相对 Naive 的 RMSE 改进值")
    ax.set_ylabel("任务")
    ax.set_title("全部任务效果排名图")
    save_figure(fig, "figure10_task_ranking")


def plot_final_vs_naive(board: pd.DataFrame) -> None:
    df = board.copy()
    df["任务"] = df["symbol"] + " @ T+" + df["horizon"].astype(str)
    df = df.sort_values(["track", "symbol", "horizon"]).reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    for i, row in df.iterrows():
        color = OKABE_ITO[2] if row["test_rmse_gain"] > 0 else OKABE_ITO[5]
        ax.plot([x[i], x[i]], [row["test_naive_rmse"], row["test_rmse"]], color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(x[i], row["test_naive_rmse"], color=OKABE_ITO[7], s=24, zorder=3)
        ax.scatter(x[i], row["test_rmse"], color=color, s=28, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["任务"], rotation=55, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Naive 与最终策略的逐任务对比")
    ax.scatter([], [], color=OKABE_ITO[7], label="Naive")
    ax.scatter([], [], color=OKABE_ITO[2], label="优于 Naive")
    ax.scatter([], [], color=OKABE_ITO[5], label="不如 Naive")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, "figure11_final_vs_naive")


def plot_win_loss_summary(board: pd.DataFrame) -> None:
    df = board.copy()
    df["结果"] = np.where(df["test_rmse_gain"] > 0, "优于Naive", "不如Naive")
    summary = (
        df.groupby(["track", "结果"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["track1", "track2", "track3"])
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(summary.index.map(CHAIN_CN), summary["优于Naive"], color=OKABE_ITO[2], label="优于Naive")
    ax.bar(
        summary.index.map(CHAIN_CN),
        summary["不如Naive"],
        bottom=summary["优于Naive"],
        color=OKABE_ITO[5],
        label="不如Naive",
    )
    ax.set_ylabel("任务数")
    ax.set_title("各赛道任务胜负分布")
    ax.legend(frameon=False)
    save_figure(fig, "figure12_win_loss_summary")


def plot_best_task_cards(board: pd.DataFrame) -> None:
    df = board.loc[board["meaningful_gain"]].copy().sort_values("test_rmse_gain", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axis("off")
    card_w = 0.17
    gap = 0.025
    start_x = 0.02
    y = 0.18
    h = 0.64
    for idx, (_, row) in enumerate(df.iterrows()):
        x = start_x + idx * (card_w + gap)
        color = OKABE_ITO[2] if row["symbol"] == "PX" else OKABE_ITO[5]
        rect = FancyBboxPatch(
            (x, y),
            card_w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3,
            edgecolor=color,
            facecolor=color + "22",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + card_w / 2, y + 0.46, f'{row["symbol"]}\nT+{int(row["horizon"])}', ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(x + card_w / 2, y + 0.24, f'增益 {row["test_rmse_gain"]:.2f}', ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.text(
            x + card_w / 2,
            y + 0.10,
            STRATEGY_CN.get(row["conservative_recommended_strategy"], row["conservative_recommended_strategy"]),
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
    ax.set_title("当前最值得保留的五个专项任务", fontsize=13)
    save_figure(fig, "figure13_best_task_cards")


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
    plot_track_horizon_gain(board)
    plot_data_coverage()
    plot_prediction_cases()
    plot_strategy_distribution(board)
    plot_workflow_diagram()
    plot_simple_task_ranking(board)
    plot_final_vs_naive(board)
    plot_win_loss_summary(board)
    plot_best_task_cards(board)
    build_thesis_summary_table(board)
    print("Saved figures to", FIGURE_DIR)


if __name__ == "__main__":
    main()
