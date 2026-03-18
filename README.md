# sinopec-07

国内能源化工品种期货价格预测项目仓库。当前仓库包含三条赛道的数据研究骨架、统一的数据读取与特征工程代码，以及面向 `T+5 / T+10 / T+20` 的基线建模脚本。

## 项目范围

- 赛道一：原油链，目标品种 `WTI / Brent / SC`
- 赛道二：聚酯产业链，目标品种 `PTA / PX / PF / PR`
- 赛道三：塑料链，目标品种 `PP / PE`

## 当前实现

- 自动识别中文 CSV 编码并清洗数值字段
- 从交易长表中抽取“主力合约”目标序列
- 将日/周/月/季基本面宽表对齐到交易日
- 生成价格、收益率、波动、均线等时序特征
- 训练全赛道基线模型并导出评估结果

## 目录结构

- `datasets/`: 原始数据集
- `src/sinopec07/`: 核心 Python 包
- `scripts/audit_datasets.py`: 数据审计脚本
- `scripts/run_all_tracks.py`: 全赛道基线训练脚本
- `results/`: 运行输出目录
- `thesis/`: 论文模板与论文工程，本地保留，不同步到远端

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/audit_datasets.py
python scripts/run_all_tracks.py
```

## 建模说明

- 预测目标为各品种主力合约未来 `5 / 10 / 20` 个交易日后的收盘价。
- 目前使用 `SimpleImputer + RandomForestRegressor(n_jobs=1)` 作为统一基线。
- 样本划分采用按时间顺序切分，避免未来信息泄露。
- 赛道一中使用无数字根代码作为主力近似标识，例如 `CL.NYM`、`B.IPE`、`SC.INE`。

## 论文与同步边界

- `thesis/` 目录用于放置论文模板和后续论文工程。
- 论文正文、编译产物和中间文件不上传到 GitHub。
