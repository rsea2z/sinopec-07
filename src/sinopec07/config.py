from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"

HORIZONS = (5, 10, 20)

TRACK_CONFIGS = {
    "track1": {
        "label": "赛道一_原油链",
        "trading": {
            "WTI": DATASETS_DIR / "赛道一" / "交易数据" / "赛道一_WTI交易数据.csv",
            "Brent": DATASETS_DIR / "赛道一" / "交易数据" / "赛道一_Brent交易数据.csv",
            "SC": DATASETS_DIR / "赛道一" / "交易数据" / "赛道一_SC交易数据.csv",
        },
        "main_contract_labels": {
            "WTI": "CL.NYM",
            "Brent": "B.IPE",
            "SC": "SC.INE",
        },
        "fundamentals": [
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_供应.csv",
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_需求.csv",
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_库存.csv",
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_利润.csv",
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_基金持仓.csv",
            DATASETS_DIR / "赛道一" / "基本面数据" / "基本面数据_宏观.csv",
        ],
    },
    "track2": {
        "label": "赛道二_聚酯产业链",
        "trading": {
            "PTA": DATASETS_DIR / "赛道二" / "交易数据" / "赛道二_PTA交易数据.csv",
            "PX": DATASETS_DIR / "赛道二" / "交易数据" / "赛道二_PX交易数据.csv",
            "PF": DATASETS_DIR / "赛道二" / "交易数据" / "赛道二_PF交易数据.csv",
            "PR": DATASETS_DIR / "赛道二" / "交易数据" / "赛道二_PR交易数据.csv",
        },
        "main_contract_labels": {
            "PTA": "主力合约",
            "PX": "主力合约",
            "PF": "主力合约",
            "PR": "主力合约",
        },
        "fundamentals": [
            DATASETS_DIR / "赛道二" / "基本面数据" / "WTI_盘面数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "Brent_盘面数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "INE_盘面数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "石脑油_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "对二甲苯PX_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "精对苯二甲酸 (PTA)_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "聚酯_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "长丝_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "涤纶短纤 (PF)_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "PET瓶片(PR)_数据.csv",
            DATASETS_DIR / "赛道二" / "基本面数据" / "PET切片_数据.csv",
        ],
    },
    "track3": {
        "label": "赛道三_塑料链",
        "trading": {
            "PP": DATASETS_DIR / "赛道三" / "交易数据" / "赛道三_PP交易数据.csv",
            "PE": DATASETS_DIR / "赛道三" / "交易数据" / "赛道三_PE_交易数据.csv",
        },
        "main_contract_labels": {
            "PP": "PP.DCE",
            "PE": "L.DCE",
        },
        "fundamentals": [
            DATASETS_DIR / "赛道三" / "基本面数据" / "赛道三_PP基本面数据.csv",
            DATASETS_DIR / "赛道三" / "基本面数据" / "赛道三_PE基本面数据.csv",
        ],
    },
}

