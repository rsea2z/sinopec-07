from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sinopec07.pipeline import run_all_tracks


def main() -> None:
    summary = run_all_tracks(save_dataset=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

