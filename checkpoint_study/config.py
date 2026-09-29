"""Paths and frozen study constants for the checkpoint study (see PLAN.md)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# orderPipe checkout that holds the SHARED live code (checkpoint_features,
# print_filter, ref_price_calculator). Override while the shared module is on a
# feature branch / worktree.
ORDERPIPE_ROOT = Path(os.getenv("ORDERPIPE_ROOT", r"C:\Users\zmbur\PycharmProjects\orderPipe"))

# Bar cache + outputs live in the MAIN backtester checkout's data dir, so a
# multi-hour Trillium fetch survives worktree removal.
DATA_DIR = Path(os.getenv("CHECKPOINT_STUDY_DATA",
                          r"C:\Users\zmbur\PycharmProjects\backtester\data\checkpoint_study"))
BARS_DIR = DATA_DIR / "bars5s"
REPORT_DIR = DATA_DIR / "report"

TRADE_LOG = Path(r"C:\Users\zmbur\PycharmProjects\ExitMonitor\data\trade_data.csv")

CHECKPOINTS = (6, 17, 30)
TRAIN_END = "2025-06-30"            # train <= this date; test after (frozen, PLAN.md)
RTH_START, RTH_END = "09:30", "15:30"   # anchor scope (live applies the same)


def load_by_path(rel: str, name: str):
    """Import an orderPipe module by file path under a unique name, so its
    package imports don't collide with backtester modules of the same name."""
    path = ORDERPIPE_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def orderpipe_on_path() -> None:
    """For modules that need orderPipe's package imports (Trillium fetch, ref
    calculator). Appended, not inserted, so backtester modules win on clashes."""
    p = str(ORDERPIPE_ROOT)
    if p not in sys.path:
        sys.path.append(p)
