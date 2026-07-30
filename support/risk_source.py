"""Bridge to ExitMonitor's R-multiple risk model (read-only).

Pragmatic copy of orderPipe/morning_watcher/risk_source.py — same pattern,
same source of truth. The dollar value of 1R and the tier->R ladder are owned
by ExitMonitor and are intentionally dynamic; consumers must read them, never
hardcode them. ExitMonitor's journal/risk_model.py is dependency-free by
design, so it's loaded directly off disk via importlib. Any failure degrades
to None so callers just disable their R features.

NOTE: third copy of this bridge (orderPipe has the canonical one). If a third
consumer ever appears, promote the bridge into ExitMonitor as a public
read-API instead of copying again.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(r"C:\Users\zmbur\PycharmProjects\ExitMonitor\journal\risk_model.py")
RISK_MODEL_PATH = Path(os.environ.get("EXITMONITOR_RISK_MODEL_PATH", str(_DEFAULT_PATH)))

_MODULE = None
_LOAD_ATTEMPTED = False


def _load():
    """Load and cache ExitMonitor's risk_model module. Returns it or None."""
    global _MODULE, _LOAD_ATTEMPTED
    if _MODULE is not None or _LOAD_ATTEMPTED:
        return _MODULE
    _LOAD_ATTEMPTED = True

    if not RISK_MODEL_PATH.exists():
        log.warning(f"ExitMonitor risk model not found: {RISK_MODEL_PATH} — R features disabled")
        return None
    try:
        spec = importlib.util.spec_from_file_location("exitmonitor_risk_model", str(RISK_MODEL_PATH))
        if spec is None or spec.loader is None:
            log.warning(f"could not build import spec for {RISK_MODEL_PATH} — R features disabled")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        log.info(f"loaded ExitMonitor risk model (ONE_R=${getattr(module, 'ONE_R', '?')})")
    except Exception as e:
        log.warning(f"failed to load ExitMonitor risk model: {e} — R features disabled")
        return None
    return _MODULE


def one_r_dollars() -> Optional[float]:
    """Dollar value of 1R from ExitMonitor (e.g. 3000.0). None if unavailable."""
    module = _load()
    if module is None:
        return None
    try:
        val = float(module.ONE_R)
        return val if val > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def tier_multiples() -> Optional[Dict[str, float]]:
    """Tier label -> R multiple from ExitMonitor (A=10, B=4, C=1, D=0.2).
    None if unavailable."""
    module = _load()
    if module is None:
        return None
    try:
        return {str(k): float(v) for k, v in module.TIER_R.items()}
    except (AttributeError, TypeError, ValueError):
        return None
