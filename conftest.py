"""Pytest configuration and shared fixtures for the backtester test suite.

Lives at the project root so that pytest inserts the root onto ``sys.path`` and the
top-level packages (``support``, ``analyzers``, ``data_queries``, ...) import the same
way they do when the app is run from the root.
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

# --- Import path -----------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DIR = ROOT / "tests" / "golden"


# --- Golden-snapshot helpers -----------------------------------------------------
def _json_default(o):
    """Coerce numpy/pandas scalars to native Python; stringify anything else."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def _normalize(obj, ndigits=6):
    """Make a structure stable for snapshot comparison.

    - rounds floats (kills last-digit jitter from harmless refactors)
    - maps NaN/inf to sentinels (NaN != NaN would otherwise always fail)
    - sorts dict keys so ordering changes don't break the snapshot
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        if math.isinf(obj):
            return "__Inf__" if obj > 0 else "__-Inf__"
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {str(k): _normalize(v, ndigits) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, ndigits) for v in obj]
    return obj


@pytest.fixture
def assert_golden():
    """Characterization-test helper: lock in the CURRENT output of a function.

    Usage::

        def test_score(assert_golden):
            result = scorer.score_setup(...)          # a plain dict
            assert_golden("bounce_score_nvda", result)

    Pass plain JSON-able structures. Convert dataclasses first, e.g.
    ``dataclasses.asdict(result)``.

    First run (or ``REGEN_GOLDEN=1``) writes ``tests/golden/<name>.json``.
    Later runs must match byte-for-byte (after normalization) or the test fails,
    flagging a behavior change for you to review and intentionally re-bless.
    """
    def _assert(name, actual):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path = GOLDEN_DIR / f"{name}.json"
        actual_json = json.loads(json.dumps(actual, default=_json_default))
        actual_norm = _normalize(actual_json)
        if os.environ.get("REGEN_GOLDEN") or not path.exists():
            path.write_text(json.dumps(actual_norm, indent=2))
        expected = _normalize(json.loads(path.read_text()))
        assert actual_norm == expected, (
            f"Golden mismatch for '{name}'. If this change is intentional, "
            f"regenerate with: REGEN_GOLDEN=1 pytest <path/to/test>"
        )
    return _assert
