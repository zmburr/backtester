"""Unified signal ledger.

Every scored signal from the three daily reports (priority_report, screener_report,
generate_report) is appended here as a fact-only row: no prediction-correctness is
stored, because correctness semantics differ by recommendation and are computed by
consumers (dispatcher scorecard, analysis scripts). Forward outcomes (same-day
OHLCV and the derived excursion columns) are filled later by
``scripts/fill_signal_outcomes.py``.

Design constraints:
  * Append-only, header-if-empty, via csv.DictWriter (mirrors the dispatcher's
    append_to_log pattern so the two files stay shape-compatible).
  * Idempotent on (date, session, source, ticker, bucket).
  * ``log_signals`` must NEVER raise into the caller — a ledger failure can never
    be allowed to break report generation, so everything is wrapped in try/except.
"""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve the data dir the same way the report scripts do: repo_root / "data".
# support/signal_ledger.py -> parent = support/, parent.parent = repo root.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEDGER_PATH = _DATA_DIR / "signal_ledger.csv"

# Emission-metric columns: the common subset shared (mostly) across the three
# reports' metrics dicts. reversal carries all of these; bounce carries
# atr_pct/gap_pct/pct_change_3/prior_day_range_atr; breakout carries
# atr_pct/gap_pct/pct_from_9ema/pct_from_50mav/premarket_rvol. Missing keys are
# written blank.
_METRIC_KEYS = [
    "atr_pct",
    "gap_pct",
    "prior_day_rvol",
    "premarket_rvol",
    "pct_change_3",
    "prior_day_range_atr",
    "pct_from_9ema",
    "pct_from_50mav",
]

# Outcome columns, blank at emission, filled by fill_signal_outcomes.py.
_OUTCOME_KEYS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "outcome_pct",
    "max_favorable_pct",
    "atr_move",
    "mfe_atr",
    "outcome_filled_date",
]

LEDGER_COLUMNS = [
    "date",
    "session",
    "source",
    "ticker",
    "bucket",
    "cap",
    "setup_type",
    "recommendation",
    "score",
    "score_str",
    *_METRIC_KEYS,
    *_OUTCOME_KEYS,
]

# Key that defines row identity for idempotency.
_IDENTITY_KEYS = ("date", "session", "source", "ticker", "bucket")


def current_session(now: Optional[datetime.datetime] = None) -> str:
    """Return 'morning' or 'evening' using the same rule as _save_signals_to_json."""
    now = now or datetime.datetime.now()
    return "morning" if now.hour < 12 else "evening"


def _extract_setup_type(entry: Dict) -> str:
    """Pull a setup-type label off the entry's score_result (dict or dataclass).

    Screener entries additionally carry an EQS 'label'; that is folded in by the
    caller path below: if a setup_type already exists it is concatenated, else the
    label stands in for it.
    """
    sr = entry.get("score_result")
    setup = None
    if isinstance(sr, dict):
        setup = sr.get("setup_type") or sr.get("archetype_detail")
    elif sr is not None:
        setup = getattr(sr, "setup_type", None)
    setup = str(setup) if setup else ""

    label = entry.get("label")
    if label:
        label = str(label)
        if setup and label not in setup:
            setup = f"{setup} [{label}]"
        elif not setup:
            setup = label
    return setup


def _extract_score(entry: Dict) -> str:
    """Best-effort numeric score. Prefer score_result.score, fall back to parsing
    score_str ('4/6' -> 4). Returns '' when nothing usable is present."""
    sr = entry.get("score_result")
    val = None
    if isinstance(sr, dict):
        val = sr.get("score")
    elif sr is not None:
        val = getattr(sr, "score", None)
    if val is not None:
        return val
    score_str = entry.get("score_str") or ""
    if "/" in score_str:
        head = score_str.split("/", 1)[0].strip()
        try:
            return int(head)
        except ValueError:
            try:
                return float(head)
            except ValueError:
                return ""
    return ""


def _row_from_entry(entry: Dict, source: str, session: str, date_str: str) -> Dict:
    """Build a full ledger row (facts only, outcome columns blank) from a scored entry."""
    metrics = entry.get("metrics") or {}
    row = {k: "" for k in LEDGER_COLUMNS}
    row.update(
        {
            "date": date_str,
            "session": session,
            "source": source,
            "ticker": entry.get("ticker", ""),
            "bucket": entry.get("bucket", ""),
            "cap": entry.get("cap", ""),
            "setup_type": _extract_setup_type(entry),
            "recommendation": entry.get("rec", ""),
            "score": _extract_score(entry),
            "score_str": entry.get("score_str", ""),
        }
    )
    for k in _METRIC_KEYS:
        v = metrics.get(k)
        if v is not None:
            row[k] = v
    return row


def _load_existing_keys() -> set:
    """Return the set of identity tuples already present in the ledger."""
    keys: set = set()
    if not LEDGER_PATH.exists():
        return keys
    try:
        with open(LEDGER_PATH, newline="") as f:
            for r in csv.DictReader(f):
                keys.add(tuple(r.get(k, "") for k in _IDENTITY_KEYS))
    except Exception as e:  # noqa: BLE001 - never let a read error break logging
        logger.warning(f"signal_ledger: failed to read existing keys: {e}")
    return keys


def log_signals(source: str, session: str, scored: List[Dict]) -> int:
    """Append every scored signal in *scored* to the ledger.

    Facts only — no prediction_correct column. Idempotent on
    (date, session, source, ticker, bucket): rows already present are skipped so
    re-running a report the same session never duplicates.

    ``recommendation`` is free text (GO / CAUTION / NO-GO / VETO are all valid and
    all logged — the whole point of the ledger is to remove the GO/CAUTION
    selection bias).

    Returns the number of rows actually written. NEVER raises: any failure is
    logged and swallowed so report generation can't break on a ledger error.
    """
    try:
        if not scored:
            return 0
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        existing = _load_existing_keys()

        rows: List[Dict] = []
        seen = set(existing)
        for entry in scored:
            try:
                row = _row_from_entry(entry, source, session, date_str)
            except Exception as e:  # noqa: BLE001 - skip a bad entry, keep going
                logger.warning(f"signal_ledger: skipping malformed entry: {e}")
                continue
            key = tuple(row.get(k, "") for k in _IDENTITY_KEYS)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        if not rows:
            return 0

        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
        with open(LEDGER_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})

        logger.info(
            f"signal_ledger: appended {len(rows)} rows from {source}/{session} "
            f"({len(scored) - len(rows)} skipped as duplicates)"
        )
        return len(rows)
    except Exception as e:  # noqa: BLE001 - contract: never propagate
        logger.warning(f"signal_ledger: log_signals failed (non-fatal): {e}")
        return 0
