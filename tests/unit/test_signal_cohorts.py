"""Cohort-isolation tests for the signal scorecard.

The control cohort (below-bar NO-GO signals) shares signal_outcomes.csv with the
surfaced cohort. The whole design rests on one invariant: adding control rows must
NOT perturb any surfaced row's episode or cluster labels, because the Signal
Analysis cron carries first-flag counters forward across cycles and re-labelling
them retroactively would silently invalidate its entire settled-questions ledger.

These tests pin that invariant, plus the backward-compatible default that a row
written before the cohort column existed reads as 'surfaced'.

Every expected value is reasoned out by hand from EPISODE_GAP and the labelling
rules — these are not snapshots.
"""
import csv

import pytest

from scripts.signal_scorecard import (
    COHORT_CONTROL,
    COHORT_SURFACED,
    COLUMNS,
    assign_clusters,
    assign_episodes,
    rolling_summary,
)


def _row(ticker, target, cohort, bucket="bounce", **kw):
    r = {
        "ticker": ticker, "bucket": bucket, "target_date": target,
        "cohort": cohort, "complete": "True", "days_available": "4",
    }
    r.update(kw)
    return r


def _keyed(rows):
    return {(r["target_date"], r["ticker"], r["bucket"], r["cohort"]): r for r in rows}


# --- Episode isolation -------------------------------------------------------


def test_control_row_does_not_chain_into_surfaced_episode():
    """A control flag one day after a surfaced flag on the same ticker must not
    become that episode's signal #2 — otherwise the surfaced row's own numbering
    and the derived first-flag counts shift underneath the analysis."""
    surfaced = _row("AMD", "2026-07-13", COHORT_SURFACED)
    control = _row("AMD", "2026-07-14", COHORT_CONTROL)
    assign_episodes(_keyed([surfaced, control]))

    assert surfaced["episode_signal_num"] == 1
    assert control["episode_signal_num"] == 1  # its own episode, not a reprint
    assert surfaced["episode_id"] != control["episode_id"]


def test_surfaced_episode_id_keeps_historical_form():
    """Surfaced ids must carry no cohort segment, so values already written to
    signal_outcomes.csv stay byte-identical."""
    r = _row("NVDA", "2026-07-13", COHORT_SURFACED)
    assign_episodes(_keyed([r]))
    assert r["episode_id"] == "NVDA_bounce_2026-07-13"


def test_control_episode_id_is_namespaced():
    r = _row("NVDA", "2026-07-13", COHORT_CONTROL)
    assign_episodes(_keyed([r]))
    assert r["episode_id"] == "NVDA_bounce_control_2026-07-13"


def test_reprints_within_a_cohort_still_chain():
    """Cohort isolation must not break normal episode chaining inside a cohort."""
    a = _row("MU", "2026-07-13", COHORT_CONTROL)
    b = _row("MU", "2026-07-14", COHORT_CONTROL)
    assign_episodes(_keyed([a, b]))
    assert (a["episode_signal_num"], b["episode_signal_num"]) == (1, 2)
    assert a["episode_id"] == b["episode_id"]


# --- Cluster isolation -------------------------------------------------------


def test_clusters_are_cohort_scoped():
    """Same bucket, same session, different cohort => different cluster, and
    cluster_size counts only same-cohort first-flags."""
    s = _row("AMD", "2026-07-13", COHORT_SURFACED)
    c = _row("MU", "2026-07-13", COHORT_CONTROL)
    keyed = _keyed([s, c])
    assign_episodes(keyed)
    assign_clusters(keyed)

    assert s["cluster_id"] == "bounce_2026-07-13"           # historical form
    assert c["cluster_id"] == "bounce_control_2026-07-13"
    assert s["cluster_size"] == 1 and c["cluster_size"] == 1


# --- Backward compatibility --------------------------------------------------


def test_missing_cohort_field_is_treated_as_surfaced():
    """Rows written before the cohort column existed carry no 'cohort' key."""
    legacy = {"ticker": "AMD", "bucket": "bounce", "target_date": "2026-07-13"}
    keyed = {("2026-07-13", "AMD", "bounce", ""): legacy}
    assign_episodes(keyed)
    assign_clusters(keyed)
    assert legacy["episode_id"] == "AMD_bounce_2026-07-13"
    assert legacy["cluster_id"] == "bounce_2026-07-13"


def test_cohort_is_a_declared_output_column():
    assert "cohort" in COLUMNS


# --- Summary exclusion -------------------------------------------------------


def test_rolling_summary_excludes_control_rows():
    """Control rows must never reach the summary groups that drive the daily
    email's ALERT_WARN / ALERT_BAD banner."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [
        _row("AMD", today, COHORT_SURFACED, recommendation="GO", tradeable_3d="True",
             days_to_1atr="1"),
        _row("MU", today, COHORT_CONTROL, recommendation="NO-GO", tradeable_3d="False",
             days_to_1atr=""),
        _row("STX", today, COHORT_CONTROL, recommendation="NO-GO", tradeable_3d="False",
             days_to_1atr=""),
    ]
    for r in rows:
        r.setdefault("episode_signal_num", 1)
    summary = rolling_summary(rows)

    assert summary["total"] == 1, "only the surfaced row should be summarised"
    bounce_all = summary["groups"]["Bounce (all)"]["tradeable_3d"]
    assert bounce_all == {"correct": 1, "total": 1, "pct": 100.0}


# --- Live-file invariant -----------------------------------------------------


def test_outcomes_file_header_matches_columns(tmp_path):
    """If signal_outcomes.csv exists, its header must be exactly COLUMNS — a
    drifted header means save_outcomes silently dropped or reordered a field."""
    from scripts.signal_scorecard import OUTCOMES_FILE
    if not OUTCOMES_FILE.exists():
        pytest.skip("no outcomes file on this box")
    with open(OUTCOMES_FILE, newline="") as f:
        header = next(csv.reader(f))
    assert header == COLUMNS
