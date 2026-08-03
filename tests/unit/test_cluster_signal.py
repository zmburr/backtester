"""Threshold and copy tests for the cluster-day signal.

The CLUSTER banner tells the trader to size up. It fired at 7 windows until
2026-08-03, a band that measured -0.50R over 10 sessions while printing
"outsized opportunity day... lean aggressive". These tests pin the corrected
behaviour so the aggressive copy can never drift back onto a no-edge band.

Expected values are reasoned from the constants, not snapshotted.
"""
import pytest

from scripts.priority_report import (
    CLUSTER_BAND_EV,
    CLUSTER_DAY_THRESHOLD,
    CLUSTER_WATCH_THRESHOLD,
    _build_cluster_banner_html,
    compute_cluster_signal,
)

AGGRESSIVE = "lean aggressive"
NO_EDGE = "No edge measured"


def _windows(n, bucket="bounce"):
    return [{"bucket": bucket} for _ in range(n)]


def _banner(n):
    return _build_cluster_banner_html(compute_cluster_signal(_windows(n)))


# --- thresholds --------------------------------------------------------------


def test_thresholds_are_ordered_and_above_the_negative_band():
    """7-14 measured -0.50R, so CLUSTER must sit above 14."""
    assert CLUSTER_WATCH_THRESHOLD < CLUSTER_DAY_THRESHOLD
    assert CLUSTER_DAY_THRESHOLD > 14


@pytest.mark.parametrize("n,expected", [
    (0, None), (2, None),
    (CLUSTER_WATCH_THRESHOLD - 1, None),
    (CLUSTER_WATCH_THRESHOLD, "WATCH"),
    (CLUSTER_DAY_THRESHOLD - 1, "WATCH"),
    (CLUSTER_DAY_THRESHOLD, "CLUSTER"),
    (CLUSTER_DAY_THRESHOLD + 5, "CLUSTER"),
])
def test_level_boundaries(n, expected):
    assert compute_cluster_signal(_windows(n))["level"] == expected


def test_only_bounce_windows_count():
    mixed = _windows(20, bucket="reversal") + _windows(2, bucket="bounce")
    sig = compute_cluster_signal(mixed)
    assert sig["bounce_open"] == 2
    assert sig["level"] is None
    assert sig["total_open"] == 22


# --- banner copy -------------------------------------------------------------


def test_watch_band_never_shows_the_aggressive_playbook():
    """The regression this whole change exists to prevent."""
    for n in range(CLUSTER_WATCH_THRESHOLD, CLUSTER_DAY_THRESHOLD):
        html = _banner(n)
        assert AGGRESSIVE not in html, f"{n} windows rendered the aggressive playbook"
        assert NO_EDGE in html, f"{n} windows missing the no-edge framing"


def test_cluster_band_keeps_the_playbook_and_the_sample_caveat():
    html = _banner(CLUSTER_DAY_THRESHOLD)
    assert AGGRESSIVE in html
    # an R figure must never ship without its session count
    assert str(CLUSTER_BAND_EV["CLUSTER"]["sessions"]) in html
    assert CLUSTER_BAND_EV["CLUSTER"]["ev"] in html


def test_quiet_day_renders_nothing():
    assert _banner(CLUSTER_WATCH_THRESHOLD - 1) == ""


def test_every_band_publishes_its_session_count():
    """Guards the rule that produced the misleading '+1.72R' read: no R value
    is ever displayed without the sessions behind it."""
    for level, ev in CLUSTER_BAND_EV.items():
        assert {"ev", "win", "sessions"} <= set(ev)
        assert isinstance(ev["sessions"], int) and ev["sessions"] > 0
