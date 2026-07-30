"""Smoke tests verifying the test harness itself works.

If these fail, the import path or the golden fixture is misconfigured and no other
test in the suite can be trusted.
"""


def test_project_root_importable():
    """Top-level packages import the same way the app imports them."""
    from support import date_utils

    assert hasattr(date_utils, "csv_date_to_iso")


def test_assert_golden_roundtrip(assert_golden):
    """The shared golden fixture creates-then-matches a snapshot."""
    assert_golden(
        "_smoke_roundtrip",
        {"score": 6, "ratio": 0.333333333, "grade": "A", "items": [1, 2, 3]},
    )
