"""10:00 AM low-of-day check in scanners/bounce_trader.py (pure helper, no stream)."""
import pandas as pd

from scanners.bounce_trader import early_low_message


def _ts(hhmm):
    return pd.Timestamp(f'2026-10-15 {hhmm}', tz='US/Eastern')


def test_open_held_as_the_low():
    assert early_low_message(None).startswith('EARLY LOW IN')


def test_low_at_935_is_early():       # SKHY 2026-10-15
    assert early_low_message(_ts('09:35')) == 'EARLY LOW IN at 9:35 — held 10 plus minutes'


def test_low_at_950_is_early():
    assert early_low_message(_ts('09:50')).startswith('EARLY LOW IN')


def test_low_at_955_is_not_confirmed():   # KORU / SOXL 2026-10-15 flush
    assert early_low_message(_ts('09:55')) == 'LOW NOT CONFIRMED — new low at 9:55, still testing'
