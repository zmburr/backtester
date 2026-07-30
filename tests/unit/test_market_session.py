"""Pure unit tests for support/market_session.py constants.

Verifies the session-time constants are 'HH:MM:SS' strings and that the
chronological ordering holds via plain string comparison (lexical order of
zero-padded 24h times equals chronological order).
"""
import re

from support import market_session as ms


HHMMSS = re.compile(r'^\d{2}:\d{2}:\d{2}$')

ALL_CONSTANTS = [
    'PREMARKET_START', 'PREMARKET_END', 'MARKET_OPEN', 'MARKET_CLOSE',
    'AFTERHOURS_END', 'NEAR_CLOSE', 'MARKET_OPEN_PLUS_2MIN',
    'MARKET_OPEN_PLUS_5MIN', 'MARKET_OPEN_PLUS_10MIN', 'MARKET_OPEN_PLUS_15MIN',
    'MARKET_OPEN_PLUS_20MIN', 'MARKET_OPEN_PLUS_25MIN', 'MARKET_OPEN_PLUS_30MIN',
]


class TestConstantFormat:
    """Every session constant must be an 'HH:MM:SS' string."""

    def test_all_constants_are_hhmmss_strings(self):
        """Each constant is a string matching the HH:MM:SS pattern."""
        for name in ALL_CONSTANTS:
            value = getattr(ms, name)
            assert isinstance(value, str), f"{name} is not a str"
            assert HHMMSS.match(value), f"{name}={value!r} not HH:MM:SS"


class TestOrdering:
    """Chronological ordering holds via lexical string comparison."""

    def test_primary_ordering(self):
        """PREMARKET_START < MARKET_OPEN < MARKET_CLOSE < AFTERHOURS_END."""
        assert ms.PREMARKET_START < ms.MARKET_OPEN < ms.MARKET_CLOSE < ms.AFTERHOURS_END

    def test_premarket_end_before_open(self):
        """PREMARKET_END (09:29:59) precedes MARKET_OPEN (09:30:00)."""
        assert ms.PREMARKET_END < ms.MARKET_OPEN

    def test_near_close_before_market_close(self):
        """NEAR_CLOSE (15:59:00) precedes MARKET_CLOSE (16:00:00)."""
        assert ms.NEAR_CLOSE < ms.MARKET_CLOSE

    def test_open_plus_minutes_ascending(self):
        """The MARKET_OPEN_PLUS_* constants are strictly increasing."""
        seq = [
            ms.MARKET_OPEN,
            ms.MARKET_OPEN_PLUS_2MIN,
            ms.MARKET_OPEN_PLUS_5MIN,
            ms.MARKET_OPEN_PLUS_10MIN,
            ms.MARKET_OPEN_PLUS_15MIN,
            ms.MARKET_OPEN_PLUS_20MIN,
            ms.MARKET_OPEN_PLUS_25MIN,
            ms.MARKET_OPEN_PLUS_30MIN,
        ]
        assert seq == sorted(seq)
        assert len(set(seq)) == len(seq)  # strictly increasing (no dupes)

    def test_known_literal_values(self):
        """Spot-check the load-bearing literals against the spec."""
        assert ms.MARKET_OPEN == '09:30:00'
        assert ms.MARKET_CLOSE == '16:00:00'
        assert ms.PREMARKET_START == '06:00:00'
        assert ms.AFTERHOURS_END == '20:00:00'
