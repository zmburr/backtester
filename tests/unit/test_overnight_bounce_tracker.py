"""Pure unit tests for scanners/overnight_bounce_tracker.py helpers (no network)."""
import json

import pandas as pd
import pytest

from scanners import overnight_bounce_tracker as ot

ET = ot.TZ


def _ts(s):
    return pd.Timestamp(s, tz=ET)


class TestChoosePrice:
    def test_manual_wins_when_newer(self):
        manual = (10.65, _ts('2026-10-14 23:00'))
        poly = (12.93, _ts('2026-10-14 20:00'))
        assert ot.choose_price(manual, poly) == (10.65, manual[1], 'M')

    def test_polygon_wins_when_newer(self):
        # 04:05 ET premarket print beats a stale 23:00 overnight entry
        manual = (10.65, _ts('2026-10-14 23:00'))
        poly = (11.20, _ts('2026-10-15 04:05'))
        assert ot.choose_price(manual, poly) == (11.20, poly[1], 'P')

    def test_tie_goes_to_manual(self):
        t = _ts('2026-10-14 23:00')
        assert ot.choose_price((1.0, t), (2.0, t))[2] == 'M'

    def test_single_source_and_none(self):
        t = _ts('2026-10-14 23:00')
        assert ot.choose_price((1.0, t), None) == (1.0, t, 'M')
        assert ot.choose_price(None, (2.0, t)) == (2.0, t, 'P')
        assert ot.choose_price(None, None) is None

    def test_yahoo_overnight_beats_polygon_8pm_print(self):
        poly = (12.93, _ts('2026-10-14 20:00'))
        yahoo = (10.59, _ts('2026-10-14 23:16'))
        assert ot.choose_price(None, poly, yahoo) == (10.59, yahoo[1], 'Y')

    def test_polygon_premarket_beats_stale_yahoo_overnight(self):
        yahoo = (10.59, _ts('2026-10-15 03:58'))
        poly = (10.80, _ts('2026-10-15 04:07'))
        assert ot.choose_price(None, poly, yahoo)[2] == 'P'

    def test_three_way_tie_prefers_manual_then_yahoo(self):
        t = _ts('2026-10-14 23:00')
        assert ot.choose_price((1.0, t), (3.0, t), (2.0, t))[2] == 'M'
        assert ot.choose_price(None, (3.0, t), (2.0, t))[2] == 'Y'


class TestParseYahooOvernight:
    PAYLOAD = {'quoteResponse': {'result': [
        {'symbol': 'SKHY', 'marketState': 'OVERNIGHT', 'overnightMarketPrice': 138.5,
         'overnightMarketTime': 1792034195, 'regularMarketPrice': 147.55},
        {'symbol': 'soxl', 'overnightMarketPrice': 104.71, 'overnightMarketTime': 1792034201},
        {'symbol': 'NOPE', 'regularMarketPrice': 5.0},                 # no overnight print
        {'symbol': 'ZERO', 'overnightMarketPrice': 0, 'overnightMarketTime': 1792034201},
    ], 'error': None}}

    def test_parses_price_and_et_timestamp(self):
        out = ot.parse_yahoo_overnight(self.PAYLOAD)
        assert set(out) == {'SKHY', 'SOXL'}
        price, asof = out['SKHY']
        assert price == 138.5
        assert asof == pd.Timestamp('2026-10-15 03:16:35', tz='UTC').tz_convert(ET)
        assert asof.strftime('%H:%M') == '23:16'

    def test_empty_and_malformed(self):
        assert ot.parse_yahoo_overnight({}) == {}
        assert ot.parse_yahoo_overnight({'quoteResponse': {'result': None}}) == {}


class TestParsePriceFlags:
    def test_parses_and_uppercases(self):
        now = _ts('2026-10-14 23:00')
        out = ot.parse_price_flags(['koru=10.65', 'SOXL=105.5'], now)
        assert out == {'KORU': (10.65, now), 'SOXL': (105.5, now)}

    def test_empty(self):
        assert ot.parse_price_flags(None, _ts('2026-10-14 23:00')) == {}

    def test_bad_flag_exits(self):
        with pytest.raises(SystemExit):
            ot.parse_price_flags(['KORU'], _ts('2026-10-14 23:00'))


class TestLoadPricesFile:
    def test_scalar_and_dict_entries(self, tmp_path):
        p = tmp_path / 'prices.json'
        p.write_text(json.dumps({
            'KORU': 10.65,
            'soxl': {'price': 105.5, 'asof': '2026-10-14 23:05'},
            'BAD': {'price': 'x'},
            'ZERO': 0,
        }))
        out = ot.load_prices_file(p)
        assert set(out) == {'KORU', 'SOXL'}
        assert out['SOXL'] == (105.5, _ts('2026-10-14 23:05'))
        # scalar entry is stamped with the file mtime, which is "now-ish"
        assert out['KORU'][0] == 10.65
        assert abs((ot.now_et() - out['KORU'][1]).total_seconds()) < 120

    def test_missing_file(self, tmp_path):
        assert ot.load_prices_file(tmp_path / 'nope.json') == {}


class TestBounceDay:
    def test_overnight_after_close_is_next_session(self):
        assert ot.bounce_day(_ts('2026-10-14 23:00')) == '2026-10-15'   # Wed night -> Thu

    def test_premarket_is_today(self):
        assert ot.bounce_day(_ts('2026-10-15 05:30')) == '2026-10-15'

    def test_friday_night_rolls_to_monday(self):
        assert ot.bounce_day(_ts('2026-10-16 21:00')) == '2026-10-19'

    def test_saturday_rolls_to_monday(self):
        assert ot.bounce_day(_ts('2026-10-17 12:00')) == '2026-10-19'


class TestTimestampConversion:
    def test_ns_and_ms(self):
        ns = int(pd.Timestamp('2026-10-15 00:00', tz='UTC').value)
        ms = ns // 1_000_000
        assert ot._ns_or_ms_to_et(ns) == pd.Timestamp('2026-10-14 20:00', tz=ET)
        assert ot._ns_or_ms_to_et(ms) == pd.Timestamp('2026-10-14 20:00', tz=ET)
        assert ot._ns_or_ms_to_et(None) is None


class TestFormatNext:
    def test_next_grade(self):
        nxt = {'grade': 'A+', 'price': 9.98, 'pct_away': -0.0629}
        assert ot.format_next(nxt) == 'A+ @ 9.98 (-6.3%)'

    def test_best_grade_reached(self):
        assert ot.format_next(None) == 'best'


class TestRotateStaleLog:
    def test_old_header_is_moved_aside(self, tmp_path):
        p = tmp_path / 'overnight_tracker_log.csv'
        p.write_text('scan_ts_et,bounce_date,ticker\n2026-10-15 08:00:00,2026-10-15,KORU\n')
        moved = ot.rotate_stale_log(p)
        assert moved is not None and moved.exists() and not p.exists()
        assert moved.name.startswith('overnight_tracker_log.') and moved.suffix == '.csv'

    def test_current_header_is_kept(self, tmp_path):
        p = tmp_path / 'overnight_tracker_log.csv'
        p.write_text(','.join(ot.LOG_FIELDS) + '\n')
        assert ot.rotate_stale_log(p) is None and p.exists()

    def test_missing_file(self, tmp_path):
        assert ot.rotate_stale_log(tmp_path / 'nope.csv') is None
