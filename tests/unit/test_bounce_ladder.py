"""Unit tests for analyzers/bounce_ladder.py and the prior-close setup lock (no network)."""
import pandas as pd
import pytest

from analyzers import bounce_ladder as bl
from scanners.premarket_bounce_scanner import setup_type_at_close


def _static(**over):
    """Weakstock ETF: below its 200-day, prior-day range 1.2x ATR.

    ETF weakstock thresholds -> flip prices off this context:
      gap -4%          -> 100 * 0.96  = 96.00
      3-day -15%       -> 115 * 0.85  = 97.75
      selloff -15%     -> 130 * 0.85  = 110.50  (already passing at the prior close)
      off 30d -30%     -> 150 * 0.70  = 105.00  (already passing)
      off 52wk -34%    -> 200 * 0.66  = 132.00  (already passing)
    """
    s = {'prior_close': 100.0, 'sma200': 120.0, 'mid_bb': 110.0, 'close_30_ago': 140.0,
         'close_15_ago': 125.0, 'close_3_ago': 115.0, 'atr': 6.0, 'prior_day_range': 7.2,
         'high_30d': 150.0, 'high_52wk': 200.0, 'consecutive_down_days': 3,
         'selloff_first_open': 130.0}
    s.update(over)
    s['setup_type'] = setup_type_at_close(s)
    return s


class TestGradeFor:
    @pytest.mark.parametrize('score,intensity,grade', [
        (6, 90, 'A+'), (5, 75, 'A+'), (5, 74.9, 'A'), (5, 60, 'A'), (5, 59.9, 'B'),
        (5, None, 'B'), (4, 90, 'B'), (3, 90, 'C'), (2, 99, '-'), (0, None, '-'),
    ])
    def test_boundaries(self, score, intensity, grade):
        assert bl.grade_for(score, intensity) == grade


class TestSetupTypeAtClose:
    def test_below_200_is_weak(self):
        assert _static()['setup_type'] == 'GapFade_weakstock'

    def test_above_200_is_strong(self):
        assert _static(sma200=90.0)['setup_type'] == 'GapFade_strongstock'

    def test_no_200_history_falls_back_to_30d_change_at_the_close(self):
        # -20% over 30 days at the close -> strong, however deep the gap later goes
        assert _static(sma200=None, close_30_ago=125.0)['setup_type'] == 'GapFade_strongstock'
        assert _static(sma200=None, close_30_ago=140.0)['setup_type'] == 'GapFade_weakstock'


class TestBuildLadder:
    def test_criteria_flip_exactly_at_their_thresholds(self):
        lad = bl.build_ladder(_static(), 'ETF', ticker='TEST')
        flips = {ev: lv['price'] for lv in lad.levels for ev in lv['events']}
        assert flips['Gap down >= 4% passes'] == pytest.approx(96.00, abs=0.011)
        assert flips['3-day momentum crash >= 15% passes'] == pytest.approx(97.75, abs=0.011)

    def test_score_counts_up_through_the_flips(self):
        lad = bl.build_ladder(_static(), 'ETF')
        assert lad.at(100.0)['score'] == 4
        assert lad.at(97.70)['score'] == 5
        assert lad.at(95.99)['score'] == 6

    def test_score_and_intensity_never_drop_as_price_falls(self):
        # No 200-day history and -20% over 30 days at the close: the old live-price
        # classification flipped this to weakstock past -25% and the score fell (SKHY 10/15)
        lad = bl.build_ladder(_static(sma200=None, close_30_ago=125.0), 'Large')
        pts = lad.points  # ordered high price -> low price
        assert pts['score'].is_monotonic_increasing
        assert pts['intensity'].dropna().is_monotonic_increasing
        ranks = pts['grade'].map(bl.GRADE_RANK)
        assert ranks.is_monotonic_increasing

    def test_static_criterion_caps_the_score(self):
        lad = bl.build_ladder(_static(prior_day_range=3.0), 'ETF')   # 0.5x ATR
        assert lad.ceiling['score'] == 5
        assert lad.ceiling['blocked'] == ['prior_day_range_atr']
        assert lad.points['score'].max() == 5

    def test_levels_are_ordered_high_to_low(self):
        lad = bl.build_ladder(_static(), 'ETF')
        prices = [lv['price'] for lv in lad.levels]
        assert prices == sorted(prices, reverse=True)

    def test_next_level_is_the_next_better_grade_below(self):
        lad = bl.build_ladder(_static(), 'ETF')
        here = lad.at(100.0)['grade']
        nxt = lad.next_level(100.0)
        assert nxt is not None and nxt['price'] < 100.0
        assert bl.GRADE_RANK[nxt['grade']] > bl.GRADE_RANK[here]
        assert nxt['pct_away'] == pytest.approx(nxt['price'] / 100.0 - 1)

    def test_next_level_none_at_the_best_grade(self):
        lad = bl.build_ladder(_static(), 'ETF')
        assert lad.next_level(lad.ceiling['floor_price']) is None


class TestOdds:
    ODDS = {
        'base': {'n': 1000, 'p_green': 0.51, 'p_bounce5': 0.10, 'med_return': 0.0},
        'buckets': [
            {'score_min': 3, 'score_max': 3, 'cap': 'ETF', 'n': 78, 'p_green': 0.5, 'p_bounce5': 0.19, 'med_return': 0.0},
            {'score_min': 3, 'score_max': 3, 'cap': None, 'n': 2358, 'p_green': 0.5, 'p_bounce5': 0.45, 'med_return': 0.0},
            {'score_min': 5, 'score_max': 6, 'cap': None, 'n': 467, 'p_green': 0.5, 'p_bounce5': 0.64, 'med_return': 0.0},
        ],
    }

    def test_cap_bucket_first(self):
        assert bl.lookup_bounce_odds(3, 'ETF', self.ODDS)['p_bounce5'] == 0.19

    def test_score_band_when_no_cap_bucket(self):
        o = bl.lookup_bounce_odds(6, 'ETF', self.ODDS)
        assert o['p_bounce5'] == 0.64 and o['bucket'] == 'score 5-6'

    def test_base_rate_when_no_band(self):
        assert bl.lookup_bounce_odds(4, 'ETF', self.ODDS)['bucket'] == 'all screened days'


class TestHistoricalGrades:
    def test_summary_by_grade(self):
        graded = pd.DataFrame([
            {'grade': 'A+', 'open_high': 0.20, 'open_close': 0.15, 'open_low': -0.05},
            {'grade': 'A+', 'open_high': 0.30, 'open_close': -0.01, 'open_low': -0.07},
            {'grade': 'C', 'open_high': 0.05, 'open_close': 0.02, 'open_low': -0.03},
        ])
        s = bl.summarize_grades(graded)
        assert s['A+']['n'] == 2
        assert s['A+']['med_open_high'] == pytest.approx(0.25)
        assert s['A+']['pct_green'] == pytest.approx(0.5)
        assert 'B' not in s

    def test_intraday_capitch_rows_are_excluded(self):
        df = pd.DataFrame([{'ticker': 'AMC', 'date': '1/27/2021', 'Setup': 'IntradayCapitch_strongstock',
                            'cap': 'Medium'}])
        assert bl.grade_trades(df).empty


class TestGameplanReminder:
    def test_command_lists_each_ticker_once(self):
        from scanners.premarket_bounce_scanner import gameplan_reminder_html
        html = gameplan_reminder_html(['KORU', 'SOXL', 'KORU'])
        assert 'python -m scanners.bounce_gameplan KORU SOXL<' in html

    def test_inserted_before_closing_body(self):
        from scanners.premarket_bounce_scanner import add_gameplan_reminder
        out = add_gameplan_reminder('<html><body><p>report</p></body></html>', ['SOXL'])
        assert out.index('bounce_gameplan SOXL') < out.index('</body>')
        assert out.endswith('</body></html>')

    def test_no_bounce_names_leaves_the_email_alone(self):
        from scanners.premarket_bounce_scanner import add_gameplan_reminder
        assert add_gameplan_reminder('<html><body></body></html>', []) == '<html><body></body></html>'
