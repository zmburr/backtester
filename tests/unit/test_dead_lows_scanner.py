"""Dead-lows close scanner: study-parity features, alert rule, and quiet main loop."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scanners import dead_lows_scanner as dl


def bars(closes, spread=0.02, volume=1_000_000):
    """Daily bars from a close path; high/low a fixed % around each close."""
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({'open': c, 'high': c * (1 + spread), 'low': c * (1 - spread),
                         'close': c, 'volume': float(volume)})


def flush_path(n_up=36, n_down=3, top=100.0, step=0.06):
    """Grind up to `top`, then `n_down` straight lower closes."""
    up = list(np.linspace(top * 0.8, top, n_up))
    down = [top * (1 - step) ** (i + 1) for i in range(n_down)]
    return up + down


def prior(**over):
    p = {'prior_close': 80.0, 'streak_prior': 3, 'down4_prior': 3, 'max30': 120.0,
         'atr': 4.0, 'adv20': 2e9, 'mid_bb': 100.0}
    p.update(over)
    return p


# ---------------------------------------------------------------- features

def test_prior_features_match_study_definitions():
    b = bars(flush_path(), volume=10_000_000)
    p = dl.prior_features(b)
    c = b['close'].to_numpy()
    assert p['prior_close'] == pytest.approx(c[-1])
    assert p['streak_prior'] == 3
    assert p['down4_prior'] == 3
    assert p['max30'] == pytest.approx(c[-30:].max())
    assert p['adv20'] == pytest.approx((c * 10_000_000)[-20:].mean())
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([b.high - b.low, abs(b.high - prev), abs(b.low - prev)]), axis=0)
    assert p['atr'] == pytest.approx(tr[-14:].mean())
    assert p['mid_bb'] == pytest.approx(c[-20:].mean())


def test_prior_features_needs_history():
    assert dl.prior_features(bars([100.0] * 10)) is None


def test_up_day_resets_streak_but_counts_in_down5():
    path = flush_path(n_down=2) + [flush_path(n_down=2)[-1] * 1.01, flush_path(n_down=2)[-1] * 0.98]
    p = dl.prior_features(bars(path))
    assert p['streak_prior'] == 1          # the up day broke the straight run
    assert p['down4_prior'] == 3           # 3 of the last 4 still down


def test_day_features_red_extends_streak_and_widens_range():
    f = dl.day_features(prior(), last=78.0, high=82.0, low=78.5)
    assert f['red'] and f['streak'] == 4 and f['down5'] == 4
    assert f['low'] == 78.0 and f['clv'] == 0.0
    assert f['dd30'] == pytest.approx(78 / 120 - 1)


# ---------------------------------------------------------------- rule

def sndk_like():
    # SNDK 2026-07-29: 4th red day, 57% off the 30-session high close, CLV 0.14.
    p = prior(prior_close=1096.10, max30=2335.0, atr=193.6, adv20=23e9)
    return dl.day_features(p, last=1015.89, high=1124.80, low=998.19)


def test_sndk_7_29_qualifies_as_speak_tier():
    f = sndk_like()
    assert dl.qualify(f) and dl.tier(f) == 'SPEAK'
    shares, stop = dl.starter(f['last'], f['atr'], 3000.0)
    assert shares == 15 and stop == pytest.approx(1015.89 - 193.6)


@pytest.mark.parametrize('over,last,high,low', [
    ({}, 85.0, 86.0, 79.0),                                  # green on the day
    ({}, 82.0, 90.0, 74.0),                                  # CLV 0.5, not at the lows
    ({'max30': 100.0}, 76.0, 80.0, 75.8),                    # only 24% off the high
    ({'adv20': 200e6}, 76.0, 80.0, 75.8),                    # below the $250M floor
    ({'streak_prior': 2, 'down4_prior': 2}, 76.0, 80.0, 75.8),  # no streak form reaches 4
])
def test_rule_rejects(over, last, high, low):
    assert not dl.qualify(dl.day_features(prior(**over), last=last, high=high, low=low))


def test_rule_accepts_the_base_case():
    assert dl.qualify(dl.day_features(prior(), last=76.0, high=80.0, low=75.8))


def test_four_of_five_qualifies_as_email_tier_when_not_straight():
    p = prior(streak_prior=1, down4_prior=3, adv20=5e9)
    f = dl.day_features(p, last=76.0, high=80.0, low=75.8)
    assert f['streak'] == 2 and f['down5'] == 4
    assert dl.qualify(f) and dl.tier(f) == 'EMAIL'


def test_breadth_counts_any_close_location():
    p = prior(adv20=10e6, streak_prior=2)
    assert dl.breadth_hit(dl.day_features(p, last=76.0, high=80.0, low=70.0))   # CLV 0.6
    assert not dl.breadth_hit(dl.day_features(prior(adv20=5e6, streak_prior=2), 76.0, 80.0, 70.0))


@pytest.mark.parametrize('breadth,streak,cap,expect', [
    (225, 4, 'ETF', 'hold to D5 (breadth 225); exit at mid-BB'),
    (225, 4, 'Medium', 'hold to D5 (breadth 225)'),
    (10, 5, 'Large', 'hold to D5 (5 red days); exit at mid-BB'),
    (10, 4, 'Large', "exit next day's close"),
])
def test_hold_plan(breadth, streak, cap, expect):
    assert dl.hold_plan(breadth, streak, cap) == expect


def test_gap_add_level_is_two_percent_under_the_close():
    assert dl.gap_add_level(100.0) == pytest.approx(98.0)


def test_split_suspect_flags_round_ratios_only():
    assert dl.split_suspect(np.array([100, 101, 50.5, 51]))
    assert not dl.split_suspect(np.array([100, 90, 81, 85]))


# ---------------------------------------------------------------- ranking / alerts

def row(ticker, tier='EMAIL', adv=5e8, breadth=60):
    return {'ticker': ticker, 'tier': tier, 'adv20': adv, 'shares': 10, 'breadth': breadth,
            'cap': 'Medium', 'src': 'T', 'streak': 4, 'down5': 4, 'dd30': -0.4, 'clv': 0.05,
            'ret0': -0.06, 'last': 50.0, 'atr': 5.0, 'stop': 45.0, 'add_below': 49.0,
            'mid_bb': 70.0, 'hold': 'hold to D5 (breadth 60)'}


def test_rank_speak_then_watchlist_then_liquidity():
    rows = [row('BIG', adv=9e9), row('WATCH', adv=3e8), row('LOUD', 'SPEAK', adv=1.5e9)]
    assert [r['ticker'] for r in dl.rank_rows(rows, {'WATCH'})] == ['LOUD', 'WATCH', 'BIG']


def test_spoken_lines_are_capped_on_broad_days():
    lines = dl.spoken_lines([row(f'S{i}', 'SPEAK') for i in range(5)] + [row('E')])
    assert len(lines) == dl.MAX_SPOKEN + 1
    assert lines[-1] == 'Plus 2 more dead lows names in the email.'
    assert dl.spoken_lines([row('E')]) == []


def test_new_alerts_dedupes_but_allows_tier_upgrade():
    state = {'alerted': {'A': 'EMAIL', 'B': 'SPEAK'}}
    new = dl.new_alerts([row('A', 'SPEAK'), row('B', 'SPEAK'), row('C')], state)
    assert [r['ticker'] for r in new] == ['A', 'C']


def test_ledger_label_carries_close_price_and_tier():
    e = dl.ledger_entries([row('SNDK', 'SPEAK')])[0]
    assert e['rec'] == 'SPEAK' and e['bucket'] == 'bounce'
    assert 'px50.00' in e['label'] and 'brd60' in e['label']


# ---------------------------------------------------------------- main loop

class FakeSession:
    def __init__(self, date, persist):
        self.cache, self.watchlist, self.rechecked = {}, set(), {}

    def shortlist(self, loose=None):
        return ['A', 'B']

    def recheck(self, tickers):
        pass


def run_main(monkeypatch, passes_rows, close='2026-10-09 16:00', now='2026-10-09 12:30'):
    monkeypatch.setattr(dl.argparse.ArgumentParser, 'parse_args',
                        lambda self: Mock(once=False, dry=False, asof=None))
    monkeypatch.setattr(dl, '_now_et', lambda: pd.Timestamp(now, tz=dl.TZ))
    monkeypatch.setattr(dl, '_market_close',
                        lambda date: pd.Timestamp(close, tz=dl.TZ) if close else None)
    monkeypatch.setattr(dl, 'Session', FakeSession)
    monkeypatch.setattr(dl, 'sessions_before', lambda d, n: [])
    monkeypatch.setattr(dl, 'live_pass', Mock(side_effect=passes_rows))
    monkeypatch.setattr(dl, 'next_trading_day', lambda d: pd.Timestamp('2026-10-12').date())
    monkeypatch.setattr(dl.time, 'sleep', lambda s: None)
    state = {'date': '2026-10-09', 'alerted': {}}
    monkeypatch.setattr(dl, 'load_state', lambda date: state)
    monkeypatch.setattr(dl, 'save_state', Mock())
    mail, voice, ledger = Mock(), Mock(), Mock(return_value=0)
    monkeypatch.setattr(dl, 'send_email', mail)
    monkeypatch.setattr(dl, 'speak', voice)
    monkeypatch.setattr(dl, 'log_signals', ledger)
    dl.main()
    return mail, voice, ledger, state, dl.live_pass


def test_main_alerts_each_name_once_and_logs_final_candidates(monkeypatch):
    p1 = (60, [row('A', 'SPEAK', adv=2e9)])
    p2 = (61, [row('A', 'SPEAK', adv=2e9), row('B')])
    mail, voice, ledger, state, _ = run_main(monkeypatch, [p1, p2])
    assert mail.call_count == 2
    assert 'A' in mail.call_args_list[0].args[1]
    assert 'B' in mail.call_args_list[1].args[1] and 'A' not in mail.call_args_list[1].args[1]
    voice.assert_called_once()                    # A spoken once, B is email-tier
    assert state['alerted'] == {'A': 'SPEAK', 'B': 'EMAIL'}
    src, session, entries = ledger.call_args.args
    assert (src, session) == ('dead_lows', 'close')
    assert ledger.call_args.kwargs['date_str'] == '2026-10-12'
    assert [e['ticker'] for e in entries] == ['A', 'B']


def test_main_is_silent_when_nothing_qualifies(monkeypatch):
    mail, voice, ledger, _, _ = run_main(monkeypatch, [(30, []), (31, [])])
    mail.assert_not_called()
    voice.assert_not_called()
    ledger.assert_not_called()


def test_main_exits_on_holidays_and_after_the_last_pass(monkeypatch):
    mail, _, ledger, _, passes = run_main(monkeypatch, [], close=None)
    passes.assert_not_called()
    mail.assert_not_called()
    ledger.assert_not_called()
    _, _, _, _, passes = run_main(monkeypatch, [], now='2026-10-09 15:58')
    passes.assert_not_called()
