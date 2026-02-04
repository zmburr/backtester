"""
Bounce Trade Alert Monitor — Live price-level and time-based alerts for bounce (long) trades.

Follows the live_watcher.py architecture (event-driven threading + TTS alerts) but purpose-built
for the bounce playbook derived from 36 historical trades.

All levels are objective — anchored to open price, prior close, and selloff high.
No entry price or personal P&L tracking. Pure market levels.

Usage:
    python scanners/bounce_trader.py NVDA Medium
    python scanners/bounce_trader.py COIN              # auto-detect cap
    python scanners/bounce_trader.py NVDA Medium --dry-run
"""

import sys
import os
import threading
import logging
import pytz
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import gtts
from playsound import playsound

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from scipy.stats import percentileofscore as _pctrank

from data_queries.polygon_queries import get_atr, get_levels_data
from data_queries.trillium_queries import get_actual_current_price_trill
from analyzers.bounce_scorer import BouncePretrade, fetch_bounce_metrics, classify_stock


# ---------------------------------------------------------------------------
# Utilities — inlined from live_watcher.py to avoid transitive import issues
# ---------------------------------------------------------------------------

def play_sounds(text):
    try:
        tts = gtts.gTTS(text)
        tempfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temps_bounce.mp3')
        tts.save(tempfile)
        playsound(tempfile, block=False)
        os.remove(tempfile)
    except Exception:
        print(f'could not play sound: {text}')


def convert_aggs(agg):
    try:
        required_keys = ['close-time', 'open', 'high', 'low', 'close', 'volume', 'vwap']
        for key in required_keys:
            if key not in agg or not isinstance(agg[key], (int, float)):
                return None
        unix_timestamp = int(agg['close-time'] / 1e9)
        gmt_time = datetime.fromtimestamp(unix_timestamp, pytz.timezone('GMT'))
        est_time = gmt_time.astimezone(pytz.timezone('US/Eastern'))
        series_data = {
            'open': agg['open'], 'high': agg['high'], 'low': agg['low'],
            'close': agg['close'], 'volume': agg['volume'], 'vwap': agg['vwap'],
        }
        return pd.Series(series_data, name=est_time)
    except Exception:
        return None


def concatenate_trade_df(trade_df, new_series):
    if new_series.name not in trade_df.index:
        trade_df = pd.concat([trade_df, new_series.to_frame().T])
    return trade_df


# ---------------------------------------------------------------------------
# Inlined from generate_report.py to avoid heavy transitive import chain
# ---------------------------------------------------------------------------

KNOWN_ETFS = {
    'GLD', 'SLV', 'GDXJ', 'QQQ', 'SPY', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY',
    'XLB', 'XLU', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL',
    'TLT', 'HYG', 'LQD', 'EEM', 'EWZ', 'EWJ', 'FXI', 'KWEB', 'SMH', 'XBI', 'IBB', 'ARKK', 'ARKG',
    'IBIT', 'ETHE',
}
_market_cap_cache: Dict[str, str] = {}


def get_ticker_cap(ticker: str) -> str:
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]
    if ticker.upper() in KNOWN_ETFS:
        _market_cap_cache[ticker] = 'ETF'
        return 'ETF'
    try:
        from polygon.rest import RESTClient
        client = RESTClient('pcwUY7TnSF66nYAPIBCApPMyVrXTckJY')
        details = client.get_ticker_details(ticker)
        if hasattr(details, 'type') and details.type in ('ETF', 'ETN'):
            _market_cap_cache[ticker] = 'ETF'
            return 'ETF'
        mc = details.market_cap
        if mc is None:
            cap = 'Medium'
        elif mc >= 100_000_000_000:
            cap = 'Large'
        elif mc >= 2_000_000_000:
            cap = 'Medium'
        elif mc >= 300_000_000:
            cap = 'Small'
        else:
            cap = 'Micro'
        _market_cap_cache[ticker] = cap
        return cap
    except Exception:
        _market_cap_cache[ticker] = 'Medium'
        return 'Medium'


# Bounce intensity spec: (metric, higher_is_better, weight)
_BOUNCE_INTENSITY_SPEC = [
    ('selloff_total_pct',              False, 0.30),
    ('consecutive_down_days',          True,  0.10),
    ('percent_of_vol_on_breakout_day', True,  0.15),
    ('pct_off_30d_high',               False, 0.20),
    ('gap_pct',                        False, 0.25),
]

# Load reference bounce data once
_bounce_csv = Path(__file__).resolve().parent.parent / 'data' / 'bounce_data.csv'
try:
    _bounce_df_all = pd.read_csv(_bounce_csv).dropna(subset=['ticker', 'date'])
except Exception:
    _bounce_df_all = pd.DataFrame()


def compute_bounce_intensity(metrics: Dict, ref_df: pd.DataFrame = None) -> Dict:
    if ref_df is None:
        ref_df = _bounce_df_all
    details = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for col, higher_is_better, weight in _BOUNCE_INTENSITY_SPEC:
        actual = metrics.get(col)
        ref_vals = ref_df[col].dropna().values if col in ref_df.columns else []
        if actual is None or pd.isna(actual) or len(ref_vals) == 0:
            details[col] = {'pctile': None, 'weight': weight, 'actual': actual}
            continue
        raw_pctile = _pctrank(ref_vals, actual, kind='rank')
        pctile = raw_pctile if higher_is_better else 100.0 - raw_pctile
        details[col] = {'pctile': round(pctile, 1), 'weight': weight, 'actual': actual}
        weighted_sum += pctile * weight
        total_weight += weight
    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    return {'composite': composite, 'details': details}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('BounceTrader')

EASTERN = pytz.timezone('US/Eastern')


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlertLevel:
    """One price level to monitor."""
    name: str
    price: float
    direction: str          # 'above' or 'below' open
    category: str           # 'target', 'drawdown', 'setup'
    detail: str             # audio message
    fired: bool = False
    priority: int = 1       # 1 = audio, 2 = log-only


@dataclass
class BounceContext:
    """All pre-computed objective setup data."""
    ticker: str
    open_price: float
    atr: float
    prior_close: float
    prior_high: float
    gap_pct: float
    selloff_pct: float
    selloff_high: Optional[float]
    setup_type: str             # 'GapFade_weakstock' / 'GapFade_strongstock'
    cap: str
    is_etf: bool
    has_exhaustion_gap: bool
    bounce_intensity: float     # 0-100
    bounce_score: int           # 0-5
    recommendation: str         # 'GO', 'CAUTION', 'NO-GO'
    consecutive_down_days: int
    metrics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BounceDataAdapter — thin wrapper around Trillium stream
# ---------------------------------------------------------------------------

class BounceDataAdapter:
    """Subscribes to bar-5s data and feeds it to BounceTradeManager."""

    def __init__(self, ticker, manager):
        self.ticker = ticker
        self.manager = manager
        self.handle = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'BounceDataAdapter starting — ticker: {self.ticker}')

    def start_data_stream(self):
        import journal
        import ctxcapmd

        with ctxcapmd.Session('10.10.1.71', 65500, journal.any_decompress) as session:
            def on_bar(obj):
                if obj['type'] == 'bar-5s':
                    self.manager.process_incoming_data(obj)

            self.handle = session.request_stream(on_bar, self.ticker, ['bar-5s'])
            self.manager.handle = self.handle
            self.handle.wait()
            try:
                self.handle.raise_on_error()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# BounceTradeManager — the core monitor
# ---------------------------------------------------------------------------

class BounceTradeManager:

    def __init__(self, ticker: str, cap: str = None, date: str = None):
        self.ticker = ticker
        self.date = date or datetime.now(EASTERN).strftime('%Y-%m-%d')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.closed = False
        self.handle = None
        self.new_data_event = threading.Event()
        self.current_bar = None
        self.current_time = None

        # ---- 1. Determine cap ----
        self.cap = cap or get_ticker_cap(ticker)
        self.logger.info(f'Cap: {self.cap}')

        # ---- 2. Get open price ----
        try:
            self.open_price = get_actual_current_price_trill(ticker)
        except Exception:
            self.logger.warning('Trillium price failed, falling back to Polygon')
            from data_queries.polygon_queries import get_daily
            daily = get_daily(ticker, self.date)
            self.open_price = daily['open'] if daily else None
        if self.open_price is None:
            raise ValueError(f'Cannot determine open price for {ticker}')
        self.logger.info(f'Open price: ${self.open_price:.2f}')

        # ---- 3. ATR ----
        self.atr = get_atr(ticker, self.date)
        self.logger.info(f'ATR: ${self.atr:.2f}')

        # ---- 4. Levels data ----
        levels = get_levels_data(ticker, self.date, 30, 1, 'day')
        self.prior_close = levels.iloc[-1]['close'] if levels is not None and not levels.empty else self.open_price
        self.prior_high = levels.iloc[-1]['high'] if levels is not None and not levels.empty else self.open_price

        # ---- 5. Gap pct ----
        self.gap_pct = (self.open_price - self.prior_close) / self.prior_close if self.prior_close != 0 else 0

        # ---- 6. Bounce metrics ----
        self.metrics = fetch_bounce_metrics(ticker, self.date)
        self.consecutive_down_days = int(self.metrics.get('consecutive_down_days', 0))

        # ---- 7. Selloff high ----
        self.selloff_high = None
        if levels is not None and not levels.empty and self.consecutive_down_days > 0:
            idx = -(self.consecutive_down_days + 1)
            if abs(idx) <= len(levels):
                self.selloff_high = levels.iloc[idx]['high']
                self.logger.info(f'Selloff high: ${self.selloff_high:.2f} ({self.consecutive_down_days} down days)')

        # ---- 8. Classify ----
        self.setup_type, _class_details = classify_stock(self.metrics)
        self.logger.info(f'Setup type: {self.setup_type}')

        # ---- 9. Pre-trade score ----
        pretrade_result = BouncePretrade().validate(ticker, self.metrics)
        self.bounce_score = pretrade_result.score
        self.recommendation = pretrade_result.recommendation

        # ---- 10. Intensity ----
        intensity_result = compute_bounce_intensity(self.metrics)
        self.bounce_intensity = intensity_result['composite']

        # ---- 11. Exhaustion gap ----
        self.is_etf = self.cap == 'ETF'
        self.has_exhaustion_gap = (self.gap_pct <= -0.05 and self.consecutive_down_days >= 3)

        # ---- 12. Build context ----
        self.ctx = BounceContext(
            ticker=ticker,
            open_price=self.open_price,
            atr=self.atr,
            prior_close=self.prior_close,
            prior_high=self.prior_high,
            gap_pct=self.gap_pct,
            selloff_pct=self.metrics.get('selloff_total_pct', 0),
            selloff_high=self.selloff_high,
            setup_type=self.setup_type,
            cap=self.cap,
            is_etf=self.is_etf,
            has_exhaustion_gap=self.has_exhaustion_gap,
            bounce_intensity=self.bounce_intensity,
            bounce_score=self.bounce_score,
            recommendation=self.recommendation,
            consecutive_down_days=self.consecutive_down_days,
            metrics=self.metrics,
        )

        # ---- 13. Build alert levels ----
        self.target_alerts: List[AlertLevel] = []
        self.drawdown_alerts: List[AlertLevel] = []
        self._build_alert_levels()

        # ---- 14. Time alerts ----
        self.time_alerts_fired: Dict[str, bool] = {}
        self._build_time_alerts()

        # ---- 15. State ----
        self.high_water_mark = self.open_price
        self.low_water_mark = self.open_price
        self.trade_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
        self.trade_df.index = pd.to_datetime(self.trade_df.index)

        # ---- 16. Print level table ----
        self._print_levels()

        # ---- 17. Setup alerts ----
        self._fire_setup_alerts()

    # -----------------------------------------------------------------------
    # Alert level construction
    # -----------------------------------------------------------------------

    def _build_alert_levels(self):
        op = self.open_price
        atr = self.atr
        pc = self.prior_close
        sh = self.selloff_high

        # --- Upside targets ---
        raw_targets = [
            ('+0.5 ATR', op + 0.5 * atr, '0.5 ATR above open'),
            ('+1.0 ATR', op + 1.0 * atr, '1 ATR above open — conservative scale zone'),
            ('+1.5 ATR', op + 1.5 * atr, '1.5 ATR'),
            ('+2.0 ATR', op + 2.0 * atr, '2 ATR — median cluster day territory'),
            ('+2.7 ATR', op + 2.7 * atr, '75th percentile cluster day high'),
        ]

        # Gap fill levels — only if stock gapped down (prior_close > open)
        if pc > op:
            half_gap = op + 0.5 * (pc - op)
            raw_targets.append(('50% Gap Fill', half_gap, '50 percent gap fill — 70 percent of trades reach this'))
            raw_targets.append(('100% Gap Fill', pc, 'Full gap fill — only 48 percent reach this'))

        # Selloff retrace — only if selloff_high known
        if sh is not None and sh > op:
            retrace_50 = op + 0.5 * (sh - op)
            retrace_65 = op + 0.65 * (sh - op)
            raw_targets.append(('50% Selloff Retrace', retrace_50, '50 percent selloff retrace — median target'))
            raw_targets.append(('65% Selloff Retrace', retrace_65, '65 percent retrace — above median'))

        # Sort by price ascending
        raw_targets.sort(key=lambda x: x[1])

        # Merge levels within 0.1% of each other
        merged = []
        for name, price, detail in raw_targets:
            if merged and abs(price - merged[-1][1]) / merged[-1][1] < 0.001:
                # Keep the one with the more informative name
                if 'gap' in name.lower() or 'retrace' in name.lower():
                    merged[-1] = (name, price, detail)
                continue
            merged.append((name, price, detail))

        for name, price, detail in merged:
            self.target_alerts.append(AlertLevel(
                name=name,
                price=price,
                direction='above',
                category='target',
                detail=detail,
                priority=1,
            ))

        # --- Downside drawdown alerts ---
        dd_levels = [
            ('-0.5 ATR', op - 0.5 * atr, 'Half ATR below open — within normal range'),
            ('-1.3 ATR', op - 1.3 * atr, 'Median drawdown before bounce — minus 1.3 ATR from open'),
            ('-2.8 ATR', op - 2.8 * atr, '75th percentile worst drawdown — ASSESS'),
        ]

        for name, price, detail in dd_levels:
            self.drawdown_alerts.append(AlertLevel(
                name=name,
                price=price,
                direction='below',
                category='drawdown',
                detail=detail,
                priority=1,
            ))

    def _build_time_alerts(self):
        self.time_alerts_fired = {
            '10:00': False,
            '14:30': False,
            '15:30': False,
            '15:55': False,
        }

    def _print_levels(self):
        print('\n' + '=' * 72)
        print(f'  BOUNCE MONITOR: {self.ticker} | Open: ${self.open_price:.2f} | ATR: ${self.atr:.2f}')
        print(f'  Cap: {self.cap} | Type: {self.setup_type} | Score: {self.bounce_score}/5 ({self.recommendation})')
        print(f'  Intensity: {self.bounce_intensity:.0f}/100 | Gap: {self.gap_pct * 100:+.1f}%')
        if self.selloff_high:
            print(f'  Selloff High: ${self.selloff_high:.2f} | Down Days: {self.consecutive_down_days}')
        print('=' * 72)

        print('\n  UPSIDE TARGETS:')
        print(f'  {"Level":<22} {"Price":>10} {"vs Open":>10}')
        print('  ' + '-' * 44)
        for a in self.target_alerts:
            pct = (a.price - self.open_price) / self.open_price * 100
            print(f'  {a.name:<22} ${a.price:>9.2f} {pct:>+9.1f}%')

        print('\n  DRAWDOWN LEVELS:')
        print(f'  {"Level":<22} {"Price":>10} {"vs Open":>10}')
        print('  ' + '-' * 44)
        for a in self.drawdown_alerts:
            pct = (a.price - self.open_price) / self.open_price * 100
            print(f'  {a.name:<22} ${a.price:>9.2f} {pct:>+9.1f}%')
        print()

    # -----------------------------------------------------------------------
    # Setup alerts — fire once at initialization
    # -----------------------------------------------------------------------

    def _fire_setup_alerts(self):
        if 'weakstock' in self.setup_type.lower():
            self._alert('WEAKSTOCK bounce — median high plus 24 percent, close plus 12', priority=1)
        else:
            self._alert('STRONGSTOCK bounce — median high plus 7 percent, close plus 2', priority=1)

        if self.is_etf:
            self._alert('ETF — more contained. Plus 7 percent median vs plus 37 for stocks on cluster days', priority=1)

        if self.has_exhaustion_gap:
            self._alert('EXHAUSTION GAP — median high plus 22 percent', priority=1)

        self._alert(f'Bounce intensity: {self.bounce_intensity:.0f} out of 100', priority=2)

        if self.recommendation == 'NO-GO':
            self._alert(f'Pretrade score {self.bounce_score} out of 5 — NO GO', priority=1)

    # -----------------------------------------------------------------------
    # Main event loop
    # -----------------------------------------------------------------------

    def run(self):
        """Event-driven main loop — blocks until self.closed is set."""
        while not self.closed:
            self.new_data_event.wait(timeout=5)

            if self.current_bar is not None:
                series = convert_aggs(self.current_bar)
                if series is not None:
                    self.trade_df = concatenate_trade_df(self.trade_df, series)
                    self.current_time = series.name

                    price = series['close']
                    high = series['high']
                    low = series['low']

                    # Update watermarks
                    if high > self.high_water_mark:
                        self.high_water_mark = high
                    if low < self.low_water_mark:
                        self.low_water_mark = low

                    # Check alerts
                    self._check_target_alerts(high)
                    self._check_drawdown_alerts(low)

            # Time alerts run on every loop regardless of new data
            self._check_time_alerts()

            self.new_data_event.clear()

    def process_incoming_data(self, bar_data):
        """Called by BounceDataAdapter with each 5s bar dict."""
        self.current_bar = bar_data
        self.new_data_event.set()

    # -----------------------------------------------------------------------
    # Price-based alert checks
    # -----------------------------------------------------------------------

    def _check_target_alerts(self, bar_high: float):
        for alert in self.target_alerts:
            if not alert.fired and bar_high >= alert.price:
                alert.fired = True
                self._alert(alert.detail, priority=alert.priority)
                self.logger.info(f'TARGET HIT: {alert.name} @ ${alert.price:.2f}')

    def _check_drawdown_alerts(self, bar_low: float):
        for alert in self.drawdown_alerts:
            if not alert.fired and bar_low <= alert.price:
                alert.fired = True
                self._alert(alert.detail, priority=alert.priority)
                self.logger.info(f'DRAWDOWN: {alert.name} @ ${alert.price:.2f}')

    # -----------------------------------------------------------------------
    # Time-based alerts
    # -----------------------------------------------------------------------

    def _check_time_alerts(self):
        if self.current_time is None:
            return

        now_et = self.current_time
        if not hasattr(now_et, 'hour'):
            return

        t = now_et.time()

        # 10:00 AM — early low check
        if not self.time_alerts_fired['10:00'] and t >= dt_time(10, 0):
            self.time_alerts_fired['10:00'] = True
            if self.low_water_mark < self.open_price:
                self._alert('LOW NOT IN FIRST 30 MIN — quality degrades to 45 percent close green', priority=1)
            else:
                self._alert('EARLY LOW CONFIRMED — 100 percent close green historically', priority=1)

        # 2:30 PM — scale reminder
        if not self.time_alerts_fired['14:30'] and t >= dt_time(14, 30):
            self.time_alerts_fired['14:30'] = True
            current = self.trade_df.iloc[-1]['close'] if not self.trade_df.empty else self.open_price
            msg = (f'SCALE REMINDER — only 51 percent of open-to-high retained at close. '
                   f'Session high was {self.high_water_mark:.2f}, current is {current:.2f}.')
            self._alert(msg, priority=1)

        # 3:30 PM — overnight reminder
        if not self.time_alerts_fired['15:30'] and t >= dt_time(15, 30):
            self.time_alerts_fired['15:30'] = True
            self._alert(
                'OVERNIGHT REMINDER — 100 percent of cluster days gapped up. '
                'Median plus 10.6 percent. Hold a portion.',
                priority=1,
            )

        # 3:55 PM — end-of-day summary
        if not self.time_alerts_fired['15:55'] and t >= dt_time(15, 55):
            self.time_alerts_fired['15:55'] = True
            self._fire_eod_summary()

    def _fire_eod_summary(self):
        current = self.trade_df.iloc[-1]['close'] if not self.trade_df.empty else self.open_price
        atr_from_open = (current - self.open_price) / self.atr if self.atr else 0

        targets_hit = sum(1 for a in self.target_alerts if a.fired)
        targets_total = len(self.target_alerts)

        msg = (
            f'END OF DAY SUMMARY — {self.ticker}. '
            f'Open {self.open_price:.2f}, current {current:.2f}, '
            f'session high {self.high_water_mark:.2f}, session low {self.low_water_mark:.2f}. '
            f'ATR move from open: {atr_from_open:+.1f}. '
            f'Targets hit {targets_hit} of {targets_total}.'
        )
        self._alert(msg, priority=1)

    # -----------------------------------------------------------------------
    # Central alert dispatch
    # -----------------------------------------------------------------------

    def _alert(self, message: str, priority: int = 1):
        self.logger.info(f'ALERT (P{priority}): {message}')
        if priority == 1:
            play_sounds(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def bounce_main(ticker: str, cap: str = None, dry_run: bool = False):
    """Entry point — build manager, optionally attach live stream, and run."""
    manager = BounceTradeManager(ticker=ticker, cap=cap)

    if dry_run:
        logger.info('DRY RUN — simulating a few bars then exiting')
        _simulate_dry_run(manager)
        return

    # Live stream
    adapter = BounceDataAdapter(ticker, manager)
    stream_thread = threading.Thread(target=adapter.start_data_stream, daemon=True)
    stream_thread.start()

    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info('Interrupted — shutting down')
        manager.closed = True
        if adapter.handle:
            adapter.handle.cancel()


def _simulate_dry_run(manager: BounceTradeManager):
    """Feed a few synthetic bars for testing init, levels, and alert logic."""
    import time as _time

    op = manager.open_price
    atr = manager.atr
    base_ts = int(datetime.now().timestamp() * 1e9)

    test_prices = [
        op,                     # at open
        op + 0.3 * atr,        # slight move up
        op + 0.6 * atr,        # should trigger +0.5 ATR
        op - 0.3 * atr,        # dip
        op - 0.6 * atr,        # should trigger -0.5 ATR
        op + 1.1 * atr,        # should trigger +1.0 ATR
    ]

    for i, price in enumerate(test_prices):
        bar = {
            'type': 'bar-5s',
            'close-time': base_ts + i * 5_000_000_000,
            'open': price - 0.01,
            'high': price + 0.02,
            'low': price - 0.02,
            'close': price,
            'volume': 10000,
            'vwap': price,
        }
        manager.process_incoming_data(bar)
        manager.new_data_event.wait(timeout=1)

        series = convert_aggs(bar)
        if series is not None:
            manager.trade_df = concatenate_trade_df(manager.trade_df, series)
            manager.current_time = series.name

            if series['high'] > manager.high_water_mark:
                manager.high_water_mark = series['high']
            if series['low'] < manager.low_water_mark:
                manager.low_water_mark = series['low']

            manager._check_target_alerts(series['high'])
            manager._check_drawdown_alerts(series['low'])

        manager.new_data_event.clear()
        _time.sleep(0.5)

    logger.info('Dry run complete — all levels and setup alerts verified')


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scanners/bounce_trader.py TICKER [CAP] [--dry-run]')
        print('  TICKER: required (e.g., NVDA)')
        print('  CAP:    optional — Large, Medium, Small, Micro, ETF (auto-detect if omitted)')
        print('  --dry-run: skip live stream, simulate test bars')
        sys.exit(1)

    _ticker = sys.argv[1].upper()
    _cap = None
    _dry_run = '--dry-run' in sys.argv

    # Cap is the second arg if it's not a flag
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        _cap = sys.argv[2]

    bounce_main(_ticker, cap=_cap, dry_run=_dry_run)
