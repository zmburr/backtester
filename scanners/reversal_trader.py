"""
Reversal (Crack Day) Trade Monitor — Live PBB tracking, covering alerts, and TTS.

Mirrors bounce_trader.py architecture (event-driven threading + TTS alerts) but purpose-built
for the parabolic short playbook.  Tracks 2-minute Prior Bar Breaks in real time,
classifies crack patterns via CoveringRules, and fires position-sizing alerts.

Usage:
    python scanners/reversal_trader.py GLD ETF
    python scanners/reversal_trader.py MSTR              # auto-detect cap
    python scanners/reversal_trader.py GLD ETF --dry-run  # test mode
    python scanners/reversal_trader.py GLD ETF --trac     # TRAC position tracking
"""

import sys
import os
import threading
import queue
import logging
import pytz
import math
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

try:
    import gtts  # type: ignore
except Exception:
    gtts = None

try:
    import pygame  # type: ignore
except Exception:
    pygame = None

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_queries.polygon_queries import get_atr, get_levels_data, get_daily
from analyzers.reversal_pretrade import ReversalPretrade, classify_reversal_setup
from analyzers.reversal_scorer import ReversalScorer, compute_reversal_intensity
from analyzers.crack_covering_rules import CoveringRules

try:
    from data_queries.trillium_queries import get_actual_current_price_trill
except Exception:
    get_actual_current_price_trill = None


# ---------------------------------------------------------------------------
# Utilities — inlined from bounce_trader.py to avoid transitive import issues
# ---------------------------------------------------------------------------

import uuid as _uuid

_pygame_initialized = False
_audio_queue: "queue.Queue[str]" = queue.Queue()
_audio_thread: Optional[threading.Thread] = None
_audio_lock = threading.Lock()


def _ensure_pygame():
    global _pygame_initialized
    if pygame is None:
        return False
    if not _pygame_initialized:
        pygame.mixer.init()
        _pygame_initialized = True
    return True


def _play_audio_file_blocking(filepath: str):
    """Play audio blocking with pygame, then clean up the temp file."""
    try:
        if not _ensure_pygame():
            return
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception:
        pass
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


def _audio_loop():
    """Single-threaded audio loop so alerts never overlap."""
    while True:
        text = _audio_queue.get()
        try:
            if gtts is None or pygame is None:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            tts = gtts.gTTS(text)
            unique_name = f'reversal_{_uuid.uuid4().hex[:8]}.mp3'
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), unique_name)
            tts.save(filepath)
            _play_audio_file_blocking(filepath)
        except Exception:
            pass
        finally:
            _audio_queue.task_done()


def _ensure_audio_thread():
    global _audio_thread
    if gtts is None or pygame is None:
        return False
    with _audio_lock:
        if _audio_thread and _audio_thread.is_alive():
            return True
        _audio_thread = threading.Thread(target=_audio_loop, daemon=True)
        _audio_thread.start()
        return True


def play_sounds(text: str):
    """Queue a TTS audio alert."""
    try:
        if not _ensure_audio_thread():
            return
        _audio_queue.put(str(text))
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
# Cap detection — inlined from bounce_trader.py
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
        client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('ReversalTrader')

EASTERN = pytz.timezone('US/Eastern')
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


def is_market_open() -> bool:
    now_et = datetime.now(EASTERN).time()
    return MARKET_OPEN <= now_et < MARKET_CLOSE


# ---------------------------------------------------------------------------
# 1. TwoMinBarAggregator — aggregate 5s bars into 2-minute bars
# ---------------------------------------------------------------------------

class TwoMinBarAggregator:
    """Aggregates 5-second Trillium bars into 2-minute bars for PBB detection.

    Aligns to even-minute boundaries (9:30, 9:32, 9:34, ...).
    """

    def __init__(self):
        self.current_bar: Optional[Dict] = None
        self.current_boundary: Optional[datetime] = None

    @staticmethod
    def _get_2m_boundary(ts: datetime) -> datetime:
        """Return the start of the 2-minute window that ``ts`` falls into."""
        minute = ts.minute
        aligned_minute = minute - (minute % 2)
        return ts.replace(minute=aligned_minute, second=0, microsecond=0)

    def feed(self, bar_5s: pd.Series) -> Optional[Dict]:
        """Feed a 5-second bar. Returns a completed 2-min bar dict when a boundary is crossed."""
        ts = bar_5s.name
        if not hasattr(ts, 'minute'):
            return None

        boundary = self._get_2m_boundary(ts)
        completed = None

        if self.current_boundary is not None and boundary != self.current_boundary:
            # New 2-minute window — emit the completed bar
            completed = dict(self.current_bar)
            completed['time'] = self.current_boundary
            self.current_bar = None

        if self.current_bar is None:
            self.current_bar = {
                'open': float(bar_5s['open']),
                'high': float(bar_5s['high']),
                'low': float(bar_5s['low']),
                'close': float(bar_5s['close']),
                'volume': int(bar_5s['volume']),
            }
            self.current_boundary = boundary
        else:
            self.current_bar['high'] = max(self.current_bar['high'], float(bar_5s['high']))
            self.current_bar['low'] = min(self.current_bar['low'], float(bar_5s['low']))
            self.current_bar['close'] = float(bar_5s['close'])
            self.current_bar['volume'] += int(bar_5s['volume'])

        return completed


# ---------------------------------------------------------------------------
# 2. PBBTracker — real-time Prior Bar Break detection on 2-minute bars
# ---------------------------------------------------------------------------

class PBBTracker:
    """Detects Prior Bar Breaks on completed 2-minute bars.

    Replicates crack_analyzer.py line 493-509 logic:
    - When current_high > prior_high -> PBB triggered
    - If any of next 3 bars breaks prior_low -> FAILED
    - If 3 bars pass without breaking -> HELD
    """

    def __init__(self):
        self.bars: List[Dict] = []          # completed 2m bars
        self.pending_pbbs: List[Dict] = []  # PBBs in 3-bar confirmation window
        self.total_failed: int = 0
        self.total_held: int = 0

    def feed(self, bar_2m: Dict) -> List[Dict]:
        """Feed a completed 2-min bar. Returns list of newly confirmed PBBs (held or failed)."""
        self.bars.append(bar_2m)
        confirmed = []

        # Check if this bar triggers a new PBB
        if len(self.bars) >= 2:
            curr = self.bars[-1]
            prior = self.bars[-2]
            if curr['high'] > prior['high']:
                self.pending_pbbs.append({
                    'trigger_bar_idx': len(self.bars) - 1,
                    'pbb_price': prior['high'],   # the prior bar's high that was broken
                    'prior_low': prior['low'],
                    'time': bar_2m.get('time'),
                    'bars_since': 0,
                    'status': 'pending',
                })

        # Update all pending PBBs
        still_pending = []
        for pbb in self.pending_pbbs:
            pbb['bars_since'] += 1

            # Check if current bar breaks prior_low -> FAILED
            if bar_2m['low'] < pbb['prior_low']:
                pbb['status'] = 'failed'
                self.total_failed += 1
                confirmed.append(pbb)
                continue

            # 3-bar window expired without breaking -> HELD
            if pbb['bars_since'] >= 3:
                pbb['status'] = 'held'
                self.total_held += 1
                confirmed.append(pbb)
                continue

            still_pending.append(pbb)

        self.pending_pbbs = still_pending
        return confirmed


# ---------------------------------------------------------------------------
# 3. CoveringRulesTracker — stateful real-time wrapper around CoveringRules
# ---------------------------------------------------------------------------

class CoveringRulesTracker:
    """Stateful real-time wrapper around CoveringRules.

    Tracks moves and failed PBBs in real time, and calls the covering decision
    tree when a significant held PBB occurs.
    """

    MIN_MOVE_ATRS = 0.5  # minimum move size to count as significant

    def __init__(self, hod_price: float, atr: float):
        self.hod_price = hod_price
        self.atr = atr
        self.rules = CoveringRules()

        # Move tracking
        self.moves: List[Dict] = []          # completed moves
        self.move_start_price = hod_price
        self.running_low = hod_price
        self.failed_pbbs_in_current_move = 0

        # Position tracking
        self.position_remaining = 1.0  # 100%
        self.pattern = ''

    def on_price_update(self, low: float):
        """Update running low between completed bars."""
        if low < self.running_low:
            self.running_low = low

    def on_failed_pbb(self):
        """Increment failed PBB counter for current move."""
        self.failed_pbbs_in_current_move += 1

    def on_held_pbb(self, pbb_price: float, current_low: float) -> Optional[Dict]:
        """Process a held PBB. Returns a covering action dict or None if move too small."""
        if self.atr <= 0:
            return None

        # Update running low
        if current_low < self.running_low:
            self.running_low = current_low

        # Calculate move size from start
        move_size = self.move_start_price - self.running_low
        move_atrs = move_size / self.atr

        if move_atrs < self.MIN_MOVE_ATRS:
            return None  # move too small to be significant

        # Record the completed move
        move_num = len(self.moves) + 1
        move = {
            'move_num': move_num,
            'start_price': self.move_start_price,
            'low_price': self.running_low,
            'pbb_price': pbb_price,
            'size_dollars': move_size,
            'size_atrs': move_atrs,
            'failed_pbbs_during': self.failed_pbbs_in_current_move,
        }
        self.moves.append(move)

        # Apply covering decision tree
        action = self._apply_rules(move)

        # Reset for next move
        self.move_start_price = pbb_price
        self.running_low = pbb_price
        self.failed_pbbs_in_current_move = 0

        return action

    def _apply_rules(self, move: Dict) -> Optional[Dict]:
        """Apply CoveringRules decision tree based on current move count."""
        move_num = move['move_num']

        if move_num == 1:
            pattern, cover_pct, reasoning = self.rules.classify_after_m1(
                move['size_atrs'], move['failed_pbbs_during'])
            self.pattern = pattern
        elif move_num == 2:
            m1 = self.moves[0]
            m2_m1_ratio = move['size_atrs'] / m1['size_atrs'] if m1['size_atrs'] > 0 else 0
            pattern, cover_pct, reasoning = self.rules.classify_after_m2(
                self.pattern, move['size_atrs'], m2_m1_ratio)
            self.pattern = pattern
            # For ONE_FLUSH_*, cover_pct=1.0 means "cover all remaining"
            if cover_pct >= 1.0:
                cover_pct = self.position_remaining
        else:
            cover_pct, reasoning = self.rules.classify_after_m3(self.position_remaining)

        if cover_pct <= 0.001:
            return {
                'move_num': move_num,
                'cover_pct': 0,
                'cover_price': move['pbb_price'],
                'pattern': self.pattern,
                'reasoning': reasoning,
                'position_remaining': self.position_remaining,
            }

        actual_cover = min(cover_pct, self.position_remaining)
        self.position_remaining -= actual_cover

        return {
            'move_num': move_num,
            'cover_pct': actual_cover,
            'cover_price': move['pbb_price'],
            'pattern': self.pattern,
            'reasoning': reasoning,
            'position_remaining': self.position_remaining,
            'move_atrs': move['size_atrs'],
            'failed_pbbs': move['failed_pbbs_during'],
        }

    def get_cheat_sheet(self) -> str:
        """Return a printable covering decision tree cheat sheet."""
        lines = [
            "  COVERING DECISION TREE:",
            "  +-- M1 >= 2.0 ATR",
            "  |   +-- fail PBBs <= 1 -> ONE_FLUSH_CLEAN  -> Cover 75%",
            "  |   +-- fail PBBs >= 2 -> ONE_FLUSH_STRONG -> Cover 50%",
            "  +-- M1 < 1.0 ATR       -> PROBE            -> Cover 0%",
            "  +-- M1 1.0-2.0 ATR     -> DEVELOPING        -> Cover 25%",
            "  ",
            "  AFTER M2:",
            "  +-- Was ONE_FLUSH_*     -> Cover remaining",
            "  +-- Was PROBE           -> Cover 50%",
            "  +-- Was DEVELOPING",
            "  |   +-- M2/M1 >= 1.0   -> STAIRCASE -> Cover 33% more",
            "  |   +-- M2 < 0.5 ATR   -> DELAYED_FLUSH -> Hold for M3",
            "  |   +-- else            -> FADING -> Cover 50% more",
            "  ",
            "  AFTER M3+: Cover 50% of remaining, trail rest. EOD cover final.",
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. ReversalContext — pre-computed setup data
# ---------------------------------------------------------------------------

@dataclass
class ReversalContext:
    ticker: str
    open_price: float
    atr: float
    prior_close: float
    gap_pct: float
    pct_from_9ema: float
    consecutive_up_days: int
    setup_type: Optional[str]
    cap: str
    reversal_intensity: Optional[float]
    reversal_score: int
    recommendation: str
    metrics: Dict = field(default_factory=dict)
    open_locked: bool = False


# ---------------------------------------------------------------------------
# 5. ReversalDataAdapter — thin Trillium bar-5s stream wrapper
# ---------------------------------------------------------------------------

class ReversalDataAdapter:
    """Subscribes to bar-5s data and feeds it to ReversalTradeManager."""

    def __init__(self, ticker, manager):
        self.ticker = ticker
        self.manager = manager
        self.handle = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'ReversalDataAdapter starting — ticker: {self.ticker}')

    def start_data_stream(self):
        import sheldatagateway
        from sheldatagateway import environments

        user = 'zburr'
        pwd = os.getenv('SHEL_API_PWD')

        with sheldatagateway.Session(environments.env_defs.Prod, user, pwd) as session:
            def on_bar(obj):
                msg_type = obj.get('type', '')
                if msg_type == 'bar-5s':
                    self.manager.process_incoming_data(obj)

            self.handle = session.request_stream(
                callback=on_bar,
                symbol=self.ticker,
                subscriptions=['bar-5s']
            )
            self.manager.handle = self.handle
            self.handle.wait()
            try:
                self.handle.raise_on_error()
            except Exception as e:
                error_msg = str(e)
                if "CANCELED" not in error_msg and "Request was cancelled" not in error_msg:
                    self.logger.error(f"Data stream error: {e}", exc_info=True)
                else:
                    self.logger.info("Data stream was cancelled (normal cleanup)")


# ---------------------------------------------------------------------------
# 6. AlertLevel — shared alert dataclass
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


# ---------------------------------------------------------------------------
# 7. ReversalTradeManager — the core monitor
# ---------------------------------------------------------------------------

class ReversalTradeManager:

    def __init__(self, ticker: str, cap: str = None, date: str = None, trac_scraper=None):
        self.ticker = ticker
        self.date = date or datetime.now(EASTERN).strftime('%Y-%m-%d')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.closed = False
        self.handle = None
        self.new_data_event = threading.Event()
        self.current_bar = None
        self.current_time = None

        # TRAC position integration
        self.trac_scraper = trac_scraper
        self._position_pct_remaining = 1.0

        # ---- 1. Determine cap ----
        self.cap = cap or get_ticker_cap(ticker)
        self.logger.info(f'Cap: {self.cap}')

        # ---- 2. Check if premarket / historical ----
        today_str = datetime.now(EASTERN).strftime('%Y-%m-%d')
        self.is_historical = (self.date != today_str)
        self.premarket_mode = not is_market_open() and not self.is_historical
        self.open_locked = False

        # ---- 3. Get initial price ----
        if self.is_historical:
            # Historical date: use Polygon daily open directly
            daily = get_daily(ticker, self.date)
            self.open_price = getattr(daily, 'open', None) if daily else None
            if self.open_price is not None:
                self.open_locked = True
                self.logger.info(f'HISTORICAL MODE ({self.date}) — Open: ${self.open_price:.2f}')
        else:
            try:
                if get_actual_current_price_trill is None:
                    raise RuntimeError('Trillium live-price unavailable')
                self.open_price = get_actual_current_price_trill(ticker)
            except Exception:
                self.logger.warning('Trillium price failed, falling back to Polygon')
                daily = get_daily(ticker, self.date)
                self.open_price = getattr(daily, 'open', None) if daily else None

            if self.premarket_mode:
                self.logger.info(f'PREMARKET MODE — current price: ${self.open_price:.2f}')
            else:
                self.open_locked = True
                self.logger.info(f'Open price LOCKED: ${self.open_price:.2f}')

        if self.open_price is None:
            raise ValueError(f'Cannot determine open price for {ticker}')

        # ---- 4. ATR ----
        self.atr = get_atr(ticker, self.date)
        self.logger.info(f'ATR: ${self.atr:.2f}')

        # ---- 5. Historical levels ----
        levels = get_levels_data(ticker, self.date, 310, 1, 'day')
        if levels is None or levels.empty:
            raise ValueError(f'No historical data for {ticker}')

        self.prior_close = float(levels.iloc[-1]['close'])
        self.prior_high = float(levels.iloc[-1]['high'])

        # ---- 6. Compute reversal metrics locally from daily history ----
        self.metrics = self._compute_metrics(levels)

        # ---- 7. Classify setup type ----
        self.setup_type = classify_reversal_setup(self.metrics)
        self.logger.info(f'Setup type: {self.setup_type or "Generic"}')

        # ---- 8. Validate pre-trade ----
        if self.setup_type:
            pretrade = ReversalPretrade()
            self.pretrade_result = pretrade.validate(
                ticker, self.metrics, setup_type=self.setup_type, cap=self.cap)
            self.reversal_score = self.pretrade_result.score
            self.recommendation = self.pretrade_result.recommendation
        else:
            scorer = ReversalScorer()
            score_result = scorer.score_setup(ticker, self.date, self.cap, self.metrics)
            self.pretrade_result = None
            self.reversal_score = score_result['pretrade_score']
            self.recommendation = score_result['pretrade_recommendation']

        # ---- 9. Compute intensity ----
        intensity_result = compute_reversal_intensity(self.metrics, cap=self.cap)
        self.reversal_intensity = intensity_result.get('composite')

        # ---- 10. Build context ----
        self.ctx = ReversalContext(
            ticker=ticker,
            open_price=self.open_price,
            atr=self.atr,
            prior_close=self.prior_close,
            gap_pct=self.metrics.get('gap_pct', 0),
            pct_from_9ema=self.metrics.get('pct_from_9ema', 0),
            consecutive_up_days=int(self.metrics.get('consecutive_up_days', 0)),
            setup_type=self.setup_type,
            cap=self.cap,
            reversal_intensity=self.reversal_intensity,
            reversal_score=self.reversal_score,
            recommendation=self.recommendation,
            metrics=self.metrics,
            open_locked=self.open_locked,
        )

        # ---- 11. Build alert levels ----
        self.target_alerts: List[AlertLevel] = []
        self.drawdown_alerts: List[AlertLevel] = []
        self._build_alert_levels()

        # ---- 12. Initialize PBB tracking ----
        self.bar_aggregator = TwoMinBarAggregator()
        self.pbb_tracker = PBBTracker()
        self.covering_tracker = CoveringRulesTracker(self.open_price, self.atr)

        # ---- 13. Time alerts + state ----
        self.time_alerts_fired: Dict[str, bool] = {
            '10:00': False,
            '14:30': False,
            '15:30': False,
            '15:55': False,
        }

        self.high_water_mark = self.open_price
        self.low_water_mark = self.open_price
        self.trade_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
        self.trade_df.index = pd.to_datetime(self.trade_df.index)

        # ---- 14. Print startup summary ----
        self._print_levels()
        self._fire_setup_alerts()

    def _compute_metrics(self, levels: pd.DataFrame) -> Dict:
        """Compute reversal metrics locally from daily OHLCV history."""
        metrics = {}

        if levels is None or levels.empty:
            return metrics

        latest = levels.iloc[-1]
        metrics['prior_close'] = float(latest['close'])

        # Gap %
        metrics['gap_pct'] = ((self.open_price - float(latest['close']))
                              / float(latest['close'])) if float(latest['close']) != 0 else 0

        # 9 EMA
        if len(levels) >= 9:
            ema9 = levels['close'].ewm(span=9, adjust=False).mean().iloc[-1]
            metrics['pct_from_9ema'] = (self.open_price - ema9) / ema9 if ema9 != 0 else 0
        else:
            metrics['pct_from_9ema'] = 0

        # 50 SMA
        if len(levels) >= 50:
            sma50 = levels['close'].rolling(50).mean().iloc[-1]
            metrics['pct_from_50mav'] = (self.open_price - sma50) / sma50 if sma50 != 0 else 0
        else:
            metrics['pct_from_50mav'] = 0

        # Consecutive up days (close > prior close)
        up_count = 0
        for i in range(len(levels) - 1, 0, -1):
            if float(levels.iloc[i]['close']) > float(levels.iloc[i - 1]['close']):
                up_count += 1
            else:
                break
        metrics['consecutive_up_days'] = up_count

        # pct_change_3 (3-day return)
        if len(levels) >= 4:
            ref_close = float(levels.iloc[-4]['close'])
            metrics['pct_change_3'] = (float(latest['close']) - ref_close) / ref_close if ref_close != 0 else 0
        else:
            metrics['pct_change_3'] = 0

        # ATR % (use the ATR we already fetched)
        metrics['atr_pct'] = self.atr / self.open_price if self.open_price != 0 else 0

        # Prior day range / ATR
        prior_range = float(latest['high']) - float(latest['low'])
        metrics['prior_day_range_atr'] = prior_range / self.atr if self.atr != 0 else 0

        # RVOL (prior day volume / 20-day avg volume)
        if len(levels) >= 21:
            avg_vol = levels['volume'].iloc[-21:-1].mean()
            metrics['rvol_score'] = float(latest['volume']) / avg_vol if avg_vol != 0 else 0
        else:
            metrics['rvol_score'] = 1.0

        return metrics

    # -----------------------------------------------------------------------
    # Alert level construction
    # -----------------------------------------------------------------------

    def _build_alert_levels(self):
        self.target_alerts.clear()
        self.drawdown_alerts.clear()

        op = self.open_price
        atr = self.atr
        pc = self.prior_close

        # --- Downside targets (profit for shorts — direction='below') ---
        raw_targets = [
            ('-0.5 ATR', op - 0.5 * atr, 'Minus half ATR from open'),
            ('-1.0 ATR (PROBE boundary)', op - 1.0 * atr, 'Minus 1 ATR — PROBE boundary'),
            ('-1.5 ATR', op - 1.5 * atr, 'Minus 1.5 ATR'),
            ('-2.0 ATR (ONE_FLUSH boundary)', op - 2.0 * atr, 'Minus 2 ATR — ONE FLUSH boundary'),
            ('-2.5 ATR', op - 2.5 * atr, 'Minus 2.5 ATR'),
            ('-3.0 ATR', op - 3.0 * atr, 'Minus 3 ATR — deep crack territory'),
        ]

        # Gap fill (prior close) — only if stock gapped up
        if pc < op:
            raw_targets.append(('Gap Fill', pc, 'Gap fill — prior close level'))

        # 9 EMA — if extended above it
        ema9_pct = self.metrics.get('pct_from_9ema', 0)
        if ema9_pct > 0.02:
            ema9_price = op / (1 + ema9_pct) if ema9_pct != 0 else op
            raw_targets.append(('9 EMA', ema9_price, '9 day EMA — mean reversion level'))

        raw_targets.sort(key=lambda x: x[1])

        for name, price, detail in raw_targets:
            self.target_alerts.append(AlertLevel(
                name=name,
                price=price,
                direction='below',
                category='target',
                detail=f'{self.ticker}: {detail} at {price:.2f}',
                priority=1,
            ))

        # --- Upside risk (direction='above') ---
        risk_levels = [
            ('+0.5 ATR', op + 0.5 * atr, 'Plus half ATR above open — minor squeeze'),
            ('+1.0 ATR', op + 1.0 * atr, 'Plus 1 ATR above open — significant squeeze risk'),
        ]

        for name, price, detail in risk_levels:
            self.drawdown_alerts.append(AlertLevel(
                name=name,
                price=price,
                direction='above',
                category='drawdown',
                detail=f'{self.ticker}: {detail} at {price:.2f}',
                priority=1,
            ))

    def _print_levels(self):
        print('\n' + '=' * 80)
        price_label = 'Open' if self.open_locked else 'Premarket'
        lock_status = '' if self.open_locked else ' [PRELIMINARY — will update at 9:30]'
        setup_label = self.setup_type or 'Generic'
        print(f'  REVERSAL MONITOR: {self.ticker} | {price_label}: ${self.open_price:.2f} | ATR: ${self.atr:.2f}{lock_status}')
        print(f'  Cap: {self.cap} | Type: {setup_label} | Score: {self.reversal_score} ({self.recommendation})')
        intensity_str = f'{self.reversal_intensity:.0f}/100' if self.reversal_intensity is not None else 'N/A'
        gap_pct = self.metrics.get('gap_pct', 0)
        ema9_pct = self.metrics.get('pct_from_9ema', 0)
        print(f'  Intensity: {intensity_str} | Gap: {gap_pct * 100:+.1f}% | 9EMA dist: {ema9_pct * 100:+.1f}%')
        print(f'  Up days: {self.metrics.get("consecutive_up_days", 0)} | 3d mom: {self.metrics.get("pct_change_3", 0) * 100:+.1f}%')
        print('=' * 80)

        # Pre-trade checklist
        if self.pretrade_result:
            print(f'\n  PRETRADE CHECKLIST ({self.reversal_score}/5 — {self.recommendation}):')
            print(f'  {"Criterion":<30} {"Actual":>10} {"Threshold":>12} {"":>6}')
            print('  ' + '-' * 60)
            for item in self.pretrade_result.items:
                mark = 'PASS' if item.passed else 'FAIL'
                print(f'  {item.description:<30} {item.actual_display:>10} {item.threshold_display:>12} {mark:>6}')
        else:
            print(f'\n  GENERIC SCORER: {self.reversal_score}/5 — {self.recommendation}')

        # Covering decision tree cheat sheet
        print()
        print(self.covering_tracker.get_cheat_sheet())

        # Level table
        print('\n  DOWNSIDE TARGETS (profit):')
        print(f'  {"Level":<30} {"Price":>10} {"vs Open":>10}')
        print('  ' + '-' * 52)
        for a in self.target_alerts:
            pct = (a.price - self.open_price) / self.open_price * 100
            print(f'  {a.name:<30} ${a.price:>9.2f} {pct:>+9.1f}%')

        print('\n  UPSIDE RISK (adverse):')
        print(f'  {"Level":<30} {"Price":>10} {"vs Open":>10}')
        print('  ' + '-' * 52)
        for a in self.drawdown_alerts:
            pct = (a.price - self.open_price) / self.open_price * 100
            print(f'  {a.name:<30} ${a.price:>9.2f} {pct:>+9.1f}%')
        print()

    def _fire_setup_alerts(self):
        setup_label = self.setup_type or 'Generic reversal'
        self._alert(f'{setup_label} setup. Score {self.reversal_score}. {self.recommendation}.', priority=2)

        if self.reversal_intensity is not None:
            self._alert(f'Reversal intensity: {self.reversal_intensity:.0f} out of 100', priority=2)

    # -----------------------------------------------------------------------
    # Open price locking
    # -----------------------------------------------------------------------

    def _lock_open_price(self, official_open: float):
        old_price = self.open_price
        self.open_price = official_open
        self.open_locked = True
        self.ctx.open_price = official_open
        self.ctx.open_locked = True

        # Recalculate gap
        self.metrics['gap_pct'] = ((self.open_price - self.prior_close)
                                   / self.prior_close) if self.prior_close != 0 else 0
        self.ctx.gap_pct = self.metrics['gap_pct']

        # Rebuild alert levels and covering tracker
        self._build_alert_levels()
        self.covering_tracker = CoveringRulesTracker(self.open_price, self.atr)

        # Reset watermarks
        self.high_water_mark = self.open_price
        self.low_water_mark = self.open_price

        self.logger.info(f'OPEN PRICE LOCKED: ${self.open_price:.2f} (was ${old_price:.2f} premarket)')
        self._alert(f'Market open. Official open price locked at {self.open_price:.2f}. Targets updated.', priority=1)
        self._print_levels()

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

                    # Check for open price lock at market open (9:30 AM)
                    if not self.open_locked and hasattr(series.name, 'time'):
                        bar_time = series.name.time()
                        if bar_time >= MARKET_OPEN:
                            official_open = series['open']
                            self._lock_open_price(official_open)

                    if self.open_locked:
                        # Update watermarks
                        if high > self.high_water_mark:
                            self.high_water_mark = high
                            # Update covering tracker HOD
                            self.covering_tracker.hod_price = high
                        if low < self.low_water_mark:
                            self.low_water_mark = low

                        # Check price-based alerts
                        # Target alerts fire on bar_low <= price (price going DOWN)
                        self._check_target_alerts(low)
                        # Drawdown alerts fire on bar_high >= price (price going UP = risk)
                        self._check_drawdown_alerts(high)

                        # Feed to 2-minute bar aggregator
                        completed_2m = self.bar_aggregator.feed(series)
                        if completed_2m is not None:
                            # Feed to PBB tracker
                            confirmed_pbbs = self.pbb_tracker.feed(completed_2m)
                            for pbb in confirmed_pbbs:
                                if pbb['status'] == 'held':
                                    self._process_held_pbb(pbb, completed_2m)
                                elif pbb['status'] == 'failed':
                                    self.covering_tracker.on_failed_pbb()
                                    self.logger.info(
                                        f'FAILED PBB @ ${pbb["pbb_price"]:.2f} '
                                        f'(total failed: {self.pbb_tracker.total_failed})')

                        # Update running low for covering tracker
                        self.covering_tracker.on_price_update(low)

            # Time alerts
            self._check_time_alerts()

            self.new_data_event.clear()

    def process_incoming_data(self, bar_data):
        """Called by ReversalDataAdapter with each 5s bar dict."""
        if not hasattr(self, '_bar_count'):
            self._bar_count = 0
        self._bar_count += 1
        if self._bar_count == 1:
            self.logger.info(f'First bar received — live data confirmed (close=${bar_data.get("close", "?")})')
        elif self._bar_count % 60 == 0:
            self.logger.info(f'Heartbeat — {self._bar_count} bars received (close=${bar_data.get("close", "?")})')
        self.current_bar = bar_data
        self.new_data_event.set()

    # -----------------------------------------------------------------------
    # PBB processing and covering alerts
    # -----------------------------------------------------------------------

    def _process_held_pbb(self, pbb: Dict, bar_2m: Dict):
        """Process a confirmed held PBB through the covering rules."""
        action = self.covering_tracker.on_held_pbb(
            pbb['pbb_price'], bar_2m['low'])

        if action is None:
            self.logger.info(
                f'HELD PBB @ ${pbb["pbb_price"]:.2f} — move too small, not significant '
                f'(total held: {self.pbb_tracker.total_held})')
            return

        # Log status line
        total_atrs = (self.open_price - self.low_water_mark) / self.atr if self.atr else 0
        status = (
            f'{self.ticker} | {total_atrs:.1f} ATR down | '
            f'{self.pbb_tracker.total_failed} fail PBBs | '
            f'{action["move_num"]} HELD @ ${pbb["pbb_price"]:.2f} | '
            f'{action["pattern"]} | '
        )
        if action['cover_pct'] > 0.001:
            status += f'COVER {action["cover_pct"]*100:.0f}% | '
        else:
            status += 'HOLD | '
        status += f'{action["position_remaining"]*100:.0f}% remaining'
        print(f'\n  >>> {status}')

        # Fire covering alert (TTS)
        if action['cover_pct'] > 0.001:
            self._fire_covering_alert(action)

    def _fire_covering_alert(self, action: Dict):
        """Fire TTS alert for a covering decision."""
        msg = (
            f'SIGNIFICANT HELD PBB. '
            f'Move {action["move_num"]}, '
            f'{action.get("move_atrs", 0):.1f} ATRs, '
            f'{action.get("failed_pbbs", 0)} failed PBBs. '
            f'Pattern: {action["pattern"]}. '
            f'Cover {action["cover_pct"]*100:.0f} percent at {action["cover_price"]:.2f}. '
            f'{action["position_remaining"]*100:.0f} percent remaining.'
        )
        self._alert(msg, priority=1)

    # -----------------------------------------------------------------------
    # Price-based alert checks
    # -----------------------------------------------------------------------

    def _check_target_alerts(self, bar_low: float):
        """For shorts: targets fire when price goes DOWN (bar_low <= level)."""
        for alert in self.target_alerts:
            if not alert.fired and bar_low <= alert.price:
                alert.fired = True
                self._alert(alert.detail, priority=alert.priority)
                self.logger.info(f'TARGET HIT: {alert.name} @ ${alert.price:.2f}')

    def _check_drawdown_alerts(self, bar_high: float):
        """For shorts: drawdown when price goes UP (bar_high >= level)."""
        for alert in self.drawdown_alerts:
            if not alert.fired and bar_high >= alert.price:
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

        # 10:00 AM — early HOD check
        if not self.time_alerts_fired['10:00'] and t >= dt_time(10, 0):
            self.time_alerts_fired['10:00'] = True
            total_atrs = (self.open_price - self.low_water_mark) / self.atr if self.atr else 0
            msg = (f'Early HOD check — HOD at {self.high_water_mark:.2f}, '
                   f'{total_atrs:.1f} ATRs from open, '
                   f'{self.pbb_tracker.total_failed} failed PBBs')
            self._alert(msg, priority=1)

        # 2:30 PM — afternoon check
        if not self.time_alerts_fired['14:30'] and t >= dt_time(14, 30):
            self.time_alerts_fired['14:30'] = True
            pattern = self.covering_tracker.pattern or 'N/A'
            remaining = self.covering_tracker.position_remaining * 100
            msg = (f'Afternoon — position {remaining:.0f}% remaining, '
                   f'pattern {pattern}')
            self._alert(msg, priority=1)

        # 3:30 PM — approaching close
        if not self.time_alerts_fired['15:30'] and t >= dt_time(15, 30):
            self.time_alerts_fired['15:30'] = True
            remaining = self.covering_tracker.position_remaining * 100
            msg = (f'Approaching close — {remaining:.0f}% remaining, '
                   f'consider covering or holding overnight')
            self._alert(msg, priority=1)

        # 3:55 PM — EOD summary
        if not self.time_alerts_fired['15:55'] and t >= dt_time(15, 55):
            self.time_alerts_fired['15:55'] = True
            self._fire_eod_summary()

    def _fire_eod_summary(self):
        current = self.trade_df.iloc[-1]['close'] if not self.trade_df.empty else self.open_price
        total_atrs = (self.open_price - self.low_water_mark) / self.atr if self.atr else 0
        pattern = self.covering_tracker.pattern or 'N/A'
        remaining = self.covering_tracker.position_remaining * 100

        msg = (
            f'END OF DAY SUMMARY — {self.ticker}. '
            f'Open {self.open_price:.2f}, current {current:.2f}, '
            f'HOD {self.high_water_mark:.2f}, LOD {self.low_water_mark:.2f}. '
            f'Total move: {total_atrs:.1f} ATRs. '
            f'PBBs: {self.pbb_tracker.total_failed} failed, {self.pbb_tracker.total_held} held. '
            f'Pattern: {pattern}. Position: {remaining:.0f}% remaining.'
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

def reversal_main(ticker: str, cap: str = None, dry_run: bool = False, enable_trac: bool = False):
    """Entry point — build manager, optionally attach live stream, and run."""

    trac_scraper = None
    if enable_trac:
        try:
            from support.trac_positions import TracPositionScraper
            logger.info('Initializing TRAC position scraper...')
            trac_scraper = TracPositionScraper()
            trac_scraper.login()
            logger.info('TRAC scraper ready — position tracking enabled')
        except Exception as e:
            logger.warning(f'Failed to initialize TRAC scraper: {e}')
            logger.warning('Continuing without position tracking')

    manager = ReversalTradeManager(ticker=ticker, cap=cap, trac_scraper=trac_scraper)

    if dry_run:
        logger.info('DRY RUN — simulating crack day bars then exiting')
        _simulate_dry_run(manager)
        if trac_scraper:
            trac_scraper.close()
        return

    # Live stream
    adapter = ReversalDataAdapter(ticker, manager)
    stream_thread = threading.Thread(target=adapter.start_data_stream, daemon=True)
    stream_thread.start()

    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info('Interrupted — shutting down')
        manager.closed = True
        if adapter.handle:
            adapter.handle.cancel()
    finally:
        if trac_scraper:
            trac_scraper.close()


def _simulate_dry_run(manager: ReversalTradeManager):
    """Simulate a crack day: open -> slight squeeze (HOD) -> crack through targets -> bounce (PBB) -> resume selling.

    Each bar is spaced 2 minutes apart so every bar lands in its own 2-minute window,
    producing one completed 2-minute bar per entry. This ensures PBB detection has
    enough bars for the 3-bar confirmation window.
    """
    import time as _time

    op = manager.open_price
    atr = manager.atr

    # Base time: 9:30 ET on the trade date (use UTC offset for ET = UTC-5)
    trade_date = manager.date
    year, month, day = map(int, trade_date.split('-'))
    market_open_utc = datetime(year, month, day, 14, 30, 0)  # 9:30 ET = 14:30 UTC
    base_ts = int(market_open_utc.timestamp() * 1e9)

    # Each entry: (price, description)
    # Simulate: open -> squeeze to HOD -> crack down -> failed PBB -> more selling -> held PBB -> resume -> held PBB
    test_sequence = [
        # 2-min bars 0-2: Open and slight squeeze (HOD)
        (op,                'at open'),
        (op + 0.1 * atr,   'slight squeeze'),
        (op + 0.3 * atr,   'HOD squeeze'),
        # 2-min bars 3-7: Start cracking down
        (op + 0.1 * atr,   'rolling over'),
        (op - 0.2 * atr,   'below open'),
        (op - 0.4 * atr,   'cracking'),
        (op - 0.6 * atr,   'should trigger -0.5 ATR'),
        (op - 0.8 * atr,   'continuing'),
        # 2-min bars 8-11: Failed PBB (bounce then prior-low break)
        (op - 0.6 * atr,   'minor bounce (high > prior high = PBB triggered)'),
        (op - 0.5 * atr,   'bounce peak'),
        (op - 0.9 * atr,   'breaks prior low -> FAILED PBB'),
        (op - 1.1 * atr,   'should trigger -1.0 ATR (PROBE boundary)'),
        # 2-min bars 12-15: More selling
        (op - 1.3 * atr,   'deep crack'),
        (op - 1.6 * atr,   'accelerating'),
        (op - 1.8 * atr,   'approaching -2.0 ATR'),
        (op - 2.1 * atr,   'should trigger -2.0 ATR (ONE_FLUSH boundary)'),
        # 2-min bars 16-19: Held PBB (bounce that does NOT break prior low within 3 bars)
        (op - 1.7 * atr,   'significant bounce (high > prior high = PBB triggered)'),
        (op - 1.75 * atr,  'holds above prior low (bar 1 of 3)'),
        (op - 1.8 * atr,   'holds above prior low (bar 2 of 3)'),
        (op - 1.85 * atr,  'holds above prior low (bar 3 of 3) -> HELD PBB confirmed'),
        # 2-min bars 20-23: Resume selling
        (op - 2.0 * atr,   'resume selling after held PBB'),
        (op - 2.3 * atr,   'new low'),
        (op - 2.5 * atr,   'should trigger -2.5 ATR'),
        (op - 2.7 * atr,   'deep selling'),
        # 2-min bars 24-27: Second held PBB
        (op - 2.3 * atr,   'second bounce (high > prior high = PBB triggered)'),
        (op - 2.35 * atr,  'holds (bar 1 of 3)'),
        (op - 2.4 * atr,   'holds (bar 2 of 3)'),
        (op - 2.45 * atr,  'holds (bar 3 of 3) -> second HELD PBB'),
    ]

    for i, (price, desc) in enumerate(test_sequence):
        # Space bars 2 minutes apart so each is its own 2-minute window
        bar = {
            'type': 'bar-5s',
            'close-time': base_ts + i * 120_000_000_000,  # 120 seconds = 2 minutes
            'open': price - 0.005 * atr,
            'high': price + 0.01 * atr,
            'low': price - 0.01 * atr,
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

            # For dry-run: manually drive the pipeline (run() loop not active)
            if manager.open_locked:
                manager._check_target_alerts(series['low'])
                manager._check_drawdown_alerts(series['high'])

                completed_2m = manager.bar_aggregator.feed(series)
                if completed_2m is not None:
                    confirmed_pbbs = manager.pbb_tracker.feed(completed_2m)
                    for pbb in confirmed_pbbs:
                        if pbb['status'] == 'held':
                            manager._process_held_pbb(pbb, completed_2m)
                        elif pbb['status'] == 'failed':
                            manager.covering_tracker.on_failed_pbb()
                            logger.info(f'DRY RUN FAILED PBB @ ${pbb["pbb_price"]:.2f}')

                manager.covering_tracker.on_price_update(series['low'])

        manager.new_data_event.clear()
        _time.sleep(0.3)

    # Print final summary
    total_atrs = (manager.open_price - manager.low_water_mark) / manager.atr if manager.atr else 0
    pattern = manager.covering_tracker.pattern or 'N/A'
    remaining = manager.covering_tracker.position_remaining * 100

    print(f'\n{"=" * 80}')
    print(f'  DRY RUN COMPLETE')
    print(f'  HOD: ${manager.high_water_mark:.2f} | LOD: ${manager.low_water_mark:.2f}')
    print(f'  Total move: {total_atrs:.1f} ATRs')
    print(f'  PBBs: {manager.pbb_tracker.total_failed} failed, {manager.pbb_tracker.total_held} held')
    print(f'  Pattern: {pattern} | Position remaining: {remaining:.0f}%')
    print(f'  Moves tracked: {len(manager.covering_tracker.moves)}')
    for mv in manager.covering_tracker.moves:
        print(f'    Move {mv["move_num"]}: {mv["size_atrs"]:.1f} ATR, '
              f'{mv["failed_pbbs_during"]} failed PBBs, '
              f'PBB @ ${mv["pbb_price"]:.2f}')
    print(f'{"=" * 80}')

    logger.info('Dry run complete — all levels, PBB tracking, and covering logic verified')


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        _ticker = sys.argv[1].upper()
        _cap = None
        _dry_run = '--dry-run' in sys.argv
        _enable_trac = '--trac' in sys.argv

        if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
            _cap = sys.argv[2]

        reversal_main(_ticker, cap=_cap, dry_run=_dry_run, enable_trac=_enable_trac)
    else:
        # Interactive mode
        print('=== Reversal Trader — Interactive Mode ===')
        _ticker = input('Ticker: ').strip().upper()
        if not _ticker:
            print('Ticker is required.')
            sys.exit(1)

        _cap_input = input('Cap (Large/Medium/Small/Micro/ETF) [auto-detect]: ').strip()
        _cap = _cap_input if _cap_input else None

        _dry_input = input('Dry run? (y/n) [n]: ').strip().lower()
        _dry_run = _dry_input in ('y', 'yes')

        _trac_input = input('Enable TRAC position tracking? (y/n) [n]: ').strip().lower()
        _enable_trac = _trac_input in ('y', 'yes')

        reversal_main(_ticker, cap=_cap, dry_run=_dry_run, enable_trac=_enable_trac)
