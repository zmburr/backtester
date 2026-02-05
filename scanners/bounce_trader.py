"""
Bounce Trade Alert Monitor — Live price-level and time-based alerts for bounce (long) trades.

Follows the live_watcher.py architecture (event-driven threading + TTS alerts) but purpose-built
for the bounce playbook derived from 36 historical trades.

All levels are objective — anchored to open price, prior close, and selloff high.
Optionally integrates with TRAC position data for position-aware alerts.

Usage:
    python scanners/bounce_trader.py NVDA Medium
    python scanners/bounce_trader.py COIN              # auto-detect cap
    python scanners/bounce_trader.py NVDA Medium --dry-run
    python scanners/bounce_trader.py NVDA Medium --trac   # enable TRAC position integration
"""

import sys
import os
import threading
import queue
import logging
import pytz
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import gtts  # type: ignore
except Exception:  # pragma: no cover
    gtts = None

try:
    import pygame  # type: ignore
except Exception:  # pragma: no cover
    pygame = None

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
try:
    # Optional dependency (large). If absent, we fall back to a small numpy implementation.
    from scipy.stats import percentileofscore as _pctrank  # type: ignore
except Exception:  # pragma: no cover
    def _pctrank(a, score, kind='rank'):
        """
        Minimal percentile-of-score implementation.

        We approximate SciPy's percentileofscore(kind='rank') with a stable definition:
        mean of strict and weak ranks: pct = 100*(count(<) + 0.5*count(==))/n
        """
        try:
            arr = np.asarray(a, dtype=float)
            if arr.size == 0:
                return np.nan
            s = float(score)
        except Exception:
            return np.nan

        if kind in ('rank', 'mean'):
            return 100.0 * (np.sum(arr < s) + 0.5 * np.sum(arr == s)) / arr.size
        if kind == 'weak':
            return 100.0 * np.sum(arr <= s) / arr.size
        if kind == 'strict':
            return 100.0 * np.sum(arr < s) / arr.size
        return 100.0 * np.sum(arr <= s) / arr.size

from data_queries.polygon_queries import get_atr, get_levels_data
try:
    from data_queries.trillium_queries import get_actual_current_price_trill
except Exception:  # pragma: no cover
    get_actual_current_price_trill = None
from analyzers.bounce_scorer import BouncePretrade, fetch_bounce_metrics, classify_stock, SETUP_PROFILES


# ---------------------------------------------------------------------------
# Utilities — inlined from live_watcher.py to avoid transitive import issues
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
            unique_name = f'bounce_{_uuid.uuid4().hex[:8]}.mp3'
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), unique_name)
            tts.save(filepath)
            _play_audio_file_blocking(filepath)
        except Exception:
            # Keep the loop alive even if TTS/audio fails intermittently.
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
    """
    Queue a TTS audio alert.

    We intentionally serialize audio playback (single worker thread) so that
    multiple alerts fired in the same second never overlap.
    """
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
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


def is_market_open() -> bool:
    """Check if US market is currently open (9:30 AM - 4:00 PM ET)."""
    now_et = datetime.now(EASTERN).time()
    return MARKET_OPEN <= now_et < MARKET_CLOSE


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
class PositionInfo:
    """Live position data from TRAC."""
    has_position: bool = False
    shares: int = 0
    side: str = ''              # 'long' or 'short'
    avg_price: Optional[float] = None
    bp_used: Optional[float] = None
    pnl_per_share: float = 0.0
    pnl_total: float = 0.0
    pnl_atr: float = 0.0        # P&L in ATR units
    # Stop tracking
    stop_price: Optional[float] = None  # Low-of-day stop (computed from full intraday tape)
    stopped_out: bool = False


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
    open_locked: bool = False   # True once official open price is set
    position: PositionInfo = field(default_factory=PositionInfo)  # TRAC position data


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
        import sheldatagateway
        from sheldatagateway import environments
        from dotenv import load_dotenv
        load_dotenv()

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
# BounceTradeManager — the core monitor
# ---------------------------------------------------------------------------

class BounceTradeManager:

    def __init__(self, ticker: str, cap: str = None, date: str = None, trac_scraper=None):
        self.ticker = ticker
        self.date = date or datetime.now(EASTERN).strftime('%Y-%m-%d')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.closed = False
        self.handle = None
        self.new_data_event = threading.Event()
        self.current_bar = None
        self.current_time = None

        # ---- TRAC position integration ----
        self.trac_scraper = trac_scraper
        self.position = PositionInfo()
        self._position_alerts_fired: Dict[str, bool] = {}

        # ---- 1. Determine cap ----
        self.cap = cap or get_ticker_cap(ticker)
        self.logger.info(f'Cap: {self.cap}')

        # ---- 2. Check if premarket ----
        self.premarket_mode = not is_market_open()
        self.open_locked = False
        self.first_bar_of_session = None  # Will capture first bar at/after 9:30

        # ---- 3. Get initial price (premarket or open) ----
        try:
            if get_actual_current_price_trill is None:
                raise RuntimeError('Trillium live-price unavailable (module not installed)')
            self.open_price = get_actual_current_price_trill(ticker)
        except Exception:
            self.logger.warning('Trillium price failed, falling back to Polygon')
            from data_queries.polygon_queries import get_daily
            daily = get_daily(ticker, self.date)
            self.open_price = getattr(daily, 'open', None) if daily else None
        if self.open_price is None:
            raise ValueError(f'Cannot determine open price for {ticker}')

        if self.premarket_mode:
            self.logger.info(f'PREMARKET MODE — current price: ${self.open_price:.2f} (targets will update at 9:30)')
        else:
            self.open_locked = True
            self.logger.info(f'Open price LOCKED: ${self.open_price:.2f}')

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
        # Use cap-specific thresholds and lock setup_type for consistency with our classification step.
        self.pretrade_result = BouncePretrade().validate(
            ticker,
            self.metrics,
            force_setup=self.setup_type,
            cap=self.cap,
        )
        self.bounce_score = self.pretrade_result.score
        self.recommendation = self.pretrade_result.recommendation

        # ---- 10. Intensity ----
        intensity_result = compute_bounce_intensity(self.metrics)
        self.bounce_intensity = intensity_result['composite']

        # ---- 11. Exhaustion gap ----
        self.is_etf = self.cap == 'ETF'
        self.has_exhaustion_gap = (self.gap_pct <= -0.05 and self.consecutive_down_days >= 3)

        # ---- 12. Build context ----
        # ---- 12a. Fetch initial TRAC position ----
        if self.trac_scraper:
            self._refresh_position()

        # ---- 12b. Build context ----
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
            open_locked=self.open_locked,
            position=self.position,
        )

        # ---- 13. Build alert levels ----
        self.target_alerts: List[AlertLevel] = []
        self.drawdown_alerts: List[AlertLevel] = []
        self.checklist_price_alerts: List[AlertLevel] = []
        self.checklist_price_levels: List[Dict] = []
        self._build_alert_levels()
        self._build_checklist_price_levels()

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

    def _build_checklist_price_levels(self):
        """
        Build downside price levels for checklist criteria that are directly price-derivable.

        Two level types:
        - Threshold: the exact price where the checklist item would flip to PASS
        - A-median: the price where the setup matches the Grade-A median for that metric

        These are independent of the official open price, so they are monitored in premarket too.
        """
        self.checklist_price_levels.clear()
        self.checklist_price_alerts.clear()

        profile = SETUP_PROFILES.get(self.setup_type) or SETUP_PROFILES.get('GapFade_strongstock')
        if profile is None:
            return

        # "Current" at startup: prefer live/trillium price baked into metrics, else our initial price.
        current_price = self.metrics.get('current_price')
        if current_price is None or (isinstance(current_price, float) and pd.isna(current_price)):
            current_price = self.open_price

        # Map pretrade PASS/FAIL (already cap-adjusted above)
        passed_map: Dict[str, bool] = {}
        try:
            for item in self.pretrade_result.items:
                passed_map[item.name] = bool(item.passed)
        except Exception:
            passed_map = {}

        def _is_valid_price(x) -> bool:
            if x is None:
                return False
            if isinstance(x, float) and pd.isna(x):
                return False
            try:
                return float(x) > 0
            except Exception:
                return False

        def _add_pct_level(criterion_key: str, title: str, reference_price, reference_label: str):
            if not _is_valid_price(reference_price):
                return

            threshold_pct = profile.get_threshold(criterion_key, self.cap)
            median_pct = profile.reference_medians.get(criterion_key)

            ref_px = float(reference_price)
            threshold_price = ref_px * (1.0 + float(threshold_pct)) if threshold_pct is not None else None
            median_price = ref_px * (1.0 + float(median_pct)) if median_pct is not None else None

            self.checklist_price_levels.append({
                'criterion': criterion_key,
                'title': title,
                'reference_label': reference_label,
                'reference_price': ref_px,
                'threshold_pct': float(threshold_pct) if threshold_pct is not None else None,
                'threshold_price': float(threshold_price) if threshold_price is not None else None,
                'median_pct': float(median_pct) if median_pct is not None else None,
                'median_price': float(median_price) if median_price is not None else None,
            })

            # Arm audible alerts when levels are BELOW current price (i.e., "downside triggers")
            if _is_valid_price(current_price):
                cp = float(current_price)

                # Threshold alert — only if currently FAILING that criterion
                if threshold_price is not None and passed_map.get(criterion_key) is False and cp > threshold_price:
                    pct_label = f"{abs(float(threshold_pct)) * 100:.0f}" if threshold_pct is not None else ""
                    msg = f"{self.ticker}: checklist threshold hit — {title} {pct_label} percent at {threshold_price:.2f}"
                    self.checklist_price_alerts.append(AlertLevel(
                        name=f'{criterion_key}_threshold',
                        price=float(threshold_price),
                        direction='below',
                        category='checklist',
                        detail=msg,
                        priority=1,
                    ))

                # A-median alert — if current price is above the median-equivalent level
                if median_price is not None and cp > median_price:
                    pct_label = f"{abs(float(median_pct)) * 100:.0f}" if median_pct is not None else ""
                    msg = f"{self.ticker}: A median level — {title} {pct_label} percent at {median_price:.2f}"
                    self.checklist_price_alerts.append(AlertLevel(
                        name=f'{criterion_key}_median',
                        price=float(median_price),
                        direction='below',
                        category='checklist',
                        detail=msg,
                        priority=1,
                    ))

        # Price-derivable checklist criteria
        _add_pct_level(
            'pct_off_30d_high',
            'discount from 30 day high',
            self.metrics.get('high_30d'),
            '30d high',
        )
        _add_pct_level(
            'selloff_total_pct',
            'selloff depth from first down day open',
            self.metrics.get('selloff_start_open'),
            'selloff start open',
        )
        # NOTE: checklist uses "gap at open", but for live monitoring we treat this as
        # "down vs prior close" since it remains meaningful throughout the session.
        _add_pct_level(
            'gap_pct',
            'down versus prior close',
            self.metrics.get('prior_close') or self.prior_close,
            'prior close',
        )

        # Keep a consistent, human-friendly display order
        _order = {'pct_off_30d_high': 1, 'selloff_total_pct': 2, 'gap_pct': 3}
        self.checklist_price_levels.sort(key=lambda x: _order.get(x.get('criterion'), 999))

    def _build_time_alerts(self):
        self.time_alerts_fired = {
            '10:00': False,
            '14:30': False,
            '15:30': False,
            '15:55': False,
        }

    def _check_checklist_price_alerts(self, bar_low: float):
        for alert in self.checklist_price_alerts:
            if not alert.fired and bar_low <= alert.price:
                alert.fired = True
                self._alert(alert.detail, priority=alert.priority)
                self.logger.info(f'CHECKLIST LEVEL: {alert.name} @ ${alert.price:.2f}')

    def _print_levels(self):
        print('\n' + '=' * 72)
        price_label = 'Open' if self.open_locked else 'Premarket'
        lock_status = '' if self.open_locked else ' [PRELIMINARY — will update at 9:30]'
        print(f'  BOUNCE MONITOR: {self.ticker} | {price_label}: ${self.open_price:.2f} | ATR: ${self.atr:.2f}{lock_status}')
        print(f'  Cap: {self.cap} | Type: {self.setup_type} | Score: {self.bounce_score}/5 ({self.recommendation})')
        print(f'  Intensity: {self.bounce_intensity:.0f}/100 | Gap: {self.gap_pct * 100:+.1f}%')
        if self.selloff_high:
            print(f'  Selloff High: ${self.selloff_high:.2f} | Down Days: {self.consecutive_down_days}')

        # Show TRAC position if available
        if self.position.has_position:
            pos_str = f'  TRAC POSITION: {self.position.side.upper()} {self.position.shares} shares'
            if self.position.avg_price:
                pos_str += f' @ ${self.position.avg_price:.2f}'
            if self.position.stop_price:
                pos_str += f' | STOP: ${self.position.stop_price:.2f}'
            print(pos_str)

        print('=' * 72)

        # Score breakdown
        print(f'\n  PRETRADE CHECKLIST ({self.bounce_score}/5 — {self.recommendation}):')
        print(f'  {"Criterion":<30} {"Actual":>10} {"Threshold":>12} {"":>6}')
        print('  ' + '-' * 60)
        for item in self.pretrade_result.items:
            mark = 'PASS' if item.passed else 'FAIL'
            ref = f'  ({item.reference})' if item.reference else ''
            print(f'  {item.description:<30} {item.actual_display:>10} {item.threshold_display:>12} {mark:>6}{ref}')
        if self.pretrade_result.bonuses:
            print(f'\n  BONUSES:')
            for b in self.pretrade_result.bonuses:
                print(f'    + {b}')
        if self.pretrade_result.warnings:
            print(f'\n  WARNINGS:')
            for w in self.pretrade_result.warnings:
                print(f'    ! {w}')

        # Checklist-derived downside "price-to-pass" levels (and Grade-A medians)
        if self.checklist_price_levels:
            now_px = self.trade_df.iloc[-1]['close'] if not self.trade_df.empty else self.open_price
            print(f'\n  CHECKLIST THRESHOLD PRICES (DOWN SIDE):  now=${now_px:.2f}')
            print(f'  {"Metric":<32} {"Now%":>10} {"Pass@":>18} {"A-med@":>18} {"Ref":>12} {"":>6}')
            print('  ' + '-' * 96)
            for lvl in self.checklist_price_levels:
                ref_px = lvl.get('reference_price')
                if ref_px and ref_px != 0:
                    now_pct = (now_px - ref_px) / ref_px
                    now_pct_str = f'{now_pct * 100:+.1f}%'
                else:
                    now_pct_str = 'N/A'

                thr_pct = lvl.get('threshold_pct')
                thr_px = lvl.get('threshold_price')
                if thr_pct is not None and thr_px is not None:
                    pass_str = f'{abs(thr_pct) * 100:.0f}% / ${thr_px:.2f}'
                    status = 'PASS' if now_px <= thr_px else 'WAIT'
                else:
                    pass_str = 'N/A'
                    status = ''

                med_pct = lvl.get('median_pct')
                med_px = lvl.get('median_price')
                med_str = f'{abs(med_pct) * 100:.0f}% / ${med_px:.2f}' if med_pct is not None and med_px is not None else 'N/A'

                ref_str = f'${ref_px:.2f}' if ref_px is not None else 'N/A'
                print(f'  {lvl.get("title",""):<32} {now_pct_str:>10} {pass_str:>18} {med_str:>18} {ref_str:>12} {status:>6}')

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
            self._alert('WEAKSTOCK bounce — median high plus 24 percent, close plus 12', priority=2)
        else:
            self._alert('STRONGSTOCK bounce — median high plus 7 percent, close plus 2', priority=2)

        if self.is_etf:
            self._alert('ETF — more contained. Plus 7 percent median vs plus 37 for stocks on cluster days', priority=2)

        if self.has_exhaustion_gap:
            self._alert('EXHAUSTION GAP — median high plus 22 percent', priority=2)

        self._alert(f'Bounce intensity: {self.bounce_intensity:.0f} out of 100', priority=2)

        # Announce score with breakdown (TTS)
        passed_names = [i.name.replace('_', ' ') for i in self.pretrade_result.items if i.passed]
        failed_names = [i.name.replace('_', ' ') for i in self.pretrade_result.items if not i.passed]
        score_msg = f'Pretrade score {self.bounce_score} out of 6 — {self.recommendation}'
        if passed_names:
            score_msg += f'. Passing: {", ".join(passed_names)}'
        if failed_names:
            score_msg += f'. Failing: {", ".join(failed_names)}'
        self._alert(score_msg, priority=1)

        # Announce position if we have one
        if self.position.has_position:
            pos_msg = f'TRAC position detected: {self.position.side} {self.position.shares} shares'
            if self.position.avg_price:
                pos_msg += f' at {self.position.avg_price:.2f}'
            self._alert(pos_msg, priority=1)

    # -----------------------------------------------------------------------
    # TRAC Position tracking
    # -----------------------------------------------------------------------

    def _compute_intraday_lows(self) -> Dict[str, Optional[float]]:
        """
        Compute intraday low(s) for `self.date`.

        This is used for TRAC position stop setting. It intentionally does NOT rely on
        `low_water_mark` because:
        - `low_water_mark` only updates after open is locked
        - you might start the monitor after the true low already printed

        Returns:
            Dict with:
              - full_day_low: min(low) across all intraday bars available for the date
              - rth_low: min(low) for 09:30–16:00 ET (if bars available)
              - bars: number of bars used (if known)
              - source: 'trillium' | 'polygon' | 'trade_df' | 'fallback'
        """
        # Fast path: if we already have a populated trade_df, use it (still may be partial-day).
        try:
            if hasattr(self, 'trade_df') and isinstance(self.trade_df, pd.DataFrame) and not self.trade_df.empty:
                if 'low' in self.trade_df.columns:
                    full_low = float(pd.to_numeric(self.trade_df['low'], errors='coerce').min())
                    rth_low = None
                    try:
                        rth = self.trade_df.between_time("09:30", "16:00")
                        if not rth.empty:
                            rth_low = float(pd.to_numeric(rth['low'], errors='coerce').min())
                    except Exception:
                        rth_low = None
                    if not np.isnan(full_low):
                        return {
                            'full_day_low': full_low,
                            'rth_low': rth_low,
                            'bars': int(len(self.trade_df)),
                            'source': 'trade_df',
                        }
        except Exception:
            pass

        df = None
        source = 'fallback'

        # Prefer Trillium if available (captures pre/post and matches the live stream).
        try:
            import data_queries.trillium_queries as trlm
            df = trlm.get_intraday(self.ticker, self.date, "bar-1min")
            source = 'trillium'
        except Exception:
            df = None

        # Fallback to Polygon intraday minute bars.
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            try:
                from data_queries.polygon_queries import get_intraday as _poly_intraday
                df = _poly_intraday(self.ticker, self.date, multiplier=1, timespan='minute')
                source = 'polygon'
            except Exception:
                df = None

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            # Last-resort fallback: use current tracked low_water_mark if available.
            lw = getattr(self, 'low_water_mark', None)
            try:
                lw = float(lw) if lw is not None else None
            except Exception:
                lw = None
            return {'full_day_low': lw, 'rth_low': lw, 'bars': None, 'source': source}

        # Normalize the low column
        low_col = None
        if 'low' in df.columns:
            low_col = 'low'
        else:
            for c in df.columns:
                if str(c).lower() == 'low':
                    low_col = c
                    break

        if low_col is None:
            return {'full_day_low': None, 'rth_low': None, 'bars': int(len(df)), 'source': source}

        lows = pd.to_numeric(df[low_col], errors='coerce')
        full_low = float(lows.min()) if not lows.empty else None

        rth_low = None
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                rth = df.between_time("09:30", "16:00")
                if not rth.empty:
                    rth_low = float(pd.to_numeric(rth[low_col], errors='coerce').min())
        except Exception:
            rth_low = None

        # Clean NaNs
        if full_low is not None and (isinstance(full_low, float) and np.isnan(full_low)):
            full_low = None
        if rth_low is not None and (isinstance(rth_low, float) and np.isnan(rth_low)):
            rth_low = None

        return {
            'full_day_low': full_low,
            'rth_low': rth_low,
            'bars': int(len(df)),
            'source': source,
        }

    def _refresh_position(self, current_price: float = None):
        """
        Refresh position data from TRAC scraper.
        Updates self.position and self.ctx.position with current P&L.
        """
        if not self.trac_scraper:
            return

        try:
            self.logger.info(f'TRAC: Checking for {self.ticker} position...')
            pos_data = self.trac_scraper.get_position(self.ticker, refresh=True)
            if pos_data:
                was_new_position = not self.position.has_position
                self.position.has_position = True
                self.position.shares = pos_data.get('shares', 0)
                self.position.side = pos_data.get('side', '')
                self.position.avg_price = pos_data.get('avg_price')
                self.position.bp_used = pos_data.get('bp_used')

                self.logger.info(f'TRAC: Found position — {self.position.side} {self.position.shares} shares @ ${self.position.avg_price}')

                # Set stop price at true low-of-day when we don't yet have one.
                # (Do not rely on low_water_mark — it can miss earlier lows.)
                if self.position.stop_price is None:
                    lows = self._compute_intraday_lows()
                    stop = lows.get('full_day_low') or lows.get('rth_low')
                    if stop is None:
                        stop = getattr(self, 'low_water_mark', None) or self.open_price
                    try:
                        stop = float(stop)
                    except Exception:
                        stop = None

                    if stop is not None:
                        self.position.stop_price = stop
                        self.position.stopped_out = False

                        rth_low = lows.get('rth_low')
                        full_low = lows.get('full_day_low')
                        src = lows.get('source')
                        bars = lows.get('bars')
                        detail = f'source={src}'
                        if bars is not None:
                            detail += f', bars={bars}'
                        if full_low is not None and rth_low is not None:
                            detail += f', full_low={full_low:.2f}, rth_low={rth_low:.2f}'
                        elif full_low is not None:
                            detail += f', full_low={full_low:.2f}'
                        elif rth_low is not None:
                            detail += f', rth_low={rth_low:.2f}'

                        self.logger.info(f'STOP SET: ${self.position.stop_price:.2f} (true LOD; {detail})')
                        self._alert(f'Stop set at low of day: {self.position.stop_price:.2f}', priority=1)

                # Calculate P&L if we have avg_price and current_price
                if self.position.avg_price and current_price:
                    if self.position.side == 'long':
                        self.position.pnl_per_share = current_price - self.position.avg_price
                    else:
                        self.position.pnl_per_share = self.position.avg_price - current_price
                    self.position.pnl_total = self.position.pnl_per_share * self.position.shares
                    self.position.pnl_atr = self.position.pnl_per_share / self.atr if self.atr else 0

            else:
                self.logger.info(f'TRAC: No {self.ticker} position found')
                # No position found — may have been closed
                if self.position.has_position:
                    self.logger.info(f'Position in {self.ticker} appears to be closed')
                    self._alert(f'{self.ticker} position closed', priority=1)
                self.position.has_position = False
                self.position.shares = 0
                self.position.stop_price = None
                self.position.stopped_out = False

            # Sync to context (if ctx exists - may not during init)
            if hasattr(self, 'ctx'):
                self.ctx.position = self.position

        except Exception as e:
            self.logger.warning(f'Failed to refresh position: {e}')

    def _check_position_alerts(self, current_price: float, bar_low: float = None):
        """
        Check for position-specific alerts based on P&L thresholds and stop price.
        Only fires if we have a position in this ticker.

        Args:
            current_price: Current bar close price
            bar_low: Current bar low (for stop checking)
        """
        if not self.position.has_position or not self.position.avg_price:
            return

        # Use bar_low for stop check, fall back to current_price
        check_price = bar_low if bar_low is not None else current_price

        # Check stop-out (LOD stop for long positions)
        if (self.position.stop_price is not None
                and not self.position.stopped_out
                and self.position.side == 'long'
                and check_price < self.position.stop_price):
            self.position.stopped_out = True
            stop_loss = (self.position.stop_price - self.position.avg_price) * self.position.shares
            self.logger.info(f'STOPPED OUT: Price ${check_price:.2f} broke stop ${self.position.stop_price:.2f}')
            self._alert(f'STOPPED OUT! Price broke stop at {self.position.stop_price:.2f}. Estimated loss ${stop_loss:,.0f}', priority=1)

        # Update P&L with current price
        if self.position.side == 'long':
            pnl_per_share = current_price - self.position.avg_price
        else:
            pnl_per_share = self.position.avg_price - current_price
        pnl_atr = pnl_per_share / self.atr if self.atr else 0
        pnl_total = pnl_per_share * self.position.shares

        # Store for reference
        self.position.pnl_per_share = pnl_per_share
        self.position.pnl_total = pnl_total
        self.position.pnl_atr = pnl_atr

        # P&L milestone alerts (in ATR units)
        atr_milestones = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5]
        for milestone in atr_milestones:
            key = f'pnl_{milestone:.1f}_atr'
            if key not in self._position_alerts_fired:
                self._position_alerts_fired[key] = False

            if not self._position_alerts_fired[key]:
                if milestone > 0 and pnl_atr >= milestone:
                    self._position_alerts_fired[key] = True
                    self._alert(f'Position P&L hit plus {milestone:.1f} ATR — total ${pnl_total:+,.0f}', priority=1)
                elif milestone < 0 and pnl_atr <= milestone:
                    self._position_alerts_fired[key] = True
                    self._alert(f'Position P&L hit minus {abs(milestone):.1f} ATR — total ${pnl_total:+,.0f}', priority=1)

    # -----------------------------------------------------------------------
    # Open price locking — capture official open and rebuild targets
    # -----------------------------------------------------------------------

    def _lock_open_price(self, official_open: float):
        """
        Lock in the official market open price and rebuild all ATR-based targets.
        Called once when first bar at/after 9:30 AM is received.
        """
        old_price = self.open_price
        self.open_price = official_open
        self.open_locked = True
        self.ctx.open_price = official_open
        self.ctx.open_locked = True

        # Recalculate gap percentage with official open
        self.gap_pct = (self.open_price - self.prior_close) / self.prior_close if self.prior_close != 0 else 0
        self.ctx.gap_pct = self.gap_pct

        # Recalculate exhaustion gap flag
        self.has_exhaustion_gap = (self.gap_pct <= -0.05 and self.consecutive_down_days >= 3)
        self.ctx.has_exhaustion_gap = self.has_exhaustion_gap

        # Clear and rebuild alert levels
        self.target_alerts.clear()
        self.drawdown_alerts.clear()
        self._build_alert_levels()

        # Reset watermarks to official open
        self.high_water_mark = self.open_price
        self.low_water_mark = self.open_price

        # Log and announce
        self.logger.info(f'OPEN PRICE LOCKED: ${self.open_price:.2f} (was ${old_price:.2f} premarket)')
        self._alert(f'Market open. Official open price locked at {self.open_price:.2f}. Targets updated.', priority=1)

        # Reprint the updated level table
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

                    # Checklist threshold levels can be monitored premarket (independent of official open)
                    self._check_checklist_price_alerts(low)

                    # Check for open price lock at market open (9:30 AM)
                    if not self.open_locked and hasattr(series.name, 'time'):
                        bar_time = series.name.time()
                        if bar_time >= MARKET_OPEN:
                            # Use the open of the first bar at/after 9:30 as official open
                            official_open = series['open']
                            self._lock_open_price(official_open)

                    # Only track watermarks and check alerts after open is locked
                    if self.open_locked:
                        # Update watermarks
                        if high > self.high_water_mark:
                            self.high_water_mark = high
                        if low < self.low_water_mark:
                            self.low_water_mark = low

                        # Check alerts
                        self._check_target_alerts(high)
                        self._check_drawdown_alerts(low)

                        # Check position P&L and stop alerts (if TRAC enabled)
                        if self.trac_scraper and self.position.has_position:
                            self._check_position_alerts(price, bar_low=low)

            # Time alerts run on every loop regardless of new data
            self._check_time_alerts()

            # Periodically refresh position data (every bar = 5 seconds)
            if self.trac_scraper:
                current = self.trade_df.iloc[-1]['close'] if not self.trade_df.empty else self.open_price
                self._refresh_position(current_price=current)

            self.new_data_event.clear()

    def process_incoming_data(self, bar_data):
        """Called by BounceDataAdapter with each 5s bar dict."""
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

def bounce_main(ticker: str, cap: str = None, dry_run: bool = False, enable_trac: bool = False):
    """Entry point — build manager, optionally attach live stream, and run."""

    # Initialize TRAC scraper if requested
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

    manager = BounceTradeManager(ticker=ticker, cap=cap, trac_scraper=trac_scraper)

    if dry_run:
        logger.info('DRY RUN — simulating a few bars then exiting')
        _simulate_dry_run(manager)
        if trac_scraper:
            trac_scraper.close()
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
    finally:
        if trac_scraper:
            trac_scraper.close()


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
            manager._check_checklist_price_alerts(series['low'])

        manager.new_data_event.clear()
        _time.sleep(0.5)

    logger.info('Dry run complete — all levels and setup alerts verified')


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # If CLI args provided, use them (original behavior)
    if len(sys.argv) >= 2:
        _ticker = sys.argv[1].upper()
        _cap = None
        _dry_run = '--dry-run' in sys.argv
        _enable_trac = '--trac' in sys.argv

        if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
            _cap = sys.argv[2]

        bounce_main(_ticker, cap=_cap, dry_run=_dry_run, enable_trac=_enable_trac)
    else:
        # Interactive mode — prompt for inputs (useful for PyCharm Run)
        print('=== Bounce Trader — Interactive Mode ===')
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

        bounce_main(_ticker, cap=_cap, dry_run=_dry_run, enable_trac=_enable_trac)
