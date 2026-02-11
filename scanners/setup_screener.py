"""
Stock Setup Screener

Screens a universe of tickers for two setup types:
1. Parabolic Short - stocks extended to the upside, ripe for reversal
2. Capitulation Bounce - stocks capitulating after extended selloff, ripe for bounce

Designed to handle 1000+ tickers efficiently using a two-phase approach:
  Phase 1 (quick_scan): Uses Polygon snapshots to get current-day data for ALL
      tickers in 1 API call. Pre-filters to ~50-100 extreme movers based on
      today's change %, gap %, and volume expansion.
  Phase 2 (screen_universe): Fetches full 310-day history ONLY for pre-filtered
      candidates. Runs concurrently with rate limiting.

Pre-trade screening criteria:
  Parabolic Short (5 of 6 reversal criteria - excludes reversal_pct):
    1. % above 9EMA (cap-adjusted)
    2. Prior day range vs ATR (range expansion)
    3. RVOL (volume expansion)
    4. Consecutive up days
    5. Gap up %

  Capitulation Bounce (6 pre-trade criteria - excludes bounce_pct):
    1. Selloff depth (total % decline)
    2. Consecutive down days
    3. % off 30-day high
    4. Gap down %
    5. Prior day range vs ATR (range expansion)
    6. Volume signal (prior day RVOL or premarket RVOL)

Usage:
    from scanners.setup_screener import SetupScreener

    screener = SetupScreener()

    # Full pipeline: fetch universe -> quick scan -> full screen
    results = screener.scan('2025-01-15')

    # Or step by step:
    universe = screener.fetch_universe(min_market_cap=2e9)  # medium + large cap
    candidates = screener.quick_scan(tickers=list(universe.keys()))
    results = screener.screen_universe(candidates, '2025-01-15', ticker_caps=universe)

    # Or with your own ticker list:
    my_tickers = ['AAPL', 'TSLA', 'NVDA', ...]  # 1000+ tickers
    candidates = screener.quick_scan(tickers=my_tickers)
    results = screener.screen_universe(candidates, '2025-01-15', cap='Large')

    # Or screen with pre-computed metrics (no API calls):
    result = screener.screen_ticker_from_metrics('NVDA', my_metrics_dict, cap='Large')
"""

import pandas as pd
import numpy as np
import logging
import time
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tabulate import tabulate
from polygon.rest import RESTClient

from data_queries.polygon_queries import get_levels_data
from analyzers.reversal_scorer import ReversalScorer, CAP_THRESHOLDS as REVERSAL_THRESHOLDS
from analyzers.bounce_scorer import BouncePretrade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Polygon client (reuses same key as polygon_queries.py)
POLYGON_API_KEY = "pcwUY7TnSF66nYAPIBCApPMyVrXTckJY"

# Market cap classification boundaries
CAP_BOUNDARIES = [
    ('Large', 10e9),    # >= $10B
    ('Medium', 2e9),    # >= $2B
    ('Small', 250e6),   # >= $250M
    ('Micro', 0),       # < $250M
]


def classify_market_cap(market_cap: float) -> str:
    """Classify a market cap value into a cap category."""
    if market_cap is None:
        return 'Medium'  # default
    for label, boundary in CAP_BOUNDARIES:
        if market_cap >= boundary:
            return label
    return 'Micro'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScreenResult:
    """Result of screening a single ticker."""
    ticker: str
    cap: str

    # Parabolic short screening
    parabolic_score: int
    parabolic_max_score: int
    parabolic_grade: str
    parabolic_recommendation: str
    parabolic_criteria: Dict

    # Capitulation bounce screening
    bounce_score: int
    bounce_max_score: int
    bounce_setup_type: str
    bounce_recommendation: str
    bounce_criteria: Dict

    # Key metrics for display
    metrics: Dict

    # Flags
    is_parabolic_candidate: bool
    is_bounce_candidate: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SetupScreener
# ---------------------------------------------------------------------------

class SetupScreener:
    """
    Screens tickers for parabolic short and capitulation bounce setups.

    Handles 1000+ tickers via two-phase approach:
      1. quick_scan() - snapshot-based pre-filter (1 API call for all tickers)
      2. screen_universe() - concurrent full screening (1 API call per candidate)

    Args:
        api_key: Polygon.io API key
        max_workers: Max concurrent threads for full screening
        rate_limit: Max Polygon API calls per second
    """

    def __init__(self, api_key: str = POLYGON_API_KEY,
                 max_workers: int = 5, rate_limit: float = 5.0):
        self.reversal_scorer = ReversalScorer()
        self.bounce_checker = BouncePretrade()
        self.poly_client = RESTClient(api_key=api_key)
        self.max_workers = max_workers
        self.rate_limit = rate_limit

    # ------------------------------------------------------------------
    # Phase 0: Fetch ticker universe
    # ------------------------------------------------------------------

    def fetch_universe(self, min_market_cap: float = 2e9,
                       max_market_cap: float = None,
                       limit: int = 3000) -> Dict[str, str]:
        """
        Fetch active common stock tickers from Polygon, filtered by market cap.

        Uses Polygon's reference/tickers endpoint. If market_cap data is not
        available (depends on Polygon plan), falls back to returning all
        active common stocks and classifying them as 'Medium' by default.

        Args:
            min_market_cap: Minimum market cap in dollars (default $2B = medium+)
            max_market_cap: Maximum market cap (None = no upper limit)
            limit: Max tickers to return

        Returns:
            Dict mapping ticker -> cap category ('Large', 'Medium', 'Small', 'Micro')
        """
        universe = {}
        count = 0
        has_market_cap_data = False

        print(f"Fetching ticker universe from Polygon (min market cap: ${min_market_cap/1e9:.1f}B)...")

        try:
            for t in self.poly_client.list_tickers(
                market='stocks', type='CS', active=True, limit=1000, order='asc'
            ):
                mcap = getattr(t, 'market_cap', None)

                if mcap is not None:
                    has_market_cap_data = True
                    if min_market_cap and mcap < min_market_cap:
                        continue
                    if max_market_cap and mcap > max_market_cap:
                        continue
                    universe[t.ticker] = classify_market_cap(mcap)
                else:
                    # No market cap data - include all and classify later
                    universe[t.ticker] = 'Medium'

                count += 1
                if count >= limit:
                    break

        except Exception as e:
            logging.error(f"Error fetching universe from Polygon: {e}")

        if not has_market_cap_data and universe:
            print(f"  Note: Market cap data not available on your Polygon plan.")
            print(f"  Returning {len(universe)} tickers classified as 'Medium' by default.")
            print(f"  Use quick_scan() to further filter by dollar volume.")
        else:
            print(f"  Found {len(universe)} tickers matching market cap criteria.")

        return universe

    # ------------------------------------------------------------------
    # Phase 1: Quick scan (snapshot-based pre-filter)
    # ------------------------------------------------------------------

    def quick_scan(self, tickers: List[str] = None,
                   min_up_pct: float = 0.03, min_down_pct: float = -0.03,
                   min_dollar_vol: float = 5e6,
                   setup_type: str = 'both') -> List[str]:
        """
        Use Polygon snapshots to quickly identify extreme movers from a large
        ticker universe. Returns only tickers worth full-screening.

        This is the key to handling 1000+ tickers efficiently: 1 API call
        returns current-day data for ALL tickers. We pre-filter based on
        today's price action and volume, reducing 1000+ tickers to ~50-100
        candidates for full historical screening.

        Pre-filter criteria:
          Parabolic candidates: today's change >= min_up_pct OR gap up >= min_up_pct
          Bounce candidates: today's change <= min_down_pct OR gap down <= min_down_pct
          Both: dollar volume >= min_dollar_vol (filters out illiquid names)

        Args:
            tickers: Restrict to these tickers (None = scan ALL tickers)
            min_up_pct: Min % move to flag as parabolic candidate (default 3%)
            min_down_pct: Min % drop to flag as bounce candidate (default -3%)
            min_dollar_vol: Min daily dollar volume (default $5M)
            setup_type: 'parabolic', 'bounce', or 'both'

        Returns:
            List of ticker symbols that passed the pre-filter
        """
        ticker_set = set(tickers) if tickers else None
        candidates = []
        total_scanned = 0
        parabolic_count = 0
        bounce_count = 0

        print(f"\nQuick scanning {'all tickers' if not tickers else f'{len(tickers)} tickers'} "
              f"via Polygon snapshots...")
        print(f"  Parabolic filter: change >= {min_up_pct*100:.0f}% or gap >= {min_up_pct*100:.0f}%")
        print(f"  Bounce filter: change <= {min_down_pct*100:.0f}% or gap <= {min_down_pct*100:.0f}%")
        print(f"  Min dollar volume: ${min_dollar_vol/1e6:.0f}M")

        try:
            for snap in self.poly_client.get_snapshot_all("stocks"):
                ticker = snap.ticker

                # Filter to requested tickers
                if ticker_set and ticker not in ticker_set:
                    continue

                total_scanned += 1

                try:
                    day = snap.day
                    prev = snap.prevDay

                    if not day or not prev or not prev.c or prev.c <= 0:
                        continue

                    price = day.c if day.c else day.o
                    if not price or price <= 0:
                        continue

                    # Dollar volume filter
                    dollar_vol = (price * day.v) if day.v else 0
                    if dollar_vol < min_dollar_vol:
                        continue

                    # Compute quick metrics
                    change_pct = (snap.todaysChangePerc / 100
                                  if snap.todaysChangePerc else 0)
                    gap_pct = (day.o - prev.c) / prev.c if day.o and prev.c else 0

                    is_parabolic = (change_pct >= min_up_pct or gap_pct >= min_up_pct)
                    is_bounce = (change_pct <= min_down_pct or gap_pct <= min_down_pct)

                    if setup_type == 'parabolic' and not is_parabolic:
                        continue
                    elif setup_type == 'bounce' and not is_bounce:
                        continue
                    elif setup_type == 'both' and not (is_parabolic or is_bounce):
                        continue

                    candidates.append(ticker)
                    if is_parabolic:
                        parabolic_count += 1
                    if is_bounce:
                        bounce_count += 1

                except (AttributeError, TypeError):
                    continue

        except Exception as e:
            logging.error(f"Snapshot API failed: {e}")
            print(f"\n  Snapshot API not available ({e}).")
            print(f"  Falling back to grouped daily bars...")
            return self._quick_scan_fallback(
                tickers, min_up_pct, min_down_pct, min_dollar_vol, setup_type
            )

        print(f"  Scanned {total_scanned} tickers -> "
              f"{len(candidates)} candidates "
              f"({parabolic_count} parabolic, {bounce_count} bounce)")

        return candidates

    def _quick_scan_fallback(self, tickers, min_up_pct, min_down_pct,
                             min_dollar_vol, setup_type):
        """
        Fallback pre-filter using grouped daily bars when snapshots aren't available.

        Uses 2 API calls (today + yesterday grouped daily) to compute change %
        and gap % for all tickers.
        """
        today = datetime.now()
        if today.weekday() == 5:
            today -= timedelta(days=1)
        elif today.weekday() == 6:
            today -= timedelta(days=2)

        date_str = today.strftime('%Y-%m-%d')
        prev_str = (today - timedelta(days=5)).strftime('%Y-%m-%d')

        ticker_set = set(tickers) if tickers else None
        candidates = []

        try:
            # Get today's grouped daily (all tickers, 1 API call)
            today_aggs = {}
            for agg in self.poly_client.get_grouped_daily_aggs(date_str):
                if ticker_set and agg.ticker not in ticker_set:
                    continue
                today_aggs[agg.ticker] = agg

            # Get previous day's grouped daily (1 API call)
            prev_aggs = {}
            for agg in self.poly_client.get_grouped_daily_aggs(
                (today - timedelta(days=1)).strftime('%Y-%m-%d')
            ):
                prev_aggs[agg.ticker] = agg

            for ticker, agg in today_aggs.items():
                try:
                    price = agg.close if agg.close else agg.open
                    if not price or price <= 0:
                        continue

                    dollar_vol = price * agg.volume if agg.volume else 0
                    if dollar_vol < min_dollar_vol:
                        continue

                    prev = prev_aggs.get(ticker)
                    if not prev or not prev.close or prev.close <= 0:
                        continue

                    change_pct = (agg.close - prev.close) / prev.close
                    gap_pct = (agg.open - prev.close) / prev.close

                    is_parabolic = (change_pct >= min_up_pct or gap_pct >= min_up_pct)
                    is_bounce = (change_pct <= min_down_pct or gap_pct <= min_down_pct)

                    if setup_type == 'parabolic' and not is_parabolic:
                        continue
                    elif setup_type == 'bounce' and not is_bounce:
                        continue
                    elif setup_type == 'both' and not (is_parabolic or is_bounce):
                        continue

                    candidates.append(ticker)

                except (AttributeError, TypeError):
                    continue

            print(f"  Grouped daily fallback: {len(candidates)} candidates from "
                  f"{len(today_aggs)} tickers")

        except Exception as e:
            logging.error(f"Grouped daily fallback also failed: {e}")
            print(f"  Fallback failed: {e}")
            print(f"  Returning original ticker list for full screening.")
            return list(tickers) if tickers else []

        return candidates

    # ------------------------------------------------------------------
    # Phase 2: Full metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self, ticker: str, date: str) -> Dict:
        """
        Compute all screening metrics from daily OHLCV data.

        Single API call to get_levels_data (310 calendar days), then all
        metrics are computed locally:
          - 9 EMA, 50/200 SMA and % distance from each
          - 14-day ATR and prior day range as multiple of ATR
          - Consecutive up/down days
          - RVOL (prior day volume / 20-day ADV)
          - Gap % (today open vs prior close)
          - % changes over 3/15/30/90/120 day windows
          - % off 30-day high and 52-week high
          - Selloff depth
          - Bollinger Band positioning

        Args:
            ticker: Stock ticker symbol
            date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary of computed metrics, or empty dict if no data
        """
        metrics = {}

        levels = get_levels_data(ticker, date, 310, 1, 'day')
        if levels is None or levels.empty:
            return {}

        # Detect whether today's bar exists (partial bar during market hours)
        has_today_bar = False
        try:
            last_bar_date = levels.index[-1].date()
            has_today_bar = (last_bar_date == pd.to_datetime(date).date())
        except Exception:
            has_today_bar = False

        # hist = completed bars only (exclude partial today bar)
        hist = levels.iloc[:-1] if has_today_bar and len(levels) > 1 else levels

        if len(hist) < 5:
            return {}

        # Reference prices
        current_close = levels.iloc[-1]['close']
        today_open = levels.iloc[-1]['open'] if has_today_bar else None
        prior_close = hist.iloc[-1]['close']
        metrics['current_price'] = current_close
        metrics['prior_close'] = prior_close

        closes = hist['close']

        # --- Moving averages ---
        if len(closes) >= 9:
            ema_9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
            metrics['pct_from_9ema'] = (current_close - ema_9) / ema_9

        if len(closes) >= 50:
            sma_50 = closes.rolling(50).mean().iloc[-1]
            metrics['pct_from_50mav'] = (current_close - sma_50) / sma_50

        if len(closes) >= 200:
            sma_200 = closes.rolling(200).mean().iloc[-1]
            metrics['pct_from_200mav'] = (current_close - sma_200) / sma_200

        # --- ATR (14-day) ---
        if len(hist) >= 2:
            hl = hist['high'] - hist['low']
            hpc = abs(hist['high'] - hist['close'].shift(1))
            lpc = abs(hist['low'] - hist['close'].shift(1))
            tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

            atr_window = min(14, len(tr))
            atr = tr.rolling(window=atr_window, min_periods=1).mean().iloc[-1]
            metrics['atr'] = atr
            metrics['atr_pct'] = atr / current_close if current_close > 0 else 0

            prior_range = hist.iloc[-1]['high'] - hist.iloc[-1]['low']
            metrics['prior_day_range_atr'] = prior_range / atr if atr > 0 else 0

        # --- Consecutive up days ---
        consecutive_up = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] > hist.iloc[i - 1]['close']:
                consecutive_up += 1
            else:
                break
        metrics['consecutive_up_days'] = consecutive_up

        # --- Consecutive down days ---
        consecutive_down = 0
        for i in range(len(hist) - 1, 0, -1):
            if hist.iloc[i]['close'] < hist.iloc[i - 1]['close']:
                consecutive_down += 1
            else:
                break
        metrics['consecutive_down_days'] = consecutive_down

        # --- Volume expansion (RVOL) ---
        adv_window = min(20, len(hist))
        adv = hist['volume'].rolling(window=adv_window, min_periods=1).mean().iloc[-1]
        prior_day_vol = hist.iloc[-1]['volume']
        metrics['avg_daily_vol'] = adv
        metrics['rvol_score'] = prior_day_vol / adv if adv > 0 else 0
        metrics['prior_day_rvol'] = metrics['rvol_score']

        # Premarket RVOL not available from daily bars
        metrics['premarket_rvol'] = None

        # --- Gap % ---
        if today_open and prior_close and prior_close > 0:
            metrics['gap_pct'] = (today_open - prior_close) / prior_close
        else:
            metrics['gap_pct'] = 0.0

        # --- Percent changes over lookback windows ---
        for days, key in [(3, 'pct_change_3'), (15, 'pct_change_15'),
                          (30, 'pct_change_30'), (90, 'pct_change_90'),
                          (120, 'pct_change_120')]:
            if len(hist) >= days:
                old_close = hist.iloc[-days]['close']
                if old_close > 0:
                    metrics[key] = (current_close - old_close) / old_close

        # --- Percent off 30-day high ---
        window_30 = hist.iloc[-30:] if len(hist) >= 30 else hist
        high_30d = window_30['high'].max()
        metrics['pct_off_30d_high'] = (current_close - high_30d) / high_30d if high_30d > 0 else 0

        # --- Percent off 52-week high ---
        high_52 = hist['high'].max()
        metrics['pct_off_52wk_high'] = (current_close - high_52) / high_52 if high_52 > 0 else 0

        # --- Selloff total pct (for bounce screening) ---
        if consecutive_down > 0:
            selloff_start_idx = len(hist) - consecutive_down
            if selloff_start_idx >= 0:
                first_open = hist.iloc[selloff_start_idx]['open']
                metrics['selloff_total_pct'] = (
                    (current_close - first_open) / first_open if first_open > 0 else 0
                )
            else:
                metrics['selloff_total_pct'] = 0.0
        else:
            metrics['selloff_total_pct'] = 0.0

        # --- Bollinger Bands (20-day SMA +/- 2 std) ---
        if len(hist) >= 20:
            last_20 = closes.values[-20:]
            sma20 = np.mean(last_20)
            std20 = np.std(last_20)
            lower_band = sma20 - 2 * std20
            upper_band = sma20 + 2 * std20
            metrics['closed_outside_lower_band'] = bool(current_close < lower_band)
            metrics['closed_outside_upper_band'] = bool(current_close > upper_band)
            metrics['bollinger_width'] = (upper_band - lower_band) / sma20 if sma20 > 0 else 0

        # --- Prior day close positioning ---
        prior_range = hist.iloc[-1]['high'] - hist.iloc[-1]['low']
        if prior_range > 0:
            metrics['prior_day_close_vs_low_pct'] = (
                (hist.iloc[-1]['close'] - hist.iloc[-1]['low']) / prior_range
            )
            metrics['prior_day_close_vs_high_pct'] = (
                (hist.iloc[-1]['high'] - hist.iloc[-1]['close']) / prior_range
            )

        return metrics

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_parabolic_short(self, metrics: Dict, cap: str) -> Tuple[int, int, str, str, Dict]:
        """
        Score pre-trade parabolic short criteria (5 of 6 reversal criteria,
        excludes reversal_pct which is the trade result).
        """
        thresholds = self.reversal_scorer._get_thresholds(cap)
        passed = []
        failed = []
        criteria_details = {}

        checks = [
            ('pct_from_9ema', metrics.get('pct_from_9ema'),
             thresholds.pct_from_9ema, 'gte', 'Extended above 9EMA'),
            ('prior_day_range_atr', metrics.get('prior_day_range_atr'),
             thresholds.prior_day_range_atr, 'gte', 'Range expansion (ATR)'),
            ('rvol_score', metrics.get('rvol_score'),
             thresholds.rvol_score, 'gte', 'Volume expansion (RVOL)'),
            ('consecutive_up_days', metrics.get('consecutive_up_days'),
             thresholds.consecutive_up_days, 'gte', 'Consecutive up days'),
            ('gap_pct', metrics.get('gap_pct'),
             thresholds.gap_pct, 'gte', 'Euphoric gap up'),
        ]

        for name, actual, threshold, direction, description in checks:
            if actual is None or (isinstance(actual, float) and pd.isna(actual)):
                check_passed = False
            elif direction == 'gte':
                check_passed = actual >= threshold
            else:
                check_passed = actual <= threshold

            if check_passed:
                passed.append(name)
            else:
                failed.append(name)

            criteria_details[name] = {
                'name': description,
                'threshold': threshold,
                'actual': actual,
                'passed': check_passed,
            }

        score = len(passed)
        max_score = 5

        if score == 5:
            grade, rec = 'A+', 'GO'
        elif score == 4:
            grade, rec = 'A', 'GO'
        elif score == 3:
            grade, rec = 'B', 'CAUTION'
        else:
            grade, rec = 'F', 'NO-GO'

        return score, max_score, grade, rec, criteria_details

    def _score_capitulation_bounce(self, ticker: str, metrics: Dict,
                                   cap: str) -> Tuple[int, int, str, str, Dict]:
        """
        Score pre-trade capitulation bounce criteria using BouncePretrade.

        Auto-classifies as weakstock/strongstock based on 200-day MA positioning.
        """
        result = self.bounce_checker.validate(ticker, metrics, cap=cap)

        criteria_details = {}
        for item in result.items:
            criteria_details[item.name] = {
                'name': item.description,
                'threshold': item.threshold,
                'actual': item.actual,
                'passed': item.passed,
            }

        return (result.score, result.max_score, result.setup_type,
                result.recommendation, criteria_details)

    # ------------------------------------------------------------------
    # Single ticker screening
    # ------------------------------------------------------------------

    def screen_ticker(self, ticker: str, date: str,
                      cap: str = 'Medium') -> ScreenResult:
        """
        Screen a single ticker for both parabolic short and capitulation bounce.

        Makes 1 API call (get_levels_data) to fetch 310 days of daily bars,
        then computes all metrics and scores locally.
        """
        try:
            metrics = self.compute_metrics(ticker, date)
            if not metrics:
                return self._empty_result(ticker, cap,
                                          error=f'No data available for {ticker}')
            return self._score_from_metrics(ticker, metrics, cap)

        except Exception as e:
            logging.error(f"Error screening {ticker}: {e}")
            return self._empty_result(ticker, cap, error=str(e))

    def screen_ticker_from_metrics(self, ticker: str, metrics: Dict,
                                   cap: str = 'Medium') -> ScreenResult:
        """
        Screen a ticker using pre-computed metrics (no API calls).

        Use this when you already have the metrics from your own data source.
        """
        try:
            return self._score_from_metrics(ticker, metrics, cap)
        except Exception as e:
            logging.error(f"Error scoring {ticker}: {e}")
            return self._empty_result(ticker, cap, error=str(e))

    def _score_from_metrics(self, ticker: str, metrics: Dict,
                            cap: str) -> ScreenResult:
        """Score a ticker from its metrics dict."""
        p_score, p_max, p_grade, p_rec, p_criteria = (
            self._score_parabolic_short(metrics, cap)
        )
        b_score, b_max, b_setup, b_rec, b_criteria = (
            self._score_capitulation_bounce(ticker, metrics, cap)
        )

        return ScreenResult(
            ticker=ticker, cap=cap,
            parabolic_score=p_score, parabolic_max_score=p_max,
            parabolic_grade=p_grade, parabolic_recommendation=p_rec,
            parabolic_criteria=p_criteria,
            bounce_score=b_score, bounce_max_score=b_max,
            bounce_setup_type=b_setup, bounce_recommendation=b_rec,
            bounce_criteria=b_criteria,
            metrics=metrics,
            is_parabolic_candidate=(p_score >= 3),
            is_bounce_candidate=(b_score >= 4),
        )

    def _empty_result(self, ticker: str, cap: str,
                      error: str = None) -> ScreenResult:
        """Return an empty ScreenResult for failed tickers."""
        return ScreenResult(
            ticker=ticker, cap=cap,
            parabolic_score=0, parabolic_max_score=5,
            parabolic_grade='F', parabolic_recommendation='NO-GO',
            parabolic_criteria={},
            bounce_score=0, bounce_max_score=6,
            bounce_setup_type='', bounce_recommendation='NO-GO',
            bounce_criteria={},
            metrics={},
            is_parabolic_candidate=False, is_bounce_candidate=False,
            error=error,
        )

    # ------------------------------------------------------------------
    # Universe screening (concurrent with rate limiting)
    # ------------------------------------------------------------------

    def screen_universe(self, tickers: List[str], date: str,
                        cap: str = 'Medium',
                        ticker_caps: Optional[Dict[str, str]] = None,
                        max_workers: int = None) -> List[ScreenResult]:
        """
        Screen a list of tickers concurrently with rate limiting.

        Each ticker requires 1 API call (get_levels_data). Concurrent execution
        with rate limiting keeps throughput high without hitting Polygon limits.

        Args:
            tickers: List of ticker symbols to screen
            date: Date to screen (YYYY-MM-DD)
            cap: Default market cap category for all tickers
            ticker_caps: Optional dict mapping ticker -> cap to override default
            max_workers: Max concurrent threads (default: self.max_workers)

        Returns:
            List of ScreenResult objects
        """
        workers = max_workers or self.max_workers
        results = []
        total = len(tickers)
        completed = 0
        interval = 1.0 / self.rate_limit

        print(f"\nFull screening {total} tickers on {date}...")
        print(f"  Concurrency: {workers} workers, rate limit: {self.rate_limit} req/s")
        print("-" * 60)

        def _screen_one(ticker):
            ticker_cap = ticker_caps.get(ticker, cap) if ticker_caps else cap
            return self.screen_ticker(ticker, date, ticker_cap)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with rate limiting on submission
            futures = {}
            for ticker in tickers:
                future = executor.submit(_screen_one, ticker)
                futures[future] = ticker
                time.sleep(interval)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)

                    if result.error:
                        print(f"  [{completed}/{total}] {ticker}: ERROR - {result.error}")
                    elif result.is_parabolic_candidate or result.is_bounce_candidate:
                        flags = []
                        if result.is_parabolic_candidate:
                            flags.append(f"PARABOLIC {result.parabolic_score}/{result.parabolic_max_score}")
                        if result.is_bounce_candidate:
                            flags.append(f"BOUNCE {result.bounce_score}/{result.bounce_max_score}")
                        print(f"  [{completed}/{total}] {ticker}: ** {' | '.join(flags)} **")
                    else:
                        print(f"  [{completed}/{total}] {ticker}: no setup")
                except Exception as e:
                    results.append(self._empty_result(ticker, cap, error=str(e)))
                    print(f"  [{completed}/{total}] {ticker}: EXCEPTION - {e}")

        return results

    # ------------------------------------------------------------------
    # Full pipeline: fetch -> quick scan -> full screen
    # ------------------------------------------------------------------

    def scan(self, date: str,
             min_market_cap: float = 2e9, max_market_cap: float = None,
             tickers: List[str] = None, ticker_caps: Dict[str, str] = None,
             min_up_pct: float = 0.03, min_down_pct: float = -0.03,
             min_dollar_vol: float = 5e6,
             setup_type: str = 'both') -> List[ScreenResult]:
        """
        Full screening pipeline: fetch universe -> quick scan -> full screen.

        This is the main entry point for scanning 1000+ tickers.

        Step 1: Build ticker universe
          - If tickers provided: use those directly
          - Otherwise: fetch from Polygon filtered by market cap

        Step 2: Quick scan (snapshot pre-filter)
          - 1 API call for ALL tickers
          - Filters to extreme movers (big gap/change + sufficient volume)

        Step 3: Full screen candidates
          - 1 API call per candidate (concurrent with rate limiting)
          - Computes all metrics from 310-day history
          - Scores against parabolic short + capitulation bounce criteria

        Args:
            date: Date to screen (YYYY-MM-DD)
            min_market_cap: Min market cap for universe fetch (default $2B)
            max_market_cap: Max market cap (None = no limit)
            tickers: Use these tickers instead of fetching from Polygon
            ticker_caps: Dict of ticker -> cap overrides
            min_up_pct: Quick scan threshold for parabolic (default 3%)
            min_down_pct: Quick scan threshold for bounce (default -3%)
            min_dollar_vol: Min dollar volume for quick scan (default $5M)
            setup_type: 'parabolic', 'bounce', or 'both'

        Returns:
            List of ScreenResult for candidates that passed the quick scan
        """
        print(f"\n{'='*70}")
        print(f"SETUP SCREENER - {date}")
        print(f"{'='*70}")

        # Step 1: Build universe
        if tickers:
            universe_tickers = tickers
            print(f"\nStep 1: Using provided ticker list ({len(tickers)} tickers)")
        else:
            universe = self.fetch_universe(min_market_cap, max_market_cap)
            universe_tickers = list(universe.keys())
            # Merge fetched cap data with any overrides
            if ticker_caps is None:
                ticker_caps = universe
            else:
                merged = dict(universe)
                merged.update(ticker_caps)
                ticker_caps = merged

        # Step 2: Quick scan
        print(f"\nStep 2: Quick scan (snapshot pre-filter)...")
        candidates = self.quick_scan(
            tickers=universe_tickers,
            min_up_pct=min_up_pct,
            min_down_pct=min_down_pct,
            min_dollar_vol=min_dollar_vol,
            setup_type=setup_type,
        )

        if not candidates:
            print("\n  No candidates found in quick scan. Try lowering thresholds.")
            return []

        # Step 3: Full screen
        print(f"\nStep 3: Full screening {len(candidates)} candidates...")
        results = self.screen_universe(
            candidates, date,
            cap='Medium',
            ticker_caps=ticker_caps,
        )

        # Print results
        self.print_results(results)

        return results

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def get_candidates(self, results: List[ScreenResult],
                       setup_type: str = 'both',
                       min_parabolic_score: int = 3,
                       min_bounce_score: int = 4) -> List[ScreenResult]:
        """
        Filter results to only setup candidates.

        Args:
            results: List of ScreenResult
            setup_type: 'parabolic', 'bounce', or 'both'
            min_parabolic_score: Min parabolic short score (default 3/5)
            min_bounce_score: Min bounce score (default 4/6)
        """
        candidates = []
        for r in results:
            if r.error:
                continue
            if setup_type in ('parabolic', 'both') and r.parabolic_score >= min_parabolic_score:
                candidates.append(r)
                continue
            if setup_type in ('bounce', 'both') and r.bounce_score >= min_bounce_score:
                candidates.append(r)
                continue
        return candidates

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def print_results(self, results: List[ScreenResult], show_all: bool = False):
        """Print screening results as formatted tables."""
        valid = [r for r in results if not r.error]
        errors = [r for r in results if r.error]

        # --- Parabolic short candidates ---
        parabolic = sorted(
            [r for r in valid if show_all or r.is_parabolic_candidate],
            key=lambda r: r.parabolic_score,
            reverse=True,
        )

        print("\n" + "=" * 90)
        print("PARABOLIC SHORT CANDIDATES")
        print(f"Pre-trade criteria: 5 of 6 reversal criteria (excludes reversal %)")
        print(f"Candidates shown: score >= 3/5")
        print("=" * 90)

        if parabolic:
            rows = []
            for r in parabolic:
                m = r.metrics
                rows.append({
                    'Ticker': r.ticker,
                    'Cap': r.cap,
                    'Score': f"{r.parabolic_score}/{r.parabolic_max_score}",
                    'Grade': r.parabolic_grade,
                    '9EMA %': _fmt_pct(m.get('pct_from_9ema')),
                    'Range/ATR': _fmt_x(m.get('prior_day_range_atr')),
                    'RVOL': _fmt_x(m.get('rvol_score')),
                    'Up Days': _fmt_int(m.get('consecutive_up_days')),
                    'Gap %': _fmt_pct(m.get('gap_pct')),
                    'Rec': r.parabolic_recommendation,
                })
            print(tabulate(rows, headers='keys', tablefmt='simple', stralign='right'))
        else:
            print("  No parabolic short candidates found.")

        # --- Capitulation bounce candidates ---
        bounces = sorted(
            [r for r in valid if show_all or r.is_bounce_candidate],
            key=lambda r: r.bounce_score,
            reverse=True,
        )

        print("\n" + "=" * 90)
        print("CAPITULATION BOUNCE CANDIDATES")
        print(f"Pre-trade criteria: 6 of 7 bounce criteria (excludes bounce %)")
        print(f"Candidates shown: score >= 4/6")
        print("=" * 90)

        if bounces:
            rows = []
            for r in bounces:
                m = r.metrics
                rows.append({
                    'Ticker': r.ticker,
                    'Cap': r.cap,
                    'Score': f"{r.bounce_score}/{r.bounce_max_score}",
                    'Type': r.bounce_setup_type.replace('GapFade_', ''),
                    'Selloff': _fmt_pct(m.get('selloff_total_pct')),
                    'Down Days': _fmt_int(m.get('consecutive_down_days')),
                    'Off 30d Hi': _fmt_pct(m.get('pct_off_30d_high')),
                    'Gap %': _fmt_pct(m.get('gap_pct')),
                    'RVOL': _fmt_x(m.get('prior_day_rvol')),
                    'Rec': r.bounce_recommendation,
                })
            print(tabulate(rows, headers='keys', tablefmt='simple', stralign='right'))
        else:
            print("  No capitulation bounce candidates found.")

        # --- Summary ---
        print("\n" + "-" * 90)
        print(f"SUMMARY: {len(results)} tickers screened | "
              f"{len([r for r in valid if r.is_parabolic_candidate])} parabolic | "
              f"{len([r for r in valid if r.is_bounce_candidate])} bounce | "
              f"{len(errors)} errors")
        if errors:
            print(f"Errors: {', '.join(r.ticker for r in errors)}")
        print()

    def print_detailed(self, result: ScreenResult):
        """Print detailed breakdown for a single ticker."""
        print(f"\n{'=' * 70}")
        print(f"DETAILED SCREENING: {result.ticker} ({result.cap})")
        print(f"{'=' * 70}")

        if result.error:
            print(f"  ERROR: {result.error}")
            return

        m = result.metrics

        print(f"\nKEY METRICS:")
        if m.get('current_price'):
            print(f"  Price: ${m['current_price']:.2f}")
        print(f"  ATR: {_fmt_pct(m.get('atr_pct'))} of price")
        print(f"  9 EMA distance: {_fmt_pct(m.get('pct_from_9ema'))}")
        print(f"  50 SMA distance: {_fmt_pct(m.get('pct_from_50mav'))}")
        print(f"  200 SMA distance: {_fmt_pct(m.get('pct_from_200mav'))}")
        print(f"  Off 30d high: {_fmt_pct(m.get('pct_off_30d_high'))}")
        print(f"  Off 52wk high: {_fmt_pct(m.get('pct_off_52wk_high'))}")

        # Parabolic short
        print(f"\nPARABOLIC SHORT: {result.parabolic_score}/{result.parabolic_max_score} "
              f"({result.parabolic_grade}) - {result.parabolic_recommendation}")
        print("-" * 60)
        for name, detail in result.parabolic_criteria.items():
            status = "[PASS]" if detail['passed'] else "[FAIL]"
            actual = detail['actual']
            threshold = detail['threshold']

            if name in ('pct_from_9ema', 'gap_pct'):
                actual_s = f"{actual * 100:+.1f}%" if actual is not None else "N/A"
                thresh_s = f"{threshold * 100:.1f}%"
            elif name == 'consecutive_up_days':
                actual_s = str(int(actual)) if actual is not None else "N/A"
                thresh_s = str(int(threshold))
            else:
                actual_s = f"{actual:.2f}x" if actual is not None else "N/A"
                thresh_s = f"{threshold:.1f}x"

            print(f"  {status} {detail['name']}")
            print(f"         Required: >= {thresh_s} | Actual: {actual_s}")

        # Capitulation bounce
        print(f"\nCAPITULATION BOUNCE: {result.bounce_score}/{result.bounce_max_score} "
              f"({result.bounce_setup_type}) - {result.bounce_recommendation}")
        print("-" * 60)
        for name, detail in result.bounce_criteria.items():
            status = "[PASS]" if detail['passed'] else "[FAIL]"
            actual = detail['actual']
            threshold = detail['threshold']

            if name in ('selloff_total_pct', 'pct_off_30d_high', 'gap_pct'):
                actual_s = f"{actual * 100:.1f}%" if actual is not None else "N/A"
                thresh_s = f"{abs(threshold) * 100:.0f}%"
                direction = "<="
            elif name == 'consecutive_down_days':
                actual_s = str(int(actual)) if actual is not None else "N/A"
                thresh_s = str(int(threshold))
                direction = ">="
            elif name in ('vol_expansion', 'prior_day_range_atr'):
                if actual is not None and not isinstance(actual, str):
                    actual_s = f"{actual:.2f}x"
                else:
                    actual_s = str(actual) if actual is not None else "N/A"
                thresh_s = f"{threshold:.1f}x"
                direction = ">="
            else:
                actual_s = str(actual) if actual is not None else "N/A"
                thresh_s = str(threshold)
                direction = ">="

            print(f"  {status} {detail['name']}")
            print(f"         Required: {direction} {thresh_s} | Actual: {actual_s}")

        # Momentum context
        print(f"\nMOMENTUM CONTEXT:")
        print("-" * 60)
        for key, label in [('pct_change_3', '3-day'), ('pct_change_15', '15-day'),
                           ('pct_change_30', '30-day'), ('pct_change_90', '90-day'),
                           ('pct_change_120', '120-day')]:
            val = m.get(key)
            print(f"  {label:>8s} return: {_fmt_pct(val)}")
        print()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    return f"{val * 100:+.1f}%"


def _fmt_x(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    return f"{val:.2f}x"


def _fmt_int(val) -> str:
    if val is None:
        return 'N/A'
    return str(int(val))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Default to today (adjusted for weekends)
    today = datetime.now()
    if today.weekday() == 5:
        adjusted_date = today - timedelta(days=1)
    elif today.weekday() == 6:
        adjusted_date = today - timedelta(days=2)
    else:
        adjusted_date = today
    date = adjusted_date.strftime('%Y-%m-%d')

    if len(sys.argv) > 1:
        date = sys.argv[1]

    print(f"\nSetup Screener - {date}")
    print(f"Looking for: Parabolic Short + Capitulation Bounce setups\n")

    screener = SetupScreener(max_workers=5, rate_limit=5)

    # Full pipeline: fetches medium+large cap universe, quick scans via
    # snapshots, then full screens only the candidates that show extreme moves.
    results = screener.scan(date)

    # Detailed reports for top candidates
    parabolic_candidates = screener.get_candidates(results, setup_type='parabolic')
    bounce_candidates = screener.get_candidates(results, setup_type='bounce')

    if parabolic_candidates:
        print("\n" + "=" * 70)
        print("DETAILED PARABOLIC SHORT REPORTS")
        print("=" * 70)
        for r in sorted(parabolic_candidates,
                        key=lambda x: x.parabolic_score, reverse=True)[:5]:
            screener.print_detailed(r)

    if bounce_candidates:
        print("\n" + "=" * 70)
        print("DETAILED CAPITULATION BOUNCE REPORTS")
        print("=" * 70)
        for r in sorted(bounce_candidates,
                        key=lambda x: x.bounce_score, reverse=True)[:5]:
            screener.print_detailed(r)
